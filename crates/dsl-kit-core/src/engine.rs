//! The `dsl-kit` engine — an externally-driven, observable state machine
//! that walks a live frame tree.
//!
//! # Boundaries (design doc Appendix A)
//!
//! - **Engine owns runtime mechanics** — frame tree, suspension identity,
//!   pending bookkeeping, reducer resolution, cancellation queueing,
//!   delta plumbing.
//! - **AST owns intent** — node kinds, join policy, reducer identifier
//!   references, scope labels.
//! - **Host owns effect execution** — resolver dispatch, transport,
//!   runtime abort handles.
//!
//! # Shape
//!
//! [`Ast`] is the DSL's contract: it exposes its nodes as opaque
//! [`NodeId`]s and answers [`Ast::node_kind`] with a
//! [`NodeKind`] classification. [`Engine`] takes ownership of an `Ast`
//! plus a [`ReducerRegistry`] and drives the interpretation.
//!
//! # Execution model
//!
//! Internally the engine keeps an arena of `InternalFrame`s indexed by a
//! [`FrameHandle`]. The public [`Stepper::frame_tree`] method projects
//! the arena into the design-doc [`FrameTree`] shape on demand.
//! `Frame::Node` is not emitted; every AST node is either a control-flow
//! shape (`Seq` / `Par` / `Scope` / `Maybe`) rendered directly as its
//! matching `Frame` variant, or a `Call` leaf rendered as
//! `Frame::PendingEffect` / `Frame::Value` / `Frame::Cancelled`.
//!
//! # Reference
//!
//! See `workspace/tasks/dsl-kit-carry/async-join-design.md` — this
//! module is the concrete implementation of §3, §4, §5, §7.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use smallvec::SmallVec;

use crate::{
    CancelReason, ChildIndex, EngineError, EnvRef, FailPolicy, Frame, FrameTree, JoinPolicy,
    JoinShape, NodeContext, NodeId, ParFrame, Path, Pending, ReducerHandle, ReducerId,
    ReducerRegistry, StepOutcome, Stepper, SuspendReason, SuspensionId,
};

/// Composite error returned by the [`Engine`] via [`Stepper::Error`].
///
/// Wraps either an engine-level error ([`EngineError`]) or a raw
/// DSL-supplied effect-side error surfaced through `resolve(id, Err(_))`
/// and then propagated by the enclosing `Par` under `FailPolicy::FailFast`.
///
/// # Conversions
///
/// - `From<EE> for ExecError<EE>` — the `Stepper::Error: From<EffectError>`
///   contract.
/// - `From<EngineError>` is intentionally NOT derived (would conflict with
///   the blanket `From<EE>` when a caller happens to pick `EE = EngineError`).
///   Wrap engine-level errors explicitly via [`ExecError::Engine`].
#[derive(Debug, thiserror::Error)]
pub enum ExecError<EE: std::error::Error + Send + Sync + 'static> {
    /// An engine-level error.
    #[error(transparent)]
    Engine(EngineError),
    /// A DSL-supplied effect-side error propagated through FailFast.
    #[error(transparent)]
    Effect(EE),
}

impl<EE> From<EE> for ExecError<EE>
where
    EE: std::error::Error + Send + Sync + 'static,
{
    fn from(e: EE) -> Self {
        ExecError::Effect(e)
    }
}

// =====================================================================
// Public trait / value types
// =====================================================================

/// The DSL's contract with the engine.
///
/// An `Ast` is a read-only oracle: given a [`NodeId`], it names the
/// node's shape via [`Self::node_kind`] and reports the root. The AST
/// itself is opaque to the engine — the DSL owns storage layout,
/// serialization, and traversal helpers.
///
/// # Associated types
///
/// - `Value` — the DSL's runtime value type (what `Call` produces on
///   resolve, what a completed interpretation returns).
/// - `Delta` — per-child state delta accumulated across a `Par`;
///   `()` for DSLs with no shared mutable state.
/// - `EffectError` — the DSL's transport / domain error surfaced via
///   `resolve(id, Err(_))`.
pub trait Ast {
    /// Runtime value type produced by the DSL.
    type Value: Clone + Debug;
    /// Per-child state delta accumulated across a `Par`.
    type Delta: Clone + Default + Debug;
    /// Effect-side failure the host reports through `resolve`.
    type EffectError: std::error::Error + Clone + Send + Sync + 'static;

    /// Root node the engine anchors its interpretation on.
    fn root(&self) -> NodeId;

    /// Classify a node's shape. Called at most once per node per
    /// interpretation.
    fn node_kind(&self, id: NodeId) -> NodeKind;

    /// Value produced by an empty control-flow branch — an empty `Seq`,
    /// a `Maybe(None)`, or a `Scope` whose body evaluates to nothing.
    /// The engine also uses this to fill a non-Call `Par` child's slot
    /// when the child's subtree drains without an explicit value.
    fn unit_value(&self) -> Self::Value;
}

/// A node's structural classification.
///
/// The engine handles all five kinds natively. `Call` is the only leaf
/// that yields a [`Pending`]; the other four are control-flow shapes
/// whose progress is derived from their children's values.
#[derive(Debug, Clone)]
pub enum NodeKind {
    /// Evaluate children left-to-right, propagating the last value.
    Seq {
        /// Child nodes, in declaration order.
        children: Vec<NodeId>,
    },
    /// Fan out children concurrently and fold their slots via a reducer.
    Par {
        /// Child nodes; each occupies one `ParFrame` slot.
        children: Vec<NodeId>,
        /// Join policy (success shape + failure handling).
        policy: JoinPolicy,
        /// Reducer identifier looked up in the [`ReducerRegistry`] at
        /// `Par` entry.
        reducer_id: ReducerId,
    },
    /// Wrap a single body node with a human-readable label.
    Scope {
        /// Label surfaced through the projected [`Frame::Scope`].
        label: String,
        /// The inner body evaluated within the scope.
        body: NodeId,
    },
    /// Optionally evaluate a body node.
    Maybe {
        /// The inner body when present.
        body: Option<NodeId>,
    },
    /// A leaf that suspends the interpretation until the host resolves
    /// the matching [`SuspensionId`].
    Call {
        /// Short human-readable label surfaced through
        /// [`crate::CallSpec::label`].
        label: String,
        /// Opaque payload the host inspects when dispatching the effect.
        payload: serde_json::Value,
    },
}

// =====================================================================
// Internal frame arena
// =====================================================================

/// Opaque handle to a frame in the engine's internal arena. Not exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FrameHandle(usize);

/// A single node in the engine's arena. Each variant corresponds to one
/// running / suspended / completed AST node.
#[allow(dead_code)]
enum InternalFrame<A: Ast> {
    /// A `Seq` node in progress. `next_child` is the index into
    /// `children` to spawn next; `current` is the handle of the
    /// currently-running child (if any).
    Seq {
        node: NodeId,
        path: Path,
        children: Vec<NodeId>,
        next_child: usize,
        current: Option<FrameHandle>,
        last_value: Option<A::Value>,
    },
    /// A `Par` node in progress.
    Par {
        node: NodeId,
        path: Path,
        policy: JoinPolicy,
        #[allow(dead_code)]
        reducer_id: ReducerId,
        reducer: ReducerHandle<A::Value, A::Delta, A::EffectError>,
        children: Vec<FrameHandle>,
        slots: Vec<Option<A::Value>>,
        failures: Vec<Option<A::EffectError>>,
        deltas: Vec<Option<A::Delta>>,
        completion_order: Vec<ChildIndex>,
        joined: bool,
    },
    /// A `Scope` in progress.
    Scope {
        node: NodeId,
        path: Path,
        label: String,
        body: FrameHandle,
        body_value: Option<A::Value>,
    },
    /// A `Maybe` in progress.
    Maybe {
        node: NodeId,
        path: Path,
        body: Option<FrameHandle>,
        body_value: Option<A::Value>,
    },
    /// A `Call` leaf awaiting a `resolve` from the host.
    Pending {
        node: NodeId,
        path: Path,
        sid: SuspensionId,
        label: String,
    },
    /// A completed leaf whose value is waiting to be consumed by its
    /// parent.
    Value {
        node: NodeId,
        value: A::Value,
    },
    /// A leaf whose `resolve` returned `Err(_)`. The next `step()`
    /// applies the enclosing `Par.policy.fail`.
    Failed {
        node: NodeId,
        path: Path,
        error: A::EffectError,
    },
    /// A sub-tree the engine cancelled. Retained so `frame_tree()` can
    /// render it.
    Cancelled {
        node: NodeId,
        reason: CancelReason,
    },
}

impl<A: Ast> InternalFrame<A> {
    fn node(&self) -> NodeId {
        match self {
            InternalFrame::Seq { node, .. }
            | InternalFrame::Par { node, .. }
            | InternalFrame::Scope { node, .. }
            | InternalFrame::Maybe { node, .. }
            | InternalFrame::Pending { node, .. }
            | InternalFrame::Value { node, .. }
            | InternalFrame::Failed { node, .. }
            | InternalFrame::Cancelled { node, .. } => *node,
        }
    }
}

// =====================================================================
// The Engine
// =====================================================================

/// The engine: takes ownership of an [`Ast`] + a [`ReducerRegistry`]
/// and drives the interpretation via the [`Stepper`] trait.
pub struct Engine<A: Ast> {
    ast: A,
    registry: Arc<ReducerRegistry<A::Value, A::Delta, A::EffectError>>,

    frames: Vec<Option<InternalFrame<A>>>,
    root: FrameHandle,
    parent: HashMap<FrameHandle, FrameHandle>,

    pending: Vec<Pending>,
    /// Pending created since the last `step()` return. Drained per
    /// `Stepper::step` call and reported as
    /// `StepOutcome::Blocked.newly_pending`. Distinct from
    /// `self.pending` because cancellations can shrink the latter.
    newly_pending: Vec<Pending>,
    sid_to_frame: HashMap<SuspensionId, FrameHandle>,
    cancellations: Vec<SuspensionId>,

    next_sid: u64,
    done: bool,
    root_value: Option<A::Value>,
}

impl<A: Ast> Engine<A> {
    /// Build a fresh engine anchored at `ast.root()` with the supplied
    /// reducer registry.
    ///
    /// Returns [`EngineError::Malformed`] if the root or any spawned
    /// node fails the design-doc §3.5 validation table.
    pub fn new(
        ast: A,
        registry: Arc<ReducerRegistry<A::Value, A::Delta, A::EffectError>>,
    ) -> Result<Self, EngineError> {
        let root_id = ast.root();
        let mut engine = Self {
            ast,
            registry,
            frames: Vec::new(),
            root: FrameHandle(0),
            parent: HashMap::new(),
            pending: Vec::new(),
            newly_pending: Vec::new(),
            sid_to_frame: HashMap::new(),
            cancellations: Vec::new(),
            next_sid: 1,
            done: false,
            root_value: None,
        };
        let root_path = Path::root().push(root_id);
        let root_frame = engine.spawn_frame(root_id, root_path)?;
        engine.root = root_frame;
        Ok(engine)
    }

    // ---- Arena helpers ---------------------------------------------

    fn allocate(&mut self, frame: InternalFrame<A>) -> FrameHandle {
        let h = FrameHandle(self.frames.len());
        self.frames.push(Some(frame));
        h
    }

    fn get(&self, h: FrameHandle) -> &InternalFrame<A> {
        self.frames[h.0].as_ref().expect("frame slot vacated")
    }

    fn get_mut(&mut self, h: FrameHandle) -> &mut InternalFrame<A> {
        self.frames[h.0].as_mut().expect("frame slot vacated")
    }

    fn set_parent(&mut self, child: FrameHandle, parent: FrameHandle) {
        self.parent.insert(child, parent);
    }

    // ---- Spawning a frame from a NodeId -----------------------------

    /// Spawn an arena frame for `node_id`. Validates `Par` shape.
    fn spawn_frame(
        &mut self,
        node_id: NodeId,
        path: Path,
    ) -> Result<FrameHandle, EngineError> {
        match self.ast.node_kind(node_id) {
            NodeKind::Seq { children } => {
                let frame = InternalFrame::Seq {
                    node: node_id,
                    path,
                    children,
                    next_child: 0,
                    current: None,
                    last_value: None,
                };
                Ok(self.allocate(frame))
            }
            NodeKind::Par { children, policy, reducer_id } => {
                Self::validate_par(node_id, &children, policy)?;
                let reducer = self.registry.resolve(&reducer_id, policy.fail)?;
                let n = children.len();
                let par_handle = self.allocate(InternalFrame::Par {
                    node: node_id,
                    path: path.clone(),
                    policy,
                    reducer_id,
                    reducer,
                    children: Vec::with_capacity(n),
                    slots: vec![None; n],
                    failures: (0..n).map(|_| None).collect(),
                    deltas: vec![None; n],
                    completion_order: Vec::new(),
                    joined: false,
                });
                // Spawn every child eagerly.
                let mut child_handles = Vec::with_capacity(n);
                for child_id in children {
                    let child_path = path.push(child_id);
                    let child = self.spawn_frame(child_id, child_path)?;
                    self.set_parent(child, par_handle);
                    child_handles.push(child);
                }
                if let InternalFrame::Par { children, .. } = self.get_mut(par_handle) {
                    *children = child_handles;
                }
                Ok(par_handle)
            }
            NodeKind::Scope { label, body } => {
                let body_path = path.push(body);
                let body_handle = self.spawn_frame(body, body_path)?;
                let scope_handle = self.allocate(InternalFrame::Scope {
                    node: node_id,
                    path,
                    label,
                    body: body_handle,
                    body_value: None,
                });
                self.set_parent(body_handle, scope_handle);
                Ok(scope_handle)
            }
            NodeKind::Maybe { body } => match body {
                Some(body_id) => {
                    let body_path = path.push(body_id);
                    let body_handle = self.spawn_frame(body_id, body_path)?;
                    let maybe_handle = self.allocate(InternalFrame::Maybe {
                        node: node_id,
                        path,
                        body: Some(body_handle),
                        body_value: None,
                    });
                    self.set_parent(body_handle, maybe_handle);
                    Ok(maybe_handle)
                }
                None => {
                    let unit = self.ast.unit_value();
                    let maybe_handle = self.allocate(InternalFrame::Maybe {
                        node: node_id,
                        path,
                        body: None,
                        body_value: Some(unit),
                    });
                    Ok(maybe_handle)
                }
            },
            NodeKind::Call { label, payload: _ } => {
                let sid = SuspensionId(self.next_sid);
                self.next_sid += 1;
                let ctx = NodeContext {
                    node: node_id,
                    path: path.clone(),
                    frame: None,
                    depth: path.depth() as u32,
                    iteration: None,
                };
                let spec = crate::CallSpec {
                    label: label.clone(),
                    payload: serde_json::Value::Null,
                };
                let pending = Pending {
                    id: sid,
                    reason: SuspendReason::Call { spec },
                    at: ctx,
                };
                self.pending.push(pending.clone());
                self.newly_pending.push(pending);
                let frame = InternalFrame::Pending {
                    node: node_id,
                    path,
                    sid,
                    label,
                };
                let h = self.allocate(frame);
                self.sid_to_frame.insert(sid, h);
                Ok(h)
            }
        }
    }

    fn validate_par(
        node_id: NodeId,
        children: &[NodeId],
        policy: JoinPolicy,
    ) -> Result<(), EngineError> {
        let n = children.len();
        let malformed = |detail: String| EngineError::Malformed {
            at: NodeContext::at(node_id, Path::root().push(node_id)),
            detail,
        };
        match policy.shape {
            JoinShape::All => Ok(()),
            JoinShape::Any => {
                if n == 0 {
                    Err(malformed("Par with JoinShape::Any requires >= 1 child".into()))
                } else {
                    Ok(())
                }
            }
            JoinShape::FirstK(k) => {
                if k > n {
                    Err(malformed(format!(
                        "Par with JoinShape::FirstK({k}) requires children.len() >= k (got {n})"
                    )))
                } else {
                    Ok(())
                }
            }
        }
    }

    // ---- The step loop ---------------------------------------------

    fn step_once(&mut self) -> Result<StepStep<A::Value>, ExecError<A::EffectError>> {
        if self.done {
            return Ok(StepStep::Done);
        }

        // First, look for a failed leaf and propagate FailFast if any.
        if let Some(failed_leaf) = self.find_failed_leaf(self.root) {
            return self.propagate_failure(failed_leaf);
        }

        // Then, try to fold a fireable Par (shape satisfied).
        if let Some(par_handle) = self.find_fireable_par(self.root) {
            return self.fire_par(par_handle).map_err(ExecError::Engine);
        }

        // Then, look for a Seq whose current child produced a value or
        // is unspawned, or a Scope/Maybe whose body produced a value.
        if self.advance_completed(self.root).map_err(ExecError::Engine)? {
            return Ok(StepStep::Ready);
        }

        // Nothing to advance — we're blocked on pending suspensions.
        Ok(StepStep::Blocked)
    }

    // ---- Value propagation upward ----------------------------------

    /// Walk the tree looking for a frame that can advance because its
    /// child produced a value (Seq: current child done; Scope/Maybe:
    /// body done; Par: never advances via this path, handled in
    /// `fire_par`). Returns true if any advance happened.
    fn advance_completed(&mut self, h: FrameHandle) -> Result<bool, EngineError> {
        // We do a bottom-up walk: descend, then act on the way back.
        match self.get(h) {
            InternalFrame::Cancelled { .. }
            | InternalFrame::Value { .. }
            | InternalFrame::Failed { .. }
            | InternalFrame::Pending { .. } => Ok(false),
            InternalFrame::Seq { .. } => self.advance_seq(h),
            InternalFrame::Par { children, .. } => {
                let children = children.clone();
                for c in children {
                    if self.advance_completed(c)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            InternalFrame::Scope { body, .. } => {
                let body = *body;
                if self.advance_completed(body)? {
                    return Ok(true);
                }
                self.try_scope_promote(h)
            }
            InternalFrame::Maybe { body, .. } => {
                if let Some(body) = *body {
                    if self.advance_completed(body)? {
                        return Ok(true);
                    }
                }
                self.try_maybe_promote(h)
            }
        }
    }

    fn advance_seq(&mut self, h: FrameHandle) -> Result<bool, EngineError> {
        // Descend into current child first.
        let current = if let InternalFrame::Seq { current, .. } = self.get(h) {
            *current
        } else {
            unreachable!()
        };
        if let Some(c) = current {
            if self.advance_completed(c)? {
                return Ok(true);
            }
            // Check if current child is a Value; if so, consume and move on.
            if let InternalFrame::Value { value, .. } = self.get(c) {
                let value = value.clone();
                if let InternalFrame::Seq {
                    current, last_value, ..
                } = self.get_mut(h)
                {
                    *current = None;
                    *last_value = Some(value);
                }
                self.vacate(c);
                return self.try_seq_next(h);
            }
            return Ok(false);
        }
        // No current child — spawn the next one, or promote to Value.
        self.try_seq_next(h)
    }

    /// If there is a next child, spawn it. Otherwise promote the Seq to
    /// a Value (last child's value, or unit).
    fn try_seq_next(&mut self, h: FrameHandle) -> Result<bool, EngineError> {
        let (node, path, next_child, children_len, next_id) = {
            let f = self.get(h);
            if let InternalFrame::Seq {
                node, path, children, next_child, ..
            } = f
            {
                let id = children.get(*next_child).copied();
                (*node, path.clone(), *next_child, children.len(), id)
            } else {
                unreachable!()
            }
        };
        if let Some(next_id) = next_id {
            let child_path = path.push(next_id);
            let child = self.spawn_frame(next_id, child_path)?;
            self.set_parent(child, h);
            if let InternalFrame::Seq {
                current, next_child, ..
            } = self.get_mut(h)
            {
                *current = Some(child);
                *next_child += 1;
            }
            Ok(true)
        } else {
            // No more children — promote Seq to a Value.
            let last = if let InternalFrame::Seq { last_value, .. } = self.get(h) {
                last_value.clone()
            } else {
                unreachable!()
            };
            let value = last.unwrap_or_else(|| self.ast.unit_value());
            let promoted = InternalFrame::Value { node, value: value.clone() };
            *self.get_mut(h) = promoted;
            self.notify_par_slot(h, value);
            let _ = (next_child, children_len);
            Ok(true)
        }
    }

    fn try_scope_promote(&mut self, h: FrameHandle) -> Result<bool, EngineError> {
        let (node, body) = if let InternalFrame::Scope { node, body, .. } = self.get(h) {
            (*node, *body)
        } else {
            unreachable!()
        };
        if let InternalFrame::Value { value, .. } = self.get(body) {
            let value = value.clone();
            self.vacate(body);
            *self.get_mut(h) = InternalFrame::Value { node, value: value.clone() };
            self.notify_par_slot(h, value);
            return Ok(true);
        }
        Ok(false)
    }

    fn try_maybe_promote(&mut self, h: FrameHandle) -> Result<bool, EngineError> {
        let (node, body) = if let InternalFrame::Maybe { node, body, .. } = self.get(h) {
            (*node, *body)
        } else {
            unreachable!()
        };
        // Maybe(None) was already given a unit body_value at spawn.
        let promote_value = if let Some(body) = body {
            if let InternalFrame::Value { value, .. } = self.get(body) {
                Some(value.clone())
            } else {
                None
            }
        } else if let InternalFrame::Maybe { body_value: Some(v), .. } = self.get(h) {
            Some(v.clone())
        } else {
            None
        };
        if let Some(v) = promote_value {
            if let Some(body) = body {
                self.vacate(body);
            }
            *self.get_mut(h) = InternalFrame::Value { node, value: v.clone() };
            self.notify_par_slot(h, v);
            return Ok(true);
        }
        Ok(false)
    }

    // ---- Par firing -----------------------------------------------

    /// Walk the tree looking for a Par whose shape has fired but not
    /// yet been folded.
    fn find_fireable_par(&self, h: FrameHandle) -> Option<FrameHandle> {
        match self.get(h) {
            InternalFrame::Par {
                joined,
                completion_order,
                policy,
                slots,
                failures,
                children,
                ..
            } => {
                if !*joined && self.par_shape_fires(policy, completion_order, slots, failures) {
                    return Some(h);
                }
                // Also descend into children.
                for c in children {
                    if let Some(inner) = self.find_fireable_par(*c) {
                        return Some(inner);
                    }
                }
                None
            }
            InternalFrame::Seq { current, .. } => {
                current.and_then(|c| self.find_fireable_par(c))
            }
            InternalFrame::Scope { body, .. } => self.find_fireable_par(*body),
            InternalFrame::Maybe { body, .. } => body.and_then(|b| self.find_fireable_par(b)),
            _ => None,
        }
    }

    fn par_shape_fires(
        &self,
        policy: &JoinPolicy,
        completion_order: &[ChildIndex],
        slots: &[Option<A::Value>],
        failures: &[Option<A::EffectError>],
    ) -> bool {
        let n = slots.len();
        let success = completion_order.len();
        let fail_count = failures.iter().filter(|f| f.is_some()).count();
        match policy.shape {
            JoinShape::All => match policy.fail {
                FailPolicy::FailFast => success == n,
                FailPolicy::CollectAll => success + fail_count == n,
            },
            JoinShape::Any => match policy.fail {
                FailPolicy::FailFast => success >= 1,
                FailPolicy::CollectAll => {
                    if success >= 1 {
                        true
                    } else {
                        fail_count == n && n > 0
                    }
                }
            },
            JoinShape::FirstK(k) => match policy.fail {
                FailPolicy::FailFast => success >= k,
                FailPolicy::CollectAll => {
                    if success >= k {
                        true
                    } else {
                        // k successes no longer attainable?
                        let remaining = n - success - fail_count;
                        success + remaining < k
                    }
                }
            },
        }
    }

    /// Fold a fireable Par, mark losers as cancelled, promote to Value.
    fn fire_par(&mut self, h: FrameHandle) -> Result<StepStep<A::Value>, EngineError> {
        // Extract fields.
        let (node, policy, reducer, slots, failures, deltas, completion_order, children) = {
            let f = self.get(h);
            if let InternalFrame::Par {
                node,
                policy,
                reducer,
                slots,
                failures,
                deltas,
                completion_order,
                children,
                ..
            } = f
            {
                (
                    *node,
                    *policy,
                    reducer.clone(),
                    slots.clone(),
                    failures.clone(),
                    deltas.clone(),
                    completion_order.clone(),
                    children.clone(),
                )
            } else {
                unreachable!()
            }
        };

        // Determine winners for shape.
        let winners: Vec<ChildIndex> = match policy.shape {
            JoinShape::All => completion_order.clone(),
            JoinShape::Any => completion_order.first().copied().into_iter().collect(),
            JoinShape::FirstK(k) => completion_order.iter().take(k).copied().collect(),
        };

        // Invoke reducer.
        let reduce_result: Result<(A::Value, A::Delta), EngineError> = match &reducer {
            ReducerHandle::FailFast(r) => r.reduce(&slots, &deltas, &winners),
            ReducerHandle::CollectAll(r) => {
                // Assemble Option<Result<V, EE>> slot vector.
                let mut assembled: Vec<Option<Result<A::Value, A::EffectError>>> =
                    Vec::with_capacity(slots.len());
                for i in 0..slots.len() {
                    if let Some(v) = &slots[i] {
                        assembled.push(Some(Ok(v.clone())));
                    } else if let Some(e) = &failures[i] {
                        assembled.push(Some(Err(e.clone())));
                    } else {
                        assembled.push(None);
                    }
                }
                r.reduce(&assembled, &deltas, &winners)
            }
        };

        let (value, _delta) = reduce_result?;

        // Cancel losing children (any child not in winners and not yet
        // succeeded / failed).
        let winner_set: std::collections::HashSet<ChildIndex> =
            winners.iter().copied().collect();
        for (i, &child) in children.iter().enumerate() {
            if slots[i].is_some() {
                // Winner or a slot-filled success — mark unused winners as
                // Cancelled (ParPolicyFired) if not in winner_set.
                if !winner_set.contains(&i) {
                    self.cancel_subtree(child, CancelReason::ParPolicyFired);
                }
            } else if failures[i].is_some() {
                // Failed; leave as is (CollectAll retains failure state).
            } else {
                // Still running or pending — cancel.
                self.cancel_subtree(child, CancelReason::ParPolicyFired);
            }
        }

        // Mark Par as joined + promote to Value.
        *self.get_mut(h) = InternalFrame::Value { node, value: value.clone() };
        // If this Par sits inside another Par slot, notify.
        self.notify_par_slot(h, value);

        Ok(StepStep::Ready)
    }

    // ---- FailFast propagation -------------------------------------

    fn find_failed_leaf(&self, h: FrameHandle) -> Option<FrameHandle> {
        match self.get(h) {
            InternalFrame::Failed { .. } => Some(h),
            InternalFrame::Seq { current, .. } => {
                current.and_then(|c| self.find_failed_leaf(c))
            }
            InternalFrame::Par {
                children, policy, ..
            } => {
                // CollectAll swallows leaf failures until fire; only
                // FailFast propagates via this path.
                if matches!(policy.fail, FailPolicy::CollectAll) {
                    return None;
                }
                for c in children {
                    if let Some(f) = self.find_failed_leaf(*c) {
                        return Some(f);
                    }
                }
                None
            }
            InternalFrame::Scope { body, .. } => self.find_failed_leaf(*body),
            InternalFrame::Maybe { body, .. } => body.and_then(|b| self.find_failed_leaf(b)),
            _ => None,
        }
    }

    fn propagate_failure(
        &mut self,
        failed: FrameHandle,
    ) -> Result<StepStep<A::Value>, ExecError<A::EffectError>> {
        // Extract the error.
        let (_node, _path, error) =
            if let InternalFrame::Failed { node, path, error } = self.get(failed) {
                (*node, path.clone(), error.clone())
            } else {
                unreachable!()
            };

        // Walk upward. If we hit an enclosing Par (FailFast), cancel
        // its other children and continue propagating.
        let mut cur = failed;
        while let Some(&parent) = self.parent.get(&cur) {
            if let InternalFrame::Par {
                policy, children, ..
            } = self.get(parent)
            {
                if matches!(policy.fail, FailPolicy::FailFast) {
                    let siblings: Vec<FrameHandle> =
                        children.iter().copied().filter(|c| *c != cur).collect();
                    for s in siblings {
                        self.cancel_subtree(s, CancelReason::SiblingFailed);
                    }
                }
            }
            cur = parent;
        }

        self.done = true;
        // Preserve the DSL's raw EffectError verbatim, per design §5.1
        // (FailFast propagates the child's error upward as
        // `Err(Self::Error)`).
        Err(ExecError::Effect(error))
    }

    // ---- Cancellation ---------------------------------------------

    /// Recursively mark a sub-tree as Cancelled, queueing pending sids.
    fn cancel_subtree(&mut self, h: FrameHandle, reason: CancelReason) {
        // Descend first (collect children handles).
        let child_handles = match self.get(h) {
            InternalFrame::Cancelled { .. }
            | InternalFrame::Value { .. }
            | InternalFrame::Failed { .. } => return,
            InternalFrame::Pending { sid, node, .. } => {
                let sid = *sid;
                let node = *node;
                self.pending.retain(|p| p.id != sid);
                self.newly_pending.retain(|p| p.id != sid);
                self.sid_to_frame.remove(&sid);
                self.cancellations.push(sid);
                *self.get_mut(h) = InternalFrame::Cancelled { node, reason };
                return;
            }
            InternalFrame::Seq { current, .. } => current.iter().copied().collect::<Vec<_>>(),
            InternalFrame::Par { children, .. } => children.clone(),
            InternalFrame::Scope { body, .. } => vec![*body],
            InternalFrame::Maybe { body, .. } => body.iter().copied().collect(),
        };
        let node = self.get(h).node();
        for c in child_handles {
            self.cancel_subtree(c, reason.clone());
        }
        *self.get_mut(h) = InternalFrame::Cancelled { node, reason };
    }

    fn vacate(&mut self, h: FrameHandle) {
        self.frames[h.0] = None;
        self.parent.remove(&h);
    }

    /// If `child` sits directly under a `Par` slot, fill that slot
    /// with `value` and record completion order. Idempotent: filling a
    /// slot that already has a value is a no-op.
    ///
    /// Called from every point that transitions a frame to
    /// `Value` (Seq / Scope / Maybe promotions; `resolve` for Call
    /// fast-path handles it inline).
    fn notify_par_slot(&mut self, child: FrameHandle, value: A::Value) {
        if let Some(&parent) = self.parent.get(&child) {
            if let InternalFrame::Par {
                children,
                slots,
                completion_order,
                ..
            } = self.get_mut(parent)
            {
                if let Some(idx) = children.iter().position(|c| *c == child) {
                    if slots[idx].is_none() {
                        slots[idx] = Some(value);
                        completion_order.push(idx);
                    }
                }
            }
        }
    }

    // ---- FrameTree projection --------------------------------------

    fn project_tree(
        &self,
        h: FrameHandle,
    ) -> FrameTree<A::Value, (), A::Delta, A::EffectError> {
        match self.get(h) {
            InternalFrame::Seq { children, current, node, path, .. } => {
                // Project each spawned child (only the current one is
                // live under the DFS-pre-order kids list).
                let mut kids = Vec::new();
                if let Some(c) = current {
                    kids.push(self.project_tree(*c));
                }
                let _ = (children, path);
                FrameTree {
                    root: Frame::Node {
                        node: *node,
                        env: EnvRef(Arc::new(crate::Env {
                            delta: A::Delta::default(),
                            parent: None,
                        })),
                        cursor: (),
                    },
                    kids,
                }
            }
            InternalFrame::Par {
                node,
                policy,
                slots,
                failures: _,
                deltas,
                completion_order,
                children,
                reducer_id,
                reducer,
                joined,
                path: _,
            } => {
                let kids = children.iter().map(|c| self.project_tree(*c)).collect();
                let par = ParFrame {
                    policy: *policy,
                    slots: slots.clone(),
                    failures: vec![],
                    deltas: deltas.clone(),
                    completion_order: completion_order.clone(),
                    reducer_id: reducer_id.clone(),
                    reducer: reducer.clone(),
                    joined: *joined,
                };
                let _ = node;
                FrameTree { root: Frame::Par(par), kids }
            }
            InternalFrame::Scope { label, body, .. } => {
                let kids = vec![self.project_tree(*body)];
                FrameTree {
                    root: Frame::Scope {
                        label: label.clone(),
                        env: EnvRef(Arc::new(crate::Env {
                            delta: A::Delta::default(),
                            parent: None,
                        })),
                    },
                    kids,
                }
            }
            InternalFrame::Maybe { body, .. } => {
                let kids = body.iter().map(|b| self.project_tree(*b)).collect();
                FrameTree {
                    root: Frame::Scope {
                        label: "<maybe>".into(),
                        env: EnvRef(Arc::new(crate::Env {
                            delta: A::Delta::default(),
                            parent: None,
                        })),
                    },
                    kids,
                }
            }
            InternalFrame::Pending { sid, .. } => FrameTree {
                root: Frame::PendingEffect { id: *sid },
                kids: vec![],
            },
            InternalFrame::Value { value, .. } => FrameTree {
                root: Frame::Value(value.clone()),
                kids: vec![],
            },
            InternalFrame::Failed { .. } => FrameTree {
                root: Frame::Cancelled { reason: CancelReason::SiblingFailed },
                kids: vec![],
            },
            InternalFrame::Cancelled { reason, .. } => FrameTree {
                root: Frame::Cancelled { reason: reason.clone() },
                kids: vec![],
            },
        }
    }
}

/// Internal step outcome — the arena loop's per-iteration signal.
enum StepStep<V> {
    Ready,
    Blocked,
    #[allow(dead_code)]
    Done,
    #[allow(dead_code)]
    _Phantom(std::marker::PhantomData<V>),
}

impl<A: Ast> std::fmt::Debug for Engine<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("frames", &self.frames.len())
            .field("pending", &self.pending.len())
            .field("cancellations", &self.cancellations.len())
            .field("done", &self.done)
            .finish()
    }
}

impl<A: Ast> Stepper for Engine<A> {
    type Value = A::Value;
    type Cursor = ();
    type Delta = A::Delta;
    type EffectError = A::EffectError;
    type Error = ExecError<A::EffectError>;

    fn step(&mut self) -> Result<StepOutcome<Self::Value>, Self::Error> {
        if self.done {
            if let Some(v) = self.root_value.clone() {
                return Ok(StepOutcome::Done(v));
            }
            // Terminal failure state; caller must treat prior Err as terminal.
            return Ok(StepOutcome::Blocked { newly_pending: SmallVec::new() });
        }

        // Root-already-Value fast path — a `Call` root whose leaf was
        // resolved before step() ran needs to be promoted to Done here
        // because no interior state machine will fire.
        if let InternalFrame::Value { value, .. } = self.get(self.root) {
            let v = value.clone();
            self.done = true;
            self.root_value = Some(v.clone());
            self.newly_pending.clear();
            return Ok(StepOutcome::Done(v));
        }

        loop {
            match self.step_once()? {
                StepStep::Ready => {
                    // Check root — extract first, then mutate.
                    let root_value = if let InternalFrame::Value { value, .. } =
                        self.get(self.root)
                    {
                        Some(value.clone())
                    } else {
                        None
                    };
                    if let Some(v) = root_value {
                        self.done = true;
                        self.root_value = Some(v.clone());
                        self.newly_pending.clear();
                        return Ok(StepOutcome::Done(v));
                    }
                    // Keep looping to advance further — per design §5.1,
                    // step() advances as far as possible; the presence
                    // of newly_pending only dominates the return
                    // *state* (Blocked, not Ready), not the loop-exit
                    // decision.
                    continue;
                }
                StepStep::Blocked => {
                    let newly: SmallVec<[Pending; 1]> =
                        std::mem::take(&mut self.newly_pending).into_iter().collect();
                    return Ok(StepOutcome::Blocked { newly_pending: newly });
                }
                StepStep::Done => {
                    if let Some(v) = self.root_value.clone() {
                        return Ok(StepOutcome::Done(v));
                    }
                    return Ok(StepOutcome::Blocked { newly_pending: SmallVec::new() });
                }
                StepStep::_Phantom(_) => unreachable!(),
            }
        }
    }

    fn resolve(
        &mut self,
        id: SuspensionId,
        result: Result<Self::Value, Self::EffectError>,
    ) -> Result<(), Self::Error> {
        let h = match self.sid_to_frame.remove(&id) {
            Some(h) => h,
            None => return Err(ExecError::Engine(EngineError::UnknownSuspension { id })),
        };
        self.pending.retain(|p| p.id != id);

        match result {
            Ok(v) => {
                let (node, _path) = if let InternalFrame::Pending { node, path, .. } = self.get(h)
                {
                    (*node, path.clone())
                } else {
                    return Err(ExecError::Engine(EngineError::UnknownSuspension { id }));
                };
                // Replace leaf with Value.
                *self.get_mut(h) = InternalFrame::Value { node, value: v.clone() };
                // If parent is a Par, fill the slot.
                if let Some(&parent) = self.parent.get(&h) {
                    if let InternalFrame::Par { children, slots, completion_order, .. } =
                        self.get_mut(parent)
                    {
                        if let Some(idx) = children.iter().position(|c| *c == h) {
                            slots[idx] = Some(v);
                            completion_order.push(idx);
                        }
                    }
                }
                Ok(())
            }
            Err(e) => {
                let (node, path) = if let InternalFrame::Pending { node, path, .. } = self.get(h) {
                    (*node, path.clone())
                } else {
                    return Err(ExecError::Engine(EngineError::UnknownSuspension { id }));
                };
                // Determine enclosing Par (if any) fail policy.
                let mut enclosing_par_fail: Option<FailPolicy> = None;
                let mut cur = h;
                while let Some(&parent) = self.parent.get(&cur) {
                    if let InternalFrame::Par { policy, .. } = self.get(parent) {
                        enclosing_par_fail = Some(policy.fail);
                        break;
                    }
                    cur = parent;
                }
                match enclosing_par_fail {
                    Some(FailPolicy::CollectAll) => {
                        // Record in Par failures + mark leaf Cancelled(SiblingFailed sentinel not appropriate;
                        // Use a Value-like slot: leaf becomes Cancelled with a sentinel, Par sees Some(e) in failures).
                        // Actually: leaf becomes an Cancelled-like state that won't trigger propagation.
                        // Fill the Par's failures[idx].
                        if let Some(&parent) = self.parent.get(&h) {
                            if let InternalFrame::Par { children, failures, .. } =
                                self.get_mut(parent)
                            {
                                if let Some(idx) = children.iter().position(|c| *c == h) {
                                    failures[idx] = Some(e);
                                }
                            }
                        }
                        // Mark leaf as Cancelled with a benign reason so it stops being live.
                        *self.get_mut(h) = InternalFrame::Cancelled {
                            node,
                            reason: CancelReason::SiblingFailed,
                        };
                    }
                    _ => {
                        // FailFast or no enclosing Par — leaf transitions
                        // to Failed and the next step() propagates.
                        *self.get_mut(h) = InternalFrame::Failed { node, path, error: e };
                    }
                }
                Ok(())
            }
        }
    }

    fn pending(&self) -> &[Pending] {
        &self.pending
    }

    fn take_cancellations(&mut self) -> Vec<SuspensionId> {
        std::mem::take(&mut self.cancellations)
    }

    fn frame_tree(
        &self,
    ) -> &FrameTree<Self::Value, Self::Cursor, Self::Delta, Self::EffectError> {
        // Projection cache: we build lazily but must return a &. Use a
        // Box behind an UnsafeCell via a helper — for now, build on
        // demand and leak (test-only). Proper caching is a follow-up.
        //
        // NOTE: this method is intentionally simple; hosts that need
        // the tree are debuggers that call it infrequently. If it
        // becomes hot, add a `RefCell<Option<FrameTree>>` cache
        // invalidated on every mutation.
        let tree = self.project_tree(self.root);
        // Leak the projected tree so we can return a `&`.
        // (Test-focused API; not called on hot paths.)
        Box::leak(Box::new(tree))
    }

    fn is_done(&self) -> bool {
        self.done && self.root_value.is_some()
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Reducer, ReducerCollectAll};

    // ---- A minimal in-core test DSL --------------------------------

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum V {
        Unit,
        S(String),
        List(Vec<V>),
    }

    #[derive(Debug, Clone, thiserror::Error)]
    #[error("test-err: {0}")]
    struct EE(String);

    #[derive(Debug, Clone)]
    enum N {
        Seq(Vec<N>),
        Par {
            children: Vec<N>,
            policy: JoinPolicy,
            reducer: &'static str,
        },
        Scope(String, Box<N>),
        Maybe(Option<Box<N>>),
        Call(String),
    }

    /// Assign NodeIds in a fixed DFS pre-order for reproducibility.
    fn assign_ids(node: &N, next: &mut u64) -> (NodeId, Node) {
        let id = NodeId(*next);
        *next += 1;
        let cooked = match node {
            N::Seq(children) => {
                let cs = children.iter().map(|c| assign_ids(c, next)).collect::<Vec<_>>();
                Node::Seq(id, cs)
            }
            N::Par { children, policy, reducer } => {
                let cs = children.iter().map(|c| assign_ids(c, next)).collect::<Vec<_>>();
                Node::Par(id, cs, *policy, ReducerId::from(*reducer))
            }
            N::Scope(label, body) => {
                let b = assign_ids(body, next);
                Node::Scope(id, label.clone(), Box::new(b))
            }
            N::Maybe(body) => {
                let b = body.as_ref().map(|b| Box::new(assign_ids(b, next)));
                Node::Maybe(id, b)
            }
            N::Call(label) => Node::Call(id, label.clone()),
        };
        (id, cooked)
    }

    enum Node {
        Seq(NodeId, Vec<(NodeId, Node)>),
        Par(NodeId, Vec<(NodeId, Node)>, JoinPolicy, ReducerId),
        Scope(NodeId, String, Box<(NodeId, Node)>),
        Maybe(NodeId, Option<Box<(NodeId, Node)>>),
        Call(NodeId, String),
    }

    struct TestAst {
        root_id: NodeId,
        by_id: HashMap<NodeId, NodeKind>,
    }

    impl TestAst {
        fn build(root: N) -> Self {
            let mut next = 1u64;
            let (root_id, cooked) = assign_ids(&root, &mut next);
            let mut by_id = HashMap::new();
            flatten(cooked, &mut by_id);
            Self { root_id, by_id }
        }
    }

    fn flatten(n: Node, out: &mut HashMap<NodeId, NodeKind>) {
        match n {
            Node::Seq(id, children) => {
                let child_ids: Vec<NodeId> = children.iter().map(|(cid, _)| *cid).collect();
                out.insert(id, NodeKind::Seq { children: child_ids });
                for (_, c) in children {
                    flatten(c, out);
                }
            }
            Node::Par(id, children, policy, reducer_id) => {
                let child_ids: Vec<NodeId> = children.iter().map(|(cid, _)| *cid).collect();
                out.insert(
                    id,
                    NodeKind::Par { children: child_ids, policy, reducer_id },
                );
                for (_, c) in children {
                    flatten(c, out);
                }
            }
            Node::Scope(id, label, body) => {
                let (bid, bn) = *body;
                out.insert(id, NodeKind::Scope { label, body: bid });
                flatten(bn, out);
            }
            Node::Maybe(id, body) => {
                let body_id = body.as_ref().map(|b| b.0);
                out.insert(id, NodeKind::Maybe { body: body_id });
                if let Some(b) = body {
                    let (_, bn) = *b;
                    flatten(bn, out);
                }
            }
            Node::Call(id, label) => {
                out.insert(
                    id,
                    NodeKind::Call { label, payload: serde_json::Value::Null },
                );
            }
        }
    }

    impl Ast for TestAst {
        type Value = V;
        type Delta = ();
        type EffectError = EE;

        fn root(&self) -> NodeId {
            self.root_id
        }
        fn node_kind(&self, id: NodeId) -> NodeKind {
            self.by_id.get(&id).cloned().expect("unknown NodeId")
        }
        fn unit_value(&self) -> V {
            V::Unit
        }
    }

    // ---- Reducers --------------------------------------------------

    struct AllOrdered;
    impl Reducer<V, ()> for AllOrdered {
        fn reduce(
            &self,
            slots: &[Option<V>],
            _deltas: &[Option<()>],
            _winners: &[ChildIndex],
        ) -> Result<(V, ()), EngineError> {
            let vs: Vec<V> = slots.iter().map(|s| s.clone().unwrap_or(V::Unit)).collect();
            Ok((V::List(vs), ()))
        }
    }

    struct AnyFirstWinner;
    impl Reducer<V, ()> for AnyFirstWinner {
        fn reduce(
            &self,
            slots: &[Option<V>],
            _deltas: &[Option<()>],
            winners: &[ChildIndex],
        ) -> Result<(V, ()), EngineError> {
            let w = winners.first().copied().unwrap_or(0);
            Ok((slots.get(w).cloned().flatten().unwrap_or(V::Unit), ()))
        }
    }

    struct FirstKOrdered;
    impl Reducer<V, ()> for FirstKOrdered {
        fn reduce(
            &self,
            slots: &[Option<V>],
            _deltas: &[Option<()>],
            winners: &[ChildIndex],
        ) -> Result<(V, ()), EngineError> {
            let mut vs = Vec::new();
            for &i in winners {
                if let Some(v) = slots.get(i).cloned().flatten() {
                    vs.push(v);
                }
            }
            Ok((V::List(vs), ()))
        }
    }

    struct CollectAllResults;
    impl ReducerCollectAll<V, (), EE> for CollectAllResults {
        fn reduce(
            &self,
            slots: &[Option<Result<V, EE>>],
            _deltas: &[Option<()>],
            _winners: &[ChildIndex],
        ) -> Result<(V, ()), EngineError> {
            let mut vs = Vec::new();
            for s in slots {
                match s {
                    Some(Ok(v)) => vs.push(v.clone()),
                    Some(Err(e)) => vs.push(V::S(format!("err:{}", e.0))),
                    None => vs.push(V::Unit),
                }
            }
            Ok((V::List(vs), ()))
        }
    }

    fn default_registry() -> Arc<ReducerRegistry<V, (), EE>> {
        let mut r: ReducerRegistry<V, (), EE> = ReducerRegistry::new();
        r.register_fail_fast("all_ordered", Arc::new(AllOrdered));
        r.register_fail_fast("any_first", Arc::new(AnyFirstWinner));
        r.register_fail_fast("first_k", Arc::new(FirstKOrdered));
        r.register_collect_all("collect_all", Arc::new(CollectAllResults));
        Arc::new(r)
    }

    fn build(root: N) -> Engine<TestAst> {
        Engine::new(TestAst::build(root), default_registry()).expect("build")
    }

    fn all_ff() -> JoinPolicy {
        JoinPolicy { shape: JoinShape::All, fail: FailPolicy::FailFast }
    }
    fn any_ff() -> JoinPolicy {
        JoinPolicy { shape: JoinShape::Any, fail: FailPolicy::FailFast }
    }
    fn first_k_ff(k: usize) -> JoinPolicy {
        JoinPolicy { shape: JoinShape::FirstK(k), fail: FailPolicy::FailFast }
    }
    fn all_ca() -> JoinPolicy {
        JoinPolicy { shape: JoinShape::All, fail: FailPolicy::CollectAll }
    }

    // ---- Actual tests ---------------------------------------------

    #[test]
    fn seq_of_two_calls_completes_in_order() {
        let ast = N::Seq(vec![N::Call("a".into()), N::Call("b".into())]);
        let mut e = build(ast);
        // First step: enter Seq, spawn first Call, produce Pending.
        let out = e.step().unwrap();
        let sid1 = match out {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 1);
                newly_pending[0].id
            }
            _ => panic!("expected Blocked"),
        };
        assert_eq!(e.pending().len(), 1);
        e.resolve(sid1, Ok(V::S("A".into()))).unwrap();
        let out = e.step().unwrap();
        let sid2 = match out {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 1);
                newly_pending[0].id
            }
            _ => panic!("expected Blocked"),
        };
        e.resolve(sid2, Ok(V::S("B".into()))).unwrap();
        let out = e.step().unwrap();
        assert!(matches!(out, StepOutcome::Done(V::S(ref s)) if s == "B"));
    }

    #[test]
    fn empty_seq_completes_with_unit() {
        let ast = N::Seq(vec![]);
        let mut e = build(ast);
        let out = e.step().unwrap();
        assert!(matches!(out, StepOutcome::Done(V::Unit)));
    }

    #[test]
    fn scope_wraps_and_forwards_body_value() {
        let ast = N::Scope("s".into(), Box::new(N::Call("x".into())));
        let mut e = build(ast);
        let out = e.step().unwrap();
        let sid = match out {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!(),
        };
        e.resolve(sid, Ok(V::S("hit".into()))).unwrap();
        let out = e.step().unwrap();
        assert!(matches!(out, StepOutcome::Done(V::S(ref s)) if s == "hit"));
    }

    #[test]
    fn maybe_none_completes_with_unit_immediately() {
        let ast = N::Maybe(None);
        let mut e = build(ast);
        let out = e.step().unwrap();
        assert!(matches!(out, StepOutcome::Done(V::Unit)));
    }

    #[test]
    fn maybe_some_evaluates_body() {
        let ast = N::Maybe(Some(Box::new(N::Call("m".into()))));
        let mut e = build(ast);
        let out = e.step().unwrap();
        let sid = match out {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!(),
        };
        e.resolve(sid, Ok(V::S("m!".into()))).unwrap();
        let out = e.step().unwrap();
        assert!(matches!(out, StepOutcome::Done(V::S(ref s)) if s == "m!"));
    }

    #[test]
    fn par_of_calls_all_ordered_fans_out() {
        let ast = N::Par {
            children: vec![N::Call("a".into()), N::Call("b".into()), N::Call("c".into())],
            policy: all_ff(),
            reducer: "all_ordered",
        };
        let mut e = build(ast);
        let out = e.step().unwrap();
        let sids: Vec<SuspensionId> = match out {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 3, "3 concurrent pending on Par entry");
                newly_pending.iter().map(|p| p.id).collect()
            }
            _ => panic!("expected Blocked"),
        };
        // Resolve out of order.
        e.resolve(sids[1], Ok(V::S("B".into()))).unwrap();
        e.resolve(sids[0], Ok(V::S("A".into()))).unwrap();
        e.resolve(sids[2], Ok(V::S("C".into()))).unwrap();
        let out = e.step().unwrap();
        assert!(matches!(&out, StepOutcome::Done(V::List(vs))
            if vs == &vec![V::S("A".into()), V::S("B".into()), V::S("C".into())]));
    }

    #[test]
    fn par_any_first_winner_cancels_siblings() {
        let ast = N::Par {
            children: vec![N::Call("a".into()), N::Call("b".into())],
            policy: any_ff(),
            reducer: "any_first",
        };
        let mut e = build(ast);
        let out = e.step().unwrap();
        let sids: Vec<SuspensionId> = match out {
            StepOutcome::Blocked { newly_pending } => newly_pending.iter().map(|p| p.id).collect(),
            _ => panic!(),
        };
        e.resolve(sids[0], Ok(V::S("win".into()))).unwrap();
        let out = e.step().unwrap();
        assert!(matches!(&out, StepOutcome::Done(V::S(s)) if s == "win"));
        let cancelled = e.take_cancellations();
        assert_eq!(cancelled.len(), 1, "sibling must be cancelled");
        assert_eq!(cancelled[0], sids[1]);
    }

    #[test]
    fn par_first_k_two_of_three() {
        let ast = N::Par {
            children: vec![
                N::Call("a".into()),
                N::Call("b".into()),
                N::Call("c".into()),
            ],
            policy: first_k_ff(2),
            reducer: "first_k",
        };
        let mut e = build(ast);
        let out = e.step().unwrap();
        let sids: Vec<SuspensionId> = match out {
            StepOutcome::Blocked { newly_pending } => newly_pending.iter().map(|p| p.id).collect(),
            _ => panic!(),
        };
        e.resolve(sids[2], Ok(V::S("C".into()))).unwrap();
        e.resolve(sids[0], Ok(V::S("A".into()))).unwrap();
        let out = e.step().unwrap();
        assert!(matches!(&out, StepOutcome::Done(V::List(vs))
            if vs == &vec![V::S("C".into()), V::S("A".into())]));
        let cancelled = e.take_cancellations();
        assert_eq!(cancelled, vec![sids[1]]);
    }

    #[test]
    fn par_of_seq_fans_out_with_real_kids() {
        // Par { Seq(a1, a2), Seq(b1, b2) }
        let ast = N::Par {
            children: vec![
                N::Seq(vec![N::Call("a1".into()), N::Call("a2".into())]),
                N::Seq(vec![N::Call("b1".into()), N::Call("b2".into())]),
            ],
            policy: all_ff(),
            reducer: "all_ordered",
        };
        let mut e = build(ast);
        // Step: enter Par, spawn 2 Seq children, each Seq spawns its first Call.
        let out = e.step().unwrap();
        let first_wave: Vec<SuspensionId> = match out {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 2, "one Call per Seq");
                newly_pending.iter().map(|p| p.id).collect()
            }
            _ => panic!("expected Blocked with 2 pending"),
        };
        // Resolve both first-wave calls.
        for sid in &first_wave {
            e.resolve(*sid, Ok(V::S("1".into()))).unwrap();
        }
        // Step: each Seq spawns its second Call.
        let out = e.step().unwrap();
        let second_wave: Vec<SuspensionId> = match out {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 2, "second wave");
                newly_pending.iter().map(|p| p.id).collect()
            }
            _ => panic!("expected second-wave Blocked"),
        };
        for sid in &second_wave {
            e.resolve(*sid, Ok(V::S("2".into()))).unwrap();
        }
        let out = e.step().unwrap();
        // Both Seqs drain to "2" (last-value semantics); Par folds them.
        assert!(matches!(&out, StepOutcome::Done(V::List(vs))
            if vs == &vec![V::S("2".into()), V::S("2".into())]));
    }

    #[test]
    fn nested_par_of_par() {
        // Par { Par { a, b }, Par { c, d } } with all_ordered
        let ast = N::Par {
            children: vec![
                N::Par {
                    children: vec![N::Call("a".into()), N::Call("b".into())],
                    policy: all_ff(),
                    reducer: "all_ordered",
                },
                N::Par {
                    children: vec![N::Call("c".into()), N::Call("d".into())],
                    policy: all_ff(),
                    reducer: "all_ordered",
                },
            ],
            policy: all_ff(),
            reducer: "all_ordered",
        };
        let mut e = build(ast);
        let out = e.step().unwrap();
        let sids: Vec<SuspensionId> = match out {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 4, "4 leaves at once");
                newly_pending.iter().map(|p| p.id).collect()
            }
            _ => panic!(),
        };
        for (i, sid) in sids.iter().enumerate() {
            e.resolve(*sid, Ok(V::S(format!("v{i}")))).unwrap();
        }
        let out = e.step().unwrap();
        assert!(matches!(&out, StepOutcome::Done(V::List(vs)) if vs.len() == 2));
    }

    #[test]
    fn failfast_propagates_and_cancels_siblings() {
        let ast = N::Par {
            children: vec![N::Call("ok".into()), N::Call("bad".into()), N::Call("ok2".into())],
            policy: all_ff(),
            reducer: "all_ordered",
        };
        let mut e = build(ast);
        let out = e.step().unwrap();
        let sids: Vec<SuspensionId> = match out {
            StepOutcome::Blocked { newly_pending } => newly_pending.iter().map(|p| p.id).collect(),
            _ => panic!(),
        };
        e.resolve(sids[1], Err(EE("boom".into()))).unwrap();
        let err = e.step().unwrap_err();
        assert!(matches!(err, ExecError::Effect(EE(ref s)) if s == "boom"));
        let cancelled = e.take_cancellations();
        assert_eq!(cancelled.len(), 2, "both siblings cancelled");
        assert!(cancelled.contains(&sids[0]));
        assert!(cancelled.contains(&sids[2]));
    }

    #[test]
    fn collect_all_swallows_errors_and_reduces() {
        let ast = N::Par {
            children: vec![N::Call("ok".into()), N::Call("bad".into())],
            policy: all_ca(),
            reducer: "collect_all",
        };
        let mut e = build(ast);
        let out = e.step().unwrap();
        let sids: Vec<SuspensionId> = match out {
            StepOutcome::Blocked { newly_pending } => newly_pending.iter().map(|p| p.id).collect(),
            _ => panic!(),
        };
        e.resolve(sids[0], Ok(V::S("A".into()))).unwrap();
        e.resolve(sids[1], Err(EE("nope".into()))).unwrap();
        let out = e.step().unwrap();
        assert!(matches!(&out, StepOutcome::Done(V::List(vs))
            if vs == &vec![V::S("A".into()), V::S("err:nope".into())]));
    }

    #[test]
    fn empty_par_all_folds_immediately() {
        let ast = N::Par {
            children: vec![],
            policy: all_ff(),
            reducer: "all_ordered",
        };
        let mut e = build(ast);
        let out = e.step().unwrap();
        // The reducer identity: empty V::List.
        assert!(matches!(&out, StepOutcome::Done(V::List(vs)) if vs.is_empty()));
    }

    #[test]
    fn any_with_zero_children_is_malformed() {
        let ast = N::Par {
            children: vec![],
            policy: any_ff(),
            reducer: "any_first",
        };
        let err = Engine::new(TestAst::build(ast), default_registry()).unwrap_err();
        assert!(matches!(err, EngineError::Malformed { .. }));
    }

    #[test]
    fn first_k_with_k_greater_than_len_is_malformed() {
        let ast = N::Par {
            children: vec![N::Call("a".into())],
            policy: first_k_ff(2),
            reducer: "first_k",
        };
        let err = Engine::new(TestAst::build(ast), default_registry()).unwrap_err();
        assert!(matches!(err, EngineError::Malformed { .. }));
    }

    #[test]
    fn first_k_zero_folds_immediately() {
        let ast = N::Par {
            children: vec![N::Call("a".into())],
            policy: first_k_ff(0),
            reducer: "first_k",
        };
        let mut e = build(ast);
        let out = e.step().unwrap();
        assert!(matches!(&out, StepOutcome::Done(V::List(vs)) if vs.is_empty()));
        // Also, the child should be cancelled since Par fired before it.
        let cancelled = e.take_cancellations();
        assert_eq!(cancelled.len(), 1);
    }

    #[test]
    fn resolve_unknown_id_is_unknown_suspension() {
        let ast = N::Call("x".into());
        let mut e = build(ast);
        let _ = e.step().unwrap();
        let err = e.resolve(SuspensionId(9999), Ok(V::Unit)).unwrap_err();
        assert!(matches!(err, ExecError::Engine(EngineError::UnknownSuspension { .. })));
    }

    #[test]
    fn resolve_after_cancel_is_unknown_suspension() {
        let ast = N::Par {
            children: vec![N::Call("a".into()), N::Call("b".into())],
            policy: any_ff(),
            reducer: "any_first",
        };
        let mut e = build(ast);
        let sids: Vec<SuspensionId> = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending.iter().map(|p| p.id).collect(),
            _ => panic!(),
        };
        e.resolve(sids[0], Ok(V::S("win".into()))).unwrap();
        let _ = e.step().unwrap();
        // sids[1] was cancelled — late resolve is unknown.
        let err = e.resolve(sids[1], Ok(V::S("late".into()))).unwrap_err();
        assert!(matches!(err, ExecError::Engine(EngineError::UnknownSuspension { .. })));
    }

    #[test]
    fn take_cancellations_after_err_returns_ids() {
        let ast = N::Par {
            children: vec![N::Call("a".into()), N::Call("b".into())],
            policy: all_ff(),
            reducer: "all_ordered",
        };
        let mut e = build(ast);
        let sids: Vec<SuspensionId> = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending.iter().map(|p| p.id).collect(),
            _ => panic!(),
        };
        e.resolve(sids[0], Err(EE("x".into()))).unwrap();
        let _err = e.step().unwrap_err();
        let cancelled = e.take_cancellations();
        assert_eq!(cancelled, vec![sids[1]]);
    }

    #[test]
    fn deep_seq_of_five_calls() {
        let ast = N::Seq(
            (0..5).map(|i| N::Call(format!("c{i}"))).collect(),
        );
        let mut e = build(ast);
        for i in 0..5 {
            let sid = match e.step().unwrap() {
                StepOutcome::Blocked { newly_pending } => {
                    assert_eq!(newly_pending.len(), 1, "step {i}");
                    newly_pending[0].id
                }
                _ => panic!("step {i}"),
            };
            e.resolve(sid, Ok(V::S(format!("v{i}")))).unwrap();
        }
        let out = e.step().unwrap();
        assert!(matches!(&out, StepOutcome::Done(V::S(s)) if s == "v4"));
    }

    #[test]
    fn scope_of_seq_of_par() {
        let ast = N::Scope(
            "outer".into(),
            Box::new(N::Seq(vec![
                N::Call("prep".into()),
                N::Par {
                    children: vec![N::Call("a".into()), N::Call("b".into())],
                    policy: all_ff(),
                    reducer: "all_ordered",
                },
            ])),
        );
        let mut e = build(ast);
        // prep first.
        let sid = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!(),
        };
        e.resolve(sid, Ok(V::S("p".into()))).unwrap();
        // Par next.
        let sids: Vec<SuspensionId> = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 2);
                newly_pending.iter().map(|p| p.id).collect()
            }
            _ => panic!(),
        };
        for sid in &sids {
            e.resolve(*sid, Ok(V::S("x".into()))).unwrap();
        }
        let out = e.step().unwrap();
        assert!(matches!(&out, StepOutcome::Done(V::List(_))));
    }

    #[test]
    fn seq_wrapping_maybe_none() {
        let ast = N::Seq(vec![N::Call("a".into()), N::Maybe(None), N::Call("b".into())]);
        let mut e = build(ast);
        let sid1 = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!(),
        };
        e.resolve(sid1, Ok(V::S("A".into()))).unwrap();
        // Maybe(None) is immediate; then Call("b") suspends.
        let sid2 = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 1);
                newly_pending[0].id
            }
            _ => panic!(),
        };
        e.resolve(sid2, Ok(V::S("B".into()))).unwrap();
        let out = e.step().unwrap();
        assert!(matches!(&out, StepOutcome::Done(V::S(s)) if s == "B"));
    }

    #[test]
    fn par_first_k_collect_all_becomes_impossible_folds_early() {
        // FirstK(2) with 3 children under CollectAll; 2 fail early —
        // shape completion (2 successes) still possible from 1 remaining
        // is FALSE, so Par folds via CollectAll reducer.
        let ast = N::Par {
            children: vec![
                N::Call("a".into()),
                N::Call("b".into()),
                N::Call("c".into()),
            ],
            policy: JoinPolicy { shape: JoinShape::FirstK(2), fail: FailPolicy::CollectAll },
            reducer: "collect_all",
        };
        let mut e = build(ast);
        let sids: Vec<SuspensionId> = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending.iter().map(|p| p.id).collect(),
            _ => panic!(),
        };
        e.resolve(sids[0], Err(EE("f1".into()))).unwrap();
        e.resolve(sids[1], Err(EE("f2".into()))).unwrap();
        // Now: 0 successes, 2 failures, 1 remaining. Cannot reach k=2 successes.
        // Third child still pending — the fold happens when we step().
        // But third child hasn't failed yet; is it possible? 0 + 1 = 1 < 2 = impossible.
        let out = e.step().unwrap();
        // CollectAll reducer sees [Err, Err, None] and folds.
        assert!(matches!(&out, StepOutcome::Done(V::List(vs)) if vs.len() == 3));
    }

    #[test]
    fn is_done_reports_completion() {
        let ast = N::Call("x".into());
        let mut e = build(ast);
        assert!(!e.is_done());
        let sid = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!(),
        };
        e.resolve(sid, Ok(V::S("done".into()))).unwrap();
        let _ = e.step().unwrap();
        assert!(e.is_done());
    }

    #[test]
    fn frame_tree_root_reflects_par_shape() {
        let ast = N::Par {
            children: vec![N::Call("a".into()), N::Call("b".into())],
            policy: all_ff(),
            reducer: "all_ordered",
        };
        let mut e = build(ast);
        let _ = e.step().unwrap();
        let tree = e.frame_tree();
        assert!(matches!(tree.root, Frame::Par(_)));
        assert_eq!(tree.kids.len(), 2, "kids populated for Par children");
    }

    #[test]
    fn frame_tree_kids_populated_for_par_of_seq() {
        let ast = N::Par {
            children: vec![
                N::Seq(vec![N::Call("a1".into()), N::Call("a2".into())]),
                N::Seq(vec![N::Call("b1".into()), N::Call("b2".into())]),
            ],
            policy: all_ff(),
            reducer: "all_ordered",
        };
        let mut e = build(ast);
        let _ = e.step().unwrap();
        let tree = e.frame_tree();
        assert!(matches!(tree.root, Frame::Par(_)));
        // Two Seq children projected as Frame::Node with their spawned
        // sub-Call as sole kid.
        assert_eq!(tree.kids.len(), 2);
        for kid in &tree.kids {
            assert!(matches!(kid.root, Frame::Node { .. }));
            assert_eq!(kid.kids.len(), 1, "Seq shows current child as sole kid");
            assert!(matches!(kid.kids[0].root, Frame::PendingEffect { .. }));
        }
    }

    #[test]
    fn take_cancellations_drains_once() {
        let ast = N::Par {
            children: vec![N::Call("a".into()), N::Call("b".into())],
            policy: any_ff(),
            reducer: "any_first",
        };
        let mut e = build(ast);
        let sids: Vec<SuspensionId> = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending.iter().map(|p| p.id).collect(),
            _ => panic!(),
        };
        e.resolve(sids[0], Ok(V::S("w".into()))).unwrap();
        let _ = e.step().unwrap();
        let first = e.take_cancellations();
        assert!(!first.is_empty());
        let second = e.take_cancellations();
        assert!(second.is_empty(), "drain: subsequent take is empty");
    }

    #[test]
    fn pending_excludes_cancelled() {
        let ast = N::Par {
            children: vec![N::Call("a".into()), N::Call("b".into())],
            policy: any_ff(),
            reducer: "any_first",
        };
        let mut e = build(ast);
        let sids: Vec<SuspensionId> = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending.iter().map(|p| p.id).collect(),
            _ => panic!(),
        };
        assert_eq!(e.pending().len(), 2);
        e.resolve(sids[0], Ok(V::S("w".into()))).unwrap();
        let _ = e.step().unwrap();
        assert_eq!(e.pending().len(), 0, "cancelled sid removed");
    }

    #[test]
    fn unknown_reducer_id_rejected_at_build() {
        // Fresh registry with no reducer registered.
        let empty: Arc<ReducerRegistry<V, (), EE>> = Arc::new(ReducerRegistry::new());
        let ast = N::Par {
            children: vec![N::Call("a".into())],
            policy: all_ff(),
            reducer: "missing_reducer_id",
        };
        let err = Engine::new(TestAst::build(ast), empty).unwrap_err();
        assert!(matches!(err, EngineError::UnknownReducer { .. }));
    }

    #[test]
    fn deep_nested_scope_and_maybe() {
        // Scope(Maybe(Scope(Call)))
        let ast = N::Scope(
            "outer".into(),
            Box::new(N::Maybe(Some(Box::new(N::Scope(
                "inner".into(),
                Box::new(N::Call("deep".into())),
            ))))),
        );
        let mut e = build(ast);
        let sid = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!(),
        };
        e.resolve(sid, Ok(V::S("v".into()))).unwrap();
        let out = e.step().unwrap();
        assert!(matches!(&out, StepOutcome::Done(V::S(s)) if s == "v"));
    }

    #[test]
    fn seq_wrapping_par_of_seq_end_to_end() {
        let ast = N::Seq(vec![
            N::Call("head".into()),
            N::Par {
                children: vec![
                    N::Seq(vec![N::Call("a1".into()), N::Call("a2".into())]),
                    N::Seq(vec![N::Call("b1".into()), N::Call("b2".into())]),
                ],
                policy: all_ff(),
                reducer: "all_ordered",
            },
            N::Call("tail".into()),
        ]);
        let mut e = build(ast);
        // head
        let sid = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!(),
        };
        e.resolve(sid, Ok(V::S("h".into()))).unwrap();
        // par wave 1
        let w1: Vec<SuspensionId> = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 2);
                newly_pending.iter().map(|p| p.id).collect()
            }
            _ => panic!(),
        };
        for s in &w1 {
            e.resolve(*s, Ok(V::S("1".into()))).unwrap();
        }
        // par wave 2
        let w2: Vec<SuspensionId> = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 2);
                newly_pending.iter().map(|p| p.id).collect()
            }
            _ => panic!(),
        };
        for s in &w2 {
            e.resolve(*s, Ok(V::S("2".into()))).unwrap();
        }
        // tail
        let tsid = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!(),
        };
        e.resolve(tsid, Ok(V::S("t".into()))).unwrap();
        let out = e.step().unwrap();
        assert!(matches!(&out, StepOutcome::Done(V::S(s)) if s == "t"));
    }
}
