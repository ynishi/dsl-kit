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
//! This module is the concrete implementation of the async-join design:
//! the frame arena and spawn schedule, structured fan-out with join
//! policies, cooperative cancellation, and the event stream. The
//! load-bearing contracts live in the item-level docs below
//! ([`Engine::step_with_breakpoints`], [`Stepper::resolve`],
//! [`Engine::events`]).

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use smallvec::SmallVec;

use crate::{
    BreakpointSet, CallFrameId, CancelReason, ChildIndex, CountingSink, EngineError, EnvRef, Event,
    EventSink, FailPolicy, Frame, FrameTree, JoinPolicy, JoinShape, NodeContext, NodeId, ParFrame,
    Path, Pending, ReducerHandle, ReducerId, ReducerRegistry, StepOutcome, Stepper, SuspendReason,
    SuspensionId,
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
    /// Per-frame cursor surfaced through the projected [`FrameTree`].
    /// Use `()` when the DSL has no per-node cursor state.
    type Cursor: Clone + Debug + Default;

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

/// One deferred spawn scheduled on [`Engine::spawn_stack`].
///
/// The stack is the engine's single spawn schedule: every frame other
/// than lazily-spawned `Seq` children enters the arena by being popped
/// from it, one frame per `step_once` pass. LIFO order with children
/// pushed in reverse preserves the original recursive DFS pre-order,
/// while giving the breakpoint peek a uniform halt point before *any*
/// spawn — `Par` kids, `Scope` / `Maybe` bodies, and the root itself.
struct SpawnEntry {
    node: NodeId,
    path: Path,
    parent: SpawnParent,
}

/// Where a popped [`SpawnEntry`] wires its freshly-allocated frame.
enum SpawnParent {
    /// The entry is the root frame; store the handle on `Engine::root`.
    Root,
    /// Append to the parent `Par`'s `children` (declaration order is
    /// preserved because kids are pushed reversed and popped LIFO).
    ParChild(FrameHandle),
    /// Fill the parent `Scope`'s `body` slot.
    ScopeBody(FrameHandle),
    /// Fill the parent `Maybe`'s `body` slot.
    MaybeBody(FrameHandle),
}

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
    /// A `Scope` in progress. `body` is `None` while the body's spawn
    /// is still queued on the spawn stack (transient, at most until the
    /// enclosing `step` call drains the stack).
    Scope {
        node: NodeId,
        path: Path,
        label: String,
        body: Option<FrameHandle>,
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
        /// Call-frame identifier allocated by the engine when this leaf
        /// was spawned. Reused so `FrameLeave` cites the same id as the
        /// earlier `FrameEnter`.
        frame: CallFrameId,
    },
    /// A completed leaf whose value is waiting to be consumed by its
    /// parent.
    Value { node: NodeId, value: A::Value },
    /// A leaf whose `resolve` returned `Err(_)`. The next `step()`
    /// applies the enclosing `Par.policy.fail`.
    Failed {
        node: NodeId,
        path: Path,
        error: A::EffectError,
    },
    /// A sub-tree the engine cancelled. Retained so `frame_tree()` can
    /// render it.
    Cancelled { node: NodeId, reason: CancelReason },
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

/// Lazy [`FrameTree`] projection slot (see the field docs on
/// `Engine::frame_tree_cache`).
type FrameTreeCache<A> = std::sync::OnceLock<
    Box<
        FrameTree<
            <A as Ast>::Value,
            <A as Ast>::Cursor,
            <A as Ast>::Delta,
            <A as Ast>::EffectError,
        >,
    >,
>;

/// The engine: takes ownership of an [`Ast`] + a [`ReducerRegistry`]
/// and drives the interpretation via the [`Stepper`] trait.
pub struct Engine<A: Ast> {
    ast: A,
    registry: Arc<ReducerRegistry<A::Value, A::Delta, A::EffectError>>,

    frames: Vec<Option<InternalFrame<A>>>,
    /// Root frame handle. `None` until the first `step` pops the root's
    /// [`SpawnEntry`] off the spawn stack (the root is scheduled, not
    /// spawned, at construction — so even root entry is breakpointable).
    root: Option<FrameHandle>,
    parent: HashMap<FrameHandle, FrameHandle>,

    /// The single spawn schedule (see [`SpawnEntry`]). Drained one
    /// entry per `step_once` pass, before join-firing and value
    /// propagation. Empty whenever the engine is blocked on the host —
    /// except while halted on a breakpoint, where the remaining spawns
    /// stay queued and the frame tree faithfully shows the partially
    /// spawned state.
    spawn_stack: Vec<SpawnEntry>,

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

    /// Silent event counter used as the engine's default observation
    /// channel (design doc §5.1). Exposed through [`Engine::events`] so
    /// hosts can produce a lightweight activity summary without wiring
    /// a custom sink.
    events: CountingSink,

    /// Synthetic pending id emitted by [`Engine::step_with_breakpoints`]
    /// when the peek-ahead saw a breakpoint match. The next call to
    /// `step_with_breakpoints` drops the marker before delegating to
    /// [`Engine::step`] so the previously-halted spawn proceeds.
    pending_bp: Option<SuspensionId>,

    /// Monotonic counter for allocating [`CallFrameId`]s to `Call`
    /// leaves as they enter (spawn) their call scope. The same id is
    /// cited by the paired `FrameLeave` event when the leaf's `Pending`
    /// transitions to `Value` / `Failed` / `Cancelled`.
    next_call_frame: u64,

    /// Lazy cache for [`Stepper::frame_tree`]. Populated on the first
    /// read after a mutation, cleared whenever `step` / `resolve` is
    /// about to mutate the internal frames. `OnceLock` gives us safe
    /// interior-mutability for the `&self` read path while keeping
    /// `Engine` `Send + Sync` (required by `DslHost`).
    frame_tree_cache: FrameTreeCache<A>,

    /// Optional user-attached sink that receives every event alongside
    /// the built-in [`CountingSink`]. Set via [`Engine::attach_sink`];
    /// unset at construction so events emitted while `Engine::new`
    /// spawns the root frame don't leak into user-visible history.
    external_sink: Option<Box<dyn EventSink + Send + Sync>>,
}

impl<A: Ast> Engine<A> {
    /// Build a fresh engine anchored at `ast.root()` with the supplied
    /// reducer registry.
    ///
    /// The whole AST is validated eagerly (design-doc §3.5 table plus
    /// reducer-id resolution) via a pure `node_kind` pre-walk, so a
    /// malformed node anywhere in the tree fails here rather than at
    /// the step that would have spawned it. No frame is spawned and no
    /// event is emitted at construction: the root itself is only
    /// *scheduled* on the spawn stack and enters the arena on the first
    /// `step` — construction is not part of the walk (design doc §5.1).
    pub fn new(
        ast: A,
        registry: Arc<ReducerRegistry<A::Value, A::Delta, A::EffectError>>,
    ) -> Result<Self, EngineError> {
        Self::validate_ast(&ast, &registry)?;
        let root_id = ast.root();
        let root_path = Path::root().push(root_id);
        Ok(Self {
            ast,
            registry,
            frames: Vec::new(),
            root: None,
            parent: HashMap::new(),
            spawn_stack: vec![SpawnEntry {
                node: root_id,
                path: root_path,
                parent: SpawnParent::Root,
            }],
            pending: Vec::new(),
            newly_pending: Vec::new(),
            sid_to_frame: HashMap::new(),
            cancellations: Vec::new(),
            next_sid: 1,
            done: false,
            root_value: None,
            events: CountingSink::default(),
            pending_bp: None,
            next_call_frame: 1,
            frame_tree_cache: std::sync::OnceLock::new(),
            external_sink: None,
        })
    }

    /// Pure structural validation of the whole AST: `Par` shape rules
    /// plus reducer-id resolution, without spawning anything. Shared
    /// node ids are validated once (`seen` guard also keeps a cyclic
    /// `node_kind` graph from looping the walk).
    fn validate_ast(
        ast: &A,
        registry: &ReducerRegistry<A::Value, A::Delta, A::EffectError>,
    ) -> Result<(), EngineError> {
        let mut stack = vec![ast.root()];
        let mut seen = std::collections::HashSet::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            match ast.node_kind(id) {
                NodeKind::Seq { children } => stack.extend(children),
                NodeKind::Par {
                    children,
                    policy,
                    reducer_id,
                } => {
                    Self::validate_par(id, &children, policy)?;
                    registry.resolve(&reducer_id, policy.fail)?;
                    stack.extend(children);
                }
                NodeKind::Scope { body, .. } => stack.push(body),
                NodeKind::Maybe { body } => {
                    if let Some(b) = body {
                        stack.push(b);
                    }
                }
                NodeKind::Call { .. } => {}
            }
        }
        Ok(())
    }

    // ---- Public projections (debugger convenience, not on Stepper) --

    /// The AST root node id this engine is anchored on. Stable across
    /// the whole interpretation; available before the first step.
    pub fn root_node(&self) -> NodeId {
        self.ast.root()
    }

    /// Snapshot of the engine's event counter. Exposed for hosts that
    /// want to surface a lightweight activity histogram (see
    /// [`CountingSink::summarise`]).
    pub fn events(&self) -> &CountingSink {
        &self.events
    }

    /// Convenience wrapper around [`CountingSink::summarise`].
    pub fn event_summary(&self) -> String {
        self.events.summarise()
    }

    /// Attaches an external sink that receives every subsequent event
    /// alongside the built-in [`CountingSink`].
    ///
    /// Multiple attachments are not supported: calling `attach_sink`
    /// twice replaces the earlier sink and returns it via the return
    /// value so the caller can drain any leftover state. Use a
    /// combinator sink (e.g. one that internally fans out) when more
    /// than one downstream observer is needed.
    ///
    /// Construction emits nothing (the root is scheduled, not spawned,
    /// in `Engine::new`), so a sink attached right after `new` observes
    /// the complete walk-time event stream from the first `step` on.
    pub fn attach_sink(
        &mut self,
        sink: Box<dyn EventSink + Send + Sync>,
    ) -> Option<Box<dyn EventSink + Send + Sync>> {
        self.external_sink.replace(sink)
    }

    /// Removes the currently attached external sink (if any) and
    /// returns it to the caller.
    pub fn detach_sink(&mut self) -> Option<Box<dyn EventSink + Send + Sync>> {
        self.external_sink.take()
    }

    /// Emits `event` to the built-in [`CountingSink`] and — if
    /// present — to the user-attached external sink.
    fn emit_event(&mut self, event: &Event) {
        // NOTE: call the CountingSink's `emit` directly on the field to
        // avoid an infinite recursion into `emit_event` itself.
        <CountingSink as EventSink>::emit(&mut self.events, event);
        if let Some(sink) = self.external_sink.as_deref_mut() {
            sink.emit(event);
        }
    }

    /// Path of the currently-active leaf (a `Pending`, or the next node
    /// the engine would spawn). Returns `None` when the interpretation
    /// has completed or nothing is live.
    pub fn current_path(&self) -> Option<Path> {
        if let Some(p) = self.pending.first() {
            return Some(p.at.path.clone());
        }
        if self.done {
            return None;
        }
        self.peek_next_spawn().map(|ctx| ctx.path)
    }

    /// Depth of [`Self::current_path`] (`0` when the interpretation is
    /// idle or complete).
    pub fn depth(&self) -> usize {
        self.current_path().map(|p| p.depth()).unwrap_or(0)
    }

    /// If the engine is currently blocked on a single `Call` suspension
    /// (the common non-fan-out case), returns `(sid, node_id, label)`.
    /// Returns `None` when there are zero or more than one pending
    /// suspensions, or when the sole pending is a non-`Call` reason
    /// (breakpoint / cooperative / user).
    pub fn suspended_call(&self) -> Option<(SuspensionId, NodeId, &str)> {
        if self.pending.len() != 1 {
            return None;
        }
        let p = &self.pending[0];
        match &p.reason {
            SuspendReason::Call { spec } => Some((p.id, p.at.node, spec.label.as_str())),
            _ => None,
        }
    }

    // ---- Breakpoint-aware stepping ---------------------------------

    /// Same as [`Stepper::step`] but yields a synthetic
    /// `Pending { reason: Breakpoint }` before spawning a frame whose
    /// context matches any entry in `bps`.
    ///
    /// Every spawn passes through the same schedule (the spawn stack
    /// for `Par` kids / `Scope` and `Maybe` bodies / the root, plus the
    /// lazy `Seq`-child site), and the check runs between every
    /// micro-pass of the internal step loop — so a breakpoint halts
    /// **before the matched frame spawns**, at any depth, including
    /// under a `Par` cascade and on the root itself. While halted, the
    /// frame tree and `pending()` faithfully expose the partially
    /// spawned state; nothing is faked or rolled back.
    ///
    /// The yield is one-shot: the next call to `step_with_breakpoints`
    /// drops the marker, performs the halted spawn unchecked, and then
    /// resumes checking (so a subsequent matching site halts again).
    /// Rarely, when an unrelated `Par` join fires between the resume
    /// and the halted spawn, the same context can report a second
    /// consecutive hit; each resume still makes progress.
    pub fn step_with_breakpoints(
        &mut self,
        bps: &BreakpointSet,
    ) -> Result<StepOutcome<A::Value>, ExecError<A::EffectError>> {
        // Resume: previous call yielded on BP — drop the synthetic
        // pending and let the first micro-pass run unchecked so the
        // halted spawn actually happens.
        let resumed = if let Some(sid) = self.pending_bp.take() {
            self.pending.retain(|p| p.id != sid);
            self.newly_pending.retain(|p| p.id != sid);
            true
        } else {
            false
        };
        self.step_inner(Some(bps), resumed)
    }

    /// Shared body of [`Stepper::step`] and
    /// [`Self::step_with_breakpoints`]: run micro-passes
    /// ([`Self::step_once`]) until the interpretation blocks or
    /// completes. When `bps` is supplied, peek the next spawn site
    /// between passes and yield a synthetic breakpoint suspension
    /// before a matching frame spawns. `skip_first_check` suppresses
    /// exactly one peek so a just-resumed halt can perform its spawn
    /// instead of re-halting on the same context.
    fn step_inner(
        &mut self,
        bps: Option<&BreakpointSet>,
        mut skip_first_check: bool,
    ) -> Result<StepOutcome<A::Value>, ExecError<A::EffectError>> {
        // Any successful step may mutate the internal frames. Drop the
        // cached projection so the next `frame_tree(&self)` rebuilds.
        self.frame_tree_cache.take();
        if self.done {
            if let Some(v) = self.root_value.clone() {
                return Ok(StepOutcome::Done(v));
            }
            // Terminal failure state; caller must treat prior Err as terminal.
            return Ok(StepOutcome::Blocked {
                newly_pending: SmallVec::new(),
            });
        }

        // Root-already-Value fast path — a `Call` root whose leaf was
        // resolved before this call ran needs to be promoted to Done
        // here because no interior state machine will fire.
        if let Some(root) = self.root {
            if let InternalFrame::Value { value, .. } = self.get(root) {
                let v = value.clone();
                self.done = true;
                self.root_value = Some(v.clone());
                self.newly_pending.clear();
                return Ok(StepOutcome::Done(v));
            }
        }

        loop {
            if let Some(bps) = bps {
                if !skip_first_check && !bps.is_empty() {
                    if let Some(ctx) = self.peek_next_spawn() {
                        if !bps.matches(&ctx).is_empty() {
                            let sid = SuspensionId(self.next_sid);
                            self.next_sid += 1;
                            let pending = Pending {
                                id: sid,
                                reason: SuspendReason::Breakpoint,
                                at: ctx.clone(),
                            };
                            self.pending.push(pending.clone());
                            self.pending_bp = Some(sid);
                            self.emit_event(&Event::Suspend {
                                at: ctx,
                                reason: SuspendReason::Breakpoint,
                            });
                            // Report suspensions spawned earlier in this
                            // call alongside the synthetic one.
                            let mut newly: SmallVec<[Pending; 1]> =
                                std::mem::take(&mut self.newly_pending)
                                    .into_iter()
                                    .collect();
                            newly.push(pending);
                            return Ok(StepOutcome::Blocked {
                                newly_pending: newly,
                            });
                        }
                    }
                }
                skip_first_check = false;
            }
            match self.step_once()? {
                StepStep::Ready => {
                    // Check root — extract first, then mutate.
                    let root_value = match self.root {
                        Some(root) => {
                            if let InternalFrame::Value { value, .. } = self.get(root) {
                                Some(value.clone())
                            } else {
                                None
                            }
                        }
                        None => None,
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
                    let newly: SmallVec<[Pending; 1]> = std::mem::take(&mut self.newly_pending)
                        .into_iter()
                        .collect();
                    return Ok(StepOutcome::Blocked {
                        newly_pending: newly,
                    });
                }
                StepStep::Done => {
                    if let Some(v) = self.root_value.clone() {
                        return Ok(StepOutcome::Done(v));
                    }
                    return Ok(StepOutcome::Blocked {
                        newly_pending: SmallVec::new(),
                    });
                }
                StepStep::_Phantom(_) => unreachable!(),
            }
        }
    }

    /// Loop [`Self::step_with_breakpoints`] until it returns anything
    /// other than [`StepOutcome::Ready`].
    pub fn run_to_yield_with_breakpoints(
        &mut self,
        bps: &BreakpointSet,
    ) -> Result<StepOutcome<A::Value>, ExecError<A::EffectError>> {
        loop {
            match self.step_with_breakpoints(bps)? {
                StepOutcome::Ready => continue,
                other => return Ok(other),
            }
        }
    }

    /// The node context that the next micro-pass would spawn a frame
    /// at, if any. The spawn stack top dominates (it is drained before
    /// anything else advances); with the stack empty, fall back to the
    /// lazy `Seq`-child peek. Used by the breakpoint check to yield a
    /// synthetic suspension before the spawn happens.
    fn peek_next_spawn(&self) -> Option<NodeContext> {
        if let Some(entry) = self.spawn_stack.last() {
            return Some(Self::synthetic_ctx(entry.node, entry.path.clone()));
        }
        self.root.and_then(|r| self.peek_at(r))
    }

    fn peek_at(&self, h: FrameHandle) -> Option<NodeContext> {
        match self.get(h) {
            InternalFrame::Seq {
                current,
                children,
                next_child,
                path,
                ..
            } => {
                if let Some(c) = current {
                    match self.get(*c) {
                        InternalFrame::Value { .. } => {
                            // Seq about to advance — peek at next child.
                            children.get(*next_child).map(|next_id| {
                                let child_path = path.push(*next_id);
                                Self::synthetic_ctx(*next_id, child_path)
                            })
                        }
                        InternalFrame::Cancelled { .. } | InternalFrame::Failed { .. } => None,
                        _ => self.peek_at(*c),
                    }
                } else {
                    children.get(*next_child).map(|next_id| {
                        let child_path = path.push(*next_id);
                        Self::synthetic_ctx(*next_id, child_path)
                    })
                }
            }
            InternalFrame::Par { children, .. } => children.iter().find_map(|c| self.peek_at(*c)),
            InternalFrame::Scope { body: Some(b), .. } => self.peek_at(*b),
            InternalFrame::Maybe { body: Some(b), .. } => self.peek_at(*b),
            InternalFrame::Pending { .. }
            | InternalFrame::Value { .. }
            | InternalFrame::Failed { .. }
            | InternalFrame::Cancelled { .. }
            | InternalFrame::Scope { body: None, .. }
            | InternalFrame::Maybe { body: None, .. } => None,
        }
    }

    fn synthetic_ctx(node: NodeId, path: Path) -> NodeContext {
        let depth = path.depth() as u32;
        NodeContext {
            node,
            path,
            frame: None,
            depth,
            iteration: None,
        }
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

    /// Pop one entry off the spawn stack, spawn its frame, and wire it
    /// to its parent slot. One call per `step_once` pass — this is the
    /// uniform halt granularity the breakpoint peek rides on.
    fn drain_spawn(&mut self, entry: SpawnEntry) -> Result<(), EngineError> {
        let h = self.spawn_one(entry.node, entry.path)?;
        match entry.parent {
            SpawnParent::Root => {
                self.root = Some(h);
            }
            SpawnParent::ParChild(p) => {
                self.set_parent(h, p);
                if let InternalFrame::Par { children, .. } = self.get_mut(p) {
                    children.push(h);
                }
            }
            SpawnParent::ScopeBody(p) => {
                self.set_parent(h, p);
                if let InternalFrame::Scope { body, .. } = self.get_mut(p) {
                    *body = Some(h);
                }
            }
            SpawnParent::MaybeBody(p) => {
                self.set_parent(h, p);
                if let InternalFrame::Maybe { body, .. } = self.get_mut(p) {
                    *body = Some(h);
                }
            }
        }
        Ok(())
    }

    /// Spawn an arena frame for `node_id` alone. Composite kinds do not
    /// cascade: their children / bodies are pushed onto the spawn stack
    /// (in reverse, preserving DFS pre-order across pops) and enter the
    /// arena on subsequent passes. Validates `Par` shape.
    fn spawn_one(&mut self, node_id: NodeId, path: Path) -> Result<FrameHandle, EngineError> {
        self.emit_event(&Event::VisitPre {
            at: Self::synthetic_ctx(node_id, path.clone()),
        });
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
            NodeKind::Par {
                children,
                policy,
                reducer_id,
            } => {
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
                // Schedule every child; reverse push keeps pop order in
                // declaration order, so `children` fills positionally
                // aligned with `slots`.
                for &child_id in children.iter().rev() {
                    self.spawn_stack.push(SpawnEntry {
                        node: child_id,
                        path: path.push(child_id),
                        parent: SpawnParent::ParChild(par_handle),
                    });
                }
                Ok(par_handle)
            }
            NodeKind::Scope { label, body } => {
                let scope_handle = self.allocate(InternalFrame::Scope {
                    node: node_id,
                    path: path.clone(),
                    label,
                    body: None,
                    body_value: None,
                });
                self.spawn_stack.push(SpawnEntry {
                    node: body,
                    path: path.push(body),
                    parent: SpawnParent::ScopeBody(scope_handle),
                });
                Ok(scope_handle)
            }
            NodeKind::Maybe { body } => match body {
                Some(body_id) => {
                    let maybe_handle = self.allocate(InternalFrame::Maybe {
                        node: node_id,
                        path: path.clone(),
                        body: None,
                        body_value: None,
                    });
                    self.spawn_stack.push(SpawnEntry {
                        node: body_id,
                        path: path.push(body_id),
                        parent: SpawnParent::MaybeBody(maybe_handle),
                    });
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
                let call_frame = CallFrameId(self.next_call_frame);
                self.next_call_frame += 1;
                let ctx = NodeContext {
                    node: node_id,
                    path: path.clone(),
                    frame: Some(call_frame),
                    depth: path.depth() as u32,
                    iteration: None,
                };
                let spec = crate::CallSpec {
                    label: label.clone(),
                    payload: serde_json::Value::Null,
                };
                self.emit_event(&Event::FrameEnter { at: ctx.clone() });
                self.emit_event(&Event::Suspend {
                    at: ctx.clone(),
                    reason: SuspendReason::Call { spec: spec.clone() },
                });
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
                    frame: call_frame,
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
                    Err(malformed(
                        "Par with JoinShape::Any requires >= 1 child".into(),
                    ))
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
        // Failure preempts everything, including scheduled spawns (any
        // entries left on the stack after a breakpoint halt simply
        // never spawn once the engine is done).
        if let Some(root) = self.root {
            if let Some(failed_leaf) = self.find_failed_leaf(root) {
                return self.propagate_failure(failed_leaf);
            }
        }

        // Then, drain one scheduled spawn. One frame per pass is the
        // uniform granularity the breakpoint peek interposes on.
        if let Some(entry) = self.spawn_stack.pop() {
            self.drain_spawn(entry).map_err(ExecError::Engine)?;
            return Ok(StepStep::Ready);
        }

        // Stack empty ⇒ the root has been spawned (Engine::new always
        // schedules it as the first entry).
        let root = self.root.expect("spawn stack drained before root spawn");

        // Then, try to fold a fireable Par (shape satisfied).
        if let Some(par_handle) = self.find_fireable_par(root) {
            return self.fire_par(par_handle).map_err(ExecError::Engine);
        }

        // Then, look for a Seq whose current child produced a value or
        // is unspawned, or a Scope/Maybe whose body produced a value.
        if self.advance_completed(root).map_err(ExecError::Engine)? {
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
                if let Some(body) = *body {
                    if self.advance_completed(body)? {
                        return Ok(true);
                    }
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
                    current,
                    last_value,
                    ..
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
                node,
                path,
                children,
                next_child,
                ..
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
            // `next_child` here is the index the child about to spawn
            // occupies (0-based); increment for IterationTick display.
            self.emit_event(&Event::IterationTick {
                at: Self::synthetic_ctx(node, path.clone()),
            });
            let child = self.spawn_one(next_id, child_path)?;
            self.set_parent(child, h);
            if let InternalFrame::Seq {
                current,
                next_child,
                ..
            } = self.get_mut(h)
            {
                *current = Some(child);
                *next_child += 1;
            }
            Ok(true)
        } else {
            // No more children — promote Seq to a Value.
            let last = if let InternalFrame::Seq {
                last_value, path, ..
            } = self.get(h)
            {
                (last_value.clone(), path.clone())
            } else {
                unreachable!()
            };
            let (last_value, path) = last;
            let value = last_value.unwrap_or_else(|| self.ast.unit_value());
            let promoted = InternalFrame::Value {
                node,
                value: value.clone(),
            };
            *self.get_mut(h) = promoted;
            self.emit_event(&Event::VisitPost {
                at: Self::synthetic_ctx(node, path),
            });
            self.notify_par_slot(h, value);
            let _ = (next_child, children_len);
            Ok(true)
        }
    }

    fn try_scope_promote(&mut self, h: FrameHandle) -> Result<bool, EngineError> {
        let (node, body, path) = if let InternalFrame::Scope {
            node, body, path, ..
        } = self.get(h)
        {
            (*node, *body, path.clone())
        } else {
            unreachable!()
        };
        let Some(body) = body else {
            // Body spawn still queued on the spawn stack.
            return Ok(false);
        };
        if let InternalFrame::Value { value, .. } = self.get(body) {
            let value = value.clone();
            self.vacate(body);
            *self.get_mut(h) = InternalFrame::Value {
                node,
                value: value.clone(),
            };
            self.emit_event(&Event::VisitPost {
                at: Self::synthetic_ctx(node, path),
            });
            self.notify_par_slot(h, value);
            return Ok(true);
        }
        Ok(false)
    }

    fn try_maybe_promote(&mut self, h: FrameHandle) -> Result<bool, EngineError> {
        let (node, body, path) = if let InternalFrame::Maybe {
            node, body, path, ..
        } = self.get(h)
        {
            (*node, *body, path.clone())
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
        } else if let InternalFrame::Maybe {
            body_value: Some(v),
            ..
        } = self.get(h)
        {
            Some(v.clone())
        } else {
            None
        };
        if let Some(v) = promote_value {
            if let Some(body) = body {
                self.vacate(body);
            }
            *self.get_mut(h) = InternalFrame::Value {
                node,
                value: v.clone(),
            };
            self.emit_event(&Event::VisitPost {
                at: Self::synthetic_ctx(node, path),
            });
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
                // A Par is not fireable until every child frame has
                // spawned (`children` fills up to `slots.len()` as the
                // spawn stack drains) — guards against a premature fold
                // while the cascade is halted on a breakpoint.
                if !*joined
                    && children.len() == slots.len()
                    && self.par_shape_fires(policy, completion_order, slots, failures)
                {
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
            InternalFrame::Seq { current, .. } => current.and_then(|c| self.find_fireable_par(c)),
            InternalFrame::Scope { body, .. } => body.and_then(|b| self.find_fireable_par(b)),
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
        let (node, path, policy, reducer, slots, failures, deltas, completion_order, children) = {
            let f = self.get(h);
            if let InternalFrame::Par {
                node,
                path,
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
                    path.clone(),
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
        let winner_set: std::collections::HashSet<ChildIndex> = winners.iter().copied().collect();
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
        *self.get_mut(h) = InternalFrame::Value {
            node,
            value: value.clone(),
        };
        self.emit_event(&Event::VisitPost {
            at: Self::synthetic_ctx(node, path),
        });
        // If this Par sits inside another Par slot, notify.
        self.notify_par_slot(h, value);

        Ok(StepStep::Ready)
    }

    // ---- FailFast propagation -------------------------------------

    fn find_failed_leaf(&self, h: FrameHandle) -> Option<FrameHandle> {
        match self.get(h) {
            InternalFrame::Failed { .. } => Some(h),
            InternalFrame::Seq { current, .. } => current.and_then(|c| self.find_failed_leaf(c)),
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
            InternalFrame::Scope { body, .. } => body.and_then(|b| self.find_failed_leaf(b)),
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
            InternalFrame::Pending {
                sid,
                node,
                path,
                frame,
                ..
            } => {
                let sid = *sid;
                let node = *node;
                let path = path.clone();
                let call_frame = *frame;
                self.pending.retain(|p| p.id != sid);
                self.newly_pending.retain(|p| p.id != sid);
                self.sid_to_frame.remove(&sid);
                self.cancellations.push(sid);
                self.emit_event(&Event::Cancel { id: sid });
                let mut leave_ctx = Self::synthetic_ctx(node, path);
                leave_ctx.frame = Some(call_frame);
                self.emit_event(&Event::FrameLeave { at: leave_ctx });
                *self.get_mut(h) = InternalFrame::Cancelled { node, reason };
                return;
            }
            InternalFrame::Seq { current, .. } => current.iter().copied().collect::<Vec<_>>(),
            InternalFrame::Par { children, .. } => children.clone(),
            InternalFrame::Scope { body, .. } => body.iter().copied().collect(),
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
    ) -> FrameTree<A::Value, A::Cursor, A::Delta, A::EffectError> {
        match self.get(h) {
            InternalFrame::Seq {
                children,
                current,
                node,
                path,
                ..
            } => {
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
                        cursor: A::Cursor::default(),
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
                FrameTree {
                    root: Frame::Par(par),
                    kids,
                }
            }
            InternalFrame::Scope { label, body, .. } => {
                // `body: None` (spawn still queued) projects as a
                // Scope with no kids — the honest partial-spawn view.
                let kids = body.iter().map(|b| self.project_tree(*b)).collect();
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
                root: Frame::Cancelled {
                    reason: CancelReason::SiblingFailed,
                },
                kids: vec![],
            },
            InternalFrame::Cancelled { reason, .. } => FrameTree {
                root: Frame::Cancelled {
                    reason: reason.clone(),
                },
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
    type Cursor = A::Cursor;
    type Delta = A::Delta;
    type EffectError = A::EffectError;
    type Error = ExecError<A::EffectError>;

    fn step(&mut self) -> Result<StepOutcome<Self::Value>, Self::Error> {
        self.step_inner(None, false)
    }

    fn resolve(
        &mut self,
        id: SuspensionId,
        result: Result<Self::Value, Self::EffectError>,
    ) -> Result<(), Self::Error> {
        // Any resolve transitions the target Pending; invalidate cache.
        self.frame_tree_cache.take();
        let h = match self.sid_to_frame.remove(&id) {
            Some(h) => h,
            None => return Err(ExecError::Engine(EngineError::UnknownSuspension { id })),
        };
        self.pending.retain(|p| p.id != id);

        match result {
            Ok(v) => {
                let (node, path, call_frame) =
                    if let InternalFrame::Pending {
                        node, path, frame, ..
                    } = self.get(h)
                    {
                        (*node, path.clone(), *frame)
                    } else {
                        return Err(ExecError::Engine(EngineError::UnknownSuspension { id }));
                    };
                let mut ctx = Self::synthetic_ctx(node, path);
                ctx.frame = Some(call_frame);
                self.emit_event(&Event::Resume { at: ctx.clone() });
                self.emit_event(&Event::FrameLeave { at: ctx.clone() });
                // Replace leaf with Value.
                *self.get_mut(h) = InternalFrame::Value {
                    node,
                    value: v.clone(),
                };
                self.emit_event(&Event::VisitPost { at: ctx });
                // If parent is a Par, fill the slot.
                if let Some(&parent) = self.parent.get(&h) {
                    if let InternalFrame::Par {
                        children,
                        slots,
                        completion_order,
                        ..
                    } = self.get_mut(parent)
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
                let (node, path, call_frame) =
                    if let InternalFrame::Pending {
                        node, path, frame, ..
                    } = self.get(h)
                    {
                        (*node, path.clone(), *frame)
                    } else {
                        return Err(ExecError::Engine(EngineError::UnknownSuspension { id }));
                    };
                let mut leave_ctx = Self::synthetic_ctx(node, path.clone());
                leave_ctx.frame = Some(call_frame);
                self.emit_event(&Event::FrameLeave { at: leave_ctx });
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
                            if let InternalFrame::Par {
                                children, failures, ..
                            } = self.get_mut(parent)
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
                        *self.get_mut(h) = InternalFrame::Failed {
                            node,
                            path,
                            error: e,
                        };
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

    fn frame_tree(&self) -> &FrameTree<Self::Value, Self::Cursor, Self::Delta, Self::EffectError> {
        // Lazy projection cache backed by `OnceLock`. `step` / `resolve`
        // clear the slot whenever they mutate the internal frames (see
        // `invalidate_frame_tree_cache`), so the first read after a
        // mutation rebuilds the tree and every subsequent read hands
        // back the same allocation until the next mutation.
        self.frame_tree_cache
            .get_or_init(|| {
                Box::new(match self.root {
                    Some(root) => self.project_tree(root),
                    // Root spawn still scheduled (no step yet, or halted
                    // on a root breakpoint): project a bare node frame
                    // for the root id with no kids.
                    None => FrameTree {
                        root: Frame::Node {
                            node: self.ast.root(),
                            env: EnvRef(Arc::new(crate::Env {
                                delta: A::Delta::default(),
                                parent: None,
                            })),
                            cursor: A::Cursor::default(),
                        },
                        kids: Vec::new(),
                    },
                })
            })
            .as_ref()
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
    use crate::{BreakCondition, BufferingSink, Reducer, ReducerCollectAll};

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
                let cs = children
                    .iter()
                    .map(|c| assign_ids(c, next))
                    .collect::<Vec<_>>();
                Node::Seq(id, cs)
            }
            N::Par {
                children,
                policy,
                reducer,
            } => {
                let cs = children
                    .iter()
                    .map(|c| assign_ids(c, next))
                    .collect::<Vec<_>>();
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
                out.insert(
                    id,
                    NodeKind::Seq {
                        children: child_ids,
                    },
                );
                for (_, c) in children {
                    flatten(c, out);
                }
            }
            Node::Par(id, children, policy, reducer_id) => {
                let child_ids: Vec<NodeId> = children.iter().map(|(cid, _)| *cid).collect();
                out.insert(
                    id,
                    NodeKind::Par {
                        children: child_ids,
                        policy,
                        reducer_id,
                    },
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
                    NodeKind::Call {
                        label,
                        payload: serde_json::Value::Null,
                    },
                );
            }
        }
    }

    impl Ast for TestAst {
        type Value = V;
        type Delta = ();
        type EffectError = EE;
        type Cursor = ();

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
        JoinPolicy {
            shape: JoinShape::All,
            fail: FailPolicy::FailFast,
        }
    }
    fn any_ff() -> JoinPolicy {
        JoinPolicy {
            shape: JoinShape::Any,
            fail: FailPolicy::FailFast,
        }
    }
    fn first_k_ff(k: usize) -> JoinPolicy {
        JoinPolicy {
            shape: JoinShape::FirstK(k),
            fail: FailPolicy::FailFast,
        }
    }
    fn all_ca() -> JoinPolicy {
        JoinPolicy {
            shape: JoinShape::All,
            fail: FailPolicy::CollectAll,
        }
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
            children: vec![
                N::Call("a".into()),
                N::Call("b".into()),
                N::Call("c".into()),
            ],
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
            children: vec![
                N::Call("ok".into()),
                N::Call("bad".into()),
                N::Call("ok2".into()),
            ],
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
        assert!(matches!(
            err,
            ExecError::Engine(EngineError::UnknownSuspension { .. })
        ));
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
        assert!(matches!(
            err,
            ExecError::Engine(EngineError::UnknownSuspension { .. })
        ));
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
        let ast = N::Seq((0..5).map(|i| N::Call(format!("c{i}"))).collect());
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
        let ast = N::Seq(vec![
            N::Call("a".into()),
            N::Maybe(None),
            N::Call("b".into()),
        ]);
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
            policy: JoinPolicy {
                shape: JoinShape::FirstK(2),
                fail: FailPolicy::CollectAll,
            },
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

    // ---- FrameEnter / FrameLeave events (F1-a) ---------------------

    #[test]
    fn frame_enter_leave_paired_on_ok_resolve() {
        // Seq wrapping a single Call: the Call is spawned lazily on the
        // first step (Round 17 semantics), so the FrameEnter here is a
        // walk-time emission (not one that gets zeroed by Engine::new).
        let ast = N::Seq(vec![N::Call("solo".into())]);
        let mut e = build(ast);
        assert_eq!(e.events().frame_enter, 0);
        assert_eq!(e.events().frame_leave, 0);
        let sid = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!("expected Blocked"),
        };
        assert_eq!(e.events().frame_enter, 1);
        assert_eq!(e.events().frame_leave, 0);
        e.resolve(sid, Ok(V::S("done".into()))).unwrap();
        assert_eq!(e.events().frame_enter, 1);
        assert_eq!(e.events().frame_leave, 1);
        let _ = e.step().unwrap();
        // Seq has no further child; counters should not spuriously advance.
        assert_eq!(e.events().frame_enter, 1);
        assert_eq!(e.events().frame_leave, 1);
    }

    #[test]
    fn frame_enter_leave_counts_match_across_seq_of_two_calls() {
        // Two sequential Calls → two Enter, two Leave (paired).
        let ast = N::Seq(vec![N::Call("a".into()), N::Call("b".into())]);
        let mut e = build(ast);
        let sid1 = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!(),
        };
        assert_eq!(e.events().frame_enter, 1);
        assert_eq!(e.events().frame_leave, 0);
        e.resolve(sid1, Ok(V::S("A".into()))).unwrap();
        assert_eq!(e.events().frame_leave, 1);
        let sid2 = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!(),
        };
        assert_eq!(e.events().frame_enter, 2);
        e.resolve(sid2, Ok(V::S("B".into()))).unwrap();
        assert_eq!(e.events().frame_leave, 2);
        let _ = e.step().unwrap();
        assert_eq!(e.events().frame_enter, 2);
        assert_eq!(e.events().frame_leave, 2);
    }

    #[test]
    fn frame_leave_fires_on_cancellation_via_par_failfast() {
        // Par of two Calls under FailFast: fail one → sibling gets cancelled.
        // Both Calls must contribute a FrameLeave (one from Err propagation,
        // one from cancel_subtree Pending arm). Wrapped in Seq so Par is
        // spawned lazily at step time (Par eager-spawns its kids, and we
        // want their FrameEnter emissions to land in the walk-time
        // histogram rather than being zeroed by Engine::new).
        let ast = N::Seq(vec![N::Par {
            children: vec![N::Call("l".into()), N::Call("r".into())],
            policy: JoinPolicy {
                shape: JoinShape::All,
                fail: FailPolicy::FailFast,
            },
            reducer: "all_ordered",
        }]);
        let mut e = build(ast);
        let sids: Vec<SuspensionId> = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 2);
                newly_pending.iter().map(|p| p.id).collect()
            }
            _ => panic!(),
        };
        assert_eq!(e.events().frame_enter, 2);
        assert_eq!(e.events().frame_leave, 0);
        // Fail the first one; second one is drained via cancellation.
        e.resolve(sids[0], Err(EE("boom".into()))).unwrap();
        assert_eq!(e.events().frame_leave, 1);
        // Advance so cancel_subtree fires on the sibling.
        let _ = e.step();
        assert_eq!(e.events().frame_leave, 2);
        // Verify cancellation channel drained the sibling sid.
        let cancels = e.take_cancellations();
        assert_eq!(cancels, vec![sids[1]]);
    }

    // ---- F1-e external sink + BufferingSink -----------------------

    #[test]
    fn attached_buffering_sink_records_walk_time_events_only() {
        // Round 17 invariant: emits from `Engine::new`'s root spawn
        // must not leak to a sink attached after construction.
        let ast = N::Seq(vec![N::Call("hello".into())]);
        let mut e = build(ast);
        let buf = std::sync::Arc::new(std::sync::Mutex::new(BufferingSink::new()));
        let handle = std::sync::Arc::clone(&buf);
        struct Shared(std::sync::Arc<std::sync::Mutex<BufferingSink>>);
        impl EventSink for Shared {
            fn emit(&mut self, event: &Event) {
                self.0.lock().unwrap().emit(event);
            }
        }
        let prev = e.attach_sink(Box::new(Shared(handle)));
        assert!(prev.is_none(), "no sink previously attached");
        // First step spawns the Call → emits VisitPre + FrameEnter + Suspend.
        let sid = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!("expected Blocked"),
        };
        let recorded = buf.lock().unwrap().drain();
        assert!(
            recorded
                .iter()
                .any(|e| matches!(e, Event::FrameEnter { .. })),
            "attached sink saw the walk-time FrameEnter"
        );
        assert!(
            recorded.iter().any(|e| matches!(e, Event::Suspend { .. })),
            "attached sink saw the walk-time Suspend"
        );
        assert!(
            recorded.iter().any(|e| matches!(e, Event::VisitPre { .. })),
            "attached sink saw the walk-time VisitPre for the Call child"
        );
        // Resolve → emits Resume + FrameLeave + VisitPost.
        e.resolve(sid, Ok(V::S("done".into()))).unwrap();
        let recorded = buf.lock().unwrap().drain();
        let kinds: Vec<&str> = recorded
            .iter()
            .map(|e| match e {
                Event::Resume { .. } => "resume",
                Event::FrameLeave { .. } => "frame_leave",
                Event::VisitPost { .. } => "visit_post",
                Event::VisitPre { .. } => "visit_pre",
                Event::FrameEnter { .. } => "frame_enter",
                Event::Suspend { .. } => "suspend",
                Event::IterationTick { .. } => "iter",
                Event::Cancel { .. } => "cancel",
            })
            .collect();
        assert!(kinds.contains(&"resume"), "resume observed: {:?}", kinds);
        assert!(
            kinds.contains(&"frame_leave"),
            "frame_leave observed: {:?}",
            kinds
        );
    }

    #[test]
    fn detach_sink_returns_the_installed_sink_and_stops_forwarding() {
        let ast = N::Seq(vec![N::Call("x".into()), N::Call("y".into())]);
        let mut e = build(ast);
        let buf = std::sync::Arc::new(std::sync::Mutex::new(BufferingSink::new()));
        struct Shared(std::sync::Arc<std::sync::Mutex<BufferingSink>>);
        impl EventSink for Shared {
            fn emit(&mut self, event: &Event) {
                self.0.lock().unwrap().emit(event);
            }
        }
        e.attach_sink(Box::new(Shared(std::sync::Arc::clone(&buf))));
        let sid = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!(),
        };
        let after_first_step = buf.lock().unwrap().len();
        assert!(
            after_first_step > 0,
            "sink accumulated events during first step"
        );
        // Detach: subsequent activity must not accrue in the buffer.
        let previous = e.detach_sink();
        assert!(previous.is_some(), "detach returned the installed sink");
        buf.lock().unwrap().clear();
        e.resolve(sid, Ok(V::S("X".into()))).unwrap();
        let after_resolve = buf.lock().unwrap().len();
        assert_eq!(
            after_resolve, 0,
            "no more events forwarded post-detach (found {after_resolve})"
        );
        // Counting sink still advances (built-in observability unaffected).
        assert!(e.events().frame_enter > 0);
        assert!(e.events().frame_leave > 0);
    }

    #[test]
    fn attach_sink_replaces_previous_and_returns_it() {
        let ast = N::Call("root".into());
        let mut e = build(ast);
        let first = Box::new(BufferingSink::new());
        let second = Box::new(BufferingSink::new());
        assert!(e.attach_sink(first).is_none());
        let replaced = e.attach_sink(second);
        assert!(replaced.is_some(), "previous sink returned to caller");
    }

    #[test]
    fn frame_tree_cache_is_reused_and_invalidated() {
        // Address-stability between reads without a mutation between
        // them is what proves the OnceLock cache is populated once and
        // reused. Post-mutation address stability cannot be checked
        // because the allocator may hand back the same slot; instead,
        // we verify the tree *content* refreshes.
        let ast = N::Seq(vec![N::Call("x".into()), N::Call("y".into())]);
        let mut e = build(ast);
        let sid = match e.step().unwrap() {
            StepOutcome::Blocked { newly_pending } => newly_pending[0].id,
            _ => panic!(),
        };
        // Cache hit reuse.
        let p1 = e.frame_tree() as *const _;
        let p2 = e.frame_tree() as *const _;
        assert_eq!(p1, p2, "consecutive reads must reuse the cached box");
        // Snapshot: root Seq has current child Pending (Call awaiting resolve).
        let before = format!("{:?}", e.frame_tree());
        assert!(
            before.contains("Pending"),
            "pre-resolve tree must contain Pending frame"
        );
        // Resolve mutates → cache must be cleared → next read projects fresh.
        e.resolve(sid, Ok(V::S("X".into()))).unwrap();
        let after = format!("{:?}", e.frame_tree());
        assert!(
            !after.contains("Pending") || after != before,
            "post-resolve tree must not still describe the resolved Pending"
        );
        // Second post-mutation read reuses the fresh cached box.
        let p3 = e.frame_tree() as *const _;
        let p4 = e.frame_tree() as *const _;
        assert_eq!(p3, p4, "post-mutation reads settle back into cache-reuse");
    }

    #[test]
    fn frame_enter_zero_immediately_after_new_when_root_is_call() {
        // Construction schedules the root on the spawn stack without
        // spawning it, so no event of any kind is emitted (and no
        // counter-reset hack is needed) even when the root is a Call.
        let e = build(N::Call("root".into()));
        assert_eq!(e.events().visit_pre, 0);
        assert_eq!(e.events().frame_enter, 0);
        assert_eq!(e.events().frame_leave, 0);
        assert_eq!(e.events().suspend, 0);
        // The Call has not spawned yet, so nothing is pending either.
        assert!(e.pending().is_empty(), "no pending before the first step");
    }

    // ---- F1-b: uniform breakpoint scope (spawn-stack schedule) ------

    fn bp_at(node: u64) -> BreakpointSet {
        let mut bps = BreakpointSet::new();
        bps.add(BreakCondition::at_node(NodeId(node)));
        bps
    }

    /// Extract the sole Breakpoint-reason pending, if the outcome is a
    /// breakpoint halt.
    fn bp_halt(outcome: &StepOutcome<V>) -> Option<NodeId> {
        if let StepOutcome::Blocked { newly_pending } = outcome {
            newly_pending
                .iter()
                .find(|p| matches!(p.reason, SuspendReason::Breakpoint))
                .map(|p| p.at.node)
        } else {
            None
        }
    }

    #[test]
    fn breakpoint_on_root_halts_before_any_spawn() {
        // Root is node 1. The BP must fire before the root frame (or
        // anything else) enters the arena.
        let ast = N::Seq(vec![N::Call("a".into())]);
        let mut e = build(ast);
        let bps = bp_at(1);
        let out = e.step_with_breakpoints(&bps).unwrap();
        assert_eq!(bp_halt(&out), Some(NodeId(1)), "halted on root ctx");
        assert_eq!(e.events().visit_pre, 0, "nothing spawned at halt");
        // Frame tree faithfully shows the not-yet-started root.
        let tree = format!("{:?}", e.frame_tree());
        assert!(
            tree.contains("Node"),
            "bare root placeholder projected: {tree}"
        );
        // Resume: the halted spawn proceeds and the run reaches the Call.
        let out = e.step_with_breakpoints(&bps).unwrap();
        match out {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 1);
                assert!(matches!(
                    newly_pending[0].reason,
                    SuspendReason::Call { .. }
                ));
            }
            other => panic!("expected Blocked on the Call, got {other:?}"),
        }
    }

    #[test]
    fn breakpoint_on_par_kid_halts_before_that_spawn() {
        // Seq(1) → Par(2) → Call a(3), Call b(4). BP on kid `b`: the
        // halt must land after `a` spawned (pending) but before `b`
        // enters the arena — the partially spawned state is the truth
        // the debugger sees.
        let ast = N::Seq(vec![N::Par {
            children: vec![N::Call("a".into()), N::Call("b".into())],
            policy: JoinPolicy {
                shape: JoinShape::All,
                fail: FailPolicy::FailFast,
            },
            reducer: "all_ordered",
        }]);
        let mut e = build(ast);
        let bps = bp_at(4);
        let out = e.step_with_breakpoints(&bps).unwrap();
        assert_eq!(bp_halt(&out), Some(NodeId(4)), "halted on kid b's ctx");
        assert_eq!(e.events().frame_enter, 1, "only kid a spawned so far");
        let calls: Vec<_> = e
            .pending()
            .iter()
            .filter(|p| matches!(p.reason, SuspendReason::Call { .. }))
            .collect();
        assert_eq!(calls.len(), 1, "kid a pending, kid b not yet spawned");
        assert_eq!(calls[0].at.node, NodeId(3));
        // Resume → b spawns; the Par is now fully fanned out.
        let out = e.step_with_breakpoints(&bps).unwrap();
        match out {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 1, "b's Call joins the pending set");
                assert_eq!(newly_pending[0].at.node, NodeId(4));
            }
            other => panic!("expected Blocked, got {other:?}"),
        }
        assert_eq!(e.events().frame_enter, 2);
        // Par must not have fired while partially spawned; resolving
        // both kids completes it normally.
        let sids: Vec<SuspensionId> = e.pending().iter().map(|p| p.id).collect();
        for (i, sid) in sids.into_iter().enumerate() {
            e.resolve(sid, Ok(V::S(format!("v{i}")))).unwrap();
        }
        match e.step_with_breakpoints(&bps).unwrap() {
            StepOutcome::Done(V::List(vs)) => assert_eq!(vs.len(), 2),
            other => panic!("expected Done(List), got {other:?}"),
        }
    }

    #[test]
    fn breakpoints_on_consecutive_par_kids_halt_twice() {
        // BP on both kids: continue from the first halt must re-arm and
        // halt again on the second kid.
        let ast = N::Seq(vec![N::Par {
            children: vec![N::Call("a".into()), N::Call("b".into())],
            policy: JoinPolicy {
                shape: JoinShape::All,
                fail: FailPolicy::FailFast,
            },
            reducer: "all_ordered",
        }]);
        let mut e = build(ast);
        let mut bps = BreakpointSet::new();
        bps.add(BreakCondition::at_node(NodeId(3)));
        bps.add(BreakCondition::at_node(NodeId(4)));
        let out = e.step_with_breakpoints(&bps).unwrap();
        assert_eq!(bp_halt(&out), Some(NodeId(3)), "first halt on kid a");
        let out = e.step_with_breakpoints(&bps).unwrap();
        assert_eq!(bp_halt(&out), Some(NodeId(4)), "second halt on kid b");
        let out = e.step_with_breakpoints(&bps).unwrap();
        match out {
            StepOutcome::Blocked { newly_pending } => {
                assert!(
                    newly_pending
                        .iter()
                        .all(|p| matches!(p.reason, SuspendReason::Call { .. }))
                );
            }
            other => panic!("expected Blocked on Calls, got {other:?}"),
        }
    }

    #[test]
    fn breakpoint_on_scope_body_halts_before_body_spawn() {
        // Seq(1) → Scope(2) → Call(3). Before the spawn-stack schedule,
        // Scope bodies spawned atomically with the Scope itself and this
        // site was unreachable for breakpoints.
        let ast = N::Seq(vec![N::Scope("sect".into(), Box::new(N::Call("x".into())))]);
        let mut e = build(ast);
        let bps = bp_at(3);
        let out = e.step_with_breakpoints(&bps).unwrap();
        assert_eq!(bp_halt(&out), Some(NodeId(3)), "halted on scope body ctx");
        // Scope frame exists, body slot still empty.
        let tree = format!("{:?}", e.frame_tree());
        assert!(
            tree.contains("Scope"),
            "scope projected while body pending: {tree}"
        );
        assert_eq!(e.events().frame_enter, 0, "the Call did not spawn yet");
        // Resume runs through to the Call suspension.
        let out = e.step_with_breakpoints(&bps).unwrap();
        match out {
            StepOutcome::Blocked { newly_pending } => {
                assert!(matches!(
                    newly_pending[0].reason,
                    SuspendReason::Call { .. }
                ));
            }
            other => panic!("expected Blocked, got {other:?}"),
        }
    }

    #[test]
    fn resolve_err_while_bp_halted_abandons_unspawned_spawns() {
        // Seq(1) → Par(2, FailFast) → Call a(3), Call b(4). Halt on b's
        // spawn site, then fail a's pending call *while halted*: the
        // failure must propagate on the next step and kid b — still an
        // unspawned entry on the spawn stack — must be abandoned, never
        // spawned and never reported as a cancellation (it has no sid).
        let ast = N::Seq(vec![N::Par {
            children: vec![N::Call("a".into()), N::Call("b".into())],
            policy: JoinPolicy {
                shape: JoinShape::All,
                fail: FailPolicy::FailFast,
            },
            reducer: "all_ordered",
        }]);
        let mut e = build(ast);
        let bps = bp_at(4);
        let out = e.step_with_breakpoints(&bps).unwrap();
        assert_eq!(bp_halt(&out), Some(NodeId(4)), "halted on kid b's ctx");
        assert_eq!(e.events().frame_enter, 1, "only kid a spawned so far");
        let a_sid = e
            .pending()
            .iter()
            .find(|p| matches!(p.reason, SuspendReason::Call { .. }))
            .map(|p| p.id)
            .expect("kid a pending while halted");
        e.resolve(a_sid, Err(EE("boom".into()))).unwrap();
        // Resume: failure propagation runs before any spawn drains.
        let err = e.step_with_breakpoints(&bps).unwrap_err();
        assert!(matches!(err, ExecError::Effect(EE(ref s)) if s == "boom"));
        assert_eq!(e.events().frame_enter, 1, "kid b never spawned");
        assert!(
            e.take_cancellations().is_empty(),
            "abandoned spawn is not a cancellation (no sid was ever issued)"
        );
        assert!(
            e.pending().is_empty(),
            "synthetic BP and a's call both gone"
        );
        // Terminal: further stepping stays blocked with nothing new.
        match e.step_with_breakpoints(&bps).unwrap() {
            StepOutcome::Blocked { newly_pending } => assert!(newly_pending.is_empty()),
            other => panic!("expected terminal Blocked, got {other:?}"),
        }
    }

    // ---- F2: sans-io drive layer -----------------------------------

    use crate::drive::{AsyncEffectResolver, DriveOutcome, EffectResolver, drive, drive_async};

    /// Echoes each Call's label back as its value and records
    /// cancellation notifications.
    struct EchoResolver {
        resolved: Vec<String>,
        cancelled: Vec<SuspensionId>,
    }

    impl EchoResolver {
        fn new() -> Self {
            Self {
                resolved: Vec::new(),
                cancelled: Vec::new(),
            }
        }

        fn answer(&mut self, pending: &Pending) -> Result<V, EE> {
            let label = match &pending.reason {
                SuspendReason::Call { spec } => spec.label.clone(),
                other => panic!("resolver saw non-Call reason {other:?}"),
            };
            self.resolved.push(label.clone());
            Ok(V::S(format!("r:{label}")))
        }
    }

    impl EffectResolver<TestAst> for EchoResolver {
        fn resolve(&mut self, pending: &Pending) -> Result<V, EE> {
            self.answer(pending)
        }

        fn cancelled(&mut self, sid: SuspensionId) {
            self.cancelled.push(sid);
        }
    }

    impl AsyncEffectResolver<TestAst> for EchoResolver {
        async fn resolve(&mut self, pending: &Pending) -> Result<V, EE> {
            self.answer(pending)
        }

        fn cancelled(&mut self, sid: SuspensionId) {
            self.cancelled.push(sid);
        }
    }

    /// Minimal executor for the always-ready futures the tests build —
    /// keeps the crate free of a dev runtime dependency.
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        const VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(std::ptr::null(), &VTABLE),
            |_| {},
            |_| {},
            |_| {},
        );
        let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
        let mut cx = Context::from_waker(&waker);
        let mut fut = std::pin::pin!(fut);
        loop {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    #[test]
    fn drive_runs_seq_of_calls_to_done() {
        let ast = N::Seq(vec![N::Call("a".into()), N::Call("b".into())]);
        let mut e = build(ast);
        let mut r = EchoResolver::new();
        let out = drive(&mut e, &mut r, &BreakpointSet::new()).unwrap();
        match out {
            DriveOutcome::Done(V::S(s)) => assert_eq!(s, "r:b", "last value propagates"),
            other => panic!("expected Done, got {other:?}"),
        }
        assert_eq!(r.resolved, vec!["a".to_string(), "b".to_string()]);
        assert!(r.cancelled.is_empty());
    }

    #[test]
    fn drive_any_join_skips_losers_and_reports_cancellation() {
        // Sequential v1 semantics: the first kid resolves, the join
        // fires, and the loser is *cancelled* — never handed to the
        // resolver.
        let ast = N::Par {
            children: vec![N::Call("a".into()), N::Call("b".into())],
            policy: any_ff(),
            reducer: "any_first",
        };
        let mut e = build(ast);
        let mut r = EchoResolver::new();
        let out = drive(&mut e, &mut r, &BreakpointSet::new()).unwrap();
        assert!(matches!(out, DriveOutcome::Done(V::S(ref s)) if s == "r:a"));
        assert_eq!(r.resolved, vec!["a".to_string()], "loser never resolved");
        assert_eq!(r.cancelled.len(), 1, "loser reported through cancelled()");
    }

    #[test]
    fn drive_breaks_on_breakpoint_and_resumes() {
        // BP on the second Call (node 3): first drive run resolves `a`,
        // halts before `b` spawns, and returns Break; the second run
        // resumes to Done.
        let ast = N::Seq(vec![N::Call("a".into()), N::Call("b".into())]);
        let mut e = build(ast);
        let mut r = EchoResolver::new();
        let bps = bp_at(3);
        let out = drive(&mut e, &mut r, &bps).unwrap();
        match out {
            DriveOutcome::Break { at } => {
                assert!(
                    at.iter()
                        .any(|p| matches!(p.reason, SuspendReason::Breakpoint))
                );
            }
            other => panic!("expected Break, got {other:?}"),
        }
        assert_eq!(r.resolved, vec!["a".to_string()], "halted before b");
        let out = drive(&mut e, &mut r, &bps).unwrap();
        assert!(matches!(out, DriveOutcome::Done(V::S(ref s)) if s == "r:b"));
        assert_eq!(r.resolved, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn drive_async_mirrors_sync_loop() {
        let ast = N::Seq(vec![
            N::Call("a".into()),
            N::Par {
                children: vec![N::Call("l".into()), N::Call("r".into())],
                policy: all_ff(),
                reducer: "all_ordered",
            },
        ]);
        let mut e = build(ast);
        let mut r = EchoResolver::new();
        let out = block_on(drive_async(&mut e, &mut r, &BreakpointSet::new())).unwrap();
        match out {
            DriveOutcome::Done(V::List(vs)) => assert_eq!(vs.len(), 2),
            other => panic!("expected Done(List), got {other:?}"),
        }
        assert_eq!(
            r.resolved,
            vec!["a".to_string(), "l".to_string(), "r".to_string()]
        );
    }

    #[test]
    fn malformed_par_behind_seq_fails_at_build() {
        // Full-AST validation pre-walk: a malformed Par that the old
        // eager cascade would only reach at step time (behind a lazy
        // Seq child) now fails at Engine::new.
        let ast = N::Seq(vec![
            N::Call("first".into()),
            N::Par {
                children: vec![],
                policy: JoinPolicy {
                    shape: JoinShape::Any,
                    fail: FailPolicy::FailFast,
                },
                reducer: "all_ordered",
            },
        ]);
        let ast = TestAst::build(ast);
        let err = Engine::new(ast, default_registry()).unwrap_err();
        assert!(matches!(err, EngineError::Malformed { .. }), "got {err:?}");
    }
}
