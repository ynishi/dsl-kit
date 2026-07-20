//! Reference flow DSL for `dsl-kit` (v3 engine shape).
//!
//! Defines a small orchestration DSL — `Seq`, `Par`, `Call`, `Scope`,
//! `Maybe` — with a stepper that satisfies the v3 [`Stepper`] trait.
//!
//! Commit B1 note: `Par` with all-`Call` children now schedules them as
//! a real fan-out — N `Pending` are emitted at Par entry, host resolves
//! them in any order, the configured reducer folds the slots into the
//! parent value. Par with non-`Call` children (nested sub-flows) still
//! falls back to the Commit A sequential path; full generalisation is
//! deferred to a later commit alongside a real `FrameTree` walk.

#![warn(missing_docs)]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;

use dsl_kit::{
    BreakpointSet, CallFrameId, ChildIndex, DslNode, EngineError, EngineResult, Event, EventSink,
    FailPolicy, Frame, FrameTree, IdGen, Iteration, JoinPolicy, JoinShape, NodeContext, NodeId,
    Path, Phase, Pending, Reducer, ReducerCollectAll, ReducerId, ReducerRegistry, StepOutcome,
    Stepper, SuspendReason, SuspensionId, Walk,
};
use smallvec::SmallVec;

// ---------- AST ---------------------------------------------------------

/// AST of the flow DSL.
#[derive(Debug, dsl_kit_macros::DslNode)]
pub enum Flow {
    /// Runs its children in order.
    Seq {
        /// Stable node id.
        id: NodeId,
        /// Children evaluated in declaration order.
        children: Vec<Flow>,
    },
    /// Runs its children concurrently in principle; the Commit A
    /// reference stepper still schedules them sequentially.
    Par {
        /// Stable node id.
        id: NodeId,
        /// Children scheduled concurrently.
        children: Vec<Flow>,
        /// Join policy (shape + fail). `None` defaults to
        /// `{ shape: All, fail: FailFast }`.
        policy: Option<JoinPolicy>,
        /// Reducer id. `None` defaults to `reduce_all_ordered`.
        reducer_id: Option<String>,
    },
    /// Denotes an external effect; the stepper yields once and resumes
    /// once the host has provided a result.
    Call {
        /// Stable node id.
        id: NodeId,
        /// Label identifying the effect to the host resolver.
        label: String,
    },
    /// Wraps a single inner flow with a label.
    Scope {
        /// Stable node id.
        id: NodeId,
        /// Human-readable label for the section.
        label: String,
        /// Inner flow evaluated within the scope.
        body: Box<Flow>,
    },
    /// Optionally runs an inner flow.
    Maybe {
        /// Stable node id.
        id: NodeId,
        /// Inner flow, evaluated when present.
        body: Option<Box<Flow>>,
    },
}

impl Flow {
    /// One-line summary of a node's shape, used by pretty-printers.
    pub fn summary(&self) -> String {
        match self {
            Flow::Seq { .. } => "Seq".into(),
            Flow::Par { .. } => "Par".into(),
            Flow::Call { label, .. } => format!("Call {label:?}"),
            Flow::Scope { label, .. } => format!("Scope {label:?}"),
            Flow::Maybe { .. } => "Maybe".into(),
        }
    }
}

/// Renders a `Flow` as an indented text tree using the derived
/// `Walk::walk` traversal.
pub fn pretty(flow: &Flow) -> String {
    let mut out = String::new();
    let mut depth: usize = 0;
    flow.walk(&mut |node, phase| match phase {
        Phase::Pre => {
            for _ in 0..depth {
                out.push_str("  ");
            }
            let _ = writeln!(out, "{} {}", node.node_id(), node.summary());
            depth += 1;
        }
        Phase::Post => {
            depth = depth.saturating_sub(1);
        }
    });
    out
}

/// Confirms that every `NodeId` in the tree is unique.
pub fn check_unique_ids(flow: &Flow) -> EngineResult<()> {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut duplicate: Option<NodeId> = None;

    flow.walk(&mut |node, phase| {
        if phase != Phase::Pre {
            return;
        }
        if !seen.insert(node.node_id()) && duplicate.is_none() {
            duplicate = Some(node.node_id());
        }
    });

    match duplicate {
        None => Ok(()),
        Some(id) => Err(EngineError::Malformed {
            at: NodeContext::at(id, Path::root().push(id)),
            detail: format!("node id {id} appears more than once"),
        }),
    }
}

/// Canned effect responses keyed by call label.
pub fn canned_response(label: &str) -> String {
    match label {
        "fetch_query" => "How does miette structure diagnostics?".into(),
        "search_arxiv" => "arxiv: 3 papers on structured diagnostics".into(),
        "search_github" => "github: miette (rust-lang), ariadne, codespan-reporting".into(),
        "search_web" => "web: rust blog posts, docs.rs entries".into(),
        "synthesise" => {
            "miette layers a Diagnostic trait over thiserror errors, adding code / severity / labels."
                .into()
        }
        "citation_check" => "citations: 3 sources cross-verified".into(),
        "write_report" => "report: 380 words, 3 citations, ready".into(),
        other => format!("<no handler for {other}>"),
    }
}

/// Builds a small research pipeline expressed in the flow DSL.
pub fn research_pipeline(ids: &IdGen) -> Flow {
    Flow::Seq {
        id: ids.node(),
        children: vec![
            Flow::Call { id: ids.node(), label: "fetch_query".into() },
            Flow::Scope {
                id: ids.node(),
                label: "web_research".into(),
                body: Box::new(Flow::Par {
                    id: ids.node(),
                    children: vec![
                        Flow::Call { id: ids.node(), label: "search_arxiv".into() },
                        Flow::Call { id: ids.node(), label: "search_github".into() },
                        Flow::Call { id: ids.node(), label: "search_web".into() },
                    ],
                    policy: None,
                    reducer_id: None,
                }),
            },
            Flow::Call { id: ids.node(), label: "synthesise".into() },
            Flow::Maybe {
                id: ids.node(),
                body: Some(Box::new(Flow::Call {
                    id: ids.node(),
                    label: "citation_check".into(),
                })),
            },
            Flow::Call { id: ids.node(), label: "write_report".into() },
        ],
    }
}

// ---------- Flow value / error types (v3 associated type instances) ----

/// Value type produced by the flow DSL.
///
/// Individual `Call` responses arrive as `Text(String)`. Aggregate
/// reducer output (Commit B) uses `List`. `Unit` is the value the whole
/// interpretation returns since the top-level flow produces "just
/// finished" and results are exposed through `results()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowValue {
    /// No meaningful value (top-level completion marker).
    Unit,
    /// A textual effect response.
    Text(String),
    /// A list of nested values (used by CollectAll reducers).
    List(Vec<FlowValue>),
}

/// Effect-side failure the host reports through `resolve(id, Err(_))`.
#[derive(Debug, Clone, thiserror::Error)]
#[error("flow effect error [{code}]: {message}")]
pub struct FlowEffectErr {
    /// Short machine-readable code.
    pub code: String,
    /// Human-readable message.
    pub message: String,
}

/// Top-level error type for the flow interpretation.
#[derive(Debug, thiserror::Error)]
pub enum FlowError {
    /// An engine-level error (from `dsl-kit-core`).
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// An effect-side failure surfaced via `resolve`.
    #[error(transparent)]
    Effect(#[from] FlowEffectErr),
}

/// Placeholder cursor type. The Commit A `FlowStepper` retains its
/// internal state machine (`Vec<Frame>` stack) and does not expose a
/// per-node cursor; Commit B may replace the shadow stack with a real
/// `FrameTree` walk and populate this type meaningfully.
#[derive(Debug, Clone, Default)]
pub struct FlowCursor;

// ---------- CountingSink -----------------------------------------------

/// Silent event sink that counts each event kind.
#[derive(Debug, Default, Clone, Copy)]
pub struct CountingSink {
    /// Number of `VisitPre` events observed.
    pub visit_pre: u32,
    /// Number of `VisitPost` events observed.
    pub visit_post: u32,
    /// Number of `FrameEnter` events observed.
    pub frame_enter: u32,
    /// Number of `FrameLeave` events observed.
    pub frame_leave: u32,
    /// Number of `IterationTick` events observed.
    pub iteration_tick: u32,
    /// Number of `Suspend` events observed.
    pub suspend: u32,
    /// Number of `Resume` events observed.
    pub resume: u32,
}

impl EventSink for CountingSink {
    fn emit(&mut self, event: &Event) {
        match event {
            Event::VisitPre { .. } => self.visit_pre += 1,
            Event::VisitPost { .. } => self.visit_post += 1,
            Event::FrameEnter { .. } => self.frame_enter += 1,
            Event::FrameLeave { .. } => self.frame_leave += 1,
            Event::IterationTick { .. } => self.iteration_tick += 1,
            Event::Suspend { .. } => self.suspend += 1,
            Event::Resume { .. } => self.resume += 1,
            _ => {}
        }
    }
}

impl CountingSink {
    /// Returns a single-line human-readable event histogram.
    pub fn summarise(&self) -> String {
        format!(
            "pre={} post={} frame_enter={} frame_leave={} iter={} suspend={} resume={}",
            self.visit_pre,
            self.visit_post,
            self.frame_enter,
            self.frame_leave,
            self.iteration_tick,
            self.suspend,
            self.resume,
        )
    }
}

// ---------- FlowStepper (internal state machine + v3 Stepper impl) -----

/// Stepper over a `Flow` program.
///
/// Internal semantics (Commit A): sequential DFS walk with a
/// `Vec<InternalFrame>` stack. `Par` schedules children sequentially.
/// Every `Call` yields one `Pending` at a time (single in-flight).
///
/// The public [`Stepper`] impl adapts this to the v3 shape: assigns a
/// [`SuspensionId`] on each yield, exposes it through
/// [`Stepper::pending`], and accepts
/// `resolve(id, Result<FlowValue, FlowEffectErr>)`.
pub struct FlowStepper<'a> {
    stack: Vec<InternalFrame<'a>>,
    events: CountingSink,
    next_frame: u64,
    suspend_pending: bool,
    breakpoint_yielded: bool,
    // v3 shadow state
    next_suspension: u64,
    pending: Vec<Pending>,
    id_to_node: HashMap<SuspensionId, NodeId>,
    results: HashMap<NodeId, String>,
    done: bool,
    frame_tree_stub: FrameTree<FlowValue, FlowCursor, (), FlowEffectErr>,
    // Commit B1: real Par fan-out state
    par_contexts: Vec<ParContext<'a>>,
    sid_to_par: HashMap<SuspensionId, (usize, ChildIndex)>,
    /// Item 1b: sids that live inside a subtree slot. Resolving a
    /// subtree sid records the result into `self.results` (like a
    /// plain Call) and, on `Err`, routes to `record_par_failure`.
    sid_to_subtree: HashMap<SuspensionId, (usize, ChildIndex)>,
    cancelled: Vec<SuspensionId>,
    registry: Arc<ReducerRegistry<FlowValue, (), FlowEffectErr>>,
}

/// State of one active `Par` fan-out.
///
/// Populated at Par entry. Each Par child owns exactly one slot; the
/// slot is filled either through the Call fast path (`child_sids[i]`
/// resolves directly to `slots[i]`) or through a subtree stack
/// (`subtrees[i] = Some(state)`) that runs its own DFS until drained
/// and then fills the slot with `FlowValue::Unit`. The reducer
/// (resolved from `registry` via `reducer_id`) folds the slots once
/// `policy.shape` fires.
struct ParContext<'a> {
    slots: Vec<Option<FlowValue>>,
    /// CollectAll only: per-child effect-side failure. `Some(e)` once
    /// the child resolved with `Err(e)`; always `None` for a FailFast
    /// Par (failure aborts before reaching the reducer).
    failures: Vec<Option<FlowEffectErr>>,
    completion_order: Vec<ChildIndex>,
    /// Sid for the Call fast-path slot; `SuspensionId(0)` sentinel
    /// for subtree slots (subtree sids live in `sid_to_subtree`).
    child_sids: Vec<SuspensionId>,
    child_node_ids: Vec<NodeId>,
    /// `Some(subtree)` for a slot whose Par child is not a direct
    /// `Flow::Call` — the subtree drives a private stack until it
    /// drains, then the slot fills with `FlowValue::Unit`. `None` for
    /// a Call fast-path slot.
    subtrees: Vec<Option<SubtreeState<'a>>>,
    policy: JoinPolicy,
    reducer_id: ReducerId,
    joined: bool,
    result: Option<FlowValue>,
    failure: Option<FlowEffectErr>,
    par_ctx: NodeContext,
}

/// Item 1b: private DFS stack for a non-Call Par child.
///
/// Each subtree is stepped by swapping `self.stack` / `self.suspend_pending`
/// with these fields, running one `step_internal`, then swapping back.
/// This lets a `Par` of `Seq` / `Scope` / `Maybe` / nested `Par` fan
/// out concurrently instead of degrading to the earlier sequential
/// `ParNext` fallback.
struct SubtreeState<'a> {
    stack: Vec<InternalFrame<'a>>,
    suspend_pending: bool,
}

struct InternalFrame<'a> {
    node: &'a Flow,
    path: Path,
    state: FrameState<'a>,
    frame_id: Option<CallFrameId>,
}

enum FrameState<'a> {
    Enter,
    SeqNext { children: std::slice::Iter<'a, Flow>, index: u64 },
    /// Real fan-out `Par` waiting for its slots to fill.
    /// `context_index` indexes into `FlowStepper.par_contexts`.
    ///
    /// Slots come from two sources: direct `Flow::Call` children take
    /// the fast path (external resolve fills the slot), and non-Call
    /// children (Seq / Scope / Maybe / nested Par) drive a private
    /// subtree stack that fills the slot with `FlowValue::Unit` on
    /// drain. See `SubtreeState` and `step_par_fanout` for details.
    ParFanOut { context_index: usize },
    ScopePending { body: &'a Flow },
    ScopeDone,
    MaybePending { body: Option<&'a Flow> },
    MaybeDone,
    CallSuspending,
    CallDone,
}

impl<'a> FlowStepper<'a> {
    /// Creates a fresh stepper anchored at `root`, using the default
    /// reducer registry.
    pub fn new(root: &'a Flow) -> Self {
        Self::with_registry(root, Arc::new(flow_default_registry()))
    }

    /// Creates a fresh stepper anchored at `root` with a
    /// caller-supplied reducer registry.
    pub fn with_registry(
        root: &'a Flow,
        registry: Arc<ReducerRegistry<FlowValue, (), FlowEffectErr>>,
    ) -> Self {
        let path = Path::root().push(root.node_id());
        Self {
            stack: vec![InternalFrame {
                node: root,
                path,
                state: FrameState::Enter,
                frame_id: None,
            }],
            events: CountingSink::default(),
            next_frame: 1,
            suspend_pending: false,
            breakpoint_yielded: false,
            next_suspension: 1,
            pending: Vec::new(),
            id_to_node: HashMap::new(),
            results: HashMap::new(),
            done: false,
            frame_tree_stub: FrameTree {
                root: Frame::Node {
                    node: root.node_id(),
                    env: dsl_kit::EnvRef(std::sync::Arc::new(dsl_kit::Env {
                        delta: (),
                        parent: None,
                    })),
                    cursor: FlowCursor,
                },
                kids: Vec::new(),
            },
            par_contexts: Vec::new(),
            sid_to_par: HashMap::new(),
            sid_to_subtree: HashMap::new(),
            cancelled: Vec::new(),
            registry,
        }
    }

    /// Breakpoint-aware step. Emits `Suspended { reason: Breakpoint }`
    /// once when the next `Enter` matches a registered condition.
    pub fn step_with_breakpoints(
        &mut self,
        breakpoints: &BreakpointSet,
    ) -> Result<InternalOutcome, FlowError> {
        if self.breakpoint_yielded {
            self.breakpoint_yielded = false;
            return self.step_internal();
        }
        if breakpoints.is_empty() || self.stack.is_empty() {
            return self.step_internal();
        }

        let frame = self.stack.last().expect("non-empty");
        if matches!(frame.state, FrameState::Enter) {
            let ctx = self.ctx(frame);
            if !breakpoints.matches(&ctx).is_empty() {
                self.breakpoint_yielded = true;
                self.events.emit(&Event::Suspend {
                    at: ctx.clone(),
                    reason: SuspendReason::Breakpoint,
                });
                return Ok(InternalOutcome::Suspended {
                    reason: SuspendReason::Breakpoint,
                    at: ctx,
                });
            }
        }

        self.step_internal()
    }

    /// Loops [`Self::step_with_breakpoints`] until suspension, completion,
    /// or error.
    pub fn run_to_yield_with_breakpoints(
        &mut self,
        breakpoints: &BreakpointSet,
    ) -> Result<InternalOutcome, FlowError> {
        loop {
            match self.step_with_breakpoints(breakpoints)? {
                InternalOutcome::Advanced => continue,
                other => return Ok(other),
            }
        }
    }

    fn ctx(&self, frame: &InternalFrame<'_>) -> NodeContext {
        NodeContext {
            node: frame.node.node_id(),
            path: frame.path.clone(),
            frame: frame.frame_id,
            depth: self.stack.len() as u32,
            iteration: None,
        }
    }

    /// If the stepper is currently paused on a `Call` node, returns
    /// `(SuspensionId, NodeId, label)`.
    pub fn suspended_call(&self) -> Option<(SuspensionId, NodeId, &str)> {
        let frame = self.stack.last()?;
        if !matches!(frame.state, FrameState::CallSuspending) {
            return None;
        }
        let (node_id, label) = match frame.node {
            Flow::Call { id, label } => (*id, label.as_str()),
            _ => return None,
        };
        let sid = self.pending.iter().find(|p| p.at.node == node_id).map(|p| p.id)?;
        Some((sid, node_id, label))
    }

    /// Records the result the host produced.
    ///
    /// This is the low-level convenience used by hosts that don't go
    /// through the [`Stepper::resolve`] adapter; the trait method wraps
    /// this after converting `Result<FlowValue, FlowEffectErr>` to a
    /// success string.
    pub fn record_result(&mut self, id: NodeId, result: String) {
        self.results.insert(id, result);
        // Drop the corresponding pending entry (if any).
        self.pending.retain(|p| p.at.node != id);
    }

    /// Read access to the results recorded so far.
    pub fn results(&self) -> &HashMap<NodeId, String> {
        &self.results
    }

    /// One-line summary of the events observed so far.
    pub fn event_summary(&self) -> String {
        self.events.summarise()
    }

    /// Snapshot of the counting sink.
    pub fn events(&self) -> CountingSink {
        self.events
    }

    /// Current stack depth.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Path to the top of the stack, if any.
    pub fn current_path(&self) -> Option<Path> {
        self.stack.last().map(|f| f.path.clone())
    }

    // ---- Internal step (returns InternalOutcome, then adapted to
    //      v3 StepOutcome by the trait impl below) ------------------

    fn step_internal(&mut self) -> Result<InternalOutcome, FlowError> {
        if self.stack.is_empty() {
            self.done = true;
            return Ok(InternalOutcome::Done);
        }
        let depth_before = self.stack.len() as u32;

        let ctx = {
            let frame = self.stack.last().expect("non-empty");
            self.ctx(frame)
        };

        let frame = self.stack.last_mut().expect("non-empty");
        let path = frame.path.clone();

        match &mut frame.state {
            FrameState::Enter => {
                self.events.emit(&Event::VisitPre { at: ctx.clone() });
                match frame.node {
                    Flow::Seq { children, .. } => {
                        frame.state =
                            FrameState::SeqNext { children: children.iter(), index: 0 };
                    }
                    Flow::Par { children, policy: par_policy, reducer_id: par_reducer, .. } => {
                        // Item 1b: always enter fan-out mode. Direct
                        // `Flow::Call` children take the fast path
                        // (suspend at Par entry, resolve directly into
                        // the slot). Non-Call children get a private
                        // subtree stack that runs concurrently with
                        // its siblings and fills its slot with
                        // `FlowValue::Unit` on drain.
                        let call_id = CallFrameId(self.next_frame);
                        self.next_frame += 1;
                        let mut par_ctx = ctx.clone();
                        par_ctx.frame = Some(call_id);
                        par_ctx.depth = depth_before;

                        let context_index = self.par_contexts.len();
                        let n = children.len();
                        let mut child_sids: Vec<SuspensionId> = Vec::with_capacity(n);
                        let mut child_node_ids: Vec<NodeId> = Vec::with_capacity(n);
                        let mut subtrees: Vec<Option<SubtreeState<'a>>> =
                            Vec::with_capacity(n);

                        for (slot_idx, child) in children.iter().enumerate() {
                            match child {
                                Flow::Call { id: node_id, label } => {
                                    let sid = SuspensionId(self.next_suspension);
                                    self.next_suspension += 1;
                                    child_sids.push(sid);
                                    child_node_ids.push(*node_id);
                                    subtrees.push(None);

                                    let child_path = path.push(*node_id);
                                    let child_ctx = NodeContext {
                                        node: *node_id,
                                        path: child_path,
                                        frame: Some(call_id),
                                        depth: depth_before + 1,
                                        iteration: Some(Iteration(slot_idx as u64 + 1)),
                                    };
                                    let spec = dsl_kit::CallSpec {
                                        label: label.clone(),
                                        payload: serde_json::Value::Null,
                                    };
                                    self.pending.push(Pending {
                                        id: sid,
                                        reason: SuspendReason::Call { spec: spec.clone() },
                                        at: child_ctx.clone(),
                                    });
                                    self.id_to_node.insert(sid, *node_id);
                                    self.sid_to_par.insert(sid, (context_index, slot_idx));
                                    self.events.emit(&Event::Suspend {
                                        at: child_ctx,
                                        reason: SuspendReason::Call { spec },
                                    });
                                }
                                _ => {
                                    // Non-Call child: spawn a subtree
                                    // with a single Enter frame rooted
                                    // at the child node. Slot filled
                                    // by `FlowValue::Unit` on drain.
                                    child_sids.push(SuspensionId(0));
                                    child_node_ids.push(child.node_id());
                                    let child_path = path.push(child.node_id());
                                    subtrees.push(Some(SubtreeState {
                                        stack: vec![InternalFrame {
                                            node: child,
                                            path: child_path,
                                            state: FrameState::Enter,
                                            frame_id: None,
                                        }],
                                        suspend_pending: false,
                                    }));
                                }
                            }
                        }

                        let resolved_policy = par_policy.unwrap_or(JoinPolicy {
                            shape: JoinShape::All,
                            fail: FailPolicy::FailFast,
                        });
                        let resolved_reducer_id = par_reducer
                            .clone()
                            .map(ReducerId::from)
                            .unwrap_or_else(|| ReducerId::from("reduce_all_ordered"));
                        self.par_contexts.push(ParContext {
                            slots: vec![None; n],
                            failures: (0..n).map(|_| None).collect(),
                            completion_order: Vec::new(),
                            child_sids,
                            child_node_ids,
                            subtrees,
                            policy: resolved_policy,
                            reducer_id: resolved_reducer_id,
                            joined: false,
                            result: None,
                            failure: None,
                            par_ctx: par_ctx.clone(),
                        });

                        let frame = self.stack.last_mut().expect("non-empty");
                        frame.frame_id = Some(call_id);
                        frame.state = FrameState::ParFanOut { context_index };
                        self.events.emit(&Event::FrameEnter { at: par_ctx });
                    }
                    Flow::Scope { body, .. } => {
                        frame.state = FrameState::ScopePending { body: body.as_ref() };
                    }
                    Flow::Maybe { body, .. } => {
                        frame.state = FrameState::MaybePending { body: body.as_deref() };
                    }
                    Flow::Call { id: node_id, .. } => {
                        frame.state = FrameState::CallSuspending;
                        self.suspend_pending = true;

                        let sid = SuspensionId(self.next_suspension);
                        self.next_suspension += 1;
                        let spec = dsl_kit::CallSpec {
                            label: match frame.node {
                                Flow::Call { label, .. } => label.clone(),
                                _ => String::new(),
                            },
                            payload: serde_json::Value::Null,
                        };
                        let reason = SuspendReason::Call { spec };
                        self.pending.push(Pending {
                            id: sid,
                            reason: reason.clone(),
                            at: ctx.clone(),
                        });
                        self.id_to_node.insert(sid, *node_id);
                        self.events.emit(&Event::Suspend { at: ctx, reason });
                    }
                }
                Ok(InternalOutcome::Advanced)
            }
            FrameState::SeqNext { children, index } => {
                if let Some(next) = children.next() {
                    let child_path = path.push(next.node_id());
                    *index += 1;
                    let iter = Iteration(*index);
                    let mut ctx = ctx.clone();
                    ctx.iteration = Some(iter);
                    self.events.emit(&Event::IterationTick { at: ctx });
                    self.stack.push(InternalFrame {
                        node: next,
                        path: child_path,
                        state: FrameState::Enter,
                        frame_id: None,
                    });
                    Ok(InternalOutcome::Advanced)
                } else {
                    self.events.emit(&Event::VisitPost { at: ctx });
                    self.stack.pop();
                    Ok(InternalOutcome::Advanced)
                }
            }
            FrameState::ParFanOut { context_index } => {
                let idx = *context_index;
                // Delegate to the fan-out driver so we can borrow
                // `self` freely (subtree stepping needs `&mut self`).
                self.step_par_fanout(idx)
            }
            FrameState::ScopePending { body } => {
                let body = *body;
                let child_path = path.push(body.node_id());
                let child_state = FrameState::Enter;
                self.stack.last_mut().expect("non-empty").state = FrameState::ScopeDone;
                self.stack.push(InternalFrame {
                    node: body,
                    path: child_path,
                    state: child_state,
                    frame_id: None,
                });
                Ok(InternalOutcome::Advanced)
            }
            FrameState::ScopeDone => {
                self.events.emit(&Event::VisitPost { at: ctx });
                self.stack.pop();
                Ok(InternalOutcome::Advanced)
            }
            FrameState::MaybePending { body } => {
                let body = *body;
                self.stack.last_mut().expect("non-empty").state = FrameState::MaybeDone;
                if let Some(body) = body {
                    let child_path = path.push(body.node_id());
                    self.stack.push(InternalFrame {
                        node: body,
                        path: child_path,
                        state: FrameState::Enter,
                        frame_id: None,
                    });
                }
                Ok(InternalOutcome::Advanced)
            }
            FrameState::MaybeDone => {
                self.events.emit(&Event::VisitPost { at: ctx });
                self.stack.pop();
                Ok(InternalOutcome::Advanced)
            }
            FrameState::CallSuspending => {
                if self.suspend_pending {
                    self.suspend_pending = false;
                    return Ok(InternalOutcome::Suspended {
                        reason: SuspendReason::Call {
                            spec: dsl_kit::CallSpec {
                                label: match frame.node {
                                    Flow::Call { label, .. } => label.clone(),
                                    _ => String::new(),
                                },
                                payload: serde_json::Value::Null,
                            },
                        },
                        at: ctx,
                    });
                }
                self.events.emit(&Event::Resume { at: ctx });
                frame.state = FrameState::CallDone;
                Ok(InternalOutcome::Advanced)
            }
            FrameState::CallDone => {
                self.events.emit(&Event::VisitPost { at: ctx });
                self.stack.pop();
                Ok(InternalOutcome::Advanced)
            }
        }
    }
}

/// Internal outcome the private step machine returns; converted to
/// `StepOutcome<FlowValue>` by the [`Stepper`] impl.
#[derive(Debug)]
pub enum InternalOutcome {
    /// Advanced one internal transition.
    Advanced,
    /// Suspended on a `Call` (or breakpoint).
    Suspended {
        /// Why suspended.
        reason: SuspendReason,
        /// Where suspended.
        at: NodeContext,
    },
    /// Blocked waiting for external resolve (Par not yet joined) —
    /// no new suspension emitted, no progress possible without host
    /// input.
    Waiting,
    /// Interpretation completed.
    Done,
}

impl<'a> FlowStepper<'a> {
    /// Applies the Par's registered reducer to its slots, storing the
    /// folded value in the ParContext and returning `(V, D)`.
    fn fold_par(&mut self, context_index: usize) -> Result<(FlowValue, ()), EngineError> {
        let (reducer_id, policy, slots, failures, deltas, winners) = {
            let ctx = &self.par_contexts[context_index];
            (
                ctx.reducer_id.clone(),
                ctx.policy,
                ctx.slots.clone(),
                ctx.failures.clone(),
                vec![Some(()); ctx.slots.len()],
                ctx.completion_order.clone(),
            )
        };
        let handle = self.registry.resolve(&reducer_id, policy.fail)?;
        let (value, delta) = match handle {
            dsl_kit::ReducerHandle::FailFast(reducer) => {
                reducer.reduce(&slots, &deltas, &winners)?
            }
            dsl_kit::ReducerHandle::CollectAll(reducer) => {
                let combined: Vec<Option<Result<FlowValue, FlowEffectErr>>> = slots
                    .into_iter()
                    .zip(failures.into_iter())
                    .map(|(s, f)| match (s, f) {
                        (Some(v), _) => Some(Ok(v)),
                        (None, Some(e)) => Some(Err(e)),
                        (None, None) => None,
                    })
                    .collect();
                reducer.reduce(&combined, &deltas, &winners)?
            }
        };
        self.par_contexts[context_index].result = Some(value.clone());
        Ok((value, delta))
    }

    /// Records a successful resolve into a Par slot, updates
    /// completion order, and checks whether the shape has fired.
    fn record_par_slot(&mut self, context_index: usize, slot_idx: usize, value: FlowValue) {
        {
            let ctx = &mut self.par_contexts[context_index];
            if ctx.joined {
                return;
            }
            ctx.slots[slot_idx] = Some(value);
            ctx.completion_order.push(slot_idx);
        }
        let (fires, n) = {
            let ctx = &self.par_contexts[context_index];
            let successes = ctx.completion_order.len();
            let n = ctx.slots.len();
            let fires = match ctx.policy.shape {
                JoinShape::All => successes == n,
                JoinShape::Any => successes >= 1,
                JoinShape::FirstK(k) => successes >= k,
            };
            (fires, n)
        };
        if fires {
            self.par_contexts[context_index].joined = true;
            self.cancel_par_children(context_index, n);
        }
    }

    /// Cancels every still-live child of a joined ParContext:
    /// Call fast-path slots have their sid pushed to `self.cancelled`;
    /// subtree slots have their subtree stack dropped and every
    /// sid mapped to that subtree cancelled.
    fn cancel_par_children(&mut self, context_index: usize, n: usize) {
        for slot in 0..n {
            let slot_filled = self.par_contexts[context_index].slots[slot].is_some();
            let slot_failed = self.par_contexts[context_index].failures[slot].is_some();
            if slot_filled || slot_failed {
                continue;
            }
            // Call fast-path slot: cancel the direct sid.
            if self.par_contexts[context_index].subtrees[slot].is_none() {
                let sid = self.par_contexts[context_index].child_sids[slot];
                if sid != SuspensionId(0) {
                    self.cancelled.push(sid);
                    self.sid_to_par.remove(&sid);
                    self.id_to_node.remove(&sid);
                    self.pending.retain(|p| p.id != sid);
                }
            } else {
                // Subtree slot: drop the stack, cancel every sid that
                // was routed to this (context_index, slot).
                self.par_contexts[context_index].subtrees[slot] = None;
                let victims: Vec<SuspensionId> = self
                    .sid_to_subtree
                    .iter()
                    .filter_map(|(sid, coord)| {
                        if *coord == (context_index, slot) {
                            Some(*sid)
                        } else {
                            None
                        }
                    })
                    .collect();
                for sid in victims {
                    self.cancelled.push(sid);
                    self.sid_to_subtree.remove(&sid);
                    self.id_to_node.remove(&sid);
                    self.pending.retain(|p| p.id != sid);
                }
            }
        }
    }

    /// Records a failed resolve into a Par slot.
    ///
    /// - FailFast: mark the ParContext as failed, cancel remaining
    ///   siblings; the next `step()` propagates the error.
    /// - CollectAll: record the failure in the per-child `failures`
    ///   slot and continue. If the shape target is no longer
    ///   attainable given the remaining live children, fire the join
    ///   with the collected slot vector; the reducer decides whether
    ///   to return a value or an aggregate `Err`.
    fn record_par_failure(
        &mut self,
        context_index: usize,
        slot_idx: usize,
        err: FlowEffectErr,
    ) {
        let (join_now, n) = {
            let ctx = &mut self.par_contexts[context_index];
            if ctx.joined {
                return;
            }
            match ctx.policy.fail {
                FailPolicy::FailFast => {
                    ctx.joined = true;
                    ctx.failure = Some(err);
                    (true, ctx.slots.len())
                }
                FailPolicy::CollectAll => {
                    ctx.failures[slot_idx] = Some(err);
                    let n = ctx.slots.len();
                    let successes = ctx.completion_order.len();
                    let failed = ctx.failures.iter().filter(|f| f.is_some()).count();
                    let live = n - successes - failed;
                    let potential_successes = successes + live;
                    let target_unattainable = match ctx.policy.shape {
                        JoinShape::All => failed >= 1,
                        JoinShape::Any => potential_successes == 0,
                        JoinShape::FirstK(k) => potential_successes < k,
                    };
                    if target_unattainable {
                        ctx.joined = true;
                        (true, n)
                    } else {
                        (false, n)
                    }
                }
            }
        };
        if join_now {
            self.cancel_par_children(context_index, n);
        }
    }

    /// Item 1b: fan-out driver invoked from the `ParFanOut` arm of
    /// `step_internal`. Handles failure propagation, join+fold, and
    /// one round-robin advance across subtree slots.
    fn step_par_fanout(&mut self, context_index: usize) -> Result<InternalOutcome, FlowError> {
        let par_ctx_clone = self.par_contexts[context_index].par_ctx.clone();
        // 1. FailFast propagation.
        if let Some(err) = self.par_contexts[context_index].failure.clone() {
            return Err(FlowError::Effect(err));
        }
        // 2. Joined → fold + pop.
        if self.par_contexts[context_index].joined {
            let _ = self.fold_par(context_index).map_err(FlowError::Engine)?;
            self.events.emit(&Event::FrameLeave { at: par_ctx_clone.clone() });
            self.events.emit(&Event::VisitPost { at: par_ctx_clone });
            self.stack.pop();
            return Ok(InternalOutcome::Advanced);
        }
        // 3. Try to advance one subtree that isn't blocked on a
        //    resolve. Round-robin: first advanceable slot wins.
        let n = self.par_contexts[context_index].slots.len();
        for slot_idx in 0..n {
            if self.par_contexts[context_index].subtrees[slot_idx].is_none() {
                continue; // Call fast-path slot (owned by external resolve).
            }
            // Skip subtree if any sid is currently mapped to it.
            let blocked = self
                .sid_to_subtree
                .values()
                .any(|coord| *coord == (context_index, slot_idx));
            if blocked {
                continue;
            }
            let advanced = self.step_subtree(context_index, slot_idx)?;
            if advanced {
                return Ok(InternalOutcome::Advanced);
            }
            // step_subtree returned Waiting — try the next slot.
        }
        Ok(InternalOutcome::Waiting)
    }

    /// Item 1b: run one internal step against the subtree at
    /// `par_contexts[context_index].subtrees[slot_idx]` by swapping
    /// `self.stack` / `self.suspend_pending` with the subtree fields.
    ///
    /// Returns `Ok(true)` if the outer scheduler should treat this as
    /// `Advanced` (the subtree progressed, drained, or suspended);
    /// `Ok(false)` if the subtree is currently `Waiting` (a nested
    /// `Par` blocked on all its own subtrees) so the outer round-
    /// robin loop should try the next slot.
    fn step_subtree(
        &mut self,
        context_index: usize,
        slot_idx: usize,
    ) -> Result<bool, FlowError> {
        // Take subtree state out.
        let mut sub = self.par_contexts[context_index].subtrees[slot_idx]
            .take()
            .expect("caller checked subtree is Some");
        if sub.stack.is_empty() {
            // Drained already: fill slot with Unit and drop the state.
            if self.par_contexts[context_index].slots[slot_idx].is_none()
                && self.par_contexts[context_index].failures[slot_idx].is_none()
            {
                self.record_par_slot(context_index, slot_idx, FlowValue::Unit);
            }
            return Ok(true);
        }
        // Swap into main position.
        std::mem::swap(&mut self.stack, &mut sub.stack);
        std::mem::swap(&mut self.suspend_pending, &mut sub.suspend_pending);
        let before_pending = self.pending.len();
        let done_before = self.done;
        let outcome = self.step_internal();
        // Any newly-added sid this step created inside the subtree is
        // owned by the subtree, not the outer world; reroute it.
        let new_sids: Vec<SuspensionId> = self
            .pending
            .iter()
            .skip(before_pending)
            .map(|p| p.id)
            .collect();
        for sid in new_sids {
            // Nested Par entry inside a subtree pre-registers its own
            // sids via `sid_to_par`. Those should NOT be shadowed by a
            // subtree mapping (they resolve into the nested Par's
            // slots directly). Only untracked sids belong to the
            // subtree's own Call chain.
            let already_owned = self.sid_to_par.contains_key(&sid);
            if !already_owned {
                self.sid_to_subtree
                    .insert(sid, (context_index, slot_idx));
            }
        }
        // Swap back — even on Err — so state is preserved.
        std::mem::swap(&mut self.stack, &mut sub.stack);
        std::mem::swap(&mut self.suspend_pending, &mut sub.suspend_pending);
        // `step_internal` sets `self.done = true` when its stack goes
        // empty; that reflects the subtree draining, NOT the outer
        // program. Undo the flag unless it was already set.
        if !done_before {
            self.done = false;
        }

        let outcome = outcome?;
        let drained = sub.stack.is_empty();
        if drained {
            // Fill slot with Unit (subtree completed).
            if self.par_contexts[context_index].slots[slot_idx].is_none()
                && self.par_contexts[context_index].failures[slot_idx].is_none()
            {
                self.record_par_slot(context_index, slot_idx, FlowValue::Unit);
            }
            // Do not restore the subtree.
        } else {
            // Restore.
            self.par_contexts[context_index].subtrees[slot_idx] = Some(sub);
        }
        match outcome {
            InternalOutcome::Advanced | InternalOutcome::Suspended { .. } => Ok(true),
            InternalOutcome::Waiting => Ok(false),
            InternalOutcome::Done => Ok(true),
        }
    }
}

// ---------- v3 Stepper impl --------------------------------------------

impl<'a> Stepper for FlowStepper<'a> {
    type Value = FlowValue;
    type Cursor = FlowCursor;
    type Delta = ();
    type EffectError = FlowEffectErr;
    type Error = FlowError;

    fn step(&mut self) -> Result<StepOutcome<Self::Value>, Self::Error> {
        let before_pending_len = self.pending.len();
        let outcome = self.step_internal()?;
        match outcome {
            InternalOutcome::Advanced => {
                if self.pending.len() > before_pending_len {
                    let newly: SmallVec<[Pending; 1]> = self
                        .pending
                        .iter()
                        .skip(before_pending_len)
                        .cloned()
                        .collect();
                    Ok(StepOutcome::Blocked { newly_pending: newly })
                } else {
                    Ok(StepOutcome::Ready)
                }
            }
            InternalOutcome::Suspended { .. } => {
                let newly: SmallVec<[Pending; 1]> = self
                    .pending
                    .iter()
                    .skip(before_pending_len)
                    .cloned()
                    .collect();
                Ok(StepOutcome::Blocked { newly_pending: newly })
            }
            InternalOutcome::Waiting => {
                Ok(StepOutcome::Blocked { newly_pending: SmallVec::new() })
            }
            InternalOutcome::Done => {
                self.done = true;
                Ok(StepOutcome::Done(FlowValue::Unit))
            }
        }
    }

    fn resolve(
        &mut self,
        id: SuspensionId,
        result: Result<Self::Value, Self::EffectError>,
    ) -> Result<(), Self::Error> {
        // Par-slot fast-path resolve.
        if let Some((context_index, slot_idx)) = self.sid_to_par.remove(&id) {
            self.pending.retain(|p| p.id != id);
            let node_id = self.par_contexts[context_index].child_node_ids[slot_idx];
            self.id_to_node.remove(&id);
            match result {
                Ok(v) => {
                    // Also record the child's individual result for
                    // downstream introspection.
                    let text = match &v {
                        FlowValue::Text(s) => s.clone(),
                        FlowValue::Unit => String::new(),
                        FlowValue::List(items) => format!("{items:?}"),
                    };
                    self.results.insert(node_id, text);
                    self.record_par_slot(context_index, slot_idx, v);
                    Ok(())
                }
                Err(e) => {
                    self.record_par_failure(context_index, slot_idx, e);
                    Ok(())
                }
            }
        } else if let Some((context_index, slot_idx)) = self.sid_to_subtree.remove(&id) {
            // Item 1b: subtree Call resolve. Route the result to the
            // subtree's frame (like a plain Call) so the subtree can
            // resume; slot-fill happens later when the subtree drains.
            // On `Err`, route to the enclosing Par (FailFast aborts
            // siblings; CollectAll records into `failures[slot_idx]`).
            let node_id = self
                .id_to_node
                .remove(&id)
                .ok_or(EngineError::UnknownSuspension { id })?;
            self.pending.retain(|p| p.id != id);
            match result {
                Ok(v) => {
                    let text = match v {
                        FlowValue::Text(s) => s,
                        FlowValue::Unit => String::new(),
                        FlowValue::List(items) => format!("{items:?}"),
                    };
                    self.results.insert(node_id, text);
                    Ok(())
                }
                Err(e) => {
                    self.record_par_failure(context_index, slot_idx, e);
                    Ok(())
                }
            }
        } else {
            // Single-Call (non-Par) resolve path.
            let node_id = self
                .id_to_node
                .remove(&id)
                .ok_or(EngineError::UnknownSuspension { id })?;
            self.pending.retain(|p| p.id != id);
            match result {
                Ok(v) => {
                    let text = match v {
                        FlowValue::Text(s) => s,
                        FlowValue::Unit => String::new(),
                        FlowValue::List(items) => format!("{items:?}"),
                    };
                    self.results.insert(node_id, text);
                    Ok(())
                }
                Err(e) => Err(FlowError::Effect(e)),
            }
        }
    }

    fn pending(&self) -> &[Pending] {
        &self.pending
    }

    fn take_cancellations(&mut self) -> Vec<SuspensionId> {
        std::mem::take(&mut self.cancelled)
    }

    fn frame_tree(
        &self,
    ) -> &FrameTree<Self::Value, Self::Cursor, Self::Delta, Self::EffectError> {
        &self.frame_tree_stub
    }

    fn is_done(&self) -> bool {
        self.done
    }
}

// ---------- Default reducers + registry --------------------------------

/// FailFast + All: returns `FlowValue::List(slots-in-declaration-order)`.
pub struct FlowReduceAllOrdered;

impl Reducer<FlowValue, ()> for FlowReduceAllOrdered {
    fn reduce(
        &self,
        slots: &[Option<FlowValue>],
        _deltas: &[Option<()>],
        _winners: &[ChildIndex],
    ) -> Result<(FlowValue, ()), EngineError> {
        let list: Vec<FlowValue> = slots.iter().filter_map(|s| s.clone()).collect();
        Ok((FlowValue::List(list), ()))
    }
}

/// FailFast + Any: returns the completion-order winner.
pub struct FlowReduceAnyFirstWinner;

impl Reducer<FlowValue, ()> for FlowReduceAnyFirstWinner {
    fn reduce(
        &self,
        slots: &[Option<FlowValue>],
        _deltas: &[Option<()>],
        winners: &[ChildIndex],
    ) -> Result<(FlowValue, ()), EngineError> {
        let winner = winners.first().copied().unwrap_or(0);
        let v = slots
            .get(winner)
            .and_then(|s| s.clone())
            .unwrap_or(FlowValue::Unit);
        Ok((v, ()))
    }
}

/// FailFast + FirstK: returns `FlowValue::List(winners-in-completion-order)`.
pub struct FlowReduceFirstKOrdered;

impl Reducer<FlowValue, ()> for FlowReduceFirstKOrdered {
    fn reduce(
        &self,
        slots: &[Option<FlowValue>],
        _deltas: &[Option<()>],
        winners: &[ChildIndex],
    ) -> Result<(FlowValue, ()), EngineError> {
        let list: Vec<FlowValue> = winners
            .iter()
            .filter_map(|&i| slots.get(i).and_then(|s| s.clone()))
            .collect();
        Ok((FlowValue::List(list), ()))
    }
}

/// CollectAll + All: returns `FlowValue::List(all-Some(Ok)-values)`.
pub struct FlowReduceCollectAllResults;

impl ReducerCollectAll<FlowValue, (), FlowEffectErr> for FlowReduceCollectAllResults {
    fn reduce(
        &self,
        slots: &[Option<Result<FlowValue, FlowEffectErr>>],
        _deltas: &[Option<()>],
        _winners: &[ChildIndex],
    ) -> Result<(FlowValue, ()), EngineError> {
        let list: Vec<FlowValue> = slots
            .iter()
            .filter_map(|s| match s {
                Some(Ok(v)) => Some(v.clone()),
                _ => None,
            })
            .collect();
        Ok((FlowValue::List(list), ()))
    }
}

/// CollectAll + Any: first success wins, else fails.
pub struct FlowReduceAnyFirstOrAllFailures;

impl ReducerCollectAll<FlowValue, (), FlowEffectErr> for FlowReduceAnyFirstOrAllFailures {
    fn reduce(
        &self,
        slots: &[Option<Result<FlowValue, FlowEffectErr>>],
        _deltas: &[Option<()>],
        winners: &[ChildIndex],
    ) -> Result<(FlowValue, ()), EngineError> {
        if let Some(&w) = winners.first() {
            if let Some(Some(Ok(v))) = slots.get(w) {
                return Ok((v.clone(), ()));
            }
        }
        Err(EngineError::Aborted {
            at: NodeContext::at(NodeId(0), Path::root()),
            reason: "reduce_any_first_or_all_failures: no successful child".into(),
        })
    }
}

/// CollectAll + FirstK: returns the first `k` successes if attainable.
pub struct FlowReduceFirstKBestEffort;

impl ReducerCollectAll<FlowValue, (), FlowEffectErr> for FlowReduceFirstKBestEffort {
    fn reduce(
        &self,
        slots: &[Option<Result<FlowValue, FlowEffectErr>>],
        _deltas: &[Option<()>],
        winners: &[ChildIndex],
    ) -> Result<(FlowValue, ()), EngineError> {
        let list: Vec<FlowValue> = winners
            .iter()
            .filter_map(|&i| match slots.get(i) {
                Some(Some(Ok(v))) => Some(v.clone()),
                _ => None,
            })
            .collect();
        Ok((FlowValue::List(list), ()))
    }
}

/// Returns the default flow-dsl reducer registry, populated with the
/// six canonical reducers (three FailFast + three CollectAll).
pub fn flow_default_registry() -> ReducerRegistry<FlowValue, (), FlowEffectErr> {
    let mut reg = ReducerRegistry::new();
    reg.register_fail_fast("reduce_all_ordered", Arc::new(FlowReduceAllOrdered));
    reg.register_fail_fast(
        "reduce_any_first_winner",
        Arc::new(FlowReduceAnyFirstWinner),
    );
    reg.register_fail_fast(
        "reduce_first_k_ordered",
        Arc::new(FlowReduceFirstKOrdered),
    );
    reg.register_collect_all(
        "reduce_collect_all_results",
        Arc::new(FlowReduceCollectAllResults),
    );
    reg.register_collect_all(
        "reduce_any_first_or_all_failures",
        Arc::new(FlowReduceAnyFirstOrAllFailures),
    );
    reg.register_collect_all(
        "reduce_first_k_best_effort",
        Arc::new(FlowReduceFirstKBestEffort),
    );
    reg
}

// ---------- Sync driver helper (host convenience) ---------------------

/// Runs a flow to completion in sync mode, resolving each `Call`
/// suspension with the corresponding canned response. Used by demos.
pub fn run_flow_sync(program: &Flow) -> Result<HashMap<NodeId, String>, FlowError> {
    let mut stepper = FlowStepper::new(program);
    let bp = BreakpointSet::new();
    let mut steps = 0u32;
    loop {
        match stepper.run_to_yield_with_breakpoints(&bp)? {
            InternalOutcome::Suspended { .. } => {
                if let Some((sid, _node, label)) = stepper.suspended_call() {
                    let response = canned_response(label);
                    stepper.resolve(sid, Ok(FlowValue::Text(response)))?;
                }
            }
            InternalOutcome::Done => break,
            InternalOutcome::Advanced => {}
            InternalOutcome::Waiting => {
                // Par is blocked; resolve any outstanding Par-slot
                // pendings with canned responses to drive the fan-out
                // forward.
                let outstanding: Vec<(SuspensionId, String)> = stepper
                    .pending()
                    .iter()
                    .filter_map(|p| match &p.reason {
                        SuspendReason::Call { spec } => {
                            Some((p.id, spec.label.clone()))
                        }
                        _ => None,
                    })
                    .collect();
                if outstanding.is_empty() {
                    break;
                }
                for (sid, label) in outstanding {
                    let response = canned_response(&label);
                    stepper.resolve(sid, Ok(FlowValue::Text(response)))?;
                }
            }
        }
        steps += 1;
        if steps > 4096 {
            return Err(FlowError::Engine(EngineError::Aborted {
                at: NodeContext::at(NodeId(0), Path::root()),
                reason: "run_flow_sync exceeded safety limit".into(),
            }));
        }
    }
    Ok(stepper.results.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn research_pipeline_runs_end_to_end() {
        let ids = IdGen::new();
        let program = research_pipeline(&ids);
        let results = run_flow_sync(&program).expect("run ok");
        // 7 Call nodes in the reference program.
        assert_eq!(results.len(), 7);
    }

    #[test]
    fn check_unique_ids_detects_duplicate() {
        let dup = NodeId(101);
        let program = Flow::Seq {
            id: dup,
            children: vec![Flow::Call { id: dup, label: "x".into() }],
        };
        let err = check_unique_ids(&program).unwrap_err();
        assert!(matches!(err, EngineError::Malformed { .. }));
    }

    #[test]
    fn pretty_renders_indented_tree() {
        let ids = IdGen::new();
        let program = research_pipeline(&ids);
        let text = pretty(&program);
        assert!(text.contains("Seq"));
        assert!(text.contains("Par"));
        assert!(text.contains("fetch_query"));
    }

    #[test]
    fn par_fan_out_emits_n_pending_and_folds_out_of_order() {
        // A Par of 3 Calls: expected shape = 3 pending emitted at Par
        // entry, resolve in reverse order, reducer folds when all
        // slots filled.
        let ids = IdGen::new();
        let program = Flow::Par {
            id: ids.node(),
            children: vec![
                Flow::Call { id: ids.node(), label: "a".into() },
                Flow::Call { id: ids.node(), label: "b".into() },
                Flow::Call { id: ids.node(), label: "c".into() },
            ],
            policy: None,
            reducer_id: None,
        };
        let mut stepper = FlowStepper::new(&program);

        // Enter the Par: emits 3 Pending in one shot.
        let out1 = stepper.step().expect("enter par");
        match out1 {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 3, "expected 3 newly pending");
            }
            other => panic!("expected Blocked with 3 newly pending, got {other:?}"),
        }
        assert_eq!(stepper.pending().len(), 3);
        let sids: Vec<SuspensionId> = stepper.pending().iter().map(|p| p.id).collect();

        // Resolve out of declaration order: 2, 0, 1.
        stepper
            .resolve(sids[2], Ok(FlowValue::Text("c-resp".into())))
            .expect("resolve c");
        assert_eq!(stepper.pending().len(), 2);

        // Next step: Par not yet joined, returns Blocked{empty}.
        let out2 = stepper.step().expect("still waiting");
        assert!(matches!(out2, StepOutcome::Blocked { newly_pending } if newly_pending.is_empty()));

        stepper
            .resolve(sids[0], Ok(FlowValue::Text("a-resp".into())))
            .expect("resolve a");
        stepper
            .resolve(sids[1], Ok(FlowValue::Text("b-resp".into())))
            .expect("resolve b");
        assert_eq!(stepper.pending().len(), 0);

        // Drive until Done. Par folds via reduce_all_ordered.
        for _ in 0..20 {
            match stepper.step().expect("step") {
                StepOutcome::Done(_) => {
                    // Per-child results recorded (labels resolved).
                    assert!(!stepper.results().is_empty());
                    return;
                }
                _ => continue,
            }
        }
        panic!("did not reach Done");
    }

    #[test]
    fn par_failfast_propagates_and_cancels_siblings() {
        let ids = IdGen::new();
        let program = Flow::Par {
            id: ids.node(),
            children: vec![
                Flow::Call { id: ids.node(), label: "x".into() },
                Flow::Call { id: ids.node(), label: "y".into() },
                Flow::Call { id: ids.node(), label: "z".into() },
            ],
            policy: None,
            reducer_id: None,
        };
        let mut stepper = FlowStepper::new(&program);
        let _ = stepper.step().expect("enter par");
        let sids: Vec<SuspensionId> = stepper.pending().iter().map(|p| p.id).collect();

        // Fail the middle slot with an EffectError.
        stepper
            .resolve(
                sids[1],
                Err(FlowEffectErr {
                    code: "timeout".into(),
                    message: "y timed out".into(),
                }),
            )
            .expect("resolve records failure");

        // Next step should propagate the error.
        let err = stepper.step().expect_err("failfast should propagate");
        match err {
            FlowError::Effect(e) => assert_eq!(e.code, "timeout"),
            FlowError::Engine(_) => panic!("expected FlowError::Effect"),
        }

        // Sibling suspensions queued for cancellation.
        let cancels = stepper.take_cancellations();
        assert!(cancels.contains(&sids[0]));
        assert!(cancels.contains(&sids[2]));
    }

    #[test]
    fn default_registry_carries_six_reducers() {
        let reg = flow_default_registry();
        // FailFast side.
        for id in ["reduce_all_ordered", "reduce_any_first_winner", "reduce_first_k_ordered"] {
            let h = reg
                .resolve(&ReducerId::from(id), FailPolicy::FailFast)
                .unwrap_or_else(|_| panic!("missing fail-fast reducer {id}"));
            assert!(matches!(h, dsl_kit::ReducerHandle::FailFast(_)));
        }
        // CollectAll side.
        for id in [
            "reduce_collect_all_results",
            "reduce_any_first_or_all_failures",
            "reduce_first_k_best_effort",
        ] {
            let h = reg
                .resolve(&ReducerId::from(id), FailPolicy::CollectAll)
                .unwrap_or_else(|_| panic!("missing collect-all reducer {id}"));
            assert!(matches!(h, dsl_kit::ReducerHandle::CollectAll(_)));
        }
    }

    #[test]
    fn par_collect_all_first_k_reaches_target_with_one_failure() {
        // FirstK(2) with 3 children, second child fails; the two
        // remaining successes still satisfy the target, so the
        // reducer sees Ok/Err/Ok slots and returns the majority.
        let ids = IdGen::new();
        let program = Flow::Par {
            id: ids.node(),
            children: vec![
                Flow::Call { id: ids.node(), label: "model_a".into() },
                Flow::Call { id: ids.node(), label: "model_b".into() },
                Flow::Call { id: ids.node(), label: "model_c".into() },
            ],
            policy: Some(JoinPolicy {
                shape: JoinShape::FirstK(2),
                fail: FailPolicy::CollectAll,
            }),
            reducer_id: Some("reduce_first_k_best_effort".into()),
        };
        let mut stepper = FlowStepper::new(&program);
        let _ = stepper.step().expect("enter par");
        let sids: Vec<SuspensionId> = stepper.pending().iter().map(|p| p.id).collect();

        stepper
            .resolve(sids[0], Ok(FlowValue::Text("answer-a".into())))
            .expect("resolve a");
        stepper
            .resolve(
                sids[1],
                Err(FlowEffectErr { code: "timeout".into(), message: "b".into() }),
            )
            .expect("resolve b failure");
        stepper
            .resolve(sids[2], Ok(FlowValue::Text("answer-c".into())))
            .expect("resolve c");

        // Drive to done — no error should propagate under CollectAll.
        for _ in 0..20 {
            match stepper.step().expect("step") {
                StepOutcome::Done(_) => {
                    // The failure did NOT abort siblings.
                    assert!(stepper.take_cancellations().is_empty());
                    return;
                }
                _ => continue,
            }
        }
        panic!("did not reach Done");
    }

    #[test]
    fn par_collect_all_any_all_failed_surfaces_reducer_err() {
        // Shape=Any + CollectAll: fail all children; once no live
        // successors remain the target becomes unattainable, the
        // reducer receives an all-Err slot vector and returns
        // EngineError::Aborted.
        let ids = IdGen::new();
        let program = Flow::Par {
            id: ids.node(),
            children: vec![
                Flow::Call { id: ids.node(), label: "x".into() },
                Flow::Call { id: ids.node(), label: "y".into() },
            ],
            policy: Some(JoinPolicy {
                shape: JoinShape::Any,
                fail: FailPolicy::CollectAll,
            }),
            reducer_id: Some("reduce_any_first_or_all_failures".into()),
        };
        let mut stepper = FlowStepper::new(&program);
        let _ = stepper.step().expect("enter par");
        let sids: Vec<SuspensionId> = stepper.pending().iter().map(|p| p.id).collect();

        // Fail both slots — nothing to cancel (target only becomes
        // unattainable on the second failure).
        stepper
            .resolve(
                sids[0],
                Err(FlowEffectErr { code: "boom-x".into(), message: "x".into() }),
            )
            .expect("resolve x failure");
        stepper
            .resolve(
                sids[1],
                Err(FlowEffectErr { code: "boom-y".into(), message: "y".into() }),
            )
            .expect("resolve y failure");

        // Both children failed; no live sibling to cancel.
        assert!(stepper.take_cancellations().is_empty());

        // Next step folds via the CollectAll reducer, which returns
        // EngineError::Aborted (no successful child).
        let err = stepper
            .step()
            .expect_err("reduce_any_first_or_all_failures should Err");
        assert!(matches!(err, FlowError::Engine(_)));
    }

    #[test]
    fn par_of_seq_fans_out_and_completes() {
        // Item 1b: Par of two Seq children, each Seq has two Calls.
        // Fan-out is real (both Seqs advance concurrently) but each
        // Seq is internally sequential — so pending count at any
        // moment is at most one per Seq (i.e. up to 2, not 4).
        let ids = IdGen::new();
        let seq_a = Flow::Seq {
            id: ids.node(),
            children: vec![
                Flow::Call { id: ids.node(), label: "a1".into() },
                Flow::Call { id: ids.node(), label: "a2".into() },
            ],
        };
        let seq_b = Flow::Seq {
            id: ids.node(),
            children: vec![
                Flow::Call { id: ids.node(), label: "b1".into() },
                Flow::Call { id: ids.node(), label: "b2".into() },
            ],
        };
        let program = Flow::Par {
            id: ids.node(),
            children: vec![seq_a, seq_b],
            policy: None,
            reducer_id: None,
        };
        let mut stepper = FlowStepper::new(&program);

        // Drive to the first yield-point: both subtrees should reach
        // their first Call and suspend, giving 2 concurrent pendings.
        for _ in 0..64 {
            if stepper.pending().len() >= 2 {
                break;
            }
            let _ = stepper.step().expect("step to first yield");
        }
        assert_eq!(
            stepper.pending().len(),
            2,
            "expected 2 concurrent pending (one per Seq subtree)"
        );
        let labels_first: Vec<String> = stepper
            .pending()
            .iter()
            .map(|p| match &p.reason {
                SuspendReason::Call { spec } => spec.label.clone(),
                _ => String::new(),
            })
            .collect();
        assert!(labels_first.contains(&"a1".to_string()));
        assert!(labels_first.contains(&"b1".to_string()));

        // Resolve both. Each subtree should advance to its second
        // Call; we expect two new pendings (a2 + b2).
        let first_sids: Vec<SuspensionId> =
            stepper.pending().iter().map(|p| p.id).collect();
        for sid in &first_sids {
            stepper
                .resolve(*sid, Ok(FlowValue::Text("ok".into())))
                .expect("resolve first-wave");
        }
        for _ in 0..64 {
            if stepper.pending().len() >= 2 {
                break;
            }
            let _ = stepper.step().expect("step to second yield");
        }
        assert_eq!(stepper.pending().len(), 2, "expected 2 concurrent second-wave pending");
        let labels_second: Vec<String> = stepper
            .pending()
            .iter()
            .map(|p| match &p.reason {
                SuspendReason::Call { spec } => spec.label.clone(),
                _ => String::new(),
            })
            .collect();
        assert!(labels_second.contains(&"a2".to_string()));
        assert!(labels_second.contains(&"b2".to_string()));

        // Resolve the second wave and drive to Done.
        let second_sids: Vec<SuspensionId> =
            stepper.pending().iter().map(|p| p.id).collect();
        for sid in second_sids {
            stepper
                .resolve(sid, Ok(FlowValue::Text("ok".into())))
                .expect("resolve second-wave");
        }
        for _ in 0..128 {
            if let StepOutcome::Done(_) = stepper.step().expect("step to done") {
                // Four Call nodes recorded a result.
                assert_eq!(stepper.results().len(), 4);
                return;
            }
        }
        panic!("did not reach Done");
    }

    #[test]
    fn par_of_scope_wrapping_call_fans_out() {
        // Item 1b: Par of three Scope-wrapped Calls. Each Scope is
        // one subtree; all three should yield concurrently (3
        // pendings simultaneously).
        let ids = IdGen::new();
        let scope = |label: &str| Flow::Scope {
            id: ids.node(),
            label: label.into(),
            body: Box::new(Flow::Call { id: ids.node(), label: label.into() }),
        };
        let program = Flow::Par {
            id: ids.node(),
            children: vec![scope("x"), scope("y"), scope("z")],
            policy: None,
            reducer_id: None,
        };
        let mut stepper = FlowStepper::new(&program);

        // Drive until all three subtrees have reached their inner
        // Call and suspended.
        for _ in 0..64 {
            if stepper.pending().len() >= 3 {
                break;
            }
            let _ = stepper.step().expect("step to yield");
        }
        assert_eq!(
            stepper.pending().len(),
            3,
            "expected 3 concurrent pending across Scope subtrees"
        );
        let labels: Vec<String> = stepper
            .pending()
            .iter()
            .map(|p| match &p.reason {
                SuspendReason::Call { spec } => spec.label.clone(),
                _ => String::new(),
            })
            .collect();
        for want in ["x", "y", "z"] {
            assert!(labels.iter().any(|l| l == want), "missing label {want}");
        }

        // Resolve all three and drive to Done.
        let sids: Vec<SuspensionId> = stepper.pending().iter().map(|p| p.id).collect();
        for sid in sids {
            stepper
                .resolve(sid, Ok(FlowValue::Text("ok".into())))
                .expect("resolve");
        }
        for _ in 0..64 {
            if let StepOutcome::Done(_) = stepper.step().expect("step to done") {
                assert_eq!(stepper.results().len(), 3);
                return;
            }
        }
        panic!("did not reach Done");
    }

    #[test]
    fn par_failfast_inside_subtree_cancels_siblings() {
        // Item 1b: FailFast propagation across subtree slots.
        // Par of two Scope-wrapped Calls; fail the first, expect the
        // second's pending sid to be cancelled and the Par to
        // propagate the Effect error on the next step.
        let ids = IdGen::new();
        let scope = |label: &str| Flow::Scope {
            id: ids.node(),
            label: label.into(),
            body: Box::new(Flow::Call { id: ids.node(), label: label.into() }),
        };
        let program = Flow::Par {
            id: ids.node(),
            children: vec![scope("boom"), scope("alive")],
            policy: None,
            reducer_id: None,
        };
        let mut stepper = FlowStepper::new(&program);
        for _ in 0..32 {
            if stepper.pending().len() >= 2 {
                break;
            }
            let _ = stepper.step().expect("step to yield");
        }
        assert_eq!(stepper.pending().len(), 2);
        let boom_sid = stepper
            .pending()
            .iter()
            .find(|p| matches!(&p.reason, SuspendReason::Call { spec } if spec.label == "boom"))
            .map(|p| p.id)
            .expect("boom pending");
        let alive_sid = stepper
            .pending()
            .iter()
            .find(|p| matches!(&p.reason, SuspendReason::Call { spec } if spec.label == "alive"))
            .map(|p| p.id)
            .expect("alive pending");

        stepper
            .resolve(
                boom_sid,
                Err(FlowEffectErr {
                    code: "detonated".into(),
                    message: "boom".into(),
                }),
            )
            .expect("resolve records failure");

        // Sibling cancellation should now be queued and drainable.
        let err = stepper.step().expect_err("failfast propagates");
        match err {
            FlowError::Effect(e) => assert_eq!(e.code, "detonated"),
            FlowError::Engine(_) => panic!("expected FlowError::Effect"),
        }
        let cancels = stepper.take_cancellations();
        assert!(cancels.contains(&alive_sid), "alive sid should be cancelled");
    }

    #[test]
    fn stepper_flow_yields_and_resolves() {
        let ids = IdGen::new();
        let program = Flow::Call { id: ids.node(), label: "one".into() };
        let mut stepper = FlowStepper::new(&program);
        // First step enters and suspends.
        let out1 = stepper.step().expect("step");
        assert!(matches!(out1, StepOutcome::Blocked { .. }));
        assert_eq!(stepper.pending().len(), 1);
        let sid = stepper.pending()[0].id;
        stepper.resolve(sid, Ok(FlowValue::Text("resp".into()))).expect("resolve");
        // Drive to done.
        for _ in 0..20 {
            match stepper.step().expect("step") {
                StepOutcome::Done(FlowValue::Unit) => return,
                _ => continue,
            }
        }
        panic!("did not reach Done");
    }
}
