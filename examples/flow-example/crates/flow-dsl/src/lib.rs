//! Reference flow DSL for `dsl-kit` (v3 engine shape).
//!
//! Defines a small orchestration DSL — `Seq`, `Par`, `Call`, `Scope`,
//! `Maybe` — plus the reducers and a value taxonomy the engine uses to
//! interpret it.
//!
//! ## Round 17 note — Engine-driven walker
//!
//! Prior rounds (13–15) carried a private in-crate walker (`FlowStepper`
//! with its own `Vec<InternalFrame>` stack, `ParContext`, and
//! `SubtreeState` fan-out shadow state) because the core [`Engine`] did
//! not yet own the frame tree. Round 16 landed the real walker in
//! `dsl-kit-core::engine`; the flow-dsl walker was left in place as a
//! duplicate that satisfied the same v3 [`Stepper`] contract via a
//! parallel implementation.
//!
//! Round 17 removes that duplication: [`FlowStepper`] is now a thin
//! adapter that owns an `Engine<FlowAst<'_>>` and forwards every
//! [`Stepper`] method to it. Fan-out, cancellation queueing, event
//! emission, and breakpoint peek-ahead all live on the engine (design
//! doc §5). The flow-dsl crate owns only the intent side: the AST, the
//! value / error taxonomy, and the concrete reducer instances the
//! registry lookups return.

#![warn(missing_docs)]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;

use dsl_kit::{
    Ast, BreakpointSet, CountingSink, DslNode, Engine, EngineError, EngineResult, ExecError,
    FailPolicy, FrameTree, IdGen, JoinPolicy, JoinShape, NodeContext, NodeId, NodeKind, Path,
    Phase, Pending, Reducer, ReducerCollectAll, ReducerId, ReducerRegistry, StepOutcome, Stepper,
    SuspendReason, SuspensionId, Walk,
};

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
    /// Runs its children concurrently. The join policy and reducer
    /// govern how their slots fold into the parent value.
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
    /// Denotes an external effect; the engine yields a `Pending` and
    /// resumes once the host has resolved the matching `SuspensionId`.
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

// ---------- FlowAst — Ast impl for the engine ---------------------------

/// [`Ast`] adapter wrapping a borrowed [`Flow`] tree.
///
/// Builds a `NodeId → &Flow` lookup at construction so `node_kind`
/// answers in O(1). Used by [`Engine`] as the intent-side oracle; the
/// engine owns runtime mechanics.
pub struct FlowAst<'a> {
    root: &'a Flow,
    lookup: HashMap<NodeId, &'a Flow>,
}

impl<'a> FlowAst<'a> {
    /// Build an `Ast` view over `root`, indexing every reachable node.
    pub fn new(root: &'a Flow) -> Self {
        let mut lookup = HashMap::new();
        Self::index(root, &mut lookup);
        Self { root, lookup }
    }

    fn index(flow: &'a Flow, out: &mut HashMap<NodeId, &'a Flow>) {
        out.insert(flow.node_id(), flow);
        match flow {
            Flow::Seq { children, .. } | Flow::Par { children, .. } => {
                for c in children {
                    Self::index(c, out);
                }
            }
            Flow::Scope { body, .. } => Self::index(body, out),
            Flow::Maybe { body: Some(b), .. } => Self::index(b, out),
            Flow::Maybe { body: None, .. } | Flow::Call { .. } => {}
        }
    }
}

impl<'a> Ast for FlowAst<'a> {
    type Value = FlowValue;
    type Delta = ();
    type EffectError = FlowEffectErr;
    type Cursor = FlowCursor;

    fn root(&self) -> NodeId {
        self.root.node_id()
    }

    fn node_kind(&self, id: NodeId) -> NodeKind {
        let flow = self
            .lookup
            .get(&id)
            .copied()
            .unwrap_or_else(|| panic!("unknown NodeId {id:?}"));
        match flow {
            Flow::Seq { children, .. } => NodeKind::Seq {
                children: children.iter().map(|c| c.node_id()).collect(),
            },
            Flow::Par {
                children,
                policy,
                reducer_id,
                ..
            } => {
                let policy = policy.unwrap_or(JoinPolicy {
                    shape: JoinShape::All,
                    fail: FailPolicy::FailFast,
                });
                let reducer_id = ReducerId::from(
                    reducer_id.clone().unwrap_or_else(|| "reduce_all_ordered".into()),
                );
                NodeKind::Par {
                    children: children.iter().map(|c| c.node_id()).collect(),
                    policy,
                    reducer_id,
                }
            }
            Flow::Scope { label, body, .. } => NodeKind::Scope {
                label: label.clone(),
                body: body.node_id(),
            },
            Flow::Maybe { body, .. } => NodeKind::Maybe {
                body: body.as_deref().map(|b| b.node_id()),
            },
            Flow::Call { label, .. } => NodeKind::Call {
                label: label.clone(),
                payload: serde_json::Value::Null,
            },
        }
    }

    fn unit_value(&self) -> FlowValue {
        FlowValue::Unit
    }
}

// ---------- Flow value / error / cursor types --------------------------

/// Value type produced by the flow DSL.
///
/// Individual `Call` responses arrive as `Text(String)`. Reducer output
/// uses `List`. `Unit` is what the whole interpretation completes with
/// (per-call results live on the [`FlowStepper::results`] projection).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum FlowValue {
    /// No meaningful value (top-level completion marker).
    #[default]
    Unit,
    /// A textual effect response.
    Text(String),
    /// A list of nested values (used by reducer output).
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

/// Top-level error type for the flow interpretation. Mirrors
/// [`dsl_kit::ExecError`] shape so the [`Stepper`] contract composes.
#[derive(Debug, thiserror::Error)]
pub enum FlowError {
    /// An engine-level error (from `dsl-kit-core`).
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// An effect-side failure surfaced via `resolve` and propagated by
    /// the enclosing `Par` under `FailPolicy::FailFast`.
    #[error(transparent)]
    Effect(#[from] FlowEffectErr),
}

impl From<ExecError<FlowEffectErr>> for FlowError {
    fn from(e: ExecError<FlowEffectErr>) -> Self {
        match e {
            ExecError::Engine(e) => FlowError::Engine(e),
            ExecError::Effect(e) => FlowError::Effect(e),
        }
    }
}

/// Per-frame cursor surfaced through the projected [`FrameTree`]. The
/// flow DSL does not carry per-node cursor state; this is a
/// zero-sized placeholder.
#[derive(Debug, Clone, Default)]
pub struct FlowCursor;

// ---------- FlowStepper — thin adapter over Engine<FlowAst> ------------

/// Stepper over a `Flow` program.
///
/// Owns an [`Engine`] anchored at a [`FlowAst`] and forwards every
/// [`Stepper`] trait method to it. The adapter maintains one piece of
/// its own state: a `NodeId → String` map of successful `Call` results,
/// populated on each `resolve(_, Ok(_))` and surfaced through
/// [`FlowStepper::results`] for debugger snapshots. Everything else —
/// pending list, cancellation drain, frame tree projection, event
/// counter, breakpoint peek-ahead, fan-out fold — lives on the engine.
pub struct FlowStepper<'a> {
    engine: Engine<FlowAst<'a>>,
    /// Successful `Call` results keyed by node id. The engine records
    /// values on `Frame::Value` leaves inside the arena but consumes
    /// them when the enclosing `Seq` advances; this map preserves them
    /// for debugger snapshots (design doc §11 "results are debugger
    /// state, not part of the completion value").
    results: HashMap<NodeId, String>,
}

impl<'a> FlowStepper<'a> {
    /// Creates a fresh stepper anchored at `root`, using the default
    /// reducer registry.
    pub fn new(root: &'a Flow) -> Self {
        Self::with_registry(root, Arc::new(flow_default_registry()))
    }

    /// Creates a fresh stepper anchored at `root` with a caller-supplied
    /// reducer registry.
    pub fn with_registry(
        root: &'a Flow,
        registry: Arc<ReducerRegistry<FlowValue, (), FlowEffectErr>>,
    ) -> Self {
        let engine = Engine::new(FlowAst::new(root), registry)
            .expect("root Ast validation");
        Self {
            engine,
            results: HashMap::new(),
        }
    }

    // ---- Debugger conveniences (projections over the engine) --------

    /// Breakpoint-aware step. Yields
    /// `StepOutcome::Blocked { newly_pending: [Pending { reason: Breakpoint }] }`
    /// once when the engine is about to spawn a frame whose context
    /// matches any registered condition. See
    /// [`Engine::step_with_breakpoints`] for the exact contract.
    pub fn step_with_breakpoints(
        &mut self,
        bps: &BreakpointSet,
    ) -> Result<StepOutcome<FlowValue>, FlowError> {
        self.engine.step_with_breakpoints(bps).map_err(FlowError::from)
    }

    /// Loops [`Self::step_with_breakpoints`] until it returns anything
    /// other than [`StepOutcome::Ready`].
    pub fn run_to_yield_with_breakpoints(
        &mut self,
        bps: &BreakpointSet,
    ) -> Result<StepOutcome<FlowValue>, FlowError> {
        self.engine
            .run_to_yield_with_breakpoints(bps)
            .map_err(FlowError::from)
    }

    /// If the engine is currently blocked on a single `Call` suspension
    /// (the non-fan-out case), returns `(SuspensionId, NodeId, label)`.
    /// Returns `None` under a `Par` fan-out (multiple pending) or when
    /// no `Call` is live.
    pub fn suspended_call(&self) -> Option<(SuspensionId, NodeId, &str)> {
        self.engine.suspended_call()
    }

    /// Successful `Call` results recorded so far, keyed by the AST
    /// node id of the resolved `Call`.
    pub fn results(&self) -> &HashMap<NodeId, String> {
        &self.results
    }

    /// Cumulative event counter snapshot.
    pub fn events(&self) -> CountingSink {
        *self.engine.events()
    }

    /// One-line summary of the events observed so far
    /// ([`CountingSink::summarise`]).
    pub fn event_summary(&self) -> String {
        self.engine.event_summary()
    }

    /// Depth of the currently-active leaf (`0` when idle / finished).
    pub fn depth(&self) -> usize {
        self.engine.depth()
    }

    /// Path to the currently-active leaf, if any.
    pub fn current_path(&self) -> Option<Path> {
        self.engine.current_path()
    }
}

// ---------- Value → text projection ------------------------------------

/// Renders a [`FlowValue`] as the debugger's text projection, used to
/// key [`FlowStepper::results`]. `Text` maps verbatim; `Unit` maps to
/// the empty string; `List` maps to the debug rendering.
fn flow_value_to_text(v: &FlowValue) -> String {
    match v {
        FlowValue::Text(s) => s.clone(),
        FlowValue::Unit => String::new(),
        FlowValue::List(items) => format!("{items:?}"),
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
        self.engine.step().map_err(FlowError::from)
    }

    fn resolve(
        &mut self,
        id: SuspensionId,
        result: Result<Self::Value, Self::EffectError>,
    ) -> Result<(), Self::Error> {
        // Debugger-view bookkeeping: record the successful text into
        // `results` BEFORE delegating so the engine's Pending → Value
        // transition can consume the value verbatim.
        if let Ok(v) = &result {
            if let Some(p) = self.engine.pending().iter().find(|p| p.id == id) {
                self.results.insert(p.at.node, flow_value_to_text(v));
            }
        }
        self.engine.resolve(id, result).map_err(FlowError::from)
    }

    fn pending(&self) -> &[Pending] {
        self.engine.pending()
    }

    fn take_cancellations(&mut self) -> Vec<SuspensionId> {
        self.engine.take_cancellations()
    }

    fn frame_tree(
        &self,
    ) -> &FrameTree<Self::Value, Self::Cursor, Self::Delta, Self::EffectError> {
        self.engine.frame_tree()
    }

    fn is_done(&self) -> bool {
        self.engine.is_done()
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
        _winners: &[dsl_kit::ChildIndex],
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
        winners: &[dsl_kit::ChildIndex],
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
        winners: &[dsl_kit::ChildIndex],
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
        _winners: &[dsl_kit::ChildIndex],
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
        winners: &[dsl_kit::ChildIndex],
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
        winners: &[dsl_kit::ChildIndex],
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
/// suspension with the corresponding canned response. Used by demos and
/// as the top-level smoke test in the flow-dsl crate.
pub fn run_flow_sync(program: &Flow) -> Result<HashMap<NodeId, String>, FlowError> {
    let mut stepper = FlowStepper::new(program);
    let bp = BreakpointSet::new();
    let mut steps = 0u32;
    loop {
        match stepper.run_to_yield_with_breakpoints(&bp)? {
            StepOutcome::Done(_) => break,
            StepOutcome::Ready => {
                // `run_to_yield_with_breakpoints` never returns `Ready`
                // by contract, but treat as a no-op just in case.
            }
            StepOutcome::Blocked { .. } => {
                // Resolve every outstanding Call pending with its
                // canned response. Non-Call pendings (breakpoints /
                // cooperative yields) are unreachable here because the
                // BreakpointSet is empty and the flow DSL never emits
                // cooperative yields on its own.
                let outstanding: Vec<(SuspensionId, String)> = stepper
                    .pending()
                    .iter()
                    .filter_map(|p| match &p.reason {
                        SuspendReason::Call { spec } => Some((p.id, spec.label.clone())),
                        _ => None,
                    })
                    .collect();
                if outstanding.is_empty() {
                    // Blocked with no Call pendings — nothing to feed;
                    // treat as termination to avoid an infinite loop.
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
    Ok(stepper.results().clone())
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

        // First step: enters the Par and emits 3 Pending.
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
        assert!(
            matches!(out2, StepOutcome::Blocked { newly_pending } if newly_pending.is_empty())
        );

        stepper
            .resolve(sids[0], Ok(FlowValue::Text("a-resp".into())))
            .expect("resolve a");
        stepper
            .resolve(sids[1], Ok(FlowValue::Text("b-resp".into())))
            .expect("resolve b");
        assert_eq!(stepper.pending().len(), 0);

        // Drive until Done. Par folds via reduce_all_ordered.
        for _ in 0..20 {
            if let StepOutcome::Done(_) = stepper.step().expect("step") {
                assert!(!stepper.results().is_empty(), "per-call results recorded");
                return;
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
            if let StepOutcome::Done(_) = stepper.step().expect("step") {
                assert!(stepper.take_cancellations().is_empty());
                return;
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

        // Fail both slots.
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
        // Par of two Seq children, each Seq has two Calls. Real fan-out
        // (both Seqs advance concurrently) but each Seq is internally
        // sequential — pending count is at most one per Seq at a time.
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

        let second_sids: Vec<SuspensionId> =
            stepper.pending().iter().map(|p| p.id).collect();
        for sid in second_sids {
            stepper
                .resolve(sid, Ok(FlowValue::Text("ok".into())))
                .expect("resolve second-wave");
        }
        for _ in 0..128 {
            if let StepOutcome::Done(_) = stepper.step().expect("step to done") {
                assert_eq!(stepper.results().len(), 4);
                return;
            }
        }
        panic!("did not reach Done");
    }

    #[test]
    fn par_of_scope_wrapping_call_fans_out() {
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
        let out1 = stepper.step().expect("step");
        assert!(matches!(out1, StepOutcome::Blocked { .. }));
        assert_eq!(stepper.pending().len(), 1);
        let sid = stepper.pending()[0].id;
        stepper.resolve(sid, Ok(FlowValue::Text("resp".into()))).expect("resolve");
        for _ in 0..20 {
            if let StepOutcome::Done(FlowValue::Text(s)) = stepper.step().expect("step") {
                assert_eq!(s, "resp");
                return;
            }
        }
        panic!("did not reach Done");
    }

    #[test]
    fn breakpoint_yields_before_matching_seq_child() {
        // Root Seq of two Calls; BP on the first Call's node id should
        // yield a synthetic Breakpoint before the Call spawns.
        let ids = IdGen::new();
        let first_call_id = ids.node();
        let second_call_id = ids.node();
        let root_id = ids.node();
        let program = Flow::Seq {
            id: root_id,
            children: vec![
                Flow::Call { id: first_call_id, label: "one".into() },
                Flow::Call { id: second_call_id, label: "two".into() },
            ],
        };
        let mut stepper = FlowStepper::new(&program);
        let mut bps = BreakpointSet::new();
        bps.add(dsl_kit::BreakCondition::at_node(first_call_id));

        // First step: yields on BP (synthetic pending with reason
        // Breakpoint).
        let out = stepper.run_to_yield_with_breakpoints(&bps).expect("step");
        match out {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 1);
                assert!(matches!(newly_pending[0].reason, SuspendReason::Breakpoint));
                assert_eq!(newly_pending[0].at.node, first_call_id);
            }
            other => panic!("expected Blocked{{Breakpoint}}, got {other:?}"),
        }

        // Second step: consumes the BP marker and yields on the real
        // Call.
        let out2 = stepper.run_to_yield_with_breakpoints(&bps).expect("step");
        match out2 {
            StepOutcome::Blocked { newly_pending } => {
                assert_eq!(newly_pending.len(), 1);
                assert!(matches!(newly_pending[0].reason, SuspendReason::Call { .. }));
                assert_eq!(newly_pending[0].at.node, first_call_id);
            }
            other => panic!("expected Blocked{{Call}}, got {other:?}"),
        }
    }
}
