//! DSL-agnostic host trait for the MCP server.
//!
//! `DslMcpHandler` speaks to any DSL implementation that provides a
//! [`DslHost`] — the trait is deliberately small and expressed in terms
//! of the engine's uniform primitives (`NodeId` / `Path` / `depth` /
//! iteration / frame), so agents see the same tool contract regardless
//! of which DSL is loaded.
//!
//! To wire a new DSL, implement `DslHost` for a struct that owns the
//! program plus a stepper, then hand it to
//! `DslMcpHandler::new(Box::new(my_host))`. The reference `flow_host`
//! module in this crate shows the shape.

/// One event kind's counter, as reported by [`DslHost::snapshot`].
#[derive(Debug, Clone, Copy, Default)]
pub struct EventCounts {
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

/// Snapshot of the pending `Call` (or equivalent effect node) the
/// stepper is currently suspended on.
#[derive(Debug, Clone)]
pub struct SuspendedCall {
    /// Node id of the pending call.
    pub node: u64,
    /// Host-defined label identifying the effect.
    pub label: String,
}

/// A call that has just been resolved.
#[derive(Debug, Clone)]
pub struct ResolvedCall {
    /// Node id whose call was resolved.
    pub node: u64,
    /// Host-defined label identifying the effect.
    pub label: String,
    /// Value the host supplied for the call, serialised as text.
    pub result: String,
}

/// Effect-side failure surfaced by the host via
/// [`DslHost::resolve_by_id`]. JSON-friendly mirror of the DSL's
/// `EffectError` associated type.
#[derive(Debug, Clone)]
pub struct HostEffectError {
    /// Short machine-readable code.
    pub code: String,
    /// Human-readable message.
    pub message: String,
}

/// A full snapshot of the host's stepper state.
///
/// Everything the MCP `dsl_kit_state` tool returns comes from here;
/// hosts do not have to worry about JSON shape.
#[derive(Debug, Clone)]
pub struct HostSnapshot {
    /// Current stack depth of the stepper (0 when idle / finished).
    pub depth: usize,
    /// Path to the currently active node, when the stepper is active.
    pub current_path: Option<Vec<u64>>,
    /// Details of the pending call, when the stepper is suspended
    /// on a single-in-flight call. When multiple suspensions are
    /// live (fan-out), see [`pending`](Self::pending).
    pub suspended_call: Option<SuspendedCall>,
    /// All currently live suspensions. In the common one-in-flight
    /// case this has zero or one entry; under a `Par` fan-out it may
    /// carry N entries. Order matches DFS pre-order over the frame
    /// tree.
    pub pending: Vec<PendingProjection>,
    /// Results recorded so far, keyed by node id.
    pub results: Vec<(u64, String)>,
    /// Cumulative event counters.
    pub events: EventCounts,
}

/// One live suspension in the projection view.
///
/// JSON-friendly mirror of the engine's `Pending` type. Cancellation
/// runtime handles are held on the host side (see
/// [`DslHost::take_cancellations`]) and are not included here.
#[derive(Debug, Clone)]
pub struct PendingProjection {
    /// Stable engine-assigned id.
    pub id: u64,
    /// Short human-readable reason.
    pub reason: String,
    /// Effect label for `Call`-shaped suspensions; empty otherwise.
    pub label: String,
    /// Location context.
    pub at: HostLocation,
}

/// Location context of a suspension, in generic (JSON-friendly) form.
#[derive(Debug, Clone)]
pub struct HostLocation {
    /// Node id at which the suspension happened.
    pub node: u64,
    /// Root-to-node id chain leading to `node`.
    pub path: Vec<u64>,
    /// Stack depth at the suspension point.
    pub depth: u32,
    /// Active call frame at the suspension point.
    pub frame: Option<u64>,
    /// Iteration counter when the surrounding node is loop-shaped.
    pub iteration: Option<u64>,
}

/// Outcome of a step against a `DslHost`.
#[derive(Debug, Clone)]
pub enum HostOutcome {
    /// The stepper advanced one node.
    Advanced,
    /// The stepper is suspended and awaiting resolution.
    Suspended {
        /// Reason for the yield (e.g. `"await-effect"`, `"breakpoint"`).
        reason: String,
        /// Where the suspension happened.
        at: HostLocation,
    },
    /// Evaluation completed.
    Done,
}

/// Error message returned by the default [`DslHost::resolve`] for
/// call-less DSLs. Hosts whose DSL has no external calls inherit this
/// via the trait default; hosts with real effects override `resolve`
/// (and typically report [`DslHost::supports_calls`] as `true`).
pub const RESOLVE_UNSUPPORTED_MSG: &str =
    "resolve not supported: this host's DSL has no external calls";

/// Builds the standard "step budget exceeded" error surfaced by the
/// default [`DslHost::step_to_done`] when a host advances past its
/// [`step_budget`](DslHost::step_budget) without reaching a terminal
/// outcome. Centralised so the wording stays pinned in one place.
fn step_budget_exceeded_msg(budget: usize) -> String {
    format!("step_to_done exceeded step budget of {budget}")
}

/// DSL-agnostic surface the MCP handler drives.
///
/// The step / resolve methods are `async` so hosts whose semantics
/// need to await external work (network calls, tool invocations,
/// MCP round-trips) can do so directly. Purely synchronous hosts
/// simply wrap sync bodies in `async { … }` at zero runtime cost.
///
/// Call-less (declarative) DSLs need only implement the required
/// methods: [`resolve`](Self::resolve) and
/// [`step_to_done`](Self::step_to_done) carry defaults suited to a host
/// that never suspends on an external effect. The default `resolve`
/// returns [`RESOLVE_UNSUPPORTED_MSG`]; the default `step_to_done`
/// drives [`step_to_yield`](Self::step_to_yield) up to
/// [`step_budget`](Self::step_budget) (`4096`) iterations, returning on
/// `Done` / `Suspended` and otherwise erroring with the standard
/// budget-exceeded message. Such hosts typically override
/// [`supports_calls`](Self::supports_calls) to return `false`.
///
/// A host is long-lived by nature (it owns its program and engine for
/// the whole MCP session, and `load` replaces the program repeatedly),
/// so build the engine over an owned AST — [`dsl_kit::OwnedDerivedAst`]
/// for derive-based DSLs, or a hand-written projection like the flow
/// example's `FlowAst` — rather than leaking the program with
/// `Box::leak` to satisfy a borrow-based `Ast`. The owned projection
/// borrows the tree only during construction, so `program: N` and the
/// engine can live in the same struct and `reset` / `load` re-project
/// without accumulating memory.
///
/// The trait uses [`async_trait`] to stay `dyn`-compatible; the MCP
/// handler holds a `Box<dyn DslHost>`.
#[async_trait::async_trait]
pub trait DslHost: Send + Sync {
    /// Short name of the DSL, e.g. `"flow"`. Reported by `dsl_kit_info`.
    fn dsl_name(&self) -> &str;

    /// Stable id of the root node.
    fn root_node_id(&self) -> u64;

    /// One-line summary of the root node (e.g. `"Seq"`).
    fn root_summary(&self) -> String;

    /// Number of nodes in the AST.
    fn ast_size(&self) -> usize;

    /// Indented pretty-print of the AST.
    fn ast_pretty(&self) -> String;

    /// A full snapshot of the stepper state.
    fn snapshot(&self) -> HostSnapshot;

    /// Whether this host's DSL can suspend on external calls.
    ///
    /// The default is `true` (the host may hit `Call`-shaped effects and
    /// use [`resolve`](Self::resolve) /
    /// [`resolve_by_id`](Self::resolve_by_id)). Call-less (declarative)
    /// hosts override this to `false`; the MCP handler then gates the
    /// resolve tools behind a clear [`RESOLVE_UNSUPPORTED_MSG`] error
    /// instead of driving a resolver.
    fn supports_calls(&self) -> bool {
        true
    }

    /// Maximum [`step_to_yield`](Self::step_to_yield) iterations the
    /// default [`step_to_done`](Self::step_to_done) will run before
    /// bailing out with a budget-exceeded error. Defaults to `4096`,
    /// matching the reference host's safety limit. Override to widen or
    /// tighten the guard.
    fn step_budget(&self) -> usize {
        4096
    }

    /// Run one step. If `breakpoints` is non-empty, the host is
    /// expected to yield `Suspended { reason: "breakpoint", .. }`
    /// before executing a node whose context matches any registered
    /// condition.
    async fn step_one(
        &mut self,
        breakpoints: &dsl_kit::BreakpointSet,
    ) -> Result<HostOutcome, String>;

    /// Run steps until the next suspend / done / error.
    async fn step_to_yield(
        &mut self,
        breakpoints: &dsl_kit::BreakpointSet,
    ) -> Result<HostOutcome, String>;

    /// Run to completion.
    ///
    /// Hosts with external calls override this to resolve suspensions
    /// with a host-defined default (typically canned responses) so the
    /// stepper keeps making progress; breakpoints are honoured mid-run
    /// and resolution is performed on breakpoint yields too.
    ///
    /// The default implementation is written purely in terms of
    /// [`step_to_yield`](Self::step_to_yield) and suits call-less DSLs:
    /// it drives the stepper forward on `Advanced`, returns immediately
    /// on `Done` or `Suspended` (preserving breakpoint / suspend
    /// semantics identical to `step_to_yield`), and bails out with the
    /// standard budget-exceeded error after
    /// [`step_budget`](Self::step_budget) iterations without reaching a
    /// terminal outcome.
    async fn step_to_done(
        &mut self,
        breakpoints: &dsl_kit::BreakpointSet,
    ) -> Result<HostOutcome, String> {
        let budget = self.step_budget();
        for _ in 0..budget {
            match self.step_to_yield(breakpoints).await? {
                HostOutcome::Done => return Ok(HostOutcome::Done),
                outcome @ HostOutcome::Suspended { .. } => return Ok(outcome),
                HostOutcome::Advanced => continue,
            }
        }
        Err(step_budget_exceeded_msg(budget))
    }

    /// Resolve the currently suspended call.
    ///
    /// When `result` is `None`, the host provides a default (usually
    /// its canned response for the call's label). Used by the
    /// single-in-flight legacy path — for fan-out, use
    /// [`resolve_by_id`](Self::resolve_by_id).
    ///
    /// The default implementation returns [`RESOLVE_UNSUPPORTED_MSG`],
    /// which is the correct behaviour for a call-less DSL that never
    /// suspends on an external effect. Hosts with real effects override
    /// this (and report [`supports_calls`](Self::supports_calls) as
    /// `true`).
    async fn resolve(&mut self, _result: Option<String>) -> Result<ResolvedCall, String> {
        Err(RESOLVE_UNSUPPORTED_MSG.to_string())
    }

    /// Resolve a specific pending suspension by its stable id.
    ///
    /// `result` carries the success payload (`Ok(text)`) or an
    /// effect-side failure (`Err(HostEffectError)`). Hosts convert
    /// the text into their DSL-specific `Value` type and route the
    /// error into the engine's FailFast / CollectAll policy path.
    ///
    /// The default implementation returns
    /// `Err("resolve_by_id not implemented")`. Hosts that want to
    /// expose fan-out via MCP override it.
    async fn resolve_by_id(
        &mut self,
        _id: u64,
        _result: Result<String, HostEffectError>,
    ) -> Result<ResolvedCall, String> {
        Err("resolve_by_id not implemented".into())
    }

    /// Drains the ids of suspensions the engine has cancelled since
    /// the last drain (typically the losing legs of an `Any` /
    /// `FirstK` policy fold, or the siblings of a FailFast failure).
    /// Default returns an empty vector.
    fn take_cancellations(&mut self) -> Vec<u64> {
        Vec::new()
    }

    /// Reset the stepper to a fresh state.
    fn reset(&mut self);

    /// Host-specific error catalogue entries appended to the built-in
    /// [`dsl_kit::engine_error_catalog`] when a client calls
    /// `dsl_kit_explain`. The default returns an empty vector.
    fn catalog(&self) -> Vec<dsl_kit::ErrorCatalogEntry> {
        Vec::new()
    }

    /// DSL-layer MCP resources this host contributes.
    ///
    /// These entries are for AI or humans **writing programs in** the
    /// loaded DSL (grammar references, sample programs, tool
    /// extensions). The recommended URI prefix is
    /// [`crate::DSL_URI_PREFIX`] (`dsl-kit://dsl/`) but any URI is
    /// accepted. The default returns an empty vector.
    ///
    /// Kit-layer entries (`dsl-kit://kit/*`) are supplied by
    /// [`crate::kit_resources`] and merged separately by the handler /
    /// builder; hosts do not need to include them here.
    fn resources(&self) -> Vec<crate::ResourceEntry> {
        Vec::new()
    }

    /// Type-level shape of the DSL as a JSON document (mirror of
    /// `dsl_kit_schema::NodeSchema::to_json`).
    ///
    /// Hosts whose DSL derives `DslSchema` return the serialized
    /// schema; hosts that have not wired schema reflection return
    /// `None` (the default), and the corresponding MCP tool reports
    /// the DSL as schema-less rather than fabricating a value.
    fn schema_json(&self) -> Option<String> {
        None
    }

    /// Lint diagnostics against the currently-loaded AST as a JSON
    /// array (`[{"rule": .., "severity": .., "node": .., "message": ..}, …]`).
    ///
    /// Hosts wire this by running a
    /// [`dsl_kit_lint::Linter`](https://docs.rs/dsl-kit-lint) over
    /// their AST and serializing the diagnostics. Hosts that opt out
    /// return `None` (the default), and the tool reports the DSL as
    /// lint-less.
    fn lint_json(&self) -> Option<String> {
        None
    }

    /// Parse `input` (JSON serialized `ParseTree` — see the serde
    /// bridge in `dsl-kit-parse`), build a typed AST, swap it in, and
    /// reset the stepper.
    ///
    /// The default returns `Err("load_json not supported by this host")`
    /// so that DSL hosts that have not opted into the parse trunk still
    /// implement the trait unchanged. Hosts that link `dsl-kit-parse`
    /// override this with:
    ///
    /// 1. Feed `input` through
    ///    `dsl_kit_parse::serde_bridge::from_json_str`.
    /// 2. Call `<Ast as dsl_kit_parse::DslBuild>::from_parse_tree`
    ///    using a fresh `IdGen`.
    /// 3. Replace the currently-owned AST and reset the internal
    ///    stepper. `NodeId`s start over at 0.
    ///
    /// On failure, the returned string is a serialized JSON array of
    /// [`Diagnostic`](https://docs.rs/dsl-kit-parse) envelopes so the
    /// caller gets machine-readable feedback in the same dialect used
    /// by conformance and lint. The
    /// [`dsl_kit_load`](crate::DslMcpHandler::dsl_kit_load) tool passes
    /// this through unchanged and additionally clears the handler's
    /// breakpoint set on success (old `NodeId`s are meaningless
    /// against the new AST).
    async fn load_json(&mut self, _input: &str) -> Result<(), String> {
        Err("load_json not supported by this host".into())
    }

    /// Like [`DslHost::load_json`], but with a named-sources bundle
    /// for `$import` resolution (see `dsl-kit-parse`'s `import`
    /// module).
    ///
    /// `sources_json` is a JSON object mapping source names to
    /// single-key `{"json": "…"}` / `{"text": "…"}` objects — the
    /// exact shape `dsl_kit_parse::import::MapResolver::from_sources_json`
    /// consumes. Hosts that opt in typically:
    ///
    /// 1. Build a `MapResolver` from `sources_json`.
    /// 2. Run `dsl_kit_parse::import::Loader` over `input` (wiring
    ///    their grammar via `with_grammar` if text sources should be
    ///    accepted).
    /// 3. Build the typed AST from the linked tree, swap it in, reset
    ///    the stepper.
    ///
    /// On success the returned string is a host-produced JSON report —
    /// by convention `{"dependencies": [<id>, …], "digest": "<hex>"}`
    /// from `Loaded::dependencies` / `Loaded::digest` — which the
    /// [`dsl_kit_load`](crate::DslMcpHandler::dsl_kit_load) tool
    /// embeds under `"imports"` in its success envelope. On failure
    /// the same serialized-diagnostics contract as
    /// [`DslHost::load_json`] applies. The handler stays a pure
    /// conduit: it never parses sources itself.
    async fn load_json_bundle(
        &mut self,
        _input: &str,
        _sources_json: &str,
    ) -> Result<String, String> {
        Err("load_json_bundle not supported by this host".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsl_kit::BreakpointSet;

    /// Behaviour script controlling the stub's `step_to_yield`.
    enum YieldScript {
        /// Return `Advanced` `n` times, then `Done` forever.
        AdvanceThenDone(usize),
        /// Return `Suspended` on the very first call.
        SuspendNow,
        /// Return `Advanced` forever (never terminates on its own).
        AdvanceForever,
    }

    /// Minimal call-less host: implements only the required trait
    /// methods and inherits the defaults for `resolve`, `step_to_done`,
    /// `supports_calls`, and `step_budget`.
    struct MinimalHost {
        script: YieldScript,
        /// Number of `step_to_yield` invocations observed.
        calls: usize,
    }

    impl MinimalHost {
        fn new(script: YieldScript) -> Self {
            Self { script, calls: 0 }
        }
    }

    fn test_location() -> HostLocation {
        HostLocation {
            node: 0,
            path: Vec::new(),
            depth: 0,
            frame: None,
            iteration: None,
        }
    }

    #[async_trait::async_trait]
    impl DslHost for MinimalHost {
        fn dsl_name(&self) -> &str {
            "minimal"
        }

        fn root_node_id(&self) -> u64 {
            0
        }

        fn root_summary(&self) -> String {
            "Minimal".into()
        }

        fn ast_size(&self) -> usize {
            1
        }

        fn ast_pretty(&self) -> String {
            "Minimal".into()
        }

        fn snapshot(&self) -> HostSnapshot {
            HostSnapshot {
                depth: 0,
                current_path: None,
                suspended_call: None,
                pending: Vec::new(),
                results: Vec::new(),
                events: EventCounts::default(),
            }
        }

        async fn step_one(&mut self, breakpoints: &BreakpointSet) -> Result<HostOutcome, String> {
            self.step_to_yield(breakpoints).await
        }

        async fn step_to_yield(
            &mut self,
            _breakpoints: &BreakpointSet,
        ) -> Result<HostOutcome, String> {
            self.calls += 1;
            let outcome = match self.script {
                YieldScript::AdvanceThenDone(n) => {
                    if self.calls > n {
                        HostOutcome::Done
                    } else {
                        HostOutcome::Advanced
                    }
                }
                YieldScript::SuspendNow => HostOutcome::Suspended {
                    reason: "await-effect".into(),
                    at: test_location(),
                },
                YieldScript::AdvanceForever => HostOutcome::Advanced,
            };
            Ok(outcome)
        }

        fn reset(&mut self) {
            self.calls = 0;
        }
    }

    #[test]
    fn defaults_reflect_call_capable_host() {
        let host = MinimalHost::new(YieldScript::AdvanceForever);
        assert!(host.supports_calls());
        assert_eq!(host.step_budget(), 4096);
    }

    #[tokio::test]
    async fn default_resolve_reports_unsupported() {
        let mut host = MinimalHost::new(YieldScript::AdvanceThenDone(0));
        let err = host
            .resolve(None)
            .await
            .expect_err("call-less host must reject resolve");
        assert_eq!(err, RESOLVE_UNSUPPORTED_MSG);
    }

    #[tokio::test]
    async fn default_step_to_done_advances_then_done() {
        let bps = BreakpointSet::new();
        let mut host = MinimalHost::new(YieldScript::AdvanceThenDone(3));
        let outcome = host.step_to_done(&bps).await.expect("reaches done");
        assert!(matches!(outcome, HostOutcome::Done));
        // 3 Advanced + 1 Done = 4 step_to_yield calls.
        assert_eq!(host.calls, 4);
    }

    #[tokio::test]
    async fn default_step_to_done_returns_on_suspend() {
        let bps = BreakpointSet::new();
        let mut host = MinimalHost::new(YieldScript::SuspendNow);
        let outcome = host.step_to_done(&bps).await.expect("suspends");
        assert!(matches!(outcome, HostOutcome::Suspended { .. }));
        // The loop must stop on the first suspension, not keep spinning.
        assert_eq!(host.calls, 1);
    }

    #[tokio::test]
    async fn default_step_to_done_exhausts_budget() {
        let bps = BreakpointSet::new();
        let mut host = MinimalHost::new(YieldScript::AdvanceForever);
        let budget = host.step_budget();
        let err = host
            .step_to_done(&bps)
            .await
            .expect_err("never-terminating host must exhaust the budget");
        assert_eq!(err, step_budget_exceeded_msg(budget));
        // Budget is the number of step_to_yield iterations attempted.
        assert_eq!(host.calls, budget);
    }
}
