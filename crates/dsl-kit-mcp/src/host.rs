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

/// DSL-agnostic surface the MCP handler drives.
///
/// The step / resolve methods are `async` so hosts whose semantics
/// need to await external work (network calls, tool invocations,
/// MCP round-trips) can do so directly. Purely synchronous hosts
/// simply wrap sync bodies in `async { … }` at zero runtime cost.
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

    /// Run to completion, resolving suspensions with a host-defined
    /// default (typically canned responses). Breakpoints are honoured
    /// mid-run — they suspend the loop just like an `AwaitEffect`, and
    /// resolution is performed on breakpoint yields too so the stepper
    /// keeps making progress.
    async fn step_to_done(
        &mut self,
        breakpoints: &dsl_kit::BreakpointSet,
    ) -> Result<HostOutcome, String>;

    /// Resolve the currently suspended call.
    ///
    /// When `result` is `None`, the host provides a default (usually
    /// its canned response for the call's label). Used by the
    /// single-in-flight legacy path — for fan-out, use
    /// [`resolve_by_id`](Self::resolve_by_id).
    async fn resolve(&mut self, result: Option<String>) -> Result<ResolvedCall, String>;

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
}
