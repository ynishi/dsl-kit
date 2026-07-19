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
    pub visit_pre: u32,
    pub visit_post: u32,
    pub frame_enter: u32,
    pub frame_leave: u32,
    pub iteration_tick: u32,
    pub suspend: u32,
    pub resume: u32,
}

/// Snapshot of the pending `Call` (or equivalent effect node) the
/// stepper is currently suspended on.
#[derive(Debug, Clone)]
pub struct SuspendedCall {
    pub node: u64,
    pub label: String,
}

/// A call that has just been resolved.
#[derive(Debug, Clone)]
pub struct ResolvedCall {
    pub node: u64,
    pub label: String,
    pub result: String,
}

/// A full snapshot of the host's stepper state.
///
/// Everything the MCP `dsl_kit_state` tool returns comes from here;
/// hosts do not have to worry about JSON shape.
#[derive(Debug, Clone)]
pub struct HostSnapshot {
    pub depth: usize,
    pub current_path: Option<Vec<u64>>,
    pub suspended_call: Option<SuspendedCall>,
    pub results: Vec<(u64, String)>,
    pub events: EventCounts,
}

/// Location context of a suspension, in generic (JSON-friendly) form.
#[derive(Debug, Clone)]
pub struct HostLocation {
    pub node: u64,
    pub path: Vec<u64>,
    pub depth: u32,
    pub frame: Option<u64>,
    pub iteration: Option<u64>,
}

/// Outcome of a step against a `DslHost`.
#[derive(Debug, Clone)]
pub enum HostOutcome {
    Advanced,
    Suspended { reason: String, at: HostLocation },
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
    /// its canned response for the call's label).
    async fn resolve(&mut self, result: Option<String>) -> Result<ResolvedCall, String>;

    /// Reset the stepper to a fresh state.
    fn reset(&mut self);
}
