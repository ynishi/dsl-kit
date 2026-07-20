//! `#[tool_router]` handler that speaks to any [`DslHost`].
//!
//! The nine MCP tools are DSL-neutral: they operate on generic
//! `NodeId` / `Path` / `depth` shapes, so a caller that swaps
//! [`DslHost`] implementations sees the same contract.
//!
//! Tools:
//!
//! - `dsl_kit_info` — kit identity and the loaded DSL's summary.
//! - `dsl_kit_ast` — indented pretty-print of the AST.
//! - `dsl_kit_state` — current stepper state (depth, pending call,
//!   accumulated results, event counters, active breakpoints).
//! - `dsl_kit_step` — advance the stepper (one step, until next yield,
//!   or until completion) and return the resulting outcome.
//! - `dsl_kit_resolve` — supply a response for the currently suspended
//!   call, so the next step can continue.
//! - `dsl_kit_breakpoint_add` — add a compound breakpoint condition.
//! - `dsl_kit_breakpoint_list` — list every active breakpoint.
//! - `dsl_kit_breakpoint_remove` — remove a breakpoint by id.
//! - `dsl_kit_explain` — look up help text for a stable diagnostic code
//!   (built-in engine codes plus any the host contributes).
//! - `dsl_kit_pending` — list every live suspension (fan-out view).
//! - `dsl_kit_take_cancellations` — drain ids the engine has cancelled.
//! - `dsl_kit_resolve_by_id` — resolve a specific pending suspension by
//!   its stable id, supporting both success (`ok`) and effect-side
//!   failure (`err`) variants.
//! - `dsl_kit_reset` — reset the host's stepper.

use std::sync::Arc;

use dsl_kit::{BreakCondition, BreakpointId, BreakpointSet, NodeId, Path};
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        ListResourcesResult, PaginatedRequestParams, RawResource, ReadResourceRequestParams,
        ReadResourceResult, Resource, ResourceContents, ServerCapabilities, ServerInfo,
    },
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::host::{DslHost, HostEffectError, HostOutcome, PendingProjection};
use crate::resources::{ResourceEntry, kit_resources};

// ---------- Parameter types ---------------------------------------------

/// Parameters accepted by the `dsl_kit_step` MCP tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StepParams {
    /// How to advance the stepper. Accepted values:
    ///
    /// - `"one"` (default): a single `step()` call.
    /// - `"to_yield"`: keep stepping until the stepper suspends,
    ///   completes, or errors.
    /// - `"to_done"`: keep running (using the host's default
    ///   resolver) until the stepper reaches `Done`.
    pub mode: Option<String>,
}

/// Parameters accepted by the `dsl_kit_resolve` MCP tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ResolveParams {
    /// Response text to record against the currently suspended call.
    /// When omitted, the host's default response is used.
    pub result: Option<String>,
}

/// Parameters accepted by the `dsl_kit_breakpoint_add` MCP tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BreakpointAddParams {
    /// Match a specific node id.
    pub at_node: Option<u64>,
    /// Match every context whose depth equals this value.
    pub at_depth: Option<u32>,
    /// Match every context whose depth is at least this value.
    pub at_depth_at_least: Option<u32>,
    /// Match every context whose depth is at most this value.
    pub at_depth_at_most: Option<u32>,
    /// Match a specific iteration counter.
    pub at_iteration: Option<u64>,
    /// Match every context whose path begins with the given sequence
    /// of node ids.
    pub under_path: Option<Vec<u64>>,
}

/// Parameters accepted by the `dsl_kit_breakpoint_remove` MCP tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BreakpointRemoveParams {
    /// Breakpoint id returned by `dsl_kit_breakpoint_add`.
    pub id: u64,
}

/// Parameters accepted by the `dsl_kit_explain` MCP tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ExplainParams {
    /// Stable diagnostic code to look up, e.g. `"dsl_kit::eval::aborted"`.
    /// When omitted, the tool lists every known code without help text.
    pub code: Option<String>,
}

/// Parameters accepted by the `dsl_kit_resolve_by_id` MCP tool.
///
/// Exactly one of `ok` / `err` must be present. `ok` carries a
/// success payload as a text response; `err` carries an effect-side
/// failure with a machine-readable `code` and human message.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ResolveByIdParams {
    /// Stable suspension id returned by `dsl_kit_pending` or
    /// `dsl_kit_state`.
    pub id: u64,
    /// Success payload (as a text response). Omit when `err` is set.
    pub ok: Option<String>,
    /// Effect-side failure. Omit when `ok` is set.
    pub err: Option<ResolveErr>,
}

/// Effect-side failure body for `dsl_kit_resolve_by_id`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ResolveErr {
    /// Short machine-readable code (e.g. `"timeout"`, `"unauthorized"`).
    pub code: String,
    /// Human-readable message.
    pub message: String,
}

/// Parameters accepted by the `dsl_kit_load` MCP tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LoadParams {
    /// JSON document for the DSL to parse, build, and swap in.
    /// The internally-tagged `"type"` convention selects the variant
    /// at every level; other keys map to declared fields / child slots
    /// per the DSL's schema (see `dsl-kit-parse::serde_bridge`).
    pub input: String,
}

// ---------- Handler state -----------------------------------------------

struct HandlerState {
    host: Box<dyn DslHost>,
    breakpoints: BreakpointSet,
}

/// MCP handler that owns a single embedded DSL host and exposes it
/// to any MCP client.
#[derive(Clone)]
pub struct DslMcpHandler {
    tool_router: ToolRouter<Self>,
    state: Arc<Mutex<HandlerState>>,
    expose_kit_resources: bool,
}

impl DslMcpHandler {
    /// Builds a handler around any [`DslHost`] implementation.
    ///
    /// By default both kit-layer resources (`dsl-kit://kit/*`) and any
    /// host-contributed DSL-layer resources
    /// (`dsl-kit://dsl/*` by convention) are exposed via
    /// `list_resources` / `read_resource`. Use
    /// [`without_kit_resources`](Self::without_kit_resources) to strip
    /// the kit layer.
    pub fn new(host: Box<dyn DslHost>) -> Self {
        Self {
            tool_router: Self::tool_router(),
            state: Arc::new(Mutex::new(HandlerState { host, breakpoints: BreakpointSet::new() })),
            expose_kit_resources: true,
        }
    }

    /// Suppresses the built-in `dsl-kit://kit/*` resources.
    ///
    /// Useful for custom MCP servers that want a clean resource surface
    /// containing only their own DSL-layer entries.
    pub fn without_kit_resources(mut self) -> Self {
        self.expose_kit_resources = false;
        self
    }

    /// Returns the full resource catalogue this handler exposes, in
    /// the order it appears in `list_resources` (kit layer first, then
    /// host-contributed entries).
    pub async fn all_resources(&self) -> Vec<ResourceEntry> {
        let mut all = if self.expose_kit_resources {
            kit_resources()
        } else {
            Vec::new()
        };
        let guard = self.state.lock().await;
        all.extend(guard.host.resources());
        all
    }
}

// ---------- Tool router --------------------------------------------------

#[tool_router]
impl DslMcpHandler {
    /// Report kit identity and a one-line DSL summary.
    #[tool(name = "dsl_kit_info")]
    pub async fn dsl_kit_info(&self) -> Result<String, String> {
        let guard = self.state.lock().await;
        let host = &*guard.host;
        let body = json!({
            "kit": "dsl-kit",
            "kit_version": env!("CARGO_PKG_VERSION"),
            "dsl": host.dsl_name(),
            "dsl_root": host.root_node_id(),
            "dsl_summary": host.root_summary(),
            "ast_size": host.ast_size(),
        });
        Ok(body.to_string())
    }

    /// Return an indented text tree of the embedded program.
    #[tool(name = "dsl_kit_ast")]
    pub async fn dsl_kit_ast(&self) -> Result<String, String> {
        let guard = self.state.lock().await;
        let host = &*guard.host;
        let body = json!({
            "root": host.root_node_id(),
            "pretty": host.ast_pretty(),
        });
        Ok(body.to_string())
    }

    /// Report the stepper's current state.
    #[tool(name = "dsl_kit_state")]
    pub async fn dsl_kit_state(&self) -> Result<String, String> {
        let guard = self.state.lock().await;
        let snap = guard.host.snapshot();

        let suspended = snap
            .suspended_call
            .map(|s| json!({ "node": s.node, "label": s.label }));

        let results_json: Vec<Value> = snap
            .results
            .into_iter()
            .map(|(node, result)| json!({ "node": node, "result": result }))
            .collect();

        let bps: Vec<Value> = guard
            .breakpoints
            .iter()
            .map(|(id, cond)| json!({ "id": id.0, "condition": describe_condition(cond) }))
            .collect();

        let pending_json: Vec<Value> = snap
            .pending
            .into_iter()
            .map(|p| pending_to_json(&p))
            .collect();

        let body = json!({
            "depth": snap.depth,
            "current_path": snap.current_path,
            "suspended_call": suspended,
            "pending": pending_json,
            "results": results_json,
            "events": {
                "visit_pre": snap.events.visit_pre,
                "visit_post": snap.events.visit_post,
                "frame_enter": snap.events.frame_enter,
                "frame_leave": snap.events.frame_leave,
                "iteration_tick": snap.events.iteration_tick,
                "suspend": snap.events.suspend,
                "resume": snap.events.resume,
            },
            "breakpoints": bps,
        });
        Ok(body.to_string())
    }

    /// List every suspension the engine currently reports through
    /// `Stepper::pending`. In the one-in-flight case this echoes the
    /// same info as `dsl_kit_state.suspended_call`; under `Par`
    /// fan-out it enumerates every live child.
    #[tool(name = "dsl_kit_pending")]
    pub async fn dsl_kit_pending(&self) -> Result<String, String> {
        let guard = self.state.lock().await;
        let snap = guard.host.snapshot();
        let pending_json: Vec<Value> =
            snap.pending.into_iter().map(|p| pending_to_json(&p)).collect();
        Ok(json!({ "pending": pending_json }).to_string())
    }

    /// Drain the ids of suspensions the engine has cancelled since
    /// the last call. Hosts should invoke this after every `Ready`
    /// / `Blocked` / `Err` return from `dsl_kit_step` and act on the
    /// drained ids (typically abort their runtime handles).
    #[tool(name = "dsl_kit_take_cancellations")]
    pub async fn dsl_kit_take_cancellations(&self) -> Result<String, String> {
        let mut guard = self.state.lock().await;
        let ids = guard.host.take_cancellations();
        Ok(json!({ "cancelled": ids }).to_string())
    }

    /// Resolve a specific pending suspension by its stable id.
    /// Supports both success (`ok`) and effect-side failure (`err`)
    /// variants.
    #[tool(name = "dsl_kit_resolve_by_id")]
    pub async fn dsl_kit_resolve_by_id(
        &self,
        Parameters(params): Parameters<ResolveByIdParams>,
    ) -> Result<String, String> {
        let result = match (params.ok, params.err) {
            (Some(text), None) => Ok(text),
            (None, Some(e)) => Err(HostEffectError {
                code: e.code,
                message: e.message,
            }),
            (Some(_), Some(_)) => {
                return Err("exactly one of ok / err must be present, not both".into());
            }
            (None, None) => {
                return Err("provide one of ok (success) or err (effect failure)".into());
            }
        };
        let mut guard = self.state.lock().await;
        let resolved = guard.host.resolve_by_id(params.id, result).await?;
        Ok(json!({
            "resolved": {
                "node": resolved.node,
                "label": resolved.label,
                "result": resolved.result,
            }
        })
        .to_string())
    }

    /// Advance the stepper. See [`StepParams::mode`].
    #[tool(name = "dsl_kit_step")]
    pub async fn dsl_kit_step(
        &self,
        Parameters(params): Parameters<StepParams>,
    ) -> Result<String, String> {
        let mode = params.mode.as_deref().unwrap_or("one");
        let mut guard = self.state.lock().await;
        let HandlerState { host, breakpoints } = &mut *guard;

        let outcome = match mode {
            "one" => host.step_one(breakpoints).await,
            "to_yield" => host.step_to_yield(breakpoints).await,
            "to_done" => host.step_to_done(breakpoints).await,
            other => {
                return Err(format!(
                    "unknown mode {other:?}; use \"one\", \"to_yield\", or \"to_done\""
                ));
            }
        }?;

        Ok(outcome_to_json(&outcome).to_string())
    }

    /// Provide a response for the currently suspended call.
    #[tool(name = "dsl_kit_resolve")]
    pub async fn dsl_kit_resolve(
        &self,
        Parameters(params): Parameters<ResolveParams>,
    ) -> Result<String, String> {
        let mut guard = self.state.lock().await;
        let resolved = guard.host.resolve(params.result).await?;
        Ok(json!({
            "resolved": {
                "node": resolved.node,
                "label": resolved.label,
                "result": resolved.result,
            }
        })
        .to_string())
    }

    /// Register a compound breakpoint condition.
    #[tool(name = "dsl_kit_breakpoint_add")]
    pub async fn dsl_kit_breakpoint_add(
        &self,
        Parameters(params): Parameters<BreakpointAddParams>,
    ) -> Result<String, String> {
        let condition = build_condition(&params)?;
        let mut guard = self.state.lock().await;
        let id = guard.breakpoints.add(condition.clone());
        Ok(json!({
            "id": id.0,
            "condition": describe_condition(&condition),
        })
        .to_string())
    }

    /// List every registered breakpoint.
    #[tool(name = "dsl_kit_breakpoint_list")]
    pub async fn dsl_kit_breakpoint_list(&self) -> Result<String, String> {
        let guard = self.state.lock().await;
        let entries: Vec<Value> = guard
            .breakpoints
            .iter()
            .map(|(id, cond)| json!({ "id": id.0, "condition": describe_condition(cond) }))
            .collect();
        Ok(json!({ "entries": entries }).to_string())
    }

    /// Remove a breakpoint by id.
    #[tool(name = "dsl_kit_breakpoint_remove")]
    pub async fn dsl_kit_breakpoint_remove(
        &self,
        Parameters(params): Parameters<BreakpointRemoveParams>,
    ) -> Result<String, String> {
        let mut guard = self.state.lock().await;
        let removed = guard.breakpoints.remove(BreakpointId(params.id));
        Ok(json!({ "removed": removed }).to_string())
    }

    /// Look up the help text for a stable diagnostic code. When `code`
    /// is omitted, the full catalogue is returned (codes only).
    #[tool(name = "dsl_kit_explain")]
    pub async fn dsl_kit_explain(
        &self,
        Parameters(params): Parameters<ExplainParams>,
    ) -> Result<String, String> {
        let guard = self.state.lock().await;
        let mut entries = dsl_kit::engine_error_catalog();
        entries.extend(guard.host.catalog());

        match params.code {
            None => {
                let codes: Vec<&str> = entries.iter().map(|e| e.code.as_str()).collect();
                Ok(json!({ "codes": codes }).to_string())
            }
            Some(code) => match entries.iter().find(|e| e.code == code) {
                Some(entry) => {
                    Ok(json!({ "code": entry.code, "help": entry.help }).to_string())
                }
                None => {
                    let known: Vec<&str> = entries.iter().map(|e| e.code.as_str()).collect();
                    Err(format!(
                        "unknown error code {code:?}. Known codes: {known:?}"
                    ))
                }
            },
        }
    }

    /// Reset the host's stepper. Breakpoints are left untouched.
    #[tool(name = "dsl_kit_reset")]
    pub async fn dsl_kit_reset(&self) -> Result<String, String> {
        let mut guard = self.state.lock().await;
        guard.host.reset();
        Ok(json!({ "reset": true }).to_string())
    }

    /// Returns the loaded DSL's type-level schema as JSON.
    ///
    /// Envelope: `{ "wired": bool, "schema": <NodeSchema JSON> | null }`.
    /// `wired=false` means the host has not implemented
    /// [`DslHost::schema_json`] — the DSL is running through the MCP
    /// surface but hasn't opted into schema reflection.
    #[tool(name = "dsl_kit_schema")]
    pub async fn dsl_kit_schema(&self) -> Result<String, String> {
        let guard = self.state.lock().await;
        match guard.host.schema_json() {
            Some(text) => {
                let value: serde_json::Value = serde_json::from_str(&text)
                    .map_err(|e| format!("host schema_json is not valid JSON: {e}"))?;
                Ok(json!({ "wired": true, "schema": value }).to_string())
            }
            None => Ok(json!({ "wired": false, "schema": null }).to_string()),
        }
    }

    /// Parse a JSON document, build a typed AST, swap it into the host,
    /// and reset the stepper. Clears the handler's breakpoint set on
    /// success (old `NodeId`s are meaningless against the new AST).
    ///
    /// Envelope on success:
    /// `{ "ok": true, "dsl": <name>, "root": <root_id>, "ast_size": N }`.
    ///
    /// Envelope on failure:
    /// `{ "ok": false, "diagnostics": [<Diagnostic>, ...] }` when the
    /// host's `load_json` returned a JSON diagnostic array; otherwise
    /// `{ "ok": false, "error": <message> }` for hosts that returned a
    /// plain-text error (including the default "not supported" path).
    #[tool(name = "dsl_kit_load")]
    pub async fn dsl_kit_load(
        &self,
        Parameters(params): Parameters<LoadParams>,
    ) -> Result<String, String> {
        let mut guard = self.state.lock().await;
        match guard.host.load_json(&params.input).await {
            Ok(()) => {
                guard.breakpoints = BreakpointSet::new();
                let host = &*guard.host;
                let body = json!({
                    "ok": true,
                    "dsl": host.dsl_name(),
                    "root": host.root_node_id(),
                    "ast_size": host.ast_size(),
                });
                Ok(body.to_string())
            }
            Err(msg) => {
                // Prefer parsing the message as JSON diagnostics; fall
                // back to a plain-text `error` field for the default
                // (unsupported) and other prose-only paths.
                if let Ok(value) = serde_json::from_str::<Value>(&msg) {
                    if value.is_array() {
                        return Ok(json!({ "ok": false, "diagnostics": value }).to_string());
                    }
                }
                Ok(json!({ "ok": false, "error": msg }).to_string())
            }
        }
    }

    /// Runs the host's lint pass over the currently-loaded AST and
    /// returns the diagnostics as JSON.
    ///
    /// Envelope: `{ "wired": bool, "diagnostics": [...] | null }`.
    /// `wired=false` means the host has not implemented
    /// [`DslHost::lint_json`].
    #[tool(name = "dsl_kit_lint")]
    pub async fn dsl_kit_lint(&self) -> Result<String, String> {
        let guard = self.state.lock().await;
        match guard.host.lint_json() {
            Some(text) => {
                let value: serde_json::Value = serde_json::from_str(&text)
                    .map_err(|e| format!("host lint_json is not valid JSON: {e}"))?;
                Ok(json!({ "wired": true, "diagnostics": value }).to_string())
            }
            None => Ok(json!({ "wired": false, "diagnostics": null }).to_string()),
        }
    }
}

// ---------- ServerHandler impl -----------------------------------------

#[tool_handler]
impl ServerHandler for DslMcpHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "dsl-kit MCP server. Drives a stepper over a DSL loaded by \
                 the host, exposing traversal, breakpoints, suspend / resume, \
                 and inspection through a debugger-style tool surface.\n\n\
                 Typical workflow:\n\
                 1. dsl_kit_info + dsl_kit_ast to see the loaded program.\n\
                 2. dsl_kit_breakpoint_add to pause on interesting nodes.\n\
                 3. dsl_kit_step (mode: one | to_yield | to_done).\n\
                 4. dsl_kit_state to inspect where the stepper is.\n\
                 5. dsl_kit_resolve to supply a response for a Call, then step again.\n\
                 6. dsl_kit_reset to start over."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
            ..Default::default()
        }
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        let entries = self.all_resources().await;
        let resources: Vec<Resource> = entries
            .iter()
            .map(|e| Resource {
                raw: RawResource {
                    uri: e.uri.clone(),
                    name: e.title.clone(),
                    title: Some(e.title.clone()),
                    description: Some(e.description.clone()),
                    mime_type: Some(e.mime_type.clone()),
                    size: None,
                    icons: None,
                    meta: None,
                },
                annotations: None,
            })
            .collect();
        Ok(ListResourcesResult::with_all_items(resources))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let entries = self.all_resources().await;
        let entry = entries.iter().find(|e| e.uri == request.uri).ok_or_else(|| {
            McpError::resource_not_found(format!("unknown resource uri: {}", request.uri), None)
        })?;
        let body = entry
            .read()
            .map_err(|e| McpError::internal_error(format!("read resource: {e}"), None))?;
        Ok(ReadResourceResult {
            contents: vec![ResourceContents::text(body, entry.uri.clone())],
        })
    }
}

// ---------- Helpers ------------------------------------------------------

fn pending_to_json(p: &PendingProjection) -> Value {
    json!({
        "id": p.id,
        "reason": p.reason,
        "label": p.label,
        "at": {
            "node": p.at.node,
            "path": p.at.path,
            "depth": p.at.depth,
            "frame": p.at.frame,
            "iteration": p.at.iteration,
        },
    })
}

fn outcome_to_json(outcome: &HostOutcome) -> Value {
    match outcome {
        HostOutcome::Advanced => json!({ "kind": "advanced" }),
        HostOutcome::Suspended { reason, at } => json!({
            "kind": "suspended",
            "reason": reason,
            "at": {
                "node": at.node,
                "path": at.path,
                "depth": at.depth,
                "frame": at.frame,
                "iteration": at.iteration,
            },
        }),
        HostOutcome::Done => json!({ "kind": "done" }),
    }
}

fn build_condition(params: &BreakpointAddParams) -> Result<BreakCondition, String> {
    let mut parts: Vec<BreakCondition> = Vec::new();
    if let Some(n) = params.at_node {
        parts.push(BreakCondition::at_node(NodeId(n)));
    }
    if let Some(d) = params.at_depth {
        parts.push(BreakCondition::at_depth(d));
    }
    if let Some(d) = params.at_depth_at_least {
        parts.push(BreakCondition::at_depth_at_least(d));
    }
    if let Some(d) = params.at_depth_at_most {
        parts.push(BreakCondition::at_depth_at_most(d));
    }
    if let Some(i) = params.at_iteration {
        parts.push(BreakCondition::at_iteration(i));
    }
    if let Some(ids) = &params.under_path {
        let path = Path(ids.iter().copied().map(NodeId).collect());
        parts.push(BreakCondition::under_path(path));
    }
    match parts.len() {
        0 => Err(
            "provide at least one of: at_node, at_depth, at_depth_at_least, \
             at_depth_at_most, at_iteration, under_path"
                .into(),
        ),
        1 => Ok(parts.remove(0)),
        _ => {
            let mut iter = parts.into_iter();
            let first = iter.next().expect("non-empty");
            Ok(iter.fold(first, |acc, next| acc.and(next)))
        }
    }
}

fn describe_condition(cond: &BreakCondition) -> Value {
    match cond {
        BreakCondition::Node(id) => json!({ "kind": "node", "id": id.0 }),
        BreakCondition::PathPrefix(path) => {
            json!({ "kind": "path_prefix", "path": path.0.iter().map(|n| n.0).collect::<Vec<_>>() })
        }
        BreakCondition::PathExact(path) => {
            json!({ "kind": "path_exact", "path": path.0.iter().map(|n| n.0).collect::<Vec<_>>() })
        }
        BreakCondition::DepthAtLeast(n) => json!({ "kind": "depth_at_least", "value": n }),
        BreakCondition::DepthAtMost(n) => json!({ "kind": "depth_at_most", "value": n }),
        BreakCondition::DepthEquals(n) => json!({ "kind": "depth", "value": n }),
        BreakCondition::Iteration(n) => json!({ "kind": "iteration", "value": n }),
        BreakCondition::CallFrame(f) => json!({ "kind": "call_frame", "value": f.0 }),
        BreakCondition::Any(children) => {
            let children: Vec<Value> = children.iter().map(describe_condition).collect();
            json!({ "kind": "any", "children": children })
        }
        BreakCondition::All(children) => {
            let children: Vec<Value> = children.iter().map(describe_condition).collect();
            json!({ "kind": "all", "children": children })
        }
        BreakCondition::Not(inner) => json!({ "kind": "not", "child": describe_condition(inner) }),
        BreakCondition::Always => json!({ "kind": "always" }),
        BreakCondition::Never => json!({ "kind": "never" }),
    }
}
