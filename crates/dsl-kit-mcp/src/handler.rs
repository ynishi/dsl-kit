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
//! - `dsl_kit_reset` — reset the host's stepper.

use std::sync::Arc;

use dsl_kit::{BreakCondition, BreakpointId, BreakpointSet, NodeId, Path};
use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::host::{DslHost, HostOutcome};

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
}

impl DslMcpHandler {
    /// Builds a handler around any [`DslHost`] implementation.
    pub fn new(host: Box<dyn DslHost>) -> Self {
        Self {
            tool_router: Self::tool_router(),
            state: Arc::new(Mutex::new(HandlerState { host, breakpoints: BreakpointSet::new() })),
        }
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

        let body = json!({
            "depth": snap.depth,
            "current_path": snap.current_path,
            "suspended_call": suspended,
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

    /// Reset the host's stepper. Breakpoints are left untouched.
    #[tool(name = "dsl_kit_reset")]
    pub async fn dsl_kit_reset(&self) -> Result<String, String> {
        let mut guard = self.state.lock().await;
        guard.host.reset();
        Ok(json!({ "reset": true }).to_string())
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
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

// ---------- Helpers ------------------------------------------------------

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
