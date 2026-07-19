//! `#[tool_router]` handler exposing the flow DSL to MCP clients.
//!
//! Tools:
//!
//! - `dsl_kit_info` — kit identity and a one-line DSL summary.
//! - `dsl_kit_ast` — indented pretty-print of the AST.
//! - `dsl_kit_state` — current stepper state (depth, pending call,
//!   accumulated results, event counters, active breakpoints).
//! - `dsl_kit_step` — advance the stepper (one step, until next yield,
//!   or until completion) and return the resulting outcome.
//! - `dsl_kit_resolve` — supply a response for the currently suspended
//!   `Call`, so the next step can continue.
//! - `dsl_kit_breakpoint_add` — add a compound breakpoint condition.
//! - `dsl_kit_breakpoint_list` — list every active breakpoint.
//! - `dsl_kit_breakpoint_remove` — remove a breakpoint by id.
//! - `dsl_kit_reset` — rebuild the stepper from the embedded program.

use std::sync::Arc;

use dsl_kit::{
    BreakCondition, BreakpointId, BreakpointSet, DslNode, IdGen, NodeContext, NodeId, Path, Phase,
    StepOutcome, Stepper, Walk,
};
use dsl_kit_flow::{Flow, FlowStepper, canned_response, pretty, research_pipeline};
use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

// ---------- Parameter types ---------------------------------------------

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StepParams {
    /// How to advance the stepper. Accepted values:
    ///
    /// - `"one"` (default): a single `step()` call.
    /// - `"to_yield"`: keep stepping until the stepper suspends,
    ///   completes, or errors.
    /// - `"to_done"`: keep running (looping through `to_yield` +
    ///   automatic canned-response resolution) until the stepper
    ///   reaches `Done`.
    pub mode: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ResolveParams {
    /// Response text to record against the currently suspended call.
    /// When omitted, the built-in canned response for the call's label
    /// is used.
    pub result: Option<String>,
}

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
    /// Match a specific iteration counter (Seq / Par).
    pub at_iteration: Option<u64>,
    /// Match every context whose path begins with the given sequence
    /// of node ids.
    pub under_path: Option<Vec<u64>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BreakpointRemoveParams {
    /// Breakpoint id returned by `dsl_kit_breakpoint_add`.
    pub id: u64,
}

// ---------- Handler state -----------------------------------------------

struct HandlerState {
    stepper: FlowStepper<'static>,
    breakpoints: BreakpointSet,
}

impl HandlerState {
    fn new(program: &'static Flow) -> Self {
        Self { stepper: FlowStepper::new(program), breakpoints: BreakpointSet::new() }
    }
}

/// MCP handler that owns a single embedded flow program and drives a
/// stepper over it.
#[derive(Clone)]
pub struct DslMcpHandler {
    tool_router: ToolRouter<Self>,
    program: &'static Flow,
    state: Arc<Mutex<HandlerState>>,
}

impl DslMcpHandler {
    /// Builds a handler around the default research-pipeline program.
    ///
    /// The program is allocated once and leaked so the stepper can
    /// hold a `'static` reference to it — the leak is bounded (a
    /// single program per server process).
    pub fn new_with_default_program() -> Self {
        let ids = IdGen::new();
        let program: &'static Flow = Box::leak(Box::new(research_pipeline(&ids)));
        Self::new(program)
    }

    /// Builds a handler around a caller-supplied program.
    pub fn new(program: &'static Flow) -> Self {
        Self {
            tool_router: Self::tool_router(),
            program,
            state: Arc::new(Mutex::new(HandlerState::new(program))),
        }
    }
}

// ---------- Tool router --------------------------------------------------

#[tool_router]
impl DslMcpHandler {
    /// Report kit identity and a one-line DSL summary.
    #[tool(name = "dsl_kit_info")]
    pub async fn dsl_kit_info(&self) -> Result<String, String> {
        let ast_size = count_nodes(self.program);
        let body = json!({
            "kit": "dsl-kit",
            "kit_version": env!("CARGO_PKG_VERSION"),
            "dsl": "flow",
            "dsl_root": self.program.node_id().0,
            "dsl_summary": self.program.summary(),
            "ast_size": ast_size,
        });
        Ok(body.to_string())
    }

    /// Return an indented text tree of the embedded program.
    #[tool(name = "dsl_kit_ast")]
    pub async fn dsl_kit_ast(&self) -> Result<String, String> {
        let body = json!({
            "root": self.program.node_id().0,
            "pretty": pretty(self.program),
        });
        Ok(body.to_string())
    }

    /// Report the stepper's current state: depth, active path, whether
    /// a call is pending, accumulated results, event counters, and the
    /// registered breakpoints.
    #[tool(name = "dsl_kit_state")]
    pub async fn dsl_kit_state(&self) -> Result<String, String> {
        let guard = self.state.lock().await;
        let counts = guard.stepper.events();
        let mut results: Vec<(NodeId, String)> =
            guard.stepper.results().iter().map(|(k, v)| (*k, v.clone())).collect();
        results.sort_by_key(|(id, _)| id.0);
        let results_json: Vec<Value> = results
            .into_iter()
            .map(|(id, text)| json!({ "node": id.0, "result": text }))
            .collect();

        let suspended = guard.stepper.suspended_call().map(|(id, label)| {
            json!({ "node": id.0, "label": label })
        });

        let path_json = guard.stepper.current_path().map(|p| path_ids(&p));

        let bps: Vec<Value> = guard
            .breakpoints
            .iter()
            .map(|(id, cond)| json!({ "id": id.0, "condition": describe_condition(cond) }))
            .collect();

        let body = json!({
            "depth": guard.stepper.depth(),
            "current_path": path_json,
            "suspended_call": suspended,
            "results": results_json,
            "events": {
                "visit_pre": counts.visit_pre,
                "visit_post": counts.visit_post,
                "frame_enter": counts.frame_enter,
                "frame_leave": counts.frame_leave,
                "iteration_tick": counts.iteration_tick,
                "suspend": counts.suspend,
                "resume": counts.resume,
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

        match mode {
            "one" => {
                let outcome = guard.stepper.step().map_err(|e| e.to_string())?;
                Ok(outcome_to_json(&outcome).to_string())
            }
            "to_yield" => {
                let outcome = guard.stepper.run_to_yield().map_err(|e| e.to_string())?;
                Ok(outcome_to_json(&outcome).to_string())
            }
            "to_done" => {
                let mut steps = 0u32;
                let final_outcome = loop {
                    let outcome = guard.stepper.run_to_yield().map_err(|e| e.to_string())?;
                    steps += 1;
                    match outcome {
                        StepOutcome::Suspended { .. } => {
                            if let Some((id, label)) = guard.stepper.suspended_call() {
                                let response = canned_response(label);
                                guard.stepper.record_result(id, response);
                            }
                        }
                        other => break other,
                    }
                    if steps > 4096 {
                        return Err("stepper exceeded to_done safety limit".into());
                    }
                };
                Ok(outcome_to_json(&final_outcome).to_string())
            }
            other => Err(format!("unknown mode {other:?}; use \"one\", \"to_yield\", or \"to_done\"")),
        }
    }

    /// Provide a response for the currently suspended call.
    ///
    /// When `result` is omitted the built-in canned response for the
    /// call's label is used.
    #[tool(name = "dsl_kit_resolve")]
    pub async fn dsl_kit_resolve(
        &self,
        Parameters(params): Parameters<ResolveParams>,
    ) -> Result<String, String> {
        let mut guard = self.state.lock().await;
        let (id, label) = guard
            .stepper
            .suspended_call()
            .map(|(id, label)| (id, label.to_string()))
            .ok_or_else(|| "no suspended call to resolve".to_string())?;
        let result = params.result.unwrap_or_else(|| canned_response(&label));
        guard.stepper.record_result(id, result.clone());
        Ok(json!({
            "resolved": { "node": id.0, "label": label, "result": result }
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

    /// Reset the stepper. The AST is rebuilt from scratch and the
    /// breakpoint set is left untouched.
    #[tool(name = "dsl_kit_reset")]
    pub async fn dsl_kit_reset(&self) -> Result<String, String> {
        let mut guard = self.state.lock().await;
        guard.stepper = FlowStepper::new(self.program);
        Ok(json!({ "reset": true }).to_string())
    }
}

// ---------- ServerHandler impl -----------------------------------------

#[tool_handler]
impl ServerHandler for DslMcpHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "dsl-kit MCP server. Exposes a small orchestration DSL (Seq / Par / \
                 Call / Scope / Maybe) through a debugger-style tool surface.\n\n\
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

fn count_nodes(flow: &Flow) -> usize {
    let mut count = 0usize;
    flow.walk(&mut |_, phase| {
        if phase == Phase::Pre {
            count += 1;
        }
    });
    count
}

fn path_ids(path: &Path) -> Vec<u64> {
    path.0.iter().map(|n| n.0).collect()
}

fn ctx_json(ctx: &NodeContext) -> Value {
    json!({
        "node": ctx.node.0,
        "path": path_ids(&ctx.path),
        "depth": ctx.depth,
        "frame": ctx.frame.map(|f| f.0),
        "iteration": ctx.iteration.map(|i| i.0),
    })
}

fn outcome_to_json(outcome: &StepOutcome<()>) -> Value {
    match outcome {
        StepOutcome::Advanced => json!({ "kind": "advanced" }),
        StepOutcome::Suspended { reason, at } => json!({
            "kind": "suspended",
            "reason": format!("{reason}"),
            "at": ctx_json(at),
        }),
        StepOutcome::Done(()) => json!({ "kind": "done" }),
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
            json!({ "kind": "path_prefix", "path": path_ids(path) })
        }
        BreakCondition::PathExact(path) => {
            json!({ "kind": "path_exact", "path": path_ids(path) })
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
