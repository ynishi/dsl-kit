//! `DslHost` adapter around the arithmetic DSL.
//!
//! The host owns the program plus a small evaluation cache: each time
//! `step_*` is called it tries to reduce the whole expression against
//! the currently known bindings. If a `Var` node has no binding, the
//! host suspends with an `AwaitEffect` reason and asks the client to
//! supply the value via `resolve`.
//!
//! Serves as the second reference `DslHost`, alongside `flow-host`, to
//! demonstrate that the MCP handler stays generic across DSLs whose
//! evaluation strategy differs (Flow yields on `Call`, Expr yields on
//! unbound `Var`).

#![warn(missing_docs)]

use std::collections::HashMap;

use dsl_kit::{BreakpointSet, DslNode, IdGen, NodeContext, NodeId, Path, Phase, Walk};
use dsl_kit_mcp::host::{
    DslHost, EventCounts, HostLocation, HostOutcome, HostSnapshot, ResolvedCall, SuspendedCall,
};
use dsl_kit_mcp::resources::ResourceEntry;
use expr_dsl::{Env, Expr, UnboundVar, count_nodes, demo_program, eval, path_to, pretty};

const EXPR_GRAMMAR: &str = include_str!("./resources_data/grammar.md");
const EXPR_DEMO_PROGRAM: &str = include_str!("./resources_data/demo-program.md");

/// `DslHost` adapter around the arithmetic DSL.
pub struct ExprHost {
    program: &'static Expr,
    resolved: HashMap<String, i64>,
    pending: Option<(NodeId, String)>,
    final_value: Option<i64>,
    breakpoint_pending: Option<NodeId>,
    breakpoint_yielded: bool,
    step_count: u32,
    seen_breakpoint_nodes: Vec<NodeId>,
}

impl ExprHost {
    /// Builds a host around the built-in demo program.
    pub fn new_with_default_program() -> Self {
        let ids = IdGen::new();
        let program: &'static Expr = Box::leak(Box::new(demo_program(&ids)));
        Self::with_program(program)
    }

    /// Builds a host around a caller-supplied `Expr` reference.
    pub fn with_program(program: &'static Expr) -> Self {
        Self {
            program,
            resolved: HashMap::new(),
            pending: None,
            final_value: None,
            breakpoint_pending: None,
            breakpoint_yielded: false,
            step_count: 0,
            seen_breakpoint_nodes: Vec::new(),
        }
    }

    fn ctx_for(&self, node: NodeId) -> HostLocation {
        let path = path_to(self.program, node)
            .map(|ids| ids.into_iter().map(|n| n.0).collect())
            .unwrap_or_default();
        let depth = path_to(self.program, node).map(|ids| ids.len() as u32).unwrap_or(0);
        HostLocation { node: node.0, path, depth, frame: None, iteration: None }
    }

    fn check_breakpoint(&mut self, breakpoints: &BreakpointSet) -> Option<HostOutcome> {
        if self.breakpoint_yielded || breakpoints.is_empty() {
            self.breakpoint_yielded = false;
            return None;
        }
        let (node, _) = self.pending.clone()?;
        if self.seen_breakpoint_nodes.contains(&node) {
            return None;
        }
        let path = path_to(self.program, node)?;
        let ctx = NodeContext {
            node,
            path: Path(path.clone()),
            frame: None,
            depth: path.len() as u32,
            iteration: None,
        };
        if breakpoints.matches(&ctx).is_empty() {
            return None;
        }
        self.breakpoint_yielded = true;
        self.breakpoint_pending = Some(node);
        self.seen_breakpoint_nodes.push(node);
        Some(HostOutcome::Suspended {
            reason: "breakpoint".into(),
            at: HostLocation {
                node: node.0,
                path: path.into_iter().map(|n| n.0).collect(),
                depth: ctx.depth,
                frame: None,
                iteration: None,
            },
        })
    }

    fn try_evaluate(&mut self) -> Result<HostOutcome, String> {
        if self.final_value.is_some() {
            return Ok(HostOutcome::Done);
        }
        let mut env: Env = Vec::new();
        match eval(self.program, &mut env, &self.resolved) {
            Ok(v) => {
                self.final_value = Some(v);
                Ok(HostOutcome::Done)
            }
            Err(UnboundVar { node, name }) => {
                self.pending = Some((node, name.clone()));
                Ok(HostOutcome::Suspended {
                    reason: "await-effect".into(),
                    at: self.ctx_for(node),
                })
            }
        }
    }

    fn resolved_var_node(&self, target_name: &str) -> u64 {
        let mut found: Option<NodeId> = None;
        self.program.walk(&mut |node, phase| {
            if phase != Phase::Pre {
                return;
            }
            if let Expr::Var { id, name } = node {
                if name == target_name && found.is_none() {
                    found = Some(*id);
                }
            }
        });
        found.map(|id| id.0).unwrap_or(0)
    }
}

#[async_trait::async_trait]
impl DslHost for ExprHost {
    fn dsl_name(&self) -> &str {
        "expr"
    }

    fn root_node_id(&self) -> u64 {
        self.program.node_id().0
    }

    fn root_summary(&self) -> String {
        self.program.summary()
    }

    fn ast_size(&self) -> usize {
        count_nodes(self.program)
    }

    fn ast_pretty(&self) -> String {
        pretty(self.program)
    }

    fn snapshot(&self) -> HostSnapshot {
        let mut results: Vec<(u64, String)> = self
            .resolved
            .iter()
            .map(|(name, v)| (self.resolved_var_node(name), format!("{name} = {v}")))
            .collect();
        if let Some(v) = self.final_value {
            results.push((self.program.node_id().0, v.to_string()));
        }
        results.sort_by_key(|(id, _)| *id);

        let suspended_call = self
            .pending
            .as_ref()
            .map(|(node, name)| SuspendedCall { node: node.0, label: name.clone() });

        HostSnapshot {
            depth: if self.final_value.is_some() { 0 } else { 1 },
            current_path: self.pending.as_ref().and_then(|(node, _)| {
                path_to(self.program, *node).map(|ids| ids.into_iter().map(|n| n.0).collect())
            }),
            suspended_call,
            pending: Vec::new(),
            results,
            events: EventCounts {
                visit_pre: self.step_count,
                visit_post: self.step_count,
                ..EventCounts::default()
            },
        }
    }

    async fn step_one(&mut self, breakpoints: &BreakpointSet) -> Result<HostOutcome, String> {
        self.step_count += 1;
        if let Some(hit) = self.check_breakpoint(breakpoints) {
            return Ok(hit);
        }
        self.try_evaluate()
    }

    async fn step_to_yield(&mut self, breakpoints: &BreakpointSet) -> Result<HostOutcome, String> {
        self.step_one(breakpoints).await
    }

    async fn step_to_done(&mut self, breakpoints: &BreakpointSet) -> Result<HostOutcome, String> {
        let mut safety = 0u32;
        loop {
            match self.step_one(breakpoints).await? {
                HostOutcome::Advanced => {}
                HostOutcome::Suspended { reason, .. } if reason == "await-effect" => {
                    if let Some((_, name)) = self.pending.clone() {
                        let default = default_resolution(&name);
                        self.resolved.insert(name.clone(), default);
                        self.pending = None;
                    }
                }
                HostOutcome::Suspended { reason, .. } if reason == "breakpoint" => {
                    self.breakpoint_pending = None;
                }
                other => return Ok(other),
            }
            safety += 1;
            if safety > 4096 {
                return Err("expr host exceeded to_done safety limit".into());
            }
        }
    }

    async fn resolve(&mut self, result: Option<String>) -> Result<ResolvedCall, String> {
        let (node, name) = self
            .pending
            .take()
            .ok_or_else(|| "no unbound variable to resolve".to_string())?;
        let text = result.ok_or_else(|| {
            "expr resolve requires `result` as an integer literal (no default provided)".to_string()
        })?;
        let value: i64 = text
            .trim()
            .parse()
            .map_err(|e| format!("invalid integer literal {text:?}: {e}"))?;
        self.resolved.insert(name.clone(), value);
        Ok(ResolvedCall { node: node.0, label: name, result: value.to_string() })
    }

    fn reset(&mut self) {
        self.resolved.clear();
        self.pending = None;
        self.final_value = None;
        self.breakpoint_pending = None;
        self.breakpoint_yielded = false;
        self.step_count = 0;
        self.seen_breakpoint_nodes.clear();
    }

    fn resources(&self) -> Vec<ResourceEntry> {
        vec![
            ResourceEntry::static_markdown(
                "dsl-kit://dsl/expr/grammar",
                "expr DSL — grammar",
                "The six variants of the Expr enum (Lit / Var / Add / Mul / Let / If) with their semantics and the unbound-variable suspension contract.",
                EXPR_GRAMMAR,
            ),
            ResourceEntry::static_markdown(
                "dsl-kit://dsl/expr/samples/demo-program",
                "expr DSL — demo program",
                "The default program ExprHost loads: `let x = 3 in (x + y) * z`. Structure, source, and drive-to-done walkthrough.",
                EXPR_DEMO_PROGRAM,
            ),
        ]
    }
}

fn default_resolution(name: &str) -> i64 {
    match name {
        "y" => 5,
        "z" => 2,
        _ => 1,
    }
}
