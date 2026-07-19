//! A tiny arithmetic expression DSL.
//!
//! `Expr` is a second reference DSL for `dsl-kit`: it exercises the
//! kit's traversal, breakpoint, and MCP host contracts against a
//! shape that is very different from `dsl-kit-flow`. Where `Flow`
//! models an orchestration graph with `AwaitEffect` suspensions on
//! `Call` nodes, `Expr` models a pure evaluator whose only reason
//! to yield is an unbound variable: the host is asked to supply
//! its value, and evaluation continues.

use std::collections::HashMap;
use std::fmt::Write as _;

use dsl_kit::{
    BreakpointSet, DslNode, EngineError, IdGen, NodeContext, NodeId, Path, Phase, Walk,
};
use dsl_kit_mcp::host::{
    DslHost, EventCounts, HostLocation, HostOutcome, HostSnapshot, ResolvedCall, SuspendedCall,
};

/// AST of the arithmetic DSL.
#[derive(Debug, DslNode)]
pub enum Expr {
    Lit { id: NodeId, value: i64 },
    Var { id: NodeId, name: String },
    Add { id: NodeId, lhs: Box<Expr>, rhs: Box<Expr> },
    Mul { id: NodeId, lhs: Box<Expr>, rhs: Box<Expr> },
    Let {
        id: NodeId,
        name: String,
        value: Box<Expr>,
        body: Box<Expr>,
    },
    If {
        id: NodeId,
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },
}

impl Expr {
    /// One-line summary of a node's shape.
    pub fn summary(&self) -> String {
        match self {
            Expr::Lit { value, .. } => format!("Lit {value}"),
            Expr::Var { name, .. } => format!("Var {name:?}"),
            Expr::Add { .. } => "Add".into(),
            Expr::Mul { .. } => "Mul".into(),
            Expr::Let { name, .. } => format!("Let {name:?}"),
            Expr::If { .. } => "If".into(),
        }
    }
}

/// Renders an `Expr` as an indented text tree via `Walk::walk`.
pub fn pretty(expr: &Expr) -> String {
    let mut out = String::new();
    let mut depth: usize = 0;
    expr.walk(&mut |node, phase| match phase {
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

/// Locates a node's path from the root, if it exists.
fn path_to(expr: &Expr, target: NodeId) -> Option<Vec<NodeId>> {
    fn go(node: &Expr, target: NodeId, acc: &mut Vec<NodeId>) -> bool {
        acc.push(node.node_id());
        if node.node_id() == target {
            return true;
        }
        for child in node.children() {
            if go(child, target, acc) {
                return true;
            }
        }
        acc.pop();
        false
    }
    let mut acc = Vec::new();
    if go(expr, target, &mut acc) { Some(acc) } else { None }
}

/// A binding stack; `Let` shadows outer bindings for `Var` lookups.
type Env = Vec<(String, i64)>;

fn lookup(env: &Env, name: &str) -> Option<i64> {
    env.iter().rev().find(|(n, _)| n == name).map(|(_, v)| *v)
}

/// Error surfaced when evaluation hits a variable that neither the
/// program's `Let` bindings nor the host's `resolved` map has a value
/// for.
#[derive(Debug, Clone)]
struct UnboundVar {
    node: NodeId,
    name: String,
}

/// Pure evaluator over `Expr`.
///
/// External bindings (typically supplied by the host through
/// `resolve`) are consulted after the syntactic `Let` stack.
fn eval(expr: &Expr, env: &mut Env, resolved: &HashMap<String, i64>) -> Result<i64, UnboundVar> {
    match expr {
        Expr::Lit { value, .. } => Ok(*value),
        Expr::Var { id, name } => {
            if let Some(v) = lookup(env, name) {
                return Ok(v);
            }
            if let Some(v) = resolved.get(name) {
                return Ok(*v);
            }
            Err(UnboundVar { node: *id, name: name.clone() })
        }
        Expr::Add { lhs, rhs, .. } => Ok(eval(lhs, env, resolved)? + eval(rhs, env, resolved)?),
        Expr::Mul { lhs, rhs, .. } => Ok(eval(lhs, env, resolved)? * eval(rhs, env, resolved)?),
        Expr::Let { name, value, body, .. } => {
            let v = eval(value, env, resolved)?;
            env.push((name.clone(), v));
            let result = eval(body, env, resolved);
            env.pop();
            result
        }
        Expr::If { cond, then_branch, else_branch, .. } => {
            let c = eval(cond, env, resolved)?;
            if c != 0 { eval(then_branch, env, resolved) } else { eval(else_branch, env, resolved) }
        }
    }
}

fn count_nodes(expr: &Expr) -> usize {
    let mut count = 0usize;
    expr.walk(&mut |_, phase| {
        if phase == Phase::Pre {
            count += 1;
        }
    });
    count
}

// ---------- ExprHost ----------------------------------------------------

/// `DslHost` adapter around the arithmetic DSL.
///
/// The host owns the program plus a small evaluation cache: each time
/// `step_*` is called it tries to reduce the whole expression against
/// the currently known bindings. If a `Var` node has no binding, the
/// host suspends with an `AwaitEffect` reason and asks the client to
/// supply the value via `resolve`.
pub struct ExprHost {
    program: &'static Expr,
    resolved: HashMap<String, i64>,
    pending: Option<(NodeId, String)>,
    final_value: Option<i64>,
    /// One-shot "we just yielded on a breakpoint" guard; identical in
    /// spirit to `FlowStepper::breakpoint_yielded`.
    breakpoint_pending: Option<NodeId>,
    breakpoint_yielded: bool,
    /// Small counter of step attempts, reported in the snapshot.
    step_count: u32,
    /// Nodes we have already yielded on so we do not repeat the same
    /// breakpoint on retries.
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
        // For Expr the natural "pause boundary" is each unresolved
        // Var lookup. Check the pending var (if any) against the
        // registered breakpoints before advancing further.
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

        let suspended_call =
            self.pending.as_ref().map(|(node, name)| SuspendedCall { node: node.0, label: name.clone() });

        HostSnapshot {
            depth: if self.final_value.is_some() { 0 } else { 1 },
            current_path: self.pending.as_ref().and_then(|(node, _)| {
                path_to(self.program, *node).map(|ids| ids.into_iter().map(|n| n.0).collect())
            }),
            suspended_call,
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
                    // Breakpoint pause — consume the one-shot guard
                    // by advancing once, letting evaluation proceed.
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
}

impl ExprHost {
    /// Best-effort locator for the `Var` node that introduced a given
    /// resolved name; used purely to give the snapshot a stable node
    /// id for the resolution entry.
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

fn default_resolution(name: &str) -> i64 {
    match name {
        "y" => 5,
        "z" => 2,
        _ => 1,
    }
}

/// Builds the demo program:
///
/// ```text
/// let x = 3 in (x + y) * z
/// ```
///
/// With external bindings `y = 5`, `z = 2`, this evaluates to `16`.
pub fn demo_program(ids: &IdGen) -> Expr {
    let x_lit = Expr::Lit { id: ids.node(), value: 3 };
    let x_ref = Expr::Var { id: ids.node(), name: "x".into() };
    let y_ref = Expr::Var { id: ids.node(), name: "y".into() };
    let z_ref = Expr::Var { id: ids.node(), name: "z".into() };
    let add = Expr::Add { id: ids.node(), lhs: Box::new(x_ref), rhs: Box::new(y_ref) };
    let mul = Expr::Mul { id: ids.node(), lhs: Box::new(add), rhs: Box::new(z_ref) };
    Expr::Let {
        id: ids.node(),
        name: "x".into(),
        value: Box::new(x_lit),
        body: Box::new(mul),
    }
}

// ---------- Direct-eval helper for tests --------------------------------

/// Runs the evaluator to completion using an explicit resolver.
///
/// Returns `EngineError::Malformed` when an unbound variable is not
/// covered by the resolver — this is the "pure eval" mode used by the
/// example binary's synchronous demo.
pub fn evaluate_all<F>(expr: &Expr, mut resolver: F) -> Result<i64, EngineError>
where
    F: FnMut(&str) -> Option<i64>,
{
    let mut resolved: HashMap<String, i64> = HashMap::new();
    let mut env: Env = Vec::new();
    loop {
        match eval(expr, &mut env, &resolved) {
            Ok(v) => return Ok(v),
            Err(UnboundVar { node, name }) => match resolver(&name) {
                Some(v) => {
                    resolved.insert(name, v);
                }
                None => {
                    return Err(EngineError::Malformed {
                        at: NodeContext::at(node, Path::root().push(node)),
                        detail: format!("unbound variable {name:?}"),
                    });
                }
            },
        }
    }
}
