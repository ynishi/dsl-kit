//! A tiny arithmetic expression DSL.
//!
//! `Expr` is a second reference DSL for `dsl-kit`: it exercises the
//! kit's traversal, breakpoint, and MCP host contracts against a shape
//! very different from `flow-dsl`. Where `Flow` models an orchestration
//! graph with `AwaitEffect` suspensions on `Call` nodes, `Expr` models
//! a pure evaluator whose only reason to yield is an unbound variable:
//! the host is asked to supply its value, and evaluation continues.
//!
//! This crate carries the AST + the pure evaluator; the `DslHost`
//! adapter lives in the sibling `expr-host` crate, and the MCP binary
//! in `expr-mcp`.

#![warn(missing_docs)]

use std::collections::HashMap;
use std::fmt::Write as _;

use dsl_kit::{
    DslNode, EngineError, IdGen, NodeContext, NodeId, Path, Phase, Walk,
};

/// AST of the arithmetic DSL.
#[derive(Debug, DslNode)]
pub enum Expr {
    /// Integer literal.
    Lit {
        /// Stable node id.
        id: NodeId,
        /// Literal value.
        value: i64,
    },
    /// Variable reference. Suspends the host when unbound.
    Var {
        /// Stable node id.
        id: NodeId,
        /// Variable name.
        name: String,
    },
    /// Addition.
    Add {
        /// Stable node id.
        id: NodeId,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// Multiplication.
    Mul {
        /// Stable node id.
        id: NodeId,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// Let-binding: evaluates `value`, binds it as `name` in `body`.
    Let {
        /// Stable node id.
        id: NodeId,
        /// Bound name.
        name: String,
        /// Expression whose value is bound.
        value: Box<Expr>,
        /// Body evaluated with `name` bound.
        body: Box<Expr>,
    },
    /// Conditional: non-zero `cond` picks `then_branch`, else `else_branch`.
    If {
        /// Stable node id.
        id: NodeId,
        /// Condition expression.
        cond: Box<Expr>,
        /// Branch taken when `cond` is non-zero.
        then_branch: Box<Expr>,
        /// Branch taken otherwise.
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

/// Locates the root-to-node id chain for `target`, if reachable.
pub fn path_to(expr: &Expr, target: NodeId) -> Option<Vec<NodeId>> {
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

/// Counts every node in the AST (pre-order visits).
pub fn count_nodes(expr: &Expr) -> usize {
    let mut count = 0usize;
    expr.walk(&mut |_, phase| {
        if phase == Phase::Pre {
            count += 1;
        }
    });
    count
}

/// A binding stack; `Let` shadows outer bindings for `Var` lookups.
pub type Env = Vec<(String, i64)>;

fn lookup(env: &Env, name: &str) -> Option<i64> {
    env.iter().rev().find(|(n, _)| n == name).map(|(_, v)| *v)
}

/// Error surfaced when evaluation hits a variable that neither the
/// program's `Let` bindings nor the host's `resolved` map has a value
/// for.
#[derive(Debug, Clone)]
pub struct UnboundVar {
    /// Node id of the unbound `Var`.
    pub node: NodeId,
    /// Name that failed to resolve.
    pub name: String,
}

/// Pure evaluator over `Expr`.
///
/// External bindings (typically supplied by the host through `resolve`)
/// are consulted after the syntactic `Let` stack.
pub fn eval(
    expr: &Expr,
    env: &mut Env,
    resolved: &HashMap<String, i64>,
) -> Result<i64, UnboundVar> {
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
            if c != 0 {
                eval(then_branch, env, resolved)
            } else {
                eval(else_branch, env, resolved)
            }
        }
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
