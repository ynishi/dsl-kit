//! A tiny arithmetic expression DSL.
//!
//! `Expr` is a second reference DSL for `dsl-kit`: it exercises the
//! kit's value-bearing node kinds (`Lit` / `Read` / `Apply` / `Bind` /
//! `Branch`) against a shape very different from `flow-dsl`. Where
//! `Flow` models an orchestration graph suspending on `Call` nodes,
//! `Expr` is an arithmetic language whose only reason to yield is an
//! unbound variable — which the engine surfaces as an ordinary
//! Call-shaped suspension the host answers through `resolve`.
//!
//! This crate carries the AST + its [`Ast`] adapter (the engine does
//! the evaluating); the `DslHost` adapter lives in the sibling
//! `expr-host` crate, and the MCP binary in `expr-mcp`.

#![warn(missing_docs)]

use std::fmt::Write as _;
use std::sync::Arc;

use dsl_kit::{
    DslExec, DslNode, DslSemantics, Engine, EngineError, ExecError, IdGen, NodeContext, NodeId, Op,
    OpRegistry, OwnedDerivedAst, Path, Phase, ReducerRegistry, StepOutcome, Stepper, SuspendReason,
    SuspensionId, Walk,
};

/// AST of the arithmetic DSL.
///
/// The `#[dsl_exec(...)]` annotations are the whole evaluation story:
/// `DslExec` derives the engine classification of every variant, and
/// [`ExprSemantics`] supplies the handful of judgments the kit cannot
/// derive (what "unit" is, what counts as true, how bindings store).
#[derive(
    Debug, DslNode, dsl_kit_macros::DslSchema, dsl_kit_macros::DslBuild, dsl_kit_macros::DslExec,
)]
pub enum Expr {
    /// Integer literal.
    #[dsl_exec(value)]
    Lit {
        /// Stable node id.
        id: NodeId,
        /// Literal value.
        value: i64,
    },
    /// Variable reference. Suspends the host when unbound.
    #[dsl_exec(read(name))]
    Var {
        /// Stable node id.
        id: NodeId,
        /// Variable name.
        name: String,
    },
    /// Addition.
    #[dsl_exec(apply = "add")]
    Add {
        /// Stable node id.
        id: NodeId,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// Multiplication.
    #[dsl_exec(apply = "mul")]
    Mul {
        /// Stable node id.
        id: NodeId,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// Let-binding: evaluates `value`, binds it as `name` in `body`.
    #[dsl_exec(bind(name))]
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
    #[dsl_exec(branch)]
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
    if go(expr, target, &mut acc) {
        Some(acc)
    } else {
        None
    }
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

/// Env delta introduced by one `Let`: a single binding level.
pub type Binding = Vec<(String, i64)>;

/// Effect-side error for `Expr` — what a host reports when it answers
/// an unbound-variable suspension with `Err(_)`.
#[derive(Debug, Clone)]
pub struct ExprEffectError {
    /// Human-readable description supplied by the host.
    pub message: String,
}

impl std::fmt::Display for ExprEffectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ExprEffectError {}

/// The semantic half of the DSL — the judgments `#[derive(DslExec)]`
/// cannot make for you: what "unit" is, what counts as true, and how
/// `Let` bindings store and resolve. Everything mechanical (variant →
/// `NodeKind`, node lookup, literal extraction) is derived.
pub struct ExprSemantics;

impl DslSemantics for ExprSemantics {
    type Value = i64;
    type Delta = Binding;
    type EffectError = ExprEffectError;
    type Cursor = ();

    fn unit_value(&self) -> i64 {
        0
    }

    fn truthy(&self, value: &i64) -> Option<bool> {
        Some(*value != 0)
    }

    fn bind_delta(&self, name: &str, value: &i64) -> Option<Binding> {
        Some(vec![(name.to_string(), *value)])
    }

    fn lookup(&self, delta: &Binding, name: &str) -> Option<i64> {
        delta.iter().rev().find(|(n, _)| n == name).map(|(_, v)| *v)
    }
}

/// Engine-ready [`Ast`] over `Expr`: derived classification zipped
/// with [`ExprSemantics`].
///
/// Owned projection ([`OwnedDerivedAst`]): the engine carries no borrow
/// of the `Expr` tree, so a long-lived host can own its program and
/// engine together without `Box::leak`.
pub type ExprAst = OwnedDerivedAst<<Expr as DslExec>::LitValue, ExprSemantics>;

/// A binary arithmetic op with strict arity.
struct BinOp {
    name: &'static str,
    f: fn(i64, i64) -> i64,
}

impl Op<i64> for BinOp {
    fn apply(&self, node: NodeId, args: &[i64]) -> Result<i64, EngineError> {
        match args {
            [lhs, rhs] => Ok((self.f)(*lhs, *rhs)),
            _ => Err(EngineError::EvalFailed {
                at: NodeContext::at(node, Path::root().push(node)),
                source: format!("op {:?} expects 2 args, got {}", self.name, args.len()).into(),
            }),
        }
    }
}

/// The op table for the arithmetic surface — the only place the DSL
/// author states what `Add` and `Mul` *mean*.
pub fn expr_ops() -> Arc<OpRegistry<i64>> {
    let mut ops: OpRegistry<i64> = OpRegistry::new();
    ops.register(
        "add",
        Arc::new(BinOp {
            name: "add",
            f: |a, b| a.wrapping_add(b),
        }),
    );
    ops.register(
        "mul",
        Arc::new(BinOp {
            name: "mul",
            f: |a, b| a.wrapping_mul(b),
        }),
    );
    Arc::new(ops)
}

/// Builds a fresh engine over `expr` with the standard op table.
///
/// The returned engine owns its [`OwnedDerivedAst`] projection and holds
/// no borrow of `expr`, so the result carries no lifetime.
pub fn expr_engine(expr: &Expr) -> Result<Engine<ExprAst>, EngineError> {
    Engine::new_with_ops(
        OwnedDerivedAst::new(expr, ExprSemantics),
        Arc::new(ReducerRegistry::new()),
        expr_ops(),
    )
}

/// Builds the demo program:
///
/// ```text
/// let x = 3 in (x + y) * z
/// ```
///
/// With external bindings `y = 5`, `z = 2`, this evaluates to `16`.
pub fn demo_program(ids: &IdGen) -> Expr {
    let x_lit = Expr::Lit {
        id: ids.node(),
        value: 3,
    };
    let x_ref = Expr::Var {
        id: ids.node(),
        name: "x".into(),
    };
    let y_ref = Expr::Var {
        id: ids.node(),
        name: "y".into(),
    };
    let z_ref = Expr::Var {
        id: ids.node(),
        name: "z".into(),
    };
    let add = Expr::Add {
        id: ids.node(),
        lhs: Box::new(x_ref),
        rhs: Box::new(y_ref),
    };
    let mul = Expr::Mul {
        id: ids.node(),
        lhs: Box::new(add),
        rhs: Box::new(z_ref),
    };
    Expr::Let {
        id: ids.node(),
        name: "x".into(),
        value: Box::new(x_lit),
        body: Box::new(mul),
    }
}

/// Runs `expr` to completion on the engine, supplying unbound
/// variables through `resolver`.
///
/// Returns `EngineError::Malformed` when an unbound variable is not
/// covered by the resolver — this is the "plain run" mode used by the
/// example binary's synchronous demo. The whole reduction (dispatch,
/// bindings, branching) happens inside the engine; this function only
/// answers the suspensions the engine yields.
pub fn evaluate_all<F>(expr: &Expr, mut resolver: F) -> Result<i64, EngineError>
where
    F: FnMut(&str) -> Option<i64>,
{
    let mut engine = expr_engine(expr)?;
    loop {
        match engine.step() {
            Ok(StepOutcome::Done(v)) => return Ok(v),
            Ok(StepOutcome::Ready) => continue,
            Ok(StepOutcome::Blocked { .. }) => {
                let outstanding: Vec<(SuspensionId, NodeId, String)> = engine
                    .pending()
                    .iter()
                    .filter_map(|p| match &p.reason {
                        SuspendReason::Call { spec } => Some((p.id, p.at.node, spec.label.clone())),
                        _ => None,
                    })
                    .collect();
                if outstanding.is_empty() {
                    return Err(EngineError::Malformed {
                        at: NodeContext::at(expr.node_id(), Path::root().push(expr.node_id())),
                        detail: "engine blocked without a resolvable suspension".into(),
                    });
                }
                for (sid, node, name) in outstanding {
                    match resolver(&name) {
                        Some(v) => {
                            engine.resolve(sid, Ok(v)).map_err(|e| match e {
                                ExecError::Engine(e) => e,
                                ExecError::Effect(e) => EngineError::EvalFailed {
                                    at: NodeContext::at(node, Path::root().push(node)),
                                    source: Box::new(e),
                                },
                            })?;
                        }
                        None => {
                            return Err(EngineError::Malformed {
                                at: NodeContext::at(node, Path::root().push(node)),
                                detail: format!("unbound variable {name:?}"),
                            });
                        }
                    }
                }
            }
            Err(ExecError::Engine(e)) => return Err(e),
            Err(ExecError::Effect(e)) => {
                return Err(EngineError::EvalFailed {
                    at: NodeContext::at(expr.node_id(), Path::root().push(expr.node_id())),
                    source: Box::new(e),
                });
            }
        }
    }
}
