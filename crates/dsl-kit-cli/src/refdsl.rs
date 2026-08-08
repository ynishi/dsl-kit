//! Built-in reference DSL served by `dsl-kit-cli mcp`.
//!
//! `Ref` is a tiny arithmetic language — `Lit` / `Var` / `Add` / `Mul`
//! / `Let` / `If` — annotated with `#[dsl_exec(...)]` so `#[derive(DslExec)]`
//! generates the engine classification, `DslSemantics` supplies the
//! handful of value-side judgments, and the engine does the rest. It
//! exists so `cargo install dsl-kit-cli && dsl-kit-cli mcp` gives a
//! developer a working stepper without writing a host first.
//!
//! For a real DSL, the pattern is the same as `flow-example` /
//! `expr-example`: a crate holding the enum + semantics + host, and a
//! thin `main.rs` wrapping it in `DslMcpHandler`. Nothing here should
//! be treated as production; when your host lands, wire yours in and
//! drop this reference.

use std::sync::Arc;

use dsl_kit::{
    BreakpointSet, DslExec, DslNode, DslSemantics, Engine, EngineError, IdGen, NodeContext, NodeId,
    Op, OpRegistry, OwnedDerivedAst, Path, Pending, ReducerRegistry, StepOutcome, Stepper,
};
use dsl_kit_mcp::host::{
    DslHost, EventCounts, HostLocation, HostOutcome, HostSnapshot, PendingProjection, ResolvedCall,
    SuspendedCall,
};
use dsl_kit_mcp::resources::ResourceEntry;

/// AST of the built-in reference DSL.
#[derive(
    Debug, DslNode, dsl_kit_macros::DslSchema, dsl_kit_macros::DslBuild, dsl_kit_macros::DslExec,
)]
pub enum Ref {
    /// Integer literal.
    #[dsl_exec(value)]
    Lit {
        /// Stable node id.
        id: NodeId,
        /// Literal value.
        value: i64,
    },
    /// Variable reference — suspends the host when unbound.
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
        lhs: Box<Ref>,
        /// Right operand.
        rhs: Box<Ref>,
    },
    /// Multiplication.
    #[dsl_exec(apply = "mul")]
    Mul {
        /// Stable node id.
        id: NodeId,
        /// Left operand.
        lhs: Box<Ref>,
        /// Right operand.
        rhs: Box<Ref>,
    },
    /// Let-binding.
    #[dsl_exec(bind(name))]
    Let {
        /// Stable node id.
        id: NodeId,
        /// Bound name.
        name: String,
        /// Value expression.
        value: Box<Ref>,
        /// Body evaluated with `name` bound.
        body: Box<Ref>,
    },
    /// Conditional — non-zero `cond` picks `then_branch`.
    #[dsl_exec(branch)]
    If {
        /// Stable node id.
        id: NodeId,
        /// Condition.
        cond: Box<Ref>,
        /// Taken when `cond` is non-zero.
        then_branch: Box<Ref>,
        /// Taken otherwise.
        else_branch: Box<Ref>,
    },
}

impl Ref {
    fn summary(&self) -> String {
        match self {
            Ref::Lit { value, .. } => format!("Lit {value}"),
            Ref::Var { name, .. } => format!("Var {name:?}"),
            Ref::Add { .. } => "Add".into(),
            Ref::Mul { .. } => "Mul".into(),
            Ref::Let { name, .. } => format!("Let {name:?}"),
            Ref::If { .. } => "If".into(),
        }
    }
}

fn pretty(root: &Ref) -> String {
    use std::fmt::Write as _;
    fn go(node: &Ref, out: &mut String, depth: usize) {
        for _ in 0..depth {
            out.push_str("  ");
        }
        let _ = writeln!(out, "{} {}", node.node_id(), node.summary());
        // Direct-Walk descent, no re-import needed.
        use dsl_kit::Walk;
        for child in node.children() {
            go(child, out, depth + 1);
        }
    }
    let mut out = String::new();
    go(root, &mut out, 0);
    out
}

fn count_nodes(root: &Ref) -> usize {
    fn go(node: &Ref, n: &mut usize) {
        *n += 1;
        use dsl_kit::Walk;
        for c in node.children() {
            go(c, n);
        }
    }
    let mut n = 0usize;
    go(root, &mut n);
    n
}

/// Env delta = one binding level (innermost last).
pub type Binding = Vec<(String, i64)>;

/// Effect error surfaced when a host answers a `Var` lookup with `Err`.
#[derive(Debug, Clone)]
pub struct RefEffectError {
    /// Human-readable message from the host.
    pub message: String,
}

impl std::fmt::Display for RefEffectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RefEffectError {}

/// Value-side judgments the derive cannot supply on its own.
pub struct RefSemantics;

impl DslSemantics for RefSemantics {
    type Value = i64;
    type Delta = Binding;
    type EffectError = RefEffectError;
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

/// Engine-ready `Ast` over `Ref`.
///
/// Owned projection (see [`OwnedDerivedAst`]): the engine holds no borrow
/// of the `Ref` tree, so [`RefHost`] can own its program and engine in
/// the same struct without `Box::leak`.
pub type RefAst = OwnedDerivedAst<<Ref as DslExec>::LitValue, RefSemantics>;

/// Binary arithmetic op.
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

fn ops() -> Arc<OpRegistry<i64>> {
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

fn engine(root: &Ref) -> Result<Engine<RefAst>, EngineError> {
    Engine::new_with_ops(
        OwnedDerivedAst::new(root, RefSemantics),
        Arc::new(ReducerRegistry::new()),
        ops(),
    )
}

/// Demo program: `let x = 3 in if x then (x + y) * z else 0`.
///
/// Exercises `Bind` / `Read` / `Branch` / `Apply` / `Lit` in one
/// tree — every value-bearing `NodeKind` this crate touches, plus a
/// pair of unbound `Var`s (`y`, `z`) so the MCP client gets to see
/// the Call-shaped suspension and the standard `resolve` path.
pub fn demo_program(ids: &IdGen) -> Ref {
    let x_lit = Ref::Lit {
        id: ids.node(),
        value: 3,
    };
    let x_ref = Ref::Var {
        id: ids.node(),
        name: "x".into(),
    };
    let y_ref = Ref::Var {
        id: ids.node(),
        name: "y".into(),
    };
    let z_ref = Ref::Var {
        id: ids.node(),
        name: "z".into(),
    };
    let cond = Ref::Var {
        id: ids.node(),
        name: "x".into(),
    };
    let then_branch = Ref::Mul {
        id: ids.node(),
        lhs: Box::new(Ref::Add {
            id: ids.node(),
            lhs: Box::new(x_ref),
            rhs: Box::new(y_ref),
        }),
        rhs: Box::new(z_ref),
    };
    let else_branch = Ref::Lit {
        id: ids.node(),
        value: 0,
    };
    let if_expr = Ref::If {
        id: ids.node(),
        cond: Box::new(cond),
        then_branch: Box::new(then_branch),
        else_branch: Box::new(else_branch),
    };
    Ref::Let {
        id: ids.node(),
        name: "x".into(),
        value: Box::new(x_lit),
        body: Box::new(if_expr),
    }
}

// ---------- DslHost adapter --------------------------------------------

const REF_DSL_OVERVIEW: &str = r#"# dsl-kit-cli built-in reference DSL

A tiny arithmetic language — `Lit` / `Var` / `Add` / `Mul` / `Let` / `If`
— shipped so `cargo install dsl-kit-cli && dsl-kit-cli mcp` gives a
developer a working stepper without writing a host first.

Wiring: the enum is annotated with `#[dsl_exec(...)]`; `#[derive(DslExec)]`
generates the engine classification, `RefSemantics` supplies value-side
judgments (`unit_value` / `truthy` / `bind_delta` / `lookup`), and the
engine does the rest — dispatch, the binding chain, value-dependent
branching, and unbound-variable suspension all happen inside
`dsl-kit-core`. An unbound `Var` suspends as a Call-shaped pending
labelled with the variable name; the host answers through `resolve`.

The default program is `let x = 3 in if x then (x + y) * z else 0`.
With `y = 5, z = 2` it completes to `16`.
"#;

/// `DslHost` adapter around the reference DSL — an [`Engine`] wrapper
/// symmetric to `expr-host`, so the MCP surface stays generic.
pub struct RefHost {
    program: Ref,
    engine: Engine<RefAst>,
    resolved_log: Vec<(u64, String)>,
    final_value: Option<i64>,
}

impl RefHost {
    /// Build a host around the built-in demo program.
    pub fn new_with_default_program() -> Self {
        let ids = IdGen::new();
        Self::with_program(demo_program(&ids))
    }

    /// Build a host that owns `program` outright. The engine projects the
    /// tree into owned storage ([`OwnedDerivedAst`]), so no `Box::leak`
    /// is needed to keep program and engine together.
    fn with_program(program: Ref) -> Self {
        let engine = engine(&program).expect("reference program validates");
        Self {
            program,
            engine,
            resolved_log: Vec::new(),
            final_value: None,
        }
    }

    fn record_done(&mut self, outcome: &StepOutcome<i64>) {
        if let StepOutcome::Done(v) = outcome {
            self.final_value = Some(*v);
        }
    }
}

#[async_trait::async_trait]
impl DslHost for RefHost {
    fn dsl_name(&self) -> &str {
        "ref"
    }

    fn root_node_id(&self) -> u64 {
        self.program.node_id().0
    }

    fn root_summary(&self) -> String {
        self.program.summary()
    }

    fn ast_size(&self) -> usize {
        count_nodes(&self.program)
    }

    fn ast_pretty(&self) -> String {
        pretty(&self.program)
    }

    fn snapshot(&self) -> HostSnapshot {
        let counts = self.engine.events();
        let mut results = self.resolved_log.clone();
        if let Some(v) = self.final_value {
            results.push((self.program.node_id().0, v.to_string()));
        }
        results.sort_by_key(|(id, _)| *id);

        let suspended_call = SuspendedCall::sole(self.engine.pending());

        let pending: Vec<PendingProjection> = self
            .engine
            .pending()
            .iter()
            .map(PendingProjection::of)
            .collect();

        HostSnapshot {
            depth: self.engine.depth(),
            current_path: self
                .engine
                .current_path()
                .map(|p| p.0.iter().map(|n| n.0).collect()),
            suspended_call,
            pending,
            results,
            events: EventCounts {
                visit_pre: counts.visit_pre,
                visit_post: counts.visit_post,
                frame_enter: counts.frame_enter,
                frame_leave: counts.frame_leave,
                iteration_tick: counts.iteration_tick,
                suspend: counts.suspend,
                resume: counts.resume,
            },
        }
    }

    async fn step_one(&mut self, breakpoints: &BreakpointSet) -> Result<HostOutcome, String> {
        let outcome = self
            .engine
            .step_with_breakpoints(breakpoints)
            .map_err(|e| e.to_string())?;
        self.record_done(&outcome);
        Ok(step_outcome_to_host(outcome, self.engine.pending()))
    }

    async fn step_to_yield(&mut self, breakpoints: &BreakpointSet) -> Result<HostOutcome, String> {
        let outcome = self
            .engine
            .run_to_yield_with_breakpoints(breakpoints)
            .map_err(|e| e.to_string())?;
        self.record_done(&outcome);
        Ok(step_outcome_to_host(outcome, self.engine.pending()))
    }

    async fn step_to_done(&mut self, breakpoints: &BreakpointSet) -> Result<HostOutcome, String> {
        let mut steps = 0u32;
        loop {
            let outcome = self
                .engine
                .run_to_yield_with_breakpoints(breakpoints)
                .map_err(|e| e.to_string())?;
            self.record_done(&outcome);
            match outcome {
                StepOutcome::Done(_) => return Ok(HostOutcome::Done),
                StepOutcome::Ready => return Ok(HostOutcome::Advanced),
                StepOutcome::Blocked { .. } => {
                    let outstanding: Vec<(dsl_kit::SuspensionId, u64, String)> = self
                        .engine
                        .pending()
                        .iter()
                        .filter_map(|p| match &p.reason {
                            dsl_kit::SuspendReason::Call { spec } => {
                                Some((p.id, p.at.node.0, spec.label.clone()))
                            }
                            _ => None,
                        })
                        .collect();
                    for (sid, node, name) in outstanding {
                        let value = default_resolution(&name);
                        self.engine
                            .resolve(sid, Ok(value))
                            .map_err(|e| e.to_string())?;
                        self.resolved_log.push((node, format!("{name} = {value}")));
                    }
                }
            }
            steps += 1;
            if steps > 4096 {
                return Err("dsl-kit-cli reference host exceeded to_done safety limit".into());
            }
        }
    }

    async fn resolve(&mut self, result: Option<String>) -> Result<ResolvedCall, String> {
        let (sid, node_id, name) = self
            .engine
            .suspended_call()
            .map(|(sid, id, label)| (sid, id, label.to_string()))
            .ok_or_else(|| "no unbound variable to resolve".to_string())?;
        let text = result.ok_or_else(|| {
            "resolve requires `result` as an integer literal (no default provided)".to_string()
        })?;
        let value: i64 = text
            .trim()
            .parse()
            .map_err(|e| format!("invalid integer literal {text:?}: {e}"))?;
        self.engine
            .resolve(sid, Ok(value))
            .map_err(|e| e.to_string())?;
        self.resolved_log
            .push((node_id.0, format!("{name} = {value}")));
        Ok(ResolvedCall {
            node: node_id.0,
            label: name,
            result: value.to_string(),
        })
    }

    fn reset(&mut self) {
        self.engine = engine(&self.program).expect("reference program validates");
        self.resolved_log.clear();
        self.final_value = None;
    }

    fn resources(&self) -> Vec<ResourceEntry> {
        vec![ResourceEntry::static_markdown(
            "dsl-kit://cli/ref-dsl",
            "dsl-kit-cli — built-in reference DSL",
            "Shape, semantics, and default program of the reference DSL that dsl-kit-cli's `mcp` subcommand serves.",
            REF_DSL_OVERVIEW,
        )]
    }

    fn schema_json(&self) -> Option<String> {
        use dsl_kit_schema::DslSchema;
        Some(Ref::schema().to_json().to_string())
    }

    fn lint_json(&self) -> Option<String> {
        use dsl_kit_lint::Linter;
        let diagnostics = Linter::<Ref>::with_defaults().lint(&self.program);
        let value: Vec<serde_json::Value> = diagnostics
            .into_iter()
            .map(|d| {
                serde_json::json!({
                    "rule": d.rule,
                    "severity": format!("{:?}", d.severity),
                    "node": d.node.0,
                    "message": d.message,
                })
            })
            .collect();
        Some(serde_json::Value::Array(value).to_string())
    }

    async fn load_json(&mut self, input: &str) -> Result<(), String> {
        use dsl_kit_parse::{DslBuild, serde_bridge::from_json_str};
        use dsl_kit_schema::DslSchema;
        let tree = from_json_str(input, &Ref::schema()).map_err(|e| e.to_json().to_string())?;
        let ids = IdGen::new();
        let program = Ref::from_parse_tree(&tree, &ids).map_err(|e| e.to_json().to_string())?;
        // Owned program: replace the previous one outright (the old `Ref`
        // is dropped here) and re-project via `reset` — no `Box::leak`.
        self.program = program;
        self.reset();
        Ok(())
    }
}

fn step_outcome_to_host(outcome: StepOutcome<i64>, pending: &[Pending]) -> HostOutcome {
    match outcome {
        StepOutcome::Done(_) => HostOutcome::Done,
        StepOutcome::Ready => HostOutcome::Advanced,
        StepOutcome::Blocked { newly_pending } => {
            let reference = newly_pending.first().or_else(|| pending.first());
            match reference {
                Some(p) => HostOutcome::Suspended {
                    reason: p.reason.to_string(),
                    at: HostLocation::of(&p.at),
                },
                None => HostOutcome::Suspended {
                    reason: "waiting".into(),
                    at: HostLocation {
                        node: 0,
                        path: Vec::new(),
                        depth: 0,
                        frame: None,
                        iteration: None,
                    },
                },
            }
        }
    }
}

fn default_resolution(name: &str) -> i64 {
    match name {
        "x" => 3,
        "y" => 5,
        "z" => 2,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_program_completes_to_sixteen() {
        // let x = 3 in if x then (x + y) * z else 0 with y=5, z=2 → 16.
        let mut host = RefHost::new_with_default_program();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let out = host.step_to_done(&BreakpointSet::new()).await.unwrap();
            assert!(matches!(out, HostOutcome::Done), "got {out:?}");
        });
        let snap = host.snapshot();
        let final_value = snap
            .results
            .iter()
            .find(|(id, _)| *id == host.program.node_id().0)
            .map(|(_, s)| s.clone())
            .expect("final value present");
        assert_eq!(final_value, "16");
    }
}
