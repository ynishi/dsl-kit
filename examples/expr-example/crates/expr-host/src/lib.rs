//! `DslHost` adapter around the arithmetic DSL.
//!
//! The host wraps a [`dsl_kit::Engine`] over [`ExprAst`] — the same
//! engine that runs `flow-dsl`. There is no evaluator here: dispatch,
//! `Let` bindings, `If` branching, and unbound-variable suspension all
//! happen inside the engine, and this adapter only projects engine
//! state into the MCP host shape.
//!
//! Serves as the second reference `DslHost`, alongside `flow-host`, to
//! demonstrate that the MCP handler stays generic across DSLs whose
//! programs differ (Flow yields on `Call` effects, Expr yields on
//! unbound `Var` reads — both are Call-shaped suspensions).

#![warn(missing_docs)]

use dsl_kit::{BreakpointSet, DslNode, Engine, IdGen, Pending, StepOutcome, Stepper};
use dsl_kit_mcp::host::{
    DslHost, EventCounts, HostLocation, HostOutcome, HostSnapshot, PendingProjection, ResolvedCall,
    SuspendedCall,
};
use dsl_kit_mcp::resources::ResourceEntry;
use expr_dsl::{Expr, ExprAst, count_nodes, demo_program, expr_engine, pretty};

const EXPR_GRAMMAR: &str = include_str!("./resources_data/grammar.md");
const EXPR_DEMO_PROGRAM: &str = include_str!("./resources_data/demo-program.md");

/// `DslHost` adapter around the arithmetic DSL.
pub struct ExprHost {
    program: Expr,
    engine: Engine<ExprAst>,
    /// Resolution history projected into `HostSnapshot::results`.
    resolved_log: Vec<(u64, String)>,
    final_value: Option<i64>,
}

impl ExprHost {
    /// Builds a host around the built-in demo program.
    pub fn new_with_default_program() -> Self {
        let ids = IdGen::new();
        Self::with_program(demo_program(&ids))
    }

    /// Builds a host that owns a caller-supplied `Expr` program.
    ///
    /// The engine projects the tree into owned storage
    /// ([`ExprAst`] = `OwnedDerivedAst`), so the host holds program and
    /// engine together with no `Box::leak` and no `'static` requirement.
    pub fn with_program(program: Expr) -> Self {
        let engine = expr_engine(&program).expect("expr program validates");
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

        let suspended_call =
            self.engine
                .suspended_call()
                .map(|(_sid, node_id, label)| SuspendedCall {
                    node: node_id.0,
                    label: label.to_string(),
                });

        let pending: Vec<PendingProjection> = self
            .engine
            .pending()
            .iter()
            .map(|p| {
                let (reason, label) = match &p.reason {
                    dsl_kit::SuspendReason::Call { spec } => {
                        ("call".to_string(), spec.label.clone())
                    }
                    dsl_kit::SuspendReason::Breakpoint => ("breakpoint".into(), String::new()),
                    dsl_kit::SuspendReason::Cooperative => ("cooperative".into(), String::new()),
                    dsl_kit::SuspendReason::User { tag } => (format!("user:{tag}"), String::new()),
                    _ => ("unknown".into(), String::new()),
                };
                PendingProjection {
                    id: p.id.0,
                    reason,
                    label,
                    at: pending_to_location(&p.at),
                }
            })
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
                    // Answer every unbound variable with its default so
                    // the run drives to completion. A synthesized
                    // Breakpoint pending has no Call reason and is
                    // skipped; the next iteration consumes the marker.
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
                return Err("expr host exceeded to_done safety limit".into());
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
            "expr resolve requires `result` as an integer literal (no default provided)".to_string()
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
        self.engine = expr_engine(&self.program).expect("expr program validates");
        self.resolved_log.clear();
        self.final_value = None;
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

    fn schema_json(&self) -> Option<String> {
        use dsl_kit_schema::DslSchema;
        Some(Expr::schema().to_json().to_string())
    }

    fn lint_json(&self) -> Option<String> {
        use dsl_kit_lint::Linter;
        let diagnostics = Linter::<Expr>::with_defaults().lint(&self.program);
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
        // Bridge → conformance-checked build. Diagnostics are serialized
        // as the shared envelope so the handler can pass them through to
        // the client unchanged (see `dsl_kit_load`).
        let tree = from_json_str(input, &Expr::schema()).map_err(|e| e.to_json().to_string())?;
        let ids = IdGen::new();
        let program = Expr::from_parse_tree(&tree, &ids).map_err(|e| e.to_json().to_string())?;
        // Owned program: the host holds `Expr` by value, so replacing it
        // drops the previous AST here — no `Box::leak`, no per-load leak.
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
                    at: pending_to_location(&p.at),
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

fn pending_to_location(ctx: &dsl_kit::NodeContext) -> HostLocation {
    HostLocation {
        node: ctx.node.0,
        path: ctx.path.0.iter().map(|n| n.0).collect(),
        depth: ctx.depth,
        frame: ctx.frame.map(|f| f.0),
        iteration: ctx.iteration.map(|i| i.0),
    }
}

fn default_resolution(name: &str) -> i64 {
    match name {
        "y" => 5,
        "z" => 2,
        _ => 1,
    }
}
