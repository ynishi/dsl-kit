//! `DslHost` adapter around the configuration DSL.
//!
//! The host wraps a [`dsl_kit::Engine`] over [`CfgAst`] — the same
//! engine that runs `flow-dsl` and `expr-dsl`. There is no evaluator
//! here: keyed-slot ordering, the `last_wins` fold and reference
//! suspension all happen inside the engine, and this adapter only
//! projects engine state into the MCP host shape.
//!
//! Its reason to exist alongside `flow-host` / `expr-host` is the
//! shape of its DSL: `Cfg` is the first reference DSL whose children
//! are **keyed**, so serving it is what puts `multiplicity: "map"` on
//! an MCP surface a client can actually talk to.

#![warn(missing_docs)]

use cfg_dsl::{Cfg, CfgAst, cfg_engine, count_nodes, demo_document, pretty};
use dsl_kit::{AllowTable, BreakpointSet, DslNode, Engine, IdGen, Pending, StepOutcome, Stepper};
use dsl_kit_mcp::host::{
    DslHost, EventCounts, HostLocation, HostOutcome, HostSnapshot, PendingProjection, ResolvedCall,
    SuspendedCall,
};
use dsl_kit_mcp::resources::ResourceEntry;

const CFG_GRAMMAR: &str = include_str!("./resources_data/grammar.md");
const CFG_DEMO_DOCUMENT: &str = include_str!("./resources_data/demo-document.md");

/// `DslHost` adapter around the configuration DSL.
pub struct CfgHost {
    document: Cfg,
    engine: Engine<CfgAst>,
    /// Usage-site lint suppressions the loaded document declared,
    /// keyed on the `NodeId` the build minted for each annotated node.
    /// Empty for a hand-supplied document — an annotation is something
    /// a document spells, so only the load paths can produce one.
    allows: AllowTable,
    /// Resolution history projected into `HostSnapshot::results`.
    resolved_log: Vec<(u64, String)>,
    final_value: Option<String>,
}

impl CfgHost {
    /// Builds a host around the built-in demo document.
    pub fn new_with_default_document() -> Self {
        let ids = IdGen::new();
        Self::with_document(demo_document(&ids))
    }

    /// Builds a host that owns a caller-supplied `Cfg` document.
    pub fn with_document(document: Cfg) -> Self {
        let engine = cfg_engine(&document).expect("cfg document validates");
        Self {
            document,
            engine,
            allows: AllowTable::default(),
            resolved_log: Vec::new(),
            final_value: None,
        }
    }

    fn record_done(&mut self, outcome: &StepOutcome<String>) {
        if let StepOutcome::Done(v) = outcome {
            self.final_value = Some(v.clone());
        }
    }
}

#[async_trait::async_trait]
impl DslHost for CfgHost {
    fn dsl_name(&self) -> &str {
        "cfg"
    }

    fn root_node_id(&self) -> u64 {
        self.document.node_id().0
    }

    fn root_summary(&self) -> String {
        self.document.summary()
    }

    fn ast_size(&self) -> usize {
        count_nodes(&self.document)
    }

    fn ast_pretty(&self) -> String {
        pretty(&self.document)
    }

    fn snapshot(&self) -> HostSnapshot {
        let counts = self.engine.events();
        let mut results = self.resolved_log.clone();
        if let Some(v) = &self.final_value {
            results.push((self.document.node_id().0, v.clone()));
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
                    // Answer every outstanding reference with the
                    // canned default so the run drives to completion. A
                    // synthesized Breakpoint pending has no Call reason
                    // and is skipped; the next iteration consumes the
                    // marker.
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
                            .resolve(sid, Ok(value.clone()))
                            .map_err(|e| e.to_string())?;
                        self.resolved_log.push((node, format!("{name} = {value}")));
                    }
                }
            }
            steps += 1;
            if steps > 4096 {
                return Err("cfg host exceeded to_done safety limit".into());
            }
        }
    }

    async fn resolve(&mut self, result: Option<String>) -> Result<ResolvedCall, String> {
        let (sid, node_id, name) = self
            .engine
            .suspended_call()
            .map(|(sid, id, label)| (sid, id, label.to_string()))
            .ok_or_else(|| "no outstanding reference to resolve".to_string())?;
        // Unlike expr, any string is a legal config value, so an
        // omitted `result` falls back to the host default rather than
        // failing the call.
        let value = result.unwrap_or_else(|| default_resolution(&name));
        self.engine
            .resolve(sid, Ok(value.clone()))
            .map_err(|e| e.to_string())?;
        self.resolved_log
            .push((node_id.0, format!("{name} = {value}")));
        Ok(ResolvedCall {
            node: node_id.0,
            label: name,
            result: value,
        })
    }

    fn reset(&mut self) {
        self.engine = cfg_engine(&self.document).expect("cfg document validates");
        self.resolved_log.clear();
        self.final_value = None;
    }

    fn resources(&self) -> Vec<ResourceEntry> {
        vec![
            ResourceEntry::static_markdown(
                "dsl-kit://dsl/cfg/grammar",
                "cfg DSL — grammar",
                "The four variants of the Cfg enum (Env / Overrides / Ref / Leaf), the keyed-slot syntax they share, and the reference-resolution contract.",
                CFG_GRAMMAR,
            ),
            ResourceEntry::static_markdown(
                "dsl-kit://dsl/cfg/samples/demo-document",
                "cfg DSL — demo document",
                "The default document CfgHost loads: a two-level Env with an override stack and one unresolved reference. Structure, source, and drive-to-done walkthrough.",
                CFG_DEMO_DOCUMENT,
            ),
        ]
    }

    fn schema_json(&self) -> Option<String> {
        use dsl_kit_schema::DslSchema;
        Some(Cfg::schema().to_json().to_string())
    }

    fn lint_json(&self) -> Option<String> {
        use dsl_kit_lint::Linter;
        // The document's own `$allow` / `@allow` annotations are
        // honoured here: a node that named a rule it accepts does not
        // report that rule again. What they silenced stays enumerable
        // in `outcome.suppressed`; this surface reports the findings
        // that survived.
        let outcome = Linter::<Cfg>::with_defaults().lint_with_allows(&self.document, &self.allows);
        let value: Vec<serde_json::Value> = outcome
            .diagnostics
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
        // Bridge → conformance-checked build. Keyed slots arrive as
        // JSON objects (`{"bindings": {"app": {…}}}`) and the bridge
        // sorts them on ingest, so the tree is key-order canonical
        // before conformance runs.
        let tree = from_json_str(input, &Cfg::schema()).map_err(|e| e.to_json().to_string())?;
        let ids = IdGen::new();
        let document = Cfg::from_parse_tree(&tree, &ids).map_err(|e| e.to_json().to_string())?;
        // Take the annotations the build recorded against the ids it
        // just minted, before the generator goes out of scope.
        self.allows = ids.take_allows();
        self.document = document;
        self.reset();
        Ok(())
    }

    async fn load_json_bundle(
        &mut self,
        input: &str,
        sources_json: &str,
    ) -> Result<String, String> {
        use dsl_kit_parse::DslBuild;
        use dsl_kit_parse::allow::add_allow_syntax;
        use dsl_kit_parse::import::{Loader, MapResolver, add_import_syntax};
        use dsl_kit_parse::schema_gen::checked_grammar_from_schema;
        use dsl_kit_schema::DslSchema;

        let mut resolver =
            MapResolver::from_sources_json(sources_json).map_err(|e| e.to_json().to_string())?;
        let schema = Cfg::schema();
        let ids = IdGen::new();
        // Text sources spell imports as `@import "name"` and
        // suppressions as `@allow("rule") <node>` — same
        // schema-generated grammar the round-trip tests exercise, with
        // both reserved spellings injected on top.
        let mut grammar =
            checked_grammar_from_schema(&schema, &ids).map_err(|e| e.to_json().to_string())?;
        add_import_syntax(&mut grammar, &ids).map_err(|e| e.to_json().to_string())?;
        add_allow_syntax(&mut grammar, &ids).map_err(|e| e.to_json().to_string())?;

        let loaded = Loader::new(&schema)
            .with_grammar(&grammar)
            .load_json_str(input, &mut resolver)
            .map_err(|e| e.to_json().to_string())?;
        let document =
            Cfg::from_parse_tree(&loaded.tree, &ids).map_err(|e| e.to_json().to_string())?;
        self.allows = ids.take_allows();
        self.document = document;
        self.reset();
        Ok(serde_json::json!({
            "dependencies": loaded
                .dependencies
                .iter()
                .map(|d| d.as_str())
                .collect::<Vec<_>>(),
            "digest": loaded.digest(),
        })
        .to_string())
    }
}

fn step_outcome_to_host(outcome: StepOutcome<String>, pending: &[Pending]) -> HostOutcome {
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

/// Canned answers for the demo document's references.
fn default_resolution(name: &str) -> String {
    match name {
        "PORT" => "8080".to_string(),
        other => format!("<unset:{other}>"),
    }
}
