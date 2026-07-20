//! Reference `DslHost` implementation over the flow DSL.
//!
//! Wires the [`flow-dsl`](flow_dsl) reference DSL into
//! [`dsl-kit-mcp`](dsl_kit_mcp). Consumed by the `flow-mcp` binary
//! (default payload of the reference MCP server) and useful as a
//! worked example when writing your own `DslHost` adapter.
//!
//! Round 17 note: [`flow_dsl::FlowStepper`] is now a thin adapter over
//! the core [`dsl_kit::Engine`], so this bridge speaks the design-doc
//! v3 [`dsl_kit::StepOutcome`] shape directly — the old
//! flow-local `InternalOutcome` intermediary is gone.

#![warn(missing_docs)]

use dsl_kit::{BreakpointSet, DslNode, IdGen, Pending, Phase, StepOutcome, Stepper, Walk};
use dsl_kit_mcp::host::{
    DslHost, EventCounts, HostEffectError, HostLocation, HostOutcome, HostSnapshot,
    PendingProjection, ResolvedCall, SuspendedCall,
};
use dsl_kit_mcp::resources::ResourceEntry;
use flow_dsl::{Flow, FlowStepper, FlowValue, canned_response, pretty, research_pipeline};

const FLOW_GRAMMAR: &str = include_str!("./resources_data/grammar.md");
const FLOW_RESEARCH_PIPELINE: &str =
    include_str!("./resources_data/research-pipeline.md");

/// `DslHost` that owns a leaked-static [`Flow`] program plus its
/// stepper.
pub struct FlowHost {
    program: &'static Flow,
    stepper: FlowStepper<'static>,
}

impl FlowHost {
    /// Builds a host around the built-in research pipeline.
    pub fn new_with_default_program() -> Self {
        let ids = IdGen::new();
        let program: &'static Flow = Box::leak(Box::new(research_pipeline(&ids)));
        Self::with_program(program)
    }

    /// Builds a host around a caller-supplied `Flow` reference.
    pub fn with_program(program: &'static Flow) -> Self {
        let stepper = FlowStepper::new(program);
        Self { program, stepper }
    }
}

fn count_nodes(flow: &Flow) -> usize {
    let mut count = 0usize;
    flow.walk(&mut |_, phase| {
        if phase == Phase::Pre {
            count += 1;
        }
    });
    count
}

#[async_trait::async_trait]
impl DslHost for FlowHost {
    fn dsl_name(&self) -> &str {
        "flow"
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
        let counts = self.stepper.events();
        let mut results: Vec<(u64, String)> = self
            .stepper
            .results()
            .iter()
            .map(|(k, v)| (k.0, v.clone()))
            .collect();
        results.sort_by_key(|(id, _)| *id);

        let suspended_call =
            self.stepper.suspended_call().map(|(_sid, node_id, label)| SuspendedCall {
                node: node_id.0,
                label: label.to_string(),
            });

        let pending: Vec<PendingProjection> = self
            .stepper
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
                    at: HostLocation {
                        node: p.at.node.0,
                        path: p.at.path.0.iter().map(|n| n.0).collect(),
                        depth: p.at.depth,
                        frame: p.at.frame.map(|f| f.0),
                        iteration: p.at.iteration.map(|i| i.0),
                    },
                }
            })
            .collect();

        HostSnapshot {
            depth: self.stepper.depth(),
            current_path: self.stepper.current_path().map(|p| p.0.iter().map(|n| n.0).collect()),
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
            .stepper
            .step_with_breakpoints(breakpoints)
            .map_err(|e| e.to_string())?;
        Ok(step_outcome_to_host(outcome, self.stepper.pending()))
    }

    async fn step_to_yield(&mut self, breakpoints: &BreakpointSet) -> Result<HostOutcome, String> {
        let outcome = self
            .stepper
            .run_to_yield_with_breakpoints(breakpoints)
            .map_err(|e| e.to_string())?;
        Ok(step_outcome_to_host(outcome, self.stepper.pending()))
    }

    async fn step_to_done(&mut self, breakpoints: &BreakpointSet) -> Result<HostOutcome, String> {
        let mut steps = 0u32;
        loop {
            let outcome = self
                .stepper
                .run_to_yield_with_breakpoints(breakpoints)
                .map_err(|e| e.to_string())?;
            steps += 1;
            match outcome {
                StepOutcome::Done(_) => return Ok(HostOutcome::Done),
                StepOutcome::Ready => {
                    // `run_to_yield_with_breakpoints` only ever returns
                    // Ready when the loop exhausted its work quota
                    // without a yield; treat as a soft-halt.
                    return Ok(HostOutcome::Advanced);
                }
                StepOutcome::Blocked { .. } => {
                    // Resolve every outstanding Call with its canned
                    // response so the pipeline can drive further. A
                    // synthesized Breakpoint pending has no `Call`
                    // reason and is silently skipped; the loop's next
                    // iteration consumes the marker.
                    let outstanding: Vec<(dsl_kit::SuspensionId, String)> = self
                        .stepper
                        .pending()
                        .iter()
                        .filter_map(|p| match &p.reason {
                            dsl_kit::SuspendReason::Call { spec } => {
                                Some((p.id, spec.label.clone()))
                            }
                            _ => None,
                        })
                        .collect();
                    if outstanding.is_empty() {
                        // Only a Breakpoint (or nothing) is pending;
                        // loop again so the BP marker gets consumed on
                        // the next `run_to_yield_with_breakpoints`.
                        // If there is truly nothing to do, the next
                        // step will return `Done` or another Blocked.
                    }
                    for (sid, label) in outstanding {
                        let response = canned_response(&label);
                        self.stepper
                            .resolve(sid, Ok(FlowValue::Text(response)))
                            .map_err(|e| e.to_string())?;
                    }
                }
            }
            if steps > 4096 {
                return Err("stepper exceeded to_done safety limit".into());
            }
        }
    }

    async fn resolve(&mut self, result: Option<String>) -> Result<ResolvedCall, String> {
        let (sid, node_id, label) = self
            .stepper
            .suspended_call()
            .map(|(sid, id, label)| (sid, id, label.to_string()))
            .ok_or_else(|| "no suspended call to resolve".to_string())?;
        let response = result.unwrap_or_else(|| canned_response(&label));
        self.stepper
            .resolve(sid, Ok(FlowValue::Text(response.clone())))
            .map_err(|e| e.to_string())?;
        Ok(ResolvedCall { node: node_id.0, label, result: response })
    }

    async fn resolve_by_id(
        &mut self,
        id: u64,
        result: Result<String, HostEffectError>,
    ) -> Result<ResolvedCall, String> {
        let sid = dsl_kit::SuspensionId(id);
        // Look up node id + label via the current pending list before
        // resolving (Stepper::resolve consumes the pending entry).
        let (node_id, label) = self
            .stepper
            .pending()
            .iter()
            .find(|p| p.id == sid)
            .map(|p| {
                let label = match &p.reason {
                    dsl_kit::SuspendReason::Call { spec } => spec.label.clone(),
                    _ => String::new(),
                };
                (p.at.node.0, label)
            })
            .ok_or_else(|| format!("no pending suspension for id {id}"))?;

        match result {
            Ok(text) => {
                self.stepper
                    .resolve(sid, Ok(FlowValue::Text(text.clone())))
                    .map_err(|e| e.to_string())?;
                Ok(ResolvedCall { node: node_id, label, result: text })
            }
            Err(err) => {
                self.stepper
                    .resolve(
                        sid,
                        Err(flow_dsl::FlowEffectErr {
                            code: err.code.clone(),
                            message: err.message.clone(),
                        }),
                    )
                    .map_err(|e| e.to_string())?;
                Ok(ResolvedCall {
                    node: node_id,
                    label,
                    result: format!("<err {}: {}>", err.code, err.message),
                })
            }
        }
    }

    fn take_cancellations(&mut self) -> Vec<u64> {
        self.stepper
            .take_cancellations()
            .into_iter()
            .map(|s| s.0)
            .collect()
    }

    fn reset(&mut self) {
        self.stepper = FlowStepper::new(self.program);
    }

    fn resources(&self) -> Vec<ResourceEntry> {
        vec![
            ResourceEntry::static_markdown(
                "dsl-kit://dsl/flow/grammar",
                "flow DSL — grammar",
                "The five variants of the flow enum (Seq / Par / Call / Scope / Maybe) with their semantics and node-id contract.",
                FLOW_GRAMMAR,
            ),
            ResourceEntry::static_markdown(
                "dsl-kit://dsl/flow/samples/research-pipeline",
                "flow DSL — research_pipeline sample",
                "The default program FlowHost loads: a Seq wrapping a Par of three searches plus a Maybe citation check. Structure, source, and drive-to-done walkthrough.",
                FLOW_RESEARCH_PIPELINE,
            ),
        ]
    }

    fn schema_json(&self) -> Option<String> {
        use dsl_kit_schema::DslSchema;
        Some(Flow::schema().to_json().to_string())
    }

    fn lint_json(&self) -> Option<String> {
        use dsl_kit_lint::Linter;
        let diagnostics = Linter::<Flow>::with_defaults().lint(self.program);
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
}

/// Adapt the v3 [`StepOutcome`] shape to the DSL-agnostic
/// [`HostOutcome`] surface the MCP handler serialises.
///
/// - `Done(_)` → `HostOutcome::Done`.
/// - `Ready` → `HostOutcome::Advanced` (rare from
///   `run_to_yield_with_breakpoints`, which loops on `Ready`).
/// - `Blocked { newly_pending }` → `HostOutcome::Suspended`. Prefer the
///   first newly-pending as the yield reason; if `newly_pending` is
///   empty (a "waiting" fold — new pending exhausted, pre-existing
///   pending still live), fall back to the first entry in the current
///   `pending()` snapshot. When both are empty (should not happen from
///   `run_to_yield` but guarded here for completeness), report a
///   sentinel `"waiting"` reason at an empty location.
fn step_outcome_to_host(outcome: StepOutcome<FlowValue>, pending: &[Pending]) -> HostOutcome {
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

#[cfg(test)]
mod tests {
    use super::*;
    use dsl_kit_mcp::host::DslHost;

    #[test]
    fn schema_json_serializes_flow_shape() {
        let host = FlowHost::new_with_default_program();
        let text = host.schema_json().expect("FlowHost wires schema_json");
        let value: serde_json::Value =
            serde_json::from_str(&text).expect("schema_json must be valid JSON");
        assert_eq!(value["name"], "Flow");
        let variants = value["variants"].as_array().expect("variants array");
        // Flow has 5 variants: Seq / Par / Call / Scope / Maybe.
        assert_eq!(variants.len(), 5);
        // Spot-check: Maybe's body child is Optional.
        let maybe = variants
            .iter()
            .find(|v| v["name"] == "Maybe")
            .expect("Maybe variant");
        assert_eq!(maybe["children"][0]["multiplicity"], "optional");
    }

    #[test]
    fn lint_json_returns_empty_array_for_clean_program() {
        let host = FlowHost::new_with_default_program();
        let text = host.lint_json().expect("FlowHost wires lint_json");
        let value: serde_json::Value =
            serde_json::from_str(&text).expect("lint_json must be valid JSON");
        let arr = value.as_array().expect("lint_json is a JSON array");
        assert!(
            arr.is_empty(),
            "reference research_pipeline should lint clean, got {arr:?}"
        );
    }

    #[test]
    fn lint_json_reports_empty_seq_via_no_empty_many_children() {
        // Build a program with an empty Seq — NoEmptyManyChildren should
        // fire, and the diagnostic must flow through lint_json unchanged.
        let ids = IdGen::new();
        let empty_seq_id = ids.node();
        let program: &'static Flow = Box::leak(Box::new(Flow::Seq {
            id: ids.node(),
            children: vec![Flow::Seq { id: empty_seq_id, children: vec![] }],
        }));
        let host = FlowHost::with_program(program);
        let text = host.lint_json().expect("FlowHost wires lint_json");
        let value: serde_json::Value =
            serde_json::from_str(&text).expect("lint_json must be valid JSON");
        let arr = value.as_array().expect("lint_json is a JSON array");
        assert_eq!(arr.len(), 1, "diags = {arr:?}");
        assert_eq!(arr[0]["rule"], "no-empty-many-children");
        assert_eq!(arr[0]["severity"], "Error");
        assert_eq!(arr[0]["node"], empty_seq_id.0);
    }
}
