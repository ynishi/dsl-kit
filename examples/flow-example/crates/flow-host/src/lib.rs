//! Reference `DslHost` implementation over the flow DSL.
//!
//! Wires the [`flow-dsl`](flow_dsl) reference DSL into
//! [`dsl-kit-mcp`](dsl_kit_mcp). Consumed by the `flow-mcp` binary
//! (default payload of the reference MCP server) and useful as a
//! worked example when writing your own `DslHost` adapter.
//!
//! Commit A landing note: the internal `FlowStepper` now satisfies the
//! v3 `Stepper` trait; `FlowHost` bridges its `DslHost` (`step_one` /
//! `resolve` / …) contract to the new stepper API. The `DslHost` trait
//! itself is untouched in Commit A; new fan-out surface (pending list,
//! cancellation drain) lands in Commit B.

#![warn(missing_docs)]

use dsl_kit::{BreakpointSet, DslNode, IdGen, Phase, Stepper, Walk};
use dsl_kit_mcp::host::{
    DslHost, EventCounts, HostLocation, HostOutcome, HostSnapshot, ResolvedCall, SuspendedCall,
};
use dsl_kit_mcp::resources::ResourceEntry;
use flow_dsl::{Flow, FlowStepper, FlowValue, InternalOutcome, canned_response, pretty, research_pipeline};

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

        HostSnapshot {
            depth: self.stepper.depth(),
            current_path: self.stepper.current_path().map(|p| p.0.iter().map(|n| n.0).collect()),
            suspended_call,
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
        Ok(outcome_to_host(outcome))
    }

    async fn step_to_yield(&mut self, breakpoints: &BreakpointSet) -> Result<HostOutcome, String> {
        let outcome = self
            .stepper
            .run_to_yield_with_breakpoints(breakpoints)
            .map_err(|e| e.to_string())?;
        Ok(outcome_to_host(outcome))
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
                InternalOutcome::Suspended { .. } => {
                    if let Some((sid, _node, label)) = self.stepper.suspended_call() {
                        let response = canned_response(label);
                        self.stepper
                            .resolve(sid, Ok(FlowValue::Text(response)))
                            .map_err(|e| e.to_string())?;
                    }
                }
                other => return Ok(outcome_to_host(other)),
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
}

fn outcome_to_host(outcome: InternalOutcome) -> HostOutcome {
    match outcome {
        InternalOutcome::Advanced => HostOutcome::Advanced,
        InternalOutcome::Suspended { reason, at } => HostOutcome::Suspended {
            reason: reason.to_string(),
            at: HostLocation {
                node: at.node.0,
                path: at.path.0.iter().map(|n| n.0).collect(),
                depth: at.depth,
                frame: at.frame.map(|f| f.0),
                iteration: at.iteration.map(|i| i.0),
            },
        },
        InternalOutcome::Done => HostOutcome::Done,
    }
}
