//! Reference `DslHost` implementation over the flow DSL.
//!
//! This module wires the `dsl-kit-flow` reference DSL into the
//! MCP handler. It exists both as the default binary payload
//! (`dsl-kit-mcp` serves this host out of the box) and as a
//! worked example for new DSL adapters.

use dsl_kit::{BreakpointSet, DslNode, IdGen, Phase, StepOutcome, Walk};
use dsl_kit_flow::{Flow, FlowStepper, canned_response, pretty, research_pipeline};

use crate::host::{
    DslHost, EventCounts, HostLocation, HostOutcome, HostSnapshot, ResolvedCall, SuspendedCall,
};
use crate::resources::ResourceEntry;

const FLOW_GRAMMAR: &str = include_str!("./resources_data/flow/grammar.md");
const FLOW_RESEARCH_PIPELINE: &str =
    include_str!("./resources_data/flow/research-pipeline.md");

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
            self.stepper.suspended_call().map(|(id, label)| SuspendedCall {
                node: id.0,
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
                StepOutcome::Suspended { .. } => {
                    if let Some((id, label)) = self.stepper.suspended_call() {
                        let response = canned_response(label);
                        self.stepper.record_result(id, response);
                    }
                    // Breakpoint yields have no call to resolve; the
                    // next iteration transitions normally because the
                    // stepper's `breakpoint_yielded` guard has been
                    // cleared.
                }
                other => return Ok(outcome_to_host(other)),
            }
            if steps > 4096 {
                return Err("stepper exceeded to_done safety limit".into());
            }
        }
    }

    async fn resolve(&mut self, result: Option<String>) -> Result<ResolvedCall, String> {
        let (id, label) = self
            .stepper
            .suspended_call()
            .map(|(id, label)| (id, label.to_string()))
            .ok_or_else(|| "no suspended call to resolve".to_string())?;
        let response = result.unwrap_or_else(|| canned_response(&label));
        self.stepper.record_result(id, response.clone());
        Ok(ResolvedCall { node: id.0, label, result: response })
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

fn outcome_to_host(outcome: StepOutcome<()>) -> HostOutcome {
    match outcome {
        StepOutcome::Advanced => HostOutcome::Advanced,
        StepOutcome::Suspended { reason, at } => HostOutcome::Suspended {
            reason: reason.to_string(),
            at: HostLocation {
                node: at.node.0,
                path: at.path.0.iter().map(|n| n.0).collect(),
                depth: at.depth,
                frame: at.frame.map(|f| f.0),
                iteration: at.iteration.map(|i| i.0),
            },
        },
        StepOutcome::Done(()) => HostOutcome::Done,
    }
}
