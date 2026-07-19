//! Reference demo for the flow DSL.
//!
//! The binary is structured in four sections:
//!
//! 1. Walks the AST with the derived traversal, prints an indented
//!    tree.
//! 2. Runs the pipeline synchronously with a canned resolver.
//! 3. Runs the same pipeline through `AsyncStepper` inside a tokio
//!    runtime, using `tokio::time::sleep` to simulate real network
//!    latency, and runs two pipelines concurrently to show that the
//!    async surface actually overlaps work.
//! 4. Exercises the breakpoint surface and renders a diagnostic error.

use std::time::{Duration, Instant};

use dsl_kit::{
    AsyncStepper, BreakCondition, BreakpointId, BreakpointSet, DslNode, IdGen, NodeContext, NodeId,
    Path, StepOutcome, Stepper, Walk,
};
use dsl_kit_flow::{
    Flow, FlowStepper, canned_response, check_unique_ids, pretty, research_pipeline,
};
use futures::future::join;

/// Illustrates the breakpoint condition surface by walking the AST once
/// and reporting which nodes each condition matches.
fn demonstrate_breakpoints(program: &Flow) {
    let mut set = BreakpointSet::new();
    let by_id = set.add(BreakCondition::at_node(NodeId(4)));
    let deep = set.add(BreakCondition::at_depth_at_least(4));
    let inside_research = set.add(BreakCondition::under_path(
        Path::root().push(NodeId(0)).push(NodeId(2)),
    ));
    let combined =
        set.add(BreakCondition::at_depth_at_least(3).and(BreakCondition::at_iteration(2)));

    println!("  registered:");
    println!("    {by_id}: at node n4");
    println!("    {deep}: depth >= 4");
    println!("    {inside_research}: under path /n0/n2");
    println!("    {combined}: depth >= 3 AND iteration == 2");

    fn walk(
        node: &Flow,
        path: &Path,
        depth: u32,
        set: &BreakpointSet,
        hits: &mut Vec<(NodeId, Vec<BreakpointId>)>,
    ) {
        let ctx =
            NodeContext { node: node.node_id(), path: path.clone(), frame: None, depth, iteration: None };
        let m = set.matches(&ctx);
        if !m.is_empty() {
            hits.push((node.node_id(), m));
        }
        for child in node.children() {
            let child_path = path.push(child.node_id());
            walk(child, &child_path, depth + 1, set, hits);
        }
    }

    let mut hits = Vec::new();
    let root_path = Path::root().push(program.node_id());
    walk(program, &root_path, 1, &set, &mut hits);

    println!("  hits:");
    for (node, ids) in &hits {
        let names: Vec<String> = ids.iter().map(ToString::to_string).collect();
        println!("    {node}: {}", names.join(", "));
    }
    if hits.is_empty() {
        println!("    (none)");
    }
}

/// Drives a flow through `AsyncStepper` using a real async resolver
/// that sleeps for `latency` before answering each `Call`.
///
/// Returns the elapsed wall-clock time so the caller can compare
/// sequential vs concurrent runs.
async fn run_flow_async(program: &Flow, tag: &str, latency: Duration) -> (Duration, usize) {
    let start = Instant::now();
    let mut stepper = FlowStepper::new(program);
    let mut resolved = 0usize;
    loop {
        // Explicit await point: every step goes through the async trait.
        let outcome = stepper.step_async().await.expect("step");
        match outcome {
            StepOutcome::Advanced => continue,
            StepOutcome::Suspended { .. } => {
                if let Some((id, label)) = stepper.suspended_call() {
                    tokio::time::sleep(latency).await;
                    let response = format!("[{tag}] {}", canned_response(label));
                    stepper.record_result(id, response);
                    resolved += 1;
                }
            }
            StepOutcome::Done(()) => break,
        }
    }
    (start.elapsed(), resolved)
}

#[tokio::main]
async fn main() -> miette::Result<()> {
    let ids = IdGen::new();
    let program = research_pipeline(&ids);

    println!("=== Research pipeline: AST ===");
    print!("{}", pretty(&program));

    check_unique_ids(&program)?;

    println!("\n=== Running the pipeline ===");
    let mut stepper = FlowStepper::new(&program);
    loop {
        match stepper.run_to_yield()? {
            StepOutcome::Advanced => {}
            StepOutcome::Suspended { reason: _, at } => {
                if let Some((id, label)) = stepper.suspended_call() {
                    let response = canned_response(label);
                    println!("  {id:>4} {label:<15} -> {response}   ({at})");
                    stepper.record_result(id, response);
                }
            }
            StepOutcome::Done(()) => break,
        }
    }

    println!("\n=== Event summary ===");
    println!("  {}", stepper.event_summary());

    println!("\n=== Recorded results ===");
    let mut results: Vec<(NodeId, String)> = stepper.into_results().into_iter().collect();
    results.sort_by_key(|(id, _)| id.0);
    for (id, text) in &results {
        println!("  {id}: {text}");
    }

    println!("\n=== Async run (single pipeline, real tokio sleeps) ===");
    let (single_elapsed, single_calls) =
        run_flow_async(&program, "solo", Duration::from_millis(50)).await;
    println!(
        "  solo pipeline: resolved {single_calls} calls in {:.0} ms (50 ms per suspend x {single_calls})",
        single_elapsed.as_millis()
    );

    println!("\n=== Async run (two pipelines concurrently) ===");
    let start = Instant::now();
    let ((elapsed_a, calls_a), (elapsed_b, calls_b)) = join(
        run_flow_async(&program, "A", Duration::from_millis(50)),
        run_flow_async(&program, "B", Duration::from_millis(50)),
    )
    .await;
    let joint_elapsed = start.elapsed();
    println!(
        "  A: {calls_a} calls in {:.0} ms  B: {calls_b} calls in {:.0} ms",
        elapsed_a.as_millis(),
        elapsed_b.as_millis()
    );
    println!(
        "  concurrent wall clock: {:.0} ms (each pipeline still {calls_a} suspends x 50 ms; overlap is real)",
        joint_elapsed.as_millis()
    );

    println!("\n=== Breakpoints ===");
    demonstrate_breakpoints(&program);

    println!("\n=== Error rendering (malformed AST) ===");
    let broken = Flow::Seq {
        id: NodeId(100),
        children: vec![
            Flow::Call { id: NodeId(101), label: "one".into() },
            Flow::Call { id: NodeId(101), label: "two".into() },
        ],
    };
    if let Err(err) = check_unique_ids(&broken) {
        println!("{:?}", miette::Report::new(err));
    }

    Ok(())
}
