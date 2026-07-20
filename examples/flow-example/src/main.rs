//! Reference demo for the flow DSL.
//!
//! The binary is structured in five sections:
//!
//! 1. Walks the AST with the derived traversal, prints an indented
//!    tree.
//! 2. Runs the pipeline synchronously with a canned resolver.
//! 3. Runs one pipeline through the v3 async pattern with **real inline
//!    fan-out**: at the `Par` node all three searches are dispatched
//!    as concurrent `tokio::spawn` tasks and their responses are
//!    plumbed back as each future completes.
//! 4. Runs a FailFast demo: one Par slot resolves with an effect
//!    error; the next step surfaces the error and the sibling ids
//!    appear in `Stepper::take_cancellations`.
//! 5. Exercises the breakpoint surface and renders a diagnostic error.

use std::time::{Duration, Instant};

use dsl_kit::{
    BreakCondition, BreakpointId, BreakpointSet, DslNode, IdGen, NodeContext, NodeId, Path,
    StepOutcome, Stepper, SuspensionId, Walk,
};
use flow_dsl::{
    Flow, FlowEffectErr, FlowStepper, FlowValue, canned_response, check_unique_ids, pretty,
    research_pipeline,
};

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

/// Drives a flow through the v3 [`Stepper`] trait with **real inline
/// fan-out**. Sequential `Call` yields are awaited one at a time
/// (`tokio::time::sleep(base_latency)`); at a `Par` node every
/// outstanding slot is dispatched as its own `tokio::spawn` task, and
/// the responses are resolved back into the stepper as each future
/// completes (`JoinSet::join_next`).
///
/// To make the parallelism visible on stdout each fan-out slot gets a
/// different simulated latency (`base_latency + i * stagger`), so the
/// resolution order in the printed log is deterministic but distinct
/// from declaration order.
async fn run_flow_async_fanout(
    program: &Flow,
    base_latency: Duration,
    stagger: Duration,
) -> (Duration, usize) {
    let start = Instant::now();
    let mut stepper = FlowStepper::new(program);
    let mut resolved = 0usize;
    let bp = BreakpointSet::new();
    loop {
        let outcome = stepper.run_to_yield_with_breakpoints(&bp).expect("step");
        match outcome {
            StepOutcome::Ready => {}
            StepOutcome::Done(_) => break,
            StepOutcome::Blocked { .. } => {
                // Collect every outstanding Call. In the single-Call
                // case (`suspended_call` returns Some), await one
                // response sequentially. In the fan-out case, dispatch
                // every slot as its own `tokio::spawn` task so the
                // wall clock is bounded by the slowest, not the sum.
                let outstanding: Vec<(SuspensionId, String)> = stepper
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
                    break;
                }
                if outstanding.len() == 1 {
                    let (sid, label) = outstanding.into_iter().next().unwrap();
                    tokio::time::sleep(base_latency).await;
                    let response = canned_response(&label);
                    println!("  seq  {label:<15} -> {response}");
                    stepper
                        .resolve(sid, Ok(FlowValue::Text(response)))
                        .expect("resolve");
                    resolved += 1;
                } else {
                    let slot_count = outstanding.len();
                    let mut set = tokio::task::JoinSet::new();
                    for (i, (sid, label)) in outstanding.into_iter().enumerate() {
                        let delay = base_latency + stagger * i as u32;
                        set.spawn(async move {
                            tokio::time::sleep(delay).await;
                            (sid, label, delay)
                        });
                    }
                    let fan_start = Instant::now();
                    while let Some(joined) = set.join_next().await {
                        let (sid, label, delay) = joined.expect("spawn");
                        let response = canned_response(&label);
                        println!(
                            "  par  {label:<15} -> {response}   (after {:>3} ms)",
                            delay.as_millis()
                        );
                        stepper
                            .resolve(sid, Ok(FlowValue::Text(response)))
                            .expect("resolve");
                        resolved += 1;
                    }
                    let fan_elapsed = fan_start.elapsed();
                    println!(
                        "  par  ({slot_count} slots resolved in {:>3} ms total; sequential would take ~{} ms)",
                        fan_elapsed.as_millis(),
                        (0..slot_count)
                            .map(|i| (base_latency + stagger * i as u32).as_millis())
                            .sum::<u128>()
                    );
                }
            }
        }
    }
    (start.elapsed(), resolved)
}

/// Small standalone `Par`-of-three-calls program used by the FailFast
/// demo. Kept separate from the research pipeline so a single failure
/// isolates the fan-out cancellation behavior.
fn par_three_searches(ids: &IdGen) -> Flow {
    Flow::Par {
        id: ids.node(),
        children: vec![
            Flow::Call { id: ids.node(), label: "search_arxiv".into() },
            Flow::Call { id: ids.node(), label: "search_github".into() },
            Flow::Call { id: ids.node(), label: "search_web".into() },
        ],
        policy: None,
        reducer_id: None,
    }
}

/// Enter the Par, fail the middle slot with an effect error, then
/// print the propagated error and the drained sibling cancellations.
fn run_failfast_demo() {
    let ids = IdGen::new();
    let program = par_three_searches(&ids);
    let mut stepper = FlowStepper::new(&program);

    // Step once: enters the Par and emits 3 Pending.
    match stepper.step().expect("enter par") {
        StepOutcome::Blocked { newly_pending } => {
            println!("  Par dispatched {} slots.", newly_pending.len());
        }
        other => {
            println!("  unexpected first-step outcome: {other:?}");
            return;
        }
    }

    let pending: Vec<(SuspensionId, String)> = stepper
        .pending()
        .iter()
        .filter_map(|p| match &p.reason {
            dsl_kit::SuspendReason::Call { spec } => Some((p.id, spec.label.clone())),
            _ => None,
        })
        .collect();
    for (sid, label) in &pending {
        println!("    slot id={:>3} label={label}", sid.0);
    }

    // Fail the middle slot.
    let (fail_sid, fail_label) = pending[1].clone();
    stepper
        .resolve(
            fail_sid,
            Err(FlowEffectErr {
                code: "timeout".into(),
                message: format!("{fail_label} timed out"),
            }),
        )
        .expect("record failure");
    println!("  slot {fail_label} resolved with Err(timeout).");

    // The next step propagates the error under FailFast.
    match stepper.step() {
        Err(err) => println!("  next step -> Err: {err}"),
        Ok(out) => println!("  next step -> Ok({out:?}) (unexpected)"),
    }

    let cancelled = stepper.take_cancellations();
    println!(
        "  take_cancellations() drained {} sibling id(s): {:?}",
        cancelled.len(),
        cancelled.iter().map(|s| s.0).collect::<Vec<_>>()
    );
    let drained_again = stepper.take_cancellations();
    println!("  take_cancellations() drained again: {drained_again:?} (empty on second call)");
}

#[tokio::main]
async fn main() -> miette::Result<()> {
    let ids = IdGen::new();
    let program = research_pipeline(&ids);

    println!("=== Research pipeline: AST ===");
    print!("{}", pretty(&program));

    check_unique_ids(&program).map_err(miette::Report::new)?;

    println!("\n=== Running the pipeline ===");
    let mut stepper = FlowStepper::new(&program);
    let bp = BreakpointSet::new();
    loop {
        let outcome = stepper.run_to_yield_with_breakpoints(&bp).expect("step");
        match outcome {
            StepOutcome::Ready => {}
            StepOutcome::Done(_) => break,
            StepOutcome::Blocked { .. } => {
                // Single-Call yield: prefer `suspended_call` for the
                // pretty single-line trace. Fan-out yield: resolve
                // every outstanding Call in one batch.
                if let Some((sid, node_id, label)) = stepper.suspended_call() {
                    let response = canned_response(label);
                    let at = stepper.current_path().map(|p| p.to_string()).unwrap_or_default();
                    println!("  {node_id:>4} {label:<15} -> {response}   ({at})");
                    stepper
                        .resolve(sid, Ok(FlowValue::Text(response)))
                        .expect("resolve");
                } else {
                    let outstanding: Vec<(dsl_kit::SuspensionId, String)> = stepper
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
                        break;
                    }
                    for (sid, label) in outstanding {
                        let response = canned_response(&label);
                        println!("  (par) {label:<15} -> {response}");
                        stepper
                            .resolve(sid, Ok(FlowValue::Text(response)))
                            .expect("resolve");
                    }
                }
            }
        }
    }

    println!("\n=== Event summary ===");
    println!("  {}", stepper.event_summary());

    println!("\n=== Recorded results ===");
    let mut results: Vec<(NodeId, String)> =
        stepper.results().iter().map(|(id, s)| (*id, s.clone())).collect();
    results.sort_by_key(|(id, _)| id.0);
    for (id, text) in &results {
        println!("  {id}: {text}");
    }

    println!("\n=== Async run (single pipeline, real inline Par fan-out) ===");
    let (elapsed, calls) = run_flow_async_fanout(
        &program,
        Duration::from_millis(50),
        Duration::from_millis(30),
    )
    .await;
    println!(
        "  pipeline resolved {calls} calls in {} ms wall clock.",
        elapsed.as_millis()
    );
    println!(
        "  The 3 Par slots ran concurrently via tokio::spawn; their combined wall clock is bounded by the slowest slot, not the sum."
    );

    println!("\n=== FailFast demo (Par of 3, middle slot fails) ===");
    run_failfast_demo();

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

    // Silence unused warning when Stepper isn't invoked directly here.
    let _ = std::marker::PhantomData::<StepOutcome<FlowValue>>;

    Ok(())
}
