//! Reference demo for the flow DSL.
//!
//! This binary uses `dsl-kit-flow` to build a small research pipeline,
//! walks it with the derived traversal, runs it through the stepper
//! while a canned resolver answers each effect, exercises the
//! breakpoint surface, and finishes by rendering a diagnostic error.

use dsl_kit::{
    BreakCondition, BreakpointId, BreakpointSet, DslNode, IdGen, NodeContext, NodeId, Path,
    StepOutcome, Stepper, Walk,
};
use dsl_kit_flow::{
    Flow, FlowStepper, canned_response, check_unique_ids, pretty, research_pipeline,
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

fn main() -> miette::Result<()> {
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
