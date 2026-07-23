//! Reference demo for the flow DSL.
//!
//! The binary is structured in eight sections:
//!
//! 1. Walks the AST with the derived traversal, prints an indented
//!    tree.
//! 2. Runs the pipeline synchronously through the sans-io drive layer
//!    (`dsl_kit::drive` + an `EffectResolver` backed by canned
//!    responses) — no hand-rolled step/resolve loop.
//! 3. Runs the pipeline through `drive_async` with an
//!    `AsyncEffectResolver` that awaits a simulated latency per call —
//!    the same engine and loop under Tokio, sequential by design.
//! 4. Runs one pipeline through the v3 async pattern with **real inline
//!    fan-out**: at the `Par` node all three searches are dispatched
//!    as concurrent `tokio::spawn` tasks and their responses are
//!    plumbed back as each future completes. This is the hand-rolled
//!    concurrent alternative to `drive_async` for hosts with an
//!    executor opinion.
//! 5. Runs a FailFast demo: one Par slot resolves with an effect
//!    error; the next step surfaces the error and the sibling ids
//!    appear in `Stepper::take_cancellations`.
//! 6. Drives the pipeline against a live breakpoint on a `Par` kid:
//!    the drive loop returns `DriveOutcome::Break` **before the kid
//!    spawns** (uniform spawn-stack breakpoint scope), then resumes to
//!    completion.
//! 7. Attempts schema-driven grammar generation for `Flow` and shows
//!    the designed fail-loud outcome: `Par`'s exotic payload types
//!    (`Option<JoinPolicy>` / `Option<ReducerId>`) have no
//!    canonical-syntax mapping, so generation reports them instead of
//!    silently dropping the variant (contrast with `expr-example`,
//!    where generation succeeds end to end).
//! 8. Exercises the static breakpoint-condition surface and renders a
//!    diagnostic error.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dsl_kit::{
    AsyncEffectResolver, BreakCondition, BreakpointId, BreakpointSet, DriveOutcome, DslNode,
    EffectResolver, Engine, IdGen, NodeContext, NodeId, Path, Pending, Phase, StepOutcome, Stepper,
    SuspendReason, SuspensionId, Walk, drive, drive_async,
};
use dsl_kit_schema::DslSchema;
use flow_dsl::{
    Flow, FlowAst, FlowEffectErr, FlowStepper, FlowValue, canned_response, check_unique_ids,
    flow_default_registry, flow_syntax_overrides, gated_pipeline, pretty, research_pipeline,
};

/// Sync effect backend for the drive layer: answers every `Call` with
/// the canned response for its label, printing a one-line trace and
/// recording the result per node so the demo can render a results
/// table afterwards.
#[derive(Default)]
struct CannedResolver {
    results: Vec<(NodeId, String)>,
}

impl EffectResolver<FlowAst> for CannedResolver {
    fn resolve(&mut self, pending: &Pending) -> Result<FlowValue, FlowEffectErr> {
        let label = match &pending.reason {
            SuspendReason::Call { spec } => spec.label.clone(),
            other => unreachable!("drive only hands Call suspensions to the resolver: {other:?}"),
        };
        let response = canned_response(&label);
        println!(
            "  {:>4} {label:<15} -> {response}   ({})",
            pending.at.node.to_string(),
            pending.at.path
        );
        self.results.push((pending.at.node, response.clone()));
        Ok(FlowValue::Text(response))
    }
}

/// Async effect backend for `drive_async`: the same canned responses,
/// but each call awaits a simulated latency first. The driver resolves
/// sequentially by design, so the wall clock is the *sum* of latencies
/// — the hand-rolled fan-out section shows the concurrent alternative
/// whose wall clock is bounded by the slowest slot.
struct SlowCannedResolver {
    latency: Duration,
    resolved: usize,
}

impl AsyncEffectResolver<FlowAst> for SlowCannedResolver {
    async fn resolve(&mut self, pending: &Pending) -> Result<FlowValue, FlowEffectErr> {
        let label = match &pending.reason {
            SuspendReason::Call { spec } => spec.label.clone(),
            other => unreachable!("drive only hands Call suspensions to the resolver: {other:?}"),
        };
        tokio::time::sleep(self.latency).await;
        let response = canned_response(&label);
        println!("  async {label:<15} -> {response}");
        self.resolved += 1;
        Ok(FlowValue::Text(response))
    }
}

/// Finds the first `Call` node carrying `label` (node IDs are assigned
/// at build time, so the demo looks the target up by label).
fn find_call_node(flow: &Flow, target: &str) -> Option<NodeId> {
    let mut found: Option<NodeId> = None;
    flow.walk(&mut |node, phase| {
        if phase != Phase::Pre {
            return;
        }
        if let Flow::Call { id, label } = node
            && label == target
            && found.is_none()
        {
            found = Some(*id);
        }
    });
    found
}

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
        let ctx = NodeContext {
            node: node.node_id(),
            path: path.clone(),
            frame: None,
            depth,
            iteration: None,
        };
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
                        dsl_kit::SuspendReason::Call { spec } => Some((p.id, spec.label.clone())),
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
/// A runtime value decides the orchestration shape: `Branch` asks
/// `cache_lookup` first, then spawns **only** the selected side. The
/// losing side — cached reply or a whole `Par` cascade — never enters
/// the engine arena, so breakpoints and the frame tree only ever see
/// the path actually taken.
fn demonstrate_value_branch() {
    struct GateResolver {
        cache: &'static str,
        labels: Vec<String>,
    }
    impl EffectResolver<FlowAst> for GateResolver {
        fn resolve(&mut self, pending: &Pending) -> Result<FlowValue, FlowEffectErr> {
            let label = match &pending.reason {
                SuspendReason::Call { spec } => spec.label.clone(),
                other => unreachable!("drive only hands Call suspensions: {other:?}"),
            };
            let response = if label == "cache_lookup" {
                self.cache.to_string()
            } else {
                canned_response(&label)
            };
            self.labels.push(label);
            Ok(FlowValue::Text(response))
        }
    }

    for cache in ["true", "false"] {
        let ids = IdGen::new();
        let program = gated_pipeline(&ids);
        let mut engine = Engine::new(FlowAst::new(&program), Arc::new(flow_default_registry()))
            .expect("engine builds");
        let mut resolver = GateResolver {
            cache,
            labels: Vec::new(),
        };
        match drive(&mut engine, &mut resolver, &BreakpointSet::new()).expect("drive") {
            DriveOutcome::Done(_) => {}
            DriveOutcome::Break { at } => panic!("unexpected break: {at:?}"),
        }
        let side = if cache == "true" {
            "cached reply; the live Par cascade never spawned"
        } else {
            "live fan-out; the cached side never spawned"
        };
        println!("  cache_lookup -> {cache:<5} took the {side}");
        println!(
            "    resolved: {}   (visit_pre = {})",
            resolver.labels.join(", "),
            engine.events().visit_pre
        );
    }
}

/// The rescue for the fail-loud section above: with
/// `flow_syntax_overrides` (grammar layer) and the
/// `#[dsl_build(with = ...)]` converters (build layer) the same schema
/// goes canonical text → generated grammar → typed `Flow` → engine.
fn demonstrate_text_round_trip() {
    use dsl_kit_parse::DslBuild;

    let text = r#"Par(policy: any_failfast, reducer_id: "reduce_any_first_winner",
                    children: [Call(label: "search_arxiv"), Call(label: "search_web")])"#;
    let grammar = dsl_kit_parse::schema_gen::checked_grammar_from_schema_with(
        &Flow::schema(),
        &IdGen::new(),
        &flow_syntax_overrides(),
    )
    .expect("overridden Flow schema generates a clean grammar");
    println!(
        "  {} rules generated with flow_syntax_overrides()",
        grammar.rules.len()
    );
    println!(
        "  text: {}",
        text.lines().map(str::trim).collect::<Vec<_>>().join(" ")
    );

    let tree = grammar.parse(text).expect("canonical Flow text parses");
    let flow = Flow::from_parse_tree(&tree, &IdGen::new()).expect("typed build");
    println!("  typed: {}", pretty(&flow).trim_end().replace('\n', " / "));

    let mut engine = Engine::new(FlowAst::new(&flow), Arc::new(flow_default_registry()))
        .expect("root Ast validation");
    let mut resolver = CannedResolver::default();
    match drive(&mut engine, &mut resolver, &BreakpointSet::new()).expect("drive to done") {
        DriveOutcome::Done(v) => println!("  engine result: {v:?}"),
        DriveOutcome::Break { .. } => unreachable!("no breakpoints registered"),
    }
}

fn par_three_searches(ids: &IdGen) -> Flow {
    Flow::Par {
        id: ids.node(),
        children: vec![
            Flow::Call {
                id: ids.node(),
                label: "search_arxiv".into(),
            },
            Flow::Call {
                id: ids.node(),
                label: "search_github".into(),
            },
            Flow::Call {
                id: ids.node(),
                label: "search_web".into(),
            },
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

    println!("\n=== Running the pipeline (sans-io drive) ===");
    let mut engine = Engine::new(FlowAst::new(&program), Arc::new(flow_default_registry()))
        .map_err(miette::Report::new)?;
    let mut resolver = CannedResolver::default();
    match drive(&mut engine, &mut resolver, &BreakpointSet::new()).expect("drive") {
        DriveOutcome::Done(v) => println!("  pipeline value: {v:?}"),
        DriveOutcome::Break { at } => {
            println!("  unexpected break with {} suspension(s)", at.len());
        }
    }

    println!("\n=== Event summary ===");
    println!("  {}", engine.event_summary());

    println!("\n=== Recorded results ===");
    let mut results = resolver.results.clone();
    results.sort_by_key(|(id, _)| id.0);
    for (id, text) in &results {
        println!("  {id}: {text}");
    }

    println!("\n=== Async run via drive_async (sequential sans-io driver) ===");
    let mut engine = Engine::new(FlowAst::new(&program), Arc::new(flow_default_registry()))
        .map_err(miette::Report::new)?;
    let mut resolver = SlowCannedResolver {
        latency: Duration::from_millis(20),
        resolved: 0,
    };
    let seq_start = Instant::now();
    match drive_async(&mut engine, &mut resolver, &BreakpointSet::new())
        .await
        .expect("drive_async")
    {
        DriveOutcome::Done(_) => println!(
            "  {} calls resolved sequentially in {} ms — the driver awaits one \
             call at a time; compare the fan-out section below.",
            resolver.resolved,
            seq_start.elapsed().as_millis()
        ),
        DriveOutcome::Break { at } => {
            println!("  unexpected break with {} suspension(s)", at.len());
        }
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

    println!("\n=== Value-dependent branch (cache gate; untaken side never spawns) ===");
    demonstrate_value_branch();

    println!("\n=== Drive with a live breakpoint (halt before a Par kid spawns) ===");
    let target = find_call_node(&program, "search_github").expect("pipeline has search_github");
    let mut engine = Engine::new(FlowAst::new(&program), Arc::new(flow_default_registry()))
        .map_err(miette::Report::new)?;
    let mut resolver = CannedResolver::default();
    let mut bps = BreakpointSet::new();
    bps.add(BreakCondition::at_node(target));
    println!("  breakpoint set on {target} (Call \"search_github\", 2nd Par kid)");
    match drive(&mut engine, &mut resolver, &bps).expect("drive to break") {
        DriveOutcome::Break { at } => {
            println!("  halted BEFORE {target} spawned; live suspensions at the halt:");
            for p in &at {
                println!("    {:>4} {}", p.at.node.to_string(), p.reason);
            }
        }
        DriveOutcome::Done(_) => println!("  unexpected completion without a break"),
    }
    match drive(&mut engine, &mut resolver, &bps).expect("drive to done") {
        DriveOutcome::Done(v) => println!("  resumed to completion: {v:?}"),
        DriveOutcome::Break { at } => {
            println!("  halted again with {} suspension(s)", at.len());
        }
    }

    println!("\n=== Schema-driven grammar generation (designed fail-loud) ===");
    match dsl_kit_parse::schema_gen::grammar_from_schema(&Flow::schema(), &IdGen::new()) {
        Ok(g) => println!("  unexpectedly generated {} rules", g.rules.len()),
        Err(e) => {
            println!("  Flow carries payload types with no canonical text syntax;");
            println!("  generation lists every offender instead of silently dropping variants:");
            for d in &e.diagnostics {
                println!("    [{}] {}", d.code, d.message);
            }
        }
    }

    println!("\n=== ...rescued by the per-field hooks (text -> typed -> engine) ===");
    demonstrate_text_round_trip();

    println!("\n=== Breakpoints (static condition matching) ===");
    demonstrate_breakpoints(&program);

    println!("\n=== Error rendering (malformed AST) ===");
    let broken = Flow::Seq {
        id: NodeId(100),
        children: vec![
            Flow::Call {
                id: NodeId(101),
                label: "one".into(),
            },
            Flow::Call {
                id: NodeId(101),
                label: "two".into(),
            },
        ],
    };
    if let Err(err) = check_unique_ids(&broken) {
        println!("{:?}", miette::Report::new(err));
    }

    // Silence unused warning when Stepper isn't invoked directly here.
    let _ = std::marker::PhantomData::<StepOutcome<FlowValue>>;

    Ok(())
}
