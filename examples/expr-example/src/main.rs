//! Runs the arithmetic DSL end to end.
//!
//! The binary shows three things:
//!
//! 1. A synchronous evaluation of the demo expression that supplies
//!    every unbound variable through a plain closure (no MCP surface
//!    involved).
//! 2. A text round-trip through the **schema-generated grammar**: the
//!    parser for the canonical `Let(name: "x", ...)` syntax is derived
//!    from `Expr::schema()` at runtime — nobody wrote a grammar. Text
//!    parses to a `ParseTree`, conformance-checks against the schema,
//!    builds a typed `Expr` via `#[derive(DslBuild)]`, and evaluates.
//! 3. The same program driven through the `DslHost` trait, proving
//!    that the MCP handler works against DSLs whose shape is very
//!    different from `flow-dsl`.
//!
//! Because the example depends on `dsl-kit-mcp`, the second half of
//! the demo can also construct a full `DslMcpHandler` around
//! `ExprHost` and call the same tools an MCP client would.

use dsl_kit::{BreakCondition, BreakpointSet, IdGen, NodeId};
use dsl_kit_mcp::host::DslHost;
use dsl_kit_parse::{DslBuild, check_conformance, schema_gen};
use dsl_kit_schema::DslSchema;
use expr_dsl::{Expr, demo_program, evaluate_all, pretty};
use expr_host::ExprHost;

/// Parses canonical text with the schema-generated grammar, builds the
/// typed AST, and evaluates it — the "two derives, parser for free"
/// path landed in Round 29.
fn run_schema_generated_grammar_demo() -> miette::Result<()> {
    let grammar = schema_gen::checked_grammar_from_schema(&Expr::schema(), &IdGen::new())
        .map_err(|e| miette::miette!("grammar generation failed: {:?}", e.diagnostics))?;
    println!(
        "generated {} rules from Expr::schema() (start rule `{}`)",
        grammar.rules.len(),
        grammar.start
    );

    // Same program as the hand-built demo: let x = 3 in (x + y) * z.
    let text = r#"
        Let(
            name: "x",
            value: Lit(value: 3),
            body: Mul(
                lhs: Add(lhs: Var(name: "x"), rhs: Var(name: "y")),
                rhs: Var(name: "z")
            )
        )
    "#;
    let tree = grammar
        .parse(text)
        .map_err(|e| miette::miette!("parse failed: {:?}", e.diagnostics))?;
    let diags = check_conformance(&tree, &Expr::schema());
    println!(
        "parsed `{}` tree; conformance diagnostics: {}",
        tree.variant,
        diags.len()
    );

    let ids = IdGen::new();
    let expr = Expr::from_parse_tree(&tree, &ids)
        .map_err(|e| miette::miette!("typed build failed: {:?}", e.diagnostics))?;
    print!("{}", pretty(&expr));
    let value = evaluate_all(&expr, |name| match name {
        "y" => Some(5),
        "z" => Some(2),
        _ => None,
    })?;
    println!("text-parsed program with y=5, z=2 -> {value}");

    // Parse failures speak the shared diagnostic dialect (farthest
    // failure + expected set).
    if let Err(e) = grammar.parse("Add(lhs: Lit(value: 1) rhs: Unit())") {
        println!("deliberate error (missing comma):");
        for d in &e.diagnostics {
            println!("  [{}] {}", d.code, d.message);
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> miette::Result<()> {
    // ---- 1. Synchronous evaluation ---------------------------------
    let ids = IdGen::new();
    let program = demo_program(&ids);

    println!("=== Expression AST ===");
    print!("{}", pretty(&program));

    let value = evaluate_all(&program, |name| match name {
        "y" => Some(5),
        "z" => Some(2),
        _ => None,
    })?;
    println!("\nsynchronous eval: (let x = 3 in (x + y) * z) with y=5, z=2 -> {value}");

    // ---- 2. Text round-trip via the schema-generated grammar -------
    println!("\n=== Schema-generated grammar (text -> typed AST -> eval) ===");
    run_schema_generated_grammar_demo()?;

    // ---- 3. Driving the same program through DslHost ---------------
    println!("\n=== DslHost run ===");
    let mut host = ExprHost::new_with_default_program();
    let bp = BreakpointSet::new();

    // First step: the host yields on the first unbound variable it sees.
    let first = host.step_one(&bp).await.expect("step ok");
    print_outcome("step 1", &first);
    let resolved = host.resolve(Some("5".into())).await.expect("resolve ok");
    println!("resolved: {} = {}", resolved.label, resolved.result);

    let second = host.step_one(&bp).await.expect("step ok");
    print_outcome("step 2", &second);
    let resolved = host.resolve(Some("2".into())).await.expect("resolve ok");
    println!("resolved: {} = {}", resolved.label, resolved.result);

    let third = host.step_one(&bp).await.expect("step ok");
    print_outcome("step 3", &third);

    let snap = host.snapshot();
    println!("\nresults:");
    for (id, entry) in &snap.results {
        println!("  n{id}: {entry}");
    }

    // ---- 4. Same host, but with a breakpoint on n1 (the Lit inside Let) --
    println!("\n=== DslHost run with a breakpoint ===");
    host.reset();
    let mut bp = BreakpointSet::new();
    // Break on the first Var lookup for `y` (node ID varies with the
    // build; find it by walking).
    if let Some(y_id) = find_var_node(&program, "y") {
        bp.add(BreakCondition::at_node(y_id));
        println!("breakpoint set on {y_id} (Var \"y\")");
    }

    let outcome = host.step_one(&bp).await.expect("step");
    print_outcome("first step (should hit await-effect on y)", &outcome);
    if let Ok(outcome) = host.step_one(&bp).await {
        print_outcome("second step (breakpoint on y should fire before eval)", &outcome);
    }

    println!("\n(For an MCP-driven session, install this binary or point");
    println!(" a client at dsl-kit-mcp with a FlowHost — the same tools work here.)");

    Ok(())
}

fn print_outcome(label: &str, outcome: &dsl_kit_mcp::host::HostOutcome) {
    match outcome {
        dsl_kit_mcp::host::HostOutcome::Advanced => println!("{label}: advanced"),
        dsl_kit_mcp::host::HostOutcome::Suspended { reason, at } => {
            println!(
                "{label}: suspended (reason={reason}, node=n{}, path={:?}, depth={})",
                at.node, at.path, at.depth
            );
        }
        dsl_kit_mcp::host::HostOutcome::Done => println!("{label}: done"),
    }
}

fn find_var_node(expr: &expr_dsl::Expr, target: &str) -> Option<NodeId> {
    use dsl_kit::{Phase, Walk};
    let mut found: Option<NodeId> = None;
    expr.walk(&mut |node, phase| {
        if phase != Phase::Pre {
            return;
        }
        if let expr_dsl::Expr::Var { id, name } = node {
            if name == target && found.is_none() {
                found = Some(*id);
            }
        }
    });
    found
}

