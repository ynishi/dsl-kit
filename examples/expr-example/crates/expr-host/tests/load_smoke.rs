//! G-1d end-to-end smoke: JSON document → `dsl_kit_load` swaps the
//! program → `dsl_kit_lint` reports on the fresh AST → `dsl_kit_step`
//! and `dsl_kit_resolve` drive it to completion, and `dsl_kit_state`
//! surfaces `16` as the final result.
//!
//! Exercises the closed consumer loop the parser-design note (§3.3)
//! promises: "write JSON → load → the already-landed lint (L4.5) and
//! debugger (L3) apply to your program".

use dsl_kit_mcp::DslMcpHandler;
use dsl_kit_mcp::handler::{BreakpointAddParams, LoadParams, ResolveParams, StepParams};
use expr_host::ExprHost;
use rmcp::handler::server::wrapper::Parameters;
use serde_json::{Value, json};

fn parse(json: &str) -> Value {
    serde_json::from_str(json).expect("valid JSON")
}

fn program() -> String {
    // let x = 3 in (x + y) * z ⇒ (3 + 5) * 2 = 16 with host defaults.
    json!({
        "type": "Let",
        "name": "x",
        "value": { "type": "Lit", "value": 3 },
        "body": {
            "type": "Mul",
            "lhs": {
                "type": "Add",
                "lhs": { "type": "Var", "name": "x" },
                "rhs": { "type": "Var", "name": "y" },
            },
            "rhs": { "type": "Var", "name": "z" },
        },
    })
    .to_string()
}

fn handler() -> DslMcpHandler {
    DslMcpHandler::new(Box::new(ExprHost::new_with_default_program()))
}

#[tokio::test]
async fn load_then_lint_then_run_yields_sixteen() {
    let h = handler();

    // Establish a breakpoint against the default program so we can
    // later confirm dsl_kit_load clears it.
    let bp_body = BreakpointAddParams {
        at_node: Some(0),
        at_depth: None,
        at_depth_at_least: None,
        at_depth_at_most: None,
        at_iteration: None,
        under_path: None,
    };
    let _ = h.dsl_kit_breakpoint_add(Parameters(bp_body)).await.unwrap();
    let before = parse(&h.dsl_kit_breakpoint_list().await.unwrap());
    assert!(
        !before["entries"].as_array().unwrap().is_empty(),
        "sanity: bp added"
    );

    // Load a fresh program via the tool surface.
    let load = parse(
        &h.dsl_kit_load(Parameters(LoadParams { input: program() }))
            .await
            .unwrap(),
    );
    assert_eq!(load["ok"], true, "load should succeed, got {load}");
    assert_eq!(load["dsl"], "expr");
    assert!(
        load["ast_size"].as_u64().unwrap() >= 7,
        "expected at least 7 nodes, got {}",
        load["ast_size"]
    );

    // Breakpoints must be cleared — the old NodeIds are meaningless
    // against the new AST.
    let after = parse(&h.dsl_kit_breakpoint_list().await.unwrap());
    assert_eq!(
        after["entries"].as_array().unwrap().len(),
        0,
        "load should clear breakpoints, got {after}"
    );

    // Lint runs against the freshly loaded AST. We don't assert a
    // specific diagnostic set — only that the tool is wired and
    // returns a well-shaped envelope.
    let lint = parse(&h.dsl_kit_lint().await.unwrap());
    assert_eq!(lint["wired"], true);
    assert!(lint["diagnostics"].is_array());

    // Drive to done. ExprHost's `step_to_done` resolves unbound vars
    // with its default map (y=5, z=2), which matches the golden 16.
    let outcome = parse(
        &h.dsl_kit_step(Parameters(StepParams {
            mode: Some("to_done".into()),
        }))
        .await
        .unwrap(),
    );
    assert_eq!(
        outcome["kind"], "done",
        "expected step_to_done to reach Done, got {outcome}"
    );

    // The final value surfaces on the root node's result row. ExprHost
    // formats non-final rows as `"x = 3"` and the final row as the raw
    // integer, so we look for a row whose `result` contains `16`.
    let state = parse(&h.dsl_kit_state().await.unwrap());
    let results = state["results"].as_array().expect("results is an array");
    assert!(
        results.iter().any(|row| {
            row["result"]
                .as_str()
                .map(|s| s.contains("16"))
                .unwrap_or(false)
        }),
        "expected a `16` row in {state}"
    );
}

#[tokio::test]
async fn load_of_bad_json_returns_diagnostics_envelope() {
    let h = handler();
    let bad = json!({
        "type": "Ad", // typo
        "lhs": { "type": "Lit", "value": 1 },
    })
    .to_string();
    let out = parse(
        &h.dsl_kit_load(Parameters(LoadParams { input: bad }))
            .await
            .unwrap(),
    );
    assert_eq!(out["ok"], false, "expected failure, got {out}");
    let diagnostics = out["diagnostics"]
        .as_array()
        .expect("diagnostics is an array");
    assert!(
        diagnostics
            .iter()
            .any(|d| d["message"].as_str().unwrap_or("").contains("Add")),
        "expected the candidate `Add` in {out}"
    );
    // Confirm the unified envelope shape at the diagnostic level too.
    let first = &diagnostics[0];
    assert!(first["severity"].is_string());
    assert!(first["code"].is_string());
    assert!(first["message"].is_string());
    assert!(first.get("location").is_some());
}

#[tokio::test]
async fn ids_restart_on_reload() {
    let h = handler();

    let a = parse(
        &h.dsl_kit_load(Parameters(LoadParams { input: program() }))
            .await
            .unwrap(),
    );
    let root_a = a["root"].as_u64().unwrap();

    let b = parse(
        &h.dsl_kit_load(Parameters(LoadParams { input: program() }))
            .await
            .unwrap(),
    );
    let root_b = b["root"].as_u64().unwrap();

    // Fresh IdGen per load ⇒ root ids match across identical inputs.
    assert_eq!(
        root_a, root_b,
        "expected identical root ids across identical inputs, got {root_a} vs {root_b}"
    );
}

#[tokio::test]
async fn resolve_after_load_records_result() {
    let h = handler();
    let _ = h
        .dsl_kit_load(Parameters(LoadParams { input: program() }))
        .await
        .unwrap();

    // Step to the first suspension.
    let outcome = parse(
        &h.dsl_kit_step(Parameters(StepParams {
            mode: Some("to_yield".into()),
        }))
        .await
        .unwrap(),
    );
    assert_eq!(outcome["kind"], "suspended");

    // Supply a value for the pending variable and confirm the resolve
    // succeeds against the freshly loaded AST.
    let resolved = parse(
        &h.dsl_kit_resolve(Parameters(ResolveParams {
            result: Some("5".into()),
        }))
        .await
        .unwrap(),
    );
    assert_eq!(resolved["resolved"]["result"], "5");
}
