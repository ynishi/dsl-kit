//! In-process smoke test for the dsl-kit MCP handler.
//!
//! Calls each tool method directly rather than through a full stdio
//! round-trip; the goal is to catch schema / logic drift, not to
//! exercise the transport (rmcp has its own tests for that).

use dsl_kit_mcp::DslMcpHandler;
use flow_host::FlowHost;
use rmcp::handler::server::wrapper::Parameters;
use serde_json::Value;

fn parse(json: &str) -> Value {
    serde_json::from_str(json).expect("valid JSON")
}

// The `#[tool]` macro on the handler exposes each method through the
// tool router; here we invoke the underlying inherent methods directly.

async fn call_info(h: &DslMcpHandler) -> Value {
    parse(&h.dsl_kit_info().await.expect("info ok"))
}

async fn call_ast(h: &DslMcpHandler) -> Value {
    parse(&h.dsl_kit_ast().await.expect("ast ok"))
}

async fn call_state(h: &DslMcpHandler) -> Value {
    parse(&h.dsl_kit_state().await.expect("state ok"))
}

async fn call_step(h: &DslMcpHandler, mode: &str) -> Value {
    let body = dsl_kit_mcp::handler::StepParams { mode: Some(mode.into()) };
    parse(&h.dsl_kit_step(Parameters(body)).await.expect("step ok"))
}

async fn call_resolve(h: &DslMcpHandler, result: Option<&str>) -> Value {
    let body = dsl_kit_mcp::handler::ResolveParams {
        result: result.map(str::to_owned),
    };
    parse(&h.dsl_kit_resolve(Parameters(body)).await.expect("resolve ok"))
}

async fn call_bp_add(h: &DslMcpHandler, node: u64) -> Value {
    let body = dsl_kit_mcp::handler::BreakpointAddParams {
        at_node: Some(node),
        at_depth: None,
        at_depth_at_least: None,
        at_depth_at_most: None,
        at_iteration: None,
        under_path: None,
    };
    parse(&h.dsl_kit_breakpoint_add(Parameters(body)).await.expect("bp add ok"))
}

async fn call_bp_list(h: &DslMcpHandler) -> Value {
    parse(&h.dsl_kit_breakpoint_list().await.expect("bp list ok"))
}

async fn call_reset(h: &DslMcpHandler) -> Value {
    parse(&h.dsl_kit_reset().await.expect("reset ok"))
}

async fn call_explain(h: &DslMcpHandler, code: Option<&str>) -> Result<Value, String> {
    let body = dsl_kit_mcp::handler::ExplainParams {
        code: code.map(str::to_owned),
    };
    h.dsl_kit_explain(Parameters(body)).await.map(|s| parse(&s))
}

#[tokio::test]
async fn info_reports_flow_dsl() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let info = call_info(&handler).await;
    assert_eq!(info["kit"], "dsl-kit");
    assert_eq!(info["dsl"], "flow");
    assert!(info["ast_size"].as_u64().unwrap() >= 10);
}

#[tokio::test]
async fn ast_pretty_contains_expected_labels() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let ast = call_ast(&handler).await;
    let pretty = ast["pretty"].as_str().expect("pretty is a string");
    assert!(pretty.contains("Seq"));
    assert!(pretty.contains("Par"));
    assert!(pretty.contains("fetch_query"));
}

#[tokio::test]
async fn stepping_to_done_resolves_every_call() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let outcome = call_step(&handler, "to_done").await;
    assert_eq!(outcome["kind"], "done");

    let state = call_state(&handler).await;
    let results = state["results"].as_array().unwrap();
    // The research pipeline defines seven calls.
    assert_eq!(results.len(), 7);
}

#[tokio::test]
async fn manual_resolve_after_yield() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    // Step until the first call yields.
    let first = call_step(&handler, "to_yield").await;
    assert_eq!(first["kind"], "suspended");

    // The state should now report a pending call.
    let state = call_state(&handler).await;
    let pending = &state["suspended_call"];
    assert!(pending.is_object());

    // Provide a custom answer.
    let resolved = call_resolve(&handler, Some("42")).await;
    assert_eq!(resolved["resolved"]["result"], "42");
}

#[tokio::test]
async fn breakpoints_survive_list() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let added = call_bp_add(&handler, 4).await;
    let id = added["id"].as_u64().unwrap();

    let list = call_bp_list(&handler).await;
    let entries = list["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["id"].as_u64().unwrap(), id);
    assert_eq!(entries[0]["condition"]["kind"], "node");
}

#[tokio::test]
async fn breakpoint_pauses_stepper_at_matching_node() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));

    // Pause on node 4 (Call "search_arxiv").
    call_bp_add(&handler, 4).await;

    // Run to yield: should first hit the initial call at n1 (AwaitEffect).
    let first = call_step(&handler, "to_yield").await;
    assert_eq!(first["kind"], "suspended");
    assert_eq!(first["reason"], "await-effect");
    assert_eq!(first["at"]["node"], 1);
    call_resolve(&handler, Some("ok")).await;

    // Continue: should now hit the breakpoint at n4 before any await.
    let second = call_step(&handler, "to_yield").await;
    assert_eq!(second["kind"], "suspended");
    assert_eq!(second["reason"], "breakpoint");
    assert_eq!(second["at"]["node"], 4);

    // Stepping again transitions past the breakpoint and reaches the
    // n4 Call's own AwaitEffect suspension.
    let third = call_step(&handler, "to_yield").await;
    assert_eq!(third["kind"], "suspended");
    assert_eq!(third["reason"], "await-effect");
    assert_eq!(third["at"]["node"], 4);
}

#[tokio::test]
async fn explain_returns_help_for_known_code() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let body = call_explain(&handler, Some("dsl_kit::eval::aborted"))
        .await
        .expect("known code");
    assert_eq!(body["code"], "dsl_kit::eval::aborted");
    assert!(body["help"].as_str().unwrap().contains("Aborted"));
}

#[tokio::test]
async fn explain_lists_catalog_when_code_omitted() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let body = call_explain(&handler, None).await.expect("catalog");
    let codes: Vec<&str> = body["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(codes.contains(&"dsl_kit::eval::aborted"));
    assert!(codes.contains(&"dsl_kit::stepper::protocol"));
}

#[tokio::test]
async fn explain_rejects_unknown_code() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let err = call_explain(&handler, Some("dsl_kit::does_not_exist"))
        .await
        .expect_err("unknown code should error");
    assert!(err.contains("unknown error code"));
}

#[tokio::test]
async fn resources_default_includes_kit_and_dsl_layers() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let entries = handler.all_resources().await;
    let uris: Vec<&str> = entries.iter().map(|e| e.uri.as_str()).collect();
    assert!(uris.contains(&"dsl-kit://kit/intro"), "missing kit intro");
    assert!(uris.contains(&"dsl-kit://kit/error-catalog"), "missing kit error-catalog");
    assert!(uris.contains(&"dsl-kit://dsl/flow/grammar"), "missing flow grammar");
    assert!(
        uris.contains(&"dsl-kit://dsl/flow/samples/research-pipeline"),
        "missing flow sample"
    );
}

#[tokio::test]
async fn without_kit_resources_drops_kit_layer_only() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()))
        .without_kit_resources();
    let entries = handler.all_resources().await;
    for entry in &entries {
        assert!(
            !entry.uri.starts_with("dsl-kit://kit/"),
            "kit layer leaked through opt-out: {}",
            entry.uri
        );
    }
    // Layer B (dsl-kit://dsl/*) must still be present.
    assert!(entries.iter().any(|e| e.uri.starts_with("dsl-kit://dsl/")));
}

#[tokio::test]
async fn reset_starts_from_scratch() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let _ = call_step(&handler, "to_done").await;
    let before = call_state(&handler).await;
    assert!(before["results"].as_array().unwrap().len() > 0);

    call_reset(&handler).await;
    let after = call_state(&handler).await;
    assert_eq!(after["results"].as_array().unwrap().len(), 0);
    assert_eq!(after["events"]["visit_pre"].as_u64().unwrap(), 0);
}
