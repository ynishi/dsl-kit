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
    let body = dsl_kit_mcp::handler::StepParams {
        mode: Some(mode.into()),
    };
    parse(&h.dsl_kit_step(Parameters(body)).await.expect("step ok"))
}

async fn call_resolve(h: &DslMcpHandler, result: Option<&str>) -> Value {
    let body = dsl_kit_mcp::handler::ResolveParams {
        result: result.map(str::to_owned),
    };
    parse(
        &h.dsl_kit_resolve(Parameters(body))
            .await
            .expect("resolve ok"),
    )
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
    parse(
        &h.dsl_kit_breakpoint_add(Parameters(body))
            .await
            .expect("bp add ok"),
    )
}

async fn call_bp_list(h: &DslMcpHandler) -> Value {
    parse(&h.dsl_kit_breakpoint_list().await.expect("bp list ok"))
}

async fn call_reset(h: &DslMcpHandler) -> Value {
    parse(&h.dsl_kit_reset().await.expect("reset ok"))
}

async fn call_pending(h: &DslMcpHandler) -> Value {
    parse(&h.dsl_kit_pending().await.expect("pending ok"))
}

async fn call_take_cancellations(h: &DslMcpHandler) -> Value {
    parse(
        &h.dsl_kit_take_cancellations()
            .await
            .expect("take_cancellations ok"),
    )
}

async fn call_resolve_by_id_ok(h: &DslMcpHandler, id: u64, ok: &str) -> Value {
    let body = dsl_kit_mcp::handler::ResolveByIdParams {
        id,
        ok: Some(ok.into()),
        err: None,
    };
    parse(
        &h.dsl_kit_resolve_by_id(Parameters(body))
            .await
            .expect("resolve_by_id ok"),
    )
}

async fn call_resolve_by_id_err(h: &DslMcpHandler, id: u64, code: &str, message: &str) -> Value {
    let body = dsl_kit_mcp::handler::ResolveByIdParams {
        id,
        ok: None,
        err: Some(dsl_kit_mcp::handler::ResolveErr {
            code: code.into(),
            message: message.into(),
        }),
    };
    parse(
        &h.dsl_kit_resolve_by_id(Parameters(body))
            .await
            .expect("resolve_by_id err"),
    )
}

/// Drive the handler forward until the Par fan-out has emitted its 3
/// pending suspensions, then return their ids in emit order (matches
/// child declaration order: search_arxiv, search_github, search_web).
async fn advance_to_par_fanout(h: &DslMcpHandler) -> Vec<u64> {
    // First yield: suspended on Call(fetch_query).
    let first = call_step(h, "to_yield").await;
    assert_eq!(first["kind"], "suspended");
    // Resolve fetch_query so the pipeline can enter the Scope + Par.
    let _ = call_resolve(h, Some("q".into())).await;
    // Keep stepping until pending has 3 entries.
    for _ in 0..16 {
        let _ = call_step(h, "to_yield").await;
        let pending = call_pending(h).await;
        let arr = pending["pending"].as_array().expect("pending is array");
        if arr.len() == 3 {
            return arr.iter().map(|p| p["id"].as_u64().unwrap()).collect();
        }
    }
    panic!("Par fan-out did not emit 3 pending within step budget");
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

    // Pause on node 1 (Call "fetch_query", outside the Par fan-out).
    // Breakpoints inside Par children are not honoured in the Commit
    // B1 fan-out schedule because those children are spawned as a
    // batch of pending suspensions rather than stepped-into one by
    // one; that semantic gap is tracked for a later commit.
    call_bp_add(&handler, 1).await;

    // Run to yield: should hit the breakpoint at n1 before its Call
    // fires.
    let first = call_step(&handler, "to_yield").await;
    assert_eq!(first["kind"], "suspended");
    assert_eq!(first["reason"], "breakpoint");
    assert_eq!(first["at"]["node"], 1);

    // Continue: transitions past the breakpoint and reaches the n1
    // Call's own suspension.
    let second = call_step(&handler, "to_yield").await;
    assert_eq!(second["kind"], "suspended");
    assert_eq!(second["reason"], "call(fetch_query)");
    assert_eq!(second["at"]["node"], 1);
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
    assert!(
        uris.contains(&"dsl-kit://kit/error-catalog"),
        "missing kit error-catalog"
    );
    assert!(
        uris.contains(&"dsl-kit://dsl/flow/grammar"),
        "missing flow grammar"
    );
    assert!(
        uris.contains(&"dsl-kit://dsl/flow/samples/research-pipeline"),
        "missing flow sample"
    );
}

#[tokio::test]
async fn without_kit_resources_drops_kit_layer_only() {
    let handler =
        DslMcpHandler::new(Box::new(FlowHost::new_with_default_program())).without_kit_resources();
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

// ---------- Fan-out surface (Commit B2) ---------------------------------

#[tokio::test]
async fn pending_lists_three_after_par_entry() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let ids = advance_to_par_fanout(&handler).await;
    assert_eq!(ids.len(), 3);

    // Every entry carries an id, a `call` reason, and a non-empty label
    // drawn from the flow program.
    let pending = call_pending(&handler).await;
    let entries = pending["pending"].as_array().unwrap();
    let mut labels: Vec<&str> = entries
        .iter()
        .map(|e| {
            assert_eq!(e["reason"], "call");
            e["label"].as_str().unwrap()
        })
        .collect();
    labels.sort();
    assert_eq!(labels, vec!["search_arxiv", "search_github", "search_web"]);
}

#[tokio::test]
async fn resolve_by_id_ok_fills_one_slot() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let ids = advance_to_par_fanout(&handler).await;

    // Pick a specific pending id (the middle one — search_github) and
    // resolve it. Pending should shrink to 2.
    let target = ids[1];
    let resolved = call_resolve_by_id_ok(&handler, target, "gh-response").await;
    assert_eq!(resolved["resolved"]["result"], "gh-response");
    assert_eq!(resolved["resolved"]["label"], "search_github");

    let pending_after = call_pending(&handler).await;
    let arr = pending_after["pending"].as_array().unwrap();
    assert_eq!(arr.len(), 2);
    let remaining_ids: Vec<u64> = arr.iter().map(|e| e["id"].as_u64().unwrap()).collect();
    assert!(!remaining_ids.contains(&target));
}

#[tokio::test]
async fn full_round_out_of_order_resolve_records_all_three_results() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let ids = advance_to_par_fanout(&handler).await;

    // Resolve in reverse order: 2, 0, 1.
    let _ = call_resolve_by_id_ok(&handler, ids[2], "web-resp").await;
    let _ = call_resolve_by_id_ok(&handler, ids[0], "arxiv-resp").await;
    let _ = call_resolve_by_id_ok(&handler, ids[1], "gh-resp").await;

    // No pending left for the Par slots.
    let pending_after = call_pending(&handler).await;
    assert!(pending_after["pending"].as_array().unwrap().is_empty());

    // Drive to completion so the pipeline records every Call's result.
    let outcome = call_step(&handler, "to_done").await;
    assert_eq!(outcome["kind"], "done");

    // The three per-Call responses we injected must appear verbatim in
    // state.results.
    let state = call_state(&handler).await;
    let result_strs: Vec<&str> = state["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["result"].as_str().unwrap())
        .collect();
    assert!(result_strs.contains(&"arxiv-resp"));
    assert!(result_strs.contains(&"gh-resp"));
    assert!(result_strs.contains(&"web-resp"));
    // Total call count for the research pipeline is 7 (fetch_query,
    // 3× Par, synthesise, citation_check, write_report).
    assert_eq!(state["results"].as_array().unwrap().len(), 7);
}

#[tokio::test]
async fn resolve_by_id_err_triggers_failfast_and_cancels_siblings() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let ids = advance_to_par_fanout(&handler).await;

    // Fail the middle slot. resolve_by_id itself records the failure
    // successfully; the FailFast propagation surfaces on the next step.
    let _ = call_resolve_by_id_err(&handler, ids[1], "timeout", "gh timed out").await;

    // Next step must report an error via the underlying handler.
    let body = dsl_kit_mcp::handler::StepParams {
        mode: Some("one".into()),
    };
    let err = handler
        .dsl_kit_step(Parameters(body))
        .await
        .expect_err("failfast must surface as tool error");
    assert!(
        err.contains("timeout") || err.contains("timed out"),
        "unexpected error message: {err}"
    );

    // The two sibling slots should now appear in the cancellation drain.
    let cancels = call_take_cancellations(&handler).await;
    let cancelled_ids: Vec<u64> = cancels["cancelled"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert!(cancelled_ids.contains(&ids[0]));
    assert!(cancelled_ids.contains(&ids[2]));
    // The failed slot itself may or may not be included in the drain
    // (implementation-defined); assert only the sibling coverage.

    // A second drain returns an empty list (drain is exhaustive).
    let cancels_again = call_take_cancellations(&handler).await;
    assert!(cancels_again["cancelled"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn take_cancellations_is_empty_on_happy_path() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let outcome = call_step(&handler, "to_done").await;
    assert_eq!(outcome["kind"], "done");

    let cancels = call_take_cancellations(&handler).await;
    assert!(cancels["cancelled"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn resolve_by_id_rejects_both_ok_and_err_together() {
    let handler = DslMcpHandler::new(Box::new(FlowHost::new_with_default_program()));
    let ids = advance_to_par_fanout(&handler).await;

    let body = dsl_kit_mcp::handler::ResolveByIdParams {
        id: ids[0],
        ok: Some("x".into()),
        err: Some(dsl_kit_mcp::handler::ResolveErr {
            code: "c".into(),
            message: "m".into(),
        }),
    };
    let err = handler
        .dsl_kit_resolve_by_id(Parameters(body))
        .await
        .expect_err("must reject");
    assert!(err.contains("exactly one"));
}
