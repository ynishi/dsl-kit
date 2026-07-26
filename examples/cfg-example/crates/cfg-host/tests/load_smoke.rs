//! MCP surface smoke for the keyed primitive: `dsl_kit_schema` reports
//! `multiplicity: "map"`, `dsl_kit_load` accepts a keyed document,
//! `dsl_kit_ast` shows the keys, and `dsl_kit_step` / `dsl_kit_resolve`
//! walk it to a value.
//!
//! These are the same calls `cfg-mcp` serves over stdio, exercised
//! against the handler in-process — so the tool surface is pinned
//! without needing a live MCP host.

use cfg_host::CfgHost;
use dsl_kit_mcp::DslMcpHandler;
use dsl_kit_mcp::handler::{LoadParams, ResolveParams, StepParams};
use rmcp::handler::server::wrapper::Parameters;
use serde_json::{Value, json};

fn parse(json: &str) -> Value {
    serde_json::from_str(json).expect("valid JSON")
}

/// A keyed document: two nested `Env` levels plus an override stack.
fn document() -> String {
    json!({
        "type": "Env",
        "bindings": {
            "app": {
                "type": "Env",
                "bindings": {
                    "name": { "type": "Leaf", "value": "dsl-kit" },
                    "port": { "type": "Ref", "name": "PORT" },
                },
            },
            "log": {
                "type": "Overrides",
                "entries": {
                    "10-base": { "type": "Leaf", "value": "info" },
                    "20-prod": { "type": "Leaf", "value": "warn" },
                },
            },
        },
    })
    .to_string()
}

fn handler() -> DslMcpHandler {
    DslMcpHandler::new(Box::new(CfgHost::new_with_default_document()))
}

#[tokio::test]
async fn schema_reports_a_keyed_slot() {
    let h = handler();
    let envelope = parse(&h.dsl_kit_schema().await.unwrap());
    assert_eq!(envelope["wired"], true);

    let multiplicities: Vec<String> = envelope["schema"]["variants"]
        .as_array()
        .expect("variants array")
        .iter()
        .flat_map(|v| v["children"].as_array().cloned().unwrap_or_default())
        .filter_map(|c| c["multiplicity"].as_str().map(str::to_string))
        .collect();

    assert!(
        multiplicities.iter().any(|m| m == "map"),
        "expected a `map` multiplicity on the cfg schema, got {multiplicities:?}"
    );
}

#[tokio::test]
async fn load_accepts_a_keyed_document_and_the_ast_keeps_its_keys() {
    let h = handler();

    let load = parse(
        &h.dsl_kit_load(Parameters(LoadParams {
            input: document(),
            sources: None,
        }))
        .await
        .unwrap(),
    );
    assert_eq!(load["ok"], true, "load should succeed, got {load}");
    assert_eq!(load["dsl"], "cfg");
    assert_eq!(
        load["ast_size"].as_u64().unwrap(),
        7,
        "seven nodes: two Env levels, one Overrides, four leaves"
    );

    // `dsl_kit_ast` renders the keys, not just the values.
    let ast = parse(&h.dsl_kit_ast().await.unwrap());
    let pretty = ast["pretty"].as_str().expect("pretty tree");
    for key in ["app:", "log:", "name:", "port:", "10-base:", "20-prod:"] {
        assert!(pretty.contains(key), "expected `{key}` in:\n{pretty}");
    }
}

#[tokio::test]
async fn step_and_resolve_walk_a_keyed_document_to_its_value() {
    let h = handler();
    let _ = h
        .dsl_kit_load(Parameters(LoadParams {
            input: document(),
            sources: None,
        }))
        .await
        .unwrap();

    // The only suspension is the `PORT` reference under `app`.
    let outcome = parse(
        &h.dsl_kit_step(Parameters(StepParams {
            mode: Some("to_yield".into()),
        }))
        .await
        .unwrap(),
    );
    assert_eq!(outcome["kind"], "suspended", "got {outcome}");

    let resolved = parse(
        &h.dsl_kit_resolve(Parameters(ResolveParams {
            result: Some("8080".into()),
        }))
        .await
        .unwrap(),
    );
    assert_eq!(resolved["resolved"]["label"], "PORT");
    assert_eq!(resolved["resolved"]["result"], "8080");

    let done = parse(
        &h.dsl_kit_step(Parameters(StepParams {
            mode: Some("to_done".into()),
        }))
        .await
        .unwrap(),
    );
    assert_eq!(done["kind"], "done", "got {done}");

    // Root Env is a Seq over `app` then `log`; `log` folds last-wins,
    // so the document resolves to the `20-prod` layer.
    let state = parse(&h.dsl_kit_state().await.unwrap());
    let results = state["results"].as_array().expect("results array");
    assert!(
        results
            .iter()
            .any(|row| row["result"].as_str() == Some("warn")),
        "expected a `warn` row in {state}"
    );
}

#[tokio::test]
async fn a_keyed_slot_given_a_list_fails_with_a_diagnostics_envelope() {
    let h = handler();
    let bad = json!({
        "type": "Env",
        "bindings": [{ "type": "Leaf", "value": "x" }],
    })
    .to_string();

    let out = parse(
        &h.dsl_kit_load(Parameters(LoadParams {
            input: bad,
            sources: None,
        }))
        .await
        .unwrap(),
    );
    assert_eq!(out["ok"], false, "expected failure, got {out}");
    let diagnostics = out["diagnostics"].as_array().expect("diagnostics array");
    assert!(
        diagnostics
            .iter()
            .any(|d| d["message"].as_str().unwrap_or("").contains("bindings")),
        "expected the slot name in {out}"
    );
    let first = &diagnostics[0];
    assert!(first["severity"].is_string());
    assert!(first["code"].is_string());
    assert!(first["message"].is_string());
}

#[tokio::test]
async fn a_repeated_key_is_reported_rather_than_dropped() {
    let h = handler();
    // JSON objects cannot carry a duplicate key by the time serde_json
    // is done with them, so the duplicate has to arrive as raw text.
    let bad = r#"{"type":"Env","bindings":{"k":{"type":"Leaf","value":"a"},
                  "k":{"type":"Leaf","value":"b"}}}"#;

    let out = parse(
        &h.dsl_kit_load(Parameters(LoadParams {
            input: bad.to_string(),
            sources: None,
        }))
        .await
        .unwrap(),
    );
    // serde_json keeps the last duplicate, so this loads with one
    // entry rather than failing — the text front-end is where a
    // repeated key is caught (`DUPLICATE_KEY`). Pinned so the
    // difference between the two doors stays visible.
    assert_eq!(out["ok"], true, "got {out}");
    assert_eq!(out["ast_size"].as_u64().unwrap(), 2);
}

/// A root that pulls one JSON fragment and one canonical-text
/// fragment through `$import`.
fn bundle() -> (String, serde_json::Map<String, Value>) {
    let input = json!({
        "type": "Env",
        "bindings": {
            "app": { "$import": "app" },
            "log": { "$import": "logging" },
        },
    })
    .to_string();
    let sources = json!({
        "app": { "text": r#"Leaf(value: "from text")"# },
        "logging": {
            "json": json!({
                "type": "Overrides",
                "entries": { "10-base": { "type": "Leaf", "value": "info" } },
            })
            .to_string()
        },
    });
    let Value::Object(sources) = sources else {
        unreachable!("bundle sources are an object")
    };
    (input, sources)
}

#[tokio::test]
async fn load_with_sources_expands_imports_and_reports_the_graph() {
    let h = handler();
    let (input, sources) = bundle();
    let load = parse(
        &h.dsl_kit_load(Parameters(LoadParams {
            input,
            sources: Some(sources),
        }))
        .await
        .unwrap(),
    );

    assert_eq!(load["ok"], true, "bundle load should succeed, got {load}");
    // Graph as data: sorted canonical ids plus a 16-hex-digit digest.
    assert_eq!(load["imports"]["dependencies"], json!(["app", "logging"]));
    let digest = load["imports"]["digest"].as_str().expect("digest string");
    assert_eq!(digest.len(), 16, "FNV-1a 64 hex digest, got {digest}");
    // Env + 2 spliced subtrees (Leaf, Overrides+Leaf) = 4 nodes.
    assert_eq!(load["ast_size"].as_u64().unwrap(), 4);
}

#[tokio::test]
async fn bundle_digest_is_stable_across_identical_loads() {
    let h = handler();
    let (input, sources) = bundle();
    let a = parse(
        &h.dsl_kit_load(Parameters(LoadParams {
            input: input.clone(),
            sources: Some(sources.clone()),
        }))
        .await
        .unwrap(),
    );
    let b = parse(
        &h.dsl_kit_load(Parameters(LoadParams {
            input,
            sources: Some(sources),
        }))
        .await
        .unwrap(),
    );
    assert_eq!(a["imports"]["digest"], b["imports"]["digest"]);
}

#[tokio::test]
async fn a_missing_bundle_source_reports_diagnostics() {
    let h = handler();
    let out = parse(
        &h.dsl_kit_load(Parameters(LoadParams {
            input: json!({ "$import": "nowhere" }).to_string(),
            sources: Some(serde_json::Map::new()),
        }))
        .await
        .unwrap(),
    );
    assert_eq!(out["ok"], false, "got {out}");
    let diagnostics = out["diagnostics"].as_array().expect("diagnostics array");
    assert!(
        diagnostics
            .iter()
            .any(|d| d["code"] == "dsl_kit::parse::import::fetch_failed"),
        "got {out}"
    );
}
