//! JSON front-end coverage for the keyed-slot DSL: JSON document →
//! `ParseTree` (serde bridge) → typed `Cfg` with fresh `NodeId`s →
//! resolution through the engine.
//!
//! This is the path `dsl_kit_load` takes on the MCP surface, so what
//! passes here is what an MCP client can send.

use cfg_dsl::{Cfg, flatten, resolve_all};
use dsl_kit::IdGen;
use dsl_kit_parse::{DslBuild, serde_bridge::from_json_value, serde_bridge::serde_codes};
use dsl_kit_schema::DslSchema;
use serde_json::json;

/// The demo document in JSON form. Keys are written out of order on
/// purpose — the bridge canonicalises them.
fn document_json() -> serde_json::Value {
    json!({
        "type": "Env",
        "bindings": {
            "log": {
                "type": "Overrides",
                "entries": {
                    "20-prod": { "type": "Leaf", "value": "warn" },
                    "10-base": { "type": "Leaf", "value": "info" },
                },
            },
            "app": {
                "type": "Env",
                "bindings": {
                    "port": { "type": "Ref", "name": "PORT" },
                    "name": { "type": "Leaf", "value": "dsl-kit" },
                },
            },
        },
    })
}

fn build(value: &serde_json::Value) -> Cfg {
    let tree = from_json_value(value, &Cfg::schema()).expect("JSON parses");
    Cfg::from_parse_tree(&tree, &IdGen::new()).expect("JSON builds")
}

#[test]
fn json_document_builds_with_keys_intact_and_sorted() {
    let document = build(&document_json());

    let paths: Vec<String> = flatten(&document)
        .into_iter()
        .map(|(path, _)| path)
        .collect();
    assert_eq!(
        paths,
        vec![
            "",
            "app",
            "app.name",
            "app.port",
            "log",
            "log.10-base",
            "log.20-prod",
        ],
        "keys survive the bridge and arrive in ascending order regardless of source order"
    );
}

#[test]
fn built_document_resolves_to_the_winning_layer() {
    let document = build(&document_json());
    let value = resolve_all(&document, |name| match name {
        "PORT" => Some("8080".to_string()),
        _ => None,
    })
    .expect("resolution settles");
    // Root Env is a Seq over `app` then `log`; `log` folds last-wins
    // over `10-base` / `20-prod`.
    assert_eq!(value, "warn");
}

#[test]
fn an_empty_keyed_slot_is_a_valid_document() {
    let document = build(&json!({ "type": "Env", "bindings": {} }));
    let Cfg::Env { bindings, .. } = &document else {
        panic!("expected Env, got {document:?}");
    };
    assert!(bindings.is_empty());
}

#[test]
fn a_keyed_slot_supplied_as_a_list_is_rejected() {
    // The slot is keyed; handing it a positional array must not read
    // as an empty slot. The bridge catches the mismatch while it is
    // still looking at JSON — `CHILD_SHAPE` rather than conformance's
    // `KEYED_SLOT_SHAPE`, which is the code for a tree that got as far
    // as being built.
    let err = from_json_value(
        &json!({
            "type": "Env",
            "bindings": [{ "type": "Leaf", "value": "x" }],
        }),
        &Cfg::schema(),
    )
    .expect_err("a keyed slot must not accept a positional list");
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.code == serde_codes::CHILD_SHAPE),
        "expected CHILD_SHAPE; got {:?}",
        err.diagnostics
            .iter()
            .map(|d| d.code.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn unknown_variant_lists_candidates() {
    let err = from_json_value(&json!({ "type": "Enve" }), &Cfg::schema())
        .expect_err("typo must not build");
    assert!(
        err.diagnostics.iter().any(|d| d.message.contains("Env")),
        "expected the candidate `Env` in {:?}",
        err.diagnostics
            .iter()
            .map(|d| d.message.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn ids_are_fresh_and_unique_across_keyed_children() {
    let document = build(&document_json());
    let mut seen = std::collections::HashSet::new();
    for (path, node) in flatten(&document) {
        use dsl_kit::DslNode;
        assert!(
            seen.insert(node.node_id().0),
            "duplicate NodeId at {path:?}: {}",
            node.node_id()
        );
    }
    assert_eq!(seen.len(), 7, "seven nodes in the demo document");
}
