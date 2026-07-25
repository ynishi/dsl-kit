//! End-to-end coverage for keyed child slots through the **text**
//! front-end: `#[derive(DslSchema)]` → generated grammar → parsed
//! source → `#[derive(DslBuild)]` → typed AST.
//!
//! Sibling of `keyed_slot_json.rs`, which drives the same AST through
//! the JSON front-end. Running both against one enum is the point: a
//! DSL author writes the Rust type once and gets two front-ends, so
//! the two had better agree. The tests below pin the agreement itself
//! (same document, same `BTreeMap`) rather than only the text path in
//! isolation.

use dsl_kit_core::{IdGen, NodeId};
use dsl_kit_macros::{DslBuild, DslNode, DslSchema};
use dsl_kit_parse::{
    DslBuild, check_conformance, codes, example_gen::examples_from_grammar,
    schema_gen::checked_grammar_from_schema, serde_bridge::from_json_value,
};
use dsl_kit_schema::DslSchema;
use serde_json::json;
use std::collections::BTreeMap;

/// Keyed-slot AST, same shape as the JSON sibling's fixture.
#[derive(Debug, DslNode, DslSchema, DslBuild)]
enum Cfg {
    /// Leaf holding a payload string.
    Leaf {
        /// Stable node id.
        id: NodeId,
        /// Payload.
        value: String,
    },
    /// Keyed slot with boxed self-recursion.
    Env {
        /// Stable node id.
        id: NodeId,
        /// Keyed children.
        entries: BTreeMap<String, Box<Cfg>>,
    },
}

/// Parses `input` with the schema-derived grammar and builds the AST,
/// asserting conformance on the way through.
fn build(input: &str) -> Cfg {
    let schema = Cfg::schema();
    let grammar = checked_grammar_from_schema(&schema, &IdGen::new())
        .expect("keyed-slot schema generates a clean grammar");
    let tree = grammar
        .parse(input)
        .unwrap_or_else(|e| panic!("parse failed for {input:?}: {:?}", e.diagnostics));
    let diags = check_conformance(&tree, &schema);
    assert!(
        diags.is_empty(),
        "conformance clean for {input:?}: {diags:?}"
    );
    Cfg::from_parse_tree(&tree, &IdGen::new())
        .unwrap_or_else(|e| panic!("build failed for {input:?}: {:?}", e.diagnostics))
}

/// Reads the payload of a `Leaf`, panicking on any other variant.
fn leaf_value(node: &Cfg) -> &str {
    match node {
        Cfg::Leaf { value, .. } => value.as_str(),
        other => panic!("expected a Leaf, got {other:?}"),
    }
}

/// Flattens an `Env` into `key=value` pairs for comparison.
fn pairs(node: &Cfg) -> Vec<String> {
    let Cfg::Env { entries, .. } = node else {
        panic!("expected Env, got {node:?}");
    };
    entries
        .iter()
        .map(|(k, v)| format!("{k}={}", leaf_value(v)))
        .collect()
}

/// A keyed slot written in the canonical text syntax reaches the typed
/// AST with its keys intact, sorted, regardless of source order. Both
/// key spellings — bare identifier and quoted string — are accepted,
/// and a quoted key may hold characters an identifier cannot.
#[test]
fn text_keyed_slot_builds_with_sorted_keys() {
    let built = build(r#"Env(entries: { zeta: Leaf(value: "z"), "a b": Leaf(value: "space") })"#);
    assert_eq!(pairs(&built), vec!["a b=space", "zeta=z"]);
}

/// An empty map is valid syntax and builds to an empty `BTreeMap`.
#[test]
fn text_empty_keyed_slot_builds_empty_map() {
    let built = build("Env(entries: {})");
    let Cfg::Env { entries, .. } = &built else {
        panic!("expected Env");
    };
    assert!(entries.is_empty());
}

/// Keyed slots nest, and each level sorts independently.
#[test]
fn text_keyed_slots_nest() {
    let built = build(
        r#"Env(entries: { outer: Env(entries: { z: Leaf(value: "deep-z"), y: Leaf(value: "deep-y") }) })"#,
    );
    let Cfg::Env { entries, .. } = &built else {
        panic!("expected Env");
    };
    assert_eq!(
        pairs(entries["outer"].as_ref()),
        vec!["y=deep-y", "z=deep-z"]
    );
}

/// The two front-ends agree: the same document written as text and as
/// JSON builds to the same typed AST. This is the property that makes
/// "write the AST in Rust, get the front-ends free" mean anything —
/// if the keyed halves disagreed, a DSL would parse differently
/// depending on which door it came through.
#[test]
fn text_and_json_front_ends_agree() {
    let from_text = build(r#"Env(entries: { b: Leaf(value: "two"), a: Leaf(value: "one") })"#);

    let value = json!({
        "type": "Env",
        "entries": {
            "b": { "type": "Leaf", "value": "two" },
            "a": { "type": "Leaf", "value": "one" },
        }
    });
    let tree = from_json_value(&value, &Cfg::schema()).expect("JSON parses");
    let from_json = Cfg::from_parse_tree(&tree, &IdGen::new()).expect("JSON builds");

    assert_eq!(pairs(&from_text), pairs(&from_json));
}

/// An empty key is a legal `BTreeMap` key and a legal JSON object
/// key, so the text syntax has to be able to spell it — otherwise a
/// slice of the AST type has no source form. Quoting is what makes it
/// writable.
#[test]
fn text_accepts_an_empty_string_key() {
    let built = build(r#"Env(entries: { "": Leaf(value: "empty") })"#);
    assert_eq!(pairs(&built), vec!["=empty"]);
}

/// The two front-ends spell an *empty* map differently in the tree —
/// JSON records the slot it saw, the parser has no entry to record —
/// and that difference is deliberate rather than a bug to chase: both
/// build to the same empty map, which is the level consumers work at.
/// Pinned so a later refactor cannot silently flip either side.
#[test]
fn empty_map_differs_in_the_tree_but_not_in_the_build() {
    let schema = Cfg::schema();

    let grammar = checked_grammar_from_schema(&schema, &IdGen::new()).expect("generates");
    let from_text = grammar.parse("Env(entries: {})").expect("text parses");
    assert!(
        from_text.keyed_child_slot("entries").is_none(),
        "the parser has no entry to record, so no slot is emitted"
    );

    let from_json =
        from_json_value(&json!({ "type": "Env", "entries": {} }), &schema).expect("JSON parses");
    assert_eq!(
        from_json.keyed_child_slot("entries"),
        Some(&[][..]),
        "the JSON bridge records the key it saw, with no entries under it"
    );

    // Both conform, and both build to the same empty map.
    assert!(check_conformance(&from_text, &schema).is_empty());
    assert!(check_conformance(&from_json, &schema).is_empty());
    for tree in [&from_text, &from_json] {
        let built = Cfg::from_parse_tree(tree, &IdGen::new()).expect("builds");
        let Cfg::Env { entries, .. } = &built else {
            panic!("expected Env");
        };
        assert!(entries.is_empty());
    }
}

/// A repeated key is the parser's to accept and the schema's to
/// reject. Splitting the stages here (rather than going through
/// `build`) pins that division: the grammar has no opinion about
/// duplicate keys, and `DUPLICATE_KEY` is what stops the build.
#[test]
fn text_duplicate_key_fails_the_build_not_the_parse() {
    let schema = Cfg::schema();
    let grammar = checked_grammar_from_schema(&schema, &IdGen::new()).expect("generates");
    let tree = grammar
        .parse(r#"Env(entries: { k: Leaf(value: "first"), k: Leaf(value: "second") })"#)
        .expect("grammar accepts a repeated key");

    let err = Cfg::from_parse_tree(&tree, &IdGen::new())
        .expect_err("build must reject the duplicate rather than drop an entry");
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.code == codes::DUPLICATE_KEY),
        "expected DUPLICATE_KEY; got {:?}",
        err.diagnostics
            .iter()
            .map(|d| d.code.as_str())
            .collect::<Vec<_>>()
    );
}

/// Every synthesized example for a keyed-slot schema parses, conforms
/// and builds. Examples are what an AI consumer reads to learn the
/// syntax, so a keyed example that does not survive its own toolchain
/// would teach a spelling the kit rejects.
#[test]
fn synthesized_examples_round_trip_through_the_build() {
    let schema = Cfg::schema();
    let grammar = checked_grammar_from_schema(&schema, &IdGen::new()).expect("generates");
    let examples = examples_from_grammar(&grammar).expect("synthesizes");

    for example in examples.per_rule.iter() {
        let tree = grammar.parse(&example.text).unwrap_or_else(|e| {
            panic!(
                "example for `{}` failed to parse: {:?}\n  text: {}",
                example.rule, e.diagnostics, example.text
            )
        });
        Cfg::from_parse_tree(&tree, &IdGen::new()).unwrap_or_else(|e| {
            panic!(
                "example for `{}` failed to build: {:?}\n  text: {}",
                example.rule, e.diagnostics, example.text
            )
        });
    }
}
