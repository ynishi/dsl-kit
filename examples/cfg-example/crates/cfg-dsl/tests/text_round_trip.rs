//! Canonical-text front-end coverage: schema-generated grammar →
//! parsed source → conformance → typed `Cfg`.
//!
//! Nobody writes a grammar for `Cfg`; `checked_grammar_from_schema`
//! derives one from `Cfg::schema()`, keyed slots included. The last
//! test pins the property that makes "one Rust type, two front-ends"
//! mean anything: the text and JSON doors build the same AST.

use cfg_dsl::{Cfg, flatten, resolve_all};
use dsl_kit::IdGen;
use dsl_kit_parse::{
    DslBuild, check_conformance, codes, example_gen::examples_from_grammar,
    schema_gen::checked_grammar_from_schema, serde_bridge::from_json_value,
};
use dsl_kit_schema::DslSchema;
use serde_json::json;

/// Parses `input` with the schema-derived grammar, asserts conformance
/// on the way through, and builds the typed AST.
fn build(input: &str) -> Cfg {
    let schema = Cfg::schema();
    let grammar = checked_grammar_from_schema(&schema, &IdGen::new())
        .expect("the keyed-slot schema generates a clean grammar");
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

/// Flattens a document into `dotted.path -> summary` lines, which is
/// enough to compare two builds without depending on `NodeId`s (those
/// are minted per build and are meant to differ).
fn shape(cfg: &Cfg) -> Vec<String> {
    flatten(cfg)
        .into_iter()
        .map(|(path, node)| format!("{path} -> {}", node.summary()))
        .collect()
}

const DEMO_TEXT: &str = r#"
    Env(bindings: {
        log: Overrides(entries: {
            "20-prod": Leaf(value: "warn"),
            "10-base": Leaf(value: "info")
        }),
        app: Env(bindings: {
            port: Ref(name: "PORT"),
            name: Leaf(value: "dsl-kit")
        })
    })
"#;

#[test]
fn text_document_builds_with_sorted_keys() {
    let document = build(DEMO_TEXT);
    assert_eq!(
        shape(&document),
        vec![
            " -> Env (2 bindings)".to_string(),
            "app -> Env (2 bindings)".to_string(),
            "app.name -> Leaf \"dsl-kit\"".to_string(),
            "app.port -> Ref \"PORT\"".to_string(),
            "log -> Overrides (2 layers)".to_string(),
            "log.10-base -> Leaf \"info\"".to_string(),
            "log.20-prod -> Leaf \"warn\"".to_string(),
        ],
    );
}

#[test]
fn text_document_resolves_through_the_engine() {
    let document = build(DEMO_TEXT);
    let value = resolve_all(&document, |name| Some(format!("<{name}>")))
        .expect("resolution settles once the reference is answered");
    assert_eq!(value, "warn");
}

#[test]
fn both_keyed_variants_accept_the_same_syntax() {
    // `Env` boxes its values and `Overrides` does not, but the source
    // form is identical — which is the point of carrying both.
    let boxed = build(r#"Env(bindings: { a: Leaf(value: "1") })"#);
    let bare = build(r#"Overrides(entries: { a: Leaf(value: "1") })"#);
    assert_eq!(shape(&boxed)[1..], shape(&bare)[1..]);
}

#[test]
fn an_empty_keyed_slot_parses() {
    let document = build("Overrides(entries: {})");
    let Cfg::Overrides { entries, .. } = &document else {
        panic!("expected Overrides, got {document:?}");
    };
    assert!(entries.is_empty());
    // An empty layer stack folds to the unit value.
    assert_eq!(resolve_all(&document, |_| None).unwrap(), "");
}

#[test]
fn a_quoted_key_may_hold_characters_an_identifier_cannot() {
    let document = build(r#"Env(bindings: { "log level": Leaf(value: "debug") })"#);
    assert_eq!(shape(&document)[1], "log level -> Leaf \"debug\"");
}

#[test]
fn a_repeated_key_fails_the_build_not_the_parse() {
    // The grammar has no opinion about duplicate keys; DUPLICATE_KEY
    // is what stops the build, so no subtree is silently dropped.
    let schema = Cfg::schema();
    let grammar = checked_grammar_from_schema(&schema, &IdGen::new()).expect("generates");
    let tree = grammar
        .parse(r#"Env(bindings: { k: Leaf(value: "first"), k: Leaf(value: "second") })"#)
        .expect("the grammar accepts a repeated key");

    let err = Cfg::from_parse_tree(&tree, &IdGen::new())
        .expect_err("the build must reject the duplicate");
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

#[test]
fn the_text_and_json_front_ends_agree() {
    let from_text = build(DEMO_TEXT);

    let value = json!({
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
    });
    let tree = from_json_value(&value, &Cfg::schema()).expect("JSON parses");
    let from_json = Cfg::from_parse_tree(&tree, &IdGen::new()).expect("JSON builds");

    assert_eq!(shape(&from_text), shape(&from_json));
}

#[test]
fn synthesized_examples_survive_their_own_toolchain() {
    // Examples are what an AI consumer reads to learn the syntax, so a
    // keyed example the kit then rejects would teach a wrong spelling.
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
