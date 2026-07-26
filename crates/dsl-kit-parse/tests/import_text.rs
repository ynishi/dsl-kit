//! End-to-end coverage for the `@import "name"` canonical-text
//! spelling (`import::add_import_syntax`) and mixed-front-end loading.
//!
//! What the tests pin:
//!
//! - **text splice semantics** — `@import "name"` works at every node
//!   position of a schema-generated grammar (root, positional slot,
//!   keyed slot value), transitively;
//! - **front-end mixing** — a text root imports JSON sources and a
//!   JSON root imports text sources, landing in one linked tree;
//! - **opt-in invisibility** — an import-enabled grammar still passes
//!   every `grammar_check` pass, and `example_gen` never spells
//!   `@import` in synthesized examples;
//! - **fail-loud text path** — text sources without a configured
//!   grammar are `text_unsupported`, not a fallback.

use dsl_kit_core::{IdGen, NodeId};
use dsl_kit_macros::{DslBuild, DslNode, DslSchema};
use dsl_kit_parse::{
    DslBuild, check_conformance,
    example_gen::examples_from_grammar,
    grammar_check,
    import::{
        IMPORT_VARIANT, ImportLimits, Loader, MapResolver, SourceId, add_import_syntax,
        import_codes, load_json_str,
    },
    peg::{self, Grammar},
    schema_gen::checked_grammar_from_schema,
};
use dsl_kit_schema::DslSchema;
use std::collections::BTreeMap;

/// Same fixture shape as `import_json.rs`, so text and JSON coverage
/// stay comparable.
#[derive(Debug, DslNode, DslSchema, DslBuild)]
enum Cfg {
    /// Leaf holding a payload string.
    Leaf {
        /// Stable node id.
        id: NodeId,
        /// Payload.
        value: String,
    },
    /// Positional list.
    #[allow(clippy::vec_box)]
    Seq {
        /// Stable node id.
        id: NodeId,
        /// Positional children.
        items: Vec<Box<Cfg>>,
    },
    /// Keyed slot.
    Env {
        /// Stable node id.
        id: NodeId,
        /// Keyed children.
        entries: BTreeMap<String, Box<Cfg>>,
    },
    /// Positional single child.
    Wrap {
        /// Stable node id.
        id: NodeId,
        /// Positional child.
        inner: Box<Cfg>,
    },
}

/// Schema-generated grammar with the import spelling enabled.
fn import_grammar(ids: &IdGen) -> Grammar {
    let mut g = checked_grammar_from_schema(&Cfg::schema(), ids).expect("grammar");
    add_import_syntax(&mut g, ids).expect("inject");
    g
}

fn build(tree: &dsl_kit_parse::ParseTree) -> Cfg {
    assert_eq!(check_conformance(tree, &Cfg::schema()), vec![]);
    Cfg::from_parse_tree(tree, &IdGen::new()).expect("typed build")
}

fn leaf_value(cfg: &Cfg) -> &str {
    match cfg {
        Cfg::Leaf { value, .. } => value,
        other => panic!("expected Leaf, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Text splice semantics
// ---------------------------------------------------------------------------

#[test]
fn text_root_imports_text_source_at_positional_slot() {
    let ids = IdGen::new();
    let grammar = import_grammar(&ids);
    let mut resolver = MapResolver::new();
    resolver.insert_text("lib", r#"Leaf(value: "shared")"#);

    let loaded = Loader::new(&Cfg::schema())
        .with_grammar(&grammar)
        .load_text(
            r#"Seq(items: [@import "lib", Leaf(value: "inline")])"#,
            &mut resolver,
        )
        .expect("load");
    assert_eq!(loaded.dependencies, vec![SourceId::new("lib")]);

    match build(&loaded.tree) {
        Cfg::Seq { items, .. } => {
            assert_eq!(items.len(), 2);
            assert_eq!(leaf_value(&items[0]), "shared");
            assert_eq!(leaf_value(&items[1]), "inline");
        }
        other => panic!("expected Seq, got {other:?}"),
    }
}

#[test]
fn text_import_at_keyed_slot_value_and_root() {
    let ids = IdGen::new();
    let grammar = import_grammar(&ids);
    let mut resolver = MapResolver::new();
    resolver.insert_text("root", r#"Env(entries: { "database": @import "db" })"#);
    resolver.insert_text("db", r#"Leaf(value: "postgres")"#);

    let loaded = Loader::new(&Cfg::schema())
        .with_grammar(&grammar)
        .load_text(r#"@import "root""#, &mut resolver)
        .expect("load");
    assert_eq!(
        loaded.dependencies,
        vec![SourceId::new("db"), SourceId::new("root")]
    );

    match build(&loaded.tree) {
        Cfg::Env { entries, .. } => assert_eq!(leaf_value(&entries["database"]), "postgres"),
        other => panic!("expected Env, got {other:?}"),
    }
}

#[test]
fn front_ends_mix_in_both_directions() {
    let ids = IdGen::new();
    let grammar = import_grammar(&ids);
    let schema = Cfg::schema();
    let loader = Loader::new(&schema).with_grammar(&grammar);

    // Text root importing a JSON source.
    let mut resolver = MapResolver::new();
    resolver.insert("j", r#"{ "type": "Leaf", "value": "from json" }"#);
    let loaded = loader
        .load_text(r#"Wrap(inner: @import "j")"#, &mut resolver)
        .expect("text root");
    match build(&loaded.tree) {
        Cfg::Wrap { inner, .. } => assert_eq!(leaf_value(&inner), "from json"),
        other => panic!("expected Wrap, got {other:?}"),
    }

    // JSON root importing a text source (which imports JSON again).
    let mut resolver = MapResolver::new();
    resolver.insert_text("t", r#"Wrap(inner: @import "j2")"#);
    resolver.insert("j2", r#"{ "type": "Leaf", "value": "deep json" }"#);
    let loaded = loader
        .load_json_str(r#"{ "$import": "t" }"#, &mut resolver)
        .expect("json root");
    match build(&loaded.tree) {
        Cfg::Wrap { inner, .. } => assert_eq!(leaf_value(&inner), "deep json"),
        other => panic!("expected Wrap, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Opt-in invisibility
// ---------------------------------------------------------------------------

#[test]
fn import_enabled_grammar_passes_all_static_checks() {
    let ids = IdGen::new();
    let grammar = import_grammar(&ids);
    assert_eq!(
        grammar_check::check_against(&grammar, &Cfg::schema()),
        vec![]
    );
}

#[test]
fn example_gen_never_spells_import() {
    let ids = IdGen::new();
    let grammar = import_grammar(&ids);
    let examples = examples_from_grammar(&grammar).expect("examples");

    assert!(
        examples.per_rule.iter().all(|r| r.rule != IMPORT_VARIANT),
        "reserved rule leaked into per-rule examples"
    );
    for r in &examples.per_rule {
        assert!(
            !r.text.contains("@import"),
            "example spells @import: {}",
            r.text
        );
    }
    assert!(
        !examples.composite.contains("@import"),
        "composite spells @import: {}",
        examples.composite
    );
}

#[test]
fn add_import_syntax_is_idempotent() {
    let ids = IdGen::new();
    let mut grammar = checked_grammar_from_schema(&Cfg::schema(), &ids).expect("grammar");
    add_import_syntax(&mut grammar, &ids).expect("first");
    let rules_after_first = grammar.rules.len();
    add_import_syntax(&mut grammar, &ids).expect("second");
    assert_eq!(grammar.rules.len(), rules_after_first);
}

#[test]
fn non_choice_start_rule_is_wrapped() {
    let ids = IdGen::new();
    // Hand grammar whose start body is a bare Node, not a Choice.
    let mut grammar = Grammar::new(
        vec![peg::rule(
            &ids,
            "s",
            peg::node(
                &ids,
                "Leaf",
                peg::seq(
                    &ids,
                    vec![
                        peg::token(&ids, "%kw:Leaf"),
                        peg::token(&ids, "("),
                        peg::token(&ids, "%kw:value"),
                        peg::token(&ids, ":"),
                        peg::field(&ids, "value", peg::token(&ids, "%str")),
                        peg::token(&ids, ")"),
                    ],
                ),
            ),
        )],
        "s",
    );
    add_import_syntax(&mut grammar, &ids).expect("inject");

    let mut resolver = MapResolver::new();
    resolver.insert_text("lib", r#"Leaf(value: "wrapped")"#);
    let loaded = Loader::new(&Cfg::schema())
        .with_grammar(&grammar)
        .load_text(r#"@import "lib""#, &mut resolver)
        .expect("load");
    assert_eq!(leaf_value(&build(&loaded.tree)), "wrapped");
}

#[test]
fn unknown_start_rule_fails_injection() {
    let ids = IdGen::new();
    let mut grammar = Grammar::new(vec![], "nope");
    let err = add_import_syntax(&mut grammar, &ids).expect_err("inject");
    assert_eq!(err.diagnostics[0].code, peg::codes::UNKNOWN_RULE);
}

// ---------------------------------------------------------------------------
// Fail-loud text path
// ---------------------------------------------------------------------------

#[test]
fn text_source_without_grammar_is_text_unsupported() {
    let mut resolver = MapResolver::new();
    resolver.insert_text("t", r#"Leaf(value: "x")"#);

    let err = load_json_str(
        r#"{ "$import": "t" }"#,
        &Cfg::schema(),
        &mut resolver,
        &ImportLimits::default(),
    )
    .expect_err("text without grammar");
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.code == import_codes::TEXT_UNSUPPORTED),
        "{err}"
    );
}

#[test]
fn text_root_without_grammar_is_text_unsupported() {
    let mut resolver = MapResolver::new();
    let err = Loader::new(&Cfg::schema())
        .load_text(r#"Leaf(value: "x")"#, &mut resolver)
        .expect_err("no grammar");
    assert_eq!(err.diagnostics[0].code, import_codes::TEXT_UNSUPPORTED);
}

#[test]
fn parse_error_in_text_source_carries_chain_context() {
    let ids = IdGen::new();
    let grammar = import_grammar(&ids);
    let mut resolver = MapResolver::new();
    resolver.insert_text("bad", r#"Leaf(value: )"#);

    let err = Loader::new(&Cfg::schema())
        .with_grammar(&grammar)
        .load_text(r#"Wrap(inner: @import "bad")"#, &mut resolver)
        .expect_err("nested parse error");
    assert_eq!(err.diagnostics[0].code, import_codes::IN_IMPORT);
    assert!(
        err.diagnostics[0].message.contains("<root> → bad"),
        "chain missing from: {}",
        err.diagnostics[0].message
    );
    assert!(err.diagnostics.len() > 1, "nested diagnostic missing");
}
