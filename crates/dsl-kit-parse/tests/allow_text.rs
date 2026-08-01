//! End-to-end coverage for the `@allow("rule") <node>` canonical-text
//! spelling (`allow::add_allow_syntax`) and the fold that collapses it.
//!
//! What the tests pin:
//!
//! - **front-end equivalence** — the same suppression written as JSON
//!   `$allow` and as text `@allow` produces the same tree and the same
//!   canonical dump, which is the whole point of having two spellings;
//! - **wrapper semantics** — several rules per annotation, stacked
//!   annotations, annotations at a nested node position, and a
//!   document with none of them;
//! - **opt-in invisibility** — a grammar that never went through
//!   `add_allow_syntax` rejects `@allow` outright, injection is
//!   idempotent, and an allow-enabled grammar still passes every
//!   `grammar_check` pass without leaking into `example_gen`;
//! - **fail-loud fold** — a malformed wrapper is a diagnostic, and a
//!   wrapper that reaches conformance is another.

use dsl_kit_core::{IdGen, NodeId};
use dsl_kit_macros::{DslBuild, DslNode, DslSchema};
use dsl_kit_parse::{
    ParseTree, RawValue, allow, check_conformance,
    example_gen::examples_from_grammar,
    grammar_check,
    peg::{self, Grammar},
    schema_gen::checked_grammar_from_schema,
    serde_bridge::{from_json_value, to_canonical_json},
};
use dsl_kit_schema::DslSchema;
use serde_json::json;

/// Same fan-out fixture as `allow_annotation.rs`, so the JSON-side and
/// text-side coverage compare like for like.
#[derive(Debug, PartialEq, DslNode, DslSchema, DslBuild)]
enum Flow {
    /// Parallel fan-out over its branches.
    Par {
        /// Stable node id.
        id: NodeId,
        /// The branches run in parallel.
        branches: Vec<Flow>,
    },
    /// Leaf unit of work.
    Task {
        /// Stable node id.
        id: NodeId,
        /// Task name.
        name: String,
    },
}

/// Schema-generated grammar with the allow spelling enabled.
fn allow_grammar(ids: &IdGen) -> Grammar {
    let mut g = checked_grammar_from_schema(&Flow::schema(), ids).expect("grammar");
    allow::add_allow_syntax(&mut g, ids).expect("inject");
    g
}

fn parse_text(input: &str) -> ParseTree {
    let ids = IdGen::new();
    let tree = allow_grammar(&ids)
        .parse(input)
        .unwrap_or_else(|e| panic!("parse failed for {input}: {:?}", e.diagnostics));
    let diags = check_conformance(&tree, &Flow::schema());
    assert!(diags.is_empty(), "conformance clean for {input}: {diags:?}");
    tree
}

fn parse_json(doc: serde_json::Value) -> ParseTree {
    let tree = from_json_value(&doc, &Flow::schema())
        .unwrap_or_else(|e| panic!("parse failed for {doc}: {:?}", e.diagnostics));
    let diags = check_conformance(&tree, &Flow::schema());
    assert!(diags.is_empty(), "conformance clean for {doc}: {diags:?}");
    tree
}

/// Variant + suppressions + slot shape, recursively.
///
/// The two front-ends are not expected to produce equal `ParseTree`
/// values: payloads arrive as `RawValue::Text` from the PEG side and
/// `RawValue::Json` from the serde side, and only the text side
/// carries spans. This projects away both, leaving the structure the
/// two must agree on; `to_canonical_json` covers the payloads.
fn shape(tree: &ParseTree) -> String {
    let mut out = format!("{}[{}]", tree.variant, tree.allows.join(","));
    for (slot, kids) in &tree.children {
        let inner: Vec<String> = kids.iter().map(shape).collect();
        out.push_str(&format!("({slot}: {})", inner.join(" ")));
    }
    for (slot, entries) in &tree.keyed_children {
        let inner: Vec<String> = entries
            .iter()
            .map(|(k, v)| format!("{k}={}", shape(v)))
            .collect();
        out.push_str(&format!("{{{slot}: {}}}", inner.join(" ")));
    }
    out
}

// ---------------------------------------------------------------------------
// Front-end equivalence
// ---------------------------------------------------------------------------

/// The acceptance criterion: one suppression, two spellings, one tree.
#[test]
fn json_and_text_spellings_produce_the_same_document() {
    let schema = Flow::schema();
    let from_json = parse_json(json!({
        "type": "Par",
        "$allow": ["max-fan-out"],
        "branches": [
            { "type": "Task", "name": "a" },
            { "type": "Task", "name": "b" },
        ],
    }));
    let from_text =
        parse_text(r#"@allow("max-fan-out") Par(branches: [Task(name: "a"), Task(name: "b")])"#);

    assert_eq!(from_text.allows, vec!["max-fan-out".to_string()]);
    assert_eq!(shape(&from_json), shape(&from_text));
    assert_eq!(
        to_canonical_json(&from_json, &schema).unwrap(),
        to_canonical_json(&from_text, &schema).unwrap(),
    );

    // And the canonical dump of the text parse is the JSON document
    // the author would have written by hand.
    assert_eq!(
        to_canonical_json(&from_text, &schema).unwrap(),
        json!({
            "type": "Par",
            "$allow": ["max-fan-out"],
            "branches": [
                { "type": "Task", "name": "a" },
                { "type": "Task", "name": "b" },
            ],
        })
    );
}

// ---------------------------------------------------------------------------
// Wrapper semantics
// ---------------------------------------------------------------------------

#[test]
fn one_annotation_can_name_several_rules() {
    let tree = parse_text(r#"@allow("max-fan-out", "no-redundant-wrap") Task(name: "t")"#);
    assert_eq!(tree.allows, vec!["max-fan-out", "no-redundant-wrap"]);

    // Order and duplicates are the author's, exactly as on the JSON
    // side — the names are echoed back in the canonical dump.
    let many = parse_text(r#"@allow("b-rule", "a-rule", "b-rule") Task(name: "t")"#);
    assert_eq!(many.allows, vec!["b-rule", "a-rule", "b-rule"]);
}

#[test]
fn stacked_annotations_fold_onto_one_node_outermost_first() {
    let tree = parse_text(r#"@allow("a") @allow("b", "c") Task(name: "t")"#);
    assert_eq!(tree.variant, "Task");
    assert_eq!(tree.allows, vec!["a", "b", "c"]);
    assert_eq!(
        tree.field("name"),
        Some(&RawValue::Text("t".to_string())),
        "the target keeps its own payload",
    );
}

#[test]
fn an_annotation_is_writable_at_a_nested_node_position() {
    let tree = parse_text(r#"Par(branches: [@allow("naming") Task(name: "a"), Task(name: "b")])"#);
    assert!(tree.allows.is_empty(), "the root annotated nothing");
    let branches = tree.child_slot("branches").expect("slot bound");
    assert_eq!(branches[0].allows, vec!["naming".to_string()]);
    assert!(branches[1].allows.is_empty());
}

#[test]
fn the_folded_node_keeps_its_own_span() {
    let input = r#"@allow("naming") Task(name: "t")"#;
    let annotated = parse_text(input);

    // The wrapper's own span would have started at byte 0, covering
    // the annotation; the surviving node points at what it annotates.
    // (Node spans open before the leading-whitespace skip, hence the
    // trim.)
    let span = annotated.span.expect("text front-end tracks spans");
    assert_eq!(
        input[span.start..span.end].trim(),
        r#"Task(name: "t")"#,
        "span {span:?} should cover the annotated node alone",
    );
    assert_eq!(span.end, input.len());
}

#[test]
fn a_document_without_an_annotation_is_unchanged() {
    let tree = parse_text(r#"Par(branches: [Task(name: "a")])"#);
    assert!(tree.allows.is_empty());
    assert!(tree.child_slot("branches").unwrap()[0].allows.is_empty());

    let canonical = to_canonical_json(&tree, &Flow::schema()).unwrap();
    assert!(canonical.get(allow::ALLOW_KEY).is_none());
    assert_eq!(
        canonical,
        json!({ "type": "Par", "branches": [{ "type": "Task", "name": "a" }] })
    );
}

// ---------------------------------------------------------------------------
// Opt-in invisibility
// ---------------------------------------------------------------------------

#[test]
fn a_grammar_without_the_injection_rejects_the_spelling() {
    let ids = IdGen::new();
    let plain = checked_grammar_from_schema(&Flow::schema(), &ids).expect("grammar");
    let err = plain
        .parse(r#"@allow("max-fan-out") Task(name: "t")"#)
        .expect_err("`@allow` is not part of an un-injected grammar");
    assert_eq!(err.diagnostics[0].code, peg::codes::UNEXPECTED);
}

#[test]
fn add_allow_syntax_is_idempotent() {
    let ids = IdGen::new();
    let mut grammar = checked_grammar_from_schema(&Flow::schema(), &ids).expect("grammar");
    allow::add_allow_syntax(&mut grammar, &ids).expect("first");
    let rules_after_first = grammar.rules.len();
    allow::add_allow_syntax(&mut grammar, &ids).expect("second");
    assert_eq!(grammar.rules.len(), rules_after_first);
}

#[test]
fn unknown_start_rule_fails_injection() {
    let ids = IdGen::new();
    let mut grammar = Grammar::new(vec![], "nope");
    let err = allow::add_allow_syntax(&mut grammar, &ids).expect_err("inject");
    assert_eq!(err.diagnostics[0].code, peg::codes::UNKNOWN_RULE);
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
                "Task",
                peg::seq(
                    &ids,
                    vec![
                        peg::token(&ids, "%kw:Task"),
                        peg::token(&ids, "("),
                        peg::token(&ids, "%kw:name"),
                        peg::token(&ids, ":"),
                        peg::field(&ids, "name", peg::token(&ids, "%str")),
                        peg::token(&ids, ")"),
                    ],
                ),
            ),
        )],
        "s",
    );
    allow::add_allow_syntax(&mut grammar, &ids).expect("inject");

    let tree = grammar
        .parse(r#"@allow("naming") Task(name: "t")"#)
        .expect("parse");
    assert_eq!(tree.variant, "Task");
    assert_eq!(tree.allows, vec!["naming".to_string()]);
}

#[test]
fn allow_enabled_grammar_passes_all_static_checks() {
    let ids = IdGen::new();
    let grammar = allow_grammar(&ids);
    assert_eq!(
        grammar_check::check_against(&grammar, &Flow::schema()),
        vec![]
    );
}

#[test]
fn example_gen_never_spells_allow() {
    let ids = IdGen::new();
    let grammar = allow_grammar(&ids);
    let examples = examples_from_grammar(&grammar).expect("examples");

    assert!(
        examples
            .per_rule
            .iter()
            .all(|r| r.rule != allow::ALLOW_VARIANT),
        "reserved rule leaked into per-rule examples"
    );
    for r in &examples.per_rule {
        assert!(
            !r.text.contains("@allow"),
            "example spells @allow: {}",
            r.text
        );
    }
    assert!(
        !examples.composite.contains("@allow"),
        "composite spells @allow: {}",
        examples.composite
    );
}

/// Both reserved spellings can live in one grammar without either
/// disturbing the other.
#[test]
fn allow_and_import_syntax_coexist() {
    use dsl_kit_parse::import::{Loader, MapResolver, add_import_syntax};

    let ids = IdGen::new();
    let mut grammar = checked_grammar_from_schema(&Flow::schema(), &ids).expect("grammar");
    add_import_syntax(&mut grammar, &ids).expect("import");
    allow::add_allow_syntax(&mut grammar, &ids).expect("allow");
    assert_eq!(
        grammar_check::check_against(&grammar, &Flow::schema()),
        vec![]
    );

    let mut resolver = MapResolver::new();
    resolver.insert_text("lib", r#"@allow("naming") Task(name: "shared")"#);
    let loaded = Loader::new(&Flow::schema())
        .with_grammar(&grammar)
        .load_text(r#"Par(branches: [@import "lib"])"#, &mut resolver)
        .expect("load");

    let branches = loaded.tree.child_slot("branches").expect("slot bound");
    assert_eq!(
        branches[0].allows,
        vec!["naming".to_string()],
        "the imported source's own annotation survives the splice",
    );
}

// ---------------------------------------------------------------------------
// Fail-loud fold
// ---------------------------------------------------------------------------

/// A wrapper that never went through the fold is caught at
/// conformance, the way an unexpanded `$import` is.
#[test]
fn an_uncollapsed_wrapper_fails_conformance() {
    let mut wrapper = ParseTree::new(allow::ALLOW_VARIANT);
    wrapper.fields.push((
        allow::ALLOW_RULES_FIELD.to_string(),
        RawValue::Text("max-fan-out".into()),
    ));
    let mut task = ParseTree::new("Task");
    task.fields
        .push(("name".to_string(), RawValue::Text("t".into())));
    wrapper
        .children
        .push((allow::ALLOW_TARGET_SLOT.to_string(), vec![task]));

    let diags = check_conformance(&wrapper, &Flow::schema());
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].code, allow::codes::UNCOLLAPSED);
    assert!(
        diags[0].message.contains(allow::ALLOW_VARIANT),
        "message names the wrapper: {}",
        diags[0].message
    );

    // Folding it first is what conformance is asking for.
    let folded = allow::collapse(wrapper).expect("fold");
    assert_eq!(folded.variant, "Task");
    assert_eq!(folded.allows, vec!["max-fan-out".to_string()]);
    assert_eq!(check_conformance(&folded, &Flow::schema()), vec![]);
}

/// Shapes the injected grammar cannot produce but a hand-built tree
/// can. Each one loses the annotated subtree if the fold guesses, so
/// each one is a diagnostic instead.
#[test]
fn a_malformed_wrapper_fails_the_fold() {
    let task = || {
        let mut t = ParseTree::new("Task");
        t.fields
            .push(("name".to_string(), RawValue::Text("t".into())));
        t
    };
    let named = |names: Vec<&str>| {
        let mut w = ParseTree::new(allow::ALLOW_VARIANT);
        for n in names {
            w.fields.push((
                allow::ALLOW_RULES_FIELD.to_string(),
                RawValue::Text(n.into()),
            ));
        }
        w
    };

    // No target at all.
    let bare = named(vec!["max-fan-out"]);

    // Two nodes where one belongs.
    let mut crowded = named(vec!["max-fan-out"]);
    crowded
        .children
        .push((allow::ALLOW_TARGET_SLOT.to_string(), vec![task(), task()]));

    // A payload that is not a rule name.
    let mut numeric = ParseTree::new(allow::ALLOW_VARIANT);
    numeric.fields.push((
        allow::ALLOW_RULES_FIELD.to_string(),
        RawValue::Json(json!(3)),
    ));
    numeric
        .children
        .push((allow::ALLOW_TARGET_SLOT.to_string(), vec![task()]));

    // A slot the wrapper has no meaning for.
    let mut stray = named(vec!["max-fan-out"]);
    stray
        .children
        .push((allow::ALLOW_TARGET_SLOT.to_string(), vec![task()]));
    stray.children.push(("branches".to_string(), vec![task()]));

    for bad in [bare, crowded, numeric, stray] {
        let err = allow::collapse(bad).expect_err("malformed wrapper must be rejected");
        assert!(
            err.diagnostics
                .iter()
                .all(|d| d.code == allow::codes::UNCOLLAPSED),
            "{err}"
        );
    }
}

/// Annotating an import placeholder is refused, the way the JSON side
/// refuses `{"$import": …, "$allow": …}`. The loader would replace the
/// placeholder with the imported tree and take the annotation with it,
/// so a fold that accepted this would drop a suppression the author
/// believed was in effect.
#[test]
fn annotating_an_import_placeholder_fails_the_fold() {
    use dsl_kit_parse::import::{IMPORT_VARIANT, add_import_syntax};

    let ids = IdGen::new();
    let mut grammar = checked_grammar_from_schema(&Flow::schema(), &ids).expect("grammar");
    add_import_syntax(&mut grammar, &ids).expect("import");
    allow::add_allow_syntax(&mut grammar, &ids).expect("allow");

    for input in [
        r#"@allow("naming") @import "lib""#,
        // Stacked wrappers reach the same placeholder, and nesting
        // does not hide it either.
        r#"@allow("a") @allow("b") @import "lib""#,
        r#"Par(branches: [@allow("naming") @import "lib"])"#,
    ] {
        let Err(err) = grammar.parse(input) else {
            panic!("annotated import placeholder must be rejected: {input}");
        };
        assert_eq!(err.diagnostics.len(), 1, "{input}: {err}");
        assert_eq!(err.diagnostics[0].code, allow::codes::UNCOLLAPSED);
        assert!(
            err.diagnostics[0].message.contains(IMPORT_VARIANT)
                && err.diagnostics[0]
                    .message
                    .contains("annotate the imported source's own nodes instead"),
            "{input}: message should name the placeholder and the way out: {}",
            err.diagnostics[0].message,
        );
    }
}

/// A wrapper nested under a well-formed node is reached too — the
/// fold walks the whole tree, not just its root.
#[test]
fn a_malformed_wrapper_below_the_root_is_reported() {
    let mut wrapper = ParseTree::new(allow::ALLOW_VARIANT);
    wrapper.fields.push((
        allow::ALLOW_RULES_FIELD.to_string(),
        RawValue::Text("naming".into()),
    ));
    let mut par = ParseTree::new("Par");
    par.children.push(("branches".to_string(), vec![wrapper]));

    let err = allow::collapse(par).expect_err("nested wrapper is malformed");
    assert_eq!(err.diagnostics[0].code, allow::codes::UNCOLLAPSED);
}
