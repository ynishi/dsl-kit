//! End-to-end coverage for declared non-emptiness on collection
//! child slots (`ChildSchema::non_empty` / `#[dsl_schema(non_empty)]`)
//! — the declaration that lets `no-empty-child-slots` (and everything
//! else) check a stated constraint instead of inferring one. Verifies
//! that:
//!
//! - `#[derive(DslSchema)]` records the declaration and the wire JSON
//!   gains `"non_empty": true` only where declared;
//! - `check_conformance` rejects a violating tree with
//!   `ARITY_NON_EMPTY`, treating an absent slot and a
//!   present-but-empty slot the same, and leaves undeclared
//!   collection slots on their zero-or-more contract;
//! - the generated canonical-text grammar requires at least one
//!   element for a declared slot (and the synthesized minimal example
//!   therefore carries one), while undeclared slots keep accepting
//!   the empty spelling;
//! - `grammar_from_schema`'s pre-flight rejects `non_empty` on
//!   multiplicities that have no empty collection to forbid.

use dsl_kit_core::{IdGen, NodeId};
use dsl_kit_macros::{DslBuild, DslNode, DslSchema};
use dsl_kit_parse::{
    DslBuild, check_conformance, codes,
    example_gen::examples_from_grammar,
    schema_gen::{self, checked_grammar_from_schema, grammar_from_schema},
    serde_bridge::from_json_value,
};
use dsl_kit_schema::{ChildSchema, DslSchema, Multiplicity, NodeSchema, VariantSchema};
use serde_json::json;
use std::collections::BTreeMap;

/// AST mixing declared and undeclared collection slots so the two
/// contracts must be told apart per slot, not per crate.
#[derive(Debug, PartialEq, DslNode, DslSchema, DslBuild)]
enum Prog {
    /// A pipeline that is meaningless without at least one stage.
    Pipeline {
        /// Stable node id.
        id: NodeId,
        /// Declared non-empty `Many` slot.
        #[dsl_schema(non_empty)]
        stages: Vec<Prog>,
    },
    /// A block where `{}` is a legal no-op — stays zero-or-more.
    Block {
        /// Stable node id.
        id: NodeId,
        /// Undeclared `Many` slot.
        stmts: Vec<Prog>,
    },
    /// An env map that must carry at least one binding.
    Env {
        /// Stable node id.
        id: NodeId,
        /// Declared non-empty scalar `Map` slot.
        #[dsl_schema(non_empty)]
        vars: BTreeMap<String, String>,
    },
    /// Leaf statement.
    Step {
        /// Stable node id.
        id: NodeId,
        /// Step label.
        label: String,
    },
}

/// The derive records the flag where declared — and only there — and
/// the wire JSON layout is unchanged for undeclared slots.
#[test]
fn schema_records_non_empty_where_declared() {
    let schema = Prog::schema();
    assert!(schema.variant("Pipeline").unwrap().children[0].non_empty);
    assert!(schema.variant("Env").unwrap().children[0].non_empty);
    assert!(!schema.variant("Block").unwrap().children[0].non_empty);

    let json = schema.to_json();
    let variants = json["variants"].as_array().unwrap();
    let pipeline = variants.iter().find(|v| v["name"] == "Pipeline").unwrap();
    assert_eq!(pipeline["children"][0]["non_empty"], json!(true));
    let block = variants.iter().find(|v| v["name"] == "Block").unwrap();
    assert!(block["children"][0].get("non_empty").is_none());
}

/// Conformance rejects declared-empty (absent or present-but-empty),
/// accepts populated, and leaves the undeclared slot alone.
#[test]
fn conformance_enforces_declared_non_emptiness() {
    let schema = Prog::schema();

    let empty = from_json_value(&json!({ "type": "Pipeline", "stages": [] }), &schema).unwrap();
    let diags = check_conformance(&empty, &schema);
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].code, codes::ARITY_NON_EMPTY);
    assert!(diags[0].message.contains("`stages`"));

    // Absent slot means the same thing as present-but-empty.
    let absent = from_json_value(&json!({ "type": "Pipeline" }), &schema).unwrap();
    assert_eq!(
        check_conformance(&absent, &schema)[0].code,
        codes::ARITY_NON_EMPTY
    );

    let keyed_empty = from_json_value(&json!({ "type": "Env", "vars": {} }), &schema).unwrap();
    let diags = check_conformance(&keyed_empty, &schema);
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].code, codes::ARITY_NON_EMPTY);
    assert!(diags[0].message.contains("`vars`"));

    let populated = from_json_value(
        &json!({ "type": "Pipeline", "stages": [{ "type": "Step", "label": "build" }] }),
        &schema,
    )
    .unwrap();
    assert!(check_conformance(&populated, &schema).is_empty());
    assert!(
        Prog::from_parse_tree(&populated, &IdGen::new()).is_ok(),
        "populated pipeline builds through to the typed AST"
    );

    // The undeclared slot keeps its zero-or-more contract verbatim.
    let noop = from_json_value(&json!({ "type": "Block", "stmts": [] }), &schema).unwrap();
    assert!(check_conformance(&noop, &schema).is_empty());
}

/// The generated grammar requires an element for the declared slots
/// and keeps accepting the empty spelling for the undeclared one.
#[test]
fn text_grammar_honours_the_declaration() {
    let schema = Prog::schema();
    let grammar = checked_grammar_from_schema(&schema, &IdGen::new())
        .expect("non_empty schema generates a clean grammar");

    assert!(
        grammar.parse("Pipeline(stages: [])").is_err(),
        "empty list must not parse for a non_empty slot"
    );
    assert!(
        grammar.parse("Env(vars: {})").is_err(),
        "empty map must not parse for a non_empty slot"
    );
    let tree = grammar
        .parse(r#"Pipeline(stages: [Step(label: "build")])"#)
        .expect("one element satisfies the declaration");
    assert!(check_conformance(&tree, &schema).is_empty());
    let keyed = grammar
        .parse(r#"Env(vars: { LOG: "info" })"#)
        .expect("one entry satisfies the declaration");
    assert!(check_conformance(&keyed, &schema).is_empty());
    let noop = grammar
        .parse("Block(stmts: [])")
        .expect("undeclared slot keeps the empty spelling");
    assert!(check_conformance(&noop, &schema).is_empty());
}

/// The synthesized minimal example for a non-empty slot is not the
/// empty collection — the constraint propagates to example_gen
/// through the grammar itself.
#[test]
fn minimal_examples_carry_an_element() {
    let schema = Prog::schema();
    let grammar = checked_grammar_from_schema(&schema, &IdGen::new()).unwrap();
    let examples = examples_from_grammar(&grammar).expect("examples derivable");
    for e in &examples.per_rule {
        match e.rule.as_str() {
            "Pipeline" => assert!(
                !e.text.contains("[]"),
                "Pipeline minimal example must carry a stage: {}",
                e.text
            ),
            "Env" => assert!(
                !e.text.contains("{}"),
                "Env minimal example must carry an entry: {}",
                e.text
            ),
            "Block" => assert!(
                e.text.contains("[]"),
                "Block minimal stays the empty list: {}",
                e.text
            ),
            _ => {}
        }
        let tree = grammar
            .parse(&e.text)
            .unwrap_or_else(|err| panic!("example {} unparsable: {:?}", e.text, err.diagnostics));
        assert!(
            check_conformance(&tree, &schema).is_empty(),
            "example for `{}` conforms: {}",
            e.rule,
            e.text
        );
    }
}

/// Pre-flight rejects `non_empty` on multiplicities that cannot be
/// empty collections.
#[test]
fn grammar_preflight_rejects_non_empty_on_one_and_optional() {
    for mult in [Multiplicity::One, Multiplicity::Optional] {
        let schema = NodeSchema {
            name: "Cfg".into(),
            variants: vec![
                VariantSchema {
                    name: "Leaf".into(),
                    fields: vec![],
                    children: vec![],
                },
                VariantSchema {
                    name: "Holder".into(),
                    fields: vec![],
                    children: vec![ChildSchema::recursive("body", mult).with_non_empty()],
                },
            ],
        };
        let err = grammar_from_schema(&schema, &IdGen::new())
            .expect_err("non_empty on a non-collection slot must fail pre-flight");
        assert!(
            err.diagnostics
                .iter()
                .all(|d| d.code == schema_gen::codes::INVALID_NON_EMPTY),
            "unexpected codes: {:?}",
            err.diagnostics
        );
    }
}
