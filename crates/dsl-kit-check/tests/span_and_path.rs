//! Acceptance 3 of the Check IR slice: the same violation is anchored
//! whichever front-end produced the tree.
//!
//! - text (PEG) front-end → the tree carries a [`Span`], so the
//!   diagnostic's `location` is a byte range and the message repeats it;
//! - serde front-end → spans are `None` by construction, so the
//!   message carries the solver's own path trail (`steps[1]`) instead.
//!
//! Both trees are shape-clean (`check_conformance` returns nothing):
//! ordering is exactly the class of error the conformance layer cannot
//! see.

use dsl_kit_check::{CheckProgram, Rule, SeqSlotDecl, atom, check_semantics, codes, fact};
use dsl_kit_core::IdGen;
use dsl_kit_parse::{
    Location, check_conformance, schema_gen::checked_grammar_from_schema,
    serde_bridge::from_json_value,
};
use dsl_kit_schema::{ChildSchema, Multiplicity, NodeSchema, VariantSchema};
use serde_json::json;

fn state(name: &str) -> dsl_kit_check::Fact {
    fact("state", [atom(name)])
}

/// Hand-written schema — no derive, so the fixture stays inside this
/// slice's "macros come later" boundary.
fn schema() -> NodeSchema {
    let step = |name: &str| VariantSchema {
        name: name.into(),
        fields: vec![],
        children: vec![],
    };
    NodeSchema {
        name: "Provision".into(),
        variants: vec![
            VariantSchema {
                name: "Plan".into(),
                fields: vec![],
                children: vec![ChildSchema::recursive("steps", Multiplicity::Many)],
            },
            step("Fetch"),
            step("Build"),
            step("Deploy"),
        ],
    }
}

fn provisioning() -> CheckProgram {
    CheckProgram::builder()
        .seq_slot(SeqSlotDecl::fold("Plan", "steps", state("Raw")))
        .rule(
            Rule::on("Fetch")
                .requires_state(state("Raw"))
                .transitions_to(state("Fetched"))
                .message(codes::CHECK_STATE_MISMATCH, "`fetch` needs {expected}"),
        )
        .rule(
            Rule::on("Build")
                .requires_state(state("Fetched"))
                .transitions_to(state("Built"))
                .message(codes::CHECK_STATE_MISMATCH, "`build` needs {expected}"),
        )
        .rule(
            Rule::on("Deploy")
                .requires_state(state("Built"))
                .transitions_to(state("Deployed"))
                .message(
                    codes::CHECK_STATE_MISMATCH,
                    "`deploy` needs {expected}, found {found}",
                ),
        )
        .build()
}

#[test]
fn the_text_front_end_anchors_on_a_span() {
    let schema = schema();
    let grammar =
        checked_grammar_from_schema(&schema, &IdGen::new()).expect("schema generates a grammar");
    let source = "Plan(steps: [Fetch(), Deploy(), Build()])";
    let tree = grammar.parse(source).expect("canonical text parses");
    assert!(
        check_conformance(&tree, &schema).is_empty(),
        "the document is shape-clean; only its ordering is wrong"
    );

    let diags = check_semantics(&tree, &provisioning());
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].code, codes::CHECK_STATE_MISMATCH);

    let Location::Span(span) = diags[0].location else {
        panic!("expected a span location, got {:?}", diags[0].location);
    };
    // The span covers the offending step, not the whole document
    // (the PEG front-end folds the leading separator into the range).
    assert_eq!(source[span.start..span.end].trim(), "Deploy()");
    // …and the message repeats the anchor, path and bytes both.
    assert!(
        diags[0].message.ends_with(&format!(
            "[at steps[1] (bytes {}..{})]",
            span.start, span.end
        )),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn the_serde_front_end_anchors_on_the_path_trail() {
    let schema = schema();
    let tree = from_json_value(
        &json!({
            "type": "Plan",
            "steps": [{ "type": "Fetch" }, { "type": "Deploy" }, { "type": "Build" }],
        }),
        &schema,
    )
    .expect("the document loads");
    assert!(check_conformance(&tree, &schema).is_empty());
    assert!(
        tree.span.is_none(),
        "the serde bridge has no source positions to hand over"
    );

    let diags = check_semantics(&tree, &provisioning());
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].location, Location::None);
    assert!(
        diags[0]
            .message
            .ends_with("`deploy` needs state(Built), found state(Fetched) [at steps[1]]"),
        "message = {}",
        diags[0].message
    );
}
