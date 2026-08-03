//! Acceptance for the S3 slice, suggestion half: what a `did you mean`
//! hint is measured against, and when it stays quiet.
//!
//! The candidate set has two halves, and both are needed:
//!
//! - **what the document produced** — a state handle
//!   (`Env(staging)`) is a value a step chose, so it appears in no
//!   schema and in no program; the only place to find it is the fact
//!   the solver actually had in hand;
//! - **what the program can mention** — every ground atom and
//!   constructor name in a premise, conclusion, transition, or fold
//!   seed, which is what catches a misspelling on the *rule* side.
//!
//! Hints are enrichment: [`check_semantics`] passes a no-op suggester,
//! so a host that wants none keeps byte-identical messages.

use dsl_kit_check::{
    CheckProgram, Rule, SeqSlotDecl, atom, check_semantics, check_semantics_with, codes, ctor,
    fact, field_ref,
};
use dsl_kit_fuzzy::FuzzySuggester;
use dsl_kit_parse::{ParseTree, RawValue};

/// "A plan provisions an environment, then deploys into it — by name."
fn deployment() -> CheckProgram {
    CheckProgram::builder()
        .seq_slot(SeqSlotDecl::fold(
            "Plan",
            "steps",
            fact("state", [atom("Raw")]),
        ))
        .rule(
            Rule::on("Provision")
                .requires_state(fact("state", [atom("Raw")]))
                .transitions_to(fact("state", [ctor("Env", [field_ref("env")])]))
                .message(codes::CHECK_STATE_MISMATCH, "`provision` starts a plan"),
        )
        .rule(
            Rule::on("Deploy")
                .requires_state(fact("state", [ctor("Env", [field_ref("target")])]))
                .message(
                    codes::CHECK_STATE_MISMATCH,
                    "`deploy` targets {expected}, but the plan is at {found} (from {provenance})",
                ),
        )
        .build()
}

fn step(variant: &str, field: &str, value: &str) -> ParseTree {
    let mut tree = ParseTree::new(variant);
    tree.fields = vec![(field.to_string(), RawValue::Text(value.to_string()))];
    tree
}

fn plan(steps: Vec<ParseTree>) -> ParseTree {
    let mut tree = ParseTree::new("Plan");
    tree.children = vec![("steps".into(), steps)];
    tree
}

#[test]
fn the_handle_the_document_produced_is_offered_back() {
    let doc = plan(vec![
        step("Provision", "env", "staging"),
        step("Deploy", "target", "stagin"),
    ]);

    let diags = check_semantics_with(&doc, &deployment(), &FuzzySuggester::default());
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert!(
        diags[0].message.starts_with(
            "`deploy` targets state(Env(stagin)), but the plan is at state(Env(staging))"
        ),
        "message = {}",
        diags[0].message
    );
    assert!(
        diags[0].message.contains("did you mean: staging"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn the_plain_entry_point_adds_nothing() {
    let doc = plan(vec![
        step("Provision", "env", "staging"),
        step("Deploy", "target", "stagin"),
    ]);

    let diags = check_semantics(&doc, &deployment());
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert!(
        !diags[0].message.contains("did you mean"),
        "message = {}",
        diags[0].message
    );
}

/// The other half of the candidate set: a name misspelt in the
/// *program* is matched against the rest of the program, which is
/// where the correct spelling lives.
#[test]
fn a_misspelt_rule_atom_is_matched_against_the_program() {
    let program = CheckProgram::builder()
        .rule(
            Rule::on("IntLit")
                .concludes(fact("type", [atom("Int")]))
                .message(codes::CHECK_TYPE_MISMATCH, "integer literal"),
        )
        .rule(
            Rule::on("BoolLit")
                .concludes(fact("type", [atom("Bool")]))
                .message(codes::CHECK_TYPE_MISMATCH, "boolean literal"),
        )
        .rule(
            // The typo: nothing concludes `type(Intt)`.
            Rule::on("Neg")
                .child("arg", fact("type", [atom("Intt")]))
                .concludes(fact("type", [atom("Int")]))
                .message(
                    codes::CHECK_TYPE_MISMATCH,
                    "`neg` wants {expected}, got {found}",
                ),
        )
        .build();

    let mut tree = ParseTree::new("Neg");
    tree.children = vec![("arg".into(), vec![ParseTree::new("IntLit")])];

    let diags = check_semantics_with(&tree, &program, &FuzzySuggester::default());
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert!(
        diags[0]
            .message
            .starts_with("`neg` wants type(Intt), got type(Int)"),
        "message = {}",
        diags[0].message
    );
    assert!(
        diags[0].message.contains("did you mean: Int"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn an_ordinary_type_disagreement_gets_no_spelling_advice() {
    // `Bool` is not a misspelt `Int`; a suggester that fires here
    // would be noise on every ordinary type error.
    let program = CheckProgram::builder()
        .rule(
            Rule::on("BoolLit")
                .concludes(fact("type", [atom("Bool")]))
                .message(codes::CHECK_TYPE_MISMATCH, "boolean literal"),
        )
        .rule(
            Rule::on("Neg")
                .child("arg", fact("type", [atom("Int")]))
                .concludes(fact("type", [atom("Int")]))
                .message(
                    codes::CHECK_TYPE_MISMATCH,
                    "`neg` wants {expected}, got {found}",
                ),
        )
        .build();

    let mut tree = ParseTree::new("Neg");
    tree.children = vec![("arg".into(), vec![ParseTree::new("BoolLit")])];

    let diags = check_semantics_with(&tree, &program, &FuzzySuggester::default());
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert!(
        !diags[0].message.contains("did you mean"),
        "message = {}",
        diags[0].message
    );
}
