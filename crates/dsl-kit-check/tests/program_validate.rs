//! Acceptance for the self-validation half of the S2 slice.
//!
//! Findings from this crate are hard errors a document cannot suppress,
//! which is only fair while the program itself is sound. These tests
//! pin the three ways a program can be unsound, and — just as
//! important — pin that a healthy program stays silent, so the check
//! does not become noise the author learns to ignore.

use dsl_kit_check::{CheckProgram, Fact, Rule, SeqSlotDecl, atom, codes, fact};
use dsl_kit_parse::Severity;

fn state(name: &str) -> Fact {
    fact("state", [atom(name)])
}

/// The three-step provisioning program from the solver suite: every
/// required state is produced, every predicate is consumed, one rule
/// per variant.
fn healthy() -> CheckProgram {
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
        .build()
}

#[test]
fn a_healthy_program_is_silent() {
    let diags = healthy().validate();
    assert!(diags.is_empty(), "diags = {diags:?}");
}

#[test]
fn a_state_nothing_reaches_is_an_error() {
    // `Fetchd` is a typo for `Fetched`: no rule produces it and no
    // fold starts from it, so `Build` could never fire — and every
    // document containing a `Build` would be rejected with no way to
    // opt out.
    let program = CheckProgram::builder()
        .seq_slot(SeqSlotDecl::fold("Plan", "steps", state("Raw")))
        .rule(
            Rule::on("Build")
                .requires_state(state("Fetchd"))
                .transitions_to(state("Built"))
                .message(codes::CHECK_STATE_MISMATCH, "`build` needs {expected}"),
        )
        .build();

    let diags = program.validate();

    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].severity, Severity::Error);
    assert_eq!(diags[0].code, codes::CHECK_PROGRAM_UNDEFINED_STATE);
    assert!(
        diags[0]
            .message
            .contains("rule `Build` requires `state(Fetchd)`"),
        "message = {}",
        diags[0].message
    );
    // The wording says what the program *can* reach, so the author can
    // spot the missing letter without re-reading the whole program.
    assert!(
        diags[0]
            .message
            .contains("(reachable: state(Built), state(Raw))"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn an_open_pattern_is_satisfied_by_anything_produced() {
    // `state($x)` is satisfiable as long as *some* state exists: the
    // check rules out the impossible, it does not demand a literal
    // match.
    let program = CheckProgram::builder()
        .seq_slot(SeqSlotDecl::fold("Plan", "steps", state("Raw")))
        .rule(
            Rule::on("Any")
                .requires_state(fact("state", [dsl_kit_check::var("x")]))
                .message(codes::CHECK_STATE_MISMATCH, "unused"),
        )
        .build();

    assert!(program.validate().is_empty());
}

#[test]
fn a_predicate_no_premise_consumes_is_a_warning() {
    // The `cap` family is produced and never asked about — legal, but
    // almost always half of a misspelt pair.
    let program = CheckProgram::builder()
        .seq_slot(SeqSlotDecl::fold("Plan", "steps", state("Raw")))
        .rule(
            Rule::on("Fetch")
                .requires_state(state("Raw"))
                .transitions_to(state("Fetched"))
                .concludes(fact("cap", [atom("Net")]))
                .message(codes::CHECK_STATE_MISMATCH, "`fetch` needs {expected}"),
        )
        .build();

    let diags = program.validate();

    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].severity, Severity::Warning);
    assert_eq!(diags[0].code, codes::CHECK_PROGRAM_UNUSED_PRED);
    assert!(
        diags[0].message.contains("predicate `cap` is produced"),
        "message = {}",
        diags[0].message
    );
    assert!(
        diags[0].message.contains("rule `Fetch`"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn a_terminal_state_is_not_an_unused_predicate() {
    // `state(Built)` is produced and never required — but the *family*
    // is consumed elsewhere, and a run has to end somewhere. Warning
    // at predicate granularity is what keeps this quiet.
    assert!(healthy().validate().is_empty());
}

#[test]
fn a_rule_behind_an_unconditional_one_is_unreachable() {
    let program = CheckProgram::builder()
        .rule(
            Rule::on("Not")
                .child("arg", fact("type", [atom("Bool")]))
                .concludes(fact("type", [atom("Bool")]))
                .message(codes::CHECK_TYPE_MISMATCH, "`not` wants {expected}"),
        )
        .rule(
            Rule::on("Lit")
                .concludes(fact("type", [atom("Int")]))
                .message(codes::CHECK_TYPE_MISMATCH, "int literal"),
        )
        .rule(
            Rule::on("Lit")
                .concludes(fact("type", [atom("Bool")]))
                .message(codes::CHECK_TYPE_MISMATCH, "bool literal"),
        )
        .build();

    let diags = program.validate();

    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].severity, Severity::Warning);
    assert_eq!(diags[0].code, codes::CHECK_PROGRAM_UNREACHABLE_RULE);
    assert!(
        diags[0]
            .message
            .contains("rule #2 for variant `Lit` can never fire"),
        "message = {}",
        diags[0].message
    );
    assert!(
        diags[0].message.contains("it is unconditional"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn a_repeated_rule_is_unreachable() {
    let program = CheckProgram::builder()
        .rule(
            Rule::on("Not")
                .child("arg", fact("type", [atom("Bool")]))
                .concludes(fact("type", [atom("Bool")]))
                .message(codes::CHECK_TYPE_MISMATCH, "`not` wants {expected}"),
        )
        .rule(
            Rule::on("Not")
                .child("arg", fact("type", [atom("Bool")]))
                .concludes(fact("type", [atom("Int")]))
                .message(codes::CHECK_TYPE_MISMATCH, "unreachable twin"),
        )
        .build();

    let diags = program.validate();

    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].code, codes::CHECK_PROGRAM_UNREACHABLE_RULE);
    assert!(
        diags[0].message.contains("it carries the same premises"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn distinct_premises_on_one_variant_stay_reachable() {
    // Two rules for `Lit`, each with its own premise: declaration
    // order picks the winner per document, and neither is dead.
    let program = CheckProgram::builder()
        .seq_slot(SeqSlotDecl::fold("Plan", "steps", state("Raw")))
        .rule(
            Rule::on("Lit")
                .requires_state(state("Raw"))
                .transitions_to(state("Seen"))
                .message(codes::CHECK_STATE_MISMATCH, "first"),
        )
        .rule(
            Rule::on("Lit")
                .requires_state(state("Seen"))
                .transitions_to(state("Seen"))
                .message(codes::CHECK_STATE_MISMATCH, "second"),
        )
        .build();

    assert!(program.validate().is_empty());
}
