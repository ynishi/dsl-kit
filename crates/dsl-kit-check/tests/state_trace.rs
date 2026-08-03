//! Acceptance 2 of the Check IR slice: a three-step state trace
//! (`Raw → Fetched → Built → Deployed`) declared as a
//! `SeqMode::Fold` slot detects an out-of-order step as
//! `CHECK_STATE_MISMATCH`.
//!
//! Also pins the two properties that make the fold usable: the seed
//! state names itself as `<initial>` in provenance, and a step whose
//! premise failed leaves the running state alone (the rule did not
//! fire, so its transition did not happen).

use dsl_kit_check::{CheckProgram, Rule, SeqSlotDecl, atom, check_semantics, codes, fact};
use dsl_kit_parse::ParseTree;

fn state(name: &str) -> dsl_kit_check::Fact {
    fact("state", [atom(name)])
}

/// `Plan { steps: [...] }` folds `state(Raw)` through its steps.
fn provisioning() -> CheckProgram {
    CheckProgram::builder()
        .seq_slot(SeqSlotDecl::fold("Plan", "steps", state("Raw")))
        .rule(
            Rule::on("Fetch")
                .requires_state(state("Raw"))
                .transitions_to(state("Fetched"))
                .message(
                    codes::CHECK_STATE_MISMATCH,
                    "`fetch` needs {expected} but the plan is at {found} (set by {provenance})",
                ),
        )
        .rule(
            Rule::on("Build")
                .requires_state(state("Fetched"))
                .transitions_to(state("Built"))
                .message(
                    codes::CHECK_STATE_MISMATCH,
                    "`build` needs {expected} but the plan is at {found} (set by {provenance})",
                ),
        )
        .rule(
            Rule::on("Deploy")
                .requires_state(state("Built"))
                .transitions_to(state("Deployed"))
                .message(
                    codes::CHECK_STATE_MISMATCH,
                    "`deploy` needs {expected} but the plan is at {found} (set by {provenance})",
                ),
        )
        .build()
}

fn plan(steps: &[&str]) -> ParseTree {
    let mut tree = ParseTree::new("Plan");
    tree.children = vec![(
        "steps".into(),
        steps.iter().map(|s| ParseTree::new(*s)).collect(),
    )];
    tree
}

#[test]
fn the_intended_order_passes() {
    assert!(check_semantics(&plan(&["Fetch", "Build", "Deploy"]), &provisioning()).is_empty());
}

#[test]
fn a_swapped_pair_is_reported_once() {
    let diags = check_semantics(&plan(&["Fetch", "Deploy", "Build"]), &provisioning());

    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].code, codes::CHECK_STATE_MISMATCH);
    assert!(
        diags[0]
            .message
            .starts_with("`deploy` needs state(Built) but the plan is at state(Fetched)"),
        "message = {}",
        diags[0].message
    );
    // Provenance names the step that last moved the state...
    assert!(
        diags[0].message.contains("(set by steps[0])"),
        "message = {}",
        diags[0].message
    );
    // ...and the anchor names the step that broke.
    assert!(
        diags[0].message.ends_with("[at steps[1]]"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn the_seed_state_names_itself() {
    // Nothing has run yet, so `build`'s complaint points at the
    // declaration rather than at a step.
    let diags = check_semantics(&plan(&["Build"]), &provisioning());

    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert!(
        diags[0]
            .message
            .starts_with("`build` needs state(Fetched) but the plan is at state(Raw)"),
        "message = {}",
        diags[0].message
    );
    assert!(
        diags[0].message.contains("(set by <initial>)"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn a_failed_step_does_not_move_the_state() {
    // `Build` fails at Raw and leaves the state Raw, so `Deploy`
    // reports against Raw too — both steps really are unreachable
    // here, and neither complaint is an artefact of the other.
    let diags = check_semantics(&plan(&["Build", "Deploy"]), &provisioning());

    assert_eq!(diags.len(), 2, "diags = {diags:?}");
    assert!(diags[0].message.starts_with("`build` needs state(Fetched)"));
    assert!(
        diags[1]
            .message
            .starts_with("`deploy` needs state(Built) but the plan is at state(Raw)"),
        "message = {}",
        diags[1].message
    );
}

#[test]
fn a_step_outside_a_declared_fold_has_no_state_to_violate() {
    // `Steps` was never declared as a fold slot, so no state is in
    // scope and the state premises pass vacuously — the check layer
    // stays opt-in per slot, not per variant.
    let mut tree = ParseTree::new("Unfolded");
    tree.children = vec![("steps".into(), vec![ParseTree::new("Deploy")])];
    assert!(check_semantics(&tree, &provisioning()).is_empty());
}

#[test]
fn each_declared_fold_starts_over() {
    // Two sibling plans: the second is not contaminated by the first.
    let mut outer = ParseTree::new("Workspace");
    outer.children = vec![(
        "plans".into(),
        vec![
            plan(&["Fetch", "Build"]),
            plan(&["Deploy"]),
            plan(&["Fetch", "Build", "Deploy"]),
        ],
    )];

    let diags = check_semantics(&outer, &provisioning());
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert!(
        diags[0].message.ends_with("[at plans[1].steps[0]]"),
        "message = {}",
        diags[0].message
    );
}
