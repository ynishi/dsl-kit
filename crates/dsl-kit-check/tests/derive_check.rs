//! Acceptance for the derive half of the S2 slice: the Stage 1
//! attribute vocabulary (`requires` / `produces` / `fold`), compiled by
//! `#[derive(DslCheck)]` into a `CheckProgram`, catches the ordering
//! mistake the design opens with.
//!
//! The enum below is the design's own sketch (`SystemPkg` →
//! `PythonInstall` → `ComfyUIInstall` → `Readiness`) with the `$var`
//! parts left for the next slice, wrapped in a `Plan` container whose
//! `steps` slot is the fold. It derives `DslNode` and `DslSchema` too:
//! the check derive has to compose with the rest of the family rather
//! than demand an enum of its own.

use dsl_kit_check::{DslCheck, SeqMode, atom, check_semantics, codes, fact};
use dsl_kit_core::NodeId;
use dsl_kit_macros::{DslCheck, DslNode, DslSchema};
use dsl_kit_parse::ParseTree;

/// A provisioning DSL: a plan is an ordered list of phases, and each
/// phase declares the state it needs and the state it leaves behind.
///
/// `#[allow(dead_code)]` at the enum level — the variants exist to be
/// annotated and are never constructed as Rust values here; the tests
/// drive the derived program against `ParseTree` fixtures, which is
/// what a front-end hands the check layer.
#[allow(dead_code)]
#[derive(Debug, DslNode, DslSchema, DslCheck)]
enum Phase {
    /// The container. Its `steps` slot is ordered, so it threads a
    /// state from `state(Raw)` through the phases left to right.
    Plan {
        id: NodeId,
        #[dsl_check(fold(initial = "state(Raw)"))]
        steps: Vec<Phase>,
    },

    /// System packages come first: nothing is required, and the plan
    /// leaves this step ready for a language runtime.
    #[dsl_check(requires = "state(Raw)", produces = "state(SystemReady)")]
    SystemPkg { id: NodeId, packages: Vec<String> },

    /// Python needs the system packages.
    #[dsl_check(requires = "state(SystemReady)", produces = "state(PythonEnv)")]
    PythonInstall { id: NodeId, version: String },

    /// The application needs Python — and says so in its own words.
    #[dsl_check(
        requires = "state(PythonEnv)",
        produces = "state(ComfyUIInstalled)",
        message = "`comfyui` needs {expected}; the plan is at {found}, set by {provenance}"
    )]
    ComfyUIInstall { id: NodeId, source: String },

    /// A terminal probe: requires the installed application, changes
    /// nothing.
    #[dsl_check(requires = "state(ComfyUIInstalled)")]
    Readiness { id: NodeId, port: u16 },
}

/// `Plan { steps: [...] }` as the front-end would hand it over.
fn plan(steps: &[&str]) -> ParseTree {
    let mut tree = ParseTree::new("Plan");
    tree.children = vec![(
        "steps".into(),
        steps.iter().map(|s| ParseTree::new(*s)).collect(),
    )];
    tree
}

#[test]
fn the_attributes_compile_into_rules_and_a_fold_declaration() {
    let program = Phase::check_program();

    // One rule per annotated variant — and none for `Plan`, which
    // carries no judgement: annotating is opt-in per variant.
    assert_eq!(program.rules.len(), 4, "rules = {:?}", program.rules);
    assert_eq!(program.rules_for("Plan").count(), 0);
    assert_eq!(program.rules_for("PythonInstall").count(), 1);

    let rule = program
        .rules_for("PythonInstall")
        .next()
        .expect("PythonInstall rule");
    assert_eq!(
        rule.premises,
        vec![dsl_kit_check::Premise::State {
            expect: fact("state", [atom("SystemReady")])
        }]
    );
    assert_eq!(rule.state_after, Some(fact("state", [atom("PythonEnv")])));
    // `produces` is a state transition, not a synthesised attribute.
    assert_eq!(rule.conclusion, None);
    assert_eq!(rule.message.code, codes::CHECK_STATE_MISMATCH);

    // The field annotation is what makes `steps` ordered.
    assert_eq!(program.seq_slots.len(), 1);
    let decl = program
        .seq_slot("Plan", "steps")
        .expect("the fold declaration");
    assert_eq!(decl.mode, SeqMode::Fold);
    assert_eq!(decl.initial, fact("state", [atom("Raw")]));
}

#[test]
fn the_intended_order_passes() {
    let doc = plan(&["SystemPkg", "PythonInstall", "ComfyUIInstall", "Readiness"]);
    let diags = check_semantics(&doc, &Phase::check_program());
    assert!(diags.is_empty(), "diags = {diags:?}");
}

#[test]
fn a_step_out_of_order_is_reported_where_it_stands() {
    // Python before the system packages: the plan is still `Raw`.
    let diags = check_semantics(
        &plan(&["PythonInstall", "SystemPkg"]),
        &Phase::check_program(),
    );

    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].code, codes::CHECK_STATE_MISMATCH);
    assert!(
        diags[0]
            .message
            .starts_with("`PythonInstall` requires state(SystemReady), found state(Raw)"),
        "message = {}",
        diags[0].message
    );
    // The default wording names the seed of the fold...
    assert!(
        diags[0].message.contains("(from <initial>)"),
        "message = {}",
        diags[0].message
    );
    // ...and the solver anchors the complaint at the offending step.
    assert!(
        diags[0].message.ends_with("[at steps[0]]"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn a_missing_prerequisite_is_reported_once() {
    // `Readiness` never gets its application: one diagnostic, not a
    // cascade, because a rule that failed leaves the state alone.
    let diags = check_semantics(&plan(&["SystemPkg", "Readiness"]), &Phase::check_program());

    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert!(
        diags[0]
            .message
            .starts_with("`Readiness` requires state(ComfyUIInstalled), found state(SystemReady)"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn a_declared_message_replaces_the_default_wording() {
    let diags = check_semantics(&plan(&["ComfyUIInstall"]), &Phase::check_program());

    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert!(
        diags[0]
            .message
            .starts_with("`comfyui` needs state(PythonEnv); the plan is at state(Raw)"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn the_derived_program_passes_its_own_validation() {
    // The derive is only worth trusting if what it emits is sound by
    // the load-time check's own standards.
    let diags = Phase::check_program().validate();
    assert!(diags.is_empty(), "diags = {diags:?}");
}
