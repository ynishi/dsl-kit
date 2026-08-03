//! Acceptance for the S3 slice, derive half: the Stage 1 sketch from
//! the design compiles through `#[derive(DslCheck)]` as written —
//! parameterised states (`state(ServiceRunning($name))`), the
//! `bind(...)` wiring that feeds a `$var` from a payload field, and the
//! child-slot form (`requires(cond = "type(Bool)")`) that carries the
//! tree-typing half.
//!
//! The three error examples the design opens with are reproduced here
//! end to end: a step out of order, a service handle that does not
//! match the one the plan started (with a `did you mean` hint), and a
//! type disagreement between operands.

use dsl_kit_check::{
    DslCheck, Premise, atom, check_semantics, check_semantics_with, codes, ctor, fact, field_ref,
    var,
};
use dsl_kit_core::NodeId;
use dsl_kit_fuzzy::FuzzySuggester;
use dsl_kit_macros::{DslCheck, DslNode, DslSchema};
use dsl_kit_parse::{ParseTree, RawValue};

// ---------------------------------------------------------------------------
// The design's Stage 1 sketch, verbatim
// ---------------------------------------------------------------------------

/// The four variants of the design's Stage 1 sketch, annotated exactly
/// as it spells them. The `id: NodeId` fields are the one addition —
/// `DslNode` requires them, and the point of the sketch is that the
/// check derive composes with the rest of the family.
///
/// `#[allow(dead_code)]`: the variants exist to be annotated. The tests
/// drive the derived program against `ParseTree` fixtures, which is
/// what a front-end hands the check layer.
#[allow(dead_code)]
#[derive(Debug, DslNode, DslSchema, DslCheck)]
enum Stage1 {
    #[dsl_check(produces = "state(SystemReady)")]
    SystemPkg { id: NodeId, packages: Vec<String> },

    #[dsl_check(requires = "state(SystemReady)", produces = "state(PythonEnv)")]
    PythonInstall { id: NodeId, version: String },

    #[dsl_check(
        requires = "state(ComfyUIInstalled)",
        produces = "state(ServiceRunning($name))",
        bind(name = "name")
    )]
    ComfyUIService { id: NodeId, name: String },

    #[dsl_check(requires = "state(ServiceRunning($target))", bind(target = "target"))]
    Readiness {
        id: NodeId,
        target: String,
        port: u16,
    },
}

#[test]
fn the_sketch_compiles_into_field_wired_terms() {
    let program = Stage1::check_program();
    assert_eq!(program.rules.len(), 4, "rules = {:?}", program.rules);

    // `$name` reaches the transition as a reference to the payload
    // field, so the state the plan moves to carries the handle the
    // document itself supplied.
    let service = program
        .rules_for("ComfyUIService")
        .next()
        .expect("ComfyUIService rule");
    assert_eq!(
        service.state_after,
        Some(fact("state", [ctor("ServiceRunning", [field_ref("name")])]))
    );
    assert_eq!(
        service.premises,
        vec![Premise::State {
            expect: fact("state", [atom("ComfyUIInstalled")])
        }]
    );

    // And the probe reads its own field, so the two are compared as
    // values rather than by position.
    let readiness = program
        .rules_for("Readiness")
        .next()
        .expect("Readiness rule");
    assert_eq!(
        readiness.premises,
        vec![Premise::State {
            expect: fact("state", [ctor("ServiceRunning", [field_ref("target")])])
        }]
    );
    assert_eq!(readiness.state_after, None);
}

#[test]
fn the_sketch_names_its_own_gap_at_load_time() {
    // The sketch is an excerpt: nothing in it produces
    // `state(ComfyUIInstalled)`, which `ComfyUIService` requires. The
    // load-time self-validation is what says so — and it has to, since
    // a check diagnostic cannot be suppressed at the document level.
    let diags = Stage1::check_program().validate();
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].code, codes::CHECK_PROGRAM_UNDEFINED_STATE);
    assert!(
        diags[0].message.contains("state(ComfyUIInstalled)"),
        "message = {}",
        diags[0].message
    );
}

// ---------------------------------------------------------------------------
// The same vocabulary, completed and run
// ---------------------------------------------------------------------------

/// The sketch with its missing step filled in (`ComfyUIInstall`) and a
/// `Plan` container whose `steps` slot is the fold — the shape a
/// document actually arrives in.
#[allow(dead_code)]
#[derive(Debug, DslNode, DslSchema, DslCheck)]
enum Provision {
    Plan {
        id: NodeId,
        #[dsl_check(fold(initial = "state(Raw)"))]
        steps: Vec<Provision>,
    },

    #[dsl_check(requires = "state(Raw)", produces = "state(SystemReady)")]
    SystemPkg { id: NodeId, packages: Vec<String> },

    #[dsl_check(requires = "state(SystemReady)", produces = "state(PythonEnv)")]
    PythonInstall { id: NodeId, version: String },

    #[dsl_check(requires = "state(PythonEnv)", produces = "state(ComfyUIInstalled)")]
    ComfyUIInstall { id: NodeId, source: String },

    #[dsl_check(
        requires = "state(ComfyUIInstalled)",
        produces = "state(ServiceRunning($name))",
        bind(name = "name")
    )]
    ComfyUIService { id: NodeId, name: String },

    #[dsl_check(requires = "state(ServiceRunning($target))", bind(target = "target"))]
    Readiness {
        id: NodeId,
        target: String,
        port: u16,
    },
}

/// A step with one payload field, as a front-end hands it over.
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

/// `SystemPkg → PythonInstall → ComfyUIInstall → ComfyUIService`, with
/// the probe's `target` left to the caller.
fn plan_up_to_readiness(target: &str) -> ParseTree {
    plan(vec![
        step("SystemPkg", "packages", "build-essential"),
        step("PythonInstall", "version", "3.12"),
        step("ComfyUIInstall", "source", "git"),
        step("ComfyUIService", "name", "comfyui"),
        step("Readiness", "target", target),
    ])
}

#[test]
fn the_completed_program_passes_its_own_validation() {
    let diags = Provision::check_program().validate();
    assert!(diags.is_empty(), "diags = {diags:?}");
}

#[test]
fn a_matching_handle_passes() {
    let diags = check_semantics(
        &plan_up_to_readiness("comfyui"),
        &Provision::check_program(),
    );
    assert!(diags.is_empty(), "diags = {diags:?}");
}

/// Error example 1: a step whose prerequisite has not happened.
#[test]
fn a_step_out_of_order_is_reported_where_it_stands() {
    let doc = plan(vec![
        step("PythonInstall", "version", "3.12"),
        step("SystemPkg", "packages", "build-essential"),
    ]);

    let diags = check_semantics(&doc, &Provision::check_program());
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].code, codes::CHECK_STATE_MISMATCH);
    assert!(
        diags[0]
            .message
            .starts_with("`PythonInstall` requires state(SystemReady), found state(Raw)"),
        "message = {}",
        diags[0].message
    );
    assert!(
        diags[0].message.ends_with("[at steps[0]]"),
        "message = {}",
        diags[0].message
    );
}

/// Error example 2: the probe waits on a handle nothing started —
/// `comfy` where the plan launched `comfyui`.
#[test]
fn a_mistyped_handle_is_told_which_one_exists() {
    let program = Provision::check_program();
    let doc = plan_up_to_readiness("comfy");

    let diags = check_semantics_with(&doc, &program, &FuzzySuggester::default());
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].code, codes::CHECK_STATE_MISMATCH);
    // What was wanted, what is actually running, and where it came
    // from — the fourth step, which is where the handle was chosen.
    assert!(
        diags[0].message.starts_with(
            "`Readiness` requires state(ServiceRunning(comfy)), \
             found state(ServiceRunning(comfyui)) (from steps[3])"
        ),
        "message = {}",
        diags[0].message
    );
    // The handle exists nowhere in the program — it is a value the
    // document itself produced, and the suggestion comes from there.
    assert!(
        diags[0].message.contains("did you mean: comfyui"),
        "message = {}",
        diags[0].message
    );
    assert!(
        diags[0].message.ends_with("[at steps[4]]"),
        "message = {}",
        diags[0].message
    );

    // Without a suggester the wording is exactly the same, minus the
    // hint: enrichment never changes what is reported.
    let plain = check_semantics(&doc, &program);
    assert_eq!(plain.len(), 1);
    assert!(
        !plain[0].message.contains("did you mean"),
        "message = {}",
        plain[0].message
    );
    assert_eq!(
        plain[0].message,
        diags[0].message.replace(" (did you mean: comfyui)", "")
    );
}

// ---------------------------------------------------------------------------
// The tree-typing half
// ---------------------------------------------------------------------------

/// `IntLit : Int`, `BoolLit : Bool`, `Add : Int × Int → Int`,
/// `If : Bool × a × a → a` — the same judgements the hand-built
/// acceptance writes with `CheckProgram::builder()`, this time as
/// annotations.
#[allow(dead_code)]
#[derive(Debug, DslNode, DslSchema, DslCheck)]
enum Expr {
    #[dsl_check(concludes = "type(Int)")]
    IntLit { id: NodeId, value: i64 },

    #[dsl_check(concludes = "type(Bool)")]
    BoolLit { id: NodeId, value: bool },

    #[dsl_check(
        requires(lhs = "type(Int)", rhs = "type(Int)"),
        concludes = "type(Int)",
        message = "`add` operand {slot} must be {expected}, found {found} (from {provenance})"
    )]
    Add {
        id: NodeId,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },

    #[dsl_check(
        requires(
            cond = "type(Bool)",
            then_branch = "type($a)",
            else_branch = "type($a)"
        ),
        concludes = "type($a)",
        message = "`if` {slot} must be {expected}, found {found}"
    )]
    If {
        id: NodeId,
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },
}

fn node(variant: &str, slots: Vec<(&str, ParseTree)>) -> ParseTree {
    let mut tree = ParseTree::new(variant);
    tree.children = slots
        .into_iter()
        .map(|(name, child)| (name.to_string(), vec![child]))
        .collect();
    tree
}

#[test]
fn child_slots_and_conclusions_compile_into_a_typing_judgement() {
    let program = Expr::check_program();
    let add = program.rules_for("Add").next().expect("Add rule");
    assert_eq!(
        add.premises,
        vec![
            Premise::Child {
                slot: "lhs".into(),
                expect: fact("type", [atom("Int")]),
            },
            Premise::Child {
                slot: "rhs".into(),
                expect: fact("type", [atom("Int")]),
            },
        ]
    );
    assert_eq!(add.conclusion, Some(fact("type", [atom("Int")])));
    assert_eq!(add.state_after, None);
    // A rule that says nothing about state reports as a type failure
    // without the author having to spell the slug.
    assert_eq!(add.message.code, codes::CHECK_TYPE_MISMATCH);

    // An unbound `$a` stays a rule-local variable: it is what makes
    // the two branches have to agree with each other rather than with
    // a fixed type.
    let branch = program.rules_for("If").next().expect("If rule");
    assert_eq!(branch.conclusion, Some(fact("type", [var("a")])));
    assert!(program.validate().is_empty());
}

/// Error example 3: operands that do not agree.
#[test]
fn a_type_mismatch_is_reported_through_the_derived_rules() {
    let program = Expr::check_program();
    let tree = node(
        "Add",
        vec![
            ("lhs", ParseTree::new("IntLit")),
            ("rhs", ParseTree::new("BoolLit")),
        ],
    );

    let diags = check_semantics(&tree, &program);
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].code, codes::CHECK_TYPE_MISMATCH);
    assert!(
        diags[0]
            .message
            .starts_with("`add` operand rhs must be type(Int), found type(Bool)"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn branches_must_agree_with_each_other() {
    let program = Expr::check_program();
    let tree = node(
        "If",
        vec![
            ("cond", ParseTree::new("BoolLit")),
            ("then_branch", ParseTree::new("IntLit")),
            ("else_branch", ParseTree::new("BoolLit")),
        ],
    );

    let diags = check_semantics(&tree, &program);
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    // `$a` was pinned to Int by the first branch, so the second is
    // measured against it.
    assert!(
        diags[0]
            .message
            .starts_with("`if` else_branch must be type(Int), found type(Bool)"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn a_well_typed_document_is_silent() {
    let program = Expr::check_program();
    let tree = node(
        "Add",
        vec![
            (
                "lhs",
                node(
                    "If",
                    vec![
                        ("cond", ParseTree::new("BoolLit")),
                        ("then_branch", ParseTree::new("IntLit")),
                        ("else_branch", ParseTree::new("IntLit")),
                    ],
                ),
            ),
            ("rhs", ParseTree::new("IntLit")),
        ],
    );
    assert!(check_semantics(&tree, &program).is_empty());
}
