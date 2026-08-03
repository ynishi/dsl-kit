//! Acceptance 1 of the Check IR slice: a tree-typing vocabulary
//! (`If` / `Add` over `IntLit` / `BoolLit`) written as a hand-built
//! `CheckProgram` — no macros — detects a `type("Bool")` requirement
//! violation and a branch-type disagreement as
//! `CHECK_TYPE_MISMATCH`.
//!
//! The same fixtures pin the two recovery contracts the solver
//! promises: a variant with no rule stays silent (opt-in), and a node
//! whose rule failed contributes no conclusion, so one mistake yields
//! one diagnostic instead of a cascade.

use dsl_kit_check::{CheckProgram, Rule, atom, check_semantics, codes, fact, var};
use dsl_kit_parse::{Location, ParseTree};

/// `IntLit : Int`, `BoolLit : Bool`, `Add : Int × Int → Int`,
/// `If : Bool × a × a → a`.
fn type_system() -> CheckProgram {
    CheckProgram::builder()
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
            Rule::on("Add")
                .child("lhs", fact("type", [atom("Int")]))
                .child("rhs", fact("type", [atom("Int")]))
                .concludes(fact("type", [atom("Int")]))
                .message(
                    codes::CHECK_TYPE_MISMATCH,
                    "`add` operand {slot} must be {expected}, found {found} (from {provenance})",
                ),
        )
        .rule(
            Rule::on("If")
                .child("cond", fact("type", [atom("Bool")]))
                .child("then_branch", fact("type", [var("a")]))
                .child("else_branch", fact("type", [var("a")]))
                .concludes(fact("type", [var("a")]))
                .message(
                    codes::CHECK_TYPE_MISMATCH,
                    "`if` {slot} must be {expected}, found {found} (from {provenance})",
                ),
        )
        .build()
}

fn leaf(variant: &str) -> ParseTree {
    ParseTree::new(variant)
}

/// Builds a node whose every child slot holds exactly one subtree.
fn node(variant: &str, slots: Vec<(&str, ParseTree)>) -> ParseTree {
    let mut tree = ParseTree::new(variant);
    tree.children = slots
        .into_iter()
        .map(|(name, child)| (name.to_string(), vec![child]))
        .collect();
    tree
}

fn if_node(cond: ParseTree, then_branch: ParseTree, else_branch: ParseTree) -> ParseTree {
    node(
        "If",
        vec![
            ("cond", cond),
            ("then_branch", then_branch),
            ("else_branch", else_branch),
        ],
    )
}

#[test]
fn well_typed_document_is_silent() {
    let program = type_system();

    let plain = if_node(leaf("BoolLit"), leaf("IntLit"), leaf("IntLit"));
    assert!(check_semantics(&plain, &program).is_empty());

    // The `If` conclusion is the grounded `$a` (= Int), so it
    // satisfies `Add`'s operand requirement.
    let nested = node(
        "Add",
        vec![
            (
                "lhs",
                if_node(leaf("BoolLit"), leaf("IntLit"), leaf("IntLit")),
            ),
            ("rhs", leaf("IntLit")),
        ],
    );
    assert!(check_semantics(&nested, &program).is_empty());
}

#[test]
fn condition_must_be_bool() {
    let program = type_system();
    // Nested one level so the diagnostic has a non-root path to name.
    let tree = node(
        "Add",
        vec![
            ("lhs", leaf("IntLit")),
            (
                "rhs",
                if_node(leaf("IntLit"), leaf("IntLit"), leaf("IntLit")),
            ),
        ],
    );

    let diags = check_semantics(&tree, &program);
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].code, codes::CHECK_TYPE_MISMATCH);
    assert!(
        diags[0]
            .message
            .starts_with("`if` cond must be type(Bool), found type(Int)"),
        "message = {}",
        diags[0].message
    );
    // The failing node is the `If`; the offending fact came from its
    // `cond` child.
    assert!(
        diags[0].message.contains("(from rhs[0].cond[0])"),
        "message = {}",
        diags[0].message
    );
    assert!(
        diags[0].message.ends_with("[at rhs[0]]"),
        "message = {}",
        diags[0].message
    );
    // Hand-built trees carry no span, so the anchor lives in the text.
    assert_eq!(diags[0].location, Location::None);
}

#[test]
fn branches_must_agree() {
    let program = type_system();
    let tree = if_node(leaf("BoolLit"), leaf("IntLit"), leaf("BoolLit"));

    let diags = check_semantics(&tree, &program);
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert_eq!(diags[0].code, codes::CHECK_TYPE_MISMATCH);
    // `$a` was bound to Int by `then_branch`, so `else_branch` is
    // measured against the branch that came first.
    assert!(
        diags[0]
            .message
            .starts_with("`if` else_branch must be type(Int), found type(Bool)"),
        "message = {}",
        diags[0].message
    );
    assert!(
        diags[0].message.contains("(from else_branch[0])"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn add_operands_must_be_int() {
    let program = type_system();
    let tree = node(
        "Add",
        vec![("lhs", leaf("IntLit")), ("rhs", leaf("BoolLit"))],
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
    // The root of the document has no path segment of its own.
    assert!(
        diags[0].message.ends_with("[at (root)]"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn un_annotated_variants_stay_out_of_the_way() {
    let program = type_system();
    // `Unknown` has no rule: it neither passes nor fails, it simply
    // contributes nothing.
    let tree = node(
        "Add",
        vec![("lhs", leaf("IntLit")), ("rhs", leaf("Unknown"))],
    );
    assert!(check_semantics(&tree, &program).is_empty());
}

#[test]
fn a_failed_node_does_not_cascade() {
    let program = type_system();
    // The inner `If` is ill-typed; the enclosing `Add` must not add a
    // second complaint about a conclusion the `If` never produced.
    let tree = node(
        "Add",
        vec![
            ("lhs", leaf("IntLit")),
            (
                "rhs",
                if_node(leaf("BoolLit"), leaf("IntLit"), leaf("BoolLit")),
            ),
        ],
    );

    let diags = check_semantics(&tree, &program);
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert!(
        diags[0].message.starts_with("`if` else_branch"),
        "message = {}",
        diags[0].message
    );
}

#[test]
fn a_well_typed_branch_of_the_wrong_type_is_still_reported() {
    let program = type_system();
    // The `If` itself is fine (both branches Bool) — it is `Add` that
    // cannot accept the Bool it synthesised.
    let tree = node(
        "Add",
        vec![
            (
                "lhs",
                if_node(leaf("BoolLit"), leaf("BoolLit"), leaf("BoolLit")),
            ),
            ("rhs", leaf("IntLit")),
        ],
    );

    let diags = check_semantics(&tree, &program);
    assert_eq!(diags.len(), 1, "diags = {diags:?}");
    assert!(
        diags[0]
            .message
            .starts_with("`add` operand lhs must be type(Int), found type(Bool)"),
        "message = {}",
        diags[0].message
    );
    assert!(
        diags[0].message.contains("(from lhs[0])"),
        "message = {}",
        diags[0].message
    );
}
