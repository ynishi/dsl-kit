//! End-to-end proof of item N on a real derived DSL: the grammar for
//! `Expr`'s canonical text syntax is **generated from the schema** the
//! `#[derive(DslSchema)]` already extracts — no hand-authored grammar,
//! no grammar JSON. Text goes generated-grammar → `ParseTree` →
//! `#[derive(DslBuild)]` → typed `Expr` → `evaluate_all`.

use dsl_kit::IdGen;
use dsl_kit_parse::peg::Grammar;
use dsl_kit_parse::{DslBuild, check_conformance};
use dsl_kit_schema::DslSchema;
use expr_dsl::{Expr, evaluate_all};

fn generated_grammar() -> Grammar {
    dsl_kit_parse::schema_gen::checked_grammar_from_schema(&Expr::schema(), &IdGen::new())
        .expect("Expr schema generates a clean grammar")
}

#[test]
fn expr_schema_generates_a_check_clean_grammar() {
    generated_grammar();
}

#[test]
fn canonical_text_round_trips_to_a_typed_ast_and_evaluates() {
    // (let x = 3 in (x + 2) * y) with y = 10 → 50
    let text = r#"
        Let(
            name: "x",
            value: Lit(value: 3),
            body: Mul(
                lhs: Add(lhs: Var(name: "x"), rhs: Lit(value: 2)),
                rhs: Var(name: "y")
            )
        )
    "#;
    let grammar = generated_grammar();
    let tree = grammar.parse(text).expect("canonical text parses");
    assert!(check_conformance(&tree, &Expr::schema()).is_empty());

    let ids = IdGen::new();
    let expr = Expr::from_parse_tree(&tree, &ids).expect("typed build succeeds");
    let value = evaluate_all(&expr, |name| match name {
        "y" => Some(10),
        _ => None,
    })
    .expect("evaluation completes");
    assert_eq!(value, 50);
}

#[test]
fn parse_error_carries_expected_set_diagnostic() {
    let grammar = generated_grammar();
    let err = grammar.parse("Add(lhs: Lit(value: 1) rhs: Lit(value: 2))").unwrap_err();
    assert!(
        !err.diagnostics.is_empty(),
        "missing comma surfaces as a diagnostic"
    );
}
