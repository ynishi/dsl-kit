//! End-to-end proof of `#[derive(DslBuild)]` on the `Expr` DSL: JSON
//! document → `ParseTree` (serde bridge) → typed `Expr` value with
//! fresh `NodeId`s → evaluation to a concrete number.
//!
//! This is the G-1c landing test. The G-1d smoke reaches through the
//! host + MCP surface; this test proves the derive itself in isolation.

use dsl_kit::IdGen;
use dsl_kit_parse::{DslBuild, serde_bridge::from_json_value};
use dsl_kit_schema::DslSchema;
use expr_dsl::{Expr, evaluate_all};
use serde_json::json;

/// Same program the flow / expr examples use, in JSON form:
/// `let x = 3 in (x + y) * z` — the evaluator should suspend twice
/// (for `y` and `z`) and settle on `(3 + 5) * 2 == 16` once the host
/// resolves them.
fn program_json() -> serde_json::Value {
    json!({
        "type": "Let",
        "name": "x",
        "value": { "type": "Lit", "value": 3 },
        "body": {
            "type": "Mul",
            "lhs": {
                "type": "Add",
                "lhs": { "type": "Var", "name": "x" },
                "rhs": { "type": "Var", "name": "y" },
            },
            "rhs": { "type": "Var", "name": "z" },
        },
    })
}

#[test]
fn json_round_trip_builds_expr_with_fresh_ids() {
    let tree = from_json_value(&program_json(), &Expr::schema()).unwrap();
    let ids = IdGen::new();
    let expr = Expr::from_parse_tree(&tree, &ids).unwrap();

    // Shape check: root is Let, one level down is Mul, then Add / Var.
    let Expr::Let { body, .. } = &expr else {
        panic!("expected top-level Let, got {expr:?}");
    };
    assert!(matches!(**body, Expr::Mul { .. }));
}

#[test]
fn built_expr_evaluates_to_sixteen() {
    let tree = from_json_value(&program_json(), &Expr::schema()).unwrap();
    let ids = IdGen::new();
    let expr = Expr::from_parse_tree(&tree, &ids).unwrap();

    // Host substitutions: y = 5, z = 2 (x = 3 comes from the Let).
    let result = evaluate_all(&expr, |name| match name {
        "y" => Some(5),
        "z" => Some(2),
        _ => None,
    })
    .expect("evaluation should succeed");
    assert_eq!(result, 16, "(3 + 5) * 2");
}

#[test]
fn missing_field_surfaces_at_build_time() {
    // Bridge accepts structurally, DslBuild's conformance check
    // rejects with MISSING_FIELD.
    let value = json!({ "type": "Lit" });
    let tree = from_json_value(&value, &Expr::schema()).unwrap();
    let ids = IdGen::new();
    let err = Expr::from_parse_tree(&tree, &ids).unwrap_err();
    assert!(
        err.diagnostics.iter().any(|d| d.message.contains("value")),
        "expected a diagnostic mentioning `value`, got: {err}"
    );
}

#[test]
fn unknown_variant_lists_candidates() {
    let value = json!({ "type": "Ad" }); // typo of Add
    let err = from_json_value(&value, &Expr::schema()).unwrap_err();
    assert!(err.diagnostics.iter().any(|d| d.message.contains("Add")));
}

#[test]
fn fresh_ids_are_unique() {
    let tree = from_json_value(&program_json(), &Expr::schema()).unwrap();
    let ids = IdGen::new();
    let expr = Expr::from_parse_tree(&tree, &ids).unwrap();

    let mut seen = std::collections::HashSet::new();
    fn collect(e: &Expr, seen: &mut std::collections::HashSet<u64>) {
        use dsl_kit::{DslNode, Walk};
        assert!(
            seen.insert(e.node_id().0),
            "duplicate NodeId {} for {:?}",
            e.node_id(),
            e.summary()
        );
        for c in e.children() {
            collect(c, seen);
        }
    }
    collect(&expr, &mut seen);
    // Program has 9 nodes: Let, Lit(3), Mul, Add, Var(x), Var(y), Var(z), plus expected boxing.
    assert!(
        seen.len() >= 7,
        "expected at least 7 unique ids, got {}",
        seen.len()
    );
}
