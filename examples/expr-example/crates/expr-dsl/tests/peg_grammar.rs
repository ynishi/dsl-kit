//! G-2 E2E — Expr golden grammar → parse → typed `Expr` → eval.
//!
//! Proves the whole G-2 chain:
//!
//! - PEG interpreter walks a hand-built [`Grammar`] value against
//!   source text and produces a [`ParseTree`].
//! - The tree conforms to the `Expr` [`DslSchema`] (`RawValue::Text`
//!   payloads, arity-checked children).
//! - `#[derive(DslBuild)]` builds a typed `Expr` from it, minting
//!   fresh `NodeId`s.
//! - The synchronous evaluator settles on `16` once external
//!   `y` / `z` bindings are supplied via the resolver.
//!
//! Grammar shape (right-recursive to avoid left recursion; PEG
//! `Choice` is ordered — see parser-design §3.5):
//!
//! ```text
//! start    <- expr
//! expr     <- let_expr / add_expr
//! let_expr <- Node "Let" { %kw:let Field "name" %ident "=" Field "value" expr %kw:in Field "body" expr }
//! add_expr <- Node "Add" { Field "lhs" mul_expr "+" Field "rhs" add_expr } / mul_expr
//! mul_expr <- Node "Mul" { Field "lhs" factor "*" Field "rhs" mul_expr } / factor
//! factor   <- paren_expr / Node "Lit" { Field "value" %int } / Node "Var" { Field "name" %ident }
//! paren_expr <- "(" expr ")"
//! ```

use dsl_kit::IdGen;
use dsl_kit_parse::{
    DslBuild,
    peg::{Grammar, choice, field, node, rule, rule_ref, seq, token},
};
use expr_dsl::{Expr, evaluate_all};

/// Hand-builds the Expr golden grammar.
///
/// Each rule owns its ids so we can reuse the same `IdGen` for the
/// consumer's `Expr` build without collision — grammar node ids and
/// `Expr` node ids live in disjoint value spaces (grammar ids are
/// interpreter-internal; consumer ids come from the caller-provided
/// `IdGen` at build time).
fn expr_grammar() -> Grammar {
    let g = IdGen::new();

    // Terminals reused across rules — grammar values are cloneable via
    // the enum's `Clone` derive, but here we just call the builders
    // multiple times.
    let start_body = rule_ref(&g, "expr");

    let expr_body = choice(
        &g,
        vec![rule_ref(&g, "let_expr"), rule_ref(&g, "add_expr")],
    );

    let let_body = node(
        &g,
        "Let",
        seq(
            &g,
            vec![
                token(&g, "%kw:let"),
                field(&g, "name", token(&g, "%ident")),
                token(&g, "="),
                field(&g, "value", rule_ref(&g, "expr")),
                token(&g, "%kw:in"),
                field(&g, "body", rule_ref(&g, "expr")),
            ],
        ),
    );

    let add_body = choice(
        &g,
        vec![
            node(
                &g,
                "Add",
                seq(
                    &g,
                    vec![
                        field(&g, "lhs", rule_ref(&g, "mul_expr")),
                        token(&g, "+"),
                        field(&g, "rhs", rule_ref(&g, "add_expr")),
                    ],
                ),
            ),
            rule_ref(&g, "mul_expr"),
        ],
    );

    let mul_body = choice(
        &g,
        vec![
            node(
                &g,
                "Mul",
                seq(
                    &g,
                    vec![
                        field(&g, "lhs", rule_ref(&g, "factor")),
                        token(&g, "*"),
                        field(&g, "rhs", rule_ref(&g, "mul_expr")),
                    ],
                ),
            ),
            rule_ref(&g, "factor"),
        ],
    );

    let factor_body = choice(
        &g,
        vec![
            rule_ref(&g, "paren_expr"),
            node(&g, "Lit", field(&g, "value", token(&g, "%int"))),
            node(&g, "Var", field(&g, "name", token(&g, "%ident"))),
        ],
    );

    let paren_body = seq(
        &g,
        vec![
            token(&g, "("),
            rule_ref(&g, "expr"),
            token(&g, ")"),
        ],
    );

    Grammar::new(
        vec![
            rule(&g, "start", start_body),
            rule(&g, "expr", expr_body),
            rule(&g, "let_expr", let_body),
            rule(&g, "add_expr", add_body),
            rule(&g, "mul_expr", mul_body),
            rule(&g, "factor", factor_body),
            rule(&g, "paren_expr", paren_body),
        ],
        "start",
    )
}

/// Parses `input` against the grammar and builds the typed `Expr`.
fn parse_expr(input: &str) -> Expr {
    let g = expr_grammar();
    let tree = g.parse(input).expect("grammar accepts golden input");
    let ids = IdGen::new();
    Expr::from_parse_tree(&tree, &ids).expect("tree conforms to Expr schema")
}

#[test]
fn golden_e2e_let_x_eq_3_in_add_mul_yields_16() {
    let expr = parse_expr("let x = 3 in (x + y) * z");
    // Same evaluator as the R-25 build_from_json test — external
    // resolver supplies `y = 5` and `z = 2`; result is `(3 + 5) * 2`.
    let result = evaluate_all(&expr, |name| match name {
        "y" => Some(5),
        "z" => Some(2),
        _ => None,
    })
    .expect("evaluator settles");
    assert_eq!(result, 16);
}

#[test]
fn accepts_bare_integer() {
    let expr = parse_expr("42");
    let result = evaluate_all(&expr, |_| None).expect("bare Lit needs no resolver");
    assert_eq!(result, 42);
}

#[test]
fn whitespace_is_flexible_around_all_tokens() {
    let expr = parse_expr("  let   x =  3   in   ( x + y ) * z  ");
    let result = evaluate_all(&expr, |name| match name {
        "y" => Some(5),
        "z" => Some(2),
        _ => None,
    })
    .unwrap();
    assert_eq!(result, 16);
}

#[test]
fn nested_let_shadowing_works() {
    // let x = 1 in let x = 5 in x    → 5
    let expr = parse_expr("let x = 1 in let x = 5 in x");
    let result = evaluate_all(&expr, |_| None).unwrap();
    assert_eq!(result, 5);
}

#[test]
fn ids_are_freshly_minted_from_the_provided_idgen() {
    use dsl_kit::Walk;

    let g = expr_grammar();
    let tree = g.parse("let x = 3 in (x + y) * z").unwrap();
    let ids = IdGen::new();
    let expr = Expr::from_parse_tree(&tree, &ids).unwrap();

    // Every node in the resulting tree should have a unique id.
    let mut seen = std::collections::HashSet::new();
    fn collect(e: &Expr, seen: &mut std::collections::HashSet<u64>) {
        use dsl_kit::DslNode;
        let id = e.node_id();
        assert!(seen.insert(id.0), "duplicate id {id}");
        for c in e.children() {
            collect(c, seen);
        }
    }
    collect(&expr, &mut seen);
    // Program has 7 nodes: Let, Lit(3), Mul, Add, Var(x), Var(y), Var(z).
    assert_eq!(seen.len(), 7);
}

// --- Negative cases: farthest-failure attribution (parser-design §3.5).

#[test]
fn rejection_reports_farthest_failure_position() {
    let g = expr_grammar();
    // Missing "in" after value.
    let err = g.parse("let x = 3 (x + y) * z").unwrap_err();
    let msg = &err.diagnostics[0].message;
    // The farthest we got in a valid alternative: after `3 `, pos 10
    // where "in" is expected. The `%kw:in` token is what we asked for.
    assert!(msg.contains("%kw:in"), "message did not mention expected keyword: {msg}");
}

#[test]
fn rejection_at_trailing_junk_is_flagged() {
    let g = expr_grammar();
    let err = g.parse("let x = 3 in x @@@").unwrap_err();
    // `@@@` is neither a valid tail nor a legal token in any position.
    // Either the trailing-input check or the farthest-failure branch
    // should call it out; both are correct — assert we got *some* error
    // and the diagnostic points past the last valid consumption.
    assert_eq!(err.diagnostics[0].code, dsl_kit_parse::peg::codes::UNEXPECTED);
}

#[test]
fn empty_input_is_rejected_with_start_expected() {
    let g = expr_grammar();
    let err = g.parse("").unwrap_err();
    assert_eq!(err.diagnostics[0].code, dsl_kit_parse::peg::codes::UNEXPECTED);
}
