//! End-to-end proof of the two per-field hooks on the DSL that
//! motivated them — grammar layer and build layer as one contract:
//!
//! - `Flow`'s `Par` carries a `policy: Option<JoinPolicy>` field whose
//!   Rust type has no built-in canonical-syntax mapping, so plain
//!   generation fails loudly; [`flow_syntax_overrides`] (the crate's
//!   `schema_gen::SyntaxOverrides`) supplies the value production and
//!   makes the derived schema generate a check-clean grammar.
//! - `Option<JoinPolicy>` also has no `FromStr` route (orphan rule),
//!   so `#[derive(DslBuild)]` maps it through
//!   `#[dsl_build(with = parse_policy)]`.
//! - `reducer_id: Option<String>` is covered by dsl-kit's built-in
//!   `Option<String>` mapping (grammar + `build_field_optional`), so
//!   no override and no `#[dsl_build(with = ...)]` are needed on this
//!   crate's side.
//!
//! Together: canonical text → generated grammar → `ParseTree` → typed
//! `Flow` → `Engine` + `drive` → joined value. The author wrote derives,
//! one value production, and one converter function — no grammar and
//! no builder by hand.

use dsl_kit::{
    DriveOutcome, EffectResolver, Engine, FailPolicy, IdGen, JoinPolicy, JoinShape, Pending,
    SuspendReason, drive,
};
use dsl_kit_parse::{DslBuild, RawValue, check_conformance, schema_gen};
use dsl_kit_schema::DslSchema;
use flow_dsl::{
    Flow, FlowAst, FlowEffectErr, FlowValue, flow_default_registry, flow_syntax_overrides,
};
use std::sync::Arc;

const CANONICAL_TEXT: &str = r#"
    Seq(children: [
        Call(label: "fetch"),
        Scope(label: "wrap", body: Maybe(body: none)),
        Par(policy: all_failfast, reducer_id: "reduce_all_ordered",
            children: [Call(label: "a"), Call(label: "b")])
    ])
"#;

fn generated_grammar() -> dsl_kit_parse::peg::Grammar {
    schema_gen::checked_grammar_from_schema_with(
        &Flow::schema(),
        &IdGen::new(),
        &flow_syntax_overrides(),
    )
    .expect("overridden Flow schema generates a clean grammar")
}

#[test]
fn flow_schema_fails_generation_without_overrides() {
    // `policy: Option<JoinPolicy>` has no built-in canonical-syntax
    // mapping — that's what `flow_syntax_overrides` supplies.
    // `reducer_id: Option<String>` used to also reject, but is now
    // covered by dsl-kit's built-in `Option<String>` mapping, so only
    // `policy` remains unmapped when overrides are absent.
    let err = schema_gen::grammar_from_schema(&Flow::schema(), &IdGen::new())
        .expect_err("Par's `policy` field has no built-in mapping");
    assert_eq!(err.diagnostics.len(), 1, "only policy is rejected");
    assert!(
        err.diagnostics
            .iter()
            .all(|d| d.code == schema_gen::codes::UNSUPPORTED_FIELD)
    );
    assert!(
        err.diagnostics[0].message.contains("policy"),
        "unmapped-field diagnostic points at `policy`",
    );
}

#[test]
fn canonical_flow_text_parses_and_conforms() {
    let tree = generated_grammar()
        .parse(CANONICAL_TEXT)
        .expect("canonical Flow text parses");
    assert!(check_conformance(&tree, &Flow::schema()).is_empty());

    assert_eq!(tree.variant, "Seq");
    let kids = tree.child_slot("children").expect("Seq children bound");
    assert_eq!(kids.len(), 3);
    let par = &kids[2];
    assert_eq!(
        par.field("policy"),
        Some(&RawValue::Text("all_failfast".into()))
    );
    assert_eq!(
        par.field("reducer_id"),
        Some(&RawValue::Text("reduce_all_ordered".into()))
    );
    assert_eq!(par.child_slot("children").unwrap().len(), 2);
}

#[test]
fn text_builds_a_typed_flow_via_dsl_build_converters() {
    let tree = generated_grammar().parse(CANONICAL_TEXT).expect("parses");
    let flow = Flow::from_parse_tree(&tree, &IdGen::new()).expect("typed build succeeds");

    let Flow::Seq { children, .. } = &flow else {
        panic!("root is Seq, got {}", flow.summary());
    };
    let Flow::Par {
        policy, reducer_id, ..
    } = &children[2]
    else {
        panic!("third child is Par, got {}", children[2].summary());
    };
    assert!(
        matches!(
            policy,
            Some(JoinPolicy {
                shape: JoinShape::All,
                fail: FailPolicy::FailFast
            })
        ),
        "parse_policy decoded all_failfast: {policy:?}"
    );
    assert_eq!(reducer_id.as_deref(), Some("reduce_all_ordered"));
}

/// Layer 1 + Layer 2 (JSON path): a `Par` document that omits the
/// optional `policy` / `reducer_id` keys entirely — the AI-emit
/// contract — builds to a typed `Flow::Par` with `None` / `None`.
///
/// This is the acceptance criterion the issue calls out first
/// ("variant with `version: Option<String>` builds from JSON that omits
/// the key entirely"), realised end-to-end against the real flow DSL.
#[test]
fn json_par_omitting_optional_keys_builds_to_none_defaults() {
    let doc = serde_json::json!({
        "type": "Par",
        "children": [
            { "type": "Call", "label": "x" },
        ],
        // NB: no `policy`, no `reducer_id`.
    });
    let tree = dsl_kit_parse::serde_bridge::from_json_value(&doc, &Flow::schema())
        .expect("JSON without optional keys parses");
    let flow = Flow::from_parse_tree(&tree, &IdGen::new())
        .expect("typed build succeeds with the optional keys omitted");
    let Flow::Par {
        policy,
        reducer_id,
        children,
        ..
    } = &flow
    else {
        panic!("root is Par");
    };
    assert!(policy.is_none(), "absent policy → None");
    assert!(reducer_id.is_none(), "absent reducer_id → None");
    assert_eq!(children.len(), 1, "one child preserved");
}

#[test]
fn none_spellings_build_to_typed_defaults() {
    let tree = generated_grammar()
        .parse("Par(policy: none, reducer_id: none, children: [Call(label: \"x\")])")
        .expect("parses");
    let flow = Flow::from_parse_tree(&tree, &IdGen::new()).expect("typed build succeeds");
    let Flow::Par {
        policy, reducer_id, ..
    } = &flow
    else {
        panic!("root is Par");
    };
    assert!(policy.is_none() && reducer_id.is_none());
}

#[test]
fn synthesized_examples_cover_every_variant_and_build_typed() {
    // Q-1 on the override case: examples are walked out of the
    // *generated grammar*, so the override spellings (`none` /
    // `all_failfast` / `%str`) surface with no extra registration.
    let grammar = generated_grammar();
    let examples =
        dsl_kit_parse::example_gen::examples_from_grammar(&grammar).expect("examples synthesize");
    assert_eq!(examples.per_rule.len(), 6, "one example per Flow variant");
    for e in &examples.per_rule {
        let tree = grammar.parse(&e.text).unwrap_or_else(|err| {
            panic!(
                "example for `{}` failed to parse: {:?}\n  text: {}",
                e.rule, err.diagnostics, e.text
            )
        });
        assert_eq!(tree.variant, e.rule);
        // Every synthesized example builds a typed Flow through the
        // #[dsl_build(with)] converters as well.
        Flow::from_parse_tree(&tree, &IdGen::new()).unwrap_or_else(|err| {
            panic!(
                "example for `{}` failed typed build: {:?}",
                e.rule, err.diagnostics
            )
        });
    }
    // Every `Par` field is optional at grammar level (policy /
    // reducer_id are Option, children is Vec), so `example_gen` picks
    // the minimal-args form `Par()`. This still builds a fully typed
    // `Flow::Par { children: vec![], policy: None, reducer_id: None }`
    // through the derive.
    let par = examples.per_rule.iter().find(|e| e.rule == "Par").unwrap();
    assert_eq!(
        par.text, "Par()",
        "minimal-args form falls out of the grammar walk when every arg is optional",
    );
    let composite_tree = grammar
        .parse(&examples.composite)
        .expect("composite parses");
    Flow::from_parse_tree(&composite_tree, &IdGen::new()).expect("composite builds typed");
}

/// Answers each `Call` with `done:<label>`.
struct EchoResolver;

impl EffectResolver<FlowAst> for EchoResolver {
    fn resolve(&mut self, pending: &Pending) -> Result<FlowValue, FlowEffectErr> {
        let SuspendReason::Call { spec } = &pending.reason else {
            unreachable!("drive only hands Call suspensions to the resolver");
        };
        Ok(FlowValue::Text(format!("done:{}", spec.label)))
    }
}

#[test]
fn text_round_trips_through_the_engine_to_a_joined_value() {
    let tree = generated_grammar().parse(CANONICAL_TEXT).expect("parses");
    let flow = Flow::from_parse_tree(&tree, &IdGen::new()).expect("typed build succeeds");

    let mut engine = Engine::new(FlowAst::new(&flow), Arc::new(flow_default_registry()))
        .expect("root Ast validation");
    let outcome = drive(
        &mut engine,
        &mut EchoResolver,
        &dsl_kit::BreakpointSet::new(),
    )
    .expect("drive completes");
    // The Seq's value is its last child's — the Par folded by the
    // reducer the *text* named.
    let DriveOutcome::Done(value) = outcome else {
        panic!("expected Done, got {outcome:?}");
    };
    assert_eq!(
        value,
        FlowValue::List(vec![
            FlowValue::Text("done:a".into()),
            FlowValue::Text("done:b".into()),
        ])
    );
}
