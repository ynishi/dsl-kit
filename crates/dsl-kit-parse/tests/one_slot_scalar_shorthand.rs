//! End-to-end coverage for declared scalar shorthands on `One` /
//! `Optional` child slots — the "promote a `String` payload field to a
//! union child slot without breaking existing documents" migration
//! path. Verifies that:
//!
//! - `#[derive(DslSchema)]` records `#[dsl_schema(scalar(...))]`
//!   declarations as `ChildSchema::scalar_shorthands`, so every
//!   consumer lowers from the same declaration;
//! - the JSON bridge lowers a bare scalar in a declared slot to the
//!   *same* `ParseTree` as the explicit node spelling — the shorthand
//!   is an input-side projection, invisible below the front-end;
//! - a slot with *no* declaration keeps today's `CHILD_SHAPE`
//!   diagnostic verbatim (the feature is opt-in and additive);
//! - `Optional` slots stay consistent: `null` → absent, scalar →
//!   coerced, object → built;
//! - the generated canonical-text grammar accepts both spellings and
//!   lands them on the same tree, so the two front-ends cannot drift;
//! - `to_canonical_json` expands the shorthand back to the long form,
//!   making canonical output (and therefore content hashes computed
//!   over it) spelling-independent;
//! - `grammar_from_schema`'s pre-flight rejects incoherent
//!   hand-written declarations loudly.

use dsl_kit_core::{IdGen, NodeId};
use dsl_kit_macros::{DslBuild, DslNode, DslSchema};
use dsl_kit_parse::{
    DslBuild, ParseTree, check_conformance,
    schema_gen::{self, checked_grammar_from_schema, grammar_from_schema},
    serde_bridge::{from_json_value, serde_codes, to_canonical_json},
};
use dsl_kit_schema::{
    ChildSchema, DslSchema, FieldSchema, Multiplicity, NodeSchema, ScalarKind, ScalarShorthand,
    VariantSchema,
};
use serde_json::json;

/// Config-shaped AST exercising every declared kind plus an
/// undeclared slot for contrast.
#[derive(Debug, PartialEq, DslNode, DslSchema, DslBuild)]
enum FsCfg {
    /// File write whose content is a union (`Literal` / `SecretRef`)
    /// but historically was a plain string — the motivating case.
    FsWrite {
        /// Stable node id.
        id: NodeId,
        /// Destination path.
        path: String,
        /// Content union. Bare strings lower to `Literal`.
        #[dsl_schema(scalar(string = Literal::value))]
        content: Box<FsCfg>,
    },
    /// A `One` slot with *no* declaration — must keep today's
    /// `CHILD_SHAPE` rejection verbatim.
    Touch {
        /// Stable node id.
        id: NodeId,
        /// Undeclared child slot.
        target: Box<FsCfg>,
    },
    /// An `Optional` slot with a declaration — the `null` / scalar /
    /// object triad must stay consistent.
    Fallback {
        /// Stable node id.
        id: NodeId,
        /// Optional content union.
        #[dsl_schema(scalar(string = Literal::value))]
        content: Option<Box<FsCfg>>,
    },
    /// A slot declaring two kinds at once — kind dispatch must be
    /// unambiguous without declaration order mattering.
    Knob {
        /// Stable node id.
        id: NodeId,
        /// Int lowers to `Count`, bool lowers to `Flag`.
        #[dsl_schema(scalar(int = Count::n, bool = Flag::on))]
        value: Box<FsCfg>,
    },
    /// Literal string payload — the string shorthand's target.
    Literal {
        /// Stable node id.
        id: NodeId,
        /// The literal content.
        value: String,
    },
    /// Secret reference — the union's other member, always explicit.
    SecretRef {
        /// Stable node id.
        id: NodeId,
        /// Secret name.
        name: String,
    },
    /// Integer payload — the int shorthand's target.
    Count {
        /// Stable node id.
        id: NodeId,
        /// The count.
        n: i64,
    },
    /// Boolean payload — the bool shorthand's target.
    Flag {
        /// Stable node id.
        id: NodeId,
        /// The flag.
        on: bool,
    },
}

fn tree_of(doc: serde_json::Value) -> ParseTree {
    let schema = FsCfg::schema();
    let tree = from_json_value(&doc, &schema)
        .unwrap_or_else(|e| panic!("parse failed for {doc}: {:?}", e.diagnostics));
    let diags = check_conformance(&tree, &schema);
    assert!(diags.is_empty(), "conformance clean for {doc}: {diags:?}");
    tree
}

/// Strips source spans recursively. The two text spellings of one
/// meaning occupy different byte ranges by construction, so
/// cross-spelling tree equality is *modulo spans* — keeping the
/// shorthand's own span on the lowered node is deliberate (diagnostics
/// should point at what the author wrote).
fn strip_spans(mut tree: ParseTree) -> ParseTree {
    tree.span = None;
    for (_, subtrees) in &mut tree.children {
        for sub in std::mem::take(subtrees) {
            subtrees.push(strip_spans(sub));
        }
    }
    for (_, entries) in &mut tree.keyed_children {
        for (key, sub) in std::mem::take(entries) {
            entries.push((key, strip_spans(sub)));
        }
    }
    tree
}

fn text_tree_of(input: &str) -> ParseTree {
    let schema = FsCfg::schema();
    let grammar = checked_grammar_from_schema(&schema, &IdGen::new())
        .expect("shorthand schema generates a clean grammar");
    let tree = grammar
        .parse(input)
        .unwrap_or_else(|e| panic!("parse failed for {input:?}: {:?}", e.diagnostics));
    let diags = check_conformance(&tree, &schema);
    assert!(
        diags.is_empty(),
        "conformance clean for {input:?}: {diags:?}"
    );
    tree
}

/// The derive records each declaration on the declaring slot — and
/// only there — so consumers read the mapping from one place.
#[test]
fn schema_records_declared_shorthands() {
    let schema = FsCfg::schema();

    let content = &schema.variant("FsWrite").unwrap().children[0];
    assert_eq!(
        content.scalar_shorthands,
        vec![ScalarShorthand {
            kind: ScalarKind::Str,
            variant: "Literal".into(),
            field: "value".into(),
        }]
    );

    let knob = &schema.variant("Knob").unwrap().children[0];
    assert_eq!(knob.scalar_shorthands.len(), 2);
    assert_eq!(
        knob.scalar_shorthand(ScalarKind::Int),
        Some(&ScalarShorthand {
            kind: ScalarKind::Int,
            variant: "Count".into(),
            field: "n".into(),
        })
    );
    assert_eq!(
        knob.scalar_shorthand(ScalarKind::Bool),
        Some(&ScalarShorthand {
            kind: ScalarKind::Bool,
            variant: "Flag".into(),
            field: "on".into(),
        })
    );

    let undeclared = &schema.variant("Touch").unwrap().children[0];
    assert!(undeclared.scalar_shorthands.is_empty());
}

/// The wire format gains a `scalar_shorthands` array on declaring
/// slots and keeps the historical layout on every other slot.
#[test]
fn schema_json_carries_shorthands_only_where_declared() {
    let schema = FsCfg::schema();
    let json = schema.to_json();
    let variants = json["variants"].as_array().unwrap();
    let fs_write = variants.iter().find(|v| v["name"] == "FsWrite").unwrap();
    assert_eq!(
        fs_write["children"][0]["scalar_shorthands"],
        json!([{ "kind": "string", "variant": "Literal", "field": "value" }])
    );
    let touch = variants.iter().find(|v| v["name"] == "Touch").unwrap();
    assert!(touch["children"][0].get("scalar_shorthands").is_none());
}

/// The motivating migration: a bare string in a declared `One` slot
/// parses to the same tree — and the same typed AST — as the explicit
/// object spelling. Existing documents survive the promotion.
#[test]
fn json_shorthand_lands_on_the_explicit_spelling_tree() {
    let short = tree_of(json!({
        "type": "FsWrite",
        "path": "/etc/app.conf",
        "content": "verbose = true",
    }));
    let explicit = tree_of(json!({
        "type": "FsWrite",
        "path": "/etc/app.conf",
        "content": { "type": "Literal", "value": "verbose = true" },
    }));
    assert_eq!(short, explicit);

    let built_short = FsCfg::from_parse_tree(&short, &IdGen::new()).unwrap();
    let built_explicit = FsCfg::from_parse_tree(&explicit, &IdGen::new()).unwrap();
    assert_eq!(built_short, built_explicit);

    // The union's other member stays reachable through the explicit
    // spelling — the shorthand takes nothing away.
    let secret = tree_of(json!({
        "type": "FsWrite",
        "path": "/etc/app.conf",
        "content": { "type": "SecretRef", "name": "app-conf" },
    }));
    let FsCfg::FsWrite { content, .. } = FsCfg::from_parse_tree(&secret, &IdGen::new()).unwrap()
    else {
        panic!("expected FsWrite");
    };
    assert!(matches!(*content, FsCfg::SecretRef { .. }));
}

/// Int and bool kinds dispatch on the input's JSON kind — two kinds
/// on one slot never collide.
#[test]
fn json_int_and_bool_shorthands_dispatch_by_kind() {
    let int_short = tree_of(json!({ "type": "Knob", "value": 3 }));
    let int_explicit = tree_of(json!({ "type": "Knob", "value": { "type": "Count", "n": 3 } }));
    assert_eq!(int_short, int_explicit);

    let bool_short = tree_of(json!({ "type": "Knob", "value": true }));
    let bool_explicit = tree_of(json!({ "type": "Knob", "value": { "type": "Flag", "on": true } }));
    assert_eq!(bool_short, bool_explicit);
}

/// A slot with no declared shorthand keeps today's `CHILD_SHAPE`
/// diagnostic **verbatim** — the feature is opt-in; nothing changes
/// for schemas that never declare one.
#[test]
fn undeclared_slot_keeps_child_shape_diagnostic_verbatim() {
    let schema = FsCfg::schema();
    let err = from_json_value(
        &json!({ "type": "Touch", "target": "not an object" }),
        &schema,
    )
    .expect_err("undeclared slot must reject a bare scalar");
    let diag = &err.diagnostics[0];
    assert_eq!(diag.code, serde_codes::CHILD_SHAPE);
    assert_eq!(
        diag.message,
        "child slot `target` on variant `Touch` requires exactly one object, got a string"
    );

    // A declared slot still rejects kinds it does not declare — the
    // string declaration says nothing about arrays.
    let err = from_json_value(
        &json!({ "type": "FsWrite", "path": "p", "content": [] }),
        &schema,
    )
    .expect_err("declared slot must reject undeclared kinds");
    assert_eq!(err.diagnostics[0].code, serde_codes::CHILD_SHAPE);
}

/// `Optional` triad: `null` → absent, scalar → coerced, object →
/// built. The three spellings coexist on one slot.
#[test]
fn optional_slot_null_scalar_object_triad() {
    let absent = tree_of(json!({ "type": "Fallback", "content": null }));
    assert_eq!(absent.child_slot("content"), Some(&[][..]));

    let coerced = tree_of(json!({ "type": "Fallback", "content": "x" }));
    let explicit = tree_of(json!({
        "type": "Fallback",
        "content": { "type": "Literal", "value": "x" },
    }));
    assert_eq!(coerced, explicit);
}

/// The generated canonical-text grammar accepts both spellings and
/// lands them on the same tree — the two front-ends lower from the
/// same declaration, so they cannot drift apart.
#[test]
fn text_shorthand_lands_on_the_explicit_spelling_tree() {
    let short = text_tree_of(r#"FsWrite(path: "/etc/app.conf", content: "verbose = true")"#);
    let explicit = text_tree_of(
        r#"FsWrite(path: "/etc/app.conf", content: Literal(value: "verbose = true"))"#,
    );
    assert_eq!(strip_spans(short.clone()), strip_spans(explicit));

    // And the typed AST agrees with the JSON front-end's.
    let from_text = FsCfg::from_parse_tree(&short, &IdGen::new()).unwrap();
    let from_json = FsCfg::from_parse_tree(
        &tree_of(json!({
            "type": "FsWrite",
            "path": "/etc/app.conf",
            "content": "verbose = true",
        })),
        &IdGen::new(),
    )
    .unwrap();
    assert_eq!(from_text, from_json);
}

/// Int and bool shorthands in canonical text, including the
/// `Optional` slot's `none` spelling staying intact next to a
/// declared shorthand.
#[test]
fn text_int_bool_and_optional_shorthands() {
    let int_short = text_tree_of("Knob(value: 3)");
    let int_explicit = text_tree_of("Knob(value: Count(n: 3))");
    assert_eq!(strip_spans(int_short), strip_spans(int_explicit));

    let bool_short = text_tree_of("Knob(value: true)");
    let bool_explicit = text_tree_of("Knob(value: Flag(on: true))");
    assert_eq!(strip_spans(bool_short), strip_spans(bool_explicit));

    let absent = text_tree_of("Fallback(content: none)");
    assert!(
        absent
            .child_slot("content")
            .is_none_or(<[ParseTree]>::is_empty)
    );

    let coerced = text_tree_of(r#"Fallback(content: "x")"#);
    let explicit = text_tree_of(r#"Fallback(content: Literal(value: "x"))"#);
    assert_eq!(strip_spans(coerced), strip_spans(explicit));
}

/// Canonical JSON always emits the long form: a shorthand document —
/// through either front-end — dumps identically to the explicit one.
/// Hash `to_canonical_json` output instead of surface bytes and a
/// shorthand introduction is hash-neutral by construction.
#[test]
fn canonical_json_expands_shorthand_to_long_form() {
    let schema = FsCfg::schema();
    let long_form = json!({
        "type": "FsWrite",
        "path": "/etc/app.conf",
        "content": { "type": "Literal", "value": "verbose = true" },
    });

    let from_short_json = tree_of(json!({
        "type": "FsWrite",
        "path": "/etc/app.conf",
        "content": "verbose = true",
    }));
    assert_eq!(
        to_canonical_json(&from_short_json, &schema).unwrap(),
        long_form
    );

    let from_short_text =
        text_tree_of(r#"FsWrite(path: "/etc/app.conf", content: "verbose = true")"#);
    assert_eq!(
        to_canonical_json(&from_short_text, &schema).unwrap(),
        long_form
    );
}

/// Pre-flight rejects every way a hand-written declaration can be
/// incoherent, with `UNSUPPORTED_SCALAR_SHORTHAND`, before any rule
/// is generated.
#[test]
fn grammar_preflight_rejects_incoherent_declarations() {
    let base_variants = || {
        vec![VariantSchema {
            name: "Literal".into(),
            fields: vec![FieldSchema::required("value", "String")],
            children: vec![],
        }]
    };
    let reject = |slot: ChildSchema| {
        let mut variants = base_variants();
        variants.push(VariantSchema {
            name: "Holder".into(),
            fields: vec![],
            children: vec![slot],
        });
        let schema = NodeSchema {
            name: "Cfg".into(),
            variants,
        };
        let err = grammar_from_schema(&schema, &IdGen::new())
            .expect_err("incoherent declaration must fail pre-flight");
        assert!(
            err.diagnostics
                .iter()
                .all(|d| d.code == schema_gen::codes::UNSUPPORTED_SCALAR_SHORTHAND),
            "unexpected codes: {:?}",
            err.diagnostics
        );
    };

    // Wrong multiplicity.
    reject(
        ChildSchema::recursive("items", Multiplicity::Many).with_scalar_shorthand(
            ScalarKind::Str,
            "Literal",
            "value",
        ),
    );
    // Unknown target variant.
    reject(
        ChildSchema::recursive("content", Multiplicity::One).with_scalar_shorthand(
            ScalarKind::Str,
            "Missing",
            "value",
        ),
    );
    // Unknown target field.
    reject(
        ChildSchema::recursive("content", Multiplicity::One).with_scalar_shorthand(
            ScalarKind::Str,
            "Literal",
            "missing",
        ),
    );
    // Kind / field-type mismatch.
    reject(
        ChildSchema::recursive("content", Multiplicity::One).with_scalar_shorthand(
            ScalarKind::Int,
            "Literal",
            "value",
        ),
    );
    // Duplicate kind.
    reject(
        ChildSchema::recursive("content", Multiplicity::One)
            .with_scalar_shorthand(ScalarKind::Str, "Literal", "value")
            .with_scalar_shorthand(ScalarKind::Str, "Literal", "value"),
    );
}
