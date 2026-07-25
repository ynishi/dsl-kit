//! Integration coverage for the `Multiplicity::Map` "not yet implemented"
//! diagnostics emitted from the three pipeline stages that touch child
//! slots: schema conformance ([`check_conformance`]), the JSON ⇒
//! `ParseTree` bridge ([`from_json_value`]), and grammar generation
//! ([`grammar_from_schema`]).
//!
//! Compiler-enforced exhaustive `match` already guarantees each site
//! carries a `Multiplicity::Map` arm; these tests guard the *slug*
//! actually emitted, so the "map support arriving" carry can be
//! grep-verified by the presence / absence of each stage's diagnostic
//! code.
//!
//! Rationale: without these tests, silently swapping a slug (e.g.
//! `codes::MAP_NOT_IMPLEMENTED` → `serde_codes::MAP_NOT_IMPLEMENTED`
//! at the wrong site, or dropping a diagnostic in favor of an
//! `unreachable!`) would compile clean while breaking consumers that
//! filter on stable slugs.

use dsl_kit_core::IdGen;
use dsl_kit_parse::{
    ParseTree, check_conformance, codes as parse_codes,
    schema_gen::{codes as schema_gen_codes, grammar_from_schema},
    serde_bridge::{from_json_value, serde_codes},
};
use dsl_kit_schema::{ChildSchema, Multiplicity, NodeSchema, VariantSchema};
use serde_json::json;

/// Hand-authored schema with one variant carrying a
/// [`Multiplicity::Map`] child slot. Reused across the three stage
/// tests so any drift is a single-file edit.
fn schema_with_map_slot() -> NodeSchema {
    NodeSchema {
        name: "Cfg".into(),
        variants: vec![VariantSchema {
            name: "Root".into(),
            fields: vec![],
            children: vec![ChildSchema {
                name: "entries".into(),
                multiplicity: Multiplicity::Map,
            }],
        }],
    }
}

/// [`check_conformance`] emits [`parse_codes::MAP_NOT_IMPLEMENTED`]
/// when a tree carries a child slot whose schema declares
/// [`Multiplicity::Map`].
#[test]
fn check_conformance_flags_map_slot() {
    let schema = schema_with_map_slot();
    // A minimal tree that *would* satisfy the shape if map runtime
    // support existed: `entries` slot present, zero children. The
    // arity check would accept zero-or-more for `Many`, but for
    // `Map` it short-circuits with the not-implemented diagnostic
    // regardless of the child count.
    let tree = ParseTree {
        variant: "Root".into(),
        fields: vec![],
        children: vec![("entries".into(), vec![])],
        span: None,
    };
    let diags = check_conformance(&tree, &schema);
    assert!(
        diags.iter().any(|d| d.code == parse_codes::MAP_NOT_IMPLEMENTED),
        "expected MAP_NOT_IMPLEMENTED among diagnostics; got {:?}",
        diags.iter().map(|d| d.code.as_str()).collect::<Vec<_>>()
    );
}

/// The JSON ⇒ [`ParseTree`] bridge emits
/// [`serde_codes::MAP_NOT_IMPLEMENTED`] (namespaced under `serde`)
/// when a JSON document names a map-declared child slot.
#[test]
fn serde_bridge_flags_map_slot() {
    let schema = schema_with_map_slot();
    // The `entries` value shape here is deliberately whatever — the
    // bridge rejects the slot on `Multiplicity` grounds before it
    // inspects the value, so any JSON payload triggers the same
    // diagnostic.
    let value = json!({
        "type": "Root",
        "entries": {}
    });
    let err =
        from_json_value(&value, &schema).expect_err("Map slot should surface an error, not a valid tree");
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.code == serde_codes::MAP_NOT_IMPLEMENTED),
        "expected serde_codes::MAP_NOT_IMPLEMENTED among diagnostics; got {:?}",
        err.diagnostics
            .iter()
            .map(|d| d.code.as_str())
            .collect::<Vec<_>>()
    );
}

/// [`grammar_from_schema`] refuses to generate a grammar when any
/// variant declares a [`Multiplicity::Map`] child slot, surfacing
/// [`schema_gen_codes::MAP_NOT_IMPLEMENTED`] up front so authors do
/// not receive a silently-missing PEG rule.
#[test]
fn schema_gen_rejects_map_slot() {
    let schema = schema_with_map_slot();
    let ids = IdGen::new();
    let err =
        grammar_from_schema(&schema, &ids).expect_err("Map slot should abort grammar generation");
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.code == schema_gen_codes::MAP_NOT_IMPLEMENTED),
        "expected schema_gen_codes::MAP_NOT_IMPLEMENTED among diagnostics; got {:?}",
        err.diagnostics
            .iter()
            .map(|d| d.code.as_str())
            .collect::<Vec<_>>()
    );
}
