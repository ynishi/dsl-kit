//! Integration coverage for the remaining `Multiplicity::Map` "not yet
//! implemented" diagnostic: grammar generation
//! ([`grammar_from_schema`]).
//!
//! Conformance ([`check_conformance`]) and the JSON ⇒ `ParseTree`
//! bridge grew real keyed-slot support, so their `MAP_NOT_IMPLEMENTED`
//! slugs are gone — their behaviour now lives in `keyed_slot_json.rs`.
//! Grammar generation still refuses map-carrying schemas up front,
//! because the canonical *text* syntax for a keyed slot is an open
//! design question; this file guards that refusal.
//!
//! Compiler-enforced exhaustive `match` already guarantees the site
//! carries a `Multiplicity::Map` arm; this test guards the *slug*
//! actually emitted, so the "map support arriving" carry can be
//! grep-verified by the presence / absence of the diagnostic code.
//!
//! Rationale: without this test, dropping the diagnostic in favour of
//! an `unreachable!` (or silently generating a rule that matches
//! nothing) would compile clean while breaking authors who declared a
//! keyed slot and expected either a grammar or a clear refusal.

use dsl_kit_core::IdGen;
use dsl_kit_parse::schema_gen::{codes as schema_gen_codes, grammar_from_schema};
use dsl_kit_schema::{ChildSchema, Multiplicity, NodeSchema, VariantSchema};

/// Hand-authored schema with one variant carrying a
/// [`Multiplicity::Map`] child slot.
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
