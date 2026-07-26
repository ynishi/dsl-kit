//! End-to-end coverage for scalar-valued keyed child slots
//! (Shape 1 of the tracking issue: `BTreeMap<String, T>` where `T`
//! is a payload type such as `String` / `i64` / `bool`).
//!
//! The consumption path a downstream host walks is
//! `JSON → from_json_value → EnvCfg::from_parse_tree → BTreeMap<K, V>`,
//! so the tests drive that whole chain rather than asserting on
//! intermediate shapes alone. What they pin:
//!
//! - the derive reports `Multiplicity::Map` **and**
//!   `ChildValueShape::Scalar { ty }` for a scalar keyed field,
//!   so schema consumers can tell the shape apart from the recursive
//!   keyed shape at type level;
//! - the JSON bridge routes each scalar value into
//!   [`ParseTree::keyed_children`] wrapped as a `value` field, so
//!   `build_scalar_map` can read it back canonically;
//! - duplicate keys — reachable via hand-built trees since JSON
//!   objects cannot carry them — emit
//!   [`codes::DUPLICATE_KEY`] rather than silently overwriting;
//! - `NodeSchema::to_json` emits the new `value` object so external
//!   consumers (editors / MCP clients) see the scalar shape without
//!   having to guess.

use dsl_kit_core::{IdGen, NodeId};
use dsl_kit_macros::{DslBuild, DslNode, DslSchema};
use dsl_kit_parse::{
    DslBuild, ParseTree, RawValue, build_scalar_map, check_conformance, codes,
    serde_bridge::from_json_value,
};
use dsl_kit_schema::{ChildValueShape, DslSchema, Multiplicity};
use serde_json::json;
use std::collections::BTreeMap;

/// AST carrying scalar-valued keyed slots alongside a recursive keyed
/// slot so the two shapes must be told apart by the derive.
#[derive(Debug, DslNode, DslSchema, DslBuild)]
enum EnvCfg {
    /// String-valued environment map. The canonical Shape 1 case.
    StringEnv {
        /// Stable node id.
        id: NodeId,
        /// Keyed scalar payload (each value is a plain string).
        entries: BTreeMap<String, String>,
    },
    /// Integer-valued knob map — exercises a non-`String` scalar so
    /// the derive doesn't hard-code the value type.
    Knobs {
        /// Stable node id.
        id: NodeId,
        /// Keyed scalar payload (each value is an `i64`).
        entries: BTreeMap<String, i64>,
    },
    /// Recursive keyed slot for contrast — same syntactic shape
    /// (`BTreeMap<String, _>`), different value semantics. Confirms
    /// the derive picks the recursive path when the value type is
    /// `Box<Self>`, and the scalar path in the two variants above.
    Nested {
        /// Stable node id.
        id: NodeId,
        /// Keyed recursive slot.
        entries: BTreeMap<String, Box<EnvCfg>>,
    },
}

/// The derive reports `Multiplicity::Map` for every keyed slot, and
/// tags the scalar-valued ones with the payload type as source text.
/// Pins the schema-side classification so an accidental fallback to
/// `ChildValueShape::Recursive` (which would misroute the JSON
/// bridge) fails at test time.
#[test]
fn schema_tags_scalar_maps_with_value_type() {
    let schema = EnvCfg::schema();

    let env = schema
        .variant("StringEnv")
        .expect("StringEnv variant declared");
    let env_entries = &env.children[0];
    assert_eq!(env_entries.name, "entries");
    assert_eq!(env_entries.multiplicity, Multiplicity::Map);
    assert_eq!(
        env_entries.value_shape,
        ChildValueShape::Scalar {
            ty: "String".into()
        }
    );

    let knobs = schema.variant("Knobs").expect("Knobs variant declared");
    let knobs_entries = &knobs.children[0];
    assert_eq!(knobs_entries.multiplicity, Multiplicity::Map);
    assert_eq!(
        knobs_entries.value_shape,
        ChildValueShape::Scalar { ty: "i64".into() }
    );

    // The recursive keyed shape keeps `ChildValueShape::Recursive`,
    // so the two paths do not collide in downstream dispatch.
    let nested = schema.variant("Nested").expect("Nested variant declared");
    let nested_entries = &nested.children[0];
    assert_eq!(nested_entries.multiplicity, Multiplicity::Map);
    assert_eq!(nested_entries.value_shape, ChildValueShape::Recursive);
}

/// JSON front-end: a scalar keyed slot arrives sorted by key, lands
/// in [`ParseTree::keyed_children`] with each entry wrapped as a
/// `value`-field leaf, conforms clean, and builds through
/// `#[derive(DslBuild)]` into a typed `BTreeMap`.
#[test]
fn json_string_scalar_map_builds_through_to_typed_ast() {
    let schema = EnvCfg::schema();
    let value = json!({
        "type": "StringEnv",
        // Deliberately unsorted to exercise the canonical-order
        // guarantee at the bridge layer.
        "entries": {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/dev",
            "SHELL": "/bin/bash",
        }
    });

    let tree = from_json_value(&value, &schema).expect("scalar keyed slot should parse");
    assert!(
        tree.children.is_empty(),
        "keyed slots must not land in the positional half; got {:?}",
        tree.children
    );
    let entries = tree
        .keyed_child_slot("entries")
        .expect("`entries` present in the keyed half");
    let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys,
        vec!["HOME", "PATH", "SHELL"],
        "bridge must sort keyed entries",
    );
    // Each entry is wrapped as a `value`-field leaf.
    for (_, leaf) in entries {
        assert!(
            leaf.field("value").is_some(),
            "scalar keyed entries must carry their payload under `value`",
        );
    }

    assert!(
        check_conformance(&tree, &schema).is_empty(),
        "a well-formed scalar keyed slot should raise no diagnostics"
    );

    let ids = IdGen::new();
    let built = EnvCfg::from_parse_tree(&tree, &ids).expect("build succeeds");
    let EnvCfg::StringEnv { entries, .. } = built else {
        panic!("expected StringEnv variant");
    };
    assert_eq!(entries.get("HOME").map(String::as_str), Some("/home/dev"));
    assert_eq!(
        entries.get("PATH").map(String::as_str),
        Some("/usr/bin:/bin")
    );
    assert_eq!(entries.get("SHELL").map(String::as_str), Some("/bin/bash"));
    assert_eq!(entries.len(), 3);
}

/// The same round-trip on an `i64` payload confirms the derive does
/// not hard-code the scalar type — `build_scalar_map::<V>` is
/// generic and every numeric primitive that satisfies
/// `serde::de::DeserializeOwned + FromStr` works unchanged.
#[test]
fn json_integer_scalar_map_builds_through_to_typed_ast() {
    let schema = EnvCfg::schema();
    let value = json!({
        "type": "Knobs",
        "entries": {
            "max_retries": 3,
            "timeout_ms": 5000,
        }
    });

    let tree = from_json_value(&value, &schema).expect("integer keyed slot should parse");
    assert!(check_conformance(&tree, &schema).is_empty());

    let ids = IdGen::new();
    let built = EnvCfg::from_parse_tree(&tree, &ids).expect("build succeeds");
    let EnvCfg::Knobs { entries, .. } = built else {
        panic!("expected Knobs variant");
    };
    assert_eq!(entries.get("max_retries").copied(), Some(3));
    assert_eq!(entries.get("timeout_ms").copied(), Some(5000));
}

/// Duplicate keys inside a scalar keyed slot cannot arise from JSON
/// (serde collapses them), but they *can* appear in hand-built trees
/// that other front-ends produce. `build_scalar_map` must reject
/// them with [`codes::DUPLICATE_KEY`] rather than let a later entry
/// silently overwrite an earlier one, so the tree round-trips
/// losslessly.
#[test]
fn duplicate_keys_in_scalar_map_are_rejected() {
    let mut tree = ParseTree::new("StringEnv");
    let leaf = |v: &str| {
        let mut t = ParseTree::new("");
        t.fields.push(("value".into(), RawValue::Json(json!(v))));
        t
    };
    tree.keyed_children.push((
        "entries".into(),
        vec![
            ("dup".into(), leaf("first")),
            ("dup".into(), leaf("second")),
        ],
    ));

    let ids = IdGen::new();
    let err = EnvCfg::from_parse_tree(&tree, &ids).expect_err("duplicate key must fail the build");
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.code == codes::DUPLICATE_KEY),
        "expected DUPLICATE_KEY diagnostic, got {:?}",
        err.diagnostics
            .iter()
            .map(|d| d.code.as_str())
            .collect::<Vec<_>>()
    );
}

/// `NodeSchema::to_json` exposes the new `value` object on scalar
/// keyed slots so external consumers (editors / MCP clients) can
/// dispatch on the shape without guessing. Recursive slots keep the
/// pre-0.6 layout — same `name` + `multiplicity` pair, no `value`
/// key — so consumers that only know about the historical shapes
/// remain unaffected.
#[test]
fn schema_json_carries_scalar_value_shape() {
    let schema_json = EnvCfg::schema().to_json();
    let variants = schema_json
        .get("variants")
        .and_then(|v| v.as_array())
        .expect("variants array");

    let string_env = variants
        .iter()
        .find(|v| v.get("name").and_then(|n| n.as_str()) == Some("StringEnv"))
        .expect("StringEnv in schema JSON");
    let entries = string_env
        .get("children")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .expect("StringEnv.entries child");
    assert_eq!(
        entries,
        &json!({
            "name": "entries",
            "multiplicity": "map",
            "value": { "kind": "scalar", "type": "String" },
        }),
        "scalar keyed slots must carry the value shape in JSON"
    );

    let nested = variants
        .iter()
        .find(|v| v.get("name").and_then(|n| n.as_str()) == Some("Nested"))
        .expect("Nested in schema JSON");
    let entries = nested
        .get("children")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .expect("Nested.entries child");
    assert_eq!(
        entries,
        &json!({ "name": "entries", "multiplicity": "map" }),
        "recursive keyed slots keep the pre-0.6 JSON layout"
    );
}

/// `build_scalar_map` is a public helper — hand-written `DslBuild`
/// impls that mix scalar keyed slots into their own conversion must
/// be able to reach it directly, not only through the derive.
#[test]
fn build_scalar_map_is_reachable_from_hand_written_impls() {
    let mut tree = ParseTree::new("StringEnv");
    let leaf = |v: &str| {
        let mut t = ParseTree::new("");
        t.fields.push(("value".into(), RawValue::Json(json!(v))));
        t
    };
    tree.keyed_children.push((
        "entries".into(),
        vec![("a".into(), leaf("1")), ("b".into(), leaf("2"))],
    ));

    let entries: BTreeMap<String, String> =
        build_scalar_map(&tree, "entries").expect("hand-built tree round-trips");
    assert_eq!(entries.get("a").map(String::as_str), Some("1"));
    assert_eq!(entries.get("b").map(String::as_str), Some("2"));

    // Absent slot → empty map (mirrors `build_child_map` semantics
    // so downstream code can treat both keyed shapes uniformly).
    let empty: BTreeMap<String, String> =
        build_scalar_map(&ParseTree::new("StringEnv"), "entries").expect("absent slot ok");
    assert!(empty.is_empty());
}
