//! Round-trip coverage for `#[derive(DslDump)]` — the inverse of
//! `#[derive(DslBuild)]`.
//!
//! The contract under test is the one downstream serializers rest on
//! (an in-memory AST cached / transported as bridge JSON must re-parse
//! to an equivalent AST):
//!
//! ```text
//! from_parse_tree(&ast.to_parse_tree()?, &ids)  ≙  ast   (modulo NodeId)
//! ```
//!
//! What the tests pin:
//!
//! - **self-conformance** — a dumped tree passes `check_conformance`
//!   against the same enum's derived schema, so the derive cannot
//!   silently drift from the shape its own build side accepts;
//! - **canonical-JSON stability** — dump → parse → dump reproduces the
//!   identical `Value`, which is the fixed point content hashing
//!   needs;
//! - **omission spellings** — absent `Option` payloads / children and
//!   empty keyed maps omit their keys; empty `Vec`s emit explicitly;
//! - **`$allow` carriage** — an [`AllowTable`] handed to the dump
//!   surfaces as `$allow` in the tree and survives a re-parse into the
//!   rebuilt AST's table;
//! - **`with` duality** — a field whose build side uses
//!   `#[dsl_build(with = ...)]` serializes through its mandatory
//!   `#[dsl_dump(with = ...)]` twin.

use dsl_kit_core::{AllowTable, IdGen, NodeId};
use dsl_kit_macros::{DslBuild, DslDump, DslNode, DslSchema};
use dsl_kit_parse::{
    BuildError, DslBuild, DslDump, ParseTree, RawValue, check_conformance, dump_canonical_json,
    dump_canonical_json_with, serde_bridge::from_json_value,
};
use dsl_kit_schema::DslSchema;
use serde_json::{Value, json};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Payload type with neither `Serialize` nor `FromStr`, so both
/// directions must route through their `with` functions.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Level {
    /// Low level.
    Low,
    /// High level.
    High,
}

fn level_from_tree(tree: &ParseTree, name: &str) -> Result<Level, BuildError> {
    match tree.field(name) {
        Some(RawValue::Json(Value::String(s))) if s == "low" => Ok(Level::Low),
        Some(RawValue::Json(Value::String(s))) if s == "high" => Ok(Level::High),
        other => panic!("unexpected `{name}` payload: {other:?}"),
    }
}

fn level_to_json(level: &Level) -> Result<Option<Value>, BuildError> {
    Ok(Some(Value::String(
        match level {
            Level::Low => "low",
            Level::High => "high",
        }
        .to_string(),
    )))
}

/// Fixture enum exercising every derive arm: bare / optional / vec
/// payloads, one / optional / many / keyed children, a scalar map, and
/// a `with`-paired custom payload.
#[derive(Debug, DslNode, DslSchema, DslBuild, DslDump)]
enum Cfg {
    /// Leaf with the payload shapes.
    Leaf {
        /// Stable node id.
        id: NodeId,
        /// Bare payload.
        name: String,
        /// Optional payload — `None` omits the key.
        note: Option<String>,
        /// Vec payload — always emits, `[]` included.
        tags: Vec<String>,
    },
    /// Custom-converted payload (`with` pair on both derives).
    Gauge {
        /// Stable node id.
        id: NodeId,
        /// Payload routed through `level_from_tree` / `level_to_json`.
        #[dsl_build(with = level_from_tree)]
        #[dsl_dump(with = level_to_json)]
        level: Level,
    },
    /// The child slot shapes.
    Node {
        /// Stable node id.
        id: NodeId,
        /// One child, boxed.
        head: Box<Cfg>,
        /// Optional child.
        fallback: Option<Box<Cfg>>,
        /// Many children.
        items: Vec<Cfg>,
        /// Keyed children.
        env: BTreeMap<String, Box<Cfg>>,
        /// Scalar-valued keyed slot.
        limits: BTreeMap<String, i64>,
    },
}

fn leaf(ids: &IdGen, name: &str) -> Cfg {
    Cfg::Leaf {
        id: ids.node(),
        name: name.to_string(),
        note: None,
        tags: Vec::new(),
    }
}

/// A node exercising every populated shape at once.
fn full_ast(ids: &IdGen) -> Cfg {
    Cfg::Node {
        id: ids.node(),
        head: Box::new(Cfg::Leaf {
            id: ids.node(),
            name: "head".to_string(),
            note: Some("annotated".to_string()),
            tags: vec!["a".to_string(), "b".to_string()],
        }),
        fallback: Some(Box::new(leaf(ids, "fb"))),
        items: vec![leaf(ids, "i1"), leaf(ids, "i2")],
        env: BTreeMap::from([
            ("x".to_string(), Box::new(leaf(ids, "ex"))),
            ("y".to_string(), Box::new(Cfg::Gauge { id: ids.node(), level: Level::High })),
        ]),
        limits: BTreeMap::from([("mem".to_string(), 512), ("cpu".to_string(), 2)]),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn dumped_tree_is_self_conformant() {
    let ids = IdGen::default();
    let tree = full_ast(&ids).to_parse_tree().expect("dump");
    let diags = check_conformance(&tree, &Cfg::schema());
    assert!(diags.is_empty(), "dumped tree failed conformance: {diags:?}");
}

#[test]
fn dump_parse_dump_is_a_fixed_point() {
    let ids = IdGen::default();
    let ast = full_ast(&ids);

    let json1 = dump_canonical_json(&ast).expect("dump 1");
    let tree = from_json_value(&json1, &Cfg::schema()).expect("re-parse");
    let rebuilt = Cfg::from_parse_tree(&tree, &IdGen::default()).expect("rebuild");
    let json2 = dump_canonical_json(&rebuilt).expect("dump 2");

    assert_eq!(json1, json2);
}

#[test]
fn rebuild_through_own_tree_round_trips() {
    let ids = IdGen::default();
    let ast = full_ast(&ids);
    let tree = ast.to_parse_tree().expect("dump");
    let rebuilt = Cfg::from_parse_tree(&tree, &IdGen::default()).expect("rebuild");
    // NodeIds are freshly minted, so compare through the canonical
    // projection rather than structural equality.
    assert_eq!(
        dump_canonical_json(&ast).expect("json a"),
        dump_canonical_json(&rebuilt).expect("json b"),
    );
}

#[test]
fn omission_spellings_match_the_canonical_rules() {
    let ids = IdGen::default();
    let ast = Cfg::Node {
        id: ids.node(),
        head: Box::new(leaf(&ids, "only")),
        fallback: None,
        items: Vec::new(),
        env: BTreeMap::new(),
        limits: BTreeMap::new(),
    };
    let json = dump_canonical_json(&ast).expect("dump");
    let obj = json.as_object().expect("object");

    // Absent optional child and empty keyed maps omit their keys.
    assert!(!obj.contains_key("fallback"));
    assert!(!obj.contains_key("env"));
    assert!(!obj.contains_key("limits"));
    // Empty Many slot emits explicitly.
    assert_eq!(obj.get("items"), Some(&json!([])));
    // The leaf inside omits its absent option and emits its empty vec.
    let head = obj.get("head").and_then(Value::as_object).expect("head");
    assert!(!head.contains_key("note"));
    assert_eq!(head.get("tags"), Some(&json!([])));
}

#[test]
fn with_pair_serializes_and_rebuilds() {
    let ids = IdGen::default();
    let ast = Cfg::Gauge {
        id: ids.node(),
        level: Level::High,
    };
    let json = dump_canonical_json(&ast).expect("dump");
    assert_eq!(json, json!({ "type": "Gauge", "level": "high" }));

    let tree = from_json_value(&json, &Cfg::schema()).expect("re-parse");
    let rebuilt = Cfg::from_parse_tree(&tree, &IdGen::default()).expect("rebuild");
    let Cfg::Gauge { level, .. } = rebuilt else {
        panic!("rebuilt wrong variant");
    };
    assert_eq!(level, Level::High);
}

/// Hygiene regression fixture: a payload field literally named
/// `allows` must not shadow the derive's `AllowTable` parameter
/// inside the generated variant arm.
#[derive(Debug, DslNode, DslSchema, DslBuild, DslDump)]
enum Shadow {
    /// Node whose payload field collides with the obvious parameter
    /// name.
    Item {
        /// Stable node id.
        id: NodeId,
        /// Payload named like the allow-table parameter.
        allows: Vec<String>,
    },
}

#[test]
fn field_named_allows_does_not_shadow_the_table() {
    let ids = IdGen::default();
    let ast = Shadow::Item {
        id: ids.node(),
        allows: vec!["payload".to_string()],
    };
    let Shadow::Item { id, .. } = &ast;
    let mut table = AllowTable::default();
    table.insert(*id, vec!["rule_x".to_string()]);

    let json = dump_canonical_json_with(&ast, &table).expect("dump");
    // `$allow` comes from the table, the `allows` key from the payload
    // field — a shadow would have fed the payload into the table
    // lookup and dropped the annotation.
    assert_eq!(json.get("$allow"), Some(&json!(["rule_x"])));
    assert_eq!(json.get("allows"), Some(&json!(["payload"])));
}

#[test]
fn allow_table_survives_the_round_trip() {
    let ids = IdGen::default();
    let ast = leaf(&ids, "guarded");
    let Cfg::Leaf { id, .. } = &ast else {
        unreachable!()
    };
    let mut allows = AllowTable::default();
    allows.insert(*id, vec!["rule_a".to_string()]);

    let json = dump_canonical_json_with(&ast, &allows).expect("dump");
    assert_eq!(json.get("$allow"), Some(&json!(["rule_a"])));

    // Re-parse: the annotation lands in the rebuilding IdGen's table.
    let tree = from_json_value(&json, &Cfg::schema()).expect("re-parse");
    let rebuild_ids = IdGen::default();
    let rebuilt = Cfg::from_parse_tree(&tree, &rebuild_ids).expect("rebuild");
    let table = rebuild_ids.take_allows();
    assert_eq!(table.len(), 1);
    let Cfg::Leaf { id: new_id, .. } = &rebuilt else {
        panic!("rebuilt wrong variant");
    };
    assert_eq!(table.get(new_id), Some(&vec!["rule_a".to_string()]));
}
