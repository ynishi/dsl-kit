//! End-to-end coverage for the reserved `$allow` annotation — the
//! usage-site half of lint suppression. Verifies that:
//!
//! - the JSON front-end carries `$allow` onto `ParseTree::allows`
//!   verbatim, leaving fields, children, and conformance untouched;
//! - every non-conforming shape fails loudly with `ALLOW_SHAPE`
//!   instead of being silently dropped;
//! - a document with no annotation is byte-identical to what it was
//!   before the feature existed — empty `allows`, no `$allow` key in
//!   the canonical dump;
//! - the canonical dump includes the annotation and round-trips it, so
//!   a content hash computed over that output sees a suppression being
//!   added or removed;
//! - `#[derive(DslBuild)]` carries the names to the typed AST as an
//!   `AllowTable` keyed on the node's minted `NodeId`, drained once.
//!
//! The lint side (actually suppressing a rule) is a separate layer;
//! nothing here reaches into `dsl-kit-lint`.

use dsl_kit_core::{IdGen, NodeId, Walk};
use dsl_kit_macros::{DslBuild, DslNode, DslSchema};
use dsl_kit_parse::{
    DslBuild, ParseTree, allow, check_conformance,
    serde_bridge::{from_json_value, to_canonical_json},
};
use dsl_kit_schema::DslSchema;
use serde_json::json;

/// Fan-out shaped AST: `max-fan-out` is the archetypal rule an author
/// wants to accept at one specific `Par`.
#[derive(Debug, PartialEq, DslNode, DslSchema, DslBuild)]
enum Flow {
    /// Parallel fan-out over its branches.
    Par {
        /// Stable node id.
        id: NodeId,
        /// The branches run in parallel.
        branches: Vec<Flow>,
    },
    /// Leaf unit of work.
    Task {
        /// Stable node id.
        id: NodeId,
        /// Task name.
        name: String,
    },
}

fn tree_of(doc: serde_json::Value) -> ParseTree {
    let schema = Flow::schema();
    let tree = from_json_value(&doc, &schema)
        .unwrap_or_else(|e| panic!("parse failed for {doc}: {:?}", e.diagnostics));
    let diags = check_conformance(&tree, &schema);
    assert!(diags.is_empty(), "conformance clean for {doc}: {diags:?}");
    tree
}

fn annotated_doc() -> serde_json::Value {
    json!({
        "type": "Par",
        "$allow": ["max-fan-out"],
        "branches": [
            { "type": "Task", "name": "a" },
            { "type": "Task", "name": "b" },
        ],
    })
}

/// The annotation lands on `allows` and nothing else moves: the same
/// document without it parses to the same fields and children, and
/// conformance never sees the key (a `$allow` is a document concern,
/// not a schema slot).
#[test]
fn json_carries_allow_names_verbatim() {
    let tree = tree_of(annotated_doc());
    assert_eq!(tree.allows, vec!["max-fan-out".to_string()]);
    assert_eq!(tree.child_slot("branches").unwrap().len(), 2);
    assert!(tree.fields.is_empty());

    // Order and duplicates are the author's; the front-end does not
    // sort or dedupe, because the names are echoed back in the
    // canonical dump.
    let many = tree_of(json!({
        "type": "Task",
        "name": "t",
        "$allow": ["b-rule", "a-rule", "b-rule"],
    }));
    assert_eq!(many.allows, vec!["b-rule", "a-rule", "b-rule"]);

    // Nested nodes carry their own annotation independently.
    let nested = tree_of(json!({
        "type": "Par",
        "branches": [{ "type": "Task", "name": "a", "$allow": ["naming"] }],
    }));
    assert!(nested.allows.is_empty());
    assert_eq!(
        nested.child_slot("branches").unwrap()[0].allows,
        vec!["naming".to_string()]
    );
}

/// Every shape that is not an array of strings fails with
/// `ALLOW_SHAPE`. A suppression the author expected to take effect must
/// never be quietly ignored.
#[test]
fn malformed_allow_reports_allow_shape() {
    let schema = Flow::schema();
    for bad in [
        json!({ "type": "Task", "name": "t", "$allow": "max-fan-out" }),
        json!({ "type": "Task", "name": "t", "$allow": 3 }),
        json!({ "type": "Task", "name": "t", "$allow": ["ok", 3] }),
    ] {
        let Err(err) = from_json_value(&bad, &schema) else {
            panic!("malformed `$allow` must be rejected: {bad}");
        };
        assert_eq!(err.diagnostics.len(), 1, "one diagnostic for {bad}");
        assert_eq!(err.diagnostics[0].code, allow::codes::ALLOW_SHAPE);
        assert!(
            err.diagnostics[0].message.contains(allow::ALLOW_KEY),
            "message names the key: {}",
            err.diagnostics[0].message
        );
    }

    // The annotation is dispatched before the schema lookup, so it is
    // never mistaken for a typo of a declared slot.
    let err = from_json_value(
        &json!({ "type": "Task", "name": "t", "$allow": 3 }),
        &schema,
    )
    .expect_err("rejected above");
    assert!(!err.diagnostics[0].message.contains("did you mean"));
}

/// A document that never spells `$allow` behaves exactly as it did
/// before the annotation existed: empty `allows` everywhere, and a
/// canonical dump with no `$allow` key.
#[test]
fn absent_allow_changes_nothing() {
    let tree = tree_of(json!({
        "type": "Par",
        "branches": [{ "type": "Task", "name": "a" }],
    }));
    assert!(tree.allows.is_empty());
    assert!(tree.child_slot("branches").unwrap()[0].allows.is_empty());

    let canonical = to_canonical_json(&tree, &Flow::schema()).unwrap();
    assert!(canonical.get(allow::ALLOW_KEY).is_none());
    assert_eq!(
        canonical,
        json!({
            "type": "Par",
            "branches": [{ "type": "Task", "name": "a" }],
        })
    );

    // An empty array means the same as no annotation at all.
    let empty = tree_of(json!({ "type": "Task", "name": "a", "$allow": [] }));
    assert!(empty.allows.is_empty());
    assert!(
        to_canonical_json(&empty, &Flow::schema())
            .unwrap()
            .get(allow::ALLOW_KEY)
            .is_none()
    );
}

/// The canonical dump carries the annotation — it is part of the
/// document's reviewed meaning — and reparsing that dump restores it.
#[test]
fn canonical_json_round_trips_the_annotation() {
    let schema = Flow::schema();
    let tree = tree_of(annotated_doc());
    let canonical = to_canonical_json(&tree, &schema).unwrap();

    assert_eq!(canonical[allow::ALLOW_KEY], json!(["max-fan-out"]));
    assert_eq!(
        canonical,
        json!({
            "type": "Par",
            "$allow": ["max-fan-out"],
            "branches": [
                { "type": "Task", "name": "a" },
                { "type": "Task", "name": "b" },
            ],
        })
    );

    let reparsed = tree_of(canonical.clone());
    assert_eq!(reparsed.allows, tree.allows);
    assert_eq!(to_canonical_json(&reparsed, &schema).unwrap(), canonical);
}

/// The build carries the names across the untyped → typed boundary:
/// the table is keyed on the `NodeId` minted for the annotated node,
/// and draining it empties the generator.
#[test]
fn derive_records_allows_against_the_minted_node_id() {
    let ids = IdGen::new();
    let program = Flow::from_parse_tree(&tree_of(annotated_doc()), &ids).unwrap();

    let table = ids.take_allows();
    assert_eq!(table.len(), 1);
    let (&node, names) = table.iter().next().unwrap();
    assert_eq!(names, &vec!["max-fan-out".to_string()]);
    assert_eq!(table.get(&node), Some(&vec!["max-fan-out".to_string()]));

    // The id names the annotated `Par`, not one of its branches.
    assert!(matches!(program.find_by_id(node), Some(Flow::Par { .. })));

    // Draining is a take: the second call sees an empty table.
    assert!(ids.take_allows().is_empty());
}

/// A build over an un-annotated document leaves the generator empty —
/// the collection channel costs nothing when nobody uses it.
#[test]
fn derive_records_nothing_without_an_annotation() {
    let ids = IdGen::new();
    let tree = tree_of(json!({
        "type": "Par",
        "branches": [{ "type": "Task", "name": "a" }],
    }));
    Flow::from_parse_tree(&tree, &ids).unwrap();
    assert!(ids.take_allows().is_empty());
}
