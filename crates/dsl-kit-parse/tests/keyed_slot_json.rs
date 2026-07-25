//! End-to-end coverage for keyed child slots (`Multiplicity::Map`)
//! across the three stages that carry them: the JSON ⇒ [`ParseTree`]
//! bridge, schema conformance, and `#[derive(DslBuild)]` → typed AST.
//!
//! The consumption path a downstream host actually walks is
//! `JSON → from_json_value → Cfg::from_parse_tree → BTreeMap`, so the
//! tests drive that whole chain rather than asserting on intermediate
//! shapes alone. What they pin:
//!
//! - **canonical key order** — a document written out of order lands
//!   sorted, so two spellings of the same map produce byte-identical
//!   trees;
//! - **keying discipline** — a keyed slot handed over positionally
//!   (or vice versa) is a diagnostic, not a silent drop;
//! - **duplicate keys** — representable in the tree precisely so they
//!   can be rejected instead of one entry quietly winning;
//! - **both derive shapes** — `BTreeMap<String, Box<Self>>` and
//!   `BTreeMap<String, Self>` build through the same helper.

use dsl_kit_core::{IdGen, NodeId};
use dsl_kit_macros::{DslBuild, DslNode, DslSchema};
use dsl_kit_parse::{
    DslBuild, ParseTree, build_child_map, check_conformance, codes,
    serde_bridge::{from_json_value, serde_codes},
};
use dsl_kit_schema::DslSchema;
use serde_json::json;
use std::collections::BTreeMap;

/// Self-recursive AST carrying both keyed-slot shapes plus a
/// positional slot, so keying mistakes have somewhere to go wrong.
#[derive(Debug, DslNode, DslSchema, DslBuild)]
enum Cfg {
    /// Leaf holding a payload string.
    Leaf {
        /// Stable node id.
        id: NodeId,
        /// Payload.
        value: String,
    },
    /// Keyed slot with boxed self-recursion — the common shape.
    Env {
        /// Stable node id.
        id: NodeId,
        /// Keyed children.
        entries: BTreeMap<String, Box<Cfg>>,
    },
    /// Keyed slot without `Box`, exercising the non-boxed derive arm.
    Bag {
        /// Stable node id.
        id: NodeId,
        /// Keyed children.
        entries: BTreeMap<String, Cfg>,
    },
    /// Positional list, so a keyed/positional mix-up is expressible.
    #[allow(clippy::vec_box)]
    Seq {
        /// Stable node id.
        id: NodeId,
        /// Positional children.
        items: Vec<Box<Cfg>>,
    },
    /// Positional single child. `Multiplicity::One` is the arity with
    /// the strictest predicate, so it is the variant that exposes
    /// whether a keying mistake also drags in a bogus arity verdict.
    Wrap {
        /// Stable node id.
        id: NodeId,
        /// Positional child.
        inner: Box<Cfg>,
    },
}

/// Every diagnostic code raised by `check_conformance`, in order.
/// Tests that care about a keying mistake assert the *whole* vector
/// rather than `.any(...)`: the point of those checks is which
/// diagnostics fire, and `.any` cannot see a spurious companion.
fn conformance_codes(tree: &ParseTree) -> Vec<String> {
    check_conformance(tree, &Cfg::schema())
        .iter()
        .map(|d| d.code.clone())
        .collect()
}

/// Builds a `Leaf` JSON object with the given payload.
fn leaf(value: &str) -> serde_json::Value {
    json!({ "type": "Leaf", "value": value })
}

/// Reads the payload of a `Leaf`, panicking on any other variant.
fn leaf_value(node: &Cfg) -> &str {
    match node {
        Cfg::Leaf { value, .. } => value.as_str(),
        other => panic!("expected a Leaf, got {other:?}"),
    }
}

/// The JSON bridge routes a map-declared slot into
/// [`ParseTree::keyed_children`] (not `children`), sorted by key, and
/// the resulting tree conforms and builds into a `BTreeMap` carrying
/// the same keys.
#[test]
fn json_keyed_slot_builds_through_to_typed_ast() {
    let schema = Cfg::schema();
    let value = json!({
        "type": "Env",
        // Deliberately unsorted: canonical order is the bridge's job.
        "entries": {
            "gamma": leaf("g"),
            "alpha": leaf("a"),
            "beta": leaf("b"),
        }
    });

    let tree = from_json_value(&value, &schema).expect("keyed slot should parse");
    assert!(
        tree.children.is_empty(),
        "keyed slots must not land in the positional half; got {:?}",
        tree.children
    );
    let entries = tree
        .keyed_child_slot("entries")
        .expect("`entries` present in the keyed half");
    let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, vec!["alpha", "beta", "gamma"]);

    assert!(
        check_conformance(&tree, &schema).is_empty(),
        "a well-formed keyed slot should raise no diagnostics"
    );

    let ids = IdGen::new();
    let built = Cfg::from_parse_tree(&tree, &ids).expect("typed build should succeed");
    let Cfg::Env { entries, .. } = &built else {
        panic!("expected Env, got {built:?}");
    };
    assert_eq!(
        entries.keys().map(String::as_str).collect::<Vec<_>>(),
        vec!["alpha", "beta", "gamma"]
    );
    assert_eq!(leaf_value(&entries["alpha"]), "a");
    assert_eq!(leaf_value(&entries["beta"]), "b");
    assert_eq!(leaf_value(&entries["gamma"]), "g");
}

/// Two documents whose keyed entries are written in different orders
/// produce the same [`ParseTree`]. This is the property the canonical
/// sort exists for — without it, key order would leak into tree
/// equality, snapshot tests, and any future re-emission.
#[test]
fn json_key_order_does_not_affect_the_tree() {
    let schema = Cfg::schema();
    let forward = json!({
        "type": "Env",
        "entries": { "a": leaf("1"), "b": leaf("2") }
    });
    let reversed = json!({
        "type": "Env",
        "entries": { "b": leaf("2"), "a": leaf("1") }
    });

    let lhs = from_json_value(&forward, &schema).expect("parses");
    let rhs = from_json_value(&reversed, &schema).expect("parses");
    assert_eq!(lhs, rhs);
}

/// An empty keyed slot is a valid shape (zero-or-more, same as
/// `Many`), and builds into an empty map rather than an error.
#[test]
fn json_empty_keyed_slot_is_valid() {
    let schema = Cfg::schema();
    let value = json!({ "type": "Env", "entries": {} });

    let tree = from_json_value(&value, &schema).expect("empty keyed slot parses");
    assert!(check_conformance(&tree, &schema).is_empty());

    let ids = IdGen::new();
    let built = Cfg::from_parse_tree(&tree, &ids).expect("builds");
    let Cfg::Env { entries, .. } = &built else {
        panic!("expected Env, got {built:?}");
    };
    assert!(entries.is_empty());
}

/// The non-boxed keyed shape (`BTreeMap<String, Self>`) goes through
/// the same helper as the boxed one — the two derive arms differ only
/// in the re-wrap, so they are pinned together.
#[test]
fn json_keyed_slot_builds_unboxed_values() {
    let schema = Cfg::schema();
    let value = json!({
        "type": "Bag",
        "entries": { "k": leaf("v") }
    });

    let tree = from_json_value(&value, &schema).expect("parses");
    let ids = IdGen::new();
    let built = Cfg::from_parse_tree(&tree, &ids).expect("builds");
    let Cfg::Bag { entries, .. } = &built else {
        panic!("expected Bag, got {built:?}");
    };
    assert_eq!(leaf_value(&entries["k"]), "v");
}

/// A keyed slot written as a JSON array (the `Many` spelling) is a
/// shape error, not an empty map.
#[test]
fn json_keyed_slot_rejects_array_shape() {
    let schema = Cfg::schema();
    let value = json!({
        "type": "Env",
        "entries": [leaf("a")]
    });

    let err = from_json_value(&value, &schema).expect_err("array shape should fail");
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.code == serde_codes::CHILD_SHAPE),
        "expected CHILD_SHAPE; got {:?}",
        err.diagnostics
            .iter()
            .map(|d| d.code.as_str())
            .collect::<Vec<_>>()
    );
}

/// An individual entry that is not an object is rejected on its own,
/// naming the offending key, while sibling entries still parse — the
/// bridge collects diagnostics rather than stopping at the first.
#[test]
fn json_keyed_slot_rejects_scalar_entry() {
    let schema = Cfg::schema();
    let value = json!({
        "type": "Env",
        "entries": { "ok": leaf("a"), "bad": 7 }
    });

    let err = from_json_value(&value, &schema).expect_err("scalar entry should fail");
    let shape: Vec<&str> = err
        .diagnostics
        .iter()
        .filter(|d| d.code == serde_codes::CHILD_SHAPE)
        .map(|d| d.message.as_str())
        .collect();
    assert_eq!(shape.len(), 1, "one bad entry, one diagnostic: {shape:?}");
    assert!(
        shape[0].contains("`bad`"),
        "diagnostic should name the offending key: {}",
        shape[0]
    );
}

/// A keyed slot supplied as a positional list is a keying error, and
/// *only* that error. The tree is hand-built because no front-end
/// would produce it — this is the shape a hand-rolled front-end gets
/// wrong.
#[test]
fn conformance_flags_keyed_slot_supplied_positionally() {
    let schema = Cfg::schema();
    let mut tree = ParseTree::new("Env");
    tree.children.push((
        "entries".into(),
        vec![from_json_value(&leaf("a"), &schema).expect("leaf parses")],
    ));

    assert_eq!(conformance_codes(&tree), vec![codes::KEYED_SLOT_SHAPE]);
}

/// The mirror case: a positional slot supplied as keyed entries.
#[test]
fn conformance_flags_positional_slot_supplied_keyed() {
    let schema = Cfg::schema();
    let mut tree = ParseTree::new("Seq");
    tree.keyed_children.push((
        "items".into(),
        vec![(
            "k".into(),
            from_json_value(&leaf("a"), &schema).expect("leaf parses"),
        )],
    ));

    assert_eq!(conformance_codes(&tree), vec![codes::KEYED_SLOT_SHAPE]);
}

/// The same mirror case on a `Multiplicity::One` slot, where the
/// arity predicate would fire if the keying check did not stand the
/// shape check down. Without that, a tree supplying exactly one child
/// collects `ARITY_ONE` claiming it supplied zero — one mistake, two
/// errors, the second untrue. Asserts the exact vector for that
/// reason.
#[test]
fn conformance_does_not_add_arity_noise_to_a_mis_keyed_one_slot() {
    let schema = Cfg::schema();
    let mut tree = ParseTree::new("Wrap");
    tree.keyed_children.push((
        "inner".into(),
        vec![(
            "k".into(),
            from_json_value(&leaf("a"), &schema).expect("leaf parses"),
        )],
    ));

    assert_eq!(conformance_codes(&tree), vec![codes::KEYED_SLOT_SHAPE]);
}

/// A slot name occupying *both* halves is a duplicate, since the two
/// vectors share one namespace. The shape check still runs here (the
/// right half is populated), so the keying diagnostic and the
/// duplicate diagnostic are both expected — unlike the wrong-half-only
/// case above.
#[test]
fn conformance_flags_a_slot_name_present_in_both_halves() {
    let schema = Cfg::schema();
    let mut tree = ParseTree::new("Env");
    tree.children.push((
        "entries".into(),
        vec![from_json_value(&leaf("a"), &schema).expect("leaf parses")],
    ));
    tree.keyed_children.push((
        "entries".into(),
        vec![(
            "k".into(),
            from_json_value(&leaf("b"), &schema).expect("leaf parses"),
        )],
    ));

    let seen = conformance_codes(&tree);
    assert!(
        seen.contains(&codes::DUPLICATE_CHILD.to_string()),
        "expected DUPLICATE_CHILD; got {seen:?}"
    );
    assert!(
        seen.contains(&codes::KEYED_SLOT_SHAPE.to_string()),
        "expected KEYED_SLOT_SHAPE; got {seen:?}"
    );
}

/// Unsorted entries are rejected. The canonical order exists so two
/// front-ends handed the same document produce equal trees; leaving it
/// as an unchecked doc promise is how a hand-rolled emitter ends up
/// shipping trees that compare unequal to the bridge's.
#[test]
fn conformance_flags_unsorted_keyed_entries() {
    let schema = Cfg::schema();
    let mut tree = ParseTree::new("Env");
    tree.keyed_children.push((
        "entries".into(),
        vec![
            (
                "b".into(),
                from_json_value(&leaf("2"), &schema).expect("leaf parses"),
            ),
            (
                "a".into(),
                from_json_value(&leaf("1"), &schema).expect("leaf parses"),
            ),
        ],
    ));

    assert_eq!(conformance_codes(&tree), vec![codes::KEYED_SLOT_UNSORTED]);
}

/// An undeclared slot name arriving through the keyed half is as
/// unknown as one arriving positionally, and a payload field name
/// misfiled as a keyed slot still reports as a field.
#[test]
fn conformance_scans_the_keyed_half_for_unknown_slots() {
    let mut unknown = ParseTree::new("Env");
    unknown
        .keyed_children
        .push(("nope".into(), vec![("k".into(), ParseTree::new("Leaf"))]));
    assert!(
        conformance_codes(&unknown).contains(&codes::UNKNOWN_CHILD.to_string()),
        "expected UNKNOWN_CHILD; got {:?}",
        conformance_codes(&unknown)
    );

    let mut as_child = ParseTree::new("Leaf");
    as_child.fields.push((
        "value".into(),
        dsl_kit_parse::RawValue::Json(json!("payload")),
    ));
    as_child
        .keyed_children
        .push(("value".into(), vec![("k".into(), ParseTree::new("Leaf"))]));
    assert!(
        conformance_codes(&as_child).contains(&codes::FIELD_AS_CHILD.to_string()),
        "expected FIELD_AS_CHILD; got {:?}",
        conformance_codes(&as_child)
    );
}

/// A keyed slot omitted entirely (rather than present-and-empty) takes
/// a different path — `keyed_child_slot` returns `None` — and must
/// land in the same place: conformant, and an empty map.
#[test]
fn absent_keyed_slot_conforms_and_builds_empty() {
    let tree = ParseTree::new("Env");
    assert_eq!(conformance_codes(&tree), Vec::<String>::new());

    let ids = IdGen::new();
    let built = Cfg::from_parse_tree(&tree, &ids).expect("builds");
    let Cfg::Env { entries, .. } = &built else {
        panic!("expected Env, got {built:?}");
    };
    assert!(entries.is_empty());
}

/// Keyed slots nest: an entry may itself be a node carrying a keyed
/// slot. Each level sorts and builds independently.
#[test]
fn json_keyed_slots_nest() {
    let schema = Cfg::schema();
    let value = json!({
        "type": "Env",
        "entries": {
            "outer": {
                "type": "Env",
                "entries": { "z": leaf("deep-z"), "y": leaf("deep-y") }
            }
        }
    });

    let tree = from_json_value(&value, &schema).expect("parses");
    let ids = IdGen::new();
    let built = Cfg::from_parse_tree(&tree, &ids).expect("builds");
    let Cfg::Env { entries, .. } = &built else {
        panic!("expected Env, got {built:?}");
    };
    let Cfg::Env { entries: inner, .. } = entries["outer"].as_ref() else {
        panic!("expected a nested Env");
    };
    assert_eq!(
        inner.keys().map(String::as_str).collect::<Vec<_>>(),
        vec!["y", "z"]
    );
    assert_eq!(leaf_value(&inner["y"]), "deep-y");
}

/// A repeated key is rejected at conformance time and again at build
/// time, so neither path can silently keep one entry and drop the
/// other. (JSON itself cannot express this — `serde_json` resolves
/// duplicates before the bridge sees them — which is exactly why the
/// tree keeps entries in a `Vec` and checks them here.)
#[test]
fn duplicate_keys_are_rejected_at_conformance_and_build() {
    let schema = Cfg::schema();
    let mut tree = ParseTree::new("Env");
    tree.keyed_children.push((
        "entries".into(),
        vec![
            (
                "dup".into(),
                from_json_value(&leaf("first"), &schema).expect("leaf parses"),
            ),
            (
                "dup".into(),
                from_json_value(&leaf("second"), &schema).expect("leaf parses"),
            ),
        ],
    ));

    assert_eq!(conformance_codes(&tree), vec![codes::DUPLICATE_KEY]);

    let ids = IdGen::new();
    let err = build_child_map::<Cfg>(&tree, "entries", &ids)
        .expect_err("build_child_map should reject the duplicate");
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.code == codes::DUPLICATE_KEY),
        "expected DUPLICATE_KEY from build_child_map; got {:?}",
        err.diagnostics
            .iter()
            .map(|d| d.code.as_str())
            .collect::<Vec<_>>()
    );
}

/// Diagnostics raised inside a keyed entry's subtree reach the caller
/// with the entry's own context intact — a broken child does not get
/// swallowed by the map machinery.
#[test]
fn build_reports_diagnostics_from_inside_keyed_entries() {
    let mut tree = ParseTree::new("Env");
    // `Leaf` without its required `value` payload.
    tree.keyed_children
        .push(("entries".into(), vec![("k".into(), ParseTree::new("Leaf"))]));

    let ids = IdGen::new();
    let err = build_child_map::<Cfg>(&tree, "entries", &ids)
        .expect_err("a malformed entry should surface");
    assert!(
        err.diagnostics
            .iter()
            .any(|d| d.code == codes::MISSING_FIELD),
        "expected MISSING_FIELD from the entry's subtree; got {:?}",
        err.diagnostics
            .iter()
            .map(|d| d.code.as_str())
            .collect::<Vec<_>>()
    );
}
