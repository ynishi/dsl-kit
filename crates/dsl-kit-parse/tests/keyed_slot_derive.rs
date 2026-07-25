//! Derive-side coverage for keyed-slot recognition
//! (`BTreeMap<String, T>` and `BTreeMap<String, Box<T>>`).
//!
//! Scope is the macro layer: `#[derive(DslNode)]` generates walk /
//! walk-mut iteration through `.values()`, and `#[derive(DslSchema)]`
//! emits `Multiplicity::Map`. The build side (JSON ⇒ ParseTree ⇒
//! typed AST) lives in `keyed_slot_json.rs`; PEG codegen still
//! refuses keyed slots outright, guarded by `map_not_implemented.rs`.
//!
//! These tests exercise a downstream-shaped enum that carries
//! keyed-slot fields alongside the pre-existing recursion shapes, so
//! any regression in the derive's type recognition or in the
//! generated iteration code fails loudly at `cargo test`.

use dsl_kit_core::{DslNode, IdGen, NodeId, Walk, WalkMut};
use dsl_kit_macros::{DslNode, DslSchema};
use dsl_kit_schema::{DslSchema, Multiplicity};
use std::collections::BTreeMap;

/// Small self-recursive AST that mixes every recognised recursion
/// shape, so the derive's shape-classification code runs against a
/// realistic input rather than a keyed-slot singleton.
///
/// `#[allow(dead_code)]` at the enum level: several variants are
/// only exercised through the compile-time derive machinery
/// (schema shape assertions in the tests below) rather than through
/// runtime construction, so rustc's dead-code pass — which does not
/// see derive-consuming code paths as a use — would otherwise warn.
#[allow(dead_code)]
#[derive(Debug, DslNode, DslSchema)]
enum Cfg {
    /// Leaf.
    Leaf { id: NodeId, value: String },
    /// Positional child (`Box<Self>`).
    Wrap { id: NodeId, inner: Box<Cfg> },
    /// Positional list (`Vec<Box<Self>>`). The `Box` is intentional —
    /// it exercises the derive's `Recursion::ManyBoxed` arm, which
    /// exists precisely for enums whose value type is not `Sized` in
    /// its own storage. Suppresses `clippy::vec_box` for that reason.
    #[allow(clippy::vec_box)]
    Seq { id: NodeId, items: Vec<Box<Cfg>> },
    /// Keyed slot with boxed self-recursion — the common shape.
    Env {
        id: NodeId,
        entries: BTreeMap<String, Box<Cfg>>,
    },
    /// Keyed slot without `Box` — same schema shape, exercises the
    /// non-boxed derive path (`Recursion::Map`) so the two arms stay
    /// in lockstep.
    Bag {
        id: NodeId,
        entries: BTreeMap<String, Cfg>,
    },
}

/// The generated schema reports `Multiplicity::Map` for keyed slots
/// regardless of whether the value type is wrapped in `Box`. This
/// pins the derive → schema mapping so an accidental
/// `Multiplicity::One` or `::Many` fallback fails at test time.
#[test]
fn schema_reports_map_for_keyed_slots() {
    let schema = Cfg::schema();

    let env = schema.variant("Env").expect("Env variant declared");
    let env_entries = &env.children[0];
    assert_eq!(env_entries.name, "entries");
    assert_eq!(env_entries.multiplicity, Multiplicity::Map);

    let bag = schema.variant("Bag").expect("Bag variant declared");
    let bag_entries = &bag.children[0];
    assert_eq!(bag_entries.name, "entries");
    assert_eq!(bag_entries.multiplicity, Multiplicity::Map);

    // Spot-check that pre-existing shapes still resolve correctly —
    // the BTreeMap detection sits *before* the single-generic path,
    // so a regression there would silently break `Vec` and `Box`.
    let wrap = schema.variant("Wrap").expect("Wrap variant declared");
    assert_eq!(wrap.children[0].multiplicity, Multiplicity::One);
    let seq = schema.variant("Seq").expect("Seq variant declared");
    assert_eq!(seq.children[0].multiplicity, Multiplicity::Many);
}

/// `Walk::children` iterates keyed slot values in the map's own
/// (sorted-by-key) order. Keys themselves are not surfaced through
/// `Walk` — a future keyed-walk API is free to expose them
/// separately.
#[test]
fn walk_iterates_map_values_in_key_order() {
    let ids = IdGen::new();
    let mut entries: BTreeMap<String, Box<Cfg>> = BTreeMap::new();
    // Insertion order deliberately not sorted, so the assertion
    // catches HashMap-style unordered iteration if the derive ever
    // regresses.
    entries.insert(
        "gamma".into(),
        Box::new(Cfg::Leaf {
            id: ids.node(),
            value: "g".into(),
        }),
    );
    entries.insert(
        "alpha".into(),
        Box::new(Cfg::Leaf {
            id: ids.node(),
            value: "a".into(),
        }),
    );
    entries.insert(
        "beta".into(),
        Box::new(Cfg::Leaf {
            id: ids.node(),
            value: "b".into(),
        }),
    );
    let node = Cfg::Env {
        id: ids.node(),
        entries,
    };
    let ordered_values: Vec<&str> = node
        .children()
        .into_iter()
        .map(|c| match c {
            Cfg::Leaf { value, .. } => value.as_str(),
            _ => unreachable!("leaves only"),
        })
        .collect();
    assert_eq!(ordered_values, vec!["a", "b", "g"]);
}

/// `WalkMut::children_mut` yields mutable references to each keyed
/// slot value in the same sorted order and lets the caller mutate
/// them in place. Confirms the `values_mut()` codegen path.
#[test]
fn walk_mut_iterates_map_values_in_key_order() {
    let ids = IdGen::new();
    let mut entries: BTreeMap<String, Box<Cfg>> = BTreeMap::new();
    entries.insert(
        "b".into(),
        Box::new(Cfg::Leaf {
            id: ids.node(),
            value: "before".into(),
        }),
    );
    entries.insert(
        "a".into(),
        Box::new(Cfg::Leaf {
            id: ids.node(),
            value: "before".into(),
        }),
    );
    let mut node = Cfg::Env {
        id: ids.node(),
        entries,
    };

    for child in node.children_mut() {
        if let Cfg::Leaf { value, .. } = child {
            value.push_str("-touched");
        }
    }

    let touched: Vec<&str> = node
        .children()
        .into_iter()
        .map(|c| match c {
            Cfg::Leaf { value, .. } => value.as_str(),
            _ => unreachable!("leaves only"),
        })
        .collect();
    assert_eq!(touched, vec!["before-touched", "before-touched"]);
}

/// `node_id` on a variant that carries a keyed slot still returns
/// the variant's own id; the derive must not accidentally use one of
/// the map values' ids.
#[test]
fn node_id_ignores_keyed_slot_values() {
    let ids = IdGen::new();
    let child_id = ids.node();
    let mut entries: BTreeMap<String, Box<Cfg>> = BTreeMap::new();
    entries.insert(
        "only".into(),
        Box::new(Cfg::Leaf {
            id: child_id,
            value: "v".into(),
        }),
    );
    let variant_id = ids.node();
    let node = Cfg::Env {
        id: variant_id,
        entries,
    };
    assert_eq!(node.node_id(), variant_id);
    assert_ne!(node.node_id(), child_id);
}
