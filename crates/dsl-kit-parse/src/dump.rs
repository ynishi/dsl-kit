//! Typed AST → [`ParseTree`] reverse walk (`DslDump`).
//!
//! [`DslBuild`](crate::DslBuild) turns a validated [`ParseTree`] into a
//! typed AST value; this module is its inverse. A type deriving
//! `#[derive(DslDump)]` (in `dsl-kit-macros`) re-emits the tree shape
//! its own `DslBuild` derive accepts, so the two derives stay a
//! round-trip pair by construction:
//!
//! ```text
//! from_parse_tree(&ast.to_parse_tree()?, &ids)  ≙  ast   (modulo NodeId)
//! ```
//!
//! Chaining the emitted tree through
//! [`to_canonical_json`](crate::serde_bridge::to_canonical_json) yields
//! the serde-bridge JSON serializer downstream consumers want — see
//! [`dump_canonical_json`]. Because `to_canonical_json` runs
//! [`check_conformance`](crate::check_conformance) on the way down, a
//! derive bug that emitted a non-conforming shape surfaces as loud
//! diagnostics rather than silently wrong JSON.
//!
//! ## Emission rules (the duals of the build defaults)
//!
//! - `id: NodeId` is **not** serialized — re-parsing mints fresh ids,
//!   which is the same contract every front-end already has. The id is
//!   only used to look up the node's `$allow` annotation in the
//!   caller-supplied [`AllowTable`].
//! - An absent `Option<T>` payload / child omits its key entirely
//!   (never `null`), matching the canonical-JSON spelling and
//!   [`build_field_optional`](crate::build_field_optional)'s absent →
//!   `None` rule.
//! - `Vec<T>` payloads always emit, including as `[]` when empty —
//!   one unambiguous spelling. ([`build_field_vec`](crate::build_field_vec)
//!   accepts the omitted spelling too, so both round-trip; the emit
//!   side just never produces it.)
//! - `Vec<Self>` child slots always emit the slot, including empty.
//!   A slot declared `non_empty` fails conformance downstream in that
//!   case, which is the declared-constraint violation surfacing — not
//!   something to paper over here.
//! - Keyed slots (`BTreeMap`) emit in map iteration order, which is
//!   already ascending by key — the sortedness
//!   [`check_conformance`](crate::check_conformance) demands. Empty
//!   maps omit the slot (the PEG front-end's "nothing to record"
//!   spelling; both spellings are accepted everywhere it matters).
//! - Spans are `None`: a synthesized tree has no source text.
//!
//! One known asymmetry, inherited rather than introduced: nested
//! `Option<Option<T>>` payloads cannot distinguish `Some(None)` from
//! `None` in JSON (both spell as an absent key / `null`), exactly as on
//! the build side.

use std::collections::BTreeMap;

use dsl_kit_core::AllowTable;
use serde_json::Value;

use crate::serde_bridge::{serde_codes, to_canonical_json};
use crate::{BuildError, Diagnostic, ParseTree, RawValue};

// ---------------------------------------------------------------------------
// DslDump
// ---------------------------------------------------------------------------

/// Contract for re-emitting a typed AST value as the [`ParseTree`] its
/// [`DslBuild`](crate::DslBuild) impl accepts.
///
/// Implemented via `#[derive(DslDump)]` in `dsl-kit-macros`;
/// hand-written impls are fine for shapes the derive cannot express,
/// as long as they uphold the round-trip contract in the module docs.
pub trait DslDump {
    /// Emits the tree, attaching each node's `$allow` annotation looked
    /// up by its `NodeId` in `allows` (the table
    /// [`IdGen::take_allows`](dsl_kit_core::IdGen::take_allows)
    /// produced when the AST was built).
    ///
    /// `$allow` participates in canonical-JSON content hashes, so a
    /// caller that carried an [`AllowTable`] alongside the AST should
    /// hand it back here rather than calling [`DslDump::to_parse_tree`]
    /// and silently dropping the suppressions.
    fn to_parse_tree_with(&self, allows: &AllowTable) -> Result<ParseTree, BuildError>;

    /// Emits the tree with no `$allow` annotations (an empty
    /// [`AllowTable`]).
    fn to_parse_tree(&self) -> Result<ParseTree, BuildError> {
        self.to_parse_tree_with(&AllowTable::default())
    }
}

// ---------------------------------------------------------------------------
// Canonical-JSON convenience
// ---------------------------------------------------------------------------

/// Serializes a typed AST straight to canonical serde-bridge JSON:
/// [`DslDump::to_parse_tree`] chained through
/// [`to_canonical_json`](crate::serde_bridge::to_canonical_json)
/// against the type's own derived schema.
///
/// The output is what the JSON front-end
/// ([`from_json_value`](crate::serde_bridge::from_json_value)) parses
/// back into an equivalent AST — the wire form for caching, transport,
/// and content hashing of an AST a program built or transformed in
/// memory.
pub fn dump_canonical_json<T>(ast: &T) -> Result<Value, BuildError>
where
    T: DslDump + dsl_kit_schema::DslSchema,
{
    dump_canonical_json_with(ast, &AllowTable::default())
}

/// [`dump_canonical_json`] with an [`AllowTable`] so `$allow`
/// annotations survive into the JSON (and therefore into content
/// hashes computed over it).
pub fn dump_canonical_json_with<T>(ast: &T, allows: &AllowTable) -> Result<Value, BuildError>
where
    T: DslDump + dsl_kit_schema::DslSchema,
{
    to_canonical_json(&ast.to_parse_tree_with(allows)?, &T::schema())
}

// ---------------------------------------------------------------------------
// Field helpers (payload side)
// ---------------------------------------------------------------------------

fn dump_value_error(context: String, err: serde_json::Error) -> BuildError {
    BuildError::single(Diagnostic::error(
        serde_codes::DUMP_FIELD,
        format!("{context}: {err}"),
    ))
}

/// Serializes one payload field into the tree as
/// [`RawValue::Json`].
///
/// Dual of [`build_field`](crate::build_field): the emitted value is
/// what `serde_json::from_value` on the build side deserializes back.
/// Serialization failure (a map with non-string keys, a non-finite
/// float, a failing custom `Serialize`) reports
/// [`serde_codes::DUMP_FIELD`].
pub fn dump_field<T: serde::Serialize>(
    tree: &mut ParseTree,
    name: &str,
    value: &T,
) -> Result<(), BuildError> {
    let v = serde_json::to_value(value)
        .map_err(|e| dump_value_error(format!("field `{name}`"), e))?;
    tree.fields.push((name.to_string(), RawValue::Json(v)));
    Ok(())
}

/// Serializes an optional payload field, omitting the key when `None`.
///
/// Dual of [`build_field_optional`](crate::build_field_optional): the
/// absent key is the canonical spelling of `None`, so `Some(x)` emits
/// `x`'s value (never a JSON `null`).
pub fn dump_field_optional<T: serde::Serialize>(
    tree: &mut ParseTree,
    name: &str,
    value: &Option<T>,
) -> Result<(), BuildError> {
    match value {
        Some(v) => dump_field(tree, name, v),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Child helpers (recursive side)
// ---------------------------------------------------------------------------

/// Emits a [`Multiplicity::One`](dsl_kit_schema::Multiplicity::One)
/// child slot. Dual of [`build_child_one`](crate::build_child_one).
pub fn dump_child_one<T: DslDump>(
    tree: &mut ParseTree,
    name: &str,
    child: &T,
    allows: &AllowTable,
) -> Result<(), BuildError> {
    let sub = child.to_parse_tree_with(allows)?;
    tree.children.push((name.to_string(), vec![sub]));
    Ok(())
}

/// Emits a
/// [`Multiplicity::Optional`](dsl_kit_schema::Multiplicity::Optional)
/// child slot, omitting the slot entirely when `None`. Dual of
/// [`build_child_optional`](crate::build_child_optional).
///
/// Takes `Option<&T>` rather than `&Option<T>` so `Option<Box<T>>`
/// fields flatten in via `as_deref()`.
pub fn dump_child_optional<T: DslDump>(
    tree: &mut ParseTree,
    name: &str,
    child: Option<&T>,
    allows: &AllowTable,
) -> Result<(), BuildError> {
    match child {
        Some(c) => dump_child_one(tree, name, c, allows),
        None => Ok(()),
    }
}

/// Emits a [`Multiplicity::Many`](dsl_kit_schema::Multiplicity::Many)
/// child slot — always, including as an empty slot, so the emitted
/// spelling is unambiguous. Dual of
/// [`build_child_many`](crate::build_child_many).
///
/// Takes an iterator of `&T` so `Vec<T>` and `Vec<Box<T>>` fields both
/// feed in without an intermediate collect.
pub fn dump_child_many<'a, T, I>(
    tree: &mut ParseTree,
    name: &str,
    children: I,
    allows: &AllowTable,
) -> Result<(), BuildError>
where
    T: DslDump + 'a,
    I: IntoIterator<Item = &'a T>,
{
    let mut subs = Vec::new();
    for child in children {
        subs.push(child.to_parse_tree_with(allows)?);
    }
    tree.children.push((name.to_string(), subs));
    Ok(())
}

/// Emits a [`Multiplicity::Map`](dsl_kit_schema::Multiplicity::Map)
/// child slot from `(key, node)` entries, omitting the slot when
/// empty. Dual of [`build_child_map`](crate::build_child_map).
///
/// The entries must arrive in ascending key order —
/// [`BTreeMap::iter`] order, which is where the derive reads them
/// from — so the emitted tree passes the
/// [`KEYED_SLOT_UNSORTED`](crate::codes::KEYED_SLOT_UNSORTED)
/// conformance gate.
pub fn dump_child_map<'a, T, I>(
    tree: &mut ParseTree,
    name: &str,
    entries: I,
    allows: &AllowTable,
) -> Result<(), BuildError>
where
    T: DslDump + 'a,
    I: IntoIterator<Item = (&'a String, &'a T)>,
{
    let mut out = Vec::new();
    for (key, child) in entries {
        out.push((key.clone(), child.to_parse_tree_with(allows)?));
    }
    if !out.is_empty() {
        tree.keyed_children.push((name.to_string(), out));
    }
    Ok(())
}

/// Emits a scalar-valued keyed slot (`BTreeMap<String, V>` where `V`
/// is a payload type) as keyed entries each carrying a single `value`
/// field — the exact shape the JSON front-end builds on ingest and
/// [`build_scalar_map`](crate::build_scalar_map) reads back. Omits the
/// slot when the map is empty.
pub fn dump_scalar_map<V: serde::Serialize>(
    tree: &mut ParseTree,
    name: &str,
    entries: &BTreeMap<String, V>,
) -> Result<(), BuildError> {
    if entries.is_empty() {
        return Ok(());
    }
    let mut out = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        let v = serde_json::to_value(value)
            .map_err(|e| dump_value_error(format!("keyed slot `{name}` entry `{key}`"), e))?;
        let mut leaf = ParseTree::new("");
        leaf.fields.push(("value".to_string(), RawValue::Json(v)));
        out.push((key.clone(), leaf));
    }
    tree.keyed_children.push((name.to_string(), out));
    Ok(())
}
