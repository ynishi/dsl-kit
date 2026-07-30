//! Serde-JSON → [`ParseTree`] bridge.
//!
//! Front-end #1 of the two mentioned in the crate root: takes a
//! [`serde_json::Value`] shaped by the **internally-tagged `"type"`**
//! convention and produces the untyped [`ParseTree`] trunk. The other
//! front-end — a PEG interpreter — lands in G-2.
//!
//! ## Convention
//!
//! An AST node is an object with a `"type"` key naming its variant.
//! The remaining keys are dispatched against the target's
//! [`NodeSchema`]:
//!
//! - keys that name a declared
//!   [`FieldSchema`](dsl_kit_schema::FieldSchema) land in
//!   [`ParseTree::fields`] wrapped as [`RawValue::Json`] — no
//!   stringify round-trip;
//! - keys that name a declared
//!   [`ChildSchema`](dsl_kit_schema::ChildSchema) land in
//!   [`ParseTree::children`], recursively parsed as nested
//!   [`ParseTree`]s. Their value shape depends on the child's
//!   [`Multiplicity`]:
//!   - `One` → a single object (the child);
//!   - `Optional` → `null` or a single object;
//!   - `Many` → an array of objects (possibly empty);
//!   - `Map` → an object mapping keys to child objects (possibly
//!     empty), landing in [`ParseTree::keyed_children`] sorted by key.
//! - keys that name neither yield an [`codes::UNKNOWN_FIELD`]
//!   diagnostic (or, when the key is a declared child slot placed as a
//!   field / vice versa, the appropriate structural code).
//!
//! Structural errors during dispatch are collected before returning so
//! that a malformed document produces one bag of diagnostics rather
//! than a chain of stop-at-first failures.
//!
//! One object shape is reserved: `{"$import": "name"}` at a node
//! position produces an import placeholder for the load phase
//! ([`crate::import`]) instead of a variant node. `$import` must be
//! the object's only key and its value a literal string; anything else
//! is a [`crate::import::import_codes::SPEC_SHAPE`] diagnostic.
//!
//! ## Example
//!
//! ```ignore
//! use dsl_kit_parse::serde_bridge::from_json_value;
//! use dsl_kit_schema::DslSchema;
//!
//! let value = serde_json::json!({
//!     "type": "Add",
//!     "lhs": { "type": "Lit", "value": 1 },
//!     "rhs": { "type": "Lit", "value": 2 },
//! });
//! let tree = from_json_value(&value, &Expr::schema()).unwrap();
//! assert_eq!(tree.variant, "Add");
//! ```

use crate::import::{self, import_codes};
use crate::{BuildError, BuiltinLevenshteinSuggester, Diagnostic, ParseTree, RawValue, codes};
use dsl_kit_core::Suggester;
use dsl_kit_schema::{
    ChildSchema, ChildValueShape, Multiplicity, NodeSchema, ScalarKind, ScalarShorthand,
    VariantSchema,
};
use serde_json::Value;

/// Additional diagnostic codes emitted specifically by the serde
/// bridge (structural JSON-shape errors distinct from schema
/// conformance).
///
/// Reuses [`codes::UNKNOWN_VARIANT`] / [`codes::UNKNOWN_FIELD`] /
/// [`codes::FIELD_AS_CHILD`] / [`codes::CHILD_AS_FIELD`] where the
/// meaning is identical to conformance.
pub mod serde_codes {
    /// The document (or a nested child) was not a JSON object.
    pub const NOT_OBJECT: &str = "dsl_kit::parse::serde::not_object";
    /// The object lacks a `"type"` key.
    pub const TYPE_MISSING: &str = "dsl_kit::parse::serde::type_missing";
    /// The `"type"` key exists but is not a string.
    pub const TYPE_NOT_STRING: &str = "dsl_kit::parse::serde::type_not_string";
    /// A child slot's value shape did not match its
    /// [`Multiplicity`](dsl_kit_schema::Multiplicity) contract.
    pub const CHILD_SHAPE: &str = "dsl_kit::parse::serde::child_shape";
    /// A [`RawValue::Text`] payload (PEG front-end) could not be
    /// rendered as canonical JSON — its field type has no built-in
    /// text → JSON mapping, or the text failed to parse as that type.
    /// Emitted by [`to_canonical_json`](super::serde_bridge::to_canonical_json).
    pub const CANONICAL_TEXT: &str = "dsl_kit::parse::serde::canonical_text";
}

/// Parses a JSON string into a [`ParseTree`], using `schema` to
/// dispatch keys.
///
/// Convenience wrapper around [`from_json_value`] that surfaces
/// `serde_json::from_str` failures as a
/// [`serde_codes::NOT_OBJECT`]-tagged diagnostic (JSON syntax errors
/// have no `Span` mapping here — the parser's own position isn't
/// exposed by `serde_json` in a stable way).
pub fn from_json_str(input: &str, schema: &NodeSchema) -> Result<ParseTree, BuildError> {
    from_json_str_with(input, schema, &BuiltinLevenshteinSuggester)
}

/// Variant of [`from_json_str`] that routes `did you mean X?` hints
/// through a caller-supplied [`Suggester`]. See
/// [`from_json_value_with`] for the same wiring on
/// [`serde_json::Value`] inputs.
pub fn from_json_str_with(
    input: &str,
    schema: &NodeSchema,
    suggester: &dyn Suggester,
) -> Result<ParseTree, BuildError> {
    let value: Value = serde_json::from_str(input).map_err(|e| {
        BuildError::single(Diagnostic::error(
            serde_codes::NOT_OBJECT,
            format!("invalid JSON: {e}"),
        ))
    })?;
    from_json_value_with(&value, schema, suggester)
}

/// Parses a [`serde_json::Value`] into a [`ParseTree`] using `schema`
/// to dispatch each key into a payload field or a child slot.
///
/// Recurses on every child value using the same `schema` (the current
/// derive's recursion is same-type; heterogenous cross-type recursion
/// would need a resolver hook — not needed for G-1).
///
/// All diagnostics found during the walk are collected before
/// returning; a malformed document yields one [`BuildError`] with
/// every problem, not the first-encountered one.
pub fn from_json_value(value: &Value, schema: &NodeSchema) -> Result<ParseTree, BuildError> {
    from_json_value_with(value, schema, &BuiltinLevenshteinSuggester)
}

/// Variant of [`from_json_value`] that routes `did you mean X?` hints
/// through a caller-supplied [`Suggester`].
///
/// The free function [`from_json_value`] delegates here with the same
/// crate-private Levenshtein backend used by
/// [`crate::check_conformance`]. Reach for this variant to plug in a
/// different similarity algorithm (e.g. `dsl-kit-fuzzy`'s
/// `FuzzySuggester`) without touching the parse trunk contract.
pub fn from_json_value_with(
    value: &Value,
    schema: &NodeSchema,
    suggester: &dyn Suggester,
) -> Result<ParseTree, BuildError> {
    let mut diags = Vec::new();
    let tree = build_tree(value, schema, &mut diags, suggester);
    if diags.is_empty() {
        Ok(tree.unwrap_or_else(|| ParseTree::new("")))
    } else {
        Err(BuildError::new(diags))
    }
}

/// Core recursive builder. Returns `Some(tree)` when a well-shaped
/// object was found; `None` when the input was so malformed that no
/// meaningful trunk could be produced (still records diagnostics).
fn build_tree(
    value: &Value,
    schema: &NodeSchema,
    diags: &mut Vec<Diagnostic>,
    suggester: &dyn Suggester,
) -> Option<ParseTree> {
    let obj = match value {
        Value::Object(map) => map,
        _ => {
            diags.push(Diagnostic::error(
                serde_codes::NOT_OBJECT,
                format!(
                    "expected a JSON object with a `type` tag, got {}",
                    kind(value)
                ),
            ));
            return None;
        }
    };

    // `{"$import": "spec"}` at a node position becomes an import
    // placeholder for the load phase (`crate::import`). Checked before
    // the `type` dispatch so a placeholder needs no `type` key.
    if let Some(spec) = obj.get(import::IMPORT_VARIANT) {
        if obj.len() != 1 {
            diags.push(Diagnostic::error(
                import_codes::SPEC_SHAPE,
                format!(
                    "`{}` must be the object's only key ({} other key(s) present)",
                    import::IMPORT_VARIANT,
                    obj.len() - 1
                ),
            ));
            return None;
        }
        return match spec {
            Value::String(_) => {
                let mut tree = ParseTree::new(import::IMPORT_VARIANT);
                tree.fields.push((
                    import::IMPORT_SPEC_FIELD.to_string(),
                    RawValue::Json(spec.clone()),
                ));
                Some(tree)
            }
            other => {
                diags.push(Diagnostic::error(
                    import_codes::SPEC_SHAPE,
                    format!(
                        "`{}` value must be a literal string, got {}",
                        import::IMPORT_VARIANT,
                        kind(other)
                    ),
                ));
                None
            }
        };
    }

    let variant_name = match obj.get("type") {
        None => {
            diags.push(Diagnostic::error(
                serde_codes::TYPE_MISSING,
                "object is missing the `type` tag (expected a variant name)".to_string(),
            ));
            return None;
        }
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            diags.push(Diagnostic::error(
                serde_codes::TYPE_NOT_STRING,
                format!("`type` tag must be a string, got {}", kind(other)),
            ));
            return None;
        }
    };

    let variant = match schema.variant(&variant_name) {
        Some(v) => v,
        None => {
            let msg = crate::format_unknown_variant(&variant_name, schema, suggester);
            diags.push(Diagnostic::error(codes::UNKNOWN_VARIANT, msg));
            return None;
        }
    };

    let mut tree = ParseTree::new(variant_name);

    for (key, val) in obj {
        if key == "type" {
            continue;
        }
        if let Some(_field) = variant.fields.iter().find(|f| &f.name == key) {
            tree.fields.push((key.clone(), RawValue::Json(val.clone())));
        } else if let Some(child) = variant.children.iter().find(|c| &c.name == key) {
            if child.multiplicity == Multiplicity::Map {
                let entries = match &child.value_shape {
                    ChildValueShape::Scalar { .. } => {
                        build_scalar_keyed_slot(val, variant, key, diags)
                    }
                    ChildValueShape::Recursive => {
                        build_keyed_child_slot(val, variant, key, schema, diags, suggester)
                    }
                    // `#[non_exhaustive]` catch-all: a future
                    // `ChildValueShape` variant added to the schema
                    // crate before this bridge is extended lands
                    // here. Falls back to the recursive path
                    // (`build_keyed_child_slot`) so the front-end
                    // stays usable, but records the shape as
                    // unknown so the drift is visible in
                    // diagnostics rather than silent misroute.
                    _ => {
                        diags.push(Diagnostic::error(
                            codes::UNKNOWN_MULTIPLICITY,
                            format!(
                                "keyed child slot `{}` on variant `{}` declares a \
                                 ChildValueShape variant this build does not recognise \
                                 (upgrade dsl-kit-parse to a version that knows about it)",
                                key, variant.name
                            ),
                        ));
                        build_keyed_child_slot(val, variant, key, schema, diags, suggester)
                    }
                };
                tree.keyed_children.push((key.clone(), entries));
            } else {
                let subtrees = build_child_slot(child, val, variant, key, schema, diags, suggester);
                tree.children.push((key.clone(), subtrees));
            }
        } else {
            // Not a declared field, not a declared child — either a
            // shape mix-up (using a field name inside "children" via
            // this bridge isn't a thing, since we dispatch by name;
            // but a client may have typoed a variant slot). We only
            // have UNKNOWN_FIELD to describe "top-level key unknown"
            // in this dialect; that reads correctly here.
            let all_slots = crate::all_slot_names(variant);
            let hint = suggester.enrich_unknown(key, &all_slots);
            let msg = match hint {
                Some(h) => format!(
                    "unknown key `{}` on variant `{}` (not a declared field or child slot; {})",
                    key, variant.name, h
                ),
                None => format!(
                    "unknown key `{}` on variant `{}` (not a declared field or child slot)",
                    key, variant.name
                ),
            };
            diags.push(Diagnostic::error(codes::UNKNOWN_FIELD, msg));
        }
    }

    Some(tree)
}

/// JSON kind of a scalar value, for shorthand dispatch. `None` for
/// shapes no shorthand can accept (objects, arrays, null, floats).
fn scalar_kind_of(val: &Value) -> Option<ScalarKind> {
    match val {
        Value::String(_) => Some(ScalarKind::Str),
        Value::Number(n) if n.is_i64() || n.is_u64() => Some(ScalarKind::Int),
        Value::Bool(_) => Some(ScalarKind::Bool),
        _ => None,
    }
}

/// Lowers a bare scalar into the canonical node spelling of its
/// declared [`ScalarShorthand`] target. The produced tree is exactly
/// what [`build_tree`] would have produced for
/// `{"type": <variant>, <field>: <scalar>}` — the shorthand is an
/// input-side projection, invisible below the front-end.
fn lower_scalar_shorthand(val: &Value, shorthand: &ScalarShorthand) -> ParseTree {
    let mut tree = ParseTree::new(shorthand.variant.clone());
    tree.fields
        .push((shorthand.field.clone(), RawValue::Json(val.clone())));
    tree
}

/// Looks up the declared shorthand matching `val`'s scalar kind, if
/// the slot declares one.
fn shorthand_for<'a>(child: &'a ChildSchema, val: &Value) -> Option<&'a ScalarShorthand> {
    scalar_kind_of(val).and_then(|k| child.scalar_shorthand(k))
}

fn build_child_slot(
    child: &ChildSchema,
    val: &Value,
    variant: &VariantSchema,
    slot: &str,
    schema: &NodeSchema,
    diags: &mut Vec<Diagnostic>,
    suggester: &dyn Suggester,
) -> Vec<ParseTree> {
    match child.multiplicity {
        Multiplicity::One => match val {
            Value::Object(_) => build_tree(val, schema, diags, suggester)
                .into_iter()
                .collect(),
            _ => {
                if let Some(shorthand) = shorthand_for(child, val) {
                    return vec![lower_scalar_shorthand(val, shorthand)];
                }
                diags.push(Diagnostic::error(
                    serde_codes::CHILD_SHAPE,
                    format!(
                        "child slot `{}` on variant `{}` requires exactly one object, got {}",
                        slot,
                        variant.name,
                        kind(val)
                    ),
                ));
                Vec::new()
            }
        },
        Multiplicity::Optional => match val {
            Value::Null => Vec::new(),
            Value::Object(_) => build_tree(val, schema, diags, suggester)
                .into_iter()
                .collect(),
            _ => {
                if let Some(shorthand) = shorthand_for(child, val) {
                    return vec![lower_scalar_shorthand(val, shorthand)];
                }
                diags.push(Diagnostic::error(
                    serde_codes::CHILD_SHAPE,
                    format!(
                        "child slot `{}` on variant `{}` requires null or an object, got {}",
                        slot,
                        variant.name,
                        kind(val)
                    ),
                ));
                Vec::new()
            }
        },
        Multiplicity::Many => match val {
            Value::Array(items) => items
                .iter()
                .filter_map(|item| build_tree(item, schema, diags, suggester))
                .collect(),
            _ => {
                diags.push(Diagnostic::error(
                    serde_codes::CHILD_SHAPE,
                    format!(
                        "child slot `{}` on variant `{}` requires an array of objects, got {}",
                        slot,
                        variant.name,
                        kind(val)
                    ),
                ));
                Vec::new()
            }
        },
        Multiplicity::Map => {
            // Keyed slots are routed to `build_keyed_child_slot` by
            // `build_tree` before reaching here, because their result
            // shape (`(key, tree)` pairs) does not fit this function's
            // return type. Arriving in this arm means that dispatch
            // lost sync with the schema — surface it rather than
            // silently dropping every entry.
            diags.push(Diagnostic::error(
                serde_codes::CHILD_SHAPE,
                format!(
                    "child slot `{}` on variant `{}` declares Multiplicity::Map but was \
                     routed through the positional builder",
                    slot, variant.name
                ),
            ));
            Vec::new()
        }
        // `#[non_exhaustive]` catch-all — see the sibling arm in
        // `check_children` (crate::lib.rs) for the versioning
        // rationale. Reuses the crate-level slug because the failure
        // mode ("this build of parse doesn't know about that
        // Multiplicity variant") is the same across all pipeline
        // stages.
        _ => {
            diags.push(Diagnostic::error(
                codes::UNKNOWN_MULTIPLICITY,
                format!(
                    "child slot `{}` on variant `{}` uses an unrecognised \
                     Multiplicity variant (upgrade dsl-kit-parse to a version \
                     that knows about it)",
                    slot, variant.name
                ),
            ));
            Vec::new()
        }
    }
}

/// Builds a keyed child slot ([`Multiplicity::Map`]) from a JSON
/// object mapping keys to child node objects.
///
/// The pairs are sorted by key before returning so that
/// [`ParseTree::keyed_children`] carries one canonical order
/// regardless of how the JSON was written, and regardless of whether
/// `serde_json` was built with the `preserve_order` feature (which
/// swaps its object map from a sorted `BTreeMap` to an
/// insertion-ordered one). Sorting here is what keeps the bridge on
/// the right side of the ordering contract
/// [`check_conformance`](crate::check_conformance) enforces
/// ([`crate::codes::KEYED_SLOT_UNSORTED`]) — a JSON document can
/// never trip it.
///
/// Duplicate keys cannot survive `serde_json`'s own object parsing
/// (the later one wins before the bridge ever sees the document), so
/// [`crate::codes::DUPLICATE_KEY`] is raised by hand-built trees and
/// by future non-JSON front-ends, not here.
fn build_keyed_child_slot(
    val: &Value,
    variant: &VariantSchema,
    slot: &str,
    schema: &NodeSchema,
    diags: &mut Vec<Diagnostic>,
    suggester: &dyn Suggester,
) -> Vec<(String, ParseTree)> {
    let Value::Object(map) = val else {
        diags.push(Diagnostic::error(
            serde_codes::CHILD_SHAPE,
            format!(
                "keyed child slot `{}` on variant `{}` requires an object mapping keys \
                 to child objects, got {}",
                slot,
                variant.name,
                kind(val)
            ),
        ));
        return Vec::new();
    };

    let mut out = Vec::with_capacity(map.len());
    for (key, item) in map {
        match item {
            Value::Object(_) => {
                if let Some(child) = build_tree(item, schema, diags, suggester) {
                    out.push((key.clone(), child));
                }
            }
            _ => diags.push(Diagnostic::error(
                serde_codes::CHILD_SHAPE,
                format!(
                    "keyed child slot `{}` on variant `{}`: entry `{}` requires an object, got {}",
                    slot,
                    variant.name,
                    key,
                    kind(item)
                ),
            )),
        }
    }
    out.sort_by(|(a, _), (b, _)| a.cmp(b));
    out
}

/// Builds a keyed child slot ([`Multiplicity::Map`]) whose values are
/// scalars (Shape 1 of the tracking issue). Each entry is wrapped as
/// a leaf [`ParseTree`] carrying its payload under the well-known
/// `"value"` field, so [`build_scalar_map`](crate::build_scalar_map)
/// can read it back with the same `build_field` route the rest of the
/// derive already uses.
///
/// JSON scalars land as [`RawValue::Json`] wholesale — `serde_json`
/// distinguishes strings, numbers, booleans, and null on its own, and
/// the build layer's [`build_field`](crate::build_field) delegates to
/// `serde_json::from_value` for that arm. That means a `String` /
/// `i64` / `bool` scalar type all deserialize without a stringify
/// round-trip. Nested arrays or objects on the value side are handed
/// straight through so a scalar map declared as
/// `BTreeMap<String, Vec<String>>` (still a scalar shape at the
/// schema level — the value is a payload type, not another AST) also
/// deserializes cleanly.
///
/// Sorted by key on emit, matching `build_keyed_child_slot` so both
/// arms feed [`ParseTree::keyed_children`] on the same canonical
/// ordering contract enforced by
/// [`check_conformance`](crate::check_conformance).
fn build_scalar_keyed_slot(
    val: &Value,
    variant: &VariantSchema,
    slot: &str,
    diags: &mut Vec<Diagnostic>,
) -> Vec<(String, ParseTree)> {
    let Value::Object(map) = val else {
        diags.push(Diagnostic::error(
            serde_codes::CHILD_SHAPE,
            format!(
                "scalar keyed child slot `{}` on variant `{}` requires an object mapping keys \
                 to scalar values, got {}",
                slot,
                variant.name,
                kind(val)
            ),
        ));
        return Vec::new();
    };

    let mut out = Vec::with_capacity(map.len());
    for (key, item) in map {
        let mut leaf = ParseTree::new("");
        leaf.fields
            .push(("value".into(), RawValue::Json(item.clone())));
        out.push((key.clone(), leaf));
    }
    out.sort_by(|(a, _), (b, _)| a.cmp(b));
    out
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

// ---------------------------------------------------------------------------
// Canonical JSON dump
// ---------------------------------------------------------------------------

/// Renders a [`ParseTree`] as **canonical JSON**: the long-form object
/// spelling with every input-side shorthand expanded.
///
/// Canonical means one shape per meaning:
///
/// - every node is the explicit `{"type": <variant>, ...}` object —
///   a tree lowered from a scalar shorthand renders identically to
///   one parsed from the explicit spelling, because the lowering
///   happened at the front-end and this function never sees the
///   difference;
/// - `Optional` slots and absent optional payloads omit their key
///   (never `null`);
/// - keyed slots render as objects in the tree's sorted-by-key order;
/// - object key order in the output is `serde_json`'s map order,
///   which is deterministic for a given tree.
///
/// Two documents that parse (through either front-end) to equal
/// meaning therefore serialize to the same `Value` — which makes this
/// output the right input for content hashing: hash
/// `serde_json::to_string(&to_canonical_json(...)?)` instead of the
/// surface bytes, and adding a shorthand spelling for an existing
/// value never invalidates a document's hash.
///
/// The tree is conformance-checked level by level on the way down;
/// a non-conforming tree returns those diagnostics unchanged.
/// [`RawValue::Text`] payloads (PEG front-end) are converted for the
/// built-in canonical-syntax types (`String`, the integer types,
/// `bool`, `Option<String>` — where the literal `none` omits the
/// key — and `Vec<String>`); other text payload types report
/// [`serde_codes::CANONICAL_TEXT`].
pub fn to_canonical_json(tree: &ParseTree, schema: &NodeSchema) -> Result<Value, BuildError> {
    let mut diags = Vec::new();
    let value = canonical_node(tree, schema, &mut diags);
    match value {
        Some(v) if diags.is_empty() => Ok(v),
        _ => {
            if diags.is_empty() {
                // Structural miss that conformance did not cover (e.g. a
                // scalar keyed entry with no `value` field) — keep the
                // error bag non-empty so the failure stays loud.
                diags.push(Diagnostic::error(
                    serde_codes::CANONICAL_TEXT,
                    "tree shape not renderable as canonical JSON".to_string(),
                ));
            }
            Err(BuildError::new(diags))
        }
    }
}

fn canonical_node(
    tree: &ParseTree,
    schema: &NodeSchema,
    diags: &mut Vec<Diagnostic>,
) -> Option<Value> {
    let level = crate::check_conformance(tree, schema);
    if !level.is_empty() {
        diags.extend(level);
        return None;
    }
    // Conformance passed, so the variant, every field name, and every
    // slot name below are declared; lookups can only miss on a bug in
    // `check_conformance` itself, which the `?`s would surface as an
    // (empty-key) omission rather than a panic.
    let variant = schema.variant(&tree.variant)?;
    let mut obj = serde_json::Map::new();
    obj.insert("type".to_string(), Value::String(tree.variant.clone()));

    for (name, raw) in &tree.fields {
        let field = variant.fields.iter().find(|f| &f.name == name)?;
        match raw {
            RawValue::Json(v) => {
                obj.insert(name.clone(), v.clone());
            }
            RawValue::Text(text) => match canonical_json_from_text(text, &field.ty) {
                Ok(Some(v)) => {
                    obj.insert(name.clone(), v);
                }
                Ok(None) => {} // absent optional — canonical form omits the key
                Err(reason) => {
                    diags.push(Diagnostic::error(
                        serde_codes::CANONICAL_TEXT,
                        format!("field `{}` on variant `{}`: {}", name, variant.name, reason),
                    ));
                }
            },
        }
    }

    for (slot, subtrees) in &tree.children {
        let child = variant.children.iter().find(|c| &c.name == slot)?;
        match child.multiplicity {
            Multiplicity::One => {
                let v = canonical_node(&subtrees[0], schema, diags)?;
                obj.insert(slot.clone(), v);
            }
            Multiplicity::Optional => {
                if let Some(sub) = subtrees.first() {
                    let v = canonical_node(sub, schema, diags)?;
                    obj.insert(slot.clone(), v);
                }
            }
            _ => {
                let items: Option<Vec<Value>> = subtrees
                    .iter()
                    .map(|sub| canonical_node(sub, schema, diags))
                    .collect();
                obj.insert(slot.clone(), Value::Array(items?));
            }
        }
    }

    for (slot, entries) in &tree.keyed_children {
        let child = variant.children.iter().find(|c| &c.name == slot)?;
        let mut map = serde_json::Map::new();
        for (key, entry) in entries {
            let v = match &child.value_shape {
                ChildValueShape::Scalar { ty } => match entry.field("value") {
                    Some(RawValue::Json(v)) => Some(v.clone()),
                    Some(RawValue::Text(text)) => match canonical_json_from_text(text, ty) {
                        Ok(v) => v,
                        Err(reason) => {
                            diags.push(Diagnostic::error(
                                serde_codes::CANONICAL_TEXT,
                                format!(
                                    "keyed slot `{}` entry `{}` on variant `{}`: {}",
                                    slot, key, variant.name, reason
                                ),
                            ));
                            None
                        }
                    },
                    None => None,
                },
                _ => canonical_node(entry, schema, diags),
            };
            map.insert(key.clone(), v?);
        }
        obj.insert(slot.clone(), Value::Object(map));
    }

    Some(Value::Object(obj))
}

/// Converts a [`RawValue::Text`] payload to its canonical JSON value
/// by the field's Rust type source text. `Ok(None)` means the field
/// is canonically *absent* (an optional payload spelled `none`).
/// Covers the same built-in type set as `schema_gen`'s canonical
/// syntax mapping, so any text a generated grammar can produce for
/// these types converts back.
fn canonical_json_from_text(text: &str, ty: &str) -> Result<Option<Value>, String> {
    const INT_TYPES: &[&str] = &[
        "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "usize", "isize",
    ];
    let ty: String = ty.chars().filter(|c| !c.is_whitespace()).collect();
    if ty == "String" {
        Ok(Some(Value::String(text.to_string())))
    } else if INT_TYPES.contains(&ty.as_str()) {
        if let Ok(n) = text.parse::<i64>() {
            Ok(Some(Value::Number(n.into())))
        } else if let Ok(n) = text.parse::<u64>() {
            Ok(Some(Value::Number(n.into())))
        } else {
            Err(format!("text `{text}` does not parse as `{ty}`"))
        }
    } else if ty == "bool" {
        match text {
            "true" => Ok(Some(Value::Bool(true))),
            "false" => Ok(Some(Value::Bool(false))),
            _ => Err(format!("text `{text}` does not parse as `bool`")),
        }
    } else if ty == "Option<String>" {
        // Mirrors `build_field_optional`: the literal `none` is the
        // absent spelling, anything else is the payload itself.
        if text == "none" {
            Ok(None)
        } else {
            Ok(Some(Value::String(text.to_string())))
        }
    } else if ty == "Vec<String>" {
        // The generated grammar contributes a JSON-compatible array
        // literal as the field text (see `field_value_peg`).
        serde_json::from_str::<Value>(text)
            .map(Some)
            .map_err(|e| format!("text `{text}` does not parse as `Vec<String>`: {e}"))
    } else {
        Err(format!(
            "type `{ty}` has no canonical text → JSON mapping (supported: String, the \
             integer types, bool, Option<String>, Vec<String>)"
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check_conformance, codes};
    use dsl_kit_schema::{
        ChildSchema, ChildValueShape, FieldSchema, Multiplicity, NodeSchema, VariantSchema,
    };
    use serde_json::json;

    fn schema() -> NodeSchema {
        NodeSchema {
            name: "Expr".into(),
            variants: vec![
                VariantSchema {
                    name: "Lit".into(),
                    fields: vec![FieldSchema {
                        name: "value".into(),
                        ty: "i64".into(),
                        optional: false,
                    }],
                    children: vec![],
                },
                VariantSchema {
                    name: "Add".into(),
                    fields: vec![],
                    children: vec![
                        ChildSchema {
                            name: "lhs".into(),
                            multiplicity: Multiplicity::One,
                            value_shape: ChildValueShape::Recursive,
                            scalar_shorthands: vec![],
                        },
                        ChildSchema {
                            name: "rhs".into(),
                            multiplicity: Multiplicity::One,
                            value_shape: ChildValueShape::Recursive,
                            scalar_shorthands: vec![],
                        },
                    ],
                },
                VariantSchema {
                    name: "Let".into(),
                    fields: vec![FieldSchema {
                        name: "name".into(),
                        ty: "String".into(),
                        optional: false,
                    }],
                    children: vec![
                        ChildSchema {
                            name: "value".into(),
                            multiplicity: Multiplicity::One,
                            value_shape: ChildValueShape::Recursive,
                            scalar_shorthands: vec![],
                        },
                        ChildSchema {
                            name: "body".into(),
                            multiplicity: Multiplicity::One,
                            value_shape: ChildValueShape::Recursive,
                            scalar_shorthands: vec![],
                        },
                    ],
                },
                VariantSchema {
                    name: "Group".into(),
                    fields: vec![],
                    children: vec![ChildSchema {
                        name: "items".into(),
                        multiplicity: Multiplicity::Many,
                        value_shape: ChildValueShape::Recursive,
                        scalar_shorthands: vec![],
                    }],
                },
                VariantSchema {
                    name: "Wrap".into(),
                    fields: vec![],
                    children: vec![ChildSchema {
                        name: "inner".into(),
                        multiplicity: Multiplicity::Optional,
                        value_shape: ChildValueShape::Recursive,
                        scalar_shorthands: vec![],
                    }],
                },
            ],
        }
    }

    #[test]
    fn simple_lit() {
        let tree = from_json_value(&json!({ "type": "Lit", "value": 7 }), &schema()).unwrap();
        assert_eq!(tree.variant, "Lit");
        assert_eq!(tree.fields.len(), 1);
        assert_eq!(tree.children.len(), 0);
        assert!(matches!(
            tree.field("value").unwrap(),
            RawValue::Json(v) if v.as_i64() == Some(7)
        ));
        assert!(check_conformance(&tree, &schema()).is_empty());
    }

    #[test]
    fn nested_add() {
        let value = json!({
            "type": "Add",
            "lhs": { "type": "Lit", "value": 1 },
            "rhs": { "type": "Lit", "value": 2 },
        });
        let tree = from_json_value(&value, &schema()).unwrap();
        assert_eq!(tree.variant, "Add");
        assert_eq!(tree.child_slot("lhs").unwrap().len(), 1);
        assert_eq!(tree.child_slot("rhs").unwrap().len(), 1);
        assert!(check_conformance(&tree, &schema()).is_empty());
    }

    #[test]
    fn many_slot_from_array() {
        let value = json!({
            "type": "Group",
            "items": [
                { "type": "Lit", "value": 1 },
                { "type": "Lit", "value": 2 },
                { "type": "Lit", "value": 3 },
            ],
        });
        let tree = from_json_value(&value, &schema()).unwrap();
        assert_eq!(tree.child_slot("items").unwrap().len(), 3);
    }

    #[test]
    fn optional_slot_from_null_is_empty() {
        let tree = from_json_value(&json!({ "type": "Wrap", "inner": null }), &schema()).unwrap();
        assert_eq!(tree.child_slot("inner").unwrap().len(), 0);
    }

    #[test]
    fn optional_slot_from_object_is_one() {
        let tree = from_json_value(
            &json!({ "type": "Wrap", "inner": { "type": "Lit", "value": 9 } }),
            &schema(),
        )
        .unwrap();
        assert_eq!(tree.child_slot("inner").unwrap().len(), 1);
    }

    #[test]
    fn unknown_variant_lists_candidates() {
        let err = from_json_value(&json!({ "type": "Ad" }), &schema()).unwrap_err();
        assert_eq!(err.diagnostics.len(), 1);
        assert_eq!(err.diagnostics[0].code, codes::UNKNOWN_VARIANT);
        assert!(err.diagnostics[0].message.contains("Add"));
    }

    #[test]
    fn unknown_key_suggests_declared_slot() {
        // The serde-bridge front-end mirrors the check_conformance
        // pair-hint: a typo of a declared field name gets `did you
        // mean` on its UNKNOWN_FIELD diagnostic.
        let value = json!({ "type": "Lit", "vlue": 1 });
        let err = from_json_value(&value, &schema()).unwrap_err();
        let d = err
            .diagnostics
            .iter()
            .find(|d| d.code == codes::UNKNOWN_FIELD)
            .expect("expected UNKNOWN_FIELD");
        assert!(
            d.message.contains("did you mean") && d.message.contains("value"),
            "expected `value` in the hint, got: {}",
            d.message
        );
    }

    #[test]
    fn missing_type_tag_reports() {
        let err = from_json_value(&json!({ "value": 1 }), &schema()).unwrap_err();
        assert_eq!(err.diagnostics[0].code, serde_codes::TYPE_MISSING);
    }

    #[test]
    fn type_not_string_reports() {
        let err = from_json_value(&json!({ "type": 42 }), &schema()).unwrap_err();
        assert_eq!(err.diagnostics[0].code, serde_codes::TYPE_NOT_STRING);
    }

    #[test]
    fn non_object_at_root_reports() {
        let err = from_json_value(&json!(42), &schema()).unwrap_err();
        assert_eq!(err.diagnostics[0].code, serde_codes::NOT_OBJECT);
    }

    #[test]
    fn many_slot_from_object_fails_shape() {
        let value = json!({
            "type": "Group",
            "items": { "type": "Lit", "value": 1 },
        });
        let err = from_json_value(&value, &schema()).unwrap_err();
        assert!(
            err.diagnostics
                .iter()
                .any(|d| d.code == serde_codes::CHILD_SHAPE)
        );
    }

    #[test]
    fn one_slot_missing_reports_via_conformance() {
        // Bridge accepts the doc; conformance catches the missing rhs.
        let value = json!({
            "type": "Add",
            "lhs": { "type": "Lit", "value": 1 },
        });
        let tree = from_json_value(&value, &schema()).unwrap();
        let diags = check_conformance(&tree, &schema());
        assert!(diags.iter().any(|d| d.code == codes::ARITY_ONE));
    }

    #[test]
    fn unknown_key_reports() {
        let value = json!({ "type": "Lit", "value": 1, "extra": 2 });
        let err = from_json_value(&value, &schema()).unwrap_err();
        assert_eq!(err.diagnostics[0].code, codes::UNKNOWN_FIELD);
    }

    #[test]
    fn multiple_errors_collected() {
        // Two Adds, each missing rhs entirely — bridge accepts, but
        // give an unknown key too so bridge sees it. Nested collection
        // proves diagnostics from recursion bubble up.
        let value = json!({
            "type": "Add",
            "lhs": { "type": "Lit", "value": 1, "bogus": 0 },
            "rhs": { "type": "Nope" },
        });
        let err = from_json_value(&value, &schema()).unwrap_err();
        // Expect: one UNKNOWN_FIELD (bogus) + one UNKNOWN_VARIANT (Nope).
        assert!(
            err.diagnostics
                .iter()
                .any(|d| d.code == codes::UNKNOWN_FIELD)
        );
        assert!(
            err.diagnostics
                .iter()
                .any(|d| d.code == codes::UNKNOWN_VARIANT)
        );
    }

    #[test]
    fn from_json_str_wraps_syntax_errors() {
        let err = from_json_str("{ not: valid }", &schema()).unwrap_err();
        assert_eq!(err.diagnostics[0].code, serde_codes::NOT_OBJECT);
    }
}
