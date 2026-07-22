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
//!   - `Many` → an array of objects (possibly empty).
//! - keys that name neither yield an [`codes::UNKNOWN_FIELD`]
//!   diagnostic (or, when the key is a declared child slot placed as a
//!   field / vice versa, the appropriate structural code).
//!
//! Structural errors during dispatch are collected before returning so
//! that a malformed document produces one bag of diagnostics rather
//! than a chain of stop-at-first failures.
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

use crate::{BuildError, BuiltinLevenshteinSuggester, Diagnostic, ParseTree, RawValue, codes};
use dsl_kit_core::Suggester;
use dsl_kit_schema::{Multiplicity, NodeSchema, VariantSchema};
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
            let subtrees = build_child_slot(
                child.multiplicity,
                val,
                variant,
                key,
                schema,
                diags,
                suggester,
            );
            tree.children.push((key.clone(), subtrees));
        } else {
            // Not a declared field, not a declared child — either a
            // shape mix-up (using a field name inside "children" via
            // this bridge isn't a thing, since we dispatch by name;
            // but a client may have typoed a variant slot). We only
            // have UNKNOWN_FIELD to describe "top-level key unknown"
            // in this dialect; that reads correctly here.
            diags.push(Diagnostic::error(
                codes::UNKNOWN_FIELD,
                format!(
                    "unknown key `{}` on variant `{}` (not a declared field or child slot)",
                    key, variant.name
                ),
            ));
        }
    }

    Some(tree)
}

fn build_child_slot(
    m: Multiplicity,
    val: &Value,
    variant: &VariantSchema,
    slot: &str,
    schema: &NodeSchema,
    diags: &mut Vec<Diagnostic>,
    suggester: &dyn Suggester,
) -> Vec<ParseTree> {
    match m {
        Multiplicity::One => match val {
            Value::Object(_) => build_tree(val, schema, diags, suggester)
                .into_iter()
                .collect(),
            _ => {
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
    }
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check_conformance, codes};
    use dsl_kit_schema::{ChildSchema, FieldSchema, Multiplicity, NodeSchema, VariantSchema};
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
                        },
                        ChildSchema {
                            name: "rhs".into(),
                            multiplicity: Multiplicity::One,
                        },
                    ],
                },
                VariantSchema {
                    name: "Let".into(),
                    fields: vec![FieldSchema {
                        name: "name".into(),
                        ty: "String".into(),
                    }],
                    children: vec![
                        ChildSchema {
                            name: "value".into(),
                            multiplicity: Multiplicity::One,
                        },
                        ChildSchema {
                            name: "body".into(),
                            multiplicity: Multiplicity::One,
                        },
                    ],
                },
                VariantSchema {
                    name: "Group".into(),
                    fields: vec![],
                    children: vec![ChildSchema {
                        name: "items".into(),
                        multiplicity: Multiplicity::Many,
                    }],
                },
                VariantSchema {
                    name: "Wrap".into(),
                    fields: vec![],
                    children: vec![ChildSchema {
                        name: "inner".into(),
                        multiplicity: Multiplicity::Optional,
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
