//! JSON Schema (2020-12 vocabulary) rendering of a [`NodeSchema`].
//!
//! [`NodeSchema::to_json`] is the kit's own layout, for tooling that
//! knows the kit. Consumers who publish a DSL as an *external*
//! interface — an OpenAPI request body, an editor's JSON validation, an
//! AI client's structured-output schema — need the standard dialect.
//! [`to_json_schema`] renders it from the same derived schema, so the
//! published contract cannot drift from the enum.
//!
//! ## What the output describes
//!
//! The document shape `dsl-kit-parse`'s serde bridge accepts: an
//! internally tagged object, `{"type": "<Variant>", <field>: …,
//! <child slot>: …}`. Rendered as one `oneOf` arm per variant —
//!
//! - `type` pinned with `const`, `discriminator.propertyName = "type"`
//!   on the union for OpenAPI consumers;
//! - `additionalProperties: false`, and every key whose absence the
//!   conformance check rejects listed in `required`;
//! - payload fields by Rust type source text, through the built-in
//!   mapping (the same set the canonical text grammar spells:
//!   `String`, the integer types, `bool`, `Option<..>` — nullable and
//!   not required — and `Vec<String>`), then a caller-supplied
//!   [`TypeMap`], else a loud [`Error::UnsupportedField`];
//! - child slots recursing through a `$ref` whose target
//!   [`RefStyle`] chooses:
//!
//! | multiplicity | rendering |
//! |---|---|
//! | `One` | `$ref`, required |
//! | `Optional` | `oneOf: [$ref, null]`, not required |
//! | `Many` | `array` of `$ref`; `minItems: 1` + required when `non_empty` |
//! | `Map` | `object` with `additionalProperties`; `minProperties: 1` + required when `non_empty` |
//!
//! Scalar-valued keyed slots map their value type like a payload
//! field. A declared [`ScalarShorthand`](crate::ScalarShorthand) adds the bare scalar as an
//! extra alternative on its slot, because the bridge accepts it there.
//!
//! ## Example
//!
//! ```
//! use dsl_kit_schema::json_schema::{RefStyle, TypeMap};
//! use dsl_kit_schema::{ChildSchema, FieldSchema, Multiplicity, NodeSchema, VariantSchema};
//!
//! let schema = NodeSchema {
//!     name: "Expr".into(),
//!     variants: vec![
//!         VariantSchema {
//!             name: "Lit".into(),
//!             fields: vec![FieldSchema::required("value", "i64")],
//!             children: vec![],
//!         },
//!         VariantSchema {
//!             name: "Neg".into(),
//!             fields: vec![],
//!             children: vec![ChildSchema::recursive("body", Multiplicity::One)],
//!         },
//!     ],
//! };
//! let doc = schema.to_json_schema(&TypeMap::new(), RefStyle::Defs).unwrap();
//! assert_eq!(doc["$ref"], "#/$defs/Expr");
//! assert_eq!(doc["$defs"]["Expr"]["oneOf"][1]["properties"]["body"]["$ref"], "#/$defs/Expr");
//! ```

use std::collections::BTreeMap;
use std::fmt;

use serde_json::{Map, Value, json};

use crate::{ChildValueShape, FieldSchema, Multiplicity, NodeSchema, ScalarKind, VariantSchema};

/// JSON Schema fragments for payload types outside the built-in
/// mapping, keyed by Rust type source text.
///
/// The JSON-Schema twin of `dsl-kit-parse`'s `SyntaxOverrides`: where
/// that supplies the text production for a type the canonical syntax
/// cannot spell, this supplies what the JSON front-end accepts for it.
/// Whitespace in the type text is ignored, so `"Option<JoinPolicy>"`
/// matches the derive-extracted `"Option < JoinPolicy >"` spelling.
#[derive(Debug, Default, Clone)]
pub struct TypeMap {
    by_type: BTreeMap<String, Value>,
}

impl TypeMap {
    /// An empty map — the built-in mapping only.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `schema` as the fragment for payload type `ty`,
    /// replacing any earlier entry for the same type.
    pub fn with(mut self, ty: impl AsRef<str>, schema: Value) -> Self {
        self.by_type.insert(strip_ws(ty.as_ref()), schema);
        self
    }

    /// The registered fragment for `ty`, if any.
    pub fn get(&self, ty: &str) -> Option<&Value> {
        self.by_type.get(&strip_ws(ty))
    }
}

/// Where the recursive `$ref`s point, and how the document is wrapped.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RefStyle {
    /// A standalone document: `{"$schema": …, "$ref": "#/$defs/<name>",
    /// "$defs": {<name>: <union>}}`. Recursion targets `#/$defs/<name>`.
    Defs,
    /// An OpenAPI component: the bare union, ready to be placed at
    /// `components.schemas.<name>`. Recursion targets
    /// `#/components/schemas/<name>`.
    OpenApiComponent,
}

/// Rendering failures.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A payload field (or scalar keyed-slot value) has a Rust type
    /// with neither a built-in mapping nor a [`TypeMap`] entry.
    UnsupportedField {
        /// Variant carrying the field.
        variant: String,
        /// Field or slot name.
        field: String,
        /// Rust type source text as the schema reports it.
        ty: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnsupportedField { variant, field, ty } => write!(
                f,
                "no JSON Schema mapping for `{variant}.{field}: {ty}` — register it in TypeMap"
            ),
        }
    }
}

impl std::error::Error for Error {}

const INT_TYPES: &[&str] = &[
    "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize",
];

fn strip_ws(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Built-in payload type mapping — the set the canonical text grammar
/// also spells without overrides, so the two front-ends and the
/// published schema agree on what a field accepts.
fn builtin(ty: &str) -> Option<Value> {
    let ty = strip_ws(ty);
    if ty == "String" {
        Some(json!({ "type": "string" }))
    } else if INT_TYPES.contains(&ty.as_str()) {
        Some(json!({ "type": "integer" }))
    } else if ty == "bool" {
        Some(json!({ "type": "boolean" }))
    } else if ty == "Vec<String>" {
        Some(json!({ "type": "array", "items": { "type": "string" } }))
    } else if let Some(inner) = ty.strip_prefix("Option<").and_then(|s| s.strip_suffix('>')) {
        // The absent key is the canonical `None`; the bridge also
        // accepts an explicit `null`.
        let inner = builtin(inner)?;
        Some(json!({ "oneOf": [inner, { "type": "null" }] }))
    } else {
        None
    }
}

fn payload(variant: &str, name: &str, ty: &str, types: &TypeMap) -> Result<Value, Error> {
    if let Some(v) = types.get(ty) {
        return Ok(v.clone());
    }
    builtin(ty).ok_or_else(|| Error::UnsupportedField {
        variant: variant.to_owned(),
        field: name.to_owned(),
        ty: ty.to_owned(),
    })
}

fn scalar_kind(kind: ScalarKind) -> Value {
    match kind {
        ScalarKind::Str => json!({ "type": "string" }),
        ScalarKind::Int => json!({ "type": "integer" }),
        ScalarKind::Bool => json!({ "type": "boolean" }),
    }
}

fn self_ref(schema: &NodeSchema, style: &RefStyle) -> Value {
    let target = match style {
        RefStyle::Defs => format!("#/$defs/{}", schema.name),
        RefStyle::OpenApiComponent => format!("#/components/schemas/{}", schema.name),
    };
    json!({ "$ref": target })
}

fn field_entry(variant: &str, f: &FieldSchema, types: &TypeMap) -> Result<Value, Error> {
    payload(variant, &f.name, &f.ty, types)
}

fn variant_arm(
    schema: &NodeSchema,
    v: &VariantSchema,
    types: &TypeMap,
    style: &RefStyle,
) -> Result<Value, Error> {
    let mut props = Map::new();
    let mut required = vec![Value::String("type".into())];
    props.insert("type".into(), json!({ "const": v.name }));

    for f in &v.fields {
        props.insert(f.name.clone(), field_entry(&v.name, f, types)?);
        if !f.optional {
            required.push(Value::String(f.name.clone()));
        }
    }

    for c in &v.children {
        let element = match &c.value_shape {
            ChildValueShape::Recursive => self_ref(schema, style),
            ChildValueShape::Scalar { ty } => payload(&v.name, &c.name, ty, types)?,
        };
        // A declared scalar shorthand lets the bridge accept a bare
        // scalar at the slot position; it is a valid document there.
        let with_shorthands = |element: Value| -> Value {
            if c.scalar_shorthands.is_empty() {
                return element;
            }
            let mut alts = vec![element];
            alts.extend(c.scalar_shorthands.iter().map(|s| scalar_kind(s.kind)));
            json!({ "oneOf": alts })
        };
        let (slot, req) = match c.multiplicity {
            Multiplicity::One => (with_shorthands(element), true),
            Multiplicity::Optional => {
                let mut alts = match with_shorthands(element) {
                    Value::Object(mut o) if o.contains_key("oneOf") => match o.remove("oneOf") {
                        Some(Value::Array(a)) => a,
                        _ => unreachable!("oneOf is always an array here"),
                    },
                    other => vec![other],
                };
                alts.push(json!({ "type": "null" }));
                (json!({ "oneOf": alts }), false)
            }
            Multiplicity::Many => {
                let mut arr = json!({ "type": "array", "items": element });
                if c.non_empty {
                    arr["minItems"] = json!(1);
                }
                (arr, c.non_empty)
            }
            Multiplicity::Map => {
                let mut obj = json!({ "type": "object", "additionalProperties": element });
                if c.non_empty {
                    obj["minProperties"] = json!(1);
                }
                (obj, c.non_empty)
            }
        };
        props.insert(c.name.clone(), slot);
        if req {
            required.push(Value::String(c.name.clone()));
        }
    }

    Ok(json!({
        "type": "object",
        "title": v.name,
        "properties": Value::Object(props),
        "required": required,
        "additionalProperties": false,
    }))
}

/// Renders `schema` as JSON Schema. See the [module docs](self) for
/// the mapping and [`RefStyle`] for the two document shapes.
pub fn to_json_schema(
    schema: &NodeSchema,
    types: &TypeMap,
    style: RefStyle,
) -> Result<Value, Error> {
    let arms = schema
        .variants
        .iter()
        .map(|v| variant_arm(schema, v, types, &style))
        .collect::<Result<Vec<_>, _>>()?;
    let union = json!({
        "title": schema.name,
        "oneOf": arms,
        "discriminator": { "propertyName": "type" },
    });
    Ok(match style {
        RefStyle::Defs => json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": format!("#/$defs/{}", schema.name),
            "$defs": { schema.name.clone(): union },
        }),
        RefStyle::OpenApiComponent => union,
    })
}

impl NodeSchema {
    /// Renders this schema as JSON Schema — see [`to_json_schema`].
    pub fn to_json_schema(&self, types: &TypeMap, style: RefStyle) -> Result<Value, Error> {
        to_json_schema(self, types, style)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChildSchema, ScalarShorthand};

    fn schema() -> NodeSchema {
        let mut items = ChildSchema::recursive("items", Multiplicity::Many);
        items.non_empty = true;
        NodeSchema {
            name: "Q".into(),
            variants: vec![
                VariantSchema {
                    name: "And".into(),
                    fields: vec![],
                    children: vec![items],
                },
                VariantSchema {
                    name: "Row".into(),
                    fields: vec![
                        FieldSchema::required("name", "String"),
                        FieldSchema::optional("tags", "Vec<String>"),
                        FieldSchema::optional("note", "Option < String >"),
                        FieldSchema::required("flag", "bool"),
                        FieldSchema::required("n", "u32"),
                    ],
                    children: vec![
                        ChildSchema::recursive("head", Multiplicity::One),
                        ChildSchema::recursive("fallback", Multiplicity::Optional),
                        ChildSchema::recursive("env", Multiplicity::Map),
                        ChildSchema::scalar_map("labels", "String"),
                    ],
                },
            ],
        }
    }

    #[test]
    fn defs_style_wraps_and_targets_defs() {
        let doc = schema()
            .to_json_schema(&TypeMap::new(), RefStyle::Defs)
            .unwrap();
        assert_eq!(doc["$ref"], "#/$defs/Q");
        let q = &doc["$defs"]["Q"];
        assert_eq!(q["discriminator"]["propertyName"], "type");
        assert_eq!(q["oneOf"].as_array().unwrap().len(), 2);
        assert_eq!(
            q["oneOf"][0]["properties"]["items"]["items"]["$ref"],
            "#/$defs/Q"
        );
    }

    #[test]
    fn openapi_style_is_the_bare_union_targeting_components() {
        let doc = schema()
            .to_json_schema(&TypeMap::new(), RefStyle::OpenApiComponent)
            .unwrap();
        assert!(doc.get("$defs").is_none());
        assert_eq!(doc["title"], "Q");
        assert_eq!(
            doc["oneOf"][1]["properties"]["head"]["$ref"],
            "#/components/schemas/Q"
        );
    }

    #[test]
    fn fields_and_slots_map_by_shape() {
        let doc = schema()
            .to_json_schema(&TypeMap::new(), RefStyle::OpenApiComponent)
            .unwrap();
        let and = &doc["oneOf"][0];
        assert_eq!(and["properties"]["type"], json!({ "const": "And" }));
        assert_eq!(and["properties"]["items"]["minItems"], 1);
        assert_eq!(and["required"], json!(["type", "items"]));
        assert_eq!(and["additionalProperties"], false);

        let row = &doc["oneOf"][1];
        let p = &row["properties"];
        assert_eq!(p["name"], json!({ "type": "string" }));
        assert_eq!(
            p["tags"],
            json!({ "type": "array", "items": { "type": "string" } })
        );
        assert_eq!(
            p["note"],
            json!({ "oneOf": [{ "type": "string" }, { "type": "null" }] })
        );
        assert_eq!(p["flag"], json!({ "type": "boolean" }));
        assert_eq!(p["n"], json!({ "type": "integer" }));
        assert_eq!(p["head"], json!({ "$ref": "#/components/schemas/Q" }));
        assert_eq!(
            p["fallback"],
            json!({ "oneOf": [{ "$ref": "#/components/schemas/Q" }, { "type": "null" }] })
        );
        assert_eq!(
            p["env"],
            json!({ "type": "object", "additionalProperties": { "$ref": "#/components/schemas/Q" } })
        );
        assert_eq!(
            p["labels"],
            json!({ "type": "object", "additionalProperties": { "type": "string" } })
        );
        assert_eq!(
            row["required"],
            json!(["type", "name", "flag", "n", "head"])
        );
    }

    #[test]
    fn scalar_shorthand_adds_the_bare_scalar_alternative() {
        let mut body = ChildSchema::recursive("body", Multiplicity::Optional);
        body.scalar_shorthands.push(ScalarShorthand {
            kind: ScalarKind::Int,
            variant: "Lit".into(),
            field: "value".into(),
        });
        let s = NodeSchema {
            name: "E".into(),
            variants: vec![VariantSchema {
                name: "Neg".into(),
                fields: vec![],
                children: vec![body],
            }],
        };
        let doc = s.to_json_schema(&TypeMap::new(), RefStyle::Defs).unwrap();
        assert_eq!(
            doc["$defs"]["E"]["oneOf"][0]["properties"]["body"],
            json!({ "oneOf": [
                { "$ref": "#/$defs/E" }, { "type": "integer" }, { "type": "null" }
            ]})
        );
    }

    #[test]
    fn type_map_wins_and_unmapped_types_fail_loudly() {
        let s = NodeSchema {
            name: "E".into(),
            variants: vec![VariantSchema {
                name: "Eq".into(),
                fields: vec![FieldSchema::required("value", "Scalar")],
                children: vec![],
            }],
        };
        let err = s
            .to_json_schema(&TypeMap::new(), RefStyle::Defs)
            .unwrap_err();
        assert_eq!(
            err,
            Error::UnsupportedField {
                variant: "Eq".into(),
                field: "value".into(),
                ty: "Scalar".into()
            }
        );
        let types = TypeMap::new().with("Scalar", json!({ "type": "number" }));
        let doc = s.to_json_schema(&types, RefStyle::Defs).unwrap();
        assert_eq!(
            doc["$defs"]["E"]["oneOf"][0]["properties"]["value"],
            json!({ "type": "number" })
        );
        // A TypeMap entry also overrides a built-in.
        let types = TypeMap::new().with("String", json!({ "type": "string", "maxLength": 8 }));
        let s2 = NodeSchema {
            name: "E".into(),
            variants: vec![VariantSchema {
                name: "N".into(),
                fields: vec![FieldSchema::required("name", "String")],
                children: vec![],
            }],
        };
        let doc = s2.to_json_schema(&types, RefStyle::Defs).unwrap();
        assert_eq!(
            doc["$defs"]["E"]["oneOf"][0]["properties"]["name"]["maxLength"],
            8
        );
    }
}
