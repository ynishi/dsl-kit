//! Compile-time schema reflection for `dsl-kit` DSLs.
//!
//! `dsl-kit-core` gives every DSL an instance-level observability surface
//! (`DslNode::node_id`, `Walk::children`), but external consumers —
//! parsers, editor tooling, MCP clients, AI prompters — need the
//! *type-level* shape: which variants exist, what fields each carries,
//! and how child recursion is spelled (`T` / `Box<T>` / `Option<T>` /
//! `Vec<T>`). None of that survives at runtime, so it is derived
//! separately from the enum definition via `#[derive(DslSchema)]`
//! (`dsl_kit_macros`) and returned by [`DslSchema::schema`].
//!
//! The shape is intentionally small and self-contained — no external
//! JSON-Schema tool feeds it, no runtime introspection re-derives it.
//! Consumers walk [`NodeSchema`] directly or serialize it with
//! [`NodeSchema::to_json`].
//!
//! ## Example
//!
//! ```ignore
//! use dsl_kit_schema::{DslSchema, Multiplicity};
//!
//! let schema = Flow::schema();
//! assert_eq!(schema.name, "Flow");
//! let seq = schema.variant("Seq").unwrap();
//! assert_eq!(seq.children[0].name, "children");
//! assert_eq!(seq.children[0].multiplicity, Multiplicity::Many);
//! ```

#![warn(missing_docs)]

use serde_json::{Value, json};

/// Compile-time schema reflection contract for a DSL AST type.
///
/// Implemented via `#[derive(DslSchema)]` from `dsl-kit-macros`.
/// Hand-written impls are fine for advanced shapes (mixed tuple / named
/// variants, indirect recursion, generics) — see the derive crate docs
/// for what it handles automatically.
pub trait DslSchema {
    /// Returns the structural schema of this DSL type.
    ///
    /// The output describes the *type*, not any particular instance —
    /// every variant that can occur is listed regardless of whether it
    /// appears in a given AST value.
    fn schema() -> NodeSchema;
}

/// Structural description of a DSL AST type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeSchema {
    /// The AST type's name (e.g. `"Flow"`).
    pub name: String,
    /// One entry per variant, in declaration order.
    pub variants: Vec<VariantSchema>,
}

impl NodeSchema {
    /// Looks up a variant by name.
    pub fn variant(&self, name: &str) -> Option<&VariantSchema> {
        self.variants.iter().find(|v| v.name == name)
    }

    /// Renders the schema as a JSON value.
    ///
    /// Layout:
    /// ```json
    /// {
    ///   "name": "Flow",
    ///   "variants": [
    ///     { "name": "Seq", "fields": [], "children": [
    ///         { "name": "children", "multiplicity": "many" }
    ///     ]},
    ///     ...
    ///   ]
    /// }
    /// ```
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "variants": self.variants.iter().map(VariantSchema::to_json).collect::<Vec<_>>(),
        })
    }
}

/// One variant of an AST enum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantSchema {
    /// Variant name (e.g. `"Seq"`).
    pub name: String,
    /// Non-recursive named fields (payload). The implementation-detail
    /// `id: NodeId` field is stripped.
    pub fields: Vec<FieldSchema>,
    /// Recursive child fields with their multiplicity.
    pub children: Vec<ChildSchema>,
}

impl VariantSchema {
    /// Renders the variant as a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "fields": self.fields.iter().map(FieldSchema::to_json).collect::<Vec<_>>(),
            "children": self.children.iter().map(ChildSchema::to_json).collect::<Vec<_>>(),
        })
    }
}

/// A non-recursive payload field on a variant.
///
/// `ty` is the Rust type source text (e.g. `"String"`,
/// `"Option<JoinPolicy>"`). Kept as a string so the schema stays
/// self-contained — consumers that need structured type information
/// should parse it or extend the derive to emit richer field info.
///
/// `optional` marks fields whose absence is a valid tree shape.
/// `#[derive(DslSchema)]` sets it automatically for payload types
/// spelled `Option<T>` (missing → `None`) and `Vec<T>` (missing →
/// empty). Hand-written schemas may set it directly. Optional fields
/// are skipped by [`check_conformance`](../dsl_kit_parse/fn.check_conformance.html)'s
/// `MISSING_FIELD` diagnostic and by `schema_gen`'s canonical PEG they
/// may be omitted from the argument list entirely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSchema {
    /// Field name.
    pub name: String,
    /// Rust type as source text.
    pub ty: String,
    /// Whether absence of this field is a valid shape (see the type
    /// docs). Defaults to `false` (required) for hand-written schemas.
    pub optional: bool,
}

impl FieldSchema {
    /// Builds a required (non-optional) field. Convenience for
    /// hand-written schemas that do not care about the optionality
    /// flag; equivalent to a struct literal with `optional: false`.
    pub fn required(name: impl Into<String>, ty: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: ty.into(),
            optional: false,
        }
    }

    /// Builds an optional field. Equivalent to a struct literal with
    /// `optional: true`.
    pub fn optional(name: impl Into<String>, ty: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: ty.into(),
            optional: true,
        }
    }

    /// Renders the field as a JSON value.
    ///
    /// `optional: true` is emitted as an extra `"optional": true` key;
    /// required fields omit the key entirely, preserving the pre-0.3
    /// JSON layout so external consumers that do not know about the
    /// flag are unaffected.
    pub fn to_json(&self) -> Value {
        if self.optional {
            json!({ "name": self.name, "type": self.ty, "optional": true })
        } else {
            json!({ "name": self.name, "type": self.ty })
        }
    }
}

/// A recursive child field on a variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildSchema {
    /// Field name (e.g. `"children"`, `"body"`).
    pub name: String,
    /// Recursion shape.
    pub multiplicity: Multiplicity,
}

impl ChildSchema {
    /// Renders the child field as a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "multiplicity": self.multiplicity.as_str(),
        })
    }
}

/// Recursion shape of a child field.
///
/// Mirrors the four shapes recognised by `#[derive(DslNode)]`.
/// `Box<T>` and `Vec<Box<T>>` are folded into their unboxed counterparts
/// because the box is a storage detail invisible to schema consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Multiplicity {
    /// `T` or `Box<T>`. Exactly one child.
    One,
    /// `Option<T>` or `Option<Box<T>>`. Zero or one child.
    Optional,
    /// `Vec<T>` or `Vec<Box<T>>`. Zero or more children in order.
    Many,
}

impl Multiplicity {
    /// Returns the wire-format string used by [`NodeSchema::to_json`].
    pub fn as_str(&self) -> &'static str {
        match self {
            Multiplicity::One => "one",
            Multiplicity::Optional => "optional",
            Multiplicity::Many => "many",
        }
    }
}
