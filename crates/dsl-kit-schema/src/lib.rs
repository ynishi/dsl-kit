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
/// The first three variants (`One` / `Optional` / `Many`) mirror the
/// shapes recognised by `#[derive(DslNode)]` for positional recursion:
/// `T` or `Box<T>`, `Option<T>` or `Option<Box<T>>`, `Vec<T>` or
/// `Vec<Box<T>>`. The `Box` is folded into its unboxed counterpart
/// because the box is a storage detail invisible to schema consumers.
///
/// [`Multiplicity::Map`] marks a **keyed** child slot — a string-keyed
/// collection of subtrees. The concrete Rust shape the derive
/// recognises (`BTreeMap<String, T>`, `HashMap<String, Box<Self>>`, …)
/// is deferred to follow-up work; this variant is defined at the
/// schema layer so consumers can construct [`NodeSchema`] values that
/// declare keyed slots today, and downstream code (derive, PEG
/// codegen, JSON ⇔ AST bridge) can grow support incrementally.
///
/// The enum is `#[non_exhaustive]` so future primitives (ordered sets,
/// fixed-arity tuple slots, non-empty lists, …) can be added as minor
/// bumps. Out-of-crate matches must therefore include a `_ =>` arm;
/// in-crate matches (this workspace) remain exhaustively checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Multiplicity {
    /// `T` or `Box<T>`. Exactly one child.
    One,
    /// `Option<T>` or `Option<Box<T>>`. Zero or one child.
    Optional,
    /// `Vec<T>` or `Vec<Box<T>>`. Zero or more children in order.
    Many,
    /// String-keyed collection of children (`Map<String, V>` shape).
    ///
    /// The schema layer only records that the slot is keyed; the value
    /// shape (scalar / other-node / self-recursive) is inferred by the
    /// derive macro from the underlying Rust type. Downstream runtime
    /// support (derive, PEG codegen, JSON ⇔ AST bridge, lint) is not
    /// yet implemented — sites that consume [`Multiplicity`] currently
    /// emit a `MAP_NOT_IMPLEMENTED` diagnostic when they encounter a
    /// [`Multiplicity::Map`] slot. See the tracking issue for the
    /// staged rollout of the three shapes.
    Map,
}

impl Multiplicity {
    /// Returns the wire-format string used by [`NodeSchema::to_json`].
    pub fn as_str(&self) -> &'static str {
        match self {
            Multiplicity::One => "one",
            Multiplicity::Optional => "optional",
            Multiplicity::Many => "many",
            Multiplicity::Map => "map",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Every variant of [`Multiplicity`] has a wire-format string.
    /// Locks the four canonical spellings so downstream consumers
    /// (parsers / editors / MCP clients) can pattern-match on stable
    /// literals.
    #[test]
    fn multiplicity_as_str_covers_all_variants() {
        assert_eq!(Multiplicity::One.as_str(), "one");
        assert_eq!(Multiplicity::Optional.as_str(), "optional");
        assert_eq!(Multiplicity::Many.as_str(), "many");
        assert_eq!(Multiplicity::Map.as_str(), "map");
    }

    /// A hand-authored [`NodeSchema`] carrying a
    /// [`Multiplicity::Map`] child slot serializes with the expected
    /// `"multiplicity": "map"` wire spelling. Guards against silent
    /// drift of the JSON layout.
    #[test]
    fn to_json_emits_map_multiplicity() {
        let schema = NodeSchema {
            name: "Cfg".into(),
            variants: vec![VariantSchema {
                name: "Root".into(),
                fields: vec![],
                children: vec![ChildSchema {
                    name: "entries".into(),
                    multiplicity: Multiplicity::Map,
                }],
            }],
        };
        assert_eq!(
            schema.to_json(),
            json!({
                "name": "Cfg",
                "variants": [{
                    "name": "Root",
                    "fields": [],
                    "children": [
                        { "name": "entries", "multiplicity": "map" }
                    ],
                }],
            })
        );
    }
}
