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
///
/// The value carried by the slot is described by
/// [`ChildSchema::value_shape`]. For the historical shapes
/// (`One` / `Optional` / `Many`, plus keyed slots whose values are
/// `Self`) the value is recursive — the same AST enum — and the
/// value shape is [`ChildValueShape::Recursive`]. Keyed slots
/// ([`Multiplicity::Map`]) may also carry scalar values
/// (`BTreeMap<String, String>`, `BTreeMap<String, i64>`, …); those
/// report [`ChildValueShape::Scalar`] and carry the scalar's Rust
/// type as source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildSchema {
    /// Field name (e.g. `"children"`, `"body"`).
    pub name: String,
    /// Recursion shape.
    pub multiplicity: Multiplicity,
    /// Value shape carried by the slot.
    ///
    /// Defaults to [`ChildValueShape::Recursive`] for the historical
    /// shapes. Scalar-valued keyed slots (Shape 1 of the tracking
    /// issue) set this to [`ChildValueShape::Scalar`] and carry the
    /// value type as source text.
    pub value_shape: ChildValueShape,
    /// Declared scalar shorthands accepted by this slot
    /// ([`ScalarShorthand`]). Empty for every slot that accepts only
    /// the canonical node spelling — the historical behaviour, and
    /// the default of every constructor helper.
    ///
    /// Only meaningful on [`Multiplicity::One`] /
    /// [`Multiplicity::Optional`] slots with
    /// [`ChildValueShape::Recursive`] values; consumers reject other
    /// combinations up front rather than guessing a meaning for them.
    /// At most one entry per [`ScalarKind`] — the front-ends dispatch
    /// on the input's kind alone, never by trial deserialization, so
    /// a duplicate kind would make the mapping ambiguous.
    pub scalar_shorthands: Vec<ScalarShorthand>,
    /// Declared non-emptiness for collection slots. `false` (the
    /// historical behaviour and every constructor's default) keeps
    /// the zero-or-more contract; `true` declares that the slot must
    /// hold at least one element.
    ///
    /// Only meaningful on [`Multiplicity::Many`] /
    /// [`Multiplicity::Map`] slots — `One` is inherently non-empty
    /// and `Optional` inherently permits absence, so consumers reject
    /// the flag there up front. The constraint is *declared*, not
    /// inferred: `check_conformance` rejects a violating tree with
    /// its own diagnostic slug, generated grammars require at least
    /// one element, and the `no-empty-child-slots` lint reports only
    /// declared violations instead of guessing from variant shape.
    pub non_empty: bool,
}

impl Default for ChildValueShape {
    /// Historical shapes carry recursive values; this matches
    /// pre-0.6 [`ChildSchema`] semantics so hand-written schemas
    /// that used struct-update syntax
    /// (`ChildSchema { name, multiplicity, ..Default::default() }`)
    /// keep the same behaviour.
    fn default() -> Self {
        ChildValueShape::Recursive
    }
}

impl ChildSchema {
    /// Builds a child slot whose value is the same AST enum
    /// (recursive shape). Convenience for hand-written schemas; the
    /// derive macro uses this path for every historical shape.
    pub fn recursive(name: impl Into<String>, multiplicity: Multiplicity) -> Self {
        Self {
            name: name.into(),
            multiplicity,
            value_shape: ChildValueShape::Recursive,
            scalar_shorthands: Vec::new(),
            non_empty: false,
        }
    }

    /// Builds a keyed child slot whose values are scalars of Rust
    /// type `ty`. The multiplicity is always [`Multiplicity::Map`];
    /// non-`Map` scalar slots are not a recognised shape.
    pub fn scalar_map(name: impl Into<String>, value_ty: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            multiplicity: Multiplicity::Map,
            value_shape: ChildValueShape::Scalar {
                ty: value_ty.into(),
            },
            scalar_shorthands: Vec::new(),
            non_empty: false,
        }
    }

    /// Marks the slot as non-empty (builder style). Only meaningful
    /// on [`Multiplicity::Many`] / [`Multiplicity::Map`] slots — see
    /// [`ChildSchema::non_empty`].
    ///
    /// ```
    /// use dsl_kit_schema::{ChildSchema, Multiplicity};
    ///
    /// let slot = ChildSchema::recursive("stmts", Multiplicity::Many).with_non_empty();
    /// assert!(slot.non_empty);
    /// ```
    pub fn with_non_empty(mut self) -> Self {
        self.non_empty = true;
        self
    }

    /// Adds a declared scalar shorthand to the slot (builder style).
    ///
    /// ```
    /// use dsl_kit_schema::{ChildSchema, Multiplicity, ScalarKind};
    ///
    /// let slot = ChildSchema::recursive("content", Multiplicity::One)
    ///     .with_scalar_shorthand(ScalarKind::Str, "Literal", "value");
    /// assert_eq!(slot.scalar_shorthands.len(), 1);
    /// ```
    pub fn with_scalar_shorthand(
        mut self,
        kind: ScalarKind,
        variant: impl Into<String>,
        field: impl Into<String>,
    ) -> Self {
        self.scalar_shorthands.push(ScalarShorthand {
            kind,
            variant: variant.into(),
            field: field.into(),
        });
        self
    }

    /// Looks up the declared shorthand for a scalar kind, if any.
    pub fn scalar_shorthand(&self, kind: ScalarKind) -> Option<&ScalarShorthand> {
        self.scalar_shorthands.iter().find(|s| s.kind == kind)
    }

    /// Renders the child field as a JSON value.
    ///
    /// [`ChildValueShape::Recursive`] slots preserve the pre-0.6 JSON
    /// layout (just `name` + `multiplicity`) so external consumers
    /// that do not know about `value_shape` are unaffected. Scalar
    /// slots gain a `"value"` object carrying the shape's kind and
    /// scalar `type` string. Declared scalar shorthands gain a
    /// `"scalar_shorthands"` array, and a declared non-empty slot
    /// gains `"non_empty": true`; slots without either keep the
    /// historical layout, again so unaware consumers see no change.
    pub fn to_json(&self) -> Value {
        let mut obj = match &self.value_shape {
            ChildValueShape::Recursive => json!({
                "name": self.name,
                "multiplicity": self.multiplicity.as_str(),
            }),
            ChildValueShape::Scalar { ty } => json!({
                "name": self.name,
                "multiplicity": self.multiplicity.as_str(),
                "value": { "kind": "scalar", "type": ty },
            }),
        };
        if !self.scalar_shorthands.is_empty() {
            obj["scalar_shorthands"] = Value::Array(
                self.scalar_shorthands
                    .iter()
                    .map(ScalarShorthand::to_json)
                    .collect(),
            );
        }
        if self.non_empty {
            obj["non_empty"] = Value::Bool(true);
        }
        obj
    }
}

/// JSON kind accepted by a declared scalar shorthand.
///
/// The front-ends dispatch on the *input's* kind — a JSON string /
/// integer / boolean, or the corresponding canonical-text token — and
/// each kind maps to at most one declared target, so resolution never
/// depends on declaration order or trial deserialization.
///
/// The enum is `#[non_exhaustive]` so future kinds can be added as
/// minor bumps. Out-of-crate matches must therefore include a `_ =>`
/// arm; in-crate matches (this workspace) remain exhaustively checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScalarKind {
    /// A JSON string (`%str` in canonical text).
    Str,
    /// A JSON integer (`%int` in canonical text).
    Int,
    /// A JSON boolean (`true` / `false` in canonical text).
    Bool,
}

impl ScalarKind {
    /// Returns the wire-format string used by
    /// [`ScalarShorthand::to_json`].
    pub fn as_str(&self) -> &'static str {
        match self {
            ScalarKind::Str => "string",
            ScalarKind::Int => "int",
            ScalarKind::Bool => "bool",
        }
    }
}

/// A declared scalar shorthand on a [`ChildSchema`] slot.
///
/// When a [`Multiplicity::One`] / [`Multiplicity::Optional`] child
/// slot receives a bare scalar of [`kind`](Self::kind) instead of the
/// canonical node spelling, the front-ends lower it to a node of
/// [`variant`](Self::variant) whose [`field`](Self::field) carries the
/// scalar — producing the *same* `ParseTree` as the explicit spelling.
/// The shorthand is an input-side projection only: canonical
/// serialization always emits the node spelling, and the lowered tree
/// is indistinguishable from one parsed from it.
///
/// The mapping is declared, never inferred: the target variant is
/// named here (via `#[dsl_schema(scalar(...))]` on the slot field, or
/// [`ChildSchema::with_scalar_shorthand`] in hand-written schemas), so
/// adding another scalar-carrying variant to the enum later cannot
/// silently change what a bare scalar means.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarShorthand {
    /// Input kind the shorthand accepts.
    pub kind: ScalarKind,
    /// Target variant name the scalar lowers to.
    pub variant: String,
    /// Payload field on the target variant that carries the scalar.
    pub field: String,
}

impl ScalarShorthand {
    /// Renders the shorthand as a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "kind": self.kind.as_str(),
            "variant": self.variant,
            "field": self.field,
        })
    }
}

/// Value shape carried by a [`ChildSchema`] slot.
///
/// [`Multiplicity`] describes cardinality (one / optional / many /
/// keyed); `ChildValueShape` describes what each element *is*.
///
/// The enum is `#[non_exhaustive]` so future shapes (values from a
/// distinct AST type — Shape 2 of the tracking issue — or other
/// non-recursive value primitives) can be added as minor bumps.
/// Out-of-crate matches must therefore include a `_ =>` arm; in-crate
/// matches (this workspace) remain exhaustively checked.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChildValueShape {
    /// The slot's value type is the enclosing AST enum itself.
    /// Every historical shape (`One` / `Optional` / `Many`, plus
    /// keyed slots spelled `BTreeMap<String, Self>` /
    /// `BTreeMap<String, Box<Self>>`) reports `Recursive`.
    Recursive,
    /// The slot's value type is a scalar payload. Only valid in
    /// combination with [`Multiplicity::Map`] — Shape 1 of the
    /// tracking issue. Carries the value type as Rust source text
    /// (e.g. `"String"`, `"i64"`, `"bool"`).
    Scalar {
        /// Rust source text of the scalar value type.
        ty: String,
    },
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
/// collection of subtrees. The derive recognises exactly
/// `BTreeMap<String, T>` and `BTreeMap<String, Box<T>>`; `HashMap` and
/// friends are not keyed shapes here, because a map slot's iteration
/// order is observable (walks, canonical text, JSON round-trips) and
/// so has to be deterministic.
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
    /// String-keyed collection of children
    /// (`BTreeMap<String, V>` shape). Zero or more entries, each
    /// reachable by its key.
    ///
    /// The schema layer only records that the slot is keyed; the value
    /// shape is inferred by the derive macro from the underlying Rust
    /// type via [`ChildSchema::value_shape`]. Self-recursive values
    /// (`BTreeMap<String, Self>` / `BTreeMap<String, Box<Self>>`) are
    /// supported end to end — derive, conformance, the JSON bridge,
    /// generated grammars and `DslBuild`. Scalar values
    /// (`BTreeMap<String, T>` where `T` is a scalar payload type —
    /// Shape 1 of the tracking issue) are supported end to end as of
    /// 0.7: the derive (`ChildValueShape::Scalar { ty }`), the
    /// JSON ⇔ ParseTree bridge, `DslBuild` via `build_scalar_map`,
    /// and the PEG grammar generator / canonical text syntax (each
    /// entry's value is a bare scalar, lowered through a synthetic
    /// `value`-field leaf node — see `schema_gen::child_arg_peg`).
    /// Keyed slots whose values are *another* AST type (Shape 2) are
    /// still being staged.
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
                    value_shape: ChildValueShape::Recursive,
                    scalar_shorthands: vec![],
                    non_empty: false,
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
