//! Parser trunk for `dsl-kit` DSLs.
//!
//! Two front-ends (serde bridge in G-1b, PEG interpreter in G-2) will
//! both feed a single [`ParseTree`] shape. One consumer,
//! `#[derive(DslBuild)]` (G-1c), converts a validated tree into a
//! typed AST value using the caller's [`IdGen`] to mint fresh
//! [`NodeId`]s.
//!
//! This module (G-1a) lands the central data types and the schema
//! conformance checker that both front-ends run against. All feedback
//! flows through one [`Diagnostic`] envelope so the consumer AI sees a
//! single dialect across parsing, conformance, lint, and debugger.
//!
//! ## Architecture (layers)
//!
//! - [`ParseTree`] — untyped trunk. Variant name + field payloads +
//!   named child slots (positional lists, plus keyed entries for
//!   [`Multiplicity::Map`] slots) + optional source [`Span`].
//! - [`RawValue`] — per-field payload representation. `Text` keeps the
//!   PEG front-end's matched source text; `Json` keeps the serde
//!   front-end's typed value with no stringify round-trip.
//! - [`check_conformance`] — validates a tree against a
//!   [`NodeSchema`] and returns diagnostics.
//! - [`Diagnostic`] / [`Severity`] / [`Location`] — the shared
//!   envelope shape.
//! - [`DslBuild`] — trait consumed by the derive to build a typed AST.
//! - [`BuildError`] — non-empty diagnostic bag returned on build
//!   failure.

#![warn(missing_docs)]

use dsl_kit_core::{IdGen, NodeId, Suggester, Suggestion};
use dsl_kit_schema::{ChildSchema, Multiplicity, NodeSchema, VariantSchema};
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

pub mod example_gen;
pub mod grammar_check;
pub mod import;
pub mod peg;
pub mod schema_gen;
pub mod serde_bridge;

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

/// Byte-offset range into the parser input.
///
/// The PEG front-end fills this in as it consumes input; the serde
/// front-end leaves it `None` because JSON payloads have no meaningful
/// source position after deserialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Inclusive start byte offset.
    pub start: usize,
    /// Exclusive end byte offset.
    pub end: usize,
}

impl Span {
    /// Constructs a new [`Span`].
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

// ---------------------------------------------------------------------------
// RawValue
// ---------------------------------------------------------------------------

/// Per-field payload representation carried by [`ParseTree`].
///
/// The two arms correspond to the two front-ends:
///
/// - [`RawValue::Text`] is the matched source text a PEG rule produced;
///   the [`DslBuild`] consumer converts it via `FromStr`.
/// - [`RawValue::Json`] is a typed value straight from serde; the
///   consumer converts it via serde deserialization. Nested payloads
///   like `Option<u32>` never take a stringify → reparse detour.
#[derive(Debug, Clone, PartialEq)]
pub enum RawValue {
    /// Matched source text (PEG front-end).
    Text(String),
    /// Typed value (serde front-end).
    Json(Value),
}

// ---------------------------------------------------------------------------
// ParseTree
// ---------------------------------------------------------------------------

/// Untyped parse trunk, produced by any front-end and consumed by
/// [`DslBuild`].
///
/// The shape deliberately mirrors [`NodeSchema`] / [`VariantSchema`]:
/// `variant` picks a variant by name; `fields` carries non-recursive
/// payload; `children` carries named **positional** child slots each
/// holding an ordered list of subtrees; `keyed_children` carries named
/// **keyed** child slots ([`Multiplicity::Map`]) holding `(key,
/// subtree)` pairs.
///
/// A given slot name lives in exactly one of the two child vectors —
/// whichever one its schema [`Multiplicity`] calls for. Putting it in
/// the wrong one is a [`codes::KEYED_SLOT_SHAPE`] diagnostic at
/// conformance time rather than a silent drop.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseTree {
    /// Variant name — must match a [`VariantSchema::name`] at
    /// conformance time.
    pub variant: String,
    /// Non-recursive payload fields, keyed by field name.
    pub fields: Vec<(String, RawValue)>,
    /// Positional child slots, keyed by slot name. Each slot's inner
    /// [`Vec`] holds zero-or-more trees per the child's
    /// [`Multiplicity`] ([`One`](Multiplicity::One) /
    /// [`Optional`](Multiplicity::Optional) /
    /// [`Many`](Multiplicity::Many)).
    pub children: Vec<(String, Vec<ParseTree>)>,
    /// Keyed child slots ([`Multiplicity::Map`]), by slot name. Each
    /// slot holds `(key, subtree)` pairs.
    ///
    /// Front-ends must emit the pairs **sorted by key**, so that two
    /// front-ends handed the same document produce equal entries. The
    /// JSON bridge sorts on ingest; hand-built trees that do not are
    /// rejected at conformance time with
    /// [`codes::KEYED_SLOT_UNSORTED`] rather than quietly compared
    /// unequal later.
    ///
    /// One shape is *not* pinned across front-ends: an **empty** map
    /// may arrive either as a present-but-empty slot or as no slot at
    /// all, depending on how the front-end spells emptiness (the JSON
    /// bridge records the key it saw; the PEG interpreter has nothing
    /// to record). Both mean the same thing everywhere it matters —
    /// [`check_conformance`] accepts both and
    /// [`build_child_map`] yields an empty map from both — so
    /// consumers should read a slot through
    /// [`ParseTree::keyed_child_slot`] rather than testing for its
    /// presence. The positional [`ParseTree::children`] half has
    /// carried the same asymmetry since it existed.
    ///
    /// Duplicate keys within one slot are representable (the storage
    /// is a [`Vec`], not a map) precisely so they can be *diagnosed* —
    /// [`codes::DUPLICATE_KEY`] — instead of one entry silently
    /// winning.
    pub keyed_children: Vec<(String, Vec<(String, ParseTree)>)>,
    /// Source-range span, if the front-end tracks it.
    pub span: Option<Span>,
}

impl ParseTree {
    /// Constructs a new [`ParseTree`] with no fields, children, or span.
    pub fn new(variant: impl Into<String>) -> Self {
        Self {
            variant: variant.into(),
            fields: Vec::new(),
            children: Vec::new(),
            keyed_children: Vec::new(),
            span: None,
        }
    }

    /// Looks up a payload field by name.
    pub fn field(&self, name: &str) -> Option<&RawValue> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    /// Looks up a positional child slot by name.
    pub fn child_slot(&self, name: &str) -> Option<&[ParseTree]> {
        self.children
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }

    /// Looks up a keyed child slot ([`Multiplicity::Map`]) by name.
    ///
    /// Returns the slot's `(key, subtree)` pairs in the order the
    /// front-end emitted them. For a tree that passed
    /// [`check_conformance`] that order is ascending by key; on an
    /// unchecked tree it is whatever the producer wrote.
    pub fn keyed_child_slot(&self, name: &str) -> Option<&[(String, ParseTree)]> {
        self.keyed_children
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }
}

// ---------------------------------------------------------------------------
// Diagnostic envelope
// ---------------------------------------------------------------------------

/// Severity level of a [`Diagnostic`].
///
/// Matches the shape used by `dsl-kit-lint` so consumers speak one
/// dialect across parse / conformance / lint / debugger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Blocks build. Multiple errors may be collected before returning.
    Error,
    /// Non-blocking warning.
    Warning,
    /// Advisory note.
    Info,
}

impl Severity {
    /// Returns the wire-format string used by [`Diagnostic::to_json`].
    pub const fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Info => "info",
        }
    }
}

/// Where a [`Diagnostic`] points into the consumer's material.
///
/// [`Location::Span`] is the parse-time answer (byte range into the
/// input text); [`Location::Node`] is the post-build answer
/// ([`NodeId`] into the typed AST); [`Location::None`] is for
/// diagnostics without a natural anchor point (e.g. "missing top-level
/// variant").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Location {
    /// Source-text range.
    Span(Span),
    /// AST node identity.
    Node(NodeId),
    /// No anchor point.
    None,
}

impl Location {
    /// Renders the location as the JSON value used by
    /// [`Diagnostic::to_json`].
    ///
    /// - `Location::Span { start, end }` → `{ "kind": "span", "start": N, "end": M }`
    /// - `Location::Node(id)`           → `{ "kind": "node", "id": N }`
    /// - `Location::None`               → JSON `null`
    pub fn to_json(&self) -> Value {
        match self {
            Location::Span(Span { start, end }) => {
                json!({ "kind": "span", "start": start, "end": end })
            }
            Location::Node(id) => json!({ "kind": "node", "id": id.0 }),
            Location::None => Value::Null,
        }
    }
}

/// One diagnostic in the unified envelope shared across parse,
/// conformance, lint, and debugger feedback.
///
/// The consumer AI is expected to read a single JSON dialect regardless
/// of which subsystem produced the diagnostic: `{ severity, code,
/// message, location }`. `code` is a stable machine-friendly slug
/// (e.g. `"dsl_kit::parse::unknown_variant"`); `message` is the human
/// (or AI) readable one-liner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// Severity level.
    pub severity: Severity,
    /// Stable machine slug.
    pub code: String,
    /// One-line human / AI readable description.
    pub message: String,
    /// Anchor point (or [`Location::None`]).
    pub location: Location,
}

impl Diagnostic {
    /// Constructs an [`Severity::Error`] diagnostic.
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            code: code.into(),
            message: message.into(),
            location: Location::None,
        }
    }

    /// Attaches a source [`Span`] as the location.
    pub fn with_span(mut self, span: Option<Span>) -> Self {
        if let Some(span) = span {
            self.location = Location::Span(span);
        }
        self
    }

    /// Attaches an AST [`NodeId`] as the location.
    pub fn with_node(mut self, id: NodeId) -> Self {
        self.location = Location::Node(id);
        self
    }

    /// Renders the diagnostic as a JSON value.
    ///
    /// Layout:
    /// ```json
    /// {
    ///   "severity": "error",
    ///   "code": "dsl_kit::parse::unknown_variant",
    ///   "message": "unknown variant `Foo` (candidates: Add, Mul, Lit)",
    ///   "location": { "kind": "span", "start": 0, "end": 3 }
    /// }
    /// ```
    pub fn to_json(&self) -> Value {
        json!({
            "severity": self.severity.as_str(),
            "code": self.code,
            "message": self.message,
            "location": self.location.to_json(),
        })
    }
}

// ---------------------------------------------------------------------------
// BuildError
// ---------------------------------------------------------------------------

/// Non-empty diagnostic bag returned by [`DslBuild::from_parse_tree`]
/// and [`check_conformance`] callers.
///
/// Multiple diagnostics are collected before failing so the consumer
/// AI sees every problem in one round-trip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildError {
    /// The diagnostics (at least one is guaranteed by
    /// [`BuildError::new`]).
    pub diagnostics: Vec<Diagnostic>,
}

impl BuildError {
    /// Wraps a non-empty vector of diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if `diagnostics` is empty — a [`BuildError`] with no
    /// diagnostics is meaningless.
    pub fn new(diagnostics: Vec<Diagnostic>) -> Self {
        assert!(
            !diagnostics.is_empty(),
            "BuildError requires at least one diagnostic"
        );
        Self { diagnostics }
    }

    /// Convenience constructor for a single-diagnostic error.
    pub fn single(diag: Diagnostic) -> Self {
        Self {
            diagnostics: vec![diag],
        }
    }

    /// Renders the whole bag as a JSON array of
    /// [`Diagnostic::to_json`] entries.
    pub fn to_json(&self) -> Value {
        Value::Array(self.diagnostics.iter().map(Diagnostic::to_json).collect())
    }
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "build failed with {} diagnostic(s):",
            self.diagnostics.len()
        )?;
        for d in &self.diagnostics {
            write!(f, "\n  [{}] {}: {}", d.severity.as_str(), d.code, d.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for BuildError {}

// ---------------------------------------------------------------------------
// DslBuild
// ---------------------------------------------------------------------------

/// Contract for building a typed AST value from a validated
/// [`ParseTree`].
///
/// Implemented via `#[derive(DslBuild)]` in `dsl-kit-macros` (G-1c);
/// hand-written impls are fine for shapes the derive cannot express
/// (mixed tuple variants, generics, cross-type recursion, etc.).
pub trait DslBuild: Sized {
    /// Builds a typed value from `tree`, minting fresh [`NodeId`]s from
    /// `ids`.
    ///
    /// Implementations should call [`check_conformance`] against
    /// `Self::schema()` before dispatching per-variant so that shape
    /// mismatches surface as diagnostics rather than panics or ad-hoc
    /// errors. `RawValue::Text` payloads convert through `FromStr`;
    /// `RawValue::Json` payloads convert through
    /// `serde_json::from_value`.
    fn from_parse_tree(tree: &ParseTree, ids: &IdGen) -> Result<Self, BuildError>;
}

// ---------------------------------------------------------------------------
// Schema conformance
// ---------------------------------------------------------------------------

/// Diagnostic codes emitted by [`check_conformance`].
///
/// Exposed as a module of `pub const &str` slugs so downstream tools
/// (lint UIs, doc generators) can reference them without matching on
/// string literals.
pub mod codes {
    /// Tree's `variant` did not match any
    /// [`VariantSchema`](dsl_kit_schema::VariantSchema) in the
    /// [`NodeSchema`](dsl_kit_schema::NodeSchema).
    pub const UNKNOWN_VARIANT: &str = "dsl_kit::parse::unknown_variant";
    /// A field required by the schema is absent from the tree.
    pub const MISSING_FIELD: &str = "dsl_kit::parse::missing_field";
    /// The tree carries a field the schema does not declare.
    pub const UNKNOWN_FIELD: &str = "dsl_kit::parse::unknown_field";
    /// The tree references a child slot the schema does not declare.
    pub const UNKNOWN_CHILD: &str = "dsl_kit::parse::unknown_child";
    /// A [`crate::Multiplicity::One`] slot did not carry exactly one
    /// child.
    pub const ARITY_ONE: &str = "dsl_kit::parse::arity_one";
    /// A [`crate::Multiplicity::Optional`] slot carried more than one
    /// child.
    pub const ARITY_OPTIONAL: &str = "dsl_kit::parse::arity_optional";
    /// A payload field appeared under `children` (structural mismatch).
    pub const FIELD_AS_CHILD: &str = "dsl_kit::parse::field_as_child";
    /// A child slot appeared under `fields` (structural mismatch).
    pub const CHILD_AS_FIELD: &str = "dsl_kit::parse::child_as_field";
    /// A field name appeared more than once.
    pub const DUPLICATE_FIELD: &str = "dsl_kit::parse::duplicate_field";
    /// A child slot name appeared more than once.
    pub const DUPLICATE_CHILD: &str = "dsl_kit::parse::duplicate_child";
    /// A payload field's value could not be converted to the target
    /// Rust type (serde deserialization failure or `FromStr` failure).
    pub const FIELD_TYPE: &str = "dsl_kit::parse::field_type";
    /// A keyed child slot ([`crate::Multiplicity::Map`]) carried the
    /// same key more than once. The map the consumer asked for cannot
    /// hold both, and picking a winner silently would make the parse
    /// lossy — so the duplicate is an error at conformance time and at
    /// [`crate::build_child_map`] time.
    pub const DUPLICATE_KEY: &str = "dsl_kit::parse::duplicate_key";
    /// A child slot was placed in the wrong half of the tree: a slot
    /// declared [`crate::Multiplicity::Map`] appeared under
    /// [`ParseTree::children`] (positional), or a slot declared
    /// `One` / `Optional` / `Many` appeared under
    /// [`ParseTree::keyed_children`].
    pub const KEYED_SLOT_SHAPE: &str = "dsl_kit::parse::keyed_slot_shape";
    /// A keyed child slot's entries were not in ascending key order.
    /// [`ParseTree::keyed_children`] documents sorted-by-key as the
    /// canonical form so that two front-ends producing the same
    /// document produce equal trees; this is that contract enforced
    /// rather than assumed. The JSON bridge sorts on ingest, so this
    /// fires only for hand-built or hand-rolled front-end trees.
    pub const KEYED_SLOT_UNSORTED: &str = "dsl_kit::parse::keyed_slot_unsorted";
    /// A child slot's multiplicity is a variant this build of
    /// `dsl-kit-parse` does not recognise. Emitted from the catch-all
    /// arm required by [`crate::Multiplicity`]'s `#[non_exhaustive]`
    /// contract when a future variant is added to the schema crate
    /// ahead of parse-side support. Not reachable in practice within
    /// the workspace (both crates are versioned in lockstep) but held
    /// as a first-class diagnostic so out-of-workspace consumers see a
    /// stable slug rather than a panic.
    pub const UNKNOWN_MULTIPLICITY: &str = "dsl_kit::parse::unknown_multiplicity";
}

/// Validates `tree` against `schema` shallowly (this level only).
///
/// The check is intentionally single-level — the recursion is the
/// caller's job (typically `DslBuild::from_parse_tree` for each child
/// slot). Every mismatch produces a [`Diagnostic`]; the returned
/// vector is empty when the tree matches.
///
/// Checks performed:
///
/// - variant name matches one of `schema.variants` (else
///   [`codes::UNKNOWN_VARIANT`] with nearest candidates in the
///   message);
/// - every declared field is present exactly once
///   ([`codes::MISSING_FIELD`], [`codes::DUPLICATE_FIELD`]);
/// - no undeclared field is present ([`codes::UNKNOWN_FIELD`],
///   or [`codes::CHILD_AS_FIELD`] when the extra name is actually a
///   declared child slot);
/// - every declared child slot is present exactly once
///   ([`codes::DUPLICATE_CHILD`]) and honours its [`Multiplicity`]
///   ([`codes::ARITY_ONE`], [`codes::ARITY_OPTIONAL`]);
/// - every child slot sits in the half its [`Multiplicity`] calls for
///   — keyed slots under [`ParseTree::keyed_children`], positional
///   ones under [`ParseTree::children`] ([`codes::KEYED_SLOT_SHAPE`]);
/// - a keyed slot carries each key at most once
///   ([`codes::DUPLICATE_KEY`]) and in ascending key order
///   ([`codes::KEYED_SLOT_UNSORTED`]);
/// - no undeclared child slot is present ([`codes::UNKNOWN_CHILD`],
///   or [`codes::FIELD_AS_CHILD`] when the extra name is actually a
///   declared payload field).
///
/// [`Multiplicity::Many`] and [`Multiplicity::Map`] both accept
/// zero-or-more children — emptiness is a domain-level judgment, not a
/// shape error, so the `no-empty-child-slots` lint rule is where it
/// belongs. That rule covers both collection shapes and reports at
/// `Warn`, since neither multiplicity promises a non-empty
/// collection.
pub fn check_conformance(tree: &ParseTree, schema: &NodeSchema) -> Vec<Diagnostic> {
    check_conformance_with(tree, schema, &BuiltinLevenshteinSuggester)
}

/// Variant of [`check_conformance`] that routes `did you mean X?`
/// hints through a caller-supplied [`Suggester`].
///
/// The free function [`check_conformance`] delegates here with a
/// crate-private [`Suggester`] backed by the same Levenshtein
/// algorithm the crate has always shipped, so existing callers see no
/// behavioural change. Reach for this variant when you want to plug
/// in a different similarity algorithm (e.g. `dsl-kit-fuzzy`'s
/// `FuzzySuggester` with Jaro-Winkler) at a specific call site.
pub fn check_conformance_with(
    tree: &ParseTree,
    schema: &NodeSchema,
    suggester: &dyn Suggester,
) -> Vec<Diagnostic> {
    let mut out = Vec::new();

    if tree.variant == import::IMPORT_VARIANT {
        out.push(
            Diagnostic::error(
                import::import_codes::UNEXPANDED,
                format!(
                    "unexpanded `{}` placeholder — run the document through \
                     `import::load_json_str` / `import::load_json_value` before conformance",
                    import::IMPORT_VARIANT
                ),
            )
            .with_span(tree.span),
        );
        return out;
    }

    let Some(variant) = schema.variant(&tree.variant) else {
        let msg = format_unknown_variant(&tree.variant, schema, suggester);
        out.push(Diagnostic::error(codes::UNKNOWN_VARIANT, msg).with_span(tree.span));
        return out;
    };

    check_fields(tree, variant, &mut out, suggester);
    check_children(tree, variant, &mut out, suggester);
    out
}

/// Shared formatter for the `UNKNOWN_VARIANT` diagnostic message,
/// used by both [`check_conformance_with`] and the serde bridge so the
/// wording stays consistent across front-ends.
pub(crate) fn format_unknown_variant(
    query: &str,
    schema: &NodeSchema,
    suggester: &dyn Suggester,
) -> String {
    let names: Vec<&str> = schema.variants.iter().map(|v| v.name.as_str()).collect();
    let base = format!("unknown variant `{}` for `{}`", query, schema.name);
    match suggester.enrich_unknown(query, &names) {
        Some(hint) => format!("{base} ({hint})"),
        None => base,
    }
}

/// Shared formatter for `UNKNOWN_FIELD` and `UNKNOWN_CHILD` messages.
///
/// `kind` is `"field"` or `"child slot"`. `all_candidates` should
/// include both declared field names and declared child slot names so
/// a `did you mean` hint can point at either. `missing_candidates`
/// carries the slot names that are declared but currently absent from
/// the tree; suggestions that appear in this set are marked
/// `(missing)` — the "pair hint" that makes payloads like
/// `taget` → `target` (missing) diagnose themselves at a glance.
pub(crate) fn format_unknown_slot(
    kind: &str,
    query: &str,
    variant_name: &str,
    all_candidates: &[&str],
    missing_candidates: &[&str],
    suggester: &dyn Suggester,
) -> String {
    let base = format!("unknown {kind} `{query}` on variant `{variant_name}`");
    let sugs = suggester.suggest(query, all_candidates);
    if sugs.is_empty() {
        return base;
    }
    let joined = sugs
        .iter()
        .take(3)
        .map(|s| {
            if missing_candidates.contains(&s.candidate) {
                format!("`{}` (missing)", s.candidate)
            } else {
                format!("`{}`", s.candidate)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{base} (did you mean: {joined})")
}

fn check_fields(
    tree: &ParseTree,
    variant: &VariantSchema,
    out: &mut Vec<Diagnostic>,
    suggester: &dyn Suggester,
) {
    // Duplicate field names.
    for i in 0..tree.fields.len() {
        let (name, _) = &tree.fields[i];
        if tree.fields[..i].iter().any(|(n, _)| n == name) {
            out.push(
                Diagnostic::error(
                    codes::DUPLICATE_FIELD,
                    format!(
                        "field `{}` appears more than once on variant `{}`",
                        name, variant.name
                    ),
                )
                .with_span(tree.span),
            );
        }
    }

    // Missing declared fields. Optional fields (typically `Option<T>`
    // / `Vec<T>` payloads recognised by `#[derive(DslSchema)]`) are
    // skipped — their absence is a valid shape and the build layer's
    // `build_field_optional` / `build_field_vec` helpers supply the
    // default (`None` / empty vec).
    for f in &variant.fields {
        if f.optional {
            continue;
        }
        if tree.field(&f.name).is_none() {
            out.push(
                Diagnostic::error(
                    codes::MISSING_FIELD,
                    format!(
                        "variant `{}` is missing required field `{}` (type `{}`)",
                        variant.name, f.name, f.ty
                    ),
                )
                .with_span(tree.span),
            );
        }
    }

    // Precompute the "missing-so-far" set once: slot names that are
    // declared but currently absent from `tree`. Used as the pair-hint
    // marker on UNKNOWN_FIELD suggestions.
    let missing = missing_slot_names(tree, variant);
    let all_slots = all_slot_names(variant);

    // Unknown fields.
    for (name, _) in &tree.fields {
        if variant.fields.iter().any(|f| &f.name == name) {
            continue;
        }
        // If it names a declared child slot, that's a structural mix-up.
        if variant.children.iter().any(|c| &c.name == name) {
            out.push(
                Diagnostic::error(
                    codes::CHILD_AS_FIELD,
                    format!(
                        "`{}` on variant `{}` is a child slot, not a payload field \
                         (move it under `children`)",
                        name, variant.name
                    ),
                )
                .with_span(tree.span),
            );
        } else {
            let msg = format_unknown_slot(
                "field",
                name,
                &variant.name,
                &all_slots,
                &missing,
                suggester,
            );
            out.push(Diagnostic::error(codes::UNKNOWN_FIELD, msg).with_span(tree.span));
        }
    }
}

fn check_children(
    tree: &ParseTree,
    variant: &VariantSchema,
    out: &mut Vec<Diagnostic>,
    suggester: &dyn Suggester,
) {
    // Duplicate child slot names. The two child vectors share one
    // namespace — a slot appearing once positionally and once keyed is
    // a duplicate, not two independent slots — so the scan walks them
    // as a single sequence.
    let mut seen: Vec<&str> = Vec::new();
    for name in slot_names_in_tree(tree) {
        if seen.contains(&name) {
            out.push(
                Diagnostic::error(
                    codes::DUPLICATE_CHILD,
                    format!(
                        "child slot `{}` appears more than once on variant `{}`",
                        name, variant.name
                    ),
                )
                .with_span(tree.span),
            );
        } else {
            seen.push(name);
        }
    }

    // Declared children: keying + arity check.
    for c in &variant.children {
        let positional = tree.child_slot(&c.name);
        let keyed = tree.keyed_child_slot(&c.name);
        let count = positional.map(<[ParseTree]>::len).unwrap_or(0);

        // Keying mismatch: the slot exists, but in the half the schema
        // did not ask for.
        let is_map = c.multiplicity == Multiplicity::Map;
        let wrong_half = if is_map {
            positional.is_some()
        } else {
            keyed.is_some()
        };
        if wrong_half {
            out.push(diag_keying(variant, c, is_map, tree.span));
        }

        // When the slot's *only* occurrence sits in the wrong half,
        // skip the shape check below: it reads the right half, finds
        // it empty, and would report "expects exactly one child but
        // got 0" about a tree that supplied exactly one child. The
        // keying diagnostic already says what went wrong; a second,
        // untrue one only misdirects. A slot present in *both* halves
        // still gets checked — `DUPLICATE_CHILD` covers the collision
        // and the shape verdict on the right half stays meaningful.
        let only_in_wrong_half = wrong_half
            && if is_map {
                keyed.is_none()
            } else {
                positional.is_none()
            };
        if only_in_wrong_half {
            continue;
        }

        match c.multiplicity {
            Multiplicity::One => {
                if count != 1 {
                    out.push(diag_arity(codes::ARITY_ONE, variant, c, count, tree.span));
                }
            }
            Multiplicity::Optional => {
                if count > 1 {
                    out.push(diag_arity(
                        codes::ARITY_OPTIONAL,
                        variant,
                        c,
                        count,
                        tree.span,
                    ));
                }
            }
            Multiplicity::Many => {
                // Zero is fine at the shape level.
            }
            Multiplicity::Map => {
                // Zero-or-more keyed entries is fine at the shape
                // level (same as `Many`); what a map *cannot* carry is
                // the same key twice.
                //
                // Walking adjacent pairs checks that in one linear
                // pass — and the same pass verifies the sorted-by-key
                // order [`ParseTree::keyed_children`] documents, so
                // that contract is enforced rather than trusted. A
                // front-end that emits out of order finds out here
                // instead of shipping trees that compare unequal to
                // every other front-end's for the same document.
                let entries = keyed.unwrap_or(&[]);
                for pair in entries.windows(2) {
                    let (lhs, rhs) = (&pair[0].0, &pair[1].0);
                    match lhs.cmp(rhs) {
                        Ordering::Equal => out.push(
                            Diagnostic::error(
                                codes::DUPLICATE_KEY,
                                format!(
                                    "keyed child slot `{}` on variant `{}` carries key \
                                     `{}` more than once",
                                    c.name, variant.name, rhs
                                ),
                            )
                            .with_span(tree.span),
                        ),
                        Ordering::Greater => out.push(
                            Diagnostic::error(
                                codes::KEYED_SLOT_UNSORTED,
                                format!(
                                    "keyed child slot `{}` on variant `{}` is not sorted by \
                                     key (`{}` precedes `{}`); front-ends must emit keyed \
                                     entries in key order",
                                    c.name, variant.name, lhs, rhs
                                ),
                            )
                            .with_span(tree.span),
                        ),
                        Ordering::Less => {}
                    }
                }
            }
            // `#[non_exhaustive]` catch-all: emitted when a future
            // Multiplicity variant lands in dsl-kit-schema before this
            // arm has been extended to handle it. Not reachable inside
            // the workspace (both crates are versioned together) but
            // required by the type contract and useful for
            // out-of-workspace consumers pinned to an older parse.
            _ => {
                out.push(
                    Diagnostic::error(
                        codes::UNKNOWN_MULTIPLICITY,
                        format!(
                            "child slot `{}` on variant `{}` uses an unrecognised \
                             Multiplicity variant (upgrade dsl-kit-parse to a version \
                             that knows about it)",
                            c.name, variant.name
                        ),
                    )
                    .with_span(tree.span),
                );
            }
        }
    }

    // Same pair-hint precomputation as check_fields, sharing the
    // "declared but currently absent" set across both diagnostics.
    let missing = missing_slot_names(tree, variant);
    let all_slots = all_slot_names(variant);

    // Unknown child slots — both halves, since an undeclared name is
    // equally unknown whether it arrived keyed or positional.
    for name in slot_names_in_tree(tree) {
        if variant.children.iter().any(|c| c.name == name) {
            continue;
        }
        if variant.fields.iter().any(|f| f.name == name) {
            out.push(
                Diagnostic::error(
                    codes::FIELD_AS_CHILD,
                    format!(
                        "`{}` on variant `{}` is a payload field, not a child slot \
                         (move it under `fields`)",
                        name, variant.name
                    ),
                )
                .with_span(tree.span),
            );
        } else {
            let msg = format_unknown_slot(
                "child slot",
                name,
                &variant.name,
                &all_slots,
                &missing,
                suggester,
            );
            out.push(Diagnostic::error(codes::UNKNOWN_CHILD, msg).with_span(tree.span));
        }
    }
}

/// Every child slot name present on `tree`, positional slots first
/// then keyed ones. The two vectors share a single namespace, so both
/// the duplicate scan and the unknown-slot scan walk this sequence
/// rather than either vector alone.
fn slot_names_in_tree(tree: &ParseTree) -> impl Iterator<Item = &str> {
    tree.children
        .iter()
        .map(|(n, _)| n.as_str())
        .chain(tree.keyed_children.iter().map(|(n, _)| n.as_str()))
}

/// Builds the [`codes::KEYED_SLOT_SHAPE`] diagnostic. `positional_seen`
/// distinguishes the two directions: `true` = a `Map`-declared slot
/// arrived under [`ParseTree::children`], `false` = a positional slot
/// arrived under [`ParseTree::keyed_children`].
fn diag_keying(
    variant: &VariantSchema,
    child: &ChildSchema,
    positional_seen: bool,
    span: Option<Span>,
) -> Diagnostic {
    let msg = if positional_seen {
        format!(
            "child slot `{}` on variant `{}` declares Multiplicity::Map — its children \
             must be supplied as keyed entries, not as a positional list",
            child.name, variant.name
        )
    } else {
        format!(
            "child slot `{}` on variant `{}` is positional ({}) — its children must be \
             supplied as a plain list, not as keyed entries",
            child.name,
            variant.name,
            child.multiplicity.as_str()
        )
    };
    Diagnostic::error(codes::KEYED_SLOT_SHAPE, msg).with_span(span)
}

/// Names of every declared payload field and child slot on
/// `variant`, in that order. Used as the `all_candidates` slice for
/// [`format_unknown_slot`].
pub(crate) fn all_slot_names(variant: &VariantSchema) -> Vec<&str> {
    variant
        .fields
        .iter()
        .map(|f| f.name.as_str())
        .chain(variant.children.iter().map(|c| c.name.as_str()))
        .collect()
}

/// Names of declared slots that are **not** currently present on
/// `tree`. Includes fields with no matching entry in `tree.fields`
/// and children whose slot has zero occurrences on the tree. Feeds
/// [`format_unknown_slot`]'s `(missing)` pair-hint marker.
pub(crate) fn missing_slot_names<'a>(tree: &ParseTree, variant: &'a VariantSchema) -> Vec<&'a str> {
    let mut out = Vec::new();
    for f in &variant.fields {
        // Optional fields are never "missing" for pair-hint purposes:
        // their absence is a valid shape, so pointing at them from an
        // unknown-slot suggestion would misdirect the reader.
        if f.optional {
            continue;
        }
        if tree.field(&f.name).is_none() {
            out.push(f.name.as_str());
        }
    }
    for c in &variant.children {
        // A slot counts as present when it carries entries in either
        // half — a keyed slot supplied positionally (or vice versa)
        // already earns its own `KEYED_SLOT_SHAPE` diagnostic, and
        // listing it as "missing" on top of that would misdirect.
        let positional = tree
            .child_slot(&c.name)
            .map(<[ParseTree]>::len)
            .unwrap_or(0);
        let keyed = tree
            .keyed_child_slot(&c.name)
            .map(<[(String, ParseTree)]>::len)
            .unwrap_or(0);
        if positional + keyed == 0 {
            out.push(c.name.as_str());
        }
    }
    out
}

fn diag_arity(
    code: &'static str,
    variant: &VariantSchema,
    child: &ChildSchema,
    count: usize,
    span: Option<Span>,
) -> Diagnostic {
    let expected = match child.multiplicity {
        Multiplicity::One => "exactly one child",
        Multiplicity::Optional => "at most one child",
        Multiplicity::Many => "zero or more children",
        // `Map` never enters `diag_arity` in practice — keyed slots
        // accept zero-or-more entries, so `check_children` has no
        // arity complaint to raise for them (only `DUPLICATE_KEY`).
        // Kept explicit (rather than folded into the `_` arm) so a
        // future change that routes Map through here gets a Map-shaped
        // label instead of the generic "unknown multiplicity" fallback.
        Multiplicity::Map => "any number of keyed children",
        // `#[non_exhaustive]` catch-all — see the sibling arm in
        // `check_children` for the versioning rationale.
        _ => "a shape this build of dsl-kit-parse does not recognise",
    };
    Diagnostic::error(
        code,
        format!(
            "child slot `{}` on variant `{}` expects {} but got {}",
            child.name, variant.name, expected, count
        ),
    )
    .with_span(span)
}

// ---------------------------------------------------------------------------
// Built-in Suggester (case-insensitive Levenshtein, cutoff = max(query.len, 1))
// ---------------------------------------------------------------------------

/// Default [`Suggester`] used by [`check_conformance`] and the serde
/// bridge when no explicit suggester is supplied.
///
/// Backed by an ASCII-case-insensitive Levenshtein edit distance with
/// cutoff `max(query.len(), 1)` — the same behaviour the crate has
/// shipped since Round 1, kept internal so the parser does not need
/// to depend on the plugin crate.
pub(crate) struct BuiltinLevenshteinSuggester;

impl Suggester for BuiltinLevenshteinSuggester {
    fn suggest<'a>(&self, query: &str, candidates: &'a [&str]) -> Vec<Suggestion<'a>> {
        let cutoff = query.chars().count().max(1);
        let mut scored: Vec<(usize, &'a str)> = candidates
            .iter()
            .map(|c| (levenshtein(query, c), *c))
            .filter(|(d, _)| *d <= cutoff)
            .collect();
        scored.sort_by(|(da, na), (db, nb)| da.cmp(db).then_with(|| na.cmp(nb)));
        scored.truncate(3);
        scored
            .into_iter()
            .map(|(d, name)| Suggestion {
                candidate: name,
                score: normalise_edit(d, query, name),
            })
            .collect()
    }
}

fn normalise_edit(distance: usize, a: &str, b: &str) -> f64 {
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    1.0 - (distance as f64 / max_len as f64)
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1].eq_ignore_ascii_case(&b[j - 1]) {
                0
            } else {
                1
            };
            curr[j] = (curr[j - 1] + 1).min(prev[j] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// ---------------------------------------------------------------------------
// Build helpers (used by #[derive(DslBuild)] code emission)
// ---------------------------------------------------------------------------

/// Extracts a payload field into `T`.
///
/// Dispatches on the field's [`RawValue`] arm:
///
/// - [`RawValue::Json`] → `serde_json::from_value::<T>`.
/// - [`RawValue::Text`] → `T::from_str` (G-2: PEG front-end path).
///
/// This means the derive works with either front-end unchanged — the
/// front-end picks the payload representation, the derive is agnostic.
/// `T` must therefore satisfy both bounds; every primitive the kit
/// currently uses (`String`, `i64`, `u32`, `bool`, `f64`) does.
///
/// This helper is intended for use by `#[derive(DslBuild)]`-generated
/// code but is usable directly by hand-written [`DslBuild`] impls.
pub fn build_field<T>(tree: &ParseTree, name: &str) -> Result<T, BuildError>
where
    T: serde::de::DeserializeOwned + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match tree.field(name) {
        Some(RawValue::Json(v)) => serde_json::from_value(v.clone()).map_err(|e| {
            BuildError::single(
                Diagnostic::error(codes::FIELD_TYPE, format!("field `{name}`: {e}"))
                    .with_span(tree.span),
            )
        }),
        Some(RawValue::Text(s)) => s.parse::<T>().map_err(|e| {
            BuildError::single(
                Diagnostic::error(codes::FIELD_TYPE, format!("field `{name}`: {e}"))
                    .with_span(tree.span),
            )
        }),
        None => Err(BuildError::single(
            Diagnostic::error(
                codes::MISSING_FIELD,
                format!("missing required field `{name}`"),
            )
            .with_span(tree.span),
        )),
    }
}

/// Extracts an optional payload field into `Option<T>`.
///
/// Companion to [`build_field`] for fields whose Rust type is
/// `Option<T>` — the `#[derive(DslBuild)]` default path routes here
/// automatically for payload `Option<T>` types (see the derive docs).
///
/// Resolution rules:
///
/// - Absent field → `Ok(None)` (this is the "AI-emit omits noise"
///   contract — schema-level `optional: true` skips the
///   `MISSING_FIELD` diagnostic so absence reaches here).
/// - [`RawValue::Json`] with a `null` value → `Ok(None)`.
/// - [`RawValue::Text`] whose text is the literal `"none"` (the
///   canonical-syntax keyword produced by `schema_gen`) → `Ok(None)`.
/// - Any other value → delegate to [`build_field`], wrap in `Some`.
///
/// `T` must satisfy the same `DeserializeOwned + FromStr` bounds as
/// [`build_field`] because the delegate path uses it.
pub fn build_field_optional<T>(tree: &ParseTree, name: &str) -> Result<Option<T>, BuildError>
where
    T: serde::de::DeserializeOwned + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match tree.field(name) {
        None => Ok(None),
        Some(RawValue::Json(Value::Null)) => Ok(None),
        Some(RawValue::Text(s)) if s == "none" => Ok(None),
        Some(_) => build_field::<T>(tree, name).map(Some),
    }
}

/// Extracts a payload field into `Vec<T>`.
///
/// Companion to [`build_field`] for fields whose Rust type is
/// `Vec<T>` — the `#[derive(DslBuild)]` default path routes here
/// automatically for payload `Vec<T>` types (see the derive docs).
///
/// Resolution rules:
///
/// - Absent field → `Ok(vec![])` (schema-level `optional: true` skips
///   the `MISSING_FIELD` diagnostic; missing = empty list).
/// - [`RawValue::Json`] with a `null` value → `Ok(vec![])`.
/// - [`RawValue::Json`] with an array → `serde_json::from_value`
///   over the whole array.
/// - [`RawValue::Text`] whose trimmed body is empty → `Ok(vec![])`.
/// - [`RawValue::Text`] otherwise → parse the text as a JSON array
///   (`serde_json::from_str::<Vec<T>>`). The canonical-syntax
///   production for `Vec<String>` (`[ "a", "b" ]`) is JSON-parsable
///   as-is; hand-authored value productions that follow the same
///   bracketed shape work unchanged.
///
/// `T` must implement `serde::de::DeserializeOwned` — no `FromStr`
/// bound is needed because the text path routes through
/// `serde_json::from_str` rather than element-wise parsing.
pub fn build_field_vec<T>(tree: &ParseTree, name: &str) -> Result<Vec<T>, BuildError>
where
    T: serde::de::DeserializeOwned,
{
    match tree.field(name) {
        None => Ok(Vec::new()),
        Some(RawValue::Json(Value::Null)) => Ok(Vec::new()),
        Some(RawValue::Json(v)) => serde_json::from_value(v.clone()).map_err(|e| {
            BuildError::single(
                Diagnostic::error(codes::FIELD_TYPE, format!("field `{name}`: {e}"))
                    .with_span(tree.span),
            )
        }),
        Some(RawValue::Text(s)) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return Ok(Vec::new());
            }
            serde_json::from_str(trimmed).map_err(|e| {
                BuildError::single(
                    Diagnostic::error(codes::FIELD_TYPE, format!("field `{name}`: {e}"))
                        .with_span(tree.span),
                )
            })
        }
    }
}

/// Extracts a payload field's raw text production.
///
/// Convenience for `#[dsl_build(with = ...)]` converter functions,
/// which typically match on the spelling the grammar (often a
/// `schema_gen::SyntaxOverrides` value production) captured. Errors
/// mirror [`build_field`]: [`codes::MISSING_FIELD`] when absent,
/// [`codes::FIELD_TYPE`] when the field carries a
/// [`RawValue::Json`] payload instead of text.
pub fn field_text<'t>(tree: &'t ParseTree, name: &str) -> Result<&'t str, BuildError> {
    match tree.field(name) {
        Some(RawValue::Text(s)) => Ok(s),
        Some(RawValue::Json(_)) => Err(BuildError::single(
            Diagnostic::error(
                codes::FIELD_TYPE,
                format!("field `{name}`: expected a text production, found JSON"),
            )
            .with_span(tree.span),
        )),
        None => Err(BuildError::single(
            Diagnostic::error(
                codes::MISSING_FIELD,
                format!("missing required field `{name}`"),
            )
            .with_span(tree.span),
        )),
    }
}

/// Builds the single child of a [`Multiplicity::One`] slot.
///
/// Returns [`codes::ARITY_ONE`] if the slot is absent or holds more
/// than one subtree. Errors from the recursive
/// [`DslBuild::from_parse_tree`] call bubble up as-is.
pub fn build_child_one<T: DslBuild>(
    tree: &ParseTree,
    name: &str,
    ids: &IdGen,
) -> Result<T, BuildError> {
    let slot = tree.child_slot(name).unwrap_or(&[]);
    if slot.len() != 1 {
        return Err(BuildError::single(
            Diagnostic::error(
                codes::ARITY_ONE,
                format!(
                    "child slot `{name}` expects exactly one child but got {}",
                    slot.len()
                ),
            )
            .with_span(tree.span),
        ));
    }
    T::from_parse_tree(&slot[0], ids)
}

/// Builds the (optional) child of a [`Multiplicity::Optional`] slot.
///
/// Returns [`codes::ARITY_OPTIONAL`] if the slot holds more than one
/// subtree. Missing / empty slots yield `Ok(None)`.
pub fn build_child_optional<T: DslBuild>(
    tree: &ParseTree,
    name: &str,
    ids: &IdGen,
) -> Result<Option<T>, BuildError> {
    let slot = tree.child_slot(name).unwrap_or(&[]);
    match slot.len() {
        0 => Ok(None),
        1 => Ok(Some(T::from_parse_tree(&slot[0], ids)?)),
        n => Err(BuildError::single(
            Diagnostic::error(
                codes::ARITY_OPTIONAL,
                format!("child slot `{name}` expects at most one child but got {n}"),
            )
            .with_span(tree.span),
        )),
    }
}

/// Builds every child of a [`Multiplicity::Many`] slot.
///
/// Diagnostics from individual children are collected across the whole
/// slot before returning, so a partially-broken batch surfaces every
/// bad subtree at once rather than the first-encountered one.
pub fn build_child_many<T: DslBuild>(
    tree: &ParseTree,
    name: &str,
    ids: &IdGen,
) -> Result<Vec<T>, BuildError> {
    let slot = tree.child_slot(name).unwrap_or(&[]);
    let mut out = Vec::with_capacity(slot.len());
    let mut diags = Vec::new();
    for child in slot {
        match T::from_parse_tree(child, ids) {
            Ok(v) => out.push(v),
            Err(mut e) => diags.append(&mut e.diagnostics),
        }
    }
    if diags.is_empty() {
        Ok(out)
    } else {
        Err(BuildError::new(diags))
    }
}

/// Builds every entry of a scalar-valued [`Multiplicity::Map`] slot
/// (Shape 1 of the tracking issue) into a [`BTreeMap`] keyed by the
/// front-end's keys.
///
/// Each entry in the tree's [`ParseTree::keyed_children`] slot is
/// expected to be a leaf [`ParseTree`] whose single payload field is
/// named `"value"` — the front-end wraps every scalar value in that
/// canonical shape. The convention gives `build_scalar_map` a stable
/// spot to fetch the payload from without the derive having to know
/// about the wrapping.
///
/// Like [`build_child_map`], duplicate keys emit a
/// [`codes::DUPLICATE_KEY`] diagnostic rather than a silent
/// last-one-wins overwrite. Missing `"value"` fields inside an entry
/// surface as [`codes::MISSING_FIELD`] and payload type errors as
/// [`codes::FIELD_TYPE`] — the same envelope as [`build_field`].
///
/// `V` must satisfy `serde::de::DeserializeOwned + FromStr` so both
/// front-ends (JSON payload / PEG text) can convert values without
/// element-wise divergence.
pub fn build_scalar_map<V>(tree: &ParseTree, name: &str) -> Result<BTreeMap<String, V>, BuildError>
where
    V: serde::de::DeserializeOwned + std::str::FromStr,
    <V as std::str::FromStr>::Err: std::fmt::Display,
{
    let slot = tree.keyed_child_slot(name).unwrap_or(&[]);
    let mut out = BTreeMap::new();
    let mut diags = Vec::new();
    for (key, entry) in slot {
        match build_field::<V>(entry, "value") {
            Ok(v) => {
                if out.insert(key.clone(), v).is_some() {
                    diags.push(
                        Diagnostic::error(
                            codes::DUPLICATE_KEY,
                            format!("keyed child slot `{name}` carries key `{key}` more than once"),
                        )
                        .with_span(tree.span),
                    );
                }
            }
            Err(mut e) => diags.append(&mut e.diagnostics),
        }
    }
    if diags.is_empty() {
        Ok(out)
    } else {
        Err(BuildError::new(diags))
    }
}

/// Builds every entry of a [`Multiplicity::Map`] slot into a
/// [`BTreeMap`] keyed by the front-end's keys.
///
/// [`BTreeMap`] rather than `HashMap` on purpose: the consumer's
/// iteration order is part of the observable AST (walks, re-emission,
/// canonical text), so it has to be deterministic. The derive
/// recognises `BTreeMap<String, T>` for the same reason.
///
/// Like [`build_child_many`], diagnostics from individual entries are
/// collected across the whole slot before returning. A key appearing
/// twice is a [`codes::DUPLICATE_KEY`] error rather than a silent
/// last-one-wins overwrite — the parse would otherwise drop a subtree
/// the author wrote.
pub fn build_child_map<T: DslBuild>(
    tree: &ParseTree,
    name: &str,
    ids: &IdGen,
) -> Result<BTreeMap<String, T>, BuildError> {
    let slot = tree.keyed_child_slot(name).unwrap_or(&[]);
    let mut out = BTreeMap::new();
    let mut diags = Vec::new();
    for (key, child) in slot {
        match T::from_parse_tree(child, ids) {
            Ok(v) => {
                if out.insert(key.clone(), v).is_some() {
                    diags.push(
                        Diagnostic::error(
                            codes::DUPLICATE_KEY,
                            format!("keyed child slot `{name}` carries key `{key}` more than once"),
                        )
                        .with_span(tree.span),
                    );
                }
            }
            Err(mut e) => diags.append(&mut e.diagnostics),
        }
    }
    if diags.is_empty() {
        Ok(out)
    } else {
        Err(BuildError::new(diags))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use dsl_kit_schema::{
        ChildSchema, ChildValueShape, FieldSchema, Multiplicity, NodeSchema, VariantSchema,
    };

    fn schema_add_lit() -> NodeSchema {
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
                        },
                        ChildSchema {
                            name: "rhs".into(),
                            multiplicity: Multiplicity::One,
                            value_shape: ChildValueShape::Recursive,
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
                        },
                        ChildSchema {
                            name: "body".into(),
                            multiplicity: Multiplicity::One,
                            value_shape: ChildValueShape::Recursive,
                        },
                    ],
                },
            ],
        }
    }

    fn lit(v: i64) -> ParseTree {
        let mut t = ParseTree::new("Lit");
        t.fields.push(("value".into(), RawValue::Json(json!(v))));
        t
    }

    #[test]
    fn ok_lit() {
        let schema = schema_add_lit();
        assert!(check_conformance(&lit(1), &schema).is_empty());
    }

    #[test]
    fn ok_add_with_two_children() {
        let schema = schema_add_lit();
        let mut add = ParseTree::new("Add");
        add.children.push(("lhs".into(), vec![lit(1)]));
        add.children.push(("rhs".into(), vec![lit(2)]));
        assert!(check_conformance(&add, &schema).is_empty());
    }

    #[test]
    fn unknown_variant_lists_candidates() {
        let schema = schema_add_lit();
        let tree = ParseTree::new("Aad"); // typo of Add
        let diags = check_conformance(&tree, &schema);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, codes::UNKNOWN_VARIANT);
        assert!(
            diags[0].message.contains("Add"),
            "expected Add to appear as a candidate; got: {}",
            diags[0].message
        );
    }

    #[test]
    fn unknown_variant_uses_injected_suggester() {
        use dsl_kit_fuzzy::FuzzySuggester;

        let schema = schema_add_lit();
        let tree = ParseTree::new("Adx"); // Jaro-Winkler prefix bias picks "Add"
        let suggester = FuzzySuggester::default();
        let diags = check_conformance_with(&tree, &schema, &suggester);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, codes::UNKNOWN_VARIANT);
        assert!(
            diags[0].message.contains("did you mean") && diags[0].message.contains("Add"),
            "expected FuzzySuggester to surface `Add`; got: {}",
            diags[0].message
        );
    }

    #[test]
    fn unknown_field_suggests_declared_field() {
        // The design's canonical example: `taget` should surface `target`
        // as a `did you mean` hint on the UNKNOWN_FIELD diagnostic.
        let schema = NodeSchema {
            name: "Cfg".into(),
            variants: vec![VariantSchema {
                name: "Load".into(),
                fields: vec![
                    FieldSchema {
                        name: "target".into(),
                        ty: "String".into(),
                        optional: false,
                    },
                    FieldSchema {
                        name: "count".into(),
                        ty: "u32".into(),
                        optional: false,
                    },
                ],
                children: vec![],
            }],
        };
        let mut tree = ParseTree::new("Load");
        tree.fields
            .push(("taget".into(), RawValue::Json(json!("a"))));
        tree.fields.push(("count".into(), RawValue::Json(json!(1))));
        let diags = check_conformance(&tree, &schema);

        let unknown = diags
            .iter()
            .find(|d| d.code == codes::UNKNOWN_FIELD)
            .expect("expected an UNKNOWN_FIELD diagnostic for `taget`");
        assert!(
            unknown.message.contains("did you mean") && unknown.message.contains("target"),
            "expected `target` in the hint, got: {}",
            unknown.message
        );
        assert!(
            unknown.message.contains("(missing)"),
            "expected `(missing)` pair-hint since `target` is a required field \
             not present on the tree; got: {}",
            unknown.message
        );
    }

    #[test]
    fn unknown_child_suggests_declared_child() {
        let schema = schema_add_lit();
        let mut add = ParseTree::new("Add");
        // `lsh` is a typo of `lhs`; `rhs` is present so only `lhs`
        // remains in the missing set.
        add.children.push(("lsh".into(), vec![lit(1)]));
        add.children.push(("rhs".into(), vec![lit(2)]));
        let diags = check_conformance(&add, &schema);

        let unknown = diags
            .iter()
            .find(|d| d.code == codes::UNKNOWN_CHILD)
            .expect("expected an UNKNOWN_CHILD diagnostic for `lsh`");
        assert!(
            unknown.message.contains("did you mean") && unknown.message.contains("lhs"),
            "expected `lhs` in the hint, got: {}",
            unknown.message
        );
        assert!(
            unknown.message.contains("(missing)"),
            "expected `(missing)` pair-hint on the `lhs` suggestion since \
             the `lhs` slot is required but absent; got: {}",
            unknown.message
        );
    }

    #[test]
    fn missing_field_is_reported() {
        let schema = schema_add_lit();
        let tree = ParseTree::new("Lit"); // no "value"
        let diags = check_conformance(&tree, &schema);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, codes::MISSING_FIELD);
        assert!(diags[0].message.contains("value"));
    }

    /// Layer 1: absent optional field does NOT trigger `MISSING_FIELD`.
    #[test]
    fn optional_field_absent_is_conformant() {
        let schema = NodeSchema {
            name: "Meta".into(),
            variants: vec![VariantSchema {
                name: "Row".into(),
                fields: vec![
                    FieldSchema::required("id", "u32"),
                    FieldSchema::optional("description", "Option<String>"),
                    FieldSchema::optional("tags", "Vec<String>"),
                ],
                children: vec![],
            }],
        };
        let mut tree = ParseTree::new("Row");
        tree.fields.push(("id".into(), RawValue::Json(json!(1))));
        // description + tags are absent — must not report MISSING_FIELD.
        let diags = check_conformance(&tree, &schema);
        assert!(
            diags.is_empty(),
            "optional fields absent should be clean: {diags:?}",
        );
    }

    /// Layer 1 negative: absent REQUIRED field alongside absent
    /// optional fields still reports only the required one.
    #[test]
    fn optional_field_absence_does_not_shadow_required_missing() {
        let schema = NodeSchema {
            name: "Meta".into(),
            variants: vec![VariantSchema {
                name: "Row".into(),
                fields: vec![
                    FieldSchema::required("id", "u32"),
                    FieldSchema::optional("description", "Option<String>"),
                ],
                children: vec![],
            }],
        };
        let tree = ParseTree::new("Row"); // absent id + absent description
        let diags = check_conformance(&tree, &schema);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, codes::MISSING_FIELD);
        assert!(
            diags[0].message.contains("id"),
            "only `id` is reported missing; got: {}",
            diags[0].message,
        );
    }

    /// Layer 1 wire contract: `to_json` emits `optional: true` only
    /// when the field is optional, preserving the pre-0.3 layout for
    /// required fields.
    #[test]
    fn field_schema_to_json_omits_optional_when_false() {
        let required = FieldSchema::required("id", "u32").to_json();
        assert_eq!(required, json!({ "name": "id", "type": "u32" }));
        assert!(
            required.get("optional").is_none(),
            "required field omits the optional key entirely (pre-0.3 layout)",
        );
        let opt = FieldSchema::optional("description", "Option<String>").to_json();
        assert_eq!(
            opt,
            json!({ "name": "description", "type": "Option<String>", "optional": true }),
        );
    }

    /// Layer 2: `build_field_optional` returns `None` for every
    /// canonical form of absence: missing field, JSON null, canonical
    /// `none` keyword. Present values flow through `build_field`.
    #[test]
    fn build_field_optional_covers_every_absence_shape() {
        let mut t = ParseTree::new("Row");
        // Absent field.
        assert_eq!(
            build_field_optional::<String>(&t, "description").unwrap(),
            None,
        );
        // JSON null.
        t.fields
            .push(("description".into(), RawValue::Json(Value::Null)));
        assert_eq!(
            build_field_optional::<String>(&t, "description").unwrap(),
            None,
        );
        // Canonical "none" keyword (Text path).
        let mut t2 = ParseTree::new("Row");
        t2.fields
            .push(("description".into(), RawValue::Text("none".into())));
        assert_eq!(
            build_field_optional::<String>(&t2, "description").unwrap(),
            None,
        );
        // Present JSON string.
        let mut t3 = ParseTree::new("Row");
        t3.fields
            .push(("description".into(), RawValue::Json(json!("hi"))));
        assert_eq!(
            build_field_optional::<String>(&t3, "description").unwrap(),
            Some("hi".to_string()),
        );
    }

    /// Layer 2: `build_field_vec` defaults to empty on absence, parses
    /// JSON arrays natively, and parses canonical bracketed text
    /// productions via `serde_json::from_str`.
    #[test]
    fn build_field_vec_covers_every_shape() {
        // Absent → empty.
        let t = ParseTree::new("Row");
        assert_eq!(
            build_field_vec::<String>(&t, "tags").unwrap(),
            Vec::<String>::new()
        );
        // JSON null → empty.
        let mut t2 = ParseTree::new("Row");
        t2.fields.push(("tags".into(), RawValue::Json(Value::Null)));
        assert_eq!(
            build_field_vec::<String>(&t2, "tags").unwrap(),
            Vec::<String>::new(),
        );
        // JSON array.
        let mut t3 = ParseTree::new("Row");
        t3.fields
            .push(("tags".into(), RawValue::Json(json!(["a", "b"]))));
        assert_eq!(
            build_field_vec::<String>(&t3, "tags").unwrap(),
            vec!["a".to_string(), "b".to_string()],
        );
        // Canonical bracketed text (JSON-parsable as-is).
        let mut t4 = ParseTree::new("Row");
        t4.fields
            .push(("tags".into(), RawValue::Text(r#"["a", "b"]"#.into())));
        assert_eq!(
            build_field_vec::<String>(&t4, "tags").unwrap(),
            vec!["a".to_string(), "b".to_string()],
        );
        // Empty bracketed text.
        let mut t5 = ParseTree::new("Row");
        t5.fields.push(("tags".into(), RawValue::Text("[]".into())));
        assert_eq!(
            build_field_vec::<String>(&t5, "tags").unwrap(),
            Vec::<String>::new(),
        );
    }

    #[test]
    fn unknown_field_is_reported() {
        let schema = schema_add_lit();
        let mut tree = lit(3);
        tree.fields.push(("extra".into(), RawValue::Json(json!(0))));
        let diags = check_conformance(&tree, &schema);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, codes::UNKNOWN_FIELD);
    }

    #[test]
    fn field_and_child_slot_swapped_reports_structural_codes() {
        let schema = schema_add_lit();

        // A child slot placed under fields.
        let mut swap1 = ParseTree::new("Add");
        swap1.fields.push(("lhs".into(), RawValue::Json(json!(0))));
        swap1.children.push(("rhs".into(), vec![lit(1)]));
        // Missing lhs child (arity_one) + child-as-field structural swap
        // → two diagnostics.
        let diags = check_conformance(&swap1, &schema);
        assert!(diags.iter().any(|d| d.code == codes::CHILD_AS_FIELD));
        assert!(diags.iter().any(|d| d.code == codes::ARITY_ONE));

        // A payload field placed under children.
        let mut swap2 = ParseTree::new("Let");
        swap2
            .children
            .push(("name".into(), vec![ParseTree::new("Lit")]));
        swap2.children.push(("value".into(), vec![lit(1)]));
        swap2.children.push(("body".into(), vec![lit(2)]));
        let diags = check_conformance(&swap2, &schema);
        assert!(diags.iter().any(|d| d.code == codes::FIELD_AS_CHILD));
        assert!(diags.iter().any(|d| d.code == codes::MISSING_FIELD));
    }

    #[test]
    fn arity_one_reports_wrong_count() {
        let schema = schema_add_lit();
        let mut add = ParseTree::new("Add");
        add.children.push(("lhs".into(), vec![lit(1), lit(2)])); // two, not one
        add.children.push(("rhs".into(), vec![lit(3)]));
        let diags = check_conformance(&add, &schema);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, codes::ARITY_ONE);
        assert!(diags[0].message.contains("lhs"));
    }

    #[test]
    fn duplicate_field_and_child_reported() {
        let schema = schema_add_lit();
        let mut t = ParseTree::new("Lit");
        t.fields.push(("value".into(), RawValue::Json(json!(1))));
        t.fields.push(("value".into(), RawValue::Json(json!(2))));
        let diags = check_conformance(&t, &schema);
        assert!(diags.iter().any(|d| d.code == codes::DUPLICATE_FIELD));

        let mut a = ParseTree::new("Add");
        a.children.push(("lhs".into(), vec![lit(1)]));
        a.children.push(("lhs".into(), vec![lit(2)]));
        a.children.push(("rhs".into(), vec![lit(3)]));
        let diags = check_conformance(&a, &schema);
        assert!(diags.iter().any(|d| d.code == codes::DUPLICATE_CHILD));
    }

    #[test]
    fn diagnostic_json_shape() {
        let diag = Diagnostic::error("x::y", "boom").with_span(Some(Span::new(3, 7)));
        let v = diag.to_json();
        assert_eq!(v["severity"], "error");
        assert_eq!(v["code"], "x::y");
        assert_eq!(v["message"], "boom");
        assert_eq!(v["location"]["kind"], "span");
        assert_eq!(v["location"]["start"], 3);
        assert_eq!(v["location"]["end"], 7);
    }

    #[test]
    fn build_error_display_lists_all() {
        let err = BuildError::new(vec![
            Diagnostic::error("a::b", "one"),
            Diagnostic::error("c::d", "two"),
        ]);
        let s = format!("{err}");
        assert!(s.contains("a::b: one"));
        assert!(s.contains("c::d: two"));
    }
}
