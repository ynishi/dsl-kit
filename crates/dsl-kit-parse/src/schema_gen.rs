//! Schema-driven grammar generation (item N).
//!
//! The kit's identity axis is "write the AST in Rust, the toolchain
//! supplies the rest". [`grammar_from_schema`] converts the type-level
//! shape a `#[derive(DslSchema)]` already extracts into a working
//! [`Grammar`] for a canonical text syntax — the DSL author writes two
//! derives and gets an external-format parser for free, instead of
//! hand-authoring a grammar (in Rust or, worse, in a Bison-style JSON
//! blob nobody wants to write).
//!
//! # Canonical syntax
//!
//! One form per variant: the variant name as a keyword, followed by a
//! parenthesised, comma-separated argument list. Arguments appear in
//! schema order — payload fields first, then child slots — each spelled
//! `name: value`:
//!
//! ```text
//! Add(lhs: Lit(value: 1), rhs: Neg(body: none))
//! List(items: [Lit(value: 1), Lit(value: 2)])
//! Name(text: "hello world")
//! Unit()
//! ```
//!
//! - **`One` child** — `name: <node>`.
//! - **`Optional` child** — `name: <node>` or `name: none`. (If a
//!   variant is literally called `none`, PEG ordered choice still
//!   parses `name: none(...)` as that variant; bare `none` means
//!   absent.)
//! - **`Many` child** — `name: [<node>, <node>, ...]`, empty list
//!   allowed.
//! - **`Map` child** — `name: { <key>: <node>, <key>: <node> }`, empty
//!   map allowed. Braces (not the `Many` brackets) mark the slot as
//!   keyed, matching the JSON bridge's object spelling. A key is
//!   either a `%str` literal or a bare `%ident`, so `"a b": …` and
//!   `path: …` are both writable. Entries may be written in any
//!   order — the parser sorts them into the canonical
//!   ascending-by-key form the `ParseTree` requires.
//! - **Payload fields** — by Rust type source text:
//!   `String` → a `%str` literal; the integer types → `%int`;
//!   `bool` → `true` / `false`. Any other payload type fails
//!   generation with [`codes::UNSUPPORTED_FIELD`] — loudly, per
//!   variant, rather than silently dropping the variant from the
//!   grammar — unless a [`SyntaxOverrides`] entry supplies its value
//!   production.
//!
//! # Syntax overrides
//!
//! [`grammar_from_schema_with`] accepts a [`SyntaxOverrides`] carrying
//! value productions for payload types the built-in mapping rejects
//! (e.g. `Option<JoinPolicy>`), keyed by type source text or pinned to
//! one `(variant, field)` site. The override supplies only the value
//! [`Peg`] — the surrounding `name: value` argument shape stays
//! canonical — and its matched text becomes the field's production, so
//! the override author owns making that text consumable downstream
//! (typically `FromStr` on the field type, via
//! [`build_field`](crate::build_field)).
//!
//! # Guarantees
//!
//! The generated grammar is left-recursion-free by construction (every
//! variant rule starts with its keyword token) and contains no nullable
//! repeats; `GrammarCheck`'s `check_left_recursion` /
//! `check_nullable_repeat` / `check_schema_consistency` all pass
//! against it — the tests pin this. Trees produced by parsing satisfy
//! [`check_conformance`](crate::check_conformance) against the source
//! schema.

use std::collections::HashMap;

use crate::grammar_check;
use crate::peg::{
    Grammar, Peg, choice, field, keyed_entry, node, repeat, rule, rule_ref, seq, token,
};
use crate::{BuildError, Diagnostic};
use dsl_kit_core::IdGen;
use dsl_kit_schema::{
    ChildSchema, ChildValueShape, FieldSchema, Multiplicity, NodeSchema, VariantSchema,
};

/// Diagnostic codes emitted by grammar generation.
pub mod codes {
    /// A payload field's Rust type has no canonical-syntax mapping.
    pub const UNSUPPORTED_FIELD: &str = "dsl_kit::schema_gen::unsupported_field";
    /// The schema declares no variants — there is nothing to parse.
    pub const EMPTY_SCHEMA: &str = "dsl_kit::schema_gen::empty_schema";
    /// A child slot's multiplicity is a variant this build of
    /// `dsl-kit-parse` does not recognise (added to the schema crate's
    /// `#[non_exhaustive]` enum ahead of parse-side support). Emitted
    /// by the pre-flight check in
    /// [`grammar_from_schema_with`] so `child_arg_peg` never sees an
    /// unknown variant.
    pub const UNKNOWN_MULTIPLICITY: &str = "dsl_kit::schema_gen::unknown_multiplicity";
    /// A keyed child slot declares a value shape the canonical text
    /// syntax has not yet grown a production for. Emitted by the
    /// pre-flight check so a scalar-valued keyed slot fails loudly
    /// at grammar-generation time rather than silently producing a
    /// grammar that expects each entry's value to be a full AST
    /// node — the shape mismatch would otherwise surface as an
    /// opaque parse failure on the DSL author's first document.
    pub const UNSUPPORTED_MAP_VALUE_SHAPE: &str =
        "dsl_kit::schema_gen::unsupported_map_value_shape";
}

/// Name of the generated start rule.
pub const START_RULE: &str = "node";

/// A deferred value-production constructor: overrides are registered
/// before the [`IdGen`] doing the generation exists, so each entry is a
/// factory invoked with the generator's `ids` at build time.
type PegFactory = Box<dyn Fn(&IdGen) -> Peg>;

/// Value-production overrides for payload fields whose Rust type has no
/// built-in canonical-syntax mapping (see the module docs).
///
/// Two granularities, resolved most-specific-first:
///
/// 1. **Per field** ([`Self::for_field`]) — pinned to one
///    `(variant, field)` site.
/// 2. **Per type** ([`Self::for_type`]) — keyed by the field's Rust
///    type source text; whitespace is ignored, so
///    `"Option<JoinPolicy>"` matches the derive-extracted
///    `"Option < JoinPolicy >"` spelling.
///
/// A field with neither entry falls back to the built-in mapping, and
/// failing that still fails generation loudly with
/// [`codes::UNSUPPORTED_FIELD`].
///
/// The factory builds only the **value** production; the enclosing
/// `name ':' <value>` argument shape (and the `Field` sink binding the
/// matched text to the field name) stays canonical, which keeps the
/// generated grammar's `GrammarCheck` guarantees intact.
#[derive(Default)]
pub struct SyntaxOverrides {
    by_type: HashMap<String, PegFactory>,
    by_field: HashMap<(String, String), PegFactory>,
}

impl SyntaxOverrides {
    /// An empty override set (equivalent to the built-in mapping only).
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a value production for every payload field of Rust
    /// type `ty` (source-text comparison, whitespace-insensitive).
    pub fn for_type(
        mut self,
        ty: impl AsRef<str>,
        value: impl Fn(&IdGen) -> Peg + 'static,
    ) -> Self {
        self.by_type.insert(strip_ws(ty.as_ref()), Box::new(value));
        self
    }

    /// Registers a value production for the single field `field` of
    /// variant `variant`, winning over any [`Self::for_type`] entry.
    pub fn for_field(
        mut self,
        variant: impl Into<String>,
        field: impl Into<String>,
        value: impl Fn(&IdGen) -> Peg + 'static,
    ) -> Self {
        self.by_field
            .insert((variant.into(), field.into()), Box::new(value));
        self
    }

    /// Most-specific matching factory for `field` of `variant`, if any.
    fn resolve(&self, variant: &str, field: &FieldSchema) -> Option<&PegFactory> {
        self.by_field
            .get(&(variant.to_string(), field.name.clone()))
            .or_else(|| self.by_type.get(&strip_ws(&field.ty)))
    }
}

/// Whitespace-insensitive type key: `Option < JoinPolicy >` (the
/// token-stream spelling `#[derive(DslSchema)]` extracts) and
/// `Option<JoinPolicy>` (what an author writes) must compare equal.
fn strip_ws(ty: &str) -> String {
    ty.chars().filter(|c| !c.is_whitespace()).collect()
}

impl Grammar {
    /// Generates the canonical-syntax grammar for `schema`. See
    /// [`grammar_from_schema`].
    pub fn from_schema(schema: &NodeSchema, ids: &IdGen) -> Result<Grammar, BuildError> {
        grammar_from_schema(schema, ids)
    }
}

/// Generates a [`Grammar`] for the canonical named-argument syntax of
/// `schema` (see the module docs for the syntax), built-in field
/// mapping only.
///
/// Fails with one [`codes::UNSUPPORTED_FIELD`] diagnostic per
/// unmappable payload field (collected across all variants, so the
/// author sees the full list at once). To map such fields instead of
/// failing, use [`grammar_from_schema_with`].
pub fn grammar_from_schema(schema: &NodeSchema, ids: &IdGen) -> Result<Grammar, BuildError> {
    grammar_from_schema_with(schema, ids, &SyntaxOverrides::default())
}

/// [`grammar_from_schema`] with [`SyntaxOverrides`] consulted before
/// the built-in field mapping (per-field entries win over per-type
/// entries win over built-ins).
pub fn grammar_from_schema_with(
    schema: &NodeSchema,
    ids: &IdGen,
    overrides: &SyntaxOverrides,
) -> Result<Grammar, BuildError> {
    if schema.variants.is_empty() {
        return Err(BuildError::single(Diagnostic::error(
            codes::EMPTY_SCHEMA,
            format!("schema `{}` declares no variants", schema.name),
        )));
    }

    // Pre-flight: collect every diagnostic that must block rule
    // generation, then return in one shot. A single pass means an
    // author with both a bad payload type and a `Multiplicity::Map`
    // slot sees both errors on the first build instead of paying two
    // edit-compile cycles for one report.
    //
    // Grouping:
    //   - `UNSUPPORTED_FIELD` for payload types with no canonical-syntax
    //     mapping (and no `SyntaxOverrides` entry).
    //   - `UNKNOWN_MULTIPLICITY` for child slots whose multiplicity
    //     this build does not know, which is what keeps
    //     `child_arg_peg`'s catch-all arm from ever running.
    let mut pre_flight = Vec::new();
    for v in &schema.variants {
        for f in &v.fields {
            if resolved_field_value_peg(&v.name, f, ids, overrides).is_none() {
                pre_flight.push(Diagnostic::error(
                    codes::UNSUPPORTED_FIELD,
                    format!(
                        "variant `{}` field `{}`: type `{}` has no canonical-syntax \
                         mapping (supported: String, bool, the integer types; or \
                         register a `SyntaxOverrides` value production)",
                        v.name, f.name, f.ty
                    ),
                ));
            }
        }
        for c in &v.children {
            match c.multiplicity {
                // Every shape `child_arg_peg` knows how to emit.
                Multiplicity::One | Multiplicity::Optional | Multiplicity::Many => {}
                // Keyed slots come in two value shapes at derive
                // level: recursive (`BTreeMap<String, Self>` /
                // `BTreeMap<String, Box<Self>>`) is supported by the
                // canonical text syntax; scalar
                // (`ChildValueShape::Scalar { .. }`, Shape 1 of the
                // tracking issue) has not grown a text production
                // yet — `child_arg_peg` would emit
                // `key: <node>` for every entry, producing an
                // unusable grammar. Fail loudly instead so the DSL
                // author is directed to the JSON front-end.
                Multiplicity::Map => {
                    if let ChildValueShape::Scalar { ty } = &c.value_shape {
                        pre_flight.push(Diagnostic::error(
                            codes::UNSUPPORTED_MAP_VALUE_SHAPE,
                            format!(
                                "variant `{}` child slot `{}`: keyed scalar values \
                                 (`BTreeMap<String, {}>`) are recognised by the derive and \
                                 the JSON front-end, but the canonical text syntax has no \
                                 production for them yet — use the JSON front-end for \
                                 scalar-map DSLs, or open the tracking issue if the text \
                                 syntax is needed",
                                v.name, c.name, ty
                            ),
                        ));
                    }
                }
                // `#[non_exhaustive]` catch-all: a future Multiplicity
                // variant added to dsl-kit-schema before this crate is
                // updated must also be rejected here — that's what
                // keeps `child_arg_peg`'s `_ => unreachable!()` arm
                // honest.
                _ => {
                    pre_flight.push(Diagnostic::error(
                        codes::UNKNOWN_MULTIPLICITY,
                        format!(
                            "variant `{}` child slot `{}`: unrecognised Multiplicity \
                             variant (upgrade dsl-kit-parse to a version that knows about it)",
                            v.name, c.name
                        ),
                    ));
                }
            }
        }
    }
    if !pre_flight.is_empty() {
        return Err(BuildError::new(pre_flight));
    }

    let mut rules = Vec::with_capacity(schema.variants.len() + 1);
    rules.push(rule(
        ids,
        START_RULE,
        choice(
            ids,
            schema
                .variants
                .iter()
                .map(|v| rule_ref(ids, v.name.clone()))
                .collect(),
        ),
    ));
    for v in &schema.variants {
        rules.push(variant_rule(v, ids, overrides));
    }
    Ok(Grammar::new(rules, START_RULE))
}

/// Builds the rule for one variant. Two shapes:
///
/// - **Strict-order** (default, variant has no optional fields):
///   `%kw:V "(" arg ("," arg)* ")"` with arguments emitted in schema
///   order (fields, then children). Preserves the pre-0.3 canonical
///   spelling for variants that never omit.
/// - **Free-order** (variant has ≥ 1 optional field): `%kw:V "(" (arg
///   ("," arg)*)? ")"` where each arg is `arg_1 | arg_2 | ...` and
///   every alternative carries its own `%kw:name` prefix. Args are
///   distinguishable by that name-keyword, so the DSL author may
///   omit any subset of optional arguments (or even all of them),
///   and `check_conformance` remains the authority on required /
///   duplicate / unknown-slot diagnostics.
fn variant_rule(v: &VariantSchema, ids: &IdGen, overrides: &SyntaxOverrides) -> Peg {
    let has_optional = v.fields.iter().any(|f| f.optional);
    let mut args: Vec<Peg> = Vec::new();
    for f in &v.fields {
        let value = resolved_field_value_peg(&v.name, f, ids, overrides)
            .expect("unsupported field types were rejected before rule generation");
        args.push(seq(
            ids,
            vec![
                token(ids, format!("%kw:{}", f.name)),
                token(ids, ":"),
                field(ids, f.name.clone(), value),
            ],
        ));
    }
    for c in &v.children {
        args.push(child_arg_peg(c, ids));
    }

    let mut items = vec![token(ids, format!("%kw:{}", v.name)), token(ids, "(")];
    if has_optional && !args.is_empty() {
        // Free-order: any of the argument shapes, comma-separated,
        // whole list optional. Each argument-slot is a Field capture
        // in its own right (built above), and the name-keyword prefix
        // keeps the alternation unambiguous at the byte level.
        let arg_choice = choice(ids, args);
        // The tail repeats `("," arg_choice)`; we rebuild arg_choice
        // per position so each Choice node gets a fresh id (Peg nodes
        // are id-unique in a grammar).
        let tail_alt = |ids: &IdGen| -> Peg {
            let mut alts: Vec<Peg> = Vec::new();
            for f in &v.fields {
                let value = resolved_field_value_peg(&v.name, f, ids, overrides)
                    .expect("unsupported field types were rejected before rule generation");
                alts.push(seq(
                    ids,
                    vec![
                        token(ids, format!("%kw:{}", f.name)),
                        token(ids, ":"),
                        field(ids, f.name.clone(), value),
                    ],
                ));
            }
            for c in &v.children {
                alts.push(child_arg_peg(c, ids));
            }
            choice(ids, alts)
        };
        let tail = repeat(ids, seq(ids, vec![token(ids, ","), tail_alt(ids)]), 0, None);
        let full = seq(ids, vec![arg_choice, tail]);
        items.push(repeat(ids, full, 0, Some(1)));
    } else {
        for (i, arg) in args.into_iter().enumerate() {
            if i > 0 {
                items.push(token(ids, ","));
            }
            items.push(arg);
        }
    }
    items.push(token(ids, ")"));

    rule(
        ids,
        v.name.clone(),
        node(ids, v.name.clone(), seq(ids, items)),
    )
}

/// Value production for a payload field: the most specific
/// [`SyntaxOverrides`] entry if registered, else the built-in mapping.
fn resolved_field_value_peg(
    variant: &str,
    f: &FieldSchema,
    ids: &IdGen,
    overrides: &SyntaxOverrides,
) -> Option<Peg> {
    match overrides.resolve(variant, f) {
        Some(factory) => Some(factory(ids)),
        None => field_value_peg(f, ids),
    }
}

/// Built-in value production for a payload field, by Rust type source
/// text. `None` when the type has no canonical-syntax mapping.
///
/// Recognised (whitespace-insensitive on the type-source spelling):
///
/// - `String` → `%str`
/// - the integer types → `%int`
/// - `bool` → `true` / `false`
/// - `Option<String>` → `none` | `%str`
/// - `Vec<String>` → `[ %str ("," %str)* ]`, empty list allowed
///
/// The last two entries realise Layer 2 of the "built-in optional
/// payload fields" contract — downstream DSLs no longer need to
/// register a [`SyntaxOverrides`] entry (nor a paired
/// `#[dsl_build(with = ...)]` converter) for these shapes.
fn field_value_peg(f: &FieldSchema, ids: &IdGen) -> Option<Peg> {
    const INT_TYPES: &[&str] = &[
        "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "usize", "isize",
    ];
    let ty = strip_ws(&f.ty);
    if ty == "String" {
        Some(token(ids, "%str"))
    } else if INT_TYPES.contains(&ty.as_str()) {
        Some(token(ids, "%int"))
    } else if ty == "bool" {
        Some(choice(
            ids,
            vec![token(ids, "%kw:true"), token(ids, "%kw:false")],
        ))
    } else if ty == "Option<String>" {
        // Canonical spelling of an absent Option-payload; mirrors the
        // `Multiplicity::Optional` child idiom.
        Some(choice(
            ids,
            vec![token(ids, "%kw:none"), token(ids, "%str")],
        ))
    } else if ty == "Vec<String>" {
        // `[ %str_raw ("," %str_raw)* ]` with empty list allowed.
        // `%str_raw` (peg-level primitive) contributes the raw
        // source slice for each element — quotes and escapes intact —
        // so the joined field text is a JSON-compatible array literal
        // that `build_field_vec` can hand straight to
        // `serde_json::from_str`.
        let elems = seq(
            ids,
            vec![
                token(ids, "%str_raw"),
                repeat(
                    ids,
                    seq(ids, vec![token(ids, ","), token(ids, "%str_raw")]),
                    0,
                    None,
                ),
            ],
        );
        Some(seq(
            ids,
            vec![
                token(ids, "["),
                repeat(ids, elems, 0, Some(1)),
                token(ids, "]"),
            ],
        ))
    } else {
        None
    }
}

/// Argument production for one child slot, by multiplicity.
fn child_arg_peg(c: &ChildSchema, ids: &IdGen) -> Peg {
    let name_kw = token(ids, format!("%kw:{}", c.name));
    let colon = token(ids, ":");
    match c.multiplicity {
        Multiplicity::One => seq(
            ids,
            vec![
                name_kw,
                colon,
                field(ids, c.name.clone(), rule_ref(ids, START_RULE)),
            ],
        ),
        // `name: <node>` or `name: none`. The Field arm is tried first,
        // so a variant literally named `none` still wins when followed
        // by `(`. The bare-`none` arm sits *outside* any Field: its
        // matched text lands on the enclosing Node sink where it is
        // dropped as syntactic noise, leaving the slot absent.
        Multiplicity::Optional => seq(
            ids,
            vec![
                name_kw,
                colon,
                choice(
                    ids,
                    vec![
                        field(ids, c.name.clone(), rule_ref(ids, START_RULE)),
                        token(ids, "%kw:none"),
                    ],
                ),
            ],
        ),
        // `name: [ <node> ("," <node>)* ]`, empty list allowed. The
        // list commas live inside the Field body; the mixed-field
        // "trees win" fallback drops them as noise.
        Multiplicity::Many => {
            let elems = seq(
                ids,
                vec![
                    rule_ref(ids, START_RULE),
                    repeat(
                        ids,
                        seq(ids, vec![token(ids, ","), rule_ref(ids, START_RULE)]),
                        0,
                        None,
                    ),
                ],
            );
            seq(
                ids,
                vec![
                    name_kw,
                    colon,
                    token(ids, "["),
                    field(ids, c.name.clone(), repeat(ids, elems, 0, Some(1))),
                    token(ids, "]"),
                ],
            )
        }
        // `name: { <key>: <node> ("," <key>: <node>)* }`, empty map
        // allowed. Braces rather than the `Many` idiom's brackets so
        // the two shapes are distinguishable at a glance, and matching
        // the JSON bridge's object spelling so a DSL reads the same
        // way in both front-ends.
        //
        // The key token is `%str | %ident`: keys are arbitrary strings
        // at the `ParseTree` level, so a quoted form has to exist, but
        // requiring quotes around every `PATH`-style key would be
        // noise. `%str` is tried first — a quoted key is unambiguous,
        // while `%ident` cannot start with `"` — and its production is
        // the decoded content, so `"a b"` and `a` both arrive as plain
        // key text.
        Multiplicity::Map => {
            //
            // The `:` separator rides on the *value* half, not the
            // key half: every token matched inside the key production
            // contributes to the key text, so a `:` there would land
            // in the key itself. On the value side it is dropped as
            // syntactic noise, the same way the `Many` list's commas
            // are.
            let entry = |ids: &IdGen| -> Peg {
                keyed_entry(
                    ids,
                    c.name.clone(),
                    choice(ids, vec![token(ids, "%str"), token(ids, "%ident")]),
                    seq(ids, vec![token(ids, ":"), rule_ref(ids, START_RULE)]),
                )
            };
            let entries = seq(
                ids,
                vec![
                    entry(ids),
                    repeat(ids, seq(ids, vec![token(ids, ","), entry(ids)]), 0, None),
                ],
            );
            seq(
                ids,
                vec![
                    name_kw,
                    colon,
                    token(ids, "{"),
                    repeat(ids, entries, 0, Some(1)),
                    token(ids, "}"),
                ],
            )
        }
        // `#[non_exhaustive]` catch-all: a future Multiplicity variant
        // added to dsl-kit-schema before this crate is updated would
        // land here. The pre-flight check in
        // `grammar_from_schema_with` should be extended in lockstep to
        // reject unknown variants; if that extension is missed the
        // panic here surfaces the omission loudly instead of emitting
        // a silently-wrong grammar rule.
        _ => unreachable!(
            "child_arg_peg encountered an unrecognised Multiplicity variant on slot \
             `{}` — extend the pre-flight check in grammar_from_schema_with to reject it",
            c.name
        ),
    }
}

/// Convenience: generate and immediately assert the grammar is clean
/// under `GrammarCheck` (left recursion, nullable repeats, schema
/// consistency). Intended for tests and one-shot setup paths; returns
/// the same grammar as [`grammar_from_schema`].
pub fn checked_grammar_from_schema(
    schema: &NodeSchema,
    ids: &IdGen,
) -> Result<Grammar, BuildError> {
    checked_grammar_from_schema_with(schema, ids, &SyntaxOverrides::default())
}

/// [`checked_grammar_from_schema`] with [`SyntaxOverrides`] consulted
/// before the built-in field mapping.
pub fn checked_grammar_from_schema_with(
    schema: &NodeSchema,
    ids: &IdGen,
    overrides: &SyntaxOverrides,
) -> Result<Grammar, BuildError> {
    let grammar = grammar_from_schema_with(schema, ids, overrides)?;
    let mut diags = grammar_check::check_left_recursion(&grammar);
    diags.extend(grammar_check::check_nullable_repeat(&grammar));
    diags.extend(
        grammar_check::check_schema_consistency(&grammar, schema)
            .into_iter()
            .filter(|d| d.severity == crate::Severity::Error),
    );
    if diags.is_empty() {
        Ok(grammar)
    } else {
        Err(BuildError::new(diags))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RawValue, check_conformance};
    use dsl_kit_schema::ChildValueShape;

    /// Expr-flavoured schema exercising every supported shape: int /
    /// string / bool fields, One / Optional / Many children, and a
    /// zero-argument variant.
    fn demo_schema() -> NodeSchema {
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
                    name: "Name".into(),
                    fields: vec![
                        FieldSchema {
                            name: "text".into(),
                            ty: "String".into(),
                            optional: false,
                        },
                        FieldSchema {
                            name: "quoted".into(),
                            ty: "bool".into(),
                            optional: false,
                        },
                    ],
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
                    name: "Neg".into(),
                    fields: vec![],
                    children: vec![ChildSchema {
                        name: "body".into(),
                        multiplicity: Multiplicity::Optional,
                        value_shape: ChildValueShape::Recursive,
                    }],
                },
                VariantSchema {
                    name: "List".into(),
                    fields: vec![],
                    children: vec![ChildSchema {
                        name: "items".into(),
                        multiplicity: Multiplicity::Many,
                        value_shape: ChildValueShape::Recursive,
                    }],
                },
                VariantSchema {
                    name: "Env".into(),
                    fields: vec![],
                    children: vec![ChildSchema {
                        name: "entries".into(),
                        multiplicity: Multiplicity::Map,
                        value_shape: ChildValueShape::Recursive,
                    }],
                },
                VariantSchema {
                    name: "Unit".into(),
                    fields: vec![],
                    children: vec![],
                },
            ],
        }
    }

    fn demo_grammar() -> Grammar {
        checked_grammar_from_schema(&demo_schema(), &IdGen::new())
            .expect("demo schema generates a clean grammar")
    }

    /// Parse + conformance-check against the demo schema in one go.
    fn parse_ok(input: &str) -> crate::ParseTree {
        let g = demo_grammar();
        let tree = g
            .parse(input)
            .unwrap_or_else(|e| panic!("parse failed for {input:?}: {:?}", e.diagnostics));
        let diags = check_conformance(&tree, &demo_schema());
        assert!(
            diags.is_empty(),
            "conformance clean for {input:?}: {diags:?}"
        );
        tree
    }

    #[test]
    fn generated_grammar_passes_all_static_checks() {
        // checked_grammar_from_schema already runs the three checks;
        // this test pins that it succeeds for the full shape matrix.
        demo_grammar();
    }

    #[test]
    fn parses_int_field_variant() {
        let tree = parse_ok("Lit(value: -42)");
        assert_eq!(tree.variant, "Lit");
        assert_eq!(tree.field("value"), Some(&RawValue::Text("-42".into())));
    }

    #[test]
    fn parses_string_and_bool_fields_with_escapes() {
        let tree = parse_ok(r#"Name(text: "hello \"world\"\n", quoted: true)"#);
        assert_eq!(
            tree.field("text"),
            Some(&RawValue::Text("hello \"world\"\n".into()))
        );
        assert_eq!(tree.field("quoted"), Some(&RawValue::Text("true".into())));
    }

    #[test]
    fn parses_nested_one_children() {
        let tree = parse_ok("Add(lhs: Lit(value: 1), rhs: Neg(body: none))");
        let lhs = tree.child_slot("lhs").unwrap();
        assert_eq!(lhs.len(), 1);
        assert_eq!(lhs[0].variant, "Lit");
        let rhs = tree.child_slot("rhs").unwrap();
        assert_eq!(rhs[0].variant, "Neg");
    }

    #[test]
    fn optional_child_none_leaves_slot_absent() {
        let tree = parse_ok("Neg(body: none)");
        assert!(
            tree.child_slot("body").is_none(),
            "bare `none` = absent slot"
        );
    }

    #[test]
    fn optional_child_present_binds_one_tree() {
        let tree = parse_ok("Neg(body: Lit(value: 3))");
        assert_eq!(tree.child_slot("body").unwrap().len(), 1);
    }

    #[test]
    fn many_child_list_binds_in_order_and_drops_commas() {
        let tree = parse_ok("List(items: [Lit(value: 1), Lit(value: 2), Unit()])");
        let items = tree.child_slot("items").unwrap();
        let variants: Vec<&str> = items.iter().map(|t| t.variant.as_str()).collect();
        assert_eq!(variants, ["Lit", "Lit", "Unit"]);
    }

    #[test]
    fn many_child_empty_list_is_conformant() {
        let tree = parse_ok("List(items: [])");
        assert!(
            tree.child_slot("items").is_none_or(<[_]>::is_empty),
            "empty list = zero children"
        );
    }

    #[test]
    fn keyed_map_binds_entries_sorted_by_key() {
        // Written deliberately out of order, with one quoted key that
        // could not be spelled as a bare identifier: the parser is
        // responsible for landing both in canonical order.
        let tree = parse_ok(r#"Env(entries: { zeta: Unit(), "a b": Lit(value: 1) })"#);
        let entries = tree.keyed_child_slot("entries").expect("keyed slot bound");
        let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["a b", "zeta"]);
        let variants: Vec<&str> = entries.iter().map(|(_, t)| t.variant.as_str()).collect();
        assert_eq!(variants, ["Lit", "Unit"]);
        assert!(
            tree.children.is_empty(),
            "keyed entries must not leak into the positional half: {:?}",
            tree.children
        );
    }

    #[test]
    fn keyed_map_source_order_does_not_change_the_tree() {
        let forward = parse_ok("Env(entries: { a: Unit(), b: Lit(value: 1) })");
        let reversed = parse_ok("Env(entries: { b: Lit(value: 1), a: Unit() })");
        // Spans differ (the two spellings put the subtrees at
        // different offsets), so compare the keyed payload rather than
        // the whole tree.
        let keys = |t: &crate::ParseTree| -> Vec<String> {
            t.keyed_child_slot("entries")
                .unwrap()
                .iter()
                .map(|(k, v)| format!("{k}={}", v.variant))
                .collect()
        };
        assert_eq!(keys(&forward), keys(&reversed));
    }

    #[test]
    fn keyed_map_empty_is_conformant() {
        let tree = parse_ok("Env(entries: {})");
        assert!(
            tree.keyed_child_slot("entries").is_none_or(<[_]>::is_empty),
            "empty map = zero entries"
        );
    }

    #[test]
    fn keyed_map_duplicate_key_parses_but_fails_conformance() {
        // The grammar has no reason to reject it — duplicate detection
        // is the schema's job, and `parse_ok` would hide the split, so
        // this test drives the two stages apart deliberately.
        let g = demo_grammar();
        let tree = g
            .parse("Env(entries: { k: Unit(), k: Lit(value: 1) })")
            .expect("grammar accepts a repeated key");
        let diags = check_conformance(&tree, &demo_schema());
        assert!(
            diags.iter().any(|d| d.code == crate::codes::DUPLICATE_KEY),
            "expected DUPLICATE_KEY; got {:?}",
            diags.iter().map(|d| d.code.as_str()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn zero_argument_variant_parses() {
        let tree = parse_ok("Unit()");
        assert_eq!(tree.variant, "Unit");
        assert!(tree.fields.is_empty() && tree.children.is_empty());
    }

    #[test]
    fn whitespace_and_newlines_are_tolerated() {
        let tree = parse_ok("Add(\n  lhs: Lit( value : 7 ),\n  rhs: Unit()\n)");
        assert_eq!(tree.child_slot("lhs").unwrap()[0].variant, "Lit");
    }

    #[test]
    fn unsupported_field_type_fails_generation_with_full_list() {
        let schema = NodeSchema {
            name: "Bad".into(),
            variants: vec![VariantSchema {
                name: "Par".into(),
                fields: vec![
                    FieldSchema {
                        name: "policy".into(),
                        ty: "Option<JoinPolicy>".into(),
                        optional: false,
                    },
                    FieldSchema {
                        name: "reducer".into(),
                        ty: "ReducerId".into(),
                        optional: false,
                    },
                ],
                children: vec![],
            }],
        };
        let err = grammar_from_schema(&schema, &IdGen::new()).unwrap_err();
        assert_eq!(err.diagnostics.len(), 2, "one diagnostic per bad field");
        assert!(
            err.diagnostics
                .iter()
                .all(|d| d.code == codes::UNSUPPORTED_FIELD)
        );
    }

    /// A scalar-valued keyed slot is recognised by the derive and
    /// the JSON front-end, but the canonical text syntax has no
    /// production for it yet — `grammar_from_schema` must fail
    /// loudly with [`codes::UNSUPPORTED_MAP_VALUE_SHAPE`] rather
    /// than silently emit a grammar that expects each entry's value
    /// to be a full AST node. Pins the pre-flight guard so a
    /// downstream feeding a Shape 1 schema through PEG generation
    /// gets a clear pointer to the JSON front-end instead of an
    /// opaque parse failure.
    #[test]
    fn scalar_keyed_slot_fails_grammar_generation_loudly() {
        let schema = NodeSchema {
            name: "Cfg".into(),
            variants: vec![VariantSchema {
                name: "Env".into(),
                fields: vec![],
                children: vec![ChildSchema::scalar_map("entries", "String")],
            }],
        };
        let err = grammar_from_schema(&schema, &IdGen::new()).unwrap_err();
        assert_eq!(err.diagnostics.len(), 1);
        assert_eq!(err.diagnostics[0].code, codes::UNSUPPORTED_MAP_VALUE_SHAPE);
        assert!(
            err.diagnostics[0].message.contains("entries"),
            "diagnostic mentions the slot name: {}",
            err.diagnostics[0].message
        );
        assert!(
            err.diagnostics[0].message.contains("String"),
            "diagnostic mentions the scalar value type: {}",
            err.diagnostics[0].message
        );
    }

    /// Recursive keyed slots still generate cleanly — the scalar
    /// gate must not accidentally reject the pre-0.6 keyed shape.
    #[test]
    fn recursive_keyed_slot_still_generates_grammar() {
        let schema = NodeSchema {
            name: "Cfg".into(),
            variants: vec![
                VariantSchema {
                    name: "Env".into(),
                    fields: vec![],
                    children: vec![ChildSchema::recursive("entries", Multiplicity::Map)],
                },
                VariantSchema {
                    name: "Unit".into(),
                    fields: vec![],
                    children: vec![],
                },
            ],
        };
        grammar_from_schema(&schema, &IdGen::new())
            .expect("recursive keyed shape still supported end-to-end in the text front-end");
    }

    /// Flow-shaped schema whose `Par` payload fields the built-in
    /// mapping rejects — the override motivating case.
    fn par_schema() -> NodeSchema {
        NodeSchema {
            name: "Flow".into(),
            variants: vec![
                VariantSchema {
                    name: "Par".into(),
                    fields: vec![
                        FieldSchema {
                            name: "policy".into(),
                            // Token-stream spelling, as the derive extracts it.
                            ty: "Option < JoinPolicy >".into(),
                            optional: false,
                        },
                        FieldSchema {
                            name: "reducer_id".into(),
                            ty: "Option < String >".into(),
                            optional: false,
                        },
                    ],
                    children: vec![ChildSchema {
                        name: "children".into(),
                        multiplicity: Multiplicity::Many,
                        value_shape: ChildValueShape::Recursive,
                    }],
                },
                VariantSchema {
                    name: "Call".into(),
                    fields: vec![FieldSchema {
                        name: "label".into(),
                        ty: "String".into(),
                        optional: false,
                    }],
                    children: vec![],
                },
            ],
        }
    }

    fn par_overrides() -> SyntaxOverrides {
        SyntaxOverrides::new()
            .for_type("Option<JoinPolicy>", |ids| {
                choice(
                    ids,
                    vec![
                        token(ids, "%kw:none"),
                        token(ids, "%kw:all_failfast"),
                        token(ids, "%kw:any_failfast"),
                    ],
                )
            })
            .for_type("Option<String>", |ids| {
                choice(ids, vec![token(ids, "%kw:none"), token(ids, "%str")])
            })
    }

    #[test]
    fn type_overrides_unlock_exotic_payload_fields() {
        // Without overrides the Flow-shaped schema fails generation
        // (pinned by unsupported_field_type_fails_generation_with_full_list);
        // with them it generates check-clean and parses. Note the
        // override keys are written without whitespace while the schema
        // carries the token-stream spelling.
        let g = checked_grammar_from_schema_with(&par_schema(), &IdGen::new(), &par_overrides())
            .expect("overridden Flow schema generates a clean grammar");
        let tree = g
            .parse(
                r#"Par(policy: all_failfast, reducer_id: "reduce_all_ordered",
                     children: [Call(label: "a"), Call(label: "b")])"#,
            )
            .expect("canonical Par text parses");
        let diags = check_conformance(&tree, &par_schema());
        assert!(diags.is_empty(), "conformance clean: {diags:?}");
        assert_eq!(
            tree.field("policy"),
            Some(&RawValue::Text("all_failfast".into()))
        );
        assert_eq!(
            tree.field("reducer_id"),
            Some(&RawValue::Text("reduce_all_ordered".into()))
        );
        assert_eq!(tree.child_slot("children").unwrap().len(), 2);
    }

    #[test]
    fn field_override_wins_over_type_override() {
        // Same type in both entries; the (variant, field)-pinned one
        // must decide the syntax.
        let overrides = par_overrides().for_field("Par", "policy", |ids| token(ids, "%kw:pinned"));
        let g = checked_grammar_from_schema_with(&par_schema(), &IdGen::new(), &overrides)
            .expect("clean grammar with field-level override");
        let tree = g
            .parse(r#"Par(policy: pinned, reducer_id: none, children: [])"#)
            .expect("field-level syntax parses");
        assert_eq!(tree.field("policy"), Some(&RawValue::Text("pinned".into())));
        // The type-level spelling is no longer accepted at that site.
        assert!(
            g.parse(r#"Par(policy: all_failfast, reducer_id: none, children: [])"#)
                .is_err(),
            "type-level syntax rejected once the field is pinned"
        );
    }

    #[test]
    fn unrelated_override_leaves_builtin_mapping_and_failure_intact() {
        // An override for some other type neither rescues the
        // unsupported fields nor perturbs built-in ones. Note that
        // `Option<String>` on `reducer_id` is now a built-in mapping
        // (Layer 2 of the "built-in optional payload fields"
        // contract), so only the `Option<JoinPolicy>` field on
        // `policy` remains unmapped without a `SyntaxOverrides` entry.
        let overrides = SyntaxOverrides::new().for_type("Uuid", |ids| token(ids, "%str"));
        let err = grammar_from_schema_with(&par_schema(), &IdGen::new(), &overrides).unwrap_err();
        assert_eq!(
            err.diagnostics.len(),
            1,
            "policy is the only Par field still unmapped without overrides",
        );
        assert!(
            err.diagnostics
                .iter()
                .all(|d| d.code == codes::UNSUPPORTED_FIELD)
        );
        assert!(
            err.diagnostics[0].message.contains("policy"),
            "unmapped-field diagnostic points at `policy`",
        );
        // Built-in demo schema is unaffected by the stray entry.
        let g = checked_grammar_from_schema_with(&demo_schema(), &IdGen::new(), &overrides)
            .expect("demo schema still generates");
        assert_eq!(g.parse("Lit(value: 7)").unwrap().variant, "Lit");
    }

    /// Layer 2: `Option<String>` payload maps to `none | %str` via the
    /// built-in mapping (no `SyntaxOverrides` entry needed). Canonical
    /// text parses both spellings and both build through the derive's
    /// `build_field_optional` route.
    #[test]
    fn builtin_option_string_mapping_accepts_none_and_str() {
        let schema = NodeSchema {
            name: "Meta".into(),
            variants: vec![VariantSchema {
                name: "Row".into(),
                fields: vec![FieldSchema {
                    name: "description".into(),
                    ty: "Option<String>".into(),
                    optional: true,
                }],
                children: vec![],
            }],
        };
        let g = checked_grammar_from_schema(&schema, &IdGen::new())
            .expect("Option<String> is a built-in mapping");
        // `none` spelling.
        let t1 = g.parse(r#"Row(description: none)"#).expect("none parses");
        assert_eq!(
            t1.field("description"),
            Some(&RawValue::Text("none".into())),
        );
        // `%str` spelling.
        let t2 = g.parse(r#"Row(description: "hi")"#).expect("string parses");
        assert_eq!(t2.field("description"), Some(&RawValue::Text("hi".into())));
        // Omitted entirely (Layer 1 optional PEG wrap).
        let t3 = g.parse(r#"Row()"#).expect("omitted parses");
        assert!(t3.field("description").is_none());
        // check_conformance is clean for all three.
        for t in [&t1, &t2, &t3] {
            assert!(
                check_conformance(t, &schema).is_empty(),
                "conformance clean for {t:?}",
            );
        }
    }

    /// Layer 2: `Vec<String>` payload maps to `[ %str ("," %str)* ]`
    /// via the built-in mapping. Empty list, single item, and multi
    /// item all parse; captured text is JSON-parsable so
    /// `build_field_vec` reads it directly.
    #[test]
    fn builtin_vec_string_mapping_accepts_bracketed_list() {
        let schema = NodeSchema {
            name: "Meta".into(),
            variants: vec![VariantSchema {
                name: "Row".into(),
                fields: vec![FieldSchema {
                    name: "tags".into(),
                    ty: "Vec<String>".into(),
                    optional: true,
                }],
                children: vec![],
            }],
        };
        let g = checked_grammar_from_schema(&schema, &IdGen::new())
            .expect("Vec<String> is a built-in mapping");
        // Empty list.
        let t1 = g.parse(r#"Row(tags: [])"#).expect("empty list parses");
        assert_eq!(t1.field("tags"), Some(&RawValue::Text("[]".into())));
        // Multi item. The PEG runtime's implicit whitespace-skip
        // strips inter-token spaces from the captured text, so the
        // field carries the compact form `["a","b"]` (still valid
        // JSON — `serde_json::from_str` accepts it).
        let t2 = g
            .parse(r#"Row(tags: ["a", "b"])"#)
            .expect("two items parses");
        assert_eq!(
            t2.field("tags"),
            Some(&RawValue::Text(r#"["a","b"]"#.into())),
        );
        // Omitted.
        let t3 = g.parse(r#"Row()"#).expect("omitted parses");
        assert!(t3.field("tags").is_none());
        for t in [&t1, &t2, &t3] {
            assert!(
                check_conformance(t, &schema).is_empty(),
                "conformance clean for {t:?}",
            );
        }
    }

    #[test]
    fn empty_schema_fails_generation() {
        let schema = NodeSchema {
            name: "Void".into(),
            variants: vec![],
        };
        let err = grammar_from_schema(&schema, &IdGen::new()).unwrap_err();
        assert_eq!(err.diagnostics[0].code, codes::EMPTY_SCHEMA);
    }

    #[test]
    fn parse_error_reports_expected_set() {
        let g = demo_grammar();
        let err = g.parse("Lit(value: banana)").unwrap_err();
        assert!(!err.diagnostics.is_empty());
    }
}
