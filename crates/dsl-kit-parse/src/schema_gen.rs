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
//! - **Payload fields** — by Rust type source text:
//!   `String` → a `%str` literal; the integer types → `%int`;
//!   `bool` → `true` / `false`. Any other payload type fails
//!   generation with [`codes::UNSUPPORTED_FIELD`] — loudly, per
//!   variant, rather than silently dropping the variant from the
//!   grammar.
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

use crate::grammar_check;
use crate::peg::{Grammar, Peg, choice, field, node, repeat, rule, rule_ref, seq, token};
use crate::{BuildError, Diagnostic};
use dsl_kit_core::IdGen;
use dsl_kit_schema::{ChildSchema, FieldSchema, Multiplicity, NodeSchema, VariantSchema};

/// Diagnostic codes emitted by grammar generation.
pub mod codes {
    /// A payload field's Rust type has no canonical-syntax mapping.
    pub const UNSUPPORTED_FIELD: &str = "dsl_kit::schema_gen::unsupported_field";
    /// The schema declares no variants — there is nothing to parse.
    pub const EMPTY_SCHEMA: &str = "dsl_kit::schema_gen::empty_schema";
}

/// Name of the generated start rule.
pub const START_RULE: &str = "node";

impl Grammar {
    /// Generates the canonical-syntax grammar for `schema`. See
    /// [`grammar_from_schema`].
    pub fn from_schema(schema: &NodeSchema, ids: &IdGen) -> Result<Grammar, BuildError> {
        grammar_from_schema(schema, ids)
    }
}

/// Generates a [`Grammar`] for the canonical named-argument syntax of
/// `schema` (see the module docs for the syntax).
///
/// Fails with one [`codes::UNSUPPORTED_FIELD`] diagnostic per
/// unmappable payload field (collected across all variants, so the
/// author sees the full list at once).
pub fn grammar_from_schema(schema: &NodeSchema, ids: &IdGen) -> Result<Grammar, BuildError> {
    if schema.variants.is_empty() {
        return Err(BuildError::single(Diagnostic::error(
            codes::EMPTY_SCHEMA,
            format!("schema `{}` declares no variants", schema.name),
        )));
    }

    let mut bad_fields = Vec::new();
    for v in &schema.variants {
        for f in &v.fields {
            if field_value_peg(f, ids).is_none() {
                bad_fields.push(Diagnostic::error(
                    codes::UNSUPPORTED_FIELD,
                    format!(
                        "variant `{}` field `{}`: type `{}` has no canonical-syntax \
                         mapping (supported: String, bool, the integer types)",
                        v.name, f.name, f.ty
                    ),
                ));
            }
        }
    }
    if !bad_fields.is_empty() {
        return Err(BuildError::new(bad_fields));
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
        rules.push(variant_rule(v, ids));
    }
    Ok(Grammar::new(rules, START_RULE))
}

/// Builds the rule for one variant:
/// `Node { variant } [ %kw:V "(" arg ("," arg)* ")" ]` with arguments
/// in schema order (fields, then children).
fn variant_rule(v: &VariantSchema, ids: &IdGen) -> Peg {
    let mut args: Vec<Peg> = Vec::new();
    for f in &v.fields {
        let value = field_value_peg(f, ids)
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
    for (i, arg) in args.into_iter().enumerate() {
        if i > 0 {
            items.push(token(ids, ","));
        }
        items.push(arg);
    }
    items.push(token(ids, ")"));

    rule(ids, v.name.clone(), node(ids, v.name.clone(), seq(ids, items)))
}

/// Value production for a payload field, by Rust type source text.
/// `None` when the type has no canonical-syntax mapping.
fn field_value_peg(f: &FieldSchema, ids: &IdGen) -> Option<Peg> {
    const INT_TYPES: &[&str] = &[
        "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "usize", "isize",
    ];
    if f.ty == "String" {
        Some(token(ids, "%str"))
    } else if INT_TYPES.contains(&f.ty.as_str()) {
        Some(token(ids, "%int"))
    } else if f.ty == "bool" {
        Some(choice(
            ids,
            vec![token(ids, "%kw:true"), token(ids, "%kw:false")],
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
    let grammar = grammar_from_schema(schema, ids)?;
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

    /// Expr-flavoured schema exercising every supported shape: int /
    /// string / bool fields, One / Optional / Many children, and a
    /// zero-argument variant.
    fn demo_schema() -> NodeSchema {
        NodeSchema {
            name: "Expr".into(),
            variants: vec![
                VariantSchema {
                    name: "Lit".into(),
                    fields: vec![FieldSchema { name: "value".into(), ty: "i64".into() }],
                    children: vec![],
                },
                VariantSchema {
                    name: "Name".into(),
                    fields: vec![
                        FieldSchema { name: "text".into(), ty: "String".into() },
                        FieldSchema { name: "quoted".into(), ty: "bool".into() },
                    ],
                    children: vec![],
                },
                VariantSchema {
                    name: "Add".into(),
                    fields: vec![],
                    children: vec![
                        ChildSchema { name: "lhs".into(), multiplicity: Multiplicity::One },
                        ChildSchema { name: "rhs".into(), multiplicity: Multiplicity::One },
                    ],
                },
                VariantSchema {
                    name: "Neg".into(),
                    fields: vec![],
                    children: vec![ChildSchema {
                        name: "body".into(),
                        multiplicity: Multiplicity::Optional,
                    }],
                },
                VariantSchema {
                    name: "List".into(),
                    fields: vec![],
                    children: vec![ChildSchema {
                        name: "items".into(),
                        multiplicity: Multiplicity::Many,
                    }],
                },
                VariantSchema { name: "Unit".into(), fields: vec![], children: vec![] },
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
        let tree = g.parse(input).unwrap_or_else(|e| {
            panic!("parse failed for {input:?}: {:?}", e.diagnostics)
        });
        let diags = check_conformance(&tree, &demo_schema());
        assert!(diags.is_empty(), "conformance clean for {input:?}: {diags:?}");
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
        assert!(tree.child_slot("body").is_none(), "bare `none` = absent slot");
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
            tree.child_slot("items").map_or(true, <[_]>::is_empty),
            "empty list = zero children"
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
                    FieldSchema { name: "policy".into(), ty: "Option<JoinPolicy>".into() },
                    FieldSchema { name: "reducer".into(), ty: "ReducerId".into() },
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

    #[test]
    fn empty_schema_fails_generation() {
        let schema = NodeSchema { name: "Void".into(), variants: vec![] };
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
