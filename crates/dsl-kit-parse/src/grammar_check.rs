//! Static grammar checks (G-3) for a [`crate::peg::Grammar`] value.
//!
//! `GrammarCheck` is the *static* side of parser-core quality.
//! The runtime interpreter in [`crate::peg`] carries backstops for left
//! recursion and nullable-body [`crate::peg::Peg::Repeat`], but the
//! grammar author wants to be told before parse time — every ill-formed
//! grammar caught here is a whole class of runtime failure the consumer
//! AI never has to reason about.
//!
//! # Rule set
//!
//! - **[`check_left_recursion`]** — flags every rule reachable from
//!   itself via first-position rule references without consuming input.
//!   Direct (`r <- r "a"`) and indirect (`a <- b; b <- a "x"`) forms
//!   are both detected. Nullable-prefix chains that expose deeper rules
//!   to first position (`a <- b a; b <- ε`) are detected too. Emits
//!   under the same [`crate::peg::codes::LEFT_RECURSION`] slug the
//!   runtime backstop uses, so downstream tools see one dialect.
//! - **[`check_nullable_repeat`]** — flags every
//!   [`crate::peg::Peg::Repeat`] whose body can match empty. At runtime
//!   the interpreter already breaks the loop (no hang) and, when the
//!   `min` bound was not yet met, emits
//!   [`crate::peg::codes::NULLABLE_REPEAT`]. The static check catches it
//!   upfront regardless of `min`, so authors see the shape mistake once
//!   during grammar review rather than the day a pathological input
//!   triggers it.
//! - **[`check_schema_consistency`]** — cross-checks every
//!   [`crate::peg::Peg::Node`] variant against the consumer's
//!   [`NodeSchema`]. Grammar variants not declared in the schema are
//!   errors (the parser would emit a
//!   [`crate::codes::UNKNOWN_VARIANT`] diagnostic at conformance time,
//!   so surface it before the first parse attempt). Schema variants
//!   never produced by the grammar are `Warning`-severity reachability
//!   hints — perfectly fine for hand-built values, worth reviewing when
//!   the grammar is the sole producer.
//!
//! # Diagnostic dialect
//!
//! Every diagnostic uses the shared [`Diagnostic`] envelope (per
//! `parser-design.md §3.2`). Grammar-node diagnostics carry
//! [`Location::Node`] pointing at the offending [`crate::peg::Peg`]
//! node id; the unreachable-variant warning has no natural anchor and
//! uses [`Location::None`].

use crate::peg::{Grammar, Peg, codes as peg_codes};
use crate::{Diagnostic, Location, Severity};
use dsl_kit_core::{NodeId, Suggester};
use dsl_kit_schema::NodeSchema;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Diagnostic codes
// ---------------------------------------------------------------------------

/// Diagnostic codes emitted by [`check_schema_consistency`].
///
/// [`check_left_recursion`] and [`check_nullable_repeat`] reuse the
/// runtime interpreter's slugs ([`crate::peg::codes::LEFT_RECURSION`],
/// [`crate::peg::codes::NULLABLE_REPEAT`]) so static + runtime speak one
/// dialect.
pub mod codes {
    /// A [`crate::peg::Peg::Node`] in the grammar references a variant
    /// name not declared in the consumer's [`dsl_kit_schema::NodeSchema`].
    pub const UNKNOWN_VARIANT: &str = "dsl_kit::parse::grammar_check::unknown_variant";
    /// A [`dsl_kit_schema::VariantSchema`] name is declared in the
    /// consumer's schema but no [`crate::peg::Peg::Node`] in the grammar
    /// ever produces it. `Warning`-severity — hand-built values may fill
    /// the gap legitimately.
    pub const UNREACHABLE_VARIANT: &str = "dsl_kit::parse::grammar_check::unreachable_variant";
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Runs every grammar-only check ([`check_left_recursion`] +
/// [`check_nullable_repeat`]) and returns the accumulated diagnostics.
pub fn check(g: &Grammar) -> Vec<Diagnostic> {
    let mut out = check_left_recursion(g);
    out.extend(check_nullable_repeat(g));
    out
}

/// Runs [`check`] and additionally [`check_schema_consistency`] against
/// `schema`.
pub fn check_against(g: &Grammar, schema: &NodeSchema) -> Vec<Diagnostic> {
    let mut out = check(g);
    out.extend(check_schema_consistency(g, schema));
    out
}

// ---------------------------------------------------------------------------
// Left-recursion detection
// ---------------------------------------------------------------------------

/// Flags every rule reachable from itself via first-position rule
/// references without consuming input.
///
/// The check runs a fixed-point analysis of two derived properties per
/// rule:
///
/// 1. `nullable(R)` — can `R`'s body succeed on zero input?
/// 2. `first_rules(R)` — the transitive set of rule names reachable at
///    first position from `R`'s body, honouring nullability across
///    [`crate::peg::Peg::Seq`] items.
///
/// `R` is left-recursive iff `R ∈ first_rules(R)`. The rules-by-name
/// map is snapshotted once per invocation; the analysis converges in a
/// small number of passes for any finite grammar.
pub fn check_left_recursion(g: &Grammar) -> Vec<Diagnostic> {
    let rules = rules_by_name(g);
    let nullable = compute_nullable(&rules);
    let first = compute_left_first(&rules, &nullable);
    let mut out = Vec::new();
    for r in &g.rules {
        if let Peg::Rule { id, name, .. } = r
            && let Some(f) = first.get(name.as_str())
            && f.contains(name.as_str())
        {
            out.push(
                Diagnostic::error(
                    peg_codes::LEFT_RECURSION,
                    format!(
                        "rule `{name}` is left-recursive (reaches itself at first \
                         position without consuming input); PEG does not support \
                         left recursion — refactor into layered rules or a Repeat"
                    ),
                )
                .with_node(*id),
            );
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Nullable-repeat detection
// ---------------------------------------------------------------------------

/// Flags every [`crate::peg::Peg::Repeat`] whose body can match empty.
///
/// A nullable body drives the interpreter to break the loop rather than
/// hang (`peg.rs` `run_repeat`), but the loop then either fails the
/// `min` bound (runtime [`crate::peg::codes::NULLABLE_REPEAT`]) or
/// silently succeeds with zero productions. Neither branch is what the
/// grammar author wanted — surface the shape mistake statically.
pub fn check_nullable_repeat(g: &Grammar) -> Vec<Diagnostic> {
    let rules = rules_by_name(g);
    let nullable = compute_nullable(&rules);
    let mut out = Vec::new();
    for r in &g.rules {
        walk_peg(r, &mut |p| {
            if let Peg::Repeat { id, body, .. } = p
                && is_nullable(body, &nullable)
            {
                out.push(
                    Diagnostic::error(
                        peg_codes::NULLABLE_REPEAT,
                        "Repeat body can match empty input; the loop would \
                         either fail its `min` bound at runtime or succeed \
                         with zero productions — tighten the body so each \
                         iteration must consume at least one byte"
                            .to_string(),
                    )
                    .with_node(*id),
                );
            }
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Grammar-schema consistency
// ---------------------------------------------------------------------------

/// Cross-checks every [`crate::peg::Peg::Node`] variant against `schema`.
///
/// Grammar variants missing from the schema are `Error`-severity —
/// building a typed AST from such a tree fails at
/// [`crate::check_conformance`] anyway, so surface it upfront.
/// Schema variants that no grammar node produces are `Warning`-severity
/// reachability hints; hand-built values may legitimately fill the gap.
pub fn check_schema_consistency(g: &Grammar, schema: &NodeSchema) -> Vec<Diagnostic> {
    check_schema_consistency_with(g, schema, &crate::BuiltinLevenshteinSuggester)
}

/// Variant of [`check_schema_consistency`] that routes
/// `did you mean X?` hints through a caller-supplied [`Suggester`].
///
/// The free function [`check_schema_consistency`] delegates here with
/// the crate's built-in Levenshtein backend, so existing callers see
/// no behavioural change. Reach for this variant to plug in a
/// different similarity algorithm (e.g. `dsl-kit-fuzzy`'s
/// `FuzzySuggester`) at a specific call site.
pub fn check_schema_consistency_with(
    g: &Grammar,
    schema: &NodeSchema,
    suggester: &dyn Suggester,
) -> Vec<Diagnostic> {
    let declared: HashSet<&str> = schema.variants.iter().map(|v| v.name.as_str()).collect();
    let declared_names: Vec<&str> = schema.variants.iter().map(|v| v.name.as_str()).collect();
    let mut referenced: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for r in &g.rules {
        walk_peg(r, &mut |p| {
            if let Peg::Node { id, variant, .. } = p {
                referenced.insert(variant.clone());
                if !declared.contains(variant.as_str()) {
                    let base = format!(
                        "grammar produces variant `{variant}` which is \
                         not declared in schema `{}`",
                        schema.name
                    );
                    let msg = match suggester.enrich_unknown(variant, &declared_names) {
                        Some(hint) => format!("{base} ({hint})"),
                        None => base,
                    };
                    out.push(Diagnostic::error(codes::UNKNOWN_VARIANT, msg).with_node(*id));
                }
            }
        });
    }
    let mut unreachable: Vec<&str> = declared
        .into_iter()
        .filter(|name| !referenced.contains(*name))
        .collect();
    unreachable.sort_unstable();
    for name in unreachable {
        out.push(Diagnostic {
            severity: Severity::Warning,
            code: codes::UNREACHABLE_VARIANT.to_string(),
            message: format!(
                "schema variant `{name}` is not produced by any grammar rule \
                 (hand-built values may still construct it)"
            ),
            location: Location::None,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Shared analysis helpers
// ---------------------------------------------------------------------------

fn rules_by_name(g: &Grammar) -> HashMap<&str, &Peg> {
    let mut m = HashMap::new();
    for r in &g.rules {
        if let Peg::Rule { name, .. } = r {
            m.insert(name.as_str(), r);
        }
    }
    m
}

/// Fixed-point nullability: a rule is nullable iff its body can succeed
/// on zero input, taking mutually-recursive rules into account.
fn compute_nullable(rules: &HashMap<&str, &Peg>) -> HashMap<String, bool> {
    let mut nullable: HashMap<String, bool> =
        rules.keys().map(|k| ((*k).to_string(), false)).collect();
    loop {
        let mut changed = false;
        for (name, rule) in rules {
            if let Peg::Rule { body, .. } = rule {
                let n = is_nullable(body, &nullable);
                let slot = nullable.entry((*name).to_string()).or_insert(false);
                if n && !*slot {
                    *slot = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    nullable
}

fn is_nullable(peg: &Peg, nullable_rules: &HashMap<String, bool>) -> bool {
    match peg {
        Peg::Rule { body, .. } | Peg::Node { body, .. } | Peg::Field { body, .. } => {
            is_nullable(body, nullable_rules)
        }
        Peg::Seq { items, .. } => items.iter().all(|i| is_nullable(i, nullable_rules)),
        Peg::Choice { alts, .. } => {
            alts.is_empty() || alts.iter().any(|a| is_nullable(a, nullable_rules))
        }
        Peg::Repeat { body, min, .. } => *min == 0 || is_nullable(body, nullable_rules),
        Peg::RuleRef { name, .. } => nullable_rules.get(name).copied().unwrap_or(false),
        // Token classes (%ident / %int / %ws / %kw:<w>) all require ≥1
        // byte; a literal is nullable iff empty. Empty literals are a
        // grammar bug in their own right but this analysis stays
        // permissive — no `Token` currently produced by the convenience
        // constructors is empty in practice.
        Peg::Token { pat, .. } => pat.is_empty(),
    }
}

/// Fixed-point first-rule set: for each rule, the transitive closure of
/// rule names reachable at first position from the rule's body without
/// consuming input.
fn compute_left_first(
    rules: &HashMap<&str, &Peg>,
    nullable: &HashMap<String, bool>,
) -> HashMap<String, HashSet<String>> {
    let mut first: HashMap<String, HashSet<String>> = rules
        .keys()
        .map(|k| ((*k).to_string(), HashSet::new()))
        .collect();
    loop {
        let mut changed = false;
        for (name, rule) in rules {
            if let Peg::Rule { body, .. } = rule {
                let mut f = HashSet::new();
                collect_first(body, nullable, &first, &mut f);
                let slot = first.entry((*name).to_string()).or_default();
                if &f != slot {
                    *slot = f;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    first
}

fn collect_first(
    peg: &Peg,
    nullable: &HashMap<String, bool>,
    first_rules: &HashMap<String, HashSet<String>>,
    out: &mut HashSet<String>,
) {
    match peg {
        Peg::Rule { body, .. }
        | Peg::Node { body, .. }
        | Peg::Field { body, .. }
        | Peg::Repeat { body, .. } => collect_first(body, nullable, first_rules, out),
        Peg::Seq { items, .. } => {
            for item in items {
                collect_first(item, nullable, first_rules, out);
                if !is_nullable(item, nullable) {
                    break;
                }
            }
        }
        Peg::Choice { alts, .. } => {
            for alt in alts {
                collect_first(alt, nullable, first_rules, out);
            }
        }
        Peg::RuleRef { name, .. } => {
            out.insert(name.clone());
            if let Some(f) = first_rules.get(name) {
                for n in f {
                    out.insert(n.clone());
                }
            }
        }
        Peg::Token { .. } => {}
    }
}

/// Depth-first pre-order walk over every [`Peg`] node in the tree
/// rooted at `peg`, invoking `f` on each. Kept module-private —
/// external callers should reach for
/// [`dsl_kit_core::Walk`]-derived traversal on the grammar value
/// itself when they need programmatic access.
fn walk_peg(peg: &Peg, f: &mut dyn FnMut(&Peg)) {
    f(peg);
    match peg {
        Peg::Rule { body, .. }
        | Peg::Repeat { body, .. }
        | Peg::Node { body, .. }
        | Peg::Field { body, .. } => walk_peg(body, f),
        Peg::Seq { items, .. } => {
            for i in items {
                walk_peg(i, f);
            }
        }
        Peg::Choice { alts, .. } => {
            for a in alts {
                walk_peg(a, f);
            }
        }
        Peg::RuleRef { .. } | Peg::Token { .. } => {}
    }
}

/// Convenience: extract every Peg node in `g` whose id equals `id`.
/// Handy in tests that want to assert the diagnostic anchor matches a
/// specific known-id in the grammar being checked.
#[cfg(test)]
fn find_by_id(g: &Grammar, id: NodeId) -> Option<&Peg> {
    fn recurse(p: &Peg, id: NodeId) -> Option<&Peg> {
        if peg_id(p) == id {
            return Some(p);
        }
        match p {
            Peg::Rule { body, .. }
            | Peg::Repeat { body, .. }
            | Peg::Node { body, .. }
            | Peg::Field { body, .. } => recurse(body, id),
            Peg::Seq { items, .. } => items.iter().find_map(|i| recurse(i, id)),
            Peg::Choice { alts, .. } => alts.iter().find_map(|a| recurse(a, id)),
            Peg::RuleRef { .. } | Peg::Token { .. } => None,
        }
    }
    g.rules.iter().find_map(|r| recurse(r, id))
}

#[cfg(test)]
fn peg_id(p: &Peg) -> NodeId {
    match p {
        Peg::Rule { id, .. }
        | Peg::Seq { id, .. }
        | Peg::Choice { id, .. }
        | Peg::Repeat { id, .. }
        | Peg::RuleRef { id, .. }
        | Peg::Token { id, .. }
        | Peg::Node { id, .. }
        | Peg::Field { id, .. } => *id,
    }
}

// Force NodeId into scope for the doc examples.
#[allow(dead_code)]
fn _touch_nodeid(_: NodeId) {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peg::{choice, field, node, repeat, rule, rule_ref, seq, token};
    use dsl_kit_core::IdGen;
    use dsl_kit_schema::{ChildSchema, FieldSchema, Multiplicity, NodeSchema, VariantSchema};

    // -- helpers --

    fn schema_expr() -> NodeSchema {
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
            ],
        }
    }

    // -- left recursion --

    #[test]
    fn direct_left_recursion_is_detected() {
        let ids = IdGen::new();
        // r <- r "a"
        let r = rule(
            &ids,
            "r",
            seq(&ids, vec![rule_ref(&ids, "r"), token(&ids, "a")]),
        );
        let g = Grammar::new(vec![r], "r");
        let diags = check_left_recursion(&g);
        assert_eq!(diags.len(), 1, "diags = {diags:?}");
        assert_eq!(diags[0].code, peg_codes::LEFT_RECURSION);
        assert!(diags[0].message.contains("`r`"), "{}", diags[0].message);
    }

    #[test]
    fn indirect_left_recursion_is_detected() {
        let ids = IdGen::new();
        // a <- b "x"; b <- a "y"
        let a = rule(
            &ids,
            "a",
            seq(&ids, vec![rule_ref(&ids, "b"), token(&ids, "x")]),
        );
        let b = rule(
            &ids,
            "b",
            seq(&ids, vec![rule_ref(&ids, "a"), token(&ids, "y")]),
        );
        let g = Grammar::new(vec![a, b], "a");
        let diags = check_left_recursion(&g);
        // Both rules should fire — each reaches itself via the other.
        assert_eq!(diags.len(), 2, "diags = {diags:?}");
        let messages: Vec<&str> = diags.iter().map(|d| d.message.as_str()).collect();
        assert!(messages.iter().any(|m| m.contains("`a`")));
        assert!(messages.iter().any(|m| m.contains("`b`")));
    }

    #[test]
    fn left_recursion_through_nullable_prefix_is_detected() {
        let ids = IdGen::new();
        // a <- b a; b <- ("y" / <empty>)
        // b is nullable — so `a`'s Seq falls through to the RuleRef(a) at
        // first position without consuming input.
        let b = rule(
            &ids,
            "b",
            choice(&ids, vec![token(&ids, "y"), seq(&ids, vec![])]),
        );
        let a = rule(
            &ids,
            "a",
            seq(&ids, vec![rule_ref(&ids, "b"), rule_ref(&ids, "a")]),
        );
        let g = Grammar::new(vec![a, b], "a");
        let diags = check_left_recursion(&g);
        assert!(
            diags
                .iter()
                .any(|d| d.code == peg_codes::LEFT_RECURSION && d.message.contains("`a`")),
            "expected `a` to be flagged, diags = {diags:?}"
        );
    }

    #[test]
    fn right_recursion_is_accepted() {
        let ids = IdGen::new();
        // a <- "x" a — legal PEG, consumes before recursing.
        let a = rule(
            &ids,
            "a",
            seq(&ids, vec![token(&ids, "x"), rule_ref(&ids, "a")]),
        );
        let g = Grammar::new(vec![a], "a");
        assert!(check_left_recursion(&g).is_empty());
    }

    #[test]
    fn layered_precedence_grammar_is_accepted() {
        // expr <- term ("+" term)*
        // term <- factor ("*" factor)*
        // factor <- "n"
        // Right-recursive-friendly layered form — no left recursion.
        let ids = IdGen::new();
        let factor = rule(&ids, "factor", node(&ids, "F", token(&ids, "n")));
        let term = rule(
            &ids,
            "term",
            seq(
                &ids,
                vec![
                    rule_ref(&ids, "factor"),
                    repeat(
                        &ids,
                        seq(&ids, vec![token(&ids, "*"), rule_ref(&ids, "factor")]),
                        0,
                        None,
                    ),
                ],
            ),
        );
        let expr = rule(
            &ids,
            "expr",
            seq(
                &ids,
                vec![
                    rule_ref(&ids, "term"),
                    repeat(
                        &ids,
                        seq(&ids, vec![token(&ids, "+"), rule_ref(&ids, "term")]),
                        0,
                        None,
                    ),
                ],
            ),
        );
        let g = Grammar::new(vec![expr, term, factor], "expr");
        assert!(check_left_recursion(&g).is_empty());
    }

    #[test]
    fn diagnostic_anchor_points_at_offending_rule() {
        let ids = IdGen::new();
        let r_peg = rule(
            &ids,
            "r",
            seq(&ids, vec![rule_ref(&ids, "r"), token(&ids, "a")]),
        );
        let r_id = peg_id(&r_peg);
        let g = Grammar::new(vec![r_peg], "r");
        let diags = check_left_recursion(&g);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].location, Location::Node(r_id));
        // Sanity-check we can round-trip: the Peg with that id is the
        // Rule node itself, not one of its descendants.
        assert!(matches!(find_by_id(&g, r_id).unwrap(), Peg::Rule { .. }));
    }

    // -- nullable repeat --

    #[test]
    fn nullable_repeat_empty_body_is_detected() {
        let ids = IdGen::new();
        // Repeat { body = Seq [] } — always matches empty.
        let rep = repeat(&ids, seq(&ids, vec![]), 0, None);
        let rep_id = peg_id(&rep);
        let r = rule(&ids, "s", node(&ids, "N", rep));
        let g = Grammar::new(vec![r], "s");
        let diags = check_nullable_repeat(&g);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, peg_codes::NULLABLE_REPEAT);
        assert_eq!(diags[0].location, Location::Node(rep_id));
    }

    #[test]
    fn nullable_repeat_choice_with_empty_alt_is_detected() {
        let ids = IdGen::new();
        // Repeat { body = Choice("a" | <empty>) } — Choice is nullable.
        let body = choice(&ids, vec![token(&ids, "a"), seq(&ids, vec![])]);
        let rep = repeat(&ids, body, 0, None);
        let r = rule(&ids, "s", node(&ids, "N", rep));
        let g = Grammar::new(vec![r], "s");
        let diags = check_nullable_repeat(&g);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, peg_codes::NULLABLE_REPEAT);
    }

    #[test]
    fn consuming_repeat_body_is_accepted() {
        let ids = IdGen::new();
        // Repeat { body = "a" } — Token is not nullable.
        let rep = repeat(&ids, token(&ids, "a"), 0, None);
        let r = rule(&ids, "s", node(&ids, "N", rep));
        let g = Grammar::new(vec![r], "s");
        assert!(check_nullable_repeat(&g).is_empty());
    }

    #[test]
    fn nullable_repeat_via_nullable_rule_is_detected() {
        let ids = IdGen::new();
        // Repeat { body = RuleRef(b) } where b is nullable.
        let b = rule(&ids, "b", seq(&ids, vec![]));
        let rep = repeat(&ids, rule_ref(&ids, "b"), 0, None);
        let s = rule(&ids, "s", node(&ids, "N", rep));
        let g = Grammar::new(vec![s, b], "s");
        let diags = check_nullable_repeat(&g);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, peg_codes::NULLABLE_REPEAT);
    }

    // -- schema consistency --

    #[test]
    fn unknown_grammar_variant_is_error() {
        let ids = IdGen::new();
        // Grammar produces "Xyz" — not in schema.
        let r = rule(&ids, "s", node(&ids, "Xyz", token(&ids, "a")));
        let g = Grammar::new(vec![r], "s");
        let diags = check_schema_consistency(&g, &schema_expr());
        let errs: Vec<&Diagnostic> = diags
            .iter()
            .filter(|d| d.code == codes::UNKNOWN_VARIANT)
            .collect();
        assert_eq!(errs.len(), 1, "diags = {diags:?}");
        assert!(errs[0].message.contains("`Xyz`"));
        assert!(errs[0].message.contains("`Expr`"));
    }

    #[test]
    fn unknown_grammar_variant_suggests_declared_variant() {
        // A typo of a declared schema variant should surface the
        // correct name via the crate's built-in Levenshtein
        // suggester.
        let ids = IdGen::new();
        let r = rule(&ids, "s", node(&ids, "Aad", token(&ids, "a")));
        let g = Grammar::new(vec![r], "s");
        let diags = check_schema_consistency(&g, &schema_expr());
        let err = diags
            .iter()
            .find(|d| d.code == codes::UNKNOWN_VARIANT)
            .expect("expected UNKNOWN_VARIANT");
        assert!(
            err.message.contains("did you mean") && err.message.contains("Add"),
            "expected `Add` in the hint, got: {}",
            err.message
        );
    }

    #[test]
    fn unreachable_schema_variant_is_warning() {
        let ids = IdGen::new();
        // Grammar produces only Lit — Add is unreached.
        let r = rule(
            &ids,
            "s",
            node(&ids, "Lit", field(&ids, "value", token(&ids, "%int"))),
        );
        let g = Grammar::new(vec![r], "s");
        let diags = check_schema_consistency(&g, &schema_expr());
        let warns: Vec<&Diagnostic> = diags
            .iter()
            .filter(|d| d.code == codes::UNREACHABLE_VARIANT)
            .collect();
        assert_eq!(warns.len(), 1, "diags = {diags:?}");
        assert_eq!(warns[0].severity, Severity::Warning);
        assert!(warns[0].message.contains("`Add`"));
    }

    #[test]
    fn schema_matched_grammar_has_no_diagnostics() {
        let ids = IdGen::new();
        // Grammar covers Lit and Add.
        let lit = rule(
            &ids,
            "lit",
            node(&ids, "Lit", field(&ids, "value", token(&ids, "%int"))),
        );
        let add = rule(
            &ids,
            "add",
            node(
                &ids,
                "Add",
                seq(
                    &ids,
                    vec![
                        field(&ids, "lhs", rule_ref(&ids, "lit")),
                        token(&ids, "+"),
                        field(&ids, "rhs", rule_ref(&ids, "lit")),
                    ],
                ),
            ),
        );
        let g = Grammar::new(vec![add, lit], "add");
        assert!(check_schema_consistency(&g, &schema_expr()).is_empty());
    }

    #[test]
    fn unknown_variant_anchors_at_the_node_peg() {
        let ids = IdGen::new();
        let n_peg = node(&ids, "Xyz", token(&ids, "a"));
        let n_id = peg_id(&n_peg);
        let r = rule(&ids, "s", n_peg);
        let g = Grammar::new(vec![r], "s");
        let diags = check_schema_consistency(&g, &schema_expr());
        let err = diags
            .iter()
            .find(|d| d.code == codes::UNKNOWN_VARIANT)
            .expect("expected unknown_variant");
        assert_eq!(err.location, Location::Node(n_id));
    }

    // -- combined --

    #[test]
    fn check_accumulates_across_rules_but_omits_schema() {
        let ids = IdGen::new();
        // Two independent issues: left recursion + nullable repeat.
        let rec = rule(&ids, "rec", rule_ref(&ids, "rec"));
        let nullrep = rule(
            &ids,
            "nullrep",
            node(&ids, "N", repeat(&ids, seq(&ids, vec![]), 0, None)),
        );
        let g = Grammar::new(vec![rec, nullrep], "rec");
        let diags = check(&g);
        assert!(diags.iter().any(|d| d.code == peg_codes::LEFT_RECURSION));
        assert!(diags.iter().any(|d| d.code == peg_codes::NULLABLE_REPEAT));
        // check(&g) never touches the schema — no UNKNOWN_VARIANT even
        // though "N" is not in any schema.
        assert!(!diags.iter().any(|d| d.code == codes::UNKNOWN_VARIANT));
        assert!(!diags.iter().any(|d| d.code == codes::UNREACHABLE_VARIANT));
    }

    #[test]
    fn check_against_composes_grammar_and_schema_diagnostics() {
        let ids = IdGen::new();
        // Grammar has a left-recursive rule AND produces an unknown
        // variant. `check_against` returns both classes at once.
        let rec = rule(&ids, "rec", node(&ids, "Xyz", rule_ref(&ids, "rec")));
        let g = Grammar::new(vec![rec], "rec");
        let diags = check_against(&g, &schema_expr());
        assert!(diags.iter().any(|d| d.code == peg_codes::LEFT_RECURSION));
        assert!(diags.iter().any(|d| d.code == codes::UNKNOWN_VARIANT));
        // Both schema variants (Lit, Add) are unreached.
        let unreached: Vec<&Diagnostic> = diags
            .iter()
            .filter(|d| d.code == codes::UNREACHABLE_VARIANT)
            .collect();
        assert_eq!(unreached.len(), 2);
    }

    #[test]
    fn clean_grammar_produces_no_diagnostics() {
        // The layered-precedence grammar wrapped in a schema-shaped
        // Node — no issues.
        let ids = IdGen::new();
        let lit = rule(
            &ids,
            "lit",
            node(&ids, "Lit", field(&ids, "value", token(&ids, "%int"))),
        );
        let g = Grammar::new(vec![lit], "lit");
        let diags = check_against(&g, &{
            NodeSchema {
                name: "Tiny".into(),
                variants: vec![VariantSchema {
                    name: "Lit".into(),
                    fields: vec![FieldSchema {
                        name: "value".into(),
                        ty: "i64".into(),
                    }],
                    children: vec![],
                }],
            }
        });
        assert!(diags.is_empty(), "diags = {diags:?}");
    }
}
