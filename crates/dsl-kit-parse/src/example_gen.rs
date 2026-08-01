//! Example synthesis from a grammar (item Q-1).
//!
//! Walks a [`Grammar`]'s [`Peg`] — not the schema — and emits input
//! strings that the grammar accepts **by construction**: every choice
//! picks a concrete arm, every token contributes matching input text
//! (`%int` → `1`, `%str` → `"example"`, keywords verbatim). Because
//! synthesis reads the grammar itself, spellings supplied through
//! `schema_gen::SyntaxOverrides` fall out automatically — no separate
//! example registration.
//!
//! Two synthesis modes drive the output of [`examples_from_grammar`]:
//!
//! - **Minimal** (per-rule examples) — at every choice take the arm
//!   with the cheapest finite derivation, at every repeat emit the
//!   minimum count. Terminates by the shortest-derivation argument:
//!   each step either emits a token or strictly reduces the remaining
//!   derivation cost.
//! - **Rich** (the composite program) — while a depth budget lasts,
//!   choices take the *most expensive* finite arm and repeats emit up
//!   to two iterations, showing nesting and list syntax; the budget
//!   decrements per rule expansion and exhaustion falls back to
//!   Minimal.
//!
//! The intended consumer is an AI writing the canonical text syntax:
//! few-shot examples beat a formal grammar table for one-shot
//! accuracy, and machine-derived examples cannot drift from the
//! grammar the way hand-written documentation does.

use std::cell::Cell;
use std::collections::HashMap;

use crate::peg::{Grammar, Peg};
use crate::{BuildError, Diagnostic};

/// Diagnostic codes emitted by example synthesis.
pub mod codes {
    /// A rule has no finite derivation (every path recurses forever),
    /// so no example input exists.
    pub const NO_FINITE_DERIVATION: &str = "dsl_kit::example_gen::no_finite_derivation";
    /// A `RuleRef` names a rule the grammar does not define.
    pub const UNKNOWN_RULE: &str = "dsl_kit::example_gen::unknown_rule";
}

/// Rich-mode depth budget used by [`examples_from_grammar`] for the
/// composite program: rule expansions beyond this depth continue
/// minimally.
pub const COMPOSITE_DEPTH: u32 = 2;

/// One rule's minimal example.
#[derive(Debug, Clone)]
pub struct RuleExample {
    /// Rule name (for `schema_gen` grammars: the variant name).
    pub rule: String,
    /// Input text accepted by the grammar, entering at this rule.
    pub text: String,
}

/// Everything [`examples_from_grammar`] synthesizes for one grammar.
#[derive(Debug, Clone)]
pub struct GrammarExamples {
    /// Minimal example per non-start rule, in rule declaration order.
    /// For `schema_gen` grammars this is exactly one example per
    /// variant.
    pub per_rule: Vec<RuleExample>,
    /// One depth-capped rich expansion of the start rule, showing
    /// nesting and repetition.
    pub composite: String,
}

/// Synthesizes [`GrammarExamples`] for `grammar`.
///
/// Fails with one [`codes::NO_FINITE_DERIVATION`] diagnostic per
/// underivable rule (collected, so the author sees the full list), or
/// [`codes::UNKNOWN_RULE`] on a dangling reference.
pub fn examples_from_grammar(grammar: &Grammar) -> Result<GrammarExamples, BuildError> {
    let synth = Synth::prepare(grammar)?;

    let mut underivable = Vec::new();
    for (name, _) in &synth.rules_in_order {
        if synth.rule_cost(name) == INFINITE {
            underivable.push(Diagnostic::error(
                codes::NO_FINITE_DERIVATION,
                format!("rule `{name}` has no finite derivation — no example input exists"),
            ));
        }
    }
    if !underivable.is_empty() {
        return Err(BuildError::new(underivable));
    }

    let mut per_rule = Vec::new();
    for (name, body) in &synth.rules_in_order {
        if *name == grammar.start {
            continue;
        }
        // The reserved rules — `$import` (load-phase plumbing) and
        // `$allow` (lint-suppression annotation) — are not the DSL the
        // examples teach, so `@import "…"` / `@allow("…")` never
        // appear in them.
        if is_reserved_rule(name) {
            continue;
        }
        let mut tokens = Vec::new();
        synth.emit(body, 0, &mut tokens)?;
        per_rule.push(RuleExample {
            rule: name.clone(),
            text: render(&tokens),
        });
    }

    let start_body = synth.rule_body(&grammar.start)?;
    let mut tokens = Vec::new();
    synth.emit(start_body, COMPOSITE_DEPTH, &mut tokens)?;
    let composite = render(&tokens);

    Ok(GrammarExamples {
        per_rule,
        composite,
    })
}

/// Sentinel cost for "no finite derivation".
const INFINITE: u64 = u64::MAX;

struct Synth<'g> {
    /// Rule name → body, for `RuleRef` expansion.
    by_name: HashMap<&'g str, &'g Peg>,
    /// `(name, body)` in grammar declaration order.
    rules_in_order: Vec<(String, &'g Peg)>,
    /// Shortest-derivation token count per rule (fixpoint result).
    costs: HashMap<String, u64>,
    /// Running count of keyed entries emitted, so each one can be
    /// given a distinct key. Interior mutability because emission
    /// takes `&self` — the alternative, threading a counter through
    /// every `emit` signature, would put keyed-slot bookkeeping in the
    /// path of shapes that have nothing to do with it.
    key_seq: Cell<u32>,
}

impl<'g> Synth<'g> {
    fn prepare(grammar: &'g Grammar) -> Result<Self, BuildError> {
        let mut by_name = HashMap::new();
        let mut rules_in_order = Vec::new();
        for r in &grammar.rules {
            if let Peg::Rule { name, body, .. } = r {
                by_name.insert(name.as_str(), body.as_ref());
                rules_in_order.push((name.clone(), body.as_ref()));
            }
        }

        // Shortest-derivation cost fixpoint: start every rule at
        // INFINITE and relax until stable. Monotone non-increasing and
        // bounded, so it terminates.
        let mut costs: HashMap<String, u64> = rules_in_order
            .iter()
            .map(|(n, _)| (n.clone(), INFINITE))
            .collect();
        loop {
            let mut changed = false;
            for (name, body) in &rules_in_order {
                let c = peg_cost(body, &costs);
                if c < costs[name] {
                    costs.insert(name.clone(), c);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        Ok(Self {
            by_name,
            rules_in_order,
            costs,
            key_seq: Cell::new(0),
        })
    }

    fn rule_cost(&self, name: &str) -> u64 {
        self.costs.get(name).copied().unwrap_or(INFINITE)
    }

    fn rule_body(&self, name: &str) -> Result<&'g Peg, BuildError> {
        self.by_name.get(name).copied().ok_or_else(|| {
            BuildError::single(Diagnostic::error(
                codes::UNKNOWN_RULE,
                format!("reference to undefined rule `{name}`"),
            ))
        })
    }

    /// Appends the input tokens of one derivation of `peg`.
    /// `rich_depth > 0` selects the expansive arm at choices and up to
    /// two iterations at repeats; `0` is the minimal mode.
    fn emit(&self, peg: &Peg, rich_depth: u32, out: &mut Vec<String>) -> Result<(), BuildError> {
        match peg {
            Peg::Token { pat, .. } => {
                out.push(token_input(pat));
                Ok(())
            }
            Peg::Seq { items, .. } => {
                for item in items {
                    self.emit(item, rich_depth, out)?;
                }
                Ok(())
            }
            Peg::Choice { alts, .. } => {
                let arm = self.pick_arm(alts, rich_depth)?;
                self.emit(arm, rich_depth, out)
            }
            Peg::Repeat { body, min, max, .. } => {
                let count = if rich_depth > 0 {
                    (*min).max(2.min(max.unwrap_or(2)))
                } else {
                    *min
                };
                for _ in 0..count {
                    self.emit(body, rich_depth, out)?;
                }
                Ok(())
            }
            Peg::RuleRef { name, .. } => {
                let body = self.rule_body(name)?;
                self.emit(body, rich_depth.saturating_sub(1), out)
            }
            Peg::Rule { body, .. } | Peg::Node { body, .. } | Peg::Field { body, .. } => {
                self.emit(body, rich_depth, out)
            }
            // A keyed entry emits its key then its value, in that
            // order — the same shape a two-item `Seq` would, since the
            // primitive adds no syntax of its own (the separator and
            // brackets around it live in the enclosing grammar).
            //
            // The key goes through `emit_key` so that repeated entries
            // get *distinct* keys. Emitting the plain token text twice
            // would synthesize `{ "example": …, "example": … }` — text
            // the grammar accepts but the schema rejects as a
            // duplicate key, which is precisely the drift this module
            // exists to prevent.
            Peg::KeyedEntry { key, value, .. } => {
                let n = self.key_seq.get() + 1;
                self.key_seq.set(n);
                self.emit_key(key, n, rich_depth, out)?;
                self.emit(value, rich_depth, out)
            }
        }
    }

    /// Emits a keyed entry's key with occurrence number `n` woven in,
    /// so sibling entries in one map get distinct keys.
    ///
    /// Handles the shapes a key production is made of (tokens, `Seq`,
    /// `Choice`). Anything richer falls through to the generic
    /// emitter: keys may then collide, but a collision surfaces as a
    /// `DUPLICATE_KEY` conformance error on the synthesized example
    /// rather than being papered over here.
    fn emit_key(
        &self,
        peg: &Peg,
        n: u32,
        rich_depth: u32,
        out: &mut Vec<String>,
    ) -> Result<(), BuildError> {
        match peg {
            Peg::Token { pat, .. } => {
                out.push(keyed_token_input(pat, n));
                Ok(())
            }
            Peg::Seq { items, .. } => {
                for item in items {
                    self.emit_key(item, n, rich_depth, out)?;
                }
                Ok(())
            }
            Peg::Choice { alts, .. } => {
                let arm = self.pick_arm(alts, rich_depth)?;
                self.emit_key(arm, n, rich_depth, out)
            }
            other => self.emit(other, rich_depth, out),
        }
    }

    /// Chooses a choice arm: cheapest finite derivation in minimal
    /// mode, most expensive finite one in rich mode (ties: first).
    ///
    /// Arms reaching a reserved rule are never picked — `@import "…"`
    /// is load-phase plumbing and `@allow("…")` is a lint annotation,
    /// neither of them the DSL the examples teach (see
    /// `crate::import::add_import_syntax` /
    /// `crate::allow::add_allow_syntax`).
    fn pick_arm<'p>(&self, alts: &'p [Peg], rich_depth: u32) -> Result<&'p Peg, BuildError> {
        let finite = alts
            .iter()
            .filter(|a| !is_reserved_arm(a))
            .map(|a| (a, peg_cost(a, &self.costs)))
            .filter(|(_, c)| *c != INFINITE);
        let picked = if rich_depth > 0 {
            finite.max_by_key(|(_, c)| *c)
        } else {
            finite.min_by_key(|(_, c)| *c)
        };
        picked.map(|(a, _)| a).ok_or_else(|| {
            BuildError::single(Diagnostic::error(
                codes::NO_FINITE_DERIVATION,
                "choice has no finite alternative".to_string(),
            ))
        })
    }
}

/// Whether a rule name is one of the reserved spellings the examples
/// never teach.
fn is_reserved_rule(name: &str) -> bool {
    name == crate::import::IMPORT_VARIANT || name == crate::allow::ALLOW_VARIANT
}

/// Whether a choice arm reaches a reserved rule / node (see
/// `pick_arm`).
fn is_reserved_arm(peg: &Peg) -> bool {
    match peg {
        Peg::RuleRef { name, .. } => is_reserved_rule(name),
        Peg::Node { variant, .. } => is_reserved_rule(variant),
        _ => false,
    }
}

/// Shortest-derivation token count of `peg` under the current rule
/// cost estimates ([`INFINITE`]-safe saturating arithmetic).
fn peg_cost(peg: &Peg, costs: &HashMap<String, u64>) -> u64 {
    match peg {
        Peg::Token { .. } => 1,
        Peg::Seq { items, .. } => items
            .iter()
            .try_fold(0u64, |acc, i| {
                let c = peg_cost(i, costs);
                if c == INFINITE {
                    None
                } else {
                    Some(acc.saturating_add(c))
                }
            })
            .unwrap_or(INFINITE),
        Peg::Choice { alts, .. } => alts
            .iter()
            .map(|a| peg_cost(a, costs))
            .min()
            .unwrap_or(INFINITE),
        Peg::Repeat { body, min, .. } => {
            if *min == 0 {
                0
            } else {
                let c = peg_cost(body, costs);
                if c == INFINITE {
                    INFINITE
                } else {
                    c.saturating_mul(u64::from(*min))
                }
            }
        }
        Peg::RuleRef { name, .. } => costs.get(name).copied().unwrap_or(INFINITE),
        Peg::Rule { body, .. } | Peg::Node { body, .. } | Peg::Field { body, .. } => {
            peg_cost(body, costs)
        }
        // Sum of both halves, `INFINITE`-poisoned like `Seq`: an entry
        // whose value has no finite derivation has none either.
        Peg::KeyedEntry { key, value, .. } => {
            let (k, v) = (peg_cost(key, costs), peg_cost(value, costs));
            if k == INFINITE || v == INFINITE {
                INFINITE
            } else {
                k.saturating_add(v)
            }
        }
    }
}

/// Input text for a keyed entry's key token, occurrence `n`.
///
/// Only the two key spellings `schema_gen` emits are disambiguated;
/// every other pattern (a literal separator like `:`) renders as
/// usual.
fn keyed_token_input(pat: &str, n: u32) -> String {
    match pat {
        "%str" => format!("\"key{n}\""),
        "%ident" => format!("key{n}"),
        other => token_input(other),
    }
}

/// Input text matching one token pattern.
fn token_input(pat: &str) -> String {
    if let Some(kw) = pat.strip_prefix("%kw:") {
        return kw.to_string();
    }
    match pat {
        "%int" => "1".to_string(),
        "%str" => "\"example\"".to_string(),
        "%ident" => "x".to_string(),
        "%ws" => " ".to_string(),
        literal => literal.to_string(),
    }
}

/// Joins tokens with canonical-syntax spacing: no space before
/// punctuation that closes or separates (`,` `)` `]` `}` `:` `(`), no
/// space after an opener (`(` `[` `{`).
fn render(tokens: &[String]) -> String {
    let mut out = String::new();
    for t in tokens {
        let tight_before = matches!(t.as_str(), "," | ")" | "]" | "}" | ":" | "(");
        let tight_after_prev = out.ends_with('(') || out.ends_with('[') || out.ends_with('{');
        if !out.is_empty() && !tight_before && !tight_after_prev {
            out.push(' ');
        }
        out.push_str(t);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check_conformance;
    use crate::schema_gen::checked_grammar_from_schema;
    use dsl_kit_core::IdGen;
    use dsl_kit_schema::{
        ChildSchema, ChildValueShape, FieldSchema, Multiplicity, NodeSchema, VariantSchema,
    };

    /// Same shape matrix as the schema_gen demo: int / string / bool
    /// fields, One / Optional / Many children, zero-argument variant.
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
                            scalar_shorthands: vec![],
                            non_empty: false,
                        },
                        ChildSchema {
                            name: "rhs".into(),
                            multiplicity: Multiplicity::One,
                            value_shape: ChildValueShape::Recursive,
                            scalar_shorthands: vec![],
                            non_empty: false,
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
                        scalar_shorthands: vec![],
                        non_empty: false,
                    }],
                },
                VariantSchema {
                    name: "List".into(),
                    fields: vec![],
                    children: vec![ChildSchema {
                        name: "items".into(),
                        multiplicity: Multiplicity::Many,
                        value_shape: ChildValueShape::Recursive,
                        scalar_shorthands: vec![],
                        non_empty: false,
                    }],
                },
                VariantSchema {
                    name: "Env".into(),
                    fields: vec![],
                    children: vec![ChildSchema {
                        name: "entries".into(),
                        multiplicity: Multiplicity::Map,
                        value_shape: ChildValueShape::Recursive,
                        scalar_shorthands: vec![],
                        non_empty: false,
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

    fn demo_examples() -> (Grammar, GrammarExamples) {
        let g = checked_grammar_from_schema(&demo_schema(), &IdGen::new())
            .expect("demo schema generates");
        let ex = examples_from_grammar(&g).expect("examples synthesize");
        (g, ex)
    }

    #[test]
    fn every_per_rule_example_parses_as_its_own_variant() {
        let (g, ex) = demo_examples();
        assert_eq!(ex.per_rule.len(), 7, "one example per variant");
        for e in &ex.per_rule {
            let tree = g.parse(&e.text).unwrap_or_else(|err| {
                panic!(
                    "example for `{}` failed to parse: {:?}\n  text: {}",
                    e.rule, err.diagnostics, e.text
                )
            });
            assert_eq!(tree.variant, e.rule, "example enters at its own variant");
            let diags = check_conformance(&tree, &demo_schema());
            assert!(
                diags.is_empty(),
                "conformance clean for {}: {diags:?}",
                e.rule
            );
        }
    }

    #[test]
    fn minimal_examples_have_the_expected_spellings() {
        let (_, ex) = demo_examples();
        let by_rule: HashMap<&str, &str> = ex
            .per_rule
            .iter()
            .map(|e| (e.rule.as_str(), e.text.as_str()))
            .collect();
        assert_eq!(by_rule["Lit"], "Lit(value: 1)");
        assert_eq!(by_rule["Name"], "Name(text: \"example\", quoted: true)");
        assert_eq!(
            by_rule["Add"], "Add(lhs: Unit(), rhs: Unit())",
            "nested slots fill with the cheapest variant"
        );
        assert_eq!(
            by_rule["Neg"], "Neg(body: none)",
            "Optional minimal = absent"
        );
        assert_eq!(
            by_rule["List"], "List(items: [])",
            "Many minimal = empty list"
        );
        assert_eq!(
            by_rule["Env"], "Env(entries: {})",
            "Map minimal = empty map, braces tight like the empty list"
        );
        assert_eq!(by_rule["Unit"], "Unit()");
    }

    /// Rich mode expands a keyed slot to two entries; those entries
    /// must carry *different* keys. Emitting the bare token text twice
    /// would produce `{ "example": …, "example": … }` — accepted by the
    /// grammar, rejected by the schema — which is exactly the
    /// grammar/schema drift machine-derived examples exist to avoid.
    #[test]
    fn rich_keyed_map_example_uses_distinct_keys() {
        let map_only = NodeSchema {
            name: "Cfg".into(),
            variants: vec![
                VariantSchema {
                    name: "Env".into(),
                    fields: vec![],
                    children: vec![ChildSchema {
                        name: "entries".into(),
                        multiplicity: Multiplicity::Map,
                        value_shape: ChildValueShape::Recursive,
                        scalar_shorthands: vec![],
                        non_empty: false,
                    }],
                },
                VariantSchema {
                    name: "Unit".into(),
                    fields: vec![],
                    children: vec![],
                },
            ],
        };
        let g = checked_grammar_from_schema(&map_only, &IdGen::new()).expect("generates");
        let ex = examples_from_grammar(&g).expect("synthesizes");

        let tree = g.parse(&ex.composite).unwrap_or_else(|err| {
            panic!(
                "composite failed to parse: {:?}\n  text: {}",
                err.diagnostics, ex.composite
            )
        });
        let entries = tree
            .keyed_child_slot("entries")
            .expect("rich mode fills the keyed slot");
        assert!(
            entries.len() >= 2,
            "rich mode should expand the map: {}",
            ex.composite
        );
        let diags = check_conformance(&tree, &map_only);
        assert!(
            diags.is_empty(),
            "synthesized example must conform: {diags:?}\n  text: {}",
            ex.composite
        );
    }

    /// A keyed entry whose value has no finite derivation poisons the
    /// entry's cost, so the rule it sits in is reported as underivable
    /// instead of sending synthesis into an endless expansion.
    #[test]
    fn keyed_entry_inherits_an_underivable_value() {
        let ids = IdGen::new();
        // `s` can only be written by writing an `s` inside a keyed
        // entry: no finite string enters the rule.
        use crate::peg::{keyed_entry, node, rule, rule_ref, token};
        let entry = keyed_entry(&ids, "entries", token(&ids, "%ident"), rule_ref(&ids, "s"));
        let g = Grammar::new(vec![rule(&ids, "s", node(&ids, "N", entry))], "s");
        let err = examples_from_grammar(&g).expect_err("no finite derivation exists");
        assert!(
            err.diagnostics
                .iter()
                .any(|d| d.code == codes::NO_FINITE_DERIVATION),
            "expected NO_FINITE_DERIVATION; got {:?}",
            err.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn composite_parses_and_shows_nesting() {
        let (g, ex) = demo_examples();
        let tree = g.parse(&ex.composite).unwrap_or_else(|err| {
            panic!(
                "composite failed to parse: {:?}\n  text: {}",
                err.diagnostics, ex.composite
            )
        });
        assert!(check_conformance(&tree, &demo_schema()).is_empty());
        // Rich mode picked an expansive variant and expanded at least
        // one nested child — pin the property, not the exact string.
        assert!(
            !tree.children.is_empty(),
            "composite has child structure: {}",
            ex.composite
        );
    }

    #[test]
    fn underivable_rules_fail_loudly_with_the_full_list() {
        // A single variant whose only child is mandatory and recursive:
        // no finite tree exists.
        let schema = NodeSchema {
            name: "Loop".into(),
            variants: vec![VariantSchema {
                name: "Rec".into(),
                fields: vec![],
                children: vec![ChildSchema {
                    name: "inner".into(),
                    multiplicity: Multiplicity::One,
                    value_shape: ChildValueShape::Recursive,
                    scalar_shorthands: vec![],
                    non_empty: false,
                }],
            }],
        };
        let g = crate::schema_gen::grammar_from_schema(&schema, &IdGen::new())
            .expect("generation itself succeeds");
        let err = examples_from_grammar(&g).expect_err("no finite derivation");
        assert!(
            err.diagnostics
                .iter()
                .all(|d| d.code == codes::NO_FINITE_DERIVATION),
            "{:?}",
            err.diagnostics
        );
        // Both the start rule and the variant rule are underivable.
        assert_eq!(err.diagnostics.len(), 2);
    }
}
