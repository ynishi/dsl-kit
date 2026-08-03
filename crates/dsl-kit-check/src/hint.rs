//! `did you mean` enrichment for the solver's diagnostics.
//!
//! A state handle is a *value* — `ServiceRunning(comfyui)` names the
//! service the document itself started — so a typo in one cannot be
//! caught by the schema layer the way an unknown variant or field can:
//! nothing declares `comfyui` anywhere. The evidence only exists at
//! check time, in two places, and this module collects both:
//!
//! | source | supplies |
//! |---|---|
//! | the [`CheckProgram`] | every ground name a rule can ever mention — atoms and constructor names in premises, conclusions, transitions, fold seeds |
//! | the fact that was actually found | the names the document produced — the fold state a previous step left behind, the conclusion a child synthesised |
//!
//! The second source is what makes the hint work at all: when
//! `ComfyUIService { name: "comfyui" }` leaves `state(ServiceRunning(comfyui))`
//! behind, `comfyui` appears in no schema and no program — only in the
//! running state a later `Readiness { target: "comfy" }` fails against.
//!
//! ## Where the wording comes from
//!
//! The formatting goes through [`Suggester::enrich_unknown`], the
//! contract `dsl-kit-core` already owns and every other layer already
//! uses (`format_unknown_variant` / `format_unknown_slot` in
//! `dsl-kit-parse`, `format_unknown_tool` / `format_unknown_mode` in
//! `dsl-kit-mcp` all bottom out in it). This crate therefore adds no
//! third spelling of the hint: it decides *what to compare against*,
//! which is the part that is genuinely check-specific, and leaves the
//! sentence to core.
//!
//! Suggestions are enrichment only — the caller passes a
//! [`Suggester`], and the default
//! [`check_semantics`](crate::check_semantics) passes a no-op one, so a
//! host that does not want a similarity algorithm keeps byte-identical
//! messages.

use std::collections::BTreeSet;

use dsl_kit_core::Suggester;

use crate::ir::{CheckProgram, Fact, Premise, Term};

/// The ground names a typo can be measured against, split by the kind
/// of position they occupy.
///
/// Atoms and constructor names are kept apart on purpose: `comfyui`
/// (a value) and `ServiceRunning` (a shape) live in different
/// positions, and offering one where the other belongs would be noise
/// rather than a hint.
#[derive(Debug, Default, Clone)]
pub(crate) struct Vocabulary {
    atoms: BTreeSet<String>,
    ctors: BTreeSet<String>,
}

impl Vocabulary {
    /// Collects every ground name the program itself can mention.
    pub(crate) fn of_program(program: &CheckProgram) -> Self {
        let mut out = Self::default();
        for decl in &program.seq_slots {
            out.absorb_fact(&decl.initial);
        }
        for rule in &program.rules {
            for premise in &rule.premises {
                match premise {
                    Premise::Child { expect, .. } | Premise::State { expect } => {
                        out.absorb_fact(expect)
                    }
                    Premise::Eq(lhs, rhs) | Premise::Neq(lhs, rhs) => {
                        out.absorb(lhs);
                        out.absorb(rhs);
                    }
                }
            }
            for fact in [rule.conclusion.as_ref(), rule.state_after.as_ref()]
                .into_iter()
                .flatten()
            {
                out.absorb_fact(fact);
            }
        }
        out
    }

    fn absorb_fact(&mut self, fact: &Fact) {
        for arg in &fact.args {
            self.absorb(arg);
        }
    }

    fn absorb(&mut self, term: &Term) {
        match term {
            Term::Atom(name) => {
                self.atoms.insert(name.clone());
            }
            Term::Ctor(name, args) => {
                self.ctors.insert(name.clone());
                for arg in args {
                    self.absorb(arg);
                }
            }
            // A variable or an unresolved field reference names no
            // value: it matches anything, so it can neither be a typo
            // nor a candidate for one.
            Term::Var(_) | Term::FieldRef(_) => {}
        }
    }

    /// A `did you mean: …` hint for the first name `expected` and
    /// `found` disagree on, or `None` when they disagree about
    /// something a spelling correction cannot fix (a different
    /// predicate, a different arity, a value against a constructor).
    ///
    /// Candidates are this vocabulary plus everything `found` itself
    /// mentions — the document's own names are the half the program
    /// cannot know.
    pub(crate) fn did_you_mean(
        &self,
        expected: &Fact,
        found: &Fact,
        suggester: &dyn Suggester,
    ) -> Option<String> {
        let (query, kind) = first_disagreement(expected, found)?;
        let mut seen = match kind {
            NameKind::Atom => self.atoms.clone(),
            NameKind::Ctor => self.ctors.clone(),
        };
        let mut from_found = Vocabulary::default();
        from_found.absorb_fact(found);
        seen.extend(match kind {
            NameKind::Atom => from_found.atoms,
            NameKind::Ctor => from_found.ctors,
        });
        seen.remove(query);
        if seen.is_empty() {
            return None;
        }
        let candidates: Vec<&str> = seen.iter().map(String::as_str).collect();
        suggester.enrich_unknown(query, &candidates)
    }
}

/// Which pool a name belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NameKind {
    Atom,
    Ctor,
}

/// The first name in `expected` that `found` contradicts, walking both
/// in parallel.
///
/// Structural disagreements (predicate, arity, a value where a
/// constructor stands) return `None`: those are not spelling mistakes,
/// and the `{expected}` / `{found}` holes already say so plainly.
fn first_disagreement<'a>(expected: &'a Fact, found: &'a Fact) -> Option<(&'a str, NameKind)> {
    if expected.pred != found.pred || expected.args.len() != found.args.len() {
        return None;
    }
    expected
        .args
        .iter()
        .zip(found.args.iter())
        .find_map(|(e, f)| first_term_disagreement(e, f))
}

fn first_term_disagreement<'a>(expected: &'a Term, found: &'a Term) -> Option<(&'a str, NameKind)> {
    match (expected, found) {
        (Term::Atom(a), Term::Atom(b)) if a != b => Some((a.as_str(), NameKind::Atom)),
        (Term::Ctor(an, aa), Term::Ctor(bn, ba)) => {
            if an != bn {
                return Some((an.as_str(), NameKind::Ctor));
            }
            if aa.len() != ba.len() {
                return None;
            }
            aa.iter()
                .zip(ba.iter())
                .find_map(|(e, f)| first_term_disagreement(e, f))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{CheckProgram, Rule, SeqSlotDecl, atom, ctor, fact, field_ref, var};
    use crate::{codes, solver};
    use dsl_kit_core::Suggestion;

    /// Suggests any candidate sharing the query's first three
    /// characters. Enough to pin the wiring without importing a
    /// similarity algorithm into a unit test.
    struct PrefixSuggester;

    impl Suggester for PrefixSuggester {
        fn suggest<'a>(&self, query: &str, candidates: &'a [&str]) -> Vec<Suggestion<'a>> {
            let head: String = query.chars().take(3).collect();
            candidates
                .iter()
                .filter(|c| c.starts_with(&head))
                .map(|c| Suggestion {
                    candidate: c,
                    score: 0.9,
                })
                .collect()
        }
    }

    fn program() -> CheckProgram {
        CheckProgram::builder()
            .seq_slot(SeqSlotDecl::fold(
                "Plan",
                "steps",
                fact("state", [atom("Raw")]),
            ))
            .rule(
                Rule::on("Service")
                    .requires_state(fact("state", [atom("Ready")]))
                    .transitions_to(fact("state", [ctor("ServiceRunning", [field_ref("name")])]))
                    .message(codes::CHECK_STATE_MISMATCH, "unused"),
            )
            .rule(
                Rule::on("Probe")
                    .requires_state(fact("state", [ctor("ServiceRunning", [var("target")])]))
                    .message(codes::CHECK_STATE_MISMATCH, "unused"),
            )
            .build()
    }

    #[test]
    fn the_program_supplies_its_ground_names_only() {
        let vocab = Vocabulary::of_program(&program());
        assert_eq!(
            vocab.atoms,
            ["Raw", "Ready"].map(String::from).into_iter().collect()
        );
        assert_eq!(
            vocab.ctors,
            ["ServiceRunning"].map(String::from).into_iter().collect()
        );
    }

    #[test]
    fn the_document_supplies_the_handle_the_program_cannot_know() {
        let vocab = Vocabulary::of_program(&program());
        // `comfyui` exists nowhere in the program — it arrived as a
        // payload value and reached the state through a FieldRef.
        let hint = vocab.did_you_mean(
            &fact("state", [ctor("ServiceRunning", [atom("comfy")])]),
            &fact("state", [ctor("ServiceRunning", [atom("comfyui")])]),
            &PrefixSuggester,
        );
        assert_eq!(hint.as_deref(), Some("did you mean: comfyui"));
    }

    #[test]
    fn a_constructor_name_is_matched_against_constructor_names() {
        let vocab = Vocabulary::of_program(&program());
        let hint = vocab.did_you_mean(
            &fact("state", [ctor("ServiceRunnin", [atom("x")])]),
            &fact("state", [ctor("Stopped", [atom("x")])]),
            &PrefixSuggester,
        );
        assert_eq!(hint.as_deref(), Some("did you mean: ServiceRunning"));
    }

    #[test]
    fn structural_disagreements_get_no_spelling_advice() {
        let vocab = Vocabulary::of_program(&program());
        // Different predicate.
        assert!(
            vocab
                .did_you_mean(
                    &fact("state", [atom("Ready")]),
                    &fact("type", [atom("Read")]),
                    &PrefixSuggester
                )
                .is_none()
        );
        // Different arity.
        assert!(
            vocab
                .did_you_mean(
                    &fact("state", [atom("Ready")]),
                    &fact("state", [atom("Read"), atom("Read")]),
                    &PrefixSuggester
                )
                .is_none()
        );
        // A value where a constructor stands.
        assert!(
            vocab
                .did_you_mean(
                    &fact("state", [atom("ServiceRunning")]),
                    &fact("state", [ctor("ServiceRunning", [atom("x")])]),
                    &PrefixSuggester
                )
                .is_none()
        );
    }

    #[test]
    fn nothing_close_yields_no_hint() {
        let vocab = Vocabulary::of_program(&program());
        assert!(
            vocab
                .did_you_mean(
                    &fact("state", [atom("Zzz")]),
                    &fact("state", [atom("Raw")]),
                    &PrefixSuggester
                )
                .is_none()
        );
    }

    #[test]
    fn a_matching_pair_is_not_reported_as_a_disagreement() {
        // Sanity: identical facts unify, so they never reach here —
        // but the walk must agree, or a hint could attach to a rule
        // that actually held.
        assert!(
            first_disagreement(
                &fact("state", [ctor("ServiceRunning", [atom("a")])]),
                &fact("state", [ctor("ServiceRunning", [atom("a")])]),
            )
            .is_none()
        );
        // Variables and field references are wildcards, not names.
        assert!(
            first_disagreement(&fact("state", [var("x")]), &fact("state", [atom("Raw")])).is_none()
        );
        // And the solver agrees they unify, which is why they cannot
        // be the cause of a failure.
        assert!(solver::may_unify_fact(
            &fact("state", [var("x")]),
            &fact("state", [atom("Raw")])
        ));
    }
}
