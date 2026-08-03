//! Load-time self-validation of a [`CheckProgram`].
//!
//! Everything this crate reports about a *document* is a hard error a
//! `$allow` annotation cannot silence — a state or type violation is a
//! won't-run condition, not advice. That is only defensible while the
//! program itself is sound: a `CheckProgram` is written by the DSL
//! author, not by this library, and a rule that requires a state
//! nothing ever produces blocks every document that mentions the
//! variant, with no escape hatch anywhere in the pipeline.
//!
//! [`CheckProgram::validate`] is the counterweight. It reads the
//! program on its own, without a document, and answers three questions:
//!
//! | finding | severity | reasoning |
//! |---|---|---|
//! | a premise requires a state nothing reaches | error | the rule can never fire; every document carrying that variant is rejected |
//! | a predicate is produced but never required | warning | legal (a terminal fact), but usually half of a misspelt pair |
//! | a rule is shadowed by an earlier one for the same variant | warning | dead vocabulary — the later rule cannot be reached |
//!
//! Hosts are expected to run it once, where they load the program:
//!
//! ```
//! use dsl_kit_check::{CheckProgram, Rule, SeqSlotDecl, atom, codes, fact};
//!
//! let program = CheckProgram::builder()
//!     .seq_slot(SeqSlotDecl::fold("Plan", "steps", fact("state", [atom("Raw")])))
//!     .rule(
//!         Rule::on("Build")
//!             // Typo: the program never reaches `state(Fetchd)`.
//!             .requires_state(fact("state", [atom("Fetchd")]))
//!             .transitions_to(fact("state", [atom("Built")]))
//!             .message(codes::CHECK_STATE_MISMATCH, "`build` needs {expected}, found {found}"),
//!     )
//!     .build();
//!
//! let diags = program.validate();
//! assert_eq!(diags.len(), 1);
//! assert_eq!(diags[0].code, codes::CHECK_PROGRAM_UNDEFINED_STATE);
//! ```
//!
//! The check is deliberately syntactic: it compares patterns, not
//! reachability through a particular document shape. A rule the walk
//! happens never to visit is out of scope — this layer only rules out
//! the judgements that cannot hold under *any* document.

use std::collections::{BTreeMap, BTreeSet};

use dsl_kit_parse::{Diagnostic, Location, Severity};

use crate::codes;
use crate::ir::{CheckProgram, Fact, Premise};
use crate::solver::may_unify_fact;

impl CheckProgram {
    /// Checks the program against itself and returns what it finds, in
    /// a stable order (errors first, then unused predicates, then
    /// unreachable rules).
    ///
    /// An empty result means every premise is satisfiable by something
    /// the program produces, every predicate is consumed somewhere, and
    /// no rule is shadowed. See the module documentation for the
    /// reasoning behind each finding and its severity.
    pub fn validate(&self) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        self.report_undefined_states(&mut out);
        self.report_unused_preds(&mut out);
        self.report_unreachable_rules(&mut out);
        out
    }

    /// Every fact the program can put into a fold state: the seed of
    /// each declared slot plus every rule's transition.
    fn state_producers(&self) -> Vec<&Fact> {
        let initials = self.seq_slots.iter().map(|d| &d.initial);
        let transitions = self.rules.iter().filter_map(|r| r.state_after.as_ref());
        initials.chain(transitions).collect()
    }

    fn report_undefined_states(&self, out: &mut Vec<Diagnostic>) {
        let producers = self.state_producers();
        for rule in &self.rules {
            for premise in &rule.premises {
                let Premise::State { expect } = premise else {
                    continue;
                };
                if producers.iter().any(|fact| may_unify_fact(expect, fact)) {
                    continue;
                }
                out.push(Diagnostic::error(
                    codes::CHECK_PROGRAM_UNDEFINED_STATE,
                    format!(
                        "rule `{}` requires `{expect}`, but nothing in the program reaches \
                         that state — no rule produces it and no fold declaration starts \
                         from it{}",
                        rule.variant,
                        reachable_summary(&producers, &expect.pred),
                    ),
                ));
            }
        }
    }

    fn report_unused_preds(&self, out: &mut Vec<Diagnostic>) {
        let mut required: BTreeSet<&str> = BTreeSet::new();
        for rule in &self.rules {
            for premise in &rule.premises {
                match premise {
                    Premise::Child { expect, .. } | Premise::State { expect } => {
                        required.insert(expect.pred.as_str());
                    }
                    // Eq / Neq compare terms, not facts: they name no
                    // predicate and so consume none.
                    Premise::Eq(..) | Premise::Neq(..) => {}
                }
            }
        }

        // First origin per predicate, so the wording points somewhere
        // concrete and the iteration order stays stable.
        let mut produced: BTreeMap<&str, String> = BTreeMap::new();
        for decl in &self.seq_slots {
            produced
                .entry(decl.initial.pred.as_str())
                .or_insert_with(|| {
                    format!("the `{}.{}` fold declaration", decl.variant, decl.slot)
                });
        }
        for rule in &self.rules {
            for fact in [rule.conclusion.as_ref(), rule.state_after.as_ref()]
                .into_iter()
                .flatten()
            {
                produced
                    .entry(fact.pred.as_str())
                    .or_insert_with(|| format!("rule `{}`", rule.variant));
            }
        }

        for (pred, origin) in produced {
            if required.contains(pred) {
                continue;
            }
            out.push(warning(
                codes::CHECK_PROGRAM_UNUSED_PRED,
                format!(
                    "predicate `{pred}` is produced (first by {origin}) but no premise ever \
                     requires it — either a rule that consumes it is missing, or one of the \
                     two spellings is a typo"
                ),
            ));
        }
    }

    fn report_unreachable_rules(&self, out: &mut Vec<Diagnostic>) {
        for (index, rule) in self.rules.iter().enumerate() {
            let earlier = self.rules[..index]
                .iter()
                .find(|e| e.variant == rule.variant && shadows(e, rule));
            let Some(earlier) = earlier else { continue };
            let ordinal = self.rules[..index]
                .iter()
                .filter(|e| e.variant == rule.variant)
                .count()
                + 1;
            let reason = if earlier.premises.is_empty() {
                "it is unconditional"
            } else {
                "it carries the same premises"
            };
            out.push(warning(
                codes::CHECK_PROGRAM_UNREACHABLE_RULE,
                format!(
                    "rule #{ordinal} for variant `{}` can never fire: an earlier rule for the \
                     same variant always wins ({reason}), and rules are tried in declaration \
                     order",
                    rule.variant
                ),
            ));
        }
    }
}

/// Whether `earlier` takes every document `later` could have taken.
///
/// Only the two decidable cases: an unconditional rule (which fires for
/// anything) and a literal repeat of the same premises. A rule that is
/// merely *more general* — `state($x)` ahead of `state(Raw)` — is left
/// alone, because deciding that in general is the subsumption problem
/// and a false "unreachable" would be worse than silence.
fn shadows(earlier: &crate::ir::Rule, later: &crate::ir::Rule) -> bool {
    earlier.premises.is_empty() || earlier.premises == later.premises
}

/// `" (reachable: state(Built), state(Raw))"`, or the empty string when
/// the predicate has no producer at all.
fn reachable_summary(producers: &[&Fact], pred: &str) -> String {
    let known: BTreeSet<String> = producers
        .iter()
        .filter(|fact| fact.pred == pred)
        .map(|fact| fact.to_string())
        .collect();
    if known.is_empty() {
        return String::new();
    }
    let listed: Vec<String> = known.into_iter().collect();
    format!(" (reachable: {})", listed.join(", "))
}

/// [`Severity::Warning`] counterpart of [`Diagnostic::error`], which the
/// parse crate does not ship.
fn warning(code: &str, message: String) -> Diagnostic {
    Diagnostic {
        severity: Severity::Warning,
        code: code.to_string(),
        message,
        location: Location::None,
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{CheckProgram, Rule, SeqSlotDecl, atom, fact};
    use crate::{codes, solver};

    fn state(name: &str) -> crate::ir::Fact {
        fact("state", [atom(name)])
    }

    #[test]
    fn producers_cover_both_seeds_and_transitions() {
        let program = CheckProgram::builder()
            .seq_slot(SeqSlotDecl::fold("Plan", "steps", state("Raw")))
            .rule(
                Rule::on("Fetch")
                    .transitions_to(state("Fetched"))
                    .message(codes::CHECK_STATE_MISMATCH, "unused"),
            )
            .build();

        let producers = program.state_producers();
        assert_eq!(producers.len(), 2);
        assert!(
            producers
                .iter()
                .any(|f| solver::may_unify_fact(&state("Raw"), f))
        );
        assert!(
            producers
                .iter()
                .any(|f| solver::may_unify_fact(&state("Fetched"), f))
        );
        assert!(
            !producers
                .iter()
                .any(|f| solver::may_unify_fact(&state("Built"), f))
        );
    }

    #[test]
    fn the_summary_lists_only_the_matching_predicate() {
        let type_int = fact("type", [atom("Int")]);
        let producers = [&state("Raw"), &state("Built"), &type_int];
        assert_eq!(
            super::reachable_summary(&producers, "state"),
            " (reachable: state(Built), state(Raw))"
        );
        assert_eq!(super::reachable_summary(&producers, "cap"), "");
    }
}
