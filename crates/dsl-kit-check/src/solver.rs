//! The engine: one bottom-up pass over a [`ParseTree`], evaluating a
//! [`CheckProgram`].
//!
//! ## Shape of the evaluation
//!
//! ```text
//! solve(node, state_in) -> (conclusion, state_out, diagnostics)
//!   1. evaluate the children first
//!        - a slot declared SeqMode::Fold threads a state left to
//!          right, starting from the declaration's `initial`; each
//!          child's `state_after` advances it
//!        - every other slot (including keyed slots) evaluates its
//!          children independently against `state_in`
//!   2. look up the rules for this variant; no rule means the node
//!      passes through with no conclusion (opt-in — an un-annotated
//!      variant never fails)
//!   3. try each rule in declaration order, unifying its premises
//!      against the collected child conclusions / the running state
//!   4. first rule whose premises all hold supplies `conclusion` and
//!      `state_after`, both grounded with the bindings collected
//!   5. no rule holds -> one diagnostic from the attempt that got
//!      furthest, and the node yields *no* conclusion and leaves the
//!      state untouched
//! ```
//!
//! No fixpoint, no worklist: every node is visited once and the
//! recursion is structural, so evaluation terminates by construction
//! and costs `O(nodes × rules-for-variant × premises)`.
//!
//! ## Why a failed node goes quiet rather than loud
//!
//! Step 5 is the error-recovery contract. A node whose rule failed
//! reports *once* and then contributes nothing: no conclusion (so a
//! parent's `Child` premise treats it as unknown instead of wrong) and
//! no state transition (the rule did not fire, so its `state_after`
//! did not happen). That is the poison-value discipline — one authoring
//! mistake yields one diagnostic instead of a cascade down the rest of
//! the document.
//!
//! ## Anchoring
//!
//! Diagnostics carry the failing node's [`Span`] when the front-end
//! tracks one (the PEG/text front-end does). The serde front-end
//! leaves spans `None`, so the solver keeps its own path trail
//! (`steps[3].argv[0]`) while it walks and every message ends with an
//! `[at …]` suffix naming the path — and the byte range too, when
//! there is one. A diagnostic is therefore never unanchored, whichever
//! front-end produced the tree.

use std::collections::BTreeMap;

use dsl_kit_parse::{Diagnostic, ParseTree, RawValue, Span};

use crate::ir::{CheckProgram, Fact, Premise, Rule, SeqMode, Term};

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

/// Evaluates `program` against `tree` and returns every semantic
/// diagnostic found, in document order.
///
/// An empty result means the document satisfies the program. This is
/// an opt-in pass: the host calls it between
/// `serde_bridge::from_json_str` / `Grammar::parse` and
/// `DslBuild::from_parse_tree`, after `check_conformance` has settled
/// the shape.
///
/// ```
/// use dsl_kit_check::{CheckProgram, Rule, atom, check_semantics, codes, fact};
/// use dsl_kit_parse::ParseTree;
///
/// let program = CheckProgram::builder()
///     .rule(
///         Rule::on("Not")
///             .child("arg", fact("type", [atom("Bool")]))
///             .concludes(fact("type", [atom("Bool")]))
///             .message(codes::CHECK_TYPE_MISMATCH, "`not` wants {expected}, got {found}"),
///     )
///     .rule(
///         Rule::on("IntLit")
///             .concludes(fact("type", [atom("Int")]))
///             .message(codes::CHECK_TYPE_MISMATCH, "int literal"),
///     )
///     .build();
///
/// let mut tree = ParseTree::new("Not");
/// tree.children = vec![("arg".into(), vec![ParseTree::new("IntLit")])];
///
/// let diags = check_semantics(&tree, &program);
/// assert_eq!(diags.len(), 1);
/// assert_eq!(diags[0].code, codes::CHECK_TYPE_MISMATCH);
/// assert!(diags[0].message.contains("`not` wants type(Bool), got type(Int)"));
/// ```
pub fn check_semantics(tree: &ParseTree, program: &CheckProgram) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    solve(tree, "", None, program, &mut diags);
    diags
}

// ---------------------------------------------------------------------------
// Internal value types
// ---------------------------------------------------------------------------

/// Where a fact came from — the path trail plus the span, when the
/// front-end tracked one.
#[derive(Debug, Clone)]
struct Provenance {
    path: String,
    span: Option<Span>,
}

impl Provenance {
    /// The seed of a fold: no node supplied it, the declaration did.
    fn initial() -> Self {
        Self {
            path: "<initial>".into(),
            span: None,
        }
    }

    fn of(path: &str, span: Option<Span>) -> Self {
        Self {
            path: path.to_string(),
            span,
        }
    }

    fn render(&self) -> String {
        location_label(&self.path, self.span)
    }
}

/// The state threaded through a [`SeqMode::Fold`] slot, together with
/// the node that last set it.
#[derive(Debug, Clone)]
struct StateVal {
    fact: Fact,
    prov: Provenance,
}

/// What one node contributes to its parent.
struct Solved {
    conclusion: Option<Fact>,
    prov: Provenance,
    state_out: Option<StateVal>,
}

/// Rule-local variable bindings, one environment per rule attempt.
type Bindings = BTreeMap<String, Term>;

/// Per-slot child conclusions: `None` marks a child that contributed
/// nothing (no rule, or a rule that failed).
type SlotVals = BTreeMap<String, Vec<(Option<Fact>, Provenance)>>;

/// A rule that fired.
struct Applied {
    conclusion: Option<Fact>,
    state_after: Option<Fact>,
}

/// A rule that did not fire, with everything the message needs.
struct Failure<'r> {
    rule: &'r Rule,
    /// How many premises held before the failing one — used to pick
    /// the most informative attempt when a variant carries several
    /// rules.
    satisfied: usize,
    slot: Option<String>,
    expected: String,
    found: String,
    provenance: String,
    bindings: Bindings,
}

// ---------------------------------------------------------------------------
// Walk
// ---------------------------------------------------------------------------

fn solve(
    node: &ParseTree,
    path: &str,
    state_in: Option<&StateVal>,
    program: &CheckProgram,
    diags: &mut Vec<Diagnostic>,
) -> Solved {
    let here = Provenance::of(path, node.span);
    let mut slots: SlotVals = BTreeMap::new();

    // 1a. Positional slots. A declared Fold slot threads its own
    //     state; everything else sees the incoming one unchanged.
    for (slot, children) in &node.children {
        let decl = program.seq_slot(&node.variant, slot);
        let folding = matches!(decl, Some(d) if d.mode == SeqMode::Fold);
        let mut running: Option<StateVal> = match decl {
            Some(d) if folding => Some(StateVal {
                fact: d.initial.clone(),
                prov: Provenance::initial(),
            }),
            _ => state_in.cloned(),
        };

        let mut vals = Vec::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            let child_path = child_path(path, slot, &i.to_string());
            let solved = solve(child, &child_path, running.as_ref(), program, diags);
            if folding {
                if let Some(next) = solved.state_out {
                    running = Some(next);
                }
            }
            vals.push((solved.conclusion, solved.prov));
        }
        slots.insert(slot.clone(), vals);
    }

    // 1b. Keyed slots are unordered by construction, so they evaluate
    //     as SeqMode::All regardless of declaration.
    for (slot, entries) in &node.keyed_children {
        let mut vals = Vec::with_capacity(entries.len());
        for (key, child) in entries {
            let child_path = child_path(path, slot, key);
            let solved = solve(child, &child_path, state_in, program, diags);
            vals.push((solved.conclusion, solved.prov));
        }
        slots.insert(slot.clone(), vals);
    }

    // 2. No rule for this variant: pass through untouched.
    let rules: Vec<&Rule> = program.rules_for(&node.variant).collect();
    if rules.is_empty() {
        return Solved {
            conclusion: None,
            prov: here,
            state_out: state_in.cloned(),
        };
    }

    // 3-4. First rule that holds wins.
    let mut best: Option<Box<Failure<'_>>> = None;
    for rule in rules {
        match try_rule(rule, node, &slots, state_in) {
            Ok(applied) => {
                let state_out = match applied.state_after {
                    Some(fact) => Some(StateVal {
                        fact,
                        prov: here.clone(),
                    }),
                    None => state_in.cloned(),
                };
                return Solved {
                    conclusion: applied.conclusion,
                    prov: here,
                    state_out,
                };
            }
            Err(failure) => {
                let better = best
                    .as_ref()
                    .is_none_or(|b| failure.satisfied > b.satisfied);
                if better {
                    best = Some(failure);
                }
            }
        }
    }

    // 5. Report once, then contribute nothing.
    if let Some(failure) = best {
        diags.push(failure.into_diagnostic(path, node.span));
    }
    Solved {
        conclusion: None,
        prov: here,
        state_out: state_in.cloned(),
    }
}

fn try_rule<'r>(
    rule: &'r Rule,
    node: &ParseTree,
    slots: &SlotVals,
    state_in: Option<&StateVal>,
) -> Result<Applied, Box<Failure<'r>>> {
    // Boxed error: the success arm is what the walk carries around,
    // and a `Failure` (four strings plus a binding map) would otherwise
    // set the size of every rule attempt's `Result`.
    let mut binds: Bindings = Bindings::new();

    for (index, premise) in rule.premises.iter().enumerate() {
        match premise {
            Premise::Child { slot, expect } => {
                let expect = ground_field_refs_fact(expect, node);
                // Absent / empty slots and children without a
                // conclusion say nothing — see the module docs.
                for (found, prov) in slots.get(slot).into_iter().flatten() {
                    let Some(found) = found else { continue };
                    let before = binds.clone();
                    if !unify_fact(&expect, found, &mut binds) {
                        return Err(Box::new(Failure {
                            rule,
                            satisfied: index,
                            slot: Some(slot.clone()),
                            expected: apply_fact(&expect, &before).to_string(),
                            found: found.to_string(),
                            provenance: prov.render(),
                            bindings: before,
                        }));
                    }
                }
            }
            Premise::State { expect } => {
                let expect = ground_field_refs_fact(expect, node);
                let Some(state) = state_in else { continue };
                let before = binds.clone();
                if !unify_fact(&expect, &state.fact, &mut binds) {
                    return Err(Box::new(Failure {
                        rule,
                        satisfied: index,
                        slot: None,
                        expected: apply_fact(&expect, &before).to_string(),
                        found: state.fact.to_string(),
                        provenance: state.prov.render(),
                        bindings: before,
                    }));
                }
            }
            Premise::Eq(lhs, rhs) => {
                let lhs = ground_field_refs(lhs, node);
                let rhs = ground_field_refs(rhs, node);
                let before = binds.clone();
                if !unify(&lhs, &rhs, &mut binds) {
                    return Err(Box::new(Failure {
                        rule,
                        satisfied: index,
                        slot: None,
                        expected: apply(&lhs, &before).to_string(),
                        found: apply(&rhs, &before).to_string(),
                        provenance: location_label(&node.variant, node.span),
                        bindings: before,
                    }));
                }
            }
            Premise::Neq(lhs, rhs) => {
                let lhs = apply(&ground_field_refs(lhs, node), &binds);
                let rhs = apply(&ground_field_refs(rhs, node), &binds);
                // Only a pair of ground, equal terms is a violation:
                // an open term is not evidence of sameness.
                if is_ground(&lhs) && is_ground(&rhs) && lhs == rhs {
                    return Err(Box::new(Failure {
                        rule,
                        satisfied: index,
                        slot: None,
                        expected: format!("anything but {lhs}"),
                        found: rhs.to_string(),
                        provenance: location_label(&node.variant, node.span),
                        bindings: binds.clone(),
                    }));
                }
            }
        }
    }

    Ok(Applied {
        conclusion: rule
            .conclusion
            .as_ref()
            .map(|f| apply_fact(&ground_field_refs_fact(f, node), &binds)),
        state_after: rule
            .state_after
            .as_ref()
            .map(|f| apply_fact(&ground_field_refs_fact(f, node), &binds)),
    })
}

impl Failure<'_> {
    fn into_diagnostic(self, path: &str, span: Option<Span>) -> Diagnostic {
        let body = render_template(
            &self.rule.message.template,
            &MsgCtx {
                slot: self.slot.as_deref(),
                expected: &self.expected,
                found: &self.found,
                provenance: &self.provenance,
                bindings: &self.bindings,
            },
        );
        let message = format!("{body} [at {}]", location_label(path, span));
        Diagnostic::error(self.rule.message.code, message).with_span(span)
    }
}

// ---------------------------------------------------------------------------
// Unification
// ---------------------------------------------------------------------------

/// Unifies two facts: same predicate, same arity, arguments pairwise.
fn unify_fact(pattern: &Fact, actual: &Fact, binds: &mut Bindings) -> bool {
    if pattern.pred != actual.pred || pattern.args.len() != actual.args.len() {
        return false;
    }
    pattern
        .args
        .iter()
        .zip(actual.args.iter())
        .all(|(p, a)| unify(p, a, binds))
}

/// Whether two fact *patterns* could ever describe the same fact.
///
/// Used by [`crate::CheckProgram::validate`], which compares a rule's
/// expectation against everything the program can produce. Both sides
/// may still carry variables, so this is an over-approximation by
/// design: it answers "is there any binding under which these agree?",
/// and reports a rule as unsatisfiable only when the answer is no for
/// every producer.
pub(crate) fn may_unify_fact(a: &Fact, b: &Fact) -> bool {
    unify_fact(a, b, &mut Bindings::new())
}

/// First-order syntactic unification.
///
/// A [`Term::FieldRef`] that survived grounding (the node has no such
/// payload field) stands for an unknown value and unifies with
/// anything — same poison-value reasoning as a child without a
/// conclusion.
fn unify(lhs: &Term, rhs: &Term, binds: &mut Bindings) -> bool {
    let lhs = resolve(lhs, binds);
    let rhs = resolve(rhs, binds);
    match (&lhs, &rhs) {
        (Term::FieldRef(_), _) | (_, Term::FieldRef(_)) => true,
        (Term::Var(name), other) | (other, Term::Var(name)) => {
            if let Term::Var(o) = other {
                if o == name {
                    return true;
                }
            }
            // Generated terms are finite, so this is defensive only.
            if occurs(name, other) {
                return false;
            }
            binds.insert(name.clone(), other.clone());
            true
        }
        (Term::Atom(a), Term::Atom(b)) => a == b,
        (Term::Ctor(an, aa), Term::Ctor(bn, ba)) => {
            an == bn
                && aa.len() == ba.len()
                && aa.iter().zip(ba.iter()).all(|(x, y)| unify(x, y, binds))
        }
        _ => false,
    }
}

fn occurs(name: &str, term: &Term) -> bool {
    match term {
        Term::Var(other) => other == name,
        Term::Ctor(_, args) => args.iter().any(|a| occurs(name, a)),
        Term::Atom(_) | Term::FieldRef(_) => false,
    }
}

/// Follows a variable chain to whatever it is currently bound to.
fn resolve(term: &Term, binds: &Bindings) -> Term {
    let mut current = term.clone();
    // The occurs check keeps chains acyclic; the counter is belt and
    // braces against a hand-built binding map.
    for _ in 0..64 {
        let Term::Var(name) = &current else { break };
        let Some(next) = binds.get(name) else { break };
        current = next.clone();
    }
    current
}

/// Substitutes bindings throughout a term.
fn apply(term: &Term, binds: &Bindings) -> Term {
    match resolve(term, binds) {
        Term::Ctor(name, args) => Term::Ctor(name, args.iter().map(|a| apply(a, binds)).collect()),
        other => other,
    }
}

fn apply_fact(f: &Fact, binds: &Bindings) -> Fact {
    Fact {
        pred: f.pred.clone(),
        args: f.args.iter().map(|a| apply(a, binds)).collect(),
    }
}

fn is_ground(term: &Term) -> bool {
    match term {
        Term::Atom(_) => true,
        Term::Var(_) | Term::FieldRef(_) => false,
        Term::Ctor(_, args) => args.iter().all(is_ground),
    }
}

/// Resolves [`Term::FieldRef`]s against the node the rule fires for.
/// A reference to a field the node does not carry is left in place and
/// treated as unknown at unification time.
fn ground_field_refs(term: &Term, node: &ParseTree) -> Term {
    match term {
        Term::FieldRef(name) => match node.field(name) {
            Some(value) => Term::Atom(raw_value_text(value)),
            None => term.clone(),
        },
        Term::Ctor(name, args) => Term::Ctor(
            name.clone(),
            args.iter().map(|a| ground_field_refs(a, node)).collect(),
        ),
        Term::Atom(_) | Term::Var(_) => term.clone(),
    }
}

fn ground_field_refs_fact(f: &Fact, node: &ParseTree) -> Fact {
    Fact {
        pred: f.pred.clone(),
        args: f.args.iter().map(|a| ground_field_refs(a, node)).collect(),
    }
}

/// Renders a payload value as the atom text a term compares against.
///
/// A JSON string keeps its contents (not its quotes) so that the two
/// front-ends agree: the text front-end already hands over the decoded
/// body of a string literal. Any other JSON value falls back to its
/// compact rendering (`1`, `true`, `[1,2]`).
fn raw_value_text(value: &RawValue) -> String {
    match value {
        RawValue::Text(text) => text.clone(),
        RawValue::Json(json) => match json.as_str() {
            Some(text) => text.to_string(),
            None => json.to_string(),
        },
    }
}

// ---------------------------------------------------------------------------
// Paths and messages
// ---------------------------------------------------------------------------

/// `steps[3]`, `steps[3].argv[0]`, `env[LOG]` — the trail the walk
/// keeps so a span-less tree still gets an anchor.
fn child_path(parent: &str, slot: &str, index: &str) -> String {
    if parent.is_empty() {
        format!("{slot}[{index}]")
    } else {
        format!("{parent}.{slot}[{index}]")
    }
}

/// `steps[1] (bytes 12..24)` when there is a span, `steps[1]`
/// otherwise; the root of the document reads `(root)`.
fn location_label(path: &str, span: Option<Span>) -> String {
    let path = if path.is_empty() { "(root)" } else { path };
    match span {
        Some(span) => format!("{path} (bytes {}..{})", span.start, span.end),
        None => path.to_string(),
    }
}

struct MsgCtx<'a> {
    slot: Option<&'a str>,
    expected: &'a str,
    found: &'a str,
    provenance: &'a str,
    bindings: &'a Bindings,
}

/// Substitutes `{…}` holes in a [`crate::MessageTemplate`].
///
/// Unknown holes are copied through verbatim: a template typo shows up
/// in the message instead of silently erasing a word.
fn render_template(template: &str, ctx: &MsgCtx<'_>) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else {
            // Unterminated hole: copy the remainder as written.
            out.push_str(&rest[open..]);
            return out;
        };
        let hole = &after[..close];
        match hole.strip_prefix('$') {
            Some(name) => match ctx.bindings.get(name) {
                Some(term) => out.push_str(&apply(term, ctx.bindings).to_string()),
                None => out.push('?'),
            },
            None => match hole {
                "slot" => out.push_str(ctx.slot.unwrap_or("?")),
                "expected" => out.push_str(ctx.expected),
                "found" => out.push_str(ctx.found),
                "provenance" => out.push_str(ctx.provenance),
                _ => {
                    out.push('{');
                    out.push_str(hole);
                    out.push('}');
                }
            },
        }
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{atom, ctor, fact, var};

    fn binds(pairs: &[(&str, Term)]) -> Bindings {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn unify_binds_variables_and_rejects_mismatches() {
        let mut b = Bindings::new();
        assert!(unify_fact(
            &fact("type", [var("a")]),
            &fact("type", [atom("Int")]),
            &mut b
        ));
        assert_eq!(b.get("a"), Some(&atom("Int")));

        // Same variable, now pinned to Int, refuses Bool.
        assert!(!unify_fact(
            &fact("type", [var("a")]),
            &fact("type", [atom("Bool")]),
            &mut b
        ));

        // Predicate and arity are part of the match.
        let mut b = Bindings::new();
        assert!(!unify_fact(
            &fact("type", [atom("Int")]),
            &fact("state", [atom("Int")]),
            &mut b
        ));
        assert!(!unify_fact(
            &fact("type", [atom("Int")]),
            &fact("type", [atom("Int"), atom("Int")]),
            &mut b
        ));
    }

    #[test]
    fn unify_walks_into_constructors() {
        let mut b = Bindings::new();
        assert!(unify(
            &ctor("Running", [var("n")]),
            &ctor("Running", [atom("comfyui")]),
            &mut b
        ));
        assert_eq!(b.get("n"), Some(&atom("comfyui")));
        assert!(!unify(
            &ctor("Running", [atom("a")]),
            &ctor("Stopped", [atom("a")]),
            &mut Bindings::new()
        ));
    }

    #[test]
    fn field_refs_are_unknown_until_grounded() {
        // Ungrounded: unifies with anything rather than inventing a
        // mismatch.
        assert!(unify(
            &Term::FieldRef("name".into()),
            &atom("comfyui"),
            &mut Bindings::new()
        ));

        let mut node = ParseTree::new("Service");
        node.fields = vec![("name".into(), RawValue::Text("comfyui".into()))];
        let grounded = ground_field_refs(&Term::FieldRef("name".into()), &node);
        assert_eq!(grounded, atom("comfyui"));
        assert!(!unify(&grounded, &atom("other"), &mut Bindings::new()));
    }

    #[test]
    fn template_holes_expand() {
        let b = binds(&[("a", atom("Int"))]);
        let ctx = MsgCtx {
            slot: Some("cond"),
            expected: "type(Bool)",
            found: "type(Int)",
            provenance: "cond[0] (bytes 3..8)",
            bindings: &b,
        };
        assert_eq!(
            render_template(
                "{slot}: want {expected}, got {found} from {provenance}; $a = {$a}, $z = {$z}",
                &ctx
            ),
            "cond: want type(Bool), got type(Int) from cond[0] (bytes 3..8); $a = Int, $z = ?"
        );
        // Unknown and unterminated holes survive verbatim.
        assert_eq!(render_template("{nope} {oops", &ctx), "{nope} {oops");
    }

    #[test]
    fn paths_and_labels_read_like_the_document() {
        assert_eq!(child_path("", "steps", "3"), "steps[3]");
        assert_eq!(child_path("steps[3]", "argv", "0"), "steps[3].argv[0]");
        assert_eq!(location_label("", None), "(root)");
        assert_eq!(
            location_label("steps[1]", Some(Span::new(4, 9))),
            "steps[1] (bytes 4..9)"
        );
    }
}
