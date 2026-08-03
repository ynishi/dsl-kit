//! Check IR — the data model a DSL author writes their type system,
//! state machine, or capability discipline in.
//!
//! Everything here is plain data: a [`CheckProgram`] is a bag of
//! [`Rule`]s (one syntax-directed judgement per variant) plus the
//! declarations that say which child slots carry a sequential meaning
//! ([`SeqSlotDecl`]). The engine that evaluates it lives in
//! [`crate::solver`] and never grows a case for a new predicate — the
//! predicate name (`"type"` / `"state"` / `"cap"` / whatever the author
//! invents) is a [`Fact::pred`] string, not an engine concept.
//!
//! Two ways to build a program, both landing on the same value:
//!
//! ```
//! use dsl_kit_check::{CheckProgram, Rule, atom, codes, fact, var};
//!
//! // Fluent (mirrors the shape a future `derive(DslCheck)` emits).
//! let program = CheckProgram::builder()
//!     .rule(
//!         Rule::on("If")
//!             .child("cond", fact("type", [atom("Bool")]))
//!             .child("then_branch", fact("type", [var("a")]))
//!             .child("else_branch", fact("type", [var("a")]))
//!             .concludes(fact("type", [var("a")]))
//!             .message(
//!                 codes::CHECK_TYPE_MISMATCH,
//!                 "branches of `if` must agree: {expected} vs {found}",
//!             ),
//!     )
//!     .build();
//! assert_eq!(program.rules.len(), 1);
//! ```
//!
//! Struct-literal construction works too — every field is public, so a
//! generator (macro, JSON loader) can emit the value directly.

use std::fmt;

// ---------------------------------------------------------------------------
// Terms and facts
// ---------------------------------------------------------------------------

/// A first-order term: the argument shape of a [`Fact`].
///
/// Terms are compared by syntactic unification (see
/// [`crate::solver`]), so `ServiceRunning($name)` matches
/// `ServiceRunning(comfyui)` and binds `$name` to `comfyui` for the
/// rest of that rule's evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    /// A ground constant — `Int`, `SystemReady`, `comfyui`.
    Atom(String),
    /// A constructor application — `ServiceRunning(comfyui)` is
    /// `Ctor("ServiceRunning", [Atom("comfyui")])`.
    Ctor(String, Vec<Term>),
    /// A rule-local variable, written `$a` in messages. Scope is one
    /// attempt at one rule on one node.
    Var(String),
    /// The value of a payload field on the node the rule fires for,
    /// lifted into a term. `FieldRef("name")` on a
    /// `ComfyUIService { name: "comfyui" }` node resolves to
    /// `Atom("comfyui")` before unification.
    FieldRef(String),
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Atom(name) => f.write_str(name),
            Term::Var(name) => write!(f, "${name}"),
            Term::FieldRef(name) => write!(f, "@{name}"),
            Term::Ctor(name, args) => {
                f.write_str(name)?;
                if args.is_empty() {
                    return Ok(());
                }
                f.write_str("(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{a}")?;
                }
                f.write_str(")")
            }
        }
    }
}

/// A judgement the solver derives or matches: predicate name plus
/// argument terms.
///
/// `type(Int)`, `state(SystemReady)`, `cap(net, outbound)` are all
/// `Fact`s — the predicate is data, so a DSL author can introduce a
/// fourth family without touching the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fact {
    /// Predicate name — `"type"`, `"state"`, `"cap"`, …
    pub pred: String,
    /// Argument terms, arity-significant (a `Fact` only unifies with
    /// another of the same predicate *and* the same arity).
    pub args: Vec<Term>,
}

impl Fact {
    /// Builds a fact from a predicate name and its arguments.
    pub fn new(pred: impl Into<String>, args: impl IntoIterator<Item = Term>) -> Self {
        Self {
            pred: pred.into(),
            args: args.into_iter().collect(),
        }
    }
}

impl fmt::Display for Fact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.pred)?;
        if self.args.is_empty() {
            return Ok(());
        }
        f.write_str("(")?;
        for (i, a) in self.args.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{a}")?;
        }
        f.write_str(")")
    }
}

/// Builds a [`Term::Atom`].
pub fn atom(name: impl Into<String>) -> Term {
    Term::Atom(name.into())
}

/// Builds a [`Term::Var`]. Write the name without the `$` sigil —
/// `var("a")` renders as `$a`.
pub fn var(name: impl Into<String>) -> Term {
    Term::Var(name.into())
}

/// Builds a [`Term::Ctor`].
pub fn ctor(name: impl Into<String>, args: impl IntoIterator<Item = Term>) -> Term {
    Term::Ctor(name.into(), args.into_iter().collect())
}

/// Builds a [`Term::FieldRef`] pointing at a payload field of the node
/// the rule fires for.
pub fn field_ref(name: impl Into<String>) -> Term {
    Term::FieldRef(name.into())
}

/// Builds a [`Fact`]. Shorthand for [`Fact::new`].
pub fn fact(pred: impl Into<String>, args: impl IntoIterator<Item = Term>) -> Fact {
    Fact::new(pred, args)
}

// ---------------------------------------------------------------------------
// Premises
// ---------------------------------------------------------------------------

/// One condition a [`Rule`] needs before it fires.
///
/// Premises are evaluated left to right and share one binding
/// environment, so an earlier premise can bind a variable a later one
/// constrains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Premise {
    /// The conclusion of every child in `slot` must unify with
    /// `expect`.
    ///
    /// A slot that is absent, empty, or whose children carry no
    /// conclusion (no rule fired for them, or their rule failed)
    /// satisfies the premise vacuously — shape is
    /// `check_conformance`'s job, and an unknown child must not
    /// manufacture a second error downstream.
    Child {
        /// Child slot name on the node the rule fires for.
        slot: String,
        /// Pattern the child's conclusion must match.
        expect: Fact,
    },
    /// The current fold state must unify with `expect`.
    ///
    /// The state is the one threaded through the enclosing
    /// [`SeqMode::Fold`] slot. Outside such a slot no state is in
    /// scope and the premise passes vacuously.
    State {
        /// Pattern the running state must match.
        expect: Fact,
    },
    /// Two terms must unify (binding variables in the process).
    Eq(Term, Term),
    /// Two terms must **not** be provably equal. Terms that are still
    /// open (unbound variable, unresolved field reference) count as
    /// not-provably-equal, so this never fires on missing information.
    Neq(Term, Term),
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// The error wording a [`Rule`] emits when its premises do not hold.
///
/// Required on every rule on purpose: a generic solver that can only
/// say "unification failed" is unusable feedback, so the vocabulary
/// author is made to write the sentence at declaration time.
///
/// Holes, substituted by [`crate::solver`]:
///
/// | hole | expands to |
/// |---|---|
/// | `{$name}` | the value bound to `$name`, or `?` if unbound |
/// | `{slot}` | the child slot under evaluation (`?` outside a `Child` premise) |
/// | `{expected}` | the premise pattern, with bindings applied |
/// | `{found}` | the fact that was actually there |
/// | `{provenance}` | where the offending fact came from — `steps[1] (bytes 12..24)` |
///
/// An unrecognised hole is left verbatim rather than swallowed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageTemplate {
    /// Stable diagnostic slug — see [`crate::codes`].
    pub code: &'static str,
    /// Template text with `{…}` holes.
    pub template: String,
}

impl MessageTemplate {
    /// Builds a template from a code slug and template text.
    pub fn new(code: &'static str, template: impl Into<String>) -> Self {
        Self {
            code,
            template: template.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

/// One syntax-directed judgement, attached to a variant by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    /// Variant this rule fires for ([`dsl_kit_parse::ParseTree::variant`]).
    pub variant: String,
    /// Conditions, evaluated in order under one binding environment.
    pub premises: Vec<Premise>,
    /// Synthesised attribute handed to the parent (`produces`). `None`
    /// means the node contributes no fact.
    pub conclusion: Option<Fact>,
    /// State the node moves the enclosing fold to. `None` leaves the
    /// state untouched.
    pub state_after: Option<Fact>,
    /// Wording emitted when a premise fails.
    pub message: MessageTemplate,
}

impl Rule {
    /// Starts building a rule for `variant`.
    ///
    /// The chain is terminated by [`RuleBuilder::message`], which is
    /// what produces the [`Rule`] — the message is not optional.
    pub fn on(variant: impl Into<String>) -> RuleBuilder {
        RuleBuilder {
            variant: variant.into(),
            premises: Vec::new(),
            conclusion: None,
            state_after: None,
        }
    }
}

/// Fluent builder for a [`Rule`]. See [`Rule::on`].
#[derive(Debug, Clone)]
pub struct RuleBuilder {
    variant: String,
    premises: Vec<Premise>,
    conclusion: Option<Fact>,
    state_after: Option<Fact>,
}

impl RuleBuilder {
    /// Adds a [`Premise::Child`].
    pub fn child(mut self, slot: impl Into<String>, expect: Fact) -> Self {
        self.premises.push(Premise::Child {
            slot: slot.into(),
            expect,
        });
        self
    }

    /// Adds a [`Premise::State`].
    pub fn requires_state(mut self, expect: Fact) -> Self {
        self.premises.push(Premise::State { expect });
        self
    }

    /// Adds a [`Premise::Eq`].
    pub fn eq(mut self, lhs: Term, rhs: Term) -> Self {
        self.premises.push(Premise::Eq(lhs, rhs));
        self
    }

    /// Adds a [`Premise::Neq`].
    pub fn neq(mut self, lhs: Term, rhs: Term) -> Self {
        self.premises.push(Premise::Neq(lhs, rhs));
        self
    }

    /// Adds an already-built premise (escape hatch for generators).
    pub fn premise(mut self, premise: Premise) -> Self {
        self.premises.push(premise);
        self
    }

    /// Sets the conclusion handed to the parent.
    pub fn concludes(mut self, conclusion: Fact) -> Self {
        self.conclusion = Some(conclusion);
        self
    }

    /// Sets the state this node moves the enclosing fold to.
    pub fn transitions_to(mut self, state: Fact) -> Self {
        self.state_after = Some(state);
        self
    }

    /// Attaches the failure wording and finishes the rule.
    pub fn message(self, code: &'static str, template: impl Into<String>) -> Rule {
        Rule {
            variant: self.variant,
            premises: self.premises,
            conclusion: self.conclusion,
            state_after: self.state_after,
            message: MessageTemplate::new(code, template),
        }
    }
}

// ---------------------------------------------------------------------------
// Sequential slots
// ---------------------------------------------------------------------------

/// How a child slot's elements relate to each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqMode {
    /// Order matters: a state is threaded left to right through the
    /// slot's children, each child's
    /// [`Rule::state_after`] updating it.
    Fold,
    /// Order does not matter: every child is evaluated against the
    /// same incoming state.
    All,
}

/// Declares that a `(variant, slot)` pair carries a sequential
/// meaning, and what state the sequence starts from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqSlotDecl {
    /// Variant owning the slot.
    pub variant: String,
    /// Slot name on that variant.
    pub slot: String,
    /// State the fold starts from (e.g. `state(Raw)`).
    pub initial: Fact,
    /// Fold or independent evaluation.
    pub mode: SeqMode,
}

impl SeqSlotDecl {
    /// Declares a [`SeqMode::Fold`] slot.
    pub fn fold(variant: impl Into<String>, slot: impl Into<String>, initial: Fact) -> Self {
        Self {
            variant: variant.into(),
            slot: slot.into(),
            initial,
            mode: SeqMode::Fold,
        }
    }

    /// Declares a [`SeqMode::All`] slot. Equivalent to leaving the
    /// slot undeclared; spelled out when the author wants the intent
    /// on the record.
    pub fn all(variant: impl Into<String>, slot: impl Into<String>, initial: Fact) -> Self {
        Self {
            variant: variant.into(),
            slot: slot.into(),
            initial,
            mode: SeqMode::All,
        }
    }
}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

/// A whole type system / state machine / capability discipline as
/// data. Fed to [`crate::check_semantics`] together with a tree.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CheckProgram {
    /// Judgements, matched against a node by [`Rule::variant`]. More
    /// than one rule per variant is allowed: they are tried in
    /// declaration order and the first whose premises all hold wins.
    pub rules: Vec<Rule>,
    /// Slots whose children carry a sequential meaning.
    pub seq_slots: Vec<SeqSlotDecl>,
}

impl CheckProgram {
    /// Builds a program from its two parts.
    pub fn new(rules: Vec<Rule>, seq_slots: Vec<SeqSlotDecl>) -> Self {
        Self { rules, seq_slots }
    }

    /// Starts a [`CheckProgramBuilder`].
    pub fn builder() -> CheckProgramBuilder {
        CheckProgramBuilder::default()
    }

    /// Rules attached to `variant`, in declaration order.
    pub fn rules_for<'a>(&'a self, variant: &'a str) -> impl Iterator<Item = &'a Rule> + 'a {
        self.rules.iter().filter(move |r| r.variant == variant)
    }

    /// The declaration for a `(variant, slot)` pair, if any.
    pub fn seq_slot(&self, variant: &str, slot: &str) -> Option<&SeqSlotDecl> {
        self.seq_slots
            .iter()
            .find(|d| d.variant == variant && d.slot == slot)
    }
}

/// Fluent builder for a [`CheckProgram`]. See [`CheckProgram::builder`].
#[derive(Debug, Clone, Default)]
pub struct CheckProgramBuilder {
    rules: Vec<Rule>,
    seq_slots: Vec<SeqSlotDecl>,
}

impl CheckProgramBuilder {
    /// Appends a rule.
    pub fn rule(mut self, rule: Rule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Appends a sequential-slot declaration.
    pub fn seq_slot(mut self, decl: SeqSlotDecl) -> Self {
        self.seq_slots.push(decl);
        self
    }

    /// Shorthand for `seq_slot(SeqSlotDecl::fold(variant, slot, initial))`.
    pub fn fold_slot(
        self,
        variant: impl Into<String>,
        slot: impl Into<String>,
        initial: Fact,
    ) -> Self {
        self.seq_slot(SeqSlotDecl::fold(variant, slot, initial))
    }

    /// Finishes the program.
    pub fn build(self) -> CheckProgram {
        CheckProgram {
            rules: self.rules,
            seq_slots: self.seq_slots,
        }
    }
}

// ---------------------------------------------------------------------------
// DslCheck
// ---------------------------------------------------------------------------

/// Contract for a DSL type that ships its own check program.
///
/// `#[derive(DslCheck)]` (in `dsl-kit-macros`) emits it from
/// `#[dsl_check(...)]` attributes, exactly as `#[derive(DslSchema)]`
/// emits [`dsl_kit_schema::DslSchema`]; writing the impl by hand is the
/// route for programs the attribute vocabulary cannot spell.
pub trait DslCheck {
    /// Returns the judgement rules for this DSL type.
    fn check_program() -> CheckProgram;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes;

    #[test]
    fn terms_render_readably() {
        assert_eq!(atom("Int").to_string(), "Int");
        assert_eq!(var("a").to_string(), "$a");
        assert_eq!(field_ref("name").to_string(), "@name");
        assert_eq!(
            ctor("ServiceRunning", [atom("comfyui")]).to_string(),
            "ServiceRunning(comfyui)"
        );
        assert_eq!(fact("type", [atom("Bool")]).to_string(), "type(Bool)");
        assert_eq!(fact("ready", []).to_string(), "ready");
    }

    #[test]
    fn builder_and_literal_agree() {
        let built = CheckProgram::builder()
            .rule(
                Rule::on("Lit")
                    .concludes(fact("type", [atom("Int")]))
                    .message(codes::CHECK_TYPE_MISMATCH, "unused"),
            )
            .fold_slot("Plan", "steps", fact("state", [atom("Raw")]))
            .build();

        let literal = CheckProgram {
            rules: vec![Rule {
                variant: "Lit".into(),
                premises: vec![],
                conclusion: Some(Fact::new("type", [atom("Int")])),
                state_after: None,
                message: MessageTemplate::new(codes::CHECK_TYPE_MISMATCH, "unused"),
            }],
            seq_slots: vec![SeqSlotDecl {
                variant: "Plan".into(),
                slot: "steps".into(),
                initial: Fact::new("state", [atom("Raw")]),
                mode: SeqMode::Fold,
            }],
        };

        assert_eq!(built, literal);
        assert_eq!(built.rules_for("Lit").count(), 1);
        assert_eq!(built.rules_for("Nope").count(), 0);
        assert!(built.seq_slot("Plan", "steps").is_some());
        assert!(built.seq_slot("Plan", "other").is_none());
    }
}
