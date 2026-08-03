//! Diagnostic codes emitted by [`crate::check_semantics`].
//!
//! Same shape as `dsl_kit_parse::codes`: stable `pub const &str` slugs
//! under a per-layer namespace, so downstream tooling (explain
//! catalogs, lint UIs, doc generators) can reference a check failure
//! without matching on message text.
//!
//! A [`crate::MessageTemplate`] names the slug it reports under, so
//! these are the *vocabulary* a DSL author picks from — not a fixed
//! list the engine assigns. Authors with a fourth predicate family are
//! free to pass their own `&'static str`.

/// A node's state premise did not hold: the sequence reached this node
/// in a state the rule does not accept (step out of order, missing
/// prerequisite step).
pub const CHECK_STATE_MISMATCH: &str = "dsl_kit::check::state_mismatch";

/// A type premise did not hold: a child's synthesised type is not the
/// one the rule requires, or two positions that must agree do not.
pub const CHECK_TYPE_MISMATCH: &str = "dsl_kit::check::type_mismatch";

/// A rule referred to a handle (service name, resource id, capability
/// subject) that nothing in scope produced.
pub const CHECK_UNBOUND_HANDLE: &str = "dsl_kit::check::unbound_handle";

/// A required capability was not available at this point in the
/// document.
pub const CHECK_CAP_MISSING: &str = "dsl_kit::check::cap_missing";

/// Catch-all for a premise that failed without a more specific slug —
/// usable while a vocabulary is being sketched, but a shipped
/// `CheckProgram` should prefer a code that says what broke.
pub const CHECK_PREMISE_FAILED: &str = "dsl_kit::check::premise_failed";

// ---------------------------------------------------------------------------
// Program self-validation (`CheckProgram::validate`)
// ---------------------------------------------------------------------------
//
// These are reported against the *program*, not against a document, so
// they live in their own `program::` sub-namespace. Unlike the codes
// above they are assigned by the engine rather than chosen by the
// vocabulary author: a `CheckProgram` is the DSL author's own work and
// its findings cannot be suppressed at the document level, so a rule
// that can never hold has to surface before it blocks every document.

/// A rule requires a state nothing in the program can reach: no rule's
/// `state_after` and no fold declaration's initial state matches it.
/// Error — the rule can never fire, so every document containing its
/// variant is rejected with no escape hatch.
pub const CHECK_PROGRAM_UNDEFINED_STATE: &str = "dsl_kit::check::program::undefined_state";

/// A predicate the program produces (a conclusion, a state transition,
/// a fold's initial state) that no premise ever requires. Warning —
/// harmless on its own, but the usual cause is a misspelt predicate
/// name in one half of the pair.
pub const CHECK_PROGRAM_UNUSED_PRED: &str = "dsl_kit::check::program::unused_pred";

/// A rule that can never be reached: an earlier rule for the same
/// variant is unconditional, or carries exactly the same premises, and
/// therefore always wins. Warning.
pub const CHECK_PROGRAM_UNREACHABLE_RULE: &str = "dsl_kit::check::program::unreachable_rule";
