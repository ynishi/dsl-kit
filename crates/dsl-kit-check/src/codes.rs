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
