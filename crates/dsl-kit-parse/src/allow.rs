//! Usage-site lint suppression: the reserved `$allow` annotation and
//! the diagnostics it can produce.
//!
//! A lint rule fires at a node. Sometimes the author of that node knows
//! better than the rule, and the only place that knowledge lives is the
//! node itself — not a project-wide configuration file, which would
//! switch the rule off everywhere and lose the reason. So a document
//! may annotate a node with the rule names it accepts there:
//!
//! ```json
//! { "type": "Par", "$allow": ["max-fan-out"], "branches": [ … ] }
//! ```
//!
//! ## Reserved key
//!
//! The annotation is spelled [`ALLOW_KEY`] (`"$allow"`). The `$` sigil
//! cannot appear in a Rust identifier, so the key can never collide
//! with a field or child slot a DSL author declares — the same
//! reservation that lets [`crate::import::IMPORT_VARIANT`] spell
//! `$import` without a namespace. Keys are dispatched against the
//! schema by name, so a reserved key is recognised before schema
//! dispatch and never reaches the unknown-key path.
//!
//! One interaction with `$import` is worth naming: an object carrying
//! both `$import` and `$allow` is rejected as
//! [`crate::import::import_codes::SPEC_SHAPE`], because an import
//! placeholder must be a single-key object. Annotate the imported
//! source's own nodes instead.
//!
//! ## Shape
//!
//! The value must be an array of rule-name strings. An empty array is
//! accepted and means the same as no annotation at all. Anything else
//! — a bare string, a number, an array with a non-string element — is
//! a [`codes::ALLOW_SHAPE`] diagnostic rather than a silently ignored
//! key, because a mis-spelled suppression that quietly does nothing is
//! worse than one that fails loudly.
//!
//! ## Where the names go
//!
//! The front-end stores them verbatim on [`ParseTree::allows`]; the
//! names are not validated against a rule registry here, since the
//! parse trunk does not know which rules exist. `#[derive(DslBuild)]`
//! carries them to [`dsl_kit_core::AllowTable`], keyed on the
//! [`NodeId`](dsl_kit_core::NodeId) minted for the annotated node,
//! and a linter resolves them there.
//!
//! [`ParseTree::allows`]: crate::ParseTree::allows

/// Reserved object key carrying a node's usage-site lint suppressions
/// in the JSON front-end.
pub const ALLOW_KEY: &str = "$allow";

/// Diagnostic codes emitted by the `$allow` annotation machinery.
pub mod codes {
    /// The [`ALLOW_KEY`](super::ALLOW_KEY) value is not an array of
    /// rule-name strings.
    pub const ALLOW_SHAPE: &str = "dsl_kit::parse::allow::allow_shape";
}
