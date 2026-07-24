//! Fuzzy-match suggestion contract shared across dsl-kit crates.
//!
//! Downstream crates (`dsl-kit-parse`, `dsl-kit-mcp`, `dsl-kit-lint`,
//! `dsl-kit-core` itself) hold an `Arc<dyn Suggester>` and enrich their
//! `unknown-*` diagnostics with `did you mean X?` hints. The actual
//! similarity algorithm lives in a plugin crate (e.g. `dsl-kit-fuzzy`);
//! this crate only owns the contract and a no-op default so consumers
//! that do not want the extra dependency stay zero-cost.
//!
//! The trait is intentionally string-only (`&[&str]`): callers extract
//! the valid set from their own compile-time schema (variant names,
//! field names, registered op ids, …) and pass it as a slice. This
//! keeps the trait leaf-level with no upward dependencies on
//! `NodeSchema`, `OpRegistry`, or any other dsl-kit type, so a plugin
//! implementation can live in a crate that itself depends on
//! `dsl-kit-core`.
//!
//! # Suggest vs. apply
//!
//! Implementations MUST return candidates only. They must not mutate
//! the caller's input or return an "auto-repaired" value: the decision
//! to substitute a suggested candidate belongs to the caller (human
//! reviewer, MCP client, IDE, …), never to the suggester.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::NodeId;

/// One candidate returned by a [`Suggester`].
///
/// `score` is normalised to `0.0..=1.0` where `1.0` is an exact match.
/// The exact scale depends on the underlying algorithm; callers should
/// treat scores as opaque and only compare them within a single
/// `Suggester`'s output.
#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion<'a> {
    /// The candidate string, borrowed from the `candidates` slice
    /// passed into [`Suggester::suggest`].
    pub candidate: &'a str,
    /// Similarity score in `0.0..=1.0`. Higher is more similar.
    pub score: f64,
}

/// Contract for fuzzy-match suggesters.
///
/// Implementations decide the similarity algorithm (Jaro-Winkler,
/// Levenshtein, Damerau-Levenshtein, …), the score threshold below
/// which candidates are suppressed, and the maximum number of results
/// returned.
///
/// A no-op implementation ([`NoopSuggester`]) is provided so consumer
/// crates can default to "no suggestions" without pulling in a
/// similarity algorithm.
pub trait Suggester: Send + Sync {
    /// Return candidates from `candidates` that are similar to `query`,
    /// ordered by descending similarity. An empty `Vec` means "no
    /// candidate is close enough to suggest".
    ///
    /// The returned [`Suggestion`]s borrow from `candidates`, so the
    /// slice must outlive the returned vector.
    fn suggest<'a>(&self, query: &str, candidates: &'a [&str]) -> Vec<Suggestion<'a>>;

    /// Format the suggestions as a `did you mean: X, Y` hint suitable
    /// for appending to a diagnostic message. Returns `None` when
    /// [`suggest`](Self::suggest) yields no candidates.
    ///
    /// The default implementation joins up to three top candidates
    /// with commas; override to change the punctuation or wording.
    fn enrich_unknown(&self, query: &str, candidates: &[&str]) -> Option<String> {
        let sugs = self.suggest(query, candidates);
        if sugs.is_empty() {
            return None;
        }
        let joined = sugs
            .iter()
            .take(3)
            .map(|s| s.candidate)
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!("did you mean: {joined}"))
    }
}

/// A [`Suggester`] that always returns no candidates.
///
/// Consumer crates use this as the default so that pulling in a real
/// suggester (e.g. `dsl-kit-fuzzy`) stays an opt-in dependency at the
/// composition root.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopSuggester;

impl Suggester for NoopSuggester {
    fn suggest<'a>(&self, _query: &str, _candidates: &'a [&str]) -> Vec<Suggestion<'a>> {
        Vec::new()
    }

    fn enrich_unknown(&self, _query: &str, _candidates: &[&str]) -> Option<String> {
        None
    }
}

/// Convenience alias for the shared trait object callers store in
/// their context (`Parser`, `Engine`, `DslMcpBuilder`, …).
pub type SuggesterHandle = Arc<dyn Suggester>;

/// Return a shared `NoopSuggester` handle. Cheap: the returned `Arc`
/// wraps a zero-sized type.
pub fn noop_handle() -> SuggesterHandle {
    Arc::new(NoopSuggester)
}

// ---------- Structured fix suggestion ----------------------------------

/// How much confidence a tool may place in auto-applying a
/// [`FixSuggestion`]'s patch.
///
/// Modelled on rustc / Clippy's `Applicability`, but deliberately
/// closed at three levels — there is **no** `Unspecified` escape hatch.
/// rustc keeps one, but "confidence unknown" is exactly the state that
/// lets mis-tagged suggestions get auto-applied by accident, so the kit
/// forces every producer to pick a real level.
///
/// # Auto-apply discipline
///
/// Only [`MachineApplicable`](Self::MachineApplicable) suggestions may
/// be applied automatically (this mirrors the rustc-dev-guide rule that
/// "only `MachineApplicable` suggestions are automatically applied by
/// rustfix"). [`MaybeIncorrect`](Self::MaybeIncorrect) and
/// [`HasPlaceholders`](Self::HasPlaceholders) suggestions must be
/// surfaced for a human / agent to confirm before the patch is written
/// back — never applied silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Applicability {
    /// The patch is correct and complete; a tool may apply it without
    /// human review. This is the only level eligible for auto-apply.
    MachineApplicable,
    /// The patch is a plausible fix but may be wrong (e.g. a fuzzy
    /// match against a set of candidates). Present it for confirmation;
    /// do not apply it automatically.
    MaybeIncorrect,
    /// The patch contains placeholder text the caller must fill in
    /// before it compiles / parses (e.g. `<expr>`). Never auto-apply.
    HasPlaceholders,
}

/// One span-anchored edit within a [`FixSuggestion`].
///
/// A suggestion's patch is always a `Vec<PatchPart>` (multipart from
/// the start, mirroring rustc's `multipart_suggestion`) so a fix that
/// spans several nodes never has to be retrofitted onto a single-span
/// assumption.
///
/// dsl-kit anchors edits on [`NodeId`] rather than a byte span: a lint
/// [`Diagnostic`](../../dsl_kit_lint/struct.Diagnostic.html) already
/// identifies the site it fires on by node, and `path` narrows the edit
/// to a sub-field of that node when the DSL needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchPart {
    /// Node the edit is anchored to.
    pub node: NodeId,
    /// Optional path to a sub-field of `node` the edit targets (e.g.
    /// `"label"`). `None` means the edit applies to the node as a
    /// whole; the exact meaning is DSL-defined.
    pub path: Option<String>,
    /// Replacement text to write at the target.
    pub replacement: String,
}

impl PatchPart {
    /// Builds a whole-node patch (`path = None`).
    pub fn node(node: NodeId, replacement: impl Into<String>) -> Self {
        Self {
            node,
            path: None,
            replacement: replacement.into(),
        }
    }

    /// Builds a patch targeting a named sub-field of `node`.
    pub fn field(node: NodeId, path: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self {
            node,
            path: Some(path.into()),
            replacement: replacement.into(),
        }
    }
}

/// A structured, auto-apply-aware fix for a diagnostic.
///
/// This is the Clippy-style layer that sits *on top of* the string-only
/// [`Suggester`] contract: a `Suggester` enumerates candidate strings,
/// and a producer (a lint rule, an MCP tool) turns a chosen candidate
/// into a [`FixSuggestion`] carrying the concrete [`patch`](Self::patch)
/// and an [`Applicability`] gate. Enumerating candidates and deciding
/// to apply one stay separate responsibilities — the suggester never
/// mutates anything.
///
/// The value is immutable once built: construct it with [`Self::new`]
/// (plus [`Self::with_part`] for extra edits) and pass it around
/// unchanged, sidestepping the toggle-style lifecycle rustc's
/// `Suggestions` enum carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixSuggestion {
    /// Human-readable one-line description of the fix.
    pub message: String,
    /// The edit(s) that make up the fix. Always multipart.
    pub patch: Vec<PatchPart>,
    /// How confidently a tool may apply [`patch`](Self::patch).
    pub applicability: Applicability,
}

impl FixSuggestion {
    /// Builds a suggestion from a message, a single patch part, and an
    /// applicability level. Use [`with_part`](Self::with_part) to add
    /// further edits for a multipart fix.
    pub fn new(message: impl Into<String>, part: PatchPart, applicability: Applicability) -> Self {
        Self {
            message: message.into(),
            patch: vec![part],
            applicability,
        }
    }

    /// Appends another [`PatchPart`] and returns the suggestion, for
    /// building a multipart fix fluently.
    #[must_use]
    pub fn with_part(mut self, part: PatchPart) -> Self {
        self.patch.push(part);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_returns_empty() {
        let s = NoopSuggester;
        assert!(s.suggest("foo", &["food", "for"]).is_empty());
        assert_eq!(s.enrich_unknown("foo", &["food", "for"]), None);
    }

    #[test]
    fn enrich_default_formats_top_three() {
        struct FixedSuggester;
        impl Suggester for FixedSuggester {
            fn suggest<'a>(&self, _q: &str, cands: &'a [&str]) -> Vec<Suggestion<'a>> {
                cands
                    .iter()
                    .enumerate()
                    .map(|(i, c)| Suggestion {
                        candidate: c,
                        score: 1.0 - (i as f64 * 0.1),
                    })
                    .collect()
            }
        }
        let s = FixedSuggester;
        let hint = s.enrich_unknown("q", &["a", "b", "c", "d", "e"]);
        assert_eq!(hint.as_deref(), Some("did you mean: a, b, c"));
    }

    #[test]
    fn noop_handle_is_shared() {
        let h: SuggesterHandle = noop_handle();
        assert!(h.suggest("x", &["y"]).is_empty());
    }

    #[test]
    fn applicability_serde_round_trips() {
        for level in [
            Applicability::MachineApplicable,
            Applicability::MaybeIncorrect,
            Applicability::HasPlaceholders,
        ] {
            let json = serde_json::to_string(&level).expect("serialize Applicability");
            let back: Applicability =
                serde_json::from_str(&json).expect("deserialize Applicability");
            assert_eq!(level, back, "round-trip changed {level:?} via {json}");
        }
        // The wire form is the variant name verbatim (external tagging).
        assert_eq!(
            serde_json::to_string(&Applicability::MachineApplicable).unwrap(),
            "\"MachineApplicable\""
        );
    }

    #[test]
    fn fix_suggestion_builds_multipart() {
        let s = FixSuggestion::new(
            "replace `Alph` with `Alpha`",
            PatchPart::node(NodeId(1), "Alpha"),
            Applicability::MaybeIncorrect,
        )
        .with_part(PatchPart::field(NodeId(2), "label", "Beta"));
        assert_eq!(s.patch.len(), 2);
        assert_eq!(s.applicability, Applicability::MaybeIncorrect);
        assert_eq!(s.patch[0].node, NodeId(1));
        assert_eq!(s.patch[0].path, None);
        assert_eq!(s.patch[1].path.as_deref(), Some("label"));
        assert_eq!(s.patch[1].replacement, "Beta");
    }
}
