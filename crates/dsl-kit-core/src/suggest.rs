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
}
