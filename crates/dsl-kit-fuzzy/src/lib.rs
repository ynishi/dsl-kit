//! Fuzzy-match [`Suggester`](dsl_kit_core::Suggester) implementation
//! for dsl-kit.
//!
//! This is the plugin half of the split described in
//! `dsl-kit-core::suggest`: the core crate owns the [`Suggester`]
//! contract and a [`NoopSuggester`](dsl_kit_core::NoopSuggester)
//! default, and this crate ships an implementation that scores
//! candidates with a string-similarity algorithm from the `strsim`
//! crate.
//!
//! # Choosing an algorithm
//!
//! - [`Algorithm::JaroWinkler`] (default): weights matching prefixes,
//!   tends to score short typos in identifiers well
//!   (e.g. `AddDeriv` → `AddDerive`).
//! - [`Algorithm::Levenshtein`]: raw edit distance, normalised into
//!   `0.0..=1.0` by the longest of the two strings.
//! - [`Algorithm::DamerauLevenshtein`]: like Levenshtein but treats a
//!   transposition of two adjacent characters as a single edit
//!   (`taget` → `target`).
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use dsl_kit_core::{Suggester, SuggesterHandle};
//! use dsl_kit_fuzzy::FuzzySuggester;
//!
//! let s: SuggesterHandle = Arc::new(FuzzySuggester::default());
//! let hint = s.enrich_unknown("AddDeriv", &["AddDerive", "AddImport", "Remove"]);
//! assert_eq!(hint.as_deref(), Some("did you mean: AddDerive"));
//! ```

#![warn(missing_docs)]

use dsl_kit_core::{Suggester, Suggestion};

/// String-similarity algorithm used by [`FuzzySuggester`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    /// Jaro-Winkler similarity. Default; biases toward matching prefixes.
    #[default]
    JaroWinkler,
    /// Levenshtein edit distance, normalised into `0.0..=1.0` by the
    /// length of the longer input.
    Levenshtein,
    /// Damerau-Levenshtein edit distance (Levenshtein + adjacent
    /// transposition), normalised the same way as [`Self::Levenshtein`].
    DamerauLevenshtein,
}

/// Configuration for [`FuzzySuggester`].
#[derive(Debug, Clone, Copy)]
pub struct SuggestOptions {
    /// Algorithm used to score candidate similarity.
    pub algorithm: Algorithm,
    /// Minimum similarity in `0.0..=1.0`; candidates scoring below
    /// this are omitted. Default `0.7`.
    pub threshold: f64,
    /// Maximum number of candidates returned by
    /// [`Suggester::suggest`]. Default `3`.
    pub max_results: usize,
}

impl Default for SuggestOptions {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::default(),
            threshold: 0.7,
            max_results: 3,
        }
    }
}

/// A [`Suggester`] implementation backed by [`strsim`].
///
/// See the crate-level docs for algorithm trade-offs and a usage
/// example.
#[derive(Debug, Clone, Copy, Default)]
pub struct FuzzySuggester {
    opts: SuggestOptions,
}

impl FuzzySuggester {
    /// Build a suggester with explicit options.
    pub fn new(opts: SuggestOptions) -> Self {
        Self { opts }
    }

    /// Override just the algorithm, leaving other options at their
    /// defaults.
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.opts.algorithm = algorithm;
        self
    }

    /// Override just the threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.opts.threshold = threshold;
        self
    }

    /// Override just the max result count.
    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.opts.max_results = max_results;
        self
    }

    fn score(&self, a: &str, b: &str) -> f64 {
        match self.opts.algorithm {
            Algorithm::JaroWinkler => strsim::jaro_winkler(a, b),
            Algorithm::Levenshtein => normalise_edit(strsim::levenshtein(a, b), a, b),
            Algorithm::DamerauLevenshtein => {
                normalise_edit(strsim::damerau_levenshtein(a, b), a, b)
            }
        }
    }
}

fn normalise_edit(distance: usize, a: &str, b: &str) -> f64 {
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    1.0 - (distance as f64 / max_len as f64)
}

impl Suggester for FuzzySuggester {
    fn suggest<'a>(&self, query: &str, candidates: &'a [&str]) -> Vec<Suggestion<'a>> {
        let mut scored: Vec<Suggestion<'a>> = candidates
            .iter()
            .map(|c| Suggestion {
                candidate: c,
                score: self.score(query, c),
            })
            .filter(|s| s.score >= self.opts.threshold)
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.opts.max_results);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_deriv_suggests_add_derive() {
        let s = FuzzySuggester::default();
        let out = s.suggest("AddDeriv", &["AddDerive", "AddImport", "Remove"]);
        assert!(!out.is_empty(), "expected at least one suggestion");
        assert_eq!(out[0].candidate, "AddDerive");
        assert!(out[0].score >= 0.7);
    }

    #[test]
    fn taget_suggests_target_via_damerau() {
        let s = FuzzySuggester::default().with_algorithm(Algorithm::DamerauLevenshtein);
        let out = s.suggest("taget", &["target", "tangent", "budget"]);
        assert_eq!(out[0].candidate, "target");
    }

    #[test]
    fn threshold_filters_far_candidates() {
        let s = FuzzySuggester::default().with_threshold(0.9);
        let out = s.suggest("foo", &["zzzz", "yyyy"]);
        assert!(out.is_empty());
    }

    #[test]
    fn max_results_caps_output() {
        let s = FuzzySuggester::default()
            .with_threshold(0.0)
            .with_max_results(2);
        let out = s.suggest("aa", &["ab", "ac", "ad", "ae"]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn enrich_unknown_uses_default_formatter() {
        let s = FuzzySuggester::default();
        let hint = s.enrich_unknown("AddDeriv", &["AddDerive", "AddImport", "Remove"]);
        assert_eq!(hint.as_deref(), Some("did you mean: AddDerive"));
    }

    #[test]
    fn empty_query_and_empty_candidates_yield_none() {
        let s = FuzzySuggester::default();
        assert!(s.suggest("", &[]).is_empty());
        assert_eq!(s.enrich_unknown("x", &[]), None);
    }
}
