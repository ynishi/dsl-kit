//! Lint framework for `dsl-kit` DSLs.
//!
//! A [`Linter`] runs a set of [`Rule`]s over an AST that implements
//! [`Walk`] (from `dsl-kit-core`) and produces [`Diagnostic`]s. Rules
//! that need type-level structural information can additionally consult
//! [`DslSchema`] (from `dsl-kit-schema`).
//!
//! ## Split of responsibility
//!
//! - `dsl-kit-core::Walk` provides the *instance* traversal every rule
//!   needs (visit every node, count depth, count children).
//! - `dsl-kit-schema::DslSchema` provides the *type-level* shape a
//!   Schema-aware rule consults (which variants declare `Many` child
//!   fields, which fields are `Optional`, …).
//!
//! Rules that only need traversal are generic over `A: Walk`; rules
//! that consult the type-level shape add a `DslSchema` bound. See the
//! built-in rules for the traversal-only flavour, and DSL-specific
//! extension rules (in `flow-dsl`'s test module) for how DSL authors
//! plug custom rules into the same harness.
//!
//! ## Example
//!
//! ```ignore
//! use dsl_kit_lint::{Linter, Severity};
//!
//! let diagnostics = Linter::<Flow>::with_defaults().lint(&program);
//! for d in &diagnostics {
//!     println!("[{:?}] {}: {}", d.severity, d.rule, d.message);
//! }
//! ```
//!
//! ## Suppression
//!
//! There is deliberately no ignore-file / per-node suppression layer.
//! The levers, in order of preference:
//!
//! 1. **Declare the constraint in the schema** so the rule checks a
//!    declaration instead of guessing — e.g. `no-empty-child-slots`
//!    fires only on slots marked
//!    [`non_empty`](dsl_kit_schema::ChildSchema) (`#[dsl_schema(non_empty)]`).
//! 2. **Compose the rule set**: build with [`Linter::new`] +
//!    [`Linter::with_rule`] instead of [`Linter::with_defaults`], and
//!    tune per-rule parameters ([`MaxDepth::new`], [`MaxFanOut::new`]).
//! 3. **Filter the output**: every [`Diagnostic`] carries its
//!    `(rule, node, severity)` publicly, so a host that needs
//!    finer-grained suppression can drop entries from
//!    [`Linter::lint`]'s return value — e.g.
//!    `diags.retain(|d| d.rule != MaxFanOut::NAME || !allowed.contains(&d.node))`.
//!    Note that [`NodeId`]s are minted per parse, so such allow-lists
//!    are session-scoped, not persistable.

#![warn(missing_docs)]

use std::collections::{HashMap, HashSet};

use dsl_kit_core::{
    Applicability, ErrorCatalogEntry, FixSuggestion, NodeId, PatchPart, Phase, SuggesterHandle,
    Walk, noop_handle,
};
use dsl_kit_schema::DslSchema;

// ---------- Diagnostics -------------------------------------------------

/// Severity of a [`Diagnostic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Rule violation the DSL author almost certainly wants to fix.
    Error,
    /// Suspicious shape worth flagging but not necessarily wrong.
    Warn,
    /// Advisory observation.
    Info,
}

/// One lint result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// Name of the [`Rule`] that produced this diagnostic.
    pub rule: &'static str,
    /// Severity level.
    pub severity: Severity,
    /// Node the diagnostic is attached to.
    pub node: NodeId,
    /// Human-readable message.
    pub message: String,
    /// Optional structured, auto-apply-aware fix. `None` for rules that
    /// only report (the pre-suggestion default); `Some` when the rule
    /// can propose a concrete [`FixSuggestion`] whose
    /// [`Applicability`](dsl_kit_core::Applicability) tells a tool
    /// whether the patch may be applied automatically.
    pub suggestion: Option<FixSuggestion>,
}

/// Accumulator handed to each rule during a lint pass.
///
/// Rules call [`LintContext::emit`] or [`LintContext::report`] for
/// each diagnostic they raise; the [`Linter`] harvests them at the
/// end of the pass.
#[derive(Debug, Default)]
pub struct LintContext {
    diagnostics: Vec<Diagnostic>,
}

impl LintContext {
    /// Creates an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a fully-formed diagnostic.
    pub fn emit(&mut self, diag: Diagnostic) {
        self.diagnostics.push(diag);
    }

    /// Convenience helper: builds a suggestion-less `Diagnostic` and
    /// records it.
    pub fn report(
        &mut self,
        rule: &'static str,
        severity: Severity,
        node: NodeId,
        message: impl Into<String>,
    ) {
        self.emit(Diagnostic {
            rule,
            severity,
            node,
            message: message.into(),
            suggestion: None,
        });
    }

    /// Convenience helper: builds a `Diagnostic` carrying a structured
    /// [`FixSuggestion`] and records it. Use this over [`report`](Self::report)
    /// when the rule can propose a concrete patch.
    pub fn report_with_suggestion(
        &mut self,
        rule: &'static str,
        severity: Severity,
        node: NodeId,
        message: impl Into<String>,
        suggestion: FixSuggestion,
    ) {
        self.emit(Diagnostic {
            rule,
            severity,
            node,
            message: message.into(),
            suggestion: Some(suggestion),
        });
    }

    /// Borrows the accumulated diagnostics.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Consumes the context and returns the accumulated diagnostics.
    pub fn into_diagnostics(self) -> Vec<Diagnostic> {
        self.diagnostics
    }
}

// ---------- Rule trait --------------------------------------------------

/// A single lint rule.
///
/// Rules are stateless (or carry only their own configuration) and
/// receive the entire AST on each pass. Traversal is the rule's own
/// responsibility — most rules use `ast.walk(&mut |node, phase| ...)`
/// to visit every node.
pub trait Rule<A: Walk> {
    /// Stable rule identifier attached to every diagnostic it emits.
    fn name(&self) -> &'static str;

    /// Runs the rule against `ast`, appending diagnostics to `ctx`.
    fn check(&self, ast: &A, ctx: &mut LintContext);
}

// ---------- Linter orchestrator -----------------------------------------

/// Runs a collection of [`Rule`]s over an AST.
pub struct Linter<A: Walk> {
    rules: Vec<Box<dyn Rule<A>>>,
}

impl<A: Walk> Default for Linter<A> {
    fn default() -> Self {
        Self { rules: Vec::new() }
    }
}

impl<A: Walk> Linter<A> {
    /// Creates a linter with no rules registered.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an additional rule; returns the linter for chaining.
    #[must_use]
    pub fn with_rule<R: Rule<A> + 'static>(mut self, rule: R) -> Self {
        self.rules.push(Box::new(rule));
        self
    }

    /// Runs every registered rule against `ast` and returns the
    /// accumulated diagnostics, in the order rules were registered.
    pub fn lint(&self, ast: &A) -> Vec<Diagnostic> {
        let mut ctx = LintContext::new();
        for rule in &self.rules {
            rule.check(ast, &mut ctx);
        }
        ctx.into_diagnostics()
    }

    /// Number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

impl<A: Walk + DslSchema> Linter<A> {
    /// Registers the built-in rule set — [`UniqueNodeIds`],
    /// [`MaxDepth`] with the [`MaxDepth::DEFAULT_LIMIT`],
    /// [`MaxFanOut`] with the [`MaxFanOut::DEFAULT_LIMIT`],
    /// [`NoEmptyChildSlots`], and [`NoRedundantWrap`].
    ///
    /// [`DeadVariants`] is intentionally *not* included — perfectly
    /// correct programs routinely exercise only a subset of a DSL's
    /// variants, so a default `Info` flood would drown genuine
    /// `Error` / `Warn` findings. Register it explicitly when a
    /// review pass wants dead-code style diagnostics.
    pub fn with_defaults() -> Self {
        Self::new()
            .with_rule(UniqueNodeIds)
            .with_rule(MaxDepth::new(MaxDepth::DEFAULT_LIMIT))
            .with_rule(MaxFanOut::new(MaxFanOut::DEFAULT_LIMIT))
            .with_rule(NoEmptyChildSlots)
            .with_rule(NoRedundantWrap)
    }
}

// ---------- Built-in rules ----------------------------------------------

/// Errors on repeated [`NodeId`] values in the AST.
///
/// Every DSL author who uses `IdGen::node()` gets unique IDs for
/// free; this rule catches hand-constructed trees that reuse an ID
/// by mistake — a bug that also breaks Engine breakpoint targeting
/// and root-to-node path resolution.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniqueNodeIds;

impl UniqueNodeIds {
    /// Rule name (also attached to every diagnostic it emits).
    pub const NAME: &'static str = "unique-node-ids";
}

impl<A: Walk> Rule<A> for UniqueNodeIds {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self, ast: &A, ctx: &mut LintContext) {
        let mut seen: HashSet<NodeId> = HashSet::new();
        ast.walk(&mut |node, phase| {
            if phase != Phase::Pre {
                return;
            }
            let id = node.node_id();
            if !seen.insert(id) {
                ctx.report(
                    Self::NAME,
                    Severity::Error,
                    id,
                    format!("node id {id} appears more than once"),
                );
            }
        });
    }
}

/// Errors on any node whose depth exceeds a configured limit.
///
/// Depth is the number of ancestors between the node and the root; the
/// root itself is depth 0. Useful for catching pathological or
/// generator-produced ASTs before they hit the engine walker.
#[derive(Debug, Clone, Copy)]
pub struct MaxDepth {
    /// Maximum allowed depth (inclusive).
    pub limit: usize,
}

impl MaxDepth {
    /// Rule name (also attached to every diagnostic it emits).
    pub const NAME: &'static str = "max-depth";

    /// Default depth limit used by [`Linter::with_defaults`].
    pub const DEFAULT_LIMIT: usize = 64;

    /// Creates a rule with the given limit.
    pub fn new(limit: usize) -> Self {
        Self { limit }
    }
}

impl<A: Walk> Rule<A> for MaxDepth {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self, ast: &A, ctx: &mut LintContext) {
        let limit = self.limit;
        let mut depth: usize = 0;
        ast.walk(&mut |node, phase| match phase {
            Phase::Pre => {
                if depth > limit {
                    ctx.report(
                        Self::NAME,
                        Severity::Error,
                        node.node_id(),
                        format!("node depth {depth} exceeds limit {limit}"),
                    );
                }
                depth += 1;
            }
            Phase::Post => {
                depth = depth.saturating_sub(1);
            }
        });
    }
}

/// Warns on any node whose direct child count exceeds a configured
/// limit.
///
/// Fan-out complexity guard. Distinct from [`MaxDepth`], which
/// bounds nesting; this bounds width. A single flat `Seq` with 200
/// children clears any depth limit but is a maintenance smell.
#[derive(Debug, Clone, Copy)]
pub struct MaxFanOut {
    /// Maximum allowed direct-child count (inclusive).
    pub limit: usize,
}

impl MaxFanOut {
    /// Rule name (also attached to every diagnostic it emits).
    pub const NAME: &'static str = "max-fan-out";

    /// Default fan-out limit used by [`Linter::with_defaults`].
    pub const DEFAULT_LIMIT: usize = 32;

    /// Creates a rule with the given limit.
    pub fn new(limit: usize) -> Self {
        Self { limit }
    }
}

impl<A: Walk> Rule<A> for MaxFanOut {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self, ast: &A, ctx: &mut LintContext) {
        let limit = self.limit;
        ast.walk(&mut |node, phase| {
            if phase != Phase::Pre {
                return;
            }
            let count = node.children().len();
            if count > limit {
                ctx.report(
                    Self::NAME,
                    Severity::Warn,
                    node.node_id(),
                    format!("node has {count} direct children, above limit {limit}"),
                );
            }
        });
    }
}

/// Warns on a node whose variant's only child slots are *collection*
/// slots (`Many` / `Map`) yet which was built with no children at all.
///
/// Schema-driven: matches the runtime node against its
/// [`VariantSchema`](dsl_kit_schema::VariantSchema) via
/// [`dsl_kit_core::DslNode::variant_name`], inspects the multiplicity
/// of the variant's child slots, and fires on variants that carry ≥1
/// collection slot and no `One` slot. Variants with a `One` slot are
/// excluded because
/// [`check_conformance`](dsl_kit_parse::check_conformance) already
/// rejects those as an arity error, and `Optional`-only variants are
/// excluded because zero children is their documented shape.
///
/// # Why this is `Suspicious` / `Warn`, not `Correctness` / `Error`
///
/// A `Many` or `Map` slot means **zero or more** — that is what
/// `dsl-kit-schema` documents and what `check_conformance` accepts. So
/// an empty collection is not *wrong*; a great many DSLs have a
/// perfectly legal empty block, empty argument list, or override map
/// with nothing in it.
///
/// Since the schema can express the constraint
/// ([`ChildSchema::non_empty`](dsl_kit_schema::ChildSchema)), this
/// rule checks the **declared** constraint: a variant with at least
/// one `non_empty` collection slot, built with no children at all,
/// violates its own schema — a provable
/// [`LintCategory::Correctness`] finding reported at
/// [`Severity::Error`]. Collection slots that do *not* declare
/// `non_empty` keep their zero-or-more contract and are never
/// flagged, so a DSL where `{}` is a legal no-op block stays silent
/// without disabling anything.
///
/// History: until 0.5 the rule *inferred* non-emptiness from variant
/// shape ("only collection slots, therefore probably needs a child")
/// and reported `Error`; 0.5 downgraded the heuristic to
/// `Suspicious` / `Warn`; with the declaration available the
/// heuristic is retired entirely — the rule reports declared
/// violations only, and does not guess.
///
/// Precision note: [`Walk::children`] flattens every slot of a node
/// into one list, so on a variant mixing a `non_empty` slot with
/// other child slots this rule can only detect the all-slots-empty
/// case (a populated sibling slot masks the empty declared one).
/// The authoritative per-slot enforcement is
/// `dsl_kit_parse::check_conformance` (`ARITY_NON_EMPTY`), which
/// sees named slots; this rule is the typed-AST backstop for
/// hand-built values that never passed through a front-end.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoEmptyChildSlots;

// No alias for the pre-declaration behaviour is kept: the rule's
// *meaning* changed (heuristic Warn on inferred shapes → Error on
// declared `non_empty` violations only), so downstreams should meet
// the CHANGELOG rather than silently see different results.

impl NoEmptyChildSlots {
    /// Rule name (also attached to every diagnostic it emits).
    pub const NAME: &'static str = "no-empty-child-slots";
}

impl<A: Walk + DslSchema> Rule<A> for NoEmptyChildSlots {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self, ast: &A, ctx: &mut LintContext) {
        // Precompute: variant name → names of its declared non-empty
        // slots. Variants without a declaration never fire.
        let schema = A::schema();
        let declared: HashMap<String, Vec<String>> = schema
            .variants
            .iter()
            .filter_map(|v| {
                let slots: Vec<String> = v
                    .children
                    .iter()
                    .filter(|c| c.non_empty)
                    .map(|c| c.name.clone())
                    .collect();
                (!slots.is_empty()).then(|| (v.name.clone(), slots))
            })
            .collect();

        if declared.is_empty() {
            return;
        }

        ast.walk(&mut |node, phase| {
            if phase != Phase::Pre {
                return;
            }
            let name = node.variant_name();
            if let Some(slots) = declared.get(name) {
                if node.children().is_empty() {
                    ctx.report(
                        Self::NAME,
                        Severity::Error,
                        node.node_id(),
                        format!(
                            "{name} declares non-empty slot(s) {} but was built with no children",
                            slots
                                .iter()
                                .map(|s| format!("`{s}`"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    );
                }
            }
        });
    }
}

/// Warns when a variant wraps exactly one child of the *same*
/// variant with no other siblings — e.g. `Seq { children: [Seq {
/// children: [...] }] }`. The outer wrap adds no structure and can
/// be inlined into the inner one.
///
/// Uses [`dsl_kit_core::DslNode::variant_name`] to compare parent
/// and child variants; needs no schema info beyond that (the rule is
/// purely structural), but ships in the built-in Schema-aware
/// bucket for consistency with the other post-R-22 rules.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoRedundantWrap;

impl NoRedundantWrap {
    /// Rule name (also attached to every diagnostic it emits).
    pub const NAME: &'static str = "no-redundant-wrap";
}

impl<A: Walk> Rule<A> for NoRedundantWrap {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self, ast: &A, ctx: &mut LintContext) {
        ast.walk(&mut |node, phase| {
            if phase != Phase::Pre {
                return;
            }
            let kids = node.children();
            if kids.len() != 1 {
                return;
            }
            let parent = node.variant_name();
            let child = kids[0].variant_name();
            if parent == child {
                ctx.report(
                    Self::NAME,
                    Severity::Warn,
                    node.node_id(),
                    format!("{parent} wraps a single {child}; the outer wrap can be inlined"),
                );
            }
        });
    }
}

/// Reports every schema variant that never appears in the AST as an
/// `Info` diagnostic anchored to the root node.
///
/// Useful for review passes (dead-code style — which parts of the
/// DSL is this program actually exercising?) and for catching stale
/// schema entries that reference variants nothing in the codebase
/// constructs anymore.
///
/// Opt-in only — not registered by [`Linter::with_defaults`], since
/// perfectly correct programs routinely exercise only a subset of a
/// DSL's variants and flooding every default pass with `Info`
/// diagnostics would drown genuine `Error` / `Warn` findings.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeadVariants;

impl DeadVariants {
    /// Rule name (also attached to every diagnostic it emits).
    pub const NAME: &'static str = "dead-variants";
}

impl<A: Walk + DslSchema> Rule<A> for DeadVariants {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self, ast: &A, ctx: &mut LintContext) {
        let schema = A::schema();
        let declared: HashSet<String> = schema.variants.iter().map(|v| v.name.clone()).collect();
        let mut seen: HashSet<String> = HashSet::new();
        ast.walk(&mut |node, phase| {
            if phase != Phase::Pre {
                return;
            }
            seen.insert(node.variant_name().to_string());
        });
        let mut dead: Vec<String> = declared.difference(&seen).cloned().collect();
        dead.sort();
        for name in dead {
            ctx.report(
                Self::NAME,
                Severity::Info,
                ast.node_id(),
                format!("schema variant {name} is not present in the AST"),
            );
        }
    }
}

/// Type of the label-extraction closure used by [`TypoHint`].
///
/// Given an AST, yields `(node, label)` pairs identifying the
/// strings the rule should cross-check against the schema. Each
/// label is checked once; the rule does not walk on its own so DSL
/// authors decide what a "label" means for their DSL (a payload
/// field, an annotation, a source-level identifier they store on a
/// node, …).
pub type TypoLabelExtractor<A> = Box<dyn Fn(&A) -> Vec<(NodeId, String)> + Send + Sync>;

/// Opt-in rule that flags strings that look like a mistyped schema
/// variant name.
///
/// Given a caller-supplied extractor that yields `(node, label)`
/// pairs from the AST, the rule runs each `label` through an
/// injected [`Suggester`](dsl_kit_core::Suggester) against the target's schema variant names
/// and reports every near-miss as a `Severity::Info` diagnostic. An
/// exact match is never flagged.
///
/// The rule is dormant unless a suggester is injected via
/// [`Self::with_suggester`]: the default is a no-op that returns no
/// candidates, so simply registering the rule keeps the lint pass
/// zero-cost. Pass
/// `Arc::new(dsl_kit_fuzzy::FuzzySuggester::default())` at the
/// composition root to enable hints.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use dsl_kit_lint::{Linter, TypoHint};
/// use dsl_kit_fuzzy::FuzzySuggester;
///
/// let linter = Linter::<MyDsl>::with_defaults().with_rule(
///     TypoHint::new(|ast: &MyDsl| ast.collect_labels())
///         .with_suggester(Arc::new(FuzzySuggester::default())),
/// );
/// ```
pub struct TypoHint<A: Walk> {
    suggester: SuggesterHandle,
    extractor: TypoLabelExtractor<A>,
}

impl<A: Walk> TypoHint<A> {
    /// Rule name (also attached to every diagnostic it emits).
    pub const NAME: &'static str = "typo-hint";

    /// Builds a rule from a label extractor. Starts with a no-op
    /// suggester so the rule is dormant until
    /// [`Self::with_suggester`] is called.
    pub fn new<F>(extractor: F) -> Self
    where
        F: Fn(&A) -> Vec<(NodeId, String)> + Send + Sync + 'static,
    {
        Self {
            suggester: noop_handle(),
            extractor: Box::new(extractor),
        }
    }

    /// Injects the [`Suggester`](dsl_kit_core::Suggester) the rule uses to score labels
    /// against schema variant names.
    pub fn with_suggester(mut self, suggester: SuggesterHandle) -> Self {
        self.suggester = suggester;
        self
    }
}

impl<A: Walk + DslSchema> Rule<A> for TypoHint<A> {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self, ast: &A, ctx: &mut LintContext) {
        let schema = A::schema();
        let names: Vec<&str> = schema.variants.iter().map(|v| v.name.as_str()).collect();
        for (node, label) in (self.extractor)(ast) {
            // An exact match is not a typo; skip it before spending
            // a similarity computation on it.
            if names.iter().any(|n| *n == label) {
                continue;
            }
            if let Some(hint) = self.suggester.enrich_unknown(&label, &names) {
                let message = format!("`{label}` looks like a schema variant name; {hint}");
                // Build a concrete patch from the top-scored candidate.
                // Fuzzy matches are never certain, so the applicability
                // is MaybeIncorrect — surface it for confirmation, never
                // auto-apply.
                match self.suggester.suggest(&label, &names).first() {
                    Some(top) => {
                        let replacement = top.candidate;
                        let suggestion = FixSuggestion::new(
                            format!("replace `{label}` with `{replacement}`"),
                            PatchPart::node(node, replacement),
                            Applicability::MaybeIncorrect,
                        );
                        ctx.report_with_suggestion(
                            Self::NAME,
                            Severity::Info,
                            node,
                            message,
                            suggestion,
                        );
                    }
                    None => {
                        // A custom suggester returned a hint but no raw
                        // candidate; fall back to a suggestion-less diagnostic.
                        ctx.report(Self::NAME, Severity::Info, node, message);
                    }
                }
            }
        }
    }
}

// ---------- Lint declaration registry -----------------------------------

/// Category of a lint rule, mirroring Clippy's category axis but
/// narrowed to the five that carry meaning for dsl-kit ASTs.
///
/// Clippy ships nine categories; `perf` / `pedantic` / `restriction` /
/// `nursery` / `cargo` have no dsl-kit analogue, so they are omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintCategory {
    /// The AST is outright wrong — a bug the author almost certainly
    /// wants to fix (e.g. duplicate node ids).
    Correctness,
    /// A shape that is probably a mistake but is not provably wrong
    /// (e.g. a label that looks like a mistyped variant name).
    Suspicious,
    /// A stylistic preference with no correctness impact.
    Style,
    /// A complexity / maintainability smell (deep nesting, wide
    /// fan-out, redundant wrapping).
    Complexity,
    /// A violation of a caller-imposed structural contract (e.g. a
    /// configured depth ceiling).
    Contract,
}

/// Declarative metadata for one lint rule, decoupled from its
/// [`Rule`] implementation.
///
/// Following rustc's `declare_lint!` + `LintStore` split, the *what*
/// (name, stable code, category, default severity, description) lives
/// here as a `'static` record, while the *how* (the traversal that
/// detects violations) lives in the `Rule` impl. Collecting the
/// declarations alone is enough to drive a registry listing or an
/// `explain`-style catalogue without instantiating any rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LintDecl {
    /// Stable rule identifier, matching the rule's `NAME` const and the
    /// `rule` field of every [`Diagnostic`] it emits.
    pub name: &'static str,
    /// Stable, machine-readable diagnostic code in the
    /// `dsl_kit::lint::<snake_case_name>` form. Shares the
    /// `dsl_kit::` namespace with engine error codes
    /// (`dsl_kit::eval::aborted`, …) so a single `explain` catalogue
    /// can resolve both.
    pub code: &'static str,
    /// Category axis of the rule.
    pub category: LintCategory,
    /// Severity the built-in rule reports at.
    pub default_severity: Severity,
    /// One-line description of what the rule flags.
    pub desc: &'static str,
}

/// Central registry of every built-in lint rule's declaration.
///
/// Kept in lock-step with the `Rule` impls above by
/// `lint_decls_cover_all_builtin_rules` — adding a built-in rule
/// without a matching entry here (or vice versa) fails that test.
pub static LINT_DECLS: &[LintDecl] = &[
    LintDecl {
        name: UniqueNodeIds::NAME,
        code: "dsl_kit::lint::unique_node_ids",
        category: LintCategory::Correctness,
        default_severity: Severity::Error,
        desc: "flags AST nodes that reuse a NodeId, which breaks breakpoint targeting and path resolution",
    },
    LintDecl {
        name: MaxDepth::NAME,
        code: "dsl_kit::lint::max_depth",
        category: LintCategory::Contract,
        default_severity: Severity::Error,
        desc: "flags nodes nested deeper than a configured depth ceiling",
    },
    LintDecl {
        name: MaxFanOut::NAME,
        code: "dsl_kit::lint::max_fan_out",
        category: LintCategory::Complexity,
        default_severity: Severity::Warn,
        desc: "flags nodes whose direct-child count exceeds a configured fan-out limit",
    },
    LintDecl {
        name: NoEmptyChildSlots::NAME,
        code: "dsl_kit::lint::no_empty_child_slots",
        category: LintCategory::Correctness,
        default_severity: Severity::Error,
        desc: "flags a variant with a declared non-empty collection slot (ChildSchema::non_empty) that was built with no children",
    },
    LintDecl {
        name: NoRedundantWrap::NAME,
        code: "dsl_kit::lint::no_redundant_wrap",
        category: LintCategory::Complexity,
        default_severity: Severity::Warn,
        desc: "flags a variant that wraps a single child of the same variant, adding no structure",
    },
    LintDecl {
        name: DeadVariants::NAME,
        code: "dsl_kit::lint::dead_variants",
        category: LintCategory::Suspicious,
        default_severity: Severity::Info,
        desc: "reports schema variants that never appear in the AST (opt-in dead-code review)",
    },
    LintDecl {
        // `TypoHint::NAME` lives on an `A: Walk`-bounded impl, so it
        // can't be named in this non-generic const context. The literal
        // is kept in sync with `TypoHint::<_>::NAME` by
        // `lint_decls_cover_all_builtin_rules`.
        name: "typo-hint",
        code: "dsl_kit::lint::typo_hint",
        category: LintCategory::Suspicious,
        default_severity: Severity::Info,
        desc: "flags labels that look like a mistyped schema variant name (opt-in, suggester-driven)",
    },
];

/// Looks up a lint declaration by either its `name` (e.g.
/// `"typo-hint"`) or its stable `code` (e.g.
/// `"dsl_kit::lint::typo_hint"`). Returns `None` when neither matches.
pub fn lint_decl(name_or_code: &str) -> Option<&'static LintDecl> {
    LINT_DECLS
        .iter()
        .find(|d| d.name == name_or_code || d.code == name_or_code)
}

/// Projects [`LINT_DECLS`] into the shared
/// [`ErrorCatalogEntry`](dsl_kit_core::ErrorCatalogEntry) shape so a
/// diagnostic-`explain` surface can list and resolve lint codes in the
/// same catalogue as engine error codes. The help text folds the rule's
/// description together with its category and default severity.
pub fn lint_catalog() -> Vec<ErrorCatalogEntry> {
    LINT_DECLS
        .iter()
        .map(|d| ErrorCatalogEntry {
            code: d.code.to_string(),
            help: format!(
                "{} [lint: category={:?}, default severity={:?}]",
                d.desc, d.category, d.default_severity
            ),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsl_kit_core::{DslNode, NodeId};
    use dsl_kit_schema::Multiplicity;

    // Minimal test AST — a linear chain with configurable fan-out —
    // to exercise Walk + rules without pulling in flow-dsl. This
    // mirrors what #[derive(DslNode)] would produce for a tiny enum.
    #[derive(Debug)]
    enum Tiny {
        Node { id: NodeId, kids: Vec<Tiny> },
    }

    impl DslNode for Tiny {
        fn node_id(&self) -> NodeId {
            match self {
                Tiny::Node { id, .. } => *id,
            }
        }

        fn variant_name(&self) -> &'static str {
            match self {
                Tiny::Node { .. } => "Node",
            }
        }
    }

    impl Walk for Tiny {
        fn children(&self) -> Vec<&Self> {
            match self {
                Tiny::Node { kids, .. } => kids.iter().collect(),
            }
        }
    }

    // Minimal DslSchema for Tiny so we can exercise NoEmptyManyChildren.
    // Tiny has one variant with a Vec<Tiny> child field = many-only.
    impl DslSchema for Tiny {
        fn schema() -> dsl_kit_schema::NodeSchema {
            dsl_kit_schema::NodeSchema {
                name: "Tiny".into(),
                variants: vec![dsl_kit_schema::VariantSchema {
                    name: "Node".into(),
                    fields: vec![],
                    children: vec![dsl_kit_schema::ChildSchema {
                        name: "kids".into(),
                        multiplicity: Multiplicity::Many,
                        value_shape: dsl_kit_schema::ChildValueShape::Recursive,
                        scalar_shorthands: vec![],
                        non_empty: false,
                    }],
                }],
            }
        }
    }

    fn leaf(n: u64) -> Tiny {
        Tiny::Node {
            id: NodeId(n),
            kids: vec![],
        }
    }

    fn node(n: u64, kids: Vec<Tiny>) -> Tiny {
        Tiny::Node {
            id: NodeId(n),
            kids,
        }
    }

    // Declared counterpart of `Tiny`: same shape, but the schema marks
    // `kids` as `non_empty` — the declaration `no-empty-child-slots`
    // checks. `Tiny` itself stays undeclared so the silent-by-default
    // contract is exercised against the same structure.
    #[derive(Debug)]
    enum Decl {
        Node { id: NodeId, kids: Vec<Decl> },
    }

    impl DslNode for Decl {
        fn node_id(&self) -> NodeId {
            match self {
                Decl::Node { id, .. } => *id,
            }
        }

        fn variant_name(&self) -> &'static str {
            match self {
                Decl::Node { .. } => "Node",
            }
        }
    }

    impl Walk for Decl {
        fn children(&self) -> Vec<&Self> {
            match self {
                Decl::Node { kids, .. } => kids.iter().collect(),
            }
        }
    }

    impl DslSchema for Decl {
        fn schema() -> dsl_kit_schema::NodeSchema {
            dsl_kit_schema::NodeSchema {
                name: "Decl".into(),
                variants: vec![dsl_kit_schema::VariantSchema {
                    name: "Node".into(),
                    fields: vec![],
                    children: vec![
                        dsl_kit_schema::ChildSchema::recursive("kids", Multiplicity::Many)
                            .with_non_empty(),
                    ],
                }],
            }
        }
    }

    fn dleaf(n: u64) -> Decl {
        Decl::Node {
            id: NodeId(n),
            kids: vec![],
        }
    }

    fn dnode(n: u64, kids: Vec<Decl>) -> Decl {
        Decl::Node {
            id: NodeId(n),
            kids,
        }
    }

    #[test]
    fn unique_node_ids_passes_on_distinct_ids() {
        let ast = node(1, vec![leaf(2), leaf(3)]);
        let diags = Linter::<Tiny>::new().with_rule(UniqueNodeIds).lint(&ast);
        assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
    }

    #[test]
    fn unique_node_ids_fires_on_duplicate() {
        let dup = NodeId(42);
        let ast = Tiny::Node {
            id: dup,
            kids: vec![Tiny::Node {
                id: dup,
                kids: vec![],
            }],
        };
        let diags = Linter::<Tiny>::new().with_rule(UniqueNodeIds).lint(&ast);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].rule, UniqueNodeIds::NAME);
        assert_eq!(diags[0].severity, Severity::Error);
        assert_eq!(diags[0].node, dup);
        assert!(diags[0].message.contains("appears more than once"));
    }

    #[test]
    fn max_depth_passes_within_limit() {
        // depth 0 root → depth 1 leaf. Limit 4 is well within.
        let ast = node(1, vec![leaf(2)]);
        let diags = Linter::<Tiny>::new().with_rule(MaxDepth::new(4)).lint(&ast);
        assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
    }

    #[test]
    fn max_depth_fires_when_exceeded() {
        // Build a chain of nested nodes: depth 0, 1, 2, 3, 4.
        // Limit 2 → depths 3 and 4 fire.
        let ast = node(
            1,
            vec![node(2, vec![node(3, vec![node(4, vec![leaf(5)])])])],
        );
        let diags = Linter::<Tiny>::new().with_rule(MaxDepth::new(2)).lint(&ast);
        assert_eq!(diags.len(), 2, "diags = {diags:?}");
        assert!(diags.iter().all(|d| d.rule == MaxDepth::NAME));
        assert!(diags.iter().all(|d| d.severity == Severity::Error));
        assert_eq!(diags[0].node, NodeId(4));
        assert_eq!(diags[1].node, NodeId(5));
    }

    #[test]
    fn max_fan_out_passes_within_limit() {
        let ast = node(1, (2..6).map(leaf).collect());
        let diags = Linter::<Tiny>::new()
            .with_rule(MaxFanOut::new(8))
            .lint(&ast);
        assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
    }

    #[test]
    fn max_fan_out_fires_at_warn_severity_when_exceeded() {
        // 5 kids under root; limit 3 → 1 diagnostic on root.
        let ast = node(1, (2..7).map(leaf).collect());
        let diags = Linter::<Tiny>::new()
            .with_rule(MaxFanOut::new(3))
            .lint(&ast);
        assert_eq!(diags.len(), 1, "diags = {diags:?}");
        assert_eq!(diags[0].rule, MaxFanOut::NAME);
        assert_eq!(diags[0].severity, Severity::Warn);
        assert_eq!(diags[0].node, NodeId(1));
        assert!(diags[0].message.contains("5 direct children"));
    }

    #[test]
    fn linter_runs_multiple_rules_in_registration_order() {
        let dup = NodeId(9);
        let ast = Tiny::Node {
            id: dup,
            kids: vec![Tiny::Node {
                id: NodeId(10),
                kids: vec![Tiny::Node {
                    id: dup,
                    kids: vec![],
                }],
            }],
        };
        let diags = Linter::<Tiny>::new()
            .with_rule(UniqueNodeIds)
            .with_rule(MaxDepth::new(0))
            .lint(&ast);
        assert!(diags.iter().any(|d| d.rule == UniqueNodeIds::NAME));
        assert!(diags.iter().any(|d| d.rule == MaxDepth::NAME));
        // UniqueNodeIds registered first: its diagnostics come first.
        let first_unique = diags.iter().position(|d| d.rule == UniqueNodeIds::NAME);
        let first_depth = diags.iter().position(|d| d.rule == MaxDepth::NAME);
        assert!(first_unique < first_depth);
    }

    #[test]
    fn rule_count_reflects_registrations() {
        let l: Linter<Tiny> = Linter::new()
            .with_rule(UniqueNodeIds)
            .with_rule(MaxDepth::new(8));
        assert_eq!(l.rule_count(), 2);
    }

    #[test]
    fn tiny_variant_name_returns_literal_ident() {
        // Sanity check on the DslNode extension: hand-written impl
        // returns the source-level variant ident.
        let ast = leaf(1);
        assert_eq!(ast.variant_name(), "Node");
    }

    /// Without a `non_empty` declaration an empty collection slot is
    /// simply its documented zero-or-more shape — the rule stays
    /// silent instead of guessing from variant shape (the pre-0.9
    /// heuristic fired `Warn` on this exact tree).
    #[test]
    fn no_empty_child_slots_silent_without_declaration() {
        let ast = leaf(1);
        let diags = Linter::<Tiny>::new()
            .with_rule(NoEmptyChildSlots)
            .lint(&ast);
        assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
    }

    /// A declared `non_empty` slot built empty is a provable schema
    /// violation — `Error`, naming the offending slot.
    #[test]
    fn no_empty_child_slots_fires_on_declared_violation() {
        let ast = dleaf(1);
        let diags = Linter::<Decl>::new()
            .with_rule(NoEmptyChildSlots)
            .lint(&ast);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].rule, NoEmptyChildSlots::NAME);
        assert_eq!(diags[0].severity, Severity::Error);
        assert_eq!(diags[0].node, NodeId(1));
        assert!(diags[0].message.contains("`kids`"));
    }

    #[test]
    fn no_empty_child_slots_passes_when_children_present() {
        let ast = dnode(1, vec![dleaf(2)]);
        // Note: dleaf(2) itself violates the declaration → fires on
        // it. Only the populated root is clean.
        let diags = Linter::<Decl>::new()
            .with_rule(NoEmptyChildSlots)
            .lint(&ast);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].node, NodeId(2));
    }

    /// A keyed (`Map`) slot can declare `non_empty` like `Many` can,
    /// and a declared violation reports the same way.
    #[test]
    fn no_empty_child_slots_covers_keyed_slots() {
        struct Keyed;
        impl DslNode for Keyed {
            fn node_id(&self) -> NodeId {
                NodeId(7)
            }
            fn variant_name(&self) -> &'static str {
                "Env"
            }
        }
        impl Walk for Keyed {
            fn children(&self) -> Vec<&Self> {
                Vec::new()
            }
        }
        impl DslSchema for Keyed {
            fn schema() -> dsl_kit_schema::NodeSchema {
                dsl_kit_schema::NodeSchema {
                    name: "Cfg".into(),
                    variants: vec![dsl_kit_schema::VariantSchema {
                        name: "Env".into(),
                        fields: vec![],
                        children: vec![
                            dsl_kit_schema::ChildSchema::recursive("entries", Multiplicity::Map)
                                .with_non_empty(),
                        ],
                    }],
                }
            }
        }

        let diags = Linter::<Keyed>::new()
            .with_rule(NoEmptyChildSlots)
            .lint(&Keyed);
        assert_eq!(diags.len(), 1, "diags = {diags:?}");
        assert_eq!(diags[0].rule, NoEmptyChildSlots::NAME);
        assert_eq!(diags[0].severity, Severity::Error);
        assert_eq!(diags[0].node, NodeId(7));
    }

    #[test]
    fn no_redundant_wrap_fires_on_parent_wrapping_single_same_variant_child() {
        // node(1, [node(2, [leaf(3)])]) — outer has 1 child of same
        // "Node" variant → fires on id=1.
        let ast = node(1, vec![node(2, vec![leaf(3)])]);
        let diags = Linter::<Tiny>::new().with_rule(NoRedundantWrap).lint(&ast);
        // Tiny only has one variant so every 1-child node fires. Both
        // id=1 (children=[node(2,..)]) and id=2 (children=[leaf(3)])
        // wrap a single Node.
        assert_eq!(diags.len(), 2, "diags = {diags:?}");
        assert!(diags.iter().all(|d| d.rule == NoRedundantWrap::NAME));
        assert!(diags.iter().all(|d| d.severity == Severity::Warn));
    }

    #[test]
    fn no_redundant_wrap_passes_on_multi_child_parent() {
        // 2 children → not a single-wrap → no fire on root. Kids are
        // leaves (0 children) → no fire on kids either.
        let ast = node(1, vec![leaf(2), leaf(3)]);
        let diags = Linter::<Tiny>::new().with_rule(NoRedundantWrap).lint(&ast);
        assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
    }

    #[test]
    fn dead_variants_reports_schema_variants_not_in_ast() {
        // Tiny's schema has only "Node" → any ast covers all variants
        // → no dead variants → no diagnostics.
        let ast = leaf(1);
        let diags = Linter::<Tiny>::new().with_rule(DeadVariants).lint(&ast);
        assert!(diags.is_empty(), "diags = {diags:?}");
    }

    // A second toy AST with two variants — one always present, one
    // never used — so we can exercise DeadVariants' positive path.
    #[derive(Debug)]
    enum TwoVar {
        Alpha {
            id: NodeId,
        },
        #[allow(dead_code)]
        Beta {
            id: NodeId,
        },
    }

    impl DslNode for TwoVar {
        fn node_id(&self) -> NodeId {
            match self {
                TwoVar::Alpha { id } | TwoVar::Beta { id } => *id,
            }
        }
        fn variant_name(&self) -> &'static str {
            match self {
                TwoVar::Alpha { .. } => "Alpha",
                TwoVar::Beta { .. } => "Beta",
            }
        }
    }

    impl Walk for TwoVar {
        fn children(&self) -> Vec<&Self> {
            Vec::new()
        }
    }

    impl DslSchema for TwoVar {
        fn schema() -> dsl_kit_schema::NodeSchema {
            dsl_kit_schema::NodeSchema {
                name: "TwoVar".into(),
                variants: vec![
                    dsl_kit_schema::VariantSchema {
                        name: "Alpha".into(),
                        fields: vec![],
                        children: vec![],
                    },
                    dsl_kit_schema::VariantSchema {
                        name: "Beta".into(),
                        fields: vec![],
                        children: vec![],
                    },
                ],
            }
        }
    }

    #[test]
    fn dead_variants_fires_info_on_unused_variant() {
        let ast = TwoVar::Alpha { id: NodeId(1) };
        let diags = Linter::<TwoVar>::new().with_rule(DeadVariants).lint(&ast);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].rule, DeadVariants::NAME);
        assert_eq!(diags[0].severity, Severity::Info);
        assert_eq!(diags[0].node, NodeId(1)); // anchored to root
        assert!(diags[0].message.contains("Beta"));
    }

    /// Fixed-output suggester used in TypoHint tests to prove the
    /// wiring without depending on `dsl-kit-fuzzy` (a dev-dep would
    /// create a `core -> fuzzy -> core` cycle that cargo cannot
    /// unify at trait level).
    struct FixedSuggester(&'static str);
    impl dsl_kit_core::Suggester for FixedSuggester {
        fn suggest<'a>(&self, _q: &str, cands: &'a [&str]) -> Vec<dsl_kit_core::Suggestion<'a>> {
            cands
                .iter()
                .find(|c| **c == self.0)
                .map(|c| dsl_kit_core::Suggestion {
                    candidate: c,
                    score: 1.0,
                })
                .into_iter()
                .collect()
        }
    }

    #[test]
    fn typo_hint_dormant_without_suggester() {
        // Rule is registered but no suggester injected: NoopSuggester
        // returns no candidates, so nothing fires even for near-miss
        // labels.
        let ast = TwoVar::Alpha { id: NodeId(1) };
        let rule = TypoHint::new(|_ast: &TwoVar| vec![(NodeId(1), "Alph".to_string())]);
        let diags = Linter::<TwoVar>::new().with_rule(rule).lint(&ast);
        assert!(diags.is_empty(), "expected zero diagnostics, got {diags:?}");
    }

    #[test]
    fn typo_hint_fires_with_suggester_on_near_miss() {
        use std::sync::Arc;
        let ast = TwoVar::Alpha { id: NodeId(1) };
        let rule = TypoHint::new(|_ast: &TwoVar| vec![(NodeId(1), "Alph".to_string())])
            .with_suggester(Arc::new(FixedSuggester("Alpha")));
        let diags = Linter::<TwoVar>::new().with_rule(rule).lint(&ast);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].rule, TypoHint::<TwoVar>::NAME);
        assert_eq!(diags[0].severity, Severity::Info);
        assert_eq!(diags[0].node, NodeId(1));
        assert!(
            diags[0].message.contains("Alph")
                && diags[0].message.contains("did you mean")
                && diags[0].message.contains("Alpha"),
            "unexpected message: {}",
            diags[0].message
        );
    }

    #[test]
    fn typo_hint_attaches_maybe_incorrect_fix_suggestion() {
        use std::sync::Arc;
        let ast = TwoVar::Alpha { id: NodeId(1) };
        let rule = TypoHint::new(|_ast: &TwoVar| vec![(NodeId(1), "Alph".to_string())])
            .with_suggester(Arc::new(FixedSuggester("Alpha")));
        let diags = Linter::<TwoVar>::new().with_rule(rule).lint(&ast);
        assert_eq!(diags.len(), 1);
        let suggestion = diags[0]
            .suggestion
            .as_ref()
            .expect("TypoHint must attach a FixSuggestion on a near-miss");
        assert_eq!(suggestion.applicability, Applicability::MaybeIncorrect);
        assert_eq!(suggestion.patch.len(), 1);
        assert_eq!(suggestion.patch[0].node, NodeId(1));
        assert_eq!(suggestion.patch[0].path, None);
        assert_eq!(suggestion.patch[0].replacement, "Alpha");
        assert!(
            suggestion.message.contains("Alph") && suggestion.message.contains("Alpha"),
            "unexpected suggestion message: {}",
            suggestion.message
        );
    }

    #[test]
    fn lint_decls_cover_all_builtin_rules() {
        // The full set of built-in rule NAMEs. Adding a rule without
        // extending both this list and LINT_DECLS fails the len check.
        let rule_names = [
            UniqueNodeIds::NAME,
            MaxDepth::NAME,
            MaxFanOut::NAME,
            NoEmptyChildSlots::NAME,
            NoRedundantWrap::NAME,
            DeadVariants::NAME,
            TypoHint::<Tiny>::NAME,
        ];

        // Every built-in rule has a declaration, resolvable by name.
        for name in rule_names {
            let decl = lint_decl(name)
                .unwrap_or_else(|| panic!("no LintDecl registered for rule `{name}`"));
            assert_eq!(decl.name, name);
            // And by its own code.
            assert_eq!(lint_decl(decl.code).map(|d| d.name), Some(name));
        }

        // Every declaration points at a real built-in rule.
        for decl in LINT_DECLS {
            assert!(
                rule_names.contains(&decl.name),
                "LINT_DECLS entry `{}` does not match any built-in rule NAME",
                decl.name
            );
            assert!(
                decl.code.starts_with("dsl_kit::lint::"),
                "lint code `{}` must live in the dsl_kit::lint:: namespace",
                decl.code
            );
        }

        // No stray / missing entries.
        assert_eq!(LINT_DECLS.len(), rule_names.len());
    }

    #[test]
    fn lint_catalog_folds_category_and_severity() {
        let catalog = lint_catalog();
        assert_eq!(catalog.len(), LINT_DECLS.len());
        let typo = catalog
            .iter()
            .find(|e| e.code == "dsl_kit::lint::typo_hint")
            .expect("typo_hint lint code in catalog");
        assert!(typo.help.contains("category=Suspicious"));
        assert!(typo.help.contains("default severity=Info"));
    }

    #[test]
    fn report_leaves_suggestion_none_by_default() {
        // Rules that only call `report` must not carry a suggestion.
        let ast = dleaf(1);
        let diags = Linter::<Decl>::new()
            .with_rule(NoEmptyChildSlots)
            .lint(&ast);
        assert_eq!(diags.len(), 1);
        assert!(diags[0].suggestion.is_none());
    }

    #[test]
    fn typo_hint_skips_exact_match() {
        // An exact match against a schema variant name is not a typo
        // — the rule must not fire even with a suggester injected.
        use std::sync::Arc;
        let ast = TwoVar::Alpha { id: NodeId(1) };
        let rule = TypoHint::new(|_ast: &TwoVar| vec![(NodeId(1), "Alpha".to_string())])
            .with_suggester(Arc::new(FixedSuggester("Alpha")));
        let diags = Linter::<TwoVar>::new().with_rule(rule).lint(&ast);
        assert!(diags.is_empty(), "expected zero diagnostics, got {diags:?}");
    }

    #[test]
    fn with_defaults_includes_no_redundant_wrap_but_not_dead_variants() {
        // Redundant wrap: node(1, [leaf(2)]) — but leaf(2) is empty
        // Node → also fires NoEmptyManyChildren (many-only). To
        // isolate: use populated inner.
        let ast = node(1, vec![node(2, vec![leaf(3)])]);
        let diags = Linter::<Tiny>::with_defaults().lint(&ast);
        // Should include no-redundant-wrap warnings but no dead-variants
        // (Tiny has one variant so DeadVariants would find nothing
        // dead anyway; the absence assertion protects intent).
        assert!(
            diags.iter().any(|d| d.rule == NoRedundantWrap::NAME),
            "expected no-redundant-wrap in defaults, diags = {diags:?}"
        );
        assert!(
            !diags.iter().any(|d| d.rule == DeadVariants::NAME),
            "dead-variants should be opt-in only, diags = {diags:?}"
        );
    }
}
