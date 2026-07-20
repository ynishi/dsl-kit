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

#![warn(missing_docs)]

use std::collections::HashSet;

use dsl_kit_core::{NodeId, Phase, Walk};
use dsl_kit_schema::{DslSchema, Multiplicity};

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

    /// Convenience helper: builds a `Diagnostic` and records it.
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
    /// [`NoEmptyManyChildren`], and [`NoRedundantWrap`].
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
            .with_rule(NoEmptyManyChildren)
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

/// Errors on a node whose variant declares at least one `Many` child
/// field, no `One` child field, and is instantiated with zero direct
/// children.
///
/// Schema-driven: matches the runtime node against its
/// [`VariantSchema`](dsl_kit_schema::VariantSchema) via
/// [`dsl_kit_core::DslNode::variant_name`], inspects the multiplicity of the
/// variant's child fields, and only fires on variants whose child
/// shape guarantees at least one child (i.e. the variant has ≥1
/// `Many` child field and no `One` field — `Optional` alone would
/// permit zero children legitimately, so those are excluded).
///
/// This is the precise version enabled by R-22's
/// [`dsl_kit_core::DslNode::variant_name`] addition; a coarse `variant_name`-less
/// implementation was deferred in R-21 to avoid mis-flagging genuine
/// leaf variants.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoEmptyManyChildren;

impl NoEmptyManyChildren {
    /// Rule name (also attached to every diagnostic it emits).
    pub const NAME: &'static str = "no-empty-many-children";
}

impl<A: Walk + DslSchema> Rule<A> for NoEmptyManyChildren {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self, ast: &A, ctx: &mut LintContext) {
        // Precompute: which variant names carry the "≥1 Many field,
        // no One field" shape.
        let schema = A::schema();
        let many_only: HashSet<String> = schema
            .variants
            .iter()
            .filter(|v| {
                let has_many = v
                    .children
                    .iter()
                    .any(|c| c.multiplicity == Multiplicity::Many);
                let has_one = v
                    .children
                    .iter()
                    .any(|c| c.multiplicity == Multiplicity::One);
                has_many && !has_one
            })
            .map(|v| v.name.clone())
            .collect();

        if many_only.is_empty() {
            return;
        }

        ast.walk(&mut |node, phase| {
            if phase != Phase::Pre {
                return;
            }
            let name = node.variant_name();
            if many_only.contains(name) && node.children().is_empty() {
                ctx.report(
                    Self::NAME,
                    Severity::Error,
                    node.node_id(),
                    format!("{name} requires at least one child but has none"),
                );
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
                    format!(
                        "{parent} wraps a single {child}; the outer wrap can be inlined"
                    ),
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
        let declared: HashSet<String> =
            schema.variants.iter().map(|v| v.name.clone()).collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use dsl_kit_core::{DslNode, NodeId};

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
                    }],
                }],
            }
        }
    }

    fn leaf(n: u64) -> Tiny {
        Tiny::Node { id: NodeId(n), kids: vec![] }
    }

    fn node(n: u64, kids: Vec<Tiny>) -> Tiny {
        Tiny::Node { id: NodeId(n), kids }
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
            kids: vec![Tiny::Node { id: dup, kids: vec![] }],
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
                kids: vec![Tiny::Node { id: dup, kids: vec![] }],
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

    #[test]
    fn no_empty_many_children_fires_on_empty_many_only_variant() {
        // Tiny::Node is many-only per its Schema — empty kids → fires.
        let ast = leaf(1);
        let diags = Linter::<Tiny>::new()
            .with_rule(NoEmptyManyChildren)
            .lint(&ast);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].rule, NoEmptyManyChildren::NAME);
        assert_eq!(diags[0].severity, Severity::Error);
        assert_eq!(diags[0].node, NodeId(1));
        assert!(diags[0].message.contains("Node"));
    }

    #[test]
    fn no_empty_many_children_passes_when_children_present() {
        let ast = node(1, vec![leaf(2)]);
        // Note: leaf(2) itself is still empty → fires on it. Only the
        // populated root is clean.
        let diags = Linter::<Tiny>::new()
            .with_rule(NoEmptyManyChildren)
            .lint(&ast);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].node, NodeId(2));
    }

    #[test]
    fn no_redundant_wrap_fires_on_parent_wrapping_single_same_variant_child() {
        // node(1, [node(2, [leaf(3)])]) — outer has 1 child of same
        // "Node" variant → fires on id=1.
        let ast = node(1, vec![node(2, vec![leaf(3)])]);
        let diags = Linter::<Tiny>::new()
            .with_rule(NoRedundantWrap)
            .lint(&ast);
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
        let diags = Linter::<Tiny>::new()
            .with_rule(NoRedundantWrap)
            .lint(&ast);
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
        Alpha { id: NodeId },
        #[allow(dead_code)]
        Beta { id: NodeId },
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
