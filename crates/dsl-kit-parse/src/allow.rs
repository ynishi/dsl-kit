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
//! The text spelling is refused symmetrically: `@allow("x") @import
//! "y"` is a [`codes::UNCOLLAPSED`] diagnostic from [`collapse`]. The
//! loader replaces the placeholder with the imported tree, which would
//! carry the annotation away with the node it was written on — and a
//! suppression that quietly does nothing is exactly what the shape
//! rules here exist to prevent.
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
//! ## The text spelling
//!
//! The two front-ends are deliberately asymmetric. JSON recognises
//! `$allow` unconditionally — a reserved key costs a document nothing
//! until it spells one. The canonical text front-end is opt-in:
//! [`add_allow_syntax`] injects the `@allow("rule") <node>` spelling
//! into a [`Grammar`], and a grammar that never passes through it does
//! not accept `@allow` anywhere, the same shape
//! [`crate::import::add_import_syntax`] has.
//!
//! Where the annotation sits differs too, because the two surfaces
//! differ: an object has a spare key, a text node has only the
//! position in front of it. So text spells it as a **wrapper**:
//!
//! ```text
//! @allow("max-fan-out") Par(branches: [ … ])
//! @allow("max-fan-out", "max-depth") Par(branches: [ … ])
//! ```
//!
//! parses to a reserved [`ALLOW_VARIANT`] node holding the annotated
//! node in its [`ALLOW_TARGET_SLOT`] child, and [`collapse`] — which
//! [`Grammar::parse`] runs on every text parse — folds the wrapper
//! away onto its target's [`ParseTree::allows`]. Downstream therefore
//! never sees the wrapper, and the two front-ends hand it the same
//! tree for the same document. A wrapper that does reach
//! [`check_conformance`](crate::check_conformance) is a
//! [`codes::UNCOLLAPSED`] diagnostic there, the way an unexpanded
//! `$import` is.
//!
//! [`ParseTree::allows`]: crate::ParseTree::allows
//! [`Grammar::parse`]: crate::peg::Grammar::parse

use crate::import::IMPORT_VARIANT;
use crate::peg::{self, Grammar, Peg};
use crate::{BuildError, Diagnostic, ParseTree, RawValue, Span};
use dsl_kit_core::IdGen;
use serde_json::Value;

/// Reserved object key carrying a node's usage-site lint suppressions
/// in the JSON front-end.
pub const ALLOW_KEY: &str = "$allow";

/// Reserved [`ParseTree::variant`](crate::ParseTree::variant) spelling
/// for an un-collapsed `@allow` wrapper in the text front-end.
///
/// Same `$` sigil, same reservation argument as [`ALLOW_KEY`]: it
/// cannot collide with a variant a DSL author declares. The two
/// constants share a spelling but not a position — one is an object
/// key, the other a variant name.
pub const ALLOW_VARIANT: &str = "$allow";

/// Payload field on an [`ALLOW_VARIANT`] wrapper carrying one rule
/// name. A wrapper naming several rules carries the field once per
/// name, in the order written.
pub const ALLOW_RULES_FIELD: &str = "rules";

/// Child slot on an [`ALLOW_VARIANT`] wrapper holding the single node
/// the annotation applies to.
pub const ALLOW_TARGET_SLOT: &str = "target";

/// Diagnostic codes emitted by the `$allow` annotation machinery.
pub mod codes {
    /// The [`ALLOW_KEY`](super::ALLOW_KEY) value is not an array of
    /// rule-name strings.
    pub const ALLOW_SHAPE: &str = "dsl_kit::parse::allow::allow_shape";
    /// An [`ALLOW_VARIANT`](super::ALLOW_VARIANT) wrapper survived
    /// [`collapse`](super::collapse) — either because the document
    /// never went through it and the wrapper reached
    /// [`check_conformance`](crate::check_conformance), or because the
    /// wrapper was malformed and the fold refused to guess.
    pub const UNCOLLAPSED: &str = "dsl_kit::parse::allow::uncollapsed";
}

// ---------------------------------------------------------------------------
// Text syntax injection
// ---------------------------------------------------------------------------

/// Adds the reserved `@allow("rule") <node>` spelling to a text
/// grammar.
///
/// Appends a rule named [`ALLOW_VARIANT`] whose body is
/// `Node("$allow", %kw:@allow "(" Field("rules", %str)+ ")"
/// Field("target", <start>))`, and makes it an alternative of the
/// grammar's start rule — for grammars generated by
/// [`crate::schema_gen`], the start rule is the `node` choice every
/// child slot references, so an annotation is writable at every node
/// position. A start rule whose body is not a [`Peg::Choice`] is
/// wrapped in one.
///
/// Opt-in by design: a grammar that never passes through this function
/// accepts no `@allow` syntax at all, and the reserved rule is
/// invisible to [`crate::example_gen`] (examples never spell `@allow`)
/// and exempt from [`crate::grammar_check`]'s schema-consistency pass.
///
/// Idempotent — a grammar that already carries the reserved rule is
/// returned unchanged. Fails with [`crate::peg::codes::UNKNOWN_RULE`]
/// if the start rule is not defined.
pub fn add_allow_syntax(grammar: &mut Grammar, ids: &IdGen) -> Result<(), BuildError> {
    let already = grammar
        .rules
        .iter()
        .any(|r| matches!(r, Peg::Rule { name, .. } if name == ALLOW_VARIANT));
    if already {
        return Ok(());
    }

    let start = grammar.start.clone();
    let start_rule = grammar
        .rules
        .iter_mut()
        .find(|r| matches!(r, Peg::Rule { name, .. } if *name == start));
    let Some(Peg::Rule { body, .. }) = start_rule else {
        return Err(BuildError::single(Diagnostic::error(
            peg::codes::UNKNOWN_RULE,
            format!("cannot add allow syntax: start rule `{start}` is not defined"),
        )));
    };

    match body.as_mut() {
        Peg::Choice { alts, .. } => alts.push(peg::rule_ref(ids, ALLOW_VARIANT)),
        _ => {
            let dummy = peg::token(ids, "");
            let old = std::mem::replace(body.as_mut(), dummy);
            **body = peg::choice(ids, vec![old, peg::rule_ref(ids, ALLOW_VARIANT)]);
        }
    }

    // Each name lands in its own `Field`, so the wrapper carries one
    // `rules` entry per rule rather than one entry holding every name
    // concatenated — a `Field` joins the text productions it collects,
    // which would fuse `"a", "b"` into `ab`.
    let name = |ids: &IdGen| peg::field(ids, ALLOW_RULES_FIELD, peg::token(ids, "%str"));
    grammar.rules.push(peg::rule(
        ids,
        ALLOW_VARIANT,
        peg::node(
            ids,
            ALLOW_VARIANT,
            peg::seq(
                ids,
                vec![
                    peg::token(ids, "%kw:@allow"),
                    peg::token(ids, "("),
                    name(ids),
                    peg::repeat(
                        ids,
                        peg::seq(ids, vec![peg::token(ids, ","), name(ids)]),
                        0,
                        None,
                    ),
                    peg::token(ids, ")"),
                    peg::field(ids, ALLOW_TARGET_SLOT, peg::rule_ref(ids, start)),
                ],
            ),
        ),
    ));
    Ok(())
}

// ---------------------------------------------------------------------------
// Collapse
// ---------------------------------------------------------------------------

/// Folds every [`ALLOW_VARIANT`] wrapper in `tree` onto its target's
/// [`ParseTree::allows`](crate::ParseTree::allows).
///
/// [`Grammar::parse`](crate::peg::Grammar::parse) applies this to
/// every text parse, so a host wires up nothing: a document that never
/// spells `@allow` has no wrapper to fold and comes back as it went
/// in. Front-ends that build wrappers themselves call it directly.
///
/// Stacked wrappers (`@allow("a") @allow("b") X`) fold onto the same
/// target, outermost name first. The surviving node keeps the
/// **target's** span: the wrapper's span covers the annotation the
/// author wrote, which is not where a diagnostic about the annotated
/// node should point.
///
/// Fails with [`codes::UNCOLLAPSED`] on a wrapper the fold cannot
/// consume — no target, more than one, a `rules` payload that is not a
/// string, any other slot. The injected grammar cannot produce those
/// shapes; a hand-built tree can, and dropping the annotated subtree
/// quietly would be the wrong kind of forgiving.
///
/// It fails the same way on a wrapper whose target is an
/// [`IMPORT_VARIANT`] placeholder, which the grammar *can* produce:
/// the annotation would survive the fold only to be discarded when the
/// loader substitutes the imported tree.
pub fn collapse(tree: ParseTree) -> Result<ParseTree, BuildError> {
    fold(tree).map_err(BuildError::new)
}

/// Recursive half of [`collapse`]. Diagnostics from distinct wrappers
/// are collected before failing, as everywhere else in the trunk.
fn fold(mut tree: ParseTree) -> Result<ParseTree, Vec<Diagnostic>> {
    if tree.variant == ALLOW_VARIANT {
        return fold_wrapper(tree);
    }

    let mut diags = Vec::new();
    for (_, slot) in &mut tree.children {
        for child in std::mem::take(slot) {
            match fold(child) {
                Ok(t) => slot.push(t),
                Err(ds) => diags.extend(ds),
            }
        }
    }
    for (_, entries) in &mut tree.keyed_children {
        for (key, child) in std::mem::take(entries) {
            match fold(child) {
                Ok(t) => entries.push((key, t)),
                Err(ds) => diags.extend(ds),
            }
        }
    }
    if diags.is_empty() {
        Ok(tree)
    } else {
        Err(diags)
    }
}

/// Folds one wrapper: read its names, take its single target, fold
/// that target in turn, and prepend the names to whatever the target
/// already carries.
fn fold_wrapper(tree: ParseTree) -> Result<ParseTree, Vec<Diagnostic>> {
    let span = tree.span;
    let ParseTree {
        fields,
        children,
        keyed_children,
        allows,
        ..
    } = tree;

    let mut diags = Vec::new();
    let mut names = Vec::new();
    for (field, value) in fields {
        if field != ALLOW_RULES_FIELD {
            diags.push(malformed(
                span,
                format!("carries an unexpected field `{field}`"),
            ));
            continue;
        }
        match value {
            RawValue::Text(s) | RawValue::Json(Value::String(s)) => names.push(s),
            _ => diags.push(malformed(
                span,
                format!("carries a `{ALLOW_RULES_FIELD}` payload that is not a rule name"),
            )),
        }
    }
    if !allows.is_empty() {
        diags.push(malformed(
            span,
            "carries suppressions of its own".to_string(),
        ));
    }
    for (slot, _) in keyed_children {
        diags.push(malformed(
            span,
            format!("carries a keyed child slot `{slot}`"),
        ));
    }

    let mut target = None;
    for (slot, mut nodes) in children {
        if slot != ALLOW_TARGET_SLOT {
            diags.push(malformed(
                span,
                format!("carries an unexpected child slot `{slot}`"),
            ));
        } else if target.is_some() {
            diags.push(malformed(
                span,
                format!("carries more than one `{ALLOW_TARGET_SLOT}` slot"),
            ));
        } else if nodes.len() != 1 {
            diags.push(malformed(
                span,
                format!(
                    "holds {} nodes in its `{ALLOW_TARGET_SLOT}` slot, not the one it annotates",
                    nodes.len()
                ),
            ));
        } else {
            target = Some(nodes.remove(0));
        }
    }

    let Some(target) = target else {
        if diags.is_empty() {
            diags.push(malformed(
                span,
                format!("has no `{ALLOW_TARGET_SLOT}` child to annotate"),
            ));
        }
        return Err(diags);
    };
    // The one malformed shape the injected grammar can produce. The
    // fold would happily move the names onto the placeholder, and the
    // loader would then replace that placeholder — annotation and all
    // — with the imported tree, so the suppression the author wrote
    // would silently do nothing.
    if target.variant == IMPORT_VARIANT {
        diags.push(malformed(
            span,
            format!(
                "annotates an `{IMPORT_VARIANT}` placeholder, which cannot carry a \
                 suppression; annotate the imported source's own nodes instead"
            ),
        ));
    }
    if !diags.is_empty() {
        return Err(diags);
    }

    let mut target = fold(target)?;
    names.append(&mut target.allows);
    target.allows = names;
    Ok(target)
}

fn malformed(span: Option<Span>, what: String) -> Diagnostic {
    Diagnostic::error(
        codes::UNCOLLAPSED,
        format!("`{ALLOW_VARIANT}` wrapper {what}"),
    )
    .with_span(span)
}
