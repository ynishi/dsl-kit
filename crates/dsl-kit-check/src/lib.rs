//! Semantic check layer for `dsl-kit` DSLs — judgement rules as data.
//!
//! `check_conformance` (in `dsl-kit-parse`) answers *"is this document
//! the right shape?"*. This crate answers the next question: *"does it
//! mean anything?"* — types agreeing across an `if`, steps happening in
//! a workable order, a handle being bound before it is used. The
//! answer is computed from a [`CheckProgram`] the DSL author writes,
//! not from rules hard-coded here, so a new judgement family costs a
//! `Fact` predicate name rather than an engine change.
//!
//! ## Where it sits
//!
//! ```text
//! JSON / text source
//!     ├─ serde_bridge / peg  →  ParseTree (span-carrying)
//!     ├─ check_conformance      … shape, per node
//!     ├─ check_semantics ★      … this crate, whole tree, one pass
//!     ├─ DslBuild::from_parse_tree → typed AST
//!     └─ Linter::lint_with_allows … advisory
//! ```
//!
//! [`check_semantics`] is **opt-in**: `dsl-kit-parse` cannot call it
//! (this crate depends on parse, not the other way round), so the host
//! invokes it between loading a tree and building the typed AST. It
//! runs on the [`ParseTree`](dsl_kit_parse::ParseTree) rather than the
//! typed AST because the tree is the only representation that carries
//! source spans, and line-anchored errors are the point.
//!
//! Its findings are **not suppressible**: `$allow` / `AllowTable`
//! govern advisory lint, and a state or type violation is a won't-run
//! condition in the same league as a conformance failure.
//!
//! ## A whole program, end to end
//!
//! ```
//! use dsl_kit_check::{CheckProgram, Rule, SeqSlotDecl, atom, check_semantics, codes, fact};
//! use dsl_kit_parse::ParseTree;
//!
//! // "A plan starts Raw; fetch makes it Fetched; build needs Fetched."
//! let program = CheckProgram::builder()
//!     .seq_slot(SeqSlotDecl::fold("Plan", "steps", fact("state", [atom("Raw")])))
//!     .rule(
//!         Rule::on("Fetch")
//!             .requires_state(fact("state", [atom("Raw")]))
//!             .transitions_to(fact("state", [atom("Fetched")]))
//!             .message(codes::CHECK_STATE_MISMATCH, "`fetch` needs {expected}, found {found}"),
//!     )
//!     .rule(
//!         Rule::on("Build")
//!             .requires_state(fact("state", [atom("Fetched")]))
//!             .transitions_to(fact("state", [atom("Built")]))
//!             .message(
//!                 codes::CHECK_STATE_MISMATCH,
//!                 "`build` needs {expected}, found {found} (set by {provenance})",
//!             ),
//!     )
//!     .build();
//!
//! let mut plan = ParseTree::new("Plan");
//! plan.children = vec![(
//!     "steps".into(),
//!     vec![ParseTree::new("Build"), ParseTree::new("Fetch")],
//! )];
//!
//! let diags = check_semantics(&plan, &program);
//! assert_eq!(diags.len(), 1);
//! assert_eq!(diags[0].code, codes::CHECK_STATE_MISMATCH);
//! assert!(diags[0].message.contains("needs state(Fetched), found state(Raw)"));
//! // Span-less (hand-built) tree: the path trail anchors the message.
//! assert!(diags[0].message.contains("[at steps[0]]"));
//! ```
//!
//! ## Layout
//!
//! - [`ir`] — the data model ([`Term`], [`Fact`], [`Premise`],
//!   [`Rule`], [`MessageTemplate`], [`CheckProgram`], [`SeqSlotDecl`],
//!   [`DslCheck`]) plus builders.
//! - [`solver`] — the single bottom-up pass and its unification.
//! - [`codes`] — diagnostic slugs.

#![warn(missing_docs)]

pub mod codes;
pub mod ir;
pub mod solver;

pub use ir::{
    CheckProgram, CheckProgramBuilder, DslCheck, Fact, MessageTemplate, Premise, Rule, RuleBuilder,
    SeqMode, SeqSlotDecl, Term, atom, ctor, fact, field_ref, var,
};
pub use solver::check_semantics;
