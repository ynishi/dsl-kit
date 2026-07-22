//! PEG grammar AST + recursive-descent interpreter (G-2).
//!
//! The [`Peg`] enum is the grammar AST. It derives `DslNode` + `DslSchema`
//! from the kit — grammars therefore get the L1-L4.6 toolchain for free
//! (walk traversal, schema JSON, lint, `variant_name`, MCP pretty-print)
//! per `parser-design §2`. Parse *execution* is a small dedicated
//! recursive-descent interpreter that walks a [`Grammar`] value against
//! input text and produces a [`ParseTree`] (crate root).
//!
//! # Quality bar (parser-design §3.5, non-negotiable)
//!
//! - **PEG semantics, exactly.** Ordered [`Peg::Choice`] commits on the
//!   first alternative that matches (no reparse of committed input);
//!   [`Peg::Repeat`] is greedy and never backtracks into completed
//!   iterations; a failed alternative restores the input position fully
//!   (no partial-consume leaks).
//! - **Farthest-failure tracking from day one.** The interpreter records
//!   the rightmost failure position + the expected set at that point *as
//!   it runs*. `dsl_kit_load` sees a diagnostic with position + expected
//!   set on failure.
//! - **Termination guarantees at runtime.** Left recursion (a rule
//!   re-entered at the same input position) and nullable-body
//!   [`Peg::Repeat`] (body succeeds without consuming) both trip
//!   guards that either error out (left recursion) or break the loop
//!   (nullable repeat) rather than hanging. G-3's `GrammarCheck` will
//!   catch these statically.
//!
//! # Capture model
//!
//! Two capture primitives:
//!
//! - [`Peg::Node`] `{ variant, body }` — opens a `ParseTree` scope with
//!   the given variant name. Inside `body`, every [`Peg::Field`] binding
//!   becomes a field or child slot on the tree. On success the tree is
//!   *contributed* upwards (to an enclosing Field, or to the top-level
//!   result at the root).
//! - [`Peg::Field`] `{ name, body }` — captures body's *productions*
//!   (raw text spans from [`Peg::Token`] + trees from [`Peg::Node`]) and
//!   binds them under `name` on the enclosing Node scope. Productions
//!   that are all text → one [`RawValue::Text`] field (joined). All
//!   trees → child slot with one or more subtrees. Mixed → tree
//!   productions win (raw text is dropped as syntactic noise from
//!   sub-Nodes' brackets etc.); empty → the Field silently no-ops.
//!   G-3's `GrammarCheck` will surface Mixed as a grammar bug.
//!
//! # Whitespace policy (parser-design §3.5, decided in G-2)
//!
//! **Implicit skip of `[ \t\r\n]*` before every [`Peg::Token`]** (except
//! the `%ws` class itself, which matches one-or-more whitespace
//! literally). No per-rule opt-out yet — deferred until a concrete
//! consumer demands it. Leading and trailing whitespace at the input
//! boundary are also skipped by [`Grammar::parse`].
//!
//! # Token pattern language
//!
//! [`Peg::Token`]'s `pat` string is either a literal or a class:
//!
//! - `%ident` — `[A-Za-z_][A-Za-z_0-9]*`
//! - `%int`   — `-?[0-9]+`
//! - `%str`   — a double-quoted string literal with `\"` `\\` `\n`
//!   `\t` `\r` escapes. Unlike every other token, the production it
//!   contributes is the **decoded inner content** (quotes stripped,
//!   escapes resolved), so a `Field` wrapping `%str` binds the string
//!   value directly.
//! - `%ws`    — `[ \t\r\n]+` (skip is disabled for this class)
//! - `%kw:<word>` — literal `<word>` with a word-boundary guard: the
//!   byte immediately after the match must not be a word char. This is
//!   how you write reserved words like `let` / `in` without swallowing
//!   the prefix of identifiers such as `letme`.
//! - anything else — plain literal match with no boundary guard. Use
//!   this for punctuation like `+`, `*`, `(`, `)`, `=`.

#![allow(clippy::result_unit_err)]

use crate::{BuildError, Diagnostic, ParseTree, RawValue, Span};
use dsl_kit_core::{NodeId, Suggester};
use dsl_kit_macros::{DslNode as DslNodeDerive, DslSchema as DslSchemaDerive};
use std::collections::{BTreeSet, HashMap};

// ---------------------------------------------------------------------------
// Peg AST
// ---------------------------------------------------------------------------

/// PEG grammar AST.
///
/// One value per primitive; the [`Grammar`] wrapper carries the flat
/// list of rules and the start-rule name.
#[derive(Debug, Clone, DslNodeDerive, DslSchemaDerive)]
pub enum Peg {
    /// Named rule definition. Top-level entries in
    /// [`Grammar::rules`] must be `Rule`s; nested `Rule` variants are
    /// legal but unusual and are treated as anonymous scopes by the
    /// interpreter (they run `body` and forward its productions).
    Rule {
        /// Stable node id.
        id: NodeId,
        /// Rule name (referenced by [`Peg::RuleRef`]).
        name: String,
        /// Rule body.
        body: Box<Peg>,
    },
    /// Sequence — every item must succeed in order.
    Seq {
        /// Stable node id.
        id: NodeId,
        /// Ordered sequence items.
        items: Vec<Peg>,
    },
    /// Ordered choice — first alternative that succeeds commits.
    Choice {
        /// Stable node id.
        id: NodeId,
        /// Alternatives, tried left to right.
        alts: Vec<Peg>,
    },
    /// Greedy repetition — runs `body` between `min` and `max` times.
    Repeat {
        /// Stable node id.
        id: NodeId,
        /// Repeated body.
        body: Box<Peg>,
        /// Minimum successful iterations (inclusive).
        min: u32,
        /// Maximum successful iterations (inclusive); `None` = unbounded.
        max: Option<u32>,
    },
    /// Reference to another rule by name.
    RuleRef {
        /// Stable node id.
        id: NodeId,
        /// Referenced rule name.
        name: String,
    },
    /// Terminal — matches a literal string or a small character class
    /// (see the module docs' token pattern language).
    Token {
        /// Stable node id.
        id: NodeId,
        /// Token pattern (literal or `%ident` / `%int` / `%str` /
        /// `%ws` / `%kw:<word>`).
        pat: String,
    },
    /// Opens a `ParseTree` capture scope with the given `variant` name.
    Node {
        /// Stable node id.
        id: NodeId,
        /// Variant name on the produced `ParseTree`.
        variant: String,
        /// Scope body.
        body: Box<Peg>,
    },
    /// Captures body's productions and binds them under `name` on the
    /// enclosing [`Peg::Node`] scope.
    Field {
        /// Stable node id.
        id: NodeId,
        /// Field or child-slot name on the enclosing Node.
        name: String,
        /// Capture body.
        body: Box<Peg>,
    },
}

// ---------------------------------------------------------------------------
// Grammar wrapper
// ---------------------------------------------------------------------------

/// A PEG grammar: a flat list of top-level [`Peg::Rule`]s plus the name
/// of the start rule.
///
/// Kept as a plain wrapper struct deriving nothing so the `Peg` enum
/// stays clean (parser-design §3.4).
#[derive(Debug, Clone)]
pub struct Grammar {
    /// Top-level rules. Every element should be a [`Peg::Rule`]
    /// variant — the interpreter looks up rules by name against this
    /// list.
    pub rules: Vec<Peg>,
    /// Name of the start rule.
    pub start: String,
}

impl Grammar {
    /// Constructs a new [`Grammar`].
    pub fn new(rules: Vec<Peg>, start: impl Into<String>) -> Self {
        Self {
            rules,
            start: start.into(),
        }
    }

    /// Parses `input` against this grammar.
    ///
    /// On success returns a single top-level [`ParseTree`] (produced by
    /// exactly one [`Peg::Node`] reached from the start rule).
    ///
    /// On failure returns a [`BuildError`] whose diagnostics carry the
    /// farthest-failure position + expected set, formatted in the shared
    /// dialect (`{ severity, code, message, location }`).
    pub fn parse(&self, input: &str) -> Result<ParseTree, BuildError> {
        let mut rules_by_name: HashMap<&str, &Peg> = HashMap::new();
        for r in &self.rules {
            if let Peg::Rule { name, .. } = r {
                rules_by_name.insert(name.as_str(), r);
            }
        }
        let start_rule = rules_by_name
            .get(self.start.as_str())
            .copied()
            .ok_or_else(|| {
                let names: Vec<&str> = rules_by_name.keys().copied().collect();
                let base = format!("start rule `{}` is not defined in the grammar", self.start);
                let msg = match crate::BuiltinLevenshteinSuggester
                    .enrich_unknown(self.start.as_str(), &names)
                {
                    Some(hint) => format!("{base} ({hint})"),
                    None => base,
                };
                BuildError::single(Diagnostic::error(codes::UNKNOWN_RULE, msg))
            })?;

        let mut interp = Interpreter::new(input, rules_by_name);
        // Skip leading whitespace at the input boundary.
        interp.skip_ws();
        // Push the top-level sink.
        interp.sink_stack.push(ActiveSink::Top(TopSink::default()));

        match interp.run_peg(start_rule) {
            Ok(()) => {}
            Err(()) => {
                if let Some(fatal) = interp.fatal_error.take() {
                    return Err(BuildError::single(fatal));
                }
                return Err(BuildError::single(interp.expected_diagnostic()));
            }
        }

        // Trailing whitespace + EOF check.
        interp.skip_ws();
        if interp.pos != input.len() {
            // Try to attribute to the farthest expected set if we have
            // one at/after this position; otherwise raw "trailing input".
            let diag = if interp.farthest_pos >= interp.pos && !interp.farthest_expected.is_empty()
            {
                interp.expected_diagnostic()
            } else {
                Diagnostic::error(
                    codes::UNEXPECTED,
                    format!("unexpected trailing input at byte {}", interp.pos),
                )
                .with_span(Some(Span::new(interp.pos, input.len())))
            };
            return Err(BuildError::single(diag));
        }

        let top = match interp.sink_stack.pop() {
            Some(ActiveSink::Top(t)) => t,
            _ => unreachable!("top sink must be present"),
        };
        top.tree.ok_or_else(|| {
            BuildError::single(Diagnostic::error(
                codes::NO_TOP_TREE,
                format!(
                    "parse succeeded but produced no top-level `Node`; \
                     start rule `{}` must reach exactly one `Peg::Node`",
                    self.start
                ),
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// Diagnostic codes
// ---------------------------------------------------------------------------

/// Diagnostic codes emitted by the PEG interpreter.
pub mod codes {
    /// Input did not match at the farthest failure position; message
    /// carries the expected set.
    pub const UNEXPECTED: &str = "dsl_kit::parse::peg::unexpected";
    /// [`super::Peg::RuleRef`] pointed to a rule name not present in
    /// [`super::Grammar::rules`], or [`super::Grammar::start`] itself is
    /// unresolved.
    pub const UNKNOWN_RULE: &str = "dsl_kit::parse::peg::unknown_rule";
    /// A rule was re-entered at the same input position without any
    /// progress — indicates left recursion. G-3's `GrammarCheck` will
    /// catch this statically; the runtime guard is a backstop that
    /// errors out rather than hanging.
    pub const LEFT_RECURSION: &str = "dsl_kit::parse::peg::left_recursion";
    /// A [`super::Peg::Repeat`] iteration succeeded without consuming
    /// input. The interpreter breaks the loop (per §3.5 termination
    /// guarantee); this diagnostic is attached when the `min` bound is
    /// also not yet met, turning the situation into a hard failure.
    pub const NULLABLE_REPEAT: &str = "dsl_kit::parse::peg::nullable_repeat";
    /// Parse succeeded but the start rule never reached a
    /// [`super::Peg::Node`], so there is no top-level `ParseTree`.
    pub const NO_TOP_TREE: &str = "dsl_kit::parse::peg::no_top_tree";
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

/// A raw production captured inside a [`FieldSink`]. Text comes from
/// [`Peg::Token`] matches, trees from [`Peg::Node`] captures. The span
/// on `Text` is retained for future span-carrying `RawValue::Text`
/// support (post-G-2).
#[derive(Debug, Clone)]
enum Production {
    #[allow(dead_code)] // span kept for post-G-2 span-carrying Text values
    Text(String, Span),
    Tree(ParseTree),
}

#[derive(Debug, Default, Clone)]
struct FieldSink {
    productions: Vec<Production>,
}

#[derive(Debug, Default, Clone)]
struct NodeSink {
    fields: Vec<(String, RawValue)>,
    children: Vec<(String, Vec<ParseTree>)>,
}

#[derive(Debug, Default, Clone)]
struct TopSink {
    tree: Option<ParseTree>,
}

/// The active-sink stack entry. Only the top sink is ever mutated by
/// productions.
#[derive(Debug, Clone)]
enum ActiveSink {
    Top(TopSink),
    Node(NodeSink),
    Field(FieldSink),
}

struct Interpreter<'g, 'i> {
    input: &'i str,
    pos: usize,
    rules_by_name: HashMap<&'g str, &'g Peg>,
    sink_stack: Vec<ActiveSink>,
    // Farthest-failure tracking (parser-design §3.5).
    farthest_pos: usize,
    farthest_expected: BTreeSet<String>,
    // Left-recursion guard: (rule name, position at entry).
    call_stack: Vec<(&'g str, usize)>,
    // Fatal errors (unknown rule / left recursion) — trip Err(()) and
    // short-circuit past normal recovery.
    fatal_error: Option<Diagnostic>,
}

impl<'g, 'i> Interpreter<'g, 'i> {
    fn new(input: &'i str, rules_by_name: HashMap<&'g str, &'g Peg>) -> Self {
        Self {
            input,
            pos: 0,
            rules_by_name,
            sink_stack: Vec::new(),
            farthest_pos: 0,
            farthest_expected: BTreeSet::new(),
            call_stack: Vec::new(),
            fatal_error: None,
        }
    }

    // -----------------------------------------------------------------------
    // Sink helpers
    // -----------------------------------------------------------------------

    fn top_sink_snapshot(&self) -> ActiveSink {
        self.sink_stack
            .last()
            .cloned()
            .expect("sink stack has at least the top-level entry during parse")
    }

    fn restore_top_sink(&mut self, snap: ActiveSink) {
        let last = self
            .sink_stack
            .last_mut()
            .expect("sink stack has at least the top-level entry during parse");
        *last = snap;
    }

    fn contribute_text(&mut self, text: String, span: Span) {
        match self.sink_stack.last_mut() {
            Some(ActiveSink::Field(s)) => s.productions.push(Production::Text(text, span)),
            _ => {
                // Discard — Token outside a Field is syntactic (like "+" or "let").
            }
        }
    }

    fn contribute_tree(&mut self, tree: ParseTree) {
        match self.sink_stack.last_mut() {
            Some(ActiveSink::Field(s)) => s.productions.push(Production::Tree(tree)),
            Some(ActiveSink::Node(_)) => {
                // Nested Node not wrapped in Field — silently discarded
                // at runtime. G-3's GrammarCheck should flag such
                // grammars as author bugs.
            }
            Some(ActiveSink::Top(t)) => {
                // Top-level result: latest wins (there should be exactly
                // one; a subsequent one is unusual but harmless).
                t.tree = Some(tree);
            }
            None => unreachable!("sink stack must not be empty during parse"),
        }
    }

    fn contribute_field(&mut self, name: &str, productions: Vec<Production>) {
        if productions.is_empty() {
            return;
        }
        let all_text = productions
            .iter()
            .all(|p| matches!(p, Production::Text(..)));
        let all_tree = productions
            .iter()
            .all(|p| matches!(p, Production::Tree(..)));

        let Some(ActiveSink::Node(node_sink)) = self.sink_stack.last_mut() else {
            // Field outside a Node — silently ignored at runtime.
            return;
        };

        if all_text {
            let joined: String = productions
                .iter()
                .filter_map(|p| match p {
                    Production::Text(t, _) => Some(t.as_str()),
                    _ => None,
                })
                .collect();
            node_sink
                .fields
                .push((name.to_string(), RawValue::Text(joined)));
            return;
        }
        if all_tree {
            let trees: Vec<ParseTree> = productions
                .into_iter()
                .filter_map(|p| match p {
                    Production::Tree(t) => Some(t),
                    _ => None,
                })
                .collect();
            if let Some(slot) = node_sink.children.iter_mut().find(|(n, _)| n == name) {
                slot.1.extend(trees);
            } else {
                node_sink.children.push((name.to_string(), trees));
            }
            return;
        }
        // Mixed: keep tree productions, drop texts (grammar bug — G-3
        // GrammarCheck will flag).
        let trees: Vec<ParseTree> = productions
            .into_iter()
            .filter_map(|p| match p {
                Production::Tree(t) => Some(t),
                _ => None,
            })
            .collect();
        if trees.is_empty() {
            return;
        }
        if let Some(slot) = node_sink.children.iter_mut().find(|(n, _)| n == name) {
            slot.1.extend(trees);
        } else {
            node_sink.children.push((name.to_string(), trees));
        }
    }

    // -----------------------------------------------------------------------
    // Farthest-failure helpers
    // -----------------------------------------------------------------------

    fn expected(&mut self, what: impl Into<String>) {
        if self.pos > self.farthest_pos {
            self.farthest_pos = self.pos;
            self.farthest_expected.clear();
            self.farthest_expected.insert(what.into());
        } else if self.pos == self.farthest_pos {
            self.farthest_expected.insert(what.into());
        }
    }

    fn expected_diagnostic(&self) -> Diagnostic {
        let expected: Vec<String> = self.farthest_expected.iter().cloned().collect();
        let expected_msg = if expected.is_empty() {
            "<unknown>".to_string()
        } else {
            expected
                .iter()
                .map(|e| format!("`{e}`"))
                .collect::<Vec<_>>()
                .join(" | ")
        };
        let end = self.input.len().min(self.farthest_pos + 1);
        let got_msg = if self.farthest_pos >= self.input.len() {
            "end of input".to_string()
        } else {
            let bytes = self.input.as_bytes();
            // Grab a small snippet — up to 8 bytes — to help the reader.
            let snip_end = self.input.len().min(self.farthest_pos + 8);
            let snip =
                std::str::from_utf8(&bytes[self.farthest_pos..snip_end]).unwrap_or("<non-utf8>");
            format!("`{}`", snip)
        };
        Diagnostic::error(
            codes::UNEXPECTED,
            format!(
                "expected {expected_msg} at byte {}, got {got_msg}",
                self.farthest_pos
            ),
        )
        .with_span(Some(Span::new(self.farthest_pos, end)))
    }

    // -----------------------------------------------------------------------
    // PEG dispatch
    // -----------------------------------------------------------------------

    fn run_peg(&mut self, peg: &'g Peg) -> Result<(), ()> {
        match peg {
            Peg::Rule { body, .. } => self.run_peg(body),
            Peg::Seq { items, .. } => {
                for item in items {
                    self.run_peg(item)?;
                }
                Ok(())
            }
            Peg::Choice { alts, .. } => self.run_choice(alts),
            Peg::Repeat { body, min, max, .. } => self.run_repeat(body, *min, *max),
            Peg::RuleRef { name, .. } => self.run_rule_ref(name),
            Peg::Token { pat, .. } => self.match_token(pat),
            Peg::Node { variant, body, .. } => self.run_node(variant, body),
            Peg::Field { name, body, .. } => self.run_field(name, body),
        }
    }

    fn run_choice(&mut self, alts: &'g [Peg]) -> Result<(), ()> {
        let start_pos = self.pos;
        let saved = self.top_sink_snapshot();
        for alt in alts {
            self.pos = start_pos;
            self.restore_top_sink(saved.clone());
            if self.fatal_error.is_some() {
                return Err(());
            }
            if self.run_peg(alt).is_ok() {
                return Ok(());
            }
            if self.fatal_error.is_some() {
                return Err(());
            }
        }
        self.pos = start_pos;
        self.restore_top_sink(saved);
        Err(())
    }

    fn run_repeat(&mut self, body: &'g Peg, min: u32, max: Option<u32>) -> Result<(), ()> {
        let mut count: u32 = 0;
        loop {
            if let Some(mx) = max
                && count >= mx
            {
                break;
            }
            let saved_pos = self.pos;
            let saved_sink = self.top_sink_snapshot();
            match self.run_peg(body) {
                Ok(()) => {
                    if self.pos == saved_pos {
                        // Nullable-body iteration — break rather than
                        // hang. GrammarCheck should reject grammars that
                        // permit this, so the interpreter's job is
                        // termination, not diagnosis.
                        if count < min {
                            self.fatal_error = Some(
                                Diagnostic::error(
                                    codes::NULLABLE_REPEAT,
                                    "repeat body succeeded without consuming input".to_string(),
                                )
                                .with_span(Some(Span::new(saved_pos, saved_pos))),
                            );
                            return Err(());
                        }
                        break;
                    }
                    count += 1;
                }
                Err(()) => {
                    if self.fatal_error.is_some() {
                        return Err(());
                    }
                    self.pos = saved_pos;
                    self.restore_top_sink(saved_sink);
                    break;
                }
            }
        }
        if count < min { Err(()) } else { Ok(()) }
    }

    fn run_rule_ref(&mut self, name: &str) -> Result<(), ()> {
        let Some(&rule) = self.rules_by_name.get(name) else {
            let candidates: Vec<&str> = self.rules_by_name.keys().copied().collect();
            let base = format!("reference to undefined rule `{name}`");
            let msg =
                match crate::BuiltinLevenshteinSuggester.enrich_unknown(name, &candidates) {
                    Some(hint) => format!("{base} ({hint})"),
                    None => base,
                };
            self.fatal_error = Some(
                Diagnostic::error(codes::UNKNOWN_RULE, msg)
                    .with_span(Some(Span::new(self.pos, self.pos))),
            );
            return Err(());
        };
        // Look up by pointer so we keep the &'g lifetime attached.
        let rule_name: &'g str = self
            .rules_by_name
            .keys()
            .copied()
            .find(|k| *k == name)
            .expect("just found");
        // Left-recursion guard: rule re-entered at the same position
        // without any progress.
        if self
            .call_stack
            .iter()
            .any(|(n, p)| *n == rule_name && *p == self.pos)
        {
            self.fatal_error = Some(
                Diagnostic::error(
                    codes::LEFT_RECURSION,
                    format!(
                        "left recursion detected: rule `{name}` re-entered at byte {}",
                        self.pos
                    ),
                )
                .with_span(Some(Span::new(self.pos, self.pos))),
            );
            return Err(());
        }
        self.call_stack.push((rule_name, self.pos));
        let r = self.run_peg(rule);
        self.call_stack.pop();
        r
    }

    fn run_node(&mut self, variant: &str, body: &'g Peg) -> Result<(), ()> {
        let start_pos = self.pos;
        self.sink_stack.push(ActiveSink::Node(NodeSink::default()));
        let body_result = self.run_peg(body);
        let node_sink = match self.sink_stack.pop() {
            Some(ActiveSink::Node(s)) => s,
            _ => unreachable!("balanced push/pop of NodeSink"),
        };
        body_result?;
        let end_pos = self.pos;
        let tree = ParseTree {
            variant: variant.to_string(),
            fields: node_sink.fields,
            children: node_sink.children,
            span: Some(Span::new(start_pos, end_pos)),
        };
        self.contribute_tree(tree);
        Ok(())
    }

    fn run_field(&mut self, name: &str, body: &'g Peg) -> Result<(), ()> {
        self.sink_stack
            .push(ActiveSink::Field(FieldSink::default()));
        let body_result = self.run_peg(body);
        let field_sink = match self.sink_stack.pop() {
            Some(ActiveSink::Field(s)) => s,
            _ => unreachable!("balanced push/pop of FieldSink"),
        };
        body_result?;
        self.contribute_field(name, field_sink.productions);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Token matching
    // -----------------------------------------------------------------------

    fn match_token(&mut self, pat: &str) -> Result<(), ()> {
        if pat != "%ws" {
            self.skip_ws();
        }
        let start = self.pos;
        if pat == "%str" {
            // Special-cased because the contributed production is the
            // decoded inner content, not the matched source slice.
            return match self.match_str() {
                Some(decoded) => {
                    let end = self.pos;
                    self.contribute_text(decoded, Span::new(start, end));
                    Ok(())
                }
                None => {
                    self.expected(pat);
                    self.pos = start;
                    Err(())
                }
            };
        }
        let ok = if pat == "%ident" {
            self.match_ident()
        } else if pat == "%int" {
            self.match_int()
        } else if pat == "%ws" {
            self.match_ws_required()
        } else if let Some(word) = pat.strip_prefix("%kw:") {
            self.match_keyword(word)
        } else {
            self.match_literal(pat)
        };
        if ok {
            let end = self.pos;
            let text = self.input[start..end].to_string();
            self.contribute_text(text, Span::new(start, end));
            Ok(())
        } else {
            // Restore pos (in case skip_ws advanced but match failed —
            // farthest-failure position is the *post-skip* position,
            // which is the semantic failure spot).
            self.expected(pat);
            self.pos = start;
            Err(())
        }
    }

    fn skip_ws(&mut self) {
        let bytes = self.input.as_bytes();
        while let Some(&c) = bytes.get(self.pos) {
            if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn match_ident(&mut self) -> bool {
        let bytes = self.input.as_bytes();
        match bytes.get(self.pos) {
            Some(&c) if is_ident_start(c) => self.pos += 1,
            _ => return false,
        }
        while let Some(&c) = bytes.get(self.pos) {
            if is_ident_cont(c) {
                self.pos += 1;
            } else {
                break;
            }
        }
        true
    }

    fn match_int(&mut self) -> bool {
        let bytes = self.input.as_bytes();
        let mut p = self.pos;
        if bytes.get(p) == Some(&b'-') {
            p += 1;
        }
        let digit_start = p;
        while let Some(&c) = bytes.get(p) {
            if c.is_ascii_digit() {
                p += 1;
            } else {
                break;
            }
        }
        if p > digit_start {
            self.pos = p;
            true
        } else {
            false
        }
    }

    /// Matches a double-quoted string literal and returns the decoded
    /// inner content. `None` on no opening quote, an unterminated
    /// literal, or an unknown escape.
    fn match_str(&mut self) -> Option<String> {
        let rest = &self.input[self.pos..];
        let mut chars = rest.char_indices();
        match chars.next() {
            Some((_, '"')) => {}
            _ => return None,
        }
        let mut out = String::new();
        while let Some((i, c)) = chars.next() {
            match c {
                '"' => {
                    self.pos += i + 1;
                    return Some(out);
                }
                '\\' => match chars.next() {
                    Some((_, '"')) => out.push('"'),
                    Some((_, '\\')) => out.push('\\'),
                    Some((_, 'n')) => out.push('\n'),
                    Some((_, 't')) => out.push('\t'),
                    Some((_, 'r')) => out.push('\r'),
                    _ => return None,
                },
                other => out.push(other),
            }
        }
        None
    }

    fn match_ws_required(&mut self) -> bool {
        let start = self.pos;
        let bytes = self.input.as_bytes();
        while let Some(&c) = bytes.get(self.pos) {
            if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.pos > start
    }

    fn match_literal(&mut self, s: &str) -> bool {
        let bytes = self.input.as_bytes();
        let end = self.pos + s.len();
        if end > bytes.len() {
            return false;
        }
        if &bytes[self.pos..end] != s.as_bytes() {
            return false;
        }
        self.pos = end;
        true
    }

    fn match_keyword(&mut self, word: &str) -> bool {
        let saved = self.pos;
        if !self.match_literal(word) {
            return false;
        }
        // Word-boundary guard: the byte after must not be a word char.
        if let Some(&next) = self.input.as_bytes().get(self.pos)
            && is_ident_cont(next)
        {
            self.pos = saved;
            return false;
        }
        true
    }
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_ident_cont(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

// ---------------------------------------------------------------------------
// Convenience constructors (grammar authors build Peg values by hand
// until G-4's meta-grammar; these helpers cut boilerplate).
// ---------------------------------------------------------------------------

use dsl_kit_core::IdGen;

/// Builds a [`Peg::Rule`] node with a fresh id.
pub fn rule(ids: &IdGen, name: impl Into<String>, body: Peg) -> Peg {
    Peg::Rule {
        id: ids.node(),
        name: name.into(),
        body: Box::new(body),
    }
}

/// Builds a [`Peg::Seq`] node with a fresh id.
pub fn seq(ids: &IdGen, items: Vec<Peg>) -> Peg {
    Peg::Seq {
        id: ids.node(),
        items,
    }
}

/// Builds a [`Peg::Choice`] node with a fresh id.
pub fn choice(ids: &IdGen, alts: Vec<Peg>) -> Peg {
    Peg::Choice {
        id: ids.node(),
        alts,
    }
}

/// Builds a [`Peg::Repeat`] node with a fresh id.
pub fn repeat(ids: &IdGen, body: Peg, min: u32, max: Option<u32>) -> Peg {
    Peg::Repeat {
        id: ids.node(),
        body: Box::new(body),
        min,
        max,
    }
}

/// Builds a [`Peg::RuleRef`] node with a fresh id.
pub fn rule_ref(ids: &IdGen, name: impl Into<String>) -> Peg {
    Peg::RuleRef {
        id: ids.node(),
        name: name.into(),
    }
}

/// Builds a [`Peg::Token`] node with a fresh id.
pub fn token(ids: &IdGen, pat: impl Into<String>) -> Peg {
    Peg::Token {
        id: ids.node(),
        pat: pat.into(),
    }
}

/// Builds a [`Peg::Node`] capture node with a fresh id.
pub fn node(ids: &IdGen, variant: impl Into<String>, body: Peg) -> Peg {
    Peg::Node {
        id: ids.node(),
        variant: variant.into(),
        body: Box::new(body),
    }
}

/// Builds a [`Peg::Field`] capture node with a fresh id.
pub fn field(ids: &IdGen, name: impl Into<String>, body: Peg) -> Peg {
    Peg::Field {
        id: ids.node(),
        name: name.into(),
        body: Box::new(body),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use dsl_kit_core::IdGen;
    use dsl_kit_schema::DslSchema;

    // -- schema shape --

    #[test]
    fn peg_schema_lists_all_variants() {
        let s = Peg::schema();
        assert_eq!(s.name, "Peg");
        let names: Vec<&str> = s.variants.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "Rule", "Seq", "Choice", "Repeat", "RuleRef", "Token", "Node", "Field"
            ]
        );
        let repeat = s.variant("Repeat").unwrap();
        // body -> one child; min / max -> fields.
        assert_eq!(repeat.children.len(), 1);
        assert_eq!(repeat.children[0].name, "body");
        let field_names: Vec<&str> = repeat.fields.iter().map(|f| f.name.as_str()).collect();
        assert!(field_names.contains(&"min"));
        assert!(field_names.contains(&"max"));
    }

    // -- Token matching --

    fn parse_one(g: &Grammar, input: &str) -> Result<ParseTree, BuildError> {
        g.parse(input)
    }

    fn one_rule_grammar(ids: &IdGen, start: &str, body: Peg) -> Grammar {
        Grammar::new(vec![rule(ids, start, body)], start)
    }

    #[test]
    fn literal_token_and_wrap_node() {
        let ids = IdGen::new();
        // start = Node "K" { Field "kw" "let" }
        let g = one_rule_grammar(
            &ids,
            "start",
            node(&ids, "K", field(&ids, "kw", token(&ids, "let"))),
        );
        let tree = parse_one(&g, "let").unwrap();
        assert_eq!(tree.variant, "K");
        assert_eq!(tree.fields.len(), 1);
        assert_eq!(tree.fields[0].0, "kw");
        match &tree.fields[0].1 {
            RawValue::Text(t) => assert_eq!(t, "let"),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn keyword_guard_blocks_prefix_match() {
        let ids = IdGen::new();
        // Plain literal `"let"` would greedily eat the prefix of
        // "letme"; `%kw:let` applies the word-boundary guard.
        let g = one_rule_grammar(&ids, "s", node(&ids, "K", token(&ids, "%kw:let")));
        let err = parse_one(&g, "letme").unwrap_err();
        assert_eq!(err.diagnostics[0].code, codes::UNEXPECTED);

        // And the same pattern DOES match when the boundary is honoured.
        let g2 = one_rule_grammar(&ids, "s", node(&ids, "K", token(&ids, "%kw:let")));
        assert!(parse_one(&g2, "let ").is_ok());
    }

    #[test]
    fn plain_literal_has_no_word_boundary_guard() {
        let ids = IdGen::new();
        // Plain "a" matches even when followed by another word char —
        // it's the grammar author's job to sequence more tokens after.
        let g = one_rule_grammar(&ids, "s", node(&ids, "K", token(&ids, "a")));
        let err = parse_one(&g, "ab").unwrap_err();
        // The literal matched; trailing "b" tripped the EOF check.
        assert_eq!(err.diagnostics[0].code, codes::UNEXPECTED);
        assert!(
            err.diagnostics[0].message.contains("trailing"),
            "{}",
            err.diagnostics[0].message
        );
    }

    #[test]
    fn ident_and_int_classes() {
        let ids = IdGen::new();
        let g = one_rule_grammar(
            &ids,
            "s",
            node(&ids, "Var", field(&ids, "name", token(&ids, "%ident"))),
        );
        let t = parse_one(&g, "foo_bar1").unwrap();
        assert_eq!(t.variant, "Var");
        match &t.fields[0].1 {
            RawValue::Text(v) => assert_eq!(v, "foo_bar1"),
            _ => panic!(),
        }

        let g2 = one_rule_grammar(
            &ids,
            "s",
            node(&ids, "N", field(&ids, "v", token(&ids, "%int"))),
        );
        let t2 = parse_one(&g2, "-42").unwrap();
        match &t2.fields[0].1 {
            RawValue::Text(v) => assert_eq!(v, "-42"),
            _ => panic!(),
        }
    }

    // -- Whitespace --

    #[test]
    fn implicit_ws_skip_before_tokens_and_at_boundaries() {
        let ids = IdGen::new();
        // Node "P" { Field "a" %int Field "b" %int }
        let body = seq(
            &ids,
            vec![
                field(&ids, "a", token(&ids, "%int")),
                field(&ids, "b", token(&ids, "%int")),
            ],
        );
        let g = one_rule_grammar(&ids, "s", node(&ids, "P", body));
        let t = parse_one(&g, "  3   7 ").unwrap();
        assert_eq!(t.fields.len(), 2);
        assert_eq!(t.fields[0].0, "a");
        assert_eq!(t.fields[1].0, "b");
    }

    // -- PEG semantics (parser-design §3.5) --

    #[test]
    fn choice_commits_on_first_match() {
        let ids = IdGen::new();
        // Choice("a" -> A | "ab" -> B). Input "ab" — PEG commits to
        // first alt, "a" matches (leaves "b" behind), then EOF check
        // fails on the trailing "b".
        let alt_a = node(&ids, "A", token(&ids, "a"));
        let alt_ab = node(&ids, "B", token(&ids, "ab"));
        let g = one_rule_grammar(&ids, "s", choice(&ids, vec![alt_a, alt_ab]));
        let err = parse_one(&g, "ab").unwrap_err();
        // Should complain about trailing "b", not attempt the second
        // alt.
        assert_eq!(err.diagnostics[0].code, codes::UNEXPECTED);
        assert!(
            err.diagnostics[0].message.contains("byte 1"),
            "{}",
            err.diagnostics[0].message
        );
    }

    #[test]
    fn failed_alt_restores_position() {
        let ids = IdGen::new();
        // Choice("abc" | "a") on "ab": first fails (partial), position
        // must restore, second succeeds on "a"; trailing "b" then fails
        // EOF. Confirms no partial-consume leak.
        let alt_abc = node(&ids, "L", token(&ids, "abc"));
        let alt_a = node(&ids, "S", token(&ids, "a"));
        let g = one_rule_grammar(&ids, "s", choice(&ids, vec![alt_abc, alt_a]));
        let err = parse_one(&g, "ab").unwrap_err();
        // Trailing "b" at byte 1.
        assert!(
            err.diagnostics[0].message.contains("byte 1"),
            "{}",
            err.diagnostics[0].message
        );
    }

    #[test]
    fn repeat_is_greedy_and_no_backtrack_into_completed() {
        let ids = IdGen::new();
        // Seq( Repeat("a", 0..*), "a" ). Repeat eats all a's greedily;
        // trailing "a" required by the Seq item then fails. If Repeat
        // gave back one iteration, we'd succeed — the test proves it
        // does not.
        let g = one_rule_grammar(
            &ids,
            "s",
            node(
                &ids,
                "N",
                seq(
                    &ids,
                    vec![repeat(&ids, token(&ids, "a"), 0, None), token(&ids, "a")],
                ),
            ),
        );
        let err = parse_one(&g, "aaa").unwrap_err();
        assert_eq!(err.diagnostics[0].code, codes::UNEXPECTED);
    }

    #[test]
    fn repeat_min_enforced() {
        let ids = IdGen::new();
        let g = one_rule_grammar(
            &ids,
            "s",
            node(&ids, "N", repeat(&ids, token(&ids, "a"), 2, None)),
        );
        // Only one "a" — Repeat min=2 fails.
        assert!(parse_one(&g, "a").is_err());
        // Two "a"s — succeeds.
        assert!(parse_one(&g, "aa").is_ok());
    }

    // -- Farthest failure --

    #[test]
    fn farthest_failure_carries_position_and_expected_set() {
        let ids = IdGen::new();
        // Seq("let", "x", "="). On input "let x !": we match "let"
        // (0..3), skip ws to 4, match "x" (4..5), skip ws to 6, expect
        // "=" at 6 but get "!". Farthest-failure position = 6, expected
        // = ["="].
        let g = one_rule_grammar(
            &ids,
            "s",
            node(
                &ids,
                "N",
                seq(
                    &ids,
                    vec![token(&ids, "let"), token(&ids, "x"), token(&ids, "=")],
                ),
            ),
        );
        let err = parse_one(&g, "let x !").unwrap_err();
        let msg = &err.diagnostics[0].message;
        assert!(msg.contains("byte 6"), "{msg}");
        assert!(msg.contains("`=`"), "{msg}");
    }

    // -- Termination guards --

    #[test]
    fn left_recursion_is_detected_at_runtime() {
        let ids = IdGen::new();
        // r <- r "a" — direct left recursion. Should trip the guard
        // rather than hang.
        let r = rule(
            &ids,
            "r",
            seq(&ids, vec![rule_ref(&ids, "r"), token(&ids, "a")]),
        );
        let g = Grammar::new(vec![r], "r");
        let err = g.parse("aaa").unwrap_err();
        assert_eq!(err.diagnostics[0].code, codes::LEFT_RECURSION);
    }

    #[test]
    fn unknown_rule_ref_is_diagnosed() {
        let ids = IdGen::new();
        let r = rule(&ids, "start", rule_ref(&ids, "missing"));
        let g = Grammar::new(vec![r], "start");
        let err = g.parse("").unwrap_err();
        assert_eq!(err.diagnostics[0].code, codes::UNKNOWN_RULE);
    }

    #[test]
    fn unknown_rule_ref_suggests_declared_rule() {
        // A typo of a declared rule name should surface the correct
        // name via the built-in Levenshtein suggester.
        let ids = IdGen::new();
        let start = rule(&ids, "start", rule_ref(&ids, "helo"));
        let target = rule(&ids, "hello", token(&ids, "hi"));
        let g = Grammar::new(vec![start, target], "start");
        let err = g.parse("hi").unwrap_err();
        assert_eq!(err.diagnostics[0].code, codes::UNKNOWN_RULE);
        assert!(
            err.diagnostics[0].message.contains("did you mean")
                && err.diagnostics[0].message.contains("hello"),
            "expected hint, got: {}",
            err.diagnostics[0].message
        );
    }

    #[test]
    fn unknown_start_rule_is_diagnosed() {
        let ids = IdGen::new();
        let r = rule(&ids, "other", token(&ids, "x"));
        let g = Grammar::new(vec![r], "start");
        let err = g.parse("x").unwrap_err();
        assert_eq!(err.diagnostics[0].code, codes::UNKNOWN_RULE);
    }

    #[test]
    fn unknown_start_rule_suggests_declared_rule() {
        let ids = IdGen::new();
        let r = rule(&ids, "start", token(&ids, "x"));
        let g = Grammar::new(vec![r], "strat");
        let err = g.parse("x").unwrap_err();
        assert_eq!(err.diagnostics[0].code, codes::UNKNOWN_RULE);
        assert!(
            err.diagnostics[0].message.contains("did you mean")
                && err.diagnostics[0].message.contains("start"),
            "expected hint, got: {}",
            err.diagnostics[0].message
        );
    }

    #[test]
    fn nullable_repeat_body_below_min_is_rejected() {
        let ids = IdGen::new();
        // Repeat body is a Field with no productive body — matches
        // empty. min=1 forces a failure rather than a hang.
        let empty_body = seq(&ids, vec![]);
        let g = one_rule_grammar(
            &ids,
            "s",
            node(&ids, "N", repeat(&ids, empty_body, 1, None)),
        );
        let err = g.parse("").unwrap_err();
        assert_eq!(err.diagnostics[0].code, codes::NULLABLE_REPEAT);
    }
}
