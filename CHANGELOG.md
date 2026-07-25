# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `dsl-kit-macros` — `#[derive(DslNode)]` / `#[derive(DslSchema)]`
  now recognise `BTreeMap<String, T>` and `BTreeMap<String, Box<T>>`
  (where `T` is the derived-on enum itself) as keyed self-recursive
  child slots. `Walk` / `WalkMut` iterate `.values()` /
  `.values_mut()` in the map's own (sorted-by-key) order; the schema
  reports `Multiplicity::Map`. `#[derive(DslBuild)]` reads the tree's
  keyed half through `build_child_map` and rebuilds the
  `BTreeMap` with its keys intact; `#[derive(DslExec)]`
  collect-children treats keyed slots the same as `Vec<Box<T>>` for
  ordering. Non-`Box` keyed shapes (`Recursion::Map`) are supported
  for symmetry with the positional shapes.
- `dsl-kit-parse` — keyed child slots are now a first-class part of
  the parse trunk. `ParseTree::keyed_children` holds
  `(slot, [(key, subtree)])` alongside the positional `children`, with
  a `ParseTree::keyed_child_slot(name)` accessor; `build_child_map`
  is the `DslBuild` helper that turns a keyed slot into a
  `BTreeMap<String, T>`. `check_conformance` accepts zero-or-more
  keyed entries and rejects the ways a tree can get keying wrong (see
  the diagnostics below).
  The JSON ⇒ `ParseTree` bridge reads a `Map` slot as an object
  mapping keys to child objects (`{"entries": {"k": {…}}}`) and sorts
  the entries by key on ingest, so documents that differ only in key
  order produce identical trees regardless of how `serde_json` was
  built.
- `dsl-kit-parse` — three diagnostics guarding the keyed-slot
  contract, all errors rather than silent recovery:
  `codes::DUPLICATE_KEY` (`dsl_kit::parse::duplicate_key`) when a slot
  carries the same key twice, which would otherwise drop a subtree the
  author wrote; `codes::KEYED_SLOT_SHAPE`
  (`dsl_kit::parse::keyed_slot_shape`) when a `Map` slot is supplied
  as a positional list or a positional slot as keyed entries, which
  would otherwise read as an empty slot; and
  `codes::KEYED_SLOT_UNSORTED`
  (`dsl_kit::parse::keyed_slot_unsorted`) when entries are not in
  ascending key order, which would otherwise leave two front-ends
  producing unequal trees for the same document. A slot found only in
  the half it does not belong to is reported once, for the keying —
  the arity check is stood down so it cannot also claim the slot is
  empty.
- `dsl-kit-schema` — new `Multiplicity::Map` variant marking a
  string-keyed child slot (`Map<String, V>` shape at the Rust level).
  `Multiplicity::as_str()` returns `"map"` and `NodeSchema::to_json`
  emits `"multiplicity": "map"`. Consumers can construct schemas with
  keyed slots today; the derive macro, PEG codegen, and JSON ⇔ AST
  bridge grow support incrementally per the tracking issue.
- `dsl-kit-parse` — keyed slots have a canonical **text** syntax:
  `name: { key: <node>, "other key": <node> }`. Braces (not the `Many`
  idiom's brackets) mark the slot as keyed, matching the JSON bridge's
  object spelling so one DSL reads the same way through either
  front-end; a key is either a bare `%ident` or a `%str` literal, so
  keys that are not identifiers stay writable. Entries may be written
  in any order — the parser sorts them into canonical order — and an
  empty map is valid. `grammar_from_schema` therefore no longer
  refuses `Multiplicity::Map` schemas, and
  `schema_gen::codes::MAP_NOT_IMPLEMENTED` is gone with the last of
  its siblings.
- `dsl-kit-parse` — `Peg::KeyedEntry { slot, key, value }` plus the
  `peg::keyed_entry` constructor: the capture primitive behind the
  syntax above. A separate primitive rather than a `Field` convention
  because a `Field` binds one *fixed* name to its productions, which
  cannot express a name the input supplies. `Peg::Node` sorts each
  keyed slot before emitting, so hand-written grammars get the
  canonical order without arranging for it. **Breaking** for
  out-of-crate `match` on `Peg`, which is exhaustive. A key production
  that yields no production at all is rejected (it would key every
  entry on `""`), but a production that *is* empty — a quoted `""` —
  is a legitimate key and parses, since both the AST and the JSON
  front-end can hold one.
- `dsl-kit-parse` — `codes::UNKNOWN_MULTIPLICITY` /
  `schema_gen::codes::UNKNOWN_MULTIPLICITY` slugs backing the
  `#[non_exhaustive]` catch-all arms; surface a stable signal instead
  of a panic when a future `Multiplicity` variant is added ahead of
  parse-side support.
- `examples/cfg-example` — new example crate (`cfg-dsl` / `cfg-host` /
  `cfg-mcp` plus a stdout demo) demonstrating the keyed `Map`
  primitive end to end. `Cfg` is a small configuration DSL whose
  children are named rather than positional: `Env` carries
  `BTreeMap<String, Box<Cfg>>` and `Overrides` carries
  `BTreeMap<String, Cfg>`, so both derive arms (`MapBoxed` and `Map`)
  stay covered by one DSL, while `Ref` suspends for a host-supplied
  value and `Leaf` terminates. Tests pin the JSON front-end, the
  schema-generated text grammar, a typed-AST → text → typed-AST round
  trip, and the MCP tool surface (`dsl_kit_schema` reporting
  `multiplicity: "map"`, `dsl_kit_load` accepting a keyed document,
  `dsl_kit_step` / `dsl_kit_resolve` walking it). `cfg-mcp` installs
  with `cargo install --path examples/cfg-example/crates/cfg-mcp`.

### Changed

- `dsl-kit-schema` — `Multiplicity` is now `#[non_exhaustive]`.
  Downstream out-of-crate matches on the enum must include a `_ =>`
  arm; in-crate matches remain exhaustively checked. This bump costs
  once so future variants (ordered sets, non-empty lists, fixed-arity
  tuple slots, …) land as minor bumps per RFC 2008. **Breaking** for
  out-of-workspace consumers that matched exhaustively on
  `Multiplicity` without a catch-all.
- `dsl-kit-parse` — `ParseTree` gained the `keyed_children` field.
  **Breaking** for front-ends that construct `ParseTree` with a struct
  literal; `ParseTree::new(variant)` followed by field assignment is
  the construction path that survives future additions. The struct is
  deliberately left exhaustive (unlike `Multiplicity`) because
  hand-building a tree is a primary use case for front-end authors.
  One known limitation rides on this: a keyed entry is a
  `(String, ParseTree)` tuple, so the *key* has nowhere to carry a
  source span. The text front-end works without one, but it means a
  duplicate-key diagnostic anchors on the enclosing node rather than
  on the offending key. Giving keys their own span would change the
  entry shape, so it is deferred rather than guessed at.
- `dsl-kit-parse` — example synthesis gives each entry of a keyed slot
  a distinct key (`key1`, `key2`, …). Reusing one token's text would
  synthesize `{ "example": …, "example": … }`: grammar-legal, but
  rejected by the schema as a duplicate key — the grammar/schema drift
  machine-derived examples exist to rule out. Rendering also learned
  `{` / `}` spacing, so an empty map prints `{}` like an empty list
  prints `[]`.

### Deprecated

### Removed

### Fixed

- `dsl-kit-lint` — `NoEmptyManyChildren` is now `NoEmptyChildSlots`
  (`no-empty-many-children` → `no-empty-child-slots`,
  `dsl_kit::lint::no_empty_many_children` →
  `dsl_kit::lint::no_empty_child_slots`), reports at
  `Suspicious` / `Warn` instead of `Correctness` / `Error`, and covers
  `Multiplicity::Map` slots alongside `Many`.

  The rule claimed to fire only on variants whose shape "guarantees at
  least one child". No such guarantee exists: both collection
  multiplicities mean *zero* or more, `check_conformance` accepts
  empty, and the schema has no way to say "at least one". So a DSL
  with a legitimately empty block or argument list got an `Error` from
  a rule in the default bundle, with no way to express the intent
  short of disabling the rule. What the rule really encodes is "a
  variant that exists only to hold a collection, holding nothing, is
  more often an oversight than an intention" — a heuristic, which is
  what `Suspicious` / `Warn` is for.

  **Breaking**: no alias is kept for the old type name. The rule's
  behaviour changed along with its name, so an alias would let code
  that matched on `Severity::Error` keep compiling while quietly
  seeing different results.
- `dsl-kit-schema` — `Multiplicity::Map`'s documentation claimed the
  derive recognises `HashMap<String, Box<Self>>` and that runtime
  support was unimplemented. Neither was true: only `BTreeMap` shapes
  are keyed slots (a map slot's iteration order is observable, so it
  has to be deterministic), and every stage now carries support. An
  author following the old text would have written a field the derive
  silently treats as a non-recursive payload.

### Security

## [0.4.0] - 2026-07-24

### Added

- `dsl-kit-core` — structured, Clippy-style fix suggestions in
  `suggest`. `Applicability` (`MachineApplicable` / `MaybeIncorrect` /
  `HasPlaceholders`, serde round-trippable, no `Unspecified` escape
  hatch) gates auto-apply; only `MachineApplicable` may be applied
  without review. `FixSuggestion` pairs a message with a multipart
  `Vec<PatchPart>` patch (`PatchPart { node, path, replacement }`,
  `NodeId`-anchored) and an `Applicability`. Immutable once built via
  `FixSuggestion::new` / `with_part`. The string-only `Suggester` /
  `Suggestion` contract is unchanged — this layer sits on top of it.
- `dsl-kit-lint` — central lint declaration registry. `LintCategory`
  (`Correctness` / `Suspicious` / `Style` / `Complexity` / `Contract`)
  and `LintDecl { name, code, category, default_severity, desc }`
  decouple rule metadata from the `Rule` impl (rustc `declare_lint!`
  style). `LINT_DECLS` lists all seven built-ins; `lint_decl(name_or_code)`
  looks one up; `lint_catalog()` projects them into
  `ErrorCatalogEntry`. Lint codes use the `dsl_kit::lint::<name>` form,
  sharing the `dsl_kit::` code space with engine error codes.

### Changed

- `dsl-kit-lint` — `Diagnostic` gains an `Option<FixSuggestion>`
  `suggestion` field (`None` for report-only rules; existing rules are
  unchanged). New `LintContext::report_with_suggestion` helper.
  `TypoHint` now attaches a `FixSuggestion` (`MaybeIncorrect`,
  single-part patch replacing the label with the top fuzzy candidate)
  on each near-miss. **Breaking:** struct-literal `Diagnostic { .. }`
  callers must add `suggestion: None`.
- `dsl-kit-mcp` — `dsl_kit_explain` now merges the built-in lint codes
  (`dsl_kit_lint::lint_catalog`) into its catalogue alongside engine
  error codes and host-contributed entries; the unknown-code
  `did you mean` suggester runs over the merged code space. Adds a
  `dsl-kit-lint` dependency.

### Deprecated

### Removed

### Fixed

### Security

## [0.3.0] - 2026-07-23

Built-in optional payload fields and standard `Vec<String>` /
`Option<String>` mapping — the "AI clients can omit noise fields"
contract that closes GH issue #1.

- `dsl-kit-schema` — `FieldSchema` gains a `pub optional: bool`. Set
  to `true` for payload fields whose absence is a valid tree shape
  (typically `Option<T>` → `None`, `Vec<T>` → empty). `to_json`
  emits `"optional": true` only when set, preserving the pre-0.3
  layout for required fields. Helper constructors
  `FieldSchema::required` / `::optional` cut boilerplate for
  hand-written schemas. **Breaking:** struct-literal callers must
  add `optional: <bool>` to every `FieldSchema { ... }` site.
- `dsl-kit-macros` — `#[derive(DslSchema)]` sets `optional: true`
  automatically for payload fields typed `Option<T>` / `Vec<T>`
  (where `T` is not the enum itself). `#[derive(DslBuild)]` routes
  those types to new `build_field_optional` / `build_field_vec`
  helpers by default (no `#[dsl_build(with = ...)]` needed for
  plain `Option<String>` / `Vec<String>` / any `Option<T>` /
  `Vec<T>` where `T: DeserializeOwned` + `FromStr` for `Option`).
  `#[dsl_build(with = path)]` on an optional payload short-circuits
  to `None` / `vec![]` when the field is absent, so hand-written
  converters no longer need to defend against missing keys.
- `dsl-kit-parse` — `check_conformance` skips the `MISSING_FIELD`
  diagnostic for `optional: true` fields; the pair-hint pass in
  `missing_slot_names` skips them too so `did you mean X (missing)`
  never mislabels an optional slot. New public helpers
  `build_field_optional::<T>` / `build_field_vec::<T>` handle every
  canonical shape of absence: missing field, JSON `null`, canonical
  text `none`, empty bracketed list.
- `dsl-kit-parse::schema_gen` — built-in canonical-syntax mappings
  added for `Option<String>` (`none` | `%str`) and `Vec<String>`
  (`[ %str_raw ("," %str_raw)* ]`, empty list allowed). Variants
  that carry at least one optional field emit a permissive
  argument-list form (`(arg ("," arg)*)?`) so authors and AI
  emitters may omit any subset of optional args from the canonical
  text; `check_conformance` remains the authority on required /
  duplicate / unknown-slot diagnostics.
- `dsl-kit-parse::peg` — new `%str_raw` token: same match as
  `%str` but contributes the raw source slice (quotes + escape
  sequences intact) so `build_field_vec` can hand the joined field
  text straight to `serde_json::from_str`.
- `examples/flow-example/crates/flow-dsl` — `Par::reducer_id`
  (`Option<String>`) loses its `#[dsl_build(with = parse_reducer_id)]`
  attribute, `parse_reducer_id` is removed, and
  `flow_syntax_overrides` sheds its `Option<String>` entry. The
  built-in Layer 2 mapping now covers the shape end-to-end. Only
  `policy: Option<JoinPolicy>` (a non-`String` `Option`) still
  needs its `SyntaxOverrides` value production and `parse_policy`
  converter.

Default `DslHost` implementations for call-less DSLs — closes GH
issue #3.

- `dsl-kit-mcp` — `DslHost::resolve` and `DslHost::step_to_done`
  gain default bodies suited to hosts whose DSL never suspends on
  external calls: `resolve` returns the new
  `RESOLVE_UNSUPPORTED_MSG` constant, `step_to_done` drives
  `step_to_yield` (returning on `Done` / `Suspended`, so breakpoint
  and suspend semantics match `step_to_yield` exactly) up to a
  configurable `step_budget()` (default `4096`) before erroring
  with a standardized budget-exceeded message. New
  `supports_calls()` hook (default `true`): hosts that override it
  to `false` get `dsl_kit_resolve` / `dsl_kit_resolve_by_id` gated
  at the handler with the same `RESOLVE_UNSUPPORTED_MSG` error.
  Existing hosts that override both methods are unaffected.

Normalized type names in schema / diagnostics — closes GH issue #4.

- `dsl-kit-macros` — `#[derive(DslSchema)]` normalizes the
  `FieldSchema.ty` source text captured from the token stream, so
  types render idiomatically (`Option < String >` →
  `Option<String>`, `HashMap < String , u32 >` →
  `HashMap<String, u32>`). Every consumer (BuildError diagnostics,
  exported schema JSON, generated docs / examples) inherits the
  tidied spelling. Consumers matching on the previous spaced form
  should update; parse-side comparisons were already
  whitespace-insensitive via `strip_ws`.

Owned AST projection so long-lived hosts avoid `Box::leak` — closes
GH issue #2. Stronger than the issue's original `Arc` sketch: the
projection needs neither `Arc` nor `unsafe`, and the resulting engine
is genuinely `'static` (it borrows nothing), so a host can hold its
program and engine in one struct and drop / replace the program on
its own terms.

- `dsl-kit-core` — new `OwnedDerivedAst<L, S>`, the owned counterpart
  of `DerivedAst<'a, N, S>`. `OwnedDerivedAst::new(&root, sem)` walks
  the tree once and projects each node's `NodeKind` classification and
  literal payload into an owned side table; the borrow of `root` ends
  when the constructor returns, so the value carries no lifetime.
  Implements `Ast` under the same bounds as `DerivedAst`
  (`L: Clone + Debug`, `S: DslSemantics`, `S::Value: From<L>`) and is
  `Clone` when `L` and `S` both are, letting a host keep a pristine
  copy to rebuild from on reset. `DerivedAst` is unchanged and remains
  the right choice for a transient engine that lives no longer than the
  program it walks — this is purely additive.
- `dsl-kit` — `OwnedDerivedAst` is re-exported through the crate's
  wholesale `pub use dsl_kit_core::*` facade.
- `dsl-kit-cli` — the built-in reference host (`RefHost`) now owns its
  `Ref` program by value and builds its engine over `OwnedDerivedAst`;
  the two `Box::leak` sites (default program + `load_json`) are gone,
  and `load_json` drops the previous program instead of leaking.
- `examples/expr-example` — `ExprAst` is now
  `OwnedDerivedAst<<Expr as DslExec>::LitValue, ExprSemantics>` (no
  lifetime); `ExprHost` owns its `Expr` and drops both `Box::leak`
  sites. `expr_engine` returns a `'static` `Engine<ExprAst>`.
- `examples/flow-example` — worked example of the hand-written owned
  projection for a non-derive AST: `FlowAst` drops its lifetime
  parameter and stores each node's `NodeKind` by value (via the new
  `flow_node_kind` helper) instead of borrowing `&Flow`; `FlowStepper`
  and `FlowHost` shed their lifetimes and `FlowHost` owns its `Flow`
  program with no `Box::leak`.

## [0.2.0] - 2026-07-22

Fuzzy-match `did you mean X?` hints wired through every `unknown-*`
diagnostic in the kit, plus a new plugin crate that ships the
similarity algorithm.

- `dsl-kit-fuzzy` — new leaf crate. `FuzzySuggester` implements the
  core `Suggester` trait via `strsim ^0.11` with Jaro-Winkler default,
  Levenshtein / Damerau-Levenshtein switchable, `threshold` 0.7, and
  `max_results` 3. Depends on `dsl-kit-core` only; other kit crates
  never depend on it, so pulling in a similarity algorithm stays
  opt-in at the composition root.
- `dsl-kit-core` — new `suggest` module owns the shared contract:
  `Suggester` trait (`suggest` + `enrich_unknown` default formatter),
  `NoopSuggester` zero-cost default, `Suggestion { candidate, score }`,
  `SuggesterHandle = Arc<dyn Suggester>`, and a `noop_handle()`
  constructor. The trait is `&[&str]`-only so plugins do not need to
  depend on `NodeSchema` / `OpRegistry` / any downstream type.
- `dsl-kit-core` — `OpRegistry` and `ReducerRegistry` gain
  `with_suggester(SuggesterHandle) -> Self`. `EngineError::UnknownOp`
  and `EngineError::UnknownReducer` grow a `hint: String` field:
  empty by default preserves the historical `unknown op OpId(...)` /
  `unknown reducer ReducerId(...)` wording, populated by
  `resolve()` when a suggester is injected. `matches!` patterns keep
  working; explicit `EngineError::UnknownOp { id }` constructors need
  to add `hint: String::new()`.
- `dsl-kit-parse` — every `unknown-*` diagnostic now routes through
  the trait. `check_conformance` / `from_json_value` / `from_json_str`
  keep their signatures and use an internal `BuiltinLevenshteinSuggester`
  (case-insensitive Levenshtein, the same algorithm the crate has
  always shipped); new `check_conformance_with` / `from_json_value_with`
  / `from_json_str_with` variants accept a `&dyn Suggester` for
  injection. `check_schema_consistency_with` is the same idea on the
  grammar-check side. `UNKNOWN_FIELD` / `UNKNOWN_CHILD` also mark a
  suggested candidate `(missing)` when that slot is declared but
  currently absent — the pair-hint the design brief calls for on
  typo pairs like `taget` -> `target`. PEG's undefined-rule /
  unresolved-start-rule diagnostics get the same enrichment.
- `dsl-kit-mcp` — `DslMcpHandler` and `DslMcpBuilder` (flowed into
  `DslMcpServer`) gain `with_suggester(SuggesterHandle) -> Self`.
  `dsl_kit_step` unknown `mode`, `dsl_kit_explain` unknown `code`
  (which now uses the compact `did you mean` form when the suggester
  fires, falling back to the full `Known codes:` dump otherwise), and
  `DslMcpServer::call_tool` unknown tool name all carry hints. Default
  is `noop_handle()`, so the crate does not pull in a similarity
  algorithm on its own.
- `dsl-kit-lint` — new opt-in `TypoHint` rule. Takes a caller-supplied
  extractor closure `Fn(&A) -> Vec<(NodeId, String)>` that decides
  what "labels" mean for the DSL; `with_suggester(SuggesterHandle)`
  wires in the plugin. Reports `Severity::Info` diagnostics named
  `"typo-hint"` for every extracted label that is a near-miss but
  not an exact schema variant name. Not registered by
  `Linter::with_defaults` — same policy as `DeadVariants`.

## [0.1.0] - 2026-07-22

Initial release.

- `dsl-kit-core` — engine primitives: frame arena with an explicit spawn
  schedule, `Seq` / `Par` / `Call` / `Scope` / `Maybe` interpretation,
  structured fan-out with join policies and reducers, cooperative
  cancellation, event stream, breakpoints that halt before a frame
  spawns, and a sans-io drive layer (`drive` / `drive_async`).
- `dsl-kit-core` — value semantics in the engine: `Apply` (pure op fold
  over an `OpRegistry`), `Branch` (value-dependent control flow via a
  `truthy` hook; the untaken side never spawns), `Bind` / `Read`
  (lexical env chain; unbound reads suspend as Call-shaped pendings the
  host answers through `resolve`), `Loop` (per-iteration respawn
  through the spawn schedule, so breakpoints halt on every iteration
  boundary), and `Lit` (literal leaves). Effects compose into value
  positions: an `Apply` argument can be a `Call` or a whole `Par`
  cascade.
- `dsl-kit-core` — `DslSemantics` + `DerivedAst`: the semantic half a
  DSL author writes (value type, unit, truthiness, binding storage)
  zipped with the derived classification into an engine-ready `Ast`.
- `dsl-kit-macros` — `#[derive(DslNode)]` (traversal), `#[derive(DslSchema)]`
  (type-level schema extraction), `#[derive(DslBuild)]` (ParseTree →
  typed AST, with `#[dsl_build(with = ...)]` per-field converters),
  `#[derive(DslExec)]` (variant → engine `NodeKind` classification via
  `#[dsl_exec(...)]` annotations: `value` / `read` / `apply` / `bind` /
  `branch` / `repeat` / `seq` / `scope` / `maybe` / `call`).
- `dsl-kit-schema` — schema reflection types consumed by parsers,
  editors, and AI clients.
- `dsl-kit-parse` — parser trunk: `ParseTree`, schema conformance,
  JSON front-end (`serde_bridge`), PEG interpreter, grammar generation
  from schemas (`schema_gen`, with per-field `SyntaxOverrides`), static
  grammar checks (`grammar_check`), and parse-guaranteed example
  synthesis (`example_gen`).
- `dsl-kit-lint` — walk-driven, schema-aware, author-extensible lint
  framework.
- `dsl-kit-mcp` — stdio MCP framework exposing a debugger-style tool
  surface (`step` / `breakpoint` / `state` / `resolve` / `schema` /
  `lint`) over any `DslHost`, plus a `DslMcpBuilder` path for authors
  who want to mix typed-fn tools (schemars-derived input schemas) with
  DSL-backed tools (`tool_from_host`) in one server. Both paths emit
  MCP 2024-11-05 spec-compliant `inputSchema` (`type: "object"` even
  for no-arg tools), so strict clients like Claude Code accept the
  whole tools/list response.
- `dsl-kit` — facade crate re-exporting the kit surface.
- `dsl-kit-cli` — developer entry-point binary shipping a `mcp`
  subcommand: `cargo install dsl-kit-cli && dsl-kit-cli mcp` starts a
  stdio MCP server hosting a minimal built-in reference DSL (Lit / Var
  / Add / Mul / Let / If) through the `dsl-kit-mcp` framework, so
  developers get a working stepper without writing a host first. Room
  for more subcommands (lint / schema / grammar) as the CLI grows.
- Examples (not published): `flow-example` (orchestration DSL with
  engine round trip), `expr-example` (expression DSL with text and JSON
  round trips, interpreted entirely by the engine — no hand-written
  evaluator), `custom-mcp-example` (shipping a DSL as its own MCP
  server).
