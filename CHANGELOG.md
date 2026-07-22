# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
