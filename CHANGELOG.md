# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-21

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
  `lint`) over any `DslHost`.
- `dsl-kit` — facade crate re-exporting the kit surface.
- Examples (not published): `flow-example` (orchestration DSL with
  engine round trip), `expr-example` (expression DSL with text and JSON
  round trips, interpreted entirely by the engine — no hand-written
  evaluator), `custom-mcp-example` (shipping a DSL as its own MCP
  server).
