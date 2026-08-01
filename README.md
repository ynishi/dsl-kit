# dsl-kit

An engine kit for building small, AI-native DSLs in Rust.

Write your AST as a Rust enum and derive the rest: from a few derives,
the kit supplies a schema, a text parser, a JSON front-end, a typed
builder, lints, a debugger-grade interpreter that executes the enum
directly, and an MCP surface an AI can drive. The Rust type is the
single source of truth — everything user-facing (grammar, examples,
diagnostics, tool schemas, execution) is generated from it, so none of
it can drift.

```rust
use dsl_kit::{IdGen, NodeId};
use dsl_kit_macros::{DslBuild, DslExec, DslNode, DslSchema};

#[derive(Debug, DslNode, DslSchema, DslBuild, DslExec)]
enum Expr {
    #[dsl_exec(value)]
    Lit { id: NodeId, value: i64 },
    #[dsl_exec(apply = "add")]
    Add { id: NodeId, lhs: Box<Expr>, rhs: Box<Expr> },
    #[dsl_exec(branch)]
    If { id: NodeId, cond: Box<Expr>, then_branch: Box<Expr>, else_branch: Box<Expr> },
}
```

That single declaration already gives you:

```rust
use dsl_kit_parse::{DslBuild as _, schema_gen};
use dsl_kit_schema::DslSchema as _;

// A grammar for a canonical text syntax, generated from the schema —
// left-recursion-free by construction and verified by static checks.
let grammar = schema_gen::checked_grammar_from_schema(&Expr::schema(), &IdGen::new())?;

// Parse canonical text, then build the typed AST back out of it.
let tree = grammar.parse("Add(lhs: Lit(value: 40), rhs: Lit(value: 2))")?;
let expr = Expr::from_parse_tree(&tree, &IdGen::new())?;

// Machine-derived examples that parse by construction — few-shot
// material for an AI writing your DSL.
let examples = dsl_kit_parse::example_gen::examples_from_grammar(&grammar)?;
```

A JSON front-end (`{"type": "Add", "lhs": ...}`) targets the same
`ParseTree`, so both surfaces share one conformance check and one typed
builder. Payload types the canonical syntax cannot spell get per-field
hooks on both sides: `schema_gen::SyntaxOverrides` for the grammar,
`#[dsl_build(with = ...)]` for the builder.

## The engine

`dsl-kit-core` interprets effect-oriented, value-bearing programs —
`Seq` / `Par` / `Call` / `Scope` / `Maybe` for orchestration, `Apply` /
`Branch` / `Bind` / `Read` / `Loop` / `Lit` for computation — with the
runtime mechanics a real host needs:

- **Structured fan-out**: `Par` with join policies (`All` / `Any` /
  `FirstK`, fail-fast or collect-all) folded by pluggable reducers.
- **Suspend / resume**: every `Call` yields a suspension the host
  resolves with `Ok(value)` or `Err(effect_error)`; cancellation is
  cooperative and observable.
- **Debugger-grade breakpoints**: every spawn passes through one
  schedule, so a breakpoint halts *before* the matched frame spawns —
  at any depth, including mid-`Par`-cascade — and the frame tree
  faithfully exposes the partially spawned state.
- **Sans-io drive layer**: `drive` (sync) and `drive_async` (any
  executor; the core has zero runtime dependencies) run the
  resolve loop for you when you don't want to hand-roll it.
- **Value semantics, interpreted**: `Apply` folds evaluated children
  through a registered op, `Branch` selects on a value (the untaken
  side never spawns), `Bind` / `Read` carry a lexical env chain,
  `Loop` respawns its body per iteration — all inside the same frame
  machinery, so breakpoints halt on branch selection and on every
  loop iteration, and an unbound `Read` suspends like any other
  effect. You register what `"add"` *means*; the engine does the
  rest. No hand-written eval loop — `expr-example` runs entirely on
  the engine.

## MCP: let an AI debug your DSL

`dsl-kit-mcp` wraps any `DslHost` in a stdio MCP server with a
debugger-style tool surface: `load`, `ast`, `step` (one / to-yield /
to-done), `breakpoint add/list/remove`, `state`, `pending`, `resolve`,
`schema`, `lint`, `explain`. The `custom-mcp-example` shows how to ship
your own DSL as its own MCP server binary.

## Conformance vs lint

Two layers inspect a document, and only one of them can stop it.

**Conformance is the compiler.** The `from_parse_tree` that
`#[derive(DslBuild)]` generates runs `check_conformance` before it
builds anything, so a non-conforming document never becomes an AST: an
unknown variant, a missing or duplicated field, a slot that breaks its
declared multiplicity, an empty slot declared
`#[dsl_schema(non_empty)]` — every one of them is a hard error.
Neither the document nor the host can suppress them, and both
front-ends (canonical text and JSON) go through the same check.

**Lint is advisory and opt-in.** `Linter` sits outside parsing and
loading: it runs only where a host wires it up explicitly
(`DslHost::lint_json`, surfaced over MCP as the `dsl_kit_lint` tool).
A document with lint findings still loads and still runs — findings
are material for an author or an agent to act on, not a gate. A DSL
that never wires a linter simply reports itself as unwired.

There is no ignore-file layer. The levers — declare the constraint in
the schema, compose the rule set, filter the output — are documented
on the `dsl-kit-lint` crate.

## Crates

| crate | role |
|---|---|
| `dsl-kit` | facade — re-exports the kit surface |
| `dsl-kit-core` | engine: frames, fan-out, cancellation, events, breakpoints, drive |
| `dsl-kit-macros` | `#[derive(DslNode)]` / `#[derive(DslSchema)]` / `#[derive(DslBuild)]` / `#[derive(DslExec)]` |
| `dsl-kit-schema` | type-level schema consumed by parsers, editors, AI clients |
| `dsl-kit-parse` | `ParseTree`, conformance, JSON bridge, PEG interpreter, grammar generation, example synthesis |
| `dsl-kit-lint` | walk-driven, schema-aware, author-extensible lints |
| `dsl-kit-mcp` | stdio MCP framework over any `DslHost` |

## Examples

```sh
cargo run -p expr-example   # expression DSL: text + JSON round trips, schema-generated grammar
cargo run -p flow-example   # orchestration DSL: fan-out, value-gated Branch, breakpoints, drive layer, text round trip
cargo run -p cfg-example    # configuration DSL: keyed child slots (BTreeMap), override folds, host-resolved references
```

The examples double as reference implementations: `flow-dsl` shows the
per-field hooks on a DSL with exotic payload types, `expr-dsl` shows
the plain derive-only path end to end, and `cfg-dsl` shows keyed child
slots — named children rather than positional ones — through both
front-ends and over MCP (see
[examples/cfg-example](examples/cfg-example/README.md)).

## Documentation

API documentation is the source of truth and is written to be read:

```sh
cargo doc --open
```

## Status

The kit is young and the API is still moving. The engine semantics
(fan-out, cancellation, halt-before-spawn breakpoints) and the derive
contracts are test-pinned; surface ergonomics may change between minor
versions. See [CHANGELOG.md](CHANGELOG.md) for the release history.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
