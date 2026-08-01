# dsl-kit — Intro

`dsl-kit` is an AI-native DSL engine kit: a small set of Rust crates that
let you build a domain-specific language whose evaluator is observable,
steppable, and debuggable through an MCP surface out of the box.

## What the kit gives you

- **Observation primitives** — `NodeId`, `Path`, `NodeContext`, `Event`,
  `SuspendReason`. Every node an evaluator visits carries a stable id and
  a root-to-node path; every yield to the outside world names its reason.
- **Stepper model** — evaluators are state machines (`Stepper` /
  `AsyncStepper`) driven from the outside. External effects surface as
  `Suspended { reason: AwaitEffect, .. }` yields; a host resolves them
  and resumes.
- **Structured errors** — `EngineError` variants carry `NodeContext`
  plus stable miette diagnostic codes (`dsl_kit::eval::aborted`, …). The
  MCP surface exposes them through `dsl_kit_explain`.
- **Traversal + breakpoints** — `#[derive(DslNode)]` generates `Walk` /
  `WalkMut`; `BreakpointSet` composes conditions over `NodeContext` so
  agents can synthesise breakpoints from the observable stream alone.
- **MCP surface** — `DslMcpHandler` speaks to any `DslHost`
  implementation with a fixed, DSL-neutral debugger tool surface;
  `DslMcpBuilder` is the light framework for hand-rolled custom MCP
  servers.

## Two audiences, two resource layers

- **`dsl-kit://kit/*`** — for people **building with** the kit. Explains
  primitives, the `DslHost` trait, the MCP tool surface, and the error
  catalog. Shipped by `dsl-kit-mcp` itself.
- **`dsl-kit://dsl/*`** — for AI or humans **writing programs in** the
  DSL a host has loaded. Contributed by the current `DslHost` via
  `DslHost::resources()`; content varies per host.

Custom builders that don't want to expose the kit-side guides through
their own server can opt out via `.without_kit_resources()` on both
`DslMcpHandler` and `DslMcpBuilder`.

## Where to go next

- `dsl-kit://kit/dsl-host-authoring` — how to implement `DslHost`
  around your own DSL.
- `dsl-kit://kit/mcp-tool-reference` — the MCP tool surface, grouped by
  purpose.
- `dsl-kit://kit/error-catalog` — every `EngineError` code with its
  help text, generated fresh from the enum.
