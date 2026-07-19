# dsl-kit — Design Notes

An engine kit for building small, AI-native DSLs in Rust.

## Vision

`dsl-kit` is a framework for authoring embedded DSLs whose primary consumer is
an LLM agent, not a human at a keyboard. The author writes the AST and the
semantics; the kit provides the machinery around them — traversal, an async
stepper, structured events, and an MCP surface — so that an agent can generate
programs in the DSL, execute them step by step, inspect intermediate state,
and receive structured errors back through a validator.

The kit is deliberately small in scope. It targets DSLs up to roughly the
complexity of SQL (order of a hundred AST variants). Beyond that scale a
proper language workbench is the right tool, and `dsl-kit` steps out of the
way.

## Design Principles

1. **Author writes semantics; kit provides plumbing.** The author's code
   should describe what a node means, not how to trace it, schedule it, or
   expose it to an agent. Everything mechanical is derived or wired from
   the semantics definition.

2. **AI is the primary consumer of the resulting DSL.** The main promise of
   `dsl-kit` is that any DSL built with it comes with the affordances an
   LLM agent needs to use it well: a schema the agent can read, a
   validator that returns structured errors, and a stepper the agent can
   drive. Program generation, evaluation, and self-correction all happen
   through these surfaces, and the feedback loop between them is kept
   short by construction.

   As a secondary benefit, defining a DSL with the kit is also intended
   to be comfortable for an LLM. Both authoring paths (a single enum with
   a derive, or a set of small functions with reflected signatures) are
   shapes an agent can generate and modify without deep Rust expertise.
   This is a nice-to-have, not the primary goal.

3. **Async and stepping are engine primitives, not retrofits.** LLM calls,
   tool calls, and MCP calls are all awaited. The evaluator is a state
   machine driven by `step()` from the outside, with await points modelled
   as `Yield`. This is the same shape that a DAP-style debugger wants, and
   it composes cleanly with parallel branches and cancellation.

4. **Design for observation from the first commit.** Every AST node carries
   a stable ID; every call carries a frame ID and depth; every loop carries
   an iteration counter. Breakpoints, tracing, and replay are all
   expressible against these primitives. Adding them later has repeatedly
   proven infeasible; adding them up front costs almost nothing.

5. **Two facets, not three.** The kit exposes a Stepper MCP surface (drive
   evaluation, break, inspect) and a Schema / Validator MCP surface (parse,
   validate, describe). A language server is intentionally out of scope:
   the human-in-IDE use case is not the target, and the two remaining
   facets carry the LLM interaction end to end.

## Architecture

Two authoring styles are supported. Both compile down to the same engine
core; authors choose based on how they prefer to express semantics.

### Path α — Derive Macros (default)

The DSL is defined as a Rust `enum`. A single derive expands into the traits
the engine core needs.

```rust
#[derive(DslNode)]
enum FlowNode {
    Seq(Vec<FlowNode>),
    Par(Vec<FlowNode>),
    Call(Tool, Args),
    // …
}

impl Interpret for FlowEngine {
    async fn visit(&mut self, node: &FlowNode, env: &mut Env) -> Result<Value> {
        // semantics only
    }
}
```

The derive produces node IDs, a visitor, a JSON schema for the AST, and hooks
into the stepper and event stream. The author's `impl Interpret` block
contains only the meaning of each node.

This is the default path because a single enum plus a single impl block is
the smallest surface an author has to think about, and it is straightforward
for an LLM to generate and modify.

### Path γ — Function-Reflection (prototype-friendly)

For rapid iteration and for DSLs whose semantics are easier to express as a
collection of small functions, the kit accepts per-node evaluator functions
and wires them together by reflecting over their signatures. This borrows the
pattern popularised by Bevy's `IntoSystem`: parameters are dependency-injected
based on their types.

```rust
fn eval_seq(node: Node<Seq>, env: &mut Env, step: Stepper) -> Value { … }
fn eval_call(node: Node<Call>, env: &mut Env, step: Stepper, mcp: McpCtx) -> Value { … }
```

The engine inspects each function's parameters and provides the requested
context (stepper handle, MCP context, tracer, etc.). Authors can add
capabilities incrementally by extending signatures.

The two paths coexist; a single project can mix them, using derive for the
stable core and function-reflection for parts under active iteration.

### Engine Core

The core is small and fixed. It provides:

- **Node ID.** Assigned when the AST is built. Stable across runs for a
  given source, distinct from source location.
- **Call frame ID and depth.** Every function-like activation gets an ID;
  recursion is uniquely identified.
- **Iteration counter.** For each loop-shaped node, position within the
  current iteration.
- **Event insertion points.** `visit_pre`, `visit_post`, `suspend`, `resume`
  are trait methods; backends attach to them.
- **Path.** Chain of node IDs from root. Breakpoint conditions can be
  expressed against paths.
- **Stepper interface.** `step()` advances the evaluation by one node.
  Await points are modelled as `Yield`. Parallel branches are multiple
  steppers interleaved by a scheduler. Cancellation is a dropped stepper.

An agent can express a condition such as “break at node X where
`call_depth = 3` and `iteration = 5`” directly against these primitives.
Without them, every user has to reinvent them, and after-the-fact
reconstruction from the AST is expensive.

### Async Model

Evaluation is a stepper. Each `step()` runs one node's semantics; when the
semantics hit an `await` point (an LLM call, a tool call, an MCP call), the
stepper yields. External code chooses when to resume.

This model gives us four properties without extra work:

- The DAP-style step primitive maps one-to-one to the engine step.
- MCP `dsl_step` is a thin adapter over the same call.
- Parallel branches are multiple steppers scheduled cooperatively.
- Cancellation is dropping the stepper; no scattered cleanup logic.

Boxing every `visit` into `Pin<Box<dyn Future>>` is avoided; the stepper
holds the state that would otherwise be spread across future frames.

### MCP Surface

Two MCP-exposed tools cover the intended interaction:

- **Stepper tool.** `step`, `break`, `resume`, `inspect(node_id | path)`,
  `evaluate(expr)`. Drives the evaluator, exposes state.
- **Schema / Validator tool.** `parse(source)`, `validate(program)`,
  `describe_grammar()`. Returns structured errors an agent can act on
  without human parsing.

The schema is generated from the AST types (via `schemars`), and the tool
signatures are exposed through `rmcp` attributes. Nothing here is new; the
kit's contribution is that the same semantics definition drives both.

## Non-Goals

- General-purpose language runtime. `rhai`, `rune`, and `koto` cover that
  space well; the kit is not competing for that use case.
- JIT, garbage collection, or partial evaluation. Tree-walking with arena
  allocation is enough for the scale we target.
- Language server. Human-in-IDE is not the primary use case, and dropping
  LSP removes a large amount of surface without hurting the agent path.
- Language workbench features (projectional editor, refactoring tooling,
  multi-language embedding). A DSL of that scale should reach for a proper
  workbench.

## Naming

Working name: `dsl-kit`. The kit is small, opinionated, and aimed at
short-lived internal DSLs, so a short name that hints at scope fits.

## Related Work

The design draws on several precedents:

- **Truffle (GraalVM).** The idea that instrumentable AST nodes with an
  observable event stream can carry a debugger and profiler as derived
  artefacts. Truffle's Instrumentation API demonstrated that the split
  between semantics and observation is a productive one. The kit adopts
  the split but not the JVM-specific machinery around it.
- **Bevy `IntoSystem`.** Function-parameter reflection as a way to let
  authors add capabilities incrementally without touching a central
  registration table. Path γ is a direct application of this idea.
- **rust-analyzer / rowan / ungrammar.** For large languages, build-time
  code generation from a grammar description is the established path. The
  kit intentionally targets the smaller scale where a derive is enough,
  and defers to the workbench path when the scale is larger.
- **sqlparser-rs (`sqlparser_derive`).** In-the-wild evidence that a derive
  approach scales cleanly through a SQL-sized AST (roughly a hundred
  variants). This sets our target scale.
- **`derive-visitor`, `enum_dispatch`, `schemars`, `rmcp`, `tracing`.**
  Each of these covers one facet cleanly. The kit builds on them rather
  than replacing them, and its value is in the single semantics
  description that drives all of them at once.

Internally, two existing projects act as reference implementations for the
runtime side:

- **`mlua-isle`** — a stepper-based execution model for a Lua VM.
- **`mlua-probe`** — an MCP-facing debugger interface (`check_launch`,
  `test_launch`, `evaluate`) for that VM.

`dsl-kit` generalises the shape these two projects share.

## Roadmap

The initial milestones are small and each ends in something runnable.

1. **Engine core skeleton.** Node ID, call frame, iteration counter, event
   insertion points, and a stepper trait. No parser, no derive yet.
2. **First hand-written DSL.** A minimal expression language driven directly
   against the engine core. Confirms the primitives are sufficient.
3. **`#[derive(DslNode)]`.** Traversal and schema generation from the enum.
   Reuses `derive-visitor` / `schemars` where possible.
4. **Function-reflection path.** `IntoEvaluator` trait and dependency
   injection over parameter types. Interop with path α in the same project.
5. **MCP surface.** Stepper tool and schema / validator tool via `rmcp`.
6. **Reference DSL.** A small flow DSL (sequence, parallel, tool call) as
   both example and integration test. This is where we validate that an
   agent can generate, run, and correct a program end to end.

Each step is intended to stand on its own; if the kit is only ever used
through path α, that should be a complete experience.
