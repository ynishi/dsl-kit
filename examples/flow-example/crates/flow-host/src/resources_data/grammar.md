# Flow DSL — grammar

The `flow` DSL is the reference DSL shipped with `dsl-kit`. Its AST is
one enum with five variants, all directly recursive.

```text
Flow ::= Seq   { id, children: Vec<Flow> }
       | Par   { id, children: Vec<Flow> }
       | Call  { id, label: String }
       | Scope { id, label: String, body: Box<Flow> }
       | Maybe { id, body: Option<Box<Flow>> }
```

## Semantics

- **`Seq`** — evaluate `children` in declaration order.
- **`Par`** — evaluate `children` concurrently in principle; the
  reference stepper schedules them sequentially, which is enough to
  demonstrate the event shape.
- **`Call`** — denotes an external effect. The stepper yields
  `Suspended { reason: AwaitEffect, .. }`; the host resolves it via
  `dsl_kit_resolve` and the stepper records the response against the
  node id.
- **`Scope`** — a labelled section wrapping one inner flow. No extra
  semantics beyond delineation, useful for pretty-printing and for
  path-shaped breakpoints.
- **`Maybe`** — optionally runs the inner flow. Absent body is a
  no-op.

## Node ids and traversal

Every variant carries a stable `id: NodeId`. `#[derive(DslNode)]`
generates `Walk` / `WalkMut` alongside the enum, so `flow.walk(...)` /
`flow.find_by_id(...)` work uniformly across variants.

The `pretty(flow)` free function uses `Walk` to produce the indented
tree the `dsl_kit_ast` MCP tool returns.

## Writing a flow program

```rust,ignore
use dsl_kit::IdGen;
use flow_dsl::Flow;

let ids = IdGen::new();
let program = Flow::Seq {
    id: ids.node(),
    children: vec![
        Flow::Call { id: ids.node(), label: "fetch_query".into() },
        Flow::Par {
            id: ids.node(),
            children: vec![
                Flow::Call { id: ids.node(), label: "search_arxiv".into() },
                Flow::Call { id: ids.node(), label: "search_semantic".into() },
            ],
        },
    ],
};
```

For a bigger worked example see
`dsl-kit://dsl/flow/samples/research-pipeline`.

## Call labels and canned responses

`FlowHost::step_to_done` resolves any `Call` yield by looking up a
canned response for its `label`. The reference set covers
`fetch_query`, `search_arxiv`, `search_github`, `search_web`,
`synthesise`, `citation_check`, `write_report`, and falls back to
`"<label>: (canned response)"` for unknown labels.
