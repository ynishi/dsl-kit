# Flow DSL — `research_pipeline` sample

The default program `FlowHost` loads when it starts up. Seven `Call`
nodes wrapped by a `Seq` with an inner `Par` (web research) and a
`Maybe` (optional citation check).

## Source

```rust,ignore
pub fn research_pipeline(ids: &IdGen) -> Flow {
    Flow::Seq {
        id: ids.node(),
        children: vec![
            Flow::Call { id: ids.node(), label: "fetch_query".into() },
            Flow::Scope {
                id: ids.node(),
                label: "web_research".into(),
                body: Box::new(Flow::Par {
                    id: ids.node(),
                    children: vec![
                        Flow::Call { id: ids.node(), label: "search_arxiv".into() },
                        Flow::Call { id: ids.node(), label: "search_github".into() },
                        Flow::Call { id: ids.node(), label: "search_web".into() },
                    ],
                }),
            },
            Flow::Call { id: ids.node(), label: "synthesise".into() },
            Flow::Maybe {
                id: ids.node(),
                body: Some(Box::new(Flow::Call {
                    id: ids.node(),
                    label: "citation_check".into(),
                })),
            },
            Flow::Call { id: ids.node(), label: "write_report".into() },
        ],
    }
}
```

## Structure

```text
Seq
├── Call fetch_query
├── Scope "web_research"
│   └── Par
│       ├── Call search_arxiv
│       ├── Call search_github
│       └── Call search_web
├── Call synthesise
├── Maybe
│   └── Call citation_check
└── Call write_report
```

`dsl_kit_ast` returns the same shape with real `NodeId`s and full
`Flow::summary()` labels.

## Driving it end-to-end

The pipeline has one `Par` node (`web_research`) with three `Call`
children. Real fan-out means those three children are dispatched in
one step and resolved individually via `dsl_kit_resolve_by_id`.

```text
1. dsl_kit_step   { mode: "to_yield" }              # suspends at Call fetch_query
2. dsl_kit_resolve { result: "..." }                 # answer fetch_query
3. dsl_kit_step   { mode: "to_yield" }              # enters Par → 3 pending emitted
4. dsl_kit_pending                                   # → 3 entries with ids I1, I2, I3
5. dsl_kit_resolve_by_id { id: I1, ok: "..." }       # resolve search_arxiv
   dsl_kit_resolve_by_id { id: I2, ok: "..." }       # resolve search_github
   dsl_kit_resolve_by_id { id: I3, ok: "..." }       # resolve search_web (any order)
6. dsl_kit_step   { mode: "to_yield" }              # Par folds, advances to next Call
7. dsl_kit_resolve { result: "..." }                 # synthesise
   ... repeat until Done ...
```

Or short-circuit the whole run:

```text
dsl_kit_step { mode: "to_done" }            # host resolves every Call with its canned response
```

After `to_done` the seven `Call` results are visible under
`dsl_kit_state.results`.

### FailFast on the Par

If instead of `ok` a client resolves one slot with an effect error,
the next `dsl_kit_step` propagates it and the siblings are queued for
cancellation:

```text
5'. dsl_kit_resolve_by_id { id: I2, err: { code: "timeout", message: "..." } }
6'. dsl_kit_step                              # → Err (flow effect error [timeout])
7'. dsl_kit_take_cancellations                # → { cancelled: [I1, I3, ...] }
```

Hosts should call `dsl_kit_take_cancellations` after any step that
returns an error and abort their runtime handles for the drained ids.
