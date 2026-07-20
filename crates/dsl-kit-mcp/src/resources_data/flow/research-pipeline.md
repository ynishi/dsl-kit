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

```text
1. dsl_kit_step { mode: "to_yield" }        # suspends at Call fetch_query
2. dsl_kit_resolve { result: "..." }        # supply an answer
3. dsl_kit_step { mode: "to_yield" }        # suspends at next Call
   ... repeat until Done ...
```

Or short-circuit the whole run:

```text
dsl_kit_step { mode: "to_done" }            # host resolves every Call with its canned response
```

After `to_done` the seven `Call` results are visible under
`dsl_kit_state.results`.
