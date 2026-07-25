# cfg DSL — demo document

The default document `CfgHost` loads when it starts up: a two-level
`Env` whose `log` binding is an override stack, and whose `app` block
holds one unresolved reference.

```text
Env(bindings: {
  app: Env(bindings: { name: Leaf(value: "dsl-kit"), port: Ref(name: "PORT") }),
  log: Overrides(entries: { "10-base": Leaf(value: "info"), "20-prod": Leaf(value: "warn") }),
})
```

With `PORT = 8080` (`CfgHost`'s canned default) it resolves to
`"warn"`: the root `Env` is a `Seq`, so its value is the last binding
in key order (`log`), and that layer stack folds last-wins to
`20-prod`.

## Source

```rust,ignore
pub fn demo_document(ids: &IdGen) -> Cfg {
    let app = Cfg::Env {
        id: ids.node(),
        bindings: BTreeMap::from([
            ("name".to_string(), Box::new(Cfg::Leaf { id: ids.node(), value: "dsl-kit".into() })),
            ("port".to_string(), Box::new(Cfg::Ref  { id: ids.node(), name: "PORT".into() })),
        ]),
    };
    let log = Cfg::Overrides {
        id: ids.node(),
        entries: BTreeMap::from([
            ("10-base".to_string(), Cfg::Leaf { id: ids.node(), value: "info".into() }),
            ("20-prod".to_string(), Cfg::Leaf { id: ids.node(), value: "warn".into() }),
        ]),
    };
    Cfg::Env {
        id: ids.node(),
        bindings: BTreeMap::from([
            ("app".to_string(), Box::new(app)),
            ("log".to_string(), Box::new(log)),
        ]),
    }
}
```

## Structure

```text
Env (2 bindings)
├── app: Env (2 bindings)
│   ├── name: Leaf "dsl-kit"
│   └── port: Ref "PORT"
└── log: Overrides (2 layers)
    ├── 10-base: Leaf "info"
    └── 20-prod: Leaf "warn"
```

`dsl_kit_ast` returns the same shape with real `NodeId`s. Note that
the tree is printed with its keys: `Walk` alone would show the values
in the same order but without the names.

## The same document as JSON

`dsl_kit_load` accepts this verbatim:

```json
{
  "type": "Env",
  "bindings": {
    "app": {
      "type": "Env",
      "bindings": {
        "name": { "type": "Leaf", "value": "dsl-kit" },
        "port": { "type": "Ref", "name": "PORT" }
      }
    },
    "log": {
      "type": "Overrides",
      "entries": {
        "10-base": { "type": "Leaf", "value": "info" },
        "20-prod": { "type": "Leaf", "value": "warn" }
      }
    }
  }
}
```

## Driving it end-to-end

Only `Ref "PORT"` suspends; everything else is literal:

```text
1. dsl_kit_step { mode: "to_yield" }    # suspends at Ref "PORT"
2. dsl_kit_resolve { result: "8080" }   # supply the port
3. dsl_kit_step { mode: "to_yield" }    # Done with value "warn"
```

Or short-circuit the whole run:

```text
dsl_kit_step { mode: "to_done" }        # host supplies the canned PORT default
```

After `to_done` the final value (`warn`) and the resolved reference
appear under `dsl_kit_state.results`.
