# cfg DSL — grammar

The `cfg` DSL is the third reference DSL shipped with `dsl-kit`, and
the one that exists to exercise **keyed child slots**. Its AST is one
enum with four variants: two keyed containers and two leaves.

```text
Cfg ::= Env       { id, bindings: BTreeMap<String, Box<Cfg>> }
      | Overrides { id, entries:  BTreeMap<String, Cfg> }
      | Ref       { id, name:  String }
      | Leaf      { id, value: String }
```

`Env` and `Overrides` differ in Rust storage only — boxed values
versus bare — so the derive's two keyed arms (`MapBoxed` and `Map`)
both stay covered. In the schema both report
`"multiplicity": "map"`, and in text and JSON both read identically.

## Keyed slot syntax

A keyed slot is written with braces (positional lists use brackets):

```text
Env(bindings: { app: Leaf(value: "dsl-kit"), "log level": Leaf(value: "info") })
```

Keys are bare identifiers or quoted strings; quoting is what lets a
key hold a space, or be empty. The equivalent JSON is an object:

```json
{ "type": "Env", "bindings": { "app": { "type": "Leaf", "value": "dsl-kit" } } }
```

Entries are canonicalised to ascending key order on ingest, so two
documents that differ only in the order they were written build to the
same tree. A repeated key is an error (`dsl_kit::parse::duplicate_key`)
rather than a silently dropped subtree.

## Semantics

- **`Leaf`** — resolves to its literal string.
- **`Ref`** — a value the document does not carry. The engine looks it
  up, misses (the DSL has no binding form), and suspends as a
  Call-shaped pending labelled with the name, which the host answers
  through `dsl_kit_resolve`.
- **`Env`** — a `Seq` over its bindings in key order: every binding is
  evaluated, and the block resolves to the last one's value.
- **`Overrides`** — an `Apply` of the registered `last_wins` op over
  its layers in key order, so `20-prod` beats `10-base`. An empty
  stack folds to the empty string.

## Where the keys live

`Walk` iterates keyed slots by value, not by key — the engine sees an
ordered child sequence and nothing more. Anything that needs the names
reads them off the AST: `Cfg::keyed_children`, `pretty` (the tree
`dsl_kit_ast` returns) and `flatten` (dotted paths) are the three
places in `cfg-dsl` that do.

## Writing a cfg document

```rust,ignore
use std::collections::BTreeMap;
use cfg_dsl::Cfg;
use dsl_kit::IdGen;

let ids = IdGen::new();
let document = Cfg::Env {
    id: ids.node(),
    bindings: BTreeMap::from([(
        "port".to_string(),
        Box::new(Cfg::Ref { id: ids.node(), name: "PORT".into() }),
    )]),
};
// Resolving this suspends on `PORT` until the host supplies a value.
```

For a bigger worked example see
`dsl-kit://dsl/cfg/samples/demo-document`.

## Resolving references

`dsl_kit_resolve` takes `result` as the string value to substitute.
Omitting it falls back to the host default: `PORT` resolves to
`8080`, and any other name to `<unset:NAME>`. `CfgHost::step_to_done`
uses the same defaults when driving the document end to end.
