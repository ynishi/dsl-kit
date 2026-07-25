# cfg-example — the keyed-slot reference DSL

`Cfg` is a small configuration language whose children are **named**
rather than positional: `Env` and `Overrides` hold
`BTreeMap<String, _>` child slots, which the kit reports as
`multiplicity: "map"`. It is the worked example for the keyed `Map`
primitive — the shape `flow-dsl` and `expr-dsl` never exercise.

```text
Cfg ::= Env       { id, bindings: BTreeMap<String, Box<Cfg>> }  -- keyed, boxed
      | Overrides { id, entries:  BTreeMap<String, Cfg> }       -- keyed, bare
      | Ref       { id, name:  String }                         -- suspends
      | Leaf      { id, value: String }                         -- literal
```

A document reads the same in text and in JSON; braces mark a keyed
slot, and entries are canonicalised to ascending key order on ingest:

```text
Env(bindings: { app: Leaf(value: "dsl-kit"), port: Ref(name: "PORT") })
```

`Env` evaluates its bindings in key order and resolves to the last
one; `Overrides` folds its layers through the registered `last_wins`
op, so `20-prod` beats `10-base`; `Ref` suspends until the host
supplies a value. `Env` boxes its values and `Overrides` does not —
same schema, both derive arms covered.

## Run it

```sh
cargo run -p cfg-example        # document, schema, text round trip, DslHost run
cargo test -p cfg-dsl -p cfg-host
```

## Serve it over MCP

```sh
cargo install --path examples/cfg-example/crates/cfg-mcp
```

Register the installed `cfg-mcp` binary with your MCP host and restart
it. `dsl_kit_schema` then reports the keyed slots, `dsl_kit_load`
accepts a keyed JSON document, and `dsl_kit_step` / `dsl_kit_resolve`
walk it — the same tool surface `flow-mcp` and `expr-mcp` expose.

Tracking issue: [#7](https://github.com/ynishi/dsl-kit/issues/7)
(the `examples/` + MCP-surface layer of the keyed `Map` primitive
landed in [#5](https://github.com/ynishi/dsl-kit/issues/5)).
