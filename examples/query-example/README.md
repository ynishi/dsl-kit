# query-example — IR + parser only

`Query` is a filter language for a Web API. It is the reference for
using the kit **without the engine**: the Rust enum is the IR, the kit
supplies both front-ends, conformance, `DslDump`, the synthesized
examples and the JSON Schema — and the typed AST goes to a SQL
lowering instead of the interpreter.

```text
Query ::= And    { id, items: Vec<Query> }            -- non_empty
        | Or     { id, items: Vec<Query> }            -- non_empty
        | Not    { id, body: Box<Query> }
        | Eq     { id, field: String, value: Scalar } -- string | integer | boolean
        | Gt     { id, field: String, value: i64 }
        | Lt     { id, field: String, value: i64 }
        | Like   { id, field: String, pattern: String }
        | In     { id, field: String, values: Vec<String> }
        | IsNull { id, field: String }
```

The same query arrives on two surfaces and lowers to the same
parameterized clause (values never enter the SQL text):

```text
POST /items/search   {"type": "And", "items": [{"type": "Eq", "field": "status", "value": "active"}, …]}
GET  /items/search?q=And(items: [Eq(field: "status", value: "active"), …])

WHERE (status = $1 AND age > $2 AND NOT deleted_at IS NULL AND (name LIKE $3 OR name IN ($4, $5)))
```

`Eq.value` shows the payload-type hook pair in the parser-only setting:
`SyntaxOverrides` gives the text grammar a JSON-literal production,
`#[dsl_build(with)]` reads both front-ends into `Scalar`,
`#[dsl_dump(with)]` is its dual, and a `TypeMap` entry tells the JSON
Schema what the field accepts.

The OpenAPI 3.1 document (`openapi` module) embeds
`Query::schema().to_json_schema(..)` as `components.schemas.Query` and
attaches the machine-derived text examples to the `q` parameter.
`tests/e2e.rs` pins the contract: the JSON Schema and the serde bridge
agree on every accept / reject case, the canonical dump validates and
re-parses, and every synthesized example builds.

## Run it

```sh
cargo run -p query-example     # JSON → SQL, text → SQL, OpenAPI document, diagnostics
cargo test -p query-example
```
