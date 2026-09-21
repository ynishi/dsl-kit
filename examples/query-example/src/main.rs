//! Runs the query DSL through the Web API shape end to end:
//!
//! 1. a JSON request body → typed `Query` → parameterized SQL;
//! 2. the same query as a `?q=` text parameter through the
//!    schema-generated grammar → the same SQL;
//! 3. the OpenAPI document a server would serve, with the JSON Schema
//!    of the query language and machine-derived examples;
//! 4. a malformed request, to show the diagnostics a client sees.

use dsl_kit_core::IdGen;
use dsl_kit_parse::{dump_canonical_json, example_gen};
use dsl_kit_schema::DslSchema as _;
use dsl_kit_schema::json_schema::TypeMap;
use query_example::openapi::ApiInfo;
use query_example::{Query, grammar, openapi, query_from_json, query_from_text, to_sql};

const COLUMNS: &[&str] = &["name", "age", "status", "deleted_at"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ids = IdGen::new();

    // ---- 1. JSON body ------------------------------------------------
    let body = serde_json::json!({
        "type": "And",
        "items": [
            { "type": "Eq", "field": "status", "value": "active" },
            { "type": "Gt", "field": "age", "value": 18 },
            { "type": "Not", "body": { "type": "IsNull", "field": "deleted_at" } },
            { "type": "Or", "items": [
                { "type": "Like", "field": "name", "pattern": "A%" },
                { "type": "In", "field": "name", "values": ["bob", "eve"] }
            ]}
        ]
    });
    let query = query_from_json(&body, &ids).map_err(diag)?;
    let sql = to_sql(&query, COLUMNS)?;
    println!("=== JSON body → SQL ===");
    println!("WHERE {}", sql.text);
    println!("params: {:?}", sql.params);

    // ---- 2. text parameter ------------------------------------------
    let grammar = grammar(&ids).map_err(diag)?;
    let q = r#"And(items: [
        Eq(field: "status", value: "active"),
        Gt(field: "age", value: 18),
        Not(body: IsNull(field: "deleted_at")),
        Or(items: [Like(field: "name", pattern: "A%"), In(field: "name", values: ["bob", "eve"])])
    ])"#;
    let from_text = query_from_text(&grammar, q, &ids).map_err(diag)?;
    let sql_text = to_sql(&from_text, COLUMNS)?;
    println!("\n=== ?q= text → SQL ===");
    println!("WHERE {}", sql_text.text);
    assert_eq!(sql, sql_text, "both surfaces must lower identically");
    println!("(identical to the JSON path)");

    // ---- 3. OpenAPI document ----------------------------------------
    let examples = example_gen::examples_from_grammar(&grammar).map_err(diag)?;
    let types = TypeMap::new().with(
        "Scalar",
        serde_json::json!({ "oneOf": [
            { "type": "string" }, { "type": "integer" }, { "type": "boolean" }
        ]}),
    );
    let doc = openapi::document(
        &Query::schema(),
        &types,
        &ApiInfo {
            title: "Items API".into(),
            version: "1.0.0".into(),
            path: "/items/search".into(),
            text_examples: examples
                .per_rule
                .iter()
                .map(|e| e.text.clone())
                .chain(std::iter::once(examples.composite.clone()))
                .collect(),
            json_examples: vec![dump_canonical_json(&query).map_err(diag)?],
        },
    )?;
    println!("\n=== OpenAPI ===");
    println!("{}", serde_json::to_string_pretty(&doc)?);

    // ---- 4. a bad request -------------------------------------------
    println!("\n=== malformed request ===");
    let bad = serde_json::json!({ "type": "Eq", "feild": "name", "value": "x" });
    if let Err(e) = query_from_json(&bad, &ids) {
        for d in &e.diagnostics {
            println!("[{}] {}", d.code, d.message);
        }
    }
    if let Err(e) = query_from_text(&grammar, r#"Eq(field: "name" value: 1)"#, &ids) {
        for d in &e.diagnostics {
            println!("[{}] {}", d.code, d.message);
        }
    }
    Ok(())
}

fn diag(e: dsl_kit_parse::BuildError) -> Box<dyn std::error::Error> {
    let lines = e
        .diagnostics
        .iter()
        .map(|d| format!("[{}] {}", d.code, d.message))
        .collect::<Vec<_>>()
        .join("\n");
    lines.into()
}
