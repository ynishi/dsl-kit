//! OpenAPI 3.1 document for the query language.
//!
//! The JSON Schema of the query itself comes from the kit
//! (`dsl_kit_schema::json_schema`, [`RefStyle::OpenApiComponent`]) —
//! rendered from `Query::schema()`, so the published request-body
//! contract cannot drift from the enum. This module only places it in
//! a document with the two API surfaces (JSON body, canonical-text
//! `q` parameter) and attaches the examples.

use dsl_kit_schema::NodeSchema;
use dsl_kit_schema::json_schema::{Error, RefStyle, TypeMap};
use serde_json::{Map, Value, json};

/// Inputs for [`document`] beyond the schema itself.
#[derive(Debug, Clone)]
pub struct ApiInfo {
    /// `info.title`.
    pub title: String,
    /// `info.version`.
    pub version: String,
    /// Path of the search operation, e.g. `/items/search`.
    pub path: String,
    /// Example text-form queries for the `q` parameter (typically the
    /// kit's `example_gen` output).
    pub text_examples: Vec<String>,
    /// Example JSON-form queries for the request body.
    pub json_examples: Vec<Value>,
}

/// Renders a complete OpenAPI 3.1 document exposing `schema` as the
/// query language of one search operation, on two surfaces: a JSON
/// request body (`POST`) and a canonical-text `q` parameter (`GET`).
pub fn document(schema: &NodeSchema, types: &TypeMap, info: &ApiInfo) -> Result<Value, Error> {
    let component = schema.to_json_schema(types, RefStyle::OpenApiComponent)?;
    let body_ref = json!({ "$ref": format!("#/components/schemas/{}", schema.name) });
    let text_examples: Map<String, Value> = info
        .text_examples
        .iter()
        .enumerate()
        .map(|(i, t)| (format!("example{i}"), json!({ "value": t })))
        .collect();
    let json_examples: Map<String, Value> = info
        .json_examples
        .iter()
        .enumerate()
        .map(|(i, v)| (format!("example{i}"), json!({ "value": v })))
        .collect();
    Ok(json!({
        "openapi": "3.1.0",
        "info": { "title": info.title, "version": info.version },
        "paths": {
            info.path.clone(): {
                "get": {
                    "summary": "Search with a canonical-text query",
                    "parameters": [{
                        "name": "q",
                        "in": "query",
                        "required": true,
                        "description": "Query in the canonical text syntax generated from the schema.",
                        "schema": { "type": "string" },
                        "examples": text_examples,
                    }],
                    "responses": { "200": { "description": "Matching rows" } }
                },
                "post": {
                    "summary": "Search with a JSON query",
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": {
                            "schema": body_ref,
                            "examples": json_examples,
                        }}
                    },
                    "responses": { "200": { "description": "Matching rows" } }
                }
            }
        },
        "components": { "schemas": { schema.name.clone(): component } }
    }))
}
