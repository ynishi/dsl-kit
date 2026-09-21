//! End-to-end contract of the "IR + parser only" shape:
//!
//! - both API surfaces (JSON body, text `q`) build the same typed
//!   query and lower to the same parameterized SQL;
//! - the OpenAPI component schema rendered from `Query::schema()`
//!   accepts every document the JSON front-end accepts (the dumped
//!   canonical form of a parsed query) and rejects what the
//!   front-end rejects;
//! - values never reach the SQL text (injection goes into a
//!   parameter, not the clause);
//! - malformed requests surface as diagnostics, not panics.

use dsl_kit_core::IdGen;
use dsl_kit_parse::{DslBuild as _, check_conformance, dump_canonical_json, example_gen};
use dsl_kit_schema::DslSchema as _;
use dsl_kit_schema::json_schema::{self as jschema, RefStyle, TypeMap};
use query_example::openapi;
use query_example::{Query, Scalar, SqlError, grammar, query_from_json, query_from_text, to_sql};
use serde_json::{Value, json};

const COLUMNS: &[&str] = &["name", "age", "status", "deleted_at"];

fn types() -> TypeMap {
    TypeMap::new().with(
        "Scalar",
        json!({ "oneOf": [{ "type": "string" }, { "type": "integer" }, { "type": "boolean" }] }),
    )
}

/// The component schema, wrapped so `$ref: #/components/schemas/Query`
/// resolves inside a standalone validator.
fn validator() -> jsonschema::Validator {
    // Same component the OpenAPI document embeds, wrapped so the
    // `#/components/schemas/Query` recursion resolves standalone.
    let component = Query::schema()
        .to_json_schema(&types(), RefStyle::OpenApiComponent)
        .expect("renders");
    let root = json!({
        "$ref": "#/components/schemas/Query",
        "components": { "schemas": { "Query": component } },
    });
    jsonschema::validator_for(&root).expect("valid JSON Schema")
}

fn sample_body() -> Value {
    json!({
        "type": "And",
        "items": [
            { "type": "Eq", "field": "status", "value": "active" },
            { "type": "Eq", "field": "age", "value": 30 },
            { "type": "Gt", "field": "age", "value": 18 },
            { "type": "Not", "body": { "type": "IsNull", "field": "deleted_at" } },
            { "type": "Or", "items": [
                { "type": "Like", "field": "name", "pattern": "A%" },
                { "type": "In", "field": "name", "values": ["bob", "eve"] }
            ]}
        ]
    })
}

const SAMPLE_TEXT: &str = r#"And(items: [
    Eq(field: "status", value: "active"),
    Eq(field: "age", value: 30),
    Gt(field: "age", value: 18),
    Not(body: IsNull(field: "deleted_at")),
    Or(items: [Like(field: "name", pattern: "A%"), In(field: "name", values: ["bob", "eve"])])
])"#;

#[test]
fn json_and_text_surfaces_lower_to_the_same_sql() {
    let ids = IdGen::new();
    let from_json = query_from_json(&sample_body(), &ids).expect("json builds");
    let from_text =
        query_from_text(&grammar(&ids).unwrap(), SAMPLE_TEXT, &ids).expect("text builds");

    let a = to_sql(&from_json, COLUMNS).unwrap();
    let b = to_sql(&from_text, COLUMNS).unwrap();
    assert_eq!(a, b);
    assert_eq!(
        a.text,
        "(status = $1 AND age = $2 AND age > $3 AND NOT deleted_at IS NULL \
         AND (name LIKE $4 OR name IN ($5, $6)))"
    );
    assert_eq!(
        a.params,
        vec![
            Scalar::Str("active".into()),
            Scalar::Int(30),
            Scalar::Int(18),
            Scalar::Str("A%".into()),
            Scalar::Str("bob".into()),
            Scalar::Str("eve".into()),
        ]
    );
}

#[test]
fn heterogeneous_scalar_keeps_its_type_on_both_surfaces() {
    let ids = IdGen::new();
    let g = grammar(&ids).unwrap();
    for (json_v, text, want) in [
        (
            json!("1"),
            r#"Eq(field: "name", value: "1")"#,
            Scalar::Str("1".into()),
        ),
        (json!(1), r#"Eq(field: "name", value: 1)"#, Scalar::Int(1)),
        (
            json!(true),
            r#"Eq(field: "name", value: true)"#,
            Scalar::Bool(true),
        ),
    ] {
        let j = query_from_json(
            &json!({ "type": "Eq", "field": "name", "value": json_v }),
            &ids,
        )
        .unwrap();
        let t = query_from_text(&g, text, &ids).unwrap();
        for q in [j, t] {
            match q {
                Query::Eq { value, .. } => assert_eq!(value, want),
                other => panic!("unexpected {other:?}"),
            }
        }
    }
}

#[test]
fn dumped_query_validates_against_the_openapi_schema_and_reparses() {
    let ids = IdGen::new();
    let q = query_from_json(&sample_body(), &ids).unwrap();
    let dumped = dump_canonical_json(&q).unwrap();

    let v = validator();
    let errors: Vec<String> = v.iter_errors(&dumped).map(|e| e.to_string()).collect();
    assert!(
        errors.is_empty(),
        "schema rejected the kit's own output: {errors:?}"
    );

    let again = query_from_json(&dumped, &IdGen::new()).unwrap();
    assert_eq!(
        to_sql(&again, COLUMNS).unwrap(),
        to_sql(&q, COLUMNS).unwrap()
    );
}

#[test]
fn openapi_schema_and_json_front_end_agree_on_rejections() {
    let v = validator();
    let ids = IdGen::new();
    let rejected = [
        json!({ "type": "Eq", "feild": "name", "value": "x" }), // unknown key
        json!({ "type": "Eq", "field": "name" }),               // missing field
        json!({ "type": "Gt", "field": "age", "value": "18" }), // wrong payload type
        json!({ "type": "And", "items": [] }),                  // non_empty
        json!({ "type": "And" }),                               // non_empty, absent
        json!({ "type": "Nope" }),                              // unknown variant
        json!({ "type": "Not" }),                               // missing child
        json!({ "type": "In", "field": "name", "values": "bob" }), // Vec<String> shape
    ];
    for doc in rejected {
        assert!(!v.is_valid(&doc), "schema accepted {doc}");
        assert!(
            query_from_json(&doc, &ids).is_err(),
            "front-end accepted {doc}"
        );
    }
    // Absent `Vec<String>` payload is the canonical empty list on both.
    let empty_in = json!({ "type": "In", "field": "name" });
    assert!(v.is_valid(&empty_in));
    assert!(query_from_json(&empty_in, &ids).is_ok());
}

#[test]
fn generated_text_examples_parse_and_document_every_variant() {
    let ids = IdGen::new();
    let g = grammar(&ids).unwrap();
    let ex = example_gen::examples_from_grammar(&g).unwrap();
    let schema = Query::schema();
    let variants: Vec<&str> = schema.variants.iter().map(|v| v.name.as_str()).collect();
    let rules: Vec<&str> = ex.per_rule.iter().map(|e| e.rule.as_str()).collect();
    assert_eq!(rules, variants);
    for e in ex
        .per_rule
        .iter()
        .map(|e| &e.text)
        .chain(std::iter::once(&ex.composite))
    {
        let tree = g
            .parse(e)
            .unwrap_or_else(|err| panic!("example `{e}` failed: {err:?}"));
        assert!(check_conformance(&tree, &Query::schema()).is_empty());
        Query::from_parse_tree(&tree, &ids).unwrap_or_else(|err| panic!("`{e}`: {err:?}"));
    }
}

#[test]
fn values_never_enter_the_sql_text() {
    let ids = IdGen::new();
    let q = query_from_json(
        &json!({ "type": "Eq", "field": "name", "value": "'; DROP TABLE items; --" }),
        &ids,
    )
    .unwrap();
    let sql = to_sql(&q, COLUMNS).unwrap();
    assert_eq!(sql.text, "name = $1");
    assert!(!sql.text.contains("DROP"));
}

#[test]
fn unknown_column_is_a_lowering_error_with_the_node_id() {
    let ids = IdGen::new();
    let q = query_from_json(&json!({ "type": "IsNull", "field": "password" }), &ids).unwrap();
    match to_sql(&q, COLUMNS) {
        Err(SqlError::UnknownColumn { field, .. }) => assert_eq!(field, "password"),
        other => panic!("expected UnknownColumn, got {other:?}"),
    }
}

#[test]
fn openapi_document_is_well_formed() {
    let doc = openapi::document(
        &Query::schema(),
        &types(),
        &openapi::ApiInfo {
            title: "t".into(),
            version: "0".into(),
            path: "/search".into(),
            text_examples: vec!["IsNull(field: \"x\")".into()],
            json_examples: vec![json!({ "type": "IsNull", "field": "x" })],
        },
    )
    .unwrap();
    assert_eq!(doc["openapi"], "3.1.0");
    assert_eq!(
        doc["paths"]["/search"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"],
        "#/components/schemas/Query"
    );
    let arms = doc["components"]["schemas"]["Query"]["oneOf"]
        .as_array()
        .unwrap();
    assert_eq!(arms.len(), Query::schema().variants.len());
}

#[test]
fn unmapped_payload_type_fails_loudly() {
    let err = Query::schema()
        .to_json_schema(&TypeMap::new(), RefStyle::OpenApiComponent)
        .unwrap_err();
    assert_eq!(
        err,
        jschema::Error::UnsupportedField {
            variant: "Eq".into(),
            field: "value".into(),
            ty: "Scalar".into()
        }
    );
}
