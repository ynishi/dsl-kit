//! A query DSL used as a Web API query language — the "IR + parser
//! only" shape of `dsl-kit`.
//!
//! The AST enum is the whole definition. From it the kit supplies:
//!
//! - the JSON front-end (`{"type": "And", "items": [...]}`) for a
//!   request body,
//! - the canonical text front-end (`And(items: [Eq(field: "name",
//!   value: "x")])`) for a `?q=` query parameter, via the
//!   schema-generated grammar,
//! - conformance checking on both,
//! - `DslDump` for re-emitting a normalized query (caching, echoing the
//!   parsed query back to the client),
//! - machine-derived text examples for the API docs.
//!
//! - the JSON Schema of the request body
//!   (`dsl_kit_schema::json_schema`).
//!
//! What this crate adds on top is the part the kit has no opinion on:
//! the OpenAPI document around that schema ([`openapi`]) and a
//! lowering to a parameterized SQL `WHERE` clause ([`to_sql`]). No
//! engine, no MCP.

#![warn(missing_docs)]

use std::fmt;

use dsl_kit_core::{IdGen, NodeId};
use dsl_kit_macros::{DslBuild, DslDump, DslNode, DslSchema};
use dsl_kit_parse::peg::{Grammar, Peg, choice, token};
use dsl_kit_parse::schema_gen::{self, SyntaxOverrides};
use dsl_kit_parse::{BuildError, Diagnostic, DslBuild as _, ParseTree, RawValue, codes};
use dsl_kit_schema::DslSchema as _;
use serde::{Deserialize, Serialize};

pub mod openapi;

/// A scalar comparison operand: string, integer, or boolean.
///
/// Heterogeneous on purpose — the same `Eq` node compares a text
/// column to a string and a numeric column to an integer. The JSON
/// front-end sees it as an untagged serde enum; the text front-end
/// spells it as a JSON-compatible literal (`"x"` / `42` / `true`) via
/// the [`syntax_overrides`] production, and [`build_scalar`] reads
/// both.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Scalar {
    /// Text operand.
    Str(String),
    /// Integer operand.
    Int(i64),
    /// Boolean operand.
    Bool(bool),
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Str(s) => write!(f, "{s:?}"),
            Scalar::Int(i) => write!(f, "{i}"),
            Scalar::Bool(b) => write!(f, "{b}"),
        }
    }
}

/// Filter expression of the query language.
#[derive(Debug, Clone, PartialEq, DslNode, DslSchema, DslBuild, DslDump)]
pub enum Query {
    /// Every item must hold.
    And {
        /// Stable node id.
        id: NodeId,
        /// Conjuncts (at least one).
        #[dsl_schema(non_empty)]
        items: Vec<Query>,
    },
    /// At least one item must hold.
    Or {
        /// Stable node id.
        id: NodeId,
        /// Disjuncts (at least one).
        #[dsl_schema(non_empty)]
        items: Vec<Query>,
    },
    /// Negation.
    Not {
        /// Stable node id.
        id: NodeId,
        /// Negated expression.
        body: Box<Query>,
    },
    /// `field = value`.
    Eq {
        /// Stable node id.
        id: NodeId,
        /// Column name.
        field: String,
        /// Operand.
        #[dsl_build(with = build_scalar)]
        #[dsl_dump(with = dump_scalar)]
        value: Scalar,
    },
    /// `field > value`.
    Gt {
        /// Stable node id.
        id: NodeId,
        /// Column name.
        field: String,
        /// Operand.
        value: i64,
    },
    /// `field < value`.
    Lt {
        /// Stable node id.
        id: NodeId,
        /// Column name.
        field: String,
        /// Operand.
        value: i64,
    },
    /// `field LIKE pattern`.
    Like {
        /// Stable node id.
        id: NodeId,
        /// Column name.
        field: String,
        /// SQL `LIKE` pattern.
        pattern: String,
    },
    /// `field IN (values...)`.
    In {
        /// Stable node id.
        id: NodeId,
        /// Column name.
        field: String,
        /// Accepted values.
        values: Vec<String>,
    },
    /// `field IS NULL`.
    IsNull {
        /// Stable node id.
        id: NodeId,
        /// Column name.
        field: String,
    },
}

// ---------------------------------------------------------------------------
// Scalar hooks (both front-ends → Scalar, Scalar → tree)
// ---------------------------------------------------------------------------

/// `#[dsl_build(with)]` converter for [`Scalar`] payloads.
///
/// The JSON front-end hands a typed value (untagged serde does the
/// rest); the text front-end hands the raw literal matched by
/// [`syntax_overrides`], which is JSON-compatible by construction, so
/// both routes end in `serde_json`.
pub fn build_scalar(tree: &ParseTree, name: &str) -> Result<Scalar, BuildError> {
    let err = |msg: String| {
        BuildError::single(Diagnostic::error(codes::FIELD_TYPE, msg).with_span(tree.span))
    };
    match tree.field(name) {
        Some(RawValue::Json(v)) => {
            serde_json::from_value(v.clone()).map_err(|e| err(format!("field `{name}`: {e}")))
        }
        Some(RawValue::Text(s)) => {
            serde_json::from_str(s).map_err(|e| err(format!("field `{name}`: {e}")))
        }
        None => Err(BuildError::single(
            Diagnostic::error(
                codes::MISSING_FIELD,
                format!("missing required field `{name}`"),
            )
            .with_span(tree.span),
        )),
    }
}

/// `#[dsl_dump(with)]` serializer for [`Scalar`] payloads — the dual
/// of [`build_scalar`].
pub fn dump_scalar(value: &Scalar) -> Result<Option<serde_json::Value>, BuildError> {
    serde_json::to_value(value).map(Some).map_err(|e| {
        BuildError::single(Diagnostic::error(
            dsl_kit_parse::serde_bridge::serde_codes::DUMP_FIELD,
            format!("scalar payload: {e}"),
        ))
    })
}

/// Text productions for payload types the canonical syntax has no
/// built-in spelling for. [`Scalar`] is spelled as a JSON literal:
/// `%str_raw` keeps the quotes so the matched text round-trips through
/// `serde_json::from_str` in [`build_scalar`].
pub fn syntax_overrides() -> SyntaxOverrides {
    SyntaxOverrides::new().for_type("Scalar", |ids: &IdGen| -> Peg {
        choice(
            ids,
            vec![
                token(ids, "%str_raw"),
                token(ids, "%int"),
                token(ids, "%kw:true"),
                token(ids, "%kw:false"),
            ],
        )
    })
}

/// The text grammar for the `?q=` surface, generated from
/// `Query::schema()` and statically checked.
pub fn grammar(ids: &IdGen) -> Result<Grammar, BuildError> {
    schema_gen::checked_grammar_from_schema_with(&Query::schema(), ids, &syntax_overrides())
}

// ---------------------------------------------------------------------------
// Request handling
// ---------------------------------------------------------------------------

/// Parses a JSON request body into a typed [`Query`].
pub fn query_from_json(body: &serde_json::Value, ids: &IdGen) -> Result<Query, BuildError> {
    let tree = dsl_kit_parse::serde_bridge::from_json_value(body, &Query::schema())?;
    Query::from_parse_tree(&tree, ids)
}

/// Parses a `?q=` text parameter into a typed [`Query`].
pub fn query_from_text(grammar: &Grammar, q: &str, ids: &IdGen) -> Result<Query, BuildError> {
    let tree = grammar.parse(q)?;
    Query::from_parse_tree(&tree, ids)
}

// ---------------------------------------------------------------------------
// SQL lowering
// ---------------------------------------------------------------------------

/// A lowered `WHERE` clause: SQL text with `$n` placeholders plus the
/// bound parameters in order. Values never enter the SQL text.
#[derive(Debug, Clone, PartialEq)]
pub struct Sql {
    /// Clause text, e.g. `(name = $1 AND age > $2)`.
    pub text: String,
    /// Bound parameters, `$1` first.
    pub params: Vec<Scalar>,
}

/// Errors from [`to_sql`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SqlError {
    /// A node names a column outside the allowed set.
    UnknownColumn {
        /// Offending node.
        node: NodeId,
        /// Column as written.
        field: String,
    },
}

impl fmt::Display for SqlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SqlError::UnknownColumn { node, field } => {
                write!(f, "unknown column `{field}` at {node}")
            }
        }
    }
}

impl std::error::Error for SqlError {}

/// Lowers a [`Query`] to a parameterized SQL `WHERE` clause. Column
/// names are checked against `columns`; everything else the schema
/// already guaranteed.
pub fn to_sql(query: &Query, columns: &[&str]) -> Result<Sql, SqlError> {
    let mut params = Vec::new();
    let text = lower(query, columns, &mut params)?;
    Ok(Sql { text, params })
}

fn column<'a>(id: NodeId, field: &'a str, columns: &[&str]) -> Result<&'a str, SqlError> {
    if columns.contains(&field) {
        Ok(field)
    } else {
        Err(SqlError::UnknownColumn {
            node: id,
            field: field.to_owned(),
        })
    }
}

fn bind(params: &mut Vec<Scalar>, value: Scalar) -> String {
    params.push(value);
    format!("${}", params.len())
}

fn lower(query: &Query, columns: &[&str], params: &mut Vec<Scalar>) -> Result<String, SqlError> {
    Ok(match query {
        Query::And { items, .. } | Query::Or { items, .. } => {
            let op = if matches!(query, Query::And { .. }) {
                " AND "
            } else {
                " OR "
            };
            let parts = items
                .iter()
                .map(|q| lower(q, columns, params))
                .collect::<Result<Vec<_>, _>>()?;
            format!("({})", parts.join(op))
        }
        Query::Not { body, .. } => format!("NOT {}", lower(body, columns, params)?),
        Query::Eq { id, field, value } => {
            let col = column(*id, field, columns)?;
            let p = bind(params, value.clone());
            format!("{col} = {p}")
        }
        Query::Gt { id, field, value } => {
            let col = column(*id, field, columns)?;
            let p = bind(params, Scalar::Int(*value));
            format!("{col} > {p}")
        }
        Query::Lt { id, field, value } => {
            let col = column(*id, field, columns)?;
            let p = bind(params, Scalar::Int(*value));
            format!("{col} < {p}")
        }
        Query::Like { id, field, pattern } => {
            let col = column(*id, field, columns)?;
            let p = bind(params, Scalar::Str(pattern.clone()));
            format!("{col} LIKE {p}")
        }
        Query::In { id, field, values } => {
            let col = column(*id, field, columns)?;
            if values.is_empty() {
                // `IN ()` is not SQL; an empty set matches nothing.
                "FALSE".to_owned()
            } else {
                let ps = values
                    .iter()
                    .map(|v| bind(params, Scalar::Str(v.clone())))
                    .collect::<Vec<_>>();
                format!("{col} IN ({})", ps.join(", "))
            }
        }
        Query::IsNull { id, field } => {
            let col = column(*id, field, columns)?;
            format!("{col} IS NULL")
        }
    })
}
