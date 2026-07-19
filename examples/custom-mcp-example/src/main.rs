//! A worked example of the `dsl-kit-mcp` builder.
//!
//! The binary starts a stdio MCP server that exposes three tools:
//!
//! - `echo` — a plain typed-fn tool. The handler is an ordinary
//!   `async` closure; its input schema is derived from the argument
//!   type via `schemars`.
//! - `sum` — another typed-fn tool with a slightly larger schema, to
//!   show that arbitrary `serde::Deserialize + schemars::JsonSchema`
//!   types are accepted.
//! - `research_pipeline` — a tool whose body is a `dsl-kit` DSL: the
//!   `FlowHost` reference research pipeline is registered directly
//!   with `tool_from_host`, and each invocation runs the flow to
//!   completion and returns its accumulated results.
//!
//! The example is what an author defining a custom DSL for their own
//! problem would write: define the DSL AST + host in their crate,
//! then hand it to the builder alongside any other typed-fn tools
//! they want to expose. No `#[tool_router]` macro, no rmcp
//! boilerplate.

use dsl_kit_mcp::{DslMcpBuilder, FlowHost, ToolCtx};
use rmcp::{ServiceExt, transport::stdio};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema)]
struct EchoArgs {
    /// Message to be echoed back verbatim.
    message: String,
}

#[derive(Debug, Serialize)]
struct EchoOut {
    echoed: String,
    length: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SumArgs {
    /// A list of integers.
    numbers: Vec<i64>,
    /// Optional starting offset (default 0).
    #[serde(default)]
    start: i64,
}

#[derive(Debug, Serialize)]
struct SumOut {
    sum: i64,
    count: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("custom-mcp-example starting");

    let host = FlowHost::new_with_default_program();

    let server = DslMcpBuilder::new()
        .instructions(
            "Demo MCP server built with `dsl_kit_mcp::DslMcpBuilder`. \
             Shows two flavours of tool registration: typed-fn handlers \
             (echo, sum) and dsl-kit AST bodies (research_pipeline).",
        )
        .tool(
            "echo",
            "Echoes the supplied message back and reports its length.",
            |args: EchoArgs, _ctx: ToolCtx| async move {
                let length = args.message.chars().count();
                Ok::<_, String>(EchoOut { echoed: args.message, length })
            },
        )
        .tool(
            "sum",
            "Sums a list of integers, optionally offset by `start`.",
            |args: SumArgs, _ctx: ToolCtx| async move {
                let sum: i64 = args.numbers.iter().sum::<i64>() + args.start;
                Ok::<_, String>(SumOut { sum, count: args.numbers.len() })
            },
        )
        .tool_from_host(
            "research_pipeline",
            "Runs the built-in dsl-kit-flow research pipeline to completion \
             and returns its per-node results.",
            host,
        )
        .build();

    let service = server.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
