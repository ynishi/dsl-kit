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
//! Also demonstrates the two resource-surface controls added in
//! Round 10:
//!
//! - `.without_kit_resources()` strips the built-in
//!   `dsl-kit://kit/*` guides so this custom server's `list_resources`
//!   only shows what the server itself contributes.
//! - `.resource(...)` registers one custom entry
//!   (`example://guides/tool-usage`) explaining the three tools above.
//!
//! The example is what an author defining a custom DSL for their own
//! problem would write: define the DSL AST + host in their crate,
//! then hand it to the builder alongside any other typed-fn tools
//! they want to expose. No `#[tool_router]` macro, no rmcp
//! boilerplate.

use dsl_kit_mcp::{DslMcpBuilder, ResourceEntry, ToolCtx};
use flow_host::FlowHost;
use rmcp::{ServiceExt, transport::stdio};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const TOOL_USAGE_GUIDE: &str = r#"# custom-mcp-example — tool usage

Three tools are registered by this server:

- **echo** — `{ "message": "..." }` → echoes the message and reports
  its Unicode length.
- **sum** — `{ "numbers": [1, 2, 3], "start": 0 }` → sums the list,
  optionally offset by `start`.
- **research_pipeline** — no args. Runs the built-in flow-dsl
  research pipeline to completion and returns per-node results.

This guide is served as the `example://guides/tool-usage` resource
via the `.resource(...)` builder API. The server also calls
`.without_kit_resources()` so `dsl-kit://kit/*` entries do not appear
alongside it.
"#;

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
        .without_kit_resources()
        .resource(ResourceEntry::static_markdown(
            "example://guides/tool-usage",
            "custom-mcp-example — tool usage",
            "How to call the three tools this server exposes.",
            TOOL_USAGE_GUIDE,
        ))
        .tool(
            "echo",
            "Echoes the supplied message back and reports its length.",
            |args: EchoArgs, _ctx: ToolCtx| async move {
                let length = args.message.chars().count();
                Ok::<_, String>(EchoOut {
                    echoed: args.message,
                    length,
                })
            },
        )
        .tool(
            "sum",
            "Sums a list of integers, optionally offset by `start`.",
            |args: SumArgs, _ctx: ToolCtx| async move {
                let sum: i64 = args.numbers.iter().sum::<i64>() + args.start;
                Ok::<_, String>(SumOut {
                    sum,
                    count: args.numbers.len(),
                })
            },
        )
        .tool_from_host(
            "research_pipeline",
            "Runs the built-in flow-dsl research pipeline to completion \
             and returns its per-node results.",
            host,
        )
        .build();

    let service = server.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
