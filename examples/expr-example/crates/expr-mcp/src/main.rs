//! MCP server binary that serves `expr-host` over stdio.
//!
//! Symmetric to `flow-mcp`. Same shape, different DSL — the whole
//! point of the example is that swapping `FlowHost` for `ExprHost`
//! is the only change.

use dsl_kit_mcp::DslMcpHandler;
use expr_host::ExprHost;
use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::{self, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("expr-mcp starting");

    let host = ExprHost::new_with_default_program();
    let handler = DslMcpHandler::new(Box::new(host));
    let service = handler.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
