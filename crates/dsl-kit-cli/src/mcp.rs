//! `dsl-kit-cli mcp` — stdio MCP server around the built-in reference
//! DSL, using the [`dsl_kit_mcp`] framework.

use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::{self, EnvFilter};

use crate::refdsl::RefHost;
use dsl_kit_mcp::DslMcpHandler;

/// Entry point for the `mcp` subcommand.
pub async fn run() -> anyhow::Result<()> {
    // Reserve stdout for MCP JSON-RPC framing.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("dsl-kit-cli mcp starting");

    let host = RefHost::new_with_default_program();
    let handler = DslMcpHandler::new(Box::new(host));
    let service = handler.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
