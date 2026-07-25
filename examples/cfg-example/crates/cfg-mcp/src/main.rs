//! MCP server binary that serves `cfg-host` over stdio.
//!
//! Same scaffolding as `flow-mcp` / `expr-mcp` — swapping the host is
//! the only change. What is new is what the surface reports: `Cfg` has
//! keyed child slots, so `dsl_kit_schema` lists
//! `"multiplicity": "map"` and `dsl_kit_load` accepts a document whose
//! children are named rather than positional.

use cfg_host::CfgHost;
use dsl_kit_mcp::DslMcpHandler;
use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::{self, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("cfg-mcp starting");

    let host = CfgHost::new_with_default_document();
    let handler = DslMcpHandler::new(Box::new(host));
    let service = handler.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
