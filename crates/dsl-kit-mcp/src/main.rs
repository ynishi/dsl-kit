use dsl_kit_mcp::DslMcpHandler;
use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::{self, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Log to stderr — stdout is reserved for MCP JSON-RPC framing.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("dsl-kit-mcp starting");

    let server = DslMcpHandler::new_with_default_program();
    let service = server.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
