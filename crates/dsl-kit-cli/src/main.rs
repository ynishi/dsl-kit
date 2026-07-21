//! `dsl-kit-cli` — developer entry point for [`dsl-kit`].
//!
//! Right now the only subcommand is `mcp`, which starts a stdio MCP
//! server hosting the built-in reference DSL (see [`refdsl`]). More
//! subcommands (lint, schema dump, grammar preview) can grow here as
//! the kit's CLI surface expands.

use clap::{Parser, Subcommand};

mod mcp;
mod refdsl;

/// Top-level CLI.
#[derive(Debug, Parser)]
#[command(
    name = "dsl-kit-cli",
    version,
    about = "Developer CLI for dsl-kit — MCP mode today, more subcommands to come.",
    long_about = None,
    subcommand_required = true,
    arg_required_else_help = true,
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Available subcommands.
#[derive(Debug, Subcommand)]
enum Command {
    /// Start a stdio MCP server hosting the built-in reference DSL.
    ///
    /// Reserves stdout for MCP JSON-RPC framing; logs go to stderr.
    Mcp,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Mcp => mcp::run().await,
    }
}
