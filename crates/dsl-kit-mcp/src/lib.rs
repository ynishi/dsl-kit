//! MCP server framework for `dsl-kit`.
//!
//! ## Architecture
//!
//! Two surfaces:
//!
//! - [`DslMcpHandler`] — the debugger. A fixed, DSL-neutral tool
//!   surface driven by any [`DslHost`] implementation. Wire your DSL
//!   by implementing
//!   `DslHost` and calling `DslMcpHandler::new(Box::new(my_host))`.
//! - [`DslMcpBuilder`] / [`DslMcpServer`] — the light framework for
//!   hand-rolled custom MCP servers. Register tools as ordinary
//!   `async fn`s; input schemas are derived via `schemars`.
//!
//! This crate is DSL-agnostic — no reference DSL lives here. See the
//! `examples/flow-example/` tree (`flow-dsl`, `flow-host`, `flow-mcp`)
//! for a worked reference implementation.

#![warn(missing_docs)]

pub mod builder;
pub mod handler;
pub mod host;
pub mod resources;

pub use builder::{DslMcpBuilder, DslMcpServer, ToolCtx};
pub use handler::DslMcpHandler;
pub use host::{
    DslHost, EventCounts, HostLocation, HostOutcome, HostSnapshot, ResolvedCall, SuspendedCall,
};
pub use resources::{DSL_URI_PREFIX, KIT_URI_PREFIX, ResourceBody, ResourceEntry, kit_resources};
