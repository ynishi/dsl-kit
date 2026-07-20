//! MCP server that drives a `dsl-kit` stepper over stdio.
//!
//! The MCP handler is DSL-agnostic: it speaks to any type that
//! implements [`DslHost`], so callers can plug in their own DSL by
//! implementing that trait and handing an instance to
//! [`DslMcpHandler::new`].
//!
//! For convenience the crate also ships a reference [`FlowHost`] that
//! wraps the `dsl-kit-flow` reference DSL; the default binary shipped
//! with the crate serves that host.

#![warn(missing_docs)]

pub mod builder;
pub mod flow_host;
pub mod handler;
pub mod host;
pub mod resources;

pub use builder::{DslMcpBuilder, DslMcpServer, ToolCtx};
pub use flow_host::FlowHost;
pub use handler::DslMcpHandler;
pub use host::{
    DslHost, EventCounts, HostLocation, HostOutcome, HostSnapshot, ResolvedCall, SuspendedCall,
};
pub use resources::{DSL_URI_PREFIX, KIT_URI_PREFIX, ResourceBody, ResourceEntry, kit_resources};
