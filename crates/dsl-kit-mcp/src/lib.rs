//! MCP server that exposes a `dsl-kit` stepper over stdio.
//!
//! The current build embeds the reference flow DSL (`dsl-kit-flow`) —
//! the research pipeline defined there is what MCP clients see and
//! interact with. The tools are DSL-neutral in shape (they speak in
//! terms of `NodeId`, `Path`, `depth`, and breakpoints) so a future
//! generalisation can swap the embedded DSL without changing the
//! contract.

pub mod handler;

pub use handler::DslMcpHandler;
