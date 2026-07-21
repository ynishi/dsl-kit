//! `dsl-kit` — umbrella crate.
//!
//! This crate re-exports the engine primitives and the derive macros so that
//! downstream users can depend on a single crate.
//!
//! ## Architecture
//!
//! The kit is a workspace of focused crates; this facade pulls in the
//! two every DSL needs:
//!
//! - `dsl-kit-core` (re-exported wholesale) — engine, stepper, event
//!   stream, breakpoints, sans-io drive layer.
//! - `dsl-kit-macros` (`DslNode` re-exported) — the traversal derive.
//!
//! The parser stack (`dsl-kit-parse`), schema reflection
//! (`dsl-kit-schema`), the lint framework (`dsl-kit-lint`), and the MCP
//! surface (`dsl-kit-mcp`) stay separate crates — add them as direct
//! dependencies when a DSL grows into them.

#![warn(missing_docs)]

pub use dsl_kit_core::*;
pub use dsl_kit_macros::DslNode;
