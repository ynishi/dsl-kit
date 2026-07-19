//! `dsl-kit` — umbrella crate.
//!
//! This crate re-exports the engine primitives and the derive macros so that
//! downstream users can depend on a single crate.

#![warn(missing_docs)]

pub use dsl_kit_core::*;
pub use dsl_kit_macros::DslNode;
