//! MCP Resource surface for `dsl-kit-mcp`.
//!
//! Two audiences, two layers:
//!
//! - **`dsl-kit://kit/*`** — resources for people **building with** the
//!   kit (primitives, `DslHost` authoring, MCP tool reference, error
//!   catalog). Shipped by this crate; opt out via
//!   [`DslMcpHandler::without_kit_resources`](crate::DslMcpHandler)
//!   or [`DslMcpBuilder::without_kit_resources`](crate::DslMcpBuilder)
//!   when a custom server does not want them bleeding into its own
//!   resource surface.
//! - **`dsl-kit://dsl/*`** — resources for AI or humans **writing
//!   programs in** the DSL the current host has loaded. Contributed by
//!   the host via [`DslHost::resources`](crate::DslHost::resources).
//!   The `dsl-kit://dsl/*` URI namespace is a recommendation, not a
//!   rule; hosts may pick any URI they like.
//!
//! Both layers are surfaced through the same MCP `list_resources` /
//! `read_resource` endpoints, with kit-layer entries appearing first.

use dsl_kit::engine_error_catalog;
use serde_json::json;

/// How a [`ResourceEntry`] produces its body when read.
pub enum ResourceBody {
    /// Body baked in at compile time.
    Static(&'static str),
    /// Body generated on every read by a caller-supplied closure.
    ///
    /// Used for entries whose content depends on runtime state (the
    /// engine error catalog, host-specific schemas, etc.) so they never
    /// drift from the underlying source.
    Dynamic(fn() -> Result<String, String>),
}

impl std::fmt::Debug for ResourceBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceBody::Static(s) => f.debug_tuple("Static").field(&s.len()).finish(),
            ResourceBody::Dynamic(_) => f.write_str("Dynamic(fn)"),
        }
    }
}

/// One MCP Resource entry exposed under the `dsl-kit://` scheme (or any
/// URI a host chooses to use).
#[derive(Debug)]
pub struct ResourceEntry {
    /// Full resource URI, e.g. `"dsl-kit://kit/intro"`.
    pub uri: String,
    /// Human-readable title (used as the `resources/list` `name`).
    pub title: String,
    /// One-line description shown in `resources/list`.
    pub description: String,
    /// MIME type reported in `resources/list` and `resources/read`.
    pub mime_type: String,
    /// Body source.
    pub body: ResourceBody,
}

impl ResourceEntry {
    /// Convenience constructor for a static markdown entry.
    pub fn static_markdown(
        uri: impl Into<String>,
        title: impl Into<String>,
        description: impl Into<String>,
        body: &'static str,
    ) -> Self {
        Self {
            uri: uri.into(),
            title: title.into(),
            description: description.into(),
            mime_type: "text/markdown".into(),
            body: ResourceBody::Static(body),
        }
    }

    /// Resolve the body to a `String`.
    pub fn read(&self) -> Result<String, String> {
        match self.body {
            ResourceBody::Static(s) => Ok(s.to_string()),
            ResourceBody::Dynamic(f) => f(),
        }
    }
}

// ---------- Kit-layer entries -------------------------------------------

const KIT_INTRO: &str = include_str!("./resources_data/kit/intro.md");
const KIT_DSL_HOST_AUTHORING: &str = include_str!("./resources_data/kit/dsl-host-authoring.md");
const KIT_MCP_TOOL_REFERENCE: &str = include_str!("./resources_data/kit/mcp-tool-reference.md");

/// URI prefix reserved for kit-layer resources.
pub const KIT_URI_PREFIX: &str = "dsl-kit://kit/";

/// URI prefix recommended (not enforced) for host-contributed
/// DSL-layer resources.
pub const DSL_URI_PREFIX: &str = "dsl-kit://dsl/";

/// Returns the built-in kit-layer resource catalogue.
pub fn kit_resources() -> Vec<ResourceEntry> {
    vec![
        ResourceEntry::static_markdown(
            "dsl-kit://kit/intro",
            "dsl-kit — Intro",
            "What the kit is, its primitives, and the two-layer resource model (kit vs. DSL).",
            KIT_INTRO,
        ),
        ResourceEntry::static_markdown(
            "dsl-kit://kit/dsl-host-authoring",
            "dsl-kit — Authoring a DslHost",
            "How to implement DslHost around your own DSL. Required surface, optional surface, invariants.",
            KIT_DSL_HOST_AUTHORING,
        ),
        ResourceEntry::static_markdown(
            "dsl-kit://kit/mcp-tool-reference",
            "dsl-kit — MCP tool reference",
            "The MCP tools DslMcpHandler exposes, grouped by purpose.",
            KIT_MCP_TOOL_REFERENCE,
        ),
        ResourceEntry {
            uri: "dsl-kit://kit/error-catalog".into(),
            title: "dsl-kit — Error catalog".into(),
            description:
                "Every built-in EngineError diagnostic code with its help text, generated fresh from the enum."
                    .into(),
            mime_type: "application/json".into(),
            body: ResourceBody::Dynamic(render_error_catalog),
        },
    ]
}

fn render_error_catalog() -> Result<String, String> {
    let entries = engine_error_catalog();
    let list: Vec<_> = entries
        .into_iter()
        .map(|e| json!({ "code": e.code, "help": e.help }))
        .collect();
    serde_json::to_string_pretty(&json!({ "entries": list }))
        .map_err(|e| format!("serialize error catalog: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kit_catalog_has_expected_uris() {
        let uris: Vec<String> = kit_resources().into_iter().map(|r| r.uri).collect();
        assert!(uris.contains(&"dsl-kit://kit/intro".to_string()));
        assert!(uris.contains(&"dsl-kit://kit/dsl-host-authoring".to_string()));
        assert!(uris.contains(&"dsl-kit://kit/mcp-tool-reference".to_string()));
        assert!(uris.contains(&"dsl-kit://kit/error-catalog".to_string()));
    }

    #[test]
    fn every_kit_body_is_non_empty() {
        for entry in kit_resources() {
            let body = entry
                .read()
                .unwrap_or_else(|e| panic!("{}: {e}", entry.uri));
            assert!(!body.is_empty(), "empty body for {}", entry.uri);
        }
    }

    #[test]
    fn error_catalog_body_is_valid_json_with_entries() {
        let body = render_error_catalog().expect("catalog renders");
        let v: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        let entries = v["entries"].as_array().expect("entries array");
        assert!(!entries.is_empty());
        for e in entries {
            assert!(e["code"].as_str().is_some());
            assert!(e["help"].as_str().is_some());
        }
    }
}
