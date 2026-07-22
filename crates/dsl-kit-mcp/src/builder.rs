//! Builder for custom MCP servers backed by `dsl-kit`.
//!
//! This module offers a light framework — the "path γ" alternative to
//! rmcp's `#[tool_router]` macro — for authors who want to declare an
//! MCP server as a collection of tools, with schemas derived from
//! Rust types via `schemars` and handlers written as ordinary
//! `async fn`s. It mirrors Bevy's `IntoSystem` idea: the framework
//! inspects each handler's parameter type to derive the tool's input
//! schema, then wires up the JSON round-trip automatically.
//!
//! ```ignore
//! use dsl_kit_mcp::{DslMcpBuilder, ToolCtx};
//! use serde::{Deserialize, Serialize};
//! use schemars::JsonSchema;
//!
//! #[derive(Deserialize, JsonSchema)]
//! struct Args { query: String }
//!
//! #[derive(Serialize)]
//! struct Out { echoed: String }
//!
//! let server = DslMcpBuilder::new()
//!     .instructions("demo")
//!     .tool("echo", "echo the query", |args: Args, _ctx: ToolCtx| async move {
//!         Ok::<_, String>(Out { echoed: args.query })
//!     })
//!     .build();
//! ```
//!
//! Handlers can also delegate to a `dsl-kit` DSL: `tool_from_host`
//! runs a `DslHost` to completion and returns its results. See the
//! `custom-mcp-example` crate for a worked demonstration.

use std::borrow::Cow;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler,
    model::{
        CallToolRequestParams, CallToolResult, Content, JsonObject, ListResourcesResult,
        ListToolsResult, PaginatedRequestParams, RawResource, ReadResourceRequestParams,
        ReadResourceResult, Resource, ResourceContents, ServerCapabilities, ServerInfo, Tool,
    },
    service::RequestContext,
};
use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::host::DslHost;
use crate::resources::{ResourceEntry, kit_resources};
use dsl_kit::{SuggesterHandle, noop_handle};

/// Per-call context handed to every tool handler.
///
/// Currently a shell that carries no cross-tool state; kept as a named
/// type so future additions (session store, breakpoint set, tracing
/// span, cancellation token) do not force a breaking signature change
/// on user handlers.
#[derive(Debug, Default, Clone)]
pub struct ToolCtx;

/// Type-erased tool handler.
type BoxedHandler = Arc<
    dyn Fn(Value, ToolCtx) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>
        + Send
        + Sync,
>;

struct RegisteredTool {
    name: Cow<'static, str>,
    description: Option<Cow<'static, str>>,
    input_schema: Arc<JsonObject>,
    handler: BoxedHandler,
}

impl RegisteredTool {
    fn to_rmcp_tool(&self) -> Tool {
        Tool {
            name: self.name.clone(),
            title: None,
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
            output_schema: None,
            annotations: None,
            execution: None,
            icons: None,
            meta: None,
        }
    }
}

/// Fluent builder for a custom MCP server.
///
/// Instantiate with [`new`](Self::new), add tools with
/// [`tool`](Self::tool) or [`tool_from_host`](Self::tool_from_host),
/// optionally set [`instructions`](Self::instructions), and call
/// [`build`](Self::build) to obtain a [`DslMcpServer`] ready to be
/// handed to `rmcp::serve`.
pub struct DslMcpBuilder {
    instructions: Option<String>,
    tools: Vec<RegisteredTool>,
    expose_kit_resources: bool,
    extra_resources: Vec<ResourceEntry>,
    suggester: SuggesterHandle,
}

impl Default for DslMcpBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DslMcpBuilder {
    /// Creates an empty builder.
    ///
    /// By default, the built server exposes the kit-layer resources
    /// (`dsl-kit://kit/*`). Call [`without_kit_resources`](Self::without_kit_resources)
    /// to suppress them; call [`resource`](Self::resource) to add DSL-
    /// layer entries.
    pub fn new() -> Self {
        Self {
            instructions: None,
            tools: Vec::new(),
            expose_kit_resources: true,
            extra_resources: Vec::new(),
            suggester: noop_handle(),
        }
    }

    /// Injects a fuzzy-match [`Suggester`](dsl_kit::Suggester) used to
    /// enrich this server's `unknown tool` diagnostic with
    /// `did you mean X?` hints when an MCP client mistypes a tool
    /// name.
    ///
    /// The default is a no-op suggester, so this crate does not pull
    /// in any similarity algorithm on its own. Pass
    /// `Arc::new(dsl_kit_fuzzy::FuzzySuggester::default())` at the
    /// composition root to enable Jaro-Winkler hints.
    pub fn with_suggester(mut self, suggester: SuggesterHandle) -> Self {
        self.suggester = suggester;
        self
    }

    /// Suppresses the built-in `dsl-kit://kit/*` resources on the
    /// built server. Use when a custom server does not want the kit
    /// authoring guides bleeding into its own resource surface.
    pub fn without_kit_resources(mut self) -> Self {
        self.expose_kit_resources = false;
        self
    }

    /// Registers an additional MCP resource entry. The URI namespace
    /// is at the caller's discretion; `dsl-kit://dsl/*` is the
    /// recommended prefix for DSL-layer entries but any URI is
    /// accepted.
    pub fn resource(mut self, entry: ResourceEntry) -> Self {
        self.extra_resources.push(entry);
        self
    }

    /// Bulk variant of [`resource`](Self::resource).
    pub fn resources(mut self, entries: impl IntoIterator<Item = ResourceEntry>) -> Self {
        self.extra_resources.extend(entries);
        self
    }

    /// Sets the server's `instructions` field (shown to MCP clients as
    /// a hint about the server's purpose and typical workflow).
    pub fn instructions(mut self, text: impl Into<String>) -> Self {
        self.instructions = Some(text.into());
        self
    }

    /// Registers a tool whose handler is an ordinary `async` closure
    /// with typed input and output.
    ///
    /// The input schema is derived from `Args` via `schemars::JsonSchema`,
    /// so the handler need not repeat it. The result type must be
    /// serializable; the value is returned to the MCP client as the
    /// tool's text content.
    pub fn tool<F, Fut, Args, Out>(
        mut self,
        name: impl Into<Cow<'static, str>>,
        description: impl Into<Cow<'static, str>>,
        handler: F,
    ) -> Self
    where
        F: Fn(Args, ToolCtx) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Out, String>> + Send + 'static,
        Args: DeserializeOwned + JsonSchema + Send + 'static,
        Out: Serialize + Send + 'static,
    {
        let schema = schemars::schema_for!(Args);
        let schema_value =
            serde_json::to_value(schema).unwrap_or_else(|_| Value::Object(Default::default()));
        let schema_obj = match schema_value {
            Value::Object(map) => map,
            _ => Default::default(),
        };
        let input_schema = Arc::new(schema_obj);
        let handler = Arc::new(handler);
        let boxed: BoxedHandler = Arc::new(move |args_value: Value, ctx: ToolCtx| {
            let handler = handler.clone();
            Box::pin(async move {
                let args: Args = if args_value.is_null() {
                    serde_json::from_value(Value::Object(Default::default())).map_err(|e| {
                        format!(
                            "failed to parse empty args as {}: {e}",
                            std::any::type_name::<Args>()
                        )
                    })?
                } else {
                    serde_json::from_value(args_value).map_err(|e| {
                        format!(
                            "failed to parse args as {}: {e}",
                            std::any::type_name::<Args>()
                        )
                    })?
                };
                let out = (handler)(args, ctx).await?;
                serde_json::to_value(out).map_err(|e| format!("failed to serialize output: {e}"))
            })
        });

        self.tools.push(RegisteredTool {
            name: name.into(),
            description: Some(description.into()),
            input_schema,
            handler: boxed,
        });
        self
    }

    /// Registers a tool that drives a `dsl-kit` [`DslHost`] to
    /// completion and returns its accumulated results.
    ///
    /// The host is wrapped in a `Mutex` so it can be shared across
    /// tool invocations. When the tool is called, the host is reset,
    /// stepped to `Done`, and its recorded results are returned as a
    /// JSON object.
    pub fn tool_from_host<H>(
        mut self,
        name: impl Into<Cow<'static, str>>,
        description: impl Into<Cow<'static, str>>,
        host: H,
    ) -> Self
    where
        H: DslHost + 'static,
    {
        let host = Arc::new(Mutex::new(host));
        let breakpoints = Arc::new(dsl_kit::BreakpointSet::new());

        // Input schema for a no-arg tool. MCP spec requires
        // `inputSchema` to be a JSON Schema *object* (`type: "object"`);
        // sending an empty `{}` is rejected by strict clients (Claude
        // Code returns `tools fetch failed`), so emit the minimum
        // canonical empty-object schema.
        let input_schema = Arc::new({
            let mut m = JsonObject::new();
            m.insert("type".into(), Value::String("object".into()));
            m.insert("properties".into(), Value::Object(JsonObject::new()));
            m.insert("additionalProperties".into(), Value::Bool(false));
            m
        });

        let boxed: BoxedHandler = Arc::new(move |_args: Value, _ctx: ToolCtx| {
            let host = host.clone();
            let bp = breakpoints.clone();
            Box::pin(async move {
                let mut guard = host.lock().await;
                guard.reset();
                let outcome = guard.step_to_done(&bp).await?;
                let snap = guard.snapshot();
                let mut entries = Vec::new();
                for (node, text) in &snap.results {
                    entries.push(serde_json::json!({ "node": node, "value": text }));
                }
                Ok(serde_json::json!({
                    "dsl": guard.dsl_name(),
                    "outcome": format!("{outcome:?}"),
                    "results": entries,
                }))
            })
        });

        self.tools.push(RegisteredTool {
            name: name.into(),
            description: Some(description.into()),
            input_schema,
            handler: boxed,
        });
        self
    }

    /// Finalizes the builder.
    pub fn build(self) -> DslMcpServer {
        let mut resources = if self.expose_kit_resources {
            kit_resources()
        } else {
            Vec::new()
        };
        resources.extend(self.extra_resources);
        DslMcpServer {
            instructions: self.instructions,
            tools: Arc::new(self.tools),
            resources: Arc::new(resources),
            suggester: self.suggester,
        }
    }
}

/// MCP server built from a [`DslMcpBuilder`].
///
/// Implements `rmcp::ServerHandler` directly; hand it to
/// `rmcp::ServiceExt::serve` to run over any rmcp transport.
#[derive(Clone)]
pub struct DslMcpServer {
    instructions: Option<String>,
    tools: Arc<Vec<RegisteredTool>>,
    resources: Arc<Vec<ResourceEntry>>,
    suggester: SuggesterHandle,
}

impl DslMcpServer {
    /// Number of registered resources (for tests / introspection).
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }
}

/// Shared formatter for the `unknown tool` `invalid_params` message.
///
/// Extracted so the enrichment path can be exercised in isolation
/// without constructing an rmcp `RequestContext`; the production
/// dispatcher in [`DslMcpServer::call_tool`] and the unit tests both
/// call in through the same entry point, keeping the `did you mean`
/// wording pinned in one place.
pub(crate) fn format_unknown_tool(
    query: &str,
    tool_names: &[&str],
    suggester: &dyn dsl_kit::Suggester,
) -> String {
    let base = format!("unknown tool {query:?}");
    match suggester.enrich_unknown(query, tool_names) {
        Some(hint) => format!("{base} ({hint})"),
        None => base,
    }
}

impl ServerHandler for DslMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: self.instructions.clone(),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
            ..Default::default()
        }
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        let resources: Vec<Resource> = self
            .resources
            .iter()
            .map(|e| Resource {
                raw: RawResource {
                    uri: e.uri.clone(),
                    name: e.title.clone(),
                    title: Some(e.title.clone()),
                    description: Some(e.description.clone()),
                    mime_type: Some(e.mime_type.clone()),
                    size: None,
                    icons: None,
                    meta: None,
                },
                annotations: None,
            })
            .collect();
        Ok(ListResourcesResult::with_all_items(resources))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let entry = self
            .resources
            .iter()
            .find(|e| e.uri == request.uri)
            .ok_or_else(|| {
                McpError::resource_not_found(format!("unknown resource uri: {}", request.uri), None)
            })?;
        let body = entry
            .read()
            .map_err(|e| McpError::internal_error(format!("read resource: {e}"), None))?;
        Ok(ReadResourceResult {
            contents: vec![ResourceContents::text(body, entry.uri.clone())],
        })
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let tools: Vec<Tool> = self
            .tools
            .iter()
            .map(RegisteredTool::to_rmcp_tool)
            .collect();
        Ok(ListToolsResult {
            tools,
            meta: None,
            next_cursor: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name == request.name)
            .ok_or_else(|| {
                let names: Vec<&str> = self.tools.iter().map(|t| t.name.as_ref()).collect();
                let msg = format_unknown_tool(&request.name, &names, &*self.suggester);
                McpError::invalid_params(msg, None)
            })?;

        let args_value = request.arguments.map(Value::Object).unwrap_or(Value::Null);

        match (tool.handler)(args_value, ToolCtx).await {
            Ok(output) => {
                let text = serde_json::to_string(&output).unwrap_or_else(|_| output.to_string());
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema)]
    struct EchoArgs {
        message: String,
    }

    #[derive(Debug, Serialize)]
    struct EchoOut {
        echoed: String,
    }

    #[tokio::test]
    async fn builder_registers_tools_and_lists_them() {
        let server = DslMcpBuilder::new()
            .instructions("test server")
            .tool(
                "echo",
                "echo the message",
                |args: EchoArgs, _ctx| async move {
                    Ok::<_, String>(EchoOut {
                        echoed: args.message,
                    })
                },
            )
            .build();

        assert_eq!(server.tools.len(), 1);
        assert_eq!(server.tools[0].name.as_ref(), "echo");
        assert_eq!(
            server.tools[0].description.as_ref().map(|c| c.as_ref()),
            Some("echo the message")
        );
        assert_eq!(
            server.get_info().instructions.as_deref(),
            Some("test server")
        );
    }

    #[tokio::test]
    async fn tool_dispatch_deserializes_and_invokes_handler() {
        let server = DslMcpBuilder::new()
            .tool("echo", "echo", |args: EchoArgs, _ctx| async move {
                Ok::<_, String>(EchoOut {
                    echoed: format!("hi, {}", args.message),
                })
            })
            .build();

        let tool = &server.tools[0];
        let args = serde_json::json!({ "message": "world" });
        let result = (tool.handler)(args, ToolCtx).await.expect("handler ok");
        assert_eq!(result, serde_json::json!({ "echoed": "hi, world" }));
    }

    #[tokio::test]
    async fn builder_exposes_kit_resources_by_default() {
        let server = DslMcpBuilder::new().build();
        let uris: Vec<&str> = server.resources.iter().map(|e| e.uri.as_str()).collect();
        assert!(uris.contains(&"dsl-kit://kit/intro"));
        assert!(uris.contains(&"dsl-kit://kit/error-catalog"));
    }

    #[tokio::test]
    async fn builder_without_kit_resources_strips_kit_layer() {
        let server = DslMcpBuilder::new().without_kit_resources().build();
        for entry in server.resources.iter() {
            assert!(
                !entry.uri.starts_with("dsl-kit://kit/"),
                "kit uri leaked: {}",
                entry.uri
            );
        }
    }

    #[tokio::test]
    async fn builder_resource_method_appends_entries() {
        let server = DslMcpBuilder::new()
            .without_kit_resources()
            .resource(ResourceEntry::static_markdown(
                "example://guide/hello",
                "hello guide",
                "toy resource",
                "hello world",
            ))
            .build();
        assert_eq!(server.resource_count(), 1);
        assert_eq!(server.resources[0].uri, "example://guide/hello");
    }

    #[tokio::test]
    async fn unknown_tool_defaults_to_noop_message() {
        // With no suggester injected the message stays literal, matching
        // the historical `unknown tool "X"` wording.
        let names = ["echo", "greet"];
        let msg = format_unknown_tool("ech", &names, &*dsl_kit::noop_handle());
        assert_eq!(msg, "unknown tool \"ech\"");
    }

    #[tokio::test]
    async fn unknown_tool_with_fuzzy_suggests_closest() {
        use dsl_kit_fuzzy::FuzzySuggester;
        let names = ["echo", "greet"];
        let suggester = FuzzySuggester::default();
        let msg = format_unknown_tool("ech", &names, &suggester);
        assert!(
            msg.contains("did you mean") && msg.contains("echo"),
            "expected `echo` in the hint, got: {msg}"
        );
    }

    #[tokio::test]
    async fn tool_dispatch_returns_error_on_missing_field() {
        let server = DslMcpBuilder::new()
            .tool("echo", "echo", |args: EchoArgs, _ctx| async move {
                Ok::<_, String>(EchoOut {
                    echoed: args.message,
                })
            })
            .build();

        let tool = &server.tools[0];
        let bad = serde_json::json!({});
        let err = (tool.handler)(bad, ToolCtx)
            .await
            .expect_err("expected err");
        assert!(err.contains("failed to parse args"));
    }
}
