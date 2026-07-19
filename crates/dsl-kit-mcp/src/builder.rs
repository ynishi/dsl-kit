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
        CallToolRequestParams, CallToolResult, Content, JsonObject, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
    },
    service::RequestContext,
};
use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::host::DslHost;

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
}

impl Default for DslMcpBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DslMcpBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self { instructions: None, tools: Vec::new() }
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
        let schema_value = serde_json::to_value(schema).unwrap_or_else(|_| Value::Object(Default::default()));
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
                    serde_json::from_value(Value::Object(Default::default()))
                        .map_err(|e| format!("failed to parse empty args as {}: {e}", std::any::type_name::<Args>()))?
                } else {
                    serde_json::from_value(args_value)
                        .map_err(|e| format!("failed to parse args as {}: {e}", std::any::type_name::<Args>()))?
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

        // Input schema: an empty object; the tool accepts no args.
        let input_schema = Arc::new(Default::default());

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
        DslMcpServer {
            instructions: self.instructions,
            tools: Arc::new(self.tools),
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
}

impl ServerHandler for DslMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: self.instructions.clone(),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let tools: Vec<Tool> = self.tools.iter().map(RegisteredTool::to_rmcp_tool).collect();
        Ok(ListToolsResult { tools, meta: None, next_cursor: None })
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
            .ok_or_else(|| McpError::invalid_params(format!("unknown tool {:?}", request.name), None))?;

        let args_value = request
            .arguments
            .map(Value::Object)
            .unwrap_or(Value::Null);

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
            .tool("echo", "echo the message", |args: EchoArgs, _ctx| async move {
                Ok::<_, String>(EchoOut { echoed: args.message })
            })
            .build();

        assert_eq!(server.tools.len(), 1);
        assert_eq!(server.tools[0].name.as_ref(), "echo");
        assert_eq!(server.tools[0].description.as_ref().map(|c| c.as_ref()), Some("echo the message"));
        assert_eq!(server.get_info().instructions.as_deref(), Some("test server"));
    }

    #[tokio::test]
    async fn tool_dispatch_deserializes_and_invokes_handler() {
        let server = DslMcpBuilder::new()
            .tool("echo", "echo", |args: EchoArgs, _ctx| async move {
                Ok::<_, String>(EchoOut { echoed: format!("hi, {}", args.message) })
            })
            .build();

        let tool = &server.tools[0];
        let args = serde_json::json!({ "message": "world" });
        let result = (tool.handler)(args, ToolCtx).await.expect("handler ok");
        assert_eq!(result, serde_json::json!({ "echoed": "hi, world" }));
    }

    #[tokio::test]
    async fn tool_dispatch_returns_error_on_missing_field() {
        let server = DslMcpBuilder::new()
            .tool("echo", "echo", |args: EchoArgs, _ctx| async move {
                Ok::<_, String>(EchoOut { echoed: args.message })
            })
            .build();

        let tool = &server.tools[0];
        let bad = serde_json::json!({});
        let err = (tool.handler)(bad, ToolCtx).await.err().expect("expected err");
        assert!(err.contains("failed to parse args"));
    }
}
