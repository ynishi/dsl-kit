# dsl-kit — Authoring a `DslHost`

`DslMcpHandler` is DSL-agnostic. To wire a new DSL, implement `DslHost`
for a struct that owns your program plus a stepper, then hand it to
`DslMcpHandler::new(Box::new(my_host))`.

## Required surface

```text
async_trait DslHost:
    fn dsl_name(&self) -> &str
    fn root_node_id(&self) -> u64
    fn root_summary(&self) -> String
    fn ast_size(&self) -> usize
    fn ast_pretty(&self) -> String
    fn snapshot(&self) -> HostSnapshot
    async fn step_one(&mut self, breakpoints) -> HostOutcome
    async fn step_to_yield(&mut self, breakpoints) -> HostOutcome
    async fn step_to_done(&mut self, breakpoints) -> HostOutcome
    async fn resolve(&mut self, result: Option<String>) -> ResolvedCall
    fn reset(&mut self)
```

## Optional surface

```text
    fn catalog(&self) -> Vec<ErrorCatalogEntry>       // default: []
    fn resources(&self) -> Vec<ResourceEntry>         // default: []
    async fn resolve_by_id(
        &mut self,
        id: u64,
        result: Result<String, HostEffectError>,
    ) -> Result<ResolvedCall, String>                 // default: error
    fn take_cancellations(&mut self) -> Vec<u64>      // default: []
```

- `catalog()` adds host-specific `EngineError`-shaped codes that
  `dsl_kit_explain` should be able to look up alongside the built-ins.
- `resources()` contributes Layer B entries (`dsl-kit://dsl/*` by
  convention) — DSL grammar references, sample programs, tool
  extensions the client might want to prime itself with. Not enforced;
  any URI namespace is legal.
- `resolve_by_id()` powers the `dsl_kit_resolve_by_id` MCP tool.
  Override this on hosts whose DSL emits multiple simultaneous
  suspensions (e.g. `Par` fan-out). `result` carries either a success
  payload (`Ok(String)`) or an effect-side failure
  (`Err(HostEffectError { code, message })`). Hosts that stay
  single-in-flight can leave the default implementation, which errors
  with `"resolve_by_id not implemented"`.
- `take_cancellations()` powers the `dsl_kit_take_cancellations` MCP
  tool. Override to drain the engine-assigned ids of suspensions the
  DSL has cancelled since the last drain (typically FailFast siblings
  after an `Err` resolve, or losing legs of an `Any`/`FirstK` fold).
  The default returns an empty vector.

## Reference implementation

`flow_host::FlowHost` (in the `flow-host` example crate) is the working
reference and doubles as the payload of the `flow-mcp` binary shipped
under `examples/flow-example/`. It:

- leaks a `Flow` program (`research_pipeline`) so the stepper can hold a
  `'static` reference,
- delegates traversal to the derive-generated `Walk` impl for pretty
  printing and size counting,
- maps its own `SuspendReason` / `Path` back to the `HostOutcome` and
  `HostLocation` shapes the handler serialises,
- provides a canned response registry for `resolve(None)`.

For a hand-rolled MCP server that skips `DslMcpHandler` altogether and
uses the light builder framework instead, see `custom-mcp-example`.

## Design invariants

- The stepper's suspend / resume contract is one-shot: each
  `Suspended` yield must be resolved exactly once before the next step.
  Breaking this contract is what `EngineError::StepperProtocol` reports.
- `snapshot()` is the single source of truth for the `dsl_kit_state`
  tool. Hosts do not shape JSON directly; `HostSnapshot` fields map to
  well-known JSON keys.
- Breakpoints are host-side data. `Stepper` knows nothing about them;
  the host inspects the current `NodeContext` before executing a node
  and yields `Suspended { reason: Breakpoint, .. }` when a condition
  matches.
