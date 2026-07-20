# dsl-kit — MCP tool reference

`DslMcpHandler` exposes ten tools. All operate on the DSL-neutral
`NodeId` / `Path` / `depth` / iteration shapes, so a caller sees the
same contract regardless of which `DslHost` is loaded.

## Inspection

- **`dsl_kit_info`** — kit identity (name + version), DSL name, root
  node summary, AST size.
- **`dsl_kit_ast`** — indented text tree of the loaded program.
- **`dsl_kit_state`** — depth, current path, pending suspended call,
  recorded results, event counters, active breakpoints.

## Stepping

- **`dsl_kit_step`** — advance the stepper. `mode` accepts:
  - `"one"` (default): one `step()`.
  - `"to_yield"`: run until suspension / done / error.
  - `"to_done"`: run until completion, resolving intermediate calls
    with the host's default response.
- **`dsl_kit_resolve`** — supply a response for the currently
  suspended call. `result` is optional; when omitted the host provides
  a canned default.

## Breakpoints

- **`dsl_kit_breakpoint_add`** — register a compound condition. At
  least one of `at_node` / `at_depth` / `at_depth_at_least` /
  `at_depth_at_most` / `at_iteration` / `under_path` must be present;
  multiple fields are ANDed.
- **`dsl_kit_breakpoint_list`** — enumerate every registered entry
  with its condition JSON.
- **`dsl_kit_breakpoint_remove`** — remove an entry by id.

## Diagnostics

- **`dsl_kit_explain`** — look up help text for a stable diagnostic
  code. Omit `code` to list every known code. Built-in codes come from
  `EngineError`; hosts extend the set via `DslHost::catalog()`.

## Lifecycle

- **`dsl_kit_reset`** — reset the host's stepper. Breakpoints are
  left in place.

## Typical workflow

1. `dsl_kit_info` + `dsl_kit_ast` to see the loaded program.
2. `dsl_kit_breakpoint_add` to pause on interesting nodes.
3. `dsl_kit_step` (`mode: "to_yield"`) until suspension.
4. `dsl_kit_state` to inspect where the stepper is.
5. `dsl_kit_resolve` to supply a call response, then step again.
6. `dsl_kit_reset` when starting over.
