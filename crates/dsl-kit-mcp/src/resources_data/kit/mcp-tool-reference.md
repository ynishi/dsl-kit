# dsl-kit — MCP tool reference

`DslMcpHandler` exposes the full tool surface below. All of it operates
on the DSL-neutral `NodeId` / `Path` / `depth` / iteration shapes, so a
caller sees the same contract regardless of which `DslHost` is loaded.

## Inspection

- **`dsl_kit_info`** — kit identity (name + version), DSL name, root
  node summary, AST size.
- **`dsl_kit_ast`** — indented text tree of the loaded program.
- **`dsl_kit_state`** — depth, current path, pending suspended call,
  the full `pending` list (fan-out projection), recorded results,
  event counters, active breakpoints.

## Stepping

- **`dsl_kit_step`** — advance the stepper. `mode` accepts:
  - `"one"` (default): one `step()`.
  - `"to_yield"`: run until suspension / done / error.
  - `"to_done"`: run until completion, resolving intermediate calls
    with the host's default response.
- **`dsl_kit_resolve`** — supply a response for the currently
  suspended call (single-in-flight path). `result` is optional; when
  omitted the host provides a canned default. For `Par` fan-out use
  `dsl_kit_resolve_by_id` instead.

## Fan-out

- **`dsl_kit_pending`** — list every live suspension. In the common
  one-in-flight case this returns zero or one entry; under a `Par`
  fan-out it enumerates every live child. Each entry carries a stable
  `id`, `reason`, `label`, and `at` location.
- **`dsl_kit_resolve_by_id`** — resolve one specific pending
  suspension by its stable `id`. Body variants:
  - `{ id, ok: "response text" }` — success payload.
  - `{ id, err: { code, message } }` — effect-side failure. Under
    `FailFast` this triggers propagation on the next `dsl_kit_step`
    and queues sibling cancellations.
- **`dsl_kit_take_cancellations`** — drain the ids of suspensions the
  engine has cancelled since the last drain. Hosts should call this
  after every `dsl_kit_step` that returns an error or completes a Par
  fold and act on the drained ids (typically abort their runtime
  handles). Returns `{ cancelled: [] }` on the happy path.

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

## Schema and lint

- **`dsl_kit_schema`** — the loaded DSL's type-level schema as JSON.
  Envelope: `{ "wired": bool, "schema": <NodeSchema JSON> | null }`.
  `wired=false` means the host has not implemented
  `DslHost::schema_json`: the DSL is reachable over MCP but has not
  opted into schema reflection.
- **`dsl_kit_lint`** — run the host's lint pass over the currently
  loaded AST. Envelope:
  `{ "wired": bool, "diagnostics": [...] | null }`. `wired=false`
  means the host has not implemented `DslHost::lint_json` — a
  lint-less DSL, which is not the same as a clean one. Lint is a pull:
  `dsl_kit_load` never runs it, so call this tool when you want the
  advisory diagnostics. They never block a load.

## Lifecycle

- **`dsl_kit_load`** — parse a JSON document, build the typed AST,
  swap it into the host, and reset the stepper. Registered breakpoints
  are cleared on success, since old `NodeId`s mean nothing against the
  new AST. An optional `sources` object (names mapped to
  `{"json": "…"}` / `{"text": "…"}` entries) turns the call into a
  bundle load that resolves `{"$import": "name"}` node positions and
  adds an `imports` report (dependencies + digest) to the envelope.
  Envelopes: `{ "ok": true, "dsl", "root", "ast_size" }` on success,
  `{ "ok": false, "diagnostics": [...] }` when the document failed
  conformance, and `{ "ok": false, "error": "…" }` for prose-only
  failures, including hosts that never opted into loading.
- **`dsl_kit_reset`** — reset the host's stepper. Breakpoints are
  left in place.

## Typical workflow

1. `dsl_kit_info` + `dsl_kit_ast` to see the loaded program.
2. `dsl_kit_breakpoint_add` to pause on interesting nodes.
3. `dsl_kit_step` (`mode: "to_yield"`) until suspension.
4. `dsl_kit_state` to inspect where the stepper is.
5. `dsl_kit_resolve` to supply a call response, then step again.
6. `dsl_kit_reset` when starting over.

## Fan-out workflow (`Par` node)

When the loaded DSL supports parallel branches, a `Par` node's `step`
emits N suspensions at once (one per child `Call`). The recommended
tool sequence is:

1. `dsl_kit_step { mode: "to_yield" }` — enters the `Par` and blocks.
2. `dsl_kit_pending` — returns N entries, each with a stable `id`.
3. `dsl_kit_resolve_by_id { id, ok: "..." }` × N — resolve each slot
   in any order (does not have to match declaration order).
4. `dsl_kit_step` — the reducer folds the slot values and the
   pipeline advances.

On the FailFast variant, step 3 for one slot uses
`{ id, err: { code, message } }`. The next `dsl_kit_step` then
returns an error, and `dsl_kit_take_cancellations` returns the ids
of the siblings that were cancelled as a consequence.
