---
name: AST feature (schema / parse trunk / multiplicity)
about: A new primitive on the AST surface — a new multiplicity, a new field shape, a new capture primitive
title: "[AST] <one-line summary>"
labels: enhancement, ast, needs-layers
---

## Why this template exists

An AST-shaped feature in dsl-kit is not done when the schema knows about it. The kit's identity axis is "write the AST in Rust, the toolchain supplies the rest" — that promise only holds if the new primitive **reaches every layer the toolchain covers**, not just the layer it was added at. Landing a `Multiplicity` variant while leaving the derive silent, or wiring a new primitive through parse and grammar while leaving no DSL that actually uses it, is a promise the kit does not deliver on.

Treat the checklist below as the definition of done for this issue. If a layer does not apply, say so explicitly — do not silently skip.

## Problem / motivation

<!-- What is the AST-shape gap and why does it matter? Concrete DSL example, not "for symmetry with X". -->

## Proposal

<!-- The new schema shape / capture primitive / declaration. Design alternatives if any. -->

## Layer coverage (definition of done)

An AST feature is "all-layers-or-not-done". Every layer below either lands a change or has an explicit reason why it does not.

- [ ] **`dsl-kit-schema`** — the declaration itself: `Multiplicity` / `ChildSchema` / `FieldSchema` / `NodeSchema` shape, wire format via `NodeSchema::to_json`, `as_str`, doc comments.
- [ ] **`dsl-kit-macros`** — the Rust-side spelling the derive recognises (`BTreeMap<String, T>` / `NonEmpty<T>` / a `#[dsl(...)]` attribute, whichever fits), across `#[derive(DslNode)]` / `#[derive(DslSchema)]` / `#[derive(DslBuild)]` / `#[derive(DslExec)]`.
- [ ] **`dsl-kit-parse` — parse trunk** — `ParseTree` fields / accessors, so a well-formed tree can express the primitive at all.
- [ ] **`dsl-kit-parse` — conformance** — `check_conformance` arm plus its own diagnostic slug(s) in `codes::`; a violation surfaces as an error at build time, not a silent drop.
- [ ] **`dsl-kit-parse` — JSON bridge** — `serde_bridge::from_json_value` route, with a documented JSON shape for the primitive; front-end round-trip stays canonical (sort / dedupe / etc. as required by the trunk).
- [ ] **`dsl-kit-parse` — grammar generation** — `schema_gen::grammar_from_schema` arm producing the canonical text spelling, plus a `Peg` primitive if none of the existing ones fits.
- [ ] **`dsl-kit-parse` — grammar checks** — `is_nullable` / `collect_first` / `walk_peg` / `find_by_id` / `peg_id` arms for any new `Peg` variant.
- [ ] **`dsl-kit-parse` — example synthesis** — `example_gen::examples_from_grammar` emits an example that survives its own toolchain (parses, conforms, builds).
- [ ] **`dsl-kit-parse` — DslBuild helper** — `build_child_*` / `build_field_*` sibling so `#[derive(DslBuild)]` can turn the tree into a typed AST value.
- [ ] **`dsl-kit-lint`** — declared constraints (non-empty, arity bounds, ordering, uniqueness) that were previously invented by heuristic now check the declaration; `LINT_DECLS` in lockstep with `Rule` impls; message honest about what the schema actually promises.
- [ ] **`examples/`** — at least one example DSL exercises the primitive end to end (`#[derive]` → conformance → typed AST). Without this the feature has no `/jikki` User consumption path and no downstream regression bell.
- [ ] **MCP surface** — the example above is served by its `*-mcp` binary, so `mcp__<server>__dsl_kit_schema` / `dsl_kit_load` / `dsl_kit_step` / `dsl_kit_resolve` all reflect the new primitive. This is the layer a downstream host actually depends on.
- [ ] **CHANGELOG** — the entry names what is breaking (SemVer-Trick permitting, `#[non_exhaustive]` softens this — but only for the enum, not for `ParseTree` or `ChildSchema`).
- [ ] **Decisions and rejected alternatives recorded** on the tracking issue (or an ADR / design note in the repo), so the next person to touch this primitive does not re-argue them.

**If a layer does not apply**, tick its checkbox and add a one-line reason. "Does not apply" is a real answer; silent omission is not.

## Skip reasons (fill in only for layers that do not apply)

<!-- Example:
- MCP surface: does not apply. This change is a pure `Multiplicity` variant with no runtime shape yet; MCP coverage lands with the derive-and-examples cycle.
-->

## Rejected alternatives

<!-- What did you consider and drop? Why? Future readers need this to not re-litigate. -->

## Verification

- [ ] `cargo test --workspace` green
- [ ] `cargo clippy --workspace --all-targets` clean
- [ ] `cargo fmt --all --check` clean
- [ ] `/jikki` Phase 2 exercises the new primitive through the example DSL's MCP surface (not just `--version`)
- [ ] The 3 pre-commit checkers pass on the diff

## Cycle plan (for staged rollouts)

<!-- If the work naturally splits into cycles (schema → derive → parse → grammar → examples), list them here with per-cycle Files to Modify estimates. Otherwise delete this section. -->
