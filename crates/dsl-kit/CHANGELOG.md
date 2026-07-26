# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Keyed child slots gain a scalar-valued shape
  (`BTreeMap<String, String>`, `BTreeMap<String, i64>`, …). See the
  workspace `CHANGELOG.md` and gh issue #5 for the full breakdown
  across `dsl-kit-schema` / `dsl-kit-macros` / `dsl-kit-parse`. The
  recursive keyed shape (`BTreeMap<String, Box<Self>>`) is unchanged.

### Breaking

- `dsl_kit_schema::ChildSchema` gains a `value_shape:
  ChildValueShape` field. Hand-written struct literals must supply
  it. The recommended migration is to switch to
  `ChildSchema::recursive(name, mult)` for the historical shapes
  (`One` / `Optional` / `Many`, plus keyed slots whose values are
  the same enum), or `ChildSchema::scalar_map(name, ty)` for the
  new scalar-value keyed shape. Both constructors default the
  `value_shape` to the correct variant, so callers do not have to
  reference `ChildValueShape` directly.

## [0.5.1](https://github.com/ynishi/dsl-kit/compare/v0.5.0...v0.5.1) - 2026-07-26

### Fixed

- *(core)* record CollectAll Par failure against the Par's direct child

## [0.5.0](https://github.com/ynishi/dsl-kit/compare/v0.4.0...v0.5.0) - 2026-07-25

### Added

- cfg-example demonstrates the keyed Map primitive end to end
- keyed child slots reach ParseTree, the JSON bridge and DslBuild
- dsl-kit-macros recognise BTreeMap<String, T> keyed slots
- keyed child slots get a canonical text syntax
- dsl-kit-schema Multiplicity::Map + non_exhaustive foundation

### Fixed

- no-empty-many-children enforced a guarantee the schema never made
