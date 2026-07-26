# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
