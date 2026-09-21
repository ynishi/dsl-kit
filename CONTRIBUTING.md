# Contributing to dsl-kit

This file is the procedure of record for working on the repository.
The `justfile` carries it out; CI enforces the parts that can be
enforced.

## Setup

```sh
just setup     # routes git hooks through .githooks (commit-msg check)
```

## Branches

Work happens on a topic branch off up-to-date `main`:

```sh
just branch <name>     # -> topic/<name>
```

## Commits

Every commit header is a [Conventional Commit](https://www.conventionalcommits.org/):

```text
<type>(<scope>)!: <summary>
```

- `type`: `feat` `fix` `docs` `style` `refactor` `perf` `test` `build`
  `ci` `chore` `revert`.
- `scope` (optional): the crate short name(s) — `parse`, `schema`,
  `core,macros`, `examples` — or the tool (`release-plz`).
- `!` marks a **breaking change** (equivalently a `BREAKING CHANGE:`
  footer).

The `commit-msg` hook rejects anything else locally, and the `commits`
CI job rejects it on the pull request.

### Versioning on 0.x — read this before choosing `feat` vs `feat!`

Releases are cut by [release-plz](https://release-plz.dev), which takes
the next version from the commit headers since the last tag using
[`next_version`](https://docs.rs/next_version). While the workspace is
`0.x`:

| headers since the last tag | next version |
|---|---|
| `fix`, `feat`, anything else | **patch** (`0.11.1` → `0.11.2`) |
| any `!` / `BREAKING CHANGE:` | **minor** (`0.11.1` → `0.12.0`) |

So a feature alone does not produce a minor release. Mark a commit
`!` when it changes what the kit accepts or produces — an API shape,
the canonical text a generated grammar accepts, the JSON a front-end
takes, a diagnostic code — even when `cargo semver-checks` calls it
compatible. If a release needs a different number than the headers
imply, bump `[workspace.package] version` (and the workspace
dependency pins) on `main` yourself, then check the release PR
release-plz produces from that state before merging it.

## Pull requests

```sh
just check                       # fmt, clippy -D warnings, tests per crate
just pr "<title>" <body.md>      # lints headers, pushes, opens the PR
```

Pull requests are merged by **rebase only** — squash and merge
commits are disabled in the repository settings — so every commit
header reaches `main` exactly as written. That is what release-plz
reads for the version and the changelog, and what `git log` shows
about how something was built. Keep commits self-contained and
conventional; the PR title is for the PR page only.

CI on a pull request: the header lint, `cargo fmt --check`,
`cargo clippy --workspace --all-targets -D warnings`,
`cargo test --workspace`, and cargo-dist's `plan`.

## Releases

1. A push to `main` runs release-plz, which opens or updates
   `chore: release vX.Y.Z` with the version and changelogs computed
   from the headers (see the table above).
2. Merge it with `gh pr merge <N> --rebase`. release-plz then publishes
   the crates to crates.io in dependency order and pushes the `vX.Y.Z`
   tag.
3. The tag triggers cargo-dist (`release.yml`), which builds the
   multi-target binaries and creates the GitHub Release.

The root `CHANGELOG.md` is written by hand under `## [Unreleased]` as
features land; release-plz maintains `crates/dsl-kit/CHANGELOG.md`
from the headers.
