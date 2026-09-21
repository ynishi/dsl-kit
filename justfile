# dsl-kit developer recipes. The procedure of record is CONTRIBUTING.md;
# these recipes are how it is carried out.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# One-time per clone: route git hooks through .githooks (Conventional Commits check on commit-msg)
setup:
    git config core.hooksPath .githooks
    chmod +x .githooks/*
    @echo "core.hooksPath = $(git config core.hooksPath)"

# Start work: creates topic/<name> off up-to-date main
branch name:
    git switch main
    git pull --ff-only origin main
    git switch -c "topic/{{name}}"

# Tests run per crate on purpose: `cargo test --workspace` links every
# crate in parallel and is not welcome on shared machines.

# The CI lanes, locally: fmt, clippy -D warnings, tests per crate
check:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    for p in $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name'); do \
        echo "== cargo test -p $p"; cargo test -q -p "$p"; \
    done

# Lint the commit headers this branch adds on top of `base`
lint-commits base="origin/main":
    ./.githooks/lint-range "{{base}}..HEAD"

# Merges are rebase-only (repository setting), so every commit header
# lands on main as-is; the title is for the PR page only.

# Publish the branch and open the PR: `just pr "<title>" <body.md>`
pr title body:
    just lint-commits
    git push -u origin "$(git branch --show-current)"
    gh pr create --base main --title "{{title}}" --body-file "{{body}}"
