#!/usr/bin/env bash
# Shared Conventional Commits header check. Sourced by `commit-msg`
# (local, via `just setup`) and `lint-range` (CI + `just lint-commits`),
# so the three never disagree on what a header is.
#
#   <type>(<scope>)!: <summary>      scope and "!" optional
#
# The "!" matters: this workspace is 0.x and release-plz (next_version)
# bumps the PATCH for feat and fix; only a breaking change — "!" after
# the type/scope or a "BREAKING CHANGE:" footer — bumps the MINOR.

CONVENTIONAL_TYPES='feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert'
CONVENTIONAL_RE="^(${CONVENTIONAL_TYPES})(\([a-z0-9][a-z0-9,_ -]*\))?!?: [^ ].*"

# check_header <header>  →  0 if conventional (or a merge / revert line), 1 otherwise
check_header() {
  local header="$1"
  case "$header" in
    "Merge "*|"Revert \""*) return 0 ;;
  esac
  [[ "$header" =~ $CONVENTIONAL_RE ]]
}

explain_header() {
  local header="$1"
  cat >&2 <<MSG
not a Conventional Commit header:
    $header
expected:
    <type>(<scope>)!: <summary>
    types: ${CONVENTIONAL_TYPES//|/ }
0.x versioning (release-plz / next_version): feat and fix bump the PATCH;
only "!" (or a "BREAKING CHANGE:" footer) bumps the MINOR. See CONTRIBUTING.md.
MSG
}
