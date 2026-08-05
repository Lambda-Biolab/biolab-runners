#!/usr/bin/env bash
# lint-makefile.sh — Lint a Lambda-Biolab Makefile against the unified canonical.
# Usage: lint-makefile.sh [path/to/Makefile]
# Exit 0 = pass, 1 = failures found.
#
# This is the Lambda-Biolab variant of the qte77 linter. The qte77 linter
# requires .ONESHELL/.SILENT which the Lambda-Biolab canonical does NOT
# have. This linter checks for what Lambda-Biolab ACTUALLY uses.

set -uo pipefail

MAKEFILE="${1:-Makefile}"
ERRORS=0

fail() {
  echo "FAIL: $1"
  ERRORS=$((ERRORS + 1))
}

if [[ ! -f "$MAKEFILE" ]]; then
  echo "ERROR: $MAKEFILE not found"
  exit 1
fi

echo "Linting $MAKEFILE against Lambda-Biolab canonical patterns"

# --- Required directives ---

grep -q '\.PHONY' "$MAKEFILE" \
  || fail ".PHONY declaration not found"

# --- Tool version pins (Lambda-Biolab-specific) ---

grep -qE '^RUFF_VERSION *:=' "$MAKEFILE" \
  || fail "RUFF_VERSION pin not set (e.g. RUFF_VERSION := 0.15.10)"

# --- Required recipes (defined as actual make targets) ---

# Common: every recipe a Lambda-Biolab Python repo MUST have
COMMON_RECIPES=("help" "setup_dev" "validate" "pre-push-validate" "validate-branch" "test" "lint" "format" "type_check" "complexity" "mutate-changed" "lint-makefile" "clean")
for recipe in "${COMMON_RECIPES[@]}"; do
  if ! grep -qE "^${recipe}:" "$MAKEFILE"; then
    fail "Required recipe '${recipe}:' not found"
  fi
done

# --- Tool path variables (Lambda-Biolab uses uv) ---

if ! grep -qE '^UV *:=' "$MAKEFILE"; then
  fail "UV variable not defined (use: UV := uv)"
fi

if ! grep -qE '^PYTHON *:=' "$MAKEFILE"; then
  fail "PYTHON variable not defined (use: PYTHON := uv run python)"
fi

# --- Help recipe should have ## description ---

if ! grep -qE '^help:.*##' "$MAKEFILE"; then
  fail "help recipe missing '## description'"
fi

# --- validate should compose lint + type_check + complexity + test (in any order) ---

VALIDATE_RECIPES=$(sed -n '/^validate:/,/^[a-zA-Z_-]/p' "$MAKEFILE" | grep -oE '^[a-z_-]+' | grep -vE '^(validate|lint|type_check|complexity|test|secrets|bandit|quick_validate|##|@echo|@\\)' | head -10)
for required in lint type_check complexity test; do
  if ! sed -n '/^validate:/,/^[a-zA-Z_-]/p' "$MAKEFILE" | grep -qE "\b${required}\b"; then
    fail "validate recipe must call ${required}"
  fi
done

# --- pre-push-validate should chain validate + validate-branch + mutate-changed ---

if ! grep -qE '^pre-push-validate:.*validate' "$MAKEFILE"; then
  fail "pre-push-validate must depend on validate"
fi
if ! grep -qE '^pre-push-validate:.*validate-branch' "$MAKEFILE"; then
  fail "pre-push-validate must depend on validate-branch"
fi
if ! grep -qE '^pre-push-validate:.*mutate-changed' "$MAKEFILE"; then
  fail "pre-push-validate must depend on mutate-changed"
fi
if ! grep -qE '^pre-push-validate:.*lint-makefile' "$MAKEFILE"; then
  fail "pre-push-validate must depend on lint-makefile (catches Makefile canonical drift at pre-push)"
fi

# --- Forbidden patterns from qte77 that Lambda-Biolab does NOT use ---

if grep -qE '^\.ONESHELL' "$MAKEFILE"; then
  fail ".ONESHELL is qte77 convention but NOT used in Lambda-Biolab canonical"
fi

if grep -qE '^\.SILENT' "$MAKEFILE"; then
  fail ".SILENT is qte77 convention but NOT used in Lambda-Biolab canonical"
fi

# --- Auto-detect layout (Lambda-Biolab uses HAS_SRC_LAYOUT / PACKAGE_DIR) ---

if ! grep -qE 'HAS_SRC_LAYOUT' "$MAKEFILE"; then
  fail "HAS_SRC_LAYOUT auto-detect not present (Lambda-Biolab canonical uses this)"
fi
if ! grep -qE 'PACKAGE_DIR' "$MAKEFILE"; then
  fail "PACKAGE_DIR auto-detect not present (Lambda-Biolab canonical uses this)"
fi

# --- pre-commit integration check (sanity) ---

# If .pre-commit-config.yaml exists, ensure pre-commit install is in setup_dev
if [[ -f "${MAKEFILE%Makefile}.pre-commit-config.yaml" ]] || [[ -f ".pre-commit-config.yaml" ]]; then
  if ! grep -qE 'pre-commit install' "$MAKEFILE"; then
    fail "pre-commit-config.yaml exists but Makefile setup_dev doesn't run 'pre-commit install'"
  fi
fi

# --- Final report ---

if [[ $ERRORS -gt 0 ]]; then
  echo ""
  echo "FAILED: $ERRORS lint errors"
  echo "See skill: lambda-biolab-makefile-conventions"
  exit 1
fi

echo "OK: passes all Lambda-Biolab Makefile conventions"
exit 0
