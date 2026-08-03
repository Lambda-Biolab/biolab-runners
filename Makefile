# Makefile — one entry point for every repeatable action in this repo.
#
# CANONICAL VERSION. This is the single source of truth for all
# Lambda-Biolab Python repos. To update, edit THIS FILE in opencode-config
# and propagate via `make propagate-makefile` (see scripts/propagate.py).
#
# Per-repo overrides (source dir, test dir, pyright includes) are auto-
# detected at parse time from the repo layout. Each repo MUST use this
# exact Makefile — no per-repo modifications. If you need a per-repo
# target, add it here and add the corresponding opt-out logic.
#
# Tool versions (single source of truth):
#   - gitleaks 8.30.1 (matches GITLEAKS_VERSION in .github/workflows/secrets.yml)
#   - bandit: dev dep in pyproject.toml
#   - mutation testing: local-only at pre-push (NOT in CI)

# --- Auto-detect repo layout ---
# Detect: the package directory under the repo root. This is the
# directory containing the source code (the package that gets
# imported). Detection rules in order:
#   1. src/<pkg>/ layout: src/ contains a single Python package
#   2. Flat layout: <pkg>/ is a top-level directory next to tests/
#   3. Fallback: src/
# Detect package dir: prefer src/<pkg>/ layout, else flat layout.
# Uses separate $(shell ...) calls for clarity (make's $(shell ...) has
# issues with multi-line if/then/else/fi).
HAS_SRC_LAYOUT := $(shell [ -d src ] && [ -d src/*/  ] && [ ! -f src/__init__.py ] && echo y || echo n)
PACKAGE_DIR := $(shell \
    if [ "$(HAS_SRC_LAYOUT)" = "y" ]; then \
        ls -1 src/ | head -1; \
    else \
        ls -1d */ 2>/dev/null | grep -vE '^(tests|scripts|docs|tools|fixtures|examples|mutants|smoke_test|node_modules|batch_diagnostics|dist|build)/$$' | head -1 | tr -d '/'; \
    fi)

ifeq ($(PACKAGE_DIR),)
    PACKAGE_DIR := src
endif

# Source paths: prefer src/<pkg>/ layout if it exists, else flat
SRC := $(if $(wildcard src/$(PACKAGE_DIR)),src/$(PACKAGE_DIR),$(PACKAGE_DIR))
TEST_DIR := $(if $(wildcard tests),tests,test)

# Tools
PYTHON := uv run python
UV := uv

.PHONY: help setup_dev install_tools lint format type_check complexity test \
        validate validate-branch quick_validate pre-push-validate \
        secrets bandit \
        mutate mutate-changed mutate-stats mutate-score _mutate-prepare \
        propagate-makefile clean

# Show the detected source layout (debug aid)
debug-layout:
	@echo "PACKAGE_DIR = $(PACKAGE_DIR)"
	@echo "SRC = $(SRC)"
	@echo "TEST_DIR = $(TEST_DIR)"

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup_dev:  ## Install uv, sync all dependency groups, install pre-commit hooks
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	$(UV) sync --all-groups
	$(UV) run pre-commit install --install-hooks
	$(UV) run pre-commit install --hook-type commit-msg
	@echo "dev environment ready. Try: make pre-push-validate"

# --- Lint / format / type / complexity / test (read-only gates) ---
lint:  ## Ruff lint + format check
	$(UV) run ruff check $(SRC) $(TEST_DIR)
	$(UV) run ruff format --check $(SRC) $(TEST_DIR)

format:  ## Ruff auto-fix + format
	$(UV) run ruff check --fix $(SRC) $(TEST_DIR)
	$(UV) run ruff format $(SRC) $(TEST_DIR)

type_check:  ## Pyright on src/ AND tests/ (tests are first-class code)
	$(UV) run pyright $(SRC) $(TEST_DIR)

# Auto-detect complexity threshold from pyproject.toml [tool.complexipy].
# Falls back to 15 (org default). Override at invocation:
#   make COMPLEXITY_MAX=22 complexity
# Auto-detect complexity threshold from [tool.complexipy] in pyproject.toml.
# Falls back to 15 (org default). Override at invocation:
#   make COMPLEXITY_MAX=22 complexity
COMPLEXITY_MAX ?= $(shell uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb')).get('tool', {}).get('complexipy', {}).get('max-complexity-allowed', 15))" 2>/dev/null || echo 15)

complexity:  ## Complexipy gate (threshold from [tool.complexipy] in pyproject.toml)
	$(UV) run complexipy $(SRC) --max-complexity-allowed $(COMPLEXITY_MAX)

test:  ## Run pytest with coverage
	$(UV) run pytest $(TEST_DIR) -m "not slow" --cov=$(SRC) --cov-report=term-missing

# --- Branch coverage gate on changed source files (vs main) ---
# Only runs if a tracked file is changed. Tracked files are anything
# under the package source dir. Coverage threshold mirrors the org's
# standard cov-fail-under value (typically 80% for critical-path modules).
# This is a fast proxy for mutation testing — for a deeper check, use
# mutmut (see CONTRIBUTING.md "Mutation Testing" section).
validate-branch:
	@CHANGED=$$(git diff --name-only --diff-filter=ACMR origin/main 2>/dev/null | \
	    grep -E '^$(SRC)/.*\.py$$' || true); \
	if [ -n "$$CHANGED" ]; then \
		echo "pre-push: checking branch coverage on changed files:"; \
		echo "$$CHANGED" | sed 's/^/  /'; \
		$(UV) run pytest $(TEST_DIR) -m "not slow" --no-header -q \
			--cov=$$(echo "$$CHANGED" | tr '\n' ',' | sed 's/,\$$//' | sed 's|/|.|g;s|\.py\$$||g') \
			--cov-branch --cov-fail-under=80 2>&1 | tail -20 || \
			(echo ""; \
			echo "Branch coverage < 80% on a changed file. Add a test that"; \
			echo "  covers the new branch."; \
			exit 1); \
	else \
		echo "pre-push: no source files changed, skipping branch coverage check"; \
	fi

# --- Secrets scanning (gitleaks) ---
# Mandatory at pre-push. The gitleaks binary is installed via
# `make install_tools` (downloaded to ~/.local/bin/gitleaks).
# The CI version is pinned via GITLEAKS_VERSION in .github/workflows/secrets.yml.
# The local install_tools Makefile target uses the SAME version (single
# source of truth = install_tools target). Bump the version in BOTH places.
secrets:
	@command -v gitleaks >/dev/null 2>&1 || { echo "gitleaks not installed \u2014 run 'make install_tools'"; exit 2; }
	@if [ -f .gitleaks.toml ]; then CONFIG=.gitleaks.toml; elif [ -f .github/gitleaks.toml ]; then CONFIG=.github/gitleaks.toml; else echo "no gitleaks config (.gitleaks.toml or .github/gitleaks.toml)"; exit 2; fi; gitleaks detect --no-git --source . --config "$$CONFIG"

# --- Bandit (Python SAST) ---
# Mandatory at pre-push. Bandit is a dev dependency in pyproject.toml.
# The -ll flag shows MEDIUM+ severity (skips LOW). The -ii flag shows
# MEDIUM+ confidence (skips LOW). HIGH severity/confidence only would
# be -lll / -iii — too permissive for our use case.
bandit:
	@command -v uv >/dev/null 2>&1 || { echo "uv not installed"; exit 2; }
	@$(UV) run --active bandit -r $(SRC) -ll -ii

# --- Install tools (gitleaks binary; bandit is a dev dep) ---
# gitleaks is shipped as a standalone binary; bandit is a dev dependency
# in pyproject.toml and doesn't need a separate install step.
# Version is 8.30.1 \u2014 must match GITLEAKS_VERSION in .github/workflows/secrets.yml.
install_tools:
	@echo "Installing gitleaks 8.30.1..."
	@if ! command -v gitleaks >/dev/null 2>&1; then \
		mkdir -p $(HOME)/.local/bin; \
		curl -sL https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz \
			| tar -xz -C $(HOME)/.local/bin gitleaks; \
		chmod +x $(HOME)/.local/bin/gitleaks; \
		echo "  installed to $(HOME)/.local/bin/gitleaks"; \
	else \
		echo "  gitleaks already present at $$(command -v gitleaks)"; \
	fi
	@echo "Bandit is a dev dependency in pyproject.toml (run 'uv sync --all-groups')."

# --- Top-level gates ---
# validate: fast read-only quality gates. Mirrors what CI runs.
# pre-push-validate: validate + branch coverage on changed files + mutation
# testing on changed tracked functions. Called by scripts/git-hooks/pre-push.
# validate does NOT include mutation testing (too slow for CI).
quick_validate: lint type_check complexity  ## Fast iteration: skip tests
	@echo "quick validate passed"

validate: lint type_check complexity test secrets bandit  ## Full pre-push gate: lint + types + complexity + tests + secrets + bandit (mirrors CI)

# The mandatory pre-push gate. Runs validate (fast quality gates) PLUS
# branch coverage on changed files PLUS mutation testing on changed
# tracked functions. The mutation gate is fast (sub-second) when no
# tracked function changed thanks to mutmut's function_hashes cache;
# slow (30+ min) only when tracked functions actually changed \u2014 exactly
# when you want the gate. CI does NOT run this.
pre-push-validate: validate validate-branch mutate-changed

# --- Mutation testing (mutmut) ---
# Local-only at pre-push (NOT in CI). Weekly CI mutation was removed in
# v1.4.0 because the same code was re-tested every week; pre-push with
# function_hashes cache is faster in the common case and only slow when
# the tracked function actually changed (which is the case you want
# the gate for). See mutation-testing skill in opencode-config.
#
# Why we mirror the full source tree to mutants/ before mutmut runs:
# mutmut only copies files listed in [tool.mutmut].source_paths into
# mutants/, then runs pytest in mutants/. If a test transitively
# imports a module NOT in source_paths (e.g. test_grid_generator.py
# imports from pocket_finder.py which isn't a mutated file), the
# import fails and pytest can't even COLLECT tests, let alone run
# them. We avoid that by pre-populating mutants/$(SRC) with the full
# source tree, then running mutmut on top. The mutated files overwrite
# the originals; the rest are unchanged. This is independent of what
# source_paths lists — mutmut still only mutates the source_paths files.
mutate:
	@mkdir -p mutants && rm -rf mutants/$(SRC) && cp -r $(SRC) mutants/$(SRC)
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	$(UV) run mutmut run

# Internal target: mirror the full source tree into mutants/ so
# pytest can collect tests that transitively import non-mutated
# modules. See mutate target docstring for the rationale.
_mutate-prepare:
	@mkdir -p mutants
	@rm -rf mutants/$(SRC)
	@mkdir -p $$(dirname mutants/$(SRC))
	@cp -r $(SRC) mutants/$(SRC)
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

mutate: _mutate-prepare  ## Run mutmut on the FULL critical-path scope
	$(UV) run mutmut run

mutate-changed: _mutate-prepare
	@CHANGED=$$(git diff --name-only --diff-filter=ACMR origin/main 2>/dev/null | \
	    grep -E '^$(SRC)/.*\.py$$' || true); \
	if [ -z "$$CHANGED" ]; then \
		echo "mutate-changed: no source files changed, nothing to mutate"; \
		exit 0; \
	fi; \
	echo "mutate-changed: source files changed ($$CHANGED)"; \
	echo "  running mutmut on full source_paths (mutmut only mutates those, not the changed files)"; \
	$(UV) run mutmut run || true

mutate-stats:
	$(UV) run mutmut results

mutate-score:
	@$(UV) run mutmut results 2>&1 | tail -1 | awk -F'[/ ]' '/killed/ { \
		killed=$$5; total=$$7; \
		if (total > 0) printf "Mutation score: %.1f%% (%d/%d killed)\n", (killed/total)*100, killed, total; \
		else printf "Mutation score: 0%% (no mutants)\n" \
	}'

# --- Propagate this Makefile to other repos (opencode-config only) ---
# Reads ~/.config/opencode/repos.yaml and copies the canonical Makefile
# to every repo. Idempotent. Only available when MAKEFILE_PROPAGATE=1
# is set (so running `make propagate-makefile` in a consumer repo
# doesn't accidentally clobber it).
propagate-makefile:  ## [opencode-config only] Push this Makefile to all Lambda-Biolab repos
	@if [ "$$(MAKEFILE_PROPAGATE)" != "1" ]; then \
		echo "propagate-makefile is for opencode-config only. Set MAKEFILE_PROPAGATE=1 to use."; \
		exit 1; \
	fi
	@echo "This target is a placeholder; use scripts/propagate_makefile.py in opencode-config."

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .complexipy_cache .pyright .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned"
