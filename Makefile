.DEFAULT_GOAL := help
.PHONY: help setup_dev ruff check_types check_complexity test validate check_links check_docs

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
setup_dev: ## Install all dependencies (dev + all extras)
	uv sync --all-extras

# ──────────────────────────────────────────────────────────────────────────────
# Quality assurance
# ──────────────────────────────────────────────────────────────────────────────
ruff: ## Format + lint (ruff format + ruff check --fix)
	uv run ruff format biolab_runners/ tests/
	uv run ruff check --fix biolab_runners/ tests/

check_types: ## Static type checking (pyright basic)
	uv run pyright biolab_runners/

check_complexity: ## Check cognitive complexity (max 15/function)
	uv run complexipy biolab_runners/ --max-complexity-allowed 15

test: ## Run tests
	uv run pytest tests/ -v --tb=short

check_links: ## Check links with lychee
	@if command -v lychee > /dev/null 2>&1; then \
		lychee --config .lychee.toml .; \
	else \
		echo "lychee not installed — see https://github.com/lycheeverse/lychee"; \
	fi

check_docs: ## Lint markdown files
	@if command -v markdownlint-cli2 > /dev/null 2>&1; then \
		markdownlint-cli2 "README.md" "AGENTS.md" "CLAUDE.md"; \
	else \
		echo "markdownlint-cli2 not installed — npm install -g markdownlint-cli2"; \
	fi

validate: ## Full gate: ruff → pyright → complexity → pytest
	$(MAKE) ruff
	$(MAKE) check_types
	$(MAKE) check_complexity
	$(MAKE) test
	@echo "All checks passed."

# ──────────────────────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
