#!/usr/bin/env bash
# Pre-commit ruff hook: detect src dir, then run ruff check --fix.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SRC_DIR=$(uv run python scripts/detect_src_dir.py)
uv run ruff check --fix "$SRC_DIR" tests/
