#!/usr/bin/env bash
# Pre-commit ruff-format hook: detect src dir, then run ruff format.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SRC_DIR=$(uv run python scripts/detect_src_dir.py)
uv run ruff format "$SRC_DIR" tests/
