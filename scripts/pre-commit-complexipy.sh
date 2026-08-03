#!/usr/bin/env bash
# Pre-commit complexipy hook: detect src dir + complexity max, then run.
# Used as the entry in .pre-commit-config.yaml.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SRC_DIR=$(uv run python scripts/detect_src_dir.py)
COMPLEXITY_MAX=$(uv run python scripts/detect_complexity_max.py)
uv run complexipy "$SRC_DIR" --max-complexity-allowed "$COMPLEXITY_MAX"
