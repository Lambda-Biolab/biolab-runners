#!/usr/bin/env bash
# Pre-commit pyright hook: detect src dir, then run pyright.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SRC_DIR=$(uv run python scripts/detect_src_dir.py)
uv run pyright "$SRC_DIR" tests/
