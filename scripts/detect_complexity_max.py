"""Detect the complexipy threshold from [tool.complexipy] in pyproject.toml.

Mirrors the COMPLEXITY_MAX detection in the canonical Makefile.
Returns 15 as fallback (the org default).

Run from the repo root:
    uv run python scripts/detect_complexity_max.py

Exits 0 and prints the threshold (e.g. "10", "15", "22") to stdout.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def detect() -> int:
    pyproject = Path.cwd() / "pyproject.toml"
    if not pyproject.exists():
        return 15  # org default
    try:
        data = tomllib.loads(pyproject.read_text())
    except Exception:
        return 15
    cc = data.get("tool", {}).get("complexipy", {})
    # Both key names are used in the wild
    threshold = cc.get("max-complexity-allowed") or cc.get("max-complexity") or 15
    try:
        return int(threshold)
    except (TypeError, ValueError):
        return 15


if __name__ == "__main__":
    print(detect())
    sys.exit(0)
