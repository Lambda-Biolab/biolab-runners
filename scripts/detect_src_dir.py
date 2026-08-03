"""Detect the source directory of a Python repo.

Mirrors the PACKAGE_DIR detection in the canonical Makefile:
  1. src/<pkg>/ layout: src/ contains a single Python package
  2. Flat layout: <pkg>/ is a top-level directory next to tests/
  3. Fallback: src/

Used by the pre-commit-config hooks (ruff, ruff-format, pyright,
complexipy) to avoid hardcoding src/ in the config — the same
canonical .pre-commit-config.yaml works for all 4 Lambda-Biolab
repos regardless of their layout.

Run from the repo root:
    uv run python scripts/detect_src_dir.py

Exits 0 and prints the source dir (e.g. "bioml_tools" or
"src/vastai_gpu_runner") to stdout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def detect() -> str:
    """Return the source directory (relative to repo root)."""
    repo_root = Path.cwd()

    # Pattern 1: src/<pkg>/ layout. Check if src/ exists AND contains
    # a single package (subdirectory with __init__.py) AND src/ itself
    # has no __init__.py (i.e. it's a namespace, not a package).
    src_dir = repo_root / "src"
    if src_dir.is_dir() and not (src_dir / "__init__.py").exists():
        candidates = [
            src_dir / p
            for p in os.listdir(src_dir)
            if (src_dir / p).is_dir() and (src_dir / p / "__init__.py").exists()
        ]
        if len(candidates) == 1:
            return f"src/{candidates[0].name}"

    # Pattern 2: flat layout. Look for a top-level directory that has
    # __init__.py (i.e. it's a Python package) and is not a known
    # non-package dir.
    excluded = {
        "tests", "test", "scripts", "docs", "doc", "tools",
        "fixtures", "examples", "node_modules", "dist", "build",
        "mutants", "smoke_test", "batch_diagnostics", ".github",
        ".opencode", ".git", ".venv", "__pycache__",
    }
    for entry in sorted(os.listdir(repo_root)):
        if entry in excluded or entry.startswith("."):
            continue
        candidate = repo_root / entry
        if candidate.is_dir() and (candidate / "__init__.py").exists():
            return entry

    # Fallback: assume src/ layout
    return "src"


if __name__ == "__main__":
    print(detect())
    sys.exit(0)
