"""Per-PR smoke test gate comparator for biolab-runners.

Compares the smoke test output to a committed baseline. The actual
smoke test is run by `make smoke_test`; this script is invoked
after the smoke test succeeds and decides whether the run passes
or fails the gate.

This is the deepvariant-arm64-Linux pattern: every PR runs the
actual scientific code (OpenMM MD on a real GPU) and is gated on
the result. The comparison criteria are:

- error is None                           (run completed cleanly)
- num_atoms matches baseline              (same molecular system)
- topology_lines matches baseline         (correct system topology)
- energy_last_row within 1% of baseline   (scientific correctness)
- ns_per_day >= 80% of baseline           (no catastrophic slowdown)

The script does NOT spin up a GPU or run the smoke test itself
— that happens in the workflow (via the self-hosted GPU runner
and `make smoke_test`). This script is the post-run gate.

Usage:
    compare.py <smoke-verify.json> [--baseline PATH]

Exit codes:
    0   pass
    1   gate failure (output diverged from baseline)
    2   file not found or invalid JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Tolerance: energy must be within 1% of baseline (deterministic MD)
ENERGY_TOLERANCE = 0.01

# Performance: ns_per_day must be >= 80% of baseline (no catastrophic slowdown)
PERF_FLOOR = 0.80


def load_baseline(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: baseline not found at {path}", file=sys.stderr)
        print(f"Run 'make smoke-baseline' once on a known-good GPU to generate it.", file=sys.stderr)
        sys.exit(2)
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: baseline at {path} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)


def load_actual(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: smoke verify output not found at {path}", file=sys.stderr)
        sys.exit(2)
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: {path} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)


def compare(actual: dict, baseline: dict) -> tuple[bool, str]:
    """Return (passed, reason)."""
    if actual.get("error"):
        return False, f"❌ smoke test reported error: {actual['error']!r}"

    if actual.get("num_atoms") != baseline.get("num_atoms"):
        return False, (
            f"❌ num_atoms mismatch: actual={actual.get('num_atoms')} "
            f"baseline={baseline.get('num_atoms')}"
        )

    if actual.get("topology_lines") != baseline.get("topology_lines"):
        return False, (
            f"❌ topology_lines mismatch: actual={actual.get('topology_lines')} "
            f"baseline={baseline.get('topology_lines')} (system changed?)"
        )

    actual_pe = float(actual.get("energy_last_row", [0, 0, 0, 0])[2])
    baseline_pe = float(baseline.get("energy_last_row", [0, 0, 0, 0])[2])
    if baseline_pe == 0:
        return False, "❌ baseline PE is 0 (invalid baseline)"
    drift = abs(actual_pe - baseline_pe) / abs(baseline_pe)
    if drift > ENERGY_TOLERANCE:
        return False, (
            f"❌ energy drift: actual={actual_pe:.4f} baseline={baseline_pe:.4f} "
            f"drift={drift:.2%} (>{ENERGY_TOLERANCE:.0%} tolerance) — scientific regression"
        )

    actual_perf = actual.get("ns_per_day", 0)
    baseline_perf = baseline.get("ns_per_day", 0)
    if baseline_perf == 0:
        return False, "❌ baseline ns_per_day is 0 (invalid baseline)"
    perf_ratio = actual_perf / baseline_perf
    if perf_ratio < PERF_FLOOR:
        return False, (
            f"❌ performance regression: actual={actual_perf:.1f} ns/day "
            f"baseline={baseline_perf:.1f} ns/day "
            f"ratio={perf_ratio:.2f} (<{PERF_FLOOR:.0%})"
        )

    return True, (
        f"✅ PASS: num_atoms={actual.get('num_atoms')}, "
        f"PE drift={drift:.3%}, perf={perf_ratio:.2f}x baseline"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("smoke_verify", type=Path,
                   help="Path to smoke_verify.json from the smoke test run")
    p.add_argument("--baseline", type=Path,
                   default=Path(__file__).parent.parent / "smoke_test" / "baseline.json",
                   help="Path to baseline.json (default: smoke_test/baseline.json)")
    args = p.parse_args()

    baseline = load_baseline(args.baseline)
    actual = load_actual(args.smoke_verify)

    def fmt_pe(d): 
        row = d.get("energy_last_row")
        if not row: return "N/A"
        try: return f"{float(row[2]):.4f}"
        except (ValueError, TypeError, IndexError): return "N/A"
    def fmt_perf(d): 
        v = d.get("ns_per_day")
        return f"{v:.1f}" if v is not None else "N/A"
    print(f"baseline: num_atoms={baseline.get('num_atoms')}, "
          f"PE={fmt_pe(baseline)}, "
          f"perf={fmt_perf(baseline)} ns/day")
    print(f"actual:   num_atoms={actual.get('num_atoms')}, "
          f"PE={fmt_pe(actual)}, "
          f"perf={fmt_perf(actual)} ns/day")
    print()

    passed, reason = compare(actual, baseline)
    print(reason)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
