"""Per-PR boltz-2 gate comparator.

Compares the boltz-2 smoke run output to a committed baseline.
Run AFTER run_boltz2_smoke.py succeeds; decides whether the
PR passes the gate.

This is the deepvariant-arm64-Linux pattern for structure
prediction: every PR runs Boltz-2 and is gated on the result.
Boltz-2 is non-deterministic (GPU seeding, MSA sampling), so
the comparison uses tolerance bands rather than exact match.

Acceptance criteria (vs smoke_test/boltz2_baseline.json):
- error is None
- output structure file is non-empty (predicted structure exists)
- iptm >= baseline.iptm - 0.05            (5% tolerance)
- ptm >= baseline.ptm - 0.05               (5% tolerance)
- plddt_mean >= baseline.plddt_mean - 2.0 (absolute tolerance)
- clash_count <= baseline.clash_count + 5  (absolute tolerance)
- clash_severe_count == 0                   (hard fail — any severe clash)

Usage:
    compare.py <boltz2_smoke.json> [--baseline PATH]

Exit codes:
    0   pass
    1   gate failure (output diverged from baseline within tolerance)
    2   file not found or invalid JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Tolerance bands (boltz-2 is non-deterministic)
IPTM_TOLERANCE = 0.05
PTM_TOLERANCE = 0.05
PLDDT_TOLERANCE = 2.0
CLASH_TOLERANCE = 5
# clash_severe_count: 0 is the only acceptable value (severe clash
# means a physically impossible structure)


def load_baseline(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: baseline not found at {path}", file=sys.stderr)
        print(f"Run 'make boltz2-baseline' once on a known-good GPU to generate it.", file=sys.stderr)
        sys.exit(2)
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: baseline at {path} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)


def load_actual(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: boltz-2 smoke output not found at {path}", file=sys.stderr)
        sys.exit(2)
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: {path} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)


def compare(actual: dict, baseline: dict) -> tuple[bool, str]:
    """Return (passed, reason). Boltz-2 is non-deterministic, so use tolerance bands."""
    if actual.get("error") is not None:
        return False, f"❌ boltz-2 reported error: {actual['error']!r}"

    structure_size = actual.get("structure_size_bytes", 0)
    if structure_size == 0:
        return False, "❌ no predicted structure file produced (size_bytes=0)"

    actual_conf = actual.get("confidence") or {}
    baseline_conf = baseline.get("confidence") or {}

    # Check hard fail: severe clashes
    actual_severe = actual_conf.get("clash_severe_count", 0) or 0
    baseline_severe = baseline_conf.get("clash_severe_count", 0) or 0
    if actual_severe > 0:
        return False, (
            f"❌ SEVERE clashes detected: {actual_severe} atom pairs with sub-2.0A overlap — "
            f"structure physically impossible"
        )

    # iptm: 5% tolerance
    actual_iptm = actual_conf.get("iptm")
    baseline_iptm = baseline_conf.get("iptm")
    if actual_iptm is not None and baseline_iptm is not None:
        threshold = baseline_iptm - IPTM_TOLERANCE
        if actual_iptm < threshold:
            return False, (
                f"❌ iptm regression: actual={actual_iptm:.4f} baseline={baseline_iptm:.4f} "
                f"threshold={threshold:.4f} (>5% drop) — confidence quality drop"
            )

    # ptm: 5% tolerance
    actual_ptm = actual_conf.get("ptm")
    baseline_ptm = baseline_conf.get("ptm")
    if actual_ptm is not None and baseline_ptm is not None:
        threshold = baseline_ptm - PTM_TOLERANCE
        if actual_ptm < threshold:
            return False, (
                f"❌ ptm regression: actual={actual_ptm:.4f} baseline={baseline_ptm:.4f} "
                f"threshold={threshold:.4f} (>5% drop)"
            )

    # plddt_mean: absolute tolerance
    actual_plddt = actual_conf.get("plddt_mean")
    baseline_plddt = baseline_conf.get("plddt_mean")
    if actual_plddt is not None and baseline_plddt is not None:
        threshold = baseline_plddt - PLDDT_TOLERANCE
        if actual_plddt < threshold:
            return False, (
                f"❌ plddt_mean regression: actual={actual_plddt:.2f} baseline={baseline_plddt:.2f} "
                f"threshold={threshold:.2f} (>{PLDDT_TOLERANCE} drop) — per-residue confidence drop"
            )

    # clash_count: must not regress by more than CLASH_TOLERANCE
    actual_clash = actual_conf.get("clash_count", 0) or 0
    baseline_clash = baseline_conf.get("clash_count", 0) or 0
    if actual_clash > baseline_clash + CLASH_TOLERANCE:
        return False, (
            f"❌ clash_count regression: actual={actual_clash} baseline={baseline_clash} "
            f"threshold={baseline_clash + CLASH_TOLERANCE} (>+{CLASH_TOLERANCE})"
        )

    # Passed
    return True, (
        f"✅ PASS: iptm={actual_iptm:.4f} (baseline {baseline_iptm:.4f}), "
        f"plddt={actual_plddt:.1f} (baseline {baseline_plddt:.1f}), "
        f"clashes={actual_clash} (baseline {baseline_clash})"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("boltz2_smoke", type=Path,
                   help="Path to boltz2_smoke.json from the smoke run")
    p.add_argument("--baseline", type=Path,
                   default=Path(__file__).parent.parent / "smoke_test" / "boltz2_baseline.json",
                   help="Path to boltz2_baseline.json (default: smoke_test/boltz2_baseline.json)")
    args = p.parse_args()

    baseline = load_baseline(args.baseline)
    actual = load_actual(args.boltz2_smoke)

    b_conf = baseline.get("confidence") or {}
    a_conf = actual.get("confidence") or {}

    def fmt_iptm(c): v = c.get("iptm"); return f"{v:.4f}" if v is not None else "N/A"
    def fmt_ptm(c): v = c.get("ptm"); return f"{v:.4f}" if v is not None else "N/A"
    def fmt_plddt(c): v = c.get("plddt_mean"); return f"{v:.2f}" if v is not None else "N/A"

    print(f"baseline: iptm={fmt_iptm(b_conf)}, "
          f"ptm={fmt_ptm(b_conf)}, "
          f"plddt={fmt_plddt(b_conf)}, "
          f"clashes={b_conf.get('clash_count', 0)}")
    print(f"actual:   iptm={fmt_iptm(a_conf)}, "
          f"ptm={fmt_ptm(a_conf)}, "
          f"plddt={fmt_plddt(a_conf)}, "
          f"clashes={a_conf.get('clash_count', 0)}")
    print()

    passed, reason = compare(actual, baseline)
    print(reason)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
