"""Run a tiny Boltz-2 prediction as the per-PR boltz-2 gate.

The gate uses a fixed small input (receptor: 8 residues, peptide: 7
residues) to keep the runtime short (~2-4 min on a typical GPU).
The output is a JSON confidence summary that's compared against a
committed baseline in the per-pr-boltz2-gate workflow.

This is the deepvariant-arm64-Linux pattern for structure
prediction: every PR runs the actual scientific code (Boltz-2
structure prediction) and is gated on the result.

The test input is intentionally tiny to keep per-PR cost near zero
on the self-hosted GPU runner. The receiver + peptide fit in
~15 residues total, which Boltz-2 can predict in seconds on an
RTX 4090. Even on the cheapest Vast.ai RTX 3060 the runtime is
under 5 minutes.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

# These inputs are hard-coded. The PR gate is comparing the
# confidence scores for THIS SPECIFIC INPUT against a baseline
# generated on a known-good GPU. Changing these inputs invalidates
# the baseline.
RECEPTOR_SEQUENCE = "MVKLTAEG"  # 8 residues — small ubiquitin-like fragment
PEPTIDE_SEQUENCE = "RWKLFKK"    # 7 residues — cathelicidin fragment
PREDICTION_NAME = "per_pr_gate_smoke"
ACCELERATOR = "gpu"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: run_boltz2_smoke.py <output_dir>", file=sys.stderr)
        return 2

    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import so this script can be syntax-checked without boltz installed
    from biolab_runners.boltz2 import Boltz2Config, Boltz2Runner, QualityGate
    from biolab_runners.boltz2.config import ConfidenceScores

    # Clean previous output for idempotency
    for child in output_dir.glob("*"):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    print(f"[boltz2-smoke] Running prediction: {PREDICTION_NAME}")
    print(f"[boltz2-smoke]   receptor: {RECEPTOR_SEQUENCE} ({len(RECEPTOR_SEQUENCE)} residues)")
    print(f"[boltz2-smoke]   peptide:  {PEPTIDE_SEQUENCE} ({len(PEPTIDE_SEQUENCE)} residues)")

    runner = Boltz2Runner(Boltz2Config(accelerator=ACCELERATOR))
    result = runner.predict_complex(
        receptor_sequence=RECEPTOR_SEQUENCE,
        peptide_sequence=PEPTIDE_SEQUENCE,
        name=PREDICTION_NAME,
        output_dir=output_dir,
    )

    print(f"[boltz2-smoke] quality_gate: {result.quality_gate}")
    print(f"[boltz2-smoke] error: {result.error!r}")

    # Convert to JSON-serializable summary
    confidence_dict = None
    if result.confidence is not None:
        c = result.confidence
        confidence_dict = {
            "iptm": float(c.iptm) if c.iptm is not None else None,
            "ptm": float(c.ptm) if c.ptm is not None else None,
            "plddt_mean": float(c.plddt_mean) if c.plddt_mean is not None else None,
            "clash_count": int(c.clash_count) if c.clash_count is not None else None,
            "clash_severe_count": int(c.clash_severe_count) if c.clash_severe_count is not None else None,
        }

    summary = {
        "name": result.name,
        "quality_gate": str(result.quality_gate),
        "error": result.error,
        "structure_path": str(result.structure_path) if result.structure_path else None,
        "structure_size_bytes": (
            os.path.getsize(result.structure_path) if result.structure_path and Path(result.structure_path).exists() else 0
        ),
        "confidence": confidence_dict,
        "gate_reasons": list(result.gate_reasons) if result.gate_reasons else [],
    }

    out_path = output_dir / "boltz2_smoke.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[boltz2-smoke] Wrote summary to {out_path}")
    return 0 if result.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
