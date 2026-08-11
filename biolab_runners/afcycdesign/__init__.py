"""Optional AfCycDesign runner exposed to the activator pipeline.

Slice 11 (BLR-AFCYC-001) shipped this as a **time-boxed feasibility
spike** behind the ``AFCYCDESIGN_SPIKE`` env var. The runner returns
a fail-closed FAILED result when the spike is not enabled; the
activator's geometry / relaxation check then becomes the binding-mode
sanity gate.

To re-run the spike:

1. Push a pinned ColabDesign + AlphaFold2 weights image to
   ``ghcr.io/lambda-biolab/afcycdesign:0.1.0`` (see the Slice 11
   runbook).
2. Set ``AFCYCDESIGN_SPIKE=1`` in the step env.
3. Run with a 5–8 residue head-to-tail sequence as the input.
4. Compare the resulting mean pLDDT against the predicted secondary
   structure from Boltz-2 or OpenMM relaxation.
"""

from __future__ import annotations

from biolab_runners.afcycdesign.spike import (
    AfCycDesignResult,
    AfCycDesignRunner,
    AfCycDesignStatus,
)

__all__ = [
    "AfCycDesignResult",
    "AfCycDesignRunner",
    "AfCycDesignStatus",
]
