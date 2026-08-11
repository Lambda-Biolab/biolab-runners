"""GROMACS runner.

Thin subprocess wrapper around the upstream ``gmx`` CLI. The runner
mirrors the boltz2 shallow pattern: a config dataclass, a runner
class with submit / dry_run / idempotency, and a utils module
with availability probes and trajectory parsing.

GROMACS is published as a binary + MPI installation rather than a
pip package. The runner expects the operator to provide the
executable via the ``GROMACS_BIN`` env var (or accept the default
``gmx`` on PATH).

Slice 13 (MD-GMX-001) introduces a parallel path to OpenMM with
parity benchmarking in the Slice 15 stage. The runner supports
temperature / pressure / integrator / nsteps options; the actual
force-field and topology files are configured by the operator.
"""

from __future__ import annotations

from biolab_runners.gromacs.config import GromacsConfig
from biolab_runners.gromacs.runner import GromacsRunner
from biolab_runners.gromacs.utils import (
    GromacsRecord,
    GromacsRecordStatus,
    gromacs_available,
    parse_nthcol_energy,
)

__all__ = [
    "GromacsConfig",
    "GromacsRecord",
    "GromacsRecordStatus",
    "GromacsRunner",
    "gromacs_available",
    "parse_nthcol_energy",
]
