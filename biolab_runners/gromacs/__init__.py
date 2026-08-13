"""GROMACS runner.

Two runners live here:

- :class:`GromacsRunner` — thin subprocess wrapper around a single
  ``gmx mdrun`` invocation (preserved from S3; used by callers
  that already have a pre-built ``.tpr``).

- :class:`GromacsProtocolRunner` — production-grade protocol
  runner (added in S4). Drives the full pipeline (topology →
  solvation → ions → minimization → NVT → NPT → production) with
  per-stage checkpoint resume.

The GROMACS CLI is published as a binary + MPI installation
rather than a pip package. Both runners expect the operator to
provide the executable via the ``GROMACS_BIN`` env var (or
accept the default ``gmx`` on PATH); the runner also honours a
``binary_prefix`` argument for containerised installs
(``["docker", "run", "--rm", "gmx-image", "gmx"]``).

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

from biolab_runners.gromacs.config import GromacsConfig, GromacsProtocolConfig
from biolab_runners.gromacs.protocol import GENION_INPUT
from biolab_runners.gromacs.runner import (
    GromacsProtocolResult,
    GromacsProtocolRunner,
    GromacsResult,
    GromacsRunner,
)
from biolab_runners.gromacs.utils import (
    GromacsRecord,
    GromacsRecordStatus,
    StageStatus,
    gromacs_available,
    parse_nthcol_energy,
)

__all__ = [
    "GENION_INPUT",
    "GromacsConfig",
    "GromacsProtocolConfig",
    "GromacsProtocolResult",
    "GromacsProtocolRunner",
    "GromacsRecord",
    "GromacsRecordStatus",
    "GromacsResult",
    "GromacsRunner",
    "StageStatus",
    "gromacs_available",
    "parse_nthcol_energy",
]
