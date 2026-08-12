"""biolab_runners.mmpbsa — optional gmx_MMPBSA integration (slice 14).

Public surface:
- :class:`GmxMMPBSARunner` (re-exported from ``runner``)
- :class:`GmxMMPBSARecord` (re-exported from ``_parser``)
- :class:`GmxMMPBSAStatus` (re-exported from ``runner`` — defined
  there, adjacent to the only emitter, so the constants cannot
  drift from the code that emits them)
- :func:`gmx_mmpbsa_available` (re-exported from ``utils``)
- :func:`parse_residue_decomposition` (re-exported from ``_parser``)

The runner gracefully degrades to ``status="unsupported"`` when
gmx_MMPBSA is not on PATH (slice 14 acceptance: missing optional
tooling yields ``unsupported``, not a fabricated value).
"""

from __future__ import annotations

from biolab_runners.mmpbsa._parser import GmxMMPBSARecord, parse_residue_decomposition
from biolab_runners.mmpbsa.runner import GmxMMPBSARunner, GmxMMPBSAStatus
from biolab_runners.mmpbsa.utils import gmx_mmpbsa_available

__all__ = [
    "GmxMMPBSARecord",
    "GmxMMPBSARunner",
    "GmxMMPBSAStatus",
    "gmx_mmpbsa_available",
    "parse_residue_decomposition",
]
