"""CLI + availability helpers for the gmx_MMPBSA runner.

The runner imports from this module. Public surface (mirrors
``biolab_runners.gromacs.utils``):
- :func:`gmx_mmpbsa_available` — PATH probe.
- :func:`parse_residue_decomposition` — file parser for
  ``{prefix}_residue_decomposition_*.dat``.

The actual ``runner`` is exposed via :mod:`biolab_runners.mmpbsa`
itself (``biolab_runners.mmpbsa.GmxMMPBSARunner.run``).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from biolab_runners.mmpbsa._parser import GmxMMPBSARecord
from biolab_runners.mmpbsa._parser import parse_residue_decomposition as _parse_records

__all__ = [
    "GmxMMPBSARecord",
    "gmx_mmpbsa_available",
    "parse_residue_decomposition",
]


def gmx_mmpbsa_available(binary: str = "gmx_MMPBSA", *, timeout_seconds: int = 30) -> bool:
    """Return True when the ``gmx_MMPBSA`` CLI is callable."""
    if binary.startswith("container://"):
        return True
    if shutil.which(binary) is None:
        return False
    try:
        completed = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


def parse_residue_decomposition(path: Path) -> tuple[GmxMMPBSARecord, ...]:
    """Public re-export of the per-residue decomposition parser."""
    return _parse_records(path)


logger = logging.getLogger(__name__)
