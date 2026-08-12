"""GmxMMPBSARunner (slice 14 / BMT-MD-001 optional gmx_MMPBSA integration).

Located at the biolab_runners.mmpbsa package root for symmetry with
``biolab_runners.gromacs``. The runner class delegates parser
work to ``biolab_runners.mmpbsa._parser``.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from biolab_runners.mmpbsa._parser import (
    GmxMMPBSARecord,
    parse_residue_decomposition,
)
from biolab_runners.openmm.config import OpenMMConfig  # noqa: TC001 — runtime use as field type

logger = logging.getLogger(__name__)

__all__ = [
    "GmxMMPBSARunner",
    "GmxMMPBSAStatus",
]


class GmxMMPBSAStatus:
    """Normalized outcome values for the gmx_MMPBSA runner."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class GmxMMPBSARunner:
    """Wrapper around the gmx_MMPBSA CLI for per-residue decomposition."""

    config: OpenMMConfig
    prefix: str
    mmpbsa_binary: str = "gmx_MMPBSA"
    timeout_seconds: int = 1800

    def _build_command(self) -> list[str]:
        """Build the gmx_MMPBSA CLI invocation.

        The exact flag set varies by gmx_MMPBSA version; the
        canonical ``complex / protein / ligand`` triple plus
        ``--use-md-decomposition residue`` is the minimum needed
        for per-residue decomposition.
        """
        return [
            self.mmpbsa_binary,
            "--use-md-decomposition",
            "residue",
            "--complex-type",
            "protein-ligand",
            "--complex-md",
            str(self.config.receptor_pdb),
            "--ligand-md",
            str(self.config.peptide_pdb),
            "--prefix",
            str(self.prefix),
        ]

    def _load_records(self) -> tuple[GmxMMPBSARecord, ...]:
        """Find and parse the per-residue decomposition file.

        Different gmx_MMPBSA versions emit different filenames; we
        accept the common variants. If no matching file is found,
        returns an empty tuple — the caller treats this as
        ``unsupported``.
        """
        candidate_names = (
            "residue_decomposition_finite.dat",
            "residue_decomposition.dat",
            "FINAL_RESULTS_MMPBSA.dat",
        )
        for name in candidate_names:
            path = Path(self.config.output_dir) / f"{self.prefix}_{name}"
            if path.exists():
                return parse_residue_decomposition(path)
        return ()

    def run(self) -> dict[str, object]:
        """Execute gmx_MMPBSA for per-residue decomposition.

        Returns a JSON-stable dict with ``status``, ``records``,
        ``error``. When the binary isn't on PATH (or the --version
        probe fails), returns ``status="unsupported"`` with empty
        records — slice 14 acceptance criterion: missing optional
        tooling yields ``unsupported``, not a fabricated value.
        """
        from biolab_runners.mmpbsa.utils import gmx_mmpbsa_available

        if not gmx_mmpbsa_available(binary=self.mmpbsa_binary):
            return {
                "status": GmxMMPBSAStatus.UNSUPPORTED,
                "binary": self.mmpbsa_binary,
                "prefix": self.prefix,
                "per_residue_records": [],
                "error": f"{self.mmpbsa_binary} not on PATH",
            }
        cmd = self._build_command()
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
                cwd=self.config.output_dir,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": GmxMMPBSAStatus.FAILED,
                "binary": self.mmpbsa_binary,
                "prefix": self.prefix,
                "per_residue_records": [],
                "error": f"gmx_MMPBSA timed out after {self.timeout_seconds}s",
            }
        if completed.returncode != 0:
            return {
                "status": GmxMMPBSAStatus.FAILED,
                "binary": self.mmpbsa_binary,
                "prefix": self.prefix,
                "per_residue_records": [],
                "error": (completed.stderr or completed.stdout).strip()[-2000:],
            }
        records = self._load_records()
        return {
            "status": GmxMMPBSAStatus.SUCCEEDED,
            "binary": self.mmpbsa_binary,
            "prefix": self.prefix,
            "per_residue_records": [r.to_dict() for r in records],
        }
