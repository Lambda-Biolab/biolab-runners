"""CLI + availability helpers for the GROMACS runner."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "GromacsRecord",
    "GromacsRecordStatus",
    "gromacs_available",
    "parse_nthcol_energy",
]


class GromacsRecordStatus:
    """Normalized outcome values for GROMACS run records."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class GromacsRecord:
    """One GROMACS simulation output."""

    index: int
    path: str
    potential_energy: float
    status: str = GromacsRecordStatus.SUCCEEDED
    error: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialize the record into a JSON-safe dictionary."""
        return {
            "index": str(self.index),
            "path": self.path,
            "potential_energy": repr(self.potential_energy),
            "status": self.status,
            "error": self.error,
        }


_FLOAT_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def gromacs_available(timeout_seconds: int = 30) -> bool:
    """Return True when the GROMACS CLI is callable."""
    import os

    binary = os.environ.get("GROMACS_BIN", "gmx")
    if binary.startswith("container://"):
        return True
    if shutil.which(binary) is None:
        return False
    try:
        completed = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _resolved_binary() -> list[str]:
    """Return the command prefix used to invoke ``gmx``."""
    import os

    binary = os.environ.get("GROMACS_BIN", "gmx")
    if binary.startswith("container://"):
        spec = binary[len("container://") :]
        runtime = os.environ.get("CONTAINER_RUNTIME", "docker")
        return [runtime, "run", "--rm", spec, "gmx"]
    return [binary]


def parse_nthcol_energy(path: Path, column: int = 1) -> float:
    """Parse the ``column``-th whitespace-separated float from each line.

    Used by the GROMACS runner to read ``energy.xvg``-style files. The
    first line that contains a parseable float in the requested column
    is returned; 0.0 indicates empty or malformed output.
    """
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("@"):
            continue
        tokens = stripped.split()
        if len(tokens) <= column:
            continue
        if not _FLOAT_RE.match(tokens[column]):
            continue
        try:
            return float(tokens[column])
        except ValueError:
            continue
    return 0.0


def invoke(
    *,
    config_dict: dict[str, str],
    output_dir: Path,
    mdrun_extra: tuple[str, ...],
    binary_prefix: list[str] | None = None,
    timeout_seconds: int = 86400,
) -> int:
    """Run ``gmx mdrun`` once; returns the process exit code."""
    prefix = binary_prefix if binary_prefix is not None else _resolved_binary()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = [
        *prefix,
        "mdrun",
        "-deffnm",
        config_dict["-deffnm"],
        "-s",
        config_dict["-s"],
        "-nsteps",
        config_dict["-nsteps"],
        "-o",
        f"{config_dict['-deffnm']}.trr",
        "-e",
        f"{config_dict['-deffnm']}.edr",
        "-g",
        f"{config_dict['-deffnm']}.log",
        "-cpo",
        f"{config_dict['-deffnm']}.cpt",
        *mdrun_extra,
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            args,
            cwd=str(output_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error("GROMACS timed out after %ds", timeout_seconds)
        return 124
    logger.info(
        "GROMACS run finished rc=%d in %.1fs",
        completed.returncode,
        time.monotonic() - started,
    )
    return completed.returncode
