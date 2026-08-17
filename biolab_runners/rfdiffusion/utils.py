"""CLI + availability helpers for the RFdiffusion runner.

RFdiffusion is invoked through the in-package ``rfdiffusion`` console
script (``biolab_runners.rfdiffusion.cli``), which translates the
runner's ``--output_dir`` + dotted/key-value flag contract into Hydra
positional overrides for the stock ``scripts/run_inference.py`` under
``RFDIFFUSION_HOME``. The runner resolves the executable through
:func:`rfdiffusion_available`, which honours the ``RFDIFFUSION_BIN``
env var (a custom binary implementing the same contract) and falls
back to the installed ``rfdiffusion`` script on the system PATH.
"""

from __future__ import annotations

import functools
import logging
import operator
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from biolab_runners.provenance import InvokeResult, stderr_tail

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "RecordData",
    "RecordDataStatus",
    "invoke",
    "parse_backbone_pdb",
    "rfdiffusion_available",
]


class RecordDataStatus:
    """Normalized outcome values for per-design records."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class RecordData:
    """One per-design record produced by RFdiffusion."""

    index: int
    path: str
    sequence: str
    status: str = RecordDataStatus.SUCCEEDED
    error: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialize the record into a JSON-safe dictionary."""
        return {
            "index": str(self.index),
            "path": self.path,
            "sequence": self.sequence,
            "status": self.status,
            "error": self.error,
        }


def rfdiffusion_available(timeout_seconds: int = 30) -> bool:
    """Return True when the RFdiffusion binary can be invoked.

    Honours ``RFDIFFUSION_BIN``; falls back to the in-package
    ``rfdiffusion`` console script on the system PATH. The probe runs
    ``--help``, which the console script answers without touching
    ``RFDIFFUSION_HOME`` or any model files. ``container://`` URIs are
    no longer supported and report unavailable here (see
    :func:`_resolved_binary`).
    """
    import os

    binary = os.environ.get("RFDIFFUSION_BIN", "rfdiffusion")
    # ``which`` is cheap and avoids spawning the real binary just to
    # probe availability.
    if shutil.which(binary) is None:
        return False
    try:
        completed = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _resolved_binary() -> list[str]:
    """Return the command prefix used to invoke RFdiffusion.

    ``RFDIFFUSION_BIN`` may point at a custom binary implementing the
    runner contract; the default is the installed ``rfdiffusion``
    console script. The legacy ``container://`` URI form is rejected:
    it invoked ``run_inference.py --key value`` directly (Hydra needs
    positional ``key=value`` overrides) and hardcoded an image-internal
    path, so it could never work as written — the in-package console
    script is the supported way to reach upstream inside a container.
    """
    import os

    binary = os.environ.get("RFDIFFUSION_BIN", "rfdiffusion")
    if binary.startswith("container://"):
        raise ValueError(
            "RFDIFFUSION_BIN=container://... is no longer supported: the "
            "in-package `rfdiffusion` console script adapts to stock "
            "RFdiffusion via RFDIFFUSION_HOME. Unset RFDIFFUSION_BIN (or "
            "point it at a custom binary implementing the --output_dir + "
            "--<dotted.key> <value> contract)."
        )
    return [binary]


_PDB_LINE_RE = re.compile(r"^(ATOM|HETATM)\s+")


def parse_backbone_pdb(path: Path) -> str:
    """Return the poly-glycine backbone sequence encoded in ``path``.

    RFdiffusion emits poly-Glycine backbones; we still read the
    residue column so future non-Gly backbones are handled without
    the runner crashing.
    """
    residues: list[str] = []
    seen_chain_residue: set[tuple[str, int]] = set()
    for line in path.read_text().splitlines():
        if not _PDB_LINE_RE.match(line):
            continue
        chain = line[21:22].strip()
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue  # type: ignore[arg-type]
        resname = line[17:20].strip()
        key = (chain, resseq)
        if key in seen_chain_residue:
            continue
        seen_chain_residue.add(key)
        # Map 3-letter residue name to 1-letter. Unknown -> X.
        one_letter = _THREE_TO_ONE.get(resname, "X")
        residues.append(one_letter)
    return "".join(residues)


_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _invoke_with_metadata(
    *,
    config_dict: dict[str, str],
    output_dir: Path,
    binary_prefix: list[str] | None = None,
    timeout_seconds: int = 3600,
) -> InvokeResult:
    """Internal helper: run RFdiffusion once and capture rich metadata.

    Returns an :class:`InvokeResult` carrying the exit code, a
    512-char stderr tail, the timeout flag, and a short failure
    reason. Public callers use the legacy :func:`invoke` wrapper
    (which discards everything except the exit code); the S2
    provenance wiring uses this helper directly.
    """
    prefix = binary_prefix if binary_prefix is not None else _resolved_binary()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = [
        *prefix,
        "--output_dir",
        str(output_dir),
        *functools.reduce(
            operator.iadd,
            ([f"--{key.replace('_', '-')}", str(value)] for key, value in config_dict.items()),
            [],
        ),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("RFdiffusion timed out after %ds", timeout_seconds)
        return InvokeResult(
            exit_code=124,
            stderr_tail=stderr_tail(exc.stderr),
            timed_out=True,
            failure_reason=f"timeout after {timeout_seconds}s",
        )
    elapsed = time.monotonic() - started
    logger.info("RFdiffusion run finished rc=%d in %.1fs", completed.returncode, elapsed)
    return InvokeResult.from_stderr(exit_code=completed.returncode, stderr=completed.stderr)


def invoke(
    *,
    config_dict: dict[str, str],
    output_dir: Path,
    binary_prefix: list[str] | None = None,
    timeout_seconds: int = 3600,
) -> int:
    """Run RFdiffusion once; returns the process exit code.

    Callers should use :class:`RFdiffusionRunner` instead of invoking
    this directly; it is exposed for tests that want to stub the
    process execution. The legacy ``int`` return type is preserved
    for backward compatibility — new code that needs stderr /
    timeout metadata should call :func:`_invoke_with_metadata`
    instead (it is the implementation that backs this function).
    """
    return _invoke_with_metadata(
        config_dict=config_dict,
        output_dir=output_dir,
        binary_prefix=binary_prefix,
        timeout_seconds=timeout_seconds,
    ).exit_code
