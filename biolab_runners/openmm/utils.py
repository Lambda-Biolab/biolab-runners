"""Utility functions for OpenMM MD simulations.

After the v14 checkpoint extraction this module only carries the
non-checkpoint helpers: dependency availability checks
(:func:`openmm_available`, :func:`pdbfixer_available`) and a
diagnostic reporter (:func:`verify_production_outputs`).

The checkpoint / manifest / state-file domain — atomic save,
quarantine, manifest parsing, terminal classification, production
step math — lives in :mod:`biolab_runners.openmm.checkpoint`.
:class:`InvalidCheckpointError` is re-exported here so existing
callers that imported the exception from this module keep
working; new code should import it from
:mod:`biolab_runners.openmm.checkpoint` directly.
"""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

from biolab_runners.openmm.checkpoint import InvalidCheckpointError  # noqa: F401
from biolab_runners.openmm.paths import FileNames

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def openmm_available(platform: str = "OpenCL") -> bool:
    """Check if OpenMM is installed and the requested platform is available.

    Args:
        platform: OpenMM platform to check for (e.g. "OpenCL", "CUDA", "CPU").

    Returns:
        True if OpenMM is importable and the platform is available.
    """
    try:
        result = subprocess.run(
            [
                "python",
                "-c",
                (
                    "import openmm; "
                    f"p = openmm.Platform.getPlatformByName('{platform}'); "
                    "print(p.getName())"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.returncode == 0 and platform in result.stdout
    except FileNotFoundError:
        return False


def pdbfixer_available() -> bool:
    """Check if PDBFixer is installed.

    Returns:
        True if pdbfixer can be imported.
    """
    try:
        result = subprocess.run(
            ["python", "-c", "from pdbfixer import PDBFixer; print('ok')"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def verify_production_outputs(output_dir: Path) -> dict[str, object]:
    """Build a diagnostic report on the production output directory.

    This is a pure diagnostic helper — it reports per-file size and
    row counts but does NOT decide whether a run is complete.
    Completion is determined by :func:`is_run_complete` (which uses
    the manifest's absolute step + terminal payload), not by
    ``"files exist + size > threshold"``. The earlier "complete"
    field was removed because a mid-production checkpoint can
    produce a large trajectory and many energy rows while the run
    is still in progress — file presence does not imply completion.

    Args:
        output_dir: Directory containing MD outputs.

    Returns:
        Diagnostic report dict with file metadata (existence, size,
        row count for the energy log). The "complete" field is
        always False; callers that want a completion verdict must
        call :func:`is_run_complete` instead.
    """
    report: dict[str, object] = {
        "output_dir": str(output_dir),
        "complete": False,
        "files": {},
    }

    for filename in FileNames.PRODUCTION_OUTPUT_FILES:
        path = output_dir / filename
        exists = path.exists()
        size = path.stat().st_size if exists else 0

        file_info: dict[str, object] = {
            "exists": exists,
            "size_bytes": size,
        }

        if filename == FileNames.ENERGY and exists:
            lines = len(path.read_text().strip().splitlines())
            file_info["rows"] = lines

        if filename == FileNames.TRAJECTORY and exists:
            file_info["note"] = (
                "Trajectory size alone does not imply completion — "
                "see is_run_complete() for the authoritative verdict."
            )

        report["files"][filename] = file_info  # type: ignore[index]

    # Insert manifest info for completeness.
    manifest_path = output_dir / FileNames.CHECKPOINT_JSON
    if manifest_path.exists():
        report["files"][FileNames.CHECKPOINT_JSON] = {  # type: ignore[index]
            "exists": True,
            "size_bytes": manifest_path.stat().st_size,
        }

    return report
