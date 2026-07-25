"""Utility functions for OpenMM MD simulations."""

from __future__ import annotations

import json
import logging
import subprocess
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

from biolab_runners.openmm.paths import FileNames

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
    """Verify that production MD outputs are complete.

    Checks for expected files and validates basic integrity (file sizes,
    energy row counts).

    Args:
        output_dir: Directory containing MD outputs.

    Returns:
        Verification report dict with "complete" boolean and file details.
    """
    report: dict[str, object] = {
        "output_dir": str(output_dir),
        "complete": True,
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
            if lines < 10:
                file_info["warning"] = "Very few energy rows — run may be incomplete"
                report["complete"] = False

        if filename == FileNames.TRAJECTORY and exists and size < 10_000_000:
            file_info["warning"] = "Trajectory < 10 MB — likely incomplete"
            report["complete"] = False

        if not exists:
            report["complete"] = False

        report["files"][filename] = file_info  # type: ignore[index]

    return report


def load_checkpoint_step(output_dir: Path) -> int:
    """Load the last checkpoint step from the atomic-save manifest.

    The manifest (``checkpoint.json``) is the ONLY authoritative
    source for the saved state's step. The previous implementation
    fell back to the last row of ``energy.csv`` when the manifest
    was missing, but that was unsafe: ``energy.csv`` is written every
    ``save_every_steps`` (~10 ps) by the reporter, while
    ``state.xml`` is saved every ``checkpoint_every_steps`` (~2 hr)
    via :func:`biolab_runners.openmm.system_builder._atomic_save_checkpoint`.
    The two cadences differ by orders of magnitude — after a crash,
    the energy row can be hundreds of thousands of steps ahead of
    the saved state, and resuming from the energy step while
    loading the older state would silently shorten the run.

    The runner's resume flow (see ``runner._resolve_skip_or_resume``)
    treats a return value of 0 as "no resume" and fails fast if
    ``state.xml`` exists without a matching manifest — see the
    "Atomic checkpoint" rule in AGENTS.md.

    Args:
        output_dir: MD output directory.

    Returns:
        Step number recorded in the manifest, or 0 if the manifest
        is missing or invalid (the runner treats 0 as "no resumable
        checkpoint").
    """
    ckpt_json = output_dir / FileNames.CHECKPOINT_JSON
    if not ckpt_json.exists():
        return 0
    try:
        data = json.loads(ckpt_json.read_text())
        records = data.get("records", [])
        if records:
            return int(records[-1].get("step", 0))
    except (json.JSONDecodeError, KeyError, IndexError, OSError):
        pass
    return 0
