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
    energy row counts). The "state presence" check is now against
    the atomic-save manifest (``checkpoint.json``) — the v7 save
    format uses generation-versioned state files
    (``state.<gen>.xml``) referenced by the manifest, not a canonical
    ``state.xml``.

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

    # The manifest always exists if the run is complete — it is the
    # last thing the atomic save commits. Insert its info into the
    # report for completeness.
    manifest_path = output_dir / FileNames.CHECKPOINT_JSON
    if manifest_path.exists():
        report["files"][FileNames.CHECKPOINT_JSON] = {  # type: ignore[index]
            "exists": True,
            "size_bytes": manifest_path.stat().st_size,
        }

    return report


def load_checkpoint(output_dir: Path) -> tuple[int, str]:
    """Load the saved step AND state file from the atomic-save manifest.

    The manifest (``checkpoint.json``) is the ONLY authoritative
    source for the saved state's step and the file to load. It
    references a state file by basename (e.g.
    ``state.700000_12345_1700000000000000000.xml``). The resume
    flow reads both the step and the file from the manifest, then
    loads the state file via ``simulation.loadState``.

    The previous design saved only ``state.xml`` and inferred the
    saved step from the last row of ``energy.csv`` — unsafe because
    the two cadences differ by orders of magnitude (energy.csv at
    ~10 ps, state.xml at ~2 hr). The v7 fix commits the state file
    and the manifest as a single atomic transaction: the manifest
    rename is the only atomic commit point; the state file's
    filename uniquely identifies it, so any state file not
    referenced by the manifest is an orphan that can be safely
    garbage-collected.

    Args:
        output_dir: MD output directory.

    Returns:
        ``(absolute_step, state_file_basename)`` on success — both
        non-empty. ``(0, "")`` if no manifest exists or the manifest
        is invalid. The runner treats (0, "") as "no resumable
        checkpoint" and fails fast on any orphaned state file.
    """
    manifest_path = output_dir / FileNames.CHECKPOINT_JSON
    if not manifest_path.exists():
        return 0, ""
    try:
        data = json.loads(manifest_path.read_text())
        records = data.get("records", [])
        if not records:
            return 0, ""
        last = records[-1]
        step = int(last.get("step", 0))
        file = str(last.get("file", ""))
        if step > 0 and file:
            return step, file
    except (json.JSONDecodeError, KeyError, IndexError, OSError, ValueError):
        pass
    return 0, ""


def load_checkpoint_step(output_dir: Path) -> int:
    """Backwards-compatible step-only wrapper around :func:`load_checkpoint`.

    Returns the absolute step from the manifest, or 0 if no manifest
    exists. Kept for callers that only need the step (e.g. logs);
    new code should call :func:`load_checkpoint` directly to also
    get the file the manifest references.
    """
    step, _ = load_checkpoint(output_dir)
    return step
