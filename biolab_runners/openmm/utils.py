"""Utility functions for OpenMM MD simulations."""

from __future__ import annotations

import json
import logging
import subprocess
import typing

if typing.TYPE_CHECKING:
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
    """Verify that production MD outputs are complete.

    Checks for expected files and validates basic integrity (file sizes,
    energy row counts).

    Args:
        output_dir: Directory containing MD outputs.

    Returns:
        Verification report dict with "complete" boolean and file details.
    """
    expected = [
        "trajectory.dcd",
        "energy.csv",
        "state.xml",
    ]

    report: dict[str, object] = {
        "output_dir": str(output_dir),
        "complete": True,
        "files": {},
    }

    for filename in expected:
        path = output_dir / filename
        exists = path.exists()
        size = path.stat().st_size if exists else 0

        file_info: dict[str, object] = {
            "exists": exists,
            "size_bytes": size,
        }

        if filename == "energy.csv" and exists:
            lines = len(path.read_text().strip().splitlines())
            file_info["rows"] = lines
            if lines < 10:
                file_info["warning"] = "Very few energy rows — run may be incomplete"
                report["complete"] = False

        if filename == "trajectory.dcd" and exists and size < 10_000_000:
            file_info["warning"] = "Trajectory < 10 MB — likely incomplete"
            report["complete"] = False

        if not exists:
            report["complete"] = False

        report["files"][filename] = file_info  # type: ignore[index]

    return report


def load_checkpoint_step(output_dir: Path) -> int:
    """Load the last checkpoint step from energy.csv or checkpoint.json.

    Args:
        output_dir: MD output directory.

    Returns:
        Step number of the last checkpoint, or 0 if no checkpoint exists.
    """
    # Try checkpoint.json first
    ckpt_json = output_dir / "checkpoint.json"
    if ckpt_json.exists():
        try:
            data = json.loads(ckpt_json.read_text())
            records = data.get("records", [])
            if records:
                return int(records[-1].get("step", 0))
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # Fallback: parse last line of energy.csv
    energy_csv = output_dir / "energy.csv"
    if energy_csv.exists() and energy_csv.stat().st_size > 0:
        try:
            last_line = energy_csv.read_text().strip().rsplit("\n", 1)[-1]
            return int(last_line.split(",", 1)[0])
        except (ValueError, IndexError):
            pass

    return 0
