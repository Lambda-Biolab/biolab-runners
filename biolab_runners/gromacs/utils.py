"""CLI + availability helpers for the GROMACS runner, plus the protocol manifest.

This module owns:

- :func:`gromacs_available` — PATH probe (preserved from S3).
- :class:`GromacsRecord` and :class:`GromacsRecordStatus` — the
  per-run record (preserved from S3).
- :func:`parse_nthcol_energy` — ``.xvg`` parser (preserved from S3).
- :func:`invoke` — the thin ``gmx mdrun`` subprocess wrapper
  (preserved from S3; the S4 protocol runner calls ``subprocess.run``
  directly so it can install a SIGTERM handler before invoking the
  child).

The S4 additions are at the bottom of the file:

- :func:`load_stage_manifest` / :func:`save_stage_manifest` /
  :func:`record_stage_status` — the structured stage manifest I/O.
  The manifest is a JSON file at ``work_dir /
  GromacsFiles.STAGE_MANIFEST``; the runner reads it to decide
  skip-vs-resume-vs-fresh on every invocation and writes it
  progressively after each stage completes.

The manifest schema is versioned (``schema_version = 1``); a future
version that breaks the schema MUST bump the version and gate
the loader on it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "GromacsRecord",
    "GromacsRecordStatus",
    "StageStatus",
    "gromacs_available",
    "invoke",
    "load_stage_manifest",
    "parse_nthcol_energy",
    "record_stage_status",
    "save_stage_manifest",
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
    """Run ``gmx mdrun`` once; returns the process exit code.

    Used by the legacy one-shot runner; the S4 protocol runner calls
    ``subprocess.run`` / ``subprocess.Popen`` directly so it can
    install a SIGTERM handler before invoking the child (Spot /
    preemption semantics).
    """
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


# ---------------------------------------------------------------------------
# S4 stage manifest
# ---------------------------------------------------------------------------


class StageStatus:
    """Outcome states recorded in the structured stage manifest.

    The runner writes one of these strings into
    ``stage_record["status"]`` after each stage completes. A
    ``PENDING`` stage is one the runner has not yet attempted;
    ``RUNNING`` is set when the subprocess is launched and cleared
    when the subprocess returns (Spot interruption leaves
    ``RUNNING`` for the next invocation to detect and recover from).
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


_STAGE_MANIFEST_SCHEMA_VERSION = 1
_STAGE_MANIFEST_STATUSES = frozenset(
    {
        StageStatus.PENDING,
        StageStatus.RUNNING,
        StageStatus.COMPLETED,
        StageStatus.FAILED,
    }
)


def load_stage_manifest(work_dir: Path) -> dict[str, Any]:
    """Load the structured stage manifest from ``work_dir``.

    Returns a fresh empty manifest when the file does not exist
    (the runner treats the absent-manifest case as "no stages have
    ever completed" and starts at the topology stage). Returns
    the on-disk manifest when present, regardless of schema
    version (the runner is forward-compatible by ignoring extra
    keys).

    Returns:
        The manifest as a ``dict``; ``{"stages": {<kind>: {...}}}``
        keyed by :class:`biolab_runners.gromacs.protocol.StageKind`
        values. A fresh manifest is
        ``{"schema_version": 1, "stages": {}}``.
    """
    # Local import to avoid a circular dependency
    # (utils ↔ protocol ↔ paths are import-clean today; this comment
    # documents the seam).
    from biolab_runners.gromacs.paths import GromacsFiles

    manifest_path = work_dir / GromacsFiles.STAGE_MANIFEST
    if not manifest_path.exists():
        return {"schema_version": _STAGE_MANIFEST_SCHEMA_VERSION, "stages": {}}
    try:
        data = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Manifest at %s is unreadable (%s); starting fresh", manifest_path, exc)
        return {"schema_version": _STAGE_MANIFEST_SCHEMA_VERSION, "stages": {}}
    if not isinstance(data, dict) or "stages" not in data:
        return {"schema_version": _STAGE_MANIFEST_SCHEMA_VERSION, "stages": {}}
    if not isinstance(data["stages"], dict):
        return {"schema_version": _STAGE_MANIFEST_SCHEMA_VERSION, "stages": {}}
    return data


def save_stage_manifest(work_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically write the stage manifest to ``work_dir``.

    Atomicity is via ``os.replace`` on a temp file in the same
    directory — a crash mid-write leaves the previous manifest
    intact (the runner's skip-vs-resume decision is conservative
    under that condition: a missing stage record forces a re-run).
    """
    from biolab_runners.gromacs.paths import GromacsFiles

    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / GromacsFiles.STAGE_MANIFEST
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(str(tmp_path), str(manifest_path))


def record_stage_status(
    work_dir: Path,
    stage_kind: str,
    status: str,
    *,
    outputs: tuple[str, ...] = (),
    command: str = "",
    started_at: str | None = None,
    completed_at: str | None = None,
    error: str = "",
    prebuilt_source: dict[str, Any] | None = None,
) -> None:
    """Record a stage's outcome into the manifest.

    Loads the current manifest, updates the ``stage_kind`` record,
    and saves atomically. The status MUST be one of the
    :class:`StageStatus` values; an invalid status raises
    ``ValueError`` (the caller is responsible for translating
    subprocess exit codes into one of the four canonical states).

    The optional ``prebuilt_source`` kwarg is consumed by the
    GROMACS TOPOLOGY stage in prebuilt mode — it carries the
    caller-supplied ``.top`` / ``.gro`` paths + their digests so
    that a future invocation with a different prebuilt source
    correctly invalidates the cached stage (see
    ``_prebuilt_source_changed`` in the runner).
    """
    if status not in _STAGE_MANIFEST_STATUSES:
        raise ValueError(
            f"invalid status {status!r}; must be one of {sorted(_STAGE_MANIFEST_STATUSES)}"
        )
    manifest = load_stage_manifest(work_dir)
    record: dict[str, Any] = manifest["stages"].get(stage_kind, {})
    record["status"] = status
    if outputs:
        record["outputs"] = list(outputs)
    if command:
        record["command"] = command
    if started_at is not None:
        record["started_at"] = started_at
    if completed_at is not None:
        record["completed_at"] = completed_at
    if error:
        record["error"] = error
    if prebuilt_source is not None:
        record["prebuilt_source"] = prebuilt_source
    manifest["stages"][stage_kind] = record
    save_stage_manifest(work_dir, manifest)


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    The runner stamps ``started_at`` and ``completed_at`` on each
    stage record. The format is ``YYYY-MM-DDTHH:MM:SS.ffffff+00:00``
    (microsecond precision; matches the OpenMM checkpoint module's
    convention).
    """
    return datetime.now(tz=UTC).isoformat()
