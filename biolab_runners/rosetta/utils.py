"""CLI + availability helpers for the Rosetta runner."""

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

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "RelaxRecord",
    "RelaxRecordStatus",
    "parse_score_file",
    "rosetta_available",
]


class RelaxRecordStatus:
    """Normalized outcome values for per-structure relax records."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class RelaxRecord:
    """One relax output produced by rosetta_scripts."""

    index: int
    path: str
    total_score: float
    status: str = RelaxRecordStatus.SUCCEEDED
    error: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialize the record into a JSON-safe dictionary."""
        return {
            "index": str(self.index),
            "path": self.path,
            "total_score": repr(self.total_score),
            "status": self.status,
            "error": self.error,
        }


_FLOAT_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def rosetta_available(timeout_seconds: int = 30) -> bool:
    """Return True when the upstream Rosetta CLI is callable."""
    import os

    binary = os.environ.get("ROSETTA_BIN", "rosetta_scripts")
    if binary.startswith("container://"):
        return True
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
    """Return the command prefix used to invoke rosetta_scripts."""
    import os

    binary = os.environ.get("ROSETTA_BIN", "rosetta_scripts")
    if binary.startswith("container://"):
        spec = binary[len("container://") :]
        runtime = os.environ.get("CONTAINER_RUNTIME", "docker")
        return [runtime, "run", "--rm", spec, "rosetta_scripts"]
    return [binary]


def parse_score_file(path: Path) -> float:
    """Parse the score column from a Rosetta ``score.sc`` output.

    The score file is a whitespace-separated table with ``score`` as
    the leading total-energy column. Header lines that begin with
    ``SCORE:`` are skipped. Body lines look like ``SCORE:  -123.456  ...``;
    the ``SCORE:`` prefix is stripped before the float column is
    parsed. Returns the first row's total score, or 0.0 if the file
    is empty or malformed.
    """
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("SCORE:"):
            stripped = stripped[len("SCORE:") :].lstrip()
        tokens = stripped.split()
        if not tokens:
            continue
        if not _FLOAT_RE.match(tokens[0]):
            continue
        try:
            return float(tokens[0])
        except ValueError:
            continue
    return 0.0


def invoke(
    *,
    config: dict[str, str],
    output_dir: Path,
    binary_prefix: list[str] | None = None,
    timeout_seconds: int = 3600,
) -> int:
    """Run rosetta_scripts once; returns the process exit code."""
    prefix = binary_prefix if binary_prefix is not None else _resolved_binary()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = [
        *prefix,
        "--parser",
        "protocol",
        *functools.reduce(
            operator.iadd,
            (
                [f"-{key}" if not key.startswith("-") else key, str(value)]
                for key, value in config.items()
            ),
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
    except subprocess.TimeoutExpired:
        logger.error("Rosetta timed out after %ds", timeout_seconds)
        return 124
    logger.info(
        "Rosetta run finished rc=%d in %.1fs",
        completed.returncode,
        time.monotonic() - started,
    )
    return completed.returncode


def parse_score_files(score_files: Iterable[Path]) -> list[RelaxRecord]:
    """Convert a list of ``score.sc`` paths into :class:`RelaxRecord`."""
    records: list[RelaxRecord] = []
    for index, path in enumerate(score_files):
        try:
            score = parse_score_file(path)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("failed to parse %s: %s", path, exc)
            records.append(
                RelaxRecord(
                    index=len(records),
                    path=str(path),
                    total_score=0.0,
                    status=RelaxRecordStatus.FAILED,
                    error=str(exc),
                )
            )
            continue
        records.append(RelaxRecord(index=index, path=str(path), total_score=score))
    return records
