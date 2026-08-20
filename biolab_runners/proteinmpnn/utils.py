"""CLI + availability helpers for the ProteinMPNN runner."""

from __future__ import annotations

import dataclasses
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from biolab_runners.contracts import (
    RunnerInvocationError,
    RunnerTimeoutError,
    RunnerUnavailableError,
)
from biolab_runners.provenance import InvokeResult, stderr_tail

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DesignRecord",
    "DesignRecordStatus",
    "build_invocation_command",
    "invoke",
    "parse_fasta_sequences",
    "proteinmpnn_available",
]


class DesignRecordStatus:
    """Normalized outcome values for per-design sequence records."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class DesignRecord:
    """One designed sequence returned by ProteinMPNN."""

    index: int
    sequence: str
    score: float
    path: str
    status: str = DesignRecordStatus.SUCCEEDED
    error: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialize the record into a JSON-safe dictionary."""
        return {
            "index": str(self.index),
            "sequence": self.sequence,
            "score": repr(self.score),
            "path": self.path,
            "status": self.status,
            "error": self.error,
        }


_FASTA_HEADER_RE = re.compile(r"^>\s*(\S+)")


def proteinmpnn_available(timeout_seconds: int = 30) -> bool:
    """Return True when the upstream ProteinMPNN CLI can be invoked."""
    import os

    binary = os.environ.get("PROTEINMPNN_BIN", "proteinmpnn")
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
    """Return the command prefix used to invoke ProteinMPNN."""
    import os

    binary = os.environ.get("PROTEINMPNN_BIN", "proteinmpnn")
    if binary.startswith("container://"):
        spec = binary[len("container://") :]
        runtime = os.environ.get("CONTAINER_RUNTIME", "docker")
        return [
            runtime,
            "run",
            "--rm",
            spec,
            "python",
            "/app/ProteinMPNN/protein_mpnn_run.py",
        ]
    return [binary]


def parse_fasta_sequences(path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into a list of ``(name, sequence)`` tuples.

    ProteinMPNN emits ``<input>.fa`` with one header per sequence.
    The runner only needs the sequence strings; the parsed names
    are preserved for downstream tooling.
    """
    lines = path.read_text().splitlines()
    return _parse_fasta_lines(lines)


def build_invocation_command(
    *,
    config_dict: dict[str, str],
    input_pdb: Path,
    output_dir: Path,
    binary_prefix: list[str] | None = None,
) -> tuple[str, ...]:
    """Build the exact argv payload sent to ProteinMPNN."""
    prefix = binary_prefix if binary_prefix is not None else _resolved_binary()
    extra_args: list[str] = []
    for key, value in config_dict.items():
        extra_args.extend((f"--{key}", str(value)))
    return (
        *prefix,
        "--input_path",
        str(input_pdb.parent),
        "--output_path",
        str(output_dir),
        "--batch_size",
        "1",
        *extra_args,
    )


def _parse_fasta_lines(lines: list[str]) -> list[tuple[str, str]]:
    """Inner helper that splits a FASTA into ``(name, sequence)`` pairs."""
    records: list[tuple[str, str]] = []
    current_name: str | None = None
    current_seq: list[str] = []
    for line in lines:
        if line.startswith(">"):
            _flush_record(records, current_name, current_seq)
            current_name = _parse_header(line)
            current_seq = []
        else:
            current_seq.append(line.strip())
    _flush_record(records, current_name, current_seq)
    return records


def _flush_record(records: list[tuple[str, str]], name: str | None, seq: list[str]) -> None:
    """Append a (name, sequence) pair to ``records`` if a name is pending."""
    if name is not None:
        records.append((name, "".join(seq)))


def _parse_header(line: str) -> str:
    """Return the FASTA header name (everything after ``>`` until whitespace)."""
    match = _FASTA_HEADER_RE.match(line)
    return match.group(1) if match else ""


def _invoke_with_metadata(
    *,
    config_dict: dict[str, str],
    input_pdb: Path,
    output_dir: Path,
    binary_prefix: list[str] | None = None,
    timeout_seconds: int = 3600,
) -> InvokeResult:
    """Internal helper: run ProteinMPNN once and capture rich metadata.

    Returns an :class:`InvokeResult` carrying the exit code, a
    512-char stderr tail, the timeout flag, and a short failure
    reason. Public callers use the legacy :func:`invoke` wrapper
    (which discards everything except the exit code); the provenance
    provenance wiring uses this helper directly.
    """
    prefix = binary_prefix if binary_prefix is not None else _resolved_binary()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = list(
        build_invocation_command(
            config_dict=config_dict,
            input_pdb=input_pdb,
            output_dir=output_dir,
            binary_prefix=prefix,
        )
    )
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
        logger.error("ProteinMPNN timed out after %ds", timeout_seconds)
        return InvokeResult(
            exit_code=124,
            stderr_tail=stderr_tail(exc.stderr),
            timed_out=True,
            failure_reason=f"timeout after {timeout_seconds}s",
            command=tuple(args),
        )
    except FileNotFoundError as exc:
        raise RunnerUnavailableError(
            f"ProteinMPNN executable unavailable: {args[0]}", runner="proteinmpnn"
        ) from exc
    except OSError as exc:
        raise RunnerInvocationError(
            f"ProteinMPNN invocation failed: {exc}", runner="proteinmpnn"
        ) from exc
    elapsed = time.monotonic() - started
    logger.info("ProteinMPNN run finished rc=%d in %.1fs", completed.returncode, elapsed)
    result = InvokeResult.from_stderr(exit_code=completed.returncode, stderr=completed.stderr)
    return dataclasses.replace(result, command=tuple(args))


def invoke(
    *,
    config_dict: dict[str, str],
    input_pdb: Path,
    output_dir: Path,
    binary_prefix: list[str] | None = None,
    timeout_seconds: int = 3600,
) -> int:
    """Run ProteinMPNN once; returns the process exit code.

    Legacy ``int`` return type — preserved for backward compatibility.
    New code that needs stderr / timeout metadata should call
    :func:`_invoke_with_metadata` directly.
    """
    result = _invoke_with_metadata(
        config_dict=config_dict,
        input_pdb=input_pdb,
        output_dir=output_dir,
        binary_prefix=binary_prefix,
        timeout_seconds=timeout_seconds,
    )
    if result.timed_out:
        raise RunnerTimeoutError(result.failure_reason, runner="proteinmpnn")
    return result.exit_code
