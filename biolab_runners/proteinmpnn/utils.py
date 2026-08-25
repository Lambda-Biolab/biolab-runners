"""CLI + availability helpers for the ProteinMPNN runner."""

from __future__ import annotations

import dataclasses
import json
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
    "materialize_fixed_positions_jsonl",
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
        return False
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
        raise ValueError(
            "ProteinMPNN container:// execution is unsupported; "
            "configure PROTEINMPNN_BIN with the proteinmpnn adapter or an executable wrapper"
        )
    return [binary]


def parse_fasta_sequences(path: Path, *, full_header: bool = False) -> list[tuple[str, str]]:
    """Parse a FASTA file into a list of ``(name, sequence)`` tuples.

    ProteinMPNN emits ``<input>.fa`` with one header per sequence.
    The runner only needs the sequence strings; the parsed names
    are preserved for downstream tooling.
    """
    lines = path.read_text().splitlines()
    return _parse_fasta_lines(lines, full_header=full_header)


def build_invocation_command(
    *,
    config_dict: dict[str, str],
    input_pdb: Path,
    output_dir: Path,
    binary_prefix: list[str] | None = None,
) -> tuple[str, ...]:
    """Build the exact argv payload sent to ProteinMPNN."""
    prefix = binary_prefix if binary_prefix is not None else _resolved_binary()
    if any(token.startswith("container://") for token in prefix):
        raise ValueError(
            "ProteinMPNN container:// execution is unsupported; "
            "configure binary_prefix with an executable command"
        )
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


def materialize_fixed_positions_jsonl(
    *,
    fixed_positions: tuple[int, ...],
    pdb_path_chains: object,
    input_pdb: Path,
    output_dir: Path,
) -> Path:
    """Write the deterministic chain-aware fixed-position input for upstream."""
    chains = _parse_pdb_path_chains(pdb_path_chains)
    positions = _normalise_fixed_positions(fixed_positions)
    chain_lengths = _parse_pdb_chain_lengths(input_pdb)
    _validate_position_ranges(positions, chains, chain_lengths)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fixed_positions.jsonl"
    payload = {input_pdb.stem: dict.fromkeys(sorted(chains), positions)}
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def _parse_pdb_path_chains(value: object) -> tuple[str, ...]:
    """Validate the space-separated one-character chain contract."""
    if not isinstance(value, str):
        raise ValueError("fixed_positions requires space-separated pdb_path_chains")
    chains = value.split()
    if not chains:
        raise ValueError("fixed_positions requires pdb_path_chains")
    if any(len(chain) != 1 for chain in chains):
        raise ValueError("pdb_path_chains must contain one-character chain IDs")
    if len(set(chains)) != len(chains):
        raise ValueError("pdb_path_chains must contain unique chain IDs")
    return tuple(chains)


def _normalise_fixed_positions(fixed_positions: tuple[int, ...]) -> tuple[int, ...]:
    """Validate and sort the caller's 1-indexed positions."""
    if len(set(fixed_positions)) != len(fixed_positions):
        raise ValueError("fixed_positions must not contain duplicate positions")
    return tuple(sorted(fixed_positions))


def _parse_pdb_chain_lengths(input_pdb: Path) -> dict[str, int]:
    """Return validated sequence lengths for every PDB chain.

    ProteinMPNN addresses residues by their sequence position within a chain,
    so a PDB residue number is useful only for proving that the mapping is
    unambiguous.  Repeated atoms and alternate locations share one residue.
    """
    try:
        lines = input_pdb.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"unable to read PDB input {input_pdb}") from exc

    residues: dict[str, list[int]] = {}
    residue_names: dict[tuple[str, int], str] = {}
    atom_found = False
    for line in lines:
        if not line.startswith("ATOM  "):
            continue
        atom_found = True
        chain, number, residue_name = _parse_pdb_atom(line)
        _append_pdb_residue(residues, residue_names, chain, number, residue_name)
    if not atom_found:
        raise ValueError("PDB input contains no ATOM records")
    return {chain: len(chain_residues) for chain, chain_residues in residues.items()}


def _is_integer(value: str) -> bool:
    """Return whether a fixed-width PDB field contains a decimal integer."""
    try:
        int(value.strip())
    except ValueError:
        return False
    return True


def _parse_pdb_atom(line: str) -> tuple[str, int, str]:
    """Parse the fixed-width fields needed for sequence-position mapping."""
    if len(line) < 27:
        raise ValueError("PDB ATOM record is too short")
    if not line[6:11].strip() or not _is_integer(line[6:11]):
        raise ValueError("PDB ATOM record has an invalid atom serial")
    if not line[12:16].strip() or not line[17:20].strip():
        raise ValueError("PDB ATOM record has invalid atom fields")
    chain = line[21]
    residue_number = line[22:26].strip()
    if not residue_number:
        raise ValueError("PDB ATOM record has an empty residue number")
    if not _is_integer(residue_number):
        raise ValueError("PDB ATOM record has an invalid residue number")
    if line[26].strip():
        raise ValueError("PDB insertion codes are unsupported for fixed positions")
    if not chain.strip():
        raise ValueError("PDB ATOM record has an empty chain ID")
    return chain, int(residue_number), line[17:20].strip()


def _append_pdb_residue(
    residues: dict[str, list[int]],
    residue_names: dict[tuple[str, int], str],
    chain: str,
    number: int,
    residue_name: str,
) -> None:
    """Add one residue while ignoring duplicate atoms and alternate locations."""
    chain_residues = residues.setdefault(chain, [])
    if chain_residues and chain_residues[-1] != number:
        if number != chain_residues[-1] + 1:
            raise ValueError(f"PDB residue numbering is not contiguous for chain {chain!r}")
        chain_residues.append(number)
    elif not chain_residues:
        chain_residues.append(number)
    residue_key = (chain, number)
    previous_name = residue_names.setdefault(residue_key, residue_name)
    if previous_name != residue_name:
        raise ValueError(f"PDB residue fields are ambiguous for chain {chain!r}")


def _validate_position_ranges(
    positions: tuple[int, ...], chains: tuple[str, ...], chain_lengths: dict[str, int]
) -> None:
    """Reject positions absent from any safely parsed designed chain."""
    for chain in chains:
        if chain not in chain_lengths:
            raise ValueError(f"pdb_path_chains selects missing chain {chain!r}")
        chain_length = chain_lengths[chain]
        if any(position > chain_length for position in positions):
            raise ValueError(f"fixed_positions are out of range for chain {chain!r}")


def _parse_fasta_lines(lines: list[str], *, full_header: bool = False) -> list[tuple[str, str]]:
    """Inner helper that splits a FASTA into ``(name, sequence)`` pairs."""
    records: list[tuple[str, str]] = []
    current_name: str | None = None
    current_seq: list[str] = []
    for line in lines:
        if line.startswith(">"):
            _flush_record(records, current_name, current_seq)
            current_name = _parse_header(line, full_header=full_header)
            current_seq = []
        else:
            current_seq.append(line.strip())
    _flush_record(records, current_name, current_seq)
    return records


def _flush_record(records: list[tuple[str, str]], name: str | None, seq: list[str]) -> None:
    """Append a (name, sequence) pair to ``records`` if a name is pending."""
    if name is not None:
        records.append((name, "".join(seq)))


def _parse_header(line: str, *, full_header: bool = False) -> str:
    """Return the FASTA header name (everything after ``>`` until whitespace)."""
    if full_header:
        return line[1:].strip()
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
