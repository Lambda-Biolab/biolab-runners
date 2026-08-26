"""Pure validation and resolution of Rosetta decoy output records."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from biolab_runners.contracts import (
    ExecutionStatus,
    IncompleteOutputError,
    MalformedOutputError,
)
from biolab_runners.rosetta.artifact import ChainAudit, PDBIdentity, RosettaDecoyArtifact
from biolab_runners.rosetta.utils import (
    METRIC_ALIASES,
    RelaxScoreRow,
    parse_relax_score_rows,
)

__all__ = ["RosettaDecoyResolutionRequest", "resolve_decoy"]

ChainRole = tuple[str, str]


@dataclass(frozen=True, kw_only=True)
class RosettaDecoyResolutionRequest:
    """Inputs required to resolve one completed Rosetta decoy."""

    score_file: Path
    input_pdb: Path
    output_pdb: Path
    input_pdb_uri: str
    output_pdb_uri: str
    candidate_identity: str
    parent_input_identity: str
    protocol_identity: str
    config_identity: str
    runtime_identity: str
    expected_chain_roles: tuple[ChainRole, ...]
    decoy_description: str | None = None
    required_metrics: tuple[str, ...] = ("total_score",)
    status: ExecutionStatus = ExecutionStatus.SUCCEEDED

    def __post_init__(self) -> None:
        """Reject mutable or ambiguous request configuration."""
        _validate_request_paths((self.score_file, self.input_pdb, self.output_pdb))
        _validate_expected_chain_roles(self.expected_chain_roles)
        _validate_required_metrics(self.required_metrics)
        if self.decoy_description is not None and (
            type(self.decoy_description) is not str or not self.decoy_description
        ):
            raise ValueError("decoy_description must be a non-empty string")
        if type(self.status) is not ExecutionStatus:
            raise ValueError("status must be an ExecutionStatus")


def _validate_request_paths(paths: tuple[object, ...]) -> None:
    if any(not isinstance(path, Path) for path in paths):
        raise ValueError("score_file, input_pdb, and output_pdb must be Paths")


def _validate_expected_chain_roles(expected: object) -> None:
    if not isinstance(expected, tuple) or not expected:
        raise ValueError("expected_chain_roles must be a non-empty tuple")
    chains: list[str] = []
    for pair in expected:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("expected_chain_roles must contain (chain, role) pairs")
        chain_id, role = pair
        if not isinstance(chain_id, str) or len(chain_id) != 1 or not chain_id.strip():
            raise ValueError("expected chain IDs must be exactly one non-whitespace character")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("expected chain roles must be non-empty")
        chains.append(chain_id)
    if len(chains) != len(set(chains)):
        raise ValueError("expected_chain_roles must have unique chain IDs")


def _validate_required_metrics(metrics: object) -> None:
    if not isinstance(metrics, tuple) or not metrics:
        raise ValueError("required_metrics must be a non-empty tuple")
    unknown = [metric for metric in metrics if metric not in METRIC_ALIASES]
    if unknown:
        raise ValueError(f"unknown required metric: {unknown[0]}")
    if len(metrics) != len(set(metrics)):
        raise ValueError("required_metrics must have unique names")


def _incomplete(message: str) -> NoReturn:
    raise IncompleteOutputError(message, runner="rosetta")


def _malformed(message: str) -> NoReturn:
    raise MalformedOutputError(message, runner="rosetta")


def _read_required_file(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        _incomplete(f"{label} is missing or not a regular file: {path}")
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise MalformedOutputError(f"{label} is unreadable: {path}", runner="rosetta") from exc
    if not contents:
        _incomplete(f"{label} is empty: {path}")
    return contents


def _select_score_row(
    rows: tuple[RelaxScoreRow, ...], request: RosettaDecoyResolutionRequest
) -> RelaxScoreRow:
    if not rows:
        _malformed("score file contains no data rows")
    if request.decoy_description is None:
        if len(rows) != 1:
            _malformed("multiple score rows require an exact decoy_description")
        row = rows[0]
    else:
        matches = tuple(row for row in rows if row.description == request.decoy_description)
        if len(matches) != 1:
            _malformed("decoy_description does not identify exactly one score row")
        row = matches[0]
    description = row.description
    if (
        not description
        or description in {".", ".."}
        or "/" in description
        or "\\" in description
        or "\x00" in description
    ):
        _malformed("score description is unsafe or missing")
    if description not in {request.output_pdb.name, request.output_pdb.stem}:
        _malformed("score description does not match output PDB filename")
    return row


def _validate_score(row: RelaxScoreRow, metrics: tuple[str, ...]) -> None:
    for metric in metrics:
        value = getattr(row.score, metric)
        if value is None or not math.isfinite(value):
            _malformed(f"selected score row is missing required metric: {metric}")


def _parse_atom_line(line: str) -> tuple[str, tuple[int, str]]:
    if len(line) < 54:
        _malformed("ATOM/HETATM record is too short")
    chain_id = line[21].strip()
    if not chain_id:
        _malformed("ATOM/HETATM record has a blank chain")
    try:
        int(line[6:11].strip())
        if not line[12:16].strip() or not line[17:20].strip():
            _malformed("ATOM/HETATM record has a blank atom or residue name")
        residue_number = int(line[22:26].strip())
        coordinates = tuple(
            float(line[start:end].strip()) for start, end in ((30, 38), (38, 46), (46, 54))
        )
    except ValueError as exc:
        raise MalformedOutputError(
            "ATOM/HETATM record has malformed serial, residue number, or coordinates",
            runner="rosetta",
        ) from exc
    if not all(math.isfinite(coordinate) for coordinate in coordinates):
        _malformed("ATOM/HETATM record has non-finite coordinates")
    return chain_id, (residue_number, line[26].strip())


def _audit_output(output: bytes, expected: tuple[ChainRole, ...]) -> tuple[ChainAudit, ...]:
    try:
        text = output.decode("ascii")
    except UnicodeDecodeError as exc:
        raise MalformedOutputError("output PDB is not ASCII text", runner="rosetta") from exc
    atom_counts: dict[str, int] = {}
    residues: dict[str, set[tuple[int, str]]] = {}
    model_count = 0
    for line in text.splitlines():
        record = line[:6].strip()
        if record == "MODEL":
            model_count += 1
        if record not in {"ATOM", "HETATM"}:
            continue
        chain_id, residue = _parse_atom_line(line)
        atom_counts[chain_id] = atom_counts.get(chain_id, 0) + 1
        residues.setdefault(chain_id, set()).add(residue)
    if model_count > 1:
        _malformed("output PDB contains multiple MODEL records")
    if not atom_counts:
        _malformed("output PDB contains no atoms")
    expected_ids = {chain_id for chain_id, _ in expected}
    if set(atom_counts) != expected_ids:
        _malformed("output PDB chain set does not match expected chain roles")
    roles = dict(expected)
    return tuple(
        ChainAudit(
            chain_id=chain_id,
            role=roles[chain_id],
            residue_count=len(residues[chain_id]),
            atom_count=atom_counts[chain_id],
        )
        for chain_id, _ in expected
    )


def resolve_decoy(request: RosettaDecoyResolutionRequest) -> RosettaDecoyArtifact:
    """Resolve and validate a completed Rosetta decoy without side effects."""
    if request.status is not ExecutionStatus.SUCCEEDED:
        _malformed("only SUCCEEDED is a scientific terminal success")
    input_bytes = _read_required_file(request.input_pdb, "input PDB")
    output_bytes = _read_required_file(request.output_pdb, "output PDB")
    _read_required_file(request.score_file, "score file")
    try:
        rows = parse_relax_score_rows(request.score_file)
    except (OSError, UnicodeError) as exc:
        raise MalformedOutputError("score file is unreadable", runner="rosetta") from exc
    row = _select_score_row(rows, request)
    _validate_score(row, request.required_metrics)
    audits = _audit_output(output_bytes, request.expected_chain_roles)
    return RosettaDecoyArtifact(
        candidate_identity=request.candidate_identity,
        parent_input_identity=request.parent_input_identity,
        protocol_identity=request.protocol_identity,
        config_identity=request.config_identity,
        runtime_identity=request.runtime_identity,
        input_pdb_identity=PDBIdentity(
            request.input_pdb_uri, hashlib.sha256(input_bytes).hexdigest()
        ),
        output_pdb_identity=PDBIdentity(
            request.output_pdb_uri, hashlib.sha256(output_bytes).hexdigest()
        ),
        chain_audits=audits,
        relax_score=row.score,
        status=request.status,
    )
