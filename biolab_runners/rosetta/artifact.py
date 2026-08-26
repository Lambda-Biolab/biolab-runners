"""Serializable identity and result contracts for Rosetta decoys."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from biolab_runners.contracts import ExecutionStatus
from biolab_runners.rosetta.utils import RelaxScore

__all__ = ["ChainAudit", "PDBIdentity", "RosettaDecoyArtifact"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL_STATUSES = frozenset(ExecutionStatus) - {
    ExecutionStatus.PENDING,
    ExecutionStatus.RUNNING,
}


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_sha256(value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")


def _validate_positive_count(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_identity_strings(artifact: RosettaDecoyArtifact) -> None:
    for field_name in (
        "candidate_identity",
        "parent_input_identity",
        "protocol_identity",
        "config_identity",
        "runtime_identity",
    ):
        _require_nonempty_string(getattr(artifact, field_name), field_name)


def _validate_chain_audits(audits: object) -> None:
    if not isinstance(audits, tuple) or not audits:
        raise ValueError("chain_audits must be a non-empty tuple")
    if any(not isinstance(audit, ChainAudit) for audit in audits):
        raise ValueError("chain_audits must contain only ChainAudit values")
    chain_ids = [audit.chain_id for audit in audits]
    if len(chain_ids) != len(set(chain_ids)):
        raise ValueError("chain_audits must have unique chain IDs")


def _validate_artifact_metadata(
    relax_score: object, status: object, schema_version: object
) -> None:
    if not isinstance(relax_score, RelaxScore):
        raise ValueError("relax_score must be a RelaxScore")
    if not isinstance(status, ExecutionStatus):
        raise ValueError("status must be an ExecutionStatus")
    if status not in _TERMINAL_STATUSES:
        raise ValueError("status must be terminal")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError("schema_version must be 1")


def _validate_pdb_identity(value: object, field_name: str) -> None:
    if not isinstance(value, PDBIdentity):
        raise ValueError(f"{field_name} must be a PDBIdentity")


@dataclass(frozen=True)
class PDBIdentity:
    """Identify a PDB artifact by URI and its bare SHA-256 digest."""

    uri: str
    sha256: str

    def __post_init__(self) -> None:
        """Validate the URI and strict digest representation."""
        _require_nonempty_string(self.uri, "uri")
        _validate_sha256(self.sha256)

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe identity dictionary."""
        return {"uri": self.uri, "sha256": self.sha256}


@dataclass(frozen=True)
class ChainAudit:
    """Record generic counts and an opaque role for one PDB chain."""

    chain_id: str
    role: str
    residue_count: int
    atom_count: int

    def __post_init__(self) -> None:
        """Validate generic chain identifiers and positive counts."""
        _require_nonempty_string(self.chain_id, "chain_id")
        _require_nonempty_string(self.role, "role")
        for field_name, count in (
            ("residue_count", self.residue_count),
            ("atom_count", self.atom_count),
        ):
            _validate_positive_count(count, field_name)

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-safe chain-audit dictionary."""
        return {
            "chain_id": self.chain_id,
            "role": self.role,
            "residue_count": self.residue_count,
            "atom_count": self.atom_count,
        }


@dataclass(frozen=True)
class RosettaDecoyArtifact:
    """Describe one validated Rosetta decoy and its provenance identities."""

    candidate_identity: str
    parent_input_identity: str
    protocol_identity: str
    config_identity: str
    runtime_identity: str
    input_pdb_identity: PDBIdentity
    output_pdb_identity: PDBIdentity
    chain_audits: tuple[ChainAudit, ...]
    relax_score: RelaxScore
    status: ExecutionStatus
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Validate schema invariants without imposing workflow semantics."""
        _validate_identity_strings(self)
        _validate_pdb_identity(self.input_pdb_identity, "input_pdb_identity")
        _validate_pdb_identity(self.output_pdb_identity, "output_pdb_identity")
        _validate_chain_audits(self.chain_audits)
        _validate_artifact_metadata(self.relax_score, self.status, self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return the artifact in an explicitly JSON-safe representation."""
        return {
            "candidate_identity": self.candidate_identity,
            "parent_input_identity": self.parent_input_identity,
            "protocol_identity": self.protocol_identity,
            "config_identity": self.config_identity,
            "runtime_identity": self.runtime_identity,
            "input_pdb_identity": self.input_pdb_identity.to_dict(),
            "output_pdb_identity": self.output_pdb_identity.to_dict(),
            "chain_audits": [audit.to_dict() for audit in self.chain_audits],
            "relax_score": self.relax_score.to_dict(),
            "status": self.status.value,
            "schema_version": self.schema_version,
        }
