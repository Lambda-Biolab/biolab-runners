"""Contracts for complete receptor-plus-design-chain preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biolab_runners.contracts import ArtifactReference, ExecutionMode, ExecutionStatus
from biolab_runners.peptide_prep.config import (
    CHIRALITY_RESTRAINT_ALGORITHM_VERSION,
    GROMACS_POSITION_RESTRAINT_ALGORITHM_VERSION,
    GROMACS_TOPOLOGY_MATERIALIZER_VERSION,
    PeptideTopologyDescriptor,
)
from biolab_runners.peptide_prep.utils import THREE_LETTER
from biolab_runners.provenance import build_execution_provenance
from biolab_runners.rosetta.artifact import RosettaDecoyArtifact

RECEPTOR_CHAIN_IDS = ("A", "B")
DESIGN_CHAIN_ID = "C"
PROTEIN_FORCE_FIELD = "amber99sbildn.xml"
WATER_FORCE_FIELD = "tip3p.xml"
PREPARATION_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
PREPARED_PDB_FILENAME = "prepared.pdb"
PREPARED_TOP_FILENAME = "prepared.top"
PREPARED_GRO_FILENAME = "prepared.gro"
SELECTION_MAP_FILENAME = "selection-map.json"
INDEX_FILENAME = "index.ndx"
GROMPP_NOT_RUN = "not_run_gmx_unavailable"
GROMPP_PASSED = "passed"
COMPLEX_PREPARATION_ALGORITHM_VERSION = "complex-preparation-v3"
MUTATION_ALGORITHM_VERSION = "design-chain-mutation-v1"
CLOSURE_ALGORITHM_VERSION = "head-tail-disulfide-closure-v1"
D_COORDINATE_ALGORITHM_VERSION = "chain-local-d-coordinate-input-v2"
MANIFEST_PAYLOAD_DIGEST_FIELD = "scientific_metadata_sha256"
ALGORITHM_VERSIONS = {
    "preparation": COMPLEX_PREPARATION_ALGORITHM_VERSION,
    "mutation": MUTATION_ALGORITHM_VERSION,
    "closure": CLOSURE_ALGORITHM_VERSION,
    "d_coordinate": D_COORDINATE_ALGORITHM_VERSION,
    "chirality": CHIRALITY_RESTRAINT_ALGORITHM_VERSION,
    "gromacs_position_restraint": GROMACS_POSITION_RESTRAINT_ALGORITHM_VERSION,
    "gromacs_topology_export": GROMACS_TOPOLOGY_MATERIALIZER_VERSION,
}

_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_D_COORDINATE_INPUT_MODES = frozenset({"canonical_l", "prepared_d"})

__all__ = [
    "ALGORITHM_VERSIONS",
    "CLOSURE_ALGORITHM_VERSION",
    "COMPLEX_PREPARATION_ALGORITHM_VERSION",
    "DESIGN_CHAIN_ID",
    "D_COORDINATE_ALGORITHM_VERSION",
    "GROMPP_NOT_RUN",
    "GROMPP_PASSED",
    "INDEX_FILENAME",
    "MANIFEST_FILENAME",
    "MANIFEST_PAYLOAD_DIGEST_FIELD",
    "MUTATION_ALGORITHM_VERSION",
    "PREPARATION_SCHEMA_VERSION",
    "PREPARED_GRO_FILENAME",
    "PREPARED_PDB_FILENAME",
    "PREPARED_TOP_FILENAME",
    "PROTEIN_FORCE_FIELD",
    "RECEPTOR_CHAIN_IDS",
    "SELECTION_MAP_FILENAME",
    "WATER_FORCE_FIELD",
    "ComplexChainAudit",
    "ComplexPrepBundle",
    "ComplexPrepConfig",
    "ComplexPrepResult",
    "ComplexResidueAudit",
]


def _empty_topology() -> PeptideTopologyDescriptor:
    return PeptideTopologyDescriptor()


def _require_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_d_entry(entry: object, sequence: str) -> None:
    sequence_length = len(sequence)
    position = getattr(entry, "position", None)
    residue = getattr(entry, "residue", None)
    if (
        isinstance(position, bool)
        or not isinstance(position, int)
        or not 1 <= position <= sequence_length
    ):
        raise ValueError(
            f"topology.d_substitutions entry has invalid position {position!r}; "
            f"design sequence length is {sequence_length}"
        )
    if (
        not isinstance(residue, str)
        or len(residue) != 3
        or residue != residue.upper()
        or residue not in THREE_LETTER.values()
    ):
        raise ValueError(
            f"topology.d_substitutions entry has invalid residue {residue!r}; "
            "expected a 3-letter amino-acid code"
        )
    if residue != THREE_LETTER[sequence[position - 1]]:
        raise ValueError(
            "topology.d_substitutions residue must match the design sequence "
            f"at position {position}"
        )


def _validate_topology(topology: PeptideTopologyDescriptor, sequence: str) -> None:
    d_positions = [_validated_d_position(entry, sequence) for entry in topology.d_substitutions]
    if len(d_positions) != len(set(d_positions)):
        raise ValueError("topology.d_substitutions contains duplicate positions")
    _validate_head_to_tail(topology.head_to_tail, len(sequence))
    _validate_disulfides(topology.disulfides, sequence)


def _validated_d_position(entry: object, sequence: str) -> int:
    _validate_d_entry(entry, sequence)
    position = getattr(entry, "position", None)
    if not isinstance(position, int):
        raise ValueError("topology.d_substitutions entry has an invalid position")
    return position


def _validate_head_to_tail(entry: object | None, sequence_length: int) -> None:
    if entry is None:
        return
    head = getattr(entry, "head", None)
    tail = getattr(entry, "tail", None)
    if (
        isinstance(head, bool)
        or not isinstance(head, int)
        or isinstance(tail, bool)
        or not isinstance(tail, int)
        or head != 1
        or tail != sequence_length
    ):
        raise ValueError(
            "topology.head_to_tail must describe the design chain's true "
            f"terminals (1, {sequence_length})"
        )


def _validate_disulfides(entries: tuple[Any, ...], sequence: str) -> None:
    pairs: set[tuple[int, int]] = set()
    for bond in entries:
        first, second = _validated_disulfide_pair(bond, sequence)
        pair: tuple[int, int] = (min(first, second), max(first, second))
        if pair in pairs:
            raise ValueError("topology.disulfides contains duplicate pairs")
        pairs.add(pair)


def _validated_disulfide_pair(entry: object, sequence: str) -> tuple[int, int]:
    first = getattr(entry, "first", None)
    second = getattr(entry, "second", None)
    if any(
        isinstance(position, bool)
        or not isinstance(position, int)
        or not 1 <= position <= len(sequence)
        for position in (first, second)
    ):
        raise ValueError("topology.disulfides contains an out-of-range position")
    if not isinstance(first, int) or not isinstance(second, int):
        raise ValueError("topology.disulfides contains an out-of-range position")
    if first == second:
        raise ValueError("topology.disulfides cannot connect a residue to itself")
    if sequence[first - 1] != "C" or sequence[second - 1] != "C":
        raise ValueError("topology.disulfides requires CYS design positions")
    return first, second


@dataclass(frozen=True, kw_only=True)
class ComplexPrepConfig:
    """Fixed-policy input for one complete complex preparation bundle."""

    source_pdb: str
    source_decoy: RosettaDecoyArtifact
    output_dir: str
    design_sequence: str
    topology: PeptideTopologyDescriptor = field(default_factory=_empty_topology)
    d_coordinate_input_mode: str = "canonical_l"
    coordinate_transformer_identity: str = ""
    chirality_validator_identity: str = ""

    def __post_init__(self) -> None:
        """Validate caller-owned identity and descriptor boundaries."""
        _require_string(self.source_pdb, "source_pdb")
        _require_string(self.output_dir, "output_dir")
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.source_decoy, RosettaDecoyArtifact
        ):
            raise ValueError("source_decoy must be a RosettaDecoyArtifact")
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.topology, PeptideTopologyDescriptor
        ):
            raise ValueError("topology must be a PeptideTopologyDescriptor")

        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.design_sequence, str
        ):
            raise ValueError("design_sequence must be a string")
        sequence = self.design_sequence.upper()
        if not sequence:
            raise ValueError("design_sequence is required")
        invalid = sorted(set(sequence) - _AMINO_ACIDS)
        if invalid:
            raise ValueError(f"design_sequence contains invalid characters {invalid!r}")
        object.__setattr__(self, "design_sequence", sequence)
        _validate_topology(self.topology, sequence)
        if self.d_coordinate_input_mode not in _D_COORDINATE_INPUT_MODES:
            raise ValueError("d_coordinate_input_mode must be 'canonical_l' or 'prepared_d'")
        if self.d_coordinate_input_mode == "prepared_d" and not self.topology.d_substitutions:
            raise ValueError("d_coordinate_input_mode='prepared_d' requires D substitutions")
        if self.topology.d_substitutions:
            if self.d_coordinate_input_mode == "canonical_l":
                _require_string(
                    self.coordinate_transformer_identity, "coordinate_transformer_identity"
                )
            _require_string(self.chirality_validator_identity, "chirality_validator_identity")


@dataclass(frozen=True)
class ComplexChainAudit:
    """Source and prepared counts for one fixed-role chain."""

    chain_id: str
    role: str
    source_residue_count: int
    prepared_residue_count: int
    source_atom_count: int
    prepared_atom_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return this chain audit as JSON-native data."""
        return {
            "chain_id": self.chain_id,
            "role": self.role,
            "source_residue_count": self.source_residue_count,
            "prepared_residue_count": self.prepared_residue_count,
            "source_atom_count": self.source_atom_count,
            "prepared_atom_count": self.prepared_atom_count,
        }


@dataclass(frozen=True)
class ComplexResidueAudit:
    """Identity and atom counts for one source/prepared residue pair."""

    chain_id: str
    residue_number: int
    insertion_code: str
    source_name: str
    prepared_name: str
    source_atom_count: int
    prepared_atom_count: int
    source_index: int
    prepared_index: int

    def to_dict(self) -> dict[str, Any]:
        """Return this residue audit as JSON-native data."""
        return {
            "chain_id": self.chain_id,
            "residue_number": self.residue_number,
            "insertion_code": self.insertion_code,
            "source_name": self.source_name,
            "prepared_name": self.prepared_name,
            "source_atom_count": self.source_atom_count,
            "prepared_atom_count": self.prepared_atom_count,
            "source_index": self.source_index,
            "prepared_index": self.prepared_index,
        }


@dataclass(frozen=True, kw_only=True)
class ComplexPrepBundle:
    """Immutable references and audits for a complete prepared bundle."""

    output_dir: str
    source_digest: str
    config_digest: str
    preparation_digest: str
    prepared_pdb: ArtifactReference
    prepared_top: ArtifactReference
    prepared_gro: ArtifactReference
    selection_map: ArtifactReference
    index: ArtifactReference
    manifest: ArtifactReference
    chain_audits: tuple[ComplexChainAudit, ...]
    residue_audits: tuple[ComplexResidueAudit, ...]
    grompp_audit_status: str
    net_charge: float
    atom_count: int

    @property
    def artifacts(self) -> tuple[ArtifactReference, ...]:
        """Return all materialized bundle references in deterministic order."""
        return (
            self.prepared_pdb,
            self.prepared_top,
            self.prepared_gro,
            self.selection_map,
            self.index,
            self.manifest,
        )

    @property
    def prepared_pdb_artifact(self) -> ArtifactReference:
        """Alias for the prepared structure artifact reference."""
        return self.prepared_pdb

    @property
    def prepared_top_artifact(self) -> ArtifactReference:
        """Alias for the prepared topology artifact reference."""
        return self.prepared_top

    @property
    def prepared_gro_artifact(self) -> ArtifactReference:
        """Alias for the prepared coordinate artifact reference."""
        return self.prepared_gro

    @property
    def selection_map_artifact(self) -> ArtifactReference:
        """Alias for the selection-map artifact reference."""
        return self.selection_map

    @property
    def index_artifact(self) -> ArtifactReference:
        """Alias for the fixed index artifact reference."""
        return self.index

    @property
    def manifest_artifact(self) -> ArtifactReference:
        """Alias for the manifest artifact reference."""
        return self.manifest

    @property
    def chain_audit(self) -> tuple[ComplexChainAudit, ...]:
        """Alias for the immutable chain-audit collection."""
        return self.chain_audits

    @property
    def residue_audit(self) -> tuple[ComplexResidueAudit, ...]:
        """Alias for the immutable residue-audit collection."""
        return self.residue_audits

    @property
    def grompp_audit(self) -> str:
        """Return the grompp audit status string."""
        return self.grompp_audit_status


@dataclass(frozen=True, kw_only=True)
class ComplexPrepResult:
    """Structured complete-complex preparation outcome."""

    status: ExecutionStatus
    output_dir: str
    source_digest: str = ""
    config_digest: str = ""
    preparation_digest: str = ""
    bundle: ComplexPrepBundle | None = None
    error: str = ""
    executed: bool = False

    @property
    def success(self) -> bool:
        """Whether preparation succeeded or returned a valid cache hit."""
        return self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.CACHED)

    @property
    def reused(self) -> bool:
        """Whether this result came from the strict cache path."""
        return self.status is ExecutionStatus.CACHED

    @property
    def source_pdb_sha256(self) -> str:
        """Return the source-PDB content digest."""
        return self.source_digest

    @property
    def source_config_digest(self) -> str:
        """Return the canonical configuration digest."""
        return self.config_digest

    @property
    def source_preparation_digest(self) -> str:
        """Return the non-circular preparation identity digest."""
        return self.preparation_digest

    @property
    def execution_mode(self) -> ExecutionMode:
        """Complex preparation is always performed in-process."""
        return ExecutionMode.IN_PROCESS

    @property
    def manifest_path(self) -> str:
        """Return the manifest path, or its contract path on failure."""
        if self.bundle is not None:
            return self.bundle.manifest.path
        return str(Path(self.output_dir) / MANIFEST_FILENAME)

    @property
    def artifacts(self) -> tuple[ArtifactReference, ...]:
        """Return bundle artifacts, or no artifacts for a failure."""
        return () if self.bundle is None else self.bundle.artifacts

    @property
    def prepared_pdb(self) -> ArtifactReference | None:
        """Return the prepared PDB reference when available."""
        return None if self.bundle is None else self.bundle.prepared_pdb

    @property
    def prepared_top(self) -> ArtifactReference | None:
        """Return the prepared TOP reference when available."""
        return None if self.bundle is None else self.bundle.prepared_top

    @property
    def prepared_gro(self) -> ArtifactReference | None:
        """Return the prepared GRO reference when available."""
        return None if self.bundle is None else self.bundle.prepared_gro

    @property
    def selection_map(self) -> ArtifactReference | None:
        """Return the selection-map reference when available."""
        return None if self.bundle is None else self.bundle.selection_map

    @property
    def index(self) -> ArtifactReference | None:
        """Return the index reference when available."""
        return None if self.bundle is None else self.bundle.index

    @property
    def manifest(self) -> ArtifactReference | None:
        """Return the manifest reference when available."""
        return None if self.bundle is None else self.bundle.manifest

    @property
    def chain_audits(self) -> tuple[ComplexChainAudit, ...]:
        """Return chain audits, or an empty collection on failure."""
        return () if self.bundle is None else self.bundle.chain_audits

    @property
    def residue_audits(self) -> tuple[ComplexResidueAudit, ...]:
        """Return residue audits, or an empty collection on failure."""
        return () if self.bundle is None else self.bundle.residue_audits

    @property
    def grompp_audit_status(self) -> str:
        """Return grompp status, or an empty value on failure."""
        return "" if self.bundle is None else self.bundle.grompp_audit_status

    @property
    def net_charge(self) -> float:
        """Return the OpenMM-computed net charge for a successful bundle."""
        return 0.0 if self.bundle is None else self.bundle.net_charge

    @property
    def atom_count(self) -> int:
        """Return the final OpenMM particle count for a successful bundle."""
        return 0 if self.bundle is None else self.bundle.atom_count

    @property
    def provenance(self) -> object:
        """Return shared provenance without inventing an executed cache digest."""
        return build_execution_provenance(
            runner_name="complex_prep",
            execution_mode=self.execution_mode,
            status=self.status,
            artifacts=self.artifacts,
            source_backbone_digest=self.source_digest or None,
            requested_config_digest=self.config_digest,
            executed_config_digest=self.config_digest if self.executed else None,
            executed=self.executed,
            cache_hit=self.reused,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result representation."""
        return {
            "status": self.status.value,
            "output_dir": self.output_dir,
            "source_digest": self.source_digest,
            "config_digest": self.config_digest,
            "preparation_digest": self.preparation_digest,
            "bundle": None
            if self.bundle is None
            else {
                "output_dir": self.bundle.output_dir,
                "source_digest": self.bundle.source_digest,
                "config_digest": self.bundle.config_digest,
                "preparation_digest": self.bundle.preparation_digest,
                "artifacts": [artifact.to_dict() for artifact in self.bundle.artifacts],
                "chain_audits": [audit.to_dict() for audit in self.bundle.chain_audits],
                "residue_audits": [audit.to_dict() for audit in self.bundle.residue_audits],
                "grompp_audit_status": self.bundle.grompp_audit_status,
                "net_charge": self.bundle.net_charge,
                "atom_count": self.bundle.atom_count,
            },
            "error": self.error,
            "executed": self.executed,
            "execution_mode": self.execution_mode.value,
        }
