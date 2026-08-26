"""Tests for the Rosetta decoy artifact contract."""

import json
from dataclasses import FrozenInstanceError

import pytest
from biolab_runners.contracts import ExecutionStatus
from biolab_runners.rosetta import ChainAudit, PDBIdentity, RosettaDecoyArtifact
from biolab_runners.rosetta.utils import RelaxScore

_DIGEST = "a" * 64


def _identity(uri: str = "gs://bucket/input.pdb") -> PDBIdentity:
    return PDBIdentity(uri=uri, sha256=_DIGEST)


def _artifact(**overrides: object) -> RosettaDecoyArtifact:
    values: dict[str, object] = {
        "candidate_identity": "candidate-7",
        "parent_input_identity": "parent-3",
        "protocol_identity": "protocol-1",
        "config_identity": "config-2",
        "runtime_identity": "runtime-4",
        "input_pdb_identity": _identity(),
        "output_pdb_identity": _identity("https://example.test/output.pdb"),
        "chain_audits": (
            ChainAudit(chain_id="X", role="binder-like", residue_count=12, atom_count=91),
            ChainAudit(chain_id="q7", role="opaque partner", residue_count=8, atom_count=64),
        ),
        "relax_score": RelaxScore(total_score=-12.5, packstat=0.7),
        "status": ExecutionStatus.SUCCEEDED,
    }
    values.update(overrides)
    return RosettaDecoyArtifact(**values)  # type: ignore[arg-type]


def test_rosetta_artifact_serializes_nested_values_and_preserves_order() -> None:
    artifact = _artifact()

    expected = {
        "candidate_identity": "candidate-7",
        "parent_input_identity": "parent-3",
        "protocol_identity": "protocol-1",
        "config_identity": "config-2",
        "runtime_identity": "runtime-4",
        "input_pdb_identity": {"uri": "gs://bucket/input.pdb", "sha256": _DIGEST},
        "output_pdb_identity": {
            "uri": "https://example.test/output.pdb",
            "sha256": _DIGEST,
        },
        "chain_audits": [
            {"chain_id": "X", "role": "binder-like", "residue_count": 12, "atom_count": 91},
            {"chain_id": "q7", "role": "opaque partner", "residue_count": 8, "atom_count": 64},
        ],
        "relax_score": {
            "total_score": -12.5,
            "total_sasa": None,
            "delta_sasa": None,
            "hydrophobic_sasa": None,
            "polar_sasa": None,
            "interface_polar_sasa": None,
            "interface_hydrophobic_sasa": None,
            "interface_dG": None,
            "interface_dSASA": None,
            "buried_unsatisfied_hbonds": None,
            "cross_interface_hbonds": None,
            "hbond_energy": None,
            "hbond_energy_fraction": None,
            "shape_complementarity": None,
            "packstat": 0.7,
        },
        "status": "succeeded",
        "schema_version": 1,
    }

    assert artifact.to_dict() == expected


def test_rosetta_artifact_payload_is_json_safe_and_round_trips_as_a_dictionary() -> None:
    payload = _artifact().to_dict()

    round_trip = json.loads(json.dumps(payload))

    assert round_trip == payload


def test_identity_requires_nonempty_uri_and_bare_lowercase_sha256() -> None:
    with pytest.raises(ValueError, match="uri"):
        PDBIdentity(uri="", sha256=_DIGEST)
    with pytest.raises(ValueError, match="64 lowercase"):
        PDBIdentity(uri="pdb://one", sha256="sha256:" + _DIGEST)
    with pytest.raises(ValueError, match="64 lowercase"):
        PDBIdentity(uri="pdb://one", sha256="A" * 64)
    with pytest.raises(ValueError, match="64 lowercase"):
        PDBIdentity(uri="pdb://one", sha256="a" * 63)


@pytest.mark.parametrize(
    "field_name",
    [
        "candidate_identity",
        "parent_input_identity",
        "protocol_identity",
        "config_identity",
        "runtime_identity",
    ],
)
def test_artifact_rejects_empty_identity_strings(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _artifact(**{field_name: ""})


@pytest.mark.parametrize("field_name", ["chain_id", "role"])
def test_chain_audit_rejects_empty_strings(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        ChainAudit(
            chain_id="A" if field_name != "chain_id" else "",
            role="target" if field_name != "role" else "",
            residue_count=1,
            atom_count=1,
        )


@pytest.mark.parametrize("field_name", ["residue_count", "atom_count"])
@pytest.mark.parametrize("count", [0, -1])
def test_chain_audit_rejects_nonpositive_counts(field_name: str, count: int) -> None:
    with pytest.raises(ValueError, match=field_name):
        ChainAudit(
            chain_id="not-A",
            role="not-activin-role",
            residue_count=count if field_name == "residue_count" else 1,
            atom_count=count if field_name == "atom_count" else 1,
        )


def test_artifact_accepts_generic_non_abc_chain_ids_and_roles() -> None:
    artifact = _artifact(
        chain_audits=(ChainAudit("chain-42", "custom role", 1, 2),),
    )

    assert artifact.chain_audits[0].chain_id == "chain-42"
    assert artifact.chain_audits[0].role == "custom role"


def test_artifact_requires_nonempty_unique_chain_audits() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _artifact(chain_audits=())
    with pytest.raises(ValueError, match="unique"):
        _artifact(
            chain_audits=(
                ChainAudit("same", "one", 1, 1),
                ChainAudit("same", "two", 2, 2),
            ),
        )


@pytest.mark.parametrize("status", [ExecutionStatus.PENDING, ExecutionStatus.RUNNING])
def test_artifact_rejects_nonterminal_status(status: ExecutionStatus) -> None:
    with pytest.raises(ValueError, match="terminal"):
        _artifact(status=status)


def test_artifact_requires_schema_version_one() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        _artifact(schema_version=2)


def test_artifact_dataclasses_are_frozen() -> None:
    artifact = _artifact()

    with pytest.raises(FrozenInstanceError):
        artifact.status = ExecutionStatus.FAILED  # type: ignore[misc]
