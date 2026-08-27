"""Behavioral tests for the complete A+B+C preparation bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import openmm.app as app
import openmm.unit as unit
import pytest
from biolab_runners.complex_prep import (
    GROMPP_NOT_RUN,
    MANIFEST_PAYLOAD_DIGEST_FIELD,
    SELECTION_MAP_FILENAME,
    ComplexPrepConfig,
    ComplexPrepRunner,
    PeptideTopologyDescriptor,
    compute_complex_config_digest,
)
from biolab_runners.complex_prep import runner as complex_prep_runner
from biolab_runners.contracts import ExecutionStatus
from biolab_runners.peptide_prep import ChiralityReport, CoordinateTransformResult
from biolab_runners.rosetta import ChainAudit, PDBIdentity, RelaxScore, RosettaDecoyArtifact
from pdbfixer import PDBFixer

import openmm


def _complex_pdb_text() -> str:
    atoms = [
        ("N", "ALA", "A", 10, 0.000, 0.000, 0.000, "N"),
        ("CA", "ALA", "A", 10, 1.450, 0.000, 0.000, "C"),
        ("C", "ALA", "A", 10, 2.450, 1.200, 0.000, "C"),
        ("O", "ALA", "A", 10, 2.200, 2.350, 0.000, "O"),
        ("CB", "ALA", "A", 10, 1.400, -1.000, -1.000, "C"),
        ("N", "ALA", "A", 11, 3.550, 1.100, 0.000, "N"),
        ("CA", "ALA", "A", 11, 4.550, 2.200, 0.000, "C"),
        ("C", "ALA", "A", 11, 5.850, 1.400, 0.000, "C"),
        ("O", "ALA", "A", 11, 6.900, 1.800, 0.000, "O"),
        ("CB", "ALA", "A", 11, 4.700, 3.300, -1.000, "C"),
        ("OXT", "ALA", "A", 11, 6.800, 1.500, 0.000, "O"),
        ("N", "GLY", "B", 20, 0.000, 10.000, 0.000, "N"),
        ("CA", "GLY", "B", 20, 1.450, 10.000, 0.000, "C"),
        ("C", "GLY", "B", 20, 2.450, 11.200, 0.000, "C"),
        ("O", "GLY", "B", 20, 2.200, 12.350, 0.000, "O"),
        ("N", "GLY", "B", 21, 3.550, 11.100, 0.000, "N"),
        ("CA", "GLY", "B", 21, 4.550, 12.200, 0.000, "C"),
        ("C", "GLY", "B", 21, 5.850, 11.400, 0.000, "C"),
        ("O", "GLY", "B", 21, 6.900, 11.800, 0.000, "O"),
        ("OXT", "GLY", "B", 21, 6.800, 11.500, 0.000, "O"),
        ("N", "ALA", "C", 30, 0.000, 20.000, 0.000, "N"),
        ("CA", "ALA", "C", 30, 1.450, 20.000, 0.000, "C"),
        ("C", "ALA", "C", 30, 2.450, 21.200, 0.000, "C"),
        ("O", "ALA", "C", 30, 2.200, 22.350, 0.000, "O"),
        ("CB", "ALA", "C", 30, 1.400, 19.000, -1.000, "C"),
        ("N", "ALA", "C", 31, 3.550, 21.100, 0.000, "N"),
        ("CA", "ALA", "C", 31, 4.550, 22.200, 0.000, "C"),
        ("C", "ALA", "C", 31, 5.850, 21.400, 0.000, "C"),
        ("O", "ALA", "C", 31, 6.900, 21.800, 0.000, "O"),
        ("CB", "ALA", "C", 31, 4.700, 23.300, -1.000, "C"),
    ]
    lines = ["HEADER    complete complex fixture"]
    for serial, (atom, residue, chain, number, x, y, z, element) in enumerate(atoms, 1):
        lines.append(
            f"ATOM  {serial:5d} {atom:>4s} {residue} {chain}{number:4d}"
            f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2s}"
        )
        if (chain, number, atom) in (("A", 11, "OXT"), ("B", 21, "OXT")):
            lines.append("TER")
    return "\n".join([*lines, "TER", "END"]) + "\n"


def _covalent_head_to_tail_complex_pdb_text() -> str:
    lines = []
    for line in _complex_pdb_text().splitlines():
        if (
            line.startswith("ATOM")
            and line[21] == "C"
            and line[22:26].strip() == "31"
            and line[12:16].strip() == "C"
        ):
            line = line[:30] + f"{1.200:8.3f}{20.800:8.3f}{0.000:8.3f}" + line[54:]
        lines.append(line)
    return "\n".join(lines) + "\n"


def _identity_topology(a_names: tuple[str, ...]) -> tuple[app.Topology, object]:
    topology = app.Topology()
    positions = []
    elements = {"N": app.element.nitrogen, "C": app.element.carbon, "O": app.element.oxygen}
    for chain_id, residue_id, residue_name, names in (
        ("A", "10", "ALA", a_names),
        ("B", "20", "GLY", ("N", "CA", "C", "O", "OXT")),
        ("C", "30", "ALA", ("N", "CA", "C")),
    ):
        chain = topology.addChain(chain_id)
        residue = topology.addResidue(residue_name, chain, id=residue_id)
        for index, name in enumerate(names):
            element = elements["N" if name == "N" else "O" if name.startswith("O") else "C"]
            topology.addAtom(name, element, residue)
            positions.append(openmm.Vec3(index * 0.1, int(residue_id) * 0.01, 0.0))
    return topology, unit.Quantity(positions, unit.nanometer)


class _RecordingTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, tuple[float, float, float]], str, int]] = []

    def __call__(
        self,
        mapping: dict[str, tuple[float, float, float]],
        residue_name: str,
        residue_index: int,
        **_: object,
    ) -> CoordinateTransformResult:
        self.calls.append((dict(mapping), residue_name, residue_index))
        return CoordinateTransformResult(
            mapping=dict(mapping), residue_name=residue_name, residue_index=residue_index
        )


class _RecordingValidator:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str]] = []

    def __call__(
        self,
        mapping: dict[str, tuple[float, float, float]],
        residue_name: str,
        residue_index: int,
        *,
        expected: str,
        **kwargs: object,
    ) -> ChiralityReport:
        assert mapping
        stage = kwargs["stage"]
        assert isinstance(stage, str)
        self.calls.append((residue_index, expected, stage))
        return ChiralityReport(residue_index, residue_name, expected, expected, True)


@pytest.fixture
def source_pdb(tmp_path: Path) -> Path:
    path = tmp_path / "source.pdb"
    path.write_text(_complex_pdb_text())
    return path


def _cys_complex_pdb_text(second_sg: tuple[float, float, float] = (3.2, 21.6, -1.0)) -> str:
    lines: list[str] = []
    for line in _complex_pdb_text().splitlines():
        if line.startswith("ATOM") and line[21] == "C":
            line = line[:17] + "CYS" + line[20:]
        lines.append(line)
        if line.startswith("ATOM") and line[21] == "C" and line[12:16].strip() == "CB":
            number = line[22:26].strip()
            coordinates = (
                "  2.200  19.800  -1.000"
                if number == "30"
                else f"{second_sg[0]:8.3f}{second_sg[1]:8.3f}{second_sg[2]:8.3f}"
            )
            lines.append(
                f"ATOM  {len(lines) + 1:5d}  SG  CYS C{int(number):4d}"
                f"    {coordinates}  1.00  0.00           S"
            )
    return "\n".join(lines) + "\n"


def _artifact(source_pdb: Path, *, c_atom_count: int = 10) -> RosettaDecoyArtifact:
    digest = hashlib.sha256(source_pdb.read_bytes()).hexdigest()
    return RosettaDecoyArtifact(
        candidate_identity="candidate-1",
        parent_input_identity="parent-1",
        protocol_identity="protocol-1",
        config_identity="config-1",
        runtime_identity="runtime-1",
        input_pdb_identity=PDBIdentity("input", digest),
        output_pdb_identity=PDBIdentity("output", digest),
        chain_audits=(
            ChainAudit("A", "receptor-alpha", 2, 11),
            ChainAudit("B", "receptor-beta", 2, 9),
            ChainAudit("C", "binder", 2, c_atom_count),
        ),
        relax_score=RelaxScore(total_score=-12.5),
        status=ExecutionStatus.SUCCEEDED,
    )


def _config(source_pdb: Path, output_dir: Path, **overrides: Any) -> ComplexPrepConfig:
    values: dict[str, Any] = {
        "source_pdb": str(source_pdb),
        "source_decoy": _artifact(source_pdb),
        "output_dir": str(output_dir),
        "design_sequence": "LA",
    }
    values.update(overrides)
    return ComplexPrepConfig(**values)


def test_config_requires_successful_decoy_and_d_identities(
    source_pdb: Path, tmp_path: Path
) -> None:
    failed = _artifact(source_pdb)
    object.__setattr__(failed, "status", ExecutionStatus.FAILED)
    result = ComplexPrepRunner().run(_config(source_pdb, tmp_path / "out", source_decoy=failed))
    assert result.status is ExecutionStatus.FAILED
    assert "status must be SUCCEEDED" in result.error
    with pytest.raises(ValueError, match="coordinate_transformer_identity"):
        _config(
            source_pdb,
            tmp_path / "out",
            topology=PeptideTopologyDescriptor(
                d_substitutions=(SimpleNamespace(position=1, residue="LEU"),)
            ),
        )
    with pytest.raises(ValueError, match="must match the design sequence"):
        _config(
            source_pdb,
            tmp_path / "out-mismatch",
            topology=PeptideTopologyDescriptor(
                d_substitutions=(SimpleNamespace(position=2, residue="LEU"),)
            ),
            coordinate_transformer_identity="transform-v1",
            chirality_validator_identity="validator-v1",
        )
    with pytest.raises(ValueError, match="3-letter amino-acid code"):
        _config(
            source_pdb,
            tmp_path / "out-case",
            topology=PeptideTopologyDescriptor(
                d_substitutions=(SimpleNamespace(position=1, residue="leu"),)
            ),
            coordinate_transformer_identity="transform-v1",
            chirality_validator_identity="validator-v1",
        )
    with pytest.raises(ValueError, match="head_to_tail"):
        _config(
            source_pdb,
            tmp_path / "out-bool",
            topology=PeptideTopologyDescriptor(
                head_to_tail=SimpleNamespace(head=True, tail=2),
            ),
        )


def test_config_digest_normalizes_disulfide_pair_and_descriptor_order(
    source_pdb: Path, tmp_path: Path
) -> None:
    first = _config(
        source_pdb,
        tmp_path / "first",
        design_sequence="CCCC",
        topology=PeptideTopologyDescriptor(
            disulfides=(
                SimpleNamespace(first=3, second=4),
                SimpleNamespace(first=2, second=1),
            ),
        ),
    )
    second = _config(
        source_pdb,
        tmp_path / "second",
        design_sequence="CCCC",
        topology=PeptideTopologyDescriptor(
            disulfides=(
                SimpleNamespace(first=1, second=2),
                SimpleNamespace(first=4, second=3),
            ),
        ),
    )

    assert compute_complex_config_digest(first) == compute_complex_config_digest(second)


@pytest.mark.parametrize(
    "prepared_names",
    [
        ("N", "CA", "C", "O"),
        ("N", "CB", "C", "O", "OXT"),
        ("N", "CA", "C", "O", "OXT", "O2"),
    ],
    ids=("missing", "replaced", "extra"),
)
def test_receptor_heavy_atom_identity_mismatch_fails_closed(
    prepared_names: tuple[str, ...],
) -> None:
    source_topology, source_positions = _identity_topology(("N", "CA", "C", "O", "OXT"))
    prepared_topology, prepared_positions = _identity_topology(prepared_names)
    state = SimpleNamespace(
        source_topology=source_topology,
        source_positions=source_positions,
        topology=prepared_topology,
        positions=prepared_positions,
    )

    with pytest.raises(ValueError, match="receptor A/B heavy-atom"):
        complex_prep_runner._validate_preservation(
            cast("complex_prep_runner._StructureState", state)
        )


def test_source_symlink_is_rejected_as_a_preflight_failure(
    source_pdb: Path, tmp_path: Path
) -> None:
    source_link = tmp_path / "source-link.pdb"
    source_link.symlink_to(source_pdb)
    config = _config(source_link, tmp_path / "symlink-out", source_decoy=_artifact(source_link))

    result = ComplexPrepRunner().run(config)

    assert result.status is ExecutionStatus.FAILED
    assert "source_pdb read failed" in result.error
    assert not result.executed
    assert not result.provenance.executed


def test_empty_source_is_rejected_as_a_preflight_failure(tmp_path: Path) -> None:
    source_pdb = tmp_path / "empty-source.pdb"
    source_pdb.write_bytes(b"")
    config = _config(source_pdb, tmp_path / "empty-source-out")

    result = ComplexPrepRunner().run(config)

    assert result.status is ExecutionStatus.FAILED
    assert "source_pdb read failed" in result.error
    assert not result.executed


def test_source_is_replaced_after_snapshot_without_affecting_preparation(
    source_pdb: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(source_pdb, tmp_path / "snapshot-out")
    original_digest = hashlib.sha256(source_pdb.read_bytes()).hexdigest()
    real_load_source = complex_prep_runner._load_source

    def replace_caller_source(snapshot_path: str) -> tuple[object, object]:
        assert Path(snapshot_path) != source_pdb
        source_pdb.write_text("caller path replaced after snapshot")
        return real_load_source(snapshot_path)

    monkeypatch.setattr(complex_prep_runner, "_load_source", replace_caller_source)
    result = ComplexPrepRunner().run(config)

    assert result.success, result.error
    assert result.source_digest == original_digest


def test_existing_output_inspection_oserror_is_structured(
    source_pdb: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_inspection(*_: Any, **__: Any) -> None:
        raise OSError("inspection unavailable")

    monkeypatch.setattr(complex_prep_runner, "_inspect_existing", fail_inspection)
    result = ComplexPrepRunner().run(_config(source_pdb, tmp_path / "inspection-error"))

    assert result.status is ExecutionStatus.FAILED
    assert "inspection unavailable" in result.error
    assert not result.executed
    assert not result.provenance.executed


def test_existing_empty_output_is_not_overwritten(source_pdb: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "empty-output"
    output_dir.mkdir()

    result = ComplexPrepRunner().run(_config(source_pdb, output_dir))

    assert result.status is ExecutionStatus.INCOMPLETE
    assert "already exists" in result.error
    assert list(output_dir.iterdir()) == []


def test_full_complex_emits_exact_bundle_and_preserves_receptors(
    source_pdb: Path, tmp_path: Path
) -> None:
    result = ComplexPrepRunner().run(_config(source_pdb, tmp_path / "out"))

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.bundle is not None
    assert {path.name for path in Path(result.output_dir).iterdir()} == {
        "prepared.pdb",
        "prepared.top",
        "prepared.gro",
        "selection-map.json",
        "index.ndx",
        "manifest.json",
    }
    assert result.bundle.atom_count > 28
    assert result.bundle.net_charge == 0.0
    assert result.bundle.grompp_audit_status in {GROMPP_NOT_RUN, "passed"}
    assert [audit.chain_id for audit in result.bundle.chain_audits] == ["A", "B", "C"]
    assert [audit.prepared_name for audit in result.bundle.residue_audits[-2:]] == ["LEU", "ALA"]
    import parmed

    exported = parmed.load_file(
        result.bundle.prepared_top.path,
        xyz=result.bundle.prepared_gro.path,
    )
    assert len(exported.atoms) == result.bundle.atom_count
    assert sum(atom.charge for atom in exported.atoms) == pytest.approx(
        result.bundle.net_charge, abs=1e-6
    )

    source = PDBFixer(filename=str(source_pdb))
    prepared = PDBFixer(filename=result.bundle.prepared_pdb.path)
    source_positions = source.positions.value_in_unit(unit.nanometer)
    prepared_positions = prepared.positions.value_in_unit(unit.nanometer)
    source_atoms = {
        (atom.residue.chain.id, atom.residue.id, atom.name): atom
        for atom in source.topology.atoms()
    }
    prepared_atoms = {
        (atom.residue.chain.id, atom.residue.id, atom.name): atom
        for atom in prepared.topology.atoms()
    }
    for key, source_atom in source_atoms.items():
        if key[0] not in {"A", "B"} or source_atom.element.symbol == "H":
            continue
        prepared_atom = prepared_atoms[key]
        assert tuple(prepared_positions[prepared_atom.index]) == pytest.approx(
            tuple(source_positions[source_atom.index]), abs=1e-6
        )


def test_d_callbacks_are_scoped_to_c_and_use_local_indices(
    source_pdb: Path, tmp_path: Path
) -> None:
    transformer = _RecordingTransformer()
    validator = _RecordingValidator()
    config = _config(
        source_pdb,
        tmp_path / "out",
        topology=PeptideTopologyDescriptor(
            d_substitutions=(SimpleNamespace(position=1, residue="LEU"),)
        ),
        coordinate_transformer_identity="transform-v1",
        chirality_validator_identity="validator-v1",
    )

    result = ComplexPrepRunner().run(
        config, coordinate_transformer=transformer, chirality_validator=validator
    )

    assert result.success
    assert len(transformer.calls) == 1
    assert transformer.calls[0][2] == 0
    assert set(transformer.calls[0][0]) >= {"N", "CA", "C"}
    assert {index for index, _expected, _stage in validator.calls} == {0, 1}
    assert {stage for _index, _expected, stage in validator.calls} == {"post_h", "pre", "post"}


def test_c_local_head_to_tail_closure_has_global_indices(tmp_path: Path) -> None:
    source_pdb = tmp_path / "covalent-source.pdb"
    source_pdb.write_text(_covalent_head_to_tail_complex_pdb_text())
    config = _config(
        source_pdb,
        tmp_path / "out",
        topology=PeptideTopologyDescriptor(
            head_to_tail=SimpleNamespace(head=1, tail=2),
        ),
    )
    result = ComplexPrepRunner().run(config)

    assert result.success, result.error
    assert result.bundle is not None
    manifest = json.loads(Path(result.bundle.manifest.path).read_text())
    records = manifest["bond_records"]
    assert records[0]["bond_type"] == "head_to_tail"
    assert records[0]["residue1_index"] == 5
    assert records[0]["residue2_index"] == 4
    assert not any(
        line.startswith("ATOM") and line[21] == "C" and line[12:16].strip() == "OXT"
        for line in Path(result.bundle.prepared_pdb.path).read_text().splitlines()
    )


def test_far_head_to_tail_closure_fails_without_publishing(
    source_pdb: Path, tmp_path: Path
) -> None:
    config = _config(
        source_pdb,
        tmp_path / "far-cycle",
        topology=PeptideTopologyDescriptor(head_to_tail=SimpleNamespace(head=1, tail=2)),
    )

    result = ComplexPrepRunner().run(config)

    assert result.status is ExecutionStatus.FAILED
    assert "head_to_tail closure distance" in result.error
    assert not Path(config.output_dir).exists()


def test_selection_map_and_index_round_trip_exactly(source_pdb: Path, tmp_path: Path) -> None:
    result = ComplexPrepRunner().run(_config(source_pdb, tmp_path / "out"))
    assert result.bundle is not None
    data = json.loads(Path(result.bundle.selection_map.path).read_text())
    index_lines = Path(result.bundle.index.path).read_text().splitlines()
    parsed: dict[str, list[int]] = {}
    current = ""
    for line in index_lines:
        if line.startswith("[ "):
            current = line[2:-2]
            parsed[current] = []
        elif line.strip():
            parsed[current].extend(int(value) for value in line.split())
    assert tuple(parsed) == ("receptor_ab", "design_c", "dimer_ab")
    assert parsed == data["selections"]
    assert parsed["receptor_ab"] == parsed["dimer_ab"]
    assert data["index_bases"]["prepared_topology_atom"] == 1
    assert data["solvent_ion_boundaries"] == {"ions": "not_staged", "solvent": "not_staged"}
    assert data["preparation_digest"] == result.preparation_digest
    assert data["added_atoms"]


def test_c_local_disulfide_maps_descriptor_positions_to_global_topology(
    tmp_path: Path,
) -> None:
    source_pdb = tmp_path / "cys-source.pdb"
    source_pdb.write_text(_cys_complex_pdb_text())
    config = _config(
        source_pdb,
        tmp_path / "out",
        source_decoy=_artifact(source_pdb, c_atom_count=12),
        design_sequence="CC",
        topology=PeptideTopologyDescriptor(
            disulfides=(SimpleNamespace(first=1, second=2),),
        ),
    )

    result = ComplexPrepRunner().run(config)

    assert result.success, result.error
    assert result.bundle is not None
    manifest = json.loads(Path(result.bundle.manifest.path).read_text())
    record = manifest["bond_records"][0]
    assert record["bond_type"] == "disulfide"
    assert record["residue1_index"] == 4
    assert record["residue2_index"] == 5
    selection_map = json.loads(Path(result.bundle.selection_map.path).read_text())
    assert all(
        item["prepared_topology_residue_index"] in {5, 6}
        for item in selection_map["source_to_prepared_atoms"]
        if item["source"]["chain_id"] == "C"
    )


def test_far_disulfide_fails_without_publishing(tmp_path: Path) -> None:
    source_pdb = tmp_path / "far-cys-source.pdb"
    source_pdb.write_text(_cys_complex_pdb_text(second_sg=(3.5, 23.0, -1.0)))
    config = _config(
        source_pdb,
        tmp_path / "far-disulfide",
        source_decoy=_artifact(source_pdb, c_atom_count=12),
        design_sequence="CC",
        topology=PeptideTopologyDescriptor(
            disulfides=(SimpleNamespace(first=1, second=2),),
        ),
    )

    result = ComplexPrepRunner().run(config)

    assert result.status is ExecutionStatus.FAILED
    assert "disulfide closure distance" in result.error
    assert not Path(config.output_dir).exists()


def test_grompp_status_is_honest_about_gmx_presence(source_pdb: Path, tmp_path: Path) -> None:
    result = ComplexPrepRunner().run(_config(source_pdb, tmp_path / "out"))

    assert result.success
    expected = "passed" if shutil.which("gmx") else GROMPP_NOT_RUN
    assert result.grompp_audit_status == expected


def test_grompp_status_records_unavailable_binary(
    source_pdb: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    result = ComplexPrepRunner().run(_config(source_pdb, tmp_path / "out"))

    assert result.success
    assert result.grompp_audit_status == GROMPP_NOT_RUN


def test_cache_hit_and_partial_tampered_malformed_refusals(
    source_pdb: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    config = _config(source_pdb, output_dir)
    first = ComplexPrepRunner().run(config)
    assert first.success

    cached = ComplexPrepRunner().run(config)
    assert cached.status is ExecutionStatus.CACHED
    Path(first.bundle.prepared_gro.path).unlink()  # type: ignore[union-attr]
    incomplete = ComplexPrepRunner().run(config)
    assert incomplete.status is ExecutionStatus.INCOMPLETE

    first = ComplexPrepRunner().run(config)  # no overwrite: still incomplete
    assert first.status is ExecutionStatus.INCOMPLETE
    (output_dir / "prepared.gro").write_text("tampered")
    failed = ComplexPrepRunner().run(config)
    assert failed.status is ExecutionStatus.FAILED
    (output_dir / "manifest.json").write_text("{")
    malformed = ComplexPrepRunner().run(config)
    assert malformed.status is ExecutionStatus.MALFORMED


def _rewrite_map_and_manifest(output_dir: Path, mutate: Any) -> None:
    map_path = output_dir / SELECTION_MAP_FILENAME
    map_data = json.loads(map_path.read_text())
    mutate(map_data)
    map_path.write_text(json.dumps(map_data, indent=2, sort_keys=True) + "\n")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][SELECTION_MAP_FILENAME]["sha256"] = hashlib.sha256(
        map_path.read_bytes()
    ).hexdigest()
    manifest[MANIFEST_PAYLOAD_DIGEST_FIELD] = complex_prep_runner._manifest_payload_digest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_cache_rejects_scientific_manifest_metadata_tampering(
    source_pdb: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "metadata-tampered"
    config = _config(source_pdb, output_dir)
    first = ComplexPrepRunner().run(config)
    assert first.success
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["net_charge"] = 1.0
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    result = ComplexPrepRunner().run(config)

    assert result.status is ExecutionStatus.FAILED
    assert "scientific metadata digest" in result.error


def test_cache_rejects_malformed_manifest_metadata_types(source_pdb: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "metadata-type-tampered"
    config = _config(source_pdb, output_dir)
    first = ComplexPrepRunner().run(config)
    assert first.success
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["atom_count"] = "not-an-integer"
    manifest[MANIFEST_PAYLOAD_DIGEST_FIELD] = complex_prep_runner._manifest_payload_digest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    result = ComplexPrepRunner().run(config)

    assert result.status is ExecutionStatus.MALFORMED
    assert "atom count" in result.error


@pytest.mark.parametrize(
    ("collection", "field"),
    (
        ("source_to_prepared_atoms", "source_pdb_atom_index"),
        ("source_to_prepared_atoms", "prepared_pdb_atom_index"),
        ("source_to_prepared_atoms", "prepared_topology_atom_index"),
        ("source_to_prepared_residues", "source_pdb_residue_index"),
        ("source_to_prepared_residues", "prepared_pdb_residue_index"),
        ("source_to_prepared_residues", "prepared_topology_residue_index"),
    ),
)
def test_cache_rejects_stale_map_index_direction(
    source_pdb: Path, tmp_path: Path, collection: str, field: str
) -> None:
    output_dir = tmp_path / f"map-{field}"
    config = _config(source_pdb, output_dir)
    first = ComplexPrepRunner().run(config)
    assert first.success

    def mutate(data: dict[str, Any]) -> None:
        data[collection][0][field] += 1

    _rewrite_map_and_manifest(output_dir, mutate)
    result = ComplexPrepRunner().run(config)

    assert result.status is ExecutionStatus.FAILED
    assert "selection map mapped" in result.error


def test_cache_rejects_map_dropped_coverage_tampering(source_pdb: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "map-dropped"
    config = _config(source_pdb, output_dir)
    first = ComplexPrepRunner().run(config)
    assert first.success

    def mutate(data: dict[str, Any]) -> None:
        data["source_to_prepared_atoms"].pop()

    _rewrite_map_and_manifest(output_dir, mutate)
    result = ComplexPrepRunner().run(config)

    assert result.status is ExecutionStatus.FAILED
    assert "source atoms" in result.error


def test_cache_rejects_map_audit_tampering(source_pdb: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "map-audit"
    config = _config(source_pdb, output_dir)
    first = ComplexPrepRunner().run(config)
    assert first.success

    def mutate(data: dict[str, Any]) -> None:
        data["chain_audits"][0]["source_atom_count"] += 1

    _rewrite_map_and_manifest(output_dir, mutate)
    result = ComplexPrepRunner().run(config)

    assert result.status is ExecutionStatus.FAILED
    assert "chain audits" in result.error


def test_source_digest_mismatch_and_callback_failure_publish_nothing(
    source_pdb: Path, tmp_path: Path
) -> None:
    wrong = _artifact(source_pdb)
    object.__setattr__(wrong, "output_pdb_identity", PDBIdentity("wrong", "0" * 64))
    mismatch_config = _config(source_pdb, tmp_path / "mismatch", source_decoy=wrong)
    mismatch = ComplexPrepRunner().run(mismatch_config)
    assert mismatch.status is ExecutionStatus.FAILED
    assert not Path(mismatch_config.output_dir).exists()

    class _MovesBackbone:
        def __call__(
            self, mapping: dict[str, tuple[float, float, float]], *_: object, **__: object
        ) -> dict[str, tuple[float, float, float]]:
            changed = dict(mapping)
            x, y, z = changed["CA"]
            changed["CA"] = (x + 0.1, y, z)
            return changed

    config = _config(
        source_pdb,
        tmp_path / "callback-failure",
        topology=PeptideTopologyDescriptor(
            d_substitutions=(SimpleNamespace(position=1, residue="LEU"),)
        ),
        coordinate_transformer_identity="transform-v1",
        chirality_validator_identity="validator-v1",
    )
    failed = ComplexPrepRunner().run(
        config,
        coordinate_transformer=_MovesBackbone(),
        chirality_validator=_RecordingValidator(),
    )
    assert failed.status is ExecutionStatus.FAILED
    assert failed.executed
    assert failed.provenance.executed
    assert not Path(config.output_dir).exists()
    assert not list(Path(config.output_dir).parent.glob(".callback-failure.staging-*"))
