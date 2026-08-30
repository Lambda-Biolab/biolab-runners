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


def _receptor_disulfide_complex_pdb_text() -> str:
    lines: list[str] = []
    sg_serials: list[int] = []
    serial = 0
    for line in _complex_pdb_text().splitlines():
        if line.startswith("ATOM"):
            serial += 1
            if line[21] == "A":
                line = line[:17] + "CYS" + line[20:]
                if line[22:26].strip() == "11" and line[12:16].strip() == "CB":
                    line = line[:30] + f"{3.800:8.3f}{1.400:8.3f}{-1.000:8.3f}" + line[54:]
            line = f"{line[:6]}{serial:5d}{line[11:]}"
            lines.append(line)
            if line[21] == "A" and line[12:16].strip() == "CB":
                serial += 1
                residue_number = int(line[22:26])
                x, y = (2.300, -0.100) if residue_number == 10 else (4.050, 0.950)
                sg_serial = serial
                lines.append(
                    f"ATOM  {serial:5d}  SG  CYS A{residue_number:4d}"
                    f"    {x:8.3f}{y:8.3f}{-1.000:8.3f}  1.00  0.00           S"
                )
                sg_serials.append(sg_serial)
                serial += 1
                lines.append(
                    f"ATOM  {serial:5d}  HG  CYS A{residue_number:4d}"
                    f"    {x:8.3f}{y:8.3f}{0.300:8.3f}  1.00  0.00           H"
                )
        else:
            lines.append(line)
    lines.insert(-1, f"CONECT{sg_serials[0]:5d}{sg_serials[1]:5d}")
    lines.insert(-1, f"CONECT{sg_serials[1]:5d}{sg_serials[0]:5d}")
    return "\n".join(lines) + "\n"


def _complete_design_chain(text: str) -> str:
    lines = text.splitlines()
    terminal_index = max(index for index, line in enumerate(lines) if line == "TER")
    lines.insert(
        terminal_index,
        "ATOM     35  OXT ALA C  31       6.800  21.500   0.000  1.00  0.00           O",
    )
    return "\n".join(lines) + "\n"


def _reduced_receptor_disulfide_complex_pdb_text() -> str:
    return _complete_design_chain(
        "\n".join(
            line
            for line in _receptor_disulfide_complex_pdb_text().splitlines()
            if not line.startswith("CONECT")
        )
    )


def _mixed_receptor_disulfide_complex_pdb_text() -> str:
    lines: list[str] = []
    serial = 36
    for line in _complete_design_chain(_receptor_disulfide_complex_pdb_text()).splitlines():
        if line.startswith("ATOM") and line[21] == "B":
            line = line[:17] + "CYS" + line[20:]
        lines.append(line)
        if line.startswith("ATOM") and line[21] == "B" and line[12:16].strip() == "O":
            residue_number = int(line[22:26])
            x, y = (1.400, 9.000) if residue_number == 20 else (3.800, 11.400)
            sg_x, sg_y = (2.300, 9.900) if residue_number == 20 else (4.050, 10.950)
            lines.extend(
                [
                    f"ATOM  {serial:5d}  CB  CYS B{residue_number:4d}"
                    f"    {x:8.3f}{y:8.3f}{-1.000:8.3f}  1.00  0.00           C",
                    f"ATOM  {serial + 1:5d}  SG  CYS B{residue_number:4d}"
                    f"    {sg_x:8.3f}{sg_y:8.3f}{-1.000:8.3f}  1.00  0.00           S",
                    f"ATOM  {serial + 2:5d}  HG  CYS B{residue_number:4d}"
                    f"    {sg_x:8.3f}{sg_y:8.3f}{0.300:8.3f}  1.00  0.00           H",
                ]
            )
            serial += 3
    return "\n".join(lines) + "\n"


def _receptor_sg_pairs(topology: app.Topology) -> set[frozenset[tuple[str, str]]]:
    return {
        frozenset(
            (
                (first.residue.chain.id, first.residue.id),
                (second.residue.chain.id, second.residue.id),
            )
        )
        for first, second in topology.bonds()
        if first.name == second.name == "SG"
        and first.residue.chain.id in {"A", "B"}
        and second.residue.chain.id in {"A", "B"}
    }


def _artifact(
    source_pdb: Path, *, a_atom_count: int = 11, b_atom_count: int = 9, c_atom_count: int = 10
) -> RosettaDecoyArtifact:
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
            ChainAudit("A", "receptor-alpha", 2, a_atom_count),
            ChainAudit("B", "receptor-beta", 2, b_atom_count),
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
    with pytest.raises(ValueError, match="d_coordinate_input_mode"):
        _config(
            source_pdb,
            tmp_path / "out-mode",
            d_coordinate_input_mode="unknown",
        )
    with pytest.raises(ValueError, match="requires D substitutions"):
        _config(
            source_pdb,
            tmp_path / "out-prepared-without-d",
            d_coordinate_input_mode="prepared_d",
        )
    with pytest.raises(ValueError, match="chirality_validator_identity"):
        _config(
            source_pdb,
            tmp_path / "out-prepared-without-validator",
            topology=PeptideTopologyDescriptor(
                d_substitutions=(SimpleNamespace(position=1, residue="LEU"),)
            ),
            d_coordinate_input_mode="prepared_d",
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


def test_config_digest_binds_d_coordinate_input_mode(source_pdb: Path, tmp_path: Path) -> None:
    topology = PeptideTopologyDescriptor(
        d_substitutions=(SimpleNamespace(position=1, residue="LEU"),)
    )
    common = {
        "topology": topology,
        "coordinate_transformer_identity": "transform-v1",
        "chirality_validator_identity": "validator-v1",
    }
    canonical = _config(source_pdb, tmp_path / "canonical", **common)
    prepared = _config(
        source_pdb,
        tmp_path / "prepared",
        d_coordinate_input_mode="prepared_d",
        **common,
    )

    assert compute_complex_config_digest(canonical) != compute_complex_config_digest(prepared)


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


def test_full_complex_normalizes_preexisting_receptor_disulfide(
    tmp_path: Path,
) -> None:
    source_pdb = tmp_path / "receptor-disulfide.pdb"
    source_pdb.write_text(_receptor_disulfide_complex_pdb_text())
    source = app.PDBFile(str(source_pdb))
    assert any(first.name == second.name == "SG" for first, second in source.topology.bonds())
    config = _config(
        source_pdb,
        tmp_path / "out",
        source_decoy=_artifact(source_pdb, a_atom_count=15),
    )
    result = ComplexPrepRunner().run(config)

    assert result.success, result.error
    assert result.bundle is not None
    prepared_lines = Path(result.bundle.prepared_pdb.path).read_text().splitlines()
    assert {
        line[17:20]
        for line in prepared_lines
        if line.startswith(("ATOM", "HETATM")) and line[21] == "A"
    } == {"CYX"}
    prepared = app.PDBFile(result.bundle.prepared_pdb.path)
    receptor_cysteines = [
        residue for residue in prepared.topology.residues() if residue.chain.id == "A"
    ]
    assert len(receptor_cysteines) == 2
    assert all(
        "HG" not in {atom.name for atom in residue.atoms()} for residue in receptor_cysteines
    )
    prepared_text = Path(result.bundle.prepared_pdb.path).read_text()
    assert any(line.startswith("CONECT") for line in prepared_text.splitlines())
    source_positions = source.positions.value_in_unit(unit.nanometer)
    prepared_positions = prepared.positions.value_in_unit(unit.nanometer)
    source_sg = {
        atom.residue.id: source_positions[atom.index]
        for atom in source.topology.atoms()
        if atom.residue.chain.id == "A" and atom.name == "SG"
    }
    prepared_sg = {
        atom.residue.id: prepared_positions[atom.index]
        for atom in prepared.topology.atoms()
        if atom.residue.chain.id == "A" and atom.name == "SG"
    }
    assert prepared_sg.keys() == source_sg.keys()
    for residue_id in source_sg:
        assert tuple(prepared_sg[residue_id]) == pytest.approx(
            tuple(source_sg[residue_id]), abs=1e-6
        )
    manifest = json.loads(Path(result.bundle.manifest.path).read_text())
    assert manifest["bond_records"] == []

    cached = ComplexPrepRunner().run(config)

    assert cached.status is ExecutionStatus.CACHED, cached.error


def test_close_reduced_receptor_cysteines_remain_reduced_without_conect(
    tmp_path: Path,
) -> None:
    source_pdb = tmp_path / "reduced-receptor-cys.pdb"
    source_pdb.write_text(_reduced_receptor_disulfide_complex_pdb_text())
    source = app.PDBFile(str(source_pdb))
    assert _receptor_sg_pairs(source.topology) == set()

    config = _config(
        source_pdb,
        tmp_path / "out",
        source_decoy=_artifact(source_pdb, a_atom_count=15, c_atom_count=11),
    )
    result = ComplexPrepRunner().run(config)

    assert result.success, result.error
    assert result.bundle is not None
    prepared = app.PDBFile(result.bundle.prepared_pdb.path)
    assert _receptor_sg_pairs(prepared.topology) == set()
    receptor_cysteines = [
        residue for residue in prepared.topology.residues() if residue.chain.id == "A"
    ]
    assert {residue.name for residue in receptor_cysteines} == {"CYS"}
    assert all("HG" in {atom.name for atom in residue.atoms()} for residue in receptor_cysteines)


def test_mixed_receptor_disulfides_preserve_only_explicit_source_pair(
    tmp_path: Path,
) -> None:
    source_pdb = tmp_path / "mixed-receptor-cys.pdb"
    source_pdb.write_text(_mixed_receptor_disulfide_complex_pdb_text())
    source = app.PDBFile(str(source_pdb))
    source_pairs = _receptor_sg_pairs(source.topology)
    assert source_pairs == {frozenset({("A", "10"), ("A", "11")})}

    config = _config(
        source_pdb,
        tmp_path / "out",
        source_decoy=_artifact(source_pdb, a_atom_count=15, b_atom_count=15, c_atom_count=11),
    )
    result = ComplexPrepRunner().run(config)

    assert result.success, result.error
    assert result.bundle is not None
    prepared = app.PDBFile(result.bundle.prepared_pdb.path)
    assert _receptor_sg_pairs(prepared.topology) == source_pairs
    b_cysteines = [residue for residue in prepared.topology.residues() if residue.chain.id == "B"]
    assert {residue.name for residue in b_cysteines} == {"CYS"}
    assert all("HG" in {atom.name for atom in residue.atoms()} for residue in b_cysteines)


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


def test_prepared_d_input_skips_transform_and_validates_source_as_d(
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
        d_coordinate_input_mode="prepared_d",
        chirality_validator_identity="validator-v1",
    )

    result = ComplexPrepRunner().run(
        config, coordinate_transformer=transformer, chirality_validator=validator
    )

    assert result.success, result.error
    assert transformer.calls == []
    assert (0, "D", "source") in validator.calls
    assert {stage for _index, _expected, stage in validator.calls} == {"source", "pre", "post"}
    assert result.bundle is not None
    manifest_path = Path(result.bundle.manifest.path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["d_coordinate_input_mode"] == "prepared_d"

    manifest["d_coordinate_input_mode"] = "canonical_l"
    manifest[MANIFEST_PAYLOAD_DIGEST_FIELD] = complex_prep_runner._manifest_payload_digest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    rejected = ComplexPrepRunner().run(config)
    assert rejected.status is ExecutionStatus.FAILED
    assert "D-coordinate input mode" in rejected.error


def test_prepared_d_input_requires_runtime_chirality_validator(
    source_pdb: Path, tmp_path: Path
) -> None:
    config = _config(
        source_pdb,
        tmp_path / "out",
        topology=PeptideTopologyDescriptor(
            d_substitutions=(SimpleNamespace(position=1, residue="LEU"),)
        ),
        d_coordinate_input_mode="prepared_d",
        chirality_validator_identity="validator-v1",
    )

    result = ComplexPrepRunner().run(config)

    assert result.status is ExecutionStatus.FAILED
    assert "chirality validator callback" in result.error
    assert not Path(config.output_dir).exists()


def test_prepared_d_input_refuses_non_d_source_chirality(source_pdb: Path, tmp_path: Path) -> None:
    class _RejectingValidator(_RecordingValidator):
        def __call__(
            self,
            mapping: dict[str, tuple[float, float, float]],
            residue_name: str,
            residue_index: int,
            *,
            expected: str,
            **kwargs: object,
        ) -> ChiralityReport:
            super().__call__(
                mapping,
                residue_name,
                residue_index,
                expected=expected,
                **kwargs,
            )
            observed = "L" if expected == "D" else expected
            return ChiralityReport(
                residue_index,
                residue_name,
                expected,
                observed,
                observed == expected,
            )

    config = _config(
        source_pdb,
        tmp_path / "out-invalid-d",
        topology=PeptideTopologyDescriptor(
            d_substitutions=(SimpleNamespace(position=1, residue="LEU"),)
        ),
        d_coordinate_input_mode="prepared_d",
        chirality_validator_identity="validator-v1",
    )

    result = ComplexPrepRunner().run(config, chirality_validator=_RejectingValidator())

    assert result.status is ExecutionStatus.FAILED
    assert "chirality validation failed" in result.error
    assert not Path(config.output_dir).exists()


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
    prepared = app.PDBFile(result.bundle.prepared_pdb.path)
    atoms = list(prepared.topology.atoms())
    closure_pair = {records[0]["atom1_index"], records[0]["atom2_index"]}
    assert closure_pair in [
        {first.index, second.index} for first, second in prepared.topology.bonds()
    ]
    assert {atoms[index].name for index in closure_pair} == {"C", "N"}
    cached = ComplexPrepRunner().run(config)
    assert cached.status is ExecutionStatus.CACHED, cached.error


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
    receptor_a_count = data["chain_audits"][0]["prepared_atom_count"]
    receptor_b_count = data["chain_audits"][1]["prepared_atom_count"]
    dimer_count = receptor_a_count + receptor_b_count
    assert receptor_a_count != receptor_b_count
    assert data["interface_mapping"] == {
        "receptor_a": list(range(1, receptor_a_count + 1)),
        "receptor_b": list(range(receptor_a_count + 1, dimer_count + 1)),
        "dimer_ab": list(range(1, dimer_count + 1)),
        "design_c": list(range(dimer_count + 1, result.bundle.atom_count + 1)),
    }
    assert set(data["interface_mapping"]["dimer_ab"]).isdisjoint(
        data["interface_mapping"]["design_c"]
    )
    assert data["index_bases"]["prepared_topology_atom"] == 1
    assert data["solvent_ion_boundaries"] == {"ions": "not_staged", "solvent": "not_staged"}
    assert data["preparation_digest"] == result.preparation_digest
    assert data["added_atoms"]


def test_cache_rejects_changed_interface_mapping(source_pdb: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "interface-map-tampered"
    config = _config(source_pdb, output_dir)
    first = ComplexPrepRunner().run(config)
    assert first.success

    def mutate(data: dict[str, Any]) -> None:
        data["interface_mapping"]["receptor_a"].pop()

    _rewrite_map_and_manifest(output_dir, mutate)

    result = ComplexPrepRunner().run(config)

    assert result.status is ExecutionStatus.FAILED
    assert "interface mapping" in result.error


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
    cached = ComplexPrepRunner().run(config)
    assert cached.status is ExecutionStatus.CACHED, cached.error


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


def _refresh_bound_artifacts(output_dir: Path, *filenames: str) -> None:
    map_path = output_dir / SELECTION_MAP_FILENAME
    map_data = json.loads(map_path.read_text())
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for filename in filenames:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        manifest["artifacts"][filename]["sha256"] = digest
        if filename == "index.ndx":
            map_data["index_sha256"] = digest
        else:
            map_data["prepared_artifact_sha256"][filename] = digest
    map_path.write_text(json.dumps(map_data, indent=2, sort_keys=True) + "\n")
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


def test_cache_rejects_index_base_and_role_tampering(source_pdb: Path, tmp_path: Path) -> None:
    base_output = tmp_path / "map-base"
    base_config = _config(source_pdb, base_output)
    assert ComplexPrepRunner().run(base_config).success

    def change_base(data: dict[str, Any]) -> None:
        data["index_bases"]["prepared_topology_atom"] = 0

    _rewrite_map_and_manifest(base_output, change_base)
    base_result = ComplexPrepRunner().run(base_config)
    assert base_result.status is ExecutionStatus.MALFORMED
    assert "index bases" in base_result.error

    role_output = tmp_path / "map-role"
    role_config = _config(source_pdb, role_output)
    assert ComplexPrepRunner().run(role_config).success
    map_path = role_output / SELECTION_MAP_FILENAME
    map_data = json.loads(map_path.read_text())
    moved = map_data["selections"]["receptor_ab"].pop(0)
    map_data["selections"]["dimer_ab"].pop(0)
    map_data["selections"]["design_c"].append(moved)
    map_data["selections"]["design_c"].sort()
    map_path.write_text(json.dumps(map_data, indent=2, sort_keys=True) + "\n")
    (role_output / "index.ndx").write_text(
        complex_prep_runner._render_index(map_data["selections"])
    )
    _refresh_bound_artifacts(role_output, "index.ndx")

    role_result = ComplexPrepRunner().run(role_config)
    assert role_result.status is ExecutionStatus.FAILED
    assert "topology roles" in role_result.error


def test_cache_revalidates_prepared_structure_and_topology(
    source_pdb: Path, tmp_path: Path
) -> None:
    pdb_output = tmp_path / "pdb-science"
    pdb_config = _config(source_pdb, pdb_output)
    assert ComplexPrepRunner().run(pdb_config).success
    pdb_path = pdb_output / "prepared.pdb"
    lines = pdb_path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("ATOM") and line[21] == "A" and line[12:16].strip() == "CA":
            lines[index] = line[:30] + f"{float(line[30:38]) + 1.0:8.3f}" + line[38:]
            break
    pdb_path.write_text("\n".join(lines) + "\n")
    _refresh_bound_artifacts(pdb_output, "prepared.pdb")

    pdb_result = ComplexPrepRunner().run(pdb_config)
    assert pdb_result.status is ExecutionStatus.FAILED
    assert "receptor A/B heavy-atom" in pdb_result.error

    top_output = tmp_path / "top-science"
    top_config = _config(source_pdb, top_output)
    assert ComplexPrepRunner().run(top_config).success
    top_path = top_output / "prepared.top"
    top_lines = top_path.read_text().splitlines()
    in_atoms = False
    for index, line in enumerate(top_lines):
        if line.strip() == "[ atoms ]":
            in_atoms = True
        elif in_atoms and line.strip() and not line.lstrip().startswith(";"):
            fields = line.split()
            fields[6] = f"{float(fields[6]) + 0.1:.8f}"
            top_lines[index] = " ".join(fields)
            break
    top_path.write_text("\n".join(top_lines) + "\n")
    _refresh_bound_artifacts(top_output, "prepared.top")

    top_result = ComplexPrepRunner().run(top_config)
    assert top_result.status is ExecutionStatus.FAILED
    assert "GROMACS parity" in top_result.error


def test_atomic_publish_never_replaces_an_occupied_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    monkeypatch.setattr(complex_prep_runner.os.path, "lexists", lambda _path: False)

    with pytest.raises(ValueError, match="became occupied"):
        complex_prep_runner._publish_staging(staging, output)

    assert staging.is_dir()
    assert output.is_dir()


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
