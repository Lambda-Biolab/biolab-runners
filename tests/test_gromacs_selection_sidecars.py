"""Behavioral tests for BR-G1 GROMACS selection sidecars."""

from __future__ import annotations

import hashlib
import json
import signal
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from biolab_runners.gromacs.config import GromacsProtocolConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gro_atom(
    residue_number: int,
    residue_name: str,
    atom_name: str,
    atom_number: int,
) -> str:
    return (
        f"{residue_number:5d}{residue_name:<5}{atom_name:>5}{atom_number:5d}"
        "   0.000   0.000   0.000"
    )


def _write_gro(path: Path, atoms: list[tuple[int, str, str, int]]) -> None:
    lines = ["prepared", f"{len(atoms):5d}"]
    lines.extend(_gro_atom(*atom) for atom in atoms)
    lines.append("   1.00000   1.00000   1.00000")
    path.write_text("\n".join(lines) + "\n")


def _canonical_digest(payload: dict[str, Any], *, excluded: str | None = None) -> str:
    content = {key: value for key, value in payload.items() if key != excluded}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _write_bundle(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True)
    prepared_pdb = tmp_path / "prepared.pdb"
    prepared_top = tmp_path / "prepared.top"
    prepared_gro = tmp_path / "prepared.gro"
    index = tmp_path / "index.ndx"
    selection_map = tmp_path / "selection-map.json"
    manifest = tmp_path / "manifest.json"

    prepared_pdb.write_text(
        "ATOM      1  N   ALA A   1\nATOM      2  CA  GLY B   1\nATOM      3  CA  GLY C   1\nEND\n"
    )
    prepared_top.write_text("; prepared topology\n")
    _write_gro(
        prepared_gro,
        [(1, "ALA", "N", 1), (1, "GLY", "CA", 2), (1, "GLY", "CA", 3)],
    )
    index.write_text("[ receptor_ab ]\n1 2\n[ design_c ]\n3\n[ dimer_ab ]\n1 2\n")

    index_digest = _sha256(index)
    prepared_digests = {
        "prepared.pdb": _sha256(prepared_pdb),
        "prepared.top": _sha256(prepared_top),
        "prepared.gro": _sha256(prepared_gro),
    }
    map_payload: dict[str, Any] = {
        "schema_version": 1,
        "preparation_digest": "a" * 64,
        "source_pdb_sha256": "b" * 64,
        "prepared_artifact_sha256": prepared_digests,
        "index_sha256": index_digest,
        "index_bases": {
            "source_pdb_atom": 1,
            "source_pdb_residue": 1,
            "prepared_pdb_atom": 1,
            "prepared_pdb_residue": 1,
            "prepared_topology_atom": 1,
            "prepared_topology_residue": 1,
        },
        "source_to_prepared_atoms": [
            {
                "source": {
                    "chain_id": "A",
                    "residue_number": 1,
                    "insertion_code": "",
                    "atom_name": "N",
                    "element": "N",
                },
                "prepared": {
                    "chain_id": "A",
                    "residue_number": 1,
                    "insertion_code": "",
                    "atom_name": "N",
                    "element": "N",
                },
                "source_pdb_atom_index": 1,
                "source_pdb_residue_index": 1,
                "prepared_pdb_atom_index": 1,
                "prepared_pdb_residue_index": 1,
                "prepared_topology_atom_index": 1,
                "prepared_topology_residue_index": 1,
            },
            {
                "source": {
                    "chain_id": "B",
                    "residue_number": 1,
                    "insertion_code": "",
                    "atom_name": "CA",
                    "element": "C",
                },
                "prepared": {
                    "chain_id": "B",
                    "residue_number": 1,
                    "insertion_code": "",
                    "atom_name": "CA",
                    "element": "C",
                },
                "source_pdb_atom_index": 2,
                "source_pdb_residue_index": 1,
                "prepared_pdb_atom_index": 2,
                "prepared_pdb_residue_index": 1,
                "prepared_topology_atom_index": 2,
                "prepared_topology_residue_index": 1,
            },
            {
                "source": {
                    "chain_id": "C",
                    "residue_number": 1,
                    "insertion_code": "",
                    "atom_name": "CA",
                    "element": "C",
                },
                "prepared": {
                    "chain_id": "C",
                    "residue_number": 1,
                    "insertion_code": "",
                    "atom_name": "CA",
                    "element": "C",
                },
                "source_pdb_atom_index": 3,
                "source_pdb_residue_index": 1,
                "prepared_pdb_atom_index": 3,
                "prepared_pdb_residue_index": 1,
                "prepared_topology_atom_index": 3,
                "prepared_topology_residue_index": 1,
            },
        ],
        "added_atoms": [],
        "dropped_atoms": [],
        "source_to_prepared_residues": [],
        "added_residues": [],
        "dropped_residues": [],
        "selections": {"receptor_ab": [1, 2], "design_c": [3], "dimer_ab": [1, 2]},
        "interface_mapping": {
            "receptor_a": [1],
            "receptor_b": [2],
            "dimer_ab": [1, 2],
            "design_c": [3],
        },
        "chain_audits": [
            {
                "chain_id": "A",
                "role": "receptor",
                "source_residue_count": 1,
                "prepared_residue_count": 1,
                "source_atom_count": 1,
                "prepared_atom_count": 1,
            },
            {
                "chain_id": "B",
                "role": "receptor",
                "source_residue_count": 1,
                "prepared_residue_count": 1,
                "source_atom_count": 1,
                "prepared_atom_count": 1,
            },
            {
                "chain_id": "C",
                "role": "design",
                "source_residue_count": 1,
                "prepared_residue_count": 1,
                "source_atom_count": 1,
                "prepared_atom_count": 1,
            },
        ],
        "residue_audits": [],
        "solvent_ion_boundaries": {"solvent": "not_staged", "ions": "not_staged"},
    }
    selection_map.write_text(json.dumps(map_payload, indent=2, sort_keys=True))

    artifacts = {
        **prepared_digests,
        "selection-map.json": _sha256(selection_map),
        "index.ndx": index_digest,
    }
    manifest_payload: dict[str, Any] = {
        "schema_version": 1,
        "runner": "complex_prep",
        "preparation_digest": map_payload["preparation_digest"],
        "source_pdb_sha256": map_payload["source_pdb_sha256"],
        "chain_roles": {"receptor": ["A", "B"], "design": "C"},
        "chain_audits": map_payload["chain_audits"],
        "residue_audits": map_payload["residue_audits"],
        "artifacts": {
            name: {"path": str(tmp_path / name), "sha256": digest}
            for name, digest in artifacts.items()
        },
        "atom_count": 3,
    }
    manifest_payload["scientific_metadata_sha256"] = _canonical_digest(
        manifest_payload, excluded="scientific_metadata_sha256"
    )
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True))
    return {
        "pdb": prepared_pdb,
        "top": prepared_top,
        "gro": prepared_gro,
        "map": selection_map,
        "index": index,
        "manifest": manifest,
    }


def _rebind_bundle(bundle: dict[str, Path]) -> None:
    selection_map = json.loads(bundle["map"].read_text())
    selection_map["prepared_artifact_sha256"] = {
        "prepared.pdb": _sha256(bundle["pdb"]),
        "prepared.top": _sha256(bundle["top"]),
        "prepared.gro": _sha256(bundle["gro"]),
    }
    selection_map["index_sha256"] = _sha256(bundle["index"])
    bundle["map"].write_text(json.dumps(selection_map, indent=2, sort_keys=True))

    manifest = json.loads(bundle["manifest"].read_text())
    for name, key in (
        ("prepared.pdb", "pdb"),
        ("prepared.top", "top"),
        ("prepared.gro", "gro"),
        ("selection-map.json", "map"),
        ("index.ndx", "index"),
    ):
        manifest["artifacts"][name]["sha256"] = _sha256(bundle[key])
    manifest["atom_count"] = int(bundle["gro"].read_text().splitlines()[1])
    manifest["chain_audits"] = selection_map["chain_audits"]
    manifest["residue_audits"] = selection_map["residue_audits"]
    manifest["scientific_metadata_sha256"] = _canonical_digest(
        manifest, excluded="scientific_metadata_sha256"
    )
    bundle["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _strict_config(tmp_path: Path, bundle: dict[str, Path]) -> GromacsProtocolConfig:
    return GromacsProtocolConfig(
        name="selection-sidecars",
        output_root=str(tmp_path / "output"),
        prebuilt_topology=str(bundle["top"]),
        prebuilt_coordinates=str(bundle["gro"]),
        prebuilt_prepared_pdb=str(bundle["pdb"]),
        prebuilt_selection_map=str(bundle["map"]),
        prebuilt_index=str(bundle["index"]),
        prebuilt_bundle_manifest=str(bundle["manifest"]),
    )


def _protocol_invoker(
    commands: list[list[str]],
    observed_states: list[str],
) -> Callable[[list[str], Path, int], int]:
    solvent_atoms = [
        (1, "ALA", "N", 1),
        (1, "GLY", "CA", 2),
        (1, "GLY", "CA", 3),
        (2, "SOL", "OW", 4),
        (2, "SOL", "HW1", 5),
        (2, "SOL", "HW2", 6),
    ]
    ionized_atoms = [
        (1, "ALA", "N", 1),
        (1, "GLY", "CA", 2),
        (1, "GLY", "CA", 3),
        (2, "SOL", "OW", 4),
        (3, "NA", "NA", 5),
        (4, "CL", "CL", 6),
    ]

    def _invoke(command: list[str], work_dir: Path, _timeout: int) -> int:
        commands.append(command)
        subcommand = command[1]
        if subcommand == "editconf":
            _write_gro(
                work_dir / "boxed.gro",
                [(1, "ALA", "N", 1), (1, "GLY", "CA", 2), (1, "GLY", "CA", 3)],
            )
        elif subcommand == "solvate":
            _write_gro(work_dir / "solvated.gro", solvent_atoms)
            (work_dir / "topol.top").write_text("; topology with solvent\n")
        elif subcommand == "grompp":
            output = command[command.index("-o") + 1]
            if output in {"ions.tpr", "min.tpr"}:
                observed_states.append(
                    json.loads((work_dir / "selection-map.json").read_text())["gromacs_state"][
                        "stage"
                    ]
                )
            (work_dir / output).write_text(f"compiled {output}\n")
        elif subcommand == "genion":
            _write_gro(work_dir / "ions.gro", ionized_atoms)
            (work_dir / "topol.top").write_text("; topology with solvent and ions\n")
        elif subcommand == "mdrun":
            prefix = command[command.index("-deffnm") + 1]
            _write_gro(work_dir / f"{prefix}.gro", ionized_atoms)
            (work_dir / f"{prefix}.cpt").write_bytes(f"{prefix}-checkpoint".encode())
            (work_dir / f"{prefix}.edr").write_text("energy\n")
            (work_dir / f"{prefix}.log").write_text("log\n")
        return 0

    return _invoke


def test_config_preserves_legacy_prebuilt_pair_without_sidecars(tmp_path: Path) -> None:
    config = GromacsProtocolConfig(
        name="legacy",
        output_root=str(tmp_path),
        prebuilt_topology="prepared.top",
        prebuilt_coordinates="prepared.gro",
    )

    assert config.prebuilt_selection_map == ""


@pytest.mark.parametrize(
    "field",
    [
        "prebuilt_prepared_pdb",
        "prebuilt_selection_map",
        "prebuilt_index",
        "prebuilt_bundle_manifest",
    ],
)
def test_config_rejects_partial_selection_sidecar_bundle(tmp_path: Path, field: str) -> None:
    values: dict[str, Any] = {
        "prebuilt_prepared_pdb": "prepared.pdb",
        "prebuilt_selection_map": "selection-map.json",
        "prebuilt_index": "index.ndx",
        "prebuilt_bundle_manifest": "manifest.json",
    }
    values[field] = ""

    with pytest.raises(ValueError, match="selection sidecar bundle"):
        GromacsProtocolConfig(
            name="partial",
            output_root=str(tmp_path),
            prebuilt_topology="prepared.top",
            prebuilt_coordinates="prepared.gro",
            **values,
        )


def test_config_rejects_sidecars_without_prebuilt_topology_pair(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires prebuilt_topology"):
        GromacsProtocolConfig(
            name="partial",
            input_pdb="input.pdb",
            output_root=str(tmp_path),
            prebuilt_prepared_pdb="prepared.pdb",
            prebuilt_selection_map="selection-map.json",
            prebuilt_index="index.ndx",
            prebuilt_bundle_manifest="manifest.json",
        )


def test_stage_selection_sidecars_copies_and_binds_complete_bundle(tmp_path: Path) -> None:
    from biolab_runners.gromacs.selection_sidecars import stage_selection_sidecars

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    work_dir = tmp_path / "work"

    identity = stage_selection_sidecars(config, work_dir)

    assert (work_dir / "prepared.pdb").read_bytes() == bundle["pdb"].read_bytes()
    assert (work_dir / "selection-map.json").read_bytes() == bundle["map"].read_bytes()
    assert (work_dir / "index.ndx").read_bytes() == bundle["index"].read_bytes()
    assert (work_dir / "complex-prep-manifest.json").read_bytes() == bundle["manifest"].read_bytes()
    assert identity["preparation_digest"] == "a" * 64
    assert identity["selection_map_sha256"] == _sha256(bundle["map"])
    assert identity["index_sha256"] == _sha256(bundle["index"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda mapping: mapping.pop("receptor_b"), "exactly receptor_a"),
        (lambda mapping: mapping["receptor_a"].__setitem__(0, 0), "out-of-range"),
        (lambda mapping: mapping["receptor_b"].__setitem__(0, 1), "overlap"),
        (
            lambda mapping: mapping.update({"receptor_a": [2], "receptor_b": [1]}),
            "explicit chain A",
        ),
        (lambda mapping: mapping["dimer_ab"].pop(), r"sorted A\+B union"),
        (lambda mapping: mapping["design_c"].__setitem__(0, 1), "overlaps design C"),
    ],
    ids=("missing", "zero-based", "overlap", "wrong-chain", "wrong-union", "wrong-design"),
)
def test_stage_selection_sidecars_rejects_invalid_interface_mapping(
    tmp_path: Path,
    mutation: Callable[[dict[str, list[int]]], object],
    message: str,
) -> None:
    from biolab_runners.gromacs.selection_sidecars import stage_selection_sidecars

    bundle = _write_bundle(tmp_path / "bundle")
    selection_map = json.loads(bundle["map"].read_text())
    mutation(selection_map["interface_mapping"])
    bundle["map"].write_text(json.dumps(selection_map, indent=2, sort_keys=True))
    _rebind_bundle(bundle)

    with pytest.raises(ValueError, match=message):
        stage_selection_sidecars(_strict_config(tmp_path, bundle), tmp_path / "work")


def test_stage_selection_sidecars_accepts_preparation_added_atoms(tmp_path: Path) -> None:
    from biolab_runners.gromacs.selection_sidecars import (
        refresh_selection_sidecars,
        stage_selection_sidecars,
    )

    bundle = _write_bundle(tmp_path / "bundle")
    _write_gro(
        bundle["gro"],
        [
            (1, "ALA", "N", 1),
            (1, "GLY", "CA", 2),
            (1, "GLY", "CA", 3),
            (1, "GLY", "H", 4),
        ],
    )
    bundle["pdb"].write_text(
        "ATOM      1  N   ALA A   1\n"
        "ATOM      2  CA  GLY B   1\n"
        "ATOM      3  CA  GLY C   1\n"
        "ATOM      4  H   GLY C   1\n"
        "END\n"
    )
    selection_map = json.loads(bundle["map"].read_text())
    selection_map["added_atoms"] = [
        {
            "chain_id": "C",
            "residue_number": 1,
            "insertion_code": "",
            "atom_name": "H",
            "element": "H",
            "prepared_pdb_atom_index": 4,
            "prepared_pdb_residue_index": 1,
            "prepared_topology_atom_index": 4,
            "prepared_topology_residue_index": 1,
        }
    ]
    selection_map["selections"]["design_c"] = [3, 4]
    selection_map["interface_mapping"]["design_c"] = [3, 4]
    selection_map["chain_audits"][2]["prepared_atom_count"] = 2
    bundle["map"].write_text(json.dumps(selection_map, indent=2, sort_keys=True))
    bundle["index"].write_text("[ receptor_ab ]\n1 2\n[ design_c ]\n3 4\n[ dimer_ab ]\n1 2\n")
    _rebind_bundle(bundle)
    work_dir = tmp_path / "work"

    stage_selection_sidecars(_strict_config(tmp_path, bundle), work_dir)
    refresh_selection_sidecars(
        work_dir,
        stage="ions",
        topology_path=work_dir / "topol.top",
        coordinates_path=work_dir / "processed.gro",
    )

    final_map = json.loads((work_dir / "selection-map.json").read_text())
    assert final_map["added_atoms"][0]["final_topology_atom_index"] == 4
    assert final_map["added_atoms"][0]["final_topology_residue_index"] == 1


def test_validate_staged_sidecars_refuses_modified_prepared_bundle_file(tmp_path: Path) -> None:
    from biolab_runners.gromacs.selection_sidecars import (
        stage_selection_sidecars,
        validate_staged_selection_sidecars,
    )

    bundle = _write_bundle(tmp_path / "bundle")
    work_dir = tmp_path / "work"
    stage_selection_sidecars(_strict_config(tmp_path, bundle), work_dir)
    (work_dir / "prepared.pdb").write_text("replaced prepared structure\n")

    with pytest.raises(ValueError, match=r"prepared\.pdb digest mismatch"):
        validate_staged_selection_sidecars(work_dir)


def test_stage_selection_sidecars_rejects_manifest_artifact_mismatch(tmp_path: Path) -> None:
    from biolab_runners.gromacs.selection_sidecars import stage_selection_sidecars

    bundle = _write_bundle(tmp_path / "bundle")
    bundle["index"].write_text("[ receptor_ab ]\n2\n")
    config = _strict_config(tmp_path, bundle)

    with pytest.raises(ValueError, match=r"index\.ndx digest mismatch"):
        stage_selection_sidecars(config, tmp_path / "work")


def test_refresh_sidecars_after_solvation_binds_final_topology_and_coordinates(
    tmp_path: Path,
) -> None:
    from biolab_runners.gromacs.selection_sidecars import (
        refresh_selection_sidecars,
        stage_selection_sidecars,
    )

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    work_dir = tmp_path / "work"
    stage_selection_sidecars(config, work_dir)
    topology = work_dir / "topol.top"
    topology.write_text("; topology with solvent\n")
    coordinates = work_dir / "solvated.gro"
    _write_gro(
        coordinates,
        [
            (1, "ALA", "N", 1),
            (1, "GLY", "CA", 2),
            (1, "GLY", "CA", 3),
            (2, "SOL", "OW", 4),
            (2, "SOL", "HW1", 5),
            (2, "SOL", "HW2", 6),
        ],
    )

    identity = refresh_selection_sidecars(
        work_dir,
        stage="solvate",
        topology_path=topology,
        coordinates_path=coordinates,
    )
    final_map = json.loads((work_dir / "selection-map.json").read_text())

    assert final_map["schema_version"] == 2
    assert final_map["selections"] == {
        "receptor_ab": [1, 2],
        "design_c": [3],
        "dimer_ab": [1, 2],
    }
    assert final_map["gromacs_state"]["topology_sha256"] == _sha256(topology)
    assert final_map["gromacs_state"]["coordinates_sha256"] == _sha256(coordinates)
    assert final_map["solvent_ion_boundaries"]["solute_atom_count"] == 3
    assert final_map["solvent_ion_boundaries"]["solvent_atom_indices"] == [4, 5, 6]
    assert final_map["solvent_ion_boundaries"]["ion_atom_indices"] == []
    assert identity["selection_map_sha256"] == _sha256(work_dir / "selection-map.json")
    assert (work_dir / "index.ndx").read_text() == (
        "[ receptor_ab ]\n1 2\n[ design_c ]\n3\n[ dimer_ab ]\n1 2\n"
    )


def test_refresh_sidecars_after_ions_records_ion_indices(tmp_path: Path) -> None:
    from biolab_runners.gromacs.selection_sidecars import (
        refresh_selection_sidecars,
        stage_selection_sidecars,
    )

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    work_dir = tmp_path / "work"
    stage_selection_sidecars(config, work_dir)
    topology = work_dir / "topol.top"
    topology.write_text("; topology with ions\n")
    coordinates = work_dir / "ions.gro"
    _write_gro(
        coordinates,
        [
            (1, "ALA", "N", 1),
            (1, "GLY", "CA", 2),
            (1, "GLY", "CA", 3),
            (2, "SOL", "OW", 4),
            (3, "NA", "NA", 5),
            (4, "CL", "CL", 6),
        ],
    )

    refresh_selection_sidecars(
        work_dir,
        stage="ions",
        topology_path=topology,
        coordinates_path=coordinates,
    )
    final_map = json.loads((work_dir / "selection-map.json").read_text())

    assert final_map["solvent_ion_boundaries"]["solvent_atom_indices"] == [4]
    assert final_map["solvent_ion_boundaries"]["ion_atom_indices"] == [5, 6]
    mapped = final_map["source_to_prepared_atoms"]
    assert [entry["prepared_topology_atom_index"] for entry in mapped] == [1, 2, 3]


def test_refresh_sidecars_accepts_molecule_boundary_residue_number_restart(
    tmp_path: Path,
) -> None:
    from biolab_runners.gromacs.selection_sidecars import (
        refresh_selection_sidecars,
        stage_selection_sidecars,
    )

    bundle = _write_bundle(tmp_path / "bundle")
    _write_gro(
        bundle["gro"],
        [(114, "ALA", "N", 1), (115, "GLY", "CA", 2), (1, "GLY", "CA", 3)],
    )
    _rebind_bundle(bundle)
    work_dir = tmp_path / "work"
    stage_selection_sidecars(_strict_config(tmp_path, bundle), work_dir)
    coordinates = work_dir / "ions.gro"
    _write_gro(
        coordinates,
        [
            (114, "ALA", "N", 1),
            (1, "GLY", "CA", 2),
            (1, "GLY", "CA", 3),
            (2, "NA", "NA", 4),
        ],
    )

    refresh_selection_sidecars(
        work_dir,
        stage="ions",
        topology_path=work_dir / "topol.top",
        coordinates_path=coordinates,
    )

    final_map = json.loads((work_dir / "selection-map.json").read_text())
    assert final_map["gromacs_added_atoms"][0]["residue_number"] == 2


@pytest.mark.parametrize(
    "final_numbers, message",
    [
        ([1, 2, 3, 3], "residue identity mismatch"),
        ([1, 1, 1, 1], "merged distinct solute residues"),
        ([1, 1, 3, 3], "renumbering is not consecutive"),
    ],
    ids=("split-residue", "merged-residues", "nonconsecutive-renumbering"),
)
def test_solute_residue_renumbering_preserves_residue_partition(
    final_numbers: list[int], message: str
) -> None:
    from biolab_runners.gromacs.selection_sidecars import (
        _GroAtom,
        _validate_solute_identity,
    )

    prepared = [
        _GroAtom(114, "ALA", "N", 1),
        _GroAtom(114, "ALA", "CA", 2),
        _GroAtom(115, "GLY", "N", 3),
        _GroAtom(115, "GLY", "CA", 4),
    ]
    final = [
        _GroAtom(number, atom.residue_name, atom.atom_name, atom.atom_number)
        for number, atom in zip(final_numbers, prepared, strict=True)
    ]
    selection_map = {
        "source_to_prepared_atoms": [
            {
                "prepared": {"chain_id": "A"},
                "prepared_topology_atom_index": index,
                "prepared_topology_residue_index": 1 if index <= 2 else 2,
            }
            for index in range(1, 5)
        ],
        "added_atoms": [],
    }

    _validate_solute_identity(
        prepared,
        [
            _GroAtom(1 if index <= 2 else 2, atom.residue_name, atom.atom_name, atom.atom_number)
            for index, atom in enumerate(prepared, 1)
        ],
        selection_map,
    )
    repeated_source_numbers = [
        _GroAtom(114, atom.residue_name, atom.atom_name, atom.atom_number) for atom in prepared
    ]
    _validate_solute_identity(
        repeated_source_numbers,
        repeated_source_numbers,
        selection_map,
    )
    with pytest.raises(ValueError, match=message):
        _validate_solute_identity(prepared, final, selection_map)


def test_validate_final_sidecars_refuses_modified_prepared_provenance(tmp_path: Path) -> None:
    from biolab_runners.gromacs.selection_sidecars import (
        refresh_selection_sidecars,
        stage_selection_sidecars,
        validate_staged_selection_sidecars,
    )

    bundle = _write_bundle(tmp_path / "bundle")
    work_dir = tmp_path / "work"
    stage_selection_sidecars(_strict_config(tmp_path, bundle), work_dir)
    refresh_selection_sidecars(
        work_dir,
        stage="ions",
        topology_path=work_dir / "topol.top",
        coordinates_path=work_dir / "processed.gro",
    )
    final_map = json.loads((work_dir / "selection-map.json").read_text())
    final_map["source_pdb_sha256"] = "c" * 64
    (work_dir / "selection-map.json").write_text(
        json.dumps(final_map, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="changed prepared field source_pdb_sha256"):
        validate_staged_selection_sidecars(work_dir)


@pytest.mark.parametrize(
    "atoms",
    [
        [(1, "GLY", "CA", 2), (1, "ALA", "N", 1), (1, "GLY", "CA", 3)],
        [(1, "GLY", "N", 1), (1, "GLY", "CA", 2), (1, "GLY", "CA", 3)],
        [(1, "ALA", "CA", 1), (1, "GLY", "CA", 2), (1, "GLY", "CA", 3)],
        [(1, "ALA", "N", 9), (1, "GLY", "CA", 2), (1, "GLY", "CA", 3)],
    ],
    ids=("atom-order", "residue-name", "atom-name", "atom-number"),
)
def test_refresh_sidecars_rejects_changed_solute_atom_identity(
    tmp_path: Path,
    atoms: list[tuple[int, str, str, int]],
) -> None:
    from biolab_runners.gromacs.selection_sidecars import (
        refresh_selection_sidecars,
        stage_selection_sidecars,
    )

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    work_dir = tmp_path / "work"
    stage_selection_sidecars(config, work_dir)
    topology = work_dir / "topol.top"
    topology.write_text("; topology\n")
    coordinates = work_dir / "solvated.gro"
    _write_gro(coordinates, atoms)

    with pytest.raises(ValueError, match="solute atom identity mismatch"):
        refresh_selection_sidecars(
            work_dir,
            stage="solvate",
            topology_path=topology,
            coordinates_path=coordinates,
        )


def test_checkpoint_binding_reuses_exact_digest_and_refuses_replacement(tmp_path: Path) -> None:
    from biolab_runners.gromacs.selection_sidecars import (
        bind_checkpoint_identity,
        refresh_selection_sidecars,
        stage_selection_sidecars,
        validate_checkpoint_identity,
    )

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    work_dir = tmp_path / "work"
    stage_selection_sidecars(config, work_dir)
    refresh_selection_sidecars(
        work_dir,
        stage="ions",
        topology_path=work_dir / "topol.top",
        coordinates_path=work_dir / "processed.gro",
    )
    checkpoint = work_dir / "prod.cpt"
    checkpoint.write_bytes(b"checkpoint-v1")

    identity = bind_checkpoint_identity(work_dir, stage="production", checkpoint=checkpoint)

    assert identity["checkpoint_sha256"] == _sha256(checkpoint)
    assert validate_checkpoint_identity(work_dir, stage="production", checkpoint=checkpoint)[
        "checkpoint_sha256"
    ] == _sha256(checkpoint)

    checkpoint.write_bytes(b"checkpoint-v2")
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        validate_checkpoint_identity(work_dir, stage="production", checkpoint=checkpoint)


def test_checkpoint_binding_refreshes_checkpoint_replaced_by_resumed_stage(tmp_path: Path) -> None:
    from biolab_runners.gromacs.selection_sidecars import (
        bind_checkpoint_identity,
        refresh_selection_sidecars,
        stage_selection_sidecars,
        validate_checkpoint_identity,
    )

    bundle = _write_bundle(tmp_path / "bundle")
    work_dir = tmp_path / "work"
    stage_selection_sidecars(_strict_config(tmp_path, bundle), work_dir)
    refresh_selection_sidecars(
        work_dir,
        stage="ions",
        topology_path=work_dir / "topol.top",
        coordinates_path=work_dir / "processed.gro",
    )
    checkpoint = work_dir / "min.cpt"
    checkpoint.write_bytes(b"first-interruption")
    bind_checkpoint_identity(work_dir, stage="minimize", checkpoint=checkpoint)
    checkpoint.write_bytes(b"second-interruption")

    rebound = bind_checkpoint_identity(work_dir, stage="minimize", checkpoint=checkpoint)

    assert rebound["checkpoint_sha256"] == _sha256(checkpoint)
    assert validate_checkpoint_identity(work_dir, stage="minimize", checkpoint=checkpoint)[
        "checkpoint_sha256"
    ] == _sha256(checkpoint)


def test_validate_staged_sidecars_refuses_missing_index(tmp_path: Path) -> None:
    from biolab_runners.gromacs.selection_sidecars import (
        stage_selection_sidecars,
        validate_staged_selection_sidecars,
    )

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    work_dir = tmp_path / "work"
    stage_selection_sidecars(config, work_dir)
    (work_dir / "index.ndx").unlink()

    with pytest.raises(ValueError, match=r"index\.ndx is missing"):
        validate_staged_selection_sidecars(work_dir)


def test_stage_selection_sidecars_rejects_incompatible_extra_index_group(
    tmp_path: Path,
) -> None:
    from biolab_runners.gromacs.selection_sidecars import stage_selection_sidecars

    bundle = _write_bundle(tmp_path / "bundle")
    bundle["index"].write_text(bundle["index"].read_text() + "[ SOL ]\n3\n")
    selection_map = json.loads(bundle["map"].read_text())
    selection_map["index_sha256"] = _sha256(bundle["index"])
    bundle["map"].write_text(json.dumps(selection_map, indent=2, sort_keys=True))
    manifest = json.loads(bundle["manifest"].read_text())
    manifest["artifacts"]["selection-map.json"]["sha256"] = _sha256(bundle["map"])
    manifest["artifacts"]["index.ndx"]["sha256"] = _sha256(bundle["index"])
    manifest["scientific_metadata_sha256"] = _canonical_digest(
        manifest, excluded="scientific_metadata_sha256"
    )
    bundle["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="exactly receptor_ab, design_c, dimer_ab"):
        stage_selection_sidecars(_strict_config(tmp_path, bundle), tmp_path / "work")


def test_protocol_runner_refreshes_strict_sidecars_through_every_state_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from biolab_runners.gromacs.runner import GromacsProtocolRunner
    from biolab_runners.gromacs.utils import load_stage_manifest

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    commands: list[list[str]] = []
    observed_states: list[str] = []
    runner = GromacsProtocolRunner(binary_prefix=["gmx"])
    monkeypatch.setattr(runner, "_run_subprocess", _protocol_invoker(commands, observed_states))

    result = runner.run_protocol(config)

    work_dir = Path(result.output_dir)
    final_map = json.loads((work_dir / "selection-map.json").read_text())
    manifest = load_stage_manifest(work_dir)
    assert result.failed == 0
    assert result.succeeded == 7
    assert result.skipped == 1
    assert observed_states == ["solvate", "ions"]
    assert final_map["schema_version"] == 2
    assert final_map["gromacs_state"]["stage"] == "production"
    assert final_map["gromacs_state"]["checkpoint_file"] == "prod.cpt"
    assert final_map["gromacs_state"]["checkpoint_sha256"] == _sha256(work_dir / "prod.cpt")
    assert final_map["solvent_ion_boundaries"]["solvent_atom_indices"] == [4]
    assert final_map["solvent_ion_boundaries"]["ion_atom_indices"] == [5, 6]
    assert len(manifest["stages"]["topology"]["protocol_identity"]) == 64
    assert any(command[1] == "mdrun" for command in commands)


def test_protocol_runner_refuses_invalid_interface_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from biolab_runners.gromacs.runner import GromacsProtocolRunner

    bundle = _write_bundle(tmp_path / "bundle")
    selection_map = json.loads(bundle["map"].read_text())
    selection_map["interface_mapping"]["receptor_b"] = [1]
    bundle["map"].write_text(json.dumps(selection_map, indent=2, sort_keys=True))
    _rebind_bundle(bundle)
    commands: list[list[str]] = []
    runner = GromacsProtocolRunner(binary_prefix=["gmx"])
    monkeypatch.setattr(runner, "_run_subprocess", _protocol_invoker(commands, []))

    result = runner.run_protocol(_strict_config(tmp_path, bundle))

    assert result.failed == 1
    assert "selection sidecar staging failed" in result.error
    assert commands == []


def test_protocol_runner_binds_interruption_and_resumes_only_exact_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from biolab_runners.gromacs.runner import GromacsProtocolRunner

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    first_commands: list[list[str]] = []
    observed_states: list[str] = []
    successful = _protocol_invoker(first_commands, observed_states)
    interrupted = False

    def _interrupt_minimize(command: list[str], work_dir: Path, timeout: int) -> int:
        nonlocal interrupted
        if command[1] == "mdrun" and command[command.index("-deffnm") + 1] == "min":
            first_commands.append(command)
            (work_dir / "min.cpt").write_bytes(b"interrupted-minimize")
            interrupted = True
            return -signal.SIGTERM
        return successful(command, work_dir, timeout)

    first_runner = GromacsProtocolRunner(binary_prefix=["gmx"])
    monkeypatch.setattr(first_runner, "_run_subprocess", _interrupt_minimize)

    first_result = first_runner.run_protocol(config)

    work_dir = Path(first_result.output_dir)
    interrupted_map = json.loads((work_dir / "selection-map.json").read_text())
    assert interrupted is True
    assert first_result.interrupted == 1
    assert interrupted_map["gromacs_state"]["stage"] == "minimize"
    assert interrupted_map["gromacs_state"]["checkpoint_sha256"] == _sha256(work_dir / "min.cpt")

    resumed_commands: list[list[str]] = []
    resumed_runner = GromacsProtocolRunner(binary_prefix=["gmx"])
    monkeypatch.setattr(
        resumed_runner,
        "_run_subprocess",
        _protocol_invoker(resumed_commands, []),
    )

    resumed_result = resumed_runner.run_protocol(config)

    min_grompp = next(command for command in resumed_commands if "min.tpr" in command)
    min_mdrun = next(
        command
        for command in resumed_commands
        if command[1] == "mdrun" and command[command.index("-deffnm") + 1] == "min"
    )
    assert resumed_result.failed == 0
    assert "-t" in min_grompp
    assert min_grompp[min_grompp.index("-t") + 1] == str(work_dir / "min.cpt")
    assert min_mdrun[-3:] == ["-cpi", str(work_dir / "min.cpt"), "-append"]


def test_protocol_runner_refuses_replaced_checkpoint_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from biolab_runners.gromacs.runner import GromacsProtocolRunner

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    successful = _protocol_invoker([], [])

    def _interrupt_minimize(command: list[str], work_dir: Path, timeout: int) -> int:
        if command[1] == "mdrun" and command[command.index("-deffnm") + 1] == "min":
            (work_dir / "min.cpt").write_bytes(b"bound-checkpoint")
            return -signal.SIGTERM
        return successful(command, work_dir, timeout)

    first_runner = GromacsProtocolRunner(binary_prefix=["gmx"])
    monkeypatch.setattr(first_runner, "_run_subprocess", _interrupt_minimize)
    first_result = first_runner.run_protocol(config)
    work_dir = Path(first_result.output_dir)
    (work_dir / "min.cpt").write_bytes(b"replacement-checkpoint")
    refused_commands: list[list[str]] = []
    second_runner = GromacsProtocolRunner(binary_prefix=["gmx"])
    monkeypatch.setattr(
        second_runner,
        "_run_subprocess",
        _protocol_invoker(refused_commands, []),
    )

    result = second_runner.run_protocol(config)

    assert result.failed == 1
    assert "checkpoint digest mismatch" in result.error
    assert refused_commands == []


def test_protocol_runner_refuses_checkpoint_bound_to_different_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from biolab_runners.gromacs.runner import GromacsProtocolRunner

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    successful = _protocol_invoker([], [])

    def _interrupt_minimize(command: list[str], work_dir: Path, timeout: int) -> int:
        if command[1] == "mdrun" and command[command.index("-deffnm") + 1] == "min":
            (work_dir / "min.cpt").write_bytes(b"bound-checkpoint")
            return -signal.SIGTERM
        return successful(command, work_dir, timeout)

    first_runner = GromacsProtocolRunner(binary_prefix=["gmx"])
    monkeypatch.setattr(first_runner, "_run_subprocess", _interrupt_minimize)
    first_result = first_runner.run_protocol(config)
    work_dir = Path(first_result.output_dir)
    selection_map = json.loads((work_dir / "selection-map.json").read_text())
    selection_map["gromacs_state"]["stage"] = "production"
    (work_dir / "selection-map.json").write_text(
        json.dumps(selection_map, indent=2, sort_keys=True) + "\n"
    )
    refused_commands: list[list[str]] = []
    second_runner = GromacsProtocolRunner(binary_prefix=["gmx"])
    monkeypatch.setattr(
        second_runner,
        "_run_subprocess",
        _protocol_invoker(refused_commands, []),
    )

    result = second_runner.run_protocol(config)

    assert result.failed == 1
    assert "checkpoint stage mismatch for minimize" in result.error
    assert refused_commands == []


def test_protocol_runner_refuses_missing_staged_sidecar_without_restaging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from biolab_runners.gromacs.runner import GromacsProtocolRunner

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    first = GromacsProtocolRunner(binary_prefix=["gmx"], dry_run=True).run_protocol(config)
    work_dir = Path(first.output_dir)
    (work_dir / "index.ndx").unlink()
    refused_commands: list[list[str]] = []
    second_runner = GromacsProtocolRunner(binary_prefix=["gmx"])
    monkeypatch.setattr(
        second_runner,
        "_run_subprocess",
        _protocol_invoker(refused_commands, []),
    )

    result = second_runner.run_protocol(config)

    assert result.failed == 1
    assert "index.ndx is missing" in result.error
    assert refused_commands == []
    assert (work_dir / "topol.top").read_bytes() == bundle["top"].read_bytes()


def test_protocol_runner_binds_sidecar_source_changes_into_stage_identity(tmp_path: Path) -> None:
    from biolab_runners.gromacs.runner import GromacsProtocolRunner
    from biolab_runners.gromacs.utils import load_stage_manifest

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    first = GromacsProtocolRunner(binary_prefix=["gmx"], dry_run=True).run_protocol(config)
    work_dir = Path(first.output_dir)
    first_identity = load_stage_manifest(work_dir)["stages"]["topology"]["protocol_identity"]
    selection_map = json.loads(bundle["map"].read_text())
    selection_map["source_pdb_sha256"] = "c" * 64
    bundle["map"].write_text(json.dumps(selection_map, indent=2, sort_keys=True))
    manifest = json.loads(bundle["manifest"].read_text())
    manifest["source_pdb_sha256"] = selection_map["source_pdb_sha256"]
    manifest["artifacts"]["selection-map.json"]["sha256"] = _sha256(bundle["map"])
    manifest["scientific_metadata_sha256"] = _canonical_digest(
        manifest,
        excluded="scientific_metadata_sha256",
    )
    bundle["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True))

    second = GromacsProtocolRunner(binary_prefix=["gmx"], dry_run=True).run_protocol(config)

    second_identity = load_stage_manifest(work_dir)["stages"]["topology"]["protocol_identity"]
    assert second.error == ""
    assert second_identity != first_identity
    assert (work_dir / "selection-map.json").read_bytes() == bundle["map"].read_bytes()


def test_protocol_runner_force_restarts_strict_bundle_without_stale_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from biolab_runners.gromacs.runner import GromacsProtocolRunner

    bundle = _write_bundle(tmp_path / "bundle")
    config = _strict_config(tmp_path, bundle)
    first_runner = GromacsProtocolRunner(binary_prefix=["gmx"])
    monkeypatch.setattr(first_runner, "_run_subprocess", _protocol_invoker([], []))
    first = first_runner.run_protocol(config)
    forced_commands: list[list[str]] = []
    forced_runner = GromacsProtocolRunner(binary_prefix=["gmx"])
    monkeypatch.setattr(
        forced_runner,
        "_run_subprocess",
        _protocol_invoker(forced_commands, []),
    )

    forced = forced_runner.run_protocol(replace(config, force=True))

    assert first.failed == 0
    assert forced.failed == 0
    assert forced.succeeded == 8
    assert forced.skipped == 0
    assert all("-cpi" not in command for command in forced_commands)
