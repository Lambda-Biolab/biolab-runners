"""Behavioral tests for BR-G1 GROMACS selection sidecars."""

from __future__ import annotations

import hashlib
import json
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

    prepared_pdb.write_text("ATOM      1  N   ALA A   1\nATOM      2  CA  GLY C   1\nEND\n")
    prepared_top.write_text("; prepared topology\n")
    _write_gro(prepared_gro, [(1, "ALA", "N", 1), (2, "GLY", "CA", 2)])
    index.write_text("[ receptor_ab ]\n1\n[ design_c ]\n2\n[ dimer_ab ]\n1\n")

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
                "source_pdb_atom_index": 2,
                "source_pdb_residue_index": 2,
                "prepared_pdb_atom_index": 2,
                "prepared_pdb_residue_index": 2,
                "prepared_topology_atom_index": 2,
                "prepared_topology_residue_index": 2,
            },
        ],
        "added_atoms": [],
        "dropped_atoms": [],
        "source_to_prepared_residues": [],
        "added_residues": [],
        "dropped_residues": [],
        "selections": {"receptor_ab": [1], "design_c": [2], "dimer_ab": [1]},
        "chain_audits": [],
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
        "artifacts": {
            name: {"path": str(tmp_path / name), "sha256": digest}
            for name, digest in artifacts.items()
        },
        "atom_count": 2,
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
            (2, "GLY", "CA", 2),
            (3, "SOL", "OW", 3),
            (3, "SOL", "HW1", 4),
            (3, "SOL", "HW2", 5),
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
        "receptor_ab": [1],
        "design_c": [2],
        "dimer_ab": [1],
    }
    assert final_map["gromacs_state"]["topology_sha256"] == _sha256(topology)
    assert final_map["gromacs_state"]["coordinates_sha256"] == _sha256(coordinates)
    assert final_map["solvent_ion_boundaries"]["solute_atom_count"] == 2
    assert final_map["solvent_ion_boundaries"]["solvent_atom_indices"] == [3, 4, 5]
    assert final_map["solvent_ion_boundaries"]["ion_atom_indices"] == []
    assert identity["selection_map_sha256"] == _sha256(work_dir / "selection-map.json")
    assert (work_dir / "index.ndx").read_text() == (
        "[ receptor_ab ]\n1\n[ design_c ]\n2\n[ dimer_ab ]\n1\n"
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
            (2, "GLY", "CA", 2),
            (3, "SOL", "OW", 3),
            (4, "NA", "NA", 4),
            (5, "CL", "CL", 5),
        ],
    )

    refresh_selection_sidecars(
        work_dir,
        stage="ions",
        topology_path=topology,
        coordinates_path=coordinates,
    )
    final_map = json.loads((work_dir / "selection-map.json").read_text())

    assert final_map["solvent_ion_boundaries"]["solvent_atom_indices"] == [3]
    assert final_map["solvent_ion_boundaries"]["ion_atom_indices"] == [4, 5]
    mapped = final_map["source_to_prepared_atoms"]
    assert [entry["prepared_topology_atom_index"] for entry in mapped] == [1, 2]


def test_refresh_sidecars_rejects_changed_solute_atom_identity(tmp_path: Path) -> None:
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
    _write_gro(coordinates, [(1, "ALA", "CA", 1), (2, "GLY", "CA", 2)])

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
