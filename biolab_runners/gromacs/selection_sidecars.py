"""Strict selection-sidecar staging for full-complex GROMACS runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from biolab_runners.gromacs.config import GromacsProtocolConfig

PREPARED_PDB = "prepared.pdb"
SOURCE_TOPOLOGY = "prepared-topol.top"
SOURCE_SELECTION_MAP = "prepared-selection-map.json"
SELECTION_MAP = "selection-map.json"
SOURCE_INDEX = "prepared-index.ndx"
INDEX = "index.ndx"
BUNDLE_MANIFEST = "complex-prep-manifest.json"

_GROUP_ORDER = ("receptor_ab", "design_c", "dimer_ab")
_INTERFACE_GROUP_ORDER = ("receptor_a", "receptor_b", "dimer_ab", "design_c")
_FULL_COMPLEX_CHAINS = ("A", "B", "C")
_CHAIN_AUDIT_FIELDS: set[str] = {
    "chain_id",
    "role",
    "source_residue_count",
    "prepared_residue_count",
    "source_atom_count",
    "prepared_atom_count",
}
_CHAIN_AUDIT_COUNT_FIELDS = (
    "source_residue_count",
    "prepared_residue_count",
    "source_atom_count",
    "prepared_atom_count",
)
_MANIFEST_DIGEST_FIELD = "scientific_metadata_sha256"
_SOLVENT_RESIDUES = frozenset({"SOL", "HOH", "WAT", "TIP3", "TIP3P", "SPC", "SPCE"})
_ION_RESIDUES = frozenset({"NA", "CL", "K", "MG", "ZN"})
_GROMACS_STATE_STAGES = frozenset(
    {"solvate", "ions", "minimize", "equil_nvt", "equil_npt", "production"}
)


@dataclass(frozen=True)
class _GroAtom:
    residue_number: int
    residue_name: str
    atom_name: str
    atom_number: int

    def identity(self) -> tuple[int, str, str, int]:
        return (self.residue_number, self.residue_name, self.atom_name, self.atom_number)


@dataclass(frozen=True)
class _PdbAtom:
    chain_id: str
    residue_key: tuple[int, str]


def strict_sidecars_requested(config: GromacsProtocolConfig) -> bool:
    """Return whether the config selects the strict BR-C4 bundle path."""
    return bool(config.prebuilt_selection_map)


def stage_selection_sidecars(
    config: GromacsProtocolConfig,
    work_dir: Path,
) -> dict[str, Any]:
    """Validate and stage one complete immutable BR-C4 bundle identity."""
    if not strict_sidecars_requested(config):
        raise ValueError("selection sidecar staging requires a complete sidecar bundle")
    sources = _source_paths(config)
    manifest = _read_json(sources[BUNDLE_MANIFEST], BUNDLE_MANIFEST)
    selection_map = _read_json(sources[SELECTION_MAP], SELECTION_MAP)
    _validate_bundle_sources(sources, manifest, selection_map)
    work_dir.mkdir(parents=True, exist_ok=True)
    _copy(sources["prepared.top"], work_dir / SOURCE_TOPOLOGY)
    _copy(sources["prepared.top"], work_dir / "topol.top")
    _copy(sources["prepared.gro"], work_dir / "processed.gro")
    _copy(sources[PREPARED_PDB], work_dir / PREPARED_PDB)
    _copy(sources[BUNDLE_MANIFEST], work_dir / BUNDLE_MANIFEST)
    _copy(sources[SELECTION_MAP], work_dir / SOURCE_SELECTION_MAP)
    _copy(sources[SELECTION_MAP], work_dir / SELECTION_MAP)
    _copy(sources[INDEX], work_dir / SOURCE_INDEX)
    _copy(sources[INDEX], work_dir / INDEX)
    return validate_staged_selection_sidecars(work_dir)


def validate_staged_selection_sidecars(work_dir: Path) -> dict[str, Any]:
    """Validate the current map/index and every identity they bind."""
    return _validate_staged_selection_sidecars(work_dir)


def _validate_staged_selection_sidecars(
    work_dir: Path,
    *,
    mutable_checkpoint: Path | None = None,
) -> dict[str, Any]:
    paths = _staged_paths(work_dir)
    for label, path in paths.items():
        _required_file(path, label)
    _validate_staged_source_bundle(work_dir)
    source_map = _read_json(paths[SOURCE_SELECTION_MAP], SOURCE_SELECTION_MAP)
    current_map = _read_json(paths[SELECTION_MAP], SELECTION_MAP)
    if type(current_map.get("schema_version")) is int and current_map["schema_version"] == 1:
        if _sha256(paths[SELECTION_MAP]) != _sha256(paths[SOURCE_SELECTION_MAP]):
            raise ValueError("staged selection-map.json differs from its prepared source")
        if _sha256(paths[INDEX]) != _sha256(paths[SOURCE_INDEX]):
            raise ValueError("staged index.ndx differs from its prepared source")
        if _sha256(work_dir / "topol.top") != _sha256(paths[SOURCE_TOPOLOGY]):
            raise ValueError("staged topol.top differs from its prepared source")
    elif type(current_map.get("schema_version")) is int and current_map["schema_version"] == 2:
        _validate_final_map(
            work_dir,
            current_map,
            source_map,
            mutable_checkpoint=mutable_checkpoint,
        )
    else:
        raise ValueError("selection-map.json schema_version must be 1 or 2")
    return _identity(work_dir, current_map)


def refresh_selection_sidecars(
    work_dir: Path,
    *,
    stage: str,
    topology_path: Path,
    coordinates_path: Path,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Regenerate the final map/index after a GROMACS state transition."""
    source_map_path = work_dir / SOURCE_SELECTION_MAP
    source_index_path = work_dir / SOURCE_INDEX
    _validate_staged_source_bundle(work_dir)
    source_map = _read_json(source_map_path, SOURCE_SELECTION_MAP)
    _required_file(topology_path, topology_path.name)
    prepared_atoms = _parse_gro(work_dir / "processed.gro")
    final_atoms = _parse_gro(coordinates_path)
    _validate_solute_identity(prepared_atoms, final_atoms)
    selections = _selection_groups(source_map)
    _write_atomic(work_dir / INDEX, _render_index(selections))
    final_map = _build_final_map(
        source_map,
        source_map_path=source_map_path,
        source_index_path=source_index_path,
        stage=stage,
        topology_path=topology_path,
        coordinates_path=coordinates_path,
        prepared_atoms=prepared_atoms,
        final_atoms=final_atoms,
        checkpoint=checkpoint,
    )
    _write_json_atomic(work_dir / SELECTION_MAP, final_map)
    return validate_staged_selection_sidecars(work_dir)


def bind_checkpoint_identity(
    work_dir: Path,
    *,
    stage: str,
    checkpoint: Path,
) -> dict[str, Any]:
    """Bind the current selection space to one exact checkpoint payload."""
    _required_file(checkpoint, checkpoint.name)
    _validate_staged_selection_sidecars(work_dir, mutable_checkpoint=checkpoint)
    current = _read_json(work_dir / SELECTION_MAP, SELECTION_MAP)
    state = current.get("gromacs_state")
    if not isinstance(state, dict):
        raise ValueError("selection-map.json has no GROMACS state to bind")
    checkpoint_digest = _sha256(checkpoint)
    state["stage"] = stage
    state["checkpoint_file"] = checkpoint.name
    state["checkpoint_sha256"] = checkpoint_digest
    _write_json_atomic(work_dir / SELECTION_MAP, current)
    identity = validate_staged_selection_sidecars(work_dir)
    return {
        **identity,
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": checkpoint_digest,
    }


def validate_checkpoint_identity(
    work_dir: Path,
    *,
    stage: str,
    checkpoint: Path,
) -> dict[str, Any]:
    """Refuse a checkpoint not bound to the current named selections."""
    _required_file(checkpoint, checkpoint.name)
    current = _read_json(work_dir / SELECTION_MAP, SELECTION_MAP)
    state = current.get("gromacs_state")
    if not isinstance(state, dict) or state.get("stage") != stage:
        raise ValueError(f"checkpoint stage mismatch for {stage}")
    if state.get("checkpoint_file") != checkpoint.name:
        raise ValueError(f"checkpoint filename mismatch for {checkpoint.name}")
    digest = _sha256(checkpoint)
    if state.get("checkpoint_sha256") != digest:
        raise ValueError(f"checkpoint digest mismatch for {checkpoint.name}")
    validate_staged_selection_sidecars(work_dir)
    return {
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": digest,
        "selection_map_sha256": _sha256(work_dir / SELECTION_MAP),
        "index_sha256": _sha256(work_dir / INDEX),
    }


def _source_paths(config: GromacsProtocolConfig) -> dict[str, Path]:
    return {
        PREPARED_PDB: Path(config.prebuilt_prepared_pdb),
        "prepared.top": Path(config.prebuilt_topology),
        "prepared.gro": Path(config.prebuilt_coordinates),
        SELECTION_MAP: Path(config.prebuilt_selection_map),
        INDEX: Path(config.prebuilt_index),
        BUNDLE_MANIFEST: Path(config.prebuilt_bundle_manifest),
    }


def _staged_paths(work_dir: Path) -> dict[str, Path]:
    return {
        PREPARED_PDB: work_dir / PREPARED_PDB,
        SOURCE_TOPOLOGY: work_dir / SOURCE_TOPOLOGY,
        "prepared.gro": work_dir / "processed.gro",
        SOURCE_SELECTION_MAP: work_dir / SOURCE_SELECTION_MAP,
        SELECTION_MAP: work_dir / SELECTION_MAP,
        SOURCE_INDEX: work_dir / SOURCE_INDEX,
        INDEX: work_dir / INDEX,
        BUNDLE_MANIFEST: work_dir / BUNDLE_MANIFEST,
    }


def _validate_staged_source_bundle(work_dir: Path) -> None:
    sources = {
        PREPARED_PDB: work_dir / PREPARED_PDB,
        "prepared.top": work_dir / SOURCE_TOPOLOGY,
        "prepared.gro": work_dir / "processed.gro",
        SELECTION_MAP: work_dir / SOURCE_SELECTION_MAP,
        INDEX: work_dir / SOURCE_INDEX,
        BUNDLE_MANIFEST: work_dir / BUNDLE_MANIFEST,
    }
    _validate_bundle_sources(
        sources,
        _read_json(sources[BUNDLE_MANIFEST], BUNDLE_MANIFEST),
        _read_json(sources[SELECTION_MAP], SOURCE_SELECTION_MAP),
    )


def _validate_bundle_sources(
    sources: dict[str, Path],
    manifest: dict[str, Any],
    selection_map: dict[str, Any],
) -> None:
    for label, path in sources.items():
        _required_file(path, label)
    _validate_manifest(manifest)
    _validate_bundle_bindings(sources, manifest, selection_map)
    atoms = _parse_gro(sources["prepared.gro"])
    _validate_map_and_index(
        selection_map,
        sources[INDEX],
        _sha256(sources[INDEX]),
        atom_count=len(atoms),
        prepared_pdb_path=sources[PREPARED_PDB],
    )
    if type(manifest.get("atom_count")) is not int or manifest["atom_count"] != len(atoms):
        raise ValueError("prepared.gro atom count mismatches complex-prep manifest")


def _validate_bundle_bindings(
    sources: dict[str, Path],
    manifest: dict[str, Any],
    selection_map: dict[str, Any],
) -> None:
    _validate_manifest_chain_identity(manifest, selection_map)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("complex-prep manifest artifacts must be an object")
    source_names = (PREPARED_PDB, "prepared.top", "prepared.gro", SELECTION_MAP, INDEX)
    for name in source_names:
        record = artifacts.get(name)
        if not isinstance(record, dict) or record.get("sha256") != _sha256(sources[name]):
            raise ValueError(f"{name} digest mismatch against complex-prep manifest")
    if selection_map.get("preparation_digest") != manifest.get("preparation_digest"):
        raise ValueError("selection-map preparation_digest mismatches complex-prep manifest")
    source_digest = selection_map.get("source_pdb_sha256")
    if not _is_sha(source_digest) or source_digest != manifest.get("source_pdb_sha256"):
        raise ValueError("selection-map source digest mismatches complex-prep manifest")
    prepared = selection_map.get("prepared_artifact_sha256")
    prepared_names = {PREPARED_PDB, "prepared.top", "prepared.gro"}
    if not isinstance(prepared, dict) or set(prepared) != prepared_names:
        raise ValueError("selection-map prepared_artifact_sha256 must be an object")
    for name in (PREPARED_PDB, "prepared.top", "prepared.gro"):
        if prepared.get(name) != _sha256(sources[name]):
            raise ValueError(f"selection-map {name} digest mismatch")


def _validate_manifest_chain_identity(
    manifest: dict[str, Any], selection_map: dict[str, Any]
) -> None:
    if manifest.get("chain_roles") != {"receptor": ["A", "B"], "design": "C"}:
        raise ValueError("complex-prep manifest chain roles are not the explicit A+B+C contract")
    if manifest.get("chain_audits") != selection_map.get("chain_audits"):
        raise ValueError("complex-prep manifest chain audits differ from selection-map audits")
    if manifest.get("residue_audits") != selection_map.get("residue_audits"):
        raise ValueError("complex-prep manifest residue audits differ from selection-map audits")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("runner") != "complex_prep"
    ):
        raise ValueError("complex-prep manifest identity is invalid")
    expected = manifest.get(_MANIFEST_DIGEST_FIELD)
    if not _is_sha(expected):
        raise ValueError("complex-prep manifest scientific metadata digest is invalid")
    payload = {key: value for key, value in manifest.items() if key != _MANIFEST_DIGEST_FIELD}
    if expected != _canonical_digest(payload):
        raise ValueError("complex-prep manifest scientific metadata digest mismatch")


def _validate_map_and_index(
    selection_map: dict[str, Any],
    index_path: Path,
    index_digest: str,
    *,
    atom_count: int,
    prepared_pdb_path: Path,
) -> None:
    if type(selection_map.get("schema_version")) is not int or selection_map["schema_version"] != 1:
        raise ValueError("prepared selection-map.json schema_version must be 1")
    if not _is_sha(selection_map.get("preparation_digest")):
        raise ValueError("selection-map preparation_digest is invalid")
    if selection_map.get("index_bases") != {
        "source_pdb_atom": 1,
        "source_pdb_residue": 1,
        "prepared_pdb_atom": 1,
        "prepared_pdb_residue": 1,
        "prepared_topology_atom": 1,
        "prepared_topology_residue": 1,
    }:
        raise ValueError("selection-map index bases are invalid")
    if selection_map.get("solvent_ion_boundaries") != {
        "solvent": "not_staged",
        "ions": "not_staged",
    }:
        raise ValueError("prepared selection-map solvent/ion boundaries are invalid")
    if selection_map.get("index_sha256") != index_digest:
        raise ValueError("index.ndx digest mismatch against selection-map.json")
    groups = _parse_index(index_path)
    selections = _selection_groups(selection_map)
    if groups != selections:
        raise ValueError("index.ndx groups disagree with selection-map.json")
    indices = [value for values in selections.values() for value in values]
    if max(indices, default=0) > atom_count:
        raise ValueError("selection-map atom index exceeds prepared atom count")
    _validate_prepared_atom_partition(selection_map, atom_count)
    _validate_prepared_residue_records(selection_map)
    _validate_interface_mapping(
        selection_map,
        _parse_pdb_atoms(prepared_pdb_path),
        atom_count=atom_count,
    )


def _validate_interface_mapping(
    selection_map: dict[str, Any],
    pdb_atoms: list[_PdbAtom],
    *,
    atom_count: int,
) -> None:
    pdb_chains = _pdb_chain_groups(pdb_atoms)
    if tuple(pdb_chains) != _FULL_COMPLEX_CHAINS:
        raise ValueError("prepared.pdb must contain exactly explicit chains A, B, and C")
    if sum(len(values) for values in pdb_chains.values()) != atom_count:
        raise ValueError("prepared.pdb atom count differs from prepared.gro")
    mapping = _normalized_interface_mapping(selection_map.get("interface_mapping"), atom_count)
    _validate_interface_partition(mapping)
    _validate_interface_selections(mapping, selection_map)
    _validate_interface_chain_bindings(mapping, pdb_chains)
    _validate_full_complex_chain_audits(selection_map, pdb_atoms)


def _normalized_interface_mapping(value: object, atom_count: int) -> dict[str, list[int]]:
    if not isinstance(value, dict) or set(value) != set(_INTERFACE_GROUP_ORDER):
        raise ValueError(
            "selection map interface_mapping must contain exactly "
            "receptor_a, receptor_b, dimer_ab, design_c"
        )
    return {
        name: _validated_interface_group(value.get(name), name, atom_count)
        for name in _INTERFACE_GROUP_ORDER
    }


def _validated_interface_group(value: object, name: str, atom_count: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"interface mapping {name} must be a non-empty list")
    if any(type(item) is not int or not 1 <= item <= atom_count for item in value):
        raise ValueError(f"interface mapping {name} contains an out-of-range atom index")
    if value != sorted(set(value)):
        raise ValueError(f"interface mapping {name} must be sorted and unique")
    return list(value)


def _validate_interface_selections(
    mapping: dict[str, list[int]], selection_map: dict[str, Any]
) -> None:
    selections = selection_map.get("selections")
    if not isinstance(selections, dict):
        raise ValueError("selection map selections are malformed")
    if mapping["dimer_ab"] != selections.get("receptor_ab") or mapping["dimer_ab"] != (
        selections.get("dimer_ab")
    ):
        raise ValueError("interface mapping dimer_ab differs from receptor selections")
    if mapping["design_c"] != selections.get("design_c"):
        raise ValueError("interface mapping design_c differs from design selection")


def _validate_interface_partition(mapping: dict[str, list[int]]) -> None:
    if set(mapping["receptor_a"]) & set(mapping["receptor_b"]):
        raise ValueError("interface mapping receptor_a and receptor_b overlap")
    if mapping["dimer_ab"] != sorted(mapping["receptor_a"] + mapping["receptor_b"]):
        raise ValueError("interface mapping dimer_ab is not the sorted A+B union")
    if set(mapping["dimer_ab"]) & set(mapping["design_c"]):
        raise ValueError("interface mapping receptor dimer overlaps design C")


def _validate_interface_chain_bindings(
    mapping: dict[str, list[int]], pdb_chains: dict[str, list[int]]
) -> None:
    if mapping["receptor_a"] != pdb_chains["A"]:
        raise ValueError("interface mapping receptor_a differs from explicit chain A")
    if mapping["receptor_b"] != pdb_chains["B"]:
        raise ValueError("interface mapping receptor_b differs from explicit chain B")
    if mapping["design_c"] != pdb_chains["C"]:
        raise ValueError("interface mapping design_c differs from explicit chain C")
    if mapping["dimer_ab"] != sorted(pdb_chains["A"] + pdb_chains["B"]):
        raise ValueError("interface mapping dimer_ab differs from explicit A+B chains")


def _validate_full_complex_chain_audits(
    selection_map: dict[str, Any],
    pdb_atoms: list[_PdbAtom],
) -> None:
    audits = selection_map.get("chain_audits")
    if not isinstance(audits, list) or len(audits) != len(_FULL_COMPLEX_CHAINS):
        raise ValueError("selection map chain_audits must describe explicit A, B, and C chains")
    for audit, chain_id in zip(audits, _FULL_COMPLEX_CHAINS, strict=True):
        _validate_chain_audit(audit, chain_id, pdb_atoms)


def _validate_chain_audit(audit: object, chain_id: str, pdb_atoms: list[_PdbAtom]) -> None:
    if not isinstance(audit, dict) or set(audit) != _CHAIN_AUDIT_FIELDS:
        raise ValueError("selection map chain_audits are malformed")
    expected_role = "design" if chain_id == "C" else "receptor"
    if audit["chain_id"] != chain_id or audit["role"] != expected_role:
        raise ValueError("selection map chain_audits do not match explicit chain roles")
    if any(
        type(audit[field]) is not int or audit[field] < 1 for field in _CHAIN_AUDIT_COUNT_FIELDS
    ):
        raise ValueError("selection map chain_audits contain invalid counts")
    chain_atoms = [atom for atom in pdb_atoms if atom.chain_id == chain_id]
    if audit["prepared_atom_count"] != len(chain_atoms):
        raise ValueError("selection map prepared chain audit differs from prepared.pdb")
    if audit["prepared_residue_count"] != len({atom.residue_key for atom in chain_atoms}):
        raise ValueError("selection map prepared residue audit differs from prepared.pdb")


def _parse_pdb_atoms(path: Path) -> list[_PdbAtom]:
    _required_file(path, path.name)
    atoms: list[_PdbAtom] = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if len(line) < 26 or not line[21].strip():
            raise ValueError(f"{path.name} contains an invalid prepared atom record")
        try:
            residue_number = int(line[22:26].strip())
        except ValueError as exc:
            raise ValueError(f"{path.name} contains an invalid residue number") from exc
        insertion_code = line[26].strip() if len(line) > 26 else ""
        atoms.append(_PdbAtom(line[21], (residue_number, insertion_code)))
    if not atoms:
        raise ValueError(f"{path.name} contains no prepared atom records")
    return atoms


def _pdb_chain_groups(atoms: list[_PdbAtom]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, atom in enumerate(atoms, start=1):
        groups.setdefault(atom.chain_id, []).append(index)
    return groups


def _validate_prepared_atom_partition(selection_map: dict[str, Any], atom_count: int) -> None:
    indices: list[int] = []
    for field in ("source_to_prepared_atoms", "added_atoms"):
        indices.extend(_prepared_atom_indices(selection_map.get(field), field, atom_count))
    if sorted(indices) != list(range(1, atom_count + 1)):
        raise ValueError("selection-map prepared atom partition is incomplete or duplicated")


def _prepared_atom_indices(records: object, field: str, atom_count: int) -> list[int]:
    if not isinstance(records, list):
        raise ValueError(f"{field} must be a list")
    indices: list[int] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{field} entry must be an object")
        atom_index = record.get("prepared_topology_atom_index")
        residue_index = record.get("prepared_topology_residue_index")
        if type(atom_index) is not int or not 1 <= atom_index <= atom_count:
            raise ValueError("prepared_topology_atom_index is outside the prepared topology")
        if type(residue_index) is not int or residue_index < 1:
            raise ValueError("prepared_topology_residue_index must be a positive integer")
        indices.append(atom_index)
    return indices


def _validate_prepared_residue_records(selection_map: dict[str, Any]) -> None:
    for field in ("source_to_prepared_residues", "added_residues"):
        records = selection_map.get(field)
        if not isinstance(records, list):
            raise ValueError(f"{field} must be a list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{field} entry must be an object")
            value = record.get("prepared_topology_residue_index")
            if type(value) is not int or value < 1:
                raise ValueError("prepared_topology_residue_index must be a positive integer")


def _selection_groups(selection_map: dict[str, Any]) -> dict[str, list[int]]:
    groups = selection_map.get("selections")
    if not isinstance(groups, dict) or set(groups) != set(_GROUP_ORDER):
        raise ValueError("selection map must contain exactly receptor_ab, design_c, dimer_ab")
    result: dict[str, list[int]] = {}
    for name in _GROUP_ORDER:
        values = groups.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"selection {name} must be a non-empty list")
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError(f"selection {name} contains an invalid atom index")
        if values != sorted(set(values)):
            raise ValueError(f"selection {name} must be sorted and unique")
        result[name] = list(values)
    if result["receptor_ab"] != result["dimer_ab"]:
        raise ValueError("receptor_ab and dimer_ab selections must match")
    return result


def _parse_index(path: Path) -> dict[str, list[int]]:
    _required_file(path, path.name)
    groups: dict[str, list[int]] = {}
    current: str | None = None
    for raw in path.read_text().splitlines():
        current = _consume_index_line(raw.strip(), groups, current)
    _validate_index_groups(groups)
    return groups


def _consume_index_line(
    line: str,
    groups: dict[str, list[int]],
    current: str | None,
) -> str | None:
    if not line:
        return current
    if line.startswith("[") and line.endswith("]"):
        name = line[1:-1].strip()
        if name in groups:
            raise ValueError(f"duplicate index group {name}")
        groups[name] = []
        return name
    if current is None:
        raise ValueError("index.ndx has atom indices before its first group")
    try:
        groups[current].extend(int(value) for value in line.split())
    except ValueError as exc:
        raise ValueError("index.ndx contains a non-integer atom index") from exc
    return current


def _validate_index_groups(groups: dict[str, list[int]]) -> None:
    if tuple(groups) != _GROUP_ORDER:
        raise ValueError("index.ndx must contain exactly receptor_ab, design_c, dimer_ab")
    for name, values in groups.items():
        if not values or any(value < 1 for value in values) or values != sorted(set(values)):
            raise ValueError(f"index group {name} must be non-empty, sorted, and unique")


def _parse_gro(path: Path) -> list[_GroAtom]:
    _required_file(path, path.name)
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path.name} is not a complete GRO file")
    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"{path.name} atom count is invalid") from exc
    if atom_count < 1 or len(lines) < atom_count + 3:
        raise ValueError(f"{path.name} atom records are incomplete")
    return [_parse_gro_atom(path.name, line) for line in lines[2 : 2 + atom_count]]


def _parse_gro_atom(filename: str, line: str) -> _GroAtom:
    if len(line) < 20:
        raise ValueError(f"{filename} contains a truncated atom record")
    try:
        return _GroAtom(
            residue_number=int(line[0:5]),
            residue_name=line[5:10].strip(),
            atom_name=line[10:15].strip(),
            atom_number=int(line[15:20]),
        )
    except ValueError as exc:
        raise ValueError(f"{filename} contains an invalid atom identity") from exc


def _validate_solute_identity(prepared: list[_GroAtom], final: list[_GroAtom]) -> None:
    if len(final) < len(prepared):
        raise ValueError("final GRO has fewer atoms than the prepared solute")
    for index, (before, after) in enumerate(zip(prepared, final, strict=False), start=1):
        if before.identity() != after.identity():
            raise ValueError(f"solute atom identity mismatch at one-based index {index}")


def _build_final_map(
    source_map: dict[str, Any],
    *,
    source_map_path: Path,
    source_index_path: Path,
    stage: str,
    topology_path: Path,
    coordinates_path: Path,
    prepared_atoms: list[_GroAtom],
    final_atoms: list[_GroAtom],
    checkpoint: Path | None,
) -> dict[str, Any]:
    result = json.loads(json.dumps(source_map))
    result["schema_version"] = 2
    result["parent_selection_map_sha256"] = _sha256(source_map_path)
    result["parent_index_sha256"] = _sha256(source_index_path)
    result["index_sha256"] = _sha256(source_index_path.parent / INDEX)
    _add_final_indices(result)
    result["gromacs_added_atoms"] = [
        _gro_atom_payload(atom, index)
        for index, atom in enumerate(
            final_atoms[len(prepared_atoms) :], start=len(prepared_atoms) + 1
        )
    ]
    result["solvent_ion_boundaries"] = _environment_boundaries(prepared_atoms, final_atoms)
    state: dict[str, Any] = {
        "stage": stage,
        "topology_file": topology_path.name,
        "topology_sha256": _sha256(topology_path),
        "coordinates_file": coordinates_path.name,
        "coordinates_sha256": _sha256(coordinates_path),
        "atom_count": len(final_atoms),
        "solute_atom_count": len(prepared_atoms),
    }
    if checkpoint is not None:
        _required_file(checkpoint, checkpoint.name)
        state["checkpoint_file"] = checkpoint.name
        state["checkpoint_sha256"] = _sha256(checkpoint)
    result["gromacs_state"] = state
    return result


def _add_final_indices(selection_map: dict[str, Any]) -> None:
    for field in ("source_to_prepared_atoms", "added_atoms"):
        for record in selection_map[field]:
            record["final_topology_atom_index"] = record["prepared_topology_atom_index"]
            record["final_topology_residue_index"] = record["prepared_topology_residue_index"]
    for field in ("source_to_prepared_residues", "added_residues"):
        for record in selection_map[field]:
            record["final_topology_residue_index"] = record["prepared_topology_residue_index"]


def _gro_atom_payload(atom: _GroAtom, index: int) -> dict[str, Any]:
    return {
        "final_topology_atom_index": index,
        "residue_number": atom.residue_number,
        "residue_name": atom.residue_name,
        "atom_name": atom.atom_name,
        "gro_atom_number": atom.atom_number,
    }


def _environment_boundaries(
    prepared: list[_GroAtom],
    final: list[_GroAtom],
) -> dict[str, Any]:
    solvent: list[int] = []
    ions: list[int] = []
    for index, atom in enumerate(final[len(prepared) :], start=len(prepared) + 1):
        residue_name = atom.residue_name.upper()
        if residue_name in _SOLVENT_RESIDUES:
            solvent.append(index)
        elif residue_name in _ION_RESIDUES:
            ions.append(index)
        else:
            raise ValueError(
                f"unclassified environment residue {atom.residue_name!r} at atom {index}"
            )
    return {
        "solute_atom_count": len(prepared),
        "environment_start_atom_index": len(prepared) + 1 if len(final) > len(prepared) else None,
        "final_atom_count": len(final),
        "solvent_atom_indices": solvent,
        "ion_atom_indices": ions,
    }


def _validate_final_map(
    work_dir: Path,
    current: dict[str, Any],
    source: dict[str, Any],
    *,
    mutable_checkpoint: Path | None = None,
) -> None:
    if current.get("parent_selection_map_sha256") != _sha256(work_dir / SOURCE_SELECTION_MAP):
        raise ValueError("final selection map parent digest mismatch")
    if current.get("parent_index_sha256") != _sha256(work_dir / SOURCE_INDEX):
        raise ValueError("final selection map parent index digest mismatch")
    if current.get("index_sha256") != _sha256(work_dir / INDEX):
        raise ValueError("final selection map index digest mismatch")
    if _selection_groups(current) != _selection_groups(source):
        raise ValueError("final named selections differ from the prepared map")
    _validate_inherited_map_fields(current, source)
    state = current.get("gromacs_state")
    if not isinstance(state, dict):
        raise ValueError("final selection map gromacs_state must be an object")
    _validate_state_file(work_dir, state, "topology")
    _validate_state_file(work_dir, state, "coordinates")
    checkpoint_file = state.get("checkpoint_file")
    if checkpoint_file is not None:
        if mutable_checkpoint is not None and checkpoint_file == mutable_checkpoint.name:
            _required_file(mutable_checkpoint, mutable_checkpoint.name)
        else:
            _validate_state_file(work_dir, state, "checkpoint")
    _validate_final_state_payload(work_dir, current, state)


def _validate_inherited_map_fields(
    current: dict[str, Any],
    source: dict[str, Any],
) -> None:
    expected_source = json.loads(json.dumps(source))
    _add_final_indices(expected_source)
    for key, value in expected_source.items():
        if (
            key not in {"schema_version", "index_sha256", "solvent_ion_boundaries"}
            and current.get(key) != value
        ):
            raise ValueError(f"final selection map changed prepared field {key}")
    expected_keys = set(expected_source) | {
        "parent_selection_map_sha256",
        "parent_index_sha256",
        "gromacs_added_atoms",
        "gromacs_state",
    }
    if set(current) != expected_keys:
        raise ValueError("final selection map fields are incomplete or unexpected")


def _validate_final_state_payload(
    work_dir: Path,
    current: dict[str, Any],
    state: dict[str, Any],
) -> None:
    base_fields = {
        "stage",
        "topology_file",
        "topology_sha256",
        "coordinates_file",
        "coordinates_sha256",
        "atom_count",
        "solute_atom_count",
    }
    checkpoint_fields = {"checkpoint_file", "checkpoint_sha256"}
    state_fields = set(state)
    if state_fields != base_fields and state_fields != base_fields | checkpoint_fields:
        raise ValueError("final selection map GROMACS state fields are invalid")
    if state.get("stage") not in _GROMACS_STATE_STAGES:
        raise ValueError("final selection map GROMACS stage is invalid")
    prepared_atoms = _parse_gro(work_dir / "processed.gro")
    coordinates = work_dir / state["coordinates_file"]
    final_atoms = _parse_gro(coordinates)
    _validate_solute_identity(prepared_atoms, final_atoms)
    if type(state.get("solute_atom_count")) is not int or state["solute_atom_count"] != len(
        prepared_atoms
    ):
        raise ValueError("final selection map solute atom count mismatch")
    if type(state.get("atom_count")) is not int or state["atom_count"] != len(final_atoms):
        raise ValueError("final selection map atom count mismatch")
    expected_added = [
        _gro_atom_payload(atom, index)
        for index, atom in enumerate(
            final_atoms[len(prepared_atoms) :], start=len(prepared_atoms) + 1
        )
    ]
    if current.get("gromacs_added_atoms") != expected_added:
        raise ValueError("final selection map added atoms mismatch")
    if current.get("solvent_ion_boundaries") != _environment_boundaries(
        prepared_atoms, final_atoms
    ):
        raise ValueError("final selection map solvent/ion boundaries mismatch")


def _validate_state_file(work_dir: Path, state: dict[str, Any], label: str) -> None:
    filename = state.get(f"{label}_file")
    digest = state.get(f"{label}_sha256")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"GROMACS {label} filename is invalid")
    path = work_dir / filename
    _required_file(path, filename)
    if digest != _sha256(path):
        raise ValueError(f"GROMACS {label} digest mismatch")


def _identity(work_dir: Path, selection_map: dict[str, Any]) -> dict[str, Any]:
    return {
        "preparation_digest": selection_map.get("preparation_digest"),
        "prepared_pdb_sha256": _sha256(work_dir / PREPARED_PDB),
        "selection_map_sha256": _sha256(work_dir / SELECTION_MAP),
        "index_sha256": _sha256(work_dir / INDEX),
        "bundle_manifest_sha256": _sha256(work_dir / BUNDLE_MANIFEST),
    }


def _render_index(groups: dict[str, list[int]]) -> str:
    lines: list[str] = []
    for name in _GROUP_ORDER:
        lines.append(f"[ {name} ]")
        values = groups[name]
        lines.extend(
            " ".join(str(value) for value in values[start : start + 15])
            for start in range(0, len(values), 15)
        )
    return "\n".join(lines) + "\n"


def _required_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or not a regular file")
    if path.stat().st_size < 1:
        raise ValueError(f"{label} is empty")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _required_file(path, label)
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _copy(source: Path, destination: Path) -> None:
    _required_file(source, source.name)
    shutil.copy2(source, destination)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
