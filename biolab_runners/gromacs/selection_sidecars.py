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
SOURCE_SELECTION_MAP = "prepared-selection-map.json"
SELECTION_MAP = "selection-map.json"
SOURCE_INDEX = "prepared-index.ndx"
INDEX = "index.ndx"
BUNDLE_MANIFEST = "complex-prep-manifest.json"

_GROUP_ORDER = ("receptor_ab", "design_c", "dimer_ab")
_MANIFEST_DIGEST_FIELD = "scientific_metadata_sha256"
_SOLVENT_RESIDUES = frozenset({"SOL", "HOH", "WAT", "TIP3", "TIP3P", "SPC", "SPCE"})
_ION_RESIDUES = frozenset({"NA", "CL", "K", "MG", "ZN"})


@dataclass(frozen=True)
class _GroAtom:
    residue_number: int
    residue_name: str
    atom_name: str
    atom_number: int

    def identity(self) -> tuple[int, str, str, int]:
        return (self.residue_number, self.residue_name, self.atom_name, self.atom_number)


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
    _copy(sources["prepared.top"], work_dir / "topol.top")
    _copy(sources["prepared.gro"], work_dir / "processed.gro")
    _copy(sources[PREPARED_PDB], work_dir / PREPARED_PDB)
    _copy(sources[BUNDLE_MANIFEST], work_dir / BUNDLE_MANIFEST)
    _copy(sources[SELECTION_MAP], work_dir / SOURCE_SELECTION_MAP)
    _copy(sources[SELECTION_MAP], work_dir / SELECTION_MAP)
    _copy(sources[INDEX], work_dir / SOURCE_INDEX)
    _copy(sources[INDEX], work_dir / INDEX)
    return _identity(work_dir, selection_map)


def validate_staged_selection_sidecars(work_dir: Path) -> dict[str, Any]:
    """Validate the current map/index and every identity they bind."""
    paths = _staged_paths(work_dir)
    for label, path in paths.items():
        _required_file(path, label)
    source_map = _read_json(paths[SOURCE_SELECTION_MAP], SOURCE_SELECTION_MAP)
    current_map = _read_json(paths[SELECTION_MAP], SELECTION_MAP)
    source_index_digest = _sha256(paths[SOURCE_INDEX])
    _validate_map_and_index(source_map, paths[SOURCE_INDEX], source_index_digest)
    if current_map.get("schema_version") == 1:
        if _sha256(paths[SELECTION_MAP]) != _sha256(paths[SOURCE_SELECTION_MAP]):
            raise ValueError("staged selection-map.json differs from its prepared source")
    elif current_map.get("schema_version") == 2:
        _validate_final_map(work_dir, current_map, source_map)
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
    source_map = _read_json(source_map_path, SOURCE_SELECTION_MAP)
    _validate_map_and_index(source_map, source_index_path, _sha256(source_index_path))
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
    current = _read_json(work_dir / SELECTION_MAP, SELECTION_MAP)
    state = current.get("gromacs_state")
    if not isinstance(state, dict):
        raise ValueError("selection-map.json has no GROMACS state to bind")
    state["stage"] = stage
    state["checkpoint_file"] = checkpoint.name
    state["checkpoint_sha256"] = _sha256(checkpoint)
    _write_json_atomic(work_dir / SELECTION_MAP, current)
    identity = validate_staged_selection_sidecars(work_dir)
    return {
        **identity,
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
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
        SOURCE_SELECTION_MAP: work_dir / SOURCE_SELECTION_MAP,
        SELECTION_MAP: work_dir / SELECTION_MAP,
        SOURCE_INDEX: work_dir / SOURCE_INDEX,
        INDEX: work_dir / INDEX,
        BUNDLE_MANIFEST: work_dir / BUNDLE_MANIFEST,
    }


def _validate_bundle_sources(
    sources: dict[str, Path],
    manifest: dict[str, Any],
    selection_map: dict[str, Any],
) -> None:
    for label, path in sources.items():
        _required_file(path, label)
    _validate_manifest(manifest)
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
    prepared = selection_map.get("prepared_artifact_sha256")
    if not isinstance(prepared, dict):
        raise ValueError("selection-map prepared_artifact_sha256 must be an object")
    for name in (PREPARED_PDB, "prepared.top", "prepared.gro"):
        if prepared.get(name) != _sha256(sources[name]):
            raise ValueError(f"selection-map {name} digest mismatch")
    _validate_map_and_index(selection_map, sources[INDEX], _sha256(sources[INDEX]))
    atoms = _parse_gro(sources["prepared.gro"])
    if manifest.get("atom_count") != len(atoms):
        raise ValueError("prepared.gro atom count mismatches complex-prep manifest")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("runner") != "complex_prep":
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
) -> None:
    if selection_map.get("schema_version") != 1:
        raise ValueError("prepared selection-map.json schema_version must be 1")
    if not _is_sha(selection_map.get("preparation_digest")):
        raise ValueError("selection-map preparation_digest is invalid")
    if selection_map.get("index_sha256") != index_digest:
        raise ValueError("index.ndx digest mismatch against selection-map.json")
    groups = _parse_index(index_path)
    selections = _selection_groups(selection_map)
    if groups != selections:
        raise ValueError("index.ndx groups disagree with selection-map.json")
    indices = [value for values in selections.values() for value in values]
    atom_count = len(_parse_gro_from_map(selection_map))
    if atom_count and max(indices, default=0) > atom_count:
        raise ValueError("selection-map atom index exceeds prepared atom count")


def _parse_gro_from_map(selection_map: dict[str, Any]) -> list[int]:
    records = selection_map.get("source_to_prepared_atoms")
    if not isinstance(records, list):
        raise ValueError("source_to_prepared_atoms must be a list")
    result: list[int] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("source_to_prepared_atoms entry must be an object")
        value = record.get("prepared_topology_atom_index")
        if type(value) is not int or value < 1:
            raise ValueError("prepared_topology_atom_index must be a positive integer")
        result.append(value)
    return result


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
    for record in selection_map["source_to_prepared_atoms"]:
        record["final_topology_atom_index"] = record["prepared_topology_atom_index"]
        record["final_topology_residue_index"] = record["prepared_topology_residue_index"]
    for record in selection_map.get("source_to_prepared_residues", []):
        if isinstance(record, dict) and "prepared_topology_residue_index" in record:
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
) -> None:
    if current.get("parent_selection_map_sha256") != _sha256(work_dir / SOURCE_SELECTION_MAP):
        raise ValueError("final selection map parent digest mismatch")
    if current.get("parent_index_sha256") != _sha256(work_dir / SOURCE_INDEX):
        raise ValueError("final selection map parent index digest mismatch")
    if current.get("index_sha256") != _sha256(work_dir / INDEX):
        raise ValueError("final selection map index digest mismatch")
    if _selection_groups(current) != _selection_groups(source):
        raise ValueError("final named selections differ from the prepared map")
    state = current.get("gromacs_state")
    if not isinstance(state, dict):
        raise ValueError("final selection map gromacs_state must be an object")
    _validate_state_file(work_dir, state, "topology")
    _validate_state_file(work_dir, state, "coordinates")
    checkpoint_file = state.get("checkpoint_file")
    if checkpoint_file is not None:
        _validate_state_file(work_dir, state, "checkpoint")


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
