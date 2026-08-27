"""Atomic full-complex preparation using the peptide-prep seams."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.complex_prep.config import (
    DESIGN_CHAIN_ID,
    GROMPP_NOT_RUN,
    GROMPP_PASSED,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    PREPARATION_SCHEMA_VERSION,
    PREPARED_GRO_FILENAME,
    PREPARED_PDB_FILENAME,
    PREPARED_TOP_FILENAME,
    PROTEIN_FORCE_FIELD,
    RECEPTOR_CHAIN_IDS,
    SELECTION_MAP_FILENAME,
    WATER_FORCE_FIELD,
    ComplexChainAudit,
    ComplexPrepBundle,
    ComplexPrepConfig,
    ComplexPrepResult,
    ComplexResidueAudit,
)
from biolab_runners.contracts import ArtifactReference, ExecutionStatus
from biolab_runners.peptide_prep import design_chain
from biolab_runners.peptide_prep.utils import TopologyBondRecord, file_sha256
from biolab_runners.provenance import _canonical_json

if TYPE_CHECKING:
    from biolab_runners.peptide_prep.protocols import ChiralityValidator, CoordinateTransformer

_OUTPUT_FILENAMES = (
    PREPARED_PDB_FILENAME,
    PREPARED_TOP_FILENAME,
    PREPARED_GRO_FILENAME,
    SELECTION_MAP_FILENAME,
    INDEX_FILENAME,
)
_GROUP_ORDER = ("receptor_ab", "design_c", "dimer_ab")
_SHA256_LENGTH = 64
_PDB_ROUNDTRIP_TOLERANCE_NM = 1.1e-4


@dataclass(frozen=True)
class _StructureState:
    topology: object
    positions: object
    source_topology: object
    source_positions: object
    chain_audits: tuple[ComplexChainAudit, ...]
    residue_audits: tuple[ComplexResidueAudit, ...]
    bond_records: tuple[TopologyBondRecord, ...]
    chirality_reports: tuple[tuple[Any, ...], ...]
    system: object
    net_charge: float


class ComplexPrepRunner:
    """Prepare one complete A+B+C complex without publishing partial output."""

    def run(
        self,
        config: ComplexPrepConfig,
        *,
        coordinate_transformer: CoordinateTransformer | None = None,
        chirality_validator: ChiralityValidator | None = None,
    ) -> ComplexPrepResult:
        """Run or strictly reuse a complete preparation bundle."""
        output_dir = Path(config.output_dir)
        source_digest = file_sha256(Path(config.source_pdb)) or ""
        config_digest = compute_complex_config_digest(config)
        preparation_digest = compute_preparation_digest(source_digest, config_digest)
        if config.source_decoy.status is not ExecutionStatus.SUCCEEDED:
            return _failure(
                output_dir,
                source_digest,
                config_digest,
                preparation_digest,
                "source_decoy.status must be SUCCEEDED",
            )

        cached = _inspect_existing(
            output_dir,
            config,
            source_digest=source_digest,
            config_digest=config_digest,
            preparation_digest=preparation_digest,
        )
        if cached is not None:
            return cached
        if not source_digest:
            return _failure(
                output_dir,
                source_digest,
                config_digest,
                preparation_digest,
                f"source_pdb is missing or unreadable: {config.source_pdb}",
            )
        if source_digest != config.source_decoy.output_pdb_identity.sha256:
            return _failure(
                output_dir,
                source_digest,
                config_digest,
                preparation_digest,
                "source_pdb bytes do not match source_decoy.output_pdb_identity.sha256",
            )
        if output_dir.exists() and any(output_dir.iterdir()):
            return _failure(
                output_dir,
                source_digest,
                config_digest,
                preparation_digest,
                "output_dir is not empty but no reusable complete bundle was found",
                status=ExecutionStatus.INCOMPLETE,
            )

        try:
            source_topology, source_positions = _load_source(config.source_pdb)
            _validate_source_against_decoy(source_topology, config)
            staging = _make_staging_dir(output_dir)
            try:
                state = _prepare_structure(
                    config,
                    source_topology,
                    source_positions,
                    coordinate_transformer=coordinate_transformer,
                    chirality_validator=chirality_validator,
                )
                _validate_preservation(state)
                manifest = _materialize_bundle(
                    config,
                    state,
                    staging,
                    source_digest=source_digest,
                    config_digest=config_digest,
                    preparation_digest=preparation_digest,
                )
                _publish_staging(staging, output_dir)
                bundle = _bundle_from_disk(
                    output_dir,
                    source_digest=source_digest,
                    config_digest=config_digest,
                    preparation_digest=preparation_digest,
                    manifest=manifest,
                )
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        except Exception as exc:
            return _failure(
                output_dir,
                source_digest,
                config_digest,
                preparation_digest,
                f"complex preparation failed: {exc}",
            )
        return ComplexPrepResult(
            status=ExecutionStatus.SUCCEEDED,
            output_dir=str(output_dir),
            source_digest=source_digest,
            config_digest=config_digest,
            preparation_digest=preparation_digest,
            bundle=bundle,
            executed=True,
        )


def compute_complex_config_digest(config: ComplexPrepConfig) -> str:
    """Hash the fixed-policy scientific configuration canonically."""
    payload = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "design_sequence": config.design_sequence,
        "chain_roles": {"receptor": RECEPTOR_CHAIN_IDS, "design": DESIGN_CHAIN_ID},
        "protein_force_field": PROTEIN_FORCE_FIELD,
        "water_force_field": WATER_FORCE_FIELD,
        "pH": 7.4,
        "topology": _descriptor_payload(config),
        "coordinate_transformer_identity": config.coordinate_transformer_identity,
        "chirality_validator_identity": config.chirality_validator_identity,
        "source_decoy": config.source_decoy.to_dict(),
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def compute_preparation_digest(source_digest: str, config_digest: str) -> str:
    """Hash the non-circular identity shared by manifest and descriptors."""
    payload = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "source_pdb_sha256": source_digest,
        "config_digest": config_digest,
        "runner": "complex_prep",
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _descriptor_payload(config: ComplexPrepConfig) -> dict[str, Any]:
    descriptor = config.topology
    return {
        "d_substitutions": [
            {"position": item.position, "residue": item.residue}
            for item in descriptor.d_substitutions
        ],
        "head_to_tail": None
        if descriptor.head_to_tail is None
        else {
            "head": descriptor.head_to_tail.head,
            "tail": descriptor.head_to_tail.tail,
        },
        "disulfides": [
            {"first": item.first, "second": item.second} for item in descriptor.disulfides
        ],
    }


def _load_source(source_pdb: str) -> tuple[object, object]:
    from pdbfixer import PDBFixer

    parsed = PDBFixer(filename=source_pdb)
    return parsed.topology, parsed.positions


def _validate_source_against_decoy(topology: object, config: ComplexPrepConfig) -> None:
    chain_map = _chain_map(topology)
    expected_chains = {*RECEPTOR_CHAIN_IDS, DESIGN_CHAIN_ID}
    if set(chain_map) != expected_chains:
        raise ValueError("source PDB must contain exactly unique chains A, B, and C")
    audits = {audit.chain_id: audit for audit in config.source_decoy.chain_audits}
    if set(audits) != expected_chains:
        raise ValueError("source decoy must contain exactly unique A/B/C chain audits")
    for chain_id, chain in chain_map.items():
        residues = list(chain.residues())
        atoms = list(chain.atoms())
        audit = audits[chain_id]
        if (len(residues), len(atoms)) != (audit.residue_count, audit.atom_count):
            raise ValueError(f"source chain {chain_id!r} does not match its Rosetta chain audit")


def _chain_map(topology: object) -> dict[str, object]:
    chains = list(topology.chains())  # type: ignore[attr-defined]
    result: dict[str, object] = {}
    for chain in chains:
        if chain.id in result:
            raise ValueError(f"source PDB has an ambiguous chain {chain.id!r}")
        result[chain.id] = chain
    return result


def _make_staging_dir(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))


def _prepare_structure(
    config: ComplexPrepConfig,
    source_topology: object,
    source_positions: object,
    *,
    coordinate_transformer: CoordinateTransformer | None,
    chirality_validator: ChiralityValidator | None,
) -> _StructureState:
    topology, positions = _mutate_and_hydrate(config)
    reports: list[tuple[Any, ...]] = []
    if chirality_validator is not None:
        reports.append(
            design_chain.run_chirality_validation(
                topology,
                positions,
                config.design_sequence,
                config.topology,
                chirality_validator,
                design_chain_id=DESIGN_CHAIN_ID,
                stage="post_h",
            )
        )
    if config.topology.d_substitutions:
        if coordinate_transformer is None or chirality_validator is None:
            raise ValueError("D substitutions require both preparation callbacks")
        positions = design_chain.apply_d_coordinate_transform(
            topology,
            positions,
            config.design_sequence,
            config.topology,
            coordinate_transformer,
            design_chain_id=DESIGN_CHAIN_ID,
        )
    if chirality_validator is not None:
        reports.append(
            design_chain.run_chirality_validation(
                topology,
                positions,
                config.design_sequence,
                config.topology,
                chirality_validator,
                design_chain_id=DESIGN_CHAIN_ID,
                stage="pre",
            )
        )
    topology, positions, bond_records = _apply_chemistry(config, topology, positions)
    if chirality_validator is not None:
        reports.append(
            design_chain.run_chirality_validation(
                topology,
                positions,
                config.design_sequence,
                config.topology,
                chirality_validator,
                design_chain_id=DESIGN_CHAIN_ID,
                stage="post",
            )
        )
        if any(not report.valid for stage in reports for report in stage):
            raise ValueError("chirality validation failed")
    system, net_charge = _build_system(topology)
    if system.getNumParticles() != topology.getNumAtoms():  # type: ignore[attr-defined]
        raise ValueError("OpenMM system particle count does not match prepared topology")
    chain_audits = _build_chain_audits(source_topology, topology)
    residue_audits = _build_residue_audits(source_topology, topology)
    return _StructureState(
        topology=topology,
        positions=positions,
        source_topology=source_topology,
        source_positions=source_positions,
        chain_audits=chain_audits,
        residue_audits=residue_audits,
        bond_records=tuple(bond_records),
        chirality_reports=tuple(reports),
        system=system,
        net_charge=net_charge,
    )


def _mutate_and_hydrate(config: ComplexPrepConfig) -> tuple[object, object]:
    from pdbfixer import PDBFixer

    from biolab_runners.peptide_prep.mutation import apply_design_chain_mutation

    topology, positions = apply_design_chain_mutation(
        backbone_pdb_path=config.source_pdb,
        design_chain_id=DESIGN_CHAIN_ID,
        target_sequence=config.design_sequence,
    )
    fixer = PDBFixer(filename=config.source_pdb)
    fixer.topology = topology
    fixer.positions = positions
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    pre_hydrogen_records = _atom_records(fixer.topology, fixer.positions)
    fixer.addMissingHydrogens(pH=7.4)
    fixer.positions = _restore_preexisting_coordinates(
        fixer.topology, fixer.positions, pre_hydrogen_records
    )
    return fixer.topology, fixer.positions


def _apply_chemistry(
    config: ComplexPrepConfig,
    topology: object,
    positions: object,
) -> tuple[object, object, list[Any]]:
    import openmm.app as app

    from biolab_runners.peptide_prep import chemistry

    _, design_residues = design_chain.resolve_design_chain(
        topology,
        DESIGN_CHAIN_ID,
        expected_length=len(config.design_sequence),
    )
    disulfide_pairs = [
        (design_residues[bond.first - 1].index, design_residues[bond.second - 1].index)
        for bond in config.topology.disulfides
    ]
    modeller = app.Modeller(topology, positions)
    if disulfide_pairs:
        chemistry.rename_cysteines_to_cyx(
            modeller,
            involved_residue_indices={index for pair in disulfide_pairs for index in pair},
        )
        chemistry.apply_disulfide_bonds(
            modeller.topology,
            disulfide_pairs=tuple(disulfide_pairs),
            app_module=app,
        )
    bond_records: list[Any] = []
    if config.topology.head_to_tail is not None:
        modeller, head_index, tail_index = chemistry.remove_chain_terminal_caps_for_cyclization(
            modeller,
            design_chain_id=DESIGN_CHAIN_ID,
        )
        tail_atom, head_atom = chemistry.apply_chain_head_to_tail_closure(
            modeller.topology,
            design_chain_id=DESIGN_CHAIN_ID,
            app_module=app,
        )
        bond_records.append(
            _bond_record(
                modeller.topology, tail_atom, head_atom, tail_index, head_index, "head_to_tail"
            )
        )
    if disulfide_pairs:
        bond_records.extend(_disulfide_records(modeller.topology, tuple(disulfide_pairs)))
    return modeller.topology, modeller.positions, bond_records


def _bond_record(
    topology: object,
    atom1_index: int,
    atom2_index: int,
    residue1_index: int,
    residue2_index: int,
    bond_type: str,
) -> TopologyBondRecord:
    atoms = {atom.index: atom for atom in topology.atoms()}  # type: ignore[attr-defined]
    return TopologyBondRecord(
        atom1_index=atom1_index,
        atom2_index=atom2_index,
        atom1_name=atoms[atom1_index].name,
        atom2_name=atoms[atom2_index].name,
        residue1_index=residue1_index,
        residue2_index=residue2_index,
        bond_type=bond_type,
    )


def _disulfide_records(
    topology: object, pairs: tuple[tuple[int, int], ...]
) -> list[TopologyBondRecord]:
    sg_by_residue = {
        atom.residue.index: atom
        for atom in topology.atoms()
        if atom.name == "SG"  # type: ignore[attr-defined]
    }
    return [
        TopologyBondRecord(
            atom1_index=sg_by_residue[first].index,
            atom2_index=sg_by_residue[second].index,
            atom1_name="SG",
            atom2_name="SG",
            residue1_index=first,
            residue2_index=second,
            bond_type="disulfide",
        )
        for first, second in pairs
    ]


def _build_system(topology: object) -> tuple[object, float]:
    import openmm.app as app

    from biolab_runners.peptide_prep.export import compute_net_charge_from_openmm

    forcefield = app.ForceField(PROTEIN_FORCE_FIELD, WATER_FORCE_FIELD)
    system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff)
    return system, compute_net_charge_from_openmm(system)


def _build_chain_audits(
    source_topology: object, prepared_topology: object
) -> tuple[ComplexChainAudit, ...]:
    source = _chain_map(source_topology)
    prepared = _chain_map(prepared_topology)
    if set(source) != set(prepared):
        raise ValueError("prepared topology changed the exact A/B/C chain set")
    audits: list[ComplexChainAudit] = []
    for chain_id in (*RECEPTOR_CHAIN_IDS, DESIGN_CHAIN_ID):
        source_chain = source[chain_id]
        prepared_chain = prepared[chain_id]
        audits.append(
            ComplexChainAudit(
                chain_id=chain_id,
                role="receptor" if chain_id in RECEPTOR_CHAIN_IDS else "design",
                source_residue_count=len(list(source_chain.residues())),
                prepared_residue_count=len(list(prepared_chain.residues())),
                source_atom_count=len(list(source_chain.atoms())),
                prepared_atom_count=len(list(prepared_chain.atoms())),
            )
        )
    return tuple(audits)


def _build_residue_audits(
    source_topology: object, prepared_topology: object
) -> tuple[ComplexResidueAudit, ...]:
    source = _chain_map(source_topology)
    prepared = _chain_map(prepared_topology)
    audits: list[ComplexResidueAudit] = []
    for chain_id in (*RECEPTOR_CHAIN_IDS, DESIGN_CHAIN_ID):
        source_residues = list(source[chain_id].residues())
        prepared_residues = list(prepared[chain_id].residues())
        if len(source_residues) != len(prepared_residues):
            raise ValueError(f"chain {chain_id!r} changed residue count")
        for source_residue, prepared_residue in zip(
            source_residues, prepared_residues, strict=True
        ):
            source_key = _residue_key(source_residue)
            if source_key != _residue_key(prepared_residue):
                raise ValueError(f"chain {chain_id!r} changed residue identity or order")
            audits.append(
                ComplexResidueAudit(
                    chain_id=chain_id,
                    residue_number=source_key[1],
                    insertion_code=source_key[2],
                    source_name=source_residue.name,
                    prepared_name=prepared_residue.name,
                    source_atom_count=len(list(source_residue.atoms())),
                    prepared_atom_count=len(list(prepared_residue.atoms())),
                    source_index=source_residue.index,
                    prepared_index=prepared_residue.index,
                )
            )
    return tuple(audits)


def _validate_preservation(state: _StructureState) -> None:
    source = _atom_records(state.source_topology, state.source_positions)
    prepared = _atom_records(state.topology, state.positions)
    for key, source_record in source.items():
        prepared_record = prepared.get(key)
        if prepared_record is None:
            continue
        if key[0] in RECEPTOR_CHAIN_IDS and source_record["element"] != "H":
            _require_same_geometry(key, source_record, prepared_record, "receptor")
        if key[0] == DESIGN_CHAIN_ID and key[3] in design_chain.D_BACKBONE_INVARIANT_ATOMS:
            _require_same_geometry(key, source_record, prepared_record, "design backbone")


def _require_same_geometry(
    key: tuple[str, int, str, str],
    source_record: dict[str, Any],
    prepared_record: dict[str, Any],
    label: str,
) -> None:
    if source_record["element"] != prepared_record["element"]:
        raise ValueError(f"{label} atom identity changed for {key!r}")
    if any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
        for a, b in zip(source_record["position_nm"], prepared_record["position_nm"], strict=True)
    ):
        raise ValueError(f"{label} atom coordinates changed for {key!r}")


def _atom_records(
    topology: object, positions: object
) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    records: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for atom in topology.atoms():  # type: ignore[attr-defined]
        if atom.element is None:
            raise ValueError(f"atom {atom.name!r} is missing element metadata")
        key = (*_residue_key(atom.residue), atom.name)
        if key in records:
            raise ValueError(f"ambiguous source/prepared atom identity {key!r}")
        records[key] = {
            "chain_id": key[0],
            "residue_number": key[1],
            "insertion_code": key[2],
            "atom_name": key[3],
            "element": atom.element.symbol,
            "atom_index": atom.index,
            "residue_index": atom.residue.index,
            "position_nm": _position_at(positions, atom.index),
        }
    return records


def _residue_key(residue: object) -> tuple[str, int, str]:
    identifier = str(residue.id).strip()  # type: ignore[attr-defined]
    number_text = identifier
    suffix = ""
    while number_text and not number_text[-1].isdigit() and number_text[-1] != "-":
        suffix = number_text[-1] + suffix
        number_text = number_text[:-1]
    if not number_text or (number_text == "-" or not number_text.lstrip("-").isdigit()):
        raise ValueError(f"residue ID is not numeric with an insertion code: {identifier!r}")
    return residue.chain.id, int(number_text), suffix  # type: ignore[attr-defined]


def _position_nm(position: object) -> tuple[float, float, float]:
    import openmm.unit as unit

    return tuple(
        float(value.value_in_unit(unit.nanometer))
        if hasattr(value, "value_in_unit")
        else float(value)
        for value in position  # type: ignore[union-attr]
    )  # type: ignore[return-value]


def _position_at(positions: object, index: int) -> tuple[float, float, float]:
    import openmm.unit as unit

    if hasattr(positions, "value_in_unit"):
        values = positions.value_in_unit(unit.nanometer)  # type: ignore[attr-defined]
        return tuple(float(value) for value in values[index])  # type: ignore[index]
    return _position_nm(positions[index])  # type: ignore[index]


def _restore_preexisting_coordinates(
    topology: object,
    positions: object,
    pre_hydrogen_records: dict[tuple[str, int, str, str], dict[str, Any]],
) -> object:
    import openmm.unit as unit

    import openmm

    restored = [
        _position_at(positions, index)
        for index in range(topology.getNumAtoms())  # type: ignore[attr-defined]
    ]
    for atom in topology.atoms():  # type: ignore[attr-defined]
        record = pre_hydrogen_records.get((*_residue_key(atom.residue), atom.name))
        if record is not None:
            restored[atom.index] = record["position_nm"]
    return unit.Quantity([openmm.Vec3(*position) for position in restored], unit.nanometer)


def _materialize_bundle(
    config: ComplexPrepConfig,
    state: _StructureState,
    staging: Path,
    *,
    source_digest: str,
    config_digest: str,
    preparation_digest: str,
) -> dict[str, Any]:
    from biolab_runners.peptide_prep import export

    pdb_path = staging / PREPARED_PDB_FILENAME
    top_path = staging / PREPARED_TOP_FILENAME
    gro_path = staging / PREPARED_GRO_FILENAME
    export.write_prepared_pdb(
        pdb_path,
        state.topology,
        state.positions,
        closure_bond_records=state.bond_records,
    )
    _restore_pdb_residue_identifiers(pdb_path, state.topology)
    exported = export.export_gromacs(
        state.topology,
        state.system,
        state.positions,
        top_path=top_path,
        gro_path=gro_path,
        gromacs_include_family="amber99sb-ildn-tip3p",
        position_restraint_force_k_kjmol_nm2=1000.0,
    )
    top_path = Path(exported["top_path"])
    gro_path = Path(exported["gro_path"])
    parity_ok, parity_message = export.verify_export_parity(
        state.topology,
        state.system,
        state.positions,
        top_path=top_path,
        gro_path=gro_path,
        no_nan=_positions_are_finite(state.positions),
    )
    if not parity_ok and parity_message.startswith("atom count mismatch:"):
        parity_ok, parity_message = _verify_multichain_parity(state.topology, top_path, gro_path)
    if not parity_ok:
        raise ValueError(f"GROMACS parity check failed: {parity_message}")
    grompp_status = _grompp_status(export, top_path, gro_path, staging)
    if grompp_status[0] is False:
        raise ValueError(f"gmx grompp round-trip failed: {grompp_status[1]}")
    _validate_exported_pdb(pdb_path, state)

    prepared_digests = {
        PREPARED_PDB_FILENAME: _required_sha(pdb_path),
        PREPARED_TOP_FILENAME: _required_sha(top_path),
        PREPARED_GRO_FILENAME: _required_sha(gro_path),
    }
    selections = _selection_groups(state.topology)
    index_path = staging / INDEX_FILENAME
    index_path.write_text(_render_index(selections))
    index_digest = _required_sha(index_path)
    selection_map = _build_selection_map(
        state,
        source_digest=source_digest,
        preparation_digest=preparation_digest,
        prepared_digests=prepared_digests,
        index_digest=index_digest,
    )
    map_path = staging / SELECTION_MAP_FILENAME
    map_path.write_text(json.dumps(selection_map, indent=2, sort_keys=True) + "\n")
    map_digest = _required_sha(map_path)
    _validate_selection_map(
        json.loads(map_path.read_text()), state.topology, index_path, map_digest
    )
    manifest_path = staging / MANIFEST_FILENAME
    manifest = _build_manifest(
        config,
        state,
        output_dir=Path(config.output_dir),
        source_digest=source_digest,
        config_digest=config_digest,
        preparation_digest=preparation_digest,
        prepared_digests=prepared_digests,
        map_digest=map_digest,
        index_digest=index_digest,
        grompp_status=grompp_status[1],
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _verify_staging(staging, manifest, state.topology)
    return manifest


def _grompp_status(
    export: object, top_path: Path, gro_path: Path, staging: Path
) -> tuple[bool, str]:
    import shutil as system_shutil

    present = system_shutil.which("gmx") is not None
    okay, _message = export.gmx_grompp_pp_check(  # type: ignore[attr-defined]
        top_path,
        gro_path,
        audit_workdir=staging / ".grompp_audit",
    )
    shutil.rmtree(staging / ".grompp_audit", ignore_errors=True)
    return okay, GROMPP_PASSED if present else GROMPP_NOT_RUN


def _positions_are_finite(positions: object) -> bool:
    return all(math.isfinite(value) for position in positions for value in _position_nm(position))  # type: ignore[union-attr]


def _validate_exported_pdb(path: Path, state: _StructureState) -> None:
    import openmm.app as app

    parsed = app.PDBFile(str(path))
    if parsed.topology.getNumAtoms() != state.topology.getNumAtoms():
        raise ValueError("prepared.pdb atom count differs from prepared topology")
    expected = _atom_records(state.topology, state.positions)
    actual = _atom_records(parsed.topology, parsed.positions)
    for key, record in expected.items():
        if key not in actual:
            raise ValueError(f"prepared.pdb dropped atom identity {key!r}")
        if any(
            abs(a - b) > _PDB_ROUNDTRIP_TOLERANCE_NM
            for a, b in zip(record["position_nm"], actual[key]["position_nm"], strict=True)
        ):
            raise ValueError(f"prepared.pdb geometry differs for {key!r}")


def _restore_pdb_residue_identifiers(path: Path, topology: object) -> None:
    lines = path.read_text().splitlines()
    atoms = list(topology.atoms())  # type: ignore[attr-defined]
    atom_index = 0
    output: list[str] = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            if atom_index >= len(atoms):
                raise ValueError("prepared.pdb contains more atoms than its topology")
            _chain_id, number, insertion = _residue_key(atoms[atom_index].residue)
            if len(insertion) > 1 or not -999 <= number <= 9999:
                raise ValueError(
                    "residue identifier cannot be represented in PDB fixed-width fields"
                )
            replacement = f"{number:4d}" + insertion
            output.append(line[:22] + replacement + line[27:])
            atom_index += 1
        else:
            output.append(line)
    if atom_index != len(atoms):
        raise ValueError("prepared.pdb contains fewer atoms than its topology")
    path.write_text("\n".join(output) + "\n")


def _verify_multichain_parity(
    topology: object,
    top_path: Path,
    gro_path: Path,
) -> tuple[bool, str]:
    """Handle ParmEd's multi-molecule atom-count layout for complete complexes."""
    try:
        import parmed

        structure = parmed.load_file(str(top_path), xyz=str(gro_path))
    except Exception as exc:
        return False, f"parmed round-trip failed: {exc}"
    if len(structure.atoms) != topology.getNumAtoms():  # type: ignore[attr-defined]
        return False, "multi-chain ParmEd atom count does not match OpenMM"
    return True, ""


def _selection_groups(topology: object) -> dict[str, list[int]]:
    groups = {name: [] for name in _GROUP_ORDER}
    for atom in topology.atoms():  # type: ignore[attr-defined]
        if atom.residue.chain.id in RECEPTOR_CHAIN_IDS:
            groups["receptor_ab"].append(atom.index + 1)
            groups["dimer_ab"].append(atom.index + 1)
        elif atom.residue.chain.id == DESIGN_CHAIN_ID:
            groups["design_c"].append(atom.index + 1)
        else:
            raise ValueError(f"unexpected chain {atom.residue.chain.id!r} in index groups")
    for group in groups.values():
        if group != sorted(set(group)) or not group:
            raise ValueError("index group is not ascending and unique")
    return groups


def _render_index(groups: dict[str, list[int]]) -> str:
    lines: list[str] = []
    for group_name in _GROUP_ORDER:
        lines.append(f"[ {group_name} ]")
        values = groups[group_name]
        lines.extend(
            " ".join(str(value) for value in values[index : index + 15])
            for index in range(0, len(values), 15)
        )
        lines.append("")
    return "\n".join(lines)


def _build_selection_map(
    state: _StructureState,
    *,
    source_digest: str,
    preparation_digest: str,
    prepared_digests: dict[str, str],
    index_digest: str,
) -> dict[str, Any]:
    source = _atom_records(state.source_topology, state.source_positions)
    prepared = _atom_records(state.topology, state.positions)
    source_residues = _residue_records(state.source_topology)
    prepared_residues = _residue_records(state.topology)
    common = sorted(set(source) & set(prepared))
    mapped = [_mapping_entry(source[key], prepared[key]) for key in common]
    dropped = [_identity_record(source[key]) for key in sorted(set(source) - set(prepared))]
    added = [_identity_record(prepared[key]) for key in sorted(set(prepared) - set(source))]
    residue_common = sorted(set(source_residues) & set(prepared_residues))
    residue_mapped = [
        {"source": source_residues[key], "prepared": prepared_residues[key]}
        for key in residue_common
    ]
    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "preparation_digest": preparation_digest,
        "source_pdb_sha256": source_digest,
        "prepared_artifact_sha256": dict(prepared_digests),
        "index_sha256": index_digest,
        "index_bases": {
            "source_pdb_atom": 1,
            "source_pdb_residue": 1,
            "prepared_pdb_atom": 1,
            "prepared_pdb_residue": 1,
            "prepared_topology_atom": 1,
            "prepared_topology_residue": 1,
        },
        "source_to_prepared_atoms": mapped,
        "added_atoms": added,
        "dropped_atoms": dropped,
        "source_to_prepared_residues": residue_mapped,
        "added_residues": [
            prepared_residues[key] for key in sorted(set(prepared_residues) - set(source_residues))
        ],
        "dropped_residues": [
            source_residues[key] for key in sorted(set(source_residues) - set(prepared_residues))
        ],
        "selections": _selection_groups(state.topology),
        "chain_audits": [audit.to_dict() for audit in state.chain_audits],
        "residue_audits": [audit.to_dict() for audit in state.residue_audits],
        "solvent_ion_boundaries": {"solvent": "not_staged", "ions": "not_staged"},
    }


def _mapping_entry(source: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": _identity_record(source),
        "prepared": _identity_record(prepared),
        "source_pdb_atom_index": source["atom_index"] + 1,
        "source_pdb_residue_index": source["residue_index"] + 1,
        "prepared_pdb_atom_index": prepared["atom_index"] + 1,
        "prepared_pdb_residue_index": prepared["residue_index"] + 1,
        "prepared_topology_atom_index": prepared["atom_index"] + 1,
        "prepared_topology_residue_index": prepared["residue_index"] + 1,
    }


def _identity_record(record: dict[str, Any]) -> dict[str, Any]:
    identity = {
        key: record[key]
        for key in ("chain_id", "residue_number", "insertion_code", "atom_name", "element")
        if key in record
    }
    for key in ("atom_index", "residue_index"):
        if key in record:
            identity[key] = record[key] + 1
    return identity


def _residue_records(topology: object) -> dict[tuple[str, int, str], dict[str, Any]]:
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for residue in topology.residues():  # type: ignore[attr-defined]
        key = _residue_key(residue)
        if key in result:
            raise ValueError(f"ambiguous residue identity {key!r}")
        result[key] = {
            "chain_id": key[0],
            "residue_number": key[1],
            "insertion_code": key[2],
            "residue_name": residue.name,
            "residue_index": residue.index + 1,
        }
    return result


def _build_manifest(
    config: ComplexPrepConfig,
    state: _StructureState,
    *,
    output_dir: Path,
    source_digest: str,
    config_digest: str,
    preparation_digest: str,
    prepared_digests: dict[str, str],
    map_digest: str,
    index_digest: str,
    grompp_status: str,
) -> dict[str, Any]:
    artifact_digests = {
        **prepared_digests,
        SELECTION_MAP_FILENAME: map_digest,
        INDEX_FILENAME: index_digest,
    }
    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "runner": "complex_prep",
        "output_dir": str(output_dir),
        "source_pdb": config.source_pdb,
        "source_pdb_sha256": source_digest,
        "source_decoy": config.source_decoy.to_dict(),
        "config_digest": config_digest,
        "preparation_digest": preparation_digest,
        "design_sequence": config.design_sequence,
        "chain_roles": {"receptor": list(RECEPTOR_CHAIN_IDS), "design": DESIGN_CHAIN_ID},
        "force_field": {"protein": PROTEIN_FORCE_FIELD, "water": WATER_FORCE_FIELD, "pH": 7.4},
        "topology": _descriptor_payload(config),
        "artifacts": {
            name: {"path": str(output_dir / name), "sha256": digest}
            for name, digest in artifact_digests.items()
        },
        "chain_audits": [audit.to_dict() for audit in state.chain_audits],
        "residue_audits": [audit.to_dict() for audit in state.residue_audits],
        "grompp_audit_status": grompp_status,
        "net_charge": state.net_charge,
        "atom_count": state.topology.getNumAtoms(),  # type: ignore[attr-defined]
        "solvent_ion_boundaries": {"solvent": "not_staged", "ions": "not_staged"},
        "chirality_reports": [
            [asdict(report) for report in stage] for stage in state.chirality_reports
        ],
        "bond_records": [asdict(record) for record in state.bond_records],
    }


def _verify_staging(staging: Path, manifest: dict[str, Any], topology: object) -> None:
    for filename in _OUTPUT_FILENAMES:
        path = staging / filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"staging artifact is missing, empty, or symlinked: {filename}")
        if _required_sha(path) != manifest["artifacts"][filename]["sha256"]:
            raise ValueError(f"staging artifact digest mismatch: {filename}")
    _validate_selection_map(
        json.loads((staging / SELECTION_MAP_FILENAME).read_text()),
        topology,
        staging / INDEX_FILENAME,
        _required_sha(staging / SELECTION_MAP_FILENAME),
    )
    _required_sha(staging / MANIFEST_FILENAME)


def _publish_staging(staging: Path, output_dir: Path) -> None:
    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ValueError("output_dir became occupied before atomic publish")
        output_dir.rmdir()
    os.replace(staging, output_dir)


def _required_sha(path: Path) -> str:
    digest = file_sha256(path)
    if digest is None or path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"required regular non-empty file unavailable: {path}")
    return digest


def _inspect_existing(
    output_dir: Path,
    config: ComplexPrepConfig,
    *,
    source_digest: str,
    config_digest: str,
    preparation_digest: str,
) -> ComplexPrepResult | None:
    if not output_dir.exists():
        return None
    if output_dir.is_symlink() or not output_dir.is_dir():
        return _failure(
            output_dir,
            source_digest,
            config_digest,
            preparation_digest,
            "output_dir is not a directory",
        )
    if not any(output_dir.iterdir()):
        return None
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return _failure(
            output_dir,
            source_digest,
            config_digest,
            preparation_digest,
            "complete bundle manifest is missing",
            status=ExecutionStatus.INCOMPLETE,
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return _failure(
            output_dir,
            source_digest,
            config_digest,
            preparation_digest,
            f"manifest is malformed: {exc}",
            status=ExecutionStatus.MALFORMED,
        )
    if not isinstance(manifest, dict):
        return _failure(
            output_dir,
            source_digest,
            config_digest,
            preparation_digest,
            "manifest is malformed",
            status=ExecutionStatus.MALFORMED,
        )
    try:
        _validate_manifest_identity(
            manifest, config, source_digest, config_digest, preparation_digest
        )
        _validate_manifest_files(output_dir, manifest)
        map_data = json.loads((output_dir / SELECTION_MAP_FILENAME).read_text())
        _validate_map_bindings(
            map_data,
            manifest,
            source_digest=source_digest,
            preparation_digest=preparation_digest,
        )
        import openmm.app as app

        prepared = app.PDBFile(str(output_dir / PREPARED_PDB_FILENAME))
        _validate_selection_map(
            map_data,
            prepared.topology,
            output_dir / INDEX_FILENAME,
            _required_sha(output_dir / SELECTION_MAP_FILENAME),
        )
        return _cached_result(
            output_dir, manifest, source_digest, config_digest, preparation_digest
        )
    except IncompleteBundleError as exc:
        return _failure(
            output_dir,
            source_digest,
            config_digest,
            preparation_digest,
            str(exc),
            status=ExecutionStatus.INCOMPLETE,
        )
    except MalformedBundleError as exc:
        return _failure(
            output_dir,
            source_digest,
            config_digest,
            preparation_digest,
            str(exc),
            status=ExecutionStatus.MALFORMED,
        )
    except Exception as exc:
        return _failure(
            output_dir,
            source_digest,
            config_digest,
            preparation_digest,
            f"cached bundle rejected: {exc}",
        )


class IncompleteBundleError(ValueError):
    """Classify a validly shaped but incomplete existing bundle."""

    pass


class MalformedBundleError(ValueError):
    """Classify an existing bundle with invalid schema or data shape."""

    pass


def _validate_manifest_identity(
    manifest: dict[str, Any],
    config: ComplexPrepConfig,
    source_digest: str,
    config_digest: str,
    preparation_digest: str,
) -> None:
    if (
        manifest.get("schema_version") != PREPARATION_SCHEMA_VERSION
        or manifest.get("runner") != "complex_prep"
    ):
        raise MalformedBundleError("manifest schema is unsupported")
    for key, expected in (
        ("source_pdb_sha256", source_digest),
        ("config_digest", config_digest),
        ("preparation_digest", preparation_digest),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"cached bundle is stale: manifest {key} does not match request")
    if manifest.get("design_sequence") != config.design_sequence or manifest.get(
        "topology"
    ) != _descriptor_payload(config):
        raise ValueError("cached bundle config descriptor does not match request")
    if manifest.get("source_decoy") != config.source_decoy.to_dict():
        raise ValueError("cached bundle source decoy identity does not match request")
    if manifest.get("chain_roles") != {
        "receptor": list(RECEPTOR_CHAIN_IDS),
        "design": DESIGN_CHAIN_ID,
    }:
        raise MalformedBundleError("manifest chain roles are malformed")
    if manifest.get("force_field") != {
        "protein": PROTEIN_FORCE_FIELD,
        "water": WATER_FORCE_FIELD,
        "pH": 7.4,
    }:
        raise MalformedBundleError("manifest force-field policy is malformed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_OUTPUT_FILENAMES):
        raise MalformedBundleError("manifest artifact list is malformed")


def _validate_manifest_files(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = output_dir / MANIFEST_FILENAME
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size == 0
    ):
        raise IncompleteBundleError("cached manifest is missing or empty")
    for child in output_dir.iterdir():
        if child.name != MANIFEST_FILENAME and child.name not in _OUTPUT_FILENAMES:
            raise ValueError(f"cached bundle contains unexpected entry: {child.name}")
    for filename in _OUTPUT_FILENAMES:
        _validate_manifest_artifact(output_dir, manifest, filename)


def _validate_manifest_artifact(output_dir: Path, manifest: dict[str, Any], filename: str) -> None:
    spec = manifest["artifacts"].get(filename)
    if not isinstance(spec, dict) or spec.get("path") != str(output_dir / filename):
        raise MalformedBundleError(f"manifest path binding is malformed for {filename}")
    if not _is_sha(spec.get("sha256")):
        raise MalformedBundleError(f"manifest digest is malformed for {filename}")
    path = output_dir / filename
    if path.is_symlink():
        raise ValueError(f"cached artifact is a symlink: {filename}")
    if not path.is_file() or path.stat().st_size == 0:
        raise IncompleteBundleError(f"cached artifact is missing or empty: {filename}")
    if _required_sha(path) != spec["sha256"]:
        raise ValueError(f"cached artifact digest mismatch: {filename}")


def _validate_selection_map(
    data: object,
    topology: object,
    index_path: Path,
    map_digest: str,
) -> None:
    if not isinstance(data, dict) or data.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise MalformedBundleError("selection map schema is malformed")
    if not _is_sha(data.get("preparation_digest")) or not _is_sha(data.get("source_pdb_sha256")):
        raise MalformedBundleError("selection map identity digests are malformed")
    if data.get("index_sha256") != _required_sha(index_path):
        raise ValueError("selection map/index digest binding is stale")
    _validate_index_groups(data.get("selections"), index_path, topology.getNumAtoms())  # type: ignore[attr-defined]
    _validate_atom_mapping(data, topology.getNumAtoms())  # type: ignore[attr-defined]
    _validate_residue_mapping(data, topology.getNumResidues())  # type: ignore[attr-defined]
    prepared_digests = data.get("prepared_artifact_sha256")
    if not isinstance(prepared_digests, dict) or set(prepared_digests) != set(
        _OUTPUT_FILENAMES[:3]
    ):
        raise MalformedBundleError("selection map lacks prepared artifact bindings")
    if any(not _is_sha(value) for value in prepared_digests.values()):
        raise MalformedBundleError("selection map prepared artifact digests are malformed")
    if not _is_sha(map_digest):
        raise MalformedBundleError("selection map digest is malformed")


def _validate_map_bindings(
    data: object,
    manifest: dict[str, Any],
    *,
    source_digest: str,
    preparation_digest: str,
) -> None:
    if not isinstance(data, dict):
        raise MalformedBundleError("selection map is malformed")
    if data.get("source_pdb_sha256") != source_digest:
        raise ValueError("selection map source digest is stale")
    if data.get("preparation_digest") != preparation_digest:
        raise ValueError("selection map preparation digest is stale")
    expected = {
        filename: manifest["artifacts"][filename]["sha256"] for filename in _OUTPUT_FILENAMES[:3]
    }
    if data.get("prepared_artifact_sha256") != expected:
        raise ValueError("selection map prepared artifact bindings are stale")
    if data.get("index_sha256") != manifest["artifacts"][INDEX_FILENAME]["sha256"]:
        raise ValueError("selection map index binding is stale")


def _validate_atom_mapping(data: dict[str, Any], atom_count: int) -> None:
    entries = data.get("source_to_prepared_atoms")
    added = data.get("added_atoms")
    dropped = data.get("dropped_atoms")
    if not all(isinstance(value, list) for value in (entries, added, dropped)):
        raise MalformedBundleError("selection map atom mappings are malformed")
    if not isinstance(entries, list) or not isinstance(added, list):
        raise MalformedBundleError("selection map atom mappings are malformed")
    prepared_indices: list[int] = []
    source_keys: set[str] = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("source"), dict)
            or not isinstance(entry.get("prepared"), dict)
        ):
            raise MalformedBundleError("selection map contains malformed mapped atom")
        prepared_index = entry.get("prepared_topology_atom_index")
        prepared_index = _checked_index(
            prepared_index, atom_count, "selection map contains an out-of-range prepared atom"
        )
        prepared_indices.append(prepared_index)
        source_keys.add(_identity_key(entry["source"]))
    if len(prepared_indices) != len(set(prepared_indices)) or len(source_keys) != len(entries):
        raise ValueError("selection map atom mapping is not bijective")
    for record in added:
        if not isinstance(record, dict):
            raise ValueError("selection map added atom is out of range")
        atom_index = record.get("atom_index")
        atom_index = _checked_index(
            atom_index, atom_count, "selection map added atom is out of range"
        )
        prepared_indices.append(atom_index)
    if sorted(prepared_indices) != list(range(1, atom_count + 1)):
        raise ValueError("selection map does not cover exactly the prepared atoms")


def _validate_residue_mapping(data: dict[str, Any], residue_count: int) -> None:
    entries = data.get("source_to_prepared_residues")
    added = data.get("added_residues")
    dropped = data.get("dropped_residues")
    if not all(isinstance(value, list) for value in (entries, added, dropped)):
        raise MalformedBundleError("selection map residue mappings are malformed")
    if not isinstance(entries, list) or not isinstance(added, list):
        raise MalformedBundleError("selection map residue mappings are malformed")
    prepared_indices: list[int] = []
    source_keys: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise MalformedBundleError("selection map contains malformed mapped residue")
        source = entry.get("source")
        prepared = entry.get("prepared")
        if not isinstance(source, dict) or not isinstance(prepared, dict):
            raise MalformedBundleError("selection map contains malformed mapped residue")
        index = _checked_index(
            prepared.get("residue_index"),
            residue_count,
            "selection map contains an out-of-range prepared residue",
        )
        prepared_indices.append(index)
        source_keys.add(_residue_identity_key(source))
    if len(prepared_indices) != len(set(prepared_indices)) or len(source_keys) != len(entries):
        raise ValueError("selection map residue mapping is not bijective")
    for record in added:
        if not isinstance(record, dict):
            raise MalformedBundleError("selection map added residue is malformed")
        prepared_indices.append(
            _checked_index(
                record.get("residue_index"),
                residue_count,
                "selection map added residue is out of range",
            )
        )
    if sorted(prepared_indices) != list(range(1, residue_count + 1)):
        raise ValueError("selection map does not cover exactly the prepared residues")


def _residue_identity_key(record: dict[str, Any]) -> str:
    fields = ("chain_id", "residue_number", "insertion_code")
    if any(field not in record for field in fields):
        raise MalformedBundleError("selection map residue identity is malformed")
    return "|".join(str(record[field]) for field in fields)


def _identity_key(record: dict[str, Any]) -> str:
    fields = ("chain_id", "residue_number", "insertion_code", "atom_name")
    if any(field not in record for field in fields):
        raise MalformedBundleError("selection map atom identity is malformed")
    return "|".join(str(record[field]) for field in fields)


def _validate_index_groups(groups: object, index_path: Path, atom_count: int) -> None:
    if not isinstance(groups, dict) or set(groups) != set(_GROUP_ORDER):
        raise MalformedBundleError("selection map groups are malformed")
    expected = {name: list(values) for name, values in groups.items() if isinstance(values, list)}
    parsed = _parse_index(index_path)
    if parsed != expected:
        raise ValueError("index.ndx selections do not equal selection-map selections")
    for values in parsed.values():
        if values != sorted(set(values)) or any(
            not _valid_index(value, atom_count) for value in values
        ):
            raise ValueError("index.ndx contains duplicate or out-of-range atom indices")
    if parsed["receptor_ab"] != parsed["dimer_ab"]:
        raise ValueError("receptor_ab and dimer_ab selections differ")


def _parse_index(path: Path) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    current: str | None = None
    for line in path.read_text().splitlines():
        current = _parse_index_line(line, groups, current)
    if tuple(groups) != _GROUP_ORDER:
        raise MalformedBundleError("index.ndx group order is not the fixed contract")
    return groups


def _parse_index_line(line: str, groups: dict[str, list[int]], current: str | None) -> str | None:
    stripped = line.strip()
    if not stripped:
        return current
    if stripped.startswith("[") and stripped.endswith("]"):
        name = stripped[1:-1].strip()
        if name in groups:
            raise MalformedBundleError("index.ndx contains duplicate groups")
        groups[name] = []
        return name
    if current is None:
        raise MalformedBundleError("index.ndx has values before its first group")
    try:
        groups[current].extend(int(token) for token in stripped.split())
    except ValueError as exc:
        raise MalformedBundleError("index.ndx contains a non-integer atom index") from exc
    return current


def _valid_index(value: object, upper: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= upper


def _checked_index(value: object, upper: int, error: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= upper:
        raise ValueError(error)
    return value


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _cached_result(
    output_dir: Path,
    manifest: dict[str, Any],
    source_digest: str,
    config_digest: str,
    preparation_digest: str,
) -> ComplexPrepResult:
    return ComplexPrepResult(
        status=ExecutionStatus.CACHED,
        output_dir=str(output_dir),
        source_digest=source_digest,
        config_digest=config_digest,
        preparation_digest=preparation_digest,
        bundle=_bundle_from_disk(
            output_dir,
            source_digest=source_digest,
            config_digest=config_digest,
            preparation_digest=preparation_digest,
            manifest=manifest,
        ),
    )


def _bundle_from_disk(
    output_dir: Path,
    *,
    source_digest: str,
    config_digest: str,
    preparation_digest: str,
    manifest: dict[str, Any],
) -> ComplexPrepBundle:
    references = {
        filename: ArtifactReference.from_path(
            output_dir / filename,
            kind="structure"
            if filename == PREPARED_PDB_FILENAME
            else "topology"
            if filename == PREPARED_TOP_FILENAME
            else "coordinates"
            if filename == PREPARED_GRO_FILENAME
            else "metadata",
            root=output_dir,
        )
        for filename in (*_OUTPUT_FILENAMES, MANIFEST_FILENAME)
    }
    return ComplexPrepBundle(
        output_dir=str(output_dir),
        source_digest=source_digest,
        config_digest=config_digest,
        preparation_digest=preparation_digest,
        prepared_pdb=references[PREPARED_PDB_FILENAME],
        prepared_top=references[PREPARED_TOP_FILENAME],
        prepared_gro=references[PREPARED_GRO_FILENAME],
        selection_map=references[SELECTION_MAP_FILENAME],
        index=references[INDEX_FILENAME],
        manifest=references[MANIFEST_FILENAME],
        chain_audits=tuple(ComplexChainAudit(**audit) for audit in manifest["chain_audits"]),
        residue_audits=tuple(ComplexResidueAudit(**audit) for audit in manifest["residue_audits"]),
        grompp_audit_status=manifest["grompp_audit_status"],
        net_charge=float(manifest["net_charge"]),
        atom_count=int(manifest["atom_count"]),
    )


def _failure(
    output_dir: Path,
    source_digest: str,
    config_digest: str,
    preparation_digest: str,
    error: str,
    *,
    status: ExecutionStatus = ExecutionStatus.FAILED,
) -> ComplexPrepResult:
    return ComplexPrepResult(
        status=status,
        output_dir=str(output_dir),
        source_digest=source_digest,
        config_digest=config_digest,
        preparation_digest=preparation_digest,
        error=error,
    )
