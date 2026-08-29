"""Atomic full-complex preparation using the peptide-prep seams."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.complex_prep.config import (
    ALGORITHM_VERSIONS,
    DESIGN_CHAIN_ID,
    GROMPP_NOT_RUN,
    GROMPP_PASSED,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_PAYLOAD_DIGEST_FIELD,
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
from biolab_runners.peptide_prep.config import (
    DEFAULT_MAX_DISULFIDE_DISTANCE_A,
    DEFAULT_MAX_HEAD_TO_TAIL_DISTANCE_A,
)
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
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


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
        config_digest = compute_complex_config_digest(config)
        source_digest = ""
        preparation_digest = compute_preparation_digest(source_digest, config_digest)
        if config.source_decoy.status is not ExecutionStatus.SUCCEEDED:
            return _failure(
                output_dir,
                source_digest,
                config_digest,
                preparation_digest,
                "source_decoy.status must be SUCCEEDED",
            )

        try:
            source_bytes, source_digest = _read_source_snapshot(Path(config.source_pdb))
        except Exception as exc:
            return _failure(
                output_dir,
                source_digest,
                config_digest,
                compute_preparation_digest(source_digest, config_digest),
                f"source_pdb read failed: {exc}",
            )
        preparation_digest = compute_preparation_digest(source_digest, config_digest)
        if source_digest != config.source_decoy.output_pdb_identity.sha256:
            return _failure(
                output_dir,
                source_digest,
                config_digest,
                preparation_digest,
                "source_pdb bytes do not match source_decoy.output_pdb_identity.sha256",
            )

        return _run_from_snapshot(
            config,
            output_dir,
            source_bytes,
            source_digest=source_digest,
            config_digest=config_digest,
            preparation_digest=preparation_digest,
            coordinate_transformer=coordinate_transformer,
            chirality_validator=chirality_validator,
        )


def _run_from_snapshot(
    config: ComplexPrepConfig,
    output_dir: Path,
    source_bytes: bytes,
    *,
    source_digest: str,
    config_digest: str,
    preparation_digest: str,
    coordinate_transformer: CoordinateTransformer | None,
    chirality_validator: ChiralityValidator | None,
) -> ComplexPrepResult:
    executed = False
    snapshot_dir: Path | None = None

    try:
        snapshot_dir, snapshot_path = _make_source_snapshot(output_dir, source_bytes)
        source_topology, source_positions = _load_source(str(snapshot_path))
        _validate_source_against_decoy(source_topology, config)
        cached = _inspect_existing(
            output_dir,
            config,
            source_topology,
            source_positions,
            source_digest=source_digest,
            config_digest=config_digest,
            preparation_digest=preparation_digest,
        )
        if cached is not None:
            return cached
        if os.path.lexists(output_dir):
            return _failure(
                output_dir,
                source_digest,
                config_digest,
                preparation_digest,
                "output_dir already exists but is not an exact reusable bundle",
                status=ExecutionStatus.INCOMPLETE,
            )
        executed = True
        staging = _make_staging_dir(output_dir)
        try:
            state = _prepare_structure(
                config,
                source_topology,
                source_positions,
                source_path=str(snapshot_path),
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
            bundle = _bundle_from_manifest(
                output_dir,
                source_digest=source_digest,
                config_digest=config_digest,
                preparation_digest=preparation_digest,
                manifest=manifest,
                manifest_digest=_required_sha(staging / MANIFEST_FILENAME),
            )
            shutil.rmtree(snapshot_dir)
            snapshot_dir = None
            _publish_staging(staging, output_dir)
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
            executed=executed,
        )
    finally:
        if snapshot_dir is not None:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
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
        "algorithm_versions": ALGORITHM_VERSIONS,
        "topology": _descriptor_payload(config),
        "d_coordinate_input_mode": config.d_coordinate_input_mode,
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
            for item in sorted(descriptor.d_substitutions, key=lambda value: value.position)
        ],
        "head_to_tail": None
        if descriptor.head_to_tail is None
        else {
            "head": descriptor.head_to_tail.head,
            "tail": descriptor.head_to_tail.tail,
        },
        "disulfides": [
            {"first": first, "second": second}
            for first, second in sorted(
                (min(item.first, item.second), max(item.first, item.second))
                for item in descriptor.disulfides
            )
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


def _read_source_snapshot(path: Path) -> tuple[bytes, str]:
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"source_pdb is not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            data = handle.read()
    finally:
        if descriptor != -1:
            os.close(descriptor)
    if not data:
        raise ValueError(f"source_pdb is empty: {path}")
    return data, hashlib.sha256(data).hexdigest()


def _make_source_snapshot(output_dir: Path, source_bytes: bytes) -> tuple[Path, Path]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.source-", dir=output_dir.parent)
    )
    snapshot_path = snapshot_dir / "source.pdb"
    try:
        snapshot_path.write_bytes(source_bytes)
    except Exception:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        raise
    return snapshot_dir, snapshot_path


def _prepare_structure(
    config: ComplexPrepConfig,
    source_topology: object,
    source_positions: object,
    *,
    source_path: str,
    coordinate_transformer: CoordinateTransformer | None,
    chirality_validator: ChiralityValidator | None,
) -> _StructureState:
    topology, positions = _mutate_and_hydrate(config, source_path, source_topology)
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
                stage=_source_chirality_stage(config),
            )
        )
    positions = _prepare_d_coordinates(
        config,
        topology,
        positions,
        coordinate_transformer=coordinate_transformer,
        chirality_validator=chirality_validator,
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


def _source_chirality_stage(config: ComplexPrepConfig) -> str:
    return "source" if config.d_coordinate_input_mode == "prepared_d" else "post_h"


def _prepare_d_coordinates(
    config: ComplexPrepConfig,
    topology: object,
    positions: object,
    *,
    coordinate_transformer: CoordinateTransformer | None,
    chirality_validator: ChiralityValidator | None,
) -> object:
    if not config.topology.d_substitutions:
        return positions
    if chirality_validator is None:
        raise ValueError("D substitutions require a chirality validator callback")
    if config.d_coordinate_input_mode == "prepared_d":
        return positions
    if coordinate_transformer is None:
        raise ValueError("canonical-L D substitutions require a coordinate transformer callback")
    return design_chain.apply_d_coordinate_transform(
        topology,
        positions,
        config.design_sequence,
        config.topology,
        coordinate_transformer,
        design_chain_id=DESIGN_CHAIN_ID,
    )


def _mutate_and_hydrate(
    config: ComplexPrepConfig, source_path: str, source_topology: object
) -> tuple[object, object]:
    from pdbfixer import PDBFixer

    from biolab_runners.peptide_prep.mutation import apply_design_chain_mutation

    topology, positions = apply_design_chain_mutation(
        backbone_pdb_path=source_path,
        design_chain_id=DESIGN_CHAIN_ID,
        target_sequence=config.design_sequence,
    )
    fixer = PDBFixer(filename=source_path)
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
    return _remove_receptor_extras(fixer.topology, fixer.positions, source_topology)


def _remove_receptor_extras(
    topology: object, positions: object, source_topology: object
) -> tuple[object, object]:
    import openmm.app as app

    source_keys = {
        (*_residue_key(atom.residue), atom.name)
        for atom in source_topology.atoms()  # type: ignore[attr-defined]
        if atom.residue.chain.id in RECEPTOR_CHAIN_IDS
        and atom.element is not None
        and atom.element.symbol != "H"
    }
    extras = [
        atom
        for atom in topology.atoms()  # type: ignore[attr-defined]
        if atom.residue.chain.id in RECEPTOR_CHAIN_IDS
        and atom.element is not None
        and atom.element.symbol != "H"
        and (*_residue_key(atom.residue), atom.name) not in source_keys
    ]
    if not extras:
        return topology, positions
    modeller = app.Modeller(topology, positions)
    modeller.delete(extras)
    return modeller.topology, modeller.positions


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
    _validate_closure_geometry(modeller.topology, modeller.positions, bond_records)
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


def _validate_closure_geometry(
    topology: object, positions: object, bond_records: list[TopologyBondRecord]
) -> None:
    atoms = {atom.index: atom for atom in topology.atoms()}  # type: ignore[attr-defined]
    limits = {
        "head_to_tail": ("C", "N", DEFAULT_MAX_HEAD_TO_TAIL_DISTANCE_A),
        "disulfide": ("SG", "SG", DEFAULT_MAX_DISULFIDE_DISTANCE_A),
    }
    for record in bond_records:
        if record.bond_type not in limits:
            continue
        expected_first, expected_second, limit = limits[record.bond_type]
        actual_names = (atoms[record.atom1_index].name, atoms[record.atom2_index].name)
        if actual_names != (expected_first, expected_second):
            raise ValueError(
                f"{record.bond_type} closure does not connect the required terminal atoms"
            )
        distance_a = (
            math.dist(
                _position_at(positions, record.atom1_index),
                _position_at(positions, record.atom2_index),
            )
            * 10.0
        )
        if not math.isfinite(distance_a) or distance_a > limit:
            raise ValueError(
                f"{record.bond_type} closure distance {distance_a:.6f} Å exceeds "
                f"the physical limit of {limit:.6f} Å"
            )


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
    _validate_preserved_topologies(
        state.source_topology,
        state.source_positions,
        state.topology,
        state.positions,
    )


def _validate_preserved_topologies(
    source_topology: object,
    source_positions: object,
    prepared_topology: object,
    prepared_positions: object,
) -> None:
    source_receptor = _receptor_heavy_snapshot(source_topology, source_positions)
    prepared_receptor = _receptor_heavy_snapshot(prepared_topology, prepared_positions)
    if source_receptor != prepared_receptor:
        raise ValueError("receptor A/B heavy-atom identity or coordinates changed")
    source = _atom_records(source_topology, source_positions)
    prepared = _atom_records(prepared_topology, prepared_positions)
    for key, source_record in source.items():
        prepared_record = prepared.get(key)
        if prepared_record is None:
            continue
        if key[0] in RECEPTOR_CHAIN_IDS and source_record["element"] != "H":
            _require_same_geometry(key, source_record, prepared_record, "receptor")
        if key[0] == DESIGN_CHAIN_ID and key[3] in design_chain.D_BACKBONE_INVARIANT_ATOMS:
            _require_same_geometry(key, source_record, prepared_record, "design backbone")


def _receptor_heavy_snapshot(topology: object, positions: object) -> tuple[Any, ...]:
    chains: list[Any] = []
    for chain in topology.chains():  # type: ignore[attr-defined]
        if chain.id not in RECEPTOR_CHAIN_IDS:
            continue
        residues: list[Any] = []
        for residue in chain.residues():
            atoms = tuple(
                (atom.name, atom.element.symbol, _position_at(positions, atom.index))
                for atom in residue.atoms()
                if atom.element is not None and atom.element.symbol != "H"
            )
            residues.append((str(residue.id), residue.name, atoms))
        chains.append((chain.id, tuple(residues)))
    return tuple(chains)


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
    _rewrite_pdb_closure_records(pdb_path, state.bond_records)
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
        json.loads(map_path.read_text()),
        state.source_topology,
        state.source_positions,
        state.topology,
        state.positions,
        index_path,
        map_digest,
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
    _verify_staging(
        staging,
        manifest,
        state.source_topology,
        state.source_positions,
        state.topology,
        state.positions,
    )
    return manifest


def _grompp_status(
    export: object, top_path: Path, gro_path: Path, staging: Path
) -> tuple[bool, str]:
    import shutil as system_shutil

    present_before = system_shutil.which("gmx") is not None
    okay, _message = export.gmx_grompp_pp_check(  # type: ignore[attr-defined]
        top_path,
        gro_path,
        audit_workdir=staging / ".grompp_audit",
    )
    shutil.rmtree(staging / ".grompp_audit", ignore_errors=True)
    present_after = system_shutil.which("gmx") is not None
    return okay, GROMPP_PASSED if present_before and present_after else GROMPP_NOT_RUN


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
    if _topology_bond_pairs(parsed.topology) != _topology_bond_pairs(state.topology):
        raise ValueError("prepared.pdb bond graph differs from prepared topology")


def _restore_pdb_residue_identifiers(path: Path, topology: object) -> None:
    lines = path.read_text().splitlines()
    atoms = list(topology.atoms())  # type: ignore[attr-defined]
    atom_index = 0
    last_residue: object | None = None
    output: list[str] = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            if atom_index >= len(atoms):
                raise ValueError("prepared.pdb contains more atoms than its topology")
            last_residue = atoms[atom_index].residue
            _chain_id, number, insertion = _residue_key(last_residue)
            if len(insertion) > 1 or not -999 <= number <= 9999:
                raise ValueError(
                    "residue identifier cannot be represented in PDB fixed-width fields"
                )
            replacement = f"{number:4d}" + insertion
            output.append(line[:22] + replacement + line[27:])
            atom_index += 1
        elif line.startswith("TER") and last_residue is not None:
            _chain_id, number, insertion = _residue_key(last_residue)
            output.append(line[:22] + f"{number:4d}{insertion}" + line[27:])
        else:
            output.append(line)
    if atom_index != len(atoms):
        raise ValueError("prepared.pdb contains fewer atoms than its topology")
    path.write_text("\n".join(output) + "\n")


def _rewrite_pdb_closure_records(path: Path, bond_records: tuple[TopologyBondRecord, ...]) -> None:
    lines = path.read_text().splitlines()
    serials = [int(line[6:11]) for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    if any(
        not 0 <= index < len(serials)
        for record in bond_records
        for index in (record.atom1_index, record.atom2_index)
    ):
        raise ValueError("closure bond index cannot be represented in prepared.pdb")
    conect = [
        f"CONECT{serials[record.atom1_index]:5d}{serials[record.atom2_index]:5d}"
        for record in bond_records
    ]
    output = [line for line in lines if not line.startswith("CONECT") and line != "END"]
    path.write_text("\n".join([*output, *conect, "END"]) + "\n")


def _topology_bond_pairs(topology: object) -> set[tuple[int, int]]:
    return {
        (min(first.index, second.index), max(first.index, second.index))
        for first, second in topology.bonds()  # type: ignore[attr-defined]
    }


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
        elif atom.residue.chain.id == DESIGN_CHAIN_ID:
            groups["design_c"].append(atom.index + 1)
        else:
            raise ValueError(f"unexpected chain {atom.residue.chain.id!r} in index groups")
    groups["dimer_ab"] = list(groups["receptor_ab"])
    for group in groups.values():
        if group != sorted(set(group)) or not group:
            raise ValueError("index group is not ascending and unique")
    if groups["receptor_ab"] != groups["dimer_ab"]:
        raise ValueError("receptor_ab and dimer_ab selections differ")
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
    dropped = [
        _map_atom_record(source[key], "source") for key in sorted(set(source) - set(prepared))
    ]
    added = [
        _map_atom_record(prepared[key], "prepared") for key in sorted(set(prepared) - set(source))
    ]
    residue_common = sorted(set(source_residues) & set(prepared_residues))
    residue_mapped = [
        {
            "source": source_residues[key],
            "prepared": prepared_residues[key],
            "source_pdb_residue_index": source_residues[key]["residue_index"],
            "prepared_pdb_residue_index": prepared_residues[key]["residue_index"],
            "prepared_topology_residue_index": prepared_residues[key]["residue_index"],
        }
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
            _map_residue_record(prepared_residues[key], "prepared")
            for key in sorted(set(prepared_residues) - set(source_residues))
        ],
        "dropped_residues": [
            _map_residue_record(source_residues[key], "source")
            for key in sorted(set(source_residues) - set(prepared_residues))
        ],
        "selections": _selection_groups(state.topology),
        "interface_mapping": _interface_mapping(state.topology),
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


def _map_atom_record(record: dict[str, Any], namespace: str) -> dict[str, Any]:
    mapped = _identity_record(record)
    if namespace == "source":
        mapped.update(
            source_pdb_atom_index=record["atom_index"] + 1,
            source_pdb_residue_index=record["residue_index"] + 1,
        )
    else:
        mapped.update(
            prepared_pdb_atom_index=record["atom_index"] + 1,
            prepared_pdb_residue_index=record["residue_index"] + 1,
            prepared_topology_atom_index=record["atom_index"] + 1,
            prepared_topology_residue_index=record["residue_index"] + 1,
        )
    return mapped


def _map_residue_record(record: dict[str, Any], namespace: str) -> dict[str, Any]:
    mapped = dict(record)
    if namespace == "source":
        mapped["source_pdb_residue_index"] = record["residue_index"]
    else:
        mapped["prepared_pdb_residue_index"] = record["residue_index"]
        mapped["prepared_topology_residue_index"] = record["residue_index"]
    return mapped


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
    manifest = {
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
        "algorithm_versions": dict(ALGORITHM_VERSIONS),
        "closure_limits": {
            "head_to_tail_angstrom": DEFAULT_MAX_HEAD_TO_TAIL_DISTANCE_A,
            "disulfide_angstrom": DEFAULT_MAX_DISULFIDE_DISTANCE_A,
        },
        "topology": _descriptor_payload(config),
        "d_coordinate_input_mode": config.d_coordinate_input_mode,
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
    manifest[MANIFEST_PAYLOAD_DIGEST_FIELD] = _manifest_payload_digest(manifest)
    return manifest


def _manifest_payload_digest(manifest: dict[str, Any]) -> str:
    payload = {
        key: value for key, value in manifest.items() if key != MANIFEST_PAYLOAD_DIGEST_FIELD
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _verify_staging(
    staging: Path,
    manifest: dict[str, Any],
    source_topology: object,
    source_positions: object,
    prepared_topology: object,
    prepared_positions: object,
) -> None:
    _validate_manifest_shape(manifest)
    for filename in _OUTPUT_FILENAMES:
        path = staging / filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"staging artifact is missing, empty, or symlinked: {filename}")
        if _required_sha(path) != manifest["artifacts"][filename]["sha256"]:
            raise ValueError(f"staging artifact digest mismatch: {filename}")
    _validate_selection_map(
        json.loads((staging / SELECTION_MAP_FILENAME).read_text()),
        source_topology,
        source_positions,
        prepared_topology,
        prepared_positions,
        staging / INDEX_FILENAME,
        _required_sha(staging / SELECTION_MAP_FILENAME),
    )
    _required_sha(staging / MANIFEST_FILENAME)


def _publish_staging(staging: Path, output_dir: Path) -> None:
    try:
        _rename_no_replace(staging, output_dir)
    except FileExistsError as exc:
        raise ValueError("output_dir became occupied before atomic publish") from exc


def _rename_no_replace(source: Path, destination: Path) -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        == 0
    ):
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _required_sha(path: Path) -> str:
    digest = file_sha256(path)
    if digest is None or path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"required regular non-empty file unavailable: {path}")
    return digest


def _load_existing_manifest(
    output_dir: Path,
    source_digest: str,
    config_digest: str,
    preparation_digest: str,
) -> tuple[dict[str, Any] | None, ComplexPrepResult | None]:
    if not os.path.lexists(output_dir):
        return None, None
    if output_dir.is_symlink() or not output_dir.is_dir():
        return None, _failure(
            output_dir,
            source_digest,
            config_digest,
            preparation_digest,
            "output_dir is not a directory",
        )
    if not any(output_dir.iterdir()):
        return None, None
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return None, _failure(
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
        return None, _failure(
            output_dir,
            source_digest,
            config_digest,
            preparation_digest,
            f"manifest is malformed: {exc}",
            status=ExecutionStatus.MALFORMED,
        )
    if not isinstance(manifest, dict):
        return None, _failure(
            output_dir,
            source_digest,
            config_digest,
            preparation_digest,
            "manifest is malformed",
            status=ExecutionStatus.MALFORMED,
        )
    return manifest, None


def _inspect_existing(
    output_dir: Path,
    config: ComplexPrepConfig,
    source_topology: object,
    source_positions: object,
    *,
    source_digest: str,
    config_digest: str,
    preparation_digest: str,
) -> ComplexPrepResult | None:
    manifest, early_result = _load_existing_manifest(
        output_dir, source_digest, config_digest, preparation_digest
    )
    if early_result is not None:
        return early_result
    if manifest is None:
        return None
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
        _restore_cached_disulfide_names(prepared.topology, config)
        _validate_selection_map(
            map_data,
            source_topology,
            source_positions,
            prepared.topology,
            prepared.positions,
            output_dir / INDEX_FILENAME,
            _required_sha(output_dir / SELECTION_MAP_FILENAME),
        )
        if (
            manifest["chain_audits"] != map_data["chain_audits"]
            or manifest["residue_audits"] != map_data["residue_audits"]
        ):
            raise ValueError("cached manifest audits differ from selection-map audits")
        if manifest["atom_count"] != prepared.topology.getNumAtoms():
            raise ValueError("cached manifest atom count differs from prepared topology")
        _validate_cached_science(
            output_dir,
            config,
            manifest,
            source_topology,
            source_positions,
            prepared.topology,
            prepared.positions,
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


def _validate_cached_science(
    output_dir: Path,
    config: ComplexPrepConfig,
    manifest: dict[str, Any],
    source_topology: object,
    source_positions: object,
    prepared_topology: object,
    prepared_positions: object,
) -> None:
    from biolab_runners.peptide_prep import export

    _validate_preserved_topologies(
        source_topology,
        source_positions,
        prepared_topology,
        prepared_positions,
    )
    records = _validate_cached_bond_records(config, manifest, prepared_topology)
    _validate_closure_geometry(prepared_topology, prepared_positions, list(records))
    system, net_charge = _build_system(prepared_topology)
    if not math.isclose(net_charge, manifest["net_charge"], rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("cached manifest net charge differs from prepared topology")
    parity_ok, parity_message = export.verify_export_parity(
        prepared_topology,
        system,
        prepared_positions,
        top_path=output_dir / PREPARED_TOP_FILENAME,
        gro_path=output_dir / PREPARED_GRO_FILENAME,
        no_nan=_positions_are_finite(prepared_positions),
    )
    if not parity_ok and parity_message.startswith("atom count mismatch:"):
        parity_ok, parity_message = _verify_multichain_parity(
            prepared_topology,
            output_dir / PREPARED_TOP_FILENAME,
            output_dir / PREPARED_GRO_FILENAME,
        )
    if not parity_ok:
        raise ValueError(f"cached GROMACS parity check failed: {parity_message}")


def _restore_cached_disulfide_names(topology: object, config: ComplexPrepConfig) -> None:
    _, residues = design_chain.resolve_design_chain(
        topology,
        DESIGN_CHAIN_ID,
        expected_length=len(config.design_sequence),
    )
    positions = {
        position for bond in config.topology.disulfides for position in (bond.first, bond.second)
    }
    for position in positions:
        residue = residues[position - 1]
        if residue.name != "CYS":
            raise ValueError("cached disulfide residue is not CYS in prepared.pdb")
        residue.name = "CYX"


def _validate_cached_bond_records(
    config: ComplexPrepConfig,
    manifest: dict[str, Any],
    topology: object,
) -> tuple[TopologyBondRecord, ...]:
    expected = _expected_closure_records(config, topology)
    actual = tuple(TopologyBondRecord(**record) for record in manifest["bond_records"])
    if sorted(_canonical_json(asdict(record)) for record in actual) != sorted(
        _canonical_json(asdict(record)) for record in expected
    ):
        raise ValueError("cached manifest bond records differ from requested topology")
    topology_pairs = _topology_bond_pairs(topology)
    if any(
        (min(record.atom1_index, record.atom2_index), max(record.atom1_index, record.atom2_index))
        not in topology_pairs
        for record in actual
    ):
        raise ValueError("cached manifest closure bond is absent from prepared.pdb")
    return actual


def _expected_closure_records(
    config: ComplexPrepConfig, topology: object
) -> tuple[TopologyBondRecord, ...]:
    _, residues = design_chain.resolve_design_chain(
        topology,
        DESIGN_CHAIN_ID,
        expected_length=len(config.design_sequence),
    )
    records: list[TopologyBondRecord] = []
    if config.topology.head_to_tail is not None:
        head = residues[config.topology.head_to_tail.head - 1]
        tail = residues[config.topology.head_to_tail.tail - 1]
        tail_atom = _required_residue_atom(tail, "C")
        head_atom = _required_residue_atom(head, "N")
        records.append(
            _bond_record(
                topology,
                tail_atom.index,
                head_atom.index,
                tail.index,
                head.index,
                "head_to_tail",
            )
        )
    disulfide_pairs = tuple(
        (residues[bond.first - 1].index, residues[bond.second - 1].index)
        for bond in config.topology.disulfides
    )
    records.extend(_disulfide_records(topology, disulfide_pairs))
    return tuple(records)


def _required_residue_atom(residue: object, atom_name: str) -> object:
    atoms = [atom for atom in residue.atoms() if atom.name == atom_name]  # type: ignore[attr-defined]
    if len(atoms) != 1:
        raise ValueError(
            f"prepared topology residue {residue.index} lacks one unique {atom_name} atom"  # type: ignore[attr-defined]
        )
    return atoms[0]


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
    _validate_manifest_shape(manifest)
    for key, expected in (
        ("source_pdb_sha256", source_digest),
        ("config_digest", config_digest),
        ("preparation_digest", preparation_digest),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"cached bundle is stale: manifest {key} does not match request")
    _validate_manifest_request_values(manifest, config)


def _validate_manifest_shape(manifest: dict[str, Any]) -> None:
    if (
        not _strict_int(manifest.get("schema_version"))
        or manifest.get("schema_version") != PREPARATION_SCHEMA_VERSION
        or manifest.get("runner") != "complex_prep"
    ):
        raise MalformedBundleError("manifest schema is unsupported")
    if not _is_sha(manifest.get(MANIFEST_PAYLOAD_DIGEST_FIELD)):
        raise MalformedBundleError("manifest scientific metadata digest is malformed")
    if manifest.get(MANIFEST_PAYLOAD_DIGEST_FIELD) != _manifest_payload_digest(manifest):
        raise ValueError("cached manifest scientific metadata digest does not match payload")
    _validate_manifest_metadata_types(manifest)


def _validate_manifest_request_values(manifest: dict[str, Any], config: ComplexPrepConfig) -> None:
    if manifest["source_pdb"] != config.source_pdb:
        raise ValueError("cached bundle source path does not match request")
    if manifest["design_sequence"] != config.design_sequence or manifest[
        "topology"
    ] != _descriptor_payload(config):
        raise ValueError("cached bundle config descriptor does not match request")
    if manifest["d_coordinate_input_mode"] != config.d_coordinate_input_mode:
        raise ValueError("cached bundle D-coordinate input mode does not match request")
    if manifest["source_decoy"] != config.source_decoy.to_dict():
        raise ValueError("cached bundle source decoy identity does not match request")
    if manifest["chain_roles"] != {
        "receptor": list(RECEPTOR_CHAIN_IDS),
        "design": DESIGN_CHAIN_ID,
    }:
        raise MalformedBundleError("manifest chain roles are malformed")
    if manifest["force_field"] != {
        "protein": PROTEIN_FORCE_FIELD,
        "water": WATER_FORCE_FIELD,
        "pH": 7.4,
    }:
        raise MalformedBundleError("manifest force-field policy is malformed")
    if manifest["algorithm_versions"] != ALGORITHM_VERSIONS:
        raise MalformedBundleError("manifest algorithm versions are stale")
    if manifest["closure_limits"] != {
        "head_to_tail_angstrom": DEFAULT_MAX_HEAD_TO_TAIL_DISTANCE_A,
        "disulfide_angstrom": DEFAULT_MAX_DISULFIDE_DISTANCE_A,
    }:
        raise MalformedBundleError("manifest closure limits are stale")


def _validate_manifest_metadata_types(manifest: dict[str, Any]) -> None:
    for key in ("source_pdb_sha256", "config_digest", "preparation_digest"):
        if not _is_sha(manifest.get(key)):
            raise MalformedBundleError(f"manifest digest is malformed for {key}")
    if not isinstance(manifest.get(MANIFEST_PAYLOAD_DIGEST_FIELD), str):
        raise MalformedBundleError("manifest scientific metadata digest is malformed")
    if not isinstance(manifest.get("source_pdb"), str) or not isinstance(
        manifest.get("output_dir"), str
    ):
        raise MalformedBundleError("manifest path metadata is malformed")
    if not isinstance(manifest.get("design_sequence"), str):
        raise MalformedBundleError("manifest design sequence is malformed")
    if manifest.get("d_coordinate_input_mode") not in {"canonical_l", "prepared_d"}:
        raise MalformedBundleError("manifest D-coordinate input mode is malformed")
    if not isinstance(manifest.get("source_decoy"), dict) or not isinstance(
        manifest.get("topology"), dict
    ):
        raise MalformedBundleError("manifest source or topology metadata is malformed")
    _validate_manifest_policies(manifest)
    _validate_manifest_artifact_specs(manifest.get("artifacts"))
    _validate_chain_audit_types(manifest.get("chain_audits"))
    _validate_residue_audit_types(manifest.get("residue_audits"))
    _validate_bond_record_types(manifest.get("bond_records"))
    _validate_chirality_report_types(manifest.get("chirality_reports"))


def _validate_manifest_policies(manifest: dict[str, Any]) -> None:
    versions = manifest.get("algorithm_versions")
    if not isinstance(versions, dict) or set(versions) != set(ALGORITHM_VERSIONS):
        raise MalformedBundleError("manifest algorithm versions are malformed")
    if any(not isinstance(value, str) for value in versions.values()):
        raise MalformedBundleError("manifest algorithm versions are malformed")
    limits = manifest.get("closure_limits")
    if not isinstance(limits, dict) or set(limits) != {
        "head_to_tail_angstrom",
        "disulfide_angstrom",
    }:
        raise MalformedBundleError("manifest closure limits are malformed")
    if any(not _finite_number(value) for value in limits.values()):
        raise MalformedBundleError("manifest closure limits are malformed")
    if manifest.get("grompp_audit_status") not in {GROMPP_NOT_RUN, GROMPP_PASSED}:
        raise MalformedBundleError("manifest grompp status is malformed")
    if not _finite_number(manifest.get("net_charge")):
        raise MalformedBundleError("manifest net charge is malformed")
    atom_count = manifest.get("atom_count")
    if not isinstance(atom_count, int) or isinstance(atom_count, bool) or atom_count <= 0:
        raise MalformedBundleError("manifest atom count is malformed")
    if manifest.get("solvent_ion_boundaries") != {
        "solvent": "not_staged",
        "ions": "not_staged",
    }:
        raise MalformedBundleError("manifest solvent/ion policy is malformed")


def _validate_manifest_artifact_specs(artifacts: object) -> None:
    if not isinstance(artifacts, dict) or set(artifacts) != set(_OUTPUT_FILENAMES):
        raise MalformedBundleError("manifest artifact list is malformed")
    for filename, spec in artifacts.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            raise MalformedBundleError(f"manifest artifact spec is malformed for {filename}")
        if not _is_sha(spec.get("sha256")):
            raise MalformedBundleError(f"manifest artifact digest is malformed for {filename}")


def _validate_chain_audit_types(value: object) -> None:
    fields = {
        "chain_id",
        "role",
        "source_residue_count",
        "prepared_residue_count",
        "source_atom_count",
        "prepared_atom_count",
    }
    if not isinstance(value, list):
        raise MalformedBundleError("manifest chain audits are malformed")
    for audit in value:
        if not isinstance(audit, dict) or set(audit) != fields:
            raise MalformedBundleError("manifest chain audits are malformed")
        if not isinstance(audit["chain_id"], str) or not isinstance(audit["role"], str):
            raise MalformedBundleError("manifest chain audits are malformed")
        if any(not _strict_int(audit[key]) for key in fields - {"chain_id", "role"}):
            raise MalformedBundleError("manifest chain audits are malformed")


def _validate_residue_audit_types(value: object) -> None:
    fields = {
        "chain_id",
        "residue_number",
        "insertion_code",
        "source_name",
        "prepared_name",
        "source_atom_count",
        "prepared_atom_count",
        "source_index",
        "prepared_index",
    }
    if not isinstance(value, list):
        raise MalformedBundleError("manifest residue audits are malformed")
    for audit in value:
        if not isinstance(audit, dict) or set(audit) != fields:
            raise MalformedBundleError("manifest residue audits are malformed")
        if any(
            not isinstance(audit[key], str)
            for key in ("chain_id", "insertion_code", "source_name", "prepared_name")
        ):
            raise MalformedBundleError("manifest residue audits are malformed")
        if any(
            not _strict_int(audit[key])
            for key in fields - {"chain_id", "insertion_code", "source_name", "prepared_name"}
        ):
            raise MalformedBundleError("manifest residue audits are malformed")


def _validate_bond_record_types(value: object) -> None:
    fields = {
        "atom1_index",
        "atom2_index",
        "bond_type",
        "atom1_name",
        "atom2_name",
        "residue1_index",
        "residue2_index",
    }
    if not isinstance(value, list):
        raise MalformedBundleError("manifest bond records are malformed")
    for record in value:
        if not isinstance(record, dict) or set(record) != fields:
            raise MalformedBundleError("manifest bond records are malformed")
        if any(
            not isinstance(record[key], str) for key in ("bond_type", "atom1_name", "atom2_name")
        ):
            raise MalformedBundleError("manifest bond records are malformed")
        if any(
            not _strict_int(record[key])
            for key in fields - {"bond_type", "atom1_name", "atom2_name"}
        ):
            raise MalformedBundleError("manifest bond records are malformed")


def _validate_chirality_report_types(value: object) -> None:
    fields = {"residue_index", "residue_name", "expected", "observed", "valid", "detail"}
    if not isinstance(value, list):
        raise MalformedBundleError("manifest chirality reports are malformed")
    for stage in value:
        _validate_chirality_stage(stage, fields)


def _validate_chirality_stage(stage: object, fields: set[str]) -> None:
    if not isinstance(stage, list):
        raise MalformedBundleError("manifest chirality reports are malformed")
    for report in stage:
        _validate_chirality_report(report, fields)


def _validate_chirality_report(report: object, fields: set[str]) -> None:
    if not isinstance(report, dict) or set(report) != fields:
        raise MalformedBundleError("manifest chirality reports are malformed")
    if not _strict_int(report["residue_index"]) or not isinstance(report["valid"], bool):
        raise MalformedBundleError("manifest chirality reports are malformed")
    if any(not isinstance(report[key], str) for key in fields - {"residue_index", "valid"}):
        raise MalformedBundleError("manifest chirality reports are malformed")


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_manifest_files(output_dir: Path, manifest: dict[str, Any]) -> None:
    if manifest["output_dir"] != str(output_dir):
        raise ValueError("manifest output directory binding is stale")
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
    source_topology: object,
    source_positions: object,
    prepared_topology: object,
    prepared_positions: object,
    index_path: Path,
    map_digest: str,
) -> None:
    data = _validated_selection_map_header(data)
    if data.get("index_sha256") != _required_sha(index_path):
        raise ValueError("selection map/index digest binding is stale")
    _validate_index_groups(
        data.get("selections"),
        index_path,
        prepared_topology.getNumAtoms(),  # type: ignore[attr-defined]
    )
    if data.get("selections") != _selection_groups(prepared_topology):
        raise ValueError("selection map groups differ from prepared topology roles")
    _validate_interface_mapping(data, prepared_topology)
    source_atoms = _atom_records(source_topology, source_positions)
    prepared_atoms = _atom_records(prepared_topology, prepared_positions)
    _validate_atom_mapping(data, source_atoms, prepared_atoms)
    source_residues = _residue_records(source_topology)
    prepared_residues = _residue_records(prepared_topology)
    _validate_residue_mapping(data, source_residues, prepared_residues)
    if data.get("chain_audits") != [
        audit.to_dict() for audit in _build_chain_audits(source_topology, prepared_topology)
    ]:
        raise ValueError("selection map chain audits differ from topologies")
    if data.get("residue_audits") != [
        audit.to_dict() for audit in _build_residue_audits(source_topology, prepared_topology)
    ]:
        raise ValueError("selection map residue audits differ from topologies")
    prepared_digests = data.get("prepared_artifact_sha256")
    if not isinstance(prepared_digests, dict) or set(prepared_digests) != set(
        _OUTPUT_FILENAMES[:3]
    ):
        raise MalformedBundleError("selection map lacks prepared artifact bindings")
    if any(not _is_sha(value) for value in prepared_digests.values()):
        raise MalformedBundleError("selection map prepared artifact digests are malformed")
    if not _is_sha(map_digest):
        raise MalformedBundleError("selection map digest is malformed")
    if data.get("solvent_ion_boundaries") != {
        "solvent": "not_staged",
        "ions": "not_staged",
    }:
        raise MalformedBundleError("selection map solvent/ion policy is malformed")


def _validated_selection_map_header(data: object) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise MalformedBundleError("selection map schema is malformed")
    if data.get("index_bases") != {
        "source_pdb_atom": 1,
        "source_pdb_residue": 1,
        "prepared_pdb_atom": 1,
        "prepared_pdb_residue": 1,
        "prepared_topology_atom": 1,
        "prepared_topology_residue": 1,
    }:
        raise MalformedBundleError("selection map index bases are malformed")
    if not _is_sha(data.get("preparation_digest")) or not _is_sha(data.get("source_pdb_sha256")):
        raise MalformedBundleError("selection map identity digests are malformed")
    return data


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


def _validate_atom_mapping(
    data: dict[str, Any],
    source_atoms: dict[tuple[str, int, str, str], dict[str, Any]],
    prepared_atoms: dict[tuple[str, int, str, str], dict[str, Any]],
) -> None:
    entries = data.get("source_to_prepared_atoms")
    added = data.get("added_atoms")
    dropped = data.get("dropped_atoms")
    if not all(isinstance(value, list) for value in (entries, added, dropped)):
        raise MalformedBundleError("selection map atom mappings are malformed")
    if not isinstance(entries, list) or not isinstance(added, list):
        raise MalformedBundleError("selection map atom mappings are malformed")
    mapped_source: set[tuple[str, int, str, str]] = set()
    mapped_prepared: set[tuple[str, int, str, str]] = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("source"), dict)
            or not isinstance(entry.get("prepared"), dict)
        ):
            raise MalformedBundleError("selection map contains malformed mapped atom")
        source_key = _validate_atom_map_side(entry["source"], source_atoms, "source")
        prepared_key = _validate_atom_map_side(entry["prepared"], prepared_atoms, "prepared")
        if source_key != prepared_key:
            raise ValueError("selection map mapped atom identities do not correspond")
        if source_key in mapped_source or prepared_key in mapped_prepared:
            raise ValueError("selection map atom mapping is not bijective")
        _validate_mapped_atom_indices(entry, source_atoms[source_key], prepared_atoms[prepared_key])
        mapped_source.add(source_key)
        mapped_prepared.add(prepared_key)
    dropped_keys = _validate_partition_atoms(dropped, source_atoms, "source", mapped_source)
    added_keys = _validate_partition_atoms(added, prepared_atoms, "prepared", mapped_prepared)
    if mapped_source | dropped_keys != set(source_atoms):
        raise ValueError("selection map does not cover exactly the source atoms")
    if mapped_prepared | added_keys != set(prepared_atoms):
        raise ValueError("selection map does not cover exactly the prepared atoms")


def _validate_atom_map_side(
    record: object,
    actual: dict[tuple[str, int, str, str], dict[str, Any]],
    namespace: str,
) -> tuple[str, int, str, str]:
    if not isinstance(record, dict):
        raise MalformedBundleError(f"selection map {namespace} atom identity is malformed")
    _validate_atom_identity_shape(record, namespace)
    key = (
        record["chain_id"],
        record["residue_number"],
        record["insertion_code"],
        record["atom_name"],
    )
    expected = actual.get(key)
    if expected is None or record["element"] != expected["element"]:
        raise ValueError(f"selection map {namespace} atom identity differs from topology")
    if (
        record["atom_index"] != expected["atom_index"] + 1
        or record["residue_index"] != expected["residue_index"] + 1
    ):
        raise ValueError(f"selection map {namespace} atom index does not match topology")
    return key


def _validate_atom_identity_shape(record: dict[str, Any], namespace: str) -> None:
    fields = ("chain_id", "residue_number", "insertion_code", "atom_name", "element")
    if any(not isinstance(record.get(field), str) for field in fields if field != "residue_number"):
        raise MalformedBundleError(f"selection map {namespace} atom identity is malformed")
    if not _strict_int(record.get("residue_number")):
        raise MalformedBundleError(f"selection map {namespace} atom identity is malformed")
    if not _strict_int(record.get("atom_index")) or not _strict_int(record.get("residue_index")):
        raise MalformedBundleError(f"selection map {namespace} atom indices are malformed")


def _validate_mapped_atom_indices(
    entry: dict[str, Any],
    source: dict[str, Any],
    prepared: dict[str, Any],
) -> None:
    expected = {
        "source_pdb_atom_index": source["atom_index"] + 1,
        "source_pdb_residue_index": source["residue_index"] + 1,
        "prepared_pdb_atom_index": prepared["atom_index"] + 1,
        "prepared_pdb_residue_index": prepared["residue_index"] + 1,
        "prepared_topology_atom_index": prepared["atom_index"] + 1,
        "prepared_topology_residue_index": prepared["residue_index"] + 1,
    }
    for field, expected_value in expected.items():
        if not _strict_int(entry.get(field)) or entry[field] != expected_value:
            raise ValueError(f"selection map mapped atom index is stale: {field}")


def _validate_partition_atoms(
    records: object,
    actual: dict[tuple[str, int, str, str], dict[str, Any]],
    namespace: str,
    excluded: set[tuple[str, int, str, str]],
) -> set[tuple[str, int, str, str]]:
    if not isinstance(records, list):
        raise MalformedBundleError(f"selection map {namespace} atom partition is malformed")
    result: set[tuple[str, int, str, str]] = set()
    for record in records:
        key = _validate_atom_map_side(record, actual, namespace)
        if key in excluded or key in result:
            raise ValueError(f"selection map {namespace} atom partition is not disjoint")
        _validate_partition_atom_indices(record, actual[key], namespace)
        result.add(key)
    return result


def _validate_partition_atom_indices(
    record: dict[str, Any], actual: dict[str, Any], namespace: str
) -> None:
    fields = (
        (
            ("source_pdb_atom_index", "atom_index"),
            ("source_pdb_residue_index", "residue_index"),
        )
        if namespace == "source"
        else (
            ("prepared_pdb_atom_index", "atom_index"),
            ("prepared_pdb_residue_index", "residue_index"),
            ("prepared_topology_atom_index", "atom_index"),
            ("prepared_topology_residue_index", "residue_index"),
        )
    )
    for field, actual_field in fields:
        if not _strict_int(record.get(field)) or record[field] != actual[actual_field] + 1:
            raise ValueError(f"selection map partition index is stale: {field}")


def _validate_residue_mapping(
    data: dict[str, Any],
    source_residues: dict[tuple[str, int, str], dict[str, Any]],
    prepared_residues: dict[tuple[str, int, str], dict[str, Any]],
) -> None:
    entries = data.get("source_to_prepared_residues")
    added = data.get("added_residues")
    dropped = data.get("dropped_residues")
    if not all(isinstance(value, list) for value in (entries, added, dropped)):
        raise MalformedBundleError("selection map residue mappings are malformed")
    if not isinstance(entries, list) or not isinstance(added, list):
        raise MalformedBundleError("selection map residue mappings are malformed")
    mapped_source, mapped_prepared = _validate_mapped_residue_entries(
        entries, source_residues, prepared_residues
    )
    dropped_keys = _validate_partition_residues(dropped, source_residues, "source", mapped_source)
    added_keys = _validate_partition_residues(added, prepared_residues, "prepared", mapped_prepared)
    if mapped_source | dropped_keys != set(source_residues):
        raise ValueError("selection map does not cover exactly the source residues")
    if mapped_prepared | added_keys != set(prepared_residues):
        raise ValueError("selection map does not cover exactly the prepared residues")


def _validate_mapped_residue_entries(
    entries: list[object],
    source_residues: dict[tuple[str, int, str], dict[str, Any]],
    prepared_residues: dict[tuple[str, int, str], dict[str, Any]],
) -> tuple[set[tuple[str, int, str]], set[tuple[str, int, str]]]:
    mapped_source: set[tuple[str, int, str]] = set()
    mapped_prepared: set[tuple[str, int, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise MalformedBundleError("selection map contains malformed mapped residue")
        source = entry.get("source")
        prepared = entry.get("prepared")
        if not isinstance(source, dict) or not isinstance(prepared, dict):
            raise MalformedBundleError("selection map contains malformed mapped residue")
        source_key = _validate_residue_map_side(source, source_residues, "source")
        prepared_key = _validate_residue_map_side(prepared, prepared_residues, "prepared")
        if source_key != prepared_key:
            raise ValueError("selection map mapped residue identities do not correspond")
        if source_key in mapped_source or prepared_key in mapped_prepared:
            raise ValueError("selection map residue mapping is not bijective")
        _validate_mapped_residue_indices(
            entry, source_residues[source_key], prepared_residues[prepared_key]
        )
        mapped_source.add(source_key)
        mapped_prepared.add(prepared_key)
    return mapped_source, mapped_prepared


def _validate_residue_map_side(
    record: object,
    actual: dict[tuple[str, int, str], dict[str, Any]],
    namespace: str,
) -> tuple[str, int, str]:
    if not isinstance(record, dict):
        raise MalformedBundleError(f"selection map {namespace} residue identity is malformed")
    fields = ("chain_id", "residue_number", "insertion_code", "residue_name")
    if any(not isinstance(record.get(field), str) for field in fields if field != "residue_number"):
        raise MalformedBundleError(f"selection map {namespace} residue identity is malformed")
    if not _strict_int(record.get("residue_number")) or not _strict_int(
        record.get("residue_index")
    ):
        raise MalformedBundleError(f"selection map {namespace} residue identity is malformed")
    key = (record["chain_id"], record["residue_number"], record["insertion_code"])
    expected = actual.get(key)
    if expected is None or record["residue_name"] != expected["residue_name"]:
        raise ValueError(f"selection map {namespace} residue identity differs from topology")
    if record["residue_index"] != expected["residue_index"]:
        raise ValueError(f"selection map {namespace} residue index does not match topology")
    return key


def _validate_mapped_residue_indices(
    entry: dict[str, Any], source: dict[str, Any], prepared: dict[str, Any]
) -> None:
    expected = {
        "source_pdb_residue_index": source["residue_index"],
        "prepared_pdb_residue_index": prepared["residue_index"],
        "prepared_topology_residue_index": prepared["residue_index"],
    }
    for field, expected_value in expected.items():
        if not _strict_int(entry.get(field)) or entry[field] != expected_value:
            raise ValueError(f"selection map mapped residue index is stale: {field}")


def _validate_partition_residues(
    records: object,
    actual: dict[tuple[str, int, str], dict[str, Any]],
    namespace: str,
    excluded: set[tuple[str, int, str]],
) -> set[tuple[str, int, str]]:
    if not isinstance(records, list):
        raise MalformedBundleError(f"selection map {namespace} residue partition is malformed")
    result: set[tuple[str, int, str]] = set()
    for record in records:
        key = _validate_residue_map_side(record, actual, namespace)
        if key in excluded or key in result:
            raise ValueError(f"selection map {namespace} residue partition is not disjoint")
        _validate_partition_residue_indices(record, actual[key], namespace)
        result.add(key)
    return result


def _validate_partition_residue_indices(
    record: dict[str, Any], actual: dict[str, Any], namespace: str
) -> None:
    fields = (
        (("source_pdb_residue_index", "residue_index"),)
        if namespace == "source"
        else (
            ("prepared_pdb_residue_index", "residue_index"),
            ("prepared_topology_residue_index", "residue_index"),
        )
    )
    for field, actual_field in fields:
        if not _strict_int(record.get(field)) or record[field] != actual[actual_field]:
            raise ValueError(f"selection map partition index is stale: {field}")


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


def _interface_mapping(topology: object) -> dict[str, list[int]]:
    mapping = {name: [] for name in ("receptor_a", "receptor_b", "dimer_ab", "design_c")}
    for atom in topology.atoms():  # type: ignore[attr-defined]
        chain_id = atom.residue.chain.id
        atom_index = atom.index + 1
        if chain_id == RECEPTOR_CHAIN_IDS[0]:
            mapping["receptor_a"].append(atom_index)
        elif chain_id == RECEPTOR_CHAIN_IDS[1]:
            mapping["receptor_b"].append(atom_index)
        elif chain_id == DESIGN_CHAIN_ID:
            mapping["design_c"].append(atom_index)
        else:
            raise ValueError(f"unexpected chain {chain_id!r} in interface mapping")
    mapping["dimer_ab"] = sorted(mapping["receptor_a"] + mapping["receptor_b"])
    _validate_interface_mapping_values(mapping)
    return mapping


def _validate_interface_mapping(data: dict[str, Any], prepared_topology: object) -> None:
    atom_count = prepared_topology.getNumAtoms()  # type: ignore[attr-defined]
    mapping = _normalized_interface_mapping(data.get("interface_mapping"), atom_count)
    expected = _interface_mapping(prepared_topology)
    if mapping != expected:
        raise ValueError("interface mapping differs from prepared topology chain roles")
    selections = data.get("selections")
    if not isinstance(selections, dict):
        raise ValueError("selection map selections are malformed")
    if mapping["dimer_ab"] != selections.get("receptor_ab") or mapping["dimer_ab"] != (
        selections.get("dimer_ab")
    ):
        raise ValueError("interface mapping dimer_ab differs from receptor selections")
    if mapping["design_c"] != selections.get("design_c"):
        raise ValueError("interface mapping design_c differs from design selection")


def _normalized_interface_mapping(value: object, atom_count: int) -> dict[str, list[int]]:
    names = ("receptor_a", "receptor_b", "dimer_ab", "design_c")
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("selection map interface_mapping groups are malformed")
    return {name: _validated_interface_group(value.get(name), name, atom_count) for name in names}


def _validated_interface_group(value: object, name: str, atom_count: int) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"interface mapping {name} must be a list")
    if not value or any(not _valid_index(item, atom_count) for item in value):
        raise ValueError(f"interface mapping {name} contains an invalid atom index")
    if value != sorted(set(value)):
        raise ValueError(f"interface mapping {name} must be sorted and unique")
    return list(value)


def _validate_interface_mapping_values(mapping: dict[str, list[int]]) -> None:
    receptor_a = mapping["receptor_a"]
    receptor_b = mapping["receptor_b"]
    dimer_ab = mapping["dimer_ab"]
    design_c = mapping["design_c"]
    if set(receptor_a) & set(receptor_b):
        raise ValueError("interface mapping receptor_a and receptor_b overlap")
    if dimer_ab != sorted(receptor_a + receptor_b):
        raise ValueError("interface mapping dimer_ab is not the sorted A+B union")
    if set(dimer_ab) & set(design_c):
        raise ValueError("interface mapping receptor dimer overlaps design C")


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
    return _bundle_from_manifest(
        output_dir,
        source_digest=source_digest,
        config_digest=config_digest,
        preparation_digest=preparation_digest,
        manifest=manifest,
        manifest_digest=_required_sha(output_dir / MANIFEST_FILENAME),
    )


def _bundle_from_manifest(
    output_dir: Path,
    *,
    source_digest: str,
    config_digest: str,
    preparation_digest: str,
    manifest: dict[str, Any],
    manifest_digest: str,
) -> ComplexPrepBundle:
    references = {
        filename: ArtifactReference(
            str(output_dir / filename),
            digest=manifest_digest
            if filename == MANIFEST_FILENAME
            else manifest["artifacts"][filename]["sha256"],
            kind="structure"
            if filename == PREPARED_PDB_FILENAME
            else "topology"
            if filename == PREPARED_TOP_FILENAME
            else "coordinates"
            if filename == PREPARED_GRO_FILENAME
            else "metadata",
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
    executed: bool = False,
) -> ComplexPrepResult:
    return ComplexPrepResult(
        status=status,
        output_dir=str(output_dir),
        source_digest=source_digest,
        config_digest=config_digest,
        preparation_digest=preparation_digest,
        error=error,
        executed=executed,
    )
