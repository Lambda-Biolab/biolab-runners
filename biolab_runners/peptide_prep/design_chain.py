"""Reusable chain-local coordinate and chirality seams."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from biolab_runners.peptide_prep.protocols import ChiralityReport, extract_coordinate_mapping
from biolab_runners.peptide_prep.utils import THREE_LETTER, collect_atom_mapping, distance

if TYPE_CHECKING:
    from biolab_runners.peptide_prep.config import PeptideTopologyDescriptor
    from biolab_runners.peptide_prep.protocols import ChiralityValidator, CoordinateTransformer

__all__ = [
    "D_BACKBONE_INVARIANT_ATOMS",
    "D_TRANSFORM_BACKBONE_TOLERANCE_A",
    "apply_d_coordinate_transform",
    "resolve_design_chain",
    "run_chirality_validation",
    "three_letter_for",
    "verify_d_backbone_invariance",
    "verify_d_mapping_complete",
]

D_BACKBONE_INVARIANT_ATOMS = ("N", "CA", "C")
D_TRANSFORM_BACKBONE_TOLERANCE_A = 1e-3


def resolve_design_chain(
    topology: object,
    design_chain_id: str,
    *,
    expected_length: int,
) -> tuple[Any, list[Any]]:
    """Resolve one unique chain and verify its residue count."""
    chains = [chain for chain in topology.chains() if chain.id == design_chain_id]  # type: ignore[attr-defined]
    if not chains:
        raise ValueError(f"design chain {design_chain_id!r} not found in topology")
    if len(chains) != 1:
        raise ValueError(
            f"design chain {design_chain_id!r} is ambiguous in topology; "
            f"found {len(chains)} matching chains"
        )
    residues = list(chains[0].residues())
    if len(residues) != expected_length:
        raise ValueError(
            f"design chain {design_chain_id!r} has {len(residues)} residues but "
            f"sequence has {expected_length}; residue count must equal sequence length"
        )
    return chains[0], residues


def apply_d_coordinate_transform(
    topology: object,
    positions: object,
    sequence: str,
    topology_descriptor: PeptideTopologyDescriptor,
    transformer: CoordinateTransformer,
    *,
    design_chain_id: str,
) -> object:
    """Apply D transforms using only one chain and chain-local positions."""
    _, design_residues = resolve_design_chain(
        topology,
        design_chain_id,
        expected_length=len(sequence),
    )
    atoms = list(topology.atoms())  # type: ignore[attr-defined]
    positions_nm = [_position_to_nm(position) for position in positions]  # type: ignore[union-attr]

    for entry in topology_descriptor.d_substitutions:
        local_index = entry.position - 1
        residue = design_residues[local_index]
        mapping = collect_atom_mapping(topology, positions, residue.index)
        verify_d_backbone_invariance(mapping, local_index)
        transformed = extract_coordinate_mapping(transformer(mapping, entry.residue, local_index))
        verify_d_backbone_invariance(transformed, local_index, before=mapping)
        verify_d_mapping_complete(mapping, transformed, local_index)
        for atom in atoms:
            if atom.residue is residue and atom.name in transformed:
                tx, ty, tz = transformed[atom.name]
                positions_nm[atom.index] = (tx / 10.0, ty / 10.0, tz / 10.0)
    return _build_positions(positions_nm)


def run_chirality_validation(
    topology: object,
    positions: object,
    sequence: str,
    topology_descriptor: PeptideTopologyDescriptor,
    validator: ChiralityValidator,
    *,
    design_chain_id: str,
    stage: str,
) -> tuple[ChiralityReport, ...]:
    """Validate every non-Gly residue on the configured chain only."""
    _, residues = resolve_design_chain(
        topology,
        design_chain_id,
        expected_length=len(sequence),
    )
    apply_d_annotations = stage != "post_h"
    reports: list[ChiralityReport] = []
    for local_index, (one_letter, residue) in enumerate(zip(sequence, residues, strict=True)):
        if one_letter == "G":
            continue
        mapping = collect_atom_mapping(topology, positions, residue.index)
        is_d = apply_d_annotations and any(
            entry.position == local_index + 1 for entry in topology_descriptor.d_substitutions
        )
        expected = "D" if is_d else "L"
        reports.append(
            validator(
                mapping,
                three_letter_for(local_index, one_letter),
                local_index,
                expected=expected,
                stage=stage,
            )
        )
    return tuple(reports)


def three_letter_for(index: int, one_letter: str) -> str:  # noqa: ARG001
    """Return the canonical three-letter name for a sequence residue."""
    return THREE_LETTER[one_letter]


def verify_d_backbone_invariance(
    mapping: dict[str, tuple[float, float, float]],
    residue_index: int,
    *,
    before: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    """Reject a transform that drops or moves N, CA, or C."""
    backbone = {name: mapping[name] for name in D_BACKBONE_INVARIANT_ATOMS if name in mapping}
    if len(backbone) != len(D_BACKBONE_INVARIANT_ATOMS):
        missing = sorted(set(D_BACKBONE_INVARIANT_ATOMS) - set(backbone))
        raise ValueError(
            f"D-coordinate transform: residue {residue_index + 1} mapping is missing "
            f"backbone atom(s) {missing!r}; the transformer must preserve the "
            "full N/CA/C backbone (the D mirror reflects side chains only)"
        )
    if before is None:
        return
    for name in D_BACKBONE_INVARIANT_ATOMS:
        moved = distance(before[name], backbone[name])
        if moved > D_TRANSFORM_BACKBONE_TOLERANCE_A:
            raise ValueError(
                f"D-coordinate transform: residue {residue_index + 1} moved backbone "
                f"atom {name!r} by {moved:.6f} Å (tolerance "
                f"{D_TRANSFORM_BACKBONE_TOLERANCE_A:.6f} Å). The D mirror "
                "reflects side-chain atoms through the N-CA-C plane; "
                "N/CA/C must be invariant."
            )


def verify_d_mapping_complete(
    mapping: dict[str, tuple[float, float, float]],
    transformed: dict[str, tuple[float, float, float]],
    residue_index: int,
) -> None:
    """Reject a transform that drops any atom from its residue mapping."""
    missing = sorted(set(mapping) - set(transformed))
    if missing:
        raise ValueError(
            f"D-coordinate transform: residue {residue_index + 1} returned mapping "
            f"dropped atom(s) {missing!r}; the transformer must return every atom "
            "it was given (side chains may move, none may vanish)"
        )


def _position_to_nm(position: object) -> tuple[float, float, float]:
    import openmm.unit as unit

    return tuple(
        float(component.value_in_unit(unit.nanometer))
        if hasattr(component, "value_in_unit")
        else float(component)
        for component in position  # type: ignore[union-attr]
    )  # type: ignore[return-value]


def _build_positions(xyz_nm: list[tuple[float, float, float]]) -> object:
    import openmm.unit as unit

    import openmm

    return unit.Quantity([openmm.Vec3(x, y, z) for x, y, z in xyz_nm], unit.nanometer)
