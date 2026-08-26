"""Real sequence mutation via PDBFixer.applyMutations.

The preparation contract requires that an arbitrary same-length
ProteinMPNN sequence becomes a real all-atom per-candidate structure
on the backbone — NOT a residue-label-only mutation. The previous
runner renamed ``residue.name`` in place which silently produced
mislabeled alanines (e.g. an ``ALA→LEU`` mutation that kept the
alanine's 5 atoms and was missing the LEU's CG / CD1 / CD2).

This module owns the *correct* mutation pipeline. Heavy atoms only;
hydrogenation is the orchestrator's responsibility (it must happen
AFTER the closure-bond / cap-removal edits so the cyclized
end-residues template-match as internal peptide residues and receive
exactly one peptide-NH, not the N-terminal NH3+ triple):

1. Load the source backbone with PDBFixer.
2. ``findMissingResidues`` + ``findMissingAtoms`` + ``addMissingAtoms``
   — establish the source backbone with the source residue templates
   fully populated.
3. ``applyMutations`` — for every residue whose target 3-letter code
   differs from the source, generate a mutation string and call
   ``PDBFixer.applyMutations``. PDBFixer renames the residue AND
   deletes the source-specific atoms that don't appear in the target
   template (so an ALA→LEU mutation drops HB1/HB2/HB3 in anticipation
   of the LEU's CB-side hydrogens).
4. Re-run ``findMissingAtoms`` + ``addMissingAtoms`` after the
   mutation so the new template's side-chain atoms (CG, CD1, CD2,
   HG, HD11-13, HD21-23 for LEU) are populated.
5. Normalize every non-Gly C-alpha to L by reflecting any
   misoriented side chain through the N-CA-C plane. PDBFixer can
   otherwise place a new C-beta on either side of an achiral
   poly-Gly backbone.

The source residue's 1-letter → 3-letter table is the same canonical
amino-acid alphabet as :mod:`biolab_runners.peptide_prep.config` —
the runner uses this module to verify the source matches the
sequence length AND every mutation string is well-formed before
calling ``PDBFixer.applyMutations`` (which raises ``KeyError`` on a
bad mutation string and ``ValueError`` on a wrong source residue
name — both are surfaced as ``PeptidePrepResult(success=False)``).

Fail-closed contract (B1):

* If the source residue name does not match the supplied source
  name (e.g. caller asked to mutate ``ALA-4-LEU`` but residue 4 is
  actually ``GLY``), :func:`mutate_sequence` raises ``ValueError``
  with a specific message; PDBFixer's own message is preserved as
  the ``__cause__`` for forensic detail.
* Unsupported target residue (e.g. ``PYL``) — :class:`ValueError`
  with the PDBFixer exception chained.

This module imports openmm / pdbfixer lazily at the function
boundary (NOT at module top-level) so the package imports cleanly
when only the config dataclasses are needed (e.g. unit tests that
build a config without an OpenMM install).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from biolab_runners.peptide_prep.utils import THREE_LETTER

if TYPE_CHECKING:
    from collections.abc import MutableSequence

logger = logging.getLogger(__name__)

_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3", "HN"})
_UNCHANGED_CHAIN_COORDINATE_TOLERANCE_NM = 1e-9

__all__ = [
    "apply_design_chain_mutation",
    "apply_sequence_mutation",
    "build_mutation_list",
    "select_chain_atoms",
]


def build_mutation_list(
    source_residue_names: list[str],
    target_sequence: str,
    source_resids: list[str],
) -> list[str]:
    """Build the mutation strings PDBFixer.applyMutations expects.

    Args:
        source_residue_names: Source residue names in chain order
            (3-letter codes, ``"ALA"``, ``"GLY"`` etc.).
        target_sequence: Designed 1-letter sequence in the SAME
            order as the source residues. Caller is responsible for
            length agreement (the runner checks before this point).
        source_resids: Source residue identifiers in chain order
            (strings — PDBFixer residue IDs are strings).

    Returns:
        A list of mutation strings of the form ``"OLD-XXX-NEW"``
        where ``OLD`` is the source 3-letter code, ``XXX`` is the
        source residue ID (an integer-string), and ``NEW`` is the
        target 3-letter code. Only entries where the source differs
        from the target are included; positions where the source
        already matches the target contribute no mutation string.

    Raises:
        ValueError: If the lengths disagree, or if a 1-letter code
            does not map to a canonical amino acid.
    """
    if len(source_residue_names) != len(target_sequence):
        raise ValueError(
            f"sequence/source length mismatch: source has "
            f"{len(source_residue_names)} residues, target has "
            f"{len(target_sequence)}"
        )
    mutations: list[str] = []
    for index, (one_letter, source_name) in enumerate(
        zip(target_sequence, source_residue_names, strict=True)
    ):
        one_letter_upper = one_letter.upper()
        target_name = THREE_LETTER.get(one_letter_upper)
        if target_name is None:
            raise ValueError(
                f"target sequence position {index + 1} has invalid "
                f"1-letter code {one_letter!r}; only the canonical "
                f"20 amino-acid codes are accepted"
            )
        if source_name == target_name:
            continue
        mutations.append(f"{source_name}-{source_resids[index]}-{target_name}")
    return mutations


def select_chain_atoms(
    fixer: object,
    chain_id: str,
) -> list[Any]:
    """Return the atoms to delete so only ``chain_id`` survives.

    PDBFixer preserves every chain in the input PDB; this helper
    enumerates the non-target-chain atoms so the caller can pass
    them to ``Modeller.delete``.

    Args:
        fixer: A loaded :class:`pdbfixer.PDBFixer` instance.
        chain_id: Chain identifier to keep.

    Returns:
        A flat list of :class:`openmm.app.topology.Topology.Atom`
        instances on every chain whose ``id != chain_id``. Empty
        list when the input has only one chain.
    """
    drop: list[Any] = []
    for chain in fixer.topology.chains():
        if chain.id == chain_id:
            continue
        drop.extend(list(chain.atoms()))
    return drop


def _normalize_residue_l_chirality(
    residue: object,
    positions_nm: MutableSequence[Any],
    residue_index: int,
) -> None:
    import numpy as np

    from openmm import Vec3

    atoms = {atom.name: atom for atom in residue.atoms()}  # type: ignore[attr-defined]
    missing = {"N", "CA", "C", "CB"} - atoms.keys()
    if missing:
        raise ValueError(
            f"cannot normalize L chirality at residue {residue_index}: "
            f"missing atoms {sorted(missing)}"
        )
    coordinates = {
        name: np.asarray(positions_nm[atom.index], dtype=float) for name, atom in atoms.items()
    }
    ca = coordinates["CA"]
    normal = np.cross(coordinates["N"] - ca, coordinates["C"] - ca)
    normal_squared = float(np.dot(normal, normal))
    signed_volume = float(np.dot(normal, coordinates["CB"] - ca))
    if normal_squared == 0.0 or signed_volume == 0.0:
        raise ValueError(
            f"cannot normalize L chirality at residue {residue_index}: "
            "N, CA, C, and CB define degenerate geometry"
        )
    if signed_volume < 0.0:
        for name, atom in atoms.items():
            if name in _BACKBONE_ATOM_NAMES:
                continue
            point = coordinates[name]
            reflected = point - 2.0 * np.dot(point - ca, normal) / normal_squared * normal
            positions_nm[atom.index] = Vec3(*reflected)


def _normalize_l_sidechain_chirality(
    topology: object,
    positions: object,
    target_sequence: str,
) -> object:
    """Reflect misoriented side chains into canonical L geometry."""
    from openmm import unit

    residues = list(topology.residues())  # type: ignore[attr-defined]
    if len(residues) != len(target_sequence):
        raise ValueError(
            f"sequence/source length mismatch after mutation: topology has "
            f"{len(residues)} residues, target has {len(target_sequence)}"
        )
    positions_nm = positions.value_in_unit(unit.nanometer)  # type: ignore[attr-defined]
    for residue_index, (residue, residue_code) in enumerate(
        zip(residues, target_sequence, strict=True)
    ):
        if residue_code.upper() == "G":
            continue
        _normalize_residue_l_chirality(residue, positions_nm, residue_index)
    return positions_nm * unit.nanometer


def _discover_missing_heavy_atoms(fixer: object, design_chain_id: str) -> None:
    fixer.findMissingResidues()  # type: ignore[attr-defined]
    chains = list(fixer.topology.chains())  # type: ignore[attr-defined]
    non_design_gaps = []
    for (chain_index, residue_index), residue_names in fixer.missingResidues.items():  # type: ignore[attr-defined]
        chain = chains[chain_index]
        if chain.id != design_chain_id:
            residue_names_text = ", ".join(residue_names)
            non_design_gaps.append(f"{chain.id}[{residue_index}] ({residue_names_text})")
    if non_design_gaps:
        raise ValueError(
            "non-design chain has missing residues; refusing to repair it: "
            + "; ".join(non_design_gaps)
        )
    fixer.missingResidues = {}  # type: ignore[attr-defined]
    fixer.findMissingAtoms()  # type: ignore[attr-defined]
    fixer.missingTerminals = {}  # type: ignore[attr-defined]


def _reject_missing_non_design_atoms(fixer: object, design_chain_id: str) -> None:
    missing = [
        f"{residue.chain.id}:{residue.id} {residue.name} ({', '.join(atom.name for atom in atoms)})"
        for residue, atoms in fixer.missingAtoms.items()  # type: ignore[attr-defined]
        if residue.chain.id != design_chain_id
    ]
    if missing:
        raise ValueError(
            "non-design chain has missing heavy atoms; refusing to repair it: " + "; ".join(missing)
        )


def _capture_unchanged_chain_snapshot(
    topology: object, positions: object, design_chain_id: str
) -> tuple[Any, ...]:
    from openmm import unit

    positions_nm = positions.value_in_unit(unit.nanometer)  # type: ignore[attr-defined]
    return tuple(
        _capture_chain_snapshot(chain, positions_nm)
        for chain in topology.chains()  # type: ignore[attr-defined]
        if chain.id != design_chain_id
    )


def _capture_chain_snapshot(chain: object, positions_nm: object) -> tuple[Any, ...]:
    residues: list[Any] = []
    for residue in chain.residues():  # type: ignore[attr-defined]
        atoms = tuple(
            (
                atom.name,
                atom.element.symbol if atom.element is not None else None,
                tuple(float(value) for value in positions_nm[atom.index]),  # type: ignore[index]
            )
            for atom in residue.atoms()
        )
        residues.append((residue.id, residue.name, atoms))
    return chain.id, tuple(residues)  # type: ignore[attr-defined]


def _validate_unchanged_chain_metadata(
    source_chain: tuple[Any, ...], current_chain: tuple[Any, ...]
) -> None:
    if source_chain[0] != current_chain[0]:
        raise ValueError("non-design chain order or identity changed during mutation")
    source_residues = source_chain[1]
    current_residues = current_chain[1]
    if len(source_residues) != len(current_residues):
        raise ValueError(
            f"non-design residue count changed for chain {source_chain[0]} during mutation"
        )
    for source_residue, current_residue in zip(source_residues, current_residues, strict=True):
        if source_residue[:2] != current_residue[:2]:
            raise ValueError(
                f"non-design residue identity or order changed in chain {source_chain[0]}"
            )
        source_metadata = [(name, element) for name, element, _ in source_residue[2]]
        current_metadata = [(name, element) for name, element, _ in current_residue[2]]
        if source_metadata != current_metadata:
            raise ValueError(
                f"non-design atom names or elements changed in chain {source_chain[0]} "
                f"residue {source_residue[0]}"
            )


def _unchanged_chain_coordinate_changed(
    source: tuple[float, ...], current: tuple[float, ...]
) -> bool:
    return any(
        not math.isclose(
            source_coordinate,
            current_coordinate,
            rel_tol=0.0,
            abs_tol=_UNCHANGED_CHAIN_COORDINATE_TOLERANCE_NM,
        )
        for source_coordinate, current_coordinate in zip(source, current, strict=True)
    )


def _validate_unchanged_chain_coordinates(
    source_snapshot: tuple[Any, ...], current_snapshot: tuple[Any, ...]
) -> None:
    for source_chain, current_chain in zip(source_snapshot, current_snapshot, strict=True):
        for source_residue, current_residue in zip(source_chain[1], current_chain[1], strict=True):
            for source_atom, current_atom in zip(
                source_residue[2], current_residue[2], strict=True
            ):
                if _unchanged_chain_coordinate_changed(source_atom[2], current_atom[2]):
                    raise ValueError(
                        f"non-design atom coordinates changed in chain {source_chain[0]} "
                        f"residue {source_residue[0]} atom {source_atom[0]}"
                    )


def _validate_non_design_chains_unchanged(
    source_snapshot: tuple[Any, ...],
    topology: object,
    positions: object,
    design_chain_id: str,
) -> None:
    current_snapshot = _capture_unchanged_chain_snapshot(topology, positions, design_chain_id)
    if len(source_snapshot) != len(current_snapshot):
        raise ValueError("non-design chain count changed during design-chain mutation")
    for source_chain, current_chain in zip(source_snapshot, current_snapshot, strict=True):
        _validate_unchanged_chain_metadata(source_chain, current_chain)
    _validate_unchanged_chain_coordinates(source_snapshot, current_snapshot)


def apply_design_chain_mutation(
    *,
    backbone_pdb_path: str,
    design_chain_id: str,
    target_sequence: str,
) -> tuple[Any, Any]:
    """Mutate one design chain while retaining the complete source complex.

    The returned topology and positions contain only the source heavy atoms
    plus heavy atoms required by the target residue templates.  Missing
    residues and terminal atoms are intentionally not added: residue identity,
    numbering, chain order, and the no-hydrogen boundary belong to the caller's
    subsequent preparation stages.
    """
    import os

    from pdbfixer import PDBFixer

    if not os.path.isfile(backbone_pdb_path):
        raise FileNotFoundError(f"backbone PDB not found: {backbone_pdb_path}")

    fixer = PDBFixer(filename=backbone_pdb_path)
    source_unchanged_snapshot = _capture_unchanged_chain_snapshot(
        fixer.topology, fixer.positions, design_chain_id
    )
    chain = next(
        (candidate for candidate in fixer.topology.chains() if candidate.id == design_chain_id),
        None,
    )
    if chain is None:
        raise ValueError(
            f"design_chain_id {design_chain_id!r} not found in PDB; "
            f"available: {[candidate.id for candidate in fixer.topology.chains()]}"
        )

    source_residues = list(chain.residues())
    mutations = build_mutation_list(
        [residue.name for residue in source_residues],
        target_sequence,
        [residue.id for residue in source_residues],
    )
    _discover_missing_heavy_atoms(fixer, design_chain_id)
    _reject_missing_non_design_atoms(fixer, design_chain_id)

    if mutations:
        try:
            fixer.applyMutations(mutations, design_chain_id)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"PDBFixer mutation failed for mutations={mutations!r} "
                f"on chain {design_chain_id!r}: {exc}"
            ) from exc

    _discover_missing_heavy_atoms(fixer, design_chain_id)
    fixer.addMissingAtoms()  # type: ignore[attr-defined]
    _validate_non_design_chains_unchanged(
        source_unchanged_snapshot, fixer.topology, fixer.positions, design_chain_id
    )
    return fixer.topology, fixer.positions


def apply_sequence_mutation(
    *,
    backbone_pdb_path: str,
    chain_id: str,
    target_sequence: str,
) -> tuple[Any, Any]:
    """Run the mutation pipeline (heavy atoms + hydrogens); return threaded topology + positions.

    The pipeline:

    1. Load the source PDB with PDBFixer.
    2. ``findMissingResidues`` + ``findMissingAtoms`` + ``addMissingAtoms``
       — populate the source backbone with the source residue templates.
    3. ``applyMutations`` with the mutation strings for every
       residue whose target 3-letter code differs from the source.
    4. ``findMissingAtoms`` + ``addMissingAtoms`` — re-discover and
       add the target template's atoms (CG/CD1/CD2 for LEU etc.).
    5. Normalize newly populated side chains to L geometry without
       moving N, CA, C, O, or terminal backbone atoms.
    6. ``addMissingHydrogens`` — final hydrogen addition. This
       populates the chain with terminal NH3+ / COO- caps that
       the orchestrator removes for cyclic peptides.

    Non-target chains are deleted before the mutation step (the
    mutation API is single-chain).

    Note on hydrogenation and cyclization:

    For cyclic peptides, hydrogenation must happen BEFORE the
    cap-removal / closure-bond edits — not after. The amber99sbildn
    force field has separate templates for N-terminal (NAL),
    C-terminal (CAL), and internal (AL) residues; for a cyclic
    peptide the end residues must template-match as internal, but
    ``Modeller.addHydrogens`` only recognises them as N-terminal
    / C-terminal by chain position (first / last). Hydrogenating
    after the closure bond confuses ``addHydrogens``: it treats
    the closure-C→N bond as a missing terminal cap and tries to
    add extra hydrogens, producing a topology the template matcher
    cannot handle. Hydrogenating FIRST (giving the end residues
    NH3+ / COO- caps) and then removing the caps + adding the
    closure bond is the standard amber99sbildn cyclization idiom
    — the post-edit topology matches the internal AL template
    and the external bond to the closure-C atom satisfies the
    template's ``ExternalBond`` declarations on N and C.

    Args:
        backbone_pdb_path: Path to the source PDB.
        chain_id: Chain identifier to keep.
        target_sequence: Designed 1-letter amino-acid sequence.
            Caller validates this matches the source residue count.

    Returns:
        ``(topology, positions)`` from the PDBFixer / Modeller
        pipeline (OpenMM objects; the caller does the type-erased
        pipelining).

    Raises:
        FileNotFoundError: ``backbone_pdb_path`` does not exist.
        ValueError: Length mismatch, unsupported 1-letter code,
            wrong source residue name, or PDBFixer-side failure.
    """
    import os

    import openmm.app as app
    from pdbfixer import PDBFixer

    if not os.path.isfile(backbone_pdb_path):
        raise FileNotFoundError(f"backbone PDB not found: {backbone_pdb_path}")

    fixer = PDBFixer(filename=backbone_pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    drop = select_chain_atoms(fixer, chain_id)
    if drop:
        modeller = app.Modeller(fixer.topology, fixer.positions)
        modeller.delete(drop)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

    chain = next((c for c in fixer.topology.chains() if c.id == chain_id), None)
    if chain is None:
        raise ValueError(
            f"chain_id {chain_id!r} not found in PDB; "
            f"available: {[c.id for c in fixer.topology.chains()]}"
        )
    source_names = [r.name for r in chain.residues()]
    source_resids = [r.id for r in chain.residues()]

    mutations = build_mutation_list(source_names, target_sequence, source_resids)
    if mutations:
        try:
            fixer.applyMutations(mutations, chain_id)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"PDBFixer mutation failed for mutations={mutations!r} on chain {chain_id!r}: {exc}"
            ) from exc

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.positions = _normalize_l_sidechain_chirality(
        fixer.topology,
        fixer.positions,
        target_sequence,
    )
    fixer.addMissingHydrogens(7.4)

    return fixer.topology, fixer.positions
