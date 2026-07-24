"""Pure-numpy PBC geometry helpers for OpenMM post-equilibration checks.

These functions are intentionally free of OpenMM imports so they can be
unit-tested in isolation and reused by other tools that read OpenMM
positions / box vectors.

Public API:
    collect_chain_ca_positions(chains, positions) -> (receptor_ca, peptide_ca)
    pbc_correct(diff, box_vecs, np) -> corrected_diff
    min_pbc_distance(rec_ca, pep_ca, box_vecs, np) -> float (angstroms)

``chains`` is the OpenMM ``Topology.chains()`` iterable; ``positions`` is a
numpy array of shape ``(N, 3)`` in angstroms (e.g. the value returned by
``State.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstroms)``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def collect_chain_ca_positions(
    chains: Iterable[object], positions: object
) -> tuple[list[object], list[object]]:
    """Split Cα positions by chain index: chain 0 → receptor, others → peptide.

    Returns:
        ``(receptor_ca_positions, peptide_ca_positions)`` as lists of
        position-like objects (numpy ndarray rows or OpenMM Quantity rows).
    """
    rec_ca: list[object] = []
    pep_ca: list[object] = []
    for chain_idx, chain in enumerate(chains):
        for atom in chain.atoms():  # type: ignore[union-attr]
            if atom.name != "CA":  # type: ignore[union-attr]
                continue
            target = rec_ca if chain_idx == 0 else pep_ca
            target.append(positions[atom.index])  # type: ignore[index]
    return rec_ca, pep_ca


def pbc_correct(diff: object, box_vecs: object, np: object) -> object:
    """Apply minimum-image PBC correction to displacement vectors.

    Supports general triclinic cells (orthorhombic, dodecahedron,
    truncated octahedron). The diagonal-only implementation used
    previously was correct for rectangular boxes but produced spurious
    large distances for GROMACS-style dodecahedron cells whenever an
    atom crossed a non-orthogonal face, because the off-diagonal
    lattice components were silently dropped. Converting ``diff`` to
    fractional coordinates via the inverse lattice, snapping to the
    nearest integer image, and converting back gives the correct
    minimum image for any box shape and reduces exactly to the prior
    diagonal operation when the lattice is rectangular.

    Accepts any array whose last axis has length 3; the inverse
    lattice multiplication broadcasts over leading axes.
    """
    box = np.asarray(box_vecs)  # type: ignore[union-attr]
    inv = np.linalg.inv(box)  # type: ignore[union-attr]
    frac = diff @ inv  # type: ignore[operator]
    frac = frac - np.round(frac)  # type: ignore[union-attr]
    return frac @ box  # type: ignore[operator]


def min_pbc_distance(
    rec_ca: Sequence[object], pep_ca: Sequence[object], box_vecs: object, np: object
) -> float:
    """Compute min PBC-corrected distance between two sets of positions (angstroms)."""
    rec_arr = np.array(rec_ca)  # type: ignore[union-attr]
    pep_arr = np.array(pep_ca)  # type: ignore[union-attr]
    diffs = rec_arr[:, None, :] - pep_arr[None, :, :]
    diffs = pbc_correct(diffs, box_vecs, np)
    dists = np.sqrt((np.square(diffs)).sum(axis=-1))  # type: ignore[union-attr]
    return float(dists.min())
