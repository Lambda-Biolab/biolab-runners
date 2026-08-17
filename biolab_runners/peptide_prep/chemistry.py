"""Topology chemistry: cysteine-to-CYX rename, disulfide bonds, head-to-tail closure.

This module owns the bond-graph edits that happen AFTER sequence
mutation but BEFORE ``ForceField.createSystem``. The three
operations are:

1. **CYS → CYX rename + HG removal**: the amber99sbildn ``CYS``
   template carries an HG hydrogen on SG; a disulfide external
   bond on SG then violates the template (the template expects 1
   external bond, not 2). The ``CYX`` template has the same atoms
   minus HG, which satisfies the template matcher.
2. **Disulfide S-S bond addition**: PDBFixer's ``addMissingAtoms``
   auto-detects an S-S bond when two SG atoms are within CONECT
   distance (it also omits the HG hydrogen for such CYS residues).
   When the SGs are too far apart for auto-detection, this module
   adds the ``topology.addBond`` explicitly (deduplicated against
   PDBFixer's auto-bond). ``ForceField.createSystem`` then
   recognises the CYX-CYX disulfide automatically.
3. **Head-to-tail closure bond + terminal-cap removal**: in a cyclic
   peptide the amber99sbildn N-terminal (NH3+) and C-terminal (COO-)
   templates do NOT match — both carry extra atoms (H2/H3 on N-term,
   OXT on C-term) and extra charges (+1 / -1) that the internal
   peptide bond templates cannot accommodate. The fix is to
   delete the cap atoms BEFORE adding the closure bond, so both
   end residues are recognised as internal peptide residues (N-H
   on the N-terminal; C=O on the C-terminal).

The closure bond direction (head-to-tail cyclization semantics):

* **tail residue C** (the C-terminal carbonyl carbon of the
  sequence) **bonds to head residue N** (the N-terminal amino
  nitrogen of the sequence).
* The previous runner bonded head-C to tail-N (an inversion that
  double-bonded an internal carbonyl C and an internal amide N
  while leaving the true termini dangling). Probe 3 surfaced this
  as the ``head-C <-> tail-N`` closure.

Cap-atom removal contract:

* N-terminal residue: ``H2`` and ``H3`` deleted (the NH3+
  cap additions PDBFixer adds on top of the peptide NH); ``H``
  retained as the single head peptide N-H.
* C-terminal residue: ``OXT`` deleted (the COO- hydroxyl oxygen);
  ``C`` keeps its ``O`` (carbonyl).

The ``H`` retention matters because the runtime audit in
``topology._verify_cyclic_topology_chemistry`` asserts the head
``N`` carries exactly ONE bonded ``H`` (the peptide NH). Deleting
``H`` instead of the NH3+ additions would break the audit (and
the internal peptide template, which expects exactly one H on N).

The result is two internal peptide-residue topologies that match
the standard peptide templates; ``ForceField.createSystem`` then
assigns parameters from the same templates used between internal
residues in the chain. Charged side chains remain charged, and the
closure bond appears in the HarmonicBondForce with those
parameters. No "system-only" bond, no custom HarmonicBondForce
addBond, no fake template matching.

Failure modes:

* Missing SG atoms in a disulfide pair → ``ValueError`` (fails
  closed; non-cysteine disulfides are physically impossible).
* Missing C/N atoms for head-to-tail → ``ValueError``.
* Post-modification net charge on the threaded topology: this
  module does NOT enforce it. The runner's
  :func:`biolab_runners.peptide_prep.topology.build_modeller`
  reads the OpenMM system afterward and reports the actual net
  charge. A cyclic peptide may have a non-zero net charge when its
  side chains are charged.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "apply_disulfide_bonds",
    "apply_head_to_tail_closure",
    "remove_terminal_caps_for_cyclization",
    "rename_cysteines_to_cyx",
]


# Names of atoms on terminal residues that are NOT part of the
# internal peptide-bond template (must be deleted for the template
# matcher to recognise a cyclic residue as an internal one).
# ``H2``/``H3`` are the NH3+ cap additions PDBFixer places on top of
# the peptide NH (``H``); deleting ``H`` would break the
# amber99sbildn peptide template (which expects one H on N) AND
# the runtime exactly-one-N-H audit in
# ``_verify_cyclic_topology_chemistry``.
N_TERMINAL_CAP_ATOM_NAMES: frozenset[str] = frozenset({"H2", "H3"})
C_TERMINAL_CAP_ATOM_NAMES: frozenset[str] = frozenset({"OXT"})


def rename_cysteines_to_cyx(
    modeller: object,
    *,
    involved_residue_indices: set[int],
) -> object:
    """Rename CYS residues involved in disulfides to CYX, drop HG.

    The amber99sbildn ``CYS`` template includes the HG hydrogen on
    the SG atom; a disulfide external bond on SG then violates
    the template (the template expects 1 external bond, not 2).
    The ``CYX`` template has the same atoms minus HG, which
    satisfies the template matcher.

    Args:
        modeller: OpenMM ``app.Modeller`` carrying the threaded
            topology + positions. The function mutates the
            modeller in place (atom deletions and residue renames).
        involved_residue_indices: Set of 0-indexed residue indices
            that participate in disulfide pairs.

    Returns:
        The (mutated) modeller reference. The caller should use
        ``modeller.topology`` / ``modeller.positions`` to read
        the post-modification state.
    """
    if not involved_residue_indices:
        return modeller

    topology = modeller.topology
    hg_to_delete: list[Any] = []
    for atom in topology.atoms():
        if atom.residue.index in involved_residue_indices and atom.name == "HG":
            hg_to_delete.append(atom)
    if hg_to_delete:
        modeller.delete(hg_to_delete)
        topology = modeller.topology

    for residue in topology.residues():
        if residue.index in involved_residue_indices and residue.name == "CYS":
            residue.name = "CYX"

    return modeller


def apply_disulfide_bonds(
    topology: object,
    *,
    disulfide_pairs: tuple[tuple[int, int], ...],
    app_module: object,
) -> None:
    """Add S-S bonds to the topology via ``topology.addBond``.

    The pairs are residue INDICES (0-indexed). The function looks
    up the SG atom in each named residue; if any residue lacks
    an SG atom, it fails closed with a clear error message. A bond
    is added only if it's NOT already in the topology (PDBFixer's
    ``addMissingAtoms`` auto-adds the bond when the two SG atoms are
    within CONECT distance; calling this function on such a topology
    without the duplicate-check would create a duplicate bond and
    double-count the harmonic force — the disulfide bond would then
    pull the SGs together with twice the intended strength).

    Args:
        topology: OpenMM ``app.Topology``.
        disulfide_pairs: 0-indexed residue pairs.
        app_module: ``openmm.app`` module (passed in for
            ``Single`` bond-type access without an import-time
            dependency).
    """
    if not disulfide_pairs:
        return

    sg_by_residue: dict[int, Any] = {
        atom.residue.index: atom for atom in topology.atoms() if atom.name == "SG"
    }

    # Index existing bonds as (min_idx, max_idx) for O(1) lookup.
    existing_bond_pairs: set[tuple[int, int]] = set()
    for bond in topology.bonds():
        existing_bond_pairs.add(
            (min(bond.atom1.index, bond.atom2.index), max(bond.atom1.index, bond.atom2.index))
        )

    for idx_a, idx_b in disulfide_pairs:
        sg_a = sg_by_residue.get(idx_a)
        sg_b = sg_by_residue.get(idx_b)
        if sg_a is None:
            raise ValueError(
                f"disulfide pair ({idx_a + 1}, {idx_b + 1}): "
                f"residue {idx_a + 1} has no SG atom "
                f"(non-cysteine disulfide is physically impossible)"
            )
        if sg_b is None:
            raise ValueError(
                f"disulfide pair ({idx_a + 1}, {idx_b + 1}): residue {idx_b + 1} has no SG atom"
            )
        # Skip if the bond is already there (e.g. PDBFixer's
        # addMissingAtoms added it).
        key = (min(sg_a.index, sg_b.index), max(sg_a.index, sg_b.index))
        if key in existing_bond_pairs:
            continue
        topology.addBond(sg_a, sg_b, type=app_module.Single)


def remove_terminal_caps_for_cyclization(
    modeller: object,
) -> tuple[Any, int, int]:
    """Delete N-terminal H2/H3 and C-terminal OXT for cyclization.

    ``H`` (the head peptide N-H) is RETAINED — it is the single
    N-bound hydrogen the amber99sbildn internal peptide template
    expects, and the runtime audit
    (``topology._verify_cyclic_topology_chemistry``) verifies the
    head ``N`` has exactly one bonded ``H`` after cap removal.

    The amber99sbildn terminal-residue templates (N-terminal with
    NH3+ cap; C-terminal with COO- cap) do NOT match when an
    external C-N closure bond is added (the templates carry the
    cap atoms AND a charge that the internal peptide-bond templates
    cannot accommodate). The fix is to delete the cap atoms so
    both end residues are recognised as internal peptide residues
    and ``ForceField.createSystem`` matches the standard peptide-
    bond template.

    Args:
        modeller: OpenMM ``app.Modeller`` carrying the threaded
            topology + positions. The function mutates the
            modeller in place (atom deletions).

    Returns:
        ``(modeller, n_terminal_index, c_terminal_index)`` where
        the indices are the 0-indexed residue positions of the
        N-terminal and C-terminal residues AFTER cap removal. The
        C-terminal residue donates its C atom to the closure
        bond; the N-terminal residue donates its N atom.

    Raises:
        ValueError: When the topology has fewer than two residues
            (a single-residue "cycle" is physically impossible).
    """
    topology = modeller.topology
    residues = list(topology.residues())
    if len(residues) < 2:
        raise ValueError(f"head-to-tail closure requires at least 2 residues; got {len(residues)}")

    n_terminal_residue = residues[0]
    c_terminal_residue = residues[-1]

    atoms_to_delete: list[Any] = []
    for atom in n_terminal_residue.atoms():
        if atom.name in N_TERMINAL_CAP_ATOM_NAMES:
            atoms_to_delete.append(atom)
    for atom in c_terminal_residue.atoms():
        if atom.name in C_TERMINAL_CAP_ATOM_NAMES:
            atoms_to_delete.append(atom)

    if atoms_to_delete:
        modeller.delete(atoms_to_delete)
        topology = modeller.topology
        residues = list(topology.residues())
        n_terminal_residue = residues[0]
        c_terminal_residue = residues[-1]

    return modeller, n_terminal_residue.index, c_terminal_residue.index


def apply_head_to_tail_closure(
    topology: object,
    *,
    app_module: object,
) -> tuple[int, int]:
    """Add the head-to-tail C-N closure bond to the topology.

    Direction (H3): **tail residue C** (C-terminal carbonyl carbon)
    bonds to **head residue N** (N-terminal amino nitrogen). The
    previous runner bonded head-C to tail-N (an inversion); probe 3
    surfaced this as the ``head-C <-> tail-N`` closure.

    The caller is responsible for :func:`remove_terminal_caps_for_cyclization`
    BEFORE calling this function — the template matcher needs the
    terminal residues to match the internal peptide topology.

    Args:
        topology: OpenMM ``app.Topology`` (post-cap-removal).
        app_module: ``openmm.app`` module (for the ``Single``
            bond-type constant).

    Returns:
        ``(tail_c_index, head_n_index)`` — the OpenMM atom indices
        of the closure pair. The runner records these in the
        manifest's :class:`TopologyBondRecord` list.
    """
    residues = list(topology.residues())
    n_terminal_residue = residues[0]
    c_terminal_residue = residues[-1]

    c_atom = next(
        (
            a
            for a in topology.atoms()
            if a.residue.index == c_terminal_residue.index and a.name == "C"
        ),
        None,
    )
    n_atom = next(
        (
            a
            for a in topology.atoms()
            if a.residue.index == n_terminal_residue.index and a.name == "N"
        ),
        None,
    )
    if c_atom is None:
        raise ValueError(
            f"head-to-tail C-terminal residue {c_terminal_residue.index + 1} has no C atom"
        )
    if n_atom is None:
        raise ValueError(
            f"head-to-tail N-terminal residue {n_terminal_residue.index + 1} has no N atom"
        )

    topology.addBond(c_atom, n_atom, type=app_module.Single)
    return c_atom.index, n_atom.index
