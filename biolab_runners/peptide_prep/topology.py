"""Core topology + system construction for the peptide-prep runner.

This module owns the OpenMM/PDBFixer/ParmEd half of the runner
pipeline. The flow is:

1. **Threading**: load the source backbone with PDBFixer, select
   exactly the configured chain, mutate sidechains to the
   designed sequence using :func:`mutation.apply_sequence_mutation`
   (B1: real PDBFixer.applyMutations + addMissingAtoms +
   addMissingHydrogens so the target template's atoms AND
   hydrogens are populated).
2. **Disulfide S-S bonds**: when a disulfide is requested, add
   the S-S bond to the topology via ``topology.addBond``. PDBFixer's
   ``addMissingAtoms`` will add the bond automatically when the
   SGs are within CONECT distance; the chemistry helper dedups.
3. **CYS → CYX**: when a disulfide is requested, rename the
   involved CYS residues to CYX (which the ``amber99sbildn``
   template recognises as the disulfide-bonded form) and remove
   the HG hydrogens from those residues. This is done AFTER
   hydrogenation because the template matcher needs the S-S bond
   to recognise CYS as CYX automatically (PDBFixer / Modeller
   leave the residue named ``CYS``; the CYX rename is applied by
   :func:`chemistry.rename_cysteines_to_cyx` in this module).
4. **Head-to-tail closure (terminal cap removal + closure bond)**:
   remove the N-term H2/H3 cap additions AND the C-term OXT cap
   atom AND add the authoritative tail-C → head-N closure bond.
   The head peptide NH (``H``) is RETAINED — the internal
   peptide-bond template expects exactly one H on N. The end
   residues template-match as internal peptide residues (NAL /
   CAL have NH3+ / COO-; after cap removal they look like
   internal AL with external bonds to the closure).
5. **System construction**: ``ForceField.createSystem`` on the
   threaded topology produces the closed-system parameters
   (amber99sbildn supplies the CYX-CYX disulfide parameters
   automatically; the head-to-tail closure bond uses the same
   peptide-bond parameters as every internal peptide bond).
6. **Restraint + minimization**: the chemistry/restrained system
   carries a ``CustomExternalForce`` for the backbone; the
   runner minimises on the restrained system and reads the energy
   before/after (B2). The closed / unrestrained system is built
   separately for ParmEd export.

Module split (M7):

* :mod:`biolab_runners.peptide_prep.mutation` — real PDBFixer mutation.
* :mod:`biolab_runners.peptide_prep.chemistry` — cap removal, CYX,
  head-to-tail closure, disulfide bonds.
* :mod:`biolab_runners.peptide_prep.minimization` — restraint,
  energy reads, minimization.
* :mod:`biolab_runners.peptide_prep.export` — ParmEd export and
  parity verification.
* :mod:`biolab_runners.peptide_prep.topology` (this module) — the
  orchestrator entry point :func:`build_modeller` + the
  :class:`PreparationArtifacts` carrier.

The modules import openmm / pdbfixer / parmed lazily at the
function boundary (NOT at module top-level) so the package imports
cleanly when only the config dataclasses / Protocols are needed
(e.g. unit tests that build a config without an OpenMM install).

Public surface (the runner uses these):

* :func:`build_modeller` — entry point. Returns a fully prepared
  :class:`PreparationArtifacts` carrying the threaded topology,
  the restrained system (for minimization), the closed /
  unrestrained system (for export), the energy before/after,
  and the closure distances for the runner's manifest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from biolab_runners.peptide_prep.config import PeptidePrepConfig

logger = logging.getLogger(__name__)

__all__ = [
    "PreparationArtifacts",
    "build_modeller",
    "compute_closure_distances",
]


@dataclass
class PreparationArtifacts:
    """The mutable carrier returned by :func:`build_modeller`.

    Attributes:
        topology: The OpenMM ``app.Topology`` AFTER threading,
            CYS→CYX renaming, cap removal, and bond insertion
            (NOT yet minimised).
        positions: ``app.Modeller.positions`` (OpenMM ``Vec3``
            quantities, length topology.getNumAtoms()).
        system: The OpenMM ``System`` with the backbone restraint
            force still attached — the runner uses this for the
            energy-minimization step and the chirality
            pre-validation (B2).
        closed_system: A SECOND ``System`` with the restraint
            force removed (the XML-serialise + re-deserialise
            idiom from :func:`minimization.build_closed_system`).
            ParmEd reads this; the GROMACS minimizer does not
            understand ``CustomExternalForce``.
        net_charge: Total nonbonded charge in elementary-charge
            units (rounded to 1e-9). Computed from ``closed_system``
            so the manifest can compare OpenMM, ParmEd ``.top``, and
            ``.gro`` views to a single canonical value.
        bond_graph: A list of ``TopologyBondRecord`` describing
            the requested closure bonds. The runner reads this to
            build the manifest and to verify the ParmEd export
            preserved them.
        closure_distances_before: Å-scale distances of the
            head-to-tail C-N and S-S bonds BEFORE minimization.
        energy_before_kjmol: Total potential energy of the
            restrained system BEFORE minimization (kJ/mol).
    """

    topology: object
    positions: object
    system: object
    closed_system: object
    net_charge: float
    bond_graph: list[Any] = field(default_factory=list)
    closure_distances_before: dict[str, float] = field(default_factory=dict)
    energy_before_kjmol: float = 0.0


def build_modeller(
    config: PeptidePrepConfig,
    *,
    platform_name: str | None = None,
) -> PreparationArtifacts:
    """Build the prepared modeller + system + initial energy.

    One-call entry point used by the runner. The runner validates
    config + computes digests in a separate path; this function
    owns the OpenMM side only.

    Args:
        config: The peptide-prep configuration.
        platform_name: OpenMM platform used for the initial energy
            read (and the system construction). Defaults to
            ``config.openmm_platform``; the runner passes its
            resolved effective platform (constructor override wins)
            so the initial energy read and the minimization run on
            the SAME platform.

    The function performs the full pipeline:

    1. :func:`mutation.apply_sequence_mutation` — real
       PDBFixer.applyMutations + heavy-atom population + hydrogen
       addition. The terminal caps (NH3+ / COO-) come from this
       step; the orchestrator removes them for cyclic peptides
       BEFORE adding the closure bond so the end residues
       template-match as internal peptide residues.
    2. :func:`chemistry.apply_disulfide_bonds` — add S-S bonds to
       the topology (dedups against PDBFixer's auto-bond).
    3. :func:`chemistry.rename_cysteines_to_cyx` — rename CYS to
       CYX for every involved disulfide pair, drop HG hydrogens
       (HG was added in step 1; we drop it here so the CYX
       template matches).
    4. :func:`chemistry.remove_terminal_caps_for_cyclization` +
       :func:`chemistry.apply_head_to_tail_closure` — close the
       ring with the CORRECT direction (tail-C → head-N); the
       N-term H2/H3 additions + C-term OXT are deleted here, the
       head ``H`` (single peptide NH) is RETAINED, so the end
       residues template-match as internal peptide residues.
    5. ``ForceField.createSystem`` — produces the closed system;
       the CYX template supplies the S-S bond parameters
       automatically; the head-to-tail bond uses the standard
       peptide-bond template parameters because the cap atoms
       have been removed and head ``H`` is preserved.
    6. :func:`minimization.restrain_backbone` — attaches the
       backbone N/CA/C restraint for the minimization step.
    7. :func:`minimization.read_potential_energy` — reads the
       initial energy on the restrained system.
    8. :func:`minimization.build_closed_system` — produces the
       unrestrained COPY for ParmEd export.

    Raises:
        FileNotFoundError: When ``config.backbone_pdb`` does
            not exist on disk.
        ValueError: For any prep-time failure (chain mismatch,
            unknown residue, impossible closure, etc.).
        RuntimeError: For OpenMM template / force-field failures
            that the caller should treat as a hard error.
    """
    from biolab_runners.peptide_prep import minimization, mutation

    if not _path_exists(config.backbone_pdb):
        raise FileNotFoundError(f"backbone PDB not found: {config.backbone_pdb}")

    # 1. Real mutation (B1).
    topology, positions = mutation.apply_sequence_mutation(
        backbone_pdb_path=config.backbone_pdb,
        chain_id=config.chain_id,
        target_sequence=config.sequence,
    )

    # The topology must match the sequence length after mutation.
    residues = list(topology.residues())
    if len(residues) != len(config.sequence):
        raise ValueError(
            f"chain {config.chain_id!r} has {len(residues)} residues after "
            f"mutation but designed sequence has {len(config.sequence)}; "
            f"residue count must equal sequence length"
        )

    # 2-4. Bond-graph edits. We thread ALL topology / position
    # mutations through a single Modeller instance so the
    # atom-index bookkeeping stays consistent — PDBFixer's
    # ``Modeller.delete`` shifts atom indices when atoms are
    # removed; re-attaching a fresh Modeller after every step
    # would silently drop the position remapping.
    modeller, closure_bond_records = _apply_bond_graph(topology, positions, config)
    topology = modeller.topology
    positions = modeller.positions

    # 5-7. ForceField.createSystem + restraint + initial energy.
    _, system, restraint_force_index, energy_before_kjmol = _build_minimization_system(
        config,
        topology,
        positions,
        platform_name=platform_name,
    )

    # 8. Build the closed / unrestrained COPY for ParmEd export.
    closed_system = minimization.build_closed_system(
        system,
        restraint_force_index=restraint_force_index,
    )
    net_charge = _net_charge(closed_system)

    # Pre-minimization closure distances for the manifest.
    closure_distances_before = compute_closure_distances(positions, topology, closure_bond_records)

    # Audit assertion (blocker #2): head N must have exactly one
    # bonded H, tail C must have only one carbonyl O plus the
    # closure connectivity. A 0-H or 2-H head, or a C with a
    # residual OXT, indicates a chemistry bug — surface it as a
    # hard error rather than silently write a wrong topology.
    _verify_cyclic_topology_chemistry(
        config=config,
        topology=topology,
        positions=positions,
    )

    return PreparationArtifacts(
        topology=topology,
        positions=positions,
        system=system,
        closed_system=closed_system,
        net_charge=net_charge,
        bond_graph=closure_bond_records,
        closure_distances_before=closure_distances_before,
        energy_before_kjmol=energy_before_kjmol,
    )


def compute_closure_distances(
    positions: object,
    topology: object,  # noqa: ARG001 — symmetric with the public protocol API
    closure_bond_records: list[Any],
) -> dict[str, float]:
    """Å-scale distances for every closure bond; keys are bond labels.

    The key format is ``"<bond_type>_<residue1_index>_<residue2_index>"``
    — the runner uses the same format for the post-minimization
    distances so the before/after distance dictionaries line up.
    """
    distances: dict[str, float] = {}
    for rec in closure_bond_records:
        pi = positions[rec.atom1_index]  # type: ignore[index]
        pj = positions[rec.atom2_index]  # type: ignore[index]
        d_nm = _distance_nm(pi, pj)
        distances[f"{rec.bond_type}_{rec.residue1_index}_{rec.residue2_index}"] = d_nm * 10.0
    return distances


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _topology_disulfide_pairs(config: PeptidePrepConfig) -> tuple[tuple[int, int], ...]:
    """Translate the descriptor's 1-indexed disulfides to 0-indexed pairs."""
    return tuple((bond.first - 1, bond.second - 1) for bond in config.topology.disulfides)


def _topology_head_to_tail(config: PeptidePrepConfig) -> tuple[int, int] | None:
    """Translate the descriptor's 1-indexed head/tail to 0-indexed (or ``None``)."""
    if config.topology.head_to_tail is None:
        return None
    return (
        config.topology.head_to_tail.head - 1,
        config.topology.head_to_tail.tail - 1,
    )


def _bond_record(
    *,
    atom1_index: int,
    atom2_index: int,
    residue1_index: int,
    residue2_index: int,
    bond_type: str,
    atom1_name: str = "",
    atom2_name: str = "",
) -> object:
    """Construct a :class:`TopologyBondRecord` (avoids import cycle)."""
    from biolab_runners.peptide_prep.utils import TopologyBondRecord

    return TopologyBondRecord(
        atom1_index=atom1_index,
        atom2_index=atom2_index,
        bond_type=bond_type,
        atom1_name=atom1_name,
        atom2_name=atom2_name,
        residue1_index=residue1_index,
        residue2_index=residue2_index,
    )


def _net_charge(system: object) -> float:
    """Sum the NonbondedForce partial charges (1e-9 precision)."""
    from biolab_runners.peptide_prep.export import compute_net_charge_from_openmm

    return compute_net_charge_from_openmm(system)


def _distance_nm(p_i: object, p_j: object) -> float:
    """Euclidean distance between two OpenMM Vec3 positions in nm."""
    import math

    import openmm.unit as unit

    def to_nm(component: object) -> float:
        if hasattr(component, "value_in_unit"):
            return float(component.value_in_unit(unit.nanometer))  # type: ignore[arg-type]
        return float(component)  # type: ignore[arg-type]

    dx = to_nm(p_i[0]) - to_nm(p_j[0])  # type: ignore[index]
    dy = to_nm(p_i[1]) - to_nm(p_j[1])  # type: ignore[index]
    dz = to_nm(p_i[2]) - to_nm(p_j[2])  # type: ignore[index]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _path_exists(path: str) -> bool:
    """Return True iff ``path`` is a string pointing to an existing file."""
    import os

    return bool(path) and os.path.exists(path) and os.path.isfile(path)


def _apply_bond_graph(
    topology: object,
    positions: object,
    config: PeptidePrepConfig,
) -> tuple[object, list[Any]]:
    """Apply CYS->CYX, disulfide bonds, and head-to-tail closure.

    Returns ``(modeller, closure_bond_records)``. The modeller's
    topology / positions are the post-edit state. Atom indices in
    the returned bond records are 0-indexed into the post-edit
    topology.

    The hydrogenation step that :func:`mutation.apply_sequence_mutation`
    ran BEFORE this function gave the end residues N-terminal
    NH3+ / C-terminal COO- caps. The cap removal + closure bond
    edits in this function give the end residues an internal
    peptide-bond topology (single peptide-NH on head N, single
    carbonyl O on tail C) that template-matches via amber99sbildn's
    internal ``ALA`` template. ``ForceField.createSystem`` then
    assigns the same peptide-bond parameters to the closure bond
    as it does to every internal peptide bond.

    Index bookkeeping (blocker #9 + H5): the disulfide records
    are built AFTER both ``rename_cysteines_to_cyx`` AND
    ``apply_disulfide_bonds`` but BEFORE the head-tail cap
    removal. When the head-tail path is also active, the cap
    removal (deletes H2/H3 on N-term + OXT on C-term; head
    ``H`` is retained) shifts ALL atom indices in the residue
    atoms following the deleted atoms. For the disulfide bond
    records (which use SG indices), this means the indices can
    shift — we MUST re-resolve the records against the final
    post-cap-removal topology before returning.
    """
    import openmm.app as app

    from biolab_runners.peptide_prep import chemistry

    modeller = app.Modeller(topology, positions)
    disulfide_pairs = _topology_disulfide_pairs(config)
    head_to_tail = _topology_head_to_tail(config)

    closure_bond_records: list[Any] = []
    if disulfide_pairs:
        involved = {idx for pair in disulfide_pairs for idx in pair}
        modeller = chemistry.rename_cysteines_to_cyx(modeller, involved_residue_indices=involved)
        chemistry.apply_disulfide_bonds(
            modeller.topology, disulfide_pairs=disulfide_pairs, app_module=app
        )
        # NOTE: do NOT build disulfide records yet — the
        # cap-removal step below may shift SG atom indices.

    if head_to_tail is not None:
        modeller, n_idx, c_idx = chemistry.remove_terminal_caps_for_cyclization(modeller)
        c_atom_index, n_atom_index = chemistry.apply_head_to_tail_closure(
            modeller.topology, app_module=app
        )
        # NOTE: the closure bond record's residue1 / residue2
        # fields describe (tail C, head N) — atom1 is the
        # tail C, atom2 is the head N. ``c_idx`` is the tail's
        # residue index (the last one) and ``n_idx`` is the
        # head's residue index (the first one).
        closure_bond_records.append(
            _head_to_tail_record(modeller.topology, c_atom_index, n_atom_index, c_idx, n_idx)
        )

    # Now build the disulfide records against the FINAL topology
    # (post-cap-removal if head-tail was active; pre otherwise —
    # the indices still match either way for the disulfide case).
    if disulfide_pairs:
        closure_bond_records.extend(_disulfide_records(modeller.topology, disulfide_pairs))

    return modeller, closure_bond_records


def _head_to_tail_record(
    topology: object,
    c_atom_index: int,
    n_atom_index: int,
    c_residue_index: int,
    n_residue_index: int,
) -> object:
    """Build the head-to-tail bond record with atom names resolved."""
    c_atom = next(a for a in topology.atoms() if a.index == c_atom_index)
    n_atom = next(a for a in topology.atoms() if a.index == n_atom_index)
    return _bond_record(
        atom1_index=c_atom_index,
        atom2_index=n_atom_index,
        residue1_index=c_residue_index,
        residue2_index=n_residue_index,
        bond_type="head_to_tail",
        atom1_name=c_atom.name,
        atom2_name=n_atom.name,
    )


def _disulfide_records(
    topology: object,
    disulfide_pairs: tuple[tuple[int, int], ...],
) -> list[Any]:
    """Build the per-disulfide bond records with atom names resolved."""
    sg_lookup: dict[int, object] = {a.residue.index: a for a in topology.atoms() if a.name == "SG"}
    records: list[Any] = []
    for idx_a, idx_b in disulfide_pairs:
        records.append(
            _bond_record(
                atom1_index=sg_lookup[idx_a].index,
                atom2_index=sg_lookup[idx_b].index,
                residue1_index=idx_a,
                residue2_index=idx_b,
                bond_type="disulfide",
                atom1_name=sg_lookup[idx_a].name,
                atom2_name=sg_lookup[idx_b].name,
            )
        )
    return records


def _build_minimization_system(
    config: PeptidePrepConfig,
    topology: object,
    positions: object,
    *,
    platform_name: str | None = None,
) -> tuple[object, object, int, float]:
    """Stage 5-7: createSystem + restraint + initial energy."""
    import openmm.app as app

    from biolab_runners.peptide_prep import minimization

    forcefield = app.ForceField(config.protein_ff, config.water_ff_xml)
    system = forcefield.createSystem(topology)
    restraint_force_index = minimization.restrain_backbone(
        system,
        topology,
        positions,
        force_constant_k_kjmol_nm2=config.restraint_force_k_kjmol_nm2,
    )
    energy_before_kjmol = minimization.read_potential_energy(
        topology,
        system,
        positions,
        platform_name=platform_name or config.openmm_platform,
    )
    return forcefield, system, restraint_force_index, energy_before_kjmol


def _verify_cyclic_tail_chemistry(topology: object, tail_c: object, tail_idx: int) -> None:
    """Verify the tail C connectivity without changing cyclic chemistry."""
    bonded_names: list[str] = []
    for bond in topology.bonds():
        if bond.atom1.index == tail_c.index:
            bonded_names.append(bond.atom2.name)
        elif bond.atom2.index == tail_c.index:
            bonded_names.append(bond.atom1.name)
    if "OXT" in bonded_names:
        raise RuntimeError(
            f"cyclic-topology audit: tail C (residue {tail_idx + 1}) still "
            f"carries an OXT atom; cap removal did not run on the C-terminal. "
            f"ForceField.createSystem would template it as a C-terminal residue "
            f"(COO-, -1 charge) and the closure bond would not be parameterized."
        )
    if sorted(bonded_names) != sorted(["O", "CA", "N"]):
        raise RuntimeError(
            f"cyclic-topology audit: tail C (residue {tail_idx + 1}) bonded "
            f"heavy atoms = {sorted(bonded_names)}; expected exactly "
            f"{{'O', 'CA', 'N'}} for an internal peptide residue. Charged side "
            f"chains remain allowed; this check only enforces terminal connectivity."
        )


def _verify_cyclic_topology_chemistry(
    *,
    config: PeptidePrepConfig,
    topology: object,
    positions: object,  # noqa: ARG001 — reserved for distance-aware checks
) -> None:
    """Blocker #2 audit: head N / tail C terminal connectivity.

    Runs ONLY when the topology carries a head-to-tail closure
    bond. Asserts:

    * The head N atom carries the residue-appropriate bonded-H
      count: exactly one H for non-PRO residues (the peptide NH,
      not the N-terminal NH3+ triple), and zero H for PRO (the
      ring N is a tertiary amide after the closure bond).
    * The tail C atom has only one carbonyl O plus its CA and the
      closure N (no residual OXT, no extraneous connectivity).

    This connectivity audit does not infer or impose a whole-chain
    net charge; side chains may be charged. A violation raises
    ``RuntimeError`` so the caller surfaces a hard, structured
    failure — NOT a silent system-only workaround. The error
    message names the exact atom so the operator can inspect the
    prepared.pdb / .top.
    """
    if config.topology.head_to_tail is None:
        return

    head_tail = _topology_head_to_tail(config)
    if head_tail is None:
        return
    head_idx, tail_idx = head_tail

    # Build an atom lookup keyed by (residue_index, atom_name).
    atoms_by_residue: dict[int, dict[str, object]] = {}
    for atom in topology.atoms():
        atoms_by_residue.setdefault(atom.residue.index, {})[atom.name] = atom

    head_n = atoms_by_residue[head_idx].get("N")
    tail_c = atoms_by_residue[tail_idx].get("C")
    if head_n is None or tail_c is None:
        raise RuntimeError(
            f"cyclic-topology audit: head N or tail C missing from the post-closure "
            f"topology (head residue {head_idx + 1} / tail residue {tail_idx + 1}); "
            f"amber99sbildn cannot template this structure"
        )

    # Count bonded H atoms on head N (peptide NH count). The
    # expected count is residue-aware: PRO's backbone N is part of
    # the pyrrolidine ring (bonded to CA and CD) and, after the
    # head-to-tail closure bond to tail-C, carries NO hydrogen —
    # a tertiary amide. Every other residue's head N carries
    # exactly one H (the peptide NH, not the N-terminal NH3+
    # triple). PDBFixer hydrogenates N-terminal PRO with H2/H3
    # (NH2+ on the ring N); cap removal deletes both, leaving 0 H,
    # which is the chemically correct tertiary-amide head.
    head_is_pro = config.sequence[head_idx] == "P"
    expected_h = 0 if head_is_pro else 1
    head_n_h_count = sum(
        1
        for atom in topology.atoms()
        if atom.element is not None
        and atom.element.symbol == "H"
        and any(
            (b.atom1.index == head_n.index and b.atom2.index == atom.index)
            or (b.atom2.index == head_n.index and b.atom1.index == atom.index)
            for b in topology.bonds()
        )
    )
    if head_n_h_count != expected_h:
        expected_desc = (
            "0 (PRO's ring N is a tertiary amide after the closure bond)"
            if head_is_pro
            else "exactly 1 (the peptide NH, not the N-terminal NH3+ triple)"
        )
        raise RuntimeError(
            f"cyclic-topology audit: head N (residue {head_idx + 1}, "
            f"{'PRO' if head_is_pro else 'non-PRO'}) has "
            f"{head_n_h_count} bonded H atoms; expected {expected_desc}. "
            f"The end residues did not template-match as internal peptide residues."
        )

    _verify_cyclic_tail_chemistry(topology, tail_c, tail_idx)
