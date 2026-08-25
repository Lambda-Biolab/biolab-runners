"""Restrained energy minimization for the prepared peptide.

This module owns the second half of the runner pipeline: the
backbone restraint, the energy reads, and the minimization itself.

Critical contract (B2):

* The backbone restraint is attached to the SAME OpenMM system
  object that is used for the minimization. The energy reads
  (before AND after) and the minimization all happen on the
  restrained system. The restraint is NOT removed until after the
  post-minimization energy read.
* The exported / GROMACS-rendered system is a SEPARATE
  ``OpenMM.System`` with the same particle / bonded parameters but
  WITHOUT the restraint (the GROMACS minimizer does not understand
  ``CustomExternalForce``). The runner's
  :class:`PreparationArtifacts` carries BOTH the restrained system
  (for energy / minimization / validation) and the unrestrained
  ``closed_system`` (for ParmEd export).

Restraint physics:

* The restraint is a ``CustomExternalForce`` of the form
  ``k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)`` applied to the
  backbone N / CA / C atoms with anchor positions at the
  threaded-backbone coordinates. ``k`` is the user-supplied force
  constant in kJ/mol/nm².
* The restraint keeps the backbone near the threaded coordinates
  while side-chain clashes relax. Heavy-atom side chains (CB, CG,
  etc.) and all hydrogens are free; this is what makes the
  minimization meaningful.

Why the restrained system is NOT mutated:

* The runner's :func:`build_closed_system` builds the
  closed_system by DEEP-copying the system and removing the
  restraint from the COPY (never from the live restrained
  system). The post-minimization energy read on the restrained
  system is therefore an honest measurement — it includes the
  restraint contribution.

Energy convention:

* Energies are in kJ/mol. The ``setPositions`` + ``getState``
  pattern is the OpenMM canonical way to read a system energy
  without running dynamics; the integrator we attach is a no-op
  ``LangevinMiddleIntegrator`` at 300 K with a 1 ps⁻¹ friction
  coefficient. The integrator's parameters are irrelevant — we
  never run dynamics.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_BACKBONE_RESTRAINT_EXPRESSION = "k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
_CHIRALITY_RESTRAINT_EXPRESSION = (
    "0.5*chirality_k*step(chirality_vmin-s*v)*(chirality_vmin-s*v)^2/chirality_vmin^2;"
    "v=(x1-x2)*((y3-y2)*(z4-z2)-(z3-z2)*(y4-y2))"
    "-(y1-y2)*((x3-x2)*(z4-z2)-(z3-z2)*(x4-x2))"
    "+(z1-z2)*((x3-x2)*(y4-y2)-(y3-y2)*(x4-x2))"
)

__all__ = [
    "build_closed_system",
    "read_potential_energy",
    "restrain_backbone",
    "restrain_chirality",
    "run_minimization",
]


def restrain_backbone(
    system: object,
    topology: object,
    positions: object,
    *,
    force_constant_k_kjmol_nm2: float,
) -> int:
    """Attach a positional restraint to backbone N/CA/C atoms.

    The restraint is a ``CustomExternalForce`` with the per-atom
    anchor positions set to the threaded backbone coordinates (in
    nm). The force constant is in kJ/mol/nm² — the unit the
    OpenMM ``system.addForce`` path expects.

    The returned integer is the exact index occupied by the
    restraint in the original system. XML serialization preserves
    force order, so :func:`build_closed_system` validates and removes
    that same index from its copy.

    Args:
        system: The OpenMM ``System`` to attach the restraint to.
        topology: The OpenMM ``app.Topology``.
        positions: The OpenMM ``Vec3`` positions (length
            ``topology.getNumAtoms()``).
        force_constant_k_kjmol_nm2: Strength of the per-atom
            harmonic well in kJ/mol/nm².

    Returns:
        The zero-based force index occupied by the newly added
        backbone restraint in ``system``.
    """
    import openmm

    restraint = openmm.CustomExternalForce(_BACKBONE_RESTRAINT_EXPRESSION)
    restraint.addGlobalParameter("k", force_constant_k_kjmol_nm2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    for atom in topology.atoms():
        if atom.name not in {"N", "CA", "C"}:
            continue
        pos = positions[atom.index]  # type: ignore[index]
        restraint.addParticle(
            atom.index,
            [
                pos[0] / openmm.unit.nanometer,  # type: ignore[operator]
                pos[1] / openmm.unit.nanometer,  # type: ignore[operator]
                pos[2] / openmm.unit.nanometer,  # type: ignore[operator]
            ],
        )

    restraint_index = system.getNumForces()
    system.addForce(restraint)
    return restraint_index


def restrain_chirality(
    system: object,
    topology: object,
    positions: object,
    *,
    force_constant_k_kjmol: float,
    minimum_signed_volume_nm3: float,
) -> int:
    """Attach signed N-CA-C-CB chiral-volume wells for every non-Gly residue."""
    import openmm

    restraint = openmm.CustomCompoundBondForce(4, _CHIRALITY_RESTRAINT_EXPRESSION)
    restraint.addGlobalParameter("chirality_k", force_constant_k_kjmol)
    restraint.addGlobalParameter("chirality_vmin", minimum_signed_volume_nm3)
    restraint.addPerBondParameter("s")
    for residue in topology.residues():
        if residue.name == "GLY":
            continue
        atoms = {atom.name: atom for atom in residue.atoms()}
        missing = {"N", "CA", "C", "CB"} - atoms.keys()
        if missing:
            raise ValueError(
                f"cannot restrain chirality at residue {residue.index}: "
                f"missing atoms {sorted(missing)}"
            )
        atom_indices = tuple(atoms[name].index for name in ("N", "CA", "C", "CB"))
        signed_volume = _signed_chiral_volume(
            *(positions[index] for index in atom_indices)  # type: ignore[index]
        )
        if signed_volume == 0.0:
            raise ValueError(
                f"cannot restrain chirality at residue {residue.index}: "
                "N, CA, C, and CB define degenerate geometry"
            )
        restraint.addBond(atom_indices, [1.0 if signed_volume > 0.0 else -1.0])

    restraint_index = system.getNumForces()
    system.addForce(restraint)
    return restraint_index


def _signed_chiral_volume(p1: object, p2: object, p3: object, p4: object) -> float:
    import numpy as np
    import openmm.unit as unit

    points = [
        np.asarray(point.value_in_unit(unit.nanometer), dtype=float)  # type: ignore[attr-defined]
        for point in (p1, p2, p3, p4)
    ]
    return float(
        np.dot(np.cross(points[0] - points[1], points[2] - points[1]), points[3] - points[1])
    )


def build_closed_system(
    restrained_system: object,
    *,
    restraint_force_index: int,
    chirality_restraint_force_index: int | None = None,
) -> object:
    """Build an unrestrained copy of the system for ParmEd export.

    The restrained system still has the backbone restraint attached
    when this is called. XML serialization preserves force order;
    this function verifies that the exact returned index still names
    a ``CustomExternalForce`` with the backbone restraint energy
    expression before removing it from the copied system. The
    original ``restrained_system`` is never mutated. Any missing,
    out-of-range, wrong-type, or wrong-expression index fails
    closed rather than falling back to an unrelated force.

    Args:
        restrained_system: The OpenMM ``System`` with the restraint
            attached.
        restraint_force_index: Zero-based index returned by
            :func:`restrain_backbone`.
        chirality_restraint_force_index: Optional zero-based index
            returned by :func:`restrain_chirality`.

    Returns:
        A new OpenMM ``System`` with the validated backbone restraint
        force removed.

    Raises:
        ValueError: If ``restraint_force_index`` is not a valid
            integer force index.
        RuntimeError: If the serialized force at that index is not
            the backbone ``CustomExternalForce``.
    """
    import openmm

    if type(restraint_force_index) is not int:
        raise ValueError(f"restraint_force_index must be an integer; got {restraint_force_index!r}")
    if restraint_force_index < 0 or restraint_force_index >= restrained_system.getNumForces():
        raise ValueError(
            f"restraint_force_index {restraint_force_index} is outside system with "
            f"{restrained_system.getNumForces()} forces"
        )

    xml = openmm.XmlSerializer.serialize(restrained_system)
    closed_system = openmm.XmlSerializer.deserialize(xml)
    if closed_system.getNumForces() != restrained_system.getNumForces():
        raise RuntimeError(
            "force count changed during the OpenMM XML round-trip; refusing to remove restraint"
        )

    _validate_restraint_force(
        closed_system,
        restraint_force_index,
        force_type=openmm.CustomExternalForce,
        expression=_BACKBONE_RESTRAINT_EXPRESSION,
        index_name="restraint_force_index",
    )
    restraint_indices = [restraint_force_index]
    if chirality_restraint_force_index is not None:
        _validate_force_index(
            restrained_system,
            chirality_restraint_force_index,
            index_name="chirality_restraint_force_index",
        )
        _validate_restraint_force(
            closed_system,
            chirality_restraint_force_index,
            force_type=openmm.CustomCompoundBondForce,
            expression=_CHIRALITY_RESTRAINT_EXPRESSION,
            index_name="chirality_restraint_force_index",
        )
        restraint_indices.append(chirality_restraint_force_index)

    for force_index in sorted(restraint_indices, reverse=True):
        closed_system.removeForce(force_index)
    return closed_system


def _validate_force_index(system: object, force_index: object, *, index_name: str) -> None:
    if type(force_index) is not int:
        raise ValueError(f"{index_name} must be an integer; got {force_index!r}")
    if force_index < 0 or force_index >= system.getNumForces():  # type: ignore[operator]
        raise ValueError(
            f"{index_name} {force_index} is outside system with {system.getNumForces()} forces"
        )


def _validate_restraint_force(
    system: object,
    force_index: int,
    *,
    force_type: type,
    expression: str,
    index_name: str,
) -> None:
    force = system.getForce(force_index)  # type: ignore[attr-defined]
    if not isinstance(force, force_type):
        raise RuntimeError(
            f"force at {index_name} {force_index} is {type(force).__name__}, "
            f"expected {force_type.__name__}"
        )
    if force.getEnergyFunction() != expression:
        raise RuntimeError(
            f"{force_type.__name__} at {index_name} {force_index} has unexpected "
            f"energy expression {force.getEnergyFunction()!r}; expected {expression!r}"
        )


def read_potential_energy(
    topology: object,
    system: object,
    positions: object,
    *,
    platform_name: str,
) -> float:
    """Build a minimal ``Simulation``, set positions, read the energy.

    The integrator we attach is a no-op ``LangevinMiddleIntegrator``
    at 300 K with a 1 ps⁻¹ friction coefficient — we never run
    dynamics, just read the energy, so the integrator parameters
    are irrelevant.

    Returns:
        Potential energy in kJ/mol.
    """
    import openmm.app as app
    import openmm.unit as unit

    import openmm

    platform = openmm.Platform.getPlatformByName(platform_name)
    integrator = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin,  # type: ignore[operator]
        1.0 / unit.picoseconds,  # type: ignore[operator]
        0.002 * unit.picoseconds,  # type: ignore[operator]
    )
    sim = app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)
    state = sim.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def run_minimization(
    topology: object,
    system: object,
    positions: object,
    *,
    platform_name: str,
    max_iterations: int,
    tolerance_kjmol_nm: float,
) -> tuple[Any, float, bool]:
    """Run the restrained minimization; return ``(positions, energy, no_nan)``.

    The minimization happens on the SYSTEM PASSED IN — the runner
    passes the restrained system (B2). The integrator is
    ``LangevinMiddleIntegrator`` at 300 K / 1 ps⁻¹ (the dynamics
    parameters are irrelevant — ``LocalEnergyMinimizer`` only
    uses forces and positions).

    Returns:
        ``(positions_after, energy_kJmol, no_nan)``.
        ``no_nan`` is ``False`` iff any position or energy value
        was non-finite (NaN, inf, -inf).
    """
    sim = _build_simulation(topology, system, positions, platform_name)
    _run_minimize(
        sim,
        max_iterations=max_iterations,
        tolerance_kjmol_nm=tolerance_kjmol_nm,
    )
    positions_after, energy = _read_state(sim)
    no_nan = _check_no_nan(energy, positions_after)
    return positions_after, float(energy), no_nan  # type: ignore[arg-type]


def _build_simulation(
    topology: object,
    system: object,
    positions: object,
    platform_name: str,
) -> object:
    """Build the OpenMM Simulation with the canonical minimisation integrator."""
    import openmm.app as app
    import openmm.unit as unit

    import openmm

    platform = openmm.Platform.getPlatformByName(platform_name)
    integrator = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin,  # type: ignore[operator]
        1.0 / unit.picoseconds,  # type: ignore[operator]
        0.002 * unit.picoseconds,  # type: ignore[operator]
    )
    sim = app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)
    return sim


def _run_minimize(
    sim: object,
    *,
    max_iterations: int,
    tolerance_kjmol_nm: float,
) -> None:
    """Run ``sim.minimizeEnergy`` with the configured cap + tolerance."""
    import openmm.unit as unit

    sim.minimizeEnergy(  # type: ignore[union-attr]
        maxIterations=max_iterations,
        tolerance=tolerance_kjmol_nm * unit.kilojoule_per_mole / unit.nanometer,
    )


def _read_state(sim: object) -> tuple[object, object]:
    """Read post-minimization positions and energy from the simulation."""
    import openmm.unit as unit

    state = sim.context.getState(getEnergy=True, getPositions=True)  # type: ignore[union-attr]
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return state.getPositions(), energy


def _check_no_nan(energy: object, positions: object) -> bool:
    """Return True iff every position + the energy are finite."""
    if not _finite_number(energy):
        return False
    positions_iter: object = positions
    for pos in positions_iter:  # type: ignore[union-attr]
        for component in pos:  # type: ignore[union-attr]
            value = _component_to_nm(component)
            if not _finite_number(value):
                return False
    return True


def _component_to_nm(component: object) -> float:
    """Convert an OpenMM Vec3 component to plain nm (always finite)."""
    import openmm.unit as unit

    if hasattr(component, "value_in_unit"):
        return float(component.value_in_unit(unit.nanometer))  # type: ignore[arg-type]
    return float(component)  # type: ignore[arg-type]


def _finite_number(value: object) -> bool:
    """Return True iff ``value`` is a finite float."""
    import math

    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
