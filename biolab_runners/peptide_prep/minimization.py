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

__all__ = [
    "build_closed_system",
    "read_potential_energy",
    "restrain_backbone",
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


def build_closed_system(
    restrained_system: object,
    *,
    restraint_force_index: int,
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

    force = closed_system.getForce(restraint_force_index)
    if not isinstance(force, openmm.CustomExternalForce):
        raise RuntimeError(
            f"force at restraint index {restraint_force_index} is "
            f"{type(force).__name__}, expected CustomExternalForce"
        )
    if force.getEnergyFunction() != _BACKBONE_RESTRAINT_EXPRESSION:
        raise RuntimeError(
            f"CustomExternalForce at restraint index {restraint_force_index} has unexpected "
            f"energy expression {force.getEnergyFunction()!r}; expected "
            f"{_BACKBONE_RESTRAINT_EXPRESSION!r}"
        )

    closed_system.removeForce(restraint_force_index)
    return closed_system


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
