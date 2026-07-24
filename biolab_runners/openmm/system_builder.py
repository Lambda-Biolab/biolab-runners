"""System/forcefield/topology/integrator assembly for OpenMM MD.

This module owns the "build me a Simulation" pipeline: imports OpenMM,
picks a platform, constructs the ForceField, builds or loads the
solvated ``Modeller``, writes the topology, assembles the System +
integrator, attaches the Cα restraint, creates the ``Simulation``,
and either resumes from a checkpoint or hands off to equilibration
(via the returned :class:`SimulationContext`).

The module is intentionally framework-typed (everything that comes
from OpenMM is ``object``) because OpenMM is an optional runtime
dependency; downstream runners must work whether or not it is
installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from biolab_runners.openmm.paths import FileNames

if TYPE_CHECKING:
    from biolab_runners.openmm.config import OpenMMConfig, SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class SimulationContext:
    """Mutable carrier for OpenMM simulation state passed between run() helpers.

    Fields are typed as ``object`` to keep the module importable when
    OpenMM is not installed; downstream callers (reporters, production
    loop, finalize) re-cast the fields they need.
    """

    simulation: object = None
    modeller: object = None
    restraint_force: object = None
    ca_indices: list[int] = field(default_factory=list)
    # Pre-built list of OpenMM Chain objects from modeller.topology.chains().
    # Equilibration and the post-equilibration PBC displacement check read
    # this rather than re-walking the modeller topology; storing it avoids
    # the walk on every call and keeps the runner methods pure data-flow.
    chains: list[object] = field(default_factory=list)
    openmm_mod: object = None
    app_mod: object = None
    unit_mod: object = None
    np_mod: object = None
    platform: object = None
    is_resuming: bool = False


def build_forcefield(config: OpenMMConfig, app: object) -> object:
    """Construct the OpenMM ForceField for the configured protein FF + water.

    Uses ``config.water_ff_xml`` when provided, else falls back to
    ``{water_model}.xml``. The distinction matters: ``Modeller.addSolvent``
    takes a SHORT model key (``"tip3p"``), whereas ``app.ForceField`` needs
    an XML filename. Bare ``tip3p.xml`` ships water parameters only, so
    ionic-strength solvation raises "No template found for residue N (NA)"
    unless the XML loaded into ForceField carries ion templates. Point
    ``water_ff_xml`` at e.g. ``"amber14/tip3p.xml"`` for an AMBER water+ions
    bundle. For CHARMM36m, the built-in ``charmm36/water.xml`` already
    includes ion templates so this override is unnecessary.

    ``config.extra_forcefields`` is appended after the protein and water
    XMLs so later entries take precedence for overlapping atom types.
    """
    ff_name = config.protein_ff
    # Match CHARMM by prefix to avoid false positives like
    # "non-charmm-test". The list is open to the same prefixes the
    # OpenMM CHARMM force fields actually ship under.
    if ff_name.lower().startswith("charmm"):
        base = ["charmm36.xml", "charmm36/water.xml"]
    else:
        water_xml = config.water_ff_xml or f"{config.water_model}.xml"
        base = [f"{ff_name}.xml", water_xml]
    return app.ForceField(*base, *config.extra_forcefields)  # type: ignore[union-attr]


def resolve_pdb(config_path: str, fallback_name: str, output_dir: Path) -> str:
    """Resolve a PDB path with fallback to output_dir / cwd.

    If ``config_path`` points to an existing file, return it as-is.
    Otherwise search ``output_dir.parent``, ``output_dir``, and the
    current working directory for ``<fallback_name>``. Returns the
    empty string if no candidate is found.

    Public so callers (and tests) can reuse the same fallback
    semantics the runner uses internally.
    """
    if config_path and Path(config_path).exists():
        return config_path
    for search_dir in [output_dir.parent, output_dir, Path(".")]:
        fallback = search_dir / fallback_name
        if fallback.exists():
            return str(fallback)
    return ""


def build_solvated_complex(
    receptor_pdb: str,
    peptide_pdb: str,
    config: OpenMMConfig,
    app: object,  # openmm.app module
    forcefield: object,
    unit: object | None = None,  # openmm.unit module; optional, imported if None
) -> object | None:
    """Build the solvated peptide-protein complex.

    Returns an openmm.app.Modeller or None if no PDB files are available.
    """
    from pdbfixer import PDBFixer

    if receptor_pdb and peptide_pdb:
        fixer = PDBFixer(filename=receptor_pdb)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(config.protonation_ph)

        pep_fixer = PDBFixer(filename=peptide_pdb)
        pep_fixer.findMissingResidues()
        pep_fixer.findMissingAtoms()
        pep_fixer.addMissingAtoms()
        pep_fixer.addMissingHydrogens(config.protonation_ph)

        modeller = app.Modeller(fixer.topology, fixer.positions)  # type: ignore[union-attr]
        modeller.add(pep_fixer.topology, pep_fixer.positions)
    elif receptor_pdb:
        fixer = PDBFixer(filename=receptor_pdb)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(config.protonation_ph)
        modeller = app.Modeller(fixer.topology, fixer.positions)  # type: ignore[union-attr]
    else:
        return None

    if unit is None:
        import openmm.unit as unit

    logger.info(
        "Complex: %d atoms, %d residues, %d chains",
        modeller.topology.getNumAtoms(),
        modeller.topology.getNumResidues(),
        modeller.topology.getNumChains(),
    )

    modeller.addSolvent(  # pyright: ignore[reportOperatorIssue, reportAttributeAccessIssue]
        forcefield,
        model=config.water_model,
        padding=config.box_padding_nm * unit.nanometers,  # pyright: ignore[reportAttributeAccessIssue, reportOperatorIssue]
        boxShape=config.box_shape,
        ionicStrength=config.nacl_mol * unit.molar,  # pyright: ignore[reportOperatorIssue]
    )
    logger.info("Solvated: %d atoms", modeller.topology.getNumAtoms())

    return modeller


def build_or_load_modeller(
    config: OpenMMConfig,
    output_dir: Path,
    app: object,
    forcefield: object,
    is_resuming: bool,
    result: SimulationResult,
    unit: object | None = None,  # openmm.unit module; forwarded to build_solvated_complex
) -> object | None:
    """Build a fresh solvated modeller or load one from a prior run."""
    topo_path = output_dir / FileNames.TOPOLOGY
    existing_topo = topo_path if topo_path.exists() and topo_path.stat().st_size > 100_000 else None

    if is_resuming and existing_topo:
        logger.info("Resuming: loading solvated topology from %s", existing_topo)
        topo_pdb = app.PDBFile(str(existing_topo))  # type: ignore[union-attr]
        modeller = app.Modeller(topo_pdb.topology, topo_pdb.positions)  # type: ignore[union-attr]
        logger.info("Loaded solvated system: %d atoms", modeller.topology.getNumAtoms())
        return modeller

    receptor_pdb = resolve_pdb(config.receptor_pdb, FileNames.RECEPTOR_PDB, output_dir)
    peptide_pdb = resolve_pdb(config.peptide_pdb, FileNames.PEPTIDE_PDB, output_dir)
    modeller = build_solvated_complex(receptor_pdb, peptide_pdb, config, app, forcefield, unit)
    if modeller is None:
        result.error = "Failed to build system — no valid PDB files"
    return modeller


def write_topology(
    modeller: object, output_dir: Path, app: object, result: SimulationResult
) -> None:
    """Persist the solvated topology PDB and populate result metadata."""
    topo_path = output_dir / FileNames.TOPOLOGY
    with open(str(topo_path), "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)  # type: ignore[union-attr]
    result.num_atoms = modeller.topology.getNumAtoms()  # type: ignore[union-attr]
    result.topology_path = str(topo_path)
    logger.info("Topology: %d atoms", result.num_atoms)


def assemble_system(
    forcefield: object,
    modeller: object,
    config: OpenMMConfig,
    openmm: object,
    app: object,
    unit: object,
) -> tuple[object, object]:
    """Create the OpenMM System (with barostat) and integrator."""
    system = forcefield.createSystem(  # type: ignore[union-attr]
        modeller.topology,  # type: ignore[union-attr]
        nonbondedMethod=app.PME,  # type: ignore[union-attr]
        nonbondedCutoff=1.0 * unit.nanometers,  # type: ignore[union-attr]
        constraints=app.HBonds,  # type: ignore[union-attr]
    )
    system.addForce(
        openmm.MonteCarloBarostat(  # type: ignore[union-attr]
            config.pressure_atm * unit.atmospheres,  # type: ignore[union-attr]
            config.temperature_k * unit.kelvin,  # type: ignore[union-attr]
            25,
        )
    )
    integrator = openmm.LangevinMiddleIntegrator(  # type: ignore[union-attr]
        config.temperature_k * unit.kelvin,  # type: ignore[union-attr]
        1.0 / unit.picoseconds,  # type: ignore[union-attr]
        config.timestep_fs * unit.femtoseconds,  # type: ignore[union-attr]
    )
    return system, integrator


def add_ca_restraint(
    system: object, modeller: object, chains: list[object], openmm: object
) -> tuple[object, list[int]]:
    """Add the C-alpha CustomExternalForce restraint (k=0) to the system."""
    ca_indices: list[int] = []
    for chain in chains:
        for atom in chain.atoms():  # type: ignore[union-attr]
            if atom.name == "CA":  # type: ignore[union-attr]
                ca_indices.append(atom.index)

    restraint_force = openmm.CustomExternalForce(  # type: ignore[union-attr]
        "k*periodicdistance(x,y,z,x0,y0,z0)^2"
    )
    restraint_force.addGlobalParameter("k", 0.0)
    restraint_force.addPerParticleParameter("x0")
    restraint_force.addPerParticleParameter("y0")
    restraint_force.addPerParticleParameter("z0")
    for idx in ca_indices:
        pos = modeller.positions[idx]  # type: ignore[union-attr]
        restraint_force.addParticle(idx, [pos.x, pos.y, pos.z])
    system.addForce(restraint_force)  # type: ignore[union-attr]
    return restraint_force, ca_indices


def prepare_simulation(
    config: OpenMMConfig,
    output_dir: Path,
    resume_xml: str,
    result: SimulationResult,
) -> SimulationContext | None:
    """Import OpenMM, build system, create simulation, optionally resume.

    Populates ``result.error`` on failure. Returns the
    :class:`SimulationContext` for the freshly-built simulation, with
    ``is_resuming=True`` if a checkpoint was loaded — the caller is
    responsible for skipping equilibration in that case.

    Equilibration is deliberately NOT performed here; that lives in
    the runner so system_builder stays free of equilibration policy.
    """
    try:
        import numpy as np  # noqa: I001
        import openmm
        import openmm.app as app
        import openmm.unit as unit
    except ImportError as exc:
        result.error = f"OpenMM not installed: {exc}"
        logger.error(result.error)
        return None

    try:
        platform = openmm.Platform.getPlatformByName(config.openmm_platform)
        if config.openmm_platform == "OpenCL":
            platform.setPropertyDefaultValue("Precision", "mixed")
        logger.info("Using platform: %s", platform.getName())
    except Exception as exc:
        result.error = f"Platform {config.openmm_platform} not available: {exc}"
        logger.error(result.error)
        return None

    forcefield = build_forcefield(config, app)
    is_resuming = bool(resume_xml and Path(resume_xml).exists())

    modeller = build_or_load_modeller(
        config, output_dir, app, forcefield, is_resuming, result, unit
    )
    if modeller is None:
        return None

    # write_topology performs two things: the file write and the
    # metadata assignments (result.num_atoms, result.topology_path).
    # On resume the existing topology.pdb was just read by
    # build_or_load_modeller, so re-writing it is wasted I/O. The
    # metadata assignments are still required so callers see a
    # populated SimulationResult. Split: always set metadata; only
    # do the file write on a fresh run.
    topo_path = output_dir / FileNames.TOPOLOGY
    result.topology_path = str(topo_path)
    result.num_atoms = modeller.topology.getNumAtoms()  # type: ignore[union-attr]
    if not is_resuming:
        with open(str(topo_path), "w") as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f)  # type: ignore[union-attr]
    logger.info("Topology: %d atoms", result.num_atoms)

    system, integrator = assemble_system(forcefield, modeller, config, openmm, app, unit)
    chains = list(modeller.topology.chains())  # type: ignore[union-attr]
    restraint_force, ca_indices = add_ca_restraint(system, modeller, chains, openmm)

    simulation = app.Simulation(modeller.topology, system, integrator, platform)  # type: ignore[union-attr]
    simulation.context.setPositions(modeller.positions)  # type: ignore[union-attr]

    if is_resuming:
        logger.info("Resuming from checkpoint: %s", resume_xml)
        simulation.loadState(resume_xml)

    return SimulationContext(
        simulation=simulation,
        modeller=modeller,
        restraint_force=restraint_force,
        ca_indices=ca_indices,
        chains=chains,
        openmm_mod=openmm,
        app_mod=app,
        unit_mod=unit,
        np_mod=np,
        platform=platform,
        is_resuming=is_resuming,
    )
