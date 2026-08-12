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

The checkpoint / manifest / state-file domain lives in
:mod:`biolab_runners.openmm.checkpoint` — this module only
references ``FileNames.TOPOLOGY`` for the resume-safety check
(see :func:`_topology_intact`). The atomic save, quarantine, and
manifest parsing are deliberately NOT here so the two change
cadences (force-field parameters vs. checkpoint protocol) stay
independent.
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


# Threshold for treating ``output_dir/topology.pdb`` as "intact"
# enough to pair with a saved state. A solvated protein/peptide
# topology is well over 100 KB, so a smaller file indicates
# truncation. Used by both ``build_or_load_modeller`` (to decide
# whether to load the on-disk topology) and ``_topology_intact`` (to
# decide whether the resume check passes). One constant so they
# cannot drift apart.
_TOPOLOGY_MIN_BYTES = 100_000


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
        ionicStrength=config.nacl_mol * unit.molar,  # pyright: ignore[reportAttributeAccessIssue, reportOperatorIssue]
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
) -> tuple[object | None, bool]:
    """Build a fresh solvated modeller or load one from a prior run.

    The resume-without-topology corruption case is NOT handled here:
    when ``is_resuming=True`` but the on-disk ``topology.pdb`` is
    missing or truncated, ``prepare_simulation`` fails fast BEFORE
    calling this function (see the "Resume safety" rule in AGENTS.md).
    That avoids the expensive PDBFixer / ``addSolvent`` work that would
    otherwise be wasted on an unrecoverable checkpoint. By the time
    this function is reached, ``is_resuming=True`` implies an intact
    topology is present.

    Returns:
        ``(modeller, loaded_existing_topology)``. The second element is
        True when an existing ``topology.pdb`` was loaded from disk
        (the fresh-run path that was just constructed does NOT set it).
        Callers use this to decide whether the on-disk topology.pdb
        is already in sync with the returned modeller — the caller's
        next step is to write the new topology (or skip the write,
        when the on-disk one was just loaded) and then call
        ``simulation.loadState`` only when the modeller came from the
        on-disk file. A freshly-built modeller is by definition
        incompatible with the saved ``state.xml``.
    """
    topo_path = output_dir / FileNames.TOPOLOGY
    existing_topo = (
        topo_path if topo_path.exists() and topo_path.stat().st_size > _TOPOLOGY_MIN_BYTES else None
    )

    if is_resuming and existing_topo:
        logger.info("Resuming: loading solvated topology from %s", existing_topo)
        topo_pdb = app.PDBFile(str(existing_topo))  # type: ignore[union-attr]
        modeller = app.Modeller(topo_pdb.topology, topo_pdb.positions)  # type: ignore[union-attr]
        logger.info("Loaded solvated system: %d atoms", modeller.topology.getNumAtoms())
        return modeller, True

    receptor_pdb = resolve_pdb(config.receptor_pdb, FileNames.RECEPTOR_PDB, output_dir)
    peptide_pdb = resolve_pdb(config.peptide_pdb, FileNames.PEPTIDE_PDB, output_dir)
    modeller = build_solvated_complex(receptor_pdb, peptide_pdb, config, app, forcefield, unit)
    if modeller is None:
        result.error = "Failed to build system — no valid PDB files"
    return modeller, False


def write_topology(
    modeller: object,
    output_dir: Path,
    app: object,
    result: SimulationResult,
    *,
    write_file: bool = True,
) -> None:
    """Populate ``result`` with topology metadata, and (optionally) write the PDB.

    Args:
        modeller: OpenMM Modeller whose topology + positions to record.
        output_dir: Where ``topology.pdb`` would be written.
        app: openmm.app module (only needed when ``write_file=True``).
        result: SimulationResult whose ``num_atoms`` and ``topology_path``
            are populated.
        write_file: If True (default), persist the topology to disk.
            If False, skip the PDB write — used during resume where the
            existing topology.pdb was just loaded by
            ``build_or_load_modeller`` and re-writing would be wasted I/O.
            The metadata assignments still run.
    """
    topo_path = output_dir / FileNames.TOPOLOGY
    result.num_atoms = modeller.topology.getNumAtoms()  # type: ignore[union-attr]
    result.topology_path = str(topo_path)
    if write_file:
        with open(str(topo_path), "w") as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f)  # type: ignore[union-attr]
    logger.info("Topology: %d atoms", result.num_atoms)


def assemble_system(
    forcefield: object,
    modeller: object,
    config: OpenMMConfig,
    openmm: object,
    app: object,
    unit: object,
) -> tuple[object, object]:
    """Create the OpenMM System (with barostat) and integrator.

    ``config.pme`` selects the nonbonded method (``app.PME`` when
    ``True``, ``app.Cutoff`` when ``False``); ``config.constraints``
    selects the bond constraint algorithm — a string from
    ``{"None", "HBonds", "AllBonds", "HAngles"}`` mapped to the
    matching ``app.<NAME>`` symbol. Both fields come from
    ``MDSpec.pme`` / ``MDSpec.constraints`` via
    ``OpenMMConfig.from_md_spec`` (slice 16 / biolab-runners#189).
    """
    nonbonded_method = app.PME if config.pme else app.Cutoff  # type: ignore[union-attr]
    constraints_attr = getattr(app, config.constraints, None)
    if constraints_attr is None:
        raise ValueError(
            f"unknown OpenMM constraints algorithm: {config.constraints!r}. "
            f"Expected one of: None, HBonds, AllBonds, HAngles."
        )
    system = forcefield.createSystem(  # type: ignore[union-attr]
        modeller.topology,  # type: ignore[union-attr]
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=1.0 * unit.nanometers,  # type: ignore[union-attr]
        constraints=constraints_attr,
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

    Resume-integrity check happens BEFORE any expensive PDBFixer /
    addSolvent work: a corrupted checkpoint (state.xml present but
    topology.pdb missing/truncated) cannot be recovered by re-solvating
    (different water count → incompatible System), so we fail fast
    with a clear error pointing the user at ``force=True`` rather than
    building a System we'd just throw away.
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

    is_resuming_known = bool(resume_xml and Path(resume_xml).exists())

    # Fail-fast BEFORE the fresh build: a checkpoint without an intact
    # topology.pdb cannot be recovered by re-solvating — the saved
    # System's particle count, masses, and force parameters are encoded
    # in state.xml at save time, and a freshly-built System will not
    # match (different water counts, different atom order). The original
    # AGENTS.md "Resume safety" rule is the source of this constraint.
    # The user must invoke runner.run(force=True) to discard the
    # checkpoint; the runner quarantines the stale files at that point
    # so an interrupted forced run cannot leave the directory in a
    # corrupt state either.
    if is_resuming_known and not _topology_intact(output_dir):
        result.error = (
            f"Checkpoint {resume_xml} exists but the original "
            f"{FileNames.TOPOLOGY} is missing or truncated — the "
            f"saved state is incompatible with a freshly-built System. "
            f"Re-run with force=True to discard the checkpoint."
        )
        logger.error(result.error)
        return None

    forcefield = build_forcefield(config, app)
    modeller, loaded_existing_topology = build_or_load_modeller(
        config, output_dir, app, forcefield, is_resuming_known, result, unit
    )
    if modeller is None:
        return None

    # When the original topology was loaded from disk, persist the
    # metadata but skip the PDB write (the file is already in sync).
    write_topology(modeller, output_dir, app, result, write_file=not loaded_existing_topology)

    is_resuming = is_resuming_known and loaded_existing_topology

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


def _topology_intact(output_dir: Path) -> bool:
    """True when ``output_dir/topology.pdb`` exists and is large enough to be real.

    Used by :func:`prepare_simulation` to short-circuit the corrupted-
    checkpoint case before running PDBFixer / ``addSolvent``. The
    size threshold matches the one in :func:`build_or_load_modeller`
    — a solvated protein/peptide topology is well over 100 KB, so a
    smaller file indicates truncation.
    """
    topo_path = output_dir / FileNames.TOPOLOGY
    return topo_path.exists() and topo_path.stat().st_size > _TOPOLOGY_MIN_BYTES
