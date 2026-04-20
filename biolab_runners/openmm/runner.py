"""OpenMM molecular dynamics simulation runner.

Runs production MD simulations for peptide-protein complexes using OpenMM.
The runner handles system building (PDBFixer + solvation), multi-stage
equilibration, and production NPT with periodic checkpointing, early abort
checks, and trajectory/energy output.

Defaults to physiological PBS-like conditions (150 mM NaCl, pH 7.4, 310 K)
with CHARMM36m/TIP3P force fields on GPU (OpenCL or CUDA). Use the
``OpenMMConfig.physiological``, ``saliva``, ``gastric``, or ``intestinal``
preset classmethods to target other buffer environments.

Requires: openmm>=8.5.0, pdbfixer>=1.9

Usage::

    from biolab_runners.openmm import OpenMMRunner, OpenMMConfig

    config = OpenMMConfig(
        receptor_pdb="receptor.pdb",
        peptide_pdb="peptide.pdb",
        output_dir="results/md",
        production_ns=100.0,
    )
    runner = OpenMMRunner(config)
    result = runner.run()
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from biolab_runners.openmm.config import OpenMMConfig, SimulationResult
from biolab_runners.openmm.utils import (
    load_checkpoint_step,
    verify_production_outputs,
)


@dataclass
class _MdContext:
    """Mutable carrier for simulation state passed between run() helpers."""

    simulation: object = None
    modeller: object = None
    restraint_force: object = None
    ca_indices: list[int] = field(default_factory=list)
    chains: list[object] = field(default_factory=list)
    openmm_mod: object = None
    app_mod: object = None
    unit_mod: object = None
    np_mod: object = None
    platform: object = None
    is_resuming: bool = False


logger = logging.getLogger(__name__)

# Sub-chunk size for production loop (~5 min wall-clock on RTX 3090)
SUB_CHUNK_STEPS = 150_000

# Early-abort thresholds
_ABORT_MULTIPLIER = 2.0  # abort_thresh = irmsd_thresh * this
_EARLY_ABORT_5NS_FS = 5_000_000  # 5 ns in femtoseconds
_EARLY_ABORT_10NS_FS = 10_000_000  # 10 ns in femtoseconds
_SLOPE_THRESHOLD_A_PER_NS = 0.05  # RMSD drift threshold (A/ns)

# Post-equilibration displacement
_DISPLACEMENT_THRESHOLD_A = 8.0  # peptide-receptor Ca min distance (A)

# Energy minimization
_MAX_MINIMIZATION_ITERS = 1000

# Equilibration restraint strengths (kJ/mol/nm^2)
_RESTRAINT_K_STRONG = 1000.0
_RESTRAINT_K_MEDIUM = 100.0


class OpenMMRunner:
    """OpenMM production MD simulation runner.

    Builds a solvated peptide-protein system, runs multi-stage equilibration,
    and performs production NPT dynamics with checkpointing and early abort
    checks.

    The runner supports:
    - Resuming from checkpoints (idempotent re-runs)
    - Dry-run mode (validates config without GPU)
    - Early abort at 5 ns / 10 ns using a per-config iRMSD threshold
    - SIGTERM handling for clean shutdown on preemption
    - Periodic checkpointing to state.xml

    Args:
        config: Simulation configuration.

    Example::

        config = OpenMMConfig(
            receptor_pdb="receptor.pdb",
            peptide_pdb="peptide.pdb",
            output_dir="results/md",
            production_ns=100.0,
        )
        runner = OpenMMRunner(config)

        # Dry run first
        result = runner.run(dry_run=True)

        # Real run
        result = runner.run()
        print(f"Completed {result.total_ns} ns in {result.elapsed_seconds:.0f}s")
    """

    def __init__(self, config: OpenMMConfig) -> None:
        self.config = config

    def run(
        self,
        *,
        force: bool = False,
        dry_run: bool = False,
        enable_early_abort: bool = True,
    ) -> SimulationResult:
        """Run the full MD simulation pipeline.

        Pipeline stages:
        1. Build/load solvated system (PDBFixer + solvation)
        2. Energy minimization (fresh start only)
        3. Multi-stage equilibration (NVT -> NPT restrained -> NPT free)
        4. Production NPT with checkpointing

        Args:
            force: Re-run even if production is already complete.
            dry_run: Validate configuration without running the simulation.
            enable_early_abort: Enable 5 ns / 10 ns RMSD early abort checks.

        Returns:
            SimulationResult with trajectory/energy paths and performance metrics.
        """
        config = self.config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = SimulationResult(config=config)

        if dry_run:
            return self._dry_run(result)

        resume_state = self._resolve_skip_or_resume(force, output_dir, config, result)
        if resume_state is None:
            return result
        start_step, remaining_steps, resume_xml = resume_state

        ctx = self._prepare_simulation(config, output_dir, resume_xml, result)
        if ctx is None:
            return result

        ref_pep_ca, ref_pep_ca_idx, ref_rec_ca, ref_rec_ca_idx = self._compute_reference_ca(ctx)

        irmsd_thresh = config.target_irmsd_threshold_a
        abort_thresh = irmsd_thresh * _ABORT_MULTIPLIER
        abort_step_5ns = int(_EARLY_ABORT_5NS_FS / config.timestep_fs)
        abort_step_10ns = int(_EARLY_ABORT_10NS_FS / config.timestep_fs)

        traj_path = str(output_dir / "trajectory.dcd")
        energy_path = str(output_dir / "energy.csv")
        state_xml_path = str(output_dir / "state.xml")

        energy_fh = self._setup_reporters(ctx, config, traj_path, energy_path, remaining_steps)

        logger.info(
            "Starting production: %d steps (%.1f ns), checkpoint every %d steps",
            remaining_steps,
            remaining_steps * config.timestep_fs / 1e6,
            config.checkpoint_every_steps,
        )

        t0 = time.time()
        _, abort_reason = self._run_production_loop(
            simulation=ctx.simulation,
            config=config,
            remaining_steps=remaining_steps,
            state_xml_path=state_xml_path,
            output_dir=output_dir,
            enable_early_abort=enable_early_abort,
            ref_pep_ca=ref_pep_ca,
            ref_pep_ca_idx=ref_pep_ca_idx,
            ref_rec_ca=ref_rec_ca,
            ref_rec_ca_idx=ref_rec_ca_idx,
            abort_thresh=abort_thresh,
            irmsd_thresh=irmsd_thresh,
            abort_step_5ns=abort_step_5ns,
            abort_step_10ns=abort_step_10ns,
            t0=t0,
            unit=ctx.unit_mod,
            np=ctx.np_mod,
        )
        if abort_reason:
            result.early_abort = True
            result.abort_reason = abort_reason

        self._finalize_result(
            ctx=ctx,
            result=result,
            energy_fh=energy_fh,
            traj_path=traj_path,
            energy_path=energy_path,
            state_xml_path=state_xml_path,
            remaining_steps=remaining_steps,
            t0=t0,
            output_dir=output_dir,
        )
        _ = start_step  # consumed via remaining_steps
        return result

    def _resolve_skip_or_resume(
        self,
        force: bool,
        output_dir: Path,
        config: OpenMMConfig,
        result: SimulationResult,
    ) -> tuple[int, int, str] | None:
        """Handle idempotency + checkpoint resolution.

        Returns None if the simulation is already complete (result populated),
        otherwise (start_step, remaining_steps, resume_xml).
        """
        if not force:
            verification = verify_production_outputs(output_dir)
            if verification["complete"]:
                logger.info(
                    "Skipping MD — trajectory already complete at %s. Use force=True to re-run.",
                    output_dir / "trajectory.dcd",
                )
                result.trajectory_path = str(output_dir / "trajectory.dcd")
                result.energy_path = str(output_dir / "energy.csv")
                result.state_xml_path = str(output_dir / "state.xml")
                result.topology_path = str(output_dir / "topology.pdb")
                return None

        start_step = 0
        resume_xml = ""
        state_xml = output_dir / "state.xml"
        if not force and state_xml.exists() and state_xml.stat().st_size > 0:
            start_step = load_checkpoint_step(output_dir)
            if start_step > 0:
                resume_xml = str(state_xml)
                logger.info(
                    "Resuming from checkpoint at step %d (%.2f ns)",
                    start_step,
                    start_step * config.timestep_fs / 1e6,
                )

        remaining_steps = max(0, config.total_steps - start_step)
        if remaining_steps == 0:
            logger.info("No remaining steps — simulation already complete")
            result.trajectory_path = str(output_dir / "trajectory.dcd")
            result.energy_path = str(output_dir / "energy.csv")
            result.state_xml_path = str(state_xml)
            return None

        return start_step, remaining_steps, resume_xml

    def _prepare_simulation(
        self,
        config: OpenMMConfig,
        output_dir: Path,
        resume_xml: str,
        result: SimulationResult,
    ) -> _MdContext | None:
        """Import OpenMM, build system, create simulation, equilibrate or resume.

        Populates result.error on failure. Returns the simulation context or None.
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

        forcefield = self._build_forcefield(config, app)
        is_resuming = bool(resume_xml and Path(resume_xml).exists())

        modeller = self._build_or_load_modeller(
            config, output_dir, app, forcefield, is_resuming, result
        )
        if modeller is None:
            return None

        self._write_topology(modeller, output_dir, app, result)

        system, integrator = self._assemble_system(forcefield, modeller, config, openmm, app, unit)
        chains = list(modeller.topology.chains())  # type: ignore[union-attr]
        restraint_force, ca_indices = self._add_ca_restraint(system, modeller, chains, openmm)

        simulation = app.Simulation(modeller.topology, system, integrator, platform)  # type: ignore[union-attr]
        simulation.context.setPositions(modeller.positions)  # type: ignore[union-attr]

        if is_resuming:
            logger.info("Resuming from checkpoint: %s", resume_xml)
            simulation.loadState(resume_xml)
        else:
            self._run_equilibration(
                simulation,
                restraint_force,
                ca_indices,
                config,
                unit,
                output_dir,
                chains,
                np,
            )

        return _MdContext(
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

    @staticmethod
    def _build_forcefield(config: OpenMMConfig, app: object) -> object:
        """Construct the OpenMM ForceField for the configured protein FF + water.

        ``config.extra_forcefields`` is appended after the protein and water
        XMLs so later entries take precedence for overlapping atom types.
        """
        ff_name = config.protein_ff
        if "charmm" in ff_name.lower():
            base = ["charmm36.xml", "charmm36/water.xml"]
        else:
            base = [f"{ff_name}.xml", f"{config.water_model}.xml"]
        return app.ForceField(*base, *config.extra_forcefields)  # type: ignore[union-attr]

    def _build_or_load_modeller(
        self,
        config: OpenMMConfig,
        output_dir: Path,
        app: object,
        forcefield: object,
        is_resuming: bool,
        result: SimulationResult,
    ) -> object | None:
        """Build a fresh solvated modeller or load one from a prior run."""
        topo_path = output_dir / "topology.pdb"
        existing_topo = (
            topo_path if topo_path.exists() and topo_path.stat().st_size > 100_000 else None
        )

        if is_resuming and existing_topo:
            logger.info("Resuming: loading solvated topology from %s", existing_topo)
            topo_pdb = app.PDBFile(str(existing_topo))  # type: ignore[union-attr]
            modeller = app.Modeller(topo_pdb.topology, topo_pdb.positions)  # type: ignore[union-attr]
            logger.info("Loaded solvated system: %d atoms", modeller.topology.getNumAtoms())
            return modeller

        receptor_pdb = self._resolve_pdb(config.receptor_pdb, "receptor.pdb")
        peptide_pdb = self._resolve_pdb(config.peptide_pdb, "peptide.pdb")
        modeller = self._build_system(receptor_pdb, peptide_pdb, config, app, forcefield)
        if modeller is None:
            result.error = "Failed to build system — no valid PDB files"
        return modeller

    @staticmethod
    def _write_topology(
        modeller: object, output_dir: Path, app: object, result: SimulationResult
    ) -> None:
        """Persist the solvated topology PDB and populate result metadata."""
        topo_path = output_dir / "topology.pdb"
        with open(str(topo_path), "w") as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f)  # type: ignore[union-attr]
        result.num_atoms = modeller.topology.getNumAtoms()  # type: ignore[union-attr]
        result.topology_path = str(topo_path)
        logger.info("Topology: %d atoms", result.num_atoms)

    @staticmethod
    def _assemble_system(
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

    @staticmethod
    def _add_ca_restraint(
        system: object, modeller: object, chains: list[object], openmm: object
    ) -> tuple[object, list[int]]:
        """Add the C-alpha CustomExternalForce restraint (k=0) to the system."""
        ca_indices: list[int] = []
        for chain in chains:
            for atom in chain.atoms():  # type: ignore[union-attr]
                if atom.name == "CA":
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

    @staticmethod
    def _compute_reference_ca(
        ctx: _MdContext,
    ) -> tuple[object, list[int], object, list[int]]:
        """Compute receptor/peptide CA index lists + reference CA positions.

        Returns ``(ref_pep_ca, ref_pep_ca_idx, ref_rec_ca, ref_rec_ca_idx)``.
        The reference positions are captured once at the start of production
        and reused by the early-abort gate for receptor-Cα-aligned peptide
        RMSD measurement (see ``_peptide_ca_rmsd``).
        """
        ref_pep_ca_idx: list[int] = []
        ref_rec_ca_idx: list[int] = []
        for chain_idx, chain in enumerate(ctx.chains):
            for atom in chain.atoms():  # type: ignore[union-attr]
                if atom.name == "CA":
                    if chain_idx == 0:
                        ref_rec_ca_idx.append(atom.index)
                    else:
                        ref_pep_ca_idx.append(atom.index)

        ref_positions = (
            ctx.simulation.context.getState(getPositions=True)  # type: ignore[union-attr]
            .getPositions(asNumpy=True)
            .value_in_unit(ctx.unit_mod.angstroms)  # type: ignore[union-attr]
        )
        ref_pep_ca = ref_positions[ref_pep_ca_idx].copy() if ref_pep_ca_idx else None
        ref_rec_ca = ref_positions[ref_rec_ca_idx].copy() if ref_rec_ca_idx else None
        return ref_pep_ca, ref_pep_ca_idx, ref_rec_ca, ref_rec_ca_idx

    @staticmethod
    def _setup_reporters(
        ctx: _MdContext,
        config: OpenMMConfig,
        traj_path: str,
        energy_path: str,
        remaining_steps: int,
    ) -> object:
        """Attach DCD + energy + stdout reporters to the simulation.

        Returns the open energy file handle (caller closes on finalize).
        """
        app = ctx.app_mod
        simulation = ctx.simulation
        is_resuming = ctx.is_resuming

        dcd_append = is_resuming and Path(traj_path).exists()
        if not is_resuming and Path(traj_path).exists():
            stale = Path(traj_path).with_suffix(".dcd.stale")
            Path(traj_path).rename(stale)

        simulation.reporters.append(  # type: ignore[union-attr]
            app.DCDReporter(traj_path, config.save_every_steps, append=dcd_append)  # type: ignore[union-attr]
        )

        energy_mode = "a" if (is_resuming and Path(energy_path).exists()) else "w"
        energy_fh = open(energy_path, energy_mode)  # noqa: SIM115

        simulation.reporters.append(  # type: ignore[union-attr]
            app.StateDataReporter(  # type: ignore[union-attr]
                energy_fh,
                config.save_every_steps,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                speed=True,
            )
        )
        simulation.reporters.append(  # type: ignore[union-attr]
            app.StateDataReporter(  # type: ignore[union-attr]
                sys.stdout,
                config.save_every_steps * 10,
                step=True,
                time=True,
                speed=True,
                remainingTime=True,
                totalSteps=remaining_steps,
            )
        )
        return energy_fh

    @staticmethod
    def _finalize_result(
        *,
        ctx: _MdContext,
        result: SimulationResult,
        energy_fh: object,
        traj_path: str,
        energy_path: str,
        state_xml_path: str,
        remaining_steps: int,
        t0: float,
        output_dir: Path,
    ) -> None:
        """Save final state, close reporters, populate result, write md_summary.json."""
        config = result.config
        elapsed = time.time() - t0
        total_ns = remaining_steps * config.timestep_fs / 1e6
        ns_per_day = (total_ns / elapsed) * 86400 if elapsed > 0 else 0

        ctx.simulation.saveState(state_xml_path)  # type: ignore[union-attr]
        energy_fh.close()  # type: ignore[union-attr]

        result.trajectory_path = traj_path
        result.energy_path = energy_path
        result.state_xml_path = state_xml_path
        result.total_ns = round(total_ns, 2)
        result.elapsed_seconds = round(elapsed, 1)
        result.ns_per_day = round(ns_per_day, 1)

        summary = {
            "total_steps": remaining_steps,
            "total_ns": result.total_ns,
            "elapsed_seconds": result.elapsed_seconds,
            "ns_per_day": result.ns_per_day,
            "platform": config.openmm_platform,
            "num_atoms": result.num_atoms,
            "trajectory": traj_path,
            "energy": energy_path,
            "early_abort": result.early_abort,
            "abort_reason": result.abort_reason,
        }
        (output_dir / "md_summary.json").write_text(json.dumps(summary, indent=2))

        logger.info(
            "Done: %.1f ns in %.1f hours (%.0f ns/day)",
            total_ns,
            elapsed / 3600,
            ns_per_day,
        )

    def _dry_run(self, result: SimulationResult) -> SimulationResult:
        """Validate configuration without running the simulation."""
        config = self.config
        remaining_ns = config.production_ns
        logger.info(
            "[DRY-RUN] Would run %.1f ns MD simulation for %s vs %s",
            remaining_ns,
            config.target,
            config.peptide_id,
        )
        logger.info(
            "[DRY-RUN] Config: %s/%s, %.0f K, %.1f atm, %s platform",
            config.protein_ff,
            config.water_model,
            config.temperature_k,
            config.pressure_atm,
            config.openmm_platform,
        )
        logger.info(
            "[DRY-RUN] Ionic: NaCl=%.3f M, box_padding=%.1f nm, box=%s",
            config.nacl_mol,
            config.box_padding_nm,
            config.box_shape,
        )
        logger.info(
            "[DRY-RUN] Total steps: %d, save every %d, checkpoint every %d",
            config.total_steps,
            config.save_every_steps,
            config.checkpoint_every_steps,
        )

        # Validate PDB files exist
        for label, pdb_path in [
            ("receptor", config.receptor_pdb),
            ("peptide", config.peptide_pdb),
        ]:
            if pdb_path and not Path(pdb_path).exists():
                logger.warning("[DRY-RUN] %s PDB not found: %s", label, pdb_path)

        return result

    def _resolve_pdb(self, config_path: str, fallback_name: str) -> str:
        """Resolve a PDB path with fallback to output_dir."""
        if config_path and Path(config_path).exists():
            return config_path
        output_dir = Path(self.config.output_dir)
        for search_dir in [output_dir.parent, output_dir, Path(".")]:
            fallback = search_dir / fallback_name
            if fallback.exists():
                return str(fallback)
        return ""

    @staticmethod
    def _build_system(
        receptor_pdb: str,
        peptide_pdb: str,
        config: OpenMMConfig,
        app: object,  # openmm.app module
        forcefield: object,
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

    @staticmethod
    def _run_equilibration(
        simulation: object,
        restraint_force: object,
        ca_indices: list[int],
        config: OpenMMConfig,
        unit: object,
        output_dir: Path,
        chains: list[object],
        np: object,
    ) -> None:
        """Run 3-stage equilibration protocol.

        Stage 1: NVT 100ps with strong restraints (k=1000 kJ/mol/nm^2)
        Stage 2: NPT 100ps with reduced restraints (k=100)
        Stage 3: NPT 200ps with gradual restraint ramp (100->0) + unrestrained
        """
        logger.info("Minimizing energy...")
        simulation.minimizeEnergy(maxIterations=_MAX_MINIMIZATION_ITERS)  # type: ignore[union-attr]

        # Update restraint reference positions to post-minimization coords
        init_positions = (
            simulation.context.getState(getPositions=True).getPositions()  # type: ignore[union-attr]
        )
        for i, idx in enumerate(ca_indices):
            pos = init_positions[idx]
            restraint_force.setParticleParameters(  # type: ignore[union-attr]
                i, idx, [pos.x, pos.y, pos.z]
            )
        restraint_force.updateParametersInContext(simulation.context)  # type: ignore[union-attr]

        timestep_fs = config.timestep_fs

        # Stage 1: NVT with strong restraints
        simulation.context.setParameter("k", _RESTRAINT_K_STRONG)  # type: ignore[union-attr]
        logger.info("Equilibrating (NVT 100ps, k=%.0f kJ/mol/nm^2)...", _RESTRAINT_K_STRONG)
        simulation.step(int(100_000 / timestep_fs))  # type: ignore[union-attr]

        # Stage 2: NPT with reduced restraints
        simulation.context.setParameter("k", _RESTRAINT_K_MEDIUM)  # type: ignore[union-attr]
        logger.info("Equilibrating (NPT 100ps, k=%.0f kJ/mol/nm^2)...", _RESTRAINT_K_MEDIUM)
        simulation.step(int(100_000 / timestep_fs))  # type: ignore[union-attr]

        # Stage 3: Gradual restraint ramp + unrestrained
        ramp_k = [80.0, 50.0, 25.0, 10.0, 0.0]
        ramp_ps = 20
        ramp_steps = int(ramp_ps * 1000 / timestep_fs)
        for k in ramp_k:
            simulation.context.setParameter("k", k)  # type: ignore[union-attr]
            simulation.step(ramp_steps)  # type: ignore[union-attr]
        logger.info("Equilibrating (NPT restraint ramp 100->0 over %dps)...", len(ramp_k) * ramp_ps)

        # Final 100ps unrestrained
        simulation.step(int(100_000 / timestep_fs))  # type: ignore[union-attr]
        logger.info("Equilibrating (NPT 100ps unrestrained)...")

        OpenMMRunner._check_post_equilibration_displacement(
            simulation, chains, output_dir, unit, np
        )

    @staticmethod
    def _check_post_equilibration_displacement(
        simulation: object,
        chains: list[object],
        output_dir: Path,
        unit: object,
        np: object,
    ) -> None:
        """Measure peptide-receptor Ca min distance after equilibration and write metadata."""
        eq_positions = (
            simulation.context.getState(getPositions=True)  # type: ignore[union-attr]
            .getPositions(asNumpy=True)
            .value_in_unit(unit.angstroms)  # type: ignore[union-attr]
        )
        rec_ca, pep_ca = OpenMMRunner._collect_chain_ca_positions(chains, eq_positions)
        if not (rec_ca and pep_ca):
            return

        box_vecs = (
            simulation.context.getState()  # type: ignore[union-attr]
            .getPeriodicBoxVectors(asNumpy=True)
            .value_in_unit(unit.angstroms)  # type: ignore[union-attr]
        )
        min_dist = OpenMMRunner._min_pbc_distance(rec_ca, pep_ca, box_vecs, np)
        logger.info(
            "Post-equilibration peptide-receptor Ca min distance: %.1f A",
            min_dist,
        )
        if min_dist > _DISPLACEMENT_THRESHOLD_A:
            logger.warning(
                "DISPLACEMENT: peptide-receptor distance %.1f A after equilibration",
                min_dist,
            )

        eq_meta = {
            "min_ca_distance_A": round(min_dist, 2),
            "displaced": min_dist > _DISPLACEMENT_THRESHOLD_A,
            "threshold_A": _DISPLACEMENT_THRESHOLD_A,
        }
        (output_dir / "equilibration_metadata.json").write_text(json.dumps(eq_meta, indent=2))

    @staticmethod
    def _collect_chain_ca_positions(
        chains: list[object], positions: object
    ) -> tuple[list[object], list[object]]:
        """Return (receptor_ca_positions, peptide_ca_positions) lists from chain 0 / chain >0."""
        rec_ca: list[object] = []
        pep_ca: list[object] = []
        for chain_idx, chain in enumerate(chains):
            for atom in chain.atoms():  # type: ignore[union-attr]
                if atom.name != "CA":
                    continue
                target = rec_ca if chain_idx == 0 else pep_ca
                target.append(positions[atom.index])  # type: ignore[index]
        return rec_ca, pep_ca

    @staticmethod
    def _pbc_correct(diff: object, box_vecs: object, np: object) -> object:
        """Apply minimum-image PBC correction to displacement vectors."""
        box_diag = np.array([box_vecs[0][0], box_vecs[1][1], box_vecs[2][2]])  # type: ignore[union-attr,index]
        diff -= np.round(diff / box_diag) * box_diag  # type: ignore[union-attr]
        return diff

    @staticmethod
    def _kabsch_rotation(
        cur_centered: object,
        ref_centered: object,
        np: object,
    ) -> object:
        """Return the 3×3 rotation matrix that best aligns ``cur`` onto ``ref``.

        Both inputs must be centroid-subtracted (N, 3) arrays. Uses the
        standard SVD formulation with a reflection-guard determinant step so
        the returned matrix is always a proper rotation (det = +1).
        """
        h = cur_centered.T @ ref_centered  # type: ignore[union-attr]
        u, _s, vt = np.linalg.svd(h)  # type: ignore[union-attr]
        d = np.sign(np.linalg.det(vt.T @ u.T))  # type: ignore[union-attr]
        reflect = np.diag([1.0, 1.0, d])  # type: ignore[union-attr]
        return vt.T @ reflect @ u.T  # type: ignore[union-attr]

    @staticmethod
    def _peptide_ca_rmsd(
        simulation: object,
        ref_pep_ca: object,
        ref_pep_ca_idx: list[int],
        ref_rec_ca: object,
        ref_rec_ca_idx: list[int],
        unit: object,
        np: object,
    ) -> float:
        """Peptide Cα RMSD after Kabsch alignment on receptor Cα.

        Measures peptide displacement in the **receptor's reference frame**,
        so receptor diffusion and tumbling during production do not inflate
        the RMSD. Per-atom PBC correction is applied first to unwrap atoms
        that crossed a box edge, then a translation + rotation is fitted to
        the receptor Cα positions and applied to the current peptide Cα.

        Using this function, a trajectory in which receptor + peptide move
        as a rigid body returns RMSD ≈ 0. Lab-frame subtraction (the prior
        implementation, superseded because it reported large RMSDs for
        bound peptides that merely followed a rotating receptor) is not used.
        """
        state = simulation.context.getState(getPositions=True)  # type: ignore[union-attr]
        cur_pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstroms)  # type: ignore[union-attr]
        box_vecs = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstroms)  # type: ignore[union-attr]

        # Per-atom PBC unwrap: current_unwrapped = ref + pbc(current − ref)
        rec_diff = OpenMMRunner._pbc_correct(cur_pos[ref_rec_ca_idx] - ref_rec_ca, box_vecs, np)
        cur_rec = ref_rec_ca + rec_diff  # type: ignore[operator]
        pep_diff = OpenMMRunner._pbc_correct(cur_pos[ref_pep_ca_idx] - ref_pep_ca, box_vecs, np)
        cur_pep = ref_pep_ca + pep_diff  # type: ignore[operator]

        # Kabsch: fit rotation + translation of receptor Cα onto the reference.
        cur_centroid = cur_rec.mean(axis=0)  # type: ignore[union-attr]
        ref_centroid = ref_rec_ca.mean(axis=0)  # type: ignore[union-attr]
        rotation = OpenMMRunner._kabsch_rotation(
            cur_rec - cur_centroid,
            ref_rec_ca - ref_centroid,
            np,  # type: ignore[operator]
        )

        # Apply the same transform to peptide Cα, then RMSD vs reference.
        pep_aligned = (cur_pep - cur_centroid) @ rotation.T + ref_centroid  # type: ignore[union-attr]
        diff = pep_aligned - ref_pep_ca
        return float(np.sqrt((diff**2).sum(axis=1).mean()))  # type: ignore[union-attr]

    @staticmethod
    def _min_pbc_distance(
        rec_ca: list[object], pep_ca: list[object], box_vecs: object, np: object
    ) -> float:
        """Compute min PBC-corrected distance between two sets of positions."""
        rec_arr = np.array(rec_ca)  # type: ignore[union-attr]
        pep_arr = np.array(pep_ca)  # type: ignore[union-attr]
        diffs = rec_arr[:, None, :] - pep_arr[None, :, :]
        diffs = OpenMMRunner._pbc_correct(diffs, box_vecs, np)
        dists = np.sqrt((np.square(diffs)).sum(axis=-1))  # type: ignore[union-attr]
        return float(dists.min())

    @staticmethod
    def _check_early_abort_5ns(
        simulation: object,
        ref_pep_ca: object,
        ref_pep_ca_idx: list[int],
        ref_rec_ca: object,
        ref_rec_ca_idx: list[int],
        abort_thresh: float,
        irmsd_thresh: float,
        config: OpenMMConfig,
        steps_done: int,
        output_dir: Path,
        unit: object,
        np: object,
    ) -> float | None:
        """Check peptide RMSD at 5 ns for early dissociation.

        Returns the peptide RMSD, or None if check could not be performed.
        """
        pep_rmsd = OpenMMRunner._peptide_ca_rmsd(
            simulation, ref_pep_ca, ref_pep_ca_idx, ref_rec_ca, ref_rec_ca_idx, unit, np
        )
        ns_at_check = steps_done * config.timestep_fs / 1e6
        logger.info(
            "5 ns check: peptide Ca RMSD = %.1f A, abort threshold = %.1f A",
            pep_rmsd,
            abort_thresh,
        )

        if pep_rmsd > abort_thresh:
            logger.warning(
                "EARLY ABORT: peptide RMSD %.1f A > %.1f A at %.1f ns",
                pep_rmsd,
                abort_thresh,
                ns_at_check,
            )
            simulation.saveState(str(output_dir / "state.xml"))  # type: ignore[union-attr]
            abort_meta = {
                "aborted": True,
                "abort_reason": "early_dissociation",
                "abort_step": steps_done,
                "abort_ns": round(ns_at_check, 2),
                "peptide_ca_rmsd_A": round(pep_rmsd, 2),
                "abort_threshold_A": round(abort_thresh, 2),
                "target_irmsd_threshold_A": irmsd_thresh,
                "target": config.target,
                "peptide_id": config.peptide_id,
            }
            (output_dir / "early_abort.json").write_text(json.dumps(abort_meta, indent=2))

        return pep_rmsd

    @staticmethod
    def _check_slope_10ns(
        simulation: object,
        ref_pep_ca: object,
        ref_pep_ca_idx: list[int],
        ref_rec_ca: object,
        ref_rec_ca_idx: list[int],
        rmsd_5ns: float,
        config: OpenMMConfig,
        steps_done: int,
        output_dir: Path,
        unit: object,
        np: object,
    ) -> bool:
        """Check RMSD slope between 5 ns and 10 ns.

        Returns True if the slope exceeds the threshold (0.05 A/ns).
        """
        rmsd_10ns = OpenMMRunner._peptide_ca_rmsd(
            simulation, ref_pep_ca, ref_pep_ca_idx, ref_rec_ca, ref_rec_ca_idx, unit, np
        )
        slope = (rmsd_10ns - rmsd_5ns) / 5.0  # A/ns
        ns_at_check = steps_done * config.timestep_fs / 1e6
        logger.info(
            "10 ns check: RMSD = %.1f A, slope = %.3f A/ns (threshold %.2f)",
            rmsd_10ns,
            slope,
            _SLOPE_THRESHOLD_A_PER_NS,
        )

        if slope > _SLOPE_THRESHOLD_A_PER_NS:
            logger.warning(
                "EARLY ABORT (slope): %.3f A/ns > 0.05 at %.1f ns",
                slope,
                ns_at_check,
            )
            simulation.saveState(str(output_dir / "state.xml"))  # type: ignore[union-attr]
            abort_meta = {
                "aborted": True,
                "abort_reason": "rmsd_slope_drift",
                "abort_step": steps_done,
                "abort_ns": round(ns_at_check, 2),
                "peptide_ca_rmsd_5ns_A": round(rmsd_5ns, 2),
                "peptide_ca_rmsd_10ns_A": round(rmsd_10ns, 2),
                "slope_A_per_ns": round(slope, 4),
                "slope_threshold": _SLOPE_THRESHOLD_A_PER_NS,
                "target": config.target,
                "peptide_id": config.peptide_id,
            }
            (output_dir / "early_abort.json").write_text(json.dumps(abort_meta, indent=2))
            return True

        return False

    def _run_production_loop(
        self,
        *,
        simulation: object,
        config: OpenMMConfig,
        remaining_steps: int,
        state_xml_path: str,
        output_dir: Path,
        enable_early_abort: bool,
        ref_pep_ca: object,
        ref_pep_ca_idx: list[int],
        ref_rec_ca: object,
        ref_rec_ca_idx: list[int],
        abort_thresh: float,
        irmsd_thresh: float,
        abort_step_5ns: int,
        abort_step_10ns: int,
        t0: float,
        unit: object,
        np: object,
    ) -> tuple[int, str]:
        """Run the production MD loop with early-abort checks and checkpointing.

        Returns:
            (steps_done, abort_reason) — abort_reason is "" if no early abort.
        """
        last_ckpt_step = 0
        early_abort_done = False
        slope_check_done = False
        rmsd_5ns: float | None = None
        abort_reason = ""
        steps_box = [0]
        self._install_sigterm_handler(simulation, state_xml_path, steps_box, config)

        while steps_box[0] < remaining_steps:
            steps_done = steps_box[0]
            chunk = min(SUB_CHUNK_STEPS, remaining_steps - steps_done)
            simulation.step(chunk)  # type: ignore[union-attr]
            steps_done += chunk
            steps_box[0] = steps_done

            do_5ns = (
                enable_early_abort
                and not early_abort_done
                and ref_pep_ca is not None
                and steps_done >= abort_step_5ns
            )
            if do_5ns:
                early_abort_done = True
                rmsd_5ns, aborted = self._do_5ns_check(
                    simulation,
                    ref_pep_ca,
                    ref_pep_ca_idx,
                    ref_rec_ca,
                    ref_rec_ca_idx,
                    abort_thresh,
                    irmsd_thresh,
                    config,
                    steps_done,
                    output_dir,
                    unit,
                    np,
                )
                if aborted:
                    abort_reason = "early_dissociation"
                    break

            do_10ns = (
                enable_early_abort
                and not slope_check_done
                and rmsd_5ns is not None
                and ref_pep_ca is not None
                and steps_done >= abort_step_10ns
            )
            if do_10ns:
                slope_check_done = True
                if self._check_slope_10ns(
                    simulation,
                    ref_pep_ca,
                    ref_pep_ca_idx,
                    ref_rec_ca,
                    ref_rec_ca_idx,
                    rmsd_5ns,  # type: ignore[arg-type]
                    config,
                    steps_done,
                    output_dir,
                    unit,
                    np,
                ):
                    abort_reason = "rmsd_slope_drift"
                    break

            last_ckpt_step = self._maybe_checkpoint(
                simulation,
                state_xml_path,
                steps_done,
                last_ckpt_step,
                remaining_steps,
                config,
                t0,
            )

        return steps_box[0], abort_reason

    @staticmethod
    def _install_sigterm_handler(
        simulation: object,
        state_xml_path: str,
        steps_box: list[int],
        config: OpenMMConfig,
    ) -> None:
        """Install a SIGTERM handler that saves state using the current step count."""

        def handle_sigterm(signum: int, frame: object) -> None:  # noqa: ARG001
            steps_done = steps_box[0]
            ns_done = steps_done * config.timestep_fs / 1e6
            logger.warning(
                "SIGTERM received at step %d (%.2f ns) — saving state",
                steps_done,
                ns_done,
            )
            try:
                simulation.saveState(state_xml_path)  # type: ignore[union-attr]
            except Exception as exc:
                logger.error("Failed to save state on SIGTERM: %s", exc)
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_sigterm)

    def _do_5ns_check(
        self,
        simulation: object,
        ref_pep_ca: object,
        ref_pep_ca_idx: list[int],
        ref_rec_ca: object,
        ref_rec_ca_idx: list[int],
        abort_thresh: float,
        irmsd_thresh: float,
        config: OpenMMConfig,
        steps_done: int,
        output_dir: Path,
        unit: object,
        np: object,
    ) -> tuple[float | None, bool]:
        """Run the 5 ns early-abort check. Returns (rmsd, should_abort)."""
        rmsd_5ns = self._check_early_abort_5ns(
            simulation,
            ref_pep_ca,
            ref_pep_ca_idx,
            ref_rec_ca,
            ref_rec_ca_idx,
            abort_thresh,
            irmsd_thresh,
            config,
            steps_done,
            output_dir,
            unit,
            np,
        )
        should_abort = rmsd_5ns is not None and rmsd_5ns > abort_thresh
        return rmsd_5ns, should_abort

    @staticmethod
    def _maybe_checkpoint(
        simulation: object,
        state_xml_path: str,
        steps_done: int,
        last_ckpt_step: int,
        remaining_steps: int,
        config: OpenMMConfig,
        t0: float,
    ) -> int:
        """Write a checkpoint if interval elapsed. Returns the (possibly updated) last_ckpt_step."""
        since_ckpt = steps_done - last_ckpt_step
        if since_ckpt < config.checkpoint_every_steps and steps_done < remaining_steps:
            return last_ckpt_step
        simulation.saveState(state_xml_path)  # type: ignore[union-attr]
        elapsed = time.time() - t0
        ns_done = steps_done * config.timestep_fs / 1e6
        ns_per_day = (ns_done / elapsed) * 86400 if elapsed > 0 else 0
        logger.info(
            "Checkpoint: %d/%d steps (%.2f ns, %.0f ns/day)",
            steps_done,
            remaining_steps,
            ns_done,
            ns_per_day,
        )
        return steps_done
