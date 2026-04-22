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
from biolab_runners.openmm.offline_gate import (
    DEFAULT_GATE_10NS,
    evaluate_trajectory,
    write_verdict_file,
)
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

# Peptide-Cα RMSD abort threshold relative to the target's iRMSD scale.
# ``abort_thresh = irmsd_thresh * _ABORT_MULTIPLIER``. The 5 ns and 10 ns
# gate milestones (ns) + slope thresholds live alongside the offline gate
# in ``biolab_runners.openmm.offline_gate`` — do not duplicate here.
_ABORT_MULTIPLIER = 2.0

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

        abort_thresh = config.target_irmsd_threshold_a * _ABORT_MULTIPLIER

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
            abort_thresh=abort_thresh,
            t0=t0,
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

        production_steps_done = max(0, start_step - config.total_equil_steps)
        remaining_steps = max(0, config.total_steps - production_steps_done)
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
        if "charmm" in ff_name.lower():
            base = ["charmm36.xml", "charmm36/water.xml"]
        else:
            water_xml = config.water_ff_xml or f"{config.water_model}.xml"
            base = [f"{ff_name}.xml", water_xml]
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
        # OralBiome-AMP#175: match the gate path — use OpenMM's internal
        # unwrapped coordinates (enforcePeriodicBox=False, default). The
        # downstream _min_pbc_distance does its own PBC-correct min-image
        # math, so the input convention here only needs to stay consistent
        # with what the gate sees; unwrapped is the correct choice.
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

    def _run_production_loop(
        self,
        *,
        simulation: object,
        config: OpenMMConfig,
        remaining_steps: int,
        state_xml_path: str,
        output_dir: Path,
        enable_early_abort: bool,
        abort_thresh: float,
        t0: float,
    ) -> tuple[int, str]:
        """Run the production MD loop with early-abort checks and checkpointing.

        Returns:
            (steps_done, abort_reason) — abort_reason is "" if no early abort.
        """
        last_ckpt_step = 0
        abort_reason = ""
        steps_box = [0]
        self._install_sigterm_handler(simulation, state_xml_path, steps_box, config)

        # OralBiome-AMP task #10: the early-abort gate is an offline mdtraj
        # evaluation of the partial trajectory.dcd, not an inside-OpenMM
        # callback. After each sub-chunk we poll ``evaluate_trajectory``
        # on the replicate directory; it re-derives the reference pose
        # from frame 0 of the DCD and computes receptor-aligned peptide-
        # Cα RMSD with triclinic-aware unwrap. See the
        # ``biolab_runners.openmm.offline_gate`` module docstring for the
        # bug chain this closes ({#162, #163, #167, #174, #175}).
        gates_active = enable_early_abort
        # Skip re-polling after we're past the 10 ns checkpoint and no
        # abort fired — there's nothing else the gate can decide.
        gate_polling_done = False

        while steps_box[0] < remaining_steps:
            steps_done = steps_box[0]
            chunk = min(SUB_CHUNK_STEPS, remaining_steps - steps_done)
            simulation.step(chunk)  # type: ignore[union-attr]
            steps_done += chunk
            steps_box[0] = steps_done

            if gates_active and not gate_polling_done:
                gate_polling_done, abort_reason = self._poll_offline_gate(
                    simulation=simulation,
                    state_xml_path=state_xml_path,
                    output_dir=output_dir,
                    abort_thresh=abort_thresh,
                    config=config,
                    steps_done=steps_done,
                )
                if abort_reason:
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
    def _poll_offline_gate(
        *,
        simulation: object,
        state_xml_path: str,
        output_dir: Path,
        abort_thresh: float,
        config: OpenMMConfig,
        steps_done: int,
    ) -> tuple[bool, str]:
        """Run the offline mdtraj gate on the current partial trajectory.

        Called after every production sub-chunk (~5 min wall). Loads
        ``trajectory.dcd`` from ``output_dir`` via mdtraj, computes the
        same peptide-Cα RMSD (receptor-aligned, triclinic-aware unwrap)
        the legacy inside-OpenMM gate did, and writes a
        ``gate_verdict_{current_ns}ns.json`` file next to the trajectory
        so orchestrators + SIGTERM teardown can see the latest state.

        On ``abort=True``, saves ``state.xml`` (via ``saveState``, per
        expert caveat #1 in ``project_md_gate_architecture.md`` — NOT
        ``saveCheckpoint``) and writes an ``early_abort.json`` with the
        reason.

        Returns:
            ``(polling_done, abort_reason)``. ``polling_done`` is True
            once the trajectory is past the 10 ns checkpoint (no further
            gate decisions possible) OR an abort fired. ``abort_reason``
            is ``""`` when no abort; otherwise the verdict's reason.
        """
        ns_at_check = steps_done * config.timestep_fs / 1e6
        try:
            verdict = evaluate_trajectory(output_dir, threshold_a=abort_thresh)
        except FileNotFoundError:
            # DCD or topology not yet written (very first sub-chunk).
            return False, ""
        except Exception as exc:
            logger.warning("Offline gate evaluation failed: %s — continuing", exc)
            return False, ""

        if verdict.n_frames == 0:
            return False, ""

        try:
            write_verdict_file(verdict, output_dir)
        except OSError as exc:
            logger.warning("Could not write gate verdict file: %s", exc)

        logger.info(
            "Offline gate @ %.2f ns: max_rmsd=%.2f Å, rmsd_5ns=%s, "
            "rmsd_10ns=%s, slope=%s, receptor_fit=%.2f Å, abort=%s",
            ns_at_check,
            verdict.max_rmsd,
            f"{verdict.rmsd_5ns:.2f}" if verdict.rmsd_5ns is not None else "n/a",
            f"{verdict.rmsd_10ns:.2f}" if verdict.rmsd_10ns is not None else "n/a",
            f"{verdict.slope_a_per_ns:.3f}" if verdict.slope_a_per_ns is not None else "n/a",
            verdict.receptor_fit_residual,
            verdict.abort,
        )

        if verdict.abort:
            simulation.saveState(state_xml_path)  # type: ignore[union-attr]
            OpenMMRunner._write_abort_metadata(
                verdict, output_dir, abort_thresh, config, steps_done, ns_at_check
            )
            return True, verdict.reason

        # Past the 10 ns checkpoint and no abort fired → gate has made its
        # final decision; stop polling to avoid wasted work on long runs.
        polling_done = verdict.current_ns >= DEFAULT_GATE_10NS + 0.1
        return polling_done, ""

    @staticmethod
    def _write_abort_metadata(
        verdict: object,
        output_dir: Path,
        abort_thresh: float,
        config: OpenMMConfig,
        steps_done: int,
        ns_at_check: float,
    ) -> None:
        """Build and write early_abort.json for a gate abort verdict."""
        # Schema matches the pre-task-#10 inside-OpenMM abort contract
        # consumed by ``oral_amp.cloud.openmm_cloud``.
        primary_rmsd = (
            verdict.rmsd_5ns  # type: ignore[union-attr]
            if verdict.reason == "early_dissociation" and verdict.rmsd_5ns is not None  # type: ignore[union-attr]
            else verdict.rmsd_10ns  # type: ignore[union-attr]
            if verdict.rmsd_10ns is not None  # type: ignore[union-attr]
            else verdict.max_rmsd  # type: ignore[union-attr]
        )
        abort_meta = {
            "aborted": True,
            "abort_reason": verdict.reason,  # type: ignore[union-attr]
            "abort_step": steps_done,
            "abort_ns": round(ns_at_check, 2),
            "peptide_ca_rmsd_A": round(primary_rmsd, 2),
            "peptide_ca_rmsd_5ns_A": (
                round(verdict.rmsd_5ns, 2) if verdict.rmsd_5ns is not None else None  # type: ignore[union-attr]
            ),
            "peptide_ca_rmsd_10ns_A": (
                round(verdict.rmsd_10ns, 2) if verdict.rmsd_10ns is not None else None  # type: ignore[union-attr]
            ),
            "slope_A_per_ns": (
                round(verdict.slope_a_per_ns, 4)  # type: ignore[union-attr]
                if verdict.slope_a_per_ns is not None  # type: ignore[union-attr]
                else None
            ),
            "max_rmsd_A": round(verdict.max_rmsd, 2),  # type: ignore[union-attr]
            "abort_threshold_A": round(abort_thresh, 2),
            "receptor_fit_residual_A": round(verdict.receptor_fit_residual, 2),  # type: ignore[union-attr]
            "gate": "offline_mdtraj",
            "target": config.target,
            "peptide_id": config.peptide_id,
        }
        (output_dir / "early_abort.json").write_text(json.dumps(abort_meta, indent=2))
        logger.warning(
            "EARLY ABORT (%s): RMSD 5ns=%s 10ns=%s max=%.2f Å @ %.1f ns",
            verdict.reason,  # type: ignore[union-attr]
            f"{verdict.rmsd_5ns:.2f}" if verdict.rmsd_5ns is not None else "n/a",  # type: ignore[union-attr]
            f"{verdict.rmsd_10ns:.2f}" if verdict.rmsd_10ns is not None else "n/a",  # type: ignore[union-attr]
            verdict.max_rmsd,  # type: ignore[union-attr]
            ns_at_check,
        )

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
