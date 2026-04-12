"""OpenMM molecular dynamics simulation runner.

Runs production MD simulations for peptide-protein complexes using OpenMM.
The runner handles system building (PDBFixer + solvation), multi-stage
equilibration, and production NPT with periodic checkpointing, early abort
checks, and trajectory/energy output.

Designed for salivary ionic conditions (140 mM NaCl, 310 K) with
CHARMM36m/TIP3P force fields on GPU (OpenCL or CUDA).

Requires: openmm>=8.5.0, pdbfixer>=1.9

Usage::

    from biolab_runners.openmm import OpenMMRunner, OpenMMConfig

    config = OpenMMConfig(
        receptor_pdb="receptor.pdb",
        peptide_pdb="peptide.pdb",
        output_dir="results/md/GtfB/PEP001",
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
from pathlib import Path

from biolab_runners.openmm.config import OpenMMConfig, SimulationResult
from biolab_runners.openmm.utils import (
    load_checkpoint_step,
    verify_production_outputs,
)

logger = logging.getLogger(__name__)

# Per-target iRMSD thresholds for early abort (expert-validated)
TARGET_IRMSD_THRESHOLDS: dict[str, float] = {
    "FadA": 3.0,
    "FimA": 3.0,
    "HmuY": 3.5,
    "VicK": 3.5,
    "SpaP": 4.0,
    "GtfB": 4.0,
}
DEFAULT_IRMSD_THRESHOLD = 3.5

# Sub-chunk size for production loop (~5 min wall-clock on RTX 3090)
SUB_CHUNK_STEPS = 150_000


class OpenMMRunner:
    """OpenMM production MD simulation runner.

    Builds a solvated peptide-protein system, runs multi-stage equilibration,
    and performs production NPT dynamics with checkpointing and early abort
    checks.

    The runner supports:
    - Resuming from checkpoints (idempotent re-runs)
    - Dry-run mode (validates config without GPU)
    - Per-target early abort thresholds at 5 ns and 10 ns
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

        # Check idempotency
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
                return result

        # Check for checkpoint to resume from
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
            return result

        # Import OpenMM (deferred to avoid import errors when not installed)
        try:
            import numpy as np  # noqa: I001
            import openmm
            import openmm.app as app
            import openmm.unit as unit
        except ImportError as exc:
            result.error = f"OpenMM not installed: {exc}"
            logger.error(result.error)
            return result

        # Select platform
        try:
            platform = openmm.Platform.getPlatformByName(config.openmm_platform)
            if config.openmm_platform == "OpenCL":
                platform.setPropertyDefaultValue("Precision", "mixed")
            logger.info("Using platform: %s", platform.getName())
        except Exception as exc:
            result.error = f"Platform {config.openmm_platform} not available: {exc}"
            logger.error(result.error)
            return result

        # Resolve PDB paths
        receptor_pdb = self._resolve_pdb(config.receptor_pdb, "receptor.pdb")
        peptide_pdb = self._resolve_pdb(config.peptide_pdb, "peptide.pdb")

        # Build force field
        ff_name = config.protein_ff
        if "charmm" in ff_name.lower():
            forcefield = app.ForceField("charmm36.xml", "charmm36/water.xml")
        else:
            forcefield = app.ForceField(f"{ff_name}.xml", f"{config.water_model}.xml")

        is_resuming = bool(resume_xml and Path(resume_xml).exists())

        # Build or load solvated system
        topo_path = output_dir / "topology.pdb"
        existing_topo = (
            topo_path if topo_path.exists() and topo_path.stat().st_size > 100_000 else None
        )

        if is_resuming and existing_topo:
            logger.info("Resuming: loading solvated topology from %s", existing_topo)
            topo_pdb = app.PDBFile(str(existing_topo))
            modeller = app.Modeller(topo_pdb.topology, topo_pdb.positions)
            logger.info("Loaded solvated system: %d atoms", modeller.topology.getNumAtoms())
        else:
            modeller = self._build_system(receptor_pdb, peptide_pdb, config, app, forcefield)
            if modeller is None:
                result.error = "Failed to build system — no valid PDB files"
                return result

        # Save topology
        with open(str(topo_path), "w") as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

        result.num_atoms = modeller.topology.getNumAtoms()
        result.topology_path = str(topo_path)
        logger.info("Topology: %d atoms", result.num_atoms)

        # Create system
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
        )

        # Barostat for NPT
        system.addForce(
            openmm.MonteCarloBarostat(
                config.pressure_atm * unit.atmospheres,
                config.temperature_k * unit.kelvin,
                25,
            )
        )

        # Integrator
        integrator = openmm.LangevinMiddleIntegrator(
            config.temperature_k * unit.kelvin,
            1.0 / unit.picoseconds,
            config.timestep_fs * unit.femtoseconds,
        )

        # Restraint force (needed for both fresh and resume paths)
        chains = list(modeller.topology.chains())
        ca_indices: list[int] = []
        for chain in chains:
            for atom in chain.atoms():
                if atom.name == "CA":
                    ca_indices.append(atom.index)

        restraint_force = openmm.CustomExternalForce("k*periodicdistance(x,y,z,x0,y0,z0)^2")
        restraint_force.addGlobalParameter("k", 0.0)
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")
        for idx in ca_indices:
            pos = modeller.positions[idx]
            restraint_force.addParticle(idx, [pos.x, pos.y, pos.z])
        system.addForce(restraint_force)

        # Create simulation
        simulation = app.Simulation(modeller.topology, system, integrator, platform)
        simulation.context.setPositions(modeller.positions)

        # Resume or equilibrate
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

        # Early abort setup
        ref_pep_ca_idx: list[int] = []
        ref_rec_ca_idx: list[int] = []
        for chain_idx, chain in enumerate(chains):
            for atom in chain.atoms():
                if atom.name == "CA":
                    if chain_idx == 0:
                        ref_rec_ca_idx.append(atom.index)
                    else:
                        ref_pep_ca_idx.append(atom.index)

        ref_positions = (
            simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.angstroms)
        )
        ref_pep_ca = ref_positions[ref_pep_ca_idx].copy() if ref_pep_ca_idx else None

        irmsd_thresh = TARGET_IRMSD_THRESHOLDS.get(config.target, DEFAULT_IRMSD_THRESHOLD)
        abort_thresh = irmsd_thresh * 2.0
        abort_step_5ns = int(5_000_000 / config.timestep_fs)
        abort_step_10ns = int(10_000_000 / config.timestep_fs)

        # Production reporters
        traj_path = str(output_dir / "trajectory.dcd")
        energy_path = str(output_dir / "energy.csv")

        dcd_append = False
        if is_resuming and Path(traj_path).exists():
            dcd_append = True
        elif not is_resuming and Path(traj_path).exists():
            stale = Path(traj_path).with_suffix(".dcd.stale")
            Path(traj_path).rename(stale)

        simulation.reporters.append(
            app.DCDReporter(traj_path, config.save_every_steps, append=dcd_append)
        )

        if is_resuming and Path(energy_path).exists():
            energy_fh = open(energy_path, "a")  # noqa: SIM115
        else:
            energy_fh = open(energy_path, "w")  # noqa: SIM115

        simulation.reporters.append(
            app.StateDataReporter(
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
        simulation.reporters.append(
            app.StateDataReporter(
                sys.stdout,
                config.save_every_steps * 10,
                step=True,
                time=True,
                speed=True,
                remainingTime=True,
                totalSteps=remaining_steps,
            )
        )

        # SIGTERM handler for clean shutdown on preemption
        state_xml_path = str(output_dir / "state.xml")
        sigterm_received = False
        steps_done = 0

        def handle_sigterm(signum: int, frame: object) -> None:  # noqa: ARG001
            nonlocal sigterm_received
            sigterm_received = True
            ns_done = steps_done * config.timestep_fs / 1e6
            logger.warning(
                "SIGTERM received at step %d (%.2f ns) — saving state",
                steps_done,
                ns_done,
            )
            try:
                simulation.saveState(state_xml_path)
            except Exception as exc:
                logger.error("Failed to save state on SIGTERM: %s", exc)
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_sigterm)

        # Production loop
        logger.info(
            "Starting production: %d steps (%.1f ns), checkpoint every %d steps",
            remaining_steps,
            remaining_steps * config.timestep_fs / 1e6,
            config.checkpoint_every_steps,
        )

        t0 = time.time()
        last_ckpt_step = 0
        early_abort_done = False
        slope_check_done = False
        rmsd_5ns: float | None = None

        while steps_done < remaining_steps:
            chunk = min(SUB_CHUNK_STEPS, remaining_steps - steps_done)
            simulation.step(chunk)
            steps_done += chunk

            # 5 ns early abort check
            if (
                enable_early_abort
                and not early_abort_done
                and ref_pep_ca is not None
                and steps_done >= abort_step_5ns
            ):
                early_abort_done = True
                rmsd_5ns = self._check_early_abort_5ns(
                    simulation,
                    ref_pep_ca,
                    ref_pep_ca_idx,
                    ref_rec_ca_idx,
                    abort_thresh,
                    irmsd_thresh,
                    config,
                    steps_done,
                    output_dir,
                    unit,
                    np,
                )
                if rmsd_5ns is not None and rmsd_5ns > abort_thresh:
                    result.early_abort = True
                    result.abort_reason = "early_dissociation"
                    break

            # 10 ns slope check
            if (
                enable_early_abort
                and not slope_check_done
                and rmsd_5ns is not None
                and ref_pep_ca is not None
                and steps_done >= abort_step_10ns
            ):
                slope_check_done = True
                slope_abort = self._check_slope_10ns(
                    simulation,
                    ref_pep_ca,
                    ref_pep_ca_idx,
                    rmsd_5ns,
                    config,
                    steps_done,
                    output_dir,
                    unit,
                    np,
                )
                if slope_abort:
                    result.early_abort = True
                    result.abort_reason = "rmsd_slope_drift"
                    break

            # Periodic checkpoint
            since_ckpt = steps_done - last_ckpt_step
            if since_ckpt >= config.checkpoint_every_steps or steps_done >= remaining_steps:
                simulation.saveState(state_xml_path)
                last_ckpt_step = steps_done
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

        # Finalize
        elapsed = time.time() - t0
        total_ns = remaining_steps * config.timestep_fs / 1e6
        ns_per_day = (total_ns / elapsed) * 86400 if elapsed > 0 else 0

        simulation.saveState(state_xml_path)
        energy_fh.close()

        result.trajectory_path = traj_path
        result.energy_path = energy_path
        result.state_xml_path = state_xml_path
        result.total_ns = round(total_ns, 2)
        result.elapsed_seconds = round(elapsed, 1)
        result.ns_per_day = round(ns_per_day, 1)

        # Write summary
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

        return result

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

        modeller.addSolvent(
            forcefield,
            model=config.water_model,
            padding=config.box_padding_nm * unit.nanometers,
            boxShape=config.box_shape,
            ionicStrength=config.nacl_mol * unit.molar,
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
        simulation.minimizeEnergy(maxIterations=1000)  # type: ignore[union-attr]

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
        k_strong = 1000.0
        simulation.context.setParameter("k", k_strong)  # type: ignore[union-attr]
        logger.info("Equilibrating (NVT 100ps, k=%.0f kJ/mol/nm^2)...", k_strong)
        simulation.step(int(100_000 / timestep_fs))  # type: ignore[union-attr]

        # Stage 2: NPT with reduced restraints
        k_medium = 100.0
        simulation.context.setParameter("k", k_medium)  # type: ignore[union-attr]
        logger.info("Equilibrating (NPT 100ps, k=%.0f kJ/mol/nm^2)...", k_medium)
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

        # Post-equilibration displacement check
        eq_positions = (
            simulation.context.getState(getPositions=True)  # type: ignore[union-attr]
            .getPositions(asNumpy=True)
            .value_in_unit(unit.angstroms)  # type: ignore[union-attr]
        )
        rec_ca = []
        pep_ca = []
        for chain_idx, chain in enumerate(chains):
            for atom in chain.atoms():  # type: ignore[union-attr]
                if atom.name == "CA":
                    if chain_idx == 0:
                        rec_ca.append(eq_positions[atom.index])
                    else:
                        pep_ca.append(eq_positions[atom.index])
        if rec_ca and pep_ca:
            rec_arr = np.array(rec_ca)  # type: ignore[union-attr]
            pep_arr = np.array(pep_ca)  # type: ignore[union-attr]
            box_vecs = (
                simulation.context.getState()  # type: ignore[union-attr]
                .getPeriodicBoxVectors(asNumpy=True)
                .value_in_unit(unit.angstroms)  # type: ignore[union-attr]
            )
            box_diag = np.array([box_vecs[0][0], box_vecs[1][1], box_vecs[2][2]])  # type: ignore[union-attr]
            diffs = rec_arr[:, None, :] - pep_arr[None, :, :]
            diffs -= np.round(diffs / box_diag) * box_diag  # type: ignore[union-attr]
            dists = np.sqrt((np.square(diffs)).sum(axis=-1))  # type: ignore[union-attr]
            min_dist = float(dists.min())
            logger.info(
                "Post-equilibration peptide-receptor Ca min distance: %.1f A",
                min_dist,
            )
            if min_dist > 8.0:
                logger.warning(
                    "DISPLACEMENT: peptide-receptor distance %.1f A after equilibration",
                    min_dist,
                )

            # Write equilibration metadata
            eq_meta = {
                "min_ca_distance_A": round(min_dist, 2),
                "displaced": min_dist > 8.0,
                "threshold_A": 8.0,
            }
            (output_dir / "equilibration_metadata.json").write_text(json.dumps(eq_meta, indent=2))

    @staticmethod
    def _check_early_abort_5ns(
        simulation: object,
        ref_pep_ca: object,
        ref_pep_ca_idx: list[int],
        ref_rec_ca_idx: list[int],  # noqa: ARG004
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
        state = simulation.context.getState(getPositions=True)  # type: ignore[union-attr]
        cur_pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstroms)  # type: ignore[union-attr]
        cur_pep_ca = cur_pos[ref_pep_ca_idx]

        # PBC correction
        box_vecs = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstroms)  # type: ignore[union-attr]
        box_diag = np.array([box_vecs[0][0], box_vecs[1][1], box_vecs[2][2]])  # type: ignore[union-attr]

        diff = cur_pep_ca - ref_pep_ca
        diff -= np.round(diff / box_diag) * box_diag  # type: ignore[union-attr]
        pep_rmsd = float(np.sqrt((diff**2).sum(axis=1).mean()))  # type: ignore[union-attr]

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
        state = simulation.context.getState(getPositions=True)  # type: ignore[union-attr]
        cur_pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstroms)  # type: ignore[union-attr]
        cur_pep_ca = cur_pos[ref_pep_ca_idx]

        box_vecs = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstroms)  # type: ignore[union-attr]
        box_diag = np.array([box_vecs[0][0], box_vecs[1][1], box_vecs[2][2]])  # type: ignore[union-attr]

        diff = cur_pep_ca - ref_pep_ca
        diff -= np.round(diff / box_diag) * box_diag  # type: ignore[union-attr]
        rmsd_10ns = float(np.sqrt((diff**2).sum(axis=1).mean()))  # type: ignore[union-attr]

        slope = (rmsd_10ns - rmsd_5ns) / 5.0  # A/ns
        ns_at_check = steps_done * config.timestep_fs / 1e6
        logger.info(
            "10 ns check: RMSD = %.1f A, slope = %.3f A/ns (threshold 0.05)",
            rmsd_10ns,
            slope,
        )

        if slope > 0.05:
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
                "slope_threshold": 0.05,
                "target": config.target,
                "peptide_id": config.peptide_id,
            }
            (output_dir / "early_abort.json").write_text(json.dumps(abort_meta, indent=2))
            return True

        return False
