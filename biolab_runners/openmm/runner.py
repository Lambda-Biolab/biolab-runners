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
from pathlib import Path

from biolab_runners.openmm.config import OpenMMConfig, SimulationResult
from biolab_runners.openmm.geometry import (
    collect_chain_ca_positions,
    min_pbc_distance,
)
from biolab_runners.openmm.offline_gate import (
    DEFAULT_GATE_10NS,
    GateVerdict,
    evaluate_trajectory,
    write_verdict_file,
)
from biolab_runners.openmm.paths import FileNames
from biolab_runners.openmm.system_builder import (
    SimulationContext,
    _atomic_save_checkpoint,
    prepare_simulation,
    quarantine_stale_checkpoint,
)
from biolab_runners.openmm.utils import (
    InvalidCheckpointError,
    is_run_complete,
    load_checkpoint,
)

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
        # ``start_step`` is the ABSOLUTE step the simulation will be at
        # when the production loop starts. For a fresh run, equil runs
        # inside _prepare_simulation, so the simulation is at
        # total_equil_steps when the loop starts. For a resumed run,
        # loadState sets the simulation to the saved step. The
        # production loop computes its absolute step as
        # ``start_step + steps_done`` (local).
        start_step, remaining_steps, resume_xml = resume_state

        ctx = self._prepare_simulation(config, output_dir, resume_xml, result)
        if ctx is None:
            return result

        abort_thresh = config.target_irmsd_threshold_a * _ABORT_MULTIPLIER

        traj_path = str(output_dir / FileNames.TRAJECTORY)
        energy_path = str(output_dir / FileNames.ENERGY)

        energy_fh = self._setup_reporters(ctx, config, traj_path, energy_path, remaining_steps)

        logger.info(
            "Starting production: %d steps (%.1f ns), checkpoint every %d steps",
            remaining_steps,
            remaining_steps * config.timestep_fs / 1e6,
            config.checkpoint_every_steps,
        )

        t0 = time.time()
        steps_done, abort_reason = self._run_production_loop(
            simulation=ctx.simulation,
            config=config,
            start_step=start_step,
            remaining_steps=remaining_steps,
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
            start_step=start_step,
            steps_done=steps_done,
            t0=t0,
            output_dir=output_dir,
        )
        return result

    def _resolve_skip_or_resume(
        self,
        force: bool,
        output_dir: Path,
        config: OpenMMConfig,
        result: SimulationResult,
    ) -> tuple[int, int, str] | None:
        """Handle idempotency + checkpoint resolution.

        Decision tree (in order):

        1. ``force=True`` ⇒ quarantine all resumable files, then
           proceed to a fresh build. The quarantine moves the
           manifest, the energy log, and every state file
           (``state*.xml``) into ``output_dir/.stale/<UTC>/`` BEFORE
           we return — so the next non-forced invocation (or this
           one, if interrupted mid-build) cannot pair a stale
           state with a fresh topology.

        2. Read the manifest via :func:`load_checkpoint`. The
           function validates the referenced state file (basename,
           expected name pattern, exists, non-empty); an invalid
           reference raises :class:`InvalidCheckpointError` which
           we convert into ``result.error`` and abort. The earlier
           "verify_production_outputs(...) → complete=True" gate
           was removed because a mid-production checkpoint can
           produce a large trajectory and many energy rows WITHOUT
           the run being complete — file presence does not imply
           completion.

        3. If the manifest is valid (step > 0, state file valid),
           call :func:`is_run_complete`: a run is terminal when
           ``manifest_step >= total_equil_steps + total_steps``
           (normal completion) OR ``early_abort.json`` exists with
           ``aborted=True`` (intentional early termination). If
           complete, populate the result (including
           ``state_xml_path`` from the manifest) and return None.

        4. Otherwise, return the resume tuple ``(start_step,
           remaining_steps, resume_xml)``. ``start_step`` is the
           absolute step the simulation will be at when the
           production loop starts (the manifest's step for resumes,
           ``total_equil_steps`` for fresh runs).

        5. If no manifest exists but state files (legacy or v7)
           are on disk, the checkpoint is orphaned — fail fast
           with ``force=True`` guidance. Pairing an orphan state
           with a freshly-built System would re-introduce the
           incompatibility the "Resume safety" / "Atomic
           checkpoint" rules exist to avoid.

        Returns None if the simulation is already complete (result
        populated) or the checkpoint is orphaned / invalid (result
        error set); otherwise (start_step, remaining_steps,
        resume_xml).

        ``start_step`` is the ABSOLUTE step the simulation will be at
        when the production loop starts:
        - Fresh run: ``config.total_equil_steps`` (equil runs inside
          ``_prepare_simulation`` before the loop).
        - Resumed run: the absolute step from the manifest (the
          ``loadState`` in ``prepare_simulation`` sets the simulator
          to this step).
        """
        if force:
            moved = quarantine_stale_checkpoint(output_dir)
            if moved:
                logger.info(
                    "force=True: quarantined %d stale checkpoint file(s) to %s",
                    len(moved),
                    moved[0].parent,
                )

        # Read the manifest. ``load_checkpoint`` validates the
        # referenced state file (basename, pattern, exists, size) —
        # InvalidCheckpointError surfaces a dangling or unsafe
        # reference immediately. The previous behaviour silently
        # accepted a manifest with a non-existent state file and
        # let ``prepare_simulation`` build a fresh System with
        # production accounting inherited from the (never-loaded)
        # checkpoint — a quietly-wrong fresh build.
        manifest_step = 0
        state_file = ""
        resume_xml = ""
        try:
            manifest_step, state_file = load_checkpoint(output_dir)
        except InvalidCheckpointError as exc:
            result.error = str(exc)
            logger.error(result.error)
            return None

        if manifest_step > 0:
            # A valid manifest exists. Determine completion from
            # the manifest step + early-abort metadata — NOT from
            # ``verify_production_outputs`` (a mid-production
            # checkpoint can produce a large trajectory and many
            # energy rows while the run is still in progress).
            target_step = config.total_equil_steps + config.total_steps
            complete, reason = is_run_complete(output_dir, config)
            if complete:
                logger.info(
                    "Skipping MD — run is already terminal (%s) at step %d",
                    reason,
                    manifest_step,
                )
                result.trajectory_path = str(output_dir / FileNames.TRAJECTORY)
                result.energy_path = str(output_dir / FileNames.ENERGY)
                result.topology_path = str(output_dir / FileNames.TOPOLOGY)
                result.state_xml_path = str(output_dir / state_file)
                return None

            resume_xml = str(output_dir / state_file)
            logger.info(
                "Resuming from checkpoint at step %d (%.2f ns of %d needed)",
                manifest_step,
                manifest_step * config.timestep_fs / 1e6,
                target_step,
            )
        else:
            # No valid manifest. If a state file exists from a
            # v6 run (legacy state.xml) or from a v7 interrupted
            # save that landed before the manifest rename, it is
            # orphaned — fail fast. Pairing the orphan with a
            # freshly-built System would re-introduce the
            # incompatibility this rule exists to avoid.
            leftover_states = list(output_dir.glob("state*.xml"))
            if leftover_states:
                result.error = (
                    f"State file(s) exist at {leftover_states} but "
                    f"the manifest {FileNames.CHECKPOINT_JSON} is "
                    f"missing or invalid — the saved state's step is "
                    f"unknown. Pairing it with a freshly-built System "
                    f"would re-introduce the incompatibility this rule "
                    f"exists to avoid. Re-run with force=True to "
                    f"discard the orphaned checkpoint."
                )
                logger.error(result.error)
                return None

        # Compute ``start_step`` (the absolute step the simulator is at
        # when the production loop starts) and ``remaining_steps``
        # (production steps to run). For a fresh run, equil will run
        # inside _prepare_simulation, so start_step = total_equil_steps.
        # For a resumed run, loadState sets the simulator to
        # ``manifest_step``, so start_step = manifest_step.
        start_step = manifest_step if manifest_step > 0 else config.total_equil_steps

        production_steps_done = max(0, start_step - config.total_equil_steps)
        remaining_steps = max(0, config.total_steps - production_steps_done)
        if remaining_steps == 0 and not force:
            # Defensive: a manifest at the END step is terminal, but
            # is_run_complete should have caught it. If we reach here
            # anyway (e.g. config changed mid-run), populate the
            # result and skip.
            logger.info(
                "No remaining steps — manifest at step %d, target %d",
                manifest_step,
                config.total_equil_steps + config.total_steps,
            )
            if state_file:
                result.trajectory_path = str(output_dir / FileNames.TRAJECTORY)
                result.energy_path = str(output_dir / FileNames.ENERGY)
                result.topology_path = str(output_dir / FileNames.TOPOLOGY)
                result.state_xml_path = str(output_dir / state_file)
            return None

        return start_step, remaining_steps, resume_xml

    def _prepare_simulation(
        self,
        config: OpenMMConfig,
        output_dir: Path,
        resume_xml: str,
        result: SimulationResult,
    ) -> SimulationContext | None:
        """Build the OpenMM simulation, then equilibrate unless resuming.

        Delegates system/forcefield/topology/integrator assembly to
        :func:`biolab_runners.openmm.system_builder.prepare_simulation`,
        then runs the equilibration protocol if the run is not a
        checkpoint resume.
        """
        ctx = prepare_simulation(config, output_dir, resume_xml, result)
        if ctx is None:
            return None
        if not ctx.is_resuming:
            self._run_equilibration(ctx, config, output_dir)
        return ctx

    @staticmethod
    def _setup_reporters(
        ctx: SimulationContext,
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

        traj_exists = Path(traj_path).exists()
        dcd_append = is_resuming and traj_exists
        if not is_resuming and traj_exists:
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
        ctx: SimulationContext,
        result: SimulationResult,
        energy_fh: object,
        traj_path: str,
        energy_path: str,
        start_step: int,
        steps_done: int,
        t0: float,
        output_dir: Path,
    ) -> None:
        """Save final state atomically (or skip if already saved), close reporters, populate result.

        The final state save goes through ``_atomic_save_checkpoint``
        so the manifest is updated together with the state file. The
        ``absolute_step`` written to the manifest is ``start_step +
        steps_done`` — the ABSOLUTE OpenMM step the simulation is at
        when the production loop ends. The v6 design wrote the
        invocation-local ``steps_done`` instead, which silently
        broke accounting on resumes.

        SUGGESTION (double-save): the production loop's last
        ``_maybe_checkpoint`` typically saves at the final step
        (because ``steps_done == remaining_steps`` triggers the
        unconditional save at the end of the loop). A second save
        here would re-serialise the entire state file (potentially
        hundreds of MB) just to rewrite the same content. We read
        the manifest and skip the save when the atomic commit has
        already happened at the same absolute step. The manifest
        rename is fast (~KB), so the extra read is cheap.

        ``result.state_xml_path`` is set from the committed state
        file (the basename returned by ``_atomic_save_checkpoint``
        when a save happens, or read from the manifest when the
        save is skipped, or read from the manifest for the
        no-resumed-step case). The runner contract is that the
        returned path names the commit-time state file the
        manifest references.
        """
        config = result.config
        elapsed = time.time() - t0
        total_ns = steps_done * config.timestep_fs / 1e6
        ns_per_day = (total_ns / elapsed) * 86400 if elapsed > 0 else 0

        absolute_step = start_step + steps_done
        state_basename = ""
        try:
            existing_step, existing_file = load_checkpoint(output_dir)
        except InvalidCheckpointError:
            # The manifest is in an odd state (e.g. empty state file)
            # — fall through and force a fresh save. The
            # quarantine stays for the user to inspect; the new
            # save at absolute_step will overwrite it.
            existing_step, existing_file = 0, ""

        if existing_step == absolute_step and existing_file:
            # The last ``_maybe_checkpoint`` already committed the
            # state at this absolute step. Reuse the manifest's
            # reference instead of re-serialising the state file.
            logger.info(
                "Final atomic save skipped — commit already at step %d (%s)",
                existing_step,
                existing_file,
            )
            state_basename = existing_file
        else:
            state_basename = _atomic_save_checkpoint(ctx.simulation, output_dir, absolute_step)

        energy_fh.close()  # type: ignore[union-attr]

        result.trajectory_path = traj_path
        result.energy_path = energy_path
        result.state_xml_path = str(output_dir / state_basename)
        result.total_ns = round(total_ns, 2)
        result.elapsed_seconds = round(elapsed, 1)
        result.ns_per_day = round(ns_per_day, 1)

        summary = {
            "total_steps": steps_done,
            "total_ns": result.total_ns,
            "elapsed_seconds": result.elapsed_seconds,
            "ns_per_day": result.ns_per_day,
            "platform": config.openmm_platform,
            "num_atoms": result.num_atoms,
            "trajectory": traj_path,
            "energy": energy_path,
            "state": str(output_dir / state_basename),
            "early_abort": result.early_abort,
            "abort_reason": result.abort_reason,
            "final_absolute_step": absolute_step,
        }
        (output_dir / FileNames.MD_SUMMARY_JSON).write_text(json.dumps(summary, indent=2))

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

    @staticmethod
    def _run_equilibration(ctx: SimulationContext, config: OpenMMConfig, output_dir: Path) -> None:
        """Run 3-stage equilibration protocol.

        Stage 1: NVT 100ps with strong restraints (k=1000 kJ/mol/nm^2)
        Stage 2: NPT 100ps with reduced restraints (k=100)
        Stage 3: NPT 200ps with gradual restraint ramp (100->0) + unrestrained
        """
        simulation = ctx.simulation
        restraint_force = ctx.restraint_force
        ca_indices = ctx.ca_indices
        unit = ctx.unit_mod
        chains = ctx.chains
        np = ctx.np_mod

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
        # downstream min_pbc_distance does its own PBC-correct min-image
        # math, so the input convention here only needs to stay consistent
        # with what the gate sees; unwrapped is the correct choice.
        eq_positions = (
            simulation.context.getState(getPositions=True)  # type: ignore[union-attr]
            .getPositions(asNumpy=True)
            .value_in_unit(unit.angstroms)  # type: ignore[union-attr]
        )
        rec_ca, pep_ca = collect_chain_ca_positions(chains, eq_positions)
        if not (rec_ca and pep_ca):
            return

        box_vecs = (
            simulation.context.getState()  # type: ignore[union-attr]
            .getPeriodicBoxVectors(asNumpy=True)
            .value_in_unit(unit.angstroms)  # type: ignore[union-attr]
        )
        min_dist = min_pbc_distance(rec_ca, pep_ca, box_vecs, np)
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
        (output_dir / FileNames.EQUILIBRATION_METADATA_JSON).write_text(
            json.dumps(eq_meta, indent=2)
        )

    def _run_production_loop(
        self,
        *,
        simulation: object,
        config: OpenMMConfig,
        start_step: int,
        remaining_steps: int,
        output_dir: Path,
        enable_early_abort: bool,
        abort_thresh: float,
        t0: float,
    ) -> tuple[int, str]:
        """Run the production MD loop with early-abort checks and checkpointing.

        ``start_step`` is the ABSOLUTE step the simulation is at when
        the loop starts (``total_equil_steps`` for fresh runs, the
        saved step for resumed runs). The atomic save helper writes
        ``start_step + steps_done`` to the manifest, so the v6
        silent-shortening bug cannot recur.

        Returns:
            (steps_done, abort_reason) — ``steps_done`` is the
            INVOCATION-LOCAL count (not absolute). The caller adds
            ``start_step`` to get the absolute step. ``abort_reason``
            is ``""`` if no early abort.
        """
        last_ckpt_step = 0
        abort_reason = ""
        steps_box = [0]
        self._install_sigterm_handler(simulation, output_dir, start_step, steps_box, config)

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
                    output_dir=output_dir,
                    start_step=start_step,
                    abort_thresh=abort_thresh,
                    config=config,
                    steps_done=steps_done,
                )
                if abort_reason:
                    break

            last_ckpt_step = self._maybe_checkpoint(
                simulation,
                output_dir,
                start_step,
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
        output_dir: Path,
        start_step: int,
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

        On ``abort=True``, atomically saves the state + manifest with
        the absolute step (``start_step + steps_done``) so the next
        non-forced invocation can resume from a consistent step.

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
            # Atomic save: state file + manifest committed together.
            # The manifest's step is the absolute OpenMM step
            # (``start_step + steps_done``), not the local counter.
            absolute_step = start_step + steps_done
            _atomic_save_checkpoint(simulation, output_dir, absolute_step)
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
        verdict: GateVerdict,
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
            verdict.rmsd_5ns
            if verdict.reason == "early_dissociation" and verdict.rmsd_5ns is not None
            else verdict.rmsd_10ns
            if verdict.rmsd_10ns is not None
            else verdict.max_rmsd
        )
        abort_meta = {
            "aborted": True,
            "abort_reason": verdict.reason,
            "abort_step": steps_done,
            "abort_ns": round(ns_at_check, 2),
            "peptide_ca_rmsd_A": round(primary_rmsd, 2),
            "peptide_ca_rmsd_5ns_A": (
                round(verdict.rmsd_5ns, 2) if verdict.rmsd_5ns is not None else None
            ),
            "peptide_ca_rmsd_10ns_A": (
                round(verdict.rmsd_10ns, 2) if verdict.rmsd_10ns is not None else None
            ),
            "slope_A_per_ns": (
                round(verdict.slope_a_per_ns, 4) if verdict.slope_a_per_ns is not None else None
            ),
            "max_rmsd_A": round(verdict.max_rmsd, 2),
            "abort_threshold_A": round(abort_thresh, 2),
            "receptor_fit_residual_A": round(verdict.receptor_fit_residual, 2),
            "gate": "offline_mdtraj",
            "target": config.target,
            "peptide_id": config.peptide_id,
        }
        (output_dir / FileNames.EARLY_ABORT_JSON).write_text(json.dumps(abort_meta, indent=2))
        logger.warning(
            "EARLY ABORT (%s): RMSD 5ns=%s 10ns=%s max=%.2f Å @ %.1f ns",
            verdict.reason,
            f"{verdict.rmsd_5ns:.2f}" if verdict.rmsd_5ns is not None else "n/a",
            f"{verdict.rmsd_10ns:.2f}" if verdict.rmsd_10ns is not None else "n/a",
            verdict.max_rmsd,
            ns_at_check,
        )

    @staticmethod
    def _install_sigterm_handler(
        simulation: object,
        output_dir: Path,
        start_step: int,
        steps_box: list[int],
        config: OpenMMConfig,
    ) -> None:
        """Install a SIGTERM handler that atomically saves state + manifest.

        Writes the absolute step (``start_step + steps_box[0]``) to
        the manifest so the next invocation can resume from the
        correct step. The atomic save (single ``os.replace`` on
        ``checkpoint.json``) ensures that a SIGTERM mid-save cannot
        leave a stale state paired with the wrong manifest step. If
        the save itself fails (e.g. disk full), the canonical
        manifest is unchanged and the runner sees the previous
        (coherent) checkpoint on the next invocation.
        """

        def handle_sigterm(signum: int, frame: object) -> None:  # noqa: ARG001
            steps_done = steps_box[0]
            absolute_step = start_step + steps_done
            ns_done = absolute_step * config.timestep_fs / 1e6
            logger.warning(
                "SIGTERM received at absolute step %d (%.2f ns) — saving state",
                absolute_step,
                ns_done,
            )
            try:
                _atomic_save_checkpoint(simulation, output_dir, absolute_step)
            except Exception as exc:
                logger.error("Failed to save state on SIGTERM: %s", exc)
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_sigterm)

    @staticmethod
    def _maybe_checkpoint(
        simulation: object,
        output_dir: Path,
        start_step: int,
        steps_done: int,
        last_ckpt_step: int,
        remaining_steps: int,
        config: OpenMMConfig,
        t0: float,
    ) -> int:
        """Write a checkpoint if interval elapsed. Returns the (possibly updated) last_ckpt_step.

        The save is atomic (the manifest rename is the single commit
        point via :func:`_atomic_save_checkpoint`) so that a crash
        mid-save cannot leave a stale state whose step does not match
        the manifest. The manifest records the absolute step
        (``start_step + steps_done``), not the invocation-local
        counter — the v6 protocol wrote the local counter and
        silently shortened resumed runs.
        """
        since_ckpt = steps_done - last_ckpt_step
        if since_ckpt < config.checkpoint_every_steps and steps_done < remaining_steps:
            return last_ckpt_step
        absolute_step = start_step + steps_done
        _atomic_save_checkpoint(simulation, output_dir, absolute_step)
        elapsed = time.time() - t0
        ns_done = absolute_step * config.timestep_fs / 1e6
        ns_per_day = (ns_done / elapsed) * 86400 if elapsed > 0 else 0
        logger.info(
            "Checkpoint: %d/%d steps (%.2f ns, %.0f ns/day)",
            steps_done,
            remaining_steps,
            ns_done,
            ns_per_day,
        )
        return steps_done
