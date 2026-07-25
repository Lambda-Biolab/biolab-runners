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
    prepare_simulation,
    quarantine_stale_checkpoint,
)
from biolab_runners.openmm.utils import (
    load_checkpoint_step,
    verify_production_outputs,
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
        _, remaining_steps, resume_xml = resume_state

        ctx = self._prepare_simulation(config, output_dir, resume_xml, result)
        if ctx is None:
            return result

        abort_thresh = config.target_irmsd_threshold_a * _ABORT_MULTIPLIER

        traj_path = str(output_dir / FileNames.TRAJECTORY)
        energy_path = str(output_dir / FileNames.ENERGY)
        state_xml_path = str(output_dir / FileNames.STATE_XML)

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
        return result

    def _resolve_skip_or_resume(
        self,
        force: bool,
        output_dir: Path,
        config: OpenMMConfig,
        result: SimulationResult,
    ) -> tuple[int, int, str] | None:
        """Handle idempotency + checkpoint resolution.

        When ``force=True`` and a prior checkpoint exists, the
        resumable files (``state.xml`` + ``checkpoint.json`` +
        ``energy.csv``) are moved into a timestamped
        ``output_dir/.stale/<UTC>/`` directory BEFORE this method
        returns. This guarantees the next non-forced invocation cannot
        pair a stale ``state.xml`` with a freshly-built topology, even
        if the forced run is interrupted before producing a new
        ``state.xml`` of its own.

        Returns None if the simulation is already complete (result populated),
        otherwise (start_step, remaining_steps, resume_xml).
        """
        if force:
            moved = quarantine_stale_checkpoint(output_dir)
            if moved:
                logger.info(
                    "force=True: quarantined %d stale checkpoint file(s) to %s",
                    len(moved),
                    moved[0].parent,
                )
        else:
            verification = verify_production_outputs(output_dir)
            if verification["complete"]:
                logger.info(
                    "Skipping MD — trajectory already complete at %s. Use force=True to re-run.",
                    output_dir / FileNames.TRAJECTORY,
                )
                result.trajectory_path = str(output_dir / FileNames.TRAJECTORY)
                result.energy_path = str(output_dir / FileNames.ENERGY)
                result.state_xml_path = str(output_dir / FileNames.STATE_XML)
                result.topology_path = str(output_dir / FileNames.TOPOLOGY)
                return None

        start_step = 0
        resume_xml = ""
        state_xml = output_dir / FileNames.STATE_XML
        # After a force=True quarantine, state_xml no longer exists on
        # disk (it was moved to .stale/) — the resume branch is naturally
        # skipped, and the fresh build proceeds without a stale state
        # to pair against.
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
            result.trajectory_path = str(output_dir / FileNames.TRAJECTORY)
            result.energy_path = str(output_dir / FileNames.ENERGY)
            result.state_xml_path = str(state_xml)
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
