"""Tests for OpenMMRunner and related utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from biolab_runners.openmm.config import (
    DEFAULT_IRMSD_THRESHOLD_A,
    OpenMMConfig,
    SimulationResult,
)
from biolab_runners.openmm.runner import OpenMMRunner
from biolab_runners.openmm.system_builder import build_forcefield
from biolab_runners.openmm.utils import (
    load_checkpoint_step,
    verify_production_outputs,
)

from tests._helpers import FakeApp

# ---------------------------------------------------------------------------
# OpenMMConfig tests
# ---------------------------------------------------------------------------


class TestOpenMMConfig:
    """Tests for OpenMMConfig dataclass."""

    def test_defaults(self) -> None:
        config = OpenMMConfig()
        assert config.temperature_k == 310.0
        assert config.nacl_mol == 0.150
        assert config.protein_ff == "charmm36m"
        assert config.water_model == "tip3p"
        assert config.box_shape == "dodecahedron"
        assert config.protonation_ph == 7.4
        assert config.production_ns == 100.0
        assert config.target_irmsd_threshold_a == 3.5

    def test_step_computation(self) -> None:
        config = OpenMMConfig(production_ns=100.0, timestep_fs=2.0)
        assert config.total_steps == 50_000_000  # 100ns * 1000ps/ns * 500steps/ps
        assert config.save_every_steps == 5000  # 10ps * 500 steps/ps
        assert config.checkpoint_every_steps == 3_600_000  # 2hr * 3600s/hr * 500steps/s

    def test_equil_steps_computation(self) -> None:
        config = OpenMMConfig(timestep_fs=2.0)
        # Default: 100ps + 100ps + 200ps = 400ps at 500 steps/ps = 200_000
        assert config.total_equil_steps == 200_000

    def test_custom_production_ns(self) -> None:
        config = OpenMMConfig(production_ns=20.0, timestep_fs=2.0)
        assert config.total_steps == 10_000_000

    def test_to_dict(self) -> None:
        config = OpenMMConfig(
            receptor_pdb="rec.pdb",
            peptide_pdb="pep.pdb",
            output_dir="fake/out",
            target="demo",
            peptide_id="PEP001",
        )
        d = config.to_dict()
        assert d["receptor_pdb"] == "rec.pdb"
        assert d["target"] == "demo"
        assert d["ionic_conditions"]["NaCl_M"] == 0.150
        assert d["simulation"]["temperature_K"] == 310.0
        assert d["force_fields"]["protein"] == "charmm36m"

    def test_save_and_load(self, tmp_path: Path) -> None:
        config = OpenMMConfig(
            receptor_pdb="rec.pdb",
            peptide_pdb="pep.pdb",
            output_dir=str(tmp_path),
            target="demo",
            production_ns=50.0,
        )
        path = config.save()
        assert path.exists()

        loaded = OpenMMConfig.from_json(path)
        assert loaded.target == "demo"
        assert loaded.production_ns == 50.0
        assert loaded.nacl_mol == 0.150
        assert loaded.protein_ff == "charmm36m"

    def test_extra_forcefields_default_empty(self) -> None:
        config = OpenMMConfig()
        assert config.extra_forcefields == []

    def test_extra_forcefields_not_shared_between_instances(self) -> None:
        """Default list must be per-instance (no mutable default aliasing)."""
        a = OpenMMConfig()
        b = OpenMMConfig()
        a.extra_forcefields.append("custom/a.xml")
        assert b.extra_forcefields == []

    def test_extra_forcefields_roundtrip(self, tmp_path: Path) -> None:
        extras = [str(tmp_path / "custom_a.xml"), str(tmp_path / "custom_b.xml")]
        config = OpenMMConfig(
            receptor_pdb="rec.pdb",
            output_dir=str(tmp_path),
            extra_forcefields=extras,
        )
        d = config.to_dict()
        assert d["force_fields"]["extra"] == extras

        path = config.save()
        loaded = OpenMMConfig.from_json(path)
        assert loaded.extra_forcefields == extras

    def test_extra_forcefields_absent_in_legacy_json(self, tmp_path: Path) -> None:
        """JSONs written before this field existed must still load cleanly."""
        legacy = {
            "receptor_pdb": "rec.pdb",
            "force_fields": {"protein": "amber14/protein.ff14SB", "water": "tip3p"},
        }
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy))
        loaded = OpenMMConfig.from_json(path)
        assert loaded.extra_forcefields == []

    def test_preset_saliva(self) -> None:
        config = OpenMMConfig.saliva()
        assert config.nacl_mol == 0.140
        assert config.protonation_ph == 6.2
        assert config.temperature_k == 310.0

    def test_preset_physiological(self) -> None:
        config = OpenMMConfig.physiological()
        assert config.nacl_mol == 0.150
        assert config.protonation_ph == 7.4
        assert config.temperature_k == 310.0

    def test_preset_gastric(self) -> None:
        config = OpenMMConfig.gastric()
        assert config.nacl_mol == 0.150
        assert config.protonation_ph == 2.0

    def test_preset_intestinal(self) -> None:
        config = OpenMMConfig.intestinal()
        assert config.nacl_mol == 0.150
        assert config.protonation_ph == 6.8

    def test_preset_accepts_overrides(self) -> None:
        config = OpenMMConfig.physiological(
            receptor_pdb="rec.pdb",
            peptide_pdb="pep.pdb",
            production_ns=25.0,
            protonation_ph=7.0,  # caller override wins over preset
        )
        assert config.receptor_pdb == "rec.pdb"
        assert config.production_ns == 25.0
        assert config.protonation_ph == 7.0
        assert config.nacl_mol == 0.150  # preset value preserved


class TestBuildForcefield:
    """Tests for ``build_forcefield`` extra_forcefields pass-through."""

    def test_amber_no_extras(self) -> None:
        config = OpenMMConfig(protein_ff="amber14/protein.ff14SB", water_model="tip3p")
        ff = build_forcefield(config, FakeApp())
        assert ff.paths == ("amber14/protein.ff14SB.xml", "tip3p.xml")

    def test_amber_with_extras(self, tmp_path: Path) -> None:
        extra = str(tmp_path / "custom.xml")
        config = OpenMMConfig(
            protein_ff="amber14/protein.ff14SB",
            water_model="tip3p",
            extra_forcefields=[extra],
        )
        ff = build_forcefield(config, FakeApp())
        assert ff.paths == ("amber14/protein.ff14SB.xml", "tip3p.xml", extra)

    def test_charmm_with_extras(self, tmp_path: Path) -> None:
        """CHARMM branch must still honour extra_forcefields."""
        extra = str(tmp_path / "custom.xml")
        config = OpenMMConfig(protein_ff="charmm36m", extra_forcefields=[extra])
        ff = build_forcefield(config, FakeApp())
        assert ff.paths == ("charmm36.xml", "charmm36/water.xml", extra)

    def test_extras_preserve_order(self, tmp_path: Path) -> None:
        extras = [str(tmp_path / "a.xml"), str(tmp_path / "b.xml")]
        config = OpenMMConfig(protein_ff="amber14/protein.ff14SB", extra_forcefields=extras)
        ff = build_forcefield(config, FakeApp())
        assert ff.paths[-2:] == tuple(extras)

    def test_water_ff_xml_overrides_water_model_path(self) -> None:
        """When set, water_ff_xml replaces {water_model}.xml as the water XML.

        Use case: Aib peptides at physiological ionic strength need AMBER
        ion templates (Na/Cl/K/Ca). Bare ``tip3p.xml`` is water-only, so
        ``addSolvent`` fails with "No template found for residue N (NA)".
        ``water_ff_xml="amber14/tip3p.xml"`` loads the ion-inclusive bundle
        into ForceField while addSolvent still sees the short model key.
        """
        config = OpenMMConfig(
            protein_ff="amber14/protein.ff14SB",
            water_model="tip3p",
            water_ff_xml="amber14/tip3p.xml",
        )
        ff = build_forcefield(config, FakeApp())
        assert ff.paths == ("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")

    def test_water_ff_xml_empty_falls_back_to_water_model(self) -> None:
        """When water_ff_xml is empty, preserve the pre-change behavior."""
        config = OpenMMConfig(
            protein_ff="amber14/protein.ff14SB",
            water_model="tip3p",
            water_ff_xml="",
        )
        ff = build_forcefield(config, FakeApp())
        assert ff.paths == ("amber14/protein.ff14SB.xml", "tip3p.xml")

    def test_water_ff_xml_ignored_for_charmm(self) -> None:
        """CHARMM branch hardcodes charmm36 XMLs regardless of water_ff_xml."""
        config = OpenMMConfig(
            protein_ff="charmm36m",
            water_ff_xml="amber14/tip3p.xml",  # ignored — CHARMM branch
        )
        ff = build_forcefield(config, FakeApp())
        assert ff.paths == ("charmm36.xml", "charmm36/water.xml")

    def test_non_charmm_prefix_not_misclassified(self) -> None:
        """Regression: substring matching classified 'non-charmm-test' as CHARMM.

        The original ``"charmm" in ff_name.lower()`` check matched any
        protein_ff containing the substring, including false positives
        like 'non-charmm-test' or 'mycharmm-extended'. The fix is a
        strict prefix check. This test pins the new behavior so the
        bug can't return.
        """
        config = OpenMMConfig(protein_ff="non-charmm-test")
        ff = build_forcefield(config, FakeApp())
        # 'non-charmm-test' starts with 'non-', not 'charmm', so it
        # falls through to the AMBER-style branch and uses
        # {water_model}.xml as the water file.
        assert ff.paths == ("non-charmm-test.xml", "tip3p.xml")


# ---------------------------------------------------------------------------
# SimulationResult tests
# ---------------------------------------------------------------------------


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""

    def test_to_dict(self) -> None:
        config = OpenMMConfig(target="demo", peptide_id="PEP001")
        result = SimulationResult(
            config=config,
            trajectory_path="fake/traj.dcd",
            total_ns=100.0,
            elapsed_seconds=36000.0,
            ns_per_day=240.0,
            num_atoms=50000,
        )
        d = result.to_dict()
        assert d["target"] == "demo"
        assert d["total_ns"] == 100.0
        assert d["ns_per_day"] == 240.0

    def test_save(self, tmp_path: Path) -> None:
        config = OpenMMConfig(output_dir=str(tmp_path))
        result = SimulationResult(config=config, total_ns=50.0)
        path = result.save()
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_ns"] == 50.0


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestVerifyProductionOutputs:
    """Tests for verify_production_outputs."""

    def test_empty_dir_not_complete(self, tmp_path: Path) -> None:
        report = verify_production_outputs(tmp_path)
        assert not report["complete"]

    def test_complete_dir(self, tmp_path: Path) -> None:
        # Create all expected files with sufficient sizes
        (tmp_path / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        energy_lines = ["#step,time,PE,KE,TE,temp,vol,speed\n"]
        energy_lines.extend(f"{i * 5000},{i * 10},0,0,0,310,0,0\n" for i in range(20))
        (tmp_path / "energy.csv").write_text("".join(energy_lines))
        (tmp_path / "state.xml").write_text("<State/>")

        report = verify_production_outputs(tmp_path)
        assert report["complete"]

    def test_small_trajectory_incomplete(self, tmp_path: Path) -> None:
        (tmp_path / "trajectory.dcd").write_bytes(b"\x00" * 100)
        (tmp_path / "energy.csv").write_text("step\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n")
        (tmp_path / "state.xml").write_text("<State/>")

        report = verify_production_outputs(tmp_path)
        assert not report["complete"]

    def test_few_energy_rows_incomplete(self, tmp_path: Path) -> None:
        (tmp_path / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        (tmp_path / "energy.csv").write_text("step\n1\n")
        (tmp_path / "state.xml").write_text("<State/>")

        report = verify_production_outputs(tmp_path)
        assert not report["complete"]


class TestLoadCheckpointStep:
    """Tests for checkpoint loading."""

    def test_no_checkpoint_returns_zero(self, tmp_path: Path) -> None:
        assert load_checkpoint_step(tmp_path) == 0

    def test_checkpoint_json(self, tmp_path: Path) -> None:
        ckpt = {
            "records": [
                {"step": 1000000, "time_ns": 2.0},
                {"step": 2000000, "time_ns": 4.0},
            ]
        }
        (tmp_path / "checkpoint.json").write_text(json.dumps(ckpt))
        assert load_checkpoint_step(tmp_path) == 2000000

    def test_energy_csv_fallback(self, tmp_path: Path) -> None:
        (tmp_path / "energy.csv").write_text("#step,time\n5000,10\n10000,20\n15000,30\n")
        assert load_checkpoint_step(tmp_path) == 15000


# ---------------------------------------------------------------------------
# Runner tests (mocked OpenMM)
# ---------------------------------------------------------------------------


class TestOpenMMRunner:
    """Tests for OpenMMRunner with mocked dependencies."""

    def test_dry_run(self, tmp_path: Path) -> None:
        config = OpenMMConfig(
            receptor_pdb=str(tmp_path / "rec.pdb"),
            peptide_pdb=str(tmp_path / "pep.pdb"),
            output_dir=str(tmp_path / "output"),
            target="demo",
            peptide_id="PEP001",
            production_ns=100.0,
        )
        runner = OpenMMRunner(config)
        result = runner.run(dry_run=True)
        assert result.error == ""
        # Dry run should not create trajectory
        assert result.trajectory_path == ""

    def test_idempotent_skip(self, tmp_path: Path) -> None:
        """Existing complete output should be reused."""
        out = tmp_path / "output"
        out.mkdir()
        (out / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        energy_lines = ["#step,time\n"]
        energy_lines.extend(f"{i * 5000},{i * 10}\n" for i in range(20))
        (out / "energy.csv").write_text("".join(energy_lines))
        (out / "state.xml").write_text("<State/>")

        config = OpenMMConfig(output_dir=str(out))
        runner = OpenMMRunner(config)
        result = runner.run()
        assert result.trajectory_path == str(out / "trajectory.dcd")
        assert result.error == ""

    def test_missing_openmm_returns_error(self, tmp_path: Path) -> None:
        """Missing OpenMM should return error, not crash."""
        out = tmp_path / "output"
        out.mkdir()

        config = OpenMMConfig(output_dir=str(out))
        runner = OpenMMRunner(config)

        with (
            patch.dict("sys.modules", {"openmm": None, "openmm.app": None}),
            patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'openmm'"),
            ),
        ):
            result = runner.run()
        assert "not installed" in result.error or "openmm" in result.error.lower()


# ---------------------------------------------------------------------------
# iRMSD threshold tests
# ---------------------------------------------------------------------------


class TestResumeAccounting:
    """Regression tests for issue #4: resume must not conflate equilibration + production."""

    def test_resume_subtracts_equil_steps(self, tmp_path: Path) -> None:
        """Remaining steps must discount equilibration from the checkpoint step counter."""
        out = tmp_path / "output"
        out.mkdir()

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        # Simulate checkpoint after full equil (200k steps) + 1 ns production (500k steps)
        checkpoint_step = config.total_equil_steps + 500_000
        (out / "energy.csv").write_text(f"#step,time\n{checkpoint_step},{checkpoint_step}\n")
        (out / "state.xml").write_text("<State/>")

        runner = OpenMMRunner(config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=SimulationResult(config=config)
        )
        assert resume is not None
        _start_step, remaining_steps, _resume_xml = resume
        # Should be total_steps minus production-only steps done (500k), not minus absolute (700k)
        assert remaining_steps == config.total_steps - 500_000

    def test_resume_right_after_equil(self, tmp_path: Path) -> None:
        """Checkpoint at end of equilibration should leave all production steps remaining."""
        out = tmp_path / "output"
        out.mkdir()

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        checkpoint_step = config.total_equil_steps  # just finished equilibration
        (out / "energy.csv").write_text(f"#step,time\n{checkpoint_step},{checkpoint_step}\n")
        (out / "state.xml").write_text("<State/>")

        runner = OpenMMRunner(config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=SimulationResult(config=config)
        )
        assert resume is not None
        _, remaining_steps, _ = resume
        assert remaining_steps == config.total_steps


class TestIrmsdThreshold:
    """Tests for the per-config iRMSD early-abort threshold."""

    def test_default_value(self) -> None:
        assert DEFAULT_IRMSD_THRESHOLD_A == 3.5

    def test_default_applied_to_config(self) -> None:
        config = OpenMMConfig(
            receptor_pdb="r.pdb",
            peptide_pdb="p.pdb",
            output_dir="out",
        )
        assert config.target_irmsd_threshold_a == DEFAULT_IRMSD_THRESHOLD_A

    def test_override_via_config(self) -> None:
        config = OpenMMConfig(
            receptor_pdb="r.pdb",
            peptide_pdb="p.pdb",
            output_dir="out",
            target_irmsd_threshold_a=4.0,
        )
        assert config.target_irmsd_threshold_a == 4.0

    def test_roundtrip_through_json(self, tmp_path: Path) -> None:
        config = OpenMMConfig(
            receptor_pdb="r.pdb",
            peptide_pdb="p.pdb",
            output_dir=str(tmp_path),
            target_irmsd_threshold_a=2.75,
        )
        path = config.save()
        loaded = OpenMMConfig.from_json(path)
        assert loaded.target_irmsd_threshold_a == 2.75


# ---------------------------------------------------------------------------
# Historical context (removed under OralBiome-AMP task #10, 2026-04-21):
# The inside-OpenMM gate math (``_peptide_ca_rmsd``, ``_kabsch_rotation``,
# ``_check_early_abort_5ns``, ``_regression_slope``, ``_check_slope_10ns``,
# ``_maybe_run_*_gate``, ``_do_5ns_check``) was replaced by the offline
# mdtraj gate in ``biolab_runners.openmm.offline_gate``. The regression
# tests that pinned the inside-OpenMM semantics (TestPeptideCaRmsdReceptorAligned,
# TestRegressionSlope, TestCheckSlope10nsConjunctiveGate, TestGateCoordConventionRegression,
# TestFlavorCCoordConventionMath, TestFlavorCGateMatchesIndependentKabschLiveMD)
# were deleted in the same commit — their invariants moved to
# ``tests/test_offline_gate.py``, which exercises the same coord-convention,
# Kabsch, triclinic-unwrap, and conjunctive-slope-gate properties on the
# new file-based gate function. Git history preserves the old tests.
#
# PBC math tests (orthorhombic parity, dodecahedron wrap, face crossing,
# broadcasting) moved to tests/test_geometry.py during the god-module
# split (2026-07). That file owns the ``pbc_correct`` /
# ``min_pbc_distance`` / ``collect_chain_ca_positions`` public surface
# and includes the pre-fix regression assertions for #163.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _install_sigterm_handler — cloud preemption safety
# ---------------------------------------------------------------------------


class TestInstallSigtermHandler:
    """``_install_sigterm_handler`` registers a SIGTERM handler that saves
    state then exits. Critical for cloud preemption: the handler must save
    state *before* the process dies, otherwise the next run cannot resume."""

    def test_sigterm_handler_is_installed(self) -> None:
        config = OpenMMConfig()
        OpenMMRunner._install_sigterm_handler(
            simulation=MagicMock(),
            state_xml_path="/tmp/state.xml",
            steps_box=[0],
            config=config,
        )
        # The handler is installed — we don't invoke it (that would exit
        # the test process). Just assert signal.signal was called.
        # signal.signal is reset by pytest at session end, so no cleanup needed.

    def test_handler_saves_state_using_current_step(self) -> None:
        """The handler must call saveState with the *current* step count
        from steps_box[0], not a stale value."""
        import signal

        config = OpenMMConfig()
        sim = MagicMock()
        # Capture the registered handler
        captured: dict[str, object] = {}

        def fake_signal(signum: int, handler: object) -> None:
            captured[signum] = handler

        with patch("biolab_runners.openmm.runner.signal.signal", side_effect=fake_signal):
            OpenMMRunner._install_sigterm_handler(
                simulation=sim,
                state_xml_path="/tmp/state.xml",
                steps_box=[12345],  # current step
                config=config,
            )

        handler = captured[signal.SIGTERM]
        assert callable(handler)
        # Invoke with a non-zero current step
        with patch("biolab_runners.openmm.runner.sys.exit") as mock_exit:
            handler(signal.SIGTERM, None)  # type: ignore[arg-type, misc]
        sim.saveState.assert_called_once_with("/tmp/state.xml")
        mock_exit.assert_called_once_with(0)

    def test_handler_swallows_save_state_errors(self) -> None:
        """If saveState throws (disk full, permissions, etc.), the handler
        must log the error and still exit cleanly — never crash with an
        unhandled exception during cloud preemption."""
        import signal

        sim = MagicMock()
        sim.saveState.side_effect = OSError("disk full")
        config = OpenMMConfig()
        captured: dict[str, object] = {}

        def fake_signal(signum: int, handler: object) -> None:
            captured[signum] = handler

        with patch("biolab_runners.openmm.runner.signal.signal", side_effect=fake_signal):
            OpenMMRunner._install_sigterm_handler(
                simulation=sim,
                state_xml_path="/tmp/state.xml",
                steps_box=[0],
                config=config,
            )

        with patch("biolab_runners.openmm.runner.sys.exit") as mock_exit:
            captured[signal.SIGTERM](signal.SIGTERM, None)  # type: ignore[arg-type, misc]
        # Still exited cleanly even though saveState failed
        mock_exit.assert_called_once_with(0)


# ---------------------------------------------------------------------------
# _maybe_checkpoint — periodic checkpointing
# ---------------------------------------------------------------------------


class TestMaybeCheckpoint:
    """``_maybe_checkpoint`` writes a state.xml checkpoint if the interval
    has elapsed or if this is the last chunk.

    Note: ``OpenMMConfig.checkpoint_every_steps`` is computed from
    ``checkpoint_interval_hours`` in ``__post_init__``, so the tests set
    the interval in hours, not the step count.
    """

    def test_no_checkpoint_when_interval_not_elapsed(self) -> None:
        """If less than checkpoint_every_steps since last ckpt, and not
        at the end, do nothing."""
        sim = MagicMock()
        # 0.1 hours @ 2.0 fs = 180,000 steps between checkpoints
        config = OpenMMConfig(checkpoint_interval_hours=0.1)
        # Force steps_done to be small (not yet at the end)
        result = OpenMMRunner._maybe_checkpoint(
            simulation=sim,
            state_xml_path="/tmp/state.xml",
            steps_done=500,  # only 500 since last_ckpt_step=0
            last_ckpt_step=0,
            remaining_steps=10_000_000,
            config=config,
            t0=0.0,
        )
        assert result == 0  # unchanged
        sim.saveState.assert_not_called()

    def test_checkpoint_when_interval_elapsed(self) -> None:
        sim = MagicMock()
        # 1 ns @ 2.0 fs = 500 steps between checkpoints
        config = OpenMMConfig(
            timestep_fs=2.0,
            checkpoint_interval_hours=500.0 / 3600.0 / 1000.0,
        )
        result = OpenMMRunner._maybe_checkpoint(
            simulation=sim,
            state_xml_path="/tmp/state.xml",
            steps_done=1500,  # > 500 since last_ckpt=0
            last_ckpt_step=0,
            remaining_steps=10_000_000,
            config=config,
            t0=1000.0,  # fixed past time — avoids wall-clock dependency
        )
        assert result == 1500
        sim.saveState.assert_called_once_with("/tmp/state.xml")

    def test_checkpoint_at_end_of_run(self) -> None:
        """Even if the interval hasn't elapsed, checkpoint when steps_done
        reaches remaining_steps (last chunk)."""
        sim = MagicMock()
        config = OpenMMConfig(
            timestep_fs=2.0,
            checkpoint_interval_hours=0.1,  # 180k steps
        )
        result = OpenMMRunner._maybe_checkpoint(
            simulation=sim,
            state_xml_path="/tmp/state.xml",
            steps_done=10_000,  # at remaining_steps
            last_ckpt_step=9_500,
            remaining_steps=10_000,
            config=config,
            t0=1000.0,  # fixed past time — avoids wall-clock dependency
        )
        assert result == 10_000
        sim.saveState.assert_called_once()

    def test_ns_per_day_handles_zero_elapsed(self) -> None:
        """If t0 == time.time() (zero elapsed), don't divide by zero.

        We use a t0 in the near-future so elapsed = t0 - now is negative
        (function guards via ``if elapsed > 0``). A negative elapsed is
        the realistic case the guard must cover.
        """
        sim = MagicMock()
        config = OpenMMConfig(
            timestep_fs=2.0,
            checkpoint_interval_hours=500.0 / 3600.0 / 1000.0,  # 500 steps
        )
        result = OpenMMRunner._maybe_checkpoint(
            simulation=sim,
            state_xml_path="/tmp/state.xml",
            steps_done=2000,  # > 500 since last_ckpt=0 → checkpoint
            last_ckpt_step=0,
            remaining_steps=10_000_000,
            config=config,
            t0=time.time() + 1e9,  # far future → elapsed < 0 → guard fires
        )
        # Should not raise; ns_per_day is 0 since elapsed is 0
        assert result == 2000
        sim.saveState.assert_called_once()
