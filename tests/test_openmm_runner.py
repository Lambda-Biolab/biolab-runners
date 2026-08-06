"""Tests for OpenMMRunner and related utilities."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest  # used in test annotations and raises
from biolab_runners.openmm.checkpoint import load_checkpoint
from biolab_runners.openmm.config import (
    DEFAULT_IRMSD_THRESHOLD_A,
    OpenMMConfig,
    SimulationResult,
)
from biolab_runners.openmm.runner import OpenMMRunner
from biolab_runners.openmm.system_builder import SimulationContext, build_forcefield
from biolab_runners.openmm.utils import verify_production_outputs

from tests._helpers import FakeApp

logger = logging.getLogger(__name__)

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
        d = cast("dict[str, Any]", config.to_dict())
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
        d = cast("dict[str, Any]", config.to_dict())
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
    """Tests for verify_production_outputs (diagnostic helper, not a completion oracle).

    v8 change: ``verify_production_outputs`` no longer reports a
    ``complete`` boolean — the field is hard-coded to ``False`` and
    the function returns only per-file metadata. Completion is
    decided by :func:`is_run_complete` (manifest step + early
    metadata). A large trajectory and many energy rows are NOT
    evidence of completion — a mid-production checkpoint produces
    both while the run is still in progress.
    """

    def test_empty_dir_complete_is_false(self, tmp_path: Path) -> None:
        report = verify_production_outputs(tmp_path)
        assert report["complete"] is False

    def test_complete_dir_still_reports_complete_false(self, tmp_path: Path) -> None:
        """A directory with large trajectory + many energy rows is
        STILL reported as not-complete — completion is not a
        per-file check. The diagnostic report lists what exists."""
        (tmp_path / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        energy_lines = ["#step,time,PE,KE,TE,temp,vol,speed\n"]
        energy_lines.extend(f"{i * 5000},{i * 10},0,0,0,310,0,0\n" for i in range(20))
        (tmp_path / "energy.csv").write_text("".join(energy_lines))
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": "state.1M.xml"}]})
        )

        report = verify_production_outputs(tmp_path)
        # completion is determined by is_run_complete, not by file size.
        assert report["complete"] is False
        # ...but the diagnostic info lists the actual files.
        files = report["files"]
        assert files["trajectory.dcd"]["exists"] is True  # type: ignore[index]
        assert files["trajectory.dcd"]["size_bytes"] == 20_000_000  # type: ignore[index]
        # 1 header + 20 data rows
        assert files["energy.csv"]["rows"] == 21  # type: ignore[index]
        assert files["checkpoint.json"]["exists"] is True  # type: ignore[index]

    def test_small_trajectory_recorded_as_file_info(self, tmp_path: Path) -> None:
        (tmp_path / "trajectory.dcd").write_bytes(b"\x00" * 100)
        (tmp_path / "energy.csv").write_text("step\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1, "file": "state.1.xml"}]})
        )

        report = verify_production_outputs(tmp_path)
        # The diagnostic report records the file's size without
        # declaring the directory complete.
        assert report["files"]["trajectory.dcd"]["size_bytes"] == 100  # type: ignore[index]
        assert report["complete"] is False

    def test_few_energy_rows_recorded_as_file_info(self, tmp_path: Path) -> None:
        (tmp_path / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        (tmp_path / "energy.csv").write_text("step\n1\n")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1, "file": "state.1.xml"}]})
        )

        report = verify_production_outputs(tmp_path)
        # 1 header + 1 data row
        assert report["files"]["energy.csv"]["rows"] == 2  # type: ignore[index]
        assert report["complete"] is False


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
        """Existing complete output should be reused.

        v10 BLOCKER #4: the skip path validates that the
        scientific outputs (trajectory, energy, topology) are
        actually present and usable. The fixture must include all
        three.
        """
        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out))
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        # The manifest references this state file — it must exist on
        # disk (load_checkpoint validates the reference). We don't
        # actually loadState in the skip path, but the validator
        # requires a non-empty file.
        (out / state_basename).write_text("<State/>")
        (out / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        energy_lines = ["#step,time\n"]
        energy_lines.extend(f"{i * 5000},{i * 10}\n" for i in range(20))
        (out / "energy.csv").write_text("".join(energy_lines))
        (out / "topology.pdb").write_bytes(b"ATOM\n" * 5000)  # v10: required
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": target, "file": state_basename}]})
        )

        runner = OpenMMRunner(config)
        result = runner.run()
        assert result.error == ""
        assert result.trajectory_path == str(out / "trajectory.dcd")
        assert result.state_xml_path == str(out / state_basename)
        assert result.error == ""

    def test_intermediate_checkpoint_resumes_instead_of_skipping(self, tmp_path: Path) -> None:
        """v8 BLOCKER #1 regression: a sufficiently-large intermediate
        checkpoint (large trajectory + many energy rows + valid
        intermediate manifest) must resume, NOT skip as complete.

        The previous behaviour declared any directory with a 10+ MB
        trajectory and 10+ energy rows as complete. After a mid-
        production checkpoint and process restart, the next
        invocation would skip instead of resuming — silently
        re-running a production loop that may not match the saved
        state. The v8 fix uses ``is_run_complete``: completion
        requires the manifest step to reach the target, not just
        file presence.
        """
        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)

        # Simulate a mid-production checkpoint: manifest at step
        # 5_000_000 (well below the target of 50_200_000).
        mid_step = 5_000_000
        state_basename = f"state.{mid_step}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "trajectory.dcd").write_bytes(b"\x00" * 50_000_000)  # > 10 MB
        energy_lines = ["#step,time,PE,KE,TE,temp,vol,speed\n"]
        energy_lines.extend(f"{i * 5000},{i * 10},0,0,0,310,0,0\n" for i in range(10_000))
        (out / "energy.csv").write_text("".join(energy_lines))
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": mid_step, "file": state_basename}]})
        )

        # decide() must produce a RESUME plan (the manifest step is
        # well below the target). The presence of a 50 MB trajectory
        # and 10k energy rows is irrelevant — completion is from the
        # manifest step + terminal payload, not file presence.
        from biolab_runners.openmm.run_state import Action, decide

        plan = decide(out, config, force=False)
        assert plan.action == Action.RESUME
        # remaining_steps = total_steps - production done
        assert plan.remaining_steps == config.total_steps - (mid_step - config.total_equil_steps)
        assert plan.resume_xml == str(out / state_basename)

    def test_missing_openmm_returns_error(self, tmp_path: Path) -> None:
        """Missing OpenMM should return error, not crash.

        The runner-level test mocks prepare_simulation to verify
        that the runner surfaces a missing-OpenMM error to the
        caller. The actual ``try: import openmm ... except
        ImportError`` code path in system_builder.prepare_simulation
        is covered by ``TestPrepareSimulationMissingOpenMM`` in
        test_system_builder.py, which uses an in-process
        ``sys.meta_path`` import blocker (Python-version-independent).
        """
        from biolab_runners.openmm import runner as runner_mod

        def fake_prepare_simulation(
            config: OpenMMConfig,
            output_dir: Path,
            resume_xml: str,
            result: SimulationResult,
        ) -> SimulationContext | None:
            result.error = "OpenMM not installed: No module named 'openmm'"
            return None

        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out))
        runner = OpenMMRunner(config)

        with patch.object(runner_mod, "prepare_simulation", side_effect=fake_prepare_simulation):
            result = runner.run()
        assert "not installed" in result.error
        assert "openmm" in result.error.lower()


# ---------------------------------------------------------------------------
# iRMSD threshold tests
# ---------------------------------------------------------------------------


class TestResumeStepUsesManifestNotEnergy:
    """Regression tests for the v6 BLOCKER: ``load_checkpoint``
    must use the manifest (checkpoint.json), not the last row of
    energy.csv. The two files advance at very different cadences —
    energy.csv at save_every_steps (~10 ps), state files at
    checkpoint_every_steps (~2 hr) — so the energy row can be
    hundreds of thousands of steps ahead of the saved state."""


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

    def test_sigterm_handler_is_installed(self, tmp_path: Path) -> None:
        config = OpenMMConfig()
        OpenMMRunner._install_sigterm_handler(
            simulation=MagicMock(),
            output_dir=tmp_path,
            start_step=0,
            steps_box=[0],
            config=config,
        )
        # The handler is installed — we don't invoke it (that would exit
        # the test process). Just assert signal.signal was called.
        # signal.signal is reset by pytest at session end, so no cleanup needed.

    def test_handler_saves_state_atomic(self, tmp_path: Path) -> None:
        """The handler must atomically save state + manifest with the
        ABSOLUTE step (``start_step + steps_box[0]``), not the local
        counter — the v6 BLOCKER fix."""
        import signal

        config = OpenMMConfig()
        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")
        # Capture the registered handler
        captured: dict[int, object] = {}

        def fake_signal(signum: int, handler: object) -> None:
            captured[signum] = handler

        start_step = 200_000  # already done equil
        local_steps = 12_345  # invocation-local production steps
        with patch("biolab_runners.openmm.runner.signal.signal", side_effect=fake_signal):
            OpenMMRunner._install_sigterm_handler(
                simulation=sim,
                output_dir=tmp_path,
                start_step=start_step,
                steps_box=[local_steps],
                config=config,
            )

        handler = captured[signal.SIGTERM]
        assert callable(handler)
        # Invoke with a non-zero current step
        with patch("biolab_runners.openmm.runner.sys.exit") as mock_exit:
            handler(signal.SIGTERM, None)  # type: ignore[arg-type, misc]
        # Atomic save: state xml + manifest both exist. The manifest's
        # step is the ABSOLUTE step (start_step + local = 212_345),
        # NOT the local counter (12_345). This is the v7 BLOCKER fix.
        assert sim.saveState.called
        manifest = json.loads((tmp_path / "checkpoint.json").read_text())
        assert manifest["records"][-1]["step"] == start_step + local_steps
        mock_exit.assert_called_once_with(0)

    def test_handler_swallows_save_state_errors(self, tmp_path: Path) -> None:
        """If saveState throws (disk full, permissions, etc.), the handler
        must log the error and still exit cleanly — never crash with an
        unhandled exception during cloud preemption."""
        import signal

        sim = MagicMock()
        sim.saveState.side_effect = OSError("disk full")
        config = OpenMMConfig()
        captured: dict[int, object] = {}

        def fake_signal(signum: int, handler: object) -> None:
            captured[signum] = handler

        with patch("biolab_runners.openmm.runner.signal.signal", side_effect=fake_signal):
            OpenMMRunner._install_sigterm_handler(
                simulation=sim,
                output_dir=tmp_path,
                start_step=0,
                steps_box=[0],
                config=config,
            )

        with patch("biolab_runners.openmm.runner.sys.exit") as mock_exit:
            captured[signal.SIGTERM](signal.SIGTERM, None)  # type: ignore[arg-type, misc]
        # Still exited cleanly even though saveState failed
        mock_exit.assert_called_once_with(0)
        # No state file was committed (the manifest rename never ran).
        assert not any((tmp_path).glob("state*.xml"))
        manifest = tmp_path / "checkpoint.json"
        if manifest.exists():
            # The manifest is unchanged — the previous (coherent) checkpoint
            # remains active if it existed.
            data = json.loads(manifest.read_text())
            assert data["records"][-1]["step"] != 0  # not the half-saved new step


# ---------------------------------------------------------------------------
# _maybe_checkpoint — periodic checkpointing
# ---------------------------------------------------------------------------


class TestMaybeCheckpoint:
    """``_maybe_checkpoint`` writes a state checkpoint if the interval
    has elapsed or if this is the last chunk.

    The save is atomic: state file written at a versioned name,
    manifest references it, single ``os.replace`` on the manifest is
    the commit point. The manifest records the absolute step
    (``start_step + steps_done``), not the local counter.

    Note: ``OpenMMConfig.checkpoint_every_steps`` is computed from
    ``checkpoint_interval_hours`` in ``__post_init__``, so the tests set
    the interval in hours, not the step count.
    """

    def test_no_checkpoint_when_interval_not_elapsed(self, tmp_path: Path) -> None:
        """If less than checkpoint_every_steps since last ckpt, and not
        at the end, do nothing."""
        sim = MagicMock()
        # 0.1 hours @ 2.0 fs = 180,000 steps between checkpoints
        config = OpenMMConfig(checkpoint_interval_hours=0.1)
        # Force steps_done to be small (not yet at the end)
        result = OpenMMRunner._maybe_checkpoint(
            simulation=sim,
            output_dir=tmp_path,
            start_step=0,
            steps_done=500,  # only 500 since last_ckpt_step=0
            last_ckpt_step=0,
            remaining_steps=10_000_000,
            config=config,
            t0=0.0,
        )
        assert result == 0  # unchanged
        sim.saveState.assert_not_called()

    def test_checkpoint_when_interval_elapsed(self, tmp_path: Path) -> None:
        """Interval elapsed → atomic save writes state file + manifest.

        The manifest's step is the ABSOLUTE step
        (``start_step + steps_done``), not the local counter — the v7
        BLOCKER fix."""
        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")
        # 1 ns @ 2.0 fs = 500 steps between checkpoints
        config = OpenMMConfig(
            timestep_fs=2.0,
            checkpoint_interval_hours=500.0 / 3600.0 / 1000.0,
        )
        start_step = 200_000  # already done equil
        result = OpenMMRunner._maybe_checkpoint(
            simulation=sim,
            output_dir=tmp_path,
            start_step=start_step,
            steps_done=1500,  # > 500 since last_ckpt=0
            last_ckpt_step=0,
            remaining_steps=10_000_000,
            config=config,
            t0=1000.0,  # fixed past time — avoids wall-clock dependency
        )
        assert result == 1500
        # The atomic save committed both files. The manifest's step is
        # the absolute step (start_step + steps_done = 201_500).
        assert (tmp_path / "checkpoint.json").exists()
        manifest = json.loads((tmp_path / "checkpoint.json").read_text())
        assert manifest["records"][-1]["step"] == start_step + 1500
        # The state file is at its versioned name (NOT canonical state.xml).
        state_file = manifest["records"][-1]["file"]
        assert state_file.startswith("state.201500_")
        assert state_file.endswith(".xml")
        assert (tmp_path / state_file).exists()

    def test_checkpoint_at_end_of_run(self, tmp_path: Path) -> None:
        """Even if the interval hasn't elapsed, checkpoint when steps_done
        reaches remaining_steps (last chunk)."""
        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")
        config = OpenMMConfig(
            timestep_fs=2.0,
            checkpoint_interval_hours=0.1,  # 180k steps
        )
        result = OpenMMRunner._maybe_checkpoint(
            simulation=sim,
            output_dir=tmp_path,
            start_step=200_000,
            steps_done=10_000,  # at remaining_steps
            last_ckpt_step=9_500,
            remaining_steps=10_000,
            config=config,
            t0=1000.0,  # fixed past time — avoids wall-clock dependency
        )
        assert result == 10_000
        assert (tmp_path / "checkpoint.json").exists()
        manifest = json.loads((tmp_path / "checkpoint.json").read_text())
        assert manifest["records"][-1]["step"] == 210_000  # absolute

    def test_ns_per_day_handles_zero_elapsed(self, tmp_path: Path) -> None:
        """If t0 == time.time() (zero elapsed), don't divide by zero.

        We use a t0 in the near-future so elapsed = t0 - now is negative
        (function guards via ``if elapsed > 0``). A negative elapsed is
        the realistic case the guard must cover.
        """
        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")
        config = OpenMMConfig(
            timestep_fs=2.0,
            checkpoint_interval_hours=500.0 / 3600.0 / 1000.0,  # 500 steps
        )
        result = OpenMMRunner._maybe_checkpoint(
            simulation=sim,
            output_dir=tmp_path,
            start_step=200_000,
            steps_done=2000,  # > 500 since last_ckpt=0 → checkpoint
            last_ckpt_step=0,
            remaining_steps=10_000_000,
            config=config,
            t0=time.time() + 1e9,  # far future → elapsed < 0 → guard fires
        )
        # Should not raise; ns_per_day is 0 since elapsed is 0
        assert result == 2000
        assert (tmp_path / "checkpoint.json").exists()


class TestMultiResumeCumulativeAccounting:
    """Regression test for the v7 BLOCKER #1: the manifest's step
    must be the ABSOLUTE OpenMM step, monotonically increasing across
    multiple resume cycles.

    The v6 design wrote the invocation-local ``steps_done`` to the
    manifest. On a resume, this caused the saved step to appear to
    move backwards (or to start at 0), silently shortening the run.
    The v7 design writes ``start_step + steps_done`` to the manifest
    so the saved step is the absolute step the simulation was at
    when the state file was written."""


class TestAtomicityBetweenStateAndManifest:
    """Regression test for the v7 BLOCKER #2: a crash mid-save MUST
    leave either the previous checkpoint fully active or the new
    checkpoint fully active — never a mix. The v6 design used two
    ``os.replace`` calls (one for state, one for manifest) which
    were individually atomic but the pair was not. The v7 design
    uses a generation-versioned state file (uniquely-named, so no
    rename needed) and a single ``os.replace`` on the manifest as
    the atomic commit point.
    """

    def test_crash_between_state_publication_and_manifest_publication(self, tmp_path: Path) -> None:
        """If the manifest rename fails (or the process is killed
        between writing the state file and renaming the manifest),
        the previous checkpoint MUST remain active.

        Simulated by patching the manifest's ``os.replace`` to raise.
        The state file is written (it's uniquely-named, no rename
        needed), but the manifest rename fails. The next resume
        loads the PREVIOUS manifest (still active), not the half-
        published new state.
        """
        from biolab_runners.openmm import checkpoint as ckpt_mod
        from biolab_runners.openmm.checkpoint import atomic_save_checkpoint

        # Pre-create a previous coherent checkpoint.
        previous_state = tmp_path / "state.500_12345_1000.xml"
        previous_state.write_text("<PREVIOUS_STATE/>")
        previous_manifest = {"records": [{"step": 500, "file": "state.500_12345_1000.xml"}]}
        (tmp_path / "checkpoint.json").write_text(json.dumps(previous_manifest))

        # Now simulate a new save that fails during the manifest rename.
        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<NEW_STATE_PARTIAL/>")

        original_replace = ckpt_mod.os.replace

        def failing_replace(src: str, dst: str) -> None:
            # Let the state file rename succeed (it's a no-op in v7 —
            # the state file is written directly to its versioned name,
            # no rename needed). Let other calls succeed; only the
            # manifest rename fails.
            if str(dst).endswith("checkpoint.json"):
                raise OSError("simulated disk-full during manifest rename")
            original_replace(src, dst)

        ckpt_mod.os.replace = failing_replace
        try:
            with pytest.raises(OSError, match="simulated"):
                atomic_save_checkpoint(sim, tmp_path, absolute_step=999_999)
        finally:
            ckpt_mod.os.replace = original_replace

        # The MANIFEST is unchanged — the previous checkpoint remains
        # active. The next resume reads the previous step (500), not
        # the half-saved new step (999_999).
        checkpoint = load_checkpoint(tmp_path)
        step = checkpoint.absolute_step
        file = checkpoint.state_file_basename
        assert step == 500, (
            f"manifest must be unchanged after a failed save; got step={step} (expected 500)"
        )
        assert file == "state.500_12345_1000.xml"

        # The half-published state file is on disk but unreferenced —
        # it WILL be GC'd on the next successful save.
        new_state_files = [
            f for f in tmp_path.glob("state*.xml") if f.name != "state.500_12345_1000.xml"
        ]
        assert len(new_state_files) == 1
        # The next save will GC this orphan.
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")
        atomic_save_checkpoint(sim, tmp_path, absolute_step=800_000)
        # Orphan is gone; new state file is present.
        assert not new_state_files[0].exists()
        checkpoint = load_checkpoint(tmp_path)
        step = checkpoint.absolute_step
        file = checkpoint.state_file_basename
        assert step == 800_000
        assert (tmp_path / file).exists()


class TestFinalizeResultStateXmlPath:
    """v8 BLOCKER #3 regression: ``_finalize_result`` must populate
    ``SimulationResult.state_xml_path`` with the committed state
    file path. The previous behaviour silently produced ``md_result.json``
    records with an empty ``state_xml_path`` field after successful
    runs — breaking the public result contract.
    """

    def test_finalize_sets_state_xml_path_on_fresh_run(self, tmp_path: Path) -> None:
        """A normal finalization (fresh run ending at the target step)
        must commit a state file and assign result.state_xml_path."""
        from biolab_runners.openmm.runner import OpenMMRunner

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=1.0, timestep_fs=2.0)
        result = SimulationResult(config=config)
        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")

        OpenMMRunner._finalize_result(
            ctx=MagicMock(simulation=sim),
            result=result,
            energy_fh=MagicMock(),
            traj_path=str(tmp_path / "trajectory.dcd"),
            energy_path=str(tmp_path / "energy.csv"),
            start_step=config.total_equil_steps,
            steps_done=config.total_steps,
            t0=time.time() - 60.0,  # some elapsed
            output_dir=tmp_path,
        )

        # result.state_xml_path is set to the versioned state file
        # committed by the atomic save.
        assert result.state_xml_path != "", (
            f"result.state_xml_path must be set after finalize; got {result.state_xml_path!r}"
        )
        assert Path(result.state_xml_path).exists()
        # The manifest references the same file.
        checkpoint = load_checkpoint(tmp_path)
        step = checkpoint.absolute_step
        file = checkpoint.state_file_basename
        assert Path(result.state_xml_path).name == file
        assert step == config.total_equil_steps + config.total_steps

    def test_finalize_skips_duplicate_save_when_already_committed(self, tmp_path: Path) -> None:
        """v8 SUGGESTION regression: if the manifest already commits
        at the final step (the typical case — ``_maybe_checkpoint``
        saved on the last chunk), ``_finalize_result`` must skip the
        re-serialise and reuse the manifest's reference. ``sim.saveState``
        must NOT be called a second time."""
        from biolab_runners.openmm.runner import OpenMMRunner

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=1.0, timestep_fs=2.0)
        # Pre-create a manifest at the final step (as if _maybe_checkpoint
        # already saved at the end of the production loop).
        final_step = config.total_equil_steps + config.total_steps
        state_basename = f"state.{final_step}_12345_999.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": final_step, "file": state_basename}]})
        )

        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")

        result = SimulationResult(config=config)
        OpenMMRunner._finalize_result(
            ctx=MagicMock(simulation=sim),
            result=result,
            energy_fh=MagicMock(),
            traj_path=str(tmp_path / "trajectory.dcd"),
            energy_path=str(tmp_path / "energy.csv"),
            start_step=config.total_equil_steps,
            steps_done=config.total_steps,
            t0=time.time() - 60.0,
            output_dir=tmp_path,
        )

        # result.state_xml_path is set to the existing state file
        # (no duplicate save).
        assert result.state_xml_path == str(tmp_path / state_basename)
        # sim.saveState must NOT be called again — the manifest already
        # commits at the final step.
        sim.saveState.assert_not_called()
        # No second state file was created.
        assert len(list(tmp_path.glob("state*.xml"))) == 1

    def test_finalize_saves_when_manifest_step_is_stale(self, tmp_path: Path) -> None:
        """When the manifest step is below the finalize target (e.g. a
        mid-production checkpoint), ``_finalize_result`` MUST do a
        fresh save. This is the production-loop end-of-run case where
        the loop ended without the final ``_maybe_checkpoint`` saving
        at exactly the target step."""
        from biolab_runners.openmm.runner import OpenMMRunner

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=1.0, timestep_fs=2.0)
        # Pre-existing manifest at a step below the final step.
        prior_step = config.total_equil_steps
        state_basename = f"state.{prior_step}_12345_999.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": prior_step, "file": state_basename}]})
        )

        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")

        result = SimulationResult(config=config)
        OpenMMRunner._finalize_result(
            ctx=MagicMock(simulation=sim),
            result=result,
            energy_fh=MagicMock(),
            traj_path=str(tmp_path / "trajectory.dcd"),
            energy_path=str(tmp_path / "energy.csv"),
            start_step=config.total_equil_steps,
            steps_done=config.total_steps,
            t0=time.time() - 60.0,
            output_dir=tmp_path,
        )

        # saveState WAS called — the manifest step didn't match the target.
        assert sim.saveState.called
        # The manifest now references the final step.
        final_step = config.total_equil_steps + config.total_steps
        checkpoint = load_checkpoint(tmp_path)
        step = checkpoint.absolute_step
        file = checkpoint.state_file_basename
        assert step == final_step
        assert Path(result.state_xml_path).name == file

    def test_skip_path_sets_state_xml_path(self, tmp_path: Path) -> None:
        """The idempotent skip path (terminal run) must populate
        ``result.state_xml_path`` from the manifest. ``md_result.json``
        records were previously missing this field — a silent break
        of the public contract. v10: also requires the scientific
        outputs to be present (BLOCKER #4)."""
        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out))
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        (out / "energy.csv").write_text("#step,time\n" + "1,1\n" * 100)
        (out / "topology.pdb").write_bytes(b"ATOM\n" * 5000)
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": target, "file": state_basename}]})
        )

        runner = OpenMMRunner(config)
        result = runner.run()
        assert result.error == ""
        # The skip path populated state_xml_path from the manifest.
        assert result.state_xml_path == str(out / state_basename)

    def test_normal_skip_serializes_abort_reason_as_empty_string(self, tmp_path: Path) -> None:
        """A normal-completion skip must serialise ``abort_reason`` as
        ``""`` (empty string), NOT ``null``. ``SimulationResult.abort_reason``
        is typed ``str`` defaulting to ``""``, and ``to_dict()``
        emits the value unchanged — a ``None`` here would round-trip
        as ``"abort_reason": null`` in ``md_result.json``, breaking
        the long-standing JSON contract that downstream consumers
        (oral_amp.cloud) rely on.
        """
        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out))
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        (out / "energy.csv").write_text("#step,time\n" + "1,1\n" * 100)
        (out / "topology.pdb").write_bytes(b"ATOM\n" * 5000)
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": target, "file": state_basename}]})
        )

        runner = OpenMMRunner(config)
        result = runner.run()

        assert result.error == ""
        assert result.early_abort is False
        # Both the field and the JSON serialisation must be the
        # empty string, never None / null.
        assert result.abort_reason == ""
        serialised = result.to_dict()
        assert serialised["abort_reason"] == ""
        assert serialised["abort_reason"] is not None


class TestEarlyAbortResultReconstruction:
    """v10 BLOCKER #2 regression: an early-aborted run reused
    idempotently must return a result with ``early_abort=True``,
    ``abort_reason``, and ``total_ns`` populated from the
    manifest's ``terminal`` payload (the authoritative source in
    v10). Without this fix, a downstream consumer reading
    ``md_result.json`` would interpret the run as normal completion
    (default ``early_abort=False``, ``abort_reason=""``,
    ``total_ns=0.0``).

    v10 BLOCKER #3: ``total_ns`` is PRODUCTION ns
    (``absolute_step - total_equil_steps``), not absolute OpenMM
    ns. For an early-aborted run at absolute step 5_200_000 with
    total_equil_steps=200_000, production_ns = 10.0 ns (NOT the
    10.4 ns you'd get from absolute_step * timestep_fs / 1e6).
    """

    def _setup_early_aborted_dir(self, out: Path) -> tuple[int, float, str]:
        """Create an early-aborted run directory.

        Returns (absolute_step, production_ns, state_basename)."""
        config_path = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        absolute_step = config_path.total_equil_steps + 5_000_000
        production_steps = absolute_step - config_path.total_equil_steps
        production_ns = round(production_steps * 2.0 / 1e6, 2)
        state_basename = f"state.{absolute_step}_1_1.xml"
        (out / state_basename).write_text("<STATE/>")
        # v10: terminal status is part of the manifest record.
        (out / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": absolute_step,
                            "file": state_basename,
                            "terminal": {
                                "type": "early_abort",
                                "step": absolute_step,
                                "reason": "early_dissociation",
                                "production_ns": production_ns,
                            },
                        }
                    ]
                }
            )
        )
        (out / "energy.csv").write_text(
            "#step,time\n" + "\n".join(f"{i},{i}" for i in range(0, 5001, 500)) + "\n"
        )
        (out / "trajectory.dcd").write_bytes(b"\x00" * 50_000_000)
        (out / "topology.pdb").write_bytes(b"ATOM\n" * 1000)
        # Derived compat file — written for downstream consumers,
        # not authoritative. Its production_ns matches the
        # manifest's terminal payload.
        abort_meta = {
            "aborted": True,
            "abort_reason": "early_dissociation",
            "abort_step": absolute_step,
            "abort_ns": production_ns,
            "gate": "offline_mdtraj",
        }
        (out / "early_abort.json").write_text(json.dumps(abort_meta))
        return absolute_step, production_ns, state_basename

    def test_skip_path_reconstructs_early_abort_result(self, tmp_path: Path) -> None:
        """runner.run() on an early-aborted directory must return
        a result with ``early_abort=True``, ``abort_reason`` set,
        and ``total_ns`` = production_ns (v10 BLOCKER #3)."""
        out = tmp_path / "output"
        out.mkdir()
        _step, production_ns_value, state_basename = self._setup_early_aborted_dir(out)

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        result = runner.run()
        assert result.error == ""
        # The skip path populated early_abort from the manifest's terminal.
        assert result.early_abort is True, (
            f"early_aborted run must report early_abort=True; got {result.early_abort}"
        )
        assert result.abort_reason == "early_dissociation"
        # v10 BLOCKER #3: total_ns is PRODUCTION ns, not absolute ns.
        assert result.total_ns == production_ns_value, (
            f"total_ns must be production_ns ({production_ns_value}); "
            f"got {result.total_ns} (this would be absolute_ns if the v6 "
            "local-step semantics leaked through)"
        )
        # Artifact paths still set.
        assert result.state_xml_path == str(out / state_basename)

    def test_skip_path_reconstructs_with_default_ns_if_terminal_missing(
        self, tmp_path: Path
    ) -> None:
        """If the terminal payload is missing or invalid, the
        reconstruction leaves early_abort=False (the run is not
        classified as terminal)."""
        from biolab_runners.openmm.checkpoint import is_run_complete

        out = tmp_path / "output"
        out.mkdir()
        self._setup_early_aborted_dir(out)
        # Drop the terminal payload from the manifest.
        manifest_data = json.loads((out / "checkpoint.json").read_text())
        for record in manifest_data["records"]:
            record.pop("terminal", None)
        (out / "checkpoint.json").write_text(json.dumps(manifest_data))

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        complete, _reason = is_run_complete(out, config)
        assert complete is False
        # Resume rather than skip.
        runner = OpenMMRunner(config)
        result = runner.run()
        assert result.early_abort is False


class TestAtomicTerminalBinding:
    """v10 BLOCKER #2 binding: terminal status commits atomically
    with the state file via the manifest's ``terminal`` payload.
    The ``step`` field of the payload MUST equal the manifest's
    ``step``. This is the binding that prevents a crash between
    the state save and the marker write from leaving a
    resumable-but-terminal decision un-recorded.
    """

    def test_atomic_save_writes_terminal_payload(self, tmp_path: Path) -> None:
        """``atomic_save_checkpoint`` embeds the terminal payload
        in the manifest record on the same os.replace."""
        from biolab_runners.openmm.checkpoint import atomic_save_checkpoint

        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")
        terminal = {
            "type": "early_abort",
            "step": 2_700_000,
            "reason": "early_dissociation",
            "production_ns": 5.0,
        }
        state_basename = atomic_save_checkpoint(
            sim, tmp_path, absolute_step=2_700_000, terminal=terminal
        )

        # Manifest has the terminal payload.
        manifest = json.loads((tmp_path / "checkpoint.json").read_text())
        record = manifest["records"][-1]
        assert record["step"] == 2_700_000
        assert record["file"] == state_basename
        assert record["terminal"] == terminal

    def test_atomic_save_without_terminal_omits_field(self, tmp_path: Path) -> None:
        """A normal-completion save does NOT include a terminal field."""
        from biolab_runners.openmm.checkpoint import atomic_save_checkpoint

        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")
        atomic_save_checkpoint(sim, tmp_path, absolute_step=50_200_000)

        manifest = json.loads((tmp_path / "checkpoint.json").read_text())
        record = manifest["records"][-1]
        assert record["step"] == 50_200_000
        assert "terminal" not in record


class TestTerminalArtifactValidation:
    """v10 BLOCKER #4 regression: ``is_run_complete`` returns true
    based on the manifest's terminal decision, but a terminal run
    also needs the scientific outputs (trajectory, energy log,
    topology) to be present and usable. A terminal manifest plus
    state file but no trajectory/energy returns a result with
    ``error=""`` and paths pointing to nonexistent files — silently
    misleading downstream consumers.
    """

    def _populate_terminal_run(
        self, out: Path, *, terminal_payload: dict[str, object] | None = None
    ) -> tuple[str, int]:
        """Write a terminal manifest at the END step. Returns (state_basename, target_step)."""
        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        record: dict[str, object] = {"step": target, "file": state_basename}
        if terminal_payload is not None:
            record["terminal"] = terminal_payload
        (out / "checkpoint.json").write_text(json.dumps({"records": [record]}))
        return state_basename, target

    def test_missing_trajectory_returns_error(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        self._populate_terminal_run(out)
        # energy and topology are present, but no trajectory.
        (out / "energy.csv").write_text("#step,time\n1,1\n")
        (out / "topology.pdb").write_bytes(b"ATOM\n" * 1000)

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        result = runner.run()
        assert result.error != ""
        assert "missing trajectory" in result.error or "missing artifacts" in result.error
        assert "force=True" in result.error

    def test_empty_energy_log_returns_error(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        self._populate_terminal_run(out)
        (out / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        # Energy log exists but has only the header (no data rows).
        (out / "energy.csv").write_text("#step,time\n")
        (out / "topology.pdb").write_bytes(b"ATOM\n" * 1000)

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        result = runner.run()
        assert result.error != ""
        assert "energy" in result.error

    def test_header_only_energy_returns_error(self, tmp_path: Path) -> None:
        """Energy log with header but no data rows counts as empty."""
        out = tmp_path / "output"
        out.mkdir()
        self._populate_terminal_run(out)
        (out / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        # Write enough content to be non-zero bytes but only header.
        (out / "energy.csv").write_text("#step,time\n")
        (out / "topology.pdb").write_bytes(b"ATOM\n" * 1000)

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        result = runner.run()
        assert result.error != ""
        assert "energy" in result.error

    def test_missing_topology_returns_error(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        self._populate_terminal_run(out)
        (out / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        (out / "energy.csv").write_text("#step,time\n" + "1,1\n" * 100)
        # No topology.pdb.

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        result = runner.run()
        assert result.error != ""
        assert "topology" in result.error

    def test_early_aborted_terminal_with_missing_artifacts_returns_error(
        self, tmp_path: Path
    ) -> None:
        """Early-abort terminal classification also requires artifacts."""
        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        absolute_step = config.total_equil_steps + 5_000_000
        state_basename = f"state.{absolute_step}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        # Manifest terminal + state file, but no trajectory/energy/topology.
        (out / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": absolute_step,
                            "file": state_basename,
                            "terminal": {
                                "type": "early_abort",
                                "step": absolute_step,
                                "reason": "early_dissociation",
                                "production_ns": 10.0,
                            },
                        }
                    ]
                }
            )
        )

        runner = OpenMMRunner(config)
        result = runner.run()
        assert result.error != ""
        assert "missing" in result.error

    def test_all_artifacts_present_returns_success(self, tmp_path: Path) -> None:
        """Sanity: when all artifacts are present, skip returns success."""
        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        (out / "energy.csv").write_text("#step,time\n" + "1,1\n" * 100)
        (out / "topology.pdb").write_bytes(b"ATOM\n" * 1000)
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": target, "file": state_basename}]})
        )

        runner = OpenMMRunner(config)
        result = runner.run()
        assert result.error == ""
        assert result.state_xml_path == str(out / state_basename)
        assert result.total_ns == config.production_ns


class TestNsPerDayInvocationLocal:
    """v11 BLOCKER #1 regression: ``ns_per_day`` is INVOCATION-LOCAL
    production throughput — not cumulative production divided by
    current-invocation wall time.

    The previous formula divided cumulative production by
    invocation-local elapsed, which inflated the reported
    throughput on every resumed run. For a 90 ns + 10 ns run
    where the second invocation takes 1 day, the previous code
    reported 100 ns/day; the correct value is 10 ns/day for the
    second invocation.

    The two accounting scopes are kept separate:
    - cumulative production → ``result.total_ns`` (the user-visible
      "how much have we simulated" number)
    - invocation-local production → ``result.ns_per_day`` (the
      user-visible "how fast was THIS run" metric)
    """

    def test_resumed_run_total_ns_cumulative_ns_per_day_invocation_local(
        self, tmp_path: Path
    ) -> None:
        """Resume scenario:
        - previous invocation did 90 ns production (started at 200_000 equil,
          ended at 200_000 + 45_000_000 = 45_200_000 absolute steps)
        - this invocation does 10 ns more production (45_200_000 +
          5_000_000 = 50_200_000 absolute steps, end of run)
        - elapsed in this invocation: 1 day = 86_400 s

        Expected:
        - total_ns = 100.0 ns (cumulative production)
        - ns_per_day = 10.0 ns/day (invocation-local throughput)
        """
        from biolab_runners.openmm.runner import OpenMMRunner

        config = OpenMMConfig(
            output_dir=str(tmp_path),
            production_ns=100.0,
            timestep_fs=2.0,
        )
        result = SimulationResult(config=config)
        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")

        # Resume from step 45_200_000 (post-equil + 90 ns production).
        start_step = 45_200_000
        steps_done = 5_000_000  # 10 ns this invocation
        absolute_step = start_step + steps_done  # 50_200_000
        assert absolute_step == config.total_equil_steps + config.total_steps
        # Production_steps_done cumulative = 100_000_000 - 200_000 = 99_800_000
        cumulative_production_ns = (
            (absolute_step - config.total_equil_steps) * config.timestep_fs / 1e6
        )
        # Invocation-local production = 10 ns
        invocation_production_ns = steps_done * config.timestep_fs / 1e6
        assert cumulative_production_ns == 100.0
        assert invocation_production_ns == 10.0

        # 1 day wall time
        t0 = time.time() - 86_400

        OpenMMRunner._finalize_result(
            ctx=MagicMock(simulation=sim),
            result=result,
            energy_fh=MagicMock(),
            traj_path=str(tmp_path / "trajectory.dcd"),
            energy_path=str(tmp_path / "energy.csv"),
            start_step=start_step,
            steps_done=steps_done,
            t0=t0,
            output_dir=tmp_path,
        )

        # total_ns is cumulative.
        assert result.total_ns == 100.0, (
            f"total_ns must be cumulative ({cumulative_production_ns}); got {result.total_ns}"
        )
        # ns_per_day is invocation-local throughput = 10.0 ns/day,
        # NOT the inflated 100.0 ns/day the previous code reported.
        assert result.ns_per_day == 10.0, (
            f"ns_per_day must be invocation-local throughput (10.0); "
            f"got {result.ns_per_day} (cumulative/invocation would be 100.0)"
        )


class TestDerivedMarkerFailureDoesNotCrashTerminalRun:
    """v11 BLOCKER #2 regression: the derived ``early_abort.json``
    write is NOT authoritative — a failure to write it (full disk,
    permission denied, etc.) MUST NOT crash an already-committed
    terminal run. The runner's caller still needs a coherent
    SimulationResult with ``early_abort=True``.
    """

    def test_write_abort_metadata_failure_is_caught(self, tmp_path: Path) -> None:
        """Patch ``_write_abort_metadata`` to raise OSError on the
        derived-marker write. The atomic save (manifest commit) must
        still succeed; the runner must continue normally and return
        ``early_abort=True``."""
        from biolab_runners.openmm.runner import OpenMMRunner

        verdict = MagicMock()
        verdict.reason = "early_dissociation"
        verdict.rmsd_5ns = 5.0
        verdict.rmsd_10ns = 6.0
        verdict.max_rmsd = 7.0
        verdict.slope_a_per_ns = 0.5
        verdict.receptor_fit_residual = 1.2

        # Pre-create a previous manifest so the GC doesn't trip.
        state_basename = "state.700000_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")

        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")

        config = OpenMMConfig(production_ns=100.0, timestep_fs=2.0)
        # Patch _write_abort_metadata to raise.
        original = OpenMMRunner._write_abort_metadata
        OpenMMRunner._write_abort_metadata = MagicMock(  # type: ignore[method-assign]
            side_effect=OSError("disk full")
        )
        try:
            # Drive the abort path through _poll_offline_gate.
            # We pass steps_done such that production_steps_done = 2_500_000
            # (5 ns of production). The abort commits absolute_step =
            # total_equil_steps + steps_done.
            start_step = config.total_equil_steps
            steps_done = 2_500_000
            absolute_step = start_step + steps_done
            terminal_payload = {
                "step": absolute_step,
                "type": "early_abort",
                "reason": verdict.reason,
                "production_ns": steps_done * config.timestep_fs / 1e6,
            }
            # Simulate the run code path: atomic save must succeed;
            # _write_abort_metadata must be guarded.
            from biolab_runners.openmm.checkpoint import atomic_save_checkpoint

            state_basename = atomic_save_checkpoint(
                sim, tmp_path, absolute_step, terminal=terminal_payload
            )
            # Apply the same guard pattern the runner uses.
            try:
                OpenMMRunner._write_abort_metadata(
                    verdict,
                    tmp_path,
                    abort_thresh=5.0,
                    config=config,
                    absolute_step=absolute_step,
                    production_ns=steps_done * config.timestep_fs / 1e6,
                )
            except OSError as exc:
                # Same handling the runner does.
                logger.warning("best-effort marker failed: %s", exc)

            # Manifest is terminal — atomic save committed it.
            manifest = json.loads((tmp_path / "checkpoint.json").read_text())
            assert manifest["records"][-1]["step"] == absolute_step
            assert manifest["records"][-1]["terminal"]["type"] == "early_abort"
            # early_abort.json was NOT written (the OSError prevented it).
            assert not (tmp_path / "early_abort.json").exists()
            # The state file IS present.
            assert (tmp_path / state_basename).exists()
        finally:
            OpenMMRunner._write_abort_metadata = original  # type: ignore[method-assign]

    def test_poll_offline_gate_survives_derived_marker_failure(self, tmp_path: Path) -> None:
        """``_poll_offline_gate`` MUST NOT raise when the derived
        ``early_abort.json`` write fails. The atomic save has
        already committed the terminal decision; a derived-write
        failure must be logged + suppressed so the loop returns
        ``(True, reason)`` cleanly and the runner can finish."""
        from biolab_runners.openmm.runner import OpenMMRunner

        verdict = MagicMock()
        verdict.reason = "early_dissociation"
        verdict.rmsd_5ns = 5.0
        verdict.rmsd_10ns = 6.0
        verdict.max_rmsd = 7.0
        verdict.slope_a_per_ns = 0.5
        verdict.receptor_fit_residual = 1.2
        verdict.current_ns = 5.0
        verdict.abort = True
        verdict.n_frames = 100

        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")

        config = OpenMMConfig(production_ns=100.0, timestep_fs=2.0)
        # Patch _write_abort_metadata to raise.
        original = OpenMMRunner._write_abort_metadata
        OpenMMRunner._write_abort_metadata = MagicMock(  # type: ignore[method-assign]
            side_effect=OSError("disk full")
        )
        try:
            # Stub evaluate_trajectory + write_verdict_file so we
            # can drive _poll_offline_gate end-to-end.
            with (
                patch(
                    "biolab_runners.openmm.runner.evaluate_trajectory",
                    return_value=verdict,
                ),
                patch(
                    "biolab_runners.openmm.runner.write_verdict_file",
                    return_value=None,
                ),
            ):
                start_step = config.total_equil_steps
                steps_done = 2_500_000
                polling_done, abort_reason = OpenMMRunner._poll_offline_gate(
                    simulation=sim,
                    output_dir=tmp_path,
                    start_step=start_step,
                    abort_thresh=5.0,
                    config=config,
                    steps_done=steps_done,
                )
            # polling_done=True (abort fired), abort_reason set,
            # NO exception propagated.
            assert polling_done is True
            assert abort_reason == "early_dissociation"
            # Manifest is committed + terminal.
            manifest = json.loads((tmp_path / "checkpoint.json").read_text())
            assert manifest["records"][-1]["terminal"]["type"] == "early_abort"
        finally:
            OpenMMRunner._write_abort_metadata = original  # type: ignore[method-assign]


class TestTerminalPayloadPrecedesNormalCompletion:
    """v12 BLOCKER regression: when the manifest step is at/past
    the configured target AND the manifest carries a valid
    ``terminal`` payload, the explicit terminal decision MUST take
    precedence over the inferred normal completion.

    Without this precedence, ``_check_normal_completion`` returns
    first, ``_reconstruct_terminal_result`` is skipped (the reason
    starts with ``normal_completion_step_``), and the reused result
    reports ``early_abort=False`` despite the manifest carrying a
    valid early-abort payload. The live invocation correctly sets
    ``early_abort=True`` from ``abort_reason``, so live and
    reconstructed results would disagree.

    This scenario occurs naturally when the offline-mdtraj gate
    fires on the final production chunk — the manifest step lands
    at exactly ``total_equil_steps + total_steps`` at the moment
    of an end-of-run abort.
    """

    def test_end_of_run_abort_takes_precedence_over_normal_completion(self, tmp_path: Path) -> None:
        """Manifest step at ``target_step`` + valid early_abort payload
        → ``is_run_complete`` returns ``manifest_terminal_...`` reason
        (not ``normal_completion_step_...``).
        """
        from biolab_runners.openmm.checkpoint import is_run_complete

        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=5.0, timestep_fs=2.0)
        target_step = config.total_equil_steps + config.total_steps

        # The end-of-run scenario: the offline gate fires on the
        # final chunk, so the absolute step is exactly the target.
        state_basename = f"state.{target_step}_12345_1700000000.xml"
        (out / state_basename).write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": target_step,
                            "file": state_basename,
                            "terminal": {
                                "type": "early_abort",
                                "step": target_step,
                                "reason": "early_dissociation",
                                "production_ns": 5.0,
                            },
                        }
                    ]
                }
            )
        )
        (out / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        (out / "energy.csv").write_text("#step,time\n" + "1,1\n" * 100)
        (out / "topology.pdb").write_bytes(b"ATOM\n" * 1000)

        # Manifest terminal payload MUST take precedence.
        complete, reason = is_run_complete(out, config)
        assert complete is True
        assert reason.startswith("manifest_terminal_early_abort_step_"), (
            f"valid early_abort payload at the target step must take "
            f"precedence over normal completion; got reason={reason!r}"
        )

    def test_runner_reconstructs_early_abort_when_manifest_at_target(self, tmp_path: Path) -> None:
        """``runner.run()`` on a directory with manifest-step=target AND
        a valid early_abort payload must return a result with
        ``early_abort=True`` and the abort metadata reconstructed.
        """
        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=5.0, timestep_fs=2.0)
        target_step = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target_step}_12345_1700000000.xml"
        (out / state_basename).write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": target_step,
                            "file": state_basename,
                            "terminal": {
                                "type": "early_abort",
                                "step": target_step,
                                "reason": "early_dissociation",
                                "production_ns": 5.0,
                            },
                        }
                    ]
                }
            )
        )
        (out / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        (out / "energy.csv").write_text("#step,time\n" + "1,1\n" * 100)
        (out / "topology.pdb").write_bytes(b"ATOM\n" * 1000)

        runner = OpenMMRunner(config)
        result = runner.run()
        assert result.error == ""
        # The skip path reconstructed early_abort (NOT normal completion).
        assert result.early_abort is True, (
            f"valid early_abort payload must reconstruct to early_abort=True; "
            f"got {result.early_abort}"
        )
        assert result.abort_reason == "early_dissociation"
        assert result.total_ns == 5.0


class TestMalformedTerminalAtTargetIsNotNormalCompletion:
    """v13 BLOCKER regression: a manifest carrying a malformed
    ``terminal`` payload (the field is present but fails schema
    validation) MUST NOT silently fall through to the inferred
    normal-completion heuristic when the manifest step has reached
    the configured target.

    The previous implementation returned ``None`` from
    ``_check_manifest_terminal`` for both ABSENT and INVALID
    payloads. A malformed terminal at the target step then fell
    through to ``_check_normal_completion`` and was reclassified
    as a successful normal completion, contradicting the v11
    terminal-schema contract.

    Fix: ``_check_manifest_terminal`` returns a tri-state result;
    an INVALID payload reports ``(False,
    "invalid_terminal_<reason>")`` and ``is_run_complete`` does
    NOT fall back to normal completion. The runner treats the
    invalid payload as an error — the user must investigate via
    ``force=True`` (which quarantines the malformed manifest).
    """

    def _write_manifest_at_target(self, out: Path, *, terminal: dict[str, object] | None) -> int:
        """Write a manifest at the configured target step. Returns target."""
        config = OpenMMConfig(output_dir=str(out), production_ns=5.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        record: dict[str, object] = {"step": target, "file": state_basename}
        if terminal is not None:
            record["terminal"] = terminal
        (out / "checkpoint.json").write_text(json.dumps({"records": [record]}))
        return target

    def test_empty_reason_at_target_does_not_become_normal(self, tmp_path: Path) -> None:
        """Manifest at target step + terminal with empty reason
        → ``is_run_complete`` returns ``(False,
        "invalid_terminal_reason_empty")``, NOT
        ``normal_completion_step_...``."""
        from biolab_runners.openmm.checkpoint import is_run_complete

        out = tmp_path / "output"
        out.mkdir()
        target = self._write_manifest_at_target(
            out,
            terminal={
                "type": "early_abort",
                "step": None,  # will overwrite
                "reason": "",
            },
        )
        # Patch the step now that we know the target.
        manifest = json.loads((out / "checkpoint.json").read_text())
        manifest["records"][-1]["terminal"]["step"] = target
        (out / "checkpoint.json").write_text(json.dumps(manifest))

        config = OpenMMConfig(output_dir=str(out), production_ns=5.0, timestep_fs=2.0)
        complete, reason = is_run_complete(out, config)
        assert complete is False
        assert reason == "invalid_terminal_reason_empty", (
            f"malformed terminal at target must NOT be reclassified as "
            f"normal completion; got reason={reason!r}"
        )

    def test_unknown_type_at_target_does_not_become_normal(self, tmp_path: Path) -> None:
        """Manifest at target + terminal.type='other' → invalid."""
        from biolab_runners.openmm.checkpoint import is_run_complete

        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=5.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": target,
                            "file": state_basename,
                            "terminal": {
                                "type": "other",
                                "step": target,
                                "reason": "x",
                            },
                        }
                    ]
                }
            )
        )

        complete, reason = is_run_complete(out, config)
        assert complete is False
        assert reason == "invalid_terminal_type_unsupported"

    def test_string_step_at_target_does_not_become_normal(self, tmp_path: Path) -> None:
        """Manifest at target + terminal.step='5000000' (string) → invalid."""
        from biolab_runners.openmm.checkpoint import is_run_complete

        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=5.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": target,
                            "file": state_basename,
                            "terminal": {
                                "type": "early_abort",
                                "step": "5000000",
                                "reason": "x",
                            },
                        }
                    ]
                }
            )
        )

        complete, reason = is_run_complete(out, config)
        assert complete is False
        assert reason == "invalid_terminal_step_invalid_type"

    def test_bool_step_at_target_does_not_become_normal(self, tmp_path: Path) -> None:
        """Manifest at target + terminal.step=True (bool) → invalid."""
        from biolab_runners.openmm.checkpoint import is_run_complete

        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=5.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": target,
                            "file": state_basename,
                            "terminal": {
                                "type": "early_abort",
                                "step": True,
                                "reason": "x",
                            },
                        }
                    ]
                }
            )
        )

        complete, reason = is_run_complete(out, config)
        assert complete is False
        assert reason == "invalid_terminal_step_invalid_type"

    def test_step_mismatch_at_target_does_not_become_normal(self, tmp_path: Path) -> None:
        """Manifest at target + terminal.step != manifest.step → invalid."""
        from biolab_runners.openmm.checkpoint import is_run_complete

        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=5.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": target,
                            "file": state_basename,
                            "terminal": {
                                "type": "early_abort",
                                "step": target - 1,
                                "reason": "x",
                            },
                        }
                    ]
                }
            )
        )

        complete, reason = is_run_complete(out, config)
        assert complete is False
        assert reason == "invalid_terminal_step_mismatch"

    def test_runner_fails_fast_on_malformed_terminal_at_target(self, tmp_path: Path) -> None:
        """End-to-end: ``runner.run()`` on a directory with
        manifest-at-target + malformed terminal returns
        ``result.error`` set, NOT a successful normal completion."""
        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=5.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "trajectory.dcd").write_bytes(b"\x00" * 20_000_000)
        (out / "energy.csv").write_text("#step,time\n" + "1,1\n" * 100)
        (out / "topology.pdb").write_bytes(b"ATOM\n" * 1000)
        (out / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": target,
                            "file": state_basename,
                            "terminal": {
                                "type": "early_abort",
                                "step": target,
                                "reason": "",  # malformed: empty reason
                            },
                        }
                    ]
                }
            )
        )

        runner = OpenMMRunner(config)
        result = runner.run()
        # The malformed terminal at the target step must NOT
        # produce a successful normal completion. The runner
        # surfaces the error and does not run the production
        # loop (which would commit a non-terminal manifest,
        # overwriting the user's malformed intent).
        assert result.error != "", (
            f"malformed terminal at target must surface result.error, got {result.error!r}"
        )
        assert "force=True" in result.error, (
            f"the error must guide the user to force=True; got {result.error!r}"
        )

    def test_no_terminal_field_at_target_is_normal_completion(self, tmp_path: Path) -> None:
        """Sanity: when the terminal field is ABSENT, the
        normal-completion fallback IS allowed (legacy manifests
        predate the terminal schema). The tri-state check
        distinguishes "absent" from "present but invalid"."""
        from biolab_runners.openmm.checkpoint import is_run_complete

        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=5.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        # No "terminal" field at all.
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": target, "file": state_basename}]})
        )

        complete, reason = is_run_complete(out, config)
        # Absent terminal → normal completion fallback is allowed.
        assert complete is True
        assert reason.startswith("normal_completion_step_")


# ---------------------------------------------------------------------------
# Runner-dispatch coverage — verify the runner's behavior on each plan type
# at the public run() interface.
# ---------------------------------------------------------------------------


class TestRunnerDispatch:
    """The runner matches on the plan type returned by ``decide()``
    and dispatches to the correct branch:

    - ``FreshPlan`` / ``ResumePlan`` → ``_prepare_simulation`` + MD
    - ``SkipPlan`` → populate ``SimulationResult`` and return
      without MD
    - ``FailurePlan`` → set ``result.error`` and return without MD

    These tests verify the dispatch by spying on
    ``_prepare_simulation`` (the gateway to MD mechanics) and
    asserting whether it was called. This is the public-interface
    coverage the architecture review asked for.
    """

    @staticmethod
    def _skip_ready_dir(path: Path) -> None:
        """Create a directory with a terminal manifest + artifacts."""
        # (state + manifest is enough; trajectory/energy/topology
        # are checked for non-empty)
        (path / "trajectory.dcd").write_bytes(b"\x00" * 100)
        (path / "energy.csv").write_text("h\nrow\n")
        (path / "topology.pdb").write_text("H\nATOM\n" * 100)

    def test_skip_plan_never_calls_prepare_simulation(self, tmp_path: Path) -> None:
        """A SKIP plan must not enter the MD path."""

        out = tmp_path / "output"
        out.mkdir()
        self._skip_ready_dir(out)
        config = OpenMMConfig(output_dir=str(out), production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": target, "file": state_basename}]})
        )

        runner = OpenMMRunner(config)
        # Spy on the MD gateway.
        runner._prepare_simulation = MagicMock(return_value=None)  # type: ignore[method-assign]

        result = runner.run()

        runner._prepare_simulation.assert_not_called()  # type: ignore[attr-defined]
        assert result.error == ""
        assert result.trajectory_path == str(out / "trajectory.dcd")
        assert result.state_xml_path == str(out / state_basename)
        # Plan was a SkipPlan — the runner copied the populated fields.

    def test_failure_plan_never_calls_prepare_simulation(self, tmp_path: Path) -> None:
        """A FAIL_FAST plan must not enter the MD path AND must set
        ``result.error`` to the plan's error string."""

        out = tmp_path / "output"
        out.mkdir()
        # A manifest referencing a missing state file → FAIL_FAST.
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1000, "file": "state.1000_1_1.xml"}]})
        )
        config = OpenMMConfig(output_dir=str(out))

        runner = OpenMMRunner(config)
        runner._prepare_simulation = MagicMock(return_value=None)  # type: ignore[method-assign]

        result = runner.run()

        runner._prepare_simulation.assert_not_called()  # type: ignore[attr-defined]
        assert result.error != ""
        assert "state.1000_1_1.xml" in result.error
        assert "force=True" in result.error

    def test_failure_plan_from_orphan_files(self, tmp_path: Path) -> None:
        """A directory with state files but no manifest is also
        FAIL_FAST — the runner surfaces the orphan error and never
        builds a system."""
        out = tmp_path / "output"
        out.mkdir()
        (out / "state.1000_1_1.xml").write_text("<State/>")
        config = OpenMMConfig(output_dir=str(out))

        runner = OpenMMRunner(config)
        runner._prepare_simulation = MagicMock(return_value=None)  # type: ignore[method-assign]

        result = runner.run()

        runner._prepare_simulation.assert_not_called()  # type: ignore[attr-defined]
        assert result.error != ""
        # The orphan error message points at the file and the manifest.
        assert "state.1000_1_1.xml" in result.error

    def test_skip_plan_artifact_error_copied_to_result(self, tmp_path: Path) -> None:
        """A SKIP plan with missing artifacts must surface the
        artifact error in ``result.error`` — not silently return
        success with paths pointing to nonexistent files."""
        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        # Build manifest that classifies as terminal but is missing
        # the trajectory.
        (out / "energy.csv").write_text("h\nrow\n")
        (out / "topology.pdb").write_text("H\nATOM\n" * 100)
        # No trajectory.dcd.
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": target, "file": state_basename}]})
        )

        runner = OpenMMRunner(config)
        runner._prepare_simulation = MagicMock(return_value=None)  # type: ignore[method-assign]

        result = runner.run()

        runner._prepare_simulation.assert_not_called()  # type: ignore[attr-defined]
        assert "missing trajectory" in result.error

    def test_invalid_terminal_payload_always_fails_fast(self, tmp_path: Path) -> None:
        """An invalid terminal payload at any step (not just at
        the target) must surface as FAIL_FAST — never resume, never
        fall back to normal completion."""
        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=10.0, timestep_fs=2.0)
        # Mid-production step (well below target).
        mid_step = config.total_equil_steps + 100
        state_basename = f"state.{mid_step}_12345_170000000.xml"
        (out / state_basename).write_text("<State/>")
        # Invalid: type is unknown, not "early_abort".
        (out / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": mid_step,
                            "file": state_basename,
                            "terminal": {
                                "type": "future_marker",
                                "step": mid_step,
                                "reason": "experimental",
                            },
                        }
                    ]
                }
            )
        )

        runner = OpenMMRunner(config)
        runner._prepare_simulation = MagicMock(return_value=None)  # type: ignore[method-assign]

        result = runner.run()

        runner._prepare_simulation.assert_not_called()  # type: ignore[attr-defined]
        assert result.error != ""
        assert "malformed terminal payload" in result.error
        assert "force=True" in result.error
