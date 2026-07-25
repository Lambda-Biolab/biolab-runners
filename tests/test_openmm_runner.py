"""Tests for OpenMMRunner and related utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest  # used in test annotations and raises
from biolab_runners.openmm.config import (
    DEFAULT_IRMSD_THRESHOLD_A,
    OpenMMConfig,
    SimulationResult,
)
from biolab_runners.openmm.runner import OpenMMRunner
from biolab_runners.openmm.system_builder import build_forcefield
from biolab_runners.openmm.utils import (
    load_checkpoint,
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


class TestIsRunComplete:
    """Tests for is_run_complete — the authoritative completion oracle.

    v8 change: a run is terminal when EITHER the manifest step has
    crossed total_equil_steps + total_steps (normal completion) OR
    a valid ``early_abort.json`` exists with ``aborted=True``
    (intentional early termination). File size and energy row
    counts are NOT considered — a mid-production checkpoint can
    produce a 50 MB trajectory and tens of thousands of energy
    rows while the run is still in progress.
    """

    def test_no_manifest_returns_in_progress(self, tmp_path: Path) -> None:
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        complete, reason = is_run_complete(tmp_path, config)
        assert complete is False
        assert reason == "in_progress"

    def test_manifest_below_target_returns_in_progress(self, tmp_path: Path) -> None:
        """Manifest step below target → in progress."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        # Build the referenced state file so load_checkpoint validates.
        state_basename = "state.500000_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 500_000, "file": state_basename}]})
        )

        complete, reason = is_run_complete(tmp_path, config)
        assert complete is False
        assert reason == "in_progress"

    def test_manifest_at_target_returns_normal_completion(self, tmp_path: Path) -> None:
        """Manifest step == target → terminal (normal completion)."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": target, "file": state_basename}]})
        )

        complete, reason = is_run_complete(tmp_path, config)
        assert complete is True
        assert "normal_completion" in reason
        assert str(target) in reason

    def test_early_abort_returns_complete(self, tmp_path: Path) -> None:
        """v10: terminal status comes from the manifest's ``terminal``
        payload (committed atomically with the checkpoint). A
        manifest with ``terminal.type == \"early_abort\"`` and
        ``terminal.step == manifest.step`` is terminal."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        # Manifest with the v10 terminal payload.
        state_basename = "state.5000000_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": 5_000_000,
                            "file": state_basename,
                            "terminal": {
                                "type": "early_abort",
                                "step": 5_000_000,
                                "reason": "early_dissociation",
                                "production_ns": 4.8,
                            },
                        }
                    ]
                }
            )
        )

        complete, reason = is_run_complete(tmp_path, config)
        assert complete is True
        assert reason.startswith("manifest_terminal_early_abort_step_")

    def test_early_abort_with_aborted_false_does_not_count(self, tmp_path: Path) -> None:
        """v10: a manifest whose terminal.type is not ``early_abort``
        does not count as early-aborted (treated as in-progress for
        the purpose of result reconstruction)."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        state_basename = "state.5000000_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        # terminal.type != "early_abort" — the manifest is
        # terminal but NOT an early abort.
        (tmp_path / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": 5_000_000,
                            "file": state_basename,
                            "terminal": {"type": "other", "step": 5_000_000},
                        }
                    ]
                }
            )
        )
        complete, reason = is_run_complete(tmp_path, config)
        assert complete is True
        assert reason.startswith("manifest_terminal_other_step_")

    def test_legacy_marker_alone_is_not_terminal(self, tmp_path: Path) -> None:
        """v10 BLOCKER #2 regression: a bare ``early_abort.json``
        WITHOUT a manifest is NOT terminal. The marker is a
        derived file written after the atomic save; the manifest
        is authoritative. Without a manifest, we cannot prove
        which generation the marker belongs to — treating it as
        terminal would risk mis-classifying a fresh run as
        aborted."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        (tmp_path / "early_abort.json").write_text(
            json.dumps(
                {
                    "aborted": True,
                    "abort_reason": "early_dissociation",
                    "abort_step": 5_000_000,
                }
            )
        )
        complete, reason = is_run_complete(tmp_path, config)
        assert complete is False
        assert reason == "in_progress"

    def test_terminal_step_must_equal_manifest_step(self, tmp_path: Path) -> None:
        """v10 BLOCKER #2 binding: a terminal payload's ``step``
        MUST equal the manifest's ``step``. A mismatch indicates
        data corruption and the run is treated as in-progress
        (logged as a warning)."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        state_basename = "state.5000000_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        # terminal.step (4_000_000) != manifest.step (5_000_000) — bad.
        (tmp_path / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": 5_000_000,
                            "file": state_basename,
                            "terminal": {
                                "type": "early_abort",
                                "step": 4_000_000,
                                "reason": "early_dissociation",
                            },
                        }
                    ]
                }
            )
        )
        complete, reason = is_run_complete(tmp_path, config)
        assert complete is False
        assert reason == "in_progress"

    def test_large_trajectory_does_not_make_run_complete(self, tmp_path: Path) -> None:
        """A large trajectory + many energy rows WITHOUT a manifest at
        the target step is NOT terminal. This is the v8 BLOCKER #1
        regression: file size alone must not mark a run complete."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        (tmp_path / "trajectory.dcd").write_bytes(b"\x00" * 50_000_000)
        energy_lines = ["#step,time,PE,KE,TE,temp,vol,speed\n"]
        energy_lines.extend(f"{i * 5000},{i * 10},0,0,0,310,0,0\n" for i in range(10_000))
        (tmp_path / "energy.csv").write_text("".join(energy_lines))
        # No manifest — the previous behaviour would have declared
        # this complete based on file size alone.
        complete, reason = is_run_complete(tmp_path, config)
        assert complete is False
        assert reason == "in_progress"


class TestLoadCheckpoint:
    """Tests for the v7 manifest-based checkpoint loader."""

    def test_no_manifest_returns_zero_empty(self, tmp_path: Path) -> None:
        """No manifest → ``(0, '')``."""
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""

    def test_manifest_returns_step_and_file(self, tmp_path: Path) -> None:
        """Manifest records the step AND the state file basename.

        v8: load_checkpoint validates the referenced state file.
        The test fixture must create it on disk (non-empty)."""
        state_basename = "state.1000000_12345_1700000000000000000.xml"
        (tmp_path / state_basename).write_text("<State/>")
        manifest = {
            "records": [
                {
                    "step": 1_000_000,
                    "file": state_basename,
                }
            ]
        }
        (tmp_path / "checkpoint.json").write_text(json.dumps(manifest))
        step, file = load_checkpoint(tmp_path)
        assert step == 1_000_000
        assert file == state_basename

    def test_no_manifest_returns_zero_even_with_energy(self, tmp_path: Path) -> None:
        """energy.csv alone no longer yields a step.

        The previous energy.csv fallback was removed because energy.csv
        advances at save_every_steps while state.xml saves at
        checkpoint_every_steps — they can be many hours of steps
        apart, and using the energy row would silently shorten the
        run. The runner's fail-fast for orphaned state files is
        covered in ``TestOrphanedStateFailsFast``.
        """
        (tmp_path / "energy.csv").write_text("#step,time\n5000,10\n10000,20\n15000,30\n")
        step, _ = load_checkpoint(tmp_path)
        assert step == 0

    def test_malformed_manifest_returns_zero(self, tmp_path: Path) -> None:
        """A manifest that won't parse returns (0, '') — same as missing."""
        (tmp_path / "checkpoint.json").write_text("not json {{{")
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""

    def test_manifest_with_zero_step_returns_zero(self, tmp_path: Path) -> None:
        """A manifest with a zero step is treated as no checkpoint."""
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 0, "file": "state.0.xml"}]})
        )
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""

    def test_manifest_with_missing_file_returns_zero(self, tmp_path: Path) -> None:
        """A manifest with no ``file`` field is treated as no checkpoint."""
        (tmp_path / "checkpoint.json").write_text(json.dumps({"records": [{"step": 1_000_000}]}))
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""

    def test_manifest_with_missing_state_file_raises(self, tmp_path: Path) -> None:
        """v8 BLOCKER #2: a manifest referencing a missing state file
        raises InvalidCheckpointError.

        The previous behaviour silently accepted the manifest step
        and let ``prepare_simulation`` build a fresh System with
        production accounting inherited from the never-loaded
        checkpoint — a quietly-wrong fresh build.
        """
        from biolab_runners.openmm.utils import InvalidCheckpointError

        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": "state.1000000_1_1.xml"}]})
        )
        # No state file exists.
        with pytest.raises(InvalidCheckpointError, match="does not exist"):
            load_checkpoint(tmp_path)

    def test_manifest_with_empty_state_file_raises(self, tmp_path: Path) -> None:
        """v8 BLOCKER #2: a manifest referencing an empty state file
        raises InvalidCheckpointError — the state was likely truncated
        mid-write."""
        from biolab_runners.openmm.utils import InvalidCheckpointError

        state_basename = "state.1000000_1_1.xml"
        (tmp_path / state_basename).write_text("")  # zero bytes
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": state_basename}]})
        )
        with pytest.raises(InvalidCheckpointError, match="empty"):
            load_checkpoint(tmp_path)

    def test_manifest_with_absolute_state_path_raises(self, tmp_path: Path) -> None:
        """v8 BLOCKER #2: a manifest referencing an absolute path
        (e.g. ``/etc/passwd``) raises InvalidCheckpointError.

        The runner must not be tricked into loading arbitrary files
        via path injection."""
        from biolab_runners.openmm.utils import InvalidCheckpointError

        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": "/etc/passwd"}]})
        )
        with pytest.raises(InvalidCheckpointError, match="basename"):
            load_checkpoint(tmp_path)

    def test_manifest_with_path_traversal_raises(self, tmp_path: Path) -> None:
        """v8 BLOCKER #2: a manifest referencing ``../state.xml``
        raises InvalidCheckpointError. Path traversal is rejected."""
        from biolab_runners.openmm.utils import InvalidCheckpointError

        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": "../state.xml"}]})
        )
        with pytest.raises(InvalidCheckpointError, match="basename"):
            load_checkpoint(tmp_path)

    def test_manifest_with_subdir_state_path_raises(self, tmp_path: Path) -> None:
        """v8 BLOCKER #2: a manifest referencing ``subdir/state.xml``
        raises InvalidCheckpointError. Only basenames are allowed."""
        from biolab_runners.openmm.utils import InvalidCheckpointError

        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": "subdir/state.xml"}]})
        )
        with pytest.raises(InvalidCheckpointError, match="basename"):
            load_checkpoint(tmp_path)

    def test_manifest_with_unknown_format_raises(self, tmp_path: Path) -> None:
        """v8 BLOCKER #2: a manifest referencing ``state.42.txt``
        raises InvalidCheckpointError. Only ``state.xml`` or
        ``state.<step>_<pid>_<nanos>.xml`` are valid basenames."""
        from biolab_runners.openmm.utils import InvalidCheckpointError

        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": "state.42.txt"}]})
        )
        with pytest.raises(InvalidCheckpointError, match=r"invalid.*name"):
            load_checkpoint(tmp_path)

    def test_legacy_state_xml_basename_accepted(self, tmp_path: Path) -> None:
        """v8 BLOCKER #2: the legacy ``state.xml`` basename is still
        accepted (a pre-v7 manifest may still be readable)."""
        (tmp_path / "state.xml").write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": "state.xml"}]})
        )
        step, file = load_checkpoint(tmp_path)
        assert step == 1_000_000
        assert file == "state.xml"


class TestInvalidCheckpointErrorSurfacesToRunner:
    """v8 BLOCKER #2: a manifest with an invalid state reference
    must surface as ``result.error`` from the runner, not degrade
    into a fresh build."""

    def test_missing_state_file_sets_result_error(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        # Manifest references a state file that doesn't exist.
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": "state.1000000_1_1.xml"}]})
        )
        config = OpenMMConfig(output_dir=str(out))
        runner = OpenMMRunner(config)
        result = SimulationResult(config=config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=result
        )
        assert resume is None, "dangling manifest must not resume"
        assert result.error != ""
        assert "force=True" in result.error
        assert "state.1000000_1_1.xml" in result.error

    def test_path_traversal_in_manifest_sets_result_error(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": "../state.xml"}]})
        )
        config = OpenMMConfig(output_dir=str(out))
        runner = OpenMMRunner(config)
        result = SimulationResult(config=config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=result
        )
        assert resume is None
        assert result.error != ""
        assert "force=True" in result.error

    def test_force_true_recovers_from_invalid_manifest(self, tmp_path: Path) -> None:
        """force=True quarantines the invalid manifest (and any state
        files) so the next non-forced invocation starts fresh."""
        out = tmp_path / "output"
        out.mkdir()
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": "../state.xml"}]})
        )
        # Also drop a v7 orphan state file to make sure quarantine
        # moves everything.
        (out / "state.42_1_1.xml").write_text("<State/>")
        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        result = SimulationResult(config=config)
        resume = runner._resolve_skip_or_resume(
            force=True, output_dir=out, config=config, result=result
        )
        assert resume is not None
        # Both the manifest and the orphan state file are gone.
        assert not (out / "checkpoint.json").exists()
        assert not (out / "state.42_1_1.xml").exists()
        # Stale dir contains both.
        stale_dirs = list((out / ".stale").iterdir())
        assert len(stale_dirs) == 1
        assert (stale_dirs[0] / "checkpoint.json").exists()
        assert (stale_dirs[0] / "state.42_1_1.xml").exists()


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

        runner = OpenMMRunner(config)
        # _resolve_skip_or_resume must return a RESUME tuple, not None.
        result = SimulationResult(config=config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=result
        )
        assert resume is not None, (
            "intermediate checkpoint must not be reported as complete; "
            f"got result.error={result.error!r}"
        )
        start_step, remaining_steps, resume_xml = resume
        assert start_step == mid_step
        # remaining = total_steps - (mid_step - total_equil_steps) — same
        # accounting as the resumed-equilibrium case.
        assert remaining_steps == config.total_steps - (mid_step - config.total_equil_steps)
        assert resume_xml == str(out / state_basename)

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

        def fake_prepare_simulation(config, output_dir, resume_xml, result):
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


class TestResumeAccounting:
    """Regression tests for issue #4: resume must not conflate equilibration + production."""

    def test_resume_subtracts_equil_steps(self, tmp_path: Path) -> None:
        """Remaining steps must discount equilibration from the checkpoint step counter."""
        out = tmp_path / "output"
        out.mkdir()

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        # Simulate checkpoint after full equil (200k steps) + 1 ns production (500k steps)
        checkpoint_step = config.total_equil_steps + 500_000
        # The atomic-save manifest is the authoritative source for the
        # saved step AND the file to load. v8: load_checkpoint validates
        # the referenced state file — it must exist on disk and be
        # non-empty. The v6/v7 test fixture referenced a non-existent
        # state.test.xml and pinned the silent-degradation behaviour.
        state_basename = f"state.{checkpoint_step}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": checkpoint_step, "file": state_basename}]})
        )

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
        # v8: state file must exist on disk.
        state_basename = f"state.{checkpoint_step}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": checkpoint_step, "file": state_basename}]})
        )

        runner = OpenMMRunner(config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=SimulationResult(config=config)
        )
        assert resume is not None
        _, remaining_steps, _ = resume
        assert remaining_steps == config.total_steps


class TestForceTrueQuarantine:
    """Regression test for the v5 BLOCKER (extended for v7): ``runner.run(force=True)``
    must actually retire the stale checkpoint files (the manifest,
    the energy log, AND any state file under ``state*.xml``) before
    the fresh build, so an interrupted forced run cannot leave the
    directory with a stale state that a subsequent non-forced
    invocation could load onto a freshly-built System.

    The v7 save format uses generation-versioned state files
    (``state.<step>_<pid>_<nanos>.xml``) referenced by the manifest
    (``checkpoint.json``). The quarantine must move ALL of them.
    """

    def _populate_stale_checkpoint(self, out: Path, step: int) -> None:
        """Pre-populate a v7 stale checkpoint."""
        state_basename = f"state.{step}_1_1.xml"
        state_path = out / state_basename
        state_path.write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": step, "file": state_basename}]})
        )
        (out / "energy.csv").write_text(f"#step,time\n{step},{step}\n")

    def test_force_true_quarantines_all_files(self, tmp_path: Path) -> None:
        """force=True moves state.*.xml + checkpoint.json + energy.csv to .stale/<UTC>/."""
        out = tmp_path / "output"
        out.mkdir()

        self._populate_stale_checkpoint(out, step=10_000)

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        resume = runner._resolve_skip_or_resume(
            force=True, output_dir=out, config=config, result=SimulationResult(config=config)
        )

        # .stale/<UTC>/ directory exists with all three files
        stale_parents = list((out / ".stale").iterdir())
        assert len(stale_parents) == 1, f"expected one stale dir, got {stale_parents}"
        stale_dir = stale_parents[0]
        # The state file is in the stale directory under its versioned name.
        stale_state_files = list(stale_dir.glob("state*.xml"))
        assert len(stale_state_files) == 1
        assert stale_state_files[0].name == "state.10000_1_1.xml"
        assert (stale_dir / "checkpoint.json").exists()
        assert (stale_dir / "energy.csv").exists()

        # The original files are GONE from output_dir.
        assert not (out / "checkpoint.json").exists()
        assert not (out / "energy.csv").exists()
        assert not any(out.glob("state*.xml"))

        # resume_xml must be empty so prepare_simulation does NOT
        # try to loadState a (no-longer-existing) checkpoint.
        assert resume is not None
        _, _, resume_xml = resume
        assert resume_xml == "", f"resume_xml must be empty after quarantine, got {resume_xml!r}"

    def test_force_true_with_no_existing_checkpoint_is_a_no_op(self, tmp_path: Path) -> None:
        """force=True with no prior checkpoint must not create .stale/."""
        out = tmp_path / "output"
        out.mkdir()

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        resume = runner._resolve_skip_or_resume(
            force=True, output_dir=out, config=config, result=SimulationResult(config=config)
        )

        # No .stale directory should have been created
        assert not (out / ".stale").exists()
        # Resume proceeds normally with empty resume_xml
        assert resume is not None
        _, _, resume_xml = resume
        assert resume_xml == ""

    def test_force_true_then_interrupted_then_non_force_yields_empty_resume(
        self, tmp_path: Path
    ) -> None:
        """The v5 BLOCKER scenario end-to-end.

        1. Stale checkpoint files exist.
        2. force=True invocation quarantines them.
        3. Run is interrupted AFTER quarantine but BEFORE a new
           state file is written (simulated by writing a fresh
           topology.pdb to mimic an interrupted-during-equilibration
           outcome).
        4. A subsequent non-force invocation must NOT see a non-empty
           resume_xml — there is no checkpoint left to load.
        """
        out = tmp_path / "output"
        out.mkdir()
        self._populate_stale_checkpoint(out, step=10_000)

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)

        # Step 2: forced run quarantines the checkpoint
        resume = runner._resolve_skip_or_resume(
            force=True, output_dir=out, config=config, result=SimulationResult(config=config)
        )
        assert resume is not None
        _, _, resume_xml = resume
        assert resume_xml == ""

        # Step 3: simulate interruption by writing a fresh topology.pdb
        # (the forced run had time to overwrite topology.pdb but did
        # NOT survive long enough to write a new state file)
        (out / "topology.pdb").write_bytes(b"X" * 150_000)

        # Step 4: subsequent non-forced invocation — must not see a
        # checkpoint. resume_xml must be empty.
        result = SimulationResult(config=config)
        resume2 = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=result
        )
        assert resume2 is not None
        _, _, resume_xml2 = resume2
        assert resume_xml2 == "", (
            f"subsequent non-force invocation must not see a stale checkpoint; "
            f"got resume_xml={resume_xml2!r}"
        )


class TestOrphanedStateFailsFast:
    """Regression tests for the v6 BLOCKER: a non-empty state file
    (legacy ``state.xml`` or v7 ``state.<gen>.xml``) without a
    matching ``checkpoint.json`` manifest is treated as orphaned
    and rejected.

    The previous load_checkpoint_step fell back to ``energy.csv``'s
    last row when the manifest was missing, which silently shortened
    resumed runs (energy.csv advances at save_every_steps while
    state.xml saves at checkpoint_every_steps — a 4-orders-of-
    magnitude cadence difference). The v6 fix makes the manifest
    the only authoritative source for the saved step and fails fast
    when the state/manifest pair is broken.
    """

    def test_legacy_state_with_no_manifest_fails_fast(self, tmp_path: Path) -> None:
        """Legacy state.xml exists, checkpoint.json missing → no resume, error set."""
        out = tmp_path / "output"
        out.mkdir()
        (out / "state.xml").write_text("<State/>")
        # No checkpoint.json — orphaned state.

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        result = SimulationResult(config=config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=result
        )

        # Must NOT return a resume tuple (would proceed to fresh build
        # and overwrite topology.pdb, re-introducing the
        # incompatibility class).
        assert resume is None, "orphaned state must not be resumed"
        assert result.error != ""
        assert "state.xml" in result.error
        assert "checkpoint.json" in result.error
        assert "force=True" in result.error

    def test_v7_state_with_no_manifest_fails_fast(self, tmp_path: Path) -> None:
        """v7 state.<gen>.xml exists, checkpoint.json missing → no resume, error set."""
        out = tmp_path / "output"
        out.mkdir()
        (out / "state.500000_12345_170000000.xml").write_text("<State/>")
        # No checkpoint.json — orphaned state.

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        result = SimulationResult(config=config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=result
        )

        assert resume is None
        assert result.error != ""
        assert "force=True" in result.error

    def test_state_with_corrupt_manifest_fails_fast(self, tmp_path: Path) -> None:
        """state file exists, checkpoint.json is malformed → no resume, error set."""
        out = tmp_path / "output"
        out.mkdir()
        (out / "state.xml").write_text("<State/>")
        (out / "checkpoint.json").write_text("{this is not json")

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        result = SimulationResult(config=config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=result
        )

        assert resume is None
        assert result.error != ""
        assert "checkpoint.json" in result.error

    def test_orphaned_state_recovered_by_force(self, tmp_path: Path) -> None:
        """force=True on an orphaned state quarantines the state, then
        resumes as a fresh build (no stale state to pair against)."""
        out = tmp_path / "output"
        out.mkdir()
        (out / "state.xml").write_text("<State/>")
        # No checkpoint.json.

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        result = SimulationResult(config=config)
        # force=True → quarantine state.xml, then no resume (state is gone).
        resume = runner._resolve_skip_or_resume(
            force=True, output_dir=out, config=config, result=result
        )

        assert resume is not None
        _, _, resume_xml = resume
        assert resume_xml == ""
        # state.xml was moved to .stale/
        assert not (out / "state.xml").exists()
        stale_dirs = list((out / ".stale").iterdir())
        assert len(stale_dirs) == 1
        assert (stale_dirs[0] / "state.xml").exists()


class TestResumeStepUsesManifestNotEnergy:
    """Regression tests for the v6 BLOCKER: ``load_checkpoint``
    must use the manifest (checkpoint.json), not the last row of
    energy.csv. The two files advance at very different cadences —
    energy.csv at save_every_steps (~10 ps), state files at
    checkpoint_every_steps (~2 hr) — so the energy row can be
    hundreds of thousands of steps ahead of the saved state.
    """

    def test_state_at_step_N_energy_at_N_plus_k_uses_manifest_step(self, tmp_path: Path) -> None:
        """Manifest says 500_000, energy.csv says 700_000 → resume from 500_000.

        The previous behaviour would have read 700_000 from energy.csv
        and computed remaining_steps = total - 700_000, then loaded
        the 500_000 state — silently shortening the run by 200_000
        steps.
        """
        out = tmp_path / "output"
        out.mkdir()
        absolute_step = 500_000
        energy_step = 700_000  # 200k steps ahead of the saved state
        # v8: load_checkpoint validates the referenced state file.
        state_basename = f"state.{absolute_step}_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": absolute_step, "file": state_basename}]})
        )
        (out / "energy.csv").write_text(f"#step,time\n{energy_step},{energy_step}\n")

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=SimulationResult(config=config)
        )

        assert resume is not None
        start_step, remaining_steps, _ = resume
        # The start step must be the manifest step (500_000), not the
        # energy step (700_000). Equil was 200_000 steps, so the
        # production-done count is 500_000 - 200_000 = 300_000.
        assert start_step == absolute_step, (
            f"start_step must be the manifest step ({absolute_step}), "
            f"not the energy step ({energy_step}); got {start_step}"
        )
        # remaining = total_steps - production done = 50_000_000 - 300_000
        assert remaining_steps == config.total_steps - 300_000

    def test_load_checkpoint_ignores_energy_csv(self, tmp_path: Path) -> None:
        """energy.csv alone no longer yields a step."""
        (tmp_path / "energy.csv").write_text("#step,time\n5000,10\n10000,20\n15000,30\n")
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""


class TestQuarantineTimestampUniqueness:
    """Regression test for the v6 SUGGESTION: rapid ``force=True``
    invocations must not collide on the quarantine timestamp.

    The previous format used second-resolution (`%Y%m%dT%H%M%SZ`)
    combined with ``mkdir(parents=True, exist_ok=False)``. Two
    invocations within the same second would race the existence
    check and one would raise FileExistsError, leaving the stale
    checkpoint in place. The v6 fix uses microsecond + PID for
    uniqueness.
    """

    def test_two_rapid_force_calls_do_not_collide(self, tmp_path: Path) -> None:
        """Two consecutive force=True calls in the same millisecond produce
        distinct .stale/ directories — neither raises FileExistsError."""
        out = tmp_path / "output"
        out.mkdir()
        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)

        # First invocation: populates a stale v7 checkpoint and quarantines it.
        (out / "state.1_1_1.xml").write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1, "file": "state.1_1_1.xml"}]})
        )
        result1 = SimulationResult(config=config)
        runner._resolve_skip_or_resume(force=True, output_dir=out, config=config, result=result1)

        # Second invocation: must produce a NEW .stale/<ts>/ dir, not
        # raise FileExistsError on the first one.
        (out / "state.2_1_1.xml").write_text("<State/>")  # second stale batch
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 2, "file": "state.2_1_1.xml"}]})
        )
        result2 = SimulationResult(config=config)
        runner._resolve_skip_or_resume(force=True, output_dir=out, config=config, result=result2)

        stale_dirs = list((out / ".stale").iterdir())
        assert len(stale_dirs) == 2, f"expected 2 distinct .stale/ dirs, got {stale_dirs}"
        assert stale_dirs[0] != stale_dirs[1]


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
        captured: dict[str, object] = {}

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
        captured: dict[str, object] = {}

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
    when the state file was written.
    """

    def test_multi_resume_step_is_cumulative_and_monotonic(self, tmp_path: Path) -> None:
        """Simulate the multi-resume protocol:

        1. Start fresh: equil runs, simulation at step 200_000.
        2. Run 100_000 production steps → save at step 300_000.
        3. Resume from step 300_000, run 750_000 more → save at step 1_050_000.
        4. Resume again, run 200_000 more → save at step 1_250_000.

        Each save must record the absolute step. The next resume
        must read the absolute step and compute remaining_steps
        correctly.
        """
        out = tmp_path / "output"
        out.mkdir()

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)

        # Simulate three "_atomic_save_checkpoint" sequences.
        # Save #1: fresh, start_step=200_000 (after equil), steps_done=100_000
        from biolab_runners.openmm.system_builder import _atomic_save_checkpoint

        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")

        # Save #1: end of first invocation
        _atomic_save_checkpoint(sim, out, absolute_step=300_000)
        # Save #2: end of second invocation (resumed from 300_000)
        _atomic_save_checkpoint(sim, out, absolute_step=1_050_000)
        # Save #3: end of third invocation (resumed from 1_050_000)
        _atomic_save_checkpoint(sim, out, absolute_step=1_250_000)

        # Read the manifest.
        step, file = load_checkpoint(out)
        assert step == 1_250_000, (
            f"manifest step must be the absolute final step (1_250_000), got {step}"
        )
        assert file.startswith("state.1250000_")
        assert (out / file).exists()

        # Now resume and verify the runner computes remaining_steps
        # correctly from the absolute step.
        runner = OpenMMRunner(config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=SimulationResult(config=config)
        )
        assert resume is not None
        start_step, remaining_steps, _ = resume
        # Simulator is at step 1_250_000 when the loop starts.
        assert start_step == 1_250_000
        # Production done = 1_250_000 - 200_000 (equil) = 1_050_000.
        # remaining = total_steps - 1_050_000.
        assert remaining_steps == config.total_steps - 1_050_000

        # Sanity: the saved step is monotonically increasing. The
        # LAST save is the one referenced by the manifest; previous
        # state files were GC'd by the atomic save.
        state_files = list(out.glob("state*.xml"))
        assert len(state_files) == 1, (
            f"expected exactly 1 state file (the active one), got {state_files}"
        )
        assert state_files[0].name == file

    def test_save_writes_absolute_step_with_start_step(self, tmp_path: Path) -> None:
        """The atomic save called with start_step + steps_done
        produces a manifest whose step is the absolute value."""
        from biolab_runners.openmm.system_builder import _atomic_save_checkpoint

        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")

        # Resume scenario: start_step = 700_000 (saved step), steps_done = 100_000
        start_step = 700_000
        steps_done = 100_000
        absolute_step = start_step + steps_done
        _atomic_save_checkpoint(sim, tmp_path, absolute_step=absolute_step)

        step, _ = load_checkpoint(tmp_path)
        assert step == absolute_step, (
            f"manifest step must be the absolute step ({absolute_step}), "
            f"not the local counter ({steps_done}); got {step}"
        )


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
        from biolab_runners.openmm import system_builder as sb
        from biolab_runners.openmm.system_builder import _atomic_save_checkpoint

        # Pre-create a previous coherent checkpoint.
        previous_state = tmp_path / "state.500_12345_1000.xml"
        previous_state.write_text("<PREVIOUS_STATE/>")
        previous_manifest = {"records": [{"step": 500, "file": "state.500_12345_1000.xml"}]}
        (tmp_path / "checkpoint.json").write_text(json.dumps(previous_manifest))

        # Now simulate a new save that fails during the manifest rename.
        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<NEW_STATE_PARTIAL/>")

        original_replace = sb.os.replace

        def failing_replace(src: str, dst: str) -> None:
            # Let the state file rename succeed (it's a no-op in v7 —
            # the state file is written directly to its versioned name,
            # no rename needed). Let other calls succeed; only the
            # manifest rename fails.
            if str(dst).endswith("checkpoint.json"):
                raise OSError("simulated disk-full during manifest rename")
            original_replace(src, dst)

        sb.os.replace = failing_replace
        try:
            with pytest.raises(OSError, match="simulated"):
                _atomic_save_checkpoint(sim, tmp_path, absolute_step=999_999)
        finally:
            sb.os.replace = original_replace

        # The MANIFEST is unchanged — the previous checkpoint remains
        # active. The next resume reads the previous step (500), not
        # the half-saved new step (999_999).
        step, file = load_checkpoint(tmp_path)
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
        _atomic_save_checkpoint(sim, tmp_path, absolute_step=800_000)
        # Orphan is gone; new state file is present.
        assert not new_state_files[0].exists()
        step, file = load_checkpoint(tmp_path)
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
        step, file = load_checkpoint(tmp_path)
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
        step, file = load_checkpoint(tmp_path)
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


class TestForceTrueQuarantinesEarlyAbortMarker:
    """v9 BLOCKER #1 regression: ``runner.run(force=True)`` must
    retire the ``early_abort.json`` marker together with the
    manifest, the energy log, and every state file. A stale marker
    would otherwise be re-read by :func:`is_run_complete` and
    mis-classify a subsequent fresh run's intermediate checkpoint
    as terminal.

    The scenario from the reviewer:

    1. Run A terminates via the early-abort gate and writes
       ``early_abort.json``.
    2. The user invokes ``force=True``.
    3. Run B starts fresh and writes an intermediate checkpoint.
    4. Run B is interrupted.
    5. The next non-forced invocation reads Run B's valid
       intermediate manifest AND Run A's stale ``early_abort.json``
       and incorrectly skips Run B as terminal.

    The fix is quarantine: ``force=True`` moves the marker to
    ``.stale/<UTC>/early_abort.json`` BEFORE the fresh build, so
    Run B can never see it.
    """

    def _populate_early_aborted_run(self, out: Path, abort_step: int) -> str:
        """Set up a directory representing an early-aborted run.

        Returns the state basename (so tests can reference it)."""
        state_basename = f"state.{abort_step}_12345_1700000000000.xml"
        (out / state_basename).write_text("<STATE/>")
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": abort_step, "file": state_basename}]})
        )
        (out / "energy.csv").write_text(f"#step,time\n{abort_step},{abort_step}\n")
        abort_meta = {
            "aborted": True,
            "abort_reason": "early_dissociation",
            "abort_step": abort_step,
            "abort_ns": round(abort_step * 2.0 / 1e6, 2),
            "gate": "offline_mdtraj",
        }
        (out / "early_abort.json").write_text(json.dumps(abort_meta))
        return state_basename

    def test_force_true_quarantines_early_abort_marker(self, tmp_path: Path) -> None:
        """force=True moves early_abort.json to .stale/<UTC>/."""
        out = tmp_path / "output"
        out.mkdir()
        self._populate_early_aborted_run(out, abort_step=5_000_000)

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        resume = runner._resolve_skip_or_resume(
            force=True, output_dir=out, config=config, result=SimulationResult(config=config)
        )
        assert resume is not None

        # early_abort.json is gone from the output dir.
        assert not (out / "early_abort.json").exists()
        # And it lives in the .stale/<UTC>/ directory.
        stale_dirs = list((out / ".stale").iterdir())
        assert len(stale_dirs) == 1
        stale_dir = stale_dirs[0]
        assert (stale_dir / "early_abort.json").exists()
        assert (stale_dir / "checkpoint.json").exists()
        assert (stale_dir / "energy.csv").exists()

    def test_force_true_then_intermediate_checkpoint_resumes(self, tmp_path: Path) -> None:
        """The full BLOCKER #1 scenario end-to-end:

        1. Run A aborts, leaves early_abort.json.
        2. force=True → marker quarantined.
        3. Run B writes an intermediate checkpoint (manifest at
           step 5_000_000, below the 50_200_000 target).
        4. A subsequent non-forced invocation must RESUME (not
           skip as terminal) because the stale marker was moved
           out by force=True.
        """
        out = tmp_path / "output"
        out.mkdir()
        self._populate_early_aborted_run(out, abort_step=5_000_000)

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)

        # Step 2: forced run quarantines everything.
        runner._resolve_skip_or_resume(
            force=True,
            output_dir=out,
            config=config,
            result=SimulationResult(config=config),
        )
        assert not (out / "early_abort.json").exists()

        # Step 3: simulate Run B writing an intermediate checkpoint.
        # v9 binding: the new manifest's step and the (would-be)
        # abort step are absolute. Run B started fresh, so it would
        # reach the 200_000-equil endpoint before any production.
        intermediate_step = config.total_equil_steps + 1_000_000
        state_basename = f"state.{intermediate_step}_1_1.xml"
        (out / state_basename).write_text("<NEW_STATE/>")
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": intermediate_step, "file": state_basename}]})
        )
        (out / "energy.csv").write_text(f"#step,time\n{intermediate_step},{intermediate_step}\n")

        # Step 4: a non-forced invocation must RESUME, not skip.
        result = SimulationResult(config=config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=result
        )
        assert resume is not None, (
            f"After force=True quarantined the abort marker, a subsequent "
            f"intermediate checkpoint must resume; got result.error={result.error!r}"
        )
        start_step, remaining_steps, resume_xml = resume
        assert start_step == intermediate_step
        # remaining = total_steps - production_done
        production_done = intermediate_step - config.total_equil_steps
        assert remaining_steps == config.total_steps - production_done
        assert resume_xml == str(out / state_basename)

    def test_successful_forced_rerun_leaves_no_stale_marker(self, tmp_path: Path) -> None:
        """After force=True, a successful fresh run that completes
        normally must not leave any trace of the previous abort
        marker. A downstream consumer looking at ``early_abort.json``
        must see "no marker"."""
        out = tmp_path / "output"
        out.mkdir()
        self._populate_early_aborted_run(out, abort_step=5_000_000)

        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        # force=True quarantines the marker.
        runner._resolve_skip_or_resume(
            force=True,
            output_dir=out,
            config=config,
            result=SimulationResult(config=config),
        )
        # Simulate a normal completion: manifest at the END step.
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (out / state_basename).write_text("<FINAL_STATE/>")
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": target, "file": state_basename}]})
        )

        # No early_abort.json should be present (the successful
        # run did not abort; the stale marker was moved to .stale/).
        assert not (out / "early_abort.json").exists()
        # The is_run_complete check should classify this as
        # "normal completion", not early_abort.
        from biolab_runners.openmm.utils import is_run_complete

        complete, reason = is_run_complete(out, config)
        assert complete is True
        assert reason.startswith("normal_completion_step_"), (
            f"successful forced rerun must not be classified as early_abort; got {reason!r}"
        )


class TestIsRunCompleteValidatesTerminalPayload:
    """v10 BLOCKER #2 binding: ``is_run_complete`` validates the
    manifest's ``terminal`` payload. The ``step`` field MUST be a
    positive integer equal to the manifest's ``step`` — a missing,
    zero, malformed, or mismatched ``terminal.step`` is treated as
    "payload invalid → run is in progress" rather than terminal.

    The previous v9 tests pinned the marker-only behaviour, which
    is unsafe (a stale ``early_abort.json`` from a previous run
    could mis-classify a fresh run as terminal). v10 makes the
    manifest authoritative; the marker is a derived file.
    """

    def _write_manifest_with_terminal(
        self, out: Path, *, terminal_step: object, terminal_type: str = "early_abort"
    ) -> None:
        """Write a manifest with the given terminal payload."""
        state_basename = "state.5000000_1_1.xml"
        (out / state_basename).write_text("<State/>")
        manifest: dict[str, object] = {
            "records": [
                {
                    "step": 5_000_000,
                    "file": state_basename,
                    "terminal": {
                        "type": terminal_type,
                        "step": terminal_step,
                        "reason": "early_dissociation",
                    },
                }
            ]
        }
        (out / "checkpoint.json").write_text(json.dumps(manifest))

    def test_terminal_step_zero_is_not_terminal(self, tmp_path: Path) -> None:
        """terminal.step=0 → not terminal."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        self._write_manifest_with_terminal(tmp_path, terminal_step=0)
        complete, reason = is_run_complete(tmp_path, config)
        assert complete is False
        assert reason == "in_progress"

    def test_terminal_step_missing_is_not_terminal(self, tmp_path: Path) -> None:
        """terminal field missing step → not terminal."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        state_basename = "state.5000000_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "step": 5_000_000,
                            "file": state_basename,
                            "terminal": {"type": "early_abort"},  # no step
                        }
                    ]
                }
            )
        )
        complete, reason = is_run_complete(tmp_path, config)
        assert complete is False
        assert reason == "in_progress"

    def test_terminal_step_invalid_type_does_not_raise(self, tmp_path: Path) -> None:
        """terminal.step="not_a_number" must not raise; not terminal."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        self._write_manifest_with_terminal(tmp_path, terminal_step="not_a_number")
        complete, reason = is_run_complete(tmp_path, config)
        assert complete is False
        assert reason == "in_progress"

    def test_terminal_step_list_does_not_raise(self, tmp_path: Path) -> None:
        """terminal.step=[1,2,3] must not raise."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        self._write_manifest_with_terminal(tmp_path, terminal_step=[1, 2, 3])
        complete, reason = is_run_complete(tmp_path, config)
        assert complete is False
        assert reason == "in_progress"

    def test_terminal_step_null_does_not_raise(self, tmp_path: Path) -> None:
        """terminal.step=null must not raise; not terminal."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        self._write_manifest_with_terminal(tmp_path, terminal_step=None)
        complete, reason = is_run_complete(tmp_path, config)
        assert complete is False
        assert reason == "in_progress"

    def test_terminal_step_positive_and_equal_to_manifest_step_is_terminal(
        self, tmp_path: Path
    ) -> None:
        """terminal.step matches manifest.step → terminal."""
        from biolab_runners.openmm.utils import is_run_complete

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        self._write_manifest_with_terminal(tmp_path, terminal_step=5_000_000)
        complete, reason = is_run_complete(tmp_path, config)
        assert complete is True
        assert reason == "manifest_terminal_early_abort_step_5000000"


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
        from biolab_runners.openmm.utils import is_run_complete

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
        """``_atomic_save_checkpoint`` embeds the terminal payload
        in the manifest record on the same os.replace."""
        from biolab_runners.openmm.system_builder import _atomic_save_checkpoint

        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")
        terminal = {
            "type": "early_abort",
            "step": 2_700_000,
            "reason": "early_dissociation",
            "production_ns": 5.0,
        }
        state_basename = _atomic_save_checkpoint(
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
        from biolab_runners.openmm.system_builder import _atomic_save_checkpoint

        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")
        _atomic_save_checkpoint(sim, tmp_path, absolute_step=50_200_000)

        manifest = json.loads((tmp_path / "checkpoint.json").read_text())
        record = manifest["records"][-1]
        assert record["step"] == 50_200_000
        assert "terminal" not in record


class TestProductionNsSemantics:
    """v10 BLOCKER #3 regression: every ns reported to downstream
    consumers (``abort_ns``, ``result.total_ns``, ``md_summary.json``,
    reconstructed results) is computed from COMPLETED PRODUCTION
    steps (``absolute_step - total_equil_steps``), not absolute
    OpenMM step.

    The previous code reported ``absolute_step * timestep_fs / 1e6``
    for the abort metadata but ``steps_done * timestep_fs / 1e6``
    for the live ``result.total_ns``. After a 5 ns production
    abort the marker said 5.4 ns while the live result said 5.0
    ns — three different meanings of ns in the same run.
    """

    def test_abort_ns_in_marker_is_production_not_absolute(self, tmp_path: Path) -> None:
        """``early_abort.json.abort_ns`` is production ns."""
        from biolab_runners.openmm.runner import OpenMMRunner

        verdict = MagicMock()
        verdict.reason = "early_dissociation"
        verdict.rmsd_5ns = 5.0
        verdict.rmsd_10ns = 6.0
        verdict.max_rmsd = 7.0
        verdict.slope_a_per_ns = 0.5
        verdict.receptor_fit_residual = 1.2

        config = OpenMMConfig(production_ns=100.0, timestep_fs=2.0)
        total_equil = config.total_equil_steps  # 200_000 for the default
        # absolute_step = total_equil + 5_000_000 → production_steps = 5_000_000
        absolute_step = total_equil + 5_000_000
        production_steps = absolute_step - total_equil
        expected_production_ns = round(production_steps * 2.0 / 1e6, 2)

        OpenMMRunner._write_abort_metadata(
            verdict,
            tmp_path,
            abort_thresh=5.0,
            config=config,
            absolute_step=absolute_step,
            production_ns=expected_production_ns,
        )
        meta = json.loads((tmp_path / "early_abort.json").read_text())
        # abort_ns is production_ns, NOT absolute_ns.
        assert meta["abort_ns"] == expected_production_ns
        # Sanity: production_ns != absolute_ns in this case.
        absolute_ns = round(absolute_step * 2.0 / 1e6, 2)
        assert expected_production_ns != absolute_ns

    def test_total_ns_uses_production_steps_for_fresh_run(self, tmp_path: Path) -> None:
        """A fresh-run finalize uses production steps for total_ns."""
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
            start_step=config.total_equil_steps,  # post-equil
            steps_done=config.total_steps,  # full production
            t0=time.time() - 60.0,
            output_dir=tmp_path,
        )
        # total_ns == production_ns == config.production_ns (1.0 ns)
        assert result.total_ns == 1.0

    def test_total_ns_uses_cumulative_production_after_resume(self, tmp_path: Path) -> None:
        """A resumed run's finalize reports cumulative production,
        not the local steps_done."""
        from biolab_runners.openmm.runner import OpenMMRunner

        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0, timestep_fs=2.0)
        result = SimulationResult(config=config)
        sim = MagicMock()
        sim.saveState.side_effect = lambda path: Path(path).write_text("<State/>")

        # Resumed from step 1_000_000 (post-equil). Did 1_000_000
        # more production steps this invocation.
        start_step = 1_000_000
        steps_done = 1_000_000
        absolute_step = start_step + steps_done
        # Production steps = absolute_step - total_equil_steps.
        production_steps = absolute_step - config.total_equil_steps
        expected_total_ns = round(production_steps * 2.0 / 1e6, 2)

        OpenMMRunner._finalize_result(
            ctx=MagicMock(simulation=sim),
            result=result,
            energy_fh=MagicMock(),
            traj_path=str(tmp_path / "trajectory.dcd"),
            energy_path=str(tmp_path / "energy.csv"),
            start_step=start_step,
            steps_done=steps_done,
            t0=time.time() - 60.0,
            output_dir=tmp_path,
        )
        assert result.total_ns == expected_total_ns
        # Sanity: not the local steps_done * timestep_fs / 1e6.
        wrong_total_ns = round(steps_done * 2.0 / 1e6, 2)
        assert result.total_ns != wrong_total_ns


class TestManifestStepEqualsStateFilenameStep:
    """v10 BLOCKER #1 regression: the manifest's ``step`` MUST
    equal the step encoded in the v7 state filename. A mismatch
    indicates a corrupt or forged checkpoint — the resume path
    would silently pair a state saved at step A with resume
    accounting at step B. The runner must fail fast with
    ``force=True`` guidance, never degrade into a fresh build.
    """

    def test_v7_step_mismatch_raises_invalid(self, tmp_path: Path) -> None:
        """Manifest says step 1_000_000 but file encodes 500_000 → raises."""
        from biolab_runners.openmm.utils import InvalidCheckpointError

        # Filename encodes step 500_000 — manifest claims 1_000_000.
        state_basename = "state.500000_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": state_basename}]})
        )

        with pytest.raises(InvalidCheckpointError, match="does not match"):
            load_checkpoint(tmp_path)

    def test_v7_step_match_returns_step_and_file(self, tmp_path: Path) -> None:
        """Manifest step == embedded step → returns (step, file)."""
        state_basename = "state.1000000_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": state_basename}]})
        )
        step, file = load_checkpoint(tmp_path)
        assert step == 1_000_000
        assert file == state_basename

    def test_legacy_state_xml_skips_step_equality_check(self, tmp_path: Path) -> None:
        """Legacy ``state.xml`` has no embedded step — the manifest's
        step is trusted as-is (logged as a compatibility notice)."""
        (tmp_path / "state.xml").write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": "state.xml"}]})
        )
        # No exception — legacy compat path.
        step, file = load_checkpoint(tmp_path)
        assert step == 1_000_000
        assert file == "state.xml"

    def test_mismatch_surfaces_to_runner(self, tmp_path: Path) -> None:
        """Mismatch from the runner's perspective → ``result.error``."""
        out = tmp_path / "output"
        out.mkdir()
        state_basename = "state.500000_1_1.xml"
        (out / state_basename).write_text("<State/>")
        (out / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1_000_000, "file": state_basename}]})
        )
        config = OpenMMConfig(output_dir=str(out), production_ns=100.0, timestep_fs=2.0)
        runner = OpenMMRunner(config)
        result = SimulationResult(config=config)
        resume = runner._resolve_skip_or_resume(
            force=False, output_dir=out, config=config, result=result
        )
        assert resume is None
        assert result.error != ""
        assert "force=True" in result.error


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


class TestLoadCheckpointMalformedManifest:
    """v10 SUGGESTION regression: ``load_checkpoint`` must tolerate
    structurally malformed manifests without raising TypeError or
    AttributeError. Valid JSON shapes that aren't valid manifests
    should be treated as "no checkpoint" (return (0, "")).
    """

    def test_root_is_list_returns_zero(self, tmp_path: Path) -> None:
        """``[]`` (root is a list) — not a manifest."""
        (tmp_path / "checkpoint.json").write_text("[]")
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""

    def test_records_is_null_returns_zero(self, tmp_path: Path) -> None:
        """``{"records": null}`` — records missing."""
        (tmp_path / "checkpoint.json").write_text(json.dumps({"records": None}))
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""

    def test_records_is_empty_returns_zero(self, tmp_path: Path) -> None:
        """``{"records": []}`` — empty."""
        (tmp_path / "checkpoint.json").write_text(json.dumps({"records": []}))
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""

    def test_last_record_is_null_returns_zero(self, tmp_path: Path) -> None:
        """``{"records": [null]}`` — last record not a mapping."""
        (tmp_path / "checkpoint.json").write_text(json.dumps({"records": [None]}))
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""

    def test_step_is_null_returns_zero(self, tmp_path: Path) -> None:
        """``step: null`` — not a valid int."""
        state_basename = "state.1_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": None, "file": state_basename}]})
        )
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""

    def test_step_is_string_returns_zero(self, tmp_path: Path) -> None:
        """``step: "1000"`` — must be int, not str."""
        state_basename = "state.1000_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": "1000", "file": state_basename}]})
        )
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""

    def test_step_is_bool_returns_zero(self, tmp_path: Path) -> None:
        """``step: true`` — bool is not a valid step (it's truthy)."""
        state_basename = "state.1_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": True, "file": state_basename}]})
        )
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""

    def test_file_is_null_returns_zero(self, tmp_path: Path) -> None:
        """``file: null`` — not a valid string."""
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1000, "file": None}]})
        )
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""

    def test_file_is_int_returns_zero(self, tmp_path: Path) -> None:
        """``file: 42`` — must be str, not int."""
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"records": [{"step": 1000, "file": 42}]})
        )
        step, file = load_checkpoint(tmp_path)
        assert step == 0
        assert file == ""
