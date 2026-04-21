"""Tests for OpenMMRunner and related utilities."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from biolab_runners.openmm.config import (
    DEFAULT_IRMSD_THRESHOLD_A,
    EquilibrationStage,
    OpenMMConfig,
    SimulationResult,
)
from biolab_runners.openmm.runner import OpenMMRunner
from biolab_runners.openmm.utils import (
    load_checkpoint_step,
    verify_production_outputs,
)

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

    def test_equilibration_stages_default(self) -> None:
        config = OpenMMConfig()
        assert len(config.equilibration) == 3
        assert config.equilibration[0]["name"] == "NVT"
        assert config.equilibration[0]["restraint_k"] == 1000.0
        assert config.equilibration[2]["restraint_k"] == 0.0

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
        assert config.cacl2_mol == 0.0014
        assert config.kh2po4_mol == 0.0005
        assert config.protonation_ph == 6.2
        assert config.temperature_k == 310.0

    def test_preset_physiological(self) -> None:
        config = OpenMMConfig.physiological()
        assert config.nacl_mol == 0.150
        assert config.cacl2_mol == 0.0
        assert config.kh2po4_mol == 0.0
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


class _RecordingForceField:
    """Fake ``app.ForceField`` that records the XML paths it was constructed with."""

    def __init__(self, *paths: str) -> None:
        self.paths: tuple[str, ...] = paths


class _FakeApp:
    """Minimal stand-in for ``openmm.app`` used by ``_build_forcefield``."""

    ForceField = _RecordingForceField


class TestBuildForcefield:
    """Tests for ``OpenMMRunner._build_forcefield`` extra_forcefields pass-through."""

    def test_amber_no_extras(self) -> None:
        config = OpenMMConfig(protein_ff="amber14/protein.ff14SB", water_model="tip3p")
        ff = OpenMMRunner._build_forcefield(config, _FakeApp())
        assert ff.paths == ("amber14/protein.ff14SB.xml", "tip3p.xml")

    def test_amber_with_extras(self, tmp_path: Path) -> None:
        extra = str(tmp_path / "custom.xml")
        config = OpenMMConfig(
            protein_ff="amber14/protein.ff14SB",
            water_model="tip3p",
            extra_forcefields=[extra],
        )
        ff = OpenMMRunner._build_forcefield(config, _FakeApp())
        assert ff.paths == ("amber14/protein.ff14SB.xml", "tip3p.xml", extra)

    def test_charmm_with_extras(self, tmp_path: Path) -> None:
        """CHARMM branch must still honour extra_forcefields."""
        extra = str(tmp_path / "custom.xml")
        config = OpenMMConfig(protein_ff="charmm36m", extra_forcefields=[extra])
        ff = OpenMMRunner._build_forcefield(config, _FakeApp())
        assert ff.paths == ("charmm36.xml", "charmm36/water.xml", extra)

    def test_extras_preserve_order(self, tmp_path: Path) -> None:
        extras = [str(tmp_path / "a.xml"), str(tmp_path / "b.xml")]
        config = OpenMMConfig(protein_ff="amber14/protein.ff14SB", extra_forcefields=extras)
        ff = OpenMMRunner._build_forcefield(config, _FakeApp())
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
        ff = OpenMMRunner._build_forcefield(config, _FakeApp())
        assert ff.paths == ("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")

    def test_water_ff_xml_empty_falls_back_to_water_model(self) -> None:
        """When water_ff_xml is empty, preserve the pre-change behavior."""
        config = OpenMMConfig(
            protein_ff="amber14/protein.ff14SB",
            water_model="tip3p",
            water_ff_xml="",
        )
        ff = OpenMMRunner._build_forcefield(config, _FakeApp())
        assert ff.paths == ("amber14/protein.ff14SB.xml", "tip3p.xml")

    def test_water_ff_xml_ignored_for_charmm(self) -> None:
        """CHARMM branch hardcodes charmm36 XMLs regardless of water_ff_xml."""
        config = OpenMMConfig(
            protein_ff="charmm36m",
            water_ff_xml="amber14/tip3p.xml",  # ignored — CHARMM branch
        )
        ff = OpenMMRunner._build_forcefield(config, _FakeApp())
        assert ff.paths == ("charmm36.xml", "charmm36/water.xml")


class TestEquilibrationStage:
    """Tests for EquilibrationStage dataclass."""

    def test_creation(self) -> None:
        stage = EquilibrationStage(name="NVT", ensemble="NVT", duration_ps=100, restraint_k=1000.0)
        assert stage.name == "NVT"
        assert stage.ensemble == "NVT"

    def test_to_dict(self) -> None:
        stage = EquilibrationStage(
            name="NPT_free", ensemble="NPT", duration_ps=200, restraint_k=0.0
        )
        d = stage.to_dict()
        assert d["name"] == "NPT_free"
        assert d["restraint_k"] == 0.0

    def test_frozen(self) -> None:
        stage = EquilibrationStage(name="NVT", ensemble="NVT", duration_ps=100, restraint_k=1000.0)
        with pytest.raises(AttributeError):
            stage.name = "changed"  # type: ignore[misc]


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
# ---------------------------------------------------------------------------


class TestPbcCorrectTriclinic:
    """Regression tests pinning the #163 dodecahedron PBC fix."""

    @staticmethod
    def _dodecahedron_box(d: float = 60.0) -> object:
        """GROMACS rhombic dodecahedron (xy-square) with edge length ``d``."""
        import math

        import numpy as np

        return np.array(
            [
                [d, 0.0, 0.0],
                [0.0, d, 0.0],
                [0.5 * d, 0.5 * d, d / math.sqrt(2.0)],
            ]
        )

    def test_orthorhombic_parity_with_diagonal_formula(self) -> None:
        """For rectangular boxes the new formula must agree with the diagonal one."""
        import numpy as np

        box = np.diag([60.0, 45.0, 80.0]).astype(float)
        rng = np.random.default_rng(seed=0)
        diff = rng.uniform(-100.0, 100.0, size=(8, 3))
        out = OpenMMRunner._pbc_correct(diff.copy(), box, np)  # noqa: SLF001
        box_diag = np.array([box[0, 0], box[1, 1], box[2, 2]])
        expected = diff - np.round(diff / box_diag) * box_diag
        assert np.allclose(out, expected, atol=1e-10)

    def test_dodecahedron_single_lattice_vector_wraps_to_zero(self) -> None:
        """A displacement equal to lattice vector ``c`` must collapse to 0."""
        import numpy as np

        box = self._dodecahedron_box()
        diff = box[2].reshape(1, 3).copy()  # one c-image
        out = OpenMMRunner._pbc_correct(diff.copy(), box, np)  # noqa: SLF001
        assert np.allclose(out, 0.0, atol=1e-10)

        # Pre-fix diagonal-only formula left off-diagonal slop behind.
        box_diag = np.array([box[0, 0], box[1, 1], box[2, 2]])
        pre_fix = diff - np.round(diff / box_diag) * box_diag
        assert np.linalg.norm(pre_fix) > 30.0  # ≈ d/√2 of stale xy components

    def test_dodecahedron_face_crossing_minimum_image(self) -> None:
        """Receptor-peptide displacement that crosses a diagonal face wraps correctly.

        This is the production failure mode: in a bound complex with receptor
        near the origin and peptide images diffusing across a non-orthogonal
        face, the diagonal-only code reported min distances of 30–40 Å while
        the true minimum-image distance was < 1 Å.
        """
        import numpy as np

        box = self._dodecahedron_box(d=60.0)
        # Peptide just past one ``c`` image from receptor.
        rec = np.array([[0.0, 0.0, 0.0]])
        pep = (box[2] + np.array([0.3, 0.3, 0.1])).reshape(1, 3)
        diff = pep - rec
        out = OpenMMRunner._pbc_correct(diff.copy(), box, np)  # noqa: SLF001
        min_dist = float(np.linalg.norm(out, axis=-1).min())
        assert min_dist < 1.0  # true distance ≈ 0.436 Å

        # Pre-fix diagonal code returned ~36 Å here (huge false "dissociation").
        box_diag = np.array([box[0, 0], box[1, 1], box[2, 2]])
        pre_fix = diff - np.round(diff / box_diag) * box_diag
        assert float(np.linalg.norm(pre_fix, axis=-1).min()) > 30.0

    def test_pbc_correct_broadcasts_over_leading_axes(self) -> None:
        """``_pbc_correct`` must accept (M, N, 3) arrays used by ``_min_pbc_distance``."""
        import numpy as np

        box = self._dodecahedron_box()
        rec = np.zeros((3, 3))
        pep = np.tile(box[2], (4, 1)) + 0.5  # 4 peptide atoms past one c-image
        diffs = rec[:, None, :] - pep[None, :, :]  # shape (3, 4, 3)
        out = OpenMMRunner._pbc_correct(diffs.copy(), box, np)  # noqa: SLF001
        assert out.shape == (3, 4, 3)
        # All pairs wrap to the same ~0.866 Å displacement (−0.5,−0.5,−0.5).
        assert np.allclose(np.linalg.norm(out, axis=-1), np.sqrt(0.75), atol=1e-10)
