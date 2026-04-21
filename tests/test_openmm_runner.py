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
        a.extra_forcefields.append("/tmp/a.xml")
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
# _peptide_ca_rmsd — receptor-Cα-aligned RMSD regression tests
#
# The prior implementation subtracted peptide Cα in lab frame with only a
# PBC box-wrap correction. This made receptor diffusion/tumbling look like
# peptide dissociation: a bound peptide that moved together with a rotating
# receptor was scored as "dissociated" with RMSDs of 20+ Å when the peptide
# had never left the pocket. Confirmed in OralBiome-AMP#162 via offline
# mdtraj re-analysis (2 of 3 spot-checks overturned; 5 of 17 batch results
# overturned).
#
# These tests pin the fix: peptide RMSD must be measured in the receptor's
# reference frame after Kabsch alignment.
# ---------------------------------------------------------------------------


class _FakeState:
    """Minimal stand-in for OpenMM's State object returned by getState()."""

    def __init__(self, positions: object, box: object) -> None:
        import numpy as _np

        self._positions = _np.asarray(positions, dtype=float)
        self._box = _np.asarray(box, dtype=float)

    def getPositions(self, asNumpy: bool = False) -> object:  # noqa: ARG002, N802, N803, FBT001, FBT002
        return _QuantityStub(self._positions)

    def getPeriodicBoxVectors(self, asNumpy: bool = False) -> object:  # noqa: ARG002, N802, N803, FBT001, FBT002
        return _QuantityStub(self._box)


class _QuantityStub:
    """Return a plain numpy array from ``.value_in_unit(...)``."""

    def __init__(self, arr: object) -> None:
        self._arr = arr

    def value_in_unit(self, _unit: object) -> object:
        return self._arr


class _FakeSimulation:
    """Minimal stand-in for OpenMM's Simulation object."""

    def __init__(self, positions: object, box: object) -> None:
        self.context = _FakeContext(positions, box)

    def saveState(self, _path: str) -> None:  # noqa: N802
        """No-op — abort path writes state.xml on real sims; tests don't need it."""
        return None


class _FakeContext:
    def __init__(self, positions: object, box: object) -> None:
        self._state = _FakeState(positions, box)

    def getState(  # noqa: ARG002, N802, N803, FBT001, FBT002
        self,
        getPositions: bool = False,  # noqa: N803
        enforcePeriodicBox: bool = False,  # noqa: N803
    ) -> _FakeState:
        return self._state


class _FakeUnit:
    """Stand-in for ``openmm.unit`` — value_in_unit just ignores the argument."""

    angstroms = "angstrom_unit_marker"


class TestPeptideCaRmsdReceptorAligned:
    """Regression tests for receptor-Cα-aligned peptide RMSD (OralBiome-AMP#162)."""

    @staticmethod
    def _make_reference() -> tuple[object, object, object, list[int], list[int]]:
        """Build a simple receptor (α-helix-ish, 10 Cα) + peptide (3 Cα) frame."""
        import numpy as np

        # Receptor: 10 Cα atoms along a helix
        rec = np.array(
            [[np.cos(i * 1.0), np.sin(i * 1.0), i * 1.5] for i in range(10)],
            dtype=float,
        )
        # Peptide: 3 Cα atoms in the receptor's "pocket" (offset fixed)
        pep = np.array([[3.0, 0.0, 2.0], [3.5, 0.5, 3.5], [3.0, 1.0, 5.0]], dtype=float)
        positions = np.vstack([rec, pep])
        rec_idx = list(range(10))
        pep_idx = list(range(10, 13))
        return positions, rec, pep, rec_idx, pep_idx

    def test_identical_frame_returns_zero(self) -> None:
        """Identity: no motion → RMSD = 0."""
        import numpy as np

        positions, ref_rec, ref_pep, rec_idx, pep_idx = self._make_reference()
        sim = _FakeSimulation(positions, np.eye(3) * 100.0)
        rmsd = OpenMMRunner._peptide_ca_rmsd(  # noqa: SLF001
            sim, ref_pep, pep_idx, ref_rec, rec_idx, _FakeUnit(), np
        )
        assert rmsd == pytest.approx(0.0, abs=1e-10)

    def test_rigid_body_translation_returns_zero(self) -> None:
        """Receptor + peptide translated by the same vector → RMSD = 0."""
        import numpy as np

        positions, ref_rec, ref_pep, rec_idx, pep_idx = self._make_reference()
        shift = np.array([7.0, -3.0, 2.5])
        sim = _FakeSimulation(positions + shift, np.eye(3) * 100.0)
        rmsd = OpenMMRunner._peptide_ca_rmsd(  # noqa: SLF001
            sim, ref_pep, pep_idx, ref_rec, rec_idx, _FakeUnit(), np
        )
        assert rmsd == pytest.approx(0.0, abs=1e-10)

    def test_rigid_body_rotation_returns_zero(self) -> None:
        """Receptor + peptide rotated together by 90° about Z → RMSD ≈ 0.

        This is the load-bearing regression. The prior lab-frame implementation
        returned a large non-zero RMSD here — the exact failure mode that caused
        the 17/17 false-positive EARLY_FAIL rate on VicK + HmuY cohorts and the
        1YCR positive control.
        """
        import numpy as np

        positions, ref_rec, ref_pep, rec_idx, pep_idx = self._make_reference()
        rot_z = np.array(
            [
                [np.cos(np.pi / 2), -np.sin(np.pi / 2), 0.0],
                [np.sin(np.pi / 2), np.cos(np.pi / 2), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated = positions @ rot_z.T
        sim = _FakeSimulation(rotated, np.eye(3) * 100.0)
        rmsd = OpenMMRunner._peptide_ca_rmsd(  # noqa: SLF001
            sim, ref_pep, pep_idx, ref_rec, rec_idx, _FakeUnit(), np
        )
        assert rmsd == pytest.approx(0.0, abs=1e-8)

    def test_genuine_peptide_displacement_detected(self) -> None:
        """Peptide genuinely translated out of the pocket while receptor fixed → large RMSD."""
        import numpy as np

        positions, ref_rec, ref_pep, rec_idx, pep_idx = self._make_reference()
        moved = positions.copy()
        moved[10:13] += np.array([15.0, 0.0, 0.0])  # peptide drifts 15 Å
        sim = _FakeSimulation(moved, np.eye(3) * 100.0)
        rmsd = OpenMMRunner._peptide_ca_rmsd(  # noqa: SLF001
            sim, ref_pep, pep_idx, ref_rec, rec_idx, _FakeUnit(), np
        )
        assert rmsd == pytest.approx(15.0, abs=1e-8)

    def test_receptor_rotation_plus_peptide_drift(self) -> None:
        """Receptor rotates + peptide drifts in receptor frame → only peptide drift counted."""
        import numpy as np

        positions, ref_rec, ref_pep, rec_idx, pep_idx = self._make_reference()
        rot_z = np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Drift peptide 2 Å along +z in the reference frame, then rotate whole system.
        perturbed = positions.copy()
        perturbed[10:13] += np.array([0.0, 0.0, 2.0])
        perturbed = perturbed @ rot_z.T
        sim = _FakeSimulation(perturbed, np.eye(3) * 100.0)
        rmsd = OpenMMRunner._peptide_ca_rmsd(  # noqa: SLF001
            sim, ref_pep, pep_idx, ref_rec, rec_idx, _FakeUnit(), np
        )
        # Each of 3 peptide Cα moved 2 Å along z → RMSD = 2 Å regardless of rotation.
        assert rmsd == pytest.approx(2.0, abs=1e-8)


# ---------------------------------------------------------------------------
# _pbc_correct — triclinic / dodecahedron minimum-image regression tests
#
# OralBiome-AMP#163: the prior implementation used only the diagonal of the
# lattice (box[0][0], box[1][1], box[2][2]). For GROMACS-style rhombic
# dodecahedron cells the third lattice vector has off-diagonal components,
# so atoms crossing a non-orthogonal face were not wrapped correctly —
# minimum-image distances appeared tens of Å larger than reality.
# Discovered when VicK_g2_abc021a3_cstr's online early-abort gate fired at
# 10 ns while offline mdtraj analysis (which knows the full lattice) saw
# max Cα-RMSD < 4 Å. Fix: convert to fractional coordinates via the inverse
# lattice, snap to nearest image, convert back. Orthorhombic case reduces
# exactly to the prior diagonal operation.
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


# ---------------------------------------------------------------------------
# _check_slope_10ns — conjunctive slope gate (#167)
#
# The prior implementation fired on slope > 0.05 Å/ns alone, using the
# endpoint-to-endpoint slope (RMSD_10ns − RMSD_5ns) / 5. Both choices were
# wrong: (a) at 310 K a bound peptide's RMSD fluctuates at ~0.1-0.2 Å/ns
# from thermal noise, so the threshold triggers on stably-bound peptides;
# (b) point-to-point slope has huge variance from single-frame noise.
# This session's VicK_g2_abc021a3_cstr validation was the load-bearing
# example — online slope 0.111 Å/ns false-aborted a peptide with max 3.82 Å
# RMSD over the full 10 ns trajectory (offline mdtraj ground truth).
#
# Fix: conjunctive gate — abort only when BOTH
# (i) rmsd_10ns > abort_thresh (the same 2× iRMSD absolute bound the 5 ns
#     gate uses), AND
# (ii) regression slope over every sub-chunk sample in the 5→10 ns window
#      > 0.05 Å/ns.
# ---------------------------------------------------------------------------


class TestRegressionSlope:
    """``_regression_slope`` computes a least-squares fit, not endpoint-to-endpoint."""

    def test_matches_polyfit_on_noisy_data(self) -> None:
        """``_regression_slope`` delegates to ``np.polyfit``, bitwise-identical."""
        import numpy as np

        rng = np.random.default_rng(seed=0)
        xs = np.linspace(5.0, 10.0, 17)
        ys_clean = 3.0 + 0.1 * (xs - 5.0)  # true slope 0.1 Å/ns
        ys = ys_clean + rng.normal(0.0, 0.3, size=xs.shape)
        samples = list(zip(xs.tolist(), ys.tolist(), strict=True))
        slope = OpenMMRunner._regression_slope(samples, np)  # noqa: SLF001
        expected_slope, _ = np.polyfit(xs, ys, 1)
        assert slope == pytest.approx(float(expected_slope), abs=1e-12)

    def test_converges_to_true_slope_with_many_samples(self) -> None:
        """Averaged across many noise seeds the regression tracks the true slope.

        Endpoint-to-endpoint slope has expected value = true slope but variance
        2σ²/window²; regression over N samples has variance 12σ²/(N window²),
        so regression beats endpoint-to-endpoint by a factor of N/6. This test
        verifies that convergence behaviour across 50 seeds — an individual
        draw is not guaranteed to win, but the average absolutely does.
        """
        import numpy as np

        errs_regression: list[float] = []
        errs_endpoint: list[float] = []
        for seed in range(50):
            rng = np.random.default_rng(seed=seed)
            xs = np.linspace(5.0, 10.0, 17)
            ys = 3.0 + 0.1 * (xs - 5.0) + rng.normal(0.0, 0.3, size=xs.shape)
            samples = list(zip(xs.tolist(), ys.tolist(), strict=True))
            slope = OpenMMRunner._regression_slope(samples, np)  # noqa: SLF001
            endpoint = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
            errs_regression.append(abs(slope - 0.1))
            errs_endpoint.append(abs(endpoint - 0.1))
        assert np.mean(errs_regression) < np.mean(errs_endpoint)

    def test_two_samples_returns_endpoint_slope(self) -> None:
        import numpy as np

        samples = [(5.0, 3.1), (10.0, 3.7)]
        slope = OpenMMRunner._regression_slope(samples, np)  # noqa: SLF001
        assert slope == pytest.approx(0.12, abs=1e-10)


class TestCheckSlope10nsConjunctiveGate:
    """Regression tests pinning the #167 fix."""

    @staticmethod
    def _make_stationary_peptide_setup() -> tuple[object, list[int], object, list[int], object]:
        """A frame where the peptide coincides with its reference (RMSD = 0)."""
        import numpy as np

        rec = np.array(
            [[np.cos(i * 1.0), np.sin(i * 1.0), i * 1.5] for i in range(10)],
            dtype=float,
        )
        pep = np.array([[3.0, 0.0, 2.0], [3.5, 0.5, 3.5], [3.0, 1.0, 5.0]], dtype=float)
        positions = np.vstack([rec, pep])
        rec_idx = list(range(10))
        pep_idx = list(range(10, 13))
        box = np.eye(3) * 100.0
        sim = _FakeSimulation(positions, box)
        return sim, rec_idx, rec, pep_idx, pep

    def _make_config(self) -> OpenMMConfig:
        # production_ns > 0 satisfies the config's internal assertions;
        # the values themselves only feed into metadata fields.
        return OpenMMConfig(
            target="VicK",
            peptide_id="VicK_g2_abc021a3_cstr",
            production_ns=20.0,
            timestep_fs=2.0,
        )

    def test_thermal_fluctuation_below_abs_thresh_does_not_abort(self, tmp_path: Path) -> None:
        """The load-bearing #167 case — a bound peptide fluctuating at > 0.05 Å/ns.

        Reference positions place the peptide exactly at pocket centre so
        ``_peptide_ca_rmsd`` returns ~0 at 10 ns. Sampled history simulates
        genuine thermal fluctuation between 5 and 10 ns (mean ~3.5 Å,
        regression slope 0.11 Å/ns — the exact observed VicK signature).
        With abort_thresh = 7 Å the conjunctive gate must NOT fire:
        abs branch fails even though the slope branch would.
        """
        import numpy as np

        sim, rec_idx, ref_rec, pep_idx, ref_pep = self._make_stationary_peptide_setup()
        rng = np.random.default_rng(seed=0)
        xs = np.linspace(5.0, 10.0, 17)
        ys = 3.0 + 0.11 * (xs - 5.0) + rng.normal(0.0, 0.15, size=xs.shape)
        rmsd_samples = list(zip(xs.tolist(), ys.tolist(), strict=True))[:-1]  # drop 10 ns

        aborted = OpenMMRunner._check_slope_10ns(  # noqa: SLF001
            sim,
            ref_pep,
            pep_idx,
            ref_rec,
            rec_idx,
            rmsd_samples,
            abort_thresh=7.0,
            config=self._make_config(),
            steps_done=5_000_000,
            output_dir=tmp_path,
            unit=_FakeUnit(),
            np=np,
        )
        assert aborted is False
        assert not (tmp_path / "early_abort.json").exists()

    def test_true_dissociation_triggers_conjunctive_abort(self, tmp_path: Path) -> None:
        """Peptide drifted out of pocket (RMSD >> 7 Å) AND slope positive → abort."""
        import numpy as np

        rec = np.array(
            [[np.cos(i * 1.0), np.sin(i * 1.0), i * 1.5] for i in range(10)],
            dtype=float,
        )
        ref_pep = np.array([[3.0, 0.0, 2.0], [3.5, 0.5, 3.5], [3.0, 1.0, 5.0]], dtype=float)
        # Simulate peptide translated 9 Å out of pocket by 10 ns.
        cur_pep = ref_pep + np.array([9.0, 0.0, 0.0])
        positions = np.vstack([rec, cur_pep])
        sim = _FakeSimulation(positions, np.eye(3) * 100.0)

        xs = np.linspace(5.0, 10.0, 17)
        ys = 3.0 + 1.2 * (xs - 5.0)  # clear drift: slope = 1.2 Å/ns
        rmsd_samples = list(zip(xs.tolist(), ys.tolist(), strict=True))[:-1]

        aborted = OpenMMRunner._check_slope_10ns(  # noqa: SLF001
            sim,
            ref_pep,
            [10, 11, 12],
            rec,
            list(range(10)),
            rmsd_samples,
            abort_thresh=7.0,
            config=self._make_config(),
            steps_done=5_000_000,
            output_dir=tmp_path,
            unit=_FakeUnit(),
            np=np,
        )
        assert aborted is True
        meta = json.loads((tmp_path / "early_abort.json").read_text())
        assert meta["aborted"] is True
        assert meta["abort_reason"] == "rmsd_slope_drift"
        assert meta["gate"] == "conjunctive"
        assert meta["abs_threshold_A"] == 7.0
        assert meta["slope_A_per_ns"] > 0.05
        assert meta["peptide_ca_rmsd_10ns_A"] > 7.0

    def test_high_slope_but_rmsd_below_abs_thresh_does_not_abort(self, tmp_path: Path) -> None:
        """Slope > threshold but RMSD inside pocket → conjunctive gate holds."""
        import numpy as np

        sim, rec_idx, ref_rec, pep_idx, ref_pep = self._make_stationary_peptide_setup()
        xs = np.linspace(5.0, 10.0, 17)
        ys = 3.0 + 0.3 * (xs - 5.0)  # slope 0.3 Å/ns (would trip the old gate)
        rmsd_samples = list(zip(xs.tolist(), ys.tolist(), strict=True))[:-1]

        aborted = OpenMMRunner._check_slope_10ns(  # noqa: SLF001
            sim,
            ref_pep,
            pep_idx,
            ref_rec,
            rec_idx,
            rmsd_samples,
            abort_thresh=7.0,
            config=self._make_config(),
            steps_done=5_000_000,
            output_dir=tmp_path,
            unit=_FakeUnit(),
            np=np,
        )
        assert aborted is False

    def test_high_rmsd_but_negative_slope_does_not_abort(self, tmp_path: Path) -> None:
        """Peptide outside pocket at 10 ns but drifting back → slope branch holds."""
        import numpy as np

        rec = np.array(
            [[np.cos(i * 1.0), np.sin(i * 1.0), i * 1.5] for i in range(10)],
            dtype=float,
        )
        ref_pep = np.array([[3.0, 0.0, 2.0], [3.5, 0.5, 3.5], [3.0, 1.0, 5.0]], dtype=float)
        cur_pep = ref_pep + np.array([9.0, 0.0, 0.0])  # at 9 Å at 10 ns
        positions = np.vstack([rec, cur_pep])
        sim = _FakeSimulation(positions, np.eye(3) * 100.0)

        xs = np.linspace(5.0, 10.0, 17)
        ys = 12.0 - 0.4 * (xs - 5.0)  # was higher at 5 ns, re-binding (negative slope)
        rmsd_samples = list(zip(xs.tolist(), ys.tolist(), strict=True))[:-1]

        aborted = OpenMMRunner._check_slope_10ns(  # noqa: SLF001
            sim,
            ref_pep,
            [10, 11, 12],
            rec,
            list(range(10)),
            rmsd_samples,
            abort_thresh=7.0,
            config=self._make_config(),
            steps_done=5_000_000,
            output_dir=tmp_path,
            unit=_FakeUnit(),
            np=np,
        )
        assert aborted is False

    def test_window_too_short_skips_gate(self, tmp_path: Path) -> None:
        """Fewer than ``_MIN_SLOPE_WINDOW_NS`` of samples → return False (no abort)."""
        import numpy as np

        sim, rec_idx, ref_rec, pep_idx, ref_pep = self._make_stationary_peptide_setup()
        rmsd_samples = [(9.9, 3.0)]  # only one sample, < 2 ns window span

        aborted = OpenMMRunner._check_slope_10ns(  # noqa: SLF001
            sim,
            ref_pep,
            pep_idx,
            ref_rec,
            rec_idx,
            rmsd_samples,
            abort_thresh=7.0,
            config=self._make_config(),
            steps_done=5_000_000,
            output_dir=tmp_path,
            unit=_FakeUnit(),
            np=np,
        )
        assert aborted is False

    def test_vick_g2_abc021a3_ground_truth_would_pass(self, tmp_path: Path) -> None:
        """The exact VicK signature from 2026-04-20 must not false-abort.

        Observed online: rmsd_5ns=3.12 Å, rmsd_10ns=3.68 Å, endpoint slope
        0.111 Å/ns → EARLY_FAIL under the old gate. Offline ground truth
        (mdtraj full trajectory): max 3.82 Å, final 1.96 Å — genuinely bound.
        With the conjunctive gate, abs_rmsd < 7 Å is enough to save the run.
        """
        import numpy as np

        sim, rec_idx, ref_rec, pep_idx, ref_pep = self._make_stationary_peptide_setup()
        # 17 samples between 5 and 10 ns, endpoint RMSDs match the observed run,
        # interior points include thermal noise consistent with the online log.
        rng = np.random.default_rng(seed=42)
        xs = np.linspace(5.0, 10.0, 17)
        ys_mean = 3.12 + (3.68 - 3.12) * (xs - 5.0) / 5.0
        ys = ys_mean + rng.normal(0.0, 0.2, size=xs.shape)
        rmsd_samples = list(zip(xs.tolist(), ys.tolist(), strict=True))[:-1]

        aborted = OpenMMRunner._check_slope_10ns(  # noqa: SLF001
            sim,
            ref_pep,
            pep_idx,
            ref_rec,
            rec_idx,
            rmsd_samples,
            abort_thresh=7.0,
            config=self._make_config(),
            steps_done=5_000_000,
            output_dir=tmp_path,
            unit=_FakeUnit(),
            np=np,
        )
        assert aborted is False


# ---------------------------------------------------------------------------
# #175 regression: getState in the gate path must NOT pass
# enforcePeriodicBox=True.
#
# History:
#   #162 fix (receptor-Cα Kabsch) used enforcePeriodicBox=False + per-atom
#   _pbc_correct. Broke when rigid receptors straddled a box face — adjacent
#   atoms picked different "nearest images", fragmenting the receptor.
#
#   #174 fix switched to enforcePeriodicBox=True (per-molecule centroid
#   wrapping). Fixed fragmentation but introduced a new pathology: molecules
#   whose centroids sit near box faces get their wrapping decisions flipped
#   by thermal jitter, placing receptor and peptide in different periodic
#   images even when they're a physically bound complex. Kabsch then mis-
#   places the peptide by ~box/2, producing phantom ~50 Å RMSD on bound
#   peptides.
#
#   #175 fix (current): use OpenMM's internal unwrapped coordinates
#   (enforcePeriodicBox=False, the default). Integrated internal positions
#   are continuous across frames — no per-frame wrapping, no image flipping.
#   Molecules stay whole. Kabsch absorbs global drift via centroid
#   subtraction. Do NOT reintroduce per-atom _pbc_correct — that was the
#   original #162 failure mode and is unnecessary here.
#
# Production reproducer for #175: OralBiome-AMP
# VicK_g2_abc021a3_cstr_tier2_rep3 aborted at 5.1 ns with online gate
# reporting 51.52 Å while min-Cα PBC distance was 4.28 Å throughout
# (peptide bound to same receptor residue across all 510 frames). Ground
# truth via state.xml unwrapped Kabsch: 9.65 Å.
# ---------------------------------------------------------------------------


class TestGateCoordConventionRegression:
    """Structural regression tests for OralBiome-AMP#175.

    Guards the three requirements of the gate coord convention:
      1. ``getState(getPositions=True)`` calls in gate paths must NOT
         pass ``enforcePeriodicBox=True``.
      2. ``_peptide_ca_rmsd`` must not call ``_pbc_correct``.
      3. Math: Kabsch on receptor absorbs rigid-body translation (works
         because the internal-unwrapped convention keeps everything
         continuous).

    The math-layer dual assertion lives in the sibling OralBiome-AMP
    ``tests/test_openmm_cloud_pbc.py`` which guards the same invariant
    on the embedded cloud runner script.

    Note: this class REPLACES the older ``TestEnforcePeriodicBoxRegression``
    from PR #25 which pinned the (now-wrong) ``enforcePeriodicBox=True``
    mechanism. The history for those tests lives in git.
    """

    def test_peptide_ca_rmsd_does_not_force_periodic_box(self) -> None:
        """The gate's getState call must NOT pass ``enforcePeriodicBox=True``.

        Per #175: enforcePeriodicBox=True per-molecule-wraps centroids
        independently; when centroids sit near box faces, wrapping
        decisions flip and receptor + peptide can end up in different
        periodic images, breaking Kabsch. Use the default (False) which
        returns OpenMM's integrated internal unwrapped positions.
        """
        import ast
        import inspect
        import textwrap

        from biolab_runners.openmm.runner import OpenMMRunner

        src = textwrap.dedent(inspect.getsource(OpenMMRunner._peptide_ca_rmsd))
        tree = ast.parse(src)
        found_call = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = ast.unparse(node.func)
            if "getState" not in callee:
                continue
            kwargs = {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg}
            assert kwargs.get("enforcePeriodicBox") != "True", (
                f"getState() in _peptide_ca_rmsd must NOT pass "
                f"enforcePeriodicBox=True (guards OralBiome-AMP#175). "
                f"Got kwargs: {kwargs}. Use the default (False) which "
                f"returns internal unwrapped coordinates."
            )
            found_call = True
        assert found_call, "expected at least one getState() call in _peptide_ca_rmsd body"

    def test_peptide_ca_rmsd_removed_per_atom_pbc(self) -> None:
        """``_peptide_ca_rmsd`` must not call ``_pbc_correct``.

        Per-atom PBC unwrap was the original #162 failure mode (fragments
        rigid bodies across box faces). Unnecessary under #175 because
        OpenMM's internal unwrapped positions are already continuous.
        Walks the AST and checks ``Call`` nodes only, so the docstring
        is free to mention ``_pbc_correct`` by name.
        """
        import ast
        import inspect
        import textwrap

        from biolab_runners.openmm.runner import OpenMMRunner

        src = textwrap.dedent(inspect.getsource(OpenMMRunner._peptide_ca_rmsd))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_peptide_ca_rmsd":
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    body = body[1:]
                for stmt in body:
                    for sub in ast.walk(stmt):
                        if isinstance(sub, ast.Call):
                            callee = ast.unparse(sub.func)
                            assert "_pbc_correct" not in callee, (
                                "_peptide_ca_rmsd must not call _pbc_correct "
                                "(OralBiome-AMP#162 + #175). Per-atom PBC unwrap "
                                "fragments rigid bodies across box faces; OpenMM's "
                                "internal unwrapped coords (the default from "
                                "getState) keep molecules continuous across frames "
                                "without needing any explicit PBC math in the gate."
                            )
                return
        pytest.fail("_peptide_ca_rmsd not found via inspect.getsource")

    def test_kabsch_absorbs_rigid_body_translation(self) -> None:
        """Math sanity: Kabsch on receptor gives RMSD≈0 under rigid-body translation.

        Once inputs are in the same coordinate system (which the
        internal-unwrapped-coords convention per #175 guarantees) Kabsch
        fully absorbs any rigid-body translation of the whole complex —
        peptide RMSD should remain at ~0 regardless of how far the
        complex has drifted. This is exactly why OpenMM's default
        (enforcePeriodicBox=False) internal-coord convention is the
        right input for the Kabsch gate: global drift is absorbed by the
        centroid-subtraction step.

        The live-integration check (Flavor C) for catching per-molecule-
        wrapping regressions is tracked in OralBiome-AMP#175.
        """
        import numpy as np

        rng = np.random.default_rng(seed=123)
        ref_rec = rng.uniform(-10.0, 10.0, size=(8, 3))
        ref_pep = rng.uniform(-5.0, 5.0, size=(5, 3))
        positions_ref = np.vstack([ref_rec, ref_pep])
        rec_idx = list(range(8))
        pep_idx = list(range(8, 13))

        for shift in (
            np.array([5.0, 0.0, 0.0]),
            np.array([12.0, 8.0, -6.0]),
            np.array([20.0, 0.0, 15.0]),
        ):
            positions_cur = positions_ref + shift
            sim = _FakeSimulation(positions_cur, np.eye(3) * 100.0)
            rmsd = OpenMMRunner._peptide_ca_rmsd(  # noqa: SLF001
                sim, ref_pep, pep_idx, ref_rec, rec_idx, _FakeUnit(), np
            )
            assert rmsd == pytest.approx(0.0, abs=1e-6), (
                f"Kabsch on receptor must absorb rigid-body translation shift={shift}. "
                f"Got RMSD={rmsd:.6f} Å — if this fails, Kabsch math is broken."
            )


# ---------------------------------------------------------------------------
# Flavor C live-integration regression (placeholder — see OralBiome-AMP#175)
#
# The previous Flavor B test (PR #26, now removed) asserted that the online
# gate's output matched DCDReporter-based offline analysis. Post-#175 that
# invariant no longer holds by design: the online gate uses
# enforcePeriodicBox=False (internal unwrapped) while DCDReporter writes
# enforcePeriodicBox=True (per-molecule wrapped). They intentionally use
# different coordinate conventions.
#
# The replacement (Flavor C) should instead pin the failure mode that
# motivated #175: a complex whose receptor centroid sits near a box face,
# long enough production for thermal motion to push centroids across box
# faces, asserting online gate RMSD ≈ ground-truth unwrapped-Kabsch RMSD
# within 0.5 Å per frame — NOT DCD agreement.
#
# Implementation options per expert consultation 2026-04-21:
#   (a) Deterministic analytical synthetic test (preferred): construct
#       wrapped+unwrapped coordinate pairs manually (receptor at face+δ,
#       peptide bound but in adjacent image) and assert gate math is
#       correct. Fast, deterministic, runs in every CI pass.
#   (b) Deterministic live-MD test: CustomExternalForce applies constant
#       slow drift to receptor CoM, guaranteeing box-face crossing at a
#       predictable step. Assert online RMSD < 7 Å while peptide is
#       restrained in pocket. Slow, run nightly.
#
# Both tracked in OralBiome-AMP#175 task queue.
# ---------------------------------------------------------------------------
