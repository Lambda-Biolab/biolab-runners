"""Tests for ``biolab_runners.openmm.offline_gate``.

Ports the invariants the deleted inside-OpenMM gate tests used to pin
(TestPeptideCaRmsdReceptorAligned, TestRegressionSlope,
TestCheckSlope10nsConjunctiveGate, TestGateCoordConventionRegression,
TestFlavorCCoordConventionMath, TestFlavorCGateMatchesIndependentKabschLiveMD).
See the "Historical context" banner in test_openmm_runner.py.

Fixture strategy: synthesise an mdtraj.Trajectory in memory for each test,
save it as a DCD + topology.pdb in ``tmp_path``, and invoke
``evaluate_trajectory(tmp_path)``. The fake topology has chain 0 = receptor
(10 Cα) and chain 1 = peptide (3 Cα), matching the real pipeline's
``chainid 0`` / ``chainid 1`` selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mdtraj", reason="offline_gate requires mdtraj")

from biolab_runners.openmm.offline_gate import (  # noqa: E402
    GateVerdict,
    _decide,
    _frame_interval_ns,
    _kabsch_rotation,
    _sample_at,
    _slope_5ns_to_10ns,
    _unwrap_to_receptor_image,
    evaluate_trajectory,
    latest_verdict_file,
    load_verdict,
    write_verdict_file,
)

# Save interval that keeps tests quick but lets the 5 ns + 10 ns gates fire.
# 10 ps matches the production default — but tests generate ~1000 frames
# rather than ~2000 to keep fixtures small.
SAVE_INTERVAL_PS = 10.0
SAVE_INTERVAL_NS = SAVE_INTERVAL_PS / 1000.0


# ---------------------------------------------------------------------------
# Fixture builders


def _build_topology() -> object:
    """Two-chain topology: 10 receptor Cα + 3 peptide Cα, all CA-named.

    Ensures ``traj.topology.select("chainid 0 and name CA")`` returns 10
    indices and ``chainid 1`` returns 3 — matching production.
    """
    import mdtraj as md

    top = md.Topology()
    # Receptor chain
    rec_chain = top.add_chain()
    for i in range(10):
        res = top.add_residue(f"ALA{i}", rec_chain)
        top.add_atom("CA", md.element.carbon, res)
    # Peptide chain
    pep_chain = top.add_chain()
    for i in range(3):
        res = top.add_residue(f"GLY{i}", pep_chain)
        top.add_atom("CA", md.element.carbon, res)
    return top


def _receptor_frame() -> np.ndarray:
    """10-Cα α-helix-ish receptor layout (nm)."""
    return np.array(
        [[np.cos(i * 1.0), np.sin(i * 1.0), i * 0.15] for i in range(10)],
        dtype=float,
    )


def _peptide_frame() -> np.ndarray:
    """3-Cα peptide in the receptor's "pocket" (nm)."""
    return np.array([[0.30, 0.0, 0.20], [0.35, 0.05, 0.35], [0.30, 0.10, 0.50]], dtype=float)


def _orthorhombic_box(size_nm: float = 5.78) -> np.ndarray:
    """5.78 nm = typical production water-box size."""
    return np.eye(3) * size_nm


def _write_trajectory(
    tmp_path: Path,
    frames: np.ndarray,
    box_vectors: np.ndarray,
    save_interval_ps: float = SAVE_INTERVAL_PS,
) -> Path:
    """Save frames + topology + system_config.json to ``tmp_path``.

    ``frames`` shape: (n_frames, n_atoms, 3) in nm.
    ``box_vectors`` shape: (n_frames, 3, 3) or (3, 3) broadcast.
    Returns ``tmp_path`` (the "replicate directory").
    """
    import mdtraj as md

    top = _build_topology()
    if box_vectors.ndim == 2:
        box_vectors = np.broadcast_to(box_vectors, (frames.shape[0], 3, 3)).copy()

    traj = md.Trajectory(xyz=frames, topology=top)
    traj.unitcell_vectors = box_vectors
    traj.save_dcd(str(tmp_path / "trajectory.dcd"))

    # Write a topology.pdb (mdtraj requires it) — use frame 0 as reference.
    single = md.Trajectory(xyz=frames[:1], topology=top)
    single.unitcell_vectors = box_vectors[:1]
    single.save_pdb(str(tmp_path / "topology.pdb"))

    # Write a minimal system_config.json so _frame_interval_ns picks up
    # the save interval. Without it, the gate defaults to 10 ps (which
    # matches SAVE_INTERVAL_PS anyway but we want to exercise the code path).
    cfg = {"simulation": {"save_interval_ps": save_interval_ps, "timestep_fs": 2.0}}
    (tmp_path / "system_config.json").write_text(json.dumps(cfg))
    return tmp_path


def _make_trajectory(
    n_frames: int,
    peptide_drift_a_per_frame: float = 0.0,
    rotation_angle_per_frame: float = 0.0,
    global_translation_per_frame: float = 0.0,
    box_size_nm: float = 5.78,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct ``n_frames`` of receptor + peptide with optional perturbations.

    All perturbations accumulate linearly across frames, so the final frame
    has n_frames * per-frame drift. Receptor rotates as a rigid body (Kabsch
    must absorb this). Peptide drifts in the receptor frame (this is what
    the gate must catch).

    Returns ``(frames, box_vectors)`` arrays for ``_write_trajectory``.
    """
    rec0 = _receptor_frame()
    pep0 = _peptide_frame()
    n_rec = rec0.shape[0]
    n_pep = pep0.shape[0]
    n_atoms = n_rec + n_pep

    # Peptide drift: accumulate along +x (nm) per frame.
    # Convert Å/frame to nm/frame (1 Å = 0.1 nm).
    drift_per_frame_nm = peptide_drift_a_per_frame * 0.1

    frames = np.zeros((n_frames, n_atoms, 3), dtype=float)
    for i in range(n_frames):
        theta = rotation_angle_per_frame * i
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([global_translation_per_frame * i, 0.0, 0.0])
        rec_i = rec0 @ rot.T + translation
        pep_drift = np.array([drift_per_frame_nm * i, 0.0, 0.0])
        pep_i = pep0 @ rot.T + translation + pep_drift
        frames[i, :n_rec, :] = rec_i
        frames[i, n_rec:, :] = pep_i

    box = _orthorhombic_box(box_size_nm)
    box_vectors = np.broadcast_to(box, (n_frames, 3, 3)).copy()
    return frames, box_vectors


# ---------------------------------------------------------------------------
# Unit tests for the private math helpers


class TestKabschRotation:
    """``_kabsch_rotation`` — proper SVD Kabsch (replaces deleted inside-OpenMM version)."""

    def test_identity_returns_identity(self) -> None:
        pts = np.random.default_rng(0).normal(size=(10, 3))
        r = _kabsch_rotation(pts - pts.mean(0), pts - pts.mean(0))
        assert np.allclose(r, np.eye(3), atol=1e-10)

    def test_90deg_rotation_recovered(self) -> None:
        pts = np.random.default_rng(1).normal(size=(10, 3))
        pts_c = pts - pts.mean(0)
        rot_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        rotated = pts_c @ rot_true.T
        r = _kabsch_rotation(rotated, pts_c)
        # Kabsch fits r such that rotated @ r.T ≈ pts_c, which inverts rot_true.
        recovered = rotated @ r.T
        assert np.allclose(recovered, pts_c, atol=1e-10)

    def test_always_proper_rotation(self) -> None:
        """Reflection guard — det must be +1 even for reflected inputs."""
        pts = np.random.default_rng(2).normal(size=(10, 3))
        reflected = pts.copy()
        reflected[:, 0] *= -1
        r = _kabsch_rotation(reflected - reflected.mean(0), pts - pts.mean(0))
        assert np.linalg.det(r) > 0


class TestUnwrapToReceptorImage:
    """``_unwrap_to_receptor_image`` — triclinic-aware whole-molecule PBC unwrap."""

    def test_already_same_image_no_shift(self) -> None:
        box = _orthorhombic_box(5.0)
        rec = np.array([[2.5, 2.5, 2.5]])
        pep = np.array([[3.0, 3.0, 3.0]])
        unwrapped = _unwrap_to_receptor_image(pep, rec, box)
        assert np.allclose(unwrapped, pep, atol=1e-10)

    def test_peptide_one_box_away_gets_folded(self) -> None:
        """Peptide centroid 1 box-length away → whole-molecule shift back."""
        box_size = 5.0
        box = _orthorhombic_box(box_size)
        rec = np.array([[2.5, 2.5, 2.5]])
        pep_wrapped = np.array([[3.0 + box_size, 3.0, 3.0]])
        unwrapped = _unwrap_to_receptor_image(pep_wrapped, rec, box)
        expected = np.array([[3.0, 3.0, 3.0]])
        assert np.allclose(unwrapped, expected, atol=1e-10)

    def test_dodecahedron_diagonal_face(self) -> None:
        """OralBiome-AMP#163: triclinic dodecahedron box — diagonal face unwraps correctly.

        The pre-#163 diagonal-only formula dropped the off-diagonal
        lattice components on rhombic dodecahedron cells, causing a
        physically bound peptide that crossed the xy-diagonal face to
        report a large displacement. The fractional-lattice formula
        used here must handle this correctly.
        """
        import math

        d = 6.0
        box = np.array(
            [
                [d, 0.0, 0.0],
                [0.0, d, 0.0],
                [0.5 * d, 0.5 * d, d / math.sqrt(2.0)],
            ]
        )
        rec = np.array([[0.0, 0.0, 0.0]])
        # Peptide one "c" lattice vector away + small offset
        pep_wrapped = box[2].reshape(1, 3) + np.array([[0.03, 0.03, 0.01]])
        unwrapped = _unwrap_to_receptor_image(pep_wrapped, rec, box)
        # After unwrap: should collapse to the small-offset image.
        assert np.linalg.norm(unwrapped) < 0.1  # ≈ 0.044 nm


class TestSampleAtAndSlope:
    """``_sample_at`` and ``_slope_5ns_to_10ns`` helpers."""

    def test_sample_at_before_target_returns_none(self) -> None:
        rmsd = np.array([1.0, 2.0, 3.0])
        time_ns = np.array([0.01, 0.02, 0.03])
        assert _sample_at(rmsd, time_ns, 1.0) is None

    def test_sample_at_nearest_frame(self) -> None:
        rmsd = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        time_ns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _sample_at(rmsd, time_ns, 3.0) == 3.0
        assert _sample_at(rmsd, time_ns, 3.4) == 3.0  # nearest
        assert _sample_at(rmsd, time_ns, 3.6) == 4.0

    def test_slope_below_min_window_returns_none(self) -> None:
        """OralBiome-AMP#167 — 2 ns minimum regression window."""
        time_ns = np.array([5.0, 5.5, 6.0])  # only 1 ns window
        rmsd = np.array([3.0, 3.5, 4.0])
        assert _slope_5ns_to_10ns(rmsd, time_ns, 5.0, 10.0, min_window_ns=2.0) is None

    def test_slope_converges_to_true_slope(self) -> None:
        """Least-squares over many samples tracks true drift, not endpoint noise."""
        time_ns = np.linspace(5.0, 10.0, 50)
        # True slope = 0.1 Å/ns + normal noise
        rng = np.random.default_rng(7)
        rmsd = 2.0 + 0.1 * (time_ns - 5.0) + rng.normal(0, 0.05, size=time_ns.size)
        slope = _slope_5ns_to_10ns(rmsd, time_ns, 5.0, 10.0, min_window_ns=2.0)
        assert slope is not None
        assert abs(slope - 0.1) < 0.02


class TestDecide:
    """Two-gate abort logic — matches the pre-task-#10 inside-OpenMM semantics."""

    # Used everywhere below
    THRESHOLD = 7.0
    SLOPE = 0.05

    def test_all_none_no_abort(self) -> None:
        abort, reason = _decide(
            rmsd_5ns=None,
            rmsd_10ns=None,
            slope=None,
            threshold_a=self.THRESHOLD,
            slope_threshold=self.SLOPE,
        )
        assert abort is False
        assert reason == ""

    def test_5ns_dissociation_fires(self) -> None:
        abort, reason = _decide(
            rmsd_5ns=8.0,
            rmsd_10ns=None,
            slope=None,
            threshold_a=self.THRESHOLD,
            slope_threshold=self.SLOPE,
        )
        assert abort is True
        assert reason == "early_dissociation"

    def test_5ns_below_threshold_no_abort(self) -> None:
        abort, reason = _decide(
            rmsd_5ns=5.0,
            rmsd_10ns=None,
            slope=None,
            threshold_a=self.THRESHOLD,
            slope_threshold=self.SLOPE,
        )
        assert abort is False
        assert reason == ""

    def test_10ns_conjunctive_requires_both(self) -> None:
        """OralBiome-AMP#167: abort ONLY when both abs > threshold AND slope > 0.05."""
        # RMSD > threshold but slope tame → no abort (thermal fluctuation).
        abort, _ = _decide(
            rmsd_5ns=6.0,
            rmsd_10ns=7.5,
            slope=0.01,
            threshold_a=self.THRESHOLD,
            slope_threshold=self.SLOPE,
        )
        assert abort is False

        # RMSD below threshold but slope > 0.05 → no abort (bound-and-breathing).
        abort, _ = _decide(
            rmsd_5ns=6.0,
            rmsd_10ns=5.0,
            slope=0.1,
            threshold_a=self.THRESHOLD,
            slope_threshold=self.SLOPE,
        )
        assert abort is False

        # Both exceed → abort with the slope-drift reason.
        abort, reason = _decide(
            rmsd_5ns=6.0,
            rmsd_10ns=8.0,
            slope=0.1,
            threshold_a=self.THRESHOLD,
            slope_threshold=self.SLOPE,
        )
        assert abort is True
        assert reason == "rmsd_slope_drift"

    def test_vick_abc021a3_ground_truth_passes(self) -> None:
        """Re-verification ground truth — VicK_g2_abc021a3_cstr at 10.2 ns passed
        under the post-#175 gate with max 3.82 Å. The conjunctive gate must not
        false-abort it.
        """
        # Approximate the observed trace: 5ns=3.0, 10ns=3.5, slope=0.1
        abort, _ = _decide(
            rmsd_5ns=3.0,
            rmsd_10ns=3.5,
            slope=0.1,
            threshold_a=self.THRESHOLD,
            slope_threshold=self.SLOPE,
        )
        assert abort is False


class TestFrameIntervalNs:
    """``_frame_interval_ns`` reads save_interval_ps from system_config.json."""

    def test_default_when_missing(self, tmp_path: Path) -> None:
        assert _frame_interval_ns(tmp_path) == 0.010  # 10 ps

    def test_reads_save_interval_ps(self, tmp_path: Path) -> None:
        cfg = {"simulation": {"save_interval_ps": 25.0}}
        (tmp_path / "system_config.json").write_text(json.dumps(cfg))
        assert _frame_interval_ns(tmp_path) == 0.025

    def test_legacy_save_every_steps(self, tmp_path: Path) -> None:
        cfg = {"simulation": {"save_every_steps": 5000, "timestep_fs": 2.0}}
        (tmp_path / "system_config.json").write_text(json.dumps(cfg))
        assert _frame_interval_ns(tmp_path) == pytest.approx(0.010)


# ---------------------------------------------------------------------------
# Integration tests — evaluate_trajectory on real mdtraj trajectories


class TestEvaluateTrajectoryReceptorAligned:
    """Receptor-aligned RMSD invariants (OralBiome-AMP#162).

    Replaces the deleted ``TestPeptideCaRmsdReceptorAligned``.
    """

    def test_identical_frames_give_zero_rmsd(self, tmp_path: Path) -> None:
        frames, box = _make_trajectory(n_frames=600)  # 6 ns
        _write_trajectory(tmp_path, frames, box)
        verdict = evaluate_trajectory(tmp_path, threshold_a=7.0)
        assert verdict.max_rmsd < 0.01
        assert verdict.abort is False

    def test_rigid_body_translation_absorbed(self, tmp_path: Path) -> None:
        """Receptor + peptide move together → Kabsch absorbs → RMSD ≈ 0."""
        frames, box = _make_trajectory(n_frames=600, global_translation_per_frame=0.001)
        _write_trajectory(tmp_path, frames, box)
        verdict = evaluate_trajectory(tmp_path, threshold_a=7.0)
        assert verdict.max_rmsd < 0.01

    def test_rigid_body_rotation_absorbed(self, tmp_path: Path) -> None:
        """Complex rotates as a rigid body → Kabsch absorbs → RMSD ≈ 0."""
        frames, box = _make_trajectory(n_frames=600, rotation_angle_per_frame=0.002)
        _write_trajectory(tmp_path, frames, box)
        verdict = evaluate_trajectory(tmp_path, threshold_a=7.0)
        assert verdict.max_rmsd < 0.01

    def test_receptor_rotation_plus_peptide_drift(self, tmp_path: Path) -> None:
        """Peptide drifts ~5 Å in receptor frame while complex tumbles → RMSD ~5 Å."""
        # 5 Å total over 600 frames
        frames, box = _make_trajectory(
            n_frames=600,
            rotation_angle_per_frame=0.001,
            peptide_drift_a_per_frame=5.0 / 600.0,
        )
        _write_trajectory(tmp_path, frames, box)
        verdict = evaluate_trajectory(tmp_path, threshold_a=7.0)
        # Final peptide drift magnitude is what the gate must detect.
        assert 4.0 < verdict.max_rmsd < 6.0
        # Below threshold so no abort.
        assert verdict.abort is False


class TestEvaluateTrajectoryGateLogic:
    """End-to-end gate-abort decisions. Replaces ``TestCheckSlope10nsConjunctiveGate``."""

    def test_5ns_dissociation_aborts(self, tmp_path: Path) -> None:
        """Peptide drifts 10 Å by 5 ns → early_dissociation."""
        # 10 Å over 500 frames = 5 ns at 10 ps/frame
        frames, box = _make_trajectory(
            n_frames=1001,  # 10.01 ns — past 10 ns gate
            peptide_drift_a_per_frame=10.0 / 500.0,
        )
        _write_trajectory(tmp_path, frames, box)
        verdict = evaluate_trajectory(tmp_path, threshold_a=7.0)
        assert verdict.abort is True
        assert verdict.reason == "early_dissociation"
        assert verdict.rmsd_5ns is not None
        assert verdict.rmsd_5ns > 7.0

    def test_stable_binding_no_abort(self, tmp_path: Path) -> None:
        """Peptide drifts 2 Å over 15 ns → well below threshold, no abort."""
        frames, box = _make_trajectory(
            n_frames=1500,
            peptide_drift_a_per_frame=2.0 / 1500.0,
        )
        _write_trajectory(tmp_path, frames, box)
        verdict = evaluate_trajectory(tmp_path, threshold_a=7.0)
        assert verdict.abort is False
        assert verdict.max_rmsd < 3.0

    def test_short_trajectory_no_5ns_sample(self, tmp_path: Path) -> None:
        """Under 5 ns of simulation → gate can't fire yet."""
        frames, box = _make_trajectory(n_frames=200)  # 2 ns
        _write_trajectory(tmp_path, frames, box)
        verdict = evaluate_trajectory(tmp_path, threshold_a=7.0)
        assert verdict.rmsd_5ns is None
        assert verdict.rmsd_10ns is None
        assert verdict.abort is False


class TestHistoricalBrokenPath:
    """Regression-history (OralBiome-AMP#175): documents the bug that's now gone.

    The pre-#175 online gate fed per-molecule-wrapped coords into Kabsch,
    producing phantom ~50 Å RMSD when receptor and peptide straddled different
    periodic images. The offline gate closes this bug class by doing
    triclinic-aware whole-molecule unwrap BEFORE Kabsch.

    This test manually reproduces what the broken-path math would have produced
    and asserts the new offline gate does NOT reproduce the phantom value.
    Kept as anti-regression evidence — if someone re-introduces per-atom
    wrapping in the unwrap helper, this test catches it.
    """

    def test_wrapped_peptide_gets_unwrapped_not_phantomed(self, tmp_path: Path) -> None:
        """Peptide in receptor pocket but its xyz is ~1 box-length away → must unwrap."""
        rec = _receptor_frame()
        pep = _peptide_frame()
        box_size = 5.78
        box = _orthorhombic_box(box_size)

        # Frame 0: everything in the natural image.
        # Frames 1..N: peptide gets "wrapped" — its xyz is shifted by one
        # box-length along x, simulating the #174 pathology.
        n_frames = 600
        frames = np.zeros((n_frames, rec.shape[0] + pep.shape[0], 3), dtype=float)
        for i in range(n_frames):
            frames[i, : rec.shape[0], :] = rec
            if i == 0:
                frames[i, rec.shape[0] :, :] = pep
            else:
                # Wrap into the -x image — exactly the pathology #174 exhibited.
                frames[i, rec.shape[0] :, :] = pep + np.array([box_size, 0.0, 0.0])

        box_vectors = np.broadcast_to(box, (n_frames, 3, 3)).copy()
        _write_trajectory(tmp_path, frames, box_vectors)

        verdict = evaluate_trajectory(tmp_path, threshold_a=7.0)
        # Post-#175 fix: the offline gate's per-frame whole-molecule unwrap
        # re-images the peptide back to the receptor's image → RMSD ≈ 0.
        # Pre-fix: would have reported ~57.8 Å (one box-length).
        assert verdict.max_rmsd < 1.0, (
            f"Offline gate must unwrap whole-molecule image shifts. "
            f"Got max RMSD {verdict.max_rmsd:.2f} Å — did someone reintroduce "
            f"per-atom wrapping or remove the unwrap step?"
        )


class TestVerdictFileIO:
    """``write_verdict_file`` / ``latest_verdict_file`` / ``load_verdict``."""

    def _make_verdict(self, current_ns: float = 5.0, abort: bool = False) -> GateVerdict:
        return GateVerdict(
            abort=abort,
            reason="" if not abort else "early_dissociation",
            current_ns=current_ns,
            rmsd_5ns=2.1 if current_ns >= 5.0 else None,
            rmsd_10ns=2.5 if current_ns >= 10.0 else None,
            max_rmsd=2.5,
            mean_rmsd=1.8,
            slope_a_per_ns=0.02 if current_ns >= 10.0 else None,
            receptor_fit_residual=1.5,
            n_frames=500,
            threshold_a=7.0,
        )

    def test_write_and_load_roundtrip(self, tmp_path: Path) -> None:
        verdict = self._make_verdict(current_ns=5.2)
        path = write_verdict_file(verdict, tmp_path)
        assert path.name == "gate_verdict_5.2ns.json"
        loaded = load_verdict(path)
        assert loaded == verdict

    def test_latest_verdict_picks_newest(self, tmp_path: Path) -> None:
        import time

        v1 = self._make_verdict(current_ns=5.0)
        write_verdict_file(v1, tmp_path)
        time.sleep(0.05)
        v2 = self._make_verdict(current_ns=10.0)
        write_verdict_file(v2, tmp_path)

        latest = latest_verdict_file(tmp_path)
        assert latest is not None
        assert latest.name == "gate_verdict_10.0ns.json"

    def test_latest_verdict_empty_dir(self, tmp_path: Path) -> None:
        assert latest_verdict_file(tmp_path) is None
