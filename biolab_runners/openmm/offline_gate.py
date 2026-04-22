r"""Offline mdtraj gate for MD early-abort decisions.

OralBiome-AMP task #10: moves the peptide-Cα RMSD gate out of the OpenMM
production callback and into a pure-Python function that reads stored DCD
frames via mdtraj. This eliminates the coordinate-convention bug class
that produced #162, #163, #167, #174, #175 in one week — every failure
mode lived at the interface between OpenMM's ``getState`` positions and
our per-frame PBC arithmetic. The offline path never touches
``getState``; mdtraj's ``unitcell_vectors`` + triclinic-aware unwrap +
battle-tested Kabsch is the only code path.

Architectural role:

- ``DCDReporter`` keeps writing ``trajectory.dcd`` every save-interval
  (default 10 ps). No change.
- Every N ns the runner calls ``evaluate_trajectory`` on the partial
  DCD. The function returns a :class:`GateVerdict` with ``abort=True``
  if the 5 ns dissociation check or the 10 ns conjunctive slope check
  fires.
- The runner writes ``gate_verdict_{current_ns}ns.json`` alongside
  trajectory.dcd so orchestrators (and SIGTERM teardown paths) can
  see the latest verdict.

Invariants preserved from the inside-OpenMM gate:

1. **5 ns early abort**: ``rmsd_5ns > threshold`` → abort
   (semantics match the pre-task-#10 inside-OpenMM
   ``_check_early_abort_5ns`` it replaces).
2. **10 ns conjunctive slope gate**: abort iff BOTH
   ``rmsd_10ns > threshold`` AND least-squares slope over the 5→10 ns
   window > 0.05 Å/ns (OralBiome-AMP#167).
3. **Receptor-Cα Kabsch alignment** (OralBiome-AMP#162).
4. **Triclinic-aware per-molecule unwrap** (OralBiome-AMP#163) — handles
   orthorhombic, triclinic, and dodecahedron cells correctly; the
   pre-#175 online gate fed per-molecule-wrapped coords into Kabsch
   and produced phantom ~50 Å RMSD when receptor/peptide straddled
   different periodic images.
5. **Receptor self-fit residual monitor** — the function returns
   ``receptor_fit_residual`` so callers can warn when ``> 3 Å``. For
   flexible receptors with hinge motion, a future enhancement should
   align on a stable subdomain rather than the full receptor Cα set
   (expert consultation 2026-04-21 caveat #3).

Not in scope for the first cut (VicK + HmuY are rigid enough that the
full-receptor Kabsch residual stays well under 3 Å):

- Subdomain alignment fallback.
- Multi-chain (>2 chain) support. The function assumes
  ``chainid 0`` = receptor, ``chainid 1`` = peptide, matching the
  topology the OpenMM runner emits.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mdtraj as md

logger = logging.getLogger(__name__)

# OralBiome-AMP#167 — conjunctive 10 ns slope gate thresholds.
SLOPE_THRESHOLD_A_PER_NS = 0.05
MIN_SLOPE_WINDOW_NS = 2.0

# Expert consultation 2026-04-21 caveat #3 — warn above this for rigid-body
# Kabsch residual; switch to subdomain alignment if it becomes routine.
RECEPTOR_FIT_RESIDUAL_WARN_A = 3.0

# Gate checkpoint milestones (ns). Default matches OpenMM runner constants.
DEFAULT_GATE_5NS = 5.0
DEFAULT_GATE_10NS = 10.0


@dataclass
class GateVerdict:
    """Verdict returned by :func:`evaluate_trajectory`.

    Attributes:
        abort: True iff the runner should save state and exit immediately.
        reason: ``""`` when ``abort=False``; otherwise one of
            ``"early_dissociation"`` (5 ns gate), ``"rmsd_slope_drift"``
            (10 ns conjunctive gate).
        current_ns: Elapsed simulated time at the last analysed frame.
        rmsd_5ns: Peptide Cα RMSD at the 5 ns checkpoint, or ``None`` if
            the trajectory has not yet reached 5 ns.
        rmsd_10ns: Peptide Cα RMSD at the 10 ns checkpoint, or ``None``.
        max_rmsd: Maximum peptide Cα RMSD observed so far.
        mean_rmsd: Mean peptide Cα RMSD.
        slope_a_per_ns: Least-squares slope over the 5→10 ns window.
            ``None`` if fewer than 2 samples in-window or window < 2 ns.
        receptor_fit_residual: Maximum receptor-Cα self-fit residual
            across frames. Flag a warning above 3 Å (the rigid-body
            Kabsch assumption starts to break down for flexible
            receptors — fall back to subdomain alignment).
        n_frames: Number of frames analysed.
        threshold_a: Abort threshold used.
    """

    abort: bool
    reason: str
    current_ns: float
    rmsd_5ns: float | None
    rmsd_10ns: float | None
    max_rmsd: float
    mean_rmsd: float
    slope_a_per_ns: float | None
    receptor_fit_residual: float
    n_frames: int
    threshold_a: float


def _kabsch_rotation(cur_centered: np.ndarray, ref_centered: np.ndarray) -> np.ndarray:
    """Return the 3×3 proper-rotation matrix that best aligns cur → ref.

    Both inputs must already be centroid-subtracted (N, 3) arrays. Uses the
    SVD formulation with a reflection-guard determinant step so the returned
    matrix is always a proper rotation (det = +1).
    """
    h = cur_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    reflect = np.diag([1.0, 1.0, d])
    return vt.T @ reflect @ u.T


def _unwrap_to_receptor_image(
    mobile_xyz: np.ndarray,
    anchor_xyz: np.ndarray,
    box_vectors: np.ndarray,
) -> np.ndarray:
    """Shift ``mobile_xyz`` as a rigid body to the image closest to ``anchor_xyz``.

    Triclinic-aware: expresses the centroid delta in fractional lattice
    coordinates, rounds each component to the nearest integer, converts
    back to Cartesian. Reduces exactly to the diagonal-only operation for
    orthorhombic boxes while still giving the correct minimum image for
    triclinic or dodecahedron cells (OralBiome-AMP#163).

    Args:
        mobile_xyz: (n, 3) positions to shift, in nm.
        anchor_xyz: (m, 3) positions used to define the target image, in nm.
        box_vectors: (3, 3) lattice vectors as rows — the shape
            ``traj.unitcell_vectors[frame]`` gives directly.

    Returns:
        ``mobile_xyz`` shifted by a whole-box integer combination.
    """
    anchor_c = anchor_xyz.mean(axis=0)
    mobile_c = mobile_xyz.mean(axis=0)
    delta = mobile_c - anchor_c
    inv_box_t = np.linalg.inv(box_vectors.T)
    frac = inv_box_t @ delta
    n = np.round(frac)
    shift = box_vectors.T @ n
    return mobile_xyz - shift


def _compute_per_frame_rmsd(
    traj: md.Trajectory,
    rec_ca_idx: np.ndarray,
    pep_ca_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Receptor-aligned peptide-Cα RMSD + receptor self-fit residual per frame.

    Reference is frame 0 with the peptide unwrapped to the receptor's image.
    Returns arrays in Å.
    """
    if traj.unitcell_vectors is None:
        raise ValueError(
            "trajectory missing unitcell_vectors — the gate's triclinic-aware "
            "unwrap requires periodic box information per frame"
        )
    unitcell_vectors = np.asarray(traj.unitcell_vectors)
    xyz = np.asarray(traj.xyz)
    box_vectors_0 = unitcell_vectors[0]
    ref_rec = xyz[0, rec_ca_idx, :]
    ref_pep_raw = xyz[0, pep_ca_idx, :]
    ref_pep = _unwrap_to_receptor_image(ref_pep_raw, ref_rec, box_vectors_0)
    ref_rec_c = ref_rec.mean(axis=0)

    n_frames = traj.n_frames
    rmsd_ang = np.zeros(n_frames)
    rec_fit_ang = np.zeros(n_frames)

    for i in range(n_frames):
        cur_rec = xyz[i, rec_ca_idx, :]
        cur_pep_raw = xyz[i, pep_ca_idx, :]
        box_vectors = unitcell_vectors[i]

        cur_pep = _unwrap_to_receptor_image(cur_pep_raw, cur_rec, box_vectors)

        cur_rec_c = cur_rec.mean(axis=0)
        r = _kabsch_rotation(cur_rec - cur_rec_c, ref_rec - ref_rec_c)
        rec_aligned = (cur_rec - cur_rec_c) @ r.T + ref_rec_c
        pep_aligned = (cur_pep - cur_rec_c) @ r.T + ref_rec_c

        # Whole-molecule fold of any residual image mismatch after rotation.
        diff_raw = pep_aligned - ref_pep
        diff_c = diff_raw.mean(axis=0)
        inv_box_t = np.linalg.inv(box_vectors.T)
        n_img = np.round(inv_box_t @ diff_c)
        image_shift = box_vectors.T @ n_img
        pep_folded = pep_aligned - image_shift
        final_diff = pep_folded - ref_pep

        rmsd_ang[i] = float(np.sqrt((final_diff**2).sum(axis=1).mean())) * 10.0
        rec_fit = rec_aligned - ref_rec
        rec_fit_ang[i] = float(np.sqrt((rec_fit**2).sum(axis=1).mean())) * 10.0

    return rmsd_ang, rec_fit_ang


def _frame_interval_ns(rep_dir: Path, default_ns: float = 0.010) -> float:
    """Read save interval from ``system_config.json``; default 10 ps.

    Does NOT trust ``md_summary.json``'s ``total_ns`` — that records the
    *planned* duration, which is wrong for early-aborted runs. Reads
    ``simulation.save_interval_ps`` (preferred) or falls back to
    ``save_every_steps * timestep_fs`` for legacy configs.
    """
    cfg_path = rep_dir / "system_config.json"
    if not cfg_path.exists():
        return default_ns
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return default_ns
    sim_cfg = cfg.get("simulation", {})
    save_interval_ps = sim_cfg.get("save_interval_ps")
    if save_interval_ps is None:
        save_every_steps = sim_cfg.get("save_every_steps")
        timestep_fs = sim_cfg.get("timestep_fs")
        if save_every_steps and timestep_fs:
            save_interval_ps = save_every_steps * timestep_fs / 1e3
    if save_interval_ps:
        return float(save_interval_ps) / 1e3
    return default_ns


def evaluate_trajectory(
    rep_dir: Path | str,
    *,
    threshold_a: float = 7.0,
    gate_5ns: float = DEFAULT_GATE_5NS,
    gate_10ns: float = DEFAULT_GATE_10NS,
    slope_threshold_a_per_ns: float = SLOPE_THRESHOLD_A_PER_NS,
    min_slope_window_ns: float = MIN_SLOPE_WINDOW_NS,
) -> GateVerdict:
    """Compute a :class:`GateVerdict` from the DCD + topology at ``rep_dir``.

    Reads ``trajectory.dcd`` and ``topology.pdb`` from ``rep_dir``, computes
    per-frame receptor-aligned peptide-Cα RMSD with triclinic-aware unwrap,
    and emits a verdict that replicates the OpenMM runner's online gate
    semantics:

    - 5 ns gate: ``rmsd_at_5ns > threshold_a`` → ``abort`` with
      reason ``"early_dissociation"``.
    - 10 ns conjunctive slope gate: requires BOTH
      ``rmsd_at_10ns > threshold_a`` AND
      ``slope over 5→10 ns > slope_threshold_a_per_ns``
      (OralBiome-AMP#167).

    The 5 ns check fires first; the 10 ns check only runs if the 5 ns check
    passed (matches the inside-OpenMM runner's short-circuit behaviour).

    Args:
        rep_dir: Replicate directory containing ``trajectory.dcd``,
            ``topology.pdb``, and optionally ``system_config.json``
            (for the trajectory save interval).
        threshold_a: Abort threshold in Å (default 7.0).
        gate_5ns: Time in ns at which the 5 ns dissociation check fires.
        gate_10ns: Time in ns at which the 10 ns slope check fires.
        slope_threshold_a_per_ns: Minimum drift rate for slope-gate abort.
        min_slope_window_ns: Minimum 5→10 ns regression window for slope
            validity. If fewer samples / shorter window, the slope gate
            is skipped (too noisy to trust).

    Returns:
        A :class:`GateVerdict` populated with all observed metrics. Call
        :func:`write_verdict_file` to persist next to the trajectory.

    Raises:
        FileNotFoundError: If ``trajectory.dcd`` or ``topology.pdb`` is
            missing from ``rep_dir``.
    """
    # Imported lazily — mdtraj is a heavy dep and we want the module to
    # import cheaply so the runner can keep its cold-start fast.
    import mdtraj as md

    rep_dir = Path(rep_dir)
    traj_path = rep_dir / "trajectory.dcd"
    top_path = rep_dir / "topology.pdb"
    if not traj_path.exists() or not top_path.exists():
        raise FileNotFoundError(f"missing trajectory.dcd or topology.pdb in {rep_dir}")

    traj = md.load(str(traj_path), top=str(top_path))
    if traj.n_frames == 0:
        return GateVerdict(
            abort=False,
            reason="",
            current_ns=0.0,
            rmsd_5ns=None,
            rmsd_10ns=None,
            max_rmsd=0.0,
            mean_rmsd=0.0,
            slope_a_per_ns=None,
            receptor_fit_residual=0.0,
            n_frames=0,
            threshold_a=threshold_a,
        )

    rec_ca_idx = traj.topology.select("chainid 0 and name CA")
    pep_ca_idx = traj.topology.select("chainid 1 and name CA")
    if rec_ca_idx.size == 0 or pep_ca_idx.size == 0:
        raise ValueError(f"topology at {top_path} missing receptor or peptide Cα atoms")

    rmsd_ang, rec_fit_ang = _compute_per_frame_rmsd(traj, rec_ca_idx, pep_ca_idx)
    frame_interval_ns = _frame_interval_ns(rep_dir)
    time_ns = np.arange(1, traj.n_frames + 1) * frame_interval_ns

    current_ns = float(time_ns[-1])
    max_rmsd = float(rmsd_ang.max())
    mean_rmsd = float(rmsd_ang.mean())
    rec_fit_max = float(rec_fit_ang.max())

    rmsd_5ns = _sample_at(rmsd_ang, time_ns, gate_5ns)
    rmsd_10ns = _sample_at(rmsd_ang, time_ns, gate_10ns)

    slope = _slope_5ns_to_10ns(rmsd_ang, time_ns, gate_5ns, gate_10ns, min_slope_window_ns)

    abort, reason = _decide(
        rmsd_5ns=rmsd_5ns,
        rmsd_10ns=rmsd_10ns,
        slope=slope,
        threshold_a=threshold_a,
        slope_threshold=slope_threshold_a_per_ns,
    )

    if rec_fit_max > RECEPTOR_FIT_RESIDUAL_WARN_A:
        logger.warning(
            "receptor self-fit residual %.2f Å > %.1f Å — rigid-body Kabsch "
            "assumption degraded; consider subdomain alignment (future enhancement)",
            rec_fit_max,
            RECEPTOR_FIT_RESIDUAL_WARN_A,
        )

    return GateVerdict(
        abort=abort,
        reason=reason,
        current_ns=current_ns,
        rmsd_5ns=rmsd_5ns,
        rmsd_10ns=rmsd_10ns,
        max_rmsd=max_rmsd,
        mean_rmsd=mean_rmsd,
        slope_a_per_ns=slope,
        receptor_fit_residual=rec_fit_max,
        n_frames=traj.n_frames,
        threshold_a=threshold_a,
    )


def _sample_at(rmsd_ang: np.ndarray, time_ns: np.ndarray, t_target_ns: float) -> float | None:
    """Nearest-frame sample of ``rmsd_ang`` at ``t_target_ns``.

    Returns ``None`` if the trajectory has not reached ``t_target_ns`` yet.
    """
    if time_ns[-1] < t_target_ns:
        return None
    idx = int(np.argmin(np.abs(time_ns - t_target_ns)))
    return float(rmsd_ang[idx])


def _slope_5ns_to_10ns(
    rmsd_ang: np.ndarray,
    time_ns: np.ndarray,
    t_lo_ns: float,
    t_hi_ns: float,
    min_window_ns: float,
) -> float | None:
    """Least-squares slope of RMSD over ``[t_lo_ns, t_hi_ns]``.

    Returns ``None`` if fewer than 2 in-window samples or if the window
    span is shorter than ``min_window_ns``. The inside-OpenMM gate sampled
    the window at sub-chunk granularity (~17 points); the offline gate
    sees every DCD frame in that window (~500 points at 10 ps save
    interval), giving a dramatically more stable regression.
    """
    if time_ns[-1] < t_hi_ns:
        return None
    mask = (time_ns >= t_lo_ns) & (time_ns <= t_hi_ns)
    xs = time_ns[mask]
    ys = rmsd_ang[mask]
    if xs.size < 2 or (xs[-1] - xs[0]) < min_window_ns:
        return None
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def _decide(
    *,
    rmsd_5ns: float | None,
    rmsd_10ns: float | None,
    slope: float | None,
    threshold_a: float,
    slope_threshold: float,
) -> tuple[bool, str]:
    """Apply the two-gate abort logic. Returns ``(abort, reason)``."""
    if rmsd_5ns is not None and rmsd_5ns > threshold_a:
        return True, "early_dissociation"
    if (
        rmsd_10ns is not None
        and rmsd_10ns > threshold_a
        and slope is not None
        and slope > slope_threshold
    ):
        return True, "rmsd_slope_drift"
    return False, ""


def write_verdict_file(verdict: GateVerdict, rep_dir: Path | str) -> Path:
    """Persist ``verdict`` as ``gate_verdict_{current_ns:.1f}ns.json``.

    The filename stamps the ``current_ns`` so concurrent writers at
    different checkpoints don't clobber one another and readers can pick
    the latest by filename sort.
    """
    rep_dir = Path(rep_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    path = rep_dir / f"gate_verdict_{verdict.current_ns:.1f}ns.json"
    path.write_text(json.dumps(asdict(verdict), indent=2))
    return path


def latest_verdict_file(rep_dir: Path | str) -> Path | None:
    """Return the most recent ``gate_verdict_*.json`` in ``rep_dir``, or None."""
    rep_dir = Path(rep_dir)
    candidates = sorted(
        rep_dir.glob("gate_verdict_*ns.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def load_verdict(path: Path | str) -> GateVerdict:
    """Load a persisted verdict JSON back into a :class:`GateVerdict`."""
    data = json.loads(Path(path).read_text())
    return GateVerdict(**data)
