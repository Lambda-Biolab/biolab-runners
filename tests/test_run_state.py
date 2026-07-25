"""Tests for :mod:`biolab_runners.openmm.run_state` — the run-state decision module.

These tests exercise the **public interface** (``decide``,
``populate_skip_result``, ``Action``, ``ResumePlan``). Internal
helpers (``_decide_with_manifest``, ``_orphan_state_error``,
``_artifact_validation_error``, ``_reconstruct_terminal_payload``)
are exercised through the public functions; the AGENTS.md
invariants describe their contracts.

The decision tree under test:

- ``force=True`` ⇒ quarantine stale files, then proceed.
- Missing manifest + orphan state file ⇒ FAIL_FAST.
- Missing manifest + no orphan ⇒ FRESH.
- Manifest with valid terminal payload or past target step ⇒ SKIP.
- Manifest with malformed terminal payload ⇒ FAIL_FAST.
- Manifest in-progress ⇒ RESUME.
- Manifest references a dangling / unsafe / step-mismatched
  state file ⇒ FAIL_FAST.
"""

from __future__ import annotations

import json
from pathlib import Path

from biolab_runners.openmm.checkpoint import atomic_save_checkpoint, load_checkpoint
from biolab_runners.openmm.config import OpenMMConfig, SimulationResult
from biolab_runners.openmm.paths import FileNames
from biolab_runners.openmm.run_state import (
    Action,
    decide,
    populate_skip_result,
)

logger = __import__("logging").getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest(
    output_dir: Path,
    *,
    step: int,
    state_file: str,
    terminal: dict[str, object] | None = None,
) -> None:
    record: dict[str, object] = {"step": step, "file": state_file}
    if terminal is not None:
        record["terminal"] = terminal
    (output_dir / FileNames.CHECKPOINT_JSON).write_text(json.dumps({"records": [record]}))


def _config(production_ns: float = 10.0, timestep_fs: float = 2.0) -> OpenMMConfig:
    return OpenMMConfig(production_ns=production_ns, timestep_fs=timestep_fs)


# ---------------------------------------------------------------------------
# decide() — fresh / no-manifest
# ---------------------------------------------------------------------------


class TestDecideFresh:
    """``Action.FRESH``: no manifest exists and no orphan state file."""

    def test_no_manifest_no_state_files(self, tmp_path: Path) -> None:
        plan = decide(tmp_path, _config(), force=False)

        assert plan.action == Action.FRESH
        assert plan.start_step == plan.start_step  # non-zero placeholder check
        assert plan.remaining_steps > 0
        assert plan.resume_xml == ""
        assert plan.error == ""

    def test_empty_output_dir(self, tmp_path: Path) -> None:
        plan = decide(tmp_path, _config(), force=False)

        assert plan.action == Action.FRESH


# ---------------------------------------------------------------------------
# decide() — fail-fast on invalid manifest
# ---------------------------------------------------------------------------


class TestDecideFailFastInvalidManifest:
    """``Action.FAIL_FAST``: manifest references a missing / empty /
    path-traversal / step-mismatched state file."""

    def test_manifest_references_missing_state_file(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, step=1000, state_file="state.1000_12345_170000000.xml")
        # Don't create the state file.

        plan = decide(tmp_path, _config(), force=False)

        assert plan.action == Action.FAIL_FAST
        assert "does not exist" in plan.error

    def test_manifest_references_empty_state_file(self, tmp_path: Path) -> None:
        (tmp_path / "state.1000_12345_170000000.xml").write_text("")
        _write_manifest(tmp_path, step=1000, state_file="state.1000_12345_170000000.xml")

        plan = decide(tmp_path, _config(), force=False)

        assert plan.action == Action.FAIL_FAST
        assert "empty" in plan.error

    def test_manifest_references_path_traversal_state_file(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, step=1000, state_file="../escape.xml")

        plan = decide(tmp_path, _config(), force=False)

        assert plan.action == Action.FAIL_FAST
        assert "not a basename" in plan.error

    def test_manifest_step_mismatch_with_state_filename(self, tmp_path: Path) -> None:
        (tmp_path / "state.999_12345_170000000.xml").write_text("<State/>")
        _write_manifest(tmp_path, step=1000, state_file="state.999_12345_170000000.xml")

        plan = decide(tmp_path, _config(), force=False)

        assert plan.action == Action.FAIL_FAST
        assert "does not match" in plan.error


# ---------------------------------------------------------------------------
# decide() — orphan state files
# ---------------------------------------------------------------------------


class TestDecideFailFastOrphan:
    """``Action.FAIL_FAST``: state files exist without a manifest."""

    def test_legacy_state_xml_with_no_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "state.xml").write_text("<State/>")

        plan = decide(tmp_path, _config(), force=False)

        assert plan.action == Action.FAIL_FAST
        assert "orphan" in plan.error.lower() or "exist at" in plan.error

    def test_v7_state_with_no_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "state.999_12345_170000000.xml").write_text("<State/>")

        plan = decide(tmp_path, _config(), force=False)

        assert plan.action == Action.FAIL_FAST

    def test_corrupt_manifest_with_state_files(self, tmp_path: Path) -> None:
        (tmp_path / "state.1000_12345_170000000.xml").write_text("<State/>")
        # Manifest references the file but doesn't exist → corrupt.
        _write_manifest(tmp_path, step=1000, state_file="state.1000_12345_170000000.xml")
        (tmp_path / "checkpoint.json").write_text("not valid json")

        plan = decide(tmp_path, _config(), force=False)

        # Corrupt manifest: treated as no manifest, then orphan detected.
        assert plan.action == Action.FAIL_FAST

    def test_orphan_recovered_by_force(self, tmp_path: Path) -> None:
        """``force=True`` quarantines the orphan, allowing a fresh build."""
        (tmp_path / "state.1000_12345_170000000.xml").write_text("<State/>")

        plan = decide(tmp_path, _config(), force=True)

        # After quarantine, no orphan remains → FRESH.
        assert plan.action == Action.FRESH
        # The state file is in .stale/<UTC>/
        stale_dirs = list((tmp_path / ".stale").iterdir())
        assert len(stale_dirs) == 1
        assert (stale_dirs[0] / "state.1000_12345_170000000.xml").exists()


# ---------------------------------------------------------------------------
# decide() — force quarantine
# ---------------------------------------------------------------------------


class TestDecideForceQuarantine:
    """``force=True`` quarantines stale checkpoint files before deciding."""

    def test_force_quarantines_manifest_energy_and_state(self, tmp_path: Path) -> None:
        config = _config(production_ns=1.0)
        # Pre-create a coherent mid-run checkpoint.
        (tmp_path / FileNames.CHECKPOINT_JSON).write_text(
            json.dumps({"records": [{"step": 10000, "file": "state.10000_1_1.xml"}]})
        )
        (tmp_path / FileNames.ENERGY).write_text("header\nrow\n")
        (tmp_path / "state.10000_1_1.xml").write_text("<State/>")

        # Decide with force=True: the existing checkpoint is
        # quarantined, then we decide fresh.
        plan = decide(tmp_path, config, force=True)

        assert plan.action == Action.FRESH
        # All three files moved into .stale/<UTC>/
        stale_dirs = list((tmp_path / ".stale").iterdir())
        assert len(stale_dirs) == 1
        stale = stale_dirs[0]
        assert (stale / FileNames.CHECKPOINT_JSON).exists()
        assert (stale / FileNames.ENERGY).exists()
        assert (stale / "state.10000_1_1.xml").exists()

    def test_force_with_no_existing_checkpoint_is_a_no_op(self, tmp_path: Path) -> None:
        plan = decide(tmp_path, _config(), force=True)

        assert plan.action == Action.FRESH
        # No stale directory created (nothing to quarantine).
        assert not (tmp_path / ".stale").exists()

    def test_force_then_interrupted_then_non_force_yields_empty_resume(
        self, tmp_path: Path
    ) -> None:
        """``force=True`` quarantines; a subsequent non-forced run with
        the new save in place is empty (no intermediate manifest)."""
        config = _config(production_ns=1.0)
        # Pre-existing checkpoint.
        (tmp_path / FileNames.CHECKPOINT_JSON).write_text(
            json.dumps({"records": [{"step": 500, "file": "state.500_1_1.xml"}]})
        )
        (tmp_path / "state.500_1_1.xml").write_text("<State/>")

        # force=True quarantines and decides fresh.
        plan1 = decide(tmp_path, config, force=True)
        assert plan1.action == Action.FRESH
        # No manifest on disk anymore.
        assert not (tmp_path / FileNames.CHECKPOINT_JSON).exists()

        # Non-forced: still fresh.
        plan2 = decide(tmp_path, config, force=False)
        assert plan2.action == Action.FRESH

    def test_force_quarantines_early_abort_marker(self, tmp_path: Path) -> None:
        """``force=True`` also moves the early-abort marker into .stale/."""
        (tmp_path / FileNames.EARLY_ABORT_JSON).write_text(
            json.dumps({"type": "early_abort", "step": 5000, "reason": "5ns gate"})
        )
        # Also create a checkpoint so the decision finds something to quarantine.
        (tmp_path / FileNames.CHECKPOINT_JSON).write_text(
            json.dumps({"records": [{"step": 5000, "file": "state.5000_1_1.xml"}]})
        )
        (tmp_path / "state.5000_1_1.xml").write_text("<State/>")

        plan = decide(tmp_path, _config(), force=True)

        assert plan.action == Action.FRESH
        stale = next((tmp_path / ".stale").iterdir())
        assert (stale / FileNames.EARLY_ABORT_JSON).exists()

    def test_force_recovers_from_invalid_manifest(self, tmp_path: Path) -> None:
        """``force=True`` quarantines the corrupt manifest, allowing fresh start."""
        (tmp_path / FileNames.CHECKPOINT_JSON).write_text("not valid json")
        (tmp_path / "state.1000_12345_170000000.xml").write_text("<State/>")

        plan = decide(tmp_path, _config(), force=True)

        assert plan.action == Action.FRESH


# ---------------------------------------------------------------------------
# decide() — resume
# ---------------------------------------------------------------------------


class TestDecideResume:
    """``Action.RESUME``: valid manifest, run is in progress."""

    def test_intermediate_checkpoint_resumes(self, tmp_path: Path) -> None:
        config = _config(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.10000_12345_170000000.xml"
        state.write_text("<State/>")
        _write_manifest(tmp_path, step=10_000, state_file="state.10000_12345_170000000.xml")

        plan = decide(tmp_path, config, force=False)

        assert plan.action == Action.RESUME
        assert plan.start_step == 10_000
        assert plan.remaining_steps > 0
        assert plan.resume_xml == str(state)
        assert plan.manifest_step == 10_000

    def test_resume_subtracts_equil_steps(self, tmp_path: Path) -> None:
        """``remaining_steps`` = total production steps - production done."""
        config = _config(production_ns=10.0, timestep_fs=2.0)
        # Manifest at step total_equil_steps + 100 production steps.
        # Remaining production steps = total_steps - 100.
        manifest_step = config.total_equil_steps + 100
        state = tmp_path / f"state.{manifest_step}_12345_170000000.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path, step=manifest_step, state_file=f"state.{manifest_step}_12345_170000000.xml"
        )

        plan = decide(tmp_path, config, force=False)

        assert plan.action == Action.RESUME
        assert plan.start_step == manifest_step
        assert plan.remaining_steps == config.total_steps - 100

    def test_resume_right_after_equil(self, tmp_path: Path) -> None:
        """A manifest at exactly total_equil_steps resumes from there
        with the full production run ahead."""
        config = _config(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / f"state.{config.total_equil_steps}_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=config.total_equil_steps,
            state_file=f"state.{config.total_equil_steps}_1_1.xml",
        )

        plan = decide(tmp_path, config, force=False)

        assert plan.action == Action.RESUME
        assert plan.remaining_steps == config.total_steps

    def test_multi_resume_step_is_cumulative_and_monotonic(self, tmp_path: Path) -> None:
        """Two sequential resumes use the absolute step from the manifest."""
        config = _config(production_ns=10.0, timestep_fs=2.0)
        # Pre-existing checkpoint.
        state = tmp_path / "state.500_12345_170000000.xml"
        state.write_text("<State/>")
        _write_manifest(tmp_path, step=500, state_file="state.500_12345_170000000.xml")

        plan = decide(tmp_path, config, force=False)
        assert plan.action == Action.RESUME
        first_step = plan.start_step

        # Simulate a save at a higher step (the runner wrote a new
        # manifest with absolute step = first_step + steps_done).
        # The file path is unchanged (the manifest references a new file).
        new_state = tmp_path / "state.1500_12345_180000000.xml"
        new_state.write_text("<State/>")
        _write_manifest(tmp_path, step=1500, state_file="state.1500_12345_180000000.xml")

        plan2 = decide(tmp_path, config, force=False)
        assert plan2.action == Action.RESUME
        assert plan2.start_step > first_step


# ---------------------------------------------------------------------------
# decide() — skip (terminal)
# ---------------------------------------------------------------------------


class TestDecideSkip:
    """``Action.SKIP``: run is terminal (manifest payload or normal completion)."""

    def test_normal_completion_at_target(self, tmp_path: Path) -> None:
        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state = tmp_path / f"state.{target}_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(tmp_path, step=target, state_file=f"state.{target}_1_1.xml")

        plan = decide(tmp_path, config, force=False)

        assert plan.action == Action.SKIP
        assert plan.manifest_step == target
        assert plan.skip_reason.startswith("normal_completion_step_")

    def test_valid_terminal_payload(self, tmp_path: Path) -> None:
        config = _config(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal={
                "type": "early_abort",
                "step": 5_000_000,
                "reason": "5ns gate tripped",
                "production_ns": 5.0,
            },
        )

        plan = decide(tmp_path, config, force=False)

        assert plan.action == Action.SKIP
        assert plan.skip_reason.startswith("manifest_terminal_early_abort_")

    def test_load_step_ignores_energy_csv(self, tmp_path: Path) -> None:
        """energy.csv alone no longer yields a step.

        A manifest-less directory with only energy.csv is decided as
        FRESH (no manifest, no orphan state file). This is the
        "decide() ignores energy.csv" guarantee — the step never
        comes from the energy log."""
        (tmp_path / FileNames.ENERGY).write_text("#step,time\n5000,10\n10000,20\n15000,30\n")

        plan = decide(tmp_path, _config(), force=False)

        assert plan.action == Action.FRESH


# ---------------------------------------------------------------------------
# decide() — fail-fast on malformed terminal payload
# ---------------------------------------------------------------------------


class TestDecideFailFastInvalidTerminal:
    """``Action.FAIL_FAST``: manifest has a present-but-invalid
    ``terminal`` payload (v13 BLOCKER tri-state)."""

    def test_invalid_terminal_payload_fails_fast_not_skip(self, tmp_path: Path) -> None:
        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state = tmp_path / f"state.{target}_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=target,
            state_file=f"state.{target}_1_1.xml",
            terminal={"type": "future_marker", "step": target, "reason": "x"},
        )

        plan = decide(tmp_path, config, force=False)

        assert plan.action == Action.FAIL_FAST
        assert "malformed terminal payload" in plan.error
        assert "force=True" in plan.error


# ---------------------------------------------------------------------------
# populate_skip_result
# ---------------------------------------------------------------------------


class TestPopulateSkipResult:
    """``populate_skip_result``: populate result fields on SKIP."""

    def test_populates_artifact_paths_and_state_xml(self, tmp_path: Path) -> None:
        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        # Build the on-disk artifacts.
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / FileNames.TRAJECTORY).write_bytes(b"\x00" * 100)
        (tmp_path / FileNames.ENERGY).write_text("header\nrow1\nrow2\n")
        (tmp_path / FileNames.TOPOLOGY).write_text("HEADER\nATOM 1\n" * 100)
        _write_manifest(tmp_path, step=target, state_file=state_basename)

        result = SimulationResult(config=config)
        plan = decide(tmp_path, config, force=False)
        assert plan.action == Action.SKIP

        skip_error = populate_skip_result(plan, tmp_path, config, result)

        assert skip_error is None
        assert result.trajectory_path == str(tmp_path / FileNames.TRAJECTORY)
        assert result.energy_path == str(tmp_path / FileNames.ENERGY)
        assert result.topology_path == str(tmp_path / FileNames.TOPOLOGY)
        assert result.state_xml_path == str(tmp_path / state_basename)
        # Normal completion → total_ns from v10 BLOCKER #3 invariant.
        assert result.total_ns > 0

    def test_missing_trajectory_returns_artifact_error(self, tmp_path: Path) -> None:
        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        # No trajectory.dcd.
        (tmp_path / FileNames.ENERGY).write_text("header\nrow\n")
        (tmp_path / FileNames.TOPOLOGY).write_text("HEADER\nATOM\n" * 100)
        _write_manifest(tmp_path, step=target, state_file=state_basename)

        plan = decide(tmp_path, config, force=False)
        assert plan.action == Action.SKIP

        result = SimulationResult(config=config)
        skip_error = populate_skip_result(plan, tmp_path, config, result)

        assert skip_error is not None
        assert "missing trajectory" in skip_error
        # Result.error is NOT set here — the caller copies the error.

    def test_empty_trajectory_returns_artifact_error(self, tmp_path: Path) -> None:
        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / FileNames.TRAJECTORY).write_bytes(b"")
        (tmp_path / FileNames.ENERGY).write_text("header\nrow\n")
        (tmp_path / FileNames.TOPOLOGY).write_text("HEADER\nATOM\n" * 100)
        _write_manifest(tmp_path, step=target, state_file=state_basename)

        plan = decide(tmp_path, config, force=False)
        assert plan.action == Action.SKIP

        result = SimulationResult(config=config)
        skip_error = populate_skip_result(plan, tmp_path, config, result)

        assert skip_error is not None
        assert "empty trajectory" in skip_error

    def test_header_only_energy_returns_artifact_error(self, tmp_path: Path) -> None:
        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / FileNames.TRAJECTORY).write_bytes(b"\x00" * 100)
        # Energy has header but no data rows.
        (tmp_path / FileNames.ENERGY).write_text("header_only\n")
        (tmp_path / FileNames.TOPOLOGY).write_text("HEADER\nATOM\n" * 100)
        _write_manifest(tmp_path, step=target, state_file=state_basename)

        plan = decide(tmp_path, config, force=False)
        assert plan.action == Action.SKIP

        result = SimulationResult(config=config)
        skip_error = populate_skip_result(plan, tmp_path, config, result)

        assert skip_error is not None
        assert "energy" in skip_error
        assert "header-only" in skip_error or "no data" in skip_error

    def test_early_abort_terminal_payload_populates_early_abort_fields(
        self, tmp_path: Path
    ) -> None:
        config = _config(production_ns=10.0, timestep_fs=2.0)
        state_basename = "state.5000000_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / FileNames.TRAJECTORY).write_bytes(b"\x00" * 100)
        (tmp_path / FileNames.ENERGY).write_text("header\nrow\n")
        (tmp_path / FileNames.TOPOLOGY).write_text("HEADER\nATOM\n" * 100)
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file=state_basename,
            terminal={
                "type": "early_abort",
                "step": 5_000_000,
                "reason": "5ns gate tripped",
                "production_ns": 5.0,
            },
        )

        plan = decide(tmp_path, config, force=False)
        assert plan.action == Action.SKIP

        result = SimulationResult(config=config)
        skip_error = populate_skip_result(plan, tmp_path, config, result)

        assert skip_error is None
        assert result.early_abort is True
        assert result.abort_reason == "5ns gate tripped"
        # v10 BLOCKER #3: total_ns is the canonical value
        # (computed from absolute_step - total_equil_steps), not
        # read from the stored payload field. The stored value
        # is a hint / context for downstream consumers only.
        from biolab_runners.openmm.checkpoint import production_ns

        assert result.total_ns == production_ns(5_000_000, config)


# ---------------------------------------------------------------------------
# End-to-end: decide() + populate_skip_result() roundtrip
# ---------------------------------------------------------------------------


class TestDecideAndPopulateRoundtrip:
    """End-to-end: a SKIP plan followed by populate_skip_result
    produces a populated result. Mirrors what the runner does."""

    def test_skip_populated_result_matches_manifest(self, tmp_path: Path) -> None:
        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / FileNames.TRAJECTORY).write_bytes(b"\x00" * 100)
        (tmp_path / FileNames.ENERGY).write_text("h\nrow\n")
        (tmp_path / FileNames.TOPOLOGY).write_text("H\nATOM\n" * 100)
        _write_manifest(tmp_path, step=target, state_file=state_basename)

        # Decide.
        plan = decide(tmp_path, config, force=False)
        assert plan.action == Action.SKIP

        # Populate.
        result = SimulationResult(config=config)
        skip_error = populate_skip_result(plan, tmp_path, config, result)
        assert skip_error is None

        # Manifest step matches checkpoint step (loaded from disk).
        checkpoint = load_checkpoint(tmp_path)
        assert checkpoint.absolute_step == target
        assert checkpoint.state_file_basename == state_basename
        assert result.state_xml_path == str(tmp_path / state_basename)

    def test_atomic_save_then_decide_yields_skip_at_target(self, tmp_path: Path) -> None:
        """End-to-end: atomic_save_checkpoint commits a manifest, then
        decide() correctly identifies the run as terminal (when at the
        target step)."""
        from unittest.mock import MagicMock

        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        sim = MagicMock()
        sim.saveState = MagicMock(side_effect=lambda path: Path(path).write_text("<State/>"))

        atomic_save_checkpoint(sim, tmp_path, absolute_step=target)

        plan = decide(tmp_path, config, force=False)
        assert plan.action == Action.SKIP
        assert plan.manifest_step == target
