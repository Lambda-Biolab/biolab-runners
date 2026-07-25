"""Tests for :mod:`biolab_runners.openmm.run_state` — the run-state decision module.

These tests exercise the **public interface** (``decide``,
:class:`RunPlan`, the four plan types, :class:`Action`). Internal
helpers (``_populate_skip_plan_fields``, ``_orphan_state_error``,
``_artifact_validation_error``) are exercised through the public
functions.

The decision tree under test:

- ``force=True`` ⇒ quarantine stale files, then proceed.
- Missing manifest + orphan state file ⇒ ``FailurePlan``.
- Missing manifest + no orphan ⇒ ``FreshPlan``.
- Manifest with valid terminal payload or past target step ⇒ ``SkipPlan``.
- Manifest with malformed terminal payload ⇒ ``FailurePlan``.
- Manifest in-progress ⇒ ``ResumePlan``.
- Manifest references a dangling / unsafe / step-mismatched
  state file ⇒ ``FailurePlan``.

Each plan is one of four distinct frozen dataclasses; the runner
matches on either the type or ``plan.action``. Invalid constructions
are unrepresentable (a ``FreshPlan`` cannot carry a ``resume_xml``,
a ``FailurePlan`` cannot carry a ``start_step``).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from biolab_runners.openmm.checkpoint import (
    CompletionStatus,
    atomic_save_checkpoint,
)
from biolab_runners.openmm.config import OpenMMConfig
from biolab_runners.openmm.paths import FileNames
from biolab_runners.openmm.run_state import (
    Action,
    FailurePlan,
    FreshPlan,
    ResumePlan,
    SkipPlan,
    decide,
)

logger = __import__("logging").getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sentinel for distinguishing "key not present" from "key present with None".
# The manifest's ``terminal`` field may legally be absent (no decision
# recorded) OR present-but-null (an explicit terminal decision that is
# malformed). The reviewer flagged this distinction as a contract
# bug — see AGENTS.md / run_state tests.
_SENTINEL = object()


def _write_manifest(
    output_dir: Path,
    *,
    step: int,
    state_file: str,
    terminal: dict[str, object] | None | object = _SENTINEL,
) -> None:
    """Write a manifest record.

    The default (``_SENTINEL``) omits the ``terminal`` key entirely.
    Passing ``None`` explicitly writes ``"terminal": null`` — the
    reviewer requires us to distinguish absent key from null value.
    """
    record: dict[str, object] = {"step": step, "file": state_file}
    if terminal is not _SENTINEL:
        record["terminal"] = terminal
    (output_dir / FileNames.CHECKPOINT_JSON).write_text(json.dumps({"records": [record]}))


def _config(production_ns: float = 10.0, timestep_fs: float = 2.0) -> OpenMMConfig:
    return OpenMMConfig(production_ns=production_ns, timestep_fs=timestep_fs)


# ---------------------------------------------------------------------------
# decide() — fresh / no-manifest
# ---------------------------------------------------------------------------


class TestDecideFresh:
    """``FreshPlan``: no manifest exists and no orphan state file."""

    def test_no_manifest_no_state_files(self, tmp_path: Path) -> None:
        config = _config()
        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, FreshPlan)
        assert plan.action == Action.FRESH
        # Exact values — a tautological test (e.g. ``plan.start_step ==
        # plan.start_step``) would not catch a regression where the
        # equil offset is wrong.
        assert plan.start_step == config.total_equil_steps
        assert plan.remaining_steps == config.total_steps

    def test_empty_output_dir(self, tmp_path: Path) -> None:
        plan = decide(tmp_path, _config(), force=False)

        assert isinstance(plan, FreshPlan)
        assert plan.action == Action.FRESH


# ---------------------------------------------------------------------------
# decide() — fail-fast on invalid manifest
# ---------------------------------------------------------------------------


class TestDecideFailFastInvalidManifest:
    """``FailurePlan``: manifest references a missing / empty /
    path-traversal / step-mismatched state file."""

    def test_manifest_references_missing_state_file(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, step=1000, state_file="state.1000_12345_170000000.xml")
        # Don't create the state file.

        plan = decide(tmp_path, _config(), force=False)

        assert isinstance(plan, FailurePlan)
        assert plan.action == Action.FAIL_FAST
        assert "does not exist" in plan.error

    def test_manifest_references_empty_state_file(self, tmp_path: Path) -> None:
        (tmp_path / "state.1000_12345_170000000.xml").write_text("")
        _write_manifest(tmp_path, step=1000, state_file="state.1000_12345_170000000.xml")

        plan = decide(tmp_path, _config(), force=False)

        assert isinstance(plan, FailurePlan)
        assert "empty" in plan.error

    def test_manifest_references_path_traversal_state_file(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, step=1000, state_file="../escape.xml")

        plan = decide(tmp_path, _config(), force=False)

        assert isinstance(plan, FailurePlan)
        assert "not a basename" in plan.error

    def test_manifest_step_mismatch_with_state_filename(self, tmp_path: Path) -> None:
        (tmp_path / "state.999_12345_170000000.xml").write_text("<State/>")
        _write_manifest(tmp_path, step=1000, state_file="state.999_12345_170000000.xml")

        plan = decide(tmp_path, _config(), force=False)

        assert isinstance(plan, FailurePlan)
        assert "does not match" in plan.error


# ---------------------------------------------------------------------------
# decide() — orphan state files
# ---------------------------------------------------------------------------


class TestDecideFailFastOrphan:
    """``FailurePlan``: state files exist without a manifest."""

    def test_legacy_state_xml_with_no_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "state.xml").write_text("<State/>")

        plan = decide(tmp_path, _config(), force=False)

        assert isinstance(plan, FailurePlan)
        assert "orphan" in plan.error.lower() or "exist at" in plan.error

    def test_v7_state_with_no_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "state.999_12345_170000000.xml").write_text("<State/>")

        plan = decide(tmp_path, _config(), force=False)

        assert isinstance(plan, FailurePlan)

    def test_corrupt_manifest_with_state_files(self, tmp_path: Path) -> None:
        (tmp_path / "state.1000_12345_170000000.xml").write_text("<State/>")
        # Manifest references the file but doesn't exist → corrupt.
        _write_manifest(tmp_path, step=1000, state_file="state.1000_12345_170000000.xml")
        (tmp_path / "checkpoint.json").write_text("not valid json")

        plan = decide(tmp_path, _config(), force=False)

        # Corrupt manifest: treated as no manifest, then orphan detected.
        assert isinstance(plan, FailurePlan)

    def test_orphan_recovered_by_force(self, tmp_path: Path) -> None:
        """``force=True`` quarantines the orphan, allowing a fresh build."""
        (tmp_path / "state.1000_12345_170000000.xml").write_text("<State/>")

        plan = decide(tmp_path, _config(), force=True)

        # After quarantine, no orphan remains → FreshPlan.
        assert isinstance(plan, FreshPlan)
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

        assert isinstance(plan, FreshPlan)
        # All three files moved into .stale/<UTC>/
        stale_dirs = list((tmp_path / ".stale").iterdir())
        assert len(stale_dirs) == 1
        stale = stale_dirs[0]
        assert (stale / FileNames.CHECKPOINT_JSON).exists()
        assert (stale / FileNames.ENERGY).exists()
        assert (stale / "state.10000_1_1.xml").exists()

    def test_force_with_no_existing_checkpoint_is_a_no_op(self, tmp_path: Path) -> None:
        plan = decide(tmp_path, _config(), force=True)

        assert isinstance(plan, FreshPlan)
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
        assert isinstance(plan1, FreshPlan)
        # No manifest on disk anymore.
        assert not (tmp_path / FileNames.CHECKPOINT_JSON).exists()

        # Non-forced: still fresh.
        plan2 = decide(tmp_path, config, force=False)
        assert isinstance(plan2, FreshPlan)

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

        assert isinstance(plan, FreshPlan)
        stale = next((tmp_path / ".stale").iterdir())
        assert (stale / FileNames.EARLY_ABORT_JSON).exists()

    def test_force_recovers_from_invalid_manifest(self, tmp_path: Path) -> None:
        """``force=True`` quarantines the corrupt manifest, allowing fresh start."""
        (tmp_path / FileNames.CHECKPOINT_JSON).write_text("not valid json")
        (tmp_path / "state.1000_12345_170000000.xml").write_text("<State/>")

        plan = decide(tmp_path, _config(), force=True)

        assert isinstance(plan, FreshPlan)


# ---------------------------------------------------------------------------
# decide() — resume
# ---------------------------------------------------------------------------


class TestDecideResume:
    """``ResumePlan``: valid manifest, run is in progress."""

    def test_intermediate_checkpoint_resumes(self, tmp_path: Path) -> None:
        config = _config(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.10000_12345_170000000.xml"
        state.write_text("<State/>")
        _write_manifest(tmp_path, step=10_000, state_file="state.10000_12345_170000000.xml")

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, ResumePlan)
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

        assert isinstance(plan, ResumePlan)
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

        assert isinstance(plan, ResumePlan)
        assert plan.remaining_steps == config.total_steps

    def test_multi_resume_step_is_cumulative_and_monotonic(self, tmp_path: Path) -> None:
        """Two sequential resumes use the absolute step from the manifest."""
        config = _config(production_ns=10.0, timestep_fs=2.0)
        # Pre-existing checkpoint.
        state = tmp_path / "state.500_12345_170000000.xml"
        state.write_text("<State/>")
        _write_manifest(tmp_path, step=500, state_file="state.500_12345_170000000.xml")

        plan = decide(tmp_path, config, force=False)
        assert isinstance(plan, ResumePlan)
        first_step = plan.start_step

        # Simulate a save at a higher step (the runner wrote a new
        # manifest with absolute step = first_step + steps_done).
        # The file path is unchanged (the manifest references a new file).
        new_state = tmp_path / "state.1500_12345_180000000.xml"
        new_state.write_text("<State/>")
        _write_manifest(tmp_path, step=1500, state_file="state.1500_12345_180000000.xml")

        plan2 = decide(tmp_path, config, force=False)
        assert isinstance(plan2, ResumePlan)
        assert plan2.start_step > first_step


# ---------------------------------------------------------------------------
# decide() — skip (terminal)
# ---------------------------------------------------------------------------


class TestDecideSkip:
    """``SkipPlan``: run is terminal (manifest payload or normal completion)."""

    def test_normal_completion_at_target(self, tmp_path: Path) -> None:
        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state = tmp_path / f"state.{target}_1_1.xml"
        state.write_text("<State/>")
        # All artifacts present.
        (tmp_path / FileNames.TRAJECTORY).write_bytes(b"\x00" * 100)
        (tmp_path / FileNames.ENERGY).write_text("header\nrow\n")
        (tmp_path / FileNames.TOPOLOGY).write_text("HEADER\nATOM\n" * 100)
        _write_manifest(tmp_path, step=target, state_file=f"state.{target}_1_1.xml")

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, SkipPlan)
        assert plan.completion == CompletionStatus.NORMAL_COMPLETE
        assert plan.manifest_step == target
        assert plan.completion_reason.startswith("normal_completion_step_")
        # Artifact paths populated; total_ns from v10 BLOCKER #3 invariant.
        assert plan.trajectory_path == str(tmp_path / FileNames.TRAJECTORY)
        assert plan.energy_path == str(tmp_path / FileNames.ENERGY)
        assert plan.topology_path == str(tmp_path / FileNames.TOPOLOGY)
        assert plan.state_xml_path == str(tmp_path / f"state.{target}_1_1.xml")
        # Exact rounded value, not just > 0. The reviewer flagged
        # that the previous implementation did ``round(..., 2)`` and
        # the test should verify the rounding contract.
        assert plan.total_ns == round(config.total_steps * config.timestep_fs / 1e6, 2)
        assert plan.early_abort is False
        assert plan.abort_reason is None

    def test_normal_completion_total_ns_rounded_for_float_timestep(self, tmp_path: Path) -> None:
        """Float timestep: total_ns is the exact-rounded value (2 dp),
        not the raw float. Without ``round()``, floating-point artifacts
        in the multiplication would leak through.

        With production_ns=2.5 and timestep_fs=2.5, total_steps is
        computed as ``int(2.5 * 1000 * (1000/2.5)) = 1_000_000`` and
        ``total_ns = 1_000_000 * 2.5 / 1e6 = 2.5``. After ``round(_, 2)``
        the value is exactly ``2.5`` — the rounding contract holds.
        """
        config = OpenMMConfig(production_ns=2.5, timestep_fs=2.5)
        target = config.total_equil_steps + config.total_steps
        state = tmp_path / f"state.{target}_1_1.xml"
        state.write_text("<State/>")
        (tmp_path / FileNames.TRAJECTORY).write_bytes(b"\x00" * 100)
        (tmp_path / FileNames.ENERGY).write_text("header\nrow\n")
        (tmp_path / FileNames.TOPOLOGY).write_text("HEADER\nATOM\n" * 100)
        _write_manifest(tmp_path, step=target, state_file=f"state.{target}_1_1.xml")

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, SkipPlan)
        assert plan.completion == CompletionStatus.NORMAL_COMPLETE
        # 1_000_000 steps × 2.5 fs = 2.5 ns. The round(_, 2) is a
        # no-op for this exact value, but the assertion verifies the
        # public surface emits a 2-decimal value.
        assert plan.total_ns == round(config.total_steps * config.timestep_fs / 1e6, 2)
        assert plan.total_ns == 2.5

    def test_valid_terminal_payload(self, tmp_path: Path) -> None:
        config = _config(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        (tmp_path / FileNames.TRAJECTORY).write_bytes(b"\x00" * 100)
        (tmp_path / FileNames.ENERGY).write_text("header\nrow\n")
        (tmp_path / FileNames.TOPOLOGY).write_text("HEADER\nATOM\n" * 100)
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

        assert isinstance(plan, SkipPlan)
        assert plan.completion == CompletionStatus.EARLY_ABORT
        assert plan.completion_reason.startswith("manifest_terminal_early_abort_")
        assert plan.early_abort is True
        assert plan.abort_reason == "5ns gate tripped"

    def test_explicit_terminal_payload_at_normal_target(self, tmp_path: Path) -> None:
        """v12 BLOCKER: explicit terminal at the normal target wins
        over inferred normal completion."""
        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state = tmp_path / f"state.{target}_1_1.xml"
        state.write_text("<State/>")
        (tmp_path / FileNames.TRAJECTORY).write_bytes(b"\x00" * 100)
        (tmp_path / FileNames.ENERGY).write_text("h\nrow\n")
        (tmp_path / FileNames.TOPOLOGY).write_text("H\nATOM\n" * 100)
        _write_manifest(
            tmp_path,
            step=target,
            state_file=f"state.{target}_1_1.xml",
            terminal={
                "type": "early_abort",
                "step": target,
                "reason": "10ns gate tripped",
                "production_ns": 10.0,
            },
        )

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, SkipPlan)
        assert plan.completion == CompletionStatus.EARLY_ABORT  # NOT normal
        assert plan.early_abort is True

    def test_load_step_ignores_energy_csv(self, tmp_path: Path) -> None:
        """energy.csv alone no longer yields a step.

        A manifest-less directory with only energy.csv is decided
        as FreshPlan (no manifest, no orphan state file). This is
        the "decide() ignores energy.csv" guarantee — the step
        never comes from the energy log."""
        (tmp_path / FileNames.ENERGY).write_text("#step,time\n5000,10\n10000,20\n15000,30\n")

        plan = decide(tmp_path, _config(), force=False)

        assert isinstance(plan, FreshPlan)


# ---------------------------------------------------------------------------
# decide() — fail-fast on malformed terminal payload
# ---------------------------------------------------------------------------


class TestDecideFailFastInvalidTerminal:
    """``FailurePlan``: manifest has a present-but-invalid
    ``terminal`` payload. Per the v11 contract, this must fail fast
    — never resume, never fall back to normal completion."""

    def test_invalid_terminal_payload_at_normal_target(self, tmp_path: Path) -> None:
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

        assert isinstance(plan, FailurePlan)
        assert "malformed terminal payload" in plan.error
        assert "force=True" in plan.error

    def test_invalid_terminal_below_target_also_fails_fast(self, tmp_path: Path) -> None:
        """A malformed terminal at any step — not just at the target
        — must fail fast. The previous implementation's "treat as
        in-progress with invalid_terminal_<reason>" was incorrect
        for a target-step manifest (silently reclassifying). Below
        the target it would also be incorrect because the v11
        contract says treat as in-progress with a specific reason."""
        config = _config(production_ns=10.0, timestep_fs=2.0)
        # Mid-production step, well below the target.
        mid_step = config.total_equil_steps + 100
        state = tmp_path / f"state.{mid_step}_12345_170000000.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=mid_step,
            state_file=f"state.{mid_step}_12345_170000000.xml",
            terminal={"type": "future_marker", "step": mid_step, "reason": "x"},
        )

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, FailurePlan)
        assert "malformed terminal payload" in plan.error

    def test_invalid_terminal_not_a_dict(self, tmp_path: Path) -> None:
        config = _config(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal="not a dict",  # type: ignore[arg-type]
        )

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, FailurePlan)
        assert "invalid_terminal_not_dict" in plan.error

    def test_invalid_terminal_step_string(self, tmp_path: Path) -> None:
        config = _config(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal={"type": "early_abort", "step": "5000000", "reason": "x"},
        )

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, FailurePlan)
        assert "invalid_terminal_step_invalid_type" in plan.error

    def test_invalid_terminal_step_mismatch(self, tmp_path: Path) -> None:
        config = _config(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal={"type": "early_abort", "step": 9_999_999, "reason": "x"},
        )

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, FailurePlan)
        assert "invalid_terminal_step_mismatch" in plan.error

    def test_invalid_terminal_empty_reason(self, tmp_path: Path) -> None:
        config = _config(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal={"type": "early_abort", "step": 5_000_000, "reason": ""},
        )

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, FailurePlan)
        assert "invalid_terminal_reason_empty" in plan.error

    def test_invalid_terminal_unknown_type(self, tmp_path: Path) -> None:
        config = _config(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal={"type": "future_marker", "step": 5_000_000, "reason": "x"},
        )

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, FailurePlan)
        assert "invalid_terminal_type_unsupported" in plan.error

    def test_terminal_null_at_target_fails_fast(self, tmp_path: Path) -> None:
        """``"terminal": null`` is INVALID — distinct from the key
        being absent. At the target step, the absence of the key
        would be a valid normal completion, but the explicit null
        value is an explicit terminal decision that fails the
        schema. The runner must fail fast."""
        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state = tmp_path / f"state.{target}_1_1.xml"
        state.write_text("<State/>")
        # Explicit ``terminal: null`` — the key is present but the
        # value is None. This is the case the previous code got
        # wrong: ``last_record.get("terminal")`` returned None for
        # both the absent key and the null value, so the manifest
        # was silently treated as normal completion.
        _write_manifest(
            tmp_path,
            step=target,
            state_file=f"state.{target}_1_1.xml",
            terminal=None,
        )

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, FailurePlan)
        assert "invalid_terminal_null" in plan.error
        assert "force=True" in plan.error

    def test_terminal_null_below_target_fails_fast(self, tmp_path: Path) -> None:
        """``"terminal": null`` below the target also fails fast —
        the rule is independent of the step, since any explicit
        terminal decision at any step that fails schema is invalid."""
        config = _config(production_ns=10.0, timestep_fs=2.0)
        mid_step = config.total_equil_steps + 100
        state = tmp_path / f"state.{mid_step}_12345_170000000.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=mid_step,
            state_file=f"state.{mid_step}_12345_170000000.xml",
            terminal=None,
        )

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, FailurePlan)
        assert "invalid_terminal_null" in plan.error


# ---------------------------------------------------------------------------
# Skip plan artifact validation
# ---------------------------------------------------------------------------


class TestSkipPlanArtifactValidation:
    """``SkipPlan`` with a missing or empty artifact surfaces as
    ``FailurePlan`` (not ``SkipPlan`` with truncated paths)."""

    def _setup_skip_ready(self, tmp_path: Path) -> tuple[Path, int]:
        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state_basename = f"state.{target}_1_1.xml"
        (tmp_path / state_basename).write_text("<State/>")
        (tmp_path / FileNames.TRAJECTORY).write_bytes(b"\x00" * 100)
        (tmp_path / FileNames.ENERGY).write_text("header\nrow1\nrow2\n")
        (tmp_path / FileNames.TOPOLOGY).write_text("HEADER\nATOM 1\n" * 100)
        _write_manifest(tmp_path, step=target, state_file=state_basename)
        return tmp_path, target

    def test_all_artifacts_present_yields_skip(self, tmp_path: Path) -> None:
        self._setup_skip_ready(tmp_path)
        plan = decide(tmp_path, _config(production_ns=1.0), force=False)
        assert isinstance(plan, SkipPlan)
        assert plan.early_abort is False

    def test_missing_trajectory_yields_failure(self, tmp_path: Path) -> None:
        self._setup_skip_ready(tmp_path)
        (tmp_path / FileNames.TRAJECTORY).unlink()
        plan = decide(tmp_path, _config(production_ns=1.0), force=False)
        assert isinstance(plan, FailurePlan)
        assert "missing trajectory" in plan.error

    def test_empty_trajectory_yields_failure(self, tmp_path: Path) -> None:
        self._setup_skip_ready(tmp_path)
        (tmp_path / FileNames.TRAJECTORY).write_bytes(b"")
        plan = decide(tmp_path, _config(production_ns=1.0), force=False)
        assert isinstance(plan, FailurePlan)
        assert "empty trajectory" in plan.error

    def test_missing_topology_yields_failure(self, tmp_path: Path) -> None:
        self._setup_skip_ready(tmp_path)
        (tmp_path / FileNames.TOPOLOGY).unlink()
        plan = decide(tmp_path, _config(production_ns=1.0), force=False)
        assert isinstance(plan, FailurePlan)
        assert "missing topology" in plan.error

    def test_empty_topology_yields_failure(self, tmp_path: Path) -> None:
        self._setup_skip_ready(tmp_path)
        (tmp_path / FileNames.TOPOLOGY).write_text("")
        plan = decide(tmp_path, _config(production_ns=1.0), force=False)
        assert isinstance(plan, FailurePlan)
        assert "empty topology" in plan.error

    def test_missing_energy_yields_failure(self, tmp_path: Path) -> None:
        self._setup_skip_ready(tmp_path)
        (tmp_path / FileNames.ENERGY).unlink()
        plan = decide(tmp_path, _config(production_ns=1.0), force=False)
        assert isinstance(plan, FailurePlan)
        assert "missing energy" in plan.error

    def test_empty_energy_yields_failure(self, tmp_path: Path) -> None:
        self._setup_skip_ready(tmp_path)
        (tmp_path / FileNames.ENERGY).write_text("")
        plan = decide(tmp_path, _config(production_ns=1.0), force=False)
        assert isinstance(plan, FailurePlan)
        assert "empty energy" in plan.error

    def test_header_only_energy_yields_failure(self, tmp_path: Path) -> None:
        self._setup_skip_ready(tmp_path)
        (tmp_path / FileNames.ENERGY).write_text("header_only\n")
        plan = decide(tmp_path, _config(production_ns=1.0), force=False)
        assert isinstance(plan, FailurePlan)
        assert "energy" in plan.error
        assert "header-only" in plan.error or "no data" in plan.error


# ---------------------------------------------------------------------------
# End-to-end: decide() roundtrip with atomic_save_checkpoint
# ---------------------------------------------------------------------------


class TestDecideAndAtomicSaveRoundtrip:
    """End-to-end: atomic_save_checkpoint commits a manifest, then
    decide() correctly identifies the run as terminal (when at the
    target step) and produces a fully populated SkipPlan."""

    def test_atomic_save_then_decide_yields_skip_at_target(self, tmp_path: Path) -> None:
        config = _config(production_ns=1.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        sim = MagicMock()
        sim.saveState = MagicMock(side_effect=lambda path: Path(path).write_text("<State/>"))
        # Build the on-disk artifacts (atomic_save writes state + manifest
        # but not the trajectory / energy / topology — those are written
        # by the runner during production).
        (tmp_path / FileNames.TRAJECTORY).write_bytes(b"\x00" * 100)
        (tmp_path / FileNames.ENERGY).write_text("h\nrow\n")
        (tmp_path / FileNames.TOPOLOGY).write_text("H\nATOM\n" * 100)

        atomic_save_checkpoint(sim, tmp_path, absolute_step=target)

        plan = decide(tmp_path, config, force=False)

        assert isinstance(plan, SkipPlan)
        assert plan.manifest_step == target
        assert plan.completion == CompletionStatus.NORMAL_COMPLETE


# ---------------------------------------------------------------------------
# Plan type contract
# ---------------------------------------------------------------------------


class TestRunPlanTypeContract:
    """The plan types are distinct — each carries only the fields
    relevant to its action, so invalid states are unrepresentable."""

    def test_fresh_plan_carries_no_resume_xml(self) -> None:
        plan = FreshPlan(start_step=200_000, remaining_steps=50_000_000)
        assert not hasattr(plan, "resume_xml")
        assert not hasattr(plan, "manifest_step")
        assert not hasattr(plan, "error")

    def test_resume_plan_carries_resume_xml(self) -> None:
        plan = ResumePlan(
            start_step=10_000,
            remaining_steps=49_990_000,
            resume_xml="build/state.10000_1_1.xml",
            manifest_step=10_000,
            state_file_basename="state.10000_1_1.xml",
        )
        assert plan.resume_xml == "build/state.10000_1_1.xml"
        assert not hasattr(plan, "error")

    def test_skip_plan_carries_all_result_fields(self) -> None:
        plan = SkipPlan(
            completion=CompletionStatus.EARLY_ABORT,
            completion_reason="manifest_terminal_early_abort_step_5000000",
            manifest_step=5_000_000,
            state_file_basename="state.5000000_1_1.xml",
            trajectory_path="build/trajectory.dcd",
            energy_path="build/energy.csv",
            topology_path="build/topology.pdb",
            state_xml_path="build/state.5000000_1_1.xml",
            total_ns=5.0,
            early_abort=True,
            abort_reason="5ns gate tripped",
        )
        assert plan.early_abort is True
        assert plan.abort_reason == "5ns gate tripped"
        assert plan.total_ns == 5.0

    def test_failure_plan_carries_only_error(self) -> None:
        plan = FailurePlan(error="state file does not exist")
        assert not hasattr(plan, "start_step")
        assert not hasattr(plan, "resume_xml")
        assert not hasattr(plan, "manifest_step")


# ---------------------------------------------------------------------------
# Plan invariant enforcement — the reviewer flagged that the plan
# types still allowed invalid constructions. The fixes:
#   - ``action`` is a ``ClassVar`` (cannot be set via constructor)
#   - Required fields have no defaults (TypeError on missing)
#   - ``__post_init__`` validates domain invariants (ValueError)
# ---------------------------------------------------------------------------


class TestRunPlanInvariants:
    """The plan types refuse to construct contradictory states."""

    def test_fresh_plan_action_cannot_be_overridden(self) -> None:
        """``action`` is a ClassVar — passing it to the constructor is a TypeError."""
        with pytest.raises(TypeError):
            FreshPlan(  # type: ignore[call-arg]
                action=Action.SKIP,
                start_step=200_000,
                remaining_steps=50_000_000,
            )

    def test_resume_plan_action_cannot_be_overridden(self) -> None:
        with pytest.raises(TypeError):
            ResumePlan(  # type: ignore[call-arg]
                action=Action.FRESH,
                start_step=10_000,
                remaining_steps=49_990_000,
                resume_xml="state.10000_1_1.xml",
                manifest_step=10_000,
                state_file_basename="state.10000_1_1.xml",
            )

    def test_skip_plan_action_cannot_be_overridden(self) -> None:
        with pytest.raises(TypeError):
            SkipPlan(  # type: ignore[call-arg]
                action=Action.RESUME,
                completion=CompletionStatus.NORMAL_COMPLETE,
                completion_reason="normal_completion_step_0_of_0",
                manifest_step=1,
                state_file_basename="state.1_1_1.xml",
                trajectory_path="build/trajectory.dcd",
                energy_path="build/energy.csv",
                topology_path="build/topology.pdb",
                state_xml_path="build/state.1_1_1.xml",
                total_ns=0.0,
                early_abort=False,
                abort_reason=None,
            )

    def test_failure_plan_action_cannot_be_overridden(self) -> None:
        with pytest.raises(TypeError):
            FailurePlan(action=Action.SKIP, error="x")  # type: ignore[call-arg]

    def test_resume_plan_default_construction_fails(self) -> None:
        """Missing required fields → TypeError (no defaults on required)."""
        with pytest.raises(TypeError):
            ResumePlan()  # type: ignore[call-arg]

    def test_failure_plan_empty_error_rejected(self) -> None:
        with pytest.raises(ValueError, match="error must be non-empty"):
            FailurePlan(error="")

    def test_fresh_plan_negative_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="start_step"):
            FreshPlan(start_step=-1, remaining_steps=0)

    def test_resume_plan_empty_resume_xml_rejected(self) -> None:
        with pytest.raises(ValueError, match="resume_xml"):
            ResumePlan(
                start_step=10_000,
                remaining_steps=0,
                resume_xml="",
                manifest_step=10_000,
                state_file_basename="state.10000_1_1.xml",
            )

    def test_skip_plan_early_abort_requires_reason(self) -> None:
        with pytest.raises(ValueError, match="abort_reason"):
            SkipPlan(
                completion=CompletionStatus.EARLY_ABORT,
                completion_reason="x",
                manifest_step=1,
                state_file_basename="state.1_1_1.xml",
                trajectory_path="build/traj.dcd",
                energy_path="build/energy.csv",
                topology_path="build/top.pdb",
                state_xml_path="build/state.1_1_1.xml",
                total_ns=0.0,
                early_abort=True,
                abort_reason=None,
            )
