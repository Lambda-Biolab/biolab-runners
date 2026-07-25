"""Tests for :mod:`biolab_runners.openmm.checkpoint` — the checkpoint domain.

These tests exercise the checkpoint module's **public interface**
(``atomic_save_checkpoint``, ``quarantine_stale_checkpoint``,
``load_checkpoint``, ``is_run_complete``, ``load_terminal_payload``,
``production_ns``, ``InvalidCheckpointError``, ``LoadedCheckpoint``).

Internal helpers (``_parse_manifest``, ``_validate_state_file_reference``,
``_parse_state_filename_step``, ``_gc_orphan_states``,
``_check_normal_completion``, ``_validate_terminal_payload``,
``_classify_invalid_terminal``, ``_production_steps``) are exercised
through the public functions; the AGENTS.md invariants describe
their contracts.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from biolab_runners.openmm.checkpoint import (
    InvalidCheckpointError,
    LoadedCheckpoint,
    atomic_save_checkpoint,
    is_run_complete,
    load_checkpoint,
    load_terminal_payload,
    production_ns,
    quarantine_stale_checkpoint,
)
from biolab_runners.openmm.config import OpenMMConfig
from biolab_runners.openmm.paths import FileNames

logger = logging.getLogger(__name__)


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
    """Write a manifest with the given last record. ``terminal`` is optional."""
    record: dict[str, object] = {"step": step, "file": state_file}
    if terminal is not None:
        record["terminal"] = terminal
    manifest = {"records": [record]}
    (output_dir / FileNames.CHECKPOINT_JSON).write_text(json.dumps(manifest))


def _fake_simulator(tmp_path: Path) -> MagicMock:
    """A mock OpenMM Simulation whose ``saveState`` writes the file to disk."""
    sim = MagicMock()
    sim.saveState = MagicMock(side_effect=lambda path: Path(path).write_text("<State/>"))
    return sim


# ---------------------------------------------------------------------------
# atomic_save_checkpoint
# ---------------------------------------------------------------------------


class TestAtomicSaveCheckpoint:
    """Public-interface tests for ``atomic_save_checkpoint``.

    The v7 BLOCKER invariant: the manifest ``os.replace`` is the
    single atomic commit point. The state file is uniquely named
    so it does not need a temp+rename.
    """

    def test_writes_state_file_and_manifest(self, tmp_path: Path) -> None:
        sim = _fake_simulator(tmp_path)

        state_basename = atomic_save_checkpoint(sim, tmp_path, absolute_step=42_000)

        # State file is at the versioned name (NOT canonical state.xml).
        assert state_basename.startswith("state.42000_")
        assert state_basename.endswith(".xml")
        state_path = tmp_path / state_basename
        assert state_path.exists()
        # legacy state.xml is NOT created.
        assert not (tmp_path / "state.xml").exists()
        # Manifest exists and references the state file.
        manifest_path = tmp_path / FileNames.CHECKPOINT_JSON
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        last_record = manifest["records"][-1]
        assert last_record["step"] == 42_000
        assert last_record["file"] == state_basename

    def test_interrupted_save_state_leaves_previous_checkpoint_active(self, tmp_path: Path) -> None:
        """If ``saveState`` raises, the manifest is unchanged."""
        sim = MagicMock()
        sim.saveState = MagicMock(side_effect=RuntimeError("simulated disk-full"))

        # Pre-create a previous coherent checkpoint.
        previous_state = tmp_path / "state.1_1_1.xml"
        previous_state.write_text("<OLD_STATE/>")
        _write_manifest(tmp_path, step=1, state_file="state.1_1_1.xml")

        with pytest.raises(RuntimeError, match="simulated"):
            atomic_save_checkpoint(sim, tmp_path, absolute_step=99_999)

        # Previous manifest is unchanged.
        manifest = json.loads((tmp_path / FileNames.CHECKPOINT_JSON).read_text())
        assert manifest["records"][-1]["step"] == 1
        assert manifest["records"][-1]["file"] == "state.1_1_1.xml"
        assert previous_state.exists()

    def test_garbage_collects_orphan_state_files(self, tmp_path: Path) -> None:
        """After a save, any state.*.xml not referenced by the manifest is removed."""
        sim = _fake_simulator(tmp_path)

        # Pre-create an orphan state file (simulating a v6 leftover or an interrupted save).
        orphan = tmp_path / "state.99999_12345_170000000.xml"
        orphan.write_text("<ORPHAN/>")

        state_basename = atomic_save_checkpoint(sim, tmp_path, absolute_step=42_000)

        # Orphan is gone, the active state file is present.
        assert not orphan.exists()
        assert (tmp_path / state_basename).exists()

    def test_two_saves_produce_distinct_state_files(self, tmp_path: Path) -> None:
        """Two consecutive saves produce distinct state files; the manifest
        references the latest one and the previous is GC'd."""
        sim = _fake_simulator(tmp_path)

        first_basename = atomic_save_checkpoint(sim, tmp_path, absolute_step=10_000)
        second_basename = atomic_save_checkpoint(sim, tmp_path, absolute_step=20_000)

        assert first_basename != second_basename
        manifest = json.loads((tmp_path / FileNames.CHECKPOINT_JSON).read_text())
        assert manifest["records"][-1]["file"] == second_basename
        assert manifest["records"][-1]["step"] == 20_000
        # Previous state file was GC'd.
        assert not (tmp_path / first_basename).exists()
        assert (tmp_path / second_basename).exists()

    def test_atomicity_under_simulated_crash_between_state_and_manifest(
        self, tmp_path: Path
    ) -> None:
        """Crash mid-save: previous checkpoint remains active (v7 BLOCKER #2).

        Simulated by patching ``os.replace`` so that the manifest
        rename fails. The next ``load_checkpoint`` must return the
        previous step + state file, not the half-published new step.
        The next successful save GC's the orphan state file written
        before the failure.
        """
        import os

        # Pre-create a previous coherent checkpoint.
        previous_state = tmp_path / "state.500_12345_1000.xml"
        previous_state.write_text("<PREVIOUS_STATE/>")
        _write_manifest(tmp_path, step=500, state_file="state.500_12345_1000.xml")

        # Save the real ``os.replace`` so we can patch selectively.
        original_replace = os.replace
        state_file_writes: list[str] = []

        def failing_replace(src: str, dst: str) -> None:
            if str(dst).endswith(FileNames.CHECKPOINT_JSON):
                raise OSError("simulated disk-full during manifest rename")
            original_replace(src, dst)

        sim = MagicMock()

        def record_save(path: str) -> None:
            state_file_writes.append(path)
            Path(path).write_text("<NEW_STATE_PARTIAL/>")

        sim.saveState = MagicMock(side_effect=record_save)

        # Apply the patch on the checkpoint module's binding (not the
        # caller's). We patch the module-level ``os.replace`` inside
        # checkpoint so the atomic-save code sees the patched function.
        from biolab_runners.openmm import checkpoint as ckpt_mod

        ckpt_mod.os.replace = failing_replace
        try:
            with pytest.raises(OSError, match="simulated"):
                atomic_save_checkpoint(sim, tmp_path, absolute_step=999_999)
        finally:
            ckpt_mod.os.replace = original_replace

        # The MANIFEST is unchanged — the previous checkpoint remains active.
        checkpoint = load_checkpoint(tmp_path)
        assert checkpoint.absolute_step == 500
        assert checkpoint.state_file_basename == "state.500_12345_1000.xml"

        # The half-published state file is on disk but unreferenced.
        new_state_files = [
            f for f in tmp_path.glob("state*.xml") if f.name != "state.500_12345_1000.xml"
        ]
        assert len(new_state_files) == 1

        # The next successful save GC's the orphan.
        sim.saveState = MagicMock(side_effect=lambda path: Path(path).write_text("<State/>"))
        atomic_save_checkpoint(sim, tmp_path, absolute_step=800_000)
        assert not new_state_files[0].exists()
        next_checkpoint = load_checkpoint(tmp_path)
        assert next_checkpoint.absolute_step == 800_000
        assert (tmp_path / next_checkpoint.state_file_basename).exists()

    def test_terminal_payload_committed_atomically_with_state_file(self, tmp_path: Path) -> None:
        """v10 BLOCKER #2: the terminal payload commits in the same ``os.replace``
        as the state file — a crash between the two cannot leave a
        resumable checkpoint whose terminal decision was already made."""
        sim = _fake_simulator(tmp_path)

        terminal = {
            "type": "early_abort",
            "step": 7_500_000,
            "reason": "5ns gate tripped",
            "production_ns": 5.0,
        }
        state_basename = atomic_save_checkpoint(
            sim, tmp_path, absolute_step=7_500_000, terminal=terminal
        )

        manifest = json.loads((tmp_path / FileNames.CHECKPOINT_JSON).read_text())
        record = manifest["records"][-1]
        assert record["step"] == 7_500_000
        assert record["file"] == state_basename
        assert record["terminal"] == terminal


# ---------------------------------------------------------------------------
# quarantine_stale_checkpoint
# ---------------------------------------------------------------------------


class TestQuarantineStaleCheckpoint:
    """Public-interface tests for ``quarantine_stale_checkpoint``."""

    def test_moves_manifest_energy_and_early_abort_marker(self, tmp_path: Path) -> None:
        for name in FileNames.PRODUCTION_OUTPUT_FILES:
            (tmp_path / name).write_text("placeholder")
        (tmp_path / FileNames.CHECKPOINT_JSON).write_text("{}")
        (tmp_path / FileNames.EARLY_ABORT_JSON).write_text("{}")

        moved = quarantine_stale_checkpoint(tmp_path)

        # All three named files moved into .stale/<ts>/.
        moved_names = {p.name for p in moved}
        assert FileNames.CHECKPOINT_JSON in moved_names
        assert FileNames.ENERGY in moved_names
        assert FileNames.EARLY_ABORT_JSON in moved_names
        # Original locations are empty.
        for name in (FileNames.CHECKPOINT_JSON, FileNames.ENERGY, FileNames.EARLY_ABORT_JSON):
            assert not (tmp_path / name).exists()

    def test_moves_state_files_legacy_and_v7(self, tmp_path: Path) -> None:
        (tmp_path / FileNames.CHECKPOINT_JSON).write_text("{}")
        # Legacy state file (pre-v7).
        legacy = tmp_path / "state.xml"
        legacy.write_text("<State/>")
        # v7 generation-versioned state file.
        v7 = tmp_path / "state.500_12345_170000000.xml"
        v7.write_text("<State/>")

        moved = quarantine_stale_checkpoint(tmp_path)

        moved_names = {p.name for p in moved}
        assert "state.xml" in moved_names
        assert "state.500_12345_170000000.xml" in moved_names
        assert not legacy.exists()
        assert not v7.exists()

    def test_empty_output_dir_produces_empty_list(self, tmp_path: Path) -> None:
        moved = quarantine_stale_checkpoint(tmp_path)
        assert moved == []
        assert not (tmp_path / ".stale").exists()

    def test_two_invocations_get_distinct_timestamps(self, tmp_path: Path) -> None:
        """Rapid retries within the same second must not collide on the
        .stale/<ts>/ directory — the timestamp includes microseconds + PID."""
        (tmp_path / FileNames.CHECKPOINT_JSON).write_text("{}")
        first = quarantine_stale_checkpoint(tmp_path)
        # Restore the file (force=True consumes it), then quarantine again.
        (tmp_path / FileNames.CHECKPOINT_JSON).write_text("{}")
        # Tiny sleep so the second invocation gets a different timestamp at least at the
        # microsecond boundary; the timestamp format already includes microseconds.
        time.sleep(0.001)
        second = quarantine_stale_checkpoint(tmp_path)
        assert first[0].parent != second[0].parent


# ---------------------------------------------------------------------------
# load_checkpoint — public, structured result
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    """Public-interface tests for ``load_checkpoint``.

    The single ``load_checkpoint`` entry point returns a
    :class:`LoadedCheckpoint` with named fields for the saved
    step, state file basename, last manifest record, and any
    validated terminal payload.
    """

    def test_returns_structured_result_for_v7_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "state.700000_12345_170000000.xml").write_text("<State/>")
        _write_manifest(tmp_path, step=700_000, state_file="state.700000_12345_170000000.xml")

        checkpoint = load_checkpoint(tmp_path)

        assert isinstance(checkpoint, LoadedCheckpoint)
        assert checkpoint.absolute_step == 700_000
        assert checkpoint.state_file_basename == "state.700000_12345_170000000.xml"
        assert checkpoint.last_record["step"] == 700_000
        assert checkpoint.is_terminal is False
        assert checkpoint.terminal_reason is None

    def test_returns_empty_when_no_manifest(self, tmp_path: Path) -> None:
        checkpoint = load_checkpoint(tmp_path)

        assert checkpoint.absolute_step == 0
        assert checkpoint.state_file_basename == ""
        assert checkpoint.last_record == {}
        assert checkpoint.is_terminal is False

    def test_legacy_state_xml_accepted_with_logged_notice(self, tmp_path: Path) -> None:
        """Legacy ``state.xml`` has no embedded step; the manifest step is trusted."""
        (tmp_path / "state.xml").write_text("<State/>")
        _write_manifest(tmp_path, step=42_000, state_file="state.xml")

        checkpoint = load_checkpoint(tmp_path)

        assert checkpoint.absolute_step == 42_000
        assert checkpoint.state_file_basename == "state.xml"

    def test_dangling_state_file_reference_raises(self, tmp_path: Path) -> None:
        """Manifest references a state file that does not exist — fail fast."""
        _write_manifest(tmp_path, step=1, state_file="state.1_12345_170000000.xml")

        with pytest.raises(InvalidCheckpointError, match="does not exist"):
            load_checkpoint(tmp_path)

    def test_empty_state_file_raises(self, tmp_path: Path) -> None:
        (tmp_path / "state.1_12345_170000000.xml").write_text("")
        _write_manifest(tmp_path, step=1, state_file="state.1_12345_170000000.xml")

        with pytest.raises(InvalidCheckpointError, match="empty"):
            load_checkpoint(tmp_path)

    def test_step_mismatch_with_state_filename_raises(self, tmp_path: Path) -> None:
        """v10 BLOCKER #1: manifest step must equal the step encoded in the state filename."""
        (tmp_path / "state.999_12345_170000000.xml").write_text("<State/>")
        # Manifest claims step=1000 — the filename says 999.
        _write_manifest(tmp_path, step=1_000, state_file="state.999_12345_170000000.xml")

        with pytest.raises(InvalidCheckpointError, match="does not match"):
            load_checkpoint(tmp_path)

    def test_path_traversal_in_state_filename_raises(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, step=1, state_file="../escape.xml")

        with pytest.raises(InvalidCheckpointError, match="not a basename"):
            load_checkpoint(tmp_path)

    def test_invalid_state_filename_pattern_raises(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, step=1, state_file="random.xml")

        with pytest.raises(InvalidCheckpointError, match="invalid"):
            load_checkpoint(tmp_path)

    def test_malformed_manifest_root_returns_empty(self, tmp_path: Path) -> None:
        """A manifest whose root is not a JSON object returns empty
        (treated as "no resumable checkpoint") rather than raising."""
        (tmp_path / FileNames.CHECKPOINT_JSON).write_text(json.dumps(["not", "a", "dict"]))

        checkpoint = load_checkpoint(tmp_path)

        assert checkpoint.absolute_step == 0
        assert checkpoint.state_file_basename == ""

    def test_malformed_manifest_missing_records_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / FileNames.CHECKPOINT_JSON).write_text(json.dumps({"step": 1}))

        checkpoint = load_checkpoint(tmp_path)

        assert checkpoint.absolute_step == 0

    def test_malformed_manifest_zero_step_returns_empty(self, tmp_path: Path) -> None:
        """A manifest with ``step`` of zero or negative returns empty."""
        state = tmp_path / "state.0_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(tmp_path, step=0, state_file="state.0_1_1.xml")

        checkpoint = load_checkpoint(tmp_path)

        assert checkpoint.absolute_step == 0

    def test_malformed_manifest_string_step_returns_empty(self, tmp_path: Path) -> None:
        """A manifest with ``step`` as a string returns empty (strict-int)."""
        state = tmp_path / "state.1_1_1.xml"
        state.write_text("<State/>")
        manifest_path = tmp_path / FileNames.CHECKPOINT_JSON
        manifest_path.write_text(
            json.dumps({"records": [{"step": "5000000", "file": "state.1_1_1.xml"}]})
        )

        checkpoint = load_checkpoint(manifest_path.parent)

        assert checkpoint.absolute_step == 0

    def test_terminal_payload_surfaced_via_structured_result(self, tmp_path: Path) -> None:
        """Valid terminal payload populates ``is_terminal`` and ``terminal_reason``."""
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

        checkpoint = load_checkpoint(tmp_path)

        assert checkpoint.absolute_step == 5_000_000
        assert checkpoint.is_terminal is True
        assert checkpoint.terminal_reason == "manifest_terminal_early_abort_step_5000000"

    def test_invalid_terminal_payload_leaves_is_terminal_false(self, tmp_path: Path) -> None:
        """A present-but-invalid terminal leaves ``is_terminal=False``;
        the failure is surfaced separately via :func:`is_run_complete`."""
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        # Step mismatch — the terminal.step is NOT equal to manifest.step.
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal={
                "type": "early_abort",
                "step": 9_999_999,  # Mismatch.
                "reason": "fake",
                "production_ns": 0.0,
            },
        )

        checkpoint = load_checkpoint(tmp_path)

        assert checkpoint.is_terminal is False
        assert checkpoint.terminal_reason is None
        # ``is_run_complete`` must report the malformed payload specifically.
        config = OpenMMConfig(production_ns=100.0)
        complete, reason = is_run_complete(tmp_path, config)
        assert complete is False
        assert reason.startswith("invalid_terminal_")


# ---------------------------------------------------------------------------
# is_run_complete — tri-state terminal check
# ---------------------------------------------------------------------------


class TestIsRunComplete:
    """Public-interface tests for :func:`is_run_complete`.

    Tri-state terminal classification (v13 BLOCKER):
    - absent (no terminal field) → fall back to normal completion
    - valid → terminal
    - invalid → (False, "invalid_terminal_<reason>") — NEVER normal completion
    """

    def test_no_manifest_returns_in_progress(self, tmp_path: Path) -> None:
        config = OpenMMConfig(output_dir=str(tmp_path), production_ns=100.0)
        complete, reason = is_run_complete(tmp_path, config)

        assert (complete, reason) == (False, "in_progress")

    def test_normal_completion_via_manifest_step(self, tmp_path: Path) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        # Build a coherent manifest at the target step.
        target = config.total_equil_steps + config.total_steps
        state = tmp_path / f"state.{target}_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(tmp_path, step=target, state_file=f"state.{target}_1_1.xml")

        complete, reason = is_run_complete(tmp_path, config)

        assert complete is True
        assert reason.startswith("normal_completion_step_")

    def test_below_target_returns_in_progress(self, tmp_path: Path) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.100_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(tmp_path, step=100, state_file="state.100_1_1.xml")

        complete, reason = is_run_complete(tmp_path, config)

        assert (complete, reason) == (False, "in_progress")

    def test_valid_terminal_payload_marks_terminal(self, tmp_path: Path) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
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

        complete, reason = is_run_complete(tmp_path, config)

        assert complete is True
        assert reason == "manifest_terminal_early_abort_step_5000000"

    def test_invalid_terminal_payload_does_not_fall_back_to_normal_completion(
        self, tmp_path: Path
    ) -> None:
        """v13 BLOCKER: a malformed terminal at the target step MUST NOT
        be reclassified as normal completion."""
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state = tmp_path / f"state.{target}_1_1.xml"
        state.write_text("<State/>")
        # Valid step, INVALID type.
        _write_manifest(
            tmp_path,
            step=target,
            state_file=f"state.{target}_1_1.xml",
            terminal={
                "type": "future_marker",
                "step": target,
                "reason": "experimental",
                "production_ns": 0.0,
            },
        )

        complete, reason = is_run_complete(tmp_path, config)

        assert complete is False
        assert reason.startswith("invalid_terminal_type_unsupported")

    def test_invalid_terminal_payload_invalid_step_returns_invalid_reason(
        self, tmp_path: Path
    ) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        # String step (strict-int required).
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal={
                "type": "early_abort",
                "step": "5000000",  # string
                "reason": "fake",
                "production_ns": 0.0,
            },
        )

        complete, reason = is_run_complete(tmp_path, config)

        assert complete is False
        assert reason == "invalid_terminal_step_invalid_type"

    def test_invalid_terminal_payload_step_mismatch_returns_invalid_reason(
        self, tmp_path: Path
    ) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal={
                "type": "early_abort",
                "step": 9_999_999,  # mismatch
                "reason": "fake",
                "production_ns": 0.0,
            },
        )

        complete, reason = is_run_complete(tmp_path, config)

        assert complete is False
        assert reason == "invalid_terminal_step_mismatch"

    def test_invalid_terminal_payload_empty_reason_returns_invalid_reason(
        self, tmp_path: Path
    ) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal={
                "type": "early_abort",
                "step": 5_000_000,
                "reason": "",  # empty
                "production_ns": 0.0,
            },
        )

        complete, reason = is_run_complete(tmp_path, config)

        assert complete is False
        assert reason == "invalid_terminal_reason_empty"

    def test_invalid_terminal_payload_not_dict_returns_invalid_reason(self, tmp_path: Path) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal="not a dict",  # type: ignore[arg-type]
        )

        complete, reason = is_run_complete(tmp_path, config)

        assert complete is False
        assert reason == "invalid_terminal_not_dict"

    def test_explicit_terminal_payload_precedes_normal_completion(self, tmp_path: Path) -> None:
        """v12 BLOCKER: when both signals fire on the same step, the
        explicit terminal payload wins over inferred normal completion."""
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        state = tmp_path / f"state.{target}_1_1.xml"
        state.write_text("<State/>")
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

        complete, reason = is_run_complete(tmp_path, config)

        assert complete is True
        assert reason.startswith("manifest_terminal_early_abort_")


# ---------------------------------------------------------------------------
# load_terminal_payload — reconstructed for early-abort idempotent reuse
# ---------------------------------------------------------------------------


class TestLoadTerminalPayload:
    """Public-interface tests for :func:`load_terminal_payload`."""

    def test_returns_normalised_payload_when_valid(self, tmp_path: Path) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
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
                "gate": "5ns",
                "target": "demo",
                "peptide_id": "PEP001",
            },
        )

        payload = load_terminal_payload(tmp_path, config)

        assert payload is not None
        assert payload["step"] == 5_000_000
        assert payload["type"] == "early_abort"
        assert payload["reason"] == "5ns gate tripped"
        assert payload["gate"] == "5ns"
        assert payload["target"] == "demo"
        assert payload["peptide_id"] == "PEP001"
        # production_ns is computed from the v10 BLOCKER #3 invariant,
        # NOT read from the stored field.
        assert payload["production_ns"] == production_ns(5_000_000, config)

    def test_returns_none_when_no_terminal(self, tmp_path: Path) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.100_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(tmp_path, step=100, state_file="state.100_1_1.xml")

        assert load_terminal_payload(tmp_path, config) is None

    def test_returns_none_when_terminal_invalid(self, tmp_path: Path) -> None:
        """v11 contract: an invalid terminal payload (empty reason,
        wrong type, step mismatch, non-int step) is treated as
        invalid and ``load_terminal_payload`` returns ``None``.

        This is a deliberate tightening from the previous
        "lenient parse, caller re-validates" behavior — the new
        ``inspect_checkpoint`` does the full schema validation
        once, and ``load_terminal_payload`` (its thin wrapper)
        inherits that strictness. Callers that need raw access
        to a possibly-malformed payload should use
        :func:`inspect_checkpoint` directly and read
        ``snapshot.terminal_payload``.
        """
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal={"type": "early_abort", "step": 5_000_000, "reason": ""},
        )

        assert load_terminal_payload(tmp_path, config) is None

    def test_returns_none_when_step_mismatch(self, tmp_path: Path) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        state = tmp_path / "state.5000000_1_1.xml"
        state.write_text("<State/>")
        _write_manifest(
            tmp_path,
            step=5_000_000,
            state_file="state.5000000_1_1.xml",
            terminal={"type": "early_abort", "step": 9_999_999, "reason": "fake"},
        )

        assert load_terminal_payload(tmp_path, config) is None


# ---------------------------------------------------------------------------
# production_ns
# ---------------------------------------------------------------------------


class TestProductionNs:
    """Public-interface tests for :func:`production_ns`."""

    def test_zero_for_pre_equilibration_step(self) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        # Before equilibration ends, production_ns is 0.
        assert production_ns(0, config) == 0.0
        assert production_ns(config.total_equil_steps // 2, config) == 0.0

    def test_nonzero_after_equilibration(self) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        # 1 production step = 2 fs = 0.000002 ns.
        ns = production_ns(config.total_equil_steps + 1, config)
        assert ns == pytest.approx(0.000002, rel=1e-6)

    def test_full_production_ns_at_target(self) -> None:
        config = OpenMMConfig(production_ns=10.0, timestep_fs=2.0)
        target = config.total_equil_steps + config.total_steps
        ns = production_ns(target, config)
        assert ns == pytest.approx(10.0, rel=1e-6)
