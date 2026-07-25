"""Run state decision — what should the runner do given the on-disk checkpoint state?

This module owns the **decision** part of an OpenMM MD run: given an
output directory, a configuration, and a ``force`` flag, inspect
the on-disk checkpoint (a single coherent read via
:func:`biolab_runners.openmm.checkpoint.inspect_checkpoint`) and
return a typed :data:`RunPlan` telling the runner what to do next.

The runner in :mod:`biolab_runners.openmm.runner` is a thin
dispatcher: it calls :func:`decide`, then acts on the returned
plan. Pre-run decision logic (manifest validation, terminal
classification, orphan detection, quarantine-on-force) and
skip-population logic (artifact validation, terminal
reconstruction) live here, not on the runner.

Why a separate module:

- The decision has many branches. The four possible outcomes
  (FRESH, RESUME, SKIP, FAIL_FAST) are encoded as four
  distinct dataclass types so invalid constructions are
  unrepresentable. The runner matches on the plan type, not on
  string-prefixes from a reason field.
- The whole decision is one read of the manifest — no race
  window between fresh builds of the same plan. The previous
  design called ``load_checkpoint`` + ``is_run_complete`` +
  ``load_terminal_payload`` separately, which could combine
  Generation A's state_file with Generation B's terminal
  classification if a concurrent commit landed between the
  reads. The current design uses
  :func:`biolab_runners.openmm.checkpoint.inspect_checkpoint`
  once and threads the snapshot through every step.
- Decision logic and MD mechanics change at different cadences.
  Keeping them separate means a force-field tweak doesn't touch
  the decision tree.

The seam between this module and the runner is the public
interface: :func:`decide`, :data:`RunPlan`, and the four plan
types (:class:`FreshPlan`, :class:`ResumePlan`, :class:`SkipPlan`,
:class:`FailurePlan`). Internal helpers
(:func:`_populate_skip_plan_fields`,
:func:`_artifact_validation_error`) are part of the module's
internal seam — used by this module's own tests for specific
failure modes, not part of the public surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from biolab_runners.openmm.checkpoint import (
    CheckpointSnapshot,
    CompletionStatus,
    InvalidCheckpointError,
    inspect_checkpoint,
    quarantine_stale_checkpoint,
)
from biolab_runners.openmm.paths import FileNames

if TYPE_CHECKING:
    from pathlib import Path

    from biolab_runners.openmm.config import OpenMMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class Action(StrEnum):
    """What the runner should do, decided by :func:`decide`.

    The action is also encoded on each plan type (``FreshPlan``,
    ``ResumePlan``, ``SkipPlan``, ``FailurePlan`` set ``action`` to
    the matching value) so the runner can match on either the
    type or the action enum.
    """

    FRESH = "fresh"
    RESUME = "resume"
    SKIP = "skip"
    FAIL_FAST = "fail_fast"


@dataclass(frozen=True)
class FreshPlan:
    """The runner should build a fresh simulation from scratch."""

    action: Action = Action.FRESH
    start_step: int = 0
    remaining_steps: int = 0


@dataclass(frozen=True)
class ResumePlan:
    """The runner should resume from the saved state file."""

    action: Action = Action.RESUME
    start_step: int = 0
    remaining_steps: int = 0
    resume_xml: str = ""
    manifest_step: int = 0
    state_file_basename: str = ""


@dataclass(frozen=True)
class SkipPlan:
    """The run is terminal; the result is fully populated for the runner to copy.

    All fields are populated by :func:`decide` so the runner
    doesn't need to do any artifact validation, terminal
    reconstruction, or string-prefix parsing. The runner reads
    these fields and assigns them to ``SimulationResult``.
    """

    action: Action = Action.SKIP
    completion: CompletionStatus = CompletionStatus.NORMAL_COMPLETE
    completion_reason: str = ""
    manifest_step: int = 0
    state_file_basename: str = ""
    # Populated artifact paths (runner copies into result).
    trajectory_path: str = ""
    energy_path: str = ""
    topology_path: str = ""
    state_xml_path: str = ""
    total_ns: float = 0.0
    early_abort: bool = False
    abort_reason: str | None = None


@dataclass(frozen=True)
class FailurePlan:
    """The runner should not run MD; ``result.error`` is set."""

    action: Action = Action.FAIL_FAST
    error: str = ""


# Tagged union — the runner matches on either the type or the
# ``action`` field. Each plan type has the ``action`` field set
# to the matching enum so ``match plan.action`` is sound.
RunPlan = FreshPlan | ResumePlan | SkipPlan | FailurePlan


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decide(
    output_dir: Path,
    config: OpenMMConfig,
    force: bool,
) -> RunPlan:
    """Inspect the on-disk checkpoint and decide what the runner should do.

    Single coherent read of the manifest via
    :func:`biolab_runners.openmm.checkpoint.inspect_checkpoint`. The
    returned plan carries everything the runner needs to dispatch
    on — no second public call is required.

    Decision tree (in order):

    1. **Force quarantine**. If ``force=True``, move the manifest,
       energy log, ``early_abort.json``, and every state file
       into ``output_dir/.stale/<UTC>/``. This step is a side
       effect — it makes a fresh build safe by removing anything
       that could pair with a freshly-built topology.

    2. **Inspect the manifest**. Single read via
       :func:`inspect_checkpoint`. On
       :class:`InvalidCheckpointError` (manifest references a
       missing / empty / path-traversal / step-mismatched state
       file), return ``FailurePlan``.

    3. **Completion classification** (from the snapshot):
       - ``NORMAL_COMPLETE`` → ``SkipPlan`` with ``total_ns``
         computed from the v10 BLOCKER #3 invariant.
       - ``EARLY_ABORT`` → ``SkipPlan`` with the validated
         terminal payload's fields populated.
       - ``INVALID_TERMINAL`` → ``FailurePlan`` (the v11
         contract — never resume, never fall back to normal
         completion). The runner surfaces the
         ``invalid_terminal_<reason>`` message.
       - ``IN_PROGRESS`` → check for orphan state files (no
         valid manifest + state files on disk is an
         inconsistent state, since v7 commits the state file
         together with the manifest). If orphans exist,
         ``FailurePlan``; otherwise ``FreshPlan`` or
         ``ResumePlan`` depending on whether the snapshot has
         a manifest step.

    Args:
        output_dir: MD output directory.
        config: OpenMMConfig (used for the target step in
            normal-completion classification).
        force: ``runner.run(force=True)`` semantic — quarantine
            stale files before inspecting.

    Returns:
        A :data:`RunPlan` tagged union: ``FreshPlan``,
        ``ResumePlan``, ``SkipPlan``, or ``FailurePlan``. The
        runner matches on either the type or the ``action`` field.

    Raises:
        Nothing — all error modes are encoded in ``FailurePlan``.
        This is deliberate: the runner's decision should never
        raise; corruption surfaces as ``result.error``, never as
        an unhandled exception.
    """
    if force:
        moved = quarantine_stale_checkpoint(output_dir)
        if moved:
            logger.info(
                "force=True: quarantined %d stale checkpoint file(s) to %s",
                len(moved),
                moved[0].parent,
            )

    try:
        snapshot = inspect_checkpoint(output_dir, config)
    except InvalidCheckpointError as exc:
        return FailurePlan(
            error=(
                f"{exc} The checkpoint is in an unrecoverable state; "
                f"re-run with force=True to discard it."
            ),
        )

    if snapshot.absolute_step <= 0:
        # No manifest — check for orphan state files.
        orphan_error = _orphan_state_error(output_dir)
        if orphan_error is not None:
            return FailurePlan(error=orphan_error)
        return FreshPlan(
            start_step=config.total_equil_steps,
            remaining_steps=config.total_steps,
        )

    if snapshot.completion in (
        CompletionStatus.NORMAL_COMPLETE,
        CompletionStatus.EARLY_ABORT,
    ):
        return _populate_skip_plan_fields(output_dir, snapshot, config)

    if snapshot.completion == CompletionStatus.INVALID_TERMINAL:
        # v13 BLOCKER: a malformed terminal payload is in an
        # ambiguous state. The runner must fail loudly — never
        # resume, never fall back to normal completion. The user
        # must investigate via ``force=True``.
        return FailurePlan(
            error=(
                f"Manifest carries a malformed terminal payload at "
                f"step {snapshot.absolute_step} "
                f"(reason={snapshot.completion_reason!r}). The "
                f"user attempted to record a terminal decision but "
                f"the schema is wrong; the run is in an ambiguous "
                f"state. Re-run with force=True to quarantine the "
                f"manifest and start fresh."
            ),
        )

    # IN_PROGRESS with a valid manifest → resume.
    logger.info(
        "Resuming from checkpoint at step %d (%.2f ns of %d needed)",
        snapshot.absolute_step,
        snapshot.absolute_step * config.timestep_fs / 1e6,
        config.total_equil_steps + config.total_steps,
    )
    return ResumePlan(
        start_step=snapshot.absolute_step,
        remaining_steps=max(
            0,
            config.total_steps - max(0, snapshot.absolute_step - config.total_equil_steps),
        ),
        resume_xml=str(output_dir / snapshot.state_file_basename),
        manifest_step=snapshot.absolute_step,
        state_file_basename=snapshot.state_file_basename,
    )


# ---------------------------------------------------------------------------
# Internal helpers (module's internal seam — used by this module's tests
# for specific failure modes; not part of the public surface).
# ---------------------------------------------------------------------------


def _populate_skip_plan_fields(
    output_dir: Path,
    snapshot: CheckpointSnapshot,
    config: OpenMMConfig,
) -> SkipPlan | FailurePlan:
    """Build a fully-populated :class:`SkipPlan` from the snapshot.

    Runs artifact validation (single pass) and pulls the
    terminal payload's fields out of the snapshot. The runner
    just copies these into ``SimulationResult``.
    """
    artifact_error = _artifact_validation_error(output_dir)
    if artifact_error is not None:
        return FailurePlan(
            error=(
                artifact_error + " The checkpoint recorded completion but the "
                "scientific outputs are not usable; the user must "
                "investigate the prior run (e.g. disk full during "
                "trajectory write) and re-run with force=True."
            ),
        )

    is_early_abort = snapshot.completion == CompletionStatus.EARLY_ABORT
    payload = snapshot.terminal_payload

    # v10 BLOCKER #3: total_ns is PRODUCTION ns. For normal
    # completion, computed from the absolute step + config. For
    # early abort, carried in the validated payload (already
    # computed by inspect_checkpoint).
    if is_early_abort and payload is not None:
        total_ns = float(payload["production_ns"])
        abort_reason: str | None = str(payload.get("reason", ""))
    else:
        total_ns = _normal_completion_total_ns(snapshot.absolute_step, config)
        abort_reason = None

    return SkipPlan(
        completion=snapshot.completion,
        completion_reason=snapshot.completion_reason,
        manifest_step=snapshot.absolute_step,
        state_file_basename=snapshot.state_file_basename,
        trajectory_path=str(output_dir / FileNames.TRAJECTORY),
        energy_path=str(output_dir / FileNames.ENERGY),
        topology_path=str(output_dir / FileNames.TOPOLOGY),
        state_xml_path=str(output_dir / snapshot.state_file_basename),
        total_ns=total_ns,
        early_abort=is_early_abort,
        abort_reason=abort_reason,
    )


def _normal_completion_total_ns(absolute_step: int, config: OpenMMConfig) -> float:
    """Compute PRODUCTION ns for a normal-completion skip.

    Per the v10 BLOCKER #3 invariant, total_ns is the COMPLETED
    PRODUCTION ns: ``max(0, absolute_step - total_equil_steps) *
    timestep_fs / 1e6``.
    """
    return max(0, absolute_step - config.total_equil_steps) * config.timestep_fs / 1e6


def _orphan_state_error(output_dir: Path) -> str | None:
    """Detect orphaned state files when no manifest is present.

    No valid manifest + state files on disk (legacy ``state.xml``
    from a v6 run or v7 ``state.<gen>.xml`` from an interrupted
    save that landed before the manifest rename) is an orphan
    condition. Pairing the orphan state with a freshly-built
    System would re-introduce the incompatibility the "Resume
    safety" rule exists to avoid.

    Returns:
        The error message string if orphan files are detected,
        ``None`` if the directory is clean (no state files
        without a manifest).
    """
    leftover_states = list(output_dir.glob("state*.xml"))
    if not leftover_states:
        return None
    return (
        f"State file(s) exist at {leftover_states} but "
        f"the manifest {FileNames.CHECKPOINT_JSON} is "
        f"missing or invalid — the saved state's step is "
        f"unknown. Pairing it with a freshly-built System "
        f"would re-introduce the incompatibility this rule "
        f"exists to avoid. Re-run with force=True to "
        f"discard the orphaned checkpoint."
    )
    """Detect orphaned state files when no manifest is present.

    No valid manifest + state files on disk (legacy ``state.xml``
    from a v6 run or v7 ``state.<gen>.xml`` from an interrupted
    save that landed before the manifest rename) is an orphan
    condition. Pairing the orphan state with a freshly-built
    System would re-introduce the incompatibility the "Resume
    safety" rule exists to avoid.

    Returns:
        The error message string if orphan files are detected,
        ``None`` if the directory is clean (no state files
        without a manifest).
    """
    leftover_states = list(output_dir.glob("state*.xml"))
    if not leftover_states:
        return None
    return (
        f"State file(s) exist at {leftover_states} but "
        f"the manifest {FileNames.CHECKPOINT_JSON} is "
        f"missing or invalid — the saved state's step is "
        f"unknown. Pairing it with a freshly-built System "
        f"would re-introduce the incompatibility this rule "
        f"exists to avoid. Re-run with force=True to "
        f"discard the orphaned checkpoint."
    )


def _artifact_validation_error(output_dir: Path) -> str | None:
    """Verify that a terminal run has its promised outputs.

    ``inspect_checkpoint`` returns the terminal classification
    from the manifest, but a terminal run also needs the
    scientific outputs (trajectory, energy log, topology) to be
    present and usable. A terminal manifest plus state file but
    no trajectory / energy returns a result with paths pointing
    to nonexistent files — silently misleading downstream
    consumers.

    Returns:
        ``None`` if all required artifacts are present and
        usable. Otherwise, a human-readable error message
        naming the first missing or unusable artifact.

    Validation:
    - trajectory.dcd must exist and be > 0 bytes.
    - energy.csv must exist and have ≥1 data row.
    - topology.pdb must exist and be > 0 bytes.
    """
    traj = output_dir / FileNames.TRAJECTORY
    energy = output_dir / FileNames.ENERGY
    topo = output_dir / FileNames.TOPOLOGY
    missing: list[str] = []
    empty: list[str] = []
    for label, path in (("trajectory", traj), ("energy", energy), ("topology", topo)):
        if not path.exists():
            missing.append(label)
        elif path.stat().st_size == 0:
            empty.append(label)
    # Energy.csv is text — a 1-byte file is header only and
    # counts as empty for the purpose of scientific output.
    if energy.exists() and energy.stat().st_size > 0:
        data_rows = max(0, len(energy.read_text().strip().splitlines()) - 1)
        if data_rows == 0:
            empty.append("energy (header-only, no data rows)")
    if missing or empty:
        problems = [f"missing {n}" for n in missing] + [f"empty {n}" for n in empty]
        return "Terminal run is missing required artifacts: " + ", ".join(problems)
    return None


__all__ = [
    "Action",
    "FailurePlan",
    "FreshPlan",
    "ResumePlan",
    "RunPlan",
    "SkipPlan",
    "decide",
]
