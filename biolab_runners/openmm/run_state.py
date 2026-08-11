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
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from biolab_runners.openmm.checkpoint import (
    CheckpointSnapshot,
    CompletionStatus,
    InvalidCheckpointError,
    inspect_checkpoint,
    quarantine_stale_checkpoint,
)
from biolab_runners.openmm.paths import FileNames

if TYPE_CHECKING:
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
    """The runner should build a fresh simulation from scratch.

    The ``action`` field is a ``ClassVar`` — it cannot be set via
    the constructor and is fixed to ``Action.FRESH``. The other
    fields are required (no defaults): callers must commit to
    values that actually describe a fresh run.
    """

    action: ClassVar[Action] = Action.FRESH
    start_step: int
    remaining_steps: int

    def __post_init__(self) -> None:
        """Validate FreshPlan invariants."""
        if self.start_step < 0:
            raise ValueError(f"FreshPlan.start_step must be non-negative, got {self.start_step}")
        if self.remaining_steps < 0:
            raise ValueError(
                f"FreshPlan.remaining_steps must be non-negative, got {self.remaining_steps}"
            )


@dataclass(frozen=True)
class ResumePlan:
    """The runner should resume from the saved state file.

    ``action`` is fixed to ``Action.RESUME`` and cannot be
    overridden. All other fields are required and validated in
    ``__post_init__``: a default ``ResumePlan()`` is a
    ``TypeError`` (missing required fields); a
    ``ResumePlan(..., resume_xml="", ...)`` is a ``ValueError``
    (empty state path).
    """

    action: ClassVar[Action] = Action.RESUME
    start_step: int
    remaining_steps: int
    resume_xml: str
    manifest_step: int
    state_file_basename: str

    def __post_init__(self) -> None:
        """Validate ResumePlan invariants."""
        if self.start_step < 0:
            raise ValueError(f"ResumePlan.start_step must be non-negative, got {self.start_step}")
        if self.remaining_steps < 0:
            raise ValueError(
                f"ResumePlan.remaining_steps must be non-negative, got {self.remaining_steps}"
            )
        if not self.resume_xml:
            raise ValueError("ResumePlan.resume_xml must be a non-empty path")
        if not self.state_file_basename:
            raise ValueError("ResumePlan.state_file_basename must be non-empty")
        if self.manifest_step <= 0:
            raise ValueError(f"ResumePlan.manifest_step must be positive, got {self.manifest_step}")
        # Cross-field: ``start_step`` and ``manifest_step`` must agree.
        # ``decide()`` derives both from the same snapshot, so a
        # divergence means either a caller passed mismatched values or
        # a future refactor pulled one of them from a different source.
        # Either way: refuse the construction — the runner cannot
        # resume with state-file step X and accounting step Y.
        if self.start_step != self.manifest_step:
            raise ValueError(
                f"ResumePlan.start_step ({self.start_step}) must equal "
                f"ResumePlan.manifest_step ({self.manifest_step})"
            )
        # Cross-field: ``resume_xml`` is the absolute path the runner
        # passes to ``loadState()``, ``state_file_basename`` is the
        # manifest's record of which state file the saved step lives
        # in. The runner resolves ``resume_xml`` via Path(...).name at
        # runtime to verify the embedded step matches the manifest;
        # the basename must therefore already be the trailing
        # component of ``resume_xml`` — anything else means the
        # runner is being asked to load a path that disagrees with
        # the manifest's recorded filename.
        if Path(self.resume_xml).name != self.state_file_basename:
            raise ValueError(
                f"ResumePlan.resume_xml basename "
                f"({Path(self.resume_xml).name!r}) must equal "
                f"ResumePlan.state_file_basename ({self.state_file_basename!r})"
            )


@dataclass(frozen=True)
class SkipPlan:
    """The run is terminal; the result is fully populated for the runner to copy.

    All fields are populated by :func:`decide` so the runner
    doesn't need to do any artifact validation, terminal
    reconstruction, or string-prefix parsing. The runner reads
    these fields and assigns them to ``SimulationResult``.
    """

    action: ClassVar[Action] = Action.SKIP
    completion: CompletionStatus
    completion_reason: str
    manifest_step: int
    state_file_basename: str
    # Populated artifact paths (runner copies into result).
    trajectory_path: str
    energy_path: str
    topology_path: str
    state_xml_path: str
    total_ns: float
    early_abort: bool
    # ``abort_reason`` is typed ``str`` (NOT ``Optional[str]``). The
    # ``SimulationResult.abort_reason`` field is also ``str`` defaulting
    # to ``""``, and ``to_dict()`` serialises the value unchanged — a
    # ``None`` here would round-trip as ``"abort_reason": null`` in the
    # output JSON, breaking the long-standing contract that normal
    # completions serialise as empty string. ``""`` for normal, populated
    # string for early abort. Cross-field invariants in ``__post_init__``
    # below enforce this.
    abort_reason: str

    def __post_init__(self) -> None:
        """Validate SkipPlan invariants."""
        _check_skip_plan_fields(self)
        _check_skip_plan_completion_matches_flags(self)


def _check_skip_plan_fields(plan: SkipPlan) -> None:
    """Validate the field-level invariants of a ``SkipPlan``.

    Extracted to keep ``__post_init__`` under the C901 complexity
    ceiling. These checks treat each field independently — the
    cross-field invariants (completion vs. early_abort vs.
    abort_reason) live in ``_check_skip_plan_completion_matches_flags``.
    """
    if plan.manifest_step <= 0:
        raise ValueError(f"SkipPlan.manifest_step must be positive, got {plan.manifest_step}")
    if not plan.completion_reason:
        raise ValueError("SkipPlan.completion_reason must be non-empty")
    for name in (
        "state_file_basename",
        "trajectory_path",
        "energy_path",
        "topology_path",
        "state_xml_path",
    ):
        if not getattr(plan, name):
            raise ValueError(f"SkipPlan.{name} must be a non-empty path")
    if plan.total_ns < 0:
        raise ValueError(f"SkipPlan.total_ns must be non-negative, got {plan.total_ns}")
    # ``completion`` is a tagged-union discriminator: only terminal
    # statuses are valid here (IN_PROGRESS would never reach
    # ``_populate_skip_plan_fields``; INVALID_TERMINAL converts to
    # ``FailurePlan`` upstream). Guarding it here makes the type
    # explicit and prevents a future refactor from silently passing
    # a non-terminal status.
    if plan.completion not in (
        CompletionStatus.NORMAL_COMPLETE,
        CompletionStatus.EARLY_ABORT,
    ):
        raise ValueError(
            f"SkipPlan.completion must be NORMAL_COMPLETE or EARLY_ABORT, got {plan.completion!r}"
        )


def _check_skip_plan_completion_matches_flags(plan: SkipPlan) -> None:
    """Validate the cross-field invariants of ``SkipPlan``.

    A ``SkipPlan`` is meant to be a terminal-populated plan that
    the runner copies straight into ``SimulationResult`` — the
    JSON shape downstream consumers see is determined by these
    fields together. The combinations are:

    - EARLY_ABORT + early_abort=True + non-empty abort_reason
    - NORMAL_COMPLETE + early_abort=False + abort_reason=""
    """
    if plan.completion == CompletionStatus.EARLY_ABORT:
        if not plan.early_abort:
            raise ValueError("SkipPlan with EARLY_ABORT completion must have early_abort=True")
        if not plan.abort_reason:
            raise ValueError(
                "SkipPlan with EARLY_ABORT completion must have a non-empty abort_reason"
            )
    if plan.completion == CompletionStatus.NORMAL_COMPLETE:
        if plan.early_abort:
            raise ValueError("SkipPlan with NORMAL_COMPLETE completion must have early_abort=False")
        if plan.abort_reason:
            raise ValueError(
                "SkipPlan with NORMAL_COMPLETE completion must have abort_reason='' "
                "(got non-empty value)"
            )


@dataclass(frozen=True)
class FailurePlan:
    """The runner should not run MD; ``result.error`` is set.

    ``action`` is fixed to ``Action.FAIL_FAST``. ``error`` is
    required and must be non-empty — a FailurePlan without an
    error message would defeat the contract that ``result.error``
    is always set on FAIL_FAST.
    """

    action: ClassVar[Action] = Action.FAIL_FAST
    error: str

    def __post_init__(self) -> None:
        """Validate FailurePlan invariants."""
        if not self.error:
            raise ValueError("FailurePlan.error must be non-empty")


# Tagged union — the runner matches on either the type or the
# ``action`` field. ``action`` is a ``ClassVar`` on each plan
# type and fixed at compile time, so the runner's ``isinstance``
# check is the source of truth (not the value of ``plan.action``,
# which can't be set incorrectly anyway).
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
        The classification path (steps 2-4) raises nothing — every
        error mode is encoded in ``FailurePlan``. This is
        deliberate: the runner's classification step never raises;
        corruption surfaces as ``result.error``, never as an
        unhandled exception.

        The ``force=True`` quarantine step (step 1) is the one
        exception. ``quarantine_stale_checkpoint`` calls
        ``shutil.move`` and ``os.replace`` on the user's filesystem,
        and an ``OSError`` from a permission error, full disk,
        vanished entry, etc. propagates out of ``decide()``. This
        is intentional — a failure to safely quarantine is a
        pre-condition failure (the user asked to discard state and
        the filesystem refused), and silently returning a
        ``FailurePlan`` here would mask the problem and let the
        runner pair a freshly-built System with stale files. The
        runner catches this at the public ``run()`` boundary if it
        needs to convert it to a result-level error.
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
    #
    # ``abort_reason`` is the field that binds this ``SkipPlan`` to
    # the ``SimulationResult.abort_reason`` field (typed ``str``,
    # defaulting to ``""``). We populate it as:
    # - early abort: the validated terminal payload's ``reason``
    #   (already verified non-empty by ``inspect_checkpoint``);
    # - normal completion: ``""`` (the historical contract —
    #   serialises as ``"abort_reason": ""`` in JSON, never ``null``).
    if is_early_abort and payload is not None:
        total_ns = float(payload["production_ns"])
        abort_reason: str = str(payload["reason"])
    else:
        total_ns = _normal_completion_total_ns(snapshot.absolute_step, config)
        abort_reason = ""

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

    Rounded to 6 decimal places (sub-fs precision in ns) for the
    same reason as ``runner.py``: the historical 2-decimal round
    silently dropped sub-100ps simulations. ``config.timestep_fs``
    accepts float values, so the unrounded result can carry
    floating-point artifacts invisible at the typical 2 fs integer
    timestep — 6 decimals is enough to capture one extra digit
    beyond those artifacts without affecting production reporting.
    """
    return round(
        max(0, absolute_step - config.total_equil_steps) * config.timestep_fs / 1e6,
        6,
    )


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


def _artifact_validation_error(output_dir: Path) -> str | None:
    """Verify that a terminal run has its promised outputs.

    ``inspect_checkpoint`` returns the terminal classification
    from the manifest, but a terminal run also needs the
    scientific outputs (trajectory, energy log, topology) to be
    present and usable. A terminal manifest plus state file but
    no trajectory / energy returns a result with paths pointing
    to nonexistent files — silently misleading downstream
    consumers.

    Every failure mode (missing path, zero-byte file, header-only
    energy log, unwritable directory, directory-disguised-as-file,
    binary-corrupted energy log, permission errors) is converted
    into the same shape: a ``FailurePlan``-bound error message.
    This module's contract is that ``decide()`` raises nothing —
    corrupt artifacts are user-visible as ``result.error``, never
    as an unhandled exception.

    Returns:
        ``None`` if all required artifacts are present and
        usable. Otherwise, a human-readable error message
        naming the first missing or unusable artifact.

    Validation:
    - trajectory.dcd must exist, be a regular file, and be > 0 bytes.
    - energy.csv must exist, be a regular file, be non-empty,
      and have ≥1 data row (header-only is rejected).
    - topology.pdb must exist, be a regular file, and be > 0 bytes.
    """
    traj = output_dir / FileNames.TRAJECTORY
    energy = output_dir / FileNames.ENERGY
    topo = output_dir / FileNames.TOPOLOGY
    missing: list[str] = []
    empty: list[str] = []
    unreadable: list[str] = []

    # First pass: presence + size. ``is_file()`` excludes
    # directories and broken symlinks; ``stat()`` is wrapped so a
    # transient ``OSError`` (permissions race, vanished entry on a
    # network FS) is treated like a missing artifact rather than
    # leaking out as an unhandled exception.
    for label, path in (
        ("trajectory", traj),
        ("energy", energy),
        ("topology", topo),
    ):
        _classify_artifact(label, path, missing, empty, unreadable)

    # Second pass: energy.csv must have ≥1 data row.
    # ``read_text()`` can raise ``IsADirectoryError`` (already
    # caught by the first pass — directory check), but a binary
    # file or partially-corrupted text file raises
    # ``UnicodeDecodeError`` which is not an ``OSError`` subclass.
    # Wrap so binary corruption surfaces as a ``FailurePlan``
    # rather than crashing the runner.
    if not missing and not empty and "energy" not in unreadable:
        _classify_energy_content(energy, empty, unreadable)

    if missing or empty or unreadable:
        problems = (
            [f"missing {n}" for n in missing]
            + [f"empty {n}" for n in empty]
            + [f"unreadable {n}" for n in unreadable]
        )
        return "Terminal run is missing required artifacts: " + ", ".join(problems)
    return None


def _classify_artifact(
    label: str,
    path: Path,
    missing: list[str],
    empty: list[str],
    unreadable: list[str],
) -> None:
    """Classify one artifact into missing / empty / unreadable.

    Extracted from ``_artifact_validation_error`` to keep the
    orchestrator under the C901 complexity ceiling. ``is_file()``
    returns False for directories, sockets, broken symlinks, and
    non-existent paths — the ``exists()`` branch distinguishes a
    missing file (returns True) from a directory-disguised-as-file
    (returns False but ``exists()`` is True).
    """
    try:
        if not path.is_file():
            if path.exists():
                unreadable.append(label)
            else:
                missing.append(label)
            return
        if path.stat().st_size == 0:
            empty.append(label)
    except OSError:
        unreadable.append(label)


def _classify_energy_content(
    energy: Path,
    empty: list[str],
    unreadable: list[str],
) -> None:
    """Classify ``energy.csv`` content: ≥1 data row, or header-only.

    A 1-byte energy file passes the size check but is header-only
    by convention; this helper enforces the ≥1-row rule and
    surfaces binary / undecodable content as ``unreadable``
    rather than letting ``UnicodeDecodeError`` leak out of
    ``decide()``.
    """
    try:
        text = energy.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        unreadable.append("energy (binary or undecodable)")
        return
    data_rows = max(0, len(text.strip().splitlines()) - 1)
    if data_rows == 0:
        empty.append("energy (header-only, no data rows)")


__all__ = [
    "Action",
    "FailurePlan",
    "FreshPlan",
    "ResumePlan",
    "RunPlan",
    "SkipPlan",
    "decide",
]
