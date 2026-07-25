"""Run state decision — what should the runner do given the on-disk checkpoint state?

This module owns the **decision** part of an OpenMM MD run: given an
output directory, a configuration, and a ``force`` flag, decide
whether to build a fresh simulation, resume a saved state, skip
because the run is already terminal, or fail fast because the
checkpoint is corrupted or orphaned.

The runner in :mod:`biolab_runners.openmm.runner` is a thin
dispatcher: it calls :func:`decide`, then acts on the returned
:data:`ResumePlan`. Pre-run decision logic (manifest validation,
terminal classification, orphan detection, quarantine-on-force) and
skip-population logic (artifact validation, terminal reconstruction)
live here, not on the runner.

Why a separate module:

- The skip / resume / fresh decision has many branches. Inlining it
  in the runner with private helper methods meant 20+ tests
  bypassed ``run()`` and hit the helpers directly. The
  ``ResumePlan`` dataclass turns the question "what should the
  runner do?" into a single interface that callers and tests can
  cross once.
- The decision is the right size for one module — small enough to
  hold in your head, with clear boundaries to ``checkpoint`` (which
  owns the manifest I/O and atomic save) and ``runner`` (which owns
  the MD mechanics).
- Decision logic and MD mechanics change at different cadences.
  Keeping them separate means a force-field tweak doesn't touch
  the decision tree.

The seam between this module and the runner is the public interface:
:func:`decide`, :func:`populate_skip_result`, :class:`Action`,
:data:`ResumePlan`. Internal helpers
(:func:`_quarantine_stale_files`, :func:`_orphan_state_error`,
:func:`_artifact_validation_error`, :func:`_reconstruct_terminal_payload`)
are part of the module's **internal seam** — they exist so the
module's own tests can exercise specific failure modes without
going past the public interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from biolab_runners.openmm.checkpoint import (
    InvalidCheckpointError,
    is_run_complete,
    load_checkpoint,
    load_terminal_payload,
    production_ns,
    quarantine_stale_checkpoint,
)
from biolab_runners.openmm.paths import FileNames

if TYPE_CHECKING:
    from pathlib import Path

    from biolab_runners.openmm.config import OpenMMConfig, SimulationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class Action(StrEnum):
    """What the runner should do, decided by :func:`decide`.

    - ``FRESH``: no manifest exists (or ``force=True`` quarantined
      one); build the simulation from scratch starting at
      ``config.total_equil_steps``.
    - ``RESUME``: a valid manifest exists, the run is not terminal,
      and the saved state should be loaded. ``start_step`` is the
      absolute step from the manifest, ``remaining_steps`` is the
      production steps still owed.
    - ``SKIP``: the run is terminal (manifest terminal payload or
      normal completion). The runner populates the result from the
      on-disk artifacts and exits without MD.
    - ``FAIL_FAST``: the checkpoint is corrupted (manifest with
      dangling / unsafe / step-mismatched state file) or the
      directory is in an invalid state (orphan state file with no
      manifest). The runner sets ``result.error`` and exits.
    """

    FRESH = "fresh"
    RESUME = "resume"
    SKIP = "skip"
    FAIL_FAST = "fail_fast"


@dataclass(frozen=True)
class ResumePlan:
    """Structured answer to "what should the runner do next?".

    Constructed by :func:`decide`. The runner dispatches on
    ``action`` and reads the relevant fields. The dataclass is
    frozen so a plan cannot be mutated after construction (which
    would let a caller change the runner's behaviour mid-dispatch).

    Fields used per action:

    - ``FRESH``: ``start_step = config.total_equil_steps``,
      ``remaining_steps = config.total_steps``, ``resume_xml = ""``.
    - ``RESUME``: ``start_step = manifest_step``,
      ``remaining_steps = config.total_steps - production_done``,
      ``resume_xml = output_dir / state_file_basename``.
    - ``SKIP``: ``manifest_step``, ``state_file_basename``,
      ``skip_reason`` (the human-readable terminal reason from
      :func:`biolab_runners.openmm.checkpoint.is_run_complete`).
    - ``FAIL_FAST``: ``error`` (the message the runner should copy
      into ``result.error``).
    """

    action: Action
    start_step: int = 0
    remaining_steps: int = 0
    resume_xml: str = ""
    manifest_step: int = 0
    state_file_basename: str = ""
    skip_reason: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decide(
    output_dir: Path,
    config: OpenMMConfig,
    force: bool,
) -> ResumePlan:
    """Inspect the output directory and decide what the runner should do.

    Decision tree (in order):

    1. **Force quarantine**. If ``force=True``, move the manifest,
       energy log, ``early_abort.json``, and every state file into
       ``output_dir/.stale/<UTC>/``. This step is a side effect —
       it makes a fresh build safe by removing anything that could
       pair with a freshly-built topology.

    2. **Load the manifest**. On :class:`InvalidCheckpointError`
       (manifest references a missing / empty / path-traversal /
       step-mismatched state file), return ``FAIL_FAST``.

    3. **Manifest present**. Call
       :func:`biolab_runners.openmm.checkpoint.is_run_complete` to
       classify the run:
       - **Complete** (manifest terminal payload OR normal
         completion): return ``SKIP``.
       - **Invalid terminal** (manifest has a malformed ``terminal``
         field): return ``FAIL_FAST`` with the
         ``invalid_terminal_<reason>`` message.
       - **In progress**: return ``RESUME``.

    4. **No manifest**. If any ``state*.xml`` files remain (legacy
       or v7) — the directory is in an inconsistent state, since
       v7 commits the state file together with the manifest —
       return ``FAIL_FAST``. Otherwise, return ``FRESH``.

    Args:
        output_dir: MD output directory.
        config: OpenMMConfig (used for target step in
            normal-completion classification).
        force: ``runner.run(force=True)`` semantic — quarantine
            stale files before inspecting.

    Returns:
        A :data:`ResumePlan`. The runner dispatches on ``plan.action``.

    Raises:
        Nothing — all error modes are encoded in
        ``ResumePlan.error`` with action ``FAIL_FAST``. This is
        deliberate: the runner's decision should never raise;
        corruption surfaces as ``result.error``, never as an
        unhandled exception.
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
        checkpoint = load_checkpoint(output_dir)
    except InvalidCheckpointError as exc:
        return ResumePlan(
            action=Action.FAIL_FAST,
            error=(
                f"{exc} The checkpoint is in an unrecoverable state; "
                f"re-run with force=True to discard it."
            ),
        )

    manifest_step = checkpoint.absolute_step
    state_file_basename = checkpoint.state_file_basename

    if manifest_step > 0:
        return _decide_with_manifest(
            output_dir=output_dir,
            config=config,
            manifest_step=manifest_step,
            state_file_basename=state_file_basename,
        )

    orphan_error = _orphan_state_error(output_dir)
    if orphan_error is not None:
        return ResumePlan(action=Action.FAIL_FAST, error=orphan_error)

    return ResumePlan(
        action=Action.FRESH,
        start_step=config.total_equil_steps,
        remaining_steps=config.total_steps,
    )


def populate_skip_result(
    plan: ResumePlan,
    output_dir: Path,
    config: OpenMMConfig,
    result: SimulationResult,
) -> str | None:
    """Populate ``result`` for an idempotent skip; return artifact error if invalid.

    Skips are idempotent re-runs of a completed (or early-aborted)
    production. The manifest has already recorded the terminal
    decision; the runner just needs to populate the result's
    artifact paths and the production ns / abort reason fields.

    Validation gates (each must pass):

    - ``trajectory.dcd`` exists and is non-empty.
    - ``energy.csv`` exists, is non-empty, and has at least one
      data row (header-only counts as empty).
    - ``topology.pdb`` exists and is non-empty.

    A missing or empty artifact produces a clear ``result.error``
    rather than silently returning success with paths pointing to
    nonexistent files. Terminality and artifact validity are
    separate questions — the manifest records the decision, but
    the user-facing artefacts must also be there.

    For normal-completion terminals, ``result.total_ns`` is set
    from the v10 BLOCKER #3 invariant (``max(0, absolute_step -
    total_equil_steps) * timestep_fs / 1e6``). For manifest-terminal
    payloads (``early_abort``), the early-abort fields are populated
    from the manifest's terminal payload.

    Args:
        plan: The :data:`ResumePlan` returned by :func:`decide`.
            Must have ``action == SKIP``.
        output_dir: MD output directory.
        config: OpenMMConfig (used for production_ns computation).
        result: SimulationResult to populate.

    Returns:
        ``None`` if all required artifacts are present and the
        result was populated successfully. Otherwise, a
        human-readable error message naming the first missing or
        unusable artifact — the caller should set this on
        ``result.error`` and exit.

    Raises:
        InvalidCheckpointError: If the manifest references a
            dangling / unsafe / step-mismatched state file when
            loading the terminal payload (see
            :func:`biolab_runners.openmm.checkpoint.load_terminal_payload`).
    """
    artifact_error = _artifact_validation_error(output_dir)
    if artifact_error is not None:
        return artifact_error

    result.trajectory_path = str(output_dir / FileNames.TRAJECTORY)
    result.energy_path = str(output_dir / FileNames.ENERGY)
    result.topology_path = str(output_dir / FileNames.TOPOLOGY)
    result.state_xml_path = str(output_dir / plan.state_file_basename)

    if plan.skip_reason.startswith("normal_completion_step_"):
        # v10 BLOCKER #3: total_ns is PRODUCTION ns.
        result.total_ns = round(production_ns(plan.manifest_step, config), 2)

    _reconstruct_terminal_payload(result, output_dir, config, plan.skip_reason)
    return None


# ---------------------------------------------------------------------------
# Internal helpers (module's internal seam — used by this module's tests
# for specific failure modes; not part of the public surface).
# ---------------------------------------------------------------------------


def _decide_with_manifest(
    *,
    output_dir: Path,
    config: OpenMMConfig,
    manifest_step: int,
    state_file_basename: str,
) -> ResumePlan:
    """Classify a manifest-present run as SKIP, FAIL_FAST, or RESUME.

    Extracted from :func:`decide` so the decision tree's manifest
    branch is testable in isolation. The runner never calls this
    directly.
    """
    target_step = config.total_equil_steps + config.total_steps
    complete, reason = is_run_complete(output_dir, config)
    if complete:
        logger.info(
            "Skipping MD — run is already terminal (%s) at step %d",
            reason,
            manifest_step,
        )
        return ResumePlan(
            action=Action.SKIP,
            manifest_step=manifest_step,
            state_file_basename=state_file_basename,
            skip_reason=reason,
        )

    if reason.startswith("invalid_terminal_"):
        # v13 BLOCKER: a malformed terminal payload at the target
        # step MUST fail loudly — the manifest is in an ambiguous
        # state and the runner must not fall through to a resume
        # that would either reclassify the run as normal completion
        # (overwriting the bad terminal) or commit a new terminal
        # without addressing the schema error. The user must
        # investigate via ``force=True``.
        return ResumePlan(
            action=Action.FAIL_FAST,
            manifest_step=manifest_step,
            state_file_basename=state_file_basename,
            error=(
                f"Manifest carries a malformed terminal payload at "
                f"step {manifest_step} (reason={reason!r}). The "
                f"user attempted to record a terminal decision but "
                f"the schema is wrong; the run is in an ambiguous "
                f"state. Re-run with force=True to quarantine the "
                f"manifest and start fresh."
            ),
        )

    logger.info(
        "Resuming from checkpoint at step %d (%.2f ns of %d needed)",
        manifest_step,
        manifest_step * config.timestep_fs / 1e6,
        target_step,
    )
    return ResumePlan(
        action=Action.RESUME,
        start_step=manifest_step,
        remaining_steps=max(
            0, config.total_steps - max(0, manifest_step - config.total_equil_steps)
        ),
        resume_xml=str(output_dir / state_file_basename),
        manifest_step=manifest_step,
        state_file_basename=state_file_basename,
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

    ``is_run_complete`` returns true based on the manifest's
    terminal decision, but a terminal run also needs the
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
        return (
            "Terminal run is missing required artifacts: "
            + ", ".join(problems)
            + ". The checkpoint recorded completion but the "
            "scientific outputs are not usable; the user must "
            "investigate the prior run (e.g. disk full during "
            "trajectory write) and re-run with force=True."
        )
    return None


def _reconstruct_terminal_payload(
    result: SimulationResult,
    output_dir: Path,
    config: OpenMMConfig,
    reason: str,
) -> None:
    """Populate ``result`` from the manifest's terminal payload.

    For normal-completion terminals (no abort marker) the function
    is a no-op — the result's artifact paths and
    ``state_xml_path`` are already populated by the caller. For
    manifest-terminal payloads (early_abort type), the function
    populates ``early_abort=True``, ``abort_reason``, and
    ``total_ns`` from the payload.

    If the manifest claims terminal but the payload fails to
    re-validate, ``result.early_abort`` is left False and a
    warning is logged so the caller notices the inconsistency.
    """
    if not reason.startswith("manifest_terminal_"):
        return
    payload = load_terminal_payload(output_dir, config)
    if payload is None:
        logger.warning(
            "Run classified as terminal via manifest payload "
            "(reason=%s) but the payload failed to re-validate; "
            "result.early_abort is left False",
            reason,
        )
        return
    if payload.get("type") == "early_abort":
        result.early_abort = True
        result.abort_reason = str(payload.get("reason", ""))
        # v10 BLOCKER #3: production_ns is the canonical
        # value (computed from absolute_step - total_equil_steps).
        try:
            result.total_ns = float(str(payload.get("production_ns", 0.0)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            result.total_ns = 0.0
        logger.info(
            "Reconstructing early-abort result: reason=%r production_ns=%.2f",
            result.abort_reason,
            result.total_ns,
        )


__all__ = [
    "Action",
    "ResumePlan",
    "decide",
    "populate_skip_result",
]


# Defensive: catch any leaked ``Any`` to satisfy pyright's strict mode
# without a TYPE_CHECKING import (the runner imports ``ResumePlan``
# via this module).
_ = Any
