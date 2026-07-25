"""Checkpoint domain: manifest, state file, atomic save, orphan GC, quarantine, terminal.

This module owns the entire checkpoint lifecycle of an OpenMM MD run:

- :func:`atomic_save_checkpoint` — commits a state file and the manifest
  in one transaction. The ``os.replace`` on the manifest is the single
  atomic commit point; a crash before leaves the previous checkpoint
  active, a crash after leaves the new one active.

- :func:`quarantine_stale_checkpoint` — moves the manifest, energy log,
  early-abort marker, and every state file into a timestamped
  ``.stale/<UTC>/`` subdirectory. Called by ``runner.run(force=True)``
  before building fresh so a stale state cannot be paired with a
  freshly-built topology.

- :func:`load_checkpoint` — reads the manifest and validates the
  referenced state file. Raises :class:`InvalidCheckpointError` on
  any safety gate failure (path traversal, missing file, empty file,
  step mismatch). Returns a :class:`LoadedCheckpoint` so callers read
  named fields instead of unpacking tuples.

- :func:`is_run_complete` — the authoritative completion oracle. Tri-state
  terminal check: a present-but-invalid ``terminal`` payload returns
  ``(False, "invalid_terminal_<reason>")`` and does NOT fall back to
  the inferred normal-completion heuristic.

- :func:`load_terminal_payload` — reads and validates the manifest's
  ``terminal`` field for the early-abort result reconstruction path.

- :func:`production_ns` — the v10 BLOCKER #3 invariant:
  ``max(0, absolute_step - total_equil_steps) * timestep_fs / 1e6``.
  Equilibration steps are protocol setup, not scientific progress.

The seam between this module and the runner is the public interface.
Internal helpers (``_parse_manifest``, ``_validate_state_file_reference``,
``_parse_state_filename_step``, ``_gc_orphan_states``,
``_validate_terminal_payload``, ``_production_steps``) are part of
the module's **internal seam** — they exist so this module's own
tests can exercise specific failure modes without resorting to
setup that goes past the public interface. They are NOT exported.

The structured :class:`CompletionStatus` enum is the cross-module
terminal-classification protocol — downstream code branches on the
enum value, not on string-prefixes of the ``reason`` field. The old
string-prefix protocol (``reason.startswith("normal_completion_")``
etc.) was a documented seam leak; this version surfaces the
classification as a stable enum.

The AGENTS.md invariants around checkpoint / manifest / terminal /
quarantine / orphan GC / resume safety all describe this module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.openmm.paths import FileNames

if TYPE_CHECKING:
    from biolab_runners.openmm.config import OpenMMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedCheckpoint:
    """The parsed manifest plus the validated state-file reference.

    The single entry point for "what is the saved state of this run?"
    Callers read named fields instead of unpacking tuples.

    Fields:
        absolute_step: The absolute OpenMM step the saved state is at
            (the manifest's ``step`` field). Zero means no resumable
            checkpoint was found.
        state_file_basename: The basename of the referenced state file
            (e.g. ``state.42000_12345_170000000.xml``). Empty when no
            resumable checkpoint exists.
        last_record: The full last record dict from the manifest, exposed
            for callers that need to inspect arbitrary fields (the v10
            ``terminal`` payload is surfaced through the convenience
            fields below; ``last_record`` is for anything else).
        is_terminal: True iff ``last_record.terminal`` validated against
            the v11 schema (``type == "early_abort"``, strict-positive-int
            ``step`` equal to ``manifest_step``, non-empty ``reason``).
            A malformed terminal payload leaves ``is_terminal=False``
            and is surfaced separately via :func:`is_run_complete` —
            the runner must fail fast on it, not silently treat the run
            as in-progress or completed.
        terminal_reason: The human-readable reason string when
            ``is_terminal`` is True; ``None`` otherwise.
    """

    absolute_step: int = 0
    state_file_basename: str = ""
    last_record: dict[str, Any] = field(default_factory=dict)
    is_terminal: bool = False
    terminal_reason: str | None = None


class CompletionStatus(StrEnum):
    """Structured terminal classification returned by :func:`is_run_complete`.

    Surfaced via :class:`CheckpointSnapshot` so callers branch on
    a stable enum rather than parsing reason-string prefixes.

    Reading the manifest once and producing one of these values lets
    downstream code branch on a stable enum rather than parsing
    reason-string prefixes (which were the previous cross-module
    protocol and a documented seam leak).
    """

    IN_PROGRESS = "in_progress"
    """The run is not yet terminal. ``resume_xml`` is the file to load."""
    NORMAL_COMPLETE = "normal_complete"
    """The manifest reached the target step without a terminal payload."""
    EARLY_ABORT = "early_abort"
    """The manifest carries a valid ``terminal`` payload (early abort)."""
    INVALID_TERMINAL = "invalid_terminal"
    """The manifest has a present-but-invalid ``terminal`` field.

    Per the v11 contract, treat as in-progress with a specific
    failure reason. The runner MUST NOT fall back to the
    normal-completion heuristic. Today the runner converts this
    to FAIL_FAST — the run is in an ambiguous state and the user
    must investigate via ``force=True``.
    """


@dataclass(frozen=True)
class CheckpointSnapshot:
    """Coherent result of inspecting the on-disk checkpoint state.

    Single read of the manifest + state file. The completion
    classification, terminal payload, and produce-tolerant fields
    are all derived from a single manifest snapshot — no race
    window between fresh builds of the same plan.

    Use :func:`inspect_checkpoint` as the canonical entry point.
    A default snapshot (``IN_PROGRESS`` with empty fields) is
    returned when no manifest exists.

    Fields:
        absolute_step: The absolute OpenMM step the saved state is at,
            or 0 if no manifest exists.
        state_file_basename: The basename of the referenced state file.
        last_record: The last record dict from the manifest, exposed
            for callers that need to inspect arbitrary fields.
        completion: The structured terminal classification.
        completion_reason: Human-readable reason for the completion
            classification (e.g. ``"normal_completion_step_50_200_000"``,
            ``"manifest_terminal_early_abort_step_5000000"``,
            ``"in_progress"``, ``"invalid_terminal_<reason>"``).
        terminal_payload: The validated manifest terminal payload
            dict, populated only when ``completion == EARLY_ABORT``.
            Structured fields: ``step``, ``type``, ``reason``,
            ``production_ns``, plus optional ``gate``, ``target``,
            ``peptide_id`` when well-typed.
    """

    absolute_step: int = 0
    state_file_basename: str = ""
    last_record: dict[str, Any] = field(default_factory=dict)
    completion: CompletionStatus = CompletionStatus.IN_PROGRESS
    completion_reason: str = "in_progress"
    terminal_payload: dict[str, Any] | None = None


class InvalidCheckpointError(Exception):
    """Raised when ``checkpoint.json`` references a state file that is invalid.

    The runner converts this into a ``result.error`` and refuses to proceed —
    the user must invoke ``runner.run(force=True)`` to discard the corrupt
    checkpoint (the quarantine moves the manifest + state files into
    ``.stale/<UTC>/`` together, so the next non-forced invocation starts from
    a fresh build).
    """


# ---------------------------------------------------------------------------
# Pattern constants (internal — used by validation helpers)
# ---------------------------------------------------------------------------


# Pattern for state file basenames accepted by the manifest. The
# legacy v6 format is a bare ``state.xml``; the v7 format is the
# generation-versioned ``state.<step>_<pid>_<nanos>.xml``. The
# validator below rejects anything else (path separators, weird
# extensions, empty strings) so the resume path cannot be tricked
# into loading a non-state file or escaping the output directory.
_STATE_FILENAME_RE = re.compile(r"^state(\.xml|\.\d+_\d+_\d+\.xml)$")

# Pattern for the embedded step in a v7 state filename. The step is
# the FIRST digit group; the pid and nanos follow. Captured by
# :func:`_parse_state_filename_step` for the manifest-step ↔
# state-filename equality check (v10 BLOCKER #1).
_V7_STATE_STEP_RE = re.compile(r"^state\.(\d+)_\d+_\d+\.xml$")


# ---------------------------------------------------------------------------
# Quarantine — moved from system_builder.py
# ---------------------------------------------------------------------------


# Files that together describe a resumable run. The manifest
# (``checkpoint.json``) is the source of truth for the saved step
# and the state file to load. ``energy.csv`` is also kept in the
# quarantine because it carries the per-step reporter rows and the
# user may want to inspect a stale trajectory step-by-step.
#
# ``early_abort.json`` is the generation-scoped terminal marker
# written when the offline-mdtraj gate fires (see
# :meth:`OpenMMRunner._write_abort_metadata`). v9: a forced fresh
# run must retire this marker together with the rest, otherwise a
# stale marker from a previous abort run will be re-read by
# :func:`is_run_complete` and mis-classify a mid-production
# checkpoint as terminal. The marker is generation-scoped by the
# manifest step — see the ``force=True quarantine`` rule in AGENTS.md.
RESUMABLE_FILES: tuple[str, ...] = (
    FileNames.CHECKPOINT_JSON,
    FileNames.ENERGY,
    FileNames.EARLY_ABORT_JSON,
)
# Glob pattern for state files. Matches both legacy ``state.xml``
# (from pre-v7 runs) and the v7 ``state.<step>_<pid>_<nanos>.xml``.
RESUMABLE_STATE_GLOB = "state*.xml"


def quarantine_stale_checkpoint(output_dir: Path) -> list[Path]:
    """Move resumable files into a timestamped ``.stale/`` subdirectory.

    Used by ``runner.run(force=True)`` to ensure the next non-forced
    invocation cannot pair a stale state file with a freshly-built
    topology. The v7 save format uses generation-versioned state
    files (``state.<step>_<pid>_<nanos>.xml``) referenced by the
    manifest (``checkpoint.json``) — the manifest is the source of
    truth for the saved step AND the file to load. So the quarantine
    must move the manifest, the energy log, AND every state file
    (both legacy ``state.xml`` and the v7 ``state.<gen>.xml``).

    Returns the list of files actually moved (those that existed).
    An empty output dir produces an empty list and no ``.stale/``
    directory is created — there's nothing to quarantine.

    Args:
        output_dir: Directory holding the stale checkpoint files.

    Returns:
        List of paths (inside the new ``.stale/<timestamp>/``
        directory) for the files that were moved.
    """
    moved: list[Path] = []
    # Collect every file that participates in the resume contract:
    # the manifest, the energy log, and any state file (legacy or
    # v7 generation-versioned).
    existing: list[str] = []
    for name in RESUMABLE_FILES:
        if (output_dir / name).exists():
            existing.append(name)
    for state_file in output_dir.glob(RESUMABLE_STATE_GLOB):
        existing.append(state_file.name)
    if not existing:
        return moved

    # Use UTC + microsecond resolution + PID to avoid filename
    # collisions on rapid retries (e.g. a CI runner failing and
    # immediately re-invoking force=True within the same second).
    # The microsecond + PID combination is unique within a single
    # host; second-resolution alone was insufficient because two
    # concurrent invocations in the same second would race the
    # mkdir(parents=True, exist_ok=False) below.
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S_%f") + f"_{os.getpid()}"
    stale_dir = output_dir / ".stale" / ts
    stale_dir.mkdir(parents=True, exist_ok=False)

    for name in existing:
        src = output_dir / name
        dst = stale_dir / name
        shutil.move(str(src), str(dst))
        moved.append(dst)
        logger.info("Quarantined stale checkpoint file: %s -> %s", src, dst)

    return moved


# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------


def atomic_save_checkpoint(
    simulation: object,
    output_dir: Path,
    absolute_step: int,
    *,
    terminal: dict[str, Any] | None = None,
) -> str:
    """Atomically commit a state file plus the manifest as one transaction.

    Design (v7, generation-versioned state files; v10 atomic terminal):

    1. Write a uniquely-named state file directly:
       ``state.<step>_<pid>_<nanos>.xml``. The filename is the
       file's identity — there is no canonical ``state.xml``. The
       filename includes the step and nanosecond timestamp so two
       concurrent saves from the same process produce distinct
       files (no collisions).

    2. Write the manifest to a temp file with the new content
       referencing the state file by basename, then atomically
       ``os.replace`` it to ``checkpoint.json``. THIS is the
       single atomic commit point.

       v10 BLOCKER #2: if ``terminal`` is provided, it is embedded
       into the manifest's last record under a ``terminal`` key.
       The terminal status therefore commits together with the
       state file in the same ``os.replace`` — a crash between
       the two cannot leave a resumable checkpoint whose terminal
       decision was already made. A separate ``early_abort.json``
       is written AFTER this function returns (in
       :meth:`OpenMMRunner._write_abort_metadata`) for downstream
       consumers — it is a derived file, not authoritative.

    3. Garbage-collect any ``state.*.xml`` that is no longer
       referenced by the manifest.

    Crash semantics:
    - Crash before the manifest rename → the previous manifest is
      still active, the new state file is an orphan (GC'd next run).
      The resume path loads the previous (coherent) state.
    - Crash after the manifest rename → the new manifest is active
      and references the new state file. The resume path loads the
      new (coherent) state.
    - Crash mid-``saveState`` → the new state file may be partial
      or missing, but the manifest still references the previous
      state file. The resume path loads the previous state.

    The v6 design used two ``os.replace`` calls (one for state.xml,
    one for the manifest) — those are individually atomic but the
    pair is not. A crash between the two renames leaves a new state
    paired with an old manifest; the resume path accepts both
    because it validates only that the state exists and the manifest
    has a positive step. The v7 design fixes that by making the
    manifest rename the single commit point.

    Args:
        simulation: OpenMM Simulation exposing ``saveState(path)``.
        output_dir: Directory holding the manifest and the
            generation-versioned state files.
        absolute_step: The ABSOLUTE OpenMM step the saved state
            corresponds to. This is what the runner computed as
            ``start_step + steps_done`` (where ``start_step`` is
            the absolute step the simulation was at when the
            production loop started: ``total_equil_steps`` for
            fresh runs, ``manifest_step`` for resumed runs). The
            v6 protocol wrote the invocation-local ``steps_done``
            instead, which silently broke accounting on resumes.
        terminal: Optional dict — when provided, embedded into the
            manifest record under ``terminal``. The caller is
            responsible for ensuring ``terminal["step"] ==
            absolute_step`` (enforced by
            :func:`is_run_complete`).

    Returns:
        The basename of the saved state file. The runner uses
        this to populate ``result.state_xml_path``.
    """
    pid = os.getpid()
    nanos = time.time_ns()
    state_basename = f"state.{absolute_step}_{pid}_{nanos}.xml"
    state_path = output_dir / state_basename

    # Write the state file directly. The unique filename eliminates
    # the need for a temp+rename on the state file itself — there
    # is nothing to overwrite. OpenMM's ``saveState`` writes the
    # XML and closes the file atomically (per POSIX write+close).
    simulation.saveState(str(state_path))  # type: ignore[union-attr]

    # Manifest — THIS is the single atomic commit point. The temp
    # file is written first, then atomically renamed to the
    # canonical manifest path. If we crash before the rename, the
    # previous manifest is still active. If we crash after the
    # rename, the new manifest references the just-written state.
    manifest_path = output_dir / FileNames.CHECKPOINT_JSON
    record: dict[str, Any] = {"step": absolute_step, "file": state_basename}
    if terminal is not None:
        record["terminal"] = terminal
    manifest_payload = {"records": [record]}
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + f".tmp.{pid}.{absolute_step}")
    manifest_tmp.write_text(json.dumps(manifest_payload))
    os.replace(str(manifest_tmp), str(manifest_path))

    # Garbage-collect orphan state files (best-effort, must not
    # raise — the save above already succeeded).
    _gc_orphan_states(output_dir)

    logger.info(
        "Atomic checkpoint: state=%s manifest=checkpoint.json step=%d terminal=%s",
        state_basename,
        absolute_step,
        "yes" if terminal is not None else "no",
    )
    return state_basename


def _gc_orphan_states(output_dir: Path) -> None:
    """Remove ``state.*.xml`` files not referenced by the manifest.

    After each atomic save, the manifest references exactly one
    state file. Any other ``state.<gen>.xml`` in the directory is
    an orphan from a previous interrupted save — safe to delete
    since the resume path would never load it.

    Failures are logged but not raised (orphan cleanup must not
    interfere with the save that just succeeded).

    Uses :func:`_parse_manifest` (the same internal helper the
    resume path uses) instead of re-implementing manifest parsing
    here, so a schema change only needs to be reflected in one
    place.

    Args:
        output_dir: Directory holding the manifest and state files.
    """
    parsed = _parse_manifest(output_dir)
    if parsed is None:
        return
    _step, _last, active_file = parsed
    if not active_file:
        return

    for state_file in output_dir.glob("state.*.xml"):
        if state_file.name == active_file:
            continue
        try:
            state_file.unlink()
        except OSError as exc:
            logger.warning("Could not GC orphan state %s: %s", state_file, exc)


# ---------------------------------------------------------------------------
# Manifest read / validate / load
# ---------------------------------------------------------------------------


def load_checkpoint(output_dir: Path) -> LoadedCheckpoint:
    """Load the saved step AND state file from the atomic-save manifest.

    The manifest (``checkpoint.json``) is the ONLY authoritative
    source for the saved state's step and the file to load. It
    references a state file by basename (e.g.
    ``state.700000_12345_1700000000000000000.xml``). The resume
    flow reads both the step and the file from the manifest, then
    loads the state file via ``simulation.loadState``.

    The previous design saved only ``state.xml`` and inferred the
    saved step from the last row of ``energy.csv`` — unsafe because
    the two cadences differ by orders of magnitude (energy.csv at
    ~10 ps, state.xml at ~2 hr). The v7 fix commits the state file
    and the manifest as a single atomic transaction: the manifest
    rename is the only atomic commit point; the state file's
    filename uniquely identifies it, so any state file not
    referenced by the manifest is an orphan that can be safely
    garbage-collected.

    v10 BLOCKER #1: the manifest's ``step`` MUST equal the step
    encoded in the v7 state filename. A mismatch raises
    :class:`InvalidCheckpointError`.

    v10 SUGGESTION: malformed manifests (root is a list, records
    is missing/empty, last record is not a mapping, ``step`` is
    ``null``, ``file`` is missing) are tolerated as "no resumable
    checkpoint" — they do NOT crash the runner. The returned
    :class:`LoadedCheckpoint` carries ``absolute_step=0`` and
    ``state_file_basename=""`` for that case.

    Args:
        output_dir: MD output directory.

    Returns:
        A :class:`LoadedCheckpoint` carrying ``absolute_step``,
        ``state_file_basename``, ``last_record``, ``is_terminal``,
        and ``terminal_reason``. The ``is_terminal`` flag is set
        when ``last_record.terminal`` validates against the v11
        schema; a malformed terminal payload leaves ``is_terminal``
        False and is surfaced separately via :func:`is_run_complete`
        (so the runner fails fast instead of silently treating the
        run as in-progress or completed).

    Raises:
        InvalidCheckpointError: If the manifest references a state
            file that is not a basename, doesn't match the expected
            pattern, doesn't exist, or is empty. Also raised when
            the manifest's step doesn't match the step encoded in
            the v7 state filename (v10 BLOCKER #1). The runner
            catches this and surfaces it as a user-facing error.
    """
    parsed = _parse_manifest(output_dir)
    if parsed is None:
        return LoadedCheckpoint()
    step, last_record, state_file = parsed
    terminal_reason = _validate_terminal_payload(step, last_record)
    return LoadedCheckpoint(
        absolute_step=step,
        state_file_basename=state_file,
        last_record=last_record,
        is_terminal=terminal_reason is not None,
        terminal_reason=terminal_reason,
    )


def inspect_checkpoint(output_dir: Path, config: OpenMMConfig) -> CheckpointSnapshot:
    """Read the manifest ONCE and return a fully-classified snapshot.

    Replaces the multi-call pattern of
    ``load_checkpoint() + is_run_complete() + load_terminal_payload()``
    with a single coherent read. Callers that build a decision
    plan from the on-disk state should use this — multiple reads
    race against a concurrent commit that lands between them.

    The snapshot's ``completion`` is one of four
    :class:`CompletionStatus` values, matching the cross-module
    contract (the previous string-prefix protocol was a noted
    seam leak).

    Decision tree on the snapshot:

    - No manifest → ``IN_PROGRESS`` with empty fields.
    - Manifest step ≥ target → ``NORMAL_COMPLETE``.
    - Manifest has a valid ``terminal`` payload → ``EARLY_ABORT``,
      with the validated payload in ``terminal_payload``.
    - Manifest has a present-but-invalid ``terminal`` payload →
      ``INVALID_TERMINAL``. The runner converts this to FAIL_FAST
      (the v11 contract — never resume, never fall back to normal
      completion).
    - Otherwise → ``IN_PROGRESS``.

    Args:
        output_dir: MD output directory.
        config: OpenMMConfig (used for the target step in normal
            completion comparison and the production_ns computation
            in the terminal payload).

    Returns:
        A :class:`CheckpointSnapshot`. The default snapshot
        (``IN_PROGRESS`` with empty fields) is returned when no
        manifest exists.

    Raises:
        InvalidCheckpointError: If the manifest references a
            dangling / unsafe / step-mismatched state file.
    """
    checkpoint = load_checkpoint(output_dir)
    if checkpoint.absolute_step <= 0:
        return CheckpointSnapshot(
            absolute_step=0,
            state_file_basename="",
            last_record={},
            completion=CompletionStatus.IN_PROGRESS,
            completion_reason="in_progress",
            terminal_payload=None,
        )

    manifest_step = checkpoint.absolute_step
    last_record = checkpoint.last_record

    # Manifest terminal payload — three-way terminal classification.
    # ABSENT (no terminal key) → IN_PROGRESS, caller falls back to
    # normal completion. INVALID (present but fails schema) →
    # INVALID_TERMINAL with a specific reason. VALID → EARLY_ABORT.
    raw_terminal = last_record.get("terminal")
    if raw_terminal is not None:
        return _classify_manifest_terminal(
            raw_terminal=raw_terminal,
            manifest_step=manifest_step,
            state_file_basename=checkpoint.state_file_basename,
            last_record=last_record,
            config=config,
        )

    # No terminal field — fall back to normal completion heuristic.
    target_step = config.total_equil_steps + config.total_steps
    if manifest_step >= target_step:
        return CheckpointSnapshot(
            absolute_step=manifest_step,
            state_file_basename=checkpoint.state_file_basename,
            last_record=last_record,
            completion=CompletionStatus.NORMAL_COMPLETE,
            completion_reason=f"normal_completion_step_{manifest_step}_of_{target_step}",
            terminal_payload=None,
        )
    return CheckpointSnapshot(
        absolute_step=manifest_step,
        state_file_basename=checkpoint.state_file_basename,
        last_record=last_record,
        completion=CompletionStatus.IN_PROGRESS,
        completion_reason="in_progress",
        terminal_payload=None,
    )


def _classify_manifest_terminal(
    *,
    raw_terminal: dict[str, Any] | str | int | float | bool | None,
    manifest_step: int,
    state_file_basename: str,
    last_record: dict[str, Any],
    config: OpenMMConfig,
) -> CheckpointSnapshot:
    """Classify a manifest's ``terminal`` field into a snapshot.

    Helper for :func:`inspect_checkpoint`. Returns either a
    snapshot with ``completion == INVALID_TERMINAL`` and a
    specific ``completion_reason``, or a snapshot with
    ``completion == EARLY_ABORT`` and the validated payload.

    The v11 contract: ABSENT → IN_PROGRESS (caller falls back to
    normal completion); INVALID → INVALID_TERMINAL (caller MUST
    NOT fall back to normal completion); VALID → EARLY_ABORT.
    """
    if not isinstance(raw_terminal, dict):
        return CheckpointSnapshot(
            absolute_step=manifest_step,
            state_file_basename=state_file_basename,
            last_record=last_record,
            completion=CompletionStatus.INVALID_TERMINAL,
            completion_reason="invalid_terminal_not_dict",
            terminal_payload=None,
        )
    raw_step = raw_terminal.get("step")
    if not isinstance(raw_step, int) or isinstance(raw_step, bool) or raw_step <= 0:
        return CheckpointSnapshot(
            absolute_step=manifest_step,
            state_file_basename=state_file_basename,
            last_record=last_record,
            completion=CompletionStatus.INVALID_TERMINAL,
            completion_reason="invalid_terminal_step_invalid_type",
            terminal_payload=None,
        )
    if raw_step != manifest_step:
        return CheckpointSnapshot(
            absolute_step=manifest_step,
            state_file_basename=state_file_basename,
            last_record=last_record,
            completion=CompletionStatus.INVALID_TERMINAL,
            completion_reason="invalid_terminal_step_mismatch",
            terminal_payload=None,
        )
    if raw_terminal.get("type") != "early_abort":
        return CheckpointSnapshot(
            absolute_step=manifest_step,
            state_file_basename=state_file_basename,
            last_record=last_record,
            completion=CompletionStatus.INVALID_TERMINAL,
            completion_reason="invalid_terminal_type_unsupported",
            terminal_payload=None,
        )
    reason = raw_terminal.get("reason")
    if not isinstance(reason, str) or not reason:
        return CheckpointSnapshot(
            absolute_step=manifest_step,
            state_file_basename=state_file_basename,
            last_record=last_record,
            completion=CompletionStatus.INVALID_TERMINAL,
            completion_reason="invalid_terminal_reason_empty",
            terminal_payload=None,
        )
    # Valid terminal payload — build the normalised payload.
    payload: dict[str, Any] = {
        "step": raw_step,
        "type": str(raw_terminal.get("type", "unknown")),
        "reason": str(raw_terminal.get("reason", "")),
        "production_ns": _production_steps(raw_step, config.total_equil_steps)
        * config.timestep_fs
        / 1e6,
    }
    for opt in ("gate", "target", "peptide_id"):
        val = raw_terminal.get(opt)
        if isinstance(val, str):
            payload[opt] = val
    return CheckpointSnapshot(
        absolute_step=manifest_step,
        state_file_basename=state_file_basename,
        last_record=last_record,
        completion=CompletionStatus.EARLY_ABORT,
        completion_reason=f"manifest_terminal_{payload['type']}_step_{raw_step}",
        terminal_payload=payload,
    )


def _validate_terminal_payload(manifest_step: int, last_record: dict[str, Any]) -> str | None:
    """Validate the manifest's terminal field; return the canonical reason on success.

    Returns:
        ``"manifest_terminal_<type>_step_<step>"`` when the manifest
        has a ``terminal`` field that validates against the v11 schema
        (``type == "early_abort"``, strict-positive-int ``step`` equal
        to ``manifest_step``, non-empty ``reason``).

        ``None`` for ABSENT (no ``terminal`` key) — the run is
        in-progress (or normal-completed, which :func:`is_run_complete`
        determines separately).

        ``None`` for INVALID (present but failing any schema check) —
        the v11 contract says treat as in-progress. The runner
        surfaces the specific failure reason separately via
        :func:`is_run_complete`'s ``invalid_terminal_<field>`` return.
        Callers that need to distinguish absent from invalid check
        ``"terminal" in last_record``.

    Args:
        manifest_step: The step recorded on the manifest's last record.
        last_record: The last record dict from the manifest.
    """
    terminal = last_record.get("terminal")
    if terminal is None:
        return None
    if not isinstance(terminal, dict):
        logger.warning(
            "Manifest has terminal field but it is not a dict; "
            "treating as invalid terminal payload (NOT as "
            "normal completion)"
        )
        return None
    raw_terminal_step = terminal.get("step")
    if (
        not isinstance(raw_terminal_step, int)
        or isinstance(raw_terminal_step, bool)
        or raw_terminal_step <= 0
    ):
        logger.warning(
            "Manifest terminal.step is not a positive int "
            "(got %r); treating as invalid terminal payload",
            raw_terminal_step,
        )
        return None
    if raw_terminal_step != manifest_step:
        logger.warning(
            "Manifest has terminal.step=%d but record step=%d; "
            "treating as invalid terminal payload (NOT as "
            "normal completion)",
            raw_terminal_step,
            manifest_step,
        )
        return None
    terminal_type = terminal.get("type")
    if terminal_type != "early_abort":
        logger.warning(
            "Manifest terminal.type=%r is not supported "
            "(only 'early_abort' is implemented); treating as "
            "invalid terminal payload",
            terminal_type,
        )
        return None
    reason = terminal.get("reason")
    if not isinstance(reason, str) or not reason:
        logger.warning(
            "Manifest terminal.reason is missing or empty; treating as invalid terminal payload"
        )
        return None
    return f"manifest_terminal_{terminal_type}_step_{raw_terminal_step}"


def _parse_manifest(
    output_dir: Path,
) -> tuple[int, dict[str, Any], str] | None:
    """Internal helper: parse ``checkpoint.json`` with strict validation.

    Returns:
        ``(step, last_record, state_file)`` on success, ``None`` if
        the manifest is missing or structurally invalid. The
        ``last_record`` is the manifest record dict (so callers can
        inspect the optional ``terminal`` payload).

    Structural validation (v10 SUGGESTION):
    - Root must be a JSON object (mapping).
    - ``records`` must be a non-empty list.
    - Final record must be a JSON object.
    - ``step`` must be a positive int.
    - ``file`` must be a non-empty string.

    A structurally invalid manifest returns ``None`` rather than
    raising — callers (the resume path) treat that as "no
    resumable checkpoint" and fail fast on orphan state files.

    Raises:
        InvalidCheckpointError: if the manifest is structurally
            valid BUT the referenced state file fails the path /
            existence / size / step-equality checks. The caller
            catches this and surfaces it as a user-facing error.
    """
    manifest_path = output_dir / FileNames.CHECKPOINT_JSON
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # Structural validation (v10 SUGGESTION).
    if not isinstance(data, dict):
        return None
    records = data.get("records")
    if not isinstance(records, list) or not records:
        return None
    last = records[-1]
    if not isinstance(last, dict):
        return None

    # Field-type validation — ``step`` must be a positive int and
    # ``file`` must be a non-empty string.
    raw_step = last.get("step")
    raw_file = last.get("file")
    if not isinstance(raw_step, int) or isinstance(raw_step, bool) or raw_step <= 0:
        return None
    if not isinstance(raw_file, str) or not raw_file:
        return None

    # Path / pattern / existence / size validation. The runner
    # catches InvalidCheckpointError and surfaces it.
    _validate_state_file_reference(output_dir, raw_file)

    # v10 BLOCKER #1: manifest step MUST equal the step encoded
    # in the v7 state filename. Legacy ``state.xml`` has no
    # embedded step — for legacy compatibility we skip the
    # equality check (the legacy file can only have been written
    # by a pre-v7 run; pairing a legacy state with a manifest is
    # accepted but logged as a compatibility shim).
    embedded = _parse_state_filename_step(raw_file)
    if embedded is not None and embedded != raw_step:
        raise InvalidCheckpointError(
            f"The manifest step {raw_step} does not match the step "
            f"encoded in {raw_file!r} ({embedded}). The state file was "
            f"saved at step {embedded}; pairing it with a manifest step "
            f"of {raw_step} would silently mismatch the System. Re-run "
            f"with force=True to discard the checkpoint."
        )
    if embedded is None:
        # Legacy state.xml — no embedded step to cross-validate.
        # Log a notice so the user can decide to migrate.
        logger.info(
            "Manifest references legacy state.xml — the filename carries no "
            "embedded step, so the manifest step (%d) is trusted as-is. "
            "Future runs will produce generation-versioned filenames.",
            raw_step,
        )

    return raw_step, last, raw_file


def _validate_state_file_reference(output_dir: Path, state_file: str) -> Path:
    """Validate the state file path encoded in a manifest record.

    Returns the absolute path on success. Raises
    :class:`InvalidCheckpointError` if the reference is unsafe or
    refers to a file that is missing or empty.

    Validation gates (each must pass):

    1. ``state_file`` is a basename (``Path(s).name == s``).
       Rejects absolute paths, ``../`` traversal, ``subdir/foo``.

    2. ``state_file`` matches the expected pattern
       (``state.xml`` or ``state.<digits>_<digits>_<digits>.xml``).
       Rejects typos, unexpected extensions, anything else.

    3. ``(output_dir / state_file).is_file()`` — the file is on
       disk. Rejects dangling manifests (the v6 save would have
       detected this only after ``loadState`` raised).

    4. ``stat().st_size > 0`` — the file is non-empty. Rejects
       truncated state files (OpenMM's ``saveState`` may write
       half a file if the process is killed mid-write).

    Args:
        output_dir: Directory containing the state file.
        state_file: The basename from the manifest record.

    Returns:
        The validated absolute path to the state file.

    Raises:
        InvalidCheckpointError: If any gate fails. The error
            message includes the offending value and the required
            remediation (``force=True`` to discard).
    """
    if not state_file:
        raise InvalidCheckpointError(
            "The manifest references an empty state file path. "
            "Re-run with force=True to discard the checkpoint."
        )
    if Path(state_file).name != state_file:
        raise InvalidCheckpointError(
            f"The manifest references a state file with a path "
            f"that is not a basename: {state_file!r}. Path traversal "
            f"and absolute paths are rejected. Re-run with force=True "
            f"to discard the checkpoint."
        )
    if not _STATE_FILENAME_RE.match(state_file):
        raise InvalidCheckpointError(
            f"The manifest references a state file with an invalid "
            f"name: {state_file!r}. Expected 'state.xml' (legacy) or "
            f"'state.<step>_<pid>_<nanos>.xml' (v7). Re-run with "
            f"force=True to discard the checkpoint."
        )
    state_path = output_dir / state_file
    if not state_path.is_file():
        raise InvalidCheckpointError(
            f"The manifest references a state file that does not "
            f"exist: {state_path}. The run cannot be resumed safely. "
            f"Re-run with force=True to discard the checkpoint."
        )
    if state_path.stat().st_size == 0:
        raise InvalidCheckpointError(
            f"The manifest references an empty state file: {state_path}. "
            f"The state was likely truncated mid-write. Re-run with "
            f"force=True to discard the checkpoint."
        )
    return state_path


def _parse_state_filename_step(state_file: str) -> int | None:
    """Parse the absolute step encoded in a v7 state filename.

    The v7 format is ``state.<step>_<pid>_<nanos>.xml`` — the step
    is the first digit group. Returns the integer step, or None for
    the legacy ``state.xml`` (which carries no step encoding and
    therefore cannot be cross-validated against the manifest's
    ``step`` field).

    Used by :func:`_parse_manifest` to enforce v10 BLOCKER #1:
    the manifest's ``step`` MUST equal the step encoded in the v7
    state filename. A mismatch indicates a corrupt or forged
    checkpoint and must fail fast.

    Args:
        state_file: The basename from the manifest record.

    Returns:
        The embedded step as int, or None if the filename is
        legacy / unparseable.
    """
    m = _V7_STATE_STEP_RE.match(state_file)
    if m is None:
        return None
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Run completion
# ---------------------------------------------------------------------------


def is_run_complete(output_dir: Path, config: OpenMMConfig) -> tuple[bool, str]:
    """Determine whether a production run is terminal.

    Thin wrapper around :func:`inspect_checkpoint` that returns
    the legacy ``(complete, reason)`` tuple shape. New code
    should use :func:`inspect_checkpoint` directly and branch on
    the structured :class:`CompletionStatus` value.

    The returned ``reason`` carries the diagnostic detail for
    logging — the runner's policy is driven by the enum, not the
    string prefix.

    Returns:
        ``(complete, reason)`` where ``complete`` is True for
        ``NORMAL_COMPLETE`` and ``EARLY_ABORT`` (the two terminal
        cases). ``False`` for ``IN_PROGRESS`` and
        ``INVALID_TERMINAL`` (the runner converts
        ``INVALID_TERMINAL`` to FAIL_FAST — the v11 contract).

    Raises:
        InvalidCheckpointError: If the manifest references a
            dangling / unsafe / step-mismatched state file.
    """
    snapshot = inspect_checkpoint(output_dir, config)
    if snapshot.completion in (CompletionStatus.NORMAL_COMPLETE, CompletionStatus.EARLY_ABORT):
        return True, snapshot.completion_reason
    return False, snapshot.completion_reason


# ---------------------------------------------------------------------------
# Production-time math
# ---------------------------------------------------------------------------
# Terminal payload reconstruction
# ---------------------------------------------------------------------------


def load_terminal_payload(output_dir: Path, config: OpenMMConfig) -> dict[str, Any] | None:
    """Load the validated terminal payload from the manifest.

    Thin wrapper around :func:`inspect_checkpoint` that returns
    the legacy ``dict | None`` shape. New code should use
    :func:`inspect_checkpoint` directly and read
    ``snapshot.terminal_payload`` (None when the manifest has
    no terminal payload or the payload is invalid).

    Returns:
        The terminal payload dict (with normalised fields
        ``step``, ``type``, ``reason``, ``production_ns``), or
        ``None`` if the run is not terminal via the manifest
        binding.

    Raises:
        InvalidCheckpointError: If the manifest references a
            dangling / unsafe / step-mismatched state file.
    """
    snapshot = inspect_checkpoint(output_dir, config)
    return snapshot.terminal_payload


# ---------------------------------------------------------------------------
# Production-time math
# ---------------------------------------------------------------------------


def _production_steps(absolute_step: int, total_equil_steps: int) -> int:
    """Return the completed production steps for an absolute step.

    Centralises the v10 BLOCKER #3 invariant: every ns reported
    to downstream consumers (``abort_ns``, ``result.total_ns``,
    ``md_summary.json``, the reconstructed result) is computed
    from the COMPLETED PRODUCTION steps (``absolute_step -
    total_equil_steps``), not the absolute OpenMM step. The
    equilibration steps are not simulation progress in the
    scientific sense — they're the protocol's setup.

    Args:
        absolute_step: The absolute OpenMM step the simulator is
            at (or the saved state corresponds to).
        total_equil_steps: The equilibration length in steps
            (``config.total_equil_steps``).

    Returns:
        The number of completed production steps (>= 0).
    """
    return max(0, absolute_step - total_equil_steps)


def production_ns(absolute_step: int, config: OpenMMConfig) -> float:
    """Convenience: completed production ns for an absolute step.

    Same invariant as :func:`_production_steps`. Returns ns as a
    float (no rounding) — callers that need a particular precision
    round at the boundary.
    """
    return _production_steps(absolute_step, config.total_equil_steps) * config.timestep_fs / 1e6
