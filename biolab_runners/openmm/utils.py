"""Utility functions for OpenMM MD simulations."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from biolab_runners.openmm.config import OpenMMConfig

from biolab_runners.openmm.paths import FileNames

logger = logging.getLogger(__name__)


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


class InvalidCheckpointError(Exception):
    """Raised when ``checkpoint.json`` references a state file that is invalid.

    The runner converts this into a ``result.error`` and refuses to proceed —
    the user must invoke ``runner.run(force=True)`` to discard the corrupt
    checkpoint (the quarantine moves the manifest + state files into
    ``.stale/<UTC>/`` together, so the next non-forced invocation starts from
    a fresh build).
    """


def openmm_available(platform: str = "OpenCL") -> bool:
    """Check if OpenMM is installed and the requested platform is available.

    Args:
        platform: OpenMM platform to check for (e.g. "OpenCL", "CUDA", "CPU").

    Returns:
        True if OpenMM is importable and the platform is available.
    """
    try:
        result = subprocess.run(
            [
                "python",
                "-c",
                (
                    "import openmm; "
                    f"p = openmm.Platform.getPlatformByName('{platform}'); "
                    "print(p.getName())"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.returncode == 0 and platform in result.stdout
    except FileNotFoundError:
        return False


def pdbfixer_available() -> bool:
    """Check if PDBFixer is installed.

    Returns:
        True if pdbfixer can be imported.
    """
    try:
        result = subprocess.run(
            ["python", "-c", "from pdbfixer import PDBFixer; print('ok')"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def verify_production_outputs(output_dir: Path) -> dict[str, object]:
    """Build a diagnostic report on the production output directory.

    This is a pure diagnostic helper — it reports per-file size and
    row counts but does NOT decide whether a run is complete.
    Completion is determined by :func:`is_run_complete` (which uses
    the manifest's absolute step + terminal payload), not by
    ``"files exist + size > threshold"``. The earlier "complete"
    field was removed because a mid-production checkpoint can
    produce a large trajectory and many energy rows while the run
    is still in progress — file presence does not imply completion.

    Args:
        output_dir: Directory containing MD outputs.

    Returns:
        Diagnostic report dict with file metadata (existence, size,
        row count for the energy log). The "complete" field is
        always False; callers that want a completion verdict must
        call :func:`is_run_complete` instead.
    """
    report: dict[str, object] = {
        "output_dir": str(output_dir),
        "complete": False,
        "files": {},
    }

    for filename in FileNames.PRODUCTION_OUTPUT_FILES:
        path = output_dir / filename
        exists = path.exists()
        size = path.stat().st_size if exists else 0

        file_info: dict[str, object] = {
            "exists": exists,
            "size_bytes": size,
        }

        if filename == FileNames.ENERGY and exists:
            lines = len(path.read_text().strip().splitlines())
            file_info["rows"] = lines

        if filename == FileNames.TRAJECTORY and exists:
            file_info["note"] = (
                "Trajectory size alone does not imply completion — "
                "see is_run_complete() for the authoritative verdict."
            )

        report["files"][filename] = file_info  # type: ignore[index]

    # Insert manifest info for completeness.
    manifest_path = output_dir / FileNames.CHECKPOINT_JSON
    if manifest_path.exists():
        report["files"][FileNames.CHECKPOINT_JSON] = {  # type: ignore[index]
            "exists": True,
            "size_bytes": manifest_path.stat().st_size,
        }

    return report


def _parse_state_filename_step(state_file: str) -> int | None:
    """Parse the absolute step encoded in a v7 state filename.

    The v7 format is ``state.<step>_<pid>_<nanos>.xml`` — the step
    is the first digit group. Returns the integer step, or None for
    the legacy ``state.xml`` (which carries no step encoding and
    therefore cannot be cross-validated against the manifest's
    ``step`` field).

    Used by :func:`load_checkpoint` to enforce v10 BLOCKER #1:
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


def _parse_manifest(
    output_dir: Path,
) -> tuple[int, dict[str, object], str] | None:
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


def load_checkpoint(output_dir: Path) -> tuple[int, str]:
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

    After parsing the manifest, the returned ``state_file`` is
    validated via :func:`_validate_state_file_reference` — a
    manifest that names a missing, empty, or unsafe state file
    raises :class:`InvalidCheckpointError`. The runner converts
    this into a ``result.error`` and aborts the resume; the user
    must invoke ``runner.run(force=True)`` to discard the corrupt
    checkpoint. A dangling manifest cannot degrade into a fresh
    build because ``prepare_simulation`` would silently build a new
    system with a different water count and pair it with the
    stale state, re-introducing the incompatibility
    (:attr:`biolab_runners.openmm.config.OpenMMConfig` "Resume
    safety" rule).

    v10 BLOCKER #1: the manifest's ``step`` MUST equal the step
    encoded in the v7 state filename. A mismatch raises
    :class:`InvalidCheckpointError`.

    v10 SUGGESTION: malformed manifests (root is a list, records
    is missing/empty, last record is not a mapping, ``step`` is
    ``null``, ``file`` is missing) are tolerated as "no resumable
    checkpoint" — they do NOT crash the runner.

    Args:
        output_dir: MD output directory.

    Returns:
        ``(absolute_step, state_file_basename)`` on success — both
        non-empty. ``(0, "")`` if no manifest exists or the manifest
        is malformed (no ``records`` array, last record has no
        ``step`` or ``file``, step is non-positive, or ``file`` is
        empty). The runner treats (0, "") as "no resumable
        checkpoint" and fails fast on any orphaned state file.

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
        return 0, ""
    step, _record, file = parsed
    return step, file


def load_checkpoint_full(
    output_dir: Path,
) -> tuple[int, dict[str, object], str] | None:
    """Like :func:`load_checkpoint` but returns the full last record.

    Used by :func:`is_run_complete` and
    :meth:`OpenMMRunner._reconstruct_terminal_result` to inspect
    the optional ``terminal`` payload (the v10 atomic terminal
    classification).

    Returns:
        ``(absolute_step, last_record, state_file)`` on success,
        ``None`` if no manifest or manifest is structurally
        invalid. The caller is responsible for handling
        ``InvalidCheckpointError`` from the state-file validation.
    """
    return _parse_manifest(output_dir)


def load_checkpoint_step(output_dir: Path) -> int:
    """Backwards-compatible step-only wrapper around :func:`load_checkpoint`.

    Returns the absolute step from the manifest, or 0 if no manifest
    exists. Kept for callers that only need the step (e.g. logs);
    new code should call :func:`load_checkpoint` directly to also
    get the file the manifest references.

    Note: unlike :func:`load_checkpoint`, this function does NOT
    raise on a manifest that references a missing state file. It
    only returns the step. Callers that need to either load the
    state or detect the dangling manifest must use
    :func:`load_checkpoint`.
    """
    try:
        step, _ = load_checkpoint(output_dir)
    except InvalidCheckpointError:
        return 0
    return step


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


def is_run_complete(output_dir: Path, config: OpenMMConfig) -> tuple[bool, str]:
    """Determine whether a production run is terminal.

    v10 BLOCKER #2: terminal status is now part of the manifest
    record itself (the ``terminal`` field) and is committed
    atomically with the state file via the manifest rename — the
    ``os.replace`` is the single commit point. A separate
    ``early_abort.json`` may still exist for downstream consumers
    (``oral_amp.cloud.openmm_cloud``), but it is a derived file
    written AFTER the atomic save and is NOT consulted by this
    function for terminal classification.

    A run is terminal when EITHER:

    1. **Normal completion**: ``manifest_step >= total_equil_steps +
       total_steps``. The manifest's step is the absolute OpenMM
       step (``start_step + steps_done``), so this signal is
       unambiguous — not dependent on file sizes or energy row
       counts.

    2. **Manifest terminal payload**: the manifest's last record
       contains a ``terminal`` dict with ``step == manifest.step``.
       The ``step`` field of the terminal payload MUST equal the
       manifest's ``step`` (v10 BLOCKER #2 binding) — a mismatch
       indicates data corruption and is logged as a warning; the
       run is treated as in_progress so a fresh loadCheckpoint
       can resume.

    Otherwise the run is in progress (or interrupted) and the
    caller should resume.

    Args:
        output_dir: MD output directory.
        config: The OpenMMConfig used to compute total_equil_steps
            and total_steps.

    Returns:
        ``(complete, reason)``. ``complete`` is True for the two
        terminal cases; ``reason`` is a human-readable explanation
        (e.g. ``"normal_completion_step_50_200_000"``,
        ``"manifest_terminal_early_abort_step_5_000_000"``,
        ``"in_progress"``).
    """
    parsed = load_checkpoint_full(output_dir)
    if parsed is None:
        # No manifest or structurally invalid manifest. The
        # runner handles orphaned state files separately.
        return False, "in_progress"
    manifest_step, last_record, _state_file = parsed
    if manifest_step <= 0:
        return False, "in_progress"

    # v12 BLOCKER: the explicit terminal payload MUST be checked
    # BEFORE the inferred normal completion. The two signals can
    # coexist at the same absolute step (the offline-mdtraj gate
    # fires on the final production chunk, so the manifest's step
    # lands at exactly ``total_equil_steps + total_steps`` at the
    # moment of an end-of-run abort). Returning normal completion
    # first would skip ``_reconstruct_terminal_result``, leaving a
    # reused result reporting ``early_abort=False`` despite the
    # manifest carrying a valid ``terminal`` payload — and the
    # live invocation would have set ``early_abort=True``. The
    # two result shapes would disagree. The explicit payload is
    # the user's stated intent; inferring completion from the
    # step alone is a heuristic and must defer to it.
    terminal = _check_manifest_terminal(manifest_step, last_record)
    if terminal is not None:
        return terminal
    normal = _check_normal_completion(manifest_step, config)
    if normal is not None:
        return normal
    return False, "in_progress"


def _check_normal_completion(manifest_step: int, config: OpenMMConfig) -> tuple[bool, str] | None:
    r"""Return ``(True, "normal_completion_..._of_<target>")`` if at/past target."""
    target_step = config.total_equil_steps + config.total_steps
    if manifest_step >= target_step:
        return True, f"normal_completion_step_{manifest_step}_of_{target_step}"
    return None


def _check_manifest_terminal(
    manifest_step: int, last_record: dict[str, object]
) -> tuple[bool, str] | None:
    r"""Return ``(True, "manifest_terminal_<type>_step_<step>")`` if payload valid.

    v10 BLOCKER #2: the manifest's ``terminal`` payload MUST have
    ``step == manifest.step`` (the binding is enforced here). A
    missing, zero, malformed, or mismatched step is logged as a
    warning and treated as "no terminal payload → in progress".

    v11 BLOCKER #3: the COMPLETE terminal schema must validate.
    The terminal record MUST have:
      - ``step``: a positive ``int`` (NOT str / bool / None).
      - ``type``: the literal string ``"early_abort"``. Any other
        value (missing, unknown, or unsupported type) is logged
        as a warning and treated as non-terminal — accepting an
        unknown type as terminal would skip result reconstruction
        (``_reconstruct_terminal_result`` only knows how to fill
        in fields for ``early_abort``), leaving the result
        reporting ``early_abort=False`` despite the manifest
        claiming terminal status.
      - ``reason``: a non-empty ``str``.
    """
    terminal = last_record.get("terminal")
    if not isinstance(terminal, dict):
        return None
    # v11 BLOCKER #3: step must be a strict positive int (no
    # string coercion). JSON int → Python int; JSON str or bool
    # is rejected so a forged manifest with ``"step": "5000000"``
    # cannot pass.
    raw_terminal_step = terminal.get("step")
    if (
        not isinstance(raw_terminal_step, int)
        or isinstance(raw_terminal_step, bool)
        or raw_terminal_step <= 0
    ):
        return None
    if raw_terminal_step != manifest_step:
        logger.warning(
            "Manifest has terminal.step=%d but record step=%d; "
            "treating as in_progress (binding mismatch)",
            raw_terminal_step,
            manifest_step,
        )
        return None
    # v11 BLOCKER #3: type must be the literal ``early_abort``.
    # No ``str(...)`` coercion — anything other than the exact
    # string is logged + treated as non-terminal.
    terminal_type = terminal.get("type")
    if terminal_type != "early_abort":
        logger.warning(
            "Manifest terminal.type=%r is not supported "
            "(only 'early_abort' is implemented); treating as in_progress",
            terminal_type,
        )
        return None
    # v11 BLOCKER #3: reason must be a non-empty string.
    reason = terminal.get("reason")
    if not isinstance(reason, str) or not reason:
        logger.warning("Manifest terminal.reason is missing or empty; treating as in_progress")
        return None
    return True, f"manifest_terminal_{terminal_type}_step_{raw_terminal_step}"


def load_terminal_payload(output_dir: Path, config: OpenMMConfig) -> dict[str, object] | None:
    """Load the validated terminal payload from the manifest.

    v10: the terminal payload is the manifest's ``terminal`` field
    on the last record. Returns the payload dict only when the
    run is terminal via the manifest binding (terminal.step ==
    manifest.step > 0) AND the required fields are well-typed.
    Returns None otherwise.

    Used by :meth:`OpenMMRunner._reconstruct_terminal_result` to
    reconstruct the abort result on idempotent reuse.

    Args:
        output_dir: MD output directory.
        config: OpenMMConfig (for production_ns computation).

    Returns:
        The terminal payload dict (with normalised fields
        ``step``, ``type``, ``reason``, ``production_ns``), or
        None if the run is not terminal or the payload is
        invalid.
    """
    parsed = load_checkpoint_full(output_dir)
    if parsed is None:
        return None
    manifest_step, last_record, _state_file = parsed
    terminal = last_record.get("terminal")
    if not isinstance(terminal, dict):
        return None
    # v11 BLOCKER #3: step must be a strict positive int (no
    # string coercion). Same gate as _check_manifest_terminal so
    # the reconstruction and the completion check agree on which
    # payloads are valid.
    raw_tstep = terminal.get("step")
    if not isinstance(raw_tstep, int) or isinstance(raw_tstep, bool) or raw_tstep <= 0:
        return None
    if raw_tstep != manifest_step:
        return None
    # Normalised payload — production_ns is computed from the
    # v10 BLOCKER #3 invariant, not read from a stored field.
    payload: dict[str, object] = {
        "step": raw_tstep,
        "type": str(terminal.get("type", "unknown")),
        "reason": str(terminal.get("reason", "")),
        "production_ns": production_ns(raw_tstep, config),
    }
    # Pass through optional fields if present and well-typed.
    for opt in ("gate", "target", "peptide_id"):
        val = terminal.get(opt)
        if isinstance(val, str):
            payload[opt] = val
    return payload
