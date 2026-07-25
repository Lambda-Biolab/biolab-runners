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
    the manifest's absolute step + early-abort metadata), not by
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


def is_run_complete(output_dir: Path, config: OpenMMConfig) -> tuple[bool, str]:
    """Determine whether a production run is terminal.

    A run is terminal when EITHER:

    1. **Normal completion**: ``manifest_step >= total_equil_steps +
       total_steps``. The manifest's step is the absolute OpenMM
       step (``start_step + steps_done``), so this signal is
       unambiguous — not dependent on file sizes or energy row
       counts.

    2. **Intentional early termination**: ``early_abort.json`` exists
       with ``aborted=True`` AND a positive integer ``abort_step``.
       The atomically-committed abort metadata is the explicit
       terminal marker for the offline-mdtraj gate. v9: the
       ``abort_step`` field is required and must parse as a
       positive integer — a missing, zero, or malformed step is
       treated as "marker invalid → run is in progress" rather
       than terminal. The marker is generation-scoped (bound to the
       manifest step in :func:`OpenMMRunner._write_abort_metadata`)
       and is moved by ``force=True`` quarantine so a fresh run
       cannot be mis-classified by a stale marker.

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
        ``"early_abort_step_5_000_000"`` or ``"in_progress"``).
    """
    manifest_step, _ = load_checkpoint(output_dir)
    if manifest_step > 0:
        target_step = config.total_equil_steps + config.total_steps
        if manifest_step >= target_step:
            return True, f"normal_completion_step_{manifest_step}_of_{target_step}"

    abort_path = output_dir / FileNames.EARLY_ABORT_JSON
    if abort_path.exists():
        try:
            abort_meta = json.loads(abort_path.read_text())
        except (json.JSONDecodeError, OSError):
            abort_meta = {}
        if abort_meta.get("aborted") is True:
            # Safely validate abort_step — must be a positive
            # integer. The previous behaviour passed through
            # ``abort_step=0`` (missing field default) or raised
            # ``TypeError``/``ValueError`` on non-int-convertible
            # strings. Both cases are now treated as "marker
            # invalid" rather than terminal.
            try:
                abort_step = int(abort_meta.get("abort_step", 0))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                abort_step = 0
            if abort_step > 0:
                return True, f"early_abort_step_{abort_step}"

    return False, "in_progress"


def load_abort_metadata(output_dir: Path) -> dict[str, object] | None:
    """Load and validate ``early_abort.json`` if it is a terminal marker.

    Returns the parsed JSON dict when ``aborted is True`` AND
    ``abort_step`` parses as a positive integer. Returns ``None``
    otherwise (missing file, malformed JSON, invalid marker).
    The validation gates match :func:`is_run_complete` so a
    terminal classification is always paired with metadata that
    reconstructs the abort result.

    Args:
        output_dir: MD output directory.

    Returns:
        The parsed abort metadata dict, or ``None`` if the marker
        is missing or invalid.
    """
    abort_path = output_dir / FileNames.EARLY_ABORT_JSON
    if not abort_path.exists():
        return None
    try:
        abort_meta = json.loads(abort_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if abort_meta.get("aborted") is not True:
        return None
    try:
        abort_step = int(abort_meta.get("abort_step", 0))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if abort_step <= 0:
        return None
    return abort_meta


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
            pattern, doesn't exist, or is empty. The runner catches
            this and surfaces it as a user-facing error.
    """
    manifest_path = output_dir / FileNames.CHECKPOINT_JSON
    if not manifest_path.exists():
        return 0, ""
    try:
        data = json.loads(manifest_path.read_text())
        records = data.get("records", [])
        if not records:
            return 0, ""
        last = records[-1]
        step = int(last.get("step", 0))
        file = str(last.get("file", ""))
        if step > 0 and file:
            # Validate the referenced state file. The runner catches
            # InvalidCheckpointError and surfaces the message to the
            # user — the resume path cannot proceed with a dangling
            # or unsafe reference.
            _validate_state_file_reference(output_dir, file)
            return step, file
    except (json.JSONDecodeError, KeyError, IndexError, OSError, ValueError):
        pass
    except InvalidCheckpointError:
        # Re-raise so the runner can convert it to a result.error.
        # (The bare ``except`` above would otherwise swallow our
        # own exception class.)
        raise
    return 0, ""


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
