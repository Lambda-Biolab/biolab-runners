"""GROMACS subprocess runners.

This module hosts two runners:

- :class:`GromacsRunner` — the **legacy one-shot mdrun** runner
  (preserved from S3). Drives a single ``gmx mdrun`` invocation
  against a pre-built ``.tpr``; idempotent on ``energy.edr``
  existence.

- :class:`GromacsProtocolRunner` — the **S4 production-grade
  protocol** runner. Drives the full pipeline (topology → box →
  solvate → ions → minimization → NVT → NPT → production) with
  per-stage checkpoint resume. Each MD stage runs ``gmx grompp``
  before ``gmx mdrun`` (grompp compiles the ``.tpr`` from the
  ``.mdp`` + previous ``.gro`` + optional ``.cpt``); mdrun uses
  ``-cpi`` + ``-append`` **only** when a ``.cpt`` exists.

Both runners are thin subprocess wrappers. The protocol
**content** (``.mdp`` strings, command-line construction, stage
plan) lives in :mod:`biolab_runners.gromacs.protocol`. The
**filenames** live in :mod:`biolab_runners.gromacs.paths`. The
**manifest I/O** lives in :mod:`biolab_runners.gromacs.utils`.
This module is the orchestrator that ties them together.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.gromacs.protocol import (
    GENION_INPUT,
    ProtocolStage,
    StageKind,
    build_commands,
    build_stage_plan,
    generate_equil_npt_mdp,
    generate_equil_nvt_mdp,
    generate_minimization_mdp,
    generate_production_mdp,
    ions_mdp_content,
    stage_minimum_outputs,
)
from biolab_runners.gromacs.utils import (
    GromacsRecord,
    GromacsRecordStatus,
    StageStatus,
    invoke,
    load_stage_manifest,
    now_utc_iso,
    parse_nthcol_energy,
    record_stage_status,
)

if TYPE_CHECKING:
    from biolab_runners.gromacs.config import GromacsConfig, GromacsProtocolConfig

logger = logging.getLogger(__name__)

__all__ = [
    "GromacsProtocolResult",
    "GromacsProtocolRunner",
    "GromacsResult",
    "GromacsRunner",
]


# Sentinel return code for "child was killed by SIGTERM". Distinct
# from any plausible gmx exit code; the runner checks for it before
# marking a stage FAILED (an interrupted stage is still RESUMABLE).
_INTERRUPTED_RC = -signal.SIGTERM


def _empty_metrics_dict() -> dict[str, float]:
    return {}


@dataclass(frozen=True)
class GromacsResult:
    """Outcome of one GROMACS run."""

    name: str
    output_dir: str
    records: tuple[GromacsRecord, ...] = ()
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    exit_code: int = 0
    duration_seconds: float = 0.0
    metrics: dict[str, float] = field(default_factory=_empty_metrics_dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into a JSON-safe dictionary."""
        return {
            "name": self.name,
            "output_dir": self.output_dir,
            "records": [r.to_dict() for r in self.records],
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class GromacsProtocolResult:
    """Outcome of one production-grade GROMACS protocol run.

    Captures per-stage status (so a partial run — e.g. one where
    the production stage was interrupted by Spot reclaim — still
    reports which earlier stages completed). The
    ``duration_seconds`` field is the wall-clock time of the
    entire run; per-stage timings are visible in the manifest.

    Counters are exact:
    - ``skipped``: stages that were COMPLETED before this invocation
      (no subprocess was launched).
    - ``succeeded``: stages that ran fresh during this invocation
      (or re-ran due to ``force=True``) and exited zero.
    - ``failed``: stages that ran but exited non-zero (and were not
      interrupted by SIGTERM).
    - ``interrupted``: stages that were interrupted by SIGTERM
      (treated as resumable — the manifest preserves RUNNING
      status so the next invocation resumes from the ``.cpt``).
    - ``validated``: stages that were **dry-run** (.mdp files
      emitted for inspection, no subprocess, NO manifest record
      written — a subsequent real run will run them from
      scratch). Mutually exclusive with ``succeeded`` /
      ``skipped`` (each stage contributes to exactly one of
      these four counters per invocation).
    - ``dry_run``: True iff the runner was constructed with
      ``dry_run=True``. When True, every stage counts as
      ``validated`` (or ``failed`` for stages that could not
      emit a .mdp).
    """

    name: str
    output_dir: str
    replica_index: int
    replicas_total: int
    stage_statuses: dict[str, str] = field(default_factory=dict)
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    interrupted: int = 0
    validated: int = 0
    dry_run: bool = False
    exit_code: int = 0
    duration_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into a JSON-safe dictionary."""
        return {
            "name": self.name,
            "output_dir": self.output_dir,
            "replica_index": self.replica_index,
            "replicas_total": self.replicas_total,
            "stage_statuses": dict(self.stage_statuses),
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "interrupted": self.interrupted,
            "validated": self.validated,
            "dry_run": self.dry_run,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


# Special stage status returned by dry-run. NOT a value of
# :class:`StageStatus` (so the manifest-authority skip logic in
# ``_stage_already_complete`` ignores it) and NOT a string that
# the runner interprets as COMPLETED/FAILED. The runner counts
# stages in this status under ``validated`` rather than
# ``succeeded`` / ``skipped`` so callers can distinguish a
# dry-run validation from a real run.
_DRY_RUN_STATUS = "validated"


def _emit_mdp(work_dir: Path, stage: ProtocolStage, config: GromacsProtocolConfig) -> str | None:
    """Emit the .mdp content for a stage and write it to disk.

    Returns the mdp filename written (matches ``stage.mdp_filename``)
    or ``None`` for stages that don't need a .mdp file (e.g.
    ``BOX`` is a pure ``editconf`` invocation with no mdp input).
    The content is the canonical deterministic string from the
    protocol module — same config → byte-identical content.
    """
    if not stage.mdp_filename:
        return None
    if stage.kind == StageKind.IONS:
        content = ions_mdp_content()
    elif stage.kind == StageKind.MINIMIZE:
        content = generate_minimization_mdp(
            config.minimization_max_iterations,
            replica_index=config.replica_index,
        )
    elif stage.kind == StageKind.EQUIL_NVT:
        content = generate_equil_nvt_mdp(
            config.nvt_ps,
            config.temperature_k,
            replica_index=config.replica_index,
        )
    elif stage.kind == StageKind.EQUIL_NPT:
        content = generate_equil_npt_mdp(
            config.npt_ps,
            config.temperature_k,
            config.pressure_bar,
            replica_index=config.replica_index,
        )
    elif stage.kind == StageKind.PRODUCTION:
        content = generate_production_mdp(
            config.effective_production_ns(),
            config.temperature_k,
            config.pressure_bar,
            replica_index=config.replica_index,
        )
    else:
        return None
    mdp_path = work_dir / stage.mdp_filename
    mdp_path.write_text(content)
    return stage.mdp_filename


def _is_md_stage(kind: StageKind) -> bool:
    """True iff the stage runs ``gmx mdrun`` (and is therefore checkpoint-resumable)."""
    return kind in (
        StageKind.MINIMIZE,
        StageKind.EQUIL_NVT,
        StageKind.EQUIL_NPT,
        StageKind.PRODUCTION,
    )


def _checkpoint_for(work_dir: Path, stage: ProtocolStage) -> str | None:
    """Return the absolute path to the ``.cpt`` for an MD stage, or None.

    ``None`` signals "fresh start" — the caller (build_commands) uses
    this to decide whether to emit ``-cpi <path>``. The "no duplicate
    start path" rule from S4: GROMACS errors when ``-cpi`` points to
    a non-existent file, so the runner pre-checks and emits the flag
    only when resume is actually possible.
    """
    if not _is_md_stage(stage.kind):
        return None
    from biolab_runners.gromacs.paths import GromacsFiles

    cpt = work_dir / GromacsFiles.checkpoint(stage.prefix)
    return str(cpt) if cpt.exists() else None


def _outputs_exist(work_dir: Path, outputs: tuple[str, ...]) -> bool:
    """True iff every output filename exists and is non-empty on disk."""
    return all(
        (work_dir / name).is_file() and (work_dir / name).stat().st_size > 0 for name in outputs
    )


def _stage_already_complete(work_dir: Path, stage: ProtocolStage) -> bool:
    """True iff the manifest marks this stage COMPLETED **before** this run.

    This is the "manifest authority" check used by the runner's
    skip accounting: a manifest entry saying COMPLETED means the
    stage was completed in a previous invocation, regardless of
    whether the disk outputs still exist (the manifest is the
    source of truth — outputs may be on a shared filesystem that
    is unavailable right now, etc.).

    The disk-output check is a separate fallback (used when the
    manifest is silent) — see ``_outputs_complete_on_disk``.
    """
    manifest = load_stage_manifest(work_dir)
    record = manifest.get("stages", {}).get(stage.kind.value)
    return record is not None and record.get("status") == StageStatus.COMPLETED


def _outputs_complete_on_disk(work_dir: Path, stage: ProtocolStage) -> bool:
    """True iff every MINIMUM output for the stage is on disk and non-empty.

    Used as a fallback when the manifest is silent (e.g. a Spot
    reclaim that lost the manifest write — the outputs may have
    survived).

    **Uses ``stage_minimum_outputs``, NOT ``stage_outputs_for``.**
    The minimum set is the strict subset of files that ``gmx``
    writes UNCONDITIONALLY: ``.tpr``, ``.gro``, ``.edr``, ``.log``.
    Forbidding the intermittent artifacts (``.cpt``,
    ``.xtc``, ``.trr``) would force the runner to re-run every
    short simulation, which defeats the purpose of the fallback.

    The runner separately checks ``.cpt`` via
    :func:`_checkpoint_for` when deciding fresh-vs-resume, so
    missing .cpt correctly forces a fresh start even when the
    minimum outputs are present.
    """
    return _outputs_exist(work_dir, stage_minimum_outputs(stage.kind, stage.prefix))


def _work_dir(config: GromacsProtocolConfig) -> Path:
    """Compute the per-replica work directory for this config.

    Single-replica runs (``replicas_total == 1``) write directly
    under ``output_root / name``. Multi-replica runs write under
    ``output_root / name / f"rep{replica_index:03d}"`` so all
    replicas can coexist in one ``output_root`` without colliding
    on .cpt / .tpr / .edr filenames.
    """
    root = Path(config.output_root) / config.name
    if config.replicas_total > 1:
        return root / f"rep{config.replica_index:03d}"
    return root


class GromacsRunner:
    """Subprocess wrapper around the upstream GROMACS CLI (legacy one-shot)."""

    def __init__(
        self,
        *,
        config: GromacsConfig | None = None,
        binary_prefix: list[str] | None = None,
        output_root: Path | None = None,
        timeout_seconds: int = 86400,
    ) -> None:
        self._config_override = config
        self._binary_prefix = binary_prefix
        self._output_root = output_root or Path.cwd() / "gromacs_output"
        self._timeout_seconds = timeout_seconds

    @property
    def output_root(self) -> Path:
        """Return the root directory into which GROMACS writes outputs."""
        return self._output_root

    def is_complete(self, config: GromacsConfig) -> bool:
        """Return True if a prior ``energy.edr`` already exists."""
        energy = self._design_dir(config) / f"{config.tpr_basename}.edr"
        return energy.exists()

    def run(
        self,
        config: GromacsConfig | None = None,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> GromacsResult:
        """Run GROMACS and return the parsed result."""
        cfg = config or self._config_override
        if cfg is None:
            raise ValueError("GromacsConfig is required: pass it to run() or the runner")

        output_dir = self._design_dir(cfg)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not force and self.is_complete(cfg):
            records = self._collect_records(output_dir, cfg)
            return GromacsResult(
                name=cfg.name,
                output_dir=str(output_dir),
                records=tuple(records),
                succeeded=len(records),
                failed=0,
                skipped=len(records),
                exit_code=0,
                duration_seconds=0.0,
            )

        if dry_run:
            return GromacsResult(
                name=cfg.name,
                output_dir=str(output_dir),
                records=(),
                succeeded=0,
                failed=0,
                skipped=0,
                exit_code=0,
                duration_seconds=0.0,
            )

        config_dict = _config_to_cli(cfg)
        started = time.monotonic()
        exit_code = invoke(
            config_dict=config_dict,
            output_dir=output_dir,
            mdrun_extra=cfg.extra_mdrun_flags,
            binary_prefix=self._binary_prefix,
            timeout_seconds=self._timeout_seconds,
        )
        records = self._collect_records(output_dir, cfg)
        succeeded = sum(1 for r in records if r.status == GromacsRecordStatus.SUCCEEDED)
        failed = len(records) - succeeded
        return GromacsResult(
            name=cfg.name,
            output_dir=str(output_dir),
            records=tuple(records),
            succeeded=succeeded,
            failed=failed,
            skipped=0,
            exit_code=exit_code,
            duration_seconds=time.monotonic() - started,
        )

    def _design_dir(self, config: GromacsConfig) -> Path:
        return self._output_root / config.name

    def _collect_records(self, output_dir: Path, config: GromacsConfig) -> list[GromacsRecord]:
        """Walk ``output_dir`` and parse each ``energy.edr``-derived file."""
        energy = output_dir / f"{config.tpr_basename}.edr"
        if not energy.exists():
            return []
        records: list[GromacsRecord] = []
        try:
            potential = parse_nthcol_energy(energy, column=1)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("failed to parse %s: %s", energy, exc)
            records.append(
                GromacsRecord(
                    index=len(records),
                    path=str(energy),
                    potential_energy=0.0,
                    status=GromacsRecordStatus.FAILED,
                    error=str(exc),
                )
            )
            return records
        records.append(
            GromacsRecord(
                index=0,
                path=str(energy),
                potential_energy=potential,
            )
        )
        return records


def _config_to_cli(config: GromacsConfig) -> dict[str, str]:
    """Translate :class:`GromacsConfig` into a flat CLI kwargs dict."""
    payload: dict[str, str] = {
        "-deffnm": config.tpr_basename,
        "-s": config.structure_file,
        "-nsteps": str(config.nsteps),
    }
    for key, value in config.extra.items():
        payload[str(key)] = str(value)
    return payload


def invoke(  # noqa: F811 - re-export for the test monkeypatch seam
    *,
    config_dict: dict[str, str],
    output_dir: Path,
    mdrun_extra: tuple[str, ...],
    binary_prefix: list[str] | None = None,
    timeout_seconds: int = 86400,
) -> int:
    """Run ``gmx mdrun`` once; returns the process exit code."""
    from biolab_runners.gromacs.utils import invoke as _invoke

    return _invoke(
        config_dict=config_dict,
        output_dir=output_dir,
        mdrun_extra=mdrun_extra,
        binary_prefix=binary_prefix,
        timeout_seconds=timeout_seconds,
    )


# ---------------------------------------------------------------------------
# S4 production-grade protocol runner
# ---------------------------------------------------------------------------


class GromacsProtocolRunner:
    """Run the production-grade GROMACS protocol end-to-end.

    Drives every stage in the canonical order (topology → box →
    solvate → ions → minimize → NVT → NPT → production), honouring
    per-stage idempotency:

    - **Manifest authority**: the manifest is the source of truth.
      A stage whose manifest record is ``completed`` is **skipped**
      (no subprocess launched) — this is the skip counter.
    - **Output fallback**: when the manifest is silent (Spot
      reclaim that lost the manifest write) but every canonical
      output is on disk and non-empty, the runner treats the
      stage as complete and records it as such.
    - **MD stage resume**: an MD stage with a ``.cpt`` on disk
      resumes via ``gmx grompp -t <cpt>`` + ``gmx mdrun -cpi
      <cpt> -append``. Without ``.cpt``, the stage starts fresh
      (no ``-cpi``, no ``-append``) — the "no duplicate start
      path" rule.
    - **SIGTERM preservation**: when the parent receives SIGTERM
      (cloud preemption), the child is forwarded the signal and
      given up to 30 s to write its periodic ``.cpt`` and exit.
      The stage is recorded as ``running`` (not ``failed``) so
      the next invocation sees the on-disk ``.cpt`` and resumes.
      The runner returns ``interrupted`` (not ``failed``) in the
      per-stage accounting.
    - **``nt_threads=0``**: the runner OMITS ``-nt`` so GROMACS
      auto-detects the host thread count (the recommended default
      on cloud VMs where you don't know the vCPU count a priori).
    - **Genion stdin**: the IONS stage's ``gmx genion`` invocation
      receives its ``SOL`` group selection via subprocess stdin
      (``input=GENION_INPUT``) — NEVER ``sh -c "echo SOL | gmx
      genion ..."`` (shell injection + platform-dependent quoting).
    - **force=True**: ignores the manifest, ignores the disk
      outputs, and re-runs every stage from scratch. The IONS
      stage re-runs ``gmx grompp`` and ``gmx genion`` (the latter
      will overwrite ``ions.gro``).

    Args:
        binary_prefix: Optional override for the ``gmx`` command
            prefix (e.g. ``["docker", "run", "--rm", "gmx-image",
            "gmx"]`` for a containerised install).
        sigterm_grace_seconds: How long the parent waits for the
            child to exit cleanly after forwarding SIGTERM, before
            escalating to SIGKILL. Default 30 s — enough for
            ``gmx mdrun`` to write its periodic ``.cpt``.
        dry_run: Validate the protocol plan and emit .mdp files
            without invoking any ``gmx`` subprocess.
    """

    def __init__(
        self,
        *,
        binary_prefix: list[str] | None = None,
        sigterm_grace_seconds: float = 30.0,
        dry_run: bool = False,
    ) -> None:
        self._binary_prefix = binary_prefix
        self._sigterm_grace_seconds = sigterm_grace_seconds
        self._dry_run = dry_run

    def run_protocol(
        self,
        config: GromacsProtocolConfig,
    ) -> GromacsProtocolResult:
        """Execute the full protocol and return a per-stage result.

        See the class docstring for the skip-vs-resume-vs-fresh
        decision tree. The method is the only public entry point
        — everything else is an internal helper.
        """
        work_dir = _work_dir(config)
        work_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        stage_statuses: dict[str, str] = {}
        succeeded = 0
        skipped = 0
        failed = 0
        interrupted = 0
        validated = 0
        exit_code = 0
        error = ""

        for stage in build_stage_plan():
            status, rc, was_skipped = self._run_single_stage(work_dir, stage, config)
            stage_statuses[stage.kind.value] = status
            if status == StageStatus.COMPLETED:
                if was_skipped:
                    skipped += 1
                else:
                    succeeded += 1
            elif status == _DRY_RUN_STATUS:
                # Dry-run stage: no subprocess, no manifest record,
                # no skip. Counted under ``validated`` (NOT under
                # ``succeeded``) so callers can tell a real run from
                # a dry-run validation.
                validated += 1
            elif status == "interrupted":
                interrupted += 1
                # Don't halt: a Spot reclaim that interrupted an
                # earlier stage should NOT prevent later stages from
                # being attempted (they'll find their inputs missing
                # and fail fast, which is the right signal). However,
                # for clarity we record the first interruption's rc.
                exit_code = rc
                continue
            elif status == StageStatus.FAILED:
                failed += 1
                exit_code = rc
                error = f"stage {stage.kind.value} failed (rc={rc})"
                break

        return GromacsProtocolResult(
            name=config.name,
            output_dir=str(work_dir),
            replica_index=config.replica_index,
            replicas_total=config.replicas_total,
            stage_statuses=stage_statuses,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            interrupted=interrupted,
            validated=validated,
            dry_run=self._dry_run,
            exit_code=exit_code,
            duration_seconds=time.monotonic() - started,
            error=error,
        )

    def _run_single_stage(
        self,
        work_dir: Path,
        stage: ProtocolStage,
        config: GromacsProtocolConfig,
    ) -> tuple[str, int, bool]:
        """Run one stage; return ``(status, exit_code, was_skipped)``.

        Decision tree (the manifest is the source of truth):

        - ``force=True``: bypass manifest + outputs, re-run from
          scratch.
        - Manifest says COMPLETED → record COMPLETED, return
          (``was_skipped=True``).
        - Manifest silent + outputs on disk → mark COMPLETED in
          the manifest, return (``was_skipped=True``; this is the
          Spot-reclaim recovery path).
        - Otherwise: emit ``.mdp``, build commands, execute.

        MD-stage resume:
        - ``.cpt`` exists → resume via ``-cpi`` + ``-append``.
        - No ``.cpt`` → start fresh.
        """
        # --- Manifest authority check ---
        if not config.force and _stage_already_complete(work_dir, stage):
            # Stage was COMPLETED in a prior invocation — skip.
            record_stage_status(
                work_dir,
                stage.kind.value,
                StageStatus.COMPLETED,
                outputs=stage_minimum_outputs(stage.kind, stage.prefix),
            )
            return StageStatus.COMPLETED, 0, True

        # --- Output fallback (manifest silent but outputs present) ---
        if not config.force and _outputs_complete_on_disk(work_dir, stage):
            # Spot-reclaim recovery: manifest was lost but outputs
            # survived. Mark COMPLETED so subsequent invocations
            # honour manifest authority.
            record_stage_status(
                work_dir,
                stage.kind.value,
                StageStatus.COMPLETED,
                outputs=stage_minimum_outputs(stage.kind, stage.prefix),
            )
            return StageStatus.COMPLETED, 0, True

        # --- Emit .mdp and build commands ---
        _emit_mdp(work_dir, stage, config)
        started_at = now_utc_iso()
        checkpoint_path = _checkpoint_for(work_dir, stage)
        commands = build_commands(
            stage,
            checkpoint_path=checkpoint_path,
            config=config,
        )

        if self._dry_run:
            # Dry-run path: emit the .mdp file (so the operator can
            # inspect it via `cat work_dir/min.mdp` etc.) and emit the
            # grompp / mdrun command list (so the operator can verify
            # what would be run), but **DO NOT** write a terminal
            # manifest record. The previous implementation wrote
            # COMPLETED to the manifest here, which had a subtle
            # foot-gun: a subsequent real run on the same work_dir
            # would honour manifest authority and SKIP every stage,
            # even though no actual simulation had run. The fix is
            # to leave the manifest empty so the next invocation
            # exercises every stage from scratch (the .mdp files
            # are useful as a preview, but they don't constitute
            # completion).
            return _DRY_RUN_STATUS, 0, False

        record_stage_status(
            work_dir,
            stage.kind.value,
            StageStatus.RUNNING,
            command=" ".join(commands[0]) if commands else "",
            started_at=started_at,
        )

        # --- Execute commands ---
        rc = 0
        for cmd in commands:
            rc = self._run_subprocess(cmd, work_dir, config.timeout_seconds)
            if rc != 0:
                break
        completed_at = now_utc_iso()

        # --- Classify the outcome ---
        if rc == _INTERRUPTED_RC:
            # SIGTERM: the child was forwarded the signal and exited.
            # Preserve the manifest in RUNNING (NOT failed) so the
            # next invocation sees the on-disk .cpt and resumes.
            record_stage_status(
                work_dir,
                stage.kind.value,
                StageStatus.RUNNING,
                outputs=stage_minimum_outputs(stage.kind, stage.prefix),
                command=" ".join(commands[0]) if commands else "",
                started_at=started_at,
                completed_at=completed_at,
                error="interrupted by SIGTERM; resumable on next invocation",
            )
            return "interrupted", _INTERRUPTED_RC, False

        status = StageStatus.COMPLETED if rc == 0 else StageStatus.FAILED
        record_stage_status(
            work_dir,
            stage.kind.value,
            status,
            outputs=stage_minimum_outputs(stage.kind, stage.prefix),
            command=" ".join(commands[0]) if commands else "",
            started_at=started_at,
            completed_at=completed_at,
            error="" if rc == 0 else f"rc={rc}",
        )
        return status, rc, False

    def _run_subprocess(
        self,
        cmd: list[str],
        work_dir: Path,
        timeout_seconds: int,
    ) -> int:
        r"""Run one ``gmx`` command with SIGTERM grace + kill escalation.

        **Honest SIGTERM semantics** (no 24 h blocking):

        1. Spawn the child. The parent installs a SIGTERM handler
           that sets ``interrupted`` and forwards to the child
           (``Popen.terminate``).
        2. If SIGTERM arrives, the child has
           ``sigterm_grace_seconds`` (default 30 s) to write its
           periodic ``.cpt`` and exit.
        3. After the grace expires the parent escalates with
           ``Popen.kill()`` (SIGKILL). The child has no further
           chance to flush a .cpt — by this point the next
           invocation will start fresh from the last periodic
           checkpoint. The grace is the **only** opportunity to
           leave a resumable .cpt; the kill is the contract.
        4. The parent returns ``_INTERRUPTED_RC`` when the child
           exited via the SIGTERM path (regardless of whether
           it exited within the grace or after the kill). The
           runner classifies the stage as ``interrupted`` (NOT
           failed); the manifest stays in ``RUNNING`` so the
           next invocation resumes from the on-disk ``.cpt``
           (if the child flushed one in time) or starts fresh
           (if the kill was the actual exit path).

        **Honest timeout semantics**: ``proc.communicate(timeout=
        timeout_seconds)`` raises ``TimeoutExpired`` if the child
        exceeds the overall per-stage timeout. The parent then
        kills the child and returns rc=124 (the conventional
        "command timed out" code). The grace escalation runs on
        a separate timer (the SIGTERM signal handler) — it does
        NOT block on the communicate timeout.

        **Genion stdin**: when ``cmd`` is the IONS stage's genion
        invocation (``gmx genion ...``), the parent's stdin pipe
        is closed and ``GENION_INPUT`` ("SOL\n") is fed via
        ``Popen.communicate(input=GENION_INPUT)``. No shell, no
        quoting, no injection.
        """
        prefix = self._binary_prefix or []
        full_cmd = [*prefix, *cmd]

        # Detect genion — needs stdin (group selection).
        is_genion = "genion" in cmd

        # Track whether the parent received SIGTERM (Spot preemption).
        interrupted = threading.Event()
        sigterm_received_at: list[float | None] = [None]
        kill_dispatched = threading.Event()

        stdin_arg = subprocess.PIPE if is_genion else None

        proc = subprocess.Popen(
            full_cmd,
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=stdin_arg,
        )

        # Background grace watchdog: when SIGTERM arrives, set
        # interrupted and schedule a kill after sigterm_grace_seconds.
        # The watchdog is a daemon thread so it does not block
        # communicate() — the parent only waits on communicate
        # (which has its own timeout_seconds cap).
        def _forward_sigterm(*_unused: object) -> None:
            if interrupted.is_set():
                return  # already handling
            interrupted.set()
            sigterm_received_at[0] = time.monotonic()
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()

            def _escalate_to_kill() -> None:
                if kill_dispatched.is_set():
                    return
                kill_dispatched.set()
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                logger.warning(
                    "gmx child did not exit within %.1fs of SIGTERM; escalated to SIGKILL",
                    self._sigterm_grace_seconds,
                )

            watchdog = threading.Timer(self._sigterm_grace_seconds, _escalate_to_kill)
            watchdog.daemon = True
            watchdog.start()

        original = signal.signal(signal.SIGTERM, _forward_sigterm)
        try:
            # Communicate is in text mode (text=True on Popen); the
            # stdin payload must be a str, not bytes. The
            # communicate timeout is the OVERALL cap (timeout_seconds);
            # the SIGTERM grace runs on the separate watchdog timer.
            stdin_payload: str | None = GENION_INPUT if is_genion else None
            _, stderr = proc.communicate(input=stdin_payload, timeout=timeout_seconds)
            rc = proc.returncode if proc.returncode is not None else 0
        except subprocess.TimeoutExpired:
            logger.error("GROMACS stage timed out after %ds in %s", timeout_seconds, work_dir)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            try:
                _, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                _, stderr = "", ""
            rc = 124
        finally:
            signal.signal(signal.SIGTERM, original)

        # If the parent received SIGTERM (interrupted.is_set()), the
        # child was forwarded the signal. Return the sentinel so the
        # stage is recorded as interrupted (not failed), preserving
        # resumable status.
        if interrupted.is_set():
            return _INTERRUPTED_RC

        if rc != 0:
            logger.warning(
                "gmx stage failed rc=%d in %s; stderr=%s",
                rc,
                work_dir,
                (stderr or "")[-500:],
            )
        return rc
