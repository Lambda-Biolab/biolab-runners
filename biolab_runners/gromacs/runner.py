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
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    stage_prebuilt_topology,
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
    save_stage_manifest,
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

    Prebuilt-source binding: the TOPOLOGY stage also stores a
    ``prebuilt_source`` block with the source digests. A
    different prebuilt source (different ``.top`` /
    ``.gro`` pair supplied by the caller) MUST invalidate the
    cached stage even when the manifest status is COMPLETED.
    This helper currently only consults the status field; the
    caller (``_run_single_stage``) does the prebuilt-source
    diff check separately — see the comment block in
    :meth:`GromacsProtocolRunner.run_protocol`.
    """
    manifest = load_stage_manifest(work_dir)
    record = manifest.get("stages", {}).get(stage.kind.value)
    return record is not None and record.get("status") == StageStatus.COMPLETED


def _prebuilt_source_changed(work_dir: Path, config: GromacsProtocolConfig) -> bool:
    """Return True iff the supplied prebuilt source differs from the cached one.

    A cached TOPOLOGY stage is invalidated when EITHER of the
    prebuilt source paths OR their digests has changed. This
    prevents a stale topology from being silently reused when
    the caller supplied a new ``.top`` / ``.gro`` pair.

    Returns ``False`` when the manifest is silent (the
    no-cache case) or when the prebuilt pair is empty (legacy
    ``input_pdb`` path) — only the prebuilt path ever has a
    "previous" identity to compare against.
    """
    if not (config.prebuilt_topology and config.prebuilt_coordinates):
        return False
    manifest = load_stage_manifest(work_dir)
    record = manifest.get("stages", {}).get(StageKind.TOPOLOGY.value)
    if not record:
        return False
    cached = record.get("prebuilt_source") or {}
    new = _prebuilt_meta(config)
    return (
        cached.get("sha256_topology") != new["sha256_topology"]
        or cached.get("sha256_coordinates") != new["sha256_coordinates"]
    )


def _outputs_complete_on_disk(work_dir: Path, stage: ProtocolStage) -> bool:
    """True iff every MINIMUM output for the stage is on disk and non-empty.

    Used only by the legacy manifest-silent recovery fallback in
    :meth:`GromacsProtocolRunner._run_single_stage`; the runner's
    guard requires a silent manifest and no on-disk ``.cpt``.

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
    - **Output fallback**: when the manifest is silent, no
      ``.cpt`` is on disk, and every canonical output is
      non-empty, the runner treats the stage as complete
      (legacy recovery). An on-disk ``.cpt`` always routes
      to the execution path so ``-cpi`` resume is preserved.
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
      The runner returns ``interrupted`` (not ``failed``) and
      halts at the interrupted stage, so a missing-input FAILED
      cannot overwrite the truthful interruption result.
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

        Prebuilt mode: when ``config.prebuilt_topology`` and
        ``config.prebuilt_coordinates`` are both set, the
        runner stages the caller-supplied ``.top`` / ``.gro``
        into canonical filenames at the START of the protocol
        (BEFORE the stage loop). The TOPOLOGY stage's command
        list is empty (returned by ``build_commands``), so the
        per-stage logic skips ``pdb2gmx``. The prebuilt source
        digests are recorded into the stage manifest under
        ``stages[TOPOLOGY].prebuilt_source`` so a future
        invocation that supplies a different
        ``prebuilt_topology`` / ``prebuilt_coordinates`` pair
        correctly invalidates the cached stage (see the
        ``_stage_already_complete`` short-circuit).
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

        # Prebuilt stage (no subprocess — file copy only).
        prebuilt_failure = self._stage_prebuilt(work_dir, config, started)
        if prebuilt_failure is not None:
            return prebuilt_failure

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
                # Halt: a missing-input FAILED on the next stage
                # would overwrite the truthful interruption result.
                # The next invocation re-enters the RUNNING stage
                # via the on-disk ``.cpt`` (or restarts fresh).
                interrupted += 1
                exit_code = rc
                error = f"stage {stage.kind.value} interrupted by SIGTERM (rc={rc})"
                break
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
        - Manifest silent AND no ``.cpt`` AND minimum outputs on
          disk → mark COMPLETED, return ``was_skipped=True``
          (LEGACY recovery path; see the guard below).
        - Otherwise: emit ``.mdp``, build commands, execute.

        MD-stage resume:
        - ``.cpt`` exists → resume via ``-cpi`` + ``-append``.
        - No ``.cpt`` → start fresh.

        INTERRUPTED-STAGE RECOVERY: a RUNNING manifest or an
        on-disk ``.cpt`` is always re-entered via the execution
        path. The legacy disk-output fallback fires only when the
        manifest is silent AND no ``.cpt`` exists, so a
        post-SIGTERM state is never silently promoted to
        ``COMPLETED``.
        """
        # --- Skip accounting (manifest authority) ---
        # The skip accounting is delegated to a single helper so
        # this orchestrator stays at one branch level. The helper
        # returns ``True`` when the stage should be skipped.
        if _stage_should_skip(work_dir, stage, config):
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

    def _stage_prebuilt(
        self,
        work_dir: Path,
        config: GromacsProtocolConfig,
        started: float,
    ) -> GromacsProtocolResult | None:
        """Stage the caller-supplied prebuilt .top/.gro (B4 + cascade invalidation).

        Compares the supplied prebuilt source digests to the
        cached ones BEFORE staging the new files. If the source
        has changed, every downstream stage that depends on
        ``topol.top`` / ``processed.gro`` is invalidated —
        their on-disk outputs are quarantined to
        ``.stale/<UTC>/`` and their manifest entries are reset
        to PENDING so the per-stage loop re-runs them from
        scratch. Without this guard the staging block would
        re-record the manifest BEFORE the loop, making the
        invalidation check unreachable (probe 5 surfaced this).

        Returns ``None`` on success, or a failure
        :class:`GromacsProtocolResult` on staging error.
        """
        if not (config.prebuilt_topology and config.prebuilt_coordinates):
            return None

        if _prebuilt_source_changed(work_dir, config):
            logger.info(
                "prebuilt source digests differ from cached values; "
                "invalidating downstream dependent stages"
            )
            _invalidate_downstream_for_prebuilt_change(work_dir)

        try:
            stage_prebuilt_topology(config, work_dir)
            prebuilt_meta = _prebuilt_meta(config)
            record_stage_status(
                work_dir,
                StageKind.TOPOLOGY.value,
                StageStatus.COMPLETED,
                outputs=stage_minimum_outputs(StageKind.TOPOLOGY, "topol"),
                prebuilt_source=prebuilt_meta,
            )
        except (FileNotFoundError, ValueError) as exc:
            return GromacsProtocolResult(
                name=config.name,
                output_dir=str(work_dir),
                replica_index=config.replica_index,
                replicas_total=config.replicas_total,
                stage_statuses={},
                succeeded=0,
                failed=1,
                skipped=0,
                interrupted=0,
                validated=0,
                dry_run=self._dry_run,
                exit_code=2,
                duration_seconds=time.monotonic() - started,
                error=f"prebuilt topology staging failed: {exc}",
            )
        return None

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


def _cached_stage_invalidated_for_topology(work_dir: Path, config: GromacsProtocolConfig) -> bool:
    """Return True iff the cached TOPOLOGY stage is invalidated by a prebuilt-source mismatch.

    Helper extracted from :meth:`GromacsProtocolRunner._run_single_stage`
    so the stage's branch structure stays under the complexity
    gate. A different prebuilt source (different ``.top`` /
    ``.gro`` pair supplied by the caller) MUST invalidate the
    cached stage even when the manifest status is COMPLETED.
    """
    return _prebuilt_source_changed(work_dir, config)


def _invalidate_downstream_for_prebuilt_change(work_dir: Path) -> None:
    """Invalidate every dependent stage when the prebuilt source has changed.

    B4 — when the supplied prebuilt ``.top`` / ``.gro`` differs from
    the cached digests, every stage that depends on those files
    must be invalidated. The TOPOLOGY stage itself is RE-staged
    below by the caller; the downstream stages (BOX, SOLVATE,
    IONS, MINIMIZE, EQUIL_NVT, EQUIL_NPT, PRODUCTION) need their
    cached outputs quarantined to ``.stale/<UTC>/`` and their
    manifest entries reset to PENDING so the per-stage loop
    re-runs them from scratch.

    Implementation:

    1. Read the current manifest.
    2. For every dependent stage kind, if its manifest entry is
       ``completed`` (or ``running`` — a partial run that was
       interrupted) or ``failed`` (a downstream failure that
       can't possibly be the prebuilt-source's fault), reset the
       entry to PENDING.
    3. Quarantine every on-disk output file that corresponds to
       those stages to ``.stale/<UTC>/`` for forensic review.
    4. Save the manifest.
    """
    manifest = load_stage_manifest(work_dir)
    stages_section = manifest.setdefault("stages", {})
    dependent_kinds = (
        StageKind.BOX,
        StageKind.SOLVATE,
        StageKind.IONS,
        StageKind.MINIMIZE,
        StageKind.EQUIL_NVT,
        StageKind.EQUIL_NPT,
        StageKind.PRODUCTION,
    )
    stale_dir = _prebuilt_stale_dir(work_dir)

    for kind in dependent_kinds:
        record = stages_section.get(kind.value)
        if record is None:
            continue
        existing_outputs = record.get("outputs")
        outputs: list[str] = (
            list(existing_outputs)
            if isinstance(existing_outputs, list)
            else list(stage_minimum_outputs(kind, _prefix_for(kind)))
        )
        _move_outputs_to_stale(work_dir, outputs, stale_dir)
        # Reset to PENDING; the per-stage loop will re-run it.
        stages_section[kind.value] = {
            "status": StageStatus.PENDING,
            "invalidated_by_prebuilt_source_change": True,
        }

    save_stage_manifest(work_dir, manifest)
    logger.info(
        "invalidated downstream stages for prebuilt source change; stale dir=%s",
        stale_dir,
    )


def _prefix_for(kind: StageKind) -> str:
    """Return the canonical ``-deffnm`` prefix for a stage kind."""
    from biolab_runners.gromacs.paths import GromacsFiles

    prefixes = {
        StageKind.BOX: "box",
        StageKind.SOLVATE: "solvate",
        StageKind.IONS: "ions",
        StageKind.MINIMIZE: GromacsFiles.MIN_PREFIX,
        StageKind.EQUIL_NVT: GromacsFiles.NVT_PREFIX,
        StageKind.EQUIL_NPT: GromacsFiles.NPT_PREFIX,
        StageKind.PRODUCTION: GromacsFiles.PROD_PREFIX,
    }
    return prefixes[kind]


def _prebuilt_stale_dir(work_dir: Path) -> Path:
    """Return the per-invocation ``.stale/<UTC>/`` directory."""
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S_%f") + f"_{os.getpid()}"
    stale_dir = work_dir / ".stale" / ts
    stale_dir.mkdir(parents=True, exist_ok=True)
    return stale_dir


def _move_outputs_to_stale(work_dir: Path, outputs: list[str], stale_dir: Path) -> None:
    """Best-effort move every named output to ``stale_dir`` (errors logged)."""
    import shutil

    for name in outputs:
        src = work_dir / name
        if not src.exists():
            continue
        try:
            shutil.move(str(src), str(stale_dir / name))
            logger.info("quarantined stale %s -> %s", src, stale_dir / name)
        except OSError as exc:
            logger.warning("quarantine failed for %s: %s", src, exc)


def _stage_should_skip(
    work_dir: Path,
    stage: ProtocolStage,
    config: GromacsProtocolConfig,
) -> bool:
    """Return True iff the stage is already complete and should be skipped.

    Encapsulates the manifest-authority decision (canonical
    path) plus the prebuilt-source invalidation (prebuilt path)
    plus the legacy disk-output fallback (recovery path). When
    the helper returns ``True``, the stage is recorded as
    COMPLETED and the runner returns ``was_skipped=True``. When
    it returns ``False``, the runner falls through to the
    execution path.

    The helper is module-level (not a method) so the helper
    itself doesn't add complexity to ``_run_single_stage``.
    """
    if config.force:
        return False

    # Manifest authority — cache hit.
    if _stage_already_complete(work_dir, stage):
        # Prebuilt mode invalidates a cached TOPOLOGY stage
        # when the supplied source digests differ from the
        # recorded ones.
        if stage.kind == StageKind.TOPOLOGY and _cached_stage_invalidated_for_topology(
            work_dir, config
        ):
            logger.info(
                "TOPOLOGY stage cached as COMPLETED but prebuilt source "
                "digests differ; invalidating cached stage"
            )
            return False
        record_stage_status(
            work_dir,
            stage.kind.value,
            StageStatus.COMPLETED,
            outputs=stage_minimum_outputs(stage.kind, stage.prefix),
        )
        return True

    # Legacy disk-output fallback — manifest silent AND no
    # on-disk .cpt AND minimum outputs on disk. Without this
    # guard, a Spot-reclaim RUNNING manifest with on-disk
    # outputs would be promoted to COMPLETED here, silently
    # dropping the -cpi resume.
    manifest_record = load_stage_manifest(work_dir).get("stages", {}).get(stage.kind.value)
    manifest_status = manifest_record.get("status") if manifest_record else None
    if (
        manifest_status is None
        and _checkpoint_for(work_dir, stage) is None
        and _outputs_complete_on_disk(work_dir, stage)
    ):
        record_stage_status(
            work_dir,
            stage.kind.value,
            StageStatus.COMPLETED,
            outputs=stage_minimum_outputs(stage.kind, stage.prefix),
        )
        return True

    return False


def _prebuilt_meta(config: GromacsProtocolConfig) -> dict[str, str]:
    """Compute the prebuilt-source digest metadata for the manifest binding.

    Used by :class:`GromacsProtocolRunner.run_protocol` to record the
    prebuilt source paths + digests in the TOPOLOGY stage's
    manifest record. A future invocation with a different
    prebuilt source sees a different digest and the cached stage
    is invalidated (see :func:`_prebuilt_source_changed`).

    Lives at module level rather than inside the class so it
    can be unit-tested without instantiating a runner.
    """
    from biolab_runners.provenance import compute_file_digest

    return {
        "prebuilt_topology": config.prebuilt_topology,
        "prebuilt_coordinates": config.prebuilt_coordinates,
        "sha256_topology": compute_file_digest(Path(config.prebuilt_topology)) or "",
        "sha256_coordinates": compute_file_digest(Path(config.prebuilt_coordinates)) or "",
    }
