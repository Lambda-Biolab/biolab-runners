"""ProteinMPNN runner."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.contracts import ArtifactReference, ExecutionMode, ExecutionStatus
from biolab_runners.proteinmpnn.utils import (
    DesignRecord,
    DesignRecordStatus,
    _invoke_with_metadata,
    build_invocation_command,
    parse_fasta_sequences,
)
from biolab_runners.provenance import (
    EMPTY_PROVENANCE,
    RNG_INTENT_SINGLE_STREAM,
    ProvenanceMetadata,
    compute_config_digest,
    compute_executed_config_digest,
    compute_file_digest,
    validate_image_digest,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from biolab_runners.proteinmpnn.config import ProteinMPNNConfig

logger = logging.getLogger(__name__)

__all__ = ["ProteinMPNNResult", "ProteinMPNNRunner"]


#: Field names excluded from the *executed* config digest for ProteinMPNN.
#: Empty by default — ProteinMPNN forwards every config field the
#: dataclass exposes (``--seed``, ``--sampling_temp``, ``--model_name``,
#: ``--num_seq_per_target``, ``--ca_only``, ``--fixed_positions``,
#: ``--omit_AA``), so the requested and executed digests agree.
EXCLUDED_FROM_EXECUTED_DIGEST: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProteinMPNNResult:
    """Outcome of one or more ProteinMPNN sequence designs.

    The ``provenance`` field carries the reproducibility record
    when the runner executed; the idempotent / dry-run paths
    initialise it to :data:`EMPTY_PROVENANCE`. ``provenance.canonical_output``
    is the tuple of raw FASTA sequences *before* any downstream
    D-residue substitution — preserving the canonical output keeps the
    upstream result auditable.
    """

    name: str
    output_dir: str
    records: tuple[DesignRecord, ...] = ()
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    exit_code: int = 0
    duration_seconds: float = 0.0
    provenance: ProvenanceMetadata = EMPTY_PROVENANCE
    status: ExecutionStatus = ExecutionStatus.INCOMPLETE
    artifacts: tuple[ArtifactReference, ...] = ()
    execution_mode: ExecutionMode = ExecutionMode.SUBPROCESS

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
            "provenance": self.provenance.to_dict(),
            "status": self.status,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "execution_mode": self.execution_mode,
        }


class ProteinMPNNRunner:
    """Subprocess wrapper for the upstream ProteinMPNN CLI."""

    def __init__(
        self,
        *,
        config: ProteinMPNNConfig | None = None,
        binary_prefix: list[str] | None = None,
        output_root: Path | None = None,
        timeout_seconds: int = 3600,
    ) -> None:
        self._config_override = config
        self._binary_prefix = binary_prefix
        self._output_root = output_root or Path.cwd() / "proteinmpnn_output"
        self._timeout_seconds = timeout_seconds

    @property
    def output_root(self) -> Path:
        """Return the root directory into which FASTA outputs are written."""
        return self._output_root

    def is_complete(self, config: ProteinMPNNConfig, input_pdb: Path) -> bool:
        """Return True if the design FASTA already exists for ``input_pdb``."""
        target = self._design_dir(config, input_pdb)
        if not target.exists():
            return False
        return bool(_fasta_paths(target))

    def run(
        self,
        input_pdb: Path,
        config: ProteinMPNNConfig | None = None,
        *,
        force: bool = False,
        dry_run: bool = False,
        image_digest: str | None = None,
    ) -> ProteinMPNNResult:
        """Run ProteinMPNN on ``input_pdb`` and return the parsed result.

        Args:
            input_pdb: Backbone PDB to design sequences for.
            config: Per-invocation config. Falls back to the runner's
                default when ``None``.
            force: Re-run even if the FASTA already exists.
            dry_run: Validate inputs and log the command without
                executing. Exit code is ``0``; the ``provenance`` field
                records the canonical *requested* config digest (the
                executed digest is ``None`` — see
                :attr:`ProvenanceMetadata.executed_config_digest`).
            image_digest: Caller-supplied Docker image digest (e.g.
                ``"sha256:abc..."`` or the bare 64-char hex form).
                Validated and normalised to OCI form at the entry
                point of this method, before any subprocess work.
                See :func:`biolab_runners.provenance.validate_image_digest`.

        Returns:
            :class:`ProteinMPNNResult` with parsed records, exit code,
            duration, and shared provenance. ``provenance.canonical_output``
            holds the raw FASTA sequences *before* downstream
            D-residue substitution — that is the shared provenance contract.
        """
        # Canonicalise the caller-supplied image digest BEFORE any
        # subprocess work so downstream manifest comparison sees a
        # single form regardless of how the caller wrote it.
        image_digest = validate_image_digest(image_digest)
        cfg = config or self._config_override
        if cfg is None:
            raise ValueError("ProteinMPNNConfig is required: pass it to run() or the runner")
        binary_prefix = _effective_binary_prefix(self._binary_prefix)
        execution_mode = _execution_mode(binary_prefix)

        output_dir = self._design_dir(cfg, input_pdb)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not force and self.is_complete(cfg, input_pdb):
            records = _parse_records(output_dir)
            status = _status_from_records(0, records, cached=True)
            artifacts = _artifacts_for_records(records, output_dir)
            succeeded, failed = _record_counts(records)
            return ProteinMPNNResult(
                name=cfg.name,
                output_dir=str(output_dir),
                records=records,
                succeeded=succeeded,
                failed=failed,
                skipped=len(records),
                exit_code=0,
                duration_seconds=0.0,
                provenance=self._build_provenance(
                    cfg,
                    input_pdb,
                    records,
                    exit_code=0,
                    failure_reason="",
                    stderr_tail="",
                    image_digest=image_digest,
                    executed=False,
                    cache_hit=True,
                    status=status,
                    artifacts=artifacts,
                    execution_mode=execution_mode,
                ),
                status=status,
                artifacts=artifacts,
                execution_mode=execution_mode,
            )

        config_dict = _config_to_cli(cfg, input_pdb)
        intended_command = build_invocation_command(
            config_dict=config_dict,
            input_pdb=input_pdb,
            output_dir=output_dir,
            binary_prefix=binary_prefix,
        )
        if dry_run:
            status = ExecutionStatus.DRY_RUN
            return ProteinMPNNResult(
                name=cfg.name,
                output_dir=str(output_dir),
                records=(),
                succeeded=0,
                failed=0,
                skipped=0,
                exit_code=0,
                duration_seconds=0.0,
                provenance=self._build_provenance(
                    cfg,
                    input_pdb,
                    (),
                    exit_code=0,
                    failure_reason="",
                    stderr_tail="",
                    image_digest=image_digest,
                    executed=False,
                    cache_hit=False,
                    status=status,
                    execution_mode=execution_mode,
                    command=intended_command,
                ),
                status=status,
                execution_mode=execution_mode,
            )

        import time

        started = time.monotonic()
        result = _invoke_with_metadata(
            config_dict=config_dict,
            input_pdb=input_pdb,
            output_dir=output_dir,
            binary_prefix=binary_prefix,
            timeout_seconds=self._timeout_seconds,
        )
        records = _parse_records(output_dir)
        succeeded = sum(1 for r in records if r.status == DesignRecordStatus.SUCCEEDED)
        failed = len(records) - succeeded
        status = _status_from_records(result.exit_code, records)
        artifacts = _artifacts_for_records(records, output_dir)
        return ProteinMPNNResult(
            name=cfg.name,
            output_dir=str(output_dir),
            records=records,
            succeeded=succeeded,
            failed=failed,
            skipped=0,
            exit_code=result.exit_code,
            duration_seconds=time.monotonic() - started,
            provenance=self._build_provenance(
                cfg,
                input_pdb,
                records,
                exit_code=result.exit_code,
                failure_reason=result.failure_reason,
                stderr_tail=result.stderr_tail,
                image_digest=image_digest,
                executed=True,
                cache_hit=False,
                status=status,
                artifacts=artifacts,
                execution_mode=execution_mode,
                command=result.command or intended_command,
            ),
            status=status,
            artifacts=artifacts,
            execution_mode=execution_mode,
        )

    def run_batch(
        self,
        inputs: Iterable[Path],
        config: ProteinMPNNConfig | None = None,
        *,
        force: bool = False,
        dry_run: bool = False,
        image_digest: str | None = None,
    ) -> list[ProteinMPNNResult]:
        """Run ProteinMPNN for each pre-clustered backbone and return per-input results."""
        return [
            self.run(path, config, force=force, dry_run=dry_run, image_digest=image_digest)
            for path in inputs
        ]

    def _design_dir(self, config: ProteinMPNNConfig, input_pdb: Path) -> Path:
        return self._output_root / config.name / input_pdb.stem

    def _build_provenance(
        self,
        cfg: ProteinMPNNConfig,
        input_pdb: Path,
        records: tuple[DesignRecord, ...],
        *,
        exit_code: int,
        failure_reason: str,
        stderr_tail: str,
        image_digest: str | None,
        executed: bool,
        cache_hit: bool,
        status: ExecutionStatus = ExecutionStatus.SUCCEEDED,
        artifacts: tuple[ArtifactReference, ...] = (),
        execution_mode: ExecutionMode = ExecutionMode.SUBPROCESS,
        command: tuple[str, ...] = (),
    ) -> ProvenanceMetadata:
        """Assemble the provenance record for a ProteinMPNN run.

        ``canonical_output`` captures the raw FASTA sequences *before*
        the downstream D-residue rewrite. Preserving the canonical
        output is the provenance contract — once downstream chemistry runs,
        the raw sequences are lost from the design path but must
        remain visible in the manifest for audit.

        ProteinMPNN forwards ``--seed`` to upstream, so
        ``base_seed`` equals ``requested_seed`` and the executed
        config digest includes every config field (no
        ``exclude_fields``).
        """
        return ProvenanceMetadata(
            model_identifier=cfg.model_name,
            temperature=cfg.temperature,
            image_digest=image_digest,
            source_backbone_digest=compute_file_digest(input_pdb),
            exit_code=exit_code,
            failure_reason=failure_reason,
            stderr_tail=stderr_tail,
            base_seed=cfg.seed,
            requested_seed=cfg.seed,
            task_count=cfg.task_count,
            rng_intent=RNG_INTENT_SINGLE_STREAM,
            canonical_output=tuple(r.sequence for r in records if r.sequence),
            requested_config_digest=compute_config_digest(cfg),
            executed_config_digest=compute_executed_config_digest(
                cfg, exclude_fields=EXCLUDED_FROM_EXECUTED_DIGEST
            )
            if executed
            else None,
            executed=executed,
            cache_hit=cache_hit,
            runner_name="proteinmpnn",
            execution_mode=execution_mode,
            status=status,
            artifacts=artifacts,
            command=command,
        )


def _config_to_cli(config: ProteinMPNNConfig, input_pdb: Path) -> dict[str, str]:
    """Translate :class:`ProteinMPNNConfig` into the upstream CLI kwargs."""
    payload: dict[str, str] = {
        "model_name": config.model_name,
        "num_seq_per_target": str(config.task_count),
        "sampling_temp": str(config.temperature),
        "seed": str(config.seed),
    }
    if config.ca_only:
        payload["ca_only"] = "True"
    if config.fixed_positions:
        payload["fixed_positions"] = ",".join(str(p - 1) for p in config.fixed_positions)
    if config.omit_aa:
        payload["omit_AA"] = config.omit_aa
    payload["pdb_path"] = input_pdb.name  # upstream expects basename
    for key, value in config.extra.items():
        payload[key] = str(value)
    return payload


def _parse_records(output_dir: Path) -> tuple[DesignRecord, ...]:
    """Parse every FASTA in ``output_dir`` into a :class:`DesignRecord`."""
    records: list[DesignRecord] = []
    for path in _fasta_paths(output_dir):
        try:
            fasta = parse_fasta_sequences(path)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("failed to parse %s: %s", path, exc)
            records.append(
                DesignRecord(
                    index=len(records),
                    sequence="",
                    score=0.0,
                    path=str(path),
                    status=DesignRecordStatus.FAILED,
                    error=str(exc),
                )
            )
            continue
        for record_index, (_name, sequence) in enumerate(fasta):
            records.append(
                DesignRecord(
                    index=len(records),
                    sequence=sequence,
                    score=0.0,
                    path=str(path),
                    status=DesignRecordStatus.SUCCEEDED,
                )
            )
            _ = record_index
    return tuple(records)


def _artifacts_for_records(
    records: tuple[DesignRecord, ...], output_dir: Path | None = None
) -> tuple[ArtifactReference, ...]:
    """Describe parsed FASTA outputs without fabricating absent files."""
    return tuple(
        ArtifactReference.from_path(record.path, kind="sequence", root=output_dir)
        for record in records
        if record.path and not Path(record.path).is_symlink()
    )


def _fasta_paths(output_dir: Path) -> tuple[Path, ...]:
    """Return legacy top-level and stock ``seqs/`` FASTA outputs."""
    return tuple(
        sorted(
            path
            for path in {*output_dir.glob("*.fa"), *output_dir.glob("seqs/*.fa")}
            if not path.is_symlink()
        )
    )


def _status_from_records(
    exit_code: int,
    records: tuple[DesignRecord, ...],
    *,
    cached: bool = False,
) -> ExecutionStatus:
    """Map legacy ProteinMPNN result fields to the shared status vocabulary."""
    if exit_code == 124:
        return ExecutionStatus.TIMEOUT
    if exit_code < 0:
        return ExecutionStatus.INTERRUPTED
    if exit_code != 0:
        return ExecutionStatus.FAILED
    if not records:
        return ExecutionStatus.INCOMPLETE
    if any(record.status != DesignRecordStatus.SUCCEEDED for record in records):
        return ExecutionStatus.MALFORMED
    return ExecutionStatus.CACHED if cached else ExecutionStatus.SUCCEEDED


def _record_counts(records: tuple[DesignRecord, ...]) -> tuple[int, int]:
    """Count records by their parsed status for cache accounting."""
    succeeded = sum(1 for record in records if record.status == DesignRecordStatus.SUCCEEDED)
    failed = sum(1 for record in records if record.status != DesignRecordStatus.SUCCEEDED)
    return succeeded, failed


def _execution_mode(binary_prefix: list[str] | None) -> ExecutionMode:
    """Identify direct or container-backed ProteinMPNN execution."""
    if binary_prefix and any(token.startswith("container://") for token in binary_prefix):
        return ExecutionMode.CONTAINER_URI
    return ExecutionMode.SUBPROCESS


def _effective_binary_prefix(binary_prefix: list[str] | None) -> list[str]:
    """Resolve the command prefix used for dispatch and mode reporting."""
    if binary_prefix is not None:
        return list(binary_prefix)
    return [os.environ.get("PROTEINMPNN_BIN", "proteinmpnn")]
