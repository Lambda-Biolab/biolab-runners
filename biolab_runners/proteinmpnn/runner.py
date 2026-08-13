"""ProteinMPNN runner."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.proteinmpnn.utils import (
    DesignRecord,
    DesignRecordStatus,
    _invoke_with_metadata,
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

    The ``provenance`` field carries the S2 reproducibility record
    when the runner executed; the idempotent / dry-run paths
    initialise it to :data:`EMPTY_PROVENANCE`. ``provenance.canonical_output``
    is the tuple of raw FASTA sequences *before* any downstream
    D-residue substitution (``chem_001``) — preserving the canonical
    output is a hard requirement of the S2 plan.
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
        return any(target.glob("*.fa"))

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
            duration, and S2 provenance. ``provenance.canonical_output``
            holds the raw FASTA sequences *before* downstream
            D-residue substitution — that is the S2 contract.
        """
        # Canonicalise the caller-supplied image digest BEFORE any
        # subprocess work so downstream manifest comparison sees a
        # single form regardless of how the caller wrote it.
        image_digest = validate_image_digest(image_digest)

        cfg = config or self._config_override
        if cfg is None:
            raise ValueError("ProteinMPNNConfig is required: pass it to run() or the runner")

        output_dir = self._design_dir(cfg, input_pdb)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not force and self.is_complete(cfg, input_pdb):
            records = _parse_records(output_dir)
            return ProteinMPNNResult(
                name=cfg.name,
                output_dir=str(output_dir),
                records=records,
                succeeded=len(records),
                failed=0,
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
                ),
            )

        config_dict = _config_to_cli(cfg, input_pdb)
        if dry_run:
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
                ),
            )

        import time

        started = time.monotonic()
        result = _invoke_with_metadata(
            config_dict=config_dict,
            input_pdb=input_pdb,
            output_dir=output_dir,
            binary_prefix=self._binary_prefix,
            timeout_seconds=self._timeout_seconds,
        )
        records = _parse_records(output_dir)
        succeeded = sum(1 for r in records if r.status == DesignRecordStatus.SUCCEEDED)
        failed = len(records) - succeeded
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
            ),
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
    ) -> ProvenanceMetadata:
        """Assemble the S2 provenance record for a ProteinMPNN run.

        ``canonical_output`` captures the raw FASTA sequences *before*
        the downstream ``chem_001`` D-residue rewrite. Preserving the
        canonical output is the S2 contract — once ``chem_001`` runs,
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
    for path in sorted(output_dir.glob("*.fa")):
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
