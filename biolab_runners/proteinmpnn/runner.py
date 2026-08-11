"""ProteinMPNN runner."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.proteinmpnn.utils import (
    DesignRecord,
    DesignRecordStatus,
    invoke,
    parse_fasta_sequences,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from biolab_runners.proteinmpnn.config import ProteinMPNNConfig

logger = logging.getLogger(__name__)

__all__ = ["ProteinMPNNResult", "ProteinMPNNRunner"]


@dataclass(frozen=True)
class ProteinMPNNResult:
    """Outcome of one or more ProteinMPNN sequence designs."""

    name: str
    output_dir: str
    records: tuple[DesignRecord, ...] = ()
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    exit_code: int = 0
    duration_seconds: float = 0.0

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
    ) -> ProteinMPNNResult:
        """Run ProteinMPNN on ``input_pdb`` and return the parsed result."""
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
            )

        import time

        started = time.monotonic()
        exit_code = invoke(
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
            exit_code=exit_code,
            duration_seconds=time.monotonic() - started,
        )

    def run_batch(
        self,
        inputs: Iterable[Path],
        config: ProteinMPNNConfig | None = None,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> list[ProteinMPNNResult]:
        """Run ProteinMPNN for each pre-clustered backbone and return per-input results."""
        return [self.run(path, config, force=force, dry_run=dry_run) for path in inputs]

    def _design_dir(self, config: ProteinMPNNConfig, input_pdb: Path) -> Path:
        return self._output_root / config.name / input_pdb.stem


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
