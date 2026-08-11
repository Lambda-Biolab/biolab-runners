"""GROMACS subprocess runner."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.gromacs.utils import (
    GromacsRecord,
    GromacsRecordStatus,
    invoke,
    parse_nthcol_energy,
)

if TYPE_CHECKING:
    from biolab_runners.gromacs.config import GromacsConfig

logger = logging.getLogger(__name__)

__all__ = ["GromacsResult", "GromacsRunner"]


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


class GromacsRunner:
    """Subprocess wrapper around the upstream GROMACS CLI."""

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
    """Run ``gmx mdrun`` once; returns the process exit code.

    This thin wrapper exists so tests can monkeypatch the runner.
    Real callers use the upstream :mod:`biolab_runners.gromacs.utils`
    implementation, which is the same function re-exported here.
    """
    from biolab_runners.gromacs.utils import invoke as _invoke

    return _invoke(
        config_dict=config_dict,
        output_dir=output_dir,
        mdrun_extra=mdrun_extra,
        binary_prefix=binary_prefix,
        timeout_seconds=timeout_seconds,
    )
