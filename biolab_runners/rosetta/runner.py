"""Rosetta CLI subprocess runner."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.rosetta.utils import (
    RelaxRecord,
    RelaxRecordStatus,
    invoke,
    parse_score_files,
)

if TYPE_CHECKING:
    from biolab_runners.rosetta.config import RosettaConfig

logger = logging.getLogger(__name__)

__all__ = ["RosettaResult", "RosettaRunner"]


@dataclass(frozen=True)
class RosettaResult:
    """Outcome of one Rosetta run."""

    name: str
    output_dir: str
    records: tuple[RelaxRecord, ...] = ()
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


class RosettaRunner:
    """Subprocess wrapper around the upstream Rosetta CLI."""

    def __init__(
        self,
        *,
        config: RosettaConfig | None = None,
        binary_prefix: list[str] | None = None,
        output_root: Path | None = None,
        timeout_seconds: int = 3600,
    ) -> None:
        self._config_override = config
        self._binary_prefix = binary_prefix
        self._output_root = output_root or Path.cwd() / "rosetta_output"
        self._timeout_seconds = timeout_seconds

    @property
    def output_root(self) -> Path:
        """Return the root directory into which Rosetta writes outputs."""
        return self._output_root

    def is_complete(self, config: RosettaConfig) -> bool:
        """Return True if at least one scored output already exists."""
        directory = self._design_dir(config)
        if not directory.exists():
            return False
        return any(directory.glob("score.sc"))

    def run(
        self,
        config: RosettaConfig | None = None,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> RosettaResult:
        """Run Rosetta and return the parsed result."""
        cfg = config or self._config_override
        if cfg is None:
            raise ValueError("RosettaConfig is required: pass it to run() or the runner")

        output_dir = self._design_dir(cfg)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not force and self.is_complete(cfg):
            records = parse_score_files(sorted(output_dir.glob("score.sc")))
            return RosettaResult(
                name=cfg.name,
                output_dir=str(output_dir),
                records=tuple(records),
                succeeded=len(records),
                failed=0,
                skipped=len(records),
                exit_code=0,
                duration_seconds=0.0,
            )

        import time

        if dry_run:
            return RosettaResult(
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
            config=config_dict,
            output_dir=output_dir,
            binary_prefix=self._binary_prefix,
            timeout_seconds=self._timeout_seconds,
        )
        records = parse_score_files(sorted(output_dir.glob("score.sc")))
        succeeded = sum(1 for r in records if r.status == RelaxRecordStatus.SUCCEEDED)
        failed = len(records) - succeeded
        return RosettaResult(
            name=cfg.name,
            output_dir=str(output_dir),
            records=tuple(records),
            succeeded=succeeded,
            failed=failed,
            skipped=0,
            exit_code=exit_code,
            duration_seconds=time.monotonic() - started,
        )

    def _design_dir(self, config: RosettaConfig) -> Path:
        return self._output_root / config.name


def _config_to_cli(config: RosettaConfig) -> dict[str, str]:
    """Translate :class:`RosettaConfig` into a flat CLI kwargs dict."""
    payload: dict[str, str] = {
        "s": config.script_file,
        "in:file:s": config.input_pdb,
        "out:path:all": config.output_dir,
        "nstruct": str(config.nstruct),
    }
    for key, value in config.extra.items():
        payload[str(key)] = str(value)
    for flag in config.extra_flags:
        if "=" in flag:
            key, _, value = flag.partition("=")
            payload[key] = value
        else:
            payload[flag] = ""
    return payload
