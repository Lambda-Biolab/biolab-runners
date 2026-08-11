"""RFdiffusion runner.

A subprocess wrapper around the upstream ``RFdiffusion/run_inference.py``
CLI. The runner owns:

* ``submit()`` / ``dry_run`` semantics matching the boltz2 runner;
* ``idempotency`` via output-presence check (if ``<output_dir>/<name>/``
  has ``*.pdb`` files, the call is a no-op);
* ``force`` to bypass idempotency;
* structured result parsing into :class:`RecordData` objects.

RFdiffusion itself is not pip-installable. The runner expects the
caller to either (a) install the upstream conda env (rare), or (b)
point ``RFDIFFUSION_BIN`` at a Docker-wrapped command (the GCP Batch
path).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.rfdiffusion.utils import (
    RecordData,
    RecordDataStatus,
    invoke,
    parse_backbone_pdb,
)

if TYPE_CHECKING:
    from biolab_runners.rfdiffusion.config import RFdiffusionConfig

logger = logging.getLogger(__name__)

__all__ = ["RFdiffusionResult", "RFdiffusionRunner"]


@dataclass(frozen=True)
class RFdiffusionResult:
    """Outcome of one or more RFdiffusion design runs."""

    name: str
    output_dir: str
    records: tuple[RecordData, ...] = ()
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


class RFdiffusionRunner:
    """Subprocess wrapper for the upstream RFdiffusion CLI."""

    def __init__(
        self,
        *,
        config: RFdiffusionConfig | None = None,
        binary_prefix: list[str] | None = None,
        output_root: Path | None = None,
        timeout_seconds: int = 3600,
    ) -> None:
        self._config_override = config
        self._binary_prefix = binary_prefix
        self._output_root = output_root or Path.cwd() / "rfdiffusion_output"
        self._timeout_seconds = timeout_seconds

    @property
    def output_root(self) -> Path:
        """Return the root directory into which designs are written."""
        return self._output_root

    def is_complete(self, config: RFdiffusionConfig) -> bool:
        """Return True if ``config.name`` already has parsed backbones."""
        directory = self._design_dir(config)
        if not directory.exists():
            return False
        return any(directory.glob("*.pdb"))

    def run(
        self,
        config: RFdiffusionConfig | None = None,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> RFdiffusionResult:
        """Generate backbones under ``<output_root>/<name>/``."""
        cfg = config or self._config_override
        if cfg is None:
            raise ValueError("RFdiffusionConfig is required: pass it to run() or the runner")

        design_dir = self._design_dir(cfg)
        design_dir.mkdir(parents=True, exist_ok=True)

        if not force and self.is_complete(cfg):
            records = _parse_output_dir(design_dir)
            return RFdiffusionResult(
                name=cfg.name,
                output_dir=str(design_dir),
                records=records,
                succeeded=len(records),
                failed=0,
                skipped=len(records),
                exit_code=0,
                duration_seconds=0.0,
            )

        config_dict = _config_to_cli(cfg)
        if dry_run:
            return RFdiffusionResult(
                name=cfg.name,
                output_dir=str(design_dir),
                records=(),
                succeeded=0,
                failed=0,
                skipped=0,
                exit_code=0,
                duration_seconds=0.0,
            )

        started = time.monotonic()
        exit_code = invoke(
            config_dict=config_dict,
            output_dir=design_dir,
            binary_prefix=self._binary_prefix,
            timeout_seconds=self._timeout_seconds,
        )
        records = _parse_output_dir(design_dir)
        succeeded = sum(1 for r in records if r.status == RecordDataStatus.SUCCEEDED)
        failed = len(records) - succeeded
        return RFdiffusionResult(
            name=cfg.name,
            output_dir=str(design_dir),
            records=records,
            succeeded=succeeded,
            failed=failed,
            skipped=0,
            exit_code=exit_code,
            duration_seconds=time.monotonic() - started,
        )

    def _design_dir(self, config: RFdiffusionConfig) -> Path:
        return self._output_root / config.name


def _config_to_cli(config: RFdiffusionConfig) -> dict[str, str]:
    """Translate :class:`RFdiffusionConfig` into the upstream CLI kwargs.

    Returns a flat ``key -> str`` mapping consumed by
    :func:`biolab_runners.rfdiffusion.utils.invoke`.
    """
    payload: dict[str, str] = {
        "inference.num_designs": str(config.task_count),
        "contigmap.contigs": config.contigs,
    }
    if config.mode == "head_to_tail":
        payload["inference.cyclic"] = "True"
        payload["inference.cyc_chains"] = "a"
    elif config.mode == "disulfide":
        payload["inference.cyclic"] = "True"
        # Disulfide pairs are encoded as a comma-separated chain
        # specifier; the upstream parser turns ``X,Y`` into a Cys
        # pair between positions X and Y.
        payload["inference.cyc_chains"] = ",".join(f"{a},{b}" for a, b in config.disulfide_pairs)
    if config.deterministic:
        payload["inference.deterministic"] = "True"
    if config.hotspots:
        payload["ppi.hotspot_res"] = ",".join(config.hotspots)
    for key, value in config.extra.items():
        payload[key] = str(value)
    return payload


def _parse_output_dir(design_dir: Path) -> tuple[RecordData, ...]:
    """Walk ``design_dir`` and parse each PDB file into a :classRecordData."""
    records: list[RecordData] = []
    for path in sorted(design_dir.glob("*.pdb")):
        try:
            sequence = parse_backbone_pdb(path)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("failed to parse %s: %s", path, exc)
            records.append(
                RecordData(
                    index=len(records),
                    path=str(path),
                    sequence="",
                    status=RecordDataStatus.FAILED,
                    error=str(exc),
                )
            )
            continue
        records.append(
            RecordData(
                index=len(records),
                path=str(path),
                sequence=sequence,
                status=RecordDataStatus.SUCCEEDED,
            )
        )
    return tuple(records)
