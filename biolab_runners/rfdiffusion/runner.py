"""RFdiffusion runner.

A subprocess wrapper around the upstream ``RFdiffusion/run_inference.py``
CLI. The runner owns:

* ``submit()`` / ``dry_run`` semantics matching the boltz2 runner;
* ``idempotency`` via output-presence check (if ``<output_dir>/<name>/``
  has ``*.pdb`` files, the call is a no-op);
* ``force`` to bypass idempotency;
* structured result parsing into :class:`RecordData` objects;
* S2 provenance — every invocation attaches a
  :class:`biolab_runners.provenance.ProvenanceMetadata` record so
  downstream consumers can audit the inputs.

RFdiffusion itself is not pip-installable. The runner expects the
caller to either (a) install the upstream conda env (rare), or (b)
point ``RFDIFFUSION_BIN`` at a Docker-wrapped command (the GCP Batch
path).

Notes on seed / temperature forwarding (S2 honesty):

* The runner sets ``inference.deterministic = True`` when
  ``config.deterministic`` is true. Upstream's deterministic mode
  pins the RNG state internally — we deliberately do **not** also
  forward ``inference.seed``: that flag is supported only in
  recent RFdiffusion versions and silently breaks older wrappers.
  The provenance manifest records the caller's
  ``requested_seed`` and the upstream RNG intent
  (``"seed-not-forwarded"`` or ``"non-deterministic"``) so the
  audit can tell "the user wanted seed=42 but the runner did not
  forward it to upstream" — which is the honest RFdiffusion story.
  ``base_seed`` is ``None`` because no seed was forwarded.
* The ``executed_config_digest`` excludes ``seed`` for the same
  reason: changing only the seed must not flip the executed-config
  digest, because the seed did not affect what upstream ran.
* RFdiffusion does not expose a single ``temperature`` parameter
  (the diffusion process uses ``noise_scale_ca`` /
  ``noise_scale_frame``, which are tuned per-application and
  upstream-internal). The provenance manifest therefore records
  ``temperature=None`` for RFdiffusion runs.

Notes on cache hits (S2 honesty):

* On the idempotent path the runner returns existing files without
  invoking upstream. The provenance record sets ``executed=False``
  and ``cache_hit=True``, and ``executed_config_digest=None`` —
  the runner does not know which prior call produced the existing
  files, so it does not fabricate a digest. ``requested_config_digest``
  describes what *this* call asked for.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.provenance import (
    EMPTY_PROVENANCE,
    RNG_INTENT_NON_DETERMINISTIC,
    RNG_INTENT_SEED_NOT_FORWARDED,
    ProvenanceMetadata,
    compute_config_digest,
    compute_executed_config_digest,
    compute_file_digest,
    validate_image_digest,
)
from biolab_runners.rfdiffusion.utils import (
    RecordData,
    RecordDataStatus,
    _invoke_with_metadata,
    parse_backbone_pdb,
)

if TYPE_CHECKING:
    from biolab_runners.rfdiffusion.config import RFdiffusionConfig

logger = logging.getLogger(__name__)

__all__ = ["RFdiffusionResult", "RFdiffusionRunner"]


#: Field names excluded from the *executed* config digest for RFdiffusion.
#: ``seed`` is excluded because the runner does not forward
#: ``inference.seed`` to upstream — see the module docstring for the
#: full rationale.
EXCLUDED_FROM_EXECUTED_DIGEST: tuple[str, ...] = ("seed",)


@dataclass(frozen=True)
class RFdiffusionResult:
    """Outcome of one or more RFdiffusion design runs.

    The ``provenance`` field carries the S2 reproducibility record
    when the runner executed; the idempotent / dry-run / error
    paths initialise it to :data:`EMPTY_PROVENANCE` so the field is
    always present and the JSON serialisation is stable.
    """

    name: str
    output_dir: str
    records: tuple[RecordData, ...] = ()
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
        image_digest: str | None = None,
    ) -> RFdiffusionResult:
        """Generate backbones under ``<output_root>/<name>/``.

        Args:
            config: Per-invocation config. Falls back to the runner's
                default when ``None``.
            force: Re-run even if the design directory already contains
                PDBs. Bypasses the idempotency short-circuit.
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
            :class:`RFdiffusionResult` with parsed records, exit code,
            duration, and S2 provenance.
        """
        # Canonicalise the caller-supplied image digest BEFORE any
        # subprocess work so downstream manifest comparison sees a
        # single form regardless of how the caller wrote it.
        image_digest = validate_image_digest(image_digest)

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
                provenance=self._build_provenance(
                    cfg,
                    exit_code=0,
                    failure_reason="",
                    stderr_tail="",
                    image_digest=image_digest,
                    executed=False,
                    cache_hit=True,
                ),
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
                provenance=self._build_provenance(
                    cfg,
                    exit_code=0,
                    failure_reason="",
                    stderr_tail="",
                    image_digest=image_digest,
                    executed=False,
                    cache_hit=False,
                ),
            )

        started = time.monotonic()
        result = _invoke_with_metadata(
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
            exit_code=result.exit_code,
            duration_seconds=time.monotonic() - started,
            provenance=self._build_provenance(
                cfg,
                exit_code=result.exit_code,
                failure_reason=result.failure_reason,
                stderr_tail=result.stderr_tail,
                image_digest=image_digest,
                executed=True,
                cache_hit=False,
            ),
        )

    def _design_dir(self, config: RFdiffusionConfig) -> Path:
        return self._output_root / config.name

    def _build_provenance(
        self,
        cfg: RFdiffusionConfig,
        *,
        exit_code: int,
        failure_reason: str,
        stderr_tail: str,
        image_digest: str | None,
        executed: bool,
        cache_hit: bool,
    ) -> ProvenanceMetadata:
        """Assemble the S2 provenance record for ``cfg``.

        The ``target_pdb`` field on the config is the source backbone
        for RFdiffusion. When unset, the manifest records ``None``
        rather than fabricating a digest — absence is a signal. When
        set but the file is absent, we warn at the logger so a
        operator notices but still record ``None`` in the manifest
        (the audit can correlate the warning with the config).
        """
        if cfg.target_pdb:
            target = Path(cfg.target_pdb)
            if not target.exists():
                logger.warning(
                    "RFdiffusionConfig.target_pdb=%s does not exist; "
                    "provenance.source_backbone_digest will be None",
                    cfg.target_pdb,
                )
        else:
            target = None
        return ProvenanceMetadata(
            model_identifier=cfg.checkpoint,
            temperature=None,
            image_digest=image_digest,
            source_backbone_digest=compute_file_digest(target) if target is not None else None,
            exit_code=exit_code,
            failure_reason=failure_reason,
            stderr_tail=stderr_tail,
            base_seed=None,
            requested_seed=cfg.seed,
            task_count=cfg.task_count,
            rng_intent=RNG_INTENT_SEED_NOT_FORWARDED
            if cfg.deterministic
            else RNG_INTENT_NON_DETERMINISTIC,
            canonical_output=(),
            requested_config_digest=compute_config_digest(cfg),
            executed_config_digest=compute_executed_config_digest(
                cfg, exclude_fields=EXCLUDED_FROM_EXECUTED_DIGEST
            )
            if executed
            else None,
            executed=executed,
            cache_hit=cache_hit,
        )


def _config_to_cli(config: RFdiffusionConfig) -> dict[str, str]:
    """Translate :class:`RFdiffusionConfig` into the upstream CLI kwargs.

    Returns a flat ``key -> str`` mapping consumed by
    :func:`biolab_runners.rfdiffusion.utils._invoke_with_metadata`.

    Notes on what's deliberately **not** forwarded:

    * ``inference.seed`` — only supported in recent RFdiffusion
      versions; older wrappers reject unknown kwargs. Upstream's
      ``inference.deterministic`` already pins the RNG state, so
      the manifest still records the user's ``seed`` as
      ``requested_seed`` for audit.
    * ``diffusion.noise_scale_ca`` — RFdiffusion's diffusion
      process uses upstream-internal noise scales, not a single
      sampling temperature. Mapping ``config.temperature`` to
      ``noise_scale_ca`` would silently change upstream behaviour
      and is intentionally avoided.
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
    """Walk ``design_dir`` and parse each PDB file into a :classRecordData`."""
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
