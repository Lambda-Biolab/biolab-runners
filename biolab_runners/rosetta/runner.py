"""Rosetta CLI subprocess runner."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.contracts import (
    ArtifactReference,
    ExecutionMode,
    ExecutionStatus,
    RunnerTimeoutError,
)
from biolab_runners.provenance import (
    EMPTY_PROVENANCE,
    ProvenanceMetadata,
    build_execution_provenance,
    compute_config_digest,
    compute_executed_config_digest,
    compute_file_digest,
    validate_image_digest,
)
from biolab_runners.rosetta.utils import (
    RelaxRecord,
    RelaxRecordStatus,
    build_invocation_command,
    invoke,
    parse_score_files,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from biolab_runners.rosetta.config import ConstrainedRelaxOptions, RosettaConfig

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
        directory = self._effective_output_dir(config)
        if not directory.exists():
            return False
        return any(path.is_file() and not path.is_symlink() for path in directory.glob("score.sc"))

    def run(
        self,
        config: RosettaConfig | None = None,
        *,
        force: bool = False,
        dry_run: bool = False,
        image_digest: str | None = None,
    ) -> RosettaResult:
        """Run Rosetta and return the parsed result."""
        cfg = config or self._config_override
        if cfg is None:
            raise ValueError("RosettaConfig is required: pass it to run() or the runner")
        image_digest = validate_image_digest(image_digest)
        execution_mode = _execution_mode(self._binary_prefix)

        config_dict = _config_to_cli(cfg)
        output_dir = self._effective_output_dir(cfg, config_dict)
        config_dict["out:path:all"] = str(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        intended_command = build_invocation_command(
            config=config_dict,
            binary_prefix=self._binary_prefix,
        )

        if not force and self.is_complete(cfg):
            records = parse_score_files(sorted(output_dir.glob("score.sc")))
            # Count by per-record status so all-None / garbage
            # scorefiles (which `parse_score_files` now flags as
            # FAILED with a synthetic error) decrement ``succeeded``
            # and increment ``failed``. The runner's reported counts
            # reflect "this row meant something", not "this row was
            # syntactically readable".
            succeeded = sum(1 for r in records if r.status == RelaxRecordStatus.SUCCEEDED)
            failed = len(records) - succeeded
            status = _status_from_records(0, records, cached=True)
            artifacts = _artifacts_for_output(output_dir)
            return RosettaResult(
                name=cfg.name,
                output_dir=str(output_dir),
                records=tuple(records),
                succeeded=succeeded,
                failed=failed,
                skipped=len(records),
                exit_code=0,
                duration_seconds=0.0,
                provenance=_provenance(
                    status,
                    0,
                    artifacts,
                    image_digest,
                    execution_mode,
                    command=(),
                    requested_config_digest=compute_config_digest(cfg),
                    source_backbone_digest=compute_file_digest(Path(cfg.input_pdb)),
                    cache_hit=True,
                ),
                status=status,
                artifacts=artifacts,
                execution_mode=execution_mode,
            )

        import time

        if dry_run:
            status = ExecutionStatus.DRY_RUN
            return RosettaResult(
                name=cfg.name,
                output_dir=str(output_dir),
                records=(),
                succeeded=0,
                failed=0,
                skipped=0,
                exit_code=0,
                duration_seconds=0.0,
                provenance=_provenance(
                    status,
                    0,
                    (),
                    image_digest,
                    execution_mode,
                    command=intended_command,
                    requested_config_digest=compute_config_digest(cfg),
                    source_backbone_digest=compute_file_digest(Path(cfg.input_pdb)),
                ),
                status=status,
                execution_mode=execution_mode,
            )

        started = time.monotonic()
        try:
            exit_code = invoke(
                config=config_dict,
                output_dir=output_dir,
                binary_prefix=self._binary_prefix,
                timeout_seconds=self._timeout_seconds,
            )
        except RunnerTimeoutError:
            exit_code = 124
        records = parse_score_files(sorted(output_dir.glob("score.sc")))
        succeeded = sum(1 for r in records if r.status == RelaxRecordStatus.SUCCEEDED)
        failed = len(records) - succeeded
        status = _status_from_records(exit_code, records)
        artifacts = _artifacts_for_output(output_dir)
        return RosettaResult(
            name=cfg.name,
            output_dir=str(output_dir),
            records=tuple(records),
            succeeded=succeeded,
            failed=failed,
            skipped=0,
            exit_code=exit_code,
            duration_seconds=time.monotonic() - started,
            provenance=_provenance(
                status,
                exit_code,
                artifacts,
                image_digest,
                execution_mode,
                command=intended_command,
                requested_config_digest=compute_config_digest(cfg),
                executed_config_digest=compute_executed_config_digest(
                    {"command": list(intended_command)}
                ),
                source_backbone_digest=compute_file_digest(Path(cfg.input_pdb)),
                executed=True,
            ),
            status=status,
            artifacts=artifacts,
            execution_mode=execution_mode,
        )

    def _design_dir(self, config: RosettaConfig) -> Path:
        return Path(config.output_dir) if config.output_dir else self._output_root / config.name

    def _effective_output_dir(
        self,
        config: RosettaConfig,
        config_dict: dict[str, Any] | None = None,
    ) -> Path:
        cli_config = config_dict if config_dict is not None else _config_to_cli(config)
        output_dir = cli_config.get("out:path:all")
        return Path(str(output_dir)) if output_dir else self._design_dir(config)


def _artifacts_for_output(output_dir: Path) -> tuple[ArtifactReference, ...]:
    """Describe score outputs that exist on disk."""
    return tuple(
        ArtifactReference.from_path(path, kind="score", root=output_dir)
        for path in sorted(output_dir.glob("score.sc"))
        if not path.is_symlink()
    )


def _status_from_records(
    exit_code: int,
    records: list[RelaxRecord],
    *,
    cached: bool = False,
) -> ExecutionStatus:
    """Map Rosetta's legacy fields to the shared status vocabulary."""
    if exit_code == 124:
        return ExecutionStatus.TIMEOUT
    if exit_code < 0:
        return ExecutionStatus.INTERRUPTED
    if exit_code != 0:
        return ExecutionStatus.FAILED
    if not records:
        return ExecutionStatus.INCOMPLETE
    if any(record.status != RelaxRecordStatus.SUCCEEDED for record in records):
        return ExecutionStatus.MALFORMED
    return ExecutionStatus.CACHED if cached else ExecutionStatus.SUCCEEDED


def _provenance(
    status: ExecutionStatus,
    exit_code: int,
    artifacts: tuple[ArtifactReference, ...],
    image_digest: str | None,
    mode: ExecutionMode = ExecutionMode.SUBPROCESS,
    *,
    command: tuple[str, ...] = (),
    requested_config_digest: str = "",
    executed_config_digest: str | None = None,
    source_backbone_digest: str | None = None,
    executed: bool = False,
    cache_hit: bool = False,
) -> ProvenanceMetadata:
    """Build the generic Rosetta execution record."""
    return build_execution_provenance(
        runner_name="rosetta",
        execution_mode=mode,
        status=status,
        exit_code=exit_code,
        image_digest=image_digest,
        artifacts=artifacts,
        command=command,
        requested_config_digest=requested_config_digest,
        executed_config_digest=executed_config_digest,
        source_backbone_digest=source_backbone_digest,
        executed=executed,
        cache_hit=cache_hit,
    )


def _execution_mode(binary_prefix: list[str] | None) -> ExecutionMode:
    """Identify direct or container-backed Rosetta execution."""
    configured = os.environ.get("ROSETTA_BIN", "")
    if configured.startswith("container://") or (
        binary_prefix and binary_prefix[0].startswith("container://")
    ):
        return ExecutionMode.CONTAINER_URI
    return ExecutionMode.SUBPROCESS


def _config_to_cli(config: RosettaConfig) -> dict[str, Any]:
    """Translate :class:`RosettaConfig` into a CLI kwargs dict.

    Most entries become single ``-key value`` argv tokens at invoke
    time. The structured ``preparation_mode`` and
    ``constrained_relax`` options are translated into
    ``-parser:script_vars key=value`` pairs and stored as a
    ``list[str]`` value under that dict key — :func:`invoke` repeats
    the flag once per element so each ``%%variable%%`` lands as its
    own argv token. Pre-joining them into one space-separated
    string would silently feed Rosetta the wrong shape (its parser
    would treat the entire ``"k1=v1 k2=v2"`` blob as one variable).

    Precedence (later sources appended after earlier ones, in order;
    no source is allowed to *delete* a token already accumulated):

    1. :func:`_structured_script_vars` — the typed options.
    2. ``config.extra["parser:script_vars"]`` — the dict-channel
       escape hatch. May be ``str`` (whitespace-split into tokens)
       or ``Sequence[str]`` (each element preserved as a token).
    3. ``config.extra_flags`` entries of the form
       ``"parser:script_vars k=v ..."`` — trailing tokens after
       the literal key prefix become one script-var per token.

    All other ``config.extra`` and ``config.extra_flags`` entries
    use last-write-wins on the dict (the v0.5 contract).
    """
    payload: dict[str, Any] = {
        "parser:protocol": config.script_file,
        "in:file:s": config.input_pdb,
        "out:path:all": config.output_dir,
        "nstruct": str(config.nstruct),
    }

    script_vars: list[str] = list(_structured_script_vars(config))

    _apply_extra_mapping(payload, script_vars, config.extra)
    _apply_extra_flags(payload, script_vars, config.extra_flags)

    if script_vars:
        # Sequence value so ``invoke`` repeats the flag per element.
        # See ``_config_value_to_argv``.
        payload["parser:script_vars"] = script_vars

    return payload


def _apply_extra_mapping(
    payload: dict[str, Any],
    script_vars: list[str],
    extra: Mapping[str, Any],
) -> None:
    """Apply ``config.extra`` entries to ``payload`` / ``script_vars``.

    The ``"parser:script_vars"`` key is special-cased. A ``str``
    value is whitespace-split into separate tokens (so callers can
    pass either ``"k1=v1 k2=v2"`` or ``["k1=v1", "k2=v2"]``); a
    list / tuple value is used as-is. The accumulator stays
    ordered, so invoke emits ``-parser:script_vars k1=v1
    -parser:script_vars k2=v2`` rather than the wrong
    space-joined shape.
    """
    for key, value in extra.items():
        if key == "parser:script_vars":
            if isinstance(value, (list, tuple)):
                script_vars.extend(str(v) for v in value if str(v))
                continue
            user_value = str(value).strip()
            if user_value:
                # Whitespace-split so the dict-channel accepts the
                # user-friendly ``"k1=v1 k2=v2"`` form AND the
                # explicit ``["k1=v1", "k2=v2"]`` form interchangeably.
                script_vars.extend(user_value.split())
            continue
        payload[_canonical_cli_key(key)] = str(value)


def _apply_extra_flags(
    payload: dict[str, Any],
    script_vars: list[str],
    extra_flags: tuple[str, ...],
) -> None:
    """Apply ``config.extra_flags`` entries.

    ``parser:script_vars`` variants accumulate, everything else
    uses last-write-wins on the dict.

    Each flag entry is one of:

    - A bare flag (``"no_output"``) — sets
      ``payload[flag] = ""`` on the dict.
    - A ``key=value`` flag (``"beta=1.0"``) — sets
      ``payload[key] = value`` (the v0.5 last-write-wins contract).
    - A ``parser:script_vars`` flag — bare (no value) is ignored;
      ``parser:script_vars k1=v1 k2=v2 ...`` (one or more trailing
      tokens after the literal ``parser:script_vars`` prefix) is
      whitespace-split so each ``k=v`` becomes a separate
      script-var token.
    """
    for flag in extra_flags:
        if flag.startswith("parser:script_vars"):
            rest = flag[len("parser:script_vars") :].lstrip(" =").strip()
            if rest:
                # Split on whitespace so each ``k=v`` becomes one
                # script-var token in the accumulated list.
                script_vars.extend(rest.split())
            continue
        if "=" in flag:
            key, _, value = flag.partition("=")
            payload[_canonical_cli_key(key)] = value
        else:
            payload[_canonical_cli_key(flag)] = ""


def _canonical_cli_key(key: object) -> str:
    text = str(key)
    return {"s": "parser:protocol"}.get(text, text)


def _structured_script_vars(config: RosettaConfig) -> tuple[str, ...]:
    """Translate structured options into ``key=value`` script-var tokens.

    The returned tuple is empty when neither ``preparation_mode`` nor
    ``constrained_relax`` is set; callers then skip emitting the
    ``parser:script_vars`` flag entirely so the consumer protocol XML
    inherits the upstream default.
    """
    tokens: list[str] = []
    if config.preparation_mode is not None:
        tokens.append(f"prep_mode={config.preparation_mode}")
    if config.constrained_relax is not None:
        tokens.extend(_constrained_relax_vars(config.constrained_relax))
    return tuple(tokens)


def _constrained_relax_vars(opts: ConstrainedRelaxOptions) -> tuple[str, ...]:
    """Translate :class:`ConstrainedRelaxOptions` into ``key=value`` pairs.

    Each field is emitted only when explicitly set (non-``None``) so
    the protocol XML receives the union of what the caller requested
    rather than a fully populated bag of N/A fields.
    """
    tokens: list[str] = []
    if opts.constrain_to_start_coords is not None:
        tokens.append(f"constrain_to_start_coords={1 if opts.constrain_to_start_coords else 0}")
    if opts.ramp_constraints is not None:
        tokens.append(f"ramp_constraints={1 if opts.ramp_constraints else 0}")
    if opts.coord_constrain_sidechains is not None:
        tokens.append(f"coord_constrain_sidechains={1 if opts.coord_constrain_sidechains else 0}")
    if opts.relax_cycles is not None:
        tokens.append(f"relax_cycles={opts.relax_cycles}")
    if opts.bb_min_only is not None:
        tokens.append(f"bb_min_only={1 if opts.bb_min_only else 0}")
    return tuple(tokens)
