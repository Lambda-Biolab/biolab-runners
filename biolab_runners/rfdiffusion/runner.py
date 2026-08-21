"""RFdiffusion runner.

A subprocess wrapper around the in-package ``rfdiffusion`` console
script (``biolab_runners.rfdiffusion.cli``), which adapts the runner's
flag contract to the upstream ``RFdiffusion/run_inference.py`` Hydra
CLI. The runner owns:

* ``submit()`` / ``dry_run`` semantics matching the boltz2 runner;
* ``idempotency`` keyed by the **full canonical requested-config
  digest** — outputs live at ``<output_root>/<name>/<digest>/`` so
  seed / task_count / contigs / mode / hotspots / checkpoint /
  model-relevant ``extra`` variants cannot cross-hit (see "Output
  layout and cache key" below);
* ``force`` to bypass idempotency;
* structured result parsing into :class:`RecordData` objects;
* Shared provenance — every invocation attaches a
  :class:`biolab_runners.provenance.ProvenanceMetadata` record so
  downstream consumers can audit the inputs.

RFdiffusion itself is not pip-installable. The runner invokes the
installed ``rfdiffusion`` console script (or ``RFDIFFUSION_BIN``), which
requires ``RFDIFFUSION_HOME`` to point at the upstream clone root with
the model weights downloaded (see ``biolab_runners.rfdiffusion.cli``).

Output layout and cache key:

* The on-disk design directory is
  ``<output_root>/<safe-name>/<identity-token>/`` where
  ``<safe-name>`` is ``config.name`` (validated as a single safe path
  component) and ``<identity-token>`` is the **canonical cache
  identity** (see :func:`_cache_identity_token`): a sha256 over

  1. the full canonical requested-config digest
     (:func:`biolab_runners.provenance.compute_config_digest` —
     every field, including ``seed``, ``task_count``, ``contigs``,
     ``mode``, ``disulfide_pairs``, ``cyc_chains``, ``design_chains``,
     ``hotspots``, ``deterministic``, ``checkpoint``, and ``extra``),
  2. the **derived execution payload** — the runner-execution
     contract version (:data:`EXECUTION_CONTRACT_VERSION`) plus the
     exact CLI mapping (:func:`_config_to_cli`) — so a mapping-only
     change (same requested config, different forwarded flags)
     invalidates the cache,
  3. the **normalized image digest** when the caller supplied one,
  4. the **content digest of ``target_pdb``** when set.

  A config variant, a different image, changed bytes at the same
  target path, or a mapping change each get their own directory; the
  cache can never return another identity's outputs. ``result.name``
  still reports ``config.name``.
* **Cache identity binding is what it covers.** When ``image_digest``
  is absent the identity has no image binding — two local runs with
  the same config and source but different container images could
  cross-hit. This is a documented local-compatibility limitation; callers
  production supplies ``image_digest``, which makes the binding
  complete. A set-but-missing ``target_pdb`` is a hard error at
  run time (fail closed): the file is forwarded as
  ``inference.input_pdb``, so a dangling path would crash upstream,
  and the identity would lose its source-content binding. An
  **empty** ``target_pdb`` is unconditional generation — the
  identity then has no source binding by design (absence is intent),
  and the provenance manifest records ``source_backbone_digest=None``.
* **Identity mechanism preserved; tokens change once.** Binding the
  execution payload (item 2) means cache tokens computed by earlier
  runner versions are no longer reproduced — a deliberate, one-time
  invalidation. The identity *mechanism* (config + image + source
  content isolation, normalized forms) is unchanged; the token for a
  given (config, image, source, execution payload) tuple is stable
  from here on.
* **Migration / legacy behavior.** Runner versions before the
  digest-keyed layout wrote directly into
  ``<output_root>/<name>/*.pdb``. Those name-only outputs carry no
  proof of which config produced them, so the runner **never**
  treats them as a matching cache entry (the binding is not
  provable) and never mixes them into results; they are left in
  place untouched. New runs write under the identity subdirectory.
* **Default ``seed=0`` is reproducible, not diverse.** With the
  default ``deterministic=True`` and ``seed=0``, the same config
  always produces the same designs (per-design seeds
  ``0..task_count-1``). Callers that want distinct designs across
  runs or replicas must vary ``seed`` explicitly — each seed also
  gets its own identity-keyed output directory.

Notes on seed / temperature forwarding:

* Stock upstream RFdiffusion has **no** ``inference.seed`` key —
  ``config/inference/base.yaml`` defines ``inference.deterministic``
  and ``inference.design_startnum``, and ``scripts/run_inference.py``
  seeds each design with the design index:
  ``for i_des in range(design_startnum, design_startnum + num_designs):``
  ``if conf.inference.deterministic: make_deterministic(i_des)``.
  A wrapper that appends ``+inference.seed=...`` would have the key
  silently ignored (inert — nothing reads it); a strict override is
  rejected. Either way it never affects the RNG. When
  ``config.deterministic`` is true the runner therefore forwards
  ``inference.design_startnum=<seed>`` (the supported external base)
  plus ``inference.deterministic=True``. The per-design seeds are
  ``seed, seed+1, ..., seed + task_count - 1`` and output
  indices/names start at ``seed``. The provenance manifest records
  ``base_seed == requested_seed == config.seed`` and ``rng_intent
  == "per-design-index"``; the per-design seed range is encoded by
  ``base_seed`` + ``task_count`` (no per-seed list is fabricated).
  ``executed_config_digest`` covers the derived execution payload
  (contract version + exact CLI mapping), which includes
  ``design_startnum`` in deterministic mode — so a seed-only change
  flips the executed digest (the digest describes what was actually
  forwarded).
* When ``config.deterministic`` is false the runner forwards
  neither ``inference.design_startnum`` nor
  ``inference.deterministic`` — upstream uses system entropy, so a
  forwarded base seed would be inert. The manifest records
  ``base_seed=None`` and ``rng_intent == "non-deterministic"``, and
  the executed mapping omits the seed entirely (no
  ``inference.design_startnum``) — a seed-only change must not flip
  the executed digest.
* RFdiffusion does not expose a single ``temperature`` parameter
  (the diffusion process uses ``noise_scale_ca`` /
  ``noise_scale_frame``, which are tuned per-application and
  upstream-internal). The provenance manifest therefore records
  ``temperature=None`` for RFdiffusion runs.
* **Output parsing mirrors stock chain assignment.** Stock
  target-conditioned output PDBs contain the generated binder chain(s)
  **plus** the receptor chains copied from ``inference.input_pdb``.
  The generated design's output chain is derived exactly as stock
  assigns it (``RFdiffusionConfig.design_chains`` — the
  lexicographically first ASCII letter not used by the
  contig-referenced receptor chains: receptor A+B → ``C``, receptor A
  → ``B``, unconditional → ``A``), and ``RecordData.sequence`` is
  parsed from exactly that chain, never target+peptide;
  ``RecordData.path`` keeps the full complex PDB so downstream
  interface filtering still has receptor coordinates. An output PDB
  missing the derived chain — or yielding no parseable residues — is
  a ``failed`` record (fail closed, never a fake-empty success), on
  the execute AND cache-hit paths. ``design_chains`` is
  parse/provenance semantics, not a Hydra flag: it is bound into the
  cache identity via the requested-config digest but is never
  forwarded to the CLI (the executed digest is unaffected because the
  mapping is).
* **Target-conditioned binder design:** ``config.target_pdb`` is
  forwarded as the canonical stock Hydra key ``inference.input_pdb``
  and bound in the cache identity + provenance by its content digest.
  Stock upstream substitutes a bundled example PDB when
  ``input_pdb`` is unset, so binder contigs (chain references in
  ``contigs``) and hotspots without ``target_pdb`` are rejected at
  config construction, and a set-but-missing ``target_pdb`` file
  raises at ``run()`` / ``is_complete()`` time (fail closed).
* Topology honesty: ``inference.cyclic`` / ``inference.cyc_chains``
  express **only** head-to-tail cyclization of named chains in **HAL
  space** — the internal chain-index space of ``rfdiffusion/contigs.py``
  (generated chains labelled ``A``, ``B``, ... via ``chain_order``
  ahead of the receptor chain), matched against ``contig_map.hal``
  with internal uppercasing (``model_runners._init_cyclic_reses``).
  The runner emits them for ``mode="head_to_tail"`` and
  ``mode="head_to_tail_and_disulfide"``; ``cyc_chains`` names the
  generated binder chain (default ``"a"`` — the first generated HAL
  chain), independent of the output-PDB letter the binder gets (see
  the output-parsing note above). ``mode="disulfide"`` forwards
  **no** cyclic flags: stock RFdiffusion cannot encode residue-pair
  disulfides, and ``disulfide_pairs`` remains in config/provenance as
  downstream topology intent, applied and validated by
  ``biolab_runners.peptide_prep`` — not by RFdiffusion.

Notes on cache hits:

* On the idempotent path the runner returns existing files without
  invoking upstream. The provenance record sets ``executed=False``
  and ``cache_hit=True``, and ``executed_config_digest=None`` —
  the runner does not know which prior call produced the existing
  files, so it does not fabricate an executed digest.
  ``requested_config_digest`` describes what *this* call asked for;
  ``requested_seed`` and ``base_seed`` (and ``rng_intent``) are
  reported because the cache is **identity-bound**: the cache key IS
  the canonical identity over the full requested config, the derived
  execution payload (contract version + exact CLI mapping), the
  normalized image digest, and the source-backbone content digest,
  so the cached outputs provably correspond to this exact
  config+execution+image+source and the per-design seed range
  ``base_seed .. base_seed + task_count - 1`` describes them.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.contracts import ArtifactReference, ExecutionMode, ExecutionStatus
from biolab_runners.provenance import (
    EMPTY_PROVENANCE,
    RNG_INTENT_NON_DETERMINISTIC,
    RNG_INTENT_PER_DESIGN_INDEX,
    ProvenanceMetadata,
    compute_config_digest,
    compute_file_digest,
    validate_image_digest,
)
from biolab_runners.rfdiffusion.cli import EXECUTION_CONTRACT_VERSION
from biolab_runners.rfdiffusion.utils import (
    RecordData,
    RecordDataStatus,
    _invoke_with_metadata,
    parse_backbone_pdb,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from biolab_runners.rfdiffusion.config import RFdiffusionConfig

logger = logging.getLogger(__name__)

__all__ = ["RFdiffusionResult", "RFdiffusionRunner"]


# ---------------------------------------------------------------------------
# Canonical cache identity
# ---------------------------------------------------------------------------


def _validate_target_file(cfg: RFdiffusionConfig) -> None:
    """Fail closed when ``target_pdb`` is set but the file is missing.

    ``target_pdb`` is forwarded as ``inference.input_pdb``: a
    dangling path would crash upstream at best, and with the file
    absent the cache identity would lose its source-content binding
    (an unusable identity). Raised by ``run()`` and ``is_complete()``
    before any directory or subprocess work. An **empty**
    ``target_pdb`` is unconditional generation and is not an error.
    """
    if cfg.target_pdb and not Path(cfg.target_pdb).is_file():
        raise ValueError(
            f"RFdiffusionConfig.target_pdb={cfg.target_pdb!r} does not exist; "
            "target-conditioned design requires a readable target structure"
        )


def _cache_identity_token(config: RFdiffusionConfig, *, image_digest: str | None) -> str:
    """Canonical cache identity for ``config``.

    Binds the four things that determine upstream's output:

    1. the full canonical requested-config digest (every config field
       — ``seed``, ``task_count``, ``contigs``, ``mode``,
       ``disulfide_pairs``, ``cyc_chains``, ``design_chains``,
       ``hotspots``, ``deterministic``, ``checkpoint``, ``extra``);
    2. the **derived execution payload** — the runner-execution
       contract version plus the exact CLI mapping
       (:func:`_execution_payload`), so a mapping-only change
       invalidates the cache;
    3. the **normalized** image digest, when the caller supplied one
       (``None``/absent → the identity has no image binding — local
       compatibility; production supplies it);
    4. the content digest of ``target_pdb`` **when set** (the file
       must exist — ``run()``/``is_complete()`` fail closed
       otherwise; an empty ``target_pdb`` is unconditional
       generation with no source binding by design).

    ``run()``, ``is_complete``, and ``_design_dir`` all compute the
    directory from this single function, so the cache lookup and the
    write target are always the same identity.
    """
    payload: dict[str, object] = {
        "config": compute_config_digest(config),
        "execution": _execution_payload(config),
        "image_digest": validate_image_digest(image_digest),
        "source_backbone_digest": compute_file_digest(Path(config.target_pdb))
        if config.target_pdb
        else None,
    }
    return compute_config_digest(payload)


# ---------------------------------------------------------------------------
# Derived execution payload (contract version + exact CLI mapping)
# ---------------------------------------------------------------------------

#: Version of the runner→CLI execution contract. The constant is
#: defined in the translation-owning console-script module
#: (``biolab_runners.rfdiffusion.cli``) and imported here so there is
#: one authoritative bump location: bump it whenever the config→flag
#: mapping (:func:`_config_to_cli`) OR the flag→Hydra translation /
#: owned overrides in the CLI module change. The mapping is part of
#: the cache identity and the executed-config digest, so a mapping-only
#: change (identical requested config) invalidates cached outputs and
#: re-provenances the run.


def _execution_payload(config: RFdiffusionConfig) -> dict[str, object]:
    """The derived execution payload for ``config``.

    The exact CLI mapping the runner forwards (contract version +
    :func:`_config_to_cli` output). Bound into the cache identity and
    the executed-config digest; the *requested* digest stays
    config-based. Non-deterministic runs omit ``seed`` from the
    mapping (no ``inference.design_startnum`` is forwarded), so the
    payload naturally reflects that.
    """
    return {
        "contract_version": EXECUTION_CONTRACT_VERSION,
        "cli": _config_to_cli(config),
    }


def _executed_digest(config: RFdiffusionConfig) -> str:
    """Digest of the DERIVED execution payload for ``config``.

    Describes what upstream actually received: the runner-execution
    contract version plus the exact CLI mapping. A mapping-only change
    (e.g. a new forwarded flag, a renamed key) flips the executed
    digest; a non-deterministic seed-only change does NOT — the
    mapping omits the seed when ``deterministic=False``.
    """
    return compute_config_digest(_execution_payload(config))


@dataclass(frozen=True)
class RFdiffusionResult:
    """Outcome of one or more RFdiffusion design runs.

    The counters are **two independent axes**: ``succeeded`` /
    ``failed`` describe the parse quality of the records (usable vs
    broken outputs, always summing to ``len(records)``), while
    ``skipped`` describes the invocation — a cache hit means nothing
    was invoked, so *every* record (succeeded or failed) is also
    counted as skipped. On a cache hit ``exit_code`` is 0 only when
    every cached record parsed; a failed parse of cached output sets
    it to 1 (honest — a bad cache entry is not success). On the
    execute path ``exit_code`` is the upstream subprocess exit code.

    The ``provenance`` field carries the reproducibility record
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

    def is_complete(self, config: RFdiffusionConfig, *, image_digest: str | None = None) -> bool:
        """Return True if the identity-keyed design dir already has backbones.

        Only the directory bound to ``config`` + ``image_digest`` +
        the current ``target_pdb`` content is consulted; legacy
        name-only outputs (see the module docstring) are never
        treated as a cache entry. ``image_digest`` is normalized
        before the identity is computed, so bare-hex and OCI forms
        bind identically. A set-but-missing ``target_pdb`` fails
        closed (see :func:`_validate_target_file`) — the probe would
        otherwise consult an identity with no source-content binding.
        """
        _validate_target_file(config)
        directory = self._design_dir(config, image_digest=image_digest)
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
        """Generate backbones under ``<output_root>/<name>/<digest>/``.

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
            duration, and shared provenance.

        The cache is keyed by the canonical identity over the full
        requested-config digest + the derived execution payload
        (contract version + exact CLI mapping) + the normalized
        ``image_digest`` (when supplied) + the content digest of
        ``config.target_pdb`` (when set):
        ``<output_root>/<name>/<identity>/``.
        Two runs that differ in any config field, in the execution
        mapping, in the image digest, or in the source-backbone bytes
        never share a directory, so a cache hit can only ever return
        this exact identity's outputs.
        A set-but-missing ``target_pdb`` file fails closed (see
        :func:`_validate_target_file`) before any directory or
        subprocess work.
        """
        # Canonicalise the caller-supplied image digest BEFORE any
        # subprocess work so downstream manifest comparison sees a
        # single form regardless of how the caller wrote it.
        image_digest = validate_image_digest(image_digest)

        cfg = config or self._config_override
        if cfg is None:
            raise ValueError("RFdiffusionConfig is required: pass it to run() or the runner")
        _validate_target_file(cfg)

        design_dir = self._design_dir(cfg, image_digest=image_digest)
        design_dir.mkdir(parents=True, exist_ok=True)

        if not force and self.is_complete(cfg, image_digest=image_digest):
            records = _parse_output_dir(design_dir, chains=cfg.design_chains)
            succeeded = sum(1 for r in records if r.status == RecordDataStatus.SUCCEEDED)
            failed = len(records) - succeeded
            # Honest cache hit: nothing was invoked (every record is also
            # "skipped"), but a failed parse of the cached output is still
            # a failure — exit_code stays 0 only when every cached record
            # is usable.
            exit_code = 1 if failed else 0
            failure_reason = f"{failed} cached record(s) failed to parse" if failed else ""
            status = _cache_status(failed)
            artifacts = _artifacts_for_records(records)
            return RFdiffusionResult(
                name=cfg.name,
                output_dir=str(design_dir),
                records=records,
                succeeded=succeeded,
                failed=failed,
                skipped=len(records),
                exit_code=exit_code,
                duration_seconds=0.0,
                provenance=self._build_provenance(
                    cfg,
                    exit_code=exit_code,
                    failure_reason=failure_reason,
                    stderr_tail="",
                    image_digest=image_digest,
                    executed=False,
                    cache_hit=True,
                    status=status,
                    artifacts=artifacts,
                ),
                status=status,
                artifacts=artifacts,
                execution_mode=ExecutionMode.SUBPROCESS,
            )

        config_dict = _config_to_cli(cfg)
        if dry_run:
            status = ExecutionStatus.DRY_RUN
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
                    status=status,
                ),
                status=status,
                execution_mode=ExecutionMode.SUBPROCESS,
            )

        started = time.monotonic()
        result = _invoke_with_metadata(
            config_dict=config_dict,
            output_dir=design_dir,
            binary_prefix=self._binary_prefix,
            timeout_seconds=self._timeout_seconds,
        )
        records = _parse_output_dir(design_dir, chains=cfg.design_chains)
        succeeded = sum(1 for r in records if r.status == RecordDataStatus.SUCCEEDED)
        failed = len(records) - succeeded
        status = _status_from_invocation(result.exit_code, records)
        artifacts = _artifacts_for_records(records)
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
                status=status,
                artifacts=artifacts,
            ),
            status=status,
            artifacts=artifacts,
            execution_mode=ExecutionMode.SUBPROCESS,
        )

    def _design_dir(self, config: RFdiffusionConfig, *, image_digest: str | None = None) -> Path:
        """Return the identity-keyed design directory for ``config``.

        ``<output_root>/<name>/<identity-token>/`` where the token is
        :func:`_cache_identity_token` — the canonical cache identity
        over the requested config digest, the normalized image digest
        (when supplied), and the ``target_pdb`` content digest (when
        set). ``run()`` and ``is_complete`` use this same
        function, so the cache lookup and the write target always
        agree.
        """
        return (
            self._output_root
            / config.name
            / _cache_identity_token(config, image_digest=image_digest)
        )

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
        status: ExecutionStatus = ExecutionStatus.SUCCEEDED,
        artifacts: tuple[ArtifactReference, ...] = (),
    ) -> ProvenanceMetadata:
        """Assemble the provenance record for ``cfg``.

        The ``target_pdb`` field on the config is the source backbone
        for RFdiffusion; its content digest is recorded as
        ``source_backbone_digest``. The file is guaranteed to exist
        here — ``run()`` fails closed on a set-but-missing target
        (see :func:`_validate_target_file`). When ``target_pdb`` is
        unset (unconditional generation) the manifest records ``None``
        rather than fabricating a digest — absence is intent.
        ``executed_config_digest`` is :func:`_executed_digest` — the
        derived execution payload (contract version + exact CLI
        mapping), describing what upstream actually received; the
        ``requested_config_digest`` stays config-based.
        """
        target = Path(cfg.target_pdb) if cfg.target_pdb else None
        return ProvenanceMetadata(
            model_identifier=cfg.checkpoint,
            temperature=None,
            image_digest=image_digest,
            source_backbone_digest=compute_file_digest(target) if target is not None else None,
            exit_code=exit_code,
            failure_reason=failure_reason,
            stderr_tail=stderr_tail,
            base_seed=cfg.seed if cfg.deterministic else None,
            requested_seed=cfg.seed,
            task_count=cfg.task_count,
            rng_intent=RNG_INTENT_PER_DESIGN_INDEX
            if cfg.deterministic
            else RNG_INTENT_NON_DETERMINISTIC,
            canonical_output=(),
            requested_config_digest=compute_config_digest(cfg),
            executed_config_digest=_executed_digest(cfg) if executed else None,
            executed=executed,
            cache_hit=cache_hit,
            runner_name="rfdiffusion",
            execution_mode=ExecutionMode.SUBPROCESS,
            status=status,
            artifacts=artifacts,
        )


def _config_to_cli(config: RFdiffusionConfig) -> dict[str, str]:
    """Translate :class:`RFdiffusionConfig` into the upstream CLI kwargs.

    Returns a flat ``key -> str`` mapping consumed by
    :func:`biolab_runners.rfdiffusion.utils._invoke_with_metadata`,
    which emits ``--<dotted.key> <value>`` flags for the in-package
    ``rfdiffusion`` console script (it re-translates list-typed keys
    to Hydra list syntax). The mapping is also the derived execution
    payload bound into the cache identity and the executed-config
    digest (:func:`_execution_payload` / :func:`_executed_digest`).

    Notes on what's deliberately **not** forwarded:

    * ``design_chains`` — never. It is parse/provenance semantics
      (which output-PDB chains carry the generated design), not a
      Hydra flag: upstream always writes the full target+binder
      complex, and the field only tells the runner how to interpret
      the output. It is bound into the cache identity via the
      requested-config digest, but the executed digest / CLI mapping
      are unaffected.
    * ``inference.seed`` — never. Stock upstream RFdiffusion has no
      such Hydra key (``config/inference/base.yaml``); a wrapper that
      appends it (``+inference.seed=...``) would have it silently
      ignored — it is inert, nothing reads it — and a strict override
      is rejected. Either way it never affects the RNG. The supported
      external base is ``inference.design_startnum`` (see below).
    * ``inference.design_startnum`` / ``inference.deterministic`` —
      forwarded only when ``config.deterministic`` is true. With
      ``deterministic=False`` the runner forwards neither: upstream
      uses system entropy, so a forwarded base seed would be inert
      and the manifest records ``rng_intent="non-deterministic"``
      with ``base_seed=None``.
    * ``diffusion.noise_scale_ca`` — RFdiffusion's diffusion
      process uses upstream-internal noise scales, not a single
      sampling temperature. Mapping ``config.temperature`` to
      ``noise_scale_ca`` would silently change upstream behaviour
      and is intentionally avoided.
    * ``inference.cyclic`` / ``inference.cyc_chains`` — forwarded
      only for the head-to-tail modes. Stock RFdiffusion uses them to
      cyclize named chains head-to-tail in **HAL space** (the internal
      chain-index space of ``contigs.py``; ``cyc_chains="a"`` = first
      generated chain = the binder, independent of the output-PDB
      letter); it has no notion of residue-pair disulfides, so
      ``mode="disulfide"`` forwards neither flag (the pairs stay in
      config/provenance as downstream closure intent).

    ``config.extra`` may not override the canonical keys the runner
    emits (``inference.num_designs`` / ``inference.design_startnum`` /
    ``inference.deterministic`` / ``inference.input_pdb`` /
    ``inference.cyclic`` / ``inference.cyc_chains`` /
    ``contigmap.contigs`` / ``ppi.hotspot_res``), nor forward
    unsupported keys such as ``inference.seed`` —
    :class:`RFdiffusionConfig` rejects all of these at construction
    time (fail closed), so the merge below cannot silently clobber a
    forwarded field.
    """
    payload: dict[str, str] = {
        "inference.num_designs": str(config.task_count),
        "contigmap.contigs": config.contigs,
    }
    if config.target_pdb:
        # Canonical stock key: target-conditioned design parses the
        # chains referenced by ``contigmap.contigs`` from this file.
        payload["inference.input_pdb"] = config.target_pdb
    if config.deterministic:
        # Upstream seeds each design with its index (see
        # RFdiffusion/scripts/run_inference.py):
        #   for i_des in range(design_startnum, design_startnum + num_designs):
        #       if conf.inference.deterministic: make_deterministic(i_des)
        # so ``seed`` maps to the supported external base
        # ``inference.design_startnum``: per-design seeds are
        # seed .. seed + task_count - 1 and output indices/names
        # start at ``seed``.
        payload["inference.design_startnum"] = str(config.seed)
        payload["inference.deterministic"] = "True"
    if config.mode in {"head_to_tail", "head_to_tail_and_disulfide"}:
        payload["inference.cyclic"] = "True"
        # Stock ``inference.cyc_chains`` names chains in HAL space —
        # the internal chain-index space of rfdiffusion/contigs.py,
        # where generated chains are labelled A, B, ... via
        # chain_order ahead of the receptor chain — matched against
        # contig_map.hal with internal uppercasing
        # (model_runners._init_cyclic_reses). The default "a" is the
        # first generated chain — the binder — regardless of the
        # output-PDB letter the binder gets (design_chains).
        payload["inference.cyc_chains"] = config.cyc_chains
    # mode="disulfide": NOT cyclic. Stock inference.cyclic/cyc_chains
    # express only head-to-tail chain cyclization and cannot encode
    # residue-pair disulfides; ``disulfide_pairs`` remains in
    # config/provenance as downstream topology intent (closure is
    # applied/validated by peptide_prep, not by RFdiffusion).
    if config.hotspots:
        payload["ppi.hotspot_res"] = ",".join(config.hotspots)
    for key, value in config.extra.items():
        payload[key] = str(value)
    return payload


def _artifacts_for_records(records: tuple[RecordData, ...]) -> tuple[ArtifactReference, ...]:
    """Describe parsed PDB outputs without inventing missing artifacts."""
    return tuple(
        ArtifactReference.from_path(record.path, kind="structure")
        for record in records
        if record.path
    )


def _status_from_invocation(exit_code: int, records: tuple[RecordData, ...]) -> ExecutionStatus:
    """Map the legacy exit/parse surface to the shared status vocabulary."""
    if exit_code == 124:
        return ExecutionStatus.TIMEOUT
    if exit_code < 0:
        return ExecutionStatus.INTERRUPTED
    if exit_code != 0:
        return ExecutionStatus.FAILED
    if not records:
        return ExecutionStatus.INCOMPLETE
    if any(record.status != RecordDataStatus.SUCCEEDED for record in records):
        return ExecutionStatus.MALFORMED
    return ExecutionStatus.SUCCEEDED


def _cache_status(failed: int) -> ExecutionStatus:
    """Return the cache status without conflating parse failure and reuse."""
    if failed:
        return ExecutionStatus.MALFORMED
    return ExecutionStatus.CACHED


#: Matches upstream's output naming ``<prefix>_<i_des>.pdb`` where
#: ``i_des`` is the final numeric design index (``design_42.pdb`` -> 42).
_INDEX_SUFFIX_RE = re.compile(r"^(?P<stem>.+)_(?P<index>\d+)\.pdb$")


def _design_index_from_name(path: Path) -> int | None:
    """Return the final numeric design index from a PDB filename.

    ``design_42.pdb`` -> ``42``; ``design_0.pdb`` -> ``0``. Names whose
    final segment before ``.pdb`` is not ``_<digits>`` (e.g.
    ``weird.pdb``, ``traj_1.pdb``) return ``None`` — the caller then
    assigns a collision-free fallback index (see
    :func:`_parse_output_dir`).
    """
    match = _INDEX_SUFFIX_RE.match(path.name)
    if match is None:
        return None
    return int(match.group("index"))


def _smallest_unused_index(used: set[int]) -> int:
    """Smallest non-negative integer not in ``used``."""
    index = 0
    while index in used:
        index += 1
    return index


def _parse_output_dir(design_dir: Path, chains: Sequence[str] = ("A",)) -> tuple[RecordData, ...]:
    """Walk ``design_dir`` and parse each PDB file into a :class:`RecordData`.

    ``RecordData.index`` is the design's numeric index parsed from the
    filename's final ``_<digits>`` segment when present (``design_42.pdb``
    -> ``42``), matching upstream's ``<prefix>_<i_des>.pdb`` naming. For
    nonstandard names (no ``_<digits>`` suffix) the index falls back to
    the smallest non-negative integer not already used by a parsed or
    assigned index, so records never collide regardless of filename
    shape. With ``seed`` mapped to ``inference.design_startnum``, the
    parsed indices equal the per-design seeds
    (``seed .. seed + task_count - 1``).

    Records are returned in **numeric design-index order** (``design_2.pdb``
    before ``design_10.pdb`` — filename lexicographic order would put 10
    first once ``task_count > 9``), with the filename as a stable
    tiebreak for names without an index suffix; fallback indices are
    assigned in that deterministic order.

    ``chains`` names the output chains that carry the generated design
    (``config.design_chains``) — ``RecordData.sequence`` is parsed from
    exactly those chains, so a target-conditioned binder record never
    mixes in the receptor chains. Fail closed: a PDB that lacks any
    configured chain, or that yields no parseable residues, is recorded
    as :attr:`RecordDataStatus.FAILED` with the reason in ``error`` —
    never a fake success with an empty/truncated sequence.
    """
    # Deterministic processing order: numeric design index first, then
    # filename (stable) for names without an index suffix.
    indexed_paths: list[tuple[int | None, Path]] = [
        (_design_index_from_name(path), path) for path in design_dir.glob("*.pdb")
    ]
    indexed_paths.sort(
        key=lambda item: (item[0] if item[0] is not None else float("inf"), item[1].name)
    )
    records: list[RecordData] = []
    used_indices: set[int] = set()
    for design_index, path in indexed_paths:
        assigned_index = (
            design_index if design_index is not None else _smallest_unused_index(used_indices)
        )
        used_indices.add(assigned_index)
        try:
            sequence = parse_backbone_pdb(path, chains=chains)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            logger.warning("failed to parse %s: %s", path, exc)
            records.append(
                RecordData(
                    index=assigned_index,
                    path=str(path),
                    sequence="",
                    status=RecordDataStatus.FAILED,
                    error=str(exc),
                )
            )
            continue
        records.append(
            RecordData(
                index=assigned_index,
                path=str(path),
                sequence=sequence,
                status=RecordDataStatus.SUCCEEDED,
            )
        )
    return tuple(records)
