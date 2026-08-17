"""RFdiffusion runner.

A subprocess wrapper around the upstream ``RFdiffusion/run_inference.py``
CLI. The runner owns:

* ``submit()`` / ``dry_run`` semantics matching the boltz2 runner;
* ``idempotency`` keyed by the **full canonical requested-config
  digest** — outputs live at ``<output_root>/<name>/<digest>/`` so
  seed / task_count / contigs / mode / hotspots / checkpoint /
  model-relevant ``extra`` variants cannot cross-hit (see "Output
  layout and cache key" below);
* ``force`` to bypass idempotency;
* structured result parsing into :class:`RecordData` objects;
* S2 provenance — every invocation attaches a
  :class:`biolab_runners.provenance.ProvenanceMetadata` record so
  downstream consumers can audit the inputs.

RFdiffusion itself is not pip-installable. The runner expects the
caller to either (a) install the upstream conda env (rare), or (b)
point ``RFDIFFUSION_BIN`` at a Docker-wrapped command (the GCP Batch
path).

Output layout and cache key:

* The on-disk design directory is
  ``<output_root>/<safe-name>/<identity-token>/`` where
  ``<safe-name>`` is ``config.name`` (validated as a single safe path
  component) and ``<identity-token>`` is the **canonical cache
  identity** (see :func:`_cache_identity_token`): a sha256 over

  1. the full canonical requested-config digest
     (:func:`biolab_runners.provenance.compute_config_digest` —
     every field, including ``seed``, ``task_count``, ``contigs``,
     ``mode``, ``disulfide_pairs``, ``hotspots``, ``deterministic``,
     ``checkpoint``, and ``extra``),
  2. the **normalized image digest** when the caller supplied one,
  3. the **content digest of ``target_pdb``** when the file exists.

  A config variant, a different image, or changed bytes at the same
  target path each get their own directory; the cache can never
  return another identity's outputs. ``result.name`` still reports
  ``config.name``.
* **Cache identity binding is what it covers.** When ``image_digest``
  is absent the identity has no image binding — two local runs with
  the same config and source but different container images could
  cross-hit. This is a documented local-compat limitation; Activin
  production supplies ``image_digest``, which makes the binding
  complete. When ``target_pdb`` is set but the file is missing, the
  identity has no source-content binding either (the config digest
  still binds the path string) and the run proceeds with the
  existing warning + ``source_backbone_digest=None``; a later run
  with the file present computes a different identity and
  re-executes, so a missing-then-present source can never cross-hit.
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

Notes on seed / temperature forwarding (S2 honesty):

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
  ``executed_config_digest`` includes ``seed``, so a seed-only
  change flips the executed digest (the digest describes what was
  actually forwarded).
* When ``config.deterministic`` is false the runner forwards
  neither ``inference.design_startnum`` nor
  ``inference.deterministic`` — upstream uses system entropy, so a
  forwarded base seed would be inert. The manifest records
  ``base_seed=None`` and ``rng_intent == "non-deterministic"``, and
  the executed config digest excludes ``seed`` (it was not
  forwarded; a seed-only change must not flip the executed digest).
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
  files, so it does not fabricate an executed digest.
  ``requested_config_digest`` describes what *this* call asked for;
  ``requested_seed`` and ``base_seed`` (and ``rng_intent``) are
  reported because the cache is **identity-bound**: the cache key IS
  the canonical identity over the full requested config, the
  normalized image digest, and the source-backbone content digest,
  so the cached outputs provably correspond to this exact
  config+image+source and the per-design seed range ``base_seed ..
  base_seed + task_count - 1`` describes them.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.provenance import (
    EMPTY_PROVENANCE,
    RNG_INTENT_NON_DETERMINISTIC,
    RNG_INTENT_PER_DESIGN_INDEX,
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


# ---------------------------------------------------------------------------
# Canonical cache identity
# ---------------------------------------------------------------------------


def _cache_identity_token(config: RFdiffusionConfig, *, image_digest: str | None) -> str:
    """Canonical cache identity for ``config``.

    Binds the three things that determine upstream's output:

    1. the full canonical requested-config digest (every config field
       — ``seed``, ``task_count``, ``contigs``, ``mode``,
       ``disulfide_pairs``, ``hotspots``, ``deterministic``,
       ``checkpoint``, ``extra``);
    2. the **normalized** image digest, when the caller supplied one
       (``None``/absent → the identity has no image binding — local
       compatibility; Activin production supplies it);
    3. the content digest of ``target_pdb`` **when the file exists**
       (missing → no source-content binding; the config digest still
       binds the path string, and a later present file yields a
       different identity, so missing-then-present can never
       cross-hit).

    ``run()``, ``is_complete``, and ``_design_dir`` all compute the
    directory from this single function, so the cache lookup and the
    write target are always the same identity.
    """
    payload: dict[str, str | None] = {
        "config": compute_config_digest(config),
        "image_digest": validate_image_digest(image_digest),
        "source_backbone_digest": compute_file_digest(Path(config.target_pdb))
        if config.target_pdb
        else None,
    }
    return compute_config_digest(payload)


# ---------------------------------------------------------------------------
# Executed-digest exclusion policy
# ---------------------------------------------------------------------------


def _executed_digest_excluded_fields(cfg: RFdiffusionConfig) -> tuple[str, ...]:
    """Fields excluded from the *executed* config digest for ``cfg``.

    ``seed`` maps to ``inference.design_startnum``, which the runner
    forwards **only** when ``deterministic=True``. So:

    * deterministic — nothing is excluded: the seed changes what
      upstream ran (per-design seeds start at it), so a seed-only
      change flips the executed digest.
    * non-deterministic — ``seed`` is excluded: the base seed is not
      forwarded (upstream uses system entropy), so a seed-only change
      must NOT flip the executed digest.
    """
    return () if cfg.deterministic else ("seed",)


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

    def is_complete(self, config: RFdiffusionConfig, *, image_digest: str | None = None) -> bool:
        """Return True if the identity-keyed design dir already has backbones.

        Only the directory bound to ``config`` + ``image_digest`` +
        the current ``target_pdb`` content is consulted; legacy
        name-only outputs (see the module docstring) are never
        treated as a cache entry. ``image_digest`` is normalized
        before the identity is computed, so bare-hex and OCI forms
        bind identically.
        """
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
            duration, and S2 provenance.

        The cache is keyed by the canonical identity over the full
        requested-config digest + the normalized ``image_digest``
        (when supplied) + the content digest of ``config.target_pdb``
        (when the file exists): ``<output_root>/<name>/<identity>/``.
        Two runs that differ in any config field, in the image digest,
        or in the source-backbone bytes never share a directory, so a
        cache hit can only ever return this exact identity's outputs.
        """
        # Canonicalise the caller-supplied image digest BEFORE any
        # subprocess work so downstream manifest comparison sees a
        # single form regardless of how the caller wrote it.
        image_digest = validate_image_digest(image_digest)

        cfg = config or self._config_override
        if cfg is None:
            raise ValueError("RFdiffusionConfig is required: pass it to run() or the runner")

        design_dir = self._design_dir(cfg, image_digest=image_digest)
        design_dir.mkdir(parents=True, exist_ok=True)

        if not force and self.is_complete(cfg, image_digest=image_digest):
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

    def _design_dir(self, config: RFdiffusionConfig, *, image_digest: str | None = None) -> Path:
        """Return the identity-keyed design directory for ``config``.

        ``<output_root>/<name>/<identity-token>/`` where the token is
        :func:`_cache_identity_token` — the canonical cache identity
        over the requested config digest, the normalized image digest
        (when supplied), and the ``target_pdb`` content digest (when
        the file exists). ``run()`` and ``is_complete`` use this same
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
            base_seed=cfg.seed if cfg.deterministic else None,
            requested_seed=cfg.seed,
            task_count=cfg.task_count,
            rng_intent=RNG_INTENT_PER_DESIGN_INDEX
            if cfg.deterministic
            else RNG_INTENT_NON_DETERMINISTIC,
            canonical_output=(),
            requested_config_digest=compute_config_digest(cfg),
            executed_config_digest=compute_executed_config_digest(
                cfg, exclude_fields=_executed_digest_excluded_fields(cfg)
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

    ``config.extra`` may not override the canonical keys the runner
    emits (``inference.num_designs`` / ``inference.design_startnum`` /
    ``inference.deterministic`` / ``inference.cyclic`` /
    ``inference.cyc_chains`` / ``contigmap.contigs`` /
    ``ppi.hotspot_res``), nor forward unsupported keys such as
    ``inference.seed`` — :class:`RFdiffusionConfig` rejects all of
    these at construction time (fail closed), so the merge below
    cannot silently clobber a forwarded field.
    """
    payload: dict[str, str] = {
        "inference.num_designs": str(config.task_count),
        "contigmap.contigs": config.contigs,
    }
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
    if config.mode == "head_to_tail":
        payload["inference.cyclic"] = "True"
        payload["inference.cyc_chains"] = "a"
    elif config.mode == "disulfide":
        payload["inference.cyclic"] = "True"
        # Disulfide pairs are encoded as a comma-separated chain
        # specifier; the upstream parser turns ``X,Y`` into a Cys
        # pair between positions X and Y.
        payload["inference.cyc_chains"] = ",".join(f"{a},{b}" for a, b in config.disulfide_pairs)
    if config.hotspots:
        payload["ppi.hotspot_res"] = ",".join(config.hotspots)
    for key, value in config.extra.items():
        payload[key] = str(value)
    return payload


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


def _parse_output_dir(design_dir: Path) -> tuple[RecordData, ...]:
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
    """
    records: list[RecordData] = []
    used_indices: set[int] = set()
    for path in sorted(design_dir.glob("*.pdb")):
        design_index = _design_index_from_name(path)
        assigned_index = (
            design_index if design_index is not None else _smallest_unused_index(used_indices)
        )
        used_indices.add(assigned_index)
        try:
            sequence = parse_backbone_pdb(path)
        except (OSError, UnicodeDecodeError) as exc:
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
