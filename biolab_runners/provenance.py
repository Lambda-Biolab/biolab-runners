"""Provenance / reproducibility primitives for the biolab-runners slice.

Provides the serialization-friendly :class:`ProvenanceMetadata` record
that runners attach to their results so downstream consumers can audit
what produced a manifest.

Design constraints:

* **Manifest equivalence on exact rerun** — same inputs hash to the
  same :class:`ProvenanceMetadata` value and therefore the same JSON
  serialisation. The ``requested_config_digest`` always covers the
  full user-supplied config; the ``executed_config_digest`` covers
  the fields the runner actually forwarded to upstream. For
  ProteinMPNN the two are equal. For RFdiffusion they are equal in
  deterministic mode (``seed`` is forwarded as
  ``inference.design_startnum``) and differ in non-deterministic
  mode (the base seed is deliberately not forwarded, so a
  seed-only change must not flip the executed digest).
* **Different seeds → distinct traceable manifests.** ``base_seed``
  is the seed the runner actually forwarded (``None`` when the run
  was non-deterministic — RFdiffusion with ``deterministic=False``,
  where upstream uses system entropy); ``requested_seed`` is what
  the caller asked for. For seeded runs the two are equal.
* **Honest RNG metadata.** The runners in this slice invoke the
  upstream CLI exactly once with ``num_seq_per_target`` /
  ``num_designs`` outputs. They do **not** invoke once per task, so
  they **must not** claim per-task seeds as separate manifest
  fields. The :attr:`ProvenanceMetadata.rng_intent` field names the
  actual RNG mode: ``"single-stream"`` (ProteinMPNN — one seed
  drives all outputs), ``"per-design-index"`` (RFdiffusion with
  ``deterministic=True`` — upstream seeds each design with its
  index, so the per-design seeds are ``base_seed .. base_seed +
  task_count - 1``, a range fully encoded by the two existing
  fields, no list fabricated) or ``"non-deterministic"``
  (RFdiffusion with ``deterministic=False`` — upstream uses system
  entropy).
* **Honest cache-hit story.** When the runner returns from an
  on-disk artifact (the idempotent path), ``executed=False`` and
  ``cache_hit=True``. The ``executed_config_digest`` is ``None``
  because the runner does not know which prior call produced the
  existing files. ``requested_config_digest`` (and the
  ``requested_seed`` / ``base_seed`` / ``rng_intent`` fields)
  describe what THIS call asked for; where the runner keys its
  cache by a canonical identity over the requested config, the
  normalized image digest, and the source-backbone content digest
  (RFdiffusion), the cached outputs are provably bound to exactly
  this config+image+source, so reporting ``base_seed`` /
  ``rng_intent`` on a cache hit is honest — the per-design seed
  range describes the cached outputs.
* **Honest dry-run story.** ``executed=False`` and ``cache_hit=False``.
  The manifest records what *would* have been executed — including
  the intended forwarded ``base_seed``.
* **No bitwise GPU determinism claimed.** Provenance records the
  *intent* (which checkpoint, which inputs, which exit code) — it
  does not promise that two GPU runs of the same provenance will
  produce byte-identical structures. That is upstream's domain.

Digest conventions:

* File digests are ``sha256`` over the file's bytes, returned as a
  lowercase hex string with no ``sha256:`` prefix (callers can prepend
  the OCI prefix if their storage layer requires it).
* Config digests are ``sha256`` over the canonical JSON
  serialisation of the dataclass — sorted keys, no whitespace, no
  fallback ``default=str`` coercion (see :func:`_canonical_json`).
  Non-JSON-native values raise ``TypeError`` rather than silently
  hashing a stringified ``object``.

Image digest conventions:

* The image digest is the **caller's** responsibility: they own the
  container image and are the only party who can supply the digest.
  The runner does NOT resolve it. Both the bare 64-char hex form
  and the OCI-prefixed ``sha256:<hex>`` form are accepted; the
  helper always returns the canonical OCI form so downstream
  consumers can compare digests without worrying about which form
  the caller supplied.

Helpers in this module are deliberately framework-free: no logger,
no I/O policy, no subprocess. They take paths / dataclasses and
return values. The runners in ``biolab_runners.rfdiffusion`` and
``biolab_runners.proteinmpnn`` are the only callers in this slice;
adding a third is a one-import change.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping  # used at runtime in isinstance() guard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn

from biolab_runners.contracts import ExecutionMode, ExecutionStatus

if TYPE_CHECKING:
    from pathlib import Path

    from _typeshed import DataclassInstance

    from biolab_runners.contracts import ArtifactReference

__all__ = [
    "EMPTY_PROVENANCE",
    "RNG_INTENT_NON_DETERMINISTIC",
    "RNG_INTENT_PER_DESIGN_INDEX",
    "RNG_INTENT_SEED_NOT_FORWARDED",
    "RNG_INTENT_SINGLE_STREAM",
    "InvokeResult",
    "ProvenanceMetadata",
    "build_execution_provenance",
    "compute_config_digest",
    "compute_executed_config_digest",
    "compute_file_digest",
    "short_reason_from_stderr",
    "stderr_tail",
    "validate_image_digest",
]


# ---------------------------------------------------------------------------
# RNG intent — the canonical "how was the RNG used" labels.
# ---------------------------------------------------------------------------

#: A single seed value drives all outputs in one upstream invocation.
#: ProteinMPNN uses this label.
RNG_INTENT_SINGLE_STREAM = "single-stream"

#: Upstream seeds each output with its own index derived from a base
#: seed: per-design seeds are ``base_seed .. base_seed + task_count - 1``.
#: RFdiffusion uses this label when ``deterministic=True``: the runner
#: forwards ``inference.design_startnum=base_seed`` and upstream seeds
#: design ``i`` (output index) with ``design_startnum + i``.
RNG_INTENT_PER_DESIGN_INDEX = "per-design-index"

#: Retained for backward compatibility with consumers that imported it
#: while it was the RFdiffusion deterministic label. No runner in this
#: slice uses it anymore — RFdiffusion forwards ``inference.design_startnum``
#: (label :data:`RNG_INTENT_PER_DESIGN_INDEX`) and ProteinMPNN forwards a
#: single ``--seed`` (label :data:`RNG_INTENT_SINGLE_STREAM`).
RNG_INTENT_SEED_NOT_FORWARDED = "seed-not-forwarded"

#: The runner did not pin the RNG; upstream used system entropy.
#: RFdiffusion uses this label when ``deterministic=False``.
RNG_INTENT_NON_DETERMINISTIC = "non-deterministic"


# ---------------------------------------------------------------------------
# Image digest validation
# ---------------------------------------------------------------------------

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_OCI_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_image_digest(digest: str | None) -> str | None:
    """Validate an image digest and normalise it to OCI ``sha256:<hex>`` form.

    Accepts the two canonical forms:

    * Bare lowercase hex (64 chars) — the raw ``sha256`` digest.
    * OCI-prefixed form (``sha256:<64 hex chars>``) — the form
      Docker / podman print in ``docker images --digests``.

    Both forms are normalised to the OCI-prefixed form on return,
    so downstream consumers compare a single canonical string
    regardless of which form the caller supplied.

    ``None`` passes through (the runner records "no digest
    supplied" rather than fabricating one).

    Why this lives here: a malformed digest would silently flow
    through to the manifest and confuse downstream audits; the
    contract is small enough to enforce centrally.

    Args:
        digest: Caller-supplied image digest, or ``None`` for
            "unknown / not supplied".

    Returns:
        The canonical OCI-prefixed digest, or ``None`` when
        ``None`` was supplied.

    Raises:
        ValueError: When ``digest`` is non-``None`` but does not
            match either canonical form.
    """
    if digest is None:
        return None
    if _SHA256_OCI_RE.match(digest):
        return digest
    if _SHA256_HEX_RE.match(digest):
        return f"sha256:{digest}"
    raise ValueError(
        f"image_digest must be a 64-char lowercase sha256 hex, "
        f"with or without the 'sha256:' prefix; got {digest!r}"
    )


# ---------------------------------------------------------------------------
# Digest helpers
# ---------------------------------------------------------------------------


def compute_file_digest(path: Path) -> str | None:
    """Return the lowercase-hex ``sha256`` of ``path``, or ``None`` if missing.

    Reads the file in 64 KiB chunks so the helper is safe on multi-MB
    PDB inputs. Missing files yield ``None`` (not an exception) so the
    caller can decide whether absence is a hard failure or a soft
    signal in the manifest.
    """
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raise_unrepresentable(value: object) -> NoReturn:
    """Default callback for ``json.dumps``: refuse to stringify opaque objects.

    JSON natively handles ``dict``, ``list``, ``tuple``, ``str``,
    ``int``, ``float``, ``bool``, and ``None``. Anything else
    (``set``, ``bytes``, custom objects, ``Mapping`` proxies, ...)
    is rejected with ``TypeError`` so the digest cannot silently
    encode "stable-looking" bytes that depend on hash-randomised
    ordering, mutable bytes, or live object identities.
    """
    raise TypeError(
        f"non-JSON-native value of type {type(value).__name__!r}; "
        f"pass a JSON-native type (dict/list/tuple/str/int/float/bool/None)"
    )


def _canonical_json(obj: object) -> str:
    """Serialise ``obj`` deterministically for hashing.

    Sorted keys, no whitespace, no fallback ``default=str`` coercion
    (see :func:`_raise_unrepresentable`). Two ``dataclasses.asdict``
    passes of equal-valued configs produce byte-identical strings.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_raise_unrepresentable)


def _cast_to_dataclass(config: object) -> DataclassInstance:
    """Cast helper: ``is_dataclass(config)`` narrows but pyright needs the explicit cast."""
    return config  # type: ignore[return-value]


def _payload_from_config(config: object) -> dict[str, object]:
    """Normalise any of {dataclass, Mapping, plain object} into a plain dict.

    ``dataclasses.asdict`` recursively converts nested dataclasses to
    dicts, which is exactly what we want for canonical JSON. For
    plain ``Mapping`` inputs, ``dict(...)`` gives us a fresh dict
    so the caller can't mutate the canonical payload later.
    """
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(_cast_to_dataclass(config))
    if isinstance(config, Mapping):
        return dict(config)
    return vars(config)


def compute_config_digest(config: object) -> str:
    """Return the lowercase-hex ``sha256`` of ``config``'s canonical form.

    This is the digest of the *requested* config — every field the
    caller supplied. Use :func:`compute_executed_config_digest` for
    the digest of the config the runner *actually* forwarded to
    upstream (with runner-specific fields excluded).
    """
    payload = _payload_from_config(config)
    encoded = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_executed_config_digest(config: object, *, exclude_fields: tuple[str, ...] = ()) -> str:
    """Like :func:`compute_config_digest`, but strips ``exclude_fields``.

    Used to compute the digest of the config the runner actually
    forwarded to upstream. ProteinMPNN forwards every field (empty
    ``exclude_fields``). RFdiffusion forwards ``seed`` (as
    ``inference.design_startnum``) only in deterministic mode, so it
    passes ``("seed",)`` for non-deterministic runs — a seed-only
    change must not flip the executed digest when the base seed was
    never forwarded.

    Args:
        config: The full requested config (a dataclass, ``Mapping``,
            or plain object).
        exclude_fields: Field names to strip before hashing. Names
            not present in the payload are silently ignored (so the
            same helper works for both RFdiffusion and ProteinMPNN
            without conditional logic).
    """
    payload = _payload_from_config(config)
    for field_name in exclude_fields:
        payload.pop(field_name, None)
    encoded = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_execution_provenance(
    *,
    runner_name: str,
    execution_mode: ExecutionMode | str,
    status: ExecutionStatus | str,
    exit_code: int = 0,
    image_digest: str | None = None,
    artifacts: tuple[ArtifactReference, ...] = (),
    command: tuple[str, ...] = (),
    source_backbone_digest: str | None = None,
    requested_config_digest: str = "",
    executed_config_digest: str | None = None,
    executed: bool = False,
    cache_hit: bool = False,
) -> ProvenanceMetadata:
    """Build the generic execution portion of a runner provenance record.

    The execution flags and digests are explicit because a generic helper
    cannot infer whether a caller actually dispatched work or returned a
    cached / dry-run result.
    """
    normalized_status = ExecutionStatus(status)
    normalized_mode = ExecutionMode(execution_mode)
    if not executed:
        executed_config_digest = None
    return dataclasses.replace(
        EMPTY_PROVENANCE,
        model_identifier=runner_name,
        runner_name=runner_name,
        execution_mode=normalized_mode,
        status=normalized_status,
        exit_code=exit_code,
        image_digest=validate_image_digest(image_digest),
        source_backbone_digest=source_backbone_digest,
        requested_config_digest=requested_config_digest,
        executed_config_digest=executed_config_digest,
        executed=executed,
        cache_hit=cache_hit,
        artifacts=artifacts,
        command=command,
    )


# ---------------------------------------------------------------------------
# Structured subprocess capture (shared between runners)
# ---------------------------------------------------------------------------


def stderr_tail(stderr: str | bytes | None, *, limit: int = 512) -> str:
    """Return the last ``limit`` chars of ``stderr`` (decoded safely).

    Public helper used by the runner ``utils`` modules to build
    :class:`InvokeResult` records. Accepts ``bytes`` because
    ``subprocess.run(text=True)`` decodes with the locale but
    ``TimeoutExpired.stderr`` is still bytes when ``text=False`` —
    we normalise both.
    """
    if stderr is None:
        return ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return stderr[-limit:]


def short_reason_from_stderr(stderr_tail: str) -> str:
    """Return the first non-empty line of ``stderr_tail``, stripped.

    Empty stderr → empty string. Lines are stripped of leading /
    trailing whitespace but otherwise returned verbatim — no
    reformatting, since the upstream message is the most actionable
    signal an operator will see.
    """
    for line in stderr_tail.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


@dataclass(frozen=True)
class InvokeResult:
    """Structured outcome of one upstream CLI invocation.

    The runner consumes the full record (exit code, stderr tail,
    timeout flag, short failure reason) to populate
    :class:`ProvenanceMetadata`. The legacy ``invoke()`` helper
    returns only ``exit_code`` for backward compatibility; new
    callers should prefer the ``_invoke_with_metadata`` helper that
    returns an :class:`InvokeResult` directly.

    Attributes:
        exit_code: The subprocess exit code. ``124`` when the
            invocation timed out (the conventional GNU coreutils
            timeout return code).
        stderr_tail: The last 512 bytes of stderr, decoded with
            ``errors="replace"``. Empty when stderr was empty or
            the process never ran.
        timed_out: True when the subprocess raised
            ``subprocess.TimeoutExpired``.
        failure_reason: Short human-readable failure description
            (``"timeout after 3600s"`` or the first non-empty line
            of stderr, trimmed). Empty on success.
    """

    exit_code: int
    stderr_tail: str = ""
    timed_out: bool = False
    failure_reason: str = ""
    command: tuple[str, ...] = ()

    @staticmethod
    def from_stderr(exit_code: int, stderr: str | bytes | None) -> InvokeResult:
        """Build an :class:`InvokeResult` from an exit code + raw stderr.

        Used by ``utils._invoke_with_metadata`` to construct the
        record without re-deriving the helper logic. Keeps
        :func:`stderr_tail` and :func:`short_reason_from_stderr`
        referenced from this module so pyright's strict
        ``reportUnusedFunction`` is happy.
        """
        tail = stderr_tail(stderr)
        return InvokeResult(
            exit_code=exit_code,
            stderr_tail=tail,
            failure_reason=short_reason_from_stderr(tail) if exit_code != 0 else "",
        )


# ---------------------------------------------------------------------------
# ProvenanceMetadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceMetadata:
    """Serialization-friendly provenance for one runner invocation.

    Attributes:
        model_identifier: The model / checkpoint identifier that was
            invoked (e.g. ``"v_48_020"`` for ProteinMPNN,
            ``"RFdiffusion"`` for the RFdiffusion runner). The runner
            decides what string to record; consumers MUST NOT parse it.
        temperature: Sampling temperature when the runner exposes one
            (ProteinMPNN). ``None`` when the runner does not expose a
            temperature (RFdiffusion's diffusion noise scale is
            upstream-internal and is intentionally not surfaced here).
        image_digest: Caller-supplied image digest in canonical OCI
            form (``sha256:<hex>``). The runner normalises via
            :func:`validate_image_digest` at the ``run()`` entry
            point. ``None`` when the caller did not supply one.
        source_backbone_digest: ``sha256`` of the source backbone file
            (the target PDB for RFdiffusion; the input PDB for
            ProteinMPNN). ``None`` when the file was absent or not
            provided — absence is a signal, not an error.
        exit_code: Exit code returned by the subprocess invocation.
            ``0`` for successful runs and for the idempotent / dry-run
            paths (those paths never invoke the binary). ``124`` for
            a timeout.
        failure_reason: Short human-readable failure reason (timeout
            description or first non-empty stderr line). Empty on
            success and on the idempotent / dry-run paths.
        stderr_tail: Last 512 chars of stderr, decoded with
            ``errors="replace"``. Captures the upstream message
            operators need to debug a failed run.
        base_seed: The seed the runner *actually forwarded* to
            upstream. ``None`` when the run was non-deterministic
            (RFdiffusion with ``deterministic=False`` — upstream
            uses system entropy). When set with
            :data:`RNG_INTENT_PER_DESIGN_INDEX`, the per-design
            seeds are ``base_seed .. base_seed + task_count - 1``
            (upstream seeds design index ``i`` with
            ``design_startnum + i``); the range is fully encoded by
            ``base_seed`` + ``task_count`` — no per-seed list is
            fabricated.
        requested_seed: The seed the *caller asked for* in the
            config. Equal to ``base_seed`` for seeded runs
            (ProteinMPNN; RFdiffusion with ``deterministic=True``);
            ``base_seed`` is ``None`` when the run was
            non-deterministic.
            ``None`` when the caller did not set one.
        task_count: The number of outputs requested.
        rng_intent: How the runner actually used the RNG. One of
            :data:`RNG_INTENT_SINGLE_STREAM` (one seed drives all
            outputs), :data:`RNG_INTENT_PER_DESIGN_INDEX`
            (upstream seeds each design with
            ``base_seed + design_index``), or
            :data:`RNG_INTENT_NON_DETERMINISTIC` (upstream used
            system entropy). :data:`RNG_INTENT_SEED_NOT_FORWARDED`
            is retained for compatibility but used by no runner.
        canonical_output: The runner's canonical raw outputs *before*
            any downstream substitution (ProteinMPNN's FASTA
            sequences before any downstream D-residue rewrite).
            Empty for runners that have no downstream substitution
            (RFdiffusion) or when the run produced no output.
        requested_config_digest: ``sha256`` of the canonical
            *requested* config (see :func:`compute_config_digest`).
            Always present — describes what the caller asked for.
        executed_config_digest: ``sha256`` of the canonical
            *executed* config (see
            :func:`compute_executed_config_digest`). ``None`` when
            ``executed=False`` (cache hit, dry run) — the runner
            has no record of what config produced the existing
            files, so it does not fabricate one.
        executed: ``True`` when the runner invoked the upstream CLI
            on this call. ``False`` for cache hits, dry runs, and
            error short-circuits before subprocess dispatch.
        cache_hit: ``True`` when the result was returned from an
            existing on-disk artifact (the idempotent path).
            ``executed_config_digest`` is ``None`` in this case.
    """

    model_identifier: str
    temperature: float | None
    image_digest: str | None
    source_backbone_digest: str | None
    exit_code: int
    failure_reason: str
    stderr_tail: str
    base_seed: int | None
    requested_seed: int | None
    task_count: int
    rng_intent: str
    canonical_output: tuple[str, ...] = field(default_factory=tuple)
    requested_config_digest: str = ""
    executed_config_digest: str | None = None
    executed: bool = False
    cache_hit: bool = False
    runner_name: str = ""
    execution_mode: ExecutionMode | str = ""
    status: ExecutionStatus | str = ""
    artifacts: tuple[ArtifactReference, ...] = field(default_factory=tuple)
    command: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a JSON-safe dictionary.

        The shape is stable across runs. ``canonical_output`` becomes
        a list because JSON has no tuple. ``None`` round-trips as
        ``null``.
        """
        return {
            "model_identifier": self.model_identifier,
            "temperature": self.temperature,
            "image_digest": self.image_digest,
            "source_backbone_digest": self.source_backbone_digest,
            "exit_code": self.exit_code,
            "failure_reason": self.failure_reason,
            "stderr_tail": self.stderr_tail,
            "base_seed": self.base_seed,
            "requested_seed": self.requested_seed,
            "task_count": self.task_count,
            "rng_intent": self.rng_intent,
            "canonical_output": list(self.canonical_output),
            "requested_config_digest": self.requested_config_digest,
            "executed_config_digest": self.executed_config_digest,
            "executed": self.executed,
            "cache_hit": self.cache_hit,
            "runner_name": self.runner_name,
            "execution_mode": self.execution_mode,
            "status": self.status,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "command": list(self.command),
        }

    @staticmethod
    def from_parts(
        *,
        model_identifier: str,
        config: object,
        base_seed: int | None,
        requested_seed: int | None,
        task_count: int,
        rng_intent: str,
        temperature: float | None = None,
        image_digest: str | None = None,
        source_backbone_path: Path | None = None,
        exit_code: int = 0,
        failure_reason: str = "",
        stderr_tail: str = "",
        canonical_output: tuple[str, ...] | None = None,
        exclude_fields: tuple[str, ...] = (),
        executed: bool = True,
        cache_hit: bool = False,
        runner_name: str = "",
        execution_mode: ExecutionMode | str = "",
        status: ExecutionStatus | str = "",
        artifacts: tuple[ArtifactReference, ...] = (),
        command: tuple[str, ...] = (),
    ) -> ProvenanceMetadata:
        """Build a :class:`ProvenanceMetadata` from runner-local pieces.

        ``requested_config_digest`` always covers the full caller
        config; ``executed_config_digest`` strips ``exclude_fields``
        (use this for fields the runner does not forward to
        upstream). When ``executed=False`` the executed digest is
        recorded as ``None`` because the runner cannot honestly
        claim the requested config produced the existing bytes.

        Raises:
            ValueError: When ``image_digest`` is supplied but does
                not match the canonical sha256 form.
        """
        return ProvenanceMetadata(
            model_identifier=model_identifier,
            temperature=temperature,
            image_digest=validate_image_digest(image_digest),
            source_backbone_digest=compute_file_digest(source_backbone_path)
            if source_backbone_path is not None
            else None,
            exit_code=exit_code,
            failure_reason=failure_reason,
            stderr_tail=stderr_tail,
            base_seed=base_seed,
            requested_seed=requested_seed,
            task_count=task_count,
            rng_intent=rng_intent,
            canonical_output=canonical_output if canonical_output is not None else (),
            requested_config_digest=compute_config_digest(config),
            executed_config_digest=compute_executed_config_digest(
                config, exclude_fields=exclude_fields
            )
            if executed
            else None,
            executed=executed,
            cache_hit=cache_hit,
            runner_name=runner_name,
            execution_mode=execution_mode,
            status=status,
            artifacts=artifacts,
            command=command,
        )


# Sentinel for callers that want to attach an "empty" provenance
# without paying for the digest computation. ``RFdiffusionResult``
# and ``ProteinMPNNResult`` initialise their ``provenance`` field to
# this value when the runner has no provenance to record.
EMPTY_PROVENANCE = ProvenanceMetadata(
    model_identifier="",
    temperature=None,
    image_digest=None,
    source_backbone_digest=None,
    exit_code=0,
    failure_reason="",
    stderr_tail="",
    base_seed=None,
    requested_seed=None,
    task_count=0,
    rng_intent=RNG_INTENT_SINGLE_STREAM,
)
