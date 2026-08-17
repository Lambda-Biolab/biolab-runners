"""Tests for the shared provenance module (S2 reproducibility).

The provenance module is the contract every downstream manifest
relies on. Tests cover:

* Image digest validation + normalisation — bare-hex and OCI-prefixed
  forms both produce the canonical OCI form on return; malformed
  values raise ``ValueError``.
* Digest stability — same inputs → same digest.
* Digest sensitivity — a 1-byte change in a config or backbone flips
  the digest.
* ``executed_config_digest`` honesty — stripping a field via
  ``exclude_fields`` keeps the digest stable across calls that
  change only the stripped field.
* Manifest equivalence — same inputs → identical
  :class:`ProvenanceMetadata`.
* Manifest divergence — different seeds / checkpoints / backbones /
  image digests → distinct manifests.
* Honest RNG metadata — ``base_seed`` is ``None`` when the run was
  non-deterministic (no seed forwarded); ``requested_seed`` carries
  the caller's intent; ``rng_intent`` names the actual upstream
  behaviour.
* Cache / execution honesty — ``cache_hit=True`` clears
  ``executed_config_digest``; ``executed=True`` requires a real
  subprocess invocation.
* Stderr / timeout capture — :class:`InvokeResult` carries the
  512-char stderr tail and the timeout flag.
* Strict canonical JSON — non-JSON-native inputs (sets, custom
  objects) raise ``TypeError`` rather than silently stringifying.
* Sentinel behaviour — :data:`EMPTY_PROVENANCE` round-trips through
  :meth:`ProvenanceMetadata.to_dict`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from biolab_runners.provenance import (
    EMPTY_PROVENANCE,
    RNG_INTENT_NON_DETERMINISTIC,
    RNG_INTENT_PER_DESIGN_INDEX,
    RNG_INTENT_SEED_NOT_FORWARDED,
    RNG_INTENT_SINGLE_STREAM,
    InvokeResult,
    ProvenanceMetadata,
    compute_config_digest,
    compute_executed_config_digest,
    compute_file_digest,
    short_reason_from_stderr,
    stderr_tail,
    validate_image_digest,
)

# ---------------------------------------------------------------------------
# Image digest validation + normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "digest,expected",
    [
        # None passes through; bare hex normalises to OCI form on return.
        (None, None),
        (
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        ),
        (
            "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        ),
    ],
)
def test_validate_image_digest_accepts_and_normalises_canonical_forms(
    digest: str | None, expected: str | None
) -> None:
    assert validate_image_digest(digest) == expected


@pytest.mark.parametrize(
    "digest",
    [
        "sha256:abc",
        "sha256:0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef",
        "md5:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "0123456789abcdef",
        "not-a-digest",
        "",
    ],
)
def test_validate_image_digest_rejects_malformed(digest: str) -> None:
    with pytest.raises(ValueError, match="image_digest must be"):
        validate_image_digest(digest)


def test_validate_image_digest_normalises_to_oci_form() -> None:
    """Bare hex → OCI form on return, so downstream comparison sees one shape."""
    hex_form = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    oci_form = f"sha256:{hex_form}"
    assert validate_image_digest(hex_form) == oci_form
    assert validate_image_digest(oci_form) == oci_form  # idempotent


# ---------------------------------------------------------------------------
# Digest helpers
# ---------------------------------------------------------------------------


def test_compute_file_digest_is_stable(tmp_path: Any) -> None:
    target = tmp_path / "backbone.pdb"
    target.write_text("HEADER\nATOM      1  CA  GLY A   1       0.000   0.000   0.000\nEND\n")
    first = compute_file_digest(target)
    second = compute_file_digest(target)
    assert first == second
    assert isinstance(first, str)
    assert len(first) == 64  # sha256 hex digest length


def test_compute_file_digest_changes_on_byte_change(tmp_path: Any) -> None:
    target = tmp_path / "backbone.pdb"
    target.write_text("HEADER\nATOM      1  CA  GLY A   1       0.000   0.000   0.000\nEND\n")
    baseline = compute_file_digest(target)
    target.write_text("HEADER\nATOM      1  CA  GLY A   1       0.000   0.000   1.000\nEND\n")
    assert compute_file_digest(target) != baseline


def test_compute_file_digest_returns_none_for_missing(tmp_path: Any) -> None:
    assert compute_file_digest(tmp_path / "absent.pdb") is None


def test_compute_file_digest_handles_large_file(tmp_path: Any) -> None:
    """Stream-read path: file larger than the chunk size must hash correctly."""
    import hashlib

    target = tmp_path / "big.pdb"
    target.write_bytes(b"X" * 200_000)  # > 64 KiB chunk
    assert compute_file_digest(target) == hashlib.sha256(b"X" * 200_000).hexdigest()


# ---------------------------------------------------------------------------
# Config digest
# ---------------------------------------------------------------------------


def test_compute_config_digest_is_stable_across_calls() -> None:
    from biolab_runners.rfdiffusion.config import RFdiffusionConfig

    cfg = RFdiffusionConfig(name="x", task_count=10, seed=42)
    assert compute_config_digest(cfg) == compute_config_digest(cfg)


def test_compute_config_digest_changes_on_field_change() -> None:
    from biolab_runners.rfdiffusion.config import RFdiffusionConfig

    base = RFdiffusionConfig(name="x", task_count=10, seed=42)
    bumped = RFdiffusionConfig(name="x", task_count=11, seed=42)
    assert compute_config_digest(base) != compute_config_digest(bumped)


def test_compute_config_digest_ignores_object_identity() -> None:
    from biolab_runners.proteinmpnn.config import ProteinMPNNConfig

    a = ProteinMPNNConfig(name="n", seed=7, temperature=0.2)
    b = ProteinMPNNConfig(name="n", seed=7, temperature=0.2)
    assert compute_config_digest(a) == compute_config_digest(b)


def test_compute_config_digest_accepts_mapping() -> None:
    payload = {"name": "x", "seed": 1}
    assert compute_config_digest(payload) == compute_config_digest(dict(payload))


def test_compute_config_digest_rejects_non_json_native() -> None:
    """A non-JSON-native payload (set, custom object) must raise, not stringify."""
    payload = {"name": "x", "tags": {"a", "b", "c"}}
    with pytest.raises(TypeError, match="non-JSON-native"):
        compute_config_digest(payload)


def test_compute_executed_config_digest_excludes_named_fields() -> None:
    """The helper's contract: stripping a field via ``exclude_fields`` keeps
    the digest stable across calls that change only the stripped field.

    Generic mechanism — RFdiffusion passes ``("seed",)`` in non-deterministic
    mode (the base seed is not forwarded then), and an empty tuple otherwise.
    """
    cfg_a = {"name": "x", "seed": 1}
    cfg_b = {"name": "x", "seed": 999}
    # Requested digests differ (seed is part of the requested config).
    assert compute_config_digest(cfg_a) != compute_config_digest(cfg_b)
    # Executed digests agree (seed is excluded).
    digest_a = compute_executed_config_digest(cfg_a, exclude_fields=("seed",))
    digest_b = compute_executed_config_digest(cfg_b, exclude_fields=("seed",))
    assert digest_a == digest_b
    # An unknown field name is silently ignored (no KeyError).
    assert compute_executed_config_digest(
        cfg_a, exclude_fields=("nonexistent",)
    ) == compute_config_digest(cfg_a)


# ---------------------------------------------------------------------------
# Stderr helpers
# ---------------------------------------------------------------------------


def test_stderr_tail_handles_str_bytes_and_none() -> None:
    assert stderr_tail(None) == ""
    assert stderr_tail("hello") == "hello"
    assert stderr_tail(b"hello") == "hello"
    assert stderr_tail("a" * 1000, limit=10) == "a" * 10


def test_short_reason_from_stderr_returns_first_nonempty_line() -> None:
    assert short_reason_from_stderr("") == ""
    assert short_reason_from_stderr("   \n  \n") == ""
    assert short_reason_from_stderr("  RuntimeError: oops  \nnext line") == "RuntimeError: oops"
    assert short_reason_from_stderr("first\nsecond") == "first"


# ---------------------------------------------------------------------------
# InvokeResult structured capture
# ---------------------------------------------------------------------------


def test_invoke_result_from_stderr_sets_failure_reason() -> None:
    """The helper collapses the raw stderr → tail + first-non-empty-line."""
    result = InvokeResult.from_stderr(exit_code=1, stderr="  RuntimeError: oops  \n  at frame X")
    assert result.exit_code == 1
    assert result.stderr_tail == "  RuntimeError: oops  \n  at frame X"
    assert result.failure_reason == "RuntimeError: oops"


def test_invoke_result_from_stderr_clears_failure_reason_on_success() -> None:
    result = InvokeResult.from_stderr(exit_code=0, stderr="informational noise")
    assert result.failure_reason == ""


def test_invoke_result_round_trips_through_json() -> None:
    """The structured result must be JSON-safe (no live subprocess handles)."""
    import dataclasses

    payload = InvokeResult(
        exit_code=124,
        stderr_tail="Killed by signal 9",
        timed_out=True,
        failure_reason="timeout after 3600s",
    )
    assert json.loads(json.dumps(dataclasses.asdict(payload)))


# ---------------------------------------------------------------------------
# Manifest equivalence / divergence (S2 contract)
# ---------------------------------------------------------------------------


def _build_provenance(**overrides: object) -> ProvenanceMetadata:
    """Build a ProvenanceMetadata with sensible defaults for unit tests.

    Uses the dataclass constructor directly (not ``from_parts``) so
    tests don't pay for digest computation when they only care
    about other fields.
    """
    kwargs: dict[str, Any] = {
        "model_identifier": "RFdiffusion",
        "temperature": None,
        "image_digest": None,
        "source_backbone_digest": None,
        "exit_code": 0,
        "failure_reason": "",
        "stderr_tail": "",
        "base_seed": None,
        "requested_seed": None,
        "task_count": 4,
        "rng_intent": RNG_INTENT_SINGLE_STREAM,
        "canonical_output": (),
        "requested_config_digest": "deadbeef" * 8,
        "executed_config_digest": None,
        "executed": True,
        "cache_hit": False,
    }
    kwargs.update(overrides)
    return ProvenanceMetadata(**kwargs)


def test_manifest_equivalence_on_exact_rerun() -> None:
    """Same inputs → byte-identical to_dict() (the S2 equivalence contract)."""
    kwargs = {
        "requested_seed": 42,
        "base_seed": 42,
        "image_digest": "sha256:abc",
        "executed_config_digest": "cafebabe" * 8,
    }
    a = _build_provenance(**kwargs)
    b = _build_provenance(**kwargs)
    assert a.to_dict() == b.to_dict()


def test_manifest_divergence_on_different_requested_seeds() -> None:
    """Different requested seeds → distinct manifests (the S2 distinctness contract)."""
    a = _build_provenance(requested_seed=0)
    b = _build_provenance(requested_seed=1)
    assert a.to_dict() != b.to_dict()


def test_manifest_divergence_on_different_base_seed() -> None:
    """Different ``base_seed`` → distinct manifests (e.g. ProteinMPNN actually forwards)."""
    a = _build_provenance(base_seed=0)
    b = _build_provenance(base_seed=1)
    assert a.to_dict() != b.to_dict()


def test_manifest_divergence_on_different_task_count() -> None:
    a = _build_provenance(task_count=4)
    b = _build_provenance(task_count=5)
    assert a.to_dict() != b.to_dict()


def test_manifest_divergence_on_different_checkpoint() -> None:
    a = _build_provenance(model_identifier="RFdiffusion")
    b = _build_provenance(model_identifier="RFdiffusion-custom")
    assert a.to_dict() != b.to_dict()


def test_manifest_divergence_on_different_image_digest() -> None:
    a = _build_provenance(image_digest="sha256:aaa")
    b = _build_provenance(image_digest="sha256:bbb")
    assert a.to_dict() != b.to_dict()


def test_manifest_divergence_on_backbone_digest() -> None:
    a = _build_provenance(source_backbone_digest="deadbeef" * 8)
    b = _build_provenance(source_backbone_digest="cafebabe" * 8)
    assert a.to_dict() != b.to_dict()


def test_manifest_divergence_on_exit_code() -> None:
    a = _build_provenance(exit_code=0)
    b = _build_provenance(exit_code=1)
    assert a.to_dict() != b.to_dict()


def test_manifest_divergence_on_failure_reason() -> None:
    a = _build_provenance(failure_reason="")
    b = _build_provenance(failure_reason="OOM")
    assert a.to_dict() != b.to_dict()


def test_manifest_divergence_on_stderr_tail() -> None:
    a = _build_provenance(stderr_tail="")
    b = _build_provenance(stderr_tail="Traceback...")
    assert a.to_dict() != b.to_dict()


def test_manifest_divergence_on_executed_config_digest() -> None:
    a = _build_provenance(executed_config_digest="aaaa" * 16)
    b = _build_provenance(executed_config_digest="bbbb" * 16)
    assert a.to_dict() != b.to_dict()


def test_manifest_divergence_on_executed_flag() -> None:
    """``executed=True`` vs ``executed=False`` is a real distinction — the audit cares."""
    a = _build_provenance(executed=True, executed_config_digest="x" * 64)
    b = _build_provenance(executed=False, executed_config_digest=None)
    assert a.to_dict() != b.to_dict()


def test_manifest_divergence_on_cache_hit() -> None:
    a = _build_provenance(cache_hit=False, executed=True, executed_config_digest="x" * 64)
    b = _build_provenance(cache_hit=True, executed=False, executed_config_digest=None)
    assert a.to_dict() != b.to_dict()


def test_manifest_round_trips_through_json() -> None:
    payload = _build_provenance().to_dict()
    assert json.loads(json.dumps(payload)) == payload


def test_manifest_carries_canonical_output() -> None:
    canonical = ("ACDEFG", "ACDEFGH", "ACDEFGHK")
    p = _build_provenance(canonical_output=canonical)
    assert p.canonical_output == canonical
    assert p.to_dict()["canonical_output"] == list(canonical)


def test_manifest_records_rng_intent_single_stream() -> None:
    p = _build_provenance(rng_intent=RNG_INTENT_SINGLE_STREAM)
    assert p.rng_intent == "single-stream"


def test_manifest_records_rng_intent_per_design_index() -> None:
    """RFdiffusion deterministic: upstream seeds each design with
    ``design_startnum + i``, so the label names the per-design-index mode."""
    p = _build_provenance(rng_intent=RNG_INTENT_PER_DESIGN_INDEX, base_seed=42, task_count=3)
    assert p.rng_intent == "per-design-index"
    assert p.base_seed == 42
    assert p.task_count == 3


def test_manifest_records_rng_intent_seed_not_forwarded() -> None:
    """Compatibility: the historical label remains importable and serialises
    unchanged, but no runner in this slice emits it anymore."""
    p = _build_provenance(rng_intent=RNG_INTENT_SEED_NOT_FORWARDED, base_seed=None)
    assert p.rng_intent == "seed-not-forwarded"
    assert p.base_seed is None


def test_manifest_records_rng_intent_non_deterministic() -> None:
    p = _build_provenance(rng_intent=RNG_INTENT_NON_DETERMINISTIC, base_seed=None)
    assert p.rng_intent == "non-deterministic"


def test_manifest_has_no_per_task_seed_field() -> None:
    """S2 honesty: ``per_task_seeds`` must NOT exist on the manifest.

    The runner invokes upstream once with a single ``--seed`` flag;
    claiming per-task seeds would misrepresent the execution."""
    p = _build_provenance()
    assert not hasattr(p, "per_task_seeds")


def test_from_parts_validates_image_digest() -> None:
    payload = {"name": "x", "seed": 0, "task_count": 1}
    with pytest.raises(ValueError, match="image_digest must be"):
        ProvenanceMetadata.from_parts(
            model_identifier="x",
            config=payload,
            base_seed=0,
            requested_seed=0,
            task_count=1,
            rng_intent=RNG_INTENT_SINGLE_STREAM,
            image_digest="not-a-digest",
        )


def test_from_parts_accepts_none_image_digest() -> None:
    payload = {"name": "x", "seed": 0, "task_count": 1}
    p = ProvenanceMetadata.from_parts(
        model_identifier="x",
        config=payload,
        base_seed=0,
        requested_seed=0,
        task_count=1,
        rng_intent=RNG_INTENT_SINGLE_STREAM,
        image_digest=None,
    )
    assert p.image_digest is None


def test_from_parts_normalises_image_digest_to_oci_form() -> None:
    """Bare hex form must be normalised to OCI-prefixed form in the manifest."""
    payload = {"name": "x", "seed": 0, "task_count": 1}
    bare_hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    p = ProvenanceMetadata.from_parts(
        model_identifier="x",
        config=payload,
        base_seed=0,
        requested_seed=0,
        task_count=1,
        rng_intent=RNG_INTENT_SINGLE_STREAM,
        image_digest=bare_hex,
    )
    assert p.image_digest == f"sha256:{bare_hex}"


def test_from_parts_cache_hit_clears_executed_config_digest() -> None:
    """Cache hits MUST NOT carry an executed config digest — the runner doesn't
    know which prior call produced the existing files."""
    payload = {"name": "x", "seed": 0, "task_count": 1}
    p = ProvenanceMetadata.from_parts(
        model_identifier="x",
        config=payload,
        base_seed=0,
        requested_seed=0,
        task_count=1,
        rng_intent=RNG_INTENT_SINGLE_STREAM,
        executed=False,
        cache_hit=True,
    )
    assert p.executed is False
    assert p.cache_hit is True
    assert p.executed_config_digest is None
    assert p.requested_config_digest != ""  # what *this* call asked for IS recorded


def test_from_parts_dry_run_records_requested_digest_only() -> None:
    payload = {"name": "x", "seed": 0, "task_count": 1}
    p = ProvenanceMetadata.from_parts(
        model_identifier="x",
        config=payload,
        base_seed=0,
        requested_seed=0,
        task_count=1,
        rng_intent=RNG_INTENT_SINGLE_STREAM,
        executed=False,
        cache_hit=False,
    )
    assert p.executed is False
    assert p.cache_hit is False
    assert p.executed_config_digest is None
    assert p.requested_config_digest != ""


def test_from_parts_executed_run_includes_both_digests() -> None:
    """A real run carries both the requested digest AND the executed digest."""
    payload = {"name": "x", "seed": 0, "task_count": 1}
    p = ProvenanceMetadata.from_parts(
        model_identifier="x",
        config=payload,
        base_seed=0,
        requested_seed=0,
        task_count=1,
        rng_intent=RNG_INTENT_SINGLE_STREAM,
        executed=True,
    )
    assert p.executed is True
    assert p.executed_config_digest is not None
    assert p.requested_config_digest is not None


def test_from_parts_exclude_fields_does_not_alter_requested_digest() -> None:
    """The requested digest always covers the full config; the executed digest
    is the one that respects ``exclude_fields`` (generic mechanism — RFdiffusion
    passes ``("seed",)`` in non-deterministic mode)."""
    cfg_a = {"name": "x", "seed": 1}
    cfg_b = {"name": "x", "seed": 999}
    p_a = ProvenanceMetadata.from_parts(
        model_identifier="x",
        config=cfg_a,
        base_seed=1,
        requested_seed=1,
        task_count=10,
        rng_intent=RNG_INTENT_SINGLE_STREAM,
        executed=True,
        exclude_fields=("seed",),
    )
    p_b = ProvenanceMetadata.from_parts(
        model_identifier="x",
        config=cfg_b,
        base_seed=999,
        requested_seed=999,
        task_count=10,
        rng_intent=RNG_INTENT_SINGLE_STREAM,
        executed=True,
        exclude_fields=("seed",),
    )
    # Requested digests differ (seed is part of the requested config).
    assert p_a.requested_config_digest != p_b.requested_config_digest
    # Executed digests agree (seed is excluded).
    assert p_a.executed_config_digest == p_b.executed_config_digest


# ---------------------------------------------------------------------------
# EMPTY_PROVENANCE sentinel
# ---------------------------------------------------------------------------


def test_empty_provenance_is_serialisable() -> None:
    payload = EMPTY_PROVENANCE.to_dict()
    assert json.loads(json.dumps(payload)) == payload


def test_empty_provenance_distinct_from_built() -> None:
    assert EMPTY_PROVENANCE.to_dict() != _build_provenance().to_dict()


def test_empty_provenance_has_no_executed_config_digest() -> None:
    """The sentinel is the "no provenance" signal — must not claim an executed digest."""
    assert EMPTY_PROVENANCE.executed_config_digest is None
    assert EMPTY_PROVENANCE.executed is False
    assert EMPTY_PROVENANCE.cache_hit is False
