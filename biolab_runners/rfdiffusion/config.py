"""Configuration for an RFdiffusion backbone generation.

Mirrors the upstream CLI flags that the Activin-E pipeline actually
uses. Every field has a conservative default so a bare-minimum
campaign still produces a valid result.

The runner accepts the upstream ``contigmap.contigs`` syntax
(e.g. ``"12-18 A3-117/0 50-50"``) so callers familiar with
RFdiffusion can use the documentation directly.

S2 reproducibility fields (per the Activin-E reproducibility plan):

* ``seed`` — the user-facing non-negative base seed. Stock upstream
  RFdiffusion has **no** ``inference.seed`` key (a wrapper that
  appends it via Hydra's ``+inference.seed=...`` would have it
  silently ignored — it is inert, nothing reads it; a strict
  override is rejected; either way it never affects the RNG). Its
  deterministic mode seeds each design with the design index. The
  supported external base is ``inference.design_startnum``, and the
  runner maps ``seed`` → ``inference.design_startnum`` when
  ``deterministic=True``. Concretely, upstream's
  ``scripts/run_inference.py`` seeds design ``i_des`` with
  ``i_des`` inside ``for i_des in range(design_startnum,
  design_startnum + num_designs)``, so the per-design seeds are
  ``seed, seed+1, ..., seed + task_count - 1`` and output
  indices/names start at ``seed``. The provenance manifest records
  ``base_seed == requested_seed == seed``; the per-design range is
  encoded by ``base_seed`` + ``task_count`` (no per-seed list is
  fabricated). When ``deterministic=False`` the seed is
  deliberately **not** forwarded — upstream uses system entropy, so
  claiming a pinned seed would be dishonest (the manifest records
  ``base_seed=None`` and ``rng_intent="non-deterministic"``).
* **Default ``seed=0`` is reproducible, not diverse.** With
  ``deterministic=True`` (the default) the same config always
  produces the *same* designs (per-design seeds
  ``0..task_count-1``). Callers that want distinct designs across
  runs or replicas must vary ``seed`` explicitly.
* ``checkpoint`` — the model / checkpoint identifier that flows
  into the provenance record. Defaults to ``"RFdiffusion"``
  because the upstream container ships a single model; callers
  pinning a custom checkpoint can override.

Note on temperature: RFdiffusion does not expose a sampling
temperature in the sense ProteinMPNN does — its diffusion process
is parameterised by noise scales (``diffusion.noise_scale_ca``,
``diffusion.noise_scale_frame``) that are tuned per-application
and live upstream-internal. We deliberately do **not** surface a
``temperature`` field here; mapping to ``noise_scale_ca`` would
silently change the upstream behaviour and is not part of the
canonical contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["RESERVED_CANONICAL_KEYS", "UNSUPPORTED_UPSTREAM_KEYS", "ContigMap", "RFdiffusionConfig"]


#: Canonical Hydra keys the runner emits itself. ``extra`` may not
#: override them — a conflict raises ``ValueError`` at config
#: construction time (fail closed) so a caller cannot silently
#: change what the runner forwards to upstream.
RESERVED_CANONICAL_KEYS: tuple[str, ...] = (
    "inference.num_designs",
    "inference.design_startnum",
    "inference.deterministic",
    "inference.cyclic",
    "inference.cyc_chains",
    "contigmap.contigs",
    "ppi.hotspot_res",
)

#: Upstream keys that do NOT exist in stock RFdiffusion and must never
#: be forwarded. ``inference.seed`` is not a Hydra key in
#: ``config/inference/base.yaml``: a wrapper that appends it (``+``
#: override) would have it silently ignored — it is inert, nothing in
#: upstream reads it — and a strict override is rejected. Either way
#: it never affects the RNG. The supported base is
#: ``inference.design_startnum``. Passing these via ``extra`` raises a
#: clear ``ValueError`` instead of forwarding a key that can only be
#: inert-or-rejected upstream.
UNSUPPORTED_UPSTREAM_KEYS: tuple[str, ...] = ("inference.seed",)


@dataclass(frozen=True)
class ContigMap:
    """Parsed upstream ``contigmap.contigs`` value.

    The runner passes ``contigs`` through as a string to the CLI;
    this dataclass exists for validation and for callers that prefer
    programmatic construction.
    """

    contigs: str
    length_min: int
    length_max: int

    def __post_init__(self) -> None:
        """Validate contigs + length range."""
        if not self.contigs:
            raise ValueError("contigs must be a non-empty string")
        if self.length_min < 1 or self.length_max < self.length_min:
            raise ValueError(f"length range invalid: min={self.length_min} max={self.length_max}")


@dataclass(frozen=True)
class RFdiffusionConfig:
    """Per-invocation configuration for the RFdiffusion runner.

    The defaults correspond to linear 14–18-residue peptide binders
    with no hotspots — the regime the PDF describes as experimental.
    Head-to-tail macrocycles and disulfide pairs are opt-in via
    ``mode`` / ``disulfide_pairs``.
    """

    name: str = "backbone"
    task_count: int = 1000
    target_pdb: str = ""
    contigs: str = "14-18"
    length_min: int = 14
    length_max: int = 18
    mode: str = "linear"  # linear | head_to_tail | disulfide
    disulfide_pairs: tuple[tuple[int, int], ...] = ()
    hotspots: tuple[str, ...] = ()
    deterministic: bool = True
    seed: int = 0
    checkpoint: str = "RFdiffusion"
    extra: Mapping[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """Validate name + mode + S2 fields + extra keys."""
        _validate_name(self)
        _validate_mode_and_lengths(self)
        _validate_s2_fields(self)
        _validate_extra_keys(self)


def _validate_name(cfg: RFdiffusionConfig) -> None:
    """Validate ``name`` as a single safe path component.

    The runner places outputs at ``<output_root>/<name>/<digest>/``; a
    name containing path separators, NUL, or control characters (or
    the ``.`` / ``..`` components) would escape the per-name directory
    or produce an unusable path. Rejected at construction (fail
    closed) rather than silently mangled.
    """
    name = cfg.name
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or any(ord(ch) < 32 for ch in name)
    ):
        raise ValueError(
            "name must be a single safe path component (no separators, "
            f"NUL, or control characters); got {name!r}"
        )


def _validate_mode_and_lengths(cfg: RFdiffusionConfig) -> None:
    """Validate the mode / length / disulfide-pair contract."""
    if cfg.mode not in {"linear", "head_to_tail", "disulfide"}:
        raise ValueError(f"mode must be one of linear/head_to_tail/disulfide; got {cfg.mode!r}")
    if cfg.length_min < 1 or cfg.length_max < cfg.length_min:
        raise ValueError(f"length range invalid: min={cfg.length_min} max={cfg.length_max}")
    if cfg.mode == "disulfide" and not cfg.disulfide_pairs:
        raise ValueError("mode=disulfide requires at least one configured pair")
    if cfg.mode == "linear" and cfg.disulfide_pairs:
        raise ValueError("disulfide_pairs only valid when mode=disulfide or head_to_tail")


def _validate_s2_fields(cfg: RFdiffusionConfig) -> None:
    """Validate the S2 reproducibility fields (seed / checkpoint)."""
    if cfg.seed < 0:
        raise ValueError(f"seed must be ≥ 0; got {cfg.seed}")
    if not cfg.checkpoint:
        raise ValueError("checkpoint must be a non-empty model identifier")


def _validate_extra_keys(cfg: RFdiffusionConfig) -> None:
    """Reject ``extra`` entries that would corrupt the upstream contract.

    Fail closed, three ways:

    1. Canonical Hydra keys the runner emits itself
       (``inference.num_designs`` / ``inference.design_startnum`` /
       ``inference.deterministic`` / ``contigmap.contigs``, ...) may
       not be overridden via ``extra`` — a conflict raises
       ``ValueError`` instead of a last-write-wins dict merge.
    2. Keys that do not exist upstream (``inference.seed``) raise a
       clear ``ValueError`` rather than silently forwarding a key the
       stock Hydra schema cannot parse.
    3. Non-string keys raise ``ValueError`` — the CLI layer coerces
       every ``extra`` value to ``str``, so a non-string *key* would
       produce an unusable flag.
    """
    for key in cfg.extra:
        # Reason: ``extra`` is annotated Mapping[str, Any] but callers can
        # still pass non-str keys at runtime (the annotation is advisory);
        # the CLI layer would crash later with an opaque AttributeError, so
        # reject the config here with a clear error.
        if not isinstance(key, str):  # type: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"extra keys must be strings; got {type(key).__name__}")
        if key in UNSUPPORTED_UPSTREAM_KEYS:
            raise ValueError(
                f"extra key {key!r} is not supported by upstream RFdiffusion; "
                f"set RFdiffusionConfig.seed instead (forwarded as "
                f"inference.design_startnum when deterministic=True)"
            )
        if key in RESERVED_CANONICAL_KEYS:
            raise ValueError(f"extra cannot override reserved canonical keys: {key}")
