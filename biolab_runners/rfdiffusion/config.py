"""Configuration for an RFdiffusion backbone generation.

Mirrors the upstream CLI flags that the Activin-E pipeline actually
uses. Every field has a conservative default so a bare-minimum
campaign still produces a valid result.

The runner accepts the upstream ``contigmap.contigs`` syntax
(e.g. ``"12-18 A3-117/0 50-50"``) so callers familiar with
RFdiffusion can use the documentation directly.

S2 reproducibility fields (per the Activin-E reproducibility plan):

* ``seed`` — the base seed for the single-stream RNG. The runner
  records ``base_seed`` and ``task_count`` in the provenance
  manifest so downstream consumers can reconstruct the RNG intent
  without re-running the runner. ``inference.seed`` is **not**
  forwarded to upstream — see the runner for the rationale.
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

__all__ = ["ContigMap", "RFdiffusionConfig"]


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
        """Validate mode + disulfide-pair consistency + S2 fields."""
        _validate_mode_and_lengths(self)
        _validate_s2_fields(self)


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
