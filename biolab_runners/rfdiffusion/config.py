"""Configuration for an RFdiffusion backbone generation.

Mirrors the upstream CLI flags that the Activin-E pipeline actually
uses. Every field has a conservative default so a bare-minimum
campaign still produces a valid result.

The runner accepts the upstream ``contigmap.contigs`` syntax
(e.g. ``"12-18 A3-117/0 50-50"``) so callers familiar with
RFdiffusion can use the documentation directly.
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
    extra: Mapping[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """Validate mode + disulfide-pair consistency."""
        if self.mode not in {"linear", "head_to_tail", "disulfide"}:
            raise ValueError(
                f"mode must be one of linear/head_to_tail/disulfide; got {self.mode!r}"
            )
        if self.length_min < 1 or self.length_max < self.length_min:
            raise ValueError(f"length range invalid: min={self.length_min} max={self.length_max}")
        if self.mode == "disulfide" and not self.disulfide_pairs:
            raise ValueError("mode=disulfide requires at least one configured pair")
        if self.mode == "linear" and self.disulfide_pairs:
            raise ValueError("disulfide_pairs only valid when mode=disulfide or head_to_tail")
