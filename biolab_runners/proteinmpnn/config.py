"""Configuration for a ProteinMPNN sequence design run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["ProteinMPNNConfig"]


def _empty_int_tuple() -> tuple[int, ...]:
    return ()


@dataclass(frozen=True)
class ProteinMPNNConfig:
    """Per-invocation configuration for the ProteinMPNN runner.

    Defaults produce four canonical L-amino-acid sequences per PDB
    input via the upstream ``v_48_020`` checkpoint (one of
    ``v_48_002``, ``v_48_010``, ``v_48_020``, ``v_48_030`` — these are
    the upstream ``--model_name`` values; they are *checkpoint
    prefixes*, not folder names). The upstream script joins
    ``model_folder_path + model_name + ".pt"`` to resolve the
    checkpoint file, so passing the folder name (e.g.
    ``"vanilla_model_weights"``) instead of the checkpoint prefix
    silently breaks loading — the historical default was a folder
    name, not a checkpoint prefix.

    Cyclic and D-residue behaviour is not part of ProteinMPNN's
    vocabulary — the runner exposes positions the consumer can pin
    to Cys / D-residues, but the actual conversion happens in the
    downstream ``biolam_tools.chem_001`` stage.
    """

    name: str = "sequence"
    task_count: int = 4
    temperature: float = 0.1
    seed: int = 0
    model_name: str = "v_48_020"
    ca_only: bool = False
    fixed_positions: tuple[int, ...] = field(default_factory=_empty_int_tuple)
    omit_aa: str = ""  # e.g. "CDF" to omit these amino acids
    extra: Mapping[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """Validate task_count, temperature, and fixed_positions."""
        if self.task_count < 1:
            raise ValueError(f"task_count must be ≥ 1; got {self.task_count}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive; got {self.temperature}")
        if self.fixed_positions and any(p < 1 for p in self.fixed_positions):
            raise ValueError("fixed_positions are 1-indexed; must be ≥ 1")
