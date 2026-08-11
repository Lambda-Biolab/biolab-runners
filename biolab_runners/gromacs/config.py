"""Configuration for a GROMACS MD run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["GromacsConfig"]


def _empty_str_tuple() -> tuple[str, ...]:
    return ()


@dataclass(frozen=True)
class GromacsConfig:
    """Per-invocation configuration for a GROMACS MD run."""

    name: str = "gromacs-md"
    structure_file: str = ""
    topology_file: str = ""
    output_dir: str = ""
    tpr_basename: str = "topol"
    nsteps: int = 1000000  # 2 ns at 2 fs
    integrator: str = "md"
    temperature: float = 310.0
    pressure: float = 1.0
    timestep_fs: float = 2.0
    extra_mdrun_flags: tuple[str, ...] = field(default_factory=_empty_str_tuple)
    extra: Mapping[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """Validate required paths and parameter ranges."""
        if not self.structure_file:
            raise ValueError("GromacsConfig.structure_file is required")
        if not self.topology_file:
            raise ValueError("GromacsConfig.topology_file is required")
        if self.nsteps < 1:
            raise ValueError(f"nsteps must be >= 1; got {self.nsteps}")
        if self.timestep_fs <= 0:
            raise ValueError(f"timestep_fs must be positive; got {self.timestep_fs}")
