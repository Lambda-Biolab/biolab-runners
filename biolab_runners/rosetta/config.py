"""Configuration for a Rosetta relax / scoring run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["RosettaConfig"]


def _empty_str_tuple() -> tuple[str, ...]:
    return ()


@dataclass(frozen=True)
class RosettaConfig:
    """Per-invocation configuration for the Rosetta runner.

    ``license_acknowledged`` must be set to True; the runner refuses
    to invoke the binary otherwise. The Python-side guard is
    informational — the upstream binary is the source of truth.
    """

    name: str = "rosetta-relax"
    script_file: str = ""
    input_pdb: str = ""
    output_dir: str = ""
    nstruct: int = 1
    extra_flags: tuple[str, ...] = field(default_factory=_empty_str_tuple)
    extra: Mapping[str, Any] = field(default_factory=lambda: {})
    license_acknowledged: bool = False

    def __post_init__(self) -> None:
        """Validate the license flag and required inputs."""
        if not self.license_acknowledged:
            raise ValueError(
                "Rosetta requires an explicit license acknowledgement: "
                "set license_acknowledged=True on RosettaConfig"
            )
        if not self.script_file:
            raise ValueError("RosettaConfig.script_file is required")
        if not self.input_pdb:
            raise ValueError("RosettaConfig.input_pdb is required")
        if self.nstruct < 1:
            raise ValueError(f"nstruct must be >= 1; got {self.nstruct}")
