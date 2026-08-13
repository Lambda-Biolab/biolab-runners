"""Configuration for a Rosetta relax / scoring run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

__all__ = ["ConstrainedRelaxOptions", "PreparationMode", "RosettaConfig"]

# ``preparation_mode`` mirrors the upstream ``rosetta_scripts`` protocol
# variable convention: the protocol XML gates behaviour on
# ``%%prep_mode%%`` and the runner passes the value through
# ``-parser:script_vars prep_mode=<mode>``. Using a Literal keeps the
# contract auditable: a typo (e.g. ``"Linear"``) raises mypy/ruff at the
# call site rather than silently shipping the wrong flag.
PreparationMode = Literal["linear", "cyclic"]


def _empty_str_tuple() -> tuple[str, ...]:
    return ()


@dataclass(frozen=True)
class ConstrainedRelaxOptions:
    """Structured knobs for constrained FastRelax.

    These map to ``-parser:script_vars key=value`` pairs recognized by
    the upstream ``relax`` Movers when the protocol XML gates
    behaviour on the corresponding ``%%key%%`` variables. Each field
    is independent — set only the ones the protocol actually uses.

    Booleans emit ``1`` / ``0`` (Rosetta's flag convention) and only
    appear in the resulting command when explicitly set; ``None`` keeps
    the flag out entirely so callers can compose a subset.
    """

    constrain_to_start_coords: bool | None = None
    ramp_constraints: bool | None = None
    coord_constrain_sidechains: bool | None = None
    relax_cycles: int | None = None
    bb_min_only: bool | None = None


@dataclass(frozen=True)
class RosettaConfig:
    """Per-invocation configuration for the Rosetta runner.

    ``license_acknowledged`` must be set to True; the runner refuses
    to invoke the binary otherwise. The Python-side guard is
    informational — the upstream binary is the source of truth.

    The structured options ``preparation_mode`` and
    ``constrained_relax`` are translated to ``-parser:script_vars``
    pairs by :func:`biolab_runners.rosetta.runner._config_to_cli`; the
    consumer protocol XML is expected to gate behaviour on the
    matching ``%%variable%%`` tokens. Leaving these unset emits no
    flag at all (the script sees only what it inherits from the XML
    defaults).
    """

    name: str = "rosetta-relax"
    script_file: str = ""
    input_pdb: str = ""
    output_dir: str = ""
    nstruct: int = 1
    preparation_mode: PreparationMode | None = None
    constrained_relax: ConstrainedRelaxOptions | None = None
    extra_flags: tuple[str, ...] = field(default_factory=_empty_str_tuple)
    extra: Mapping[str, Any] = field(default_factory=lambda: {})
    license_acknowledged: bool = False

    def __post_init__(self) -> None:
        """Validate the license flag, required inputs, and structured options."""
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
        if self.preparation_mode is not None and self.preparation_mode not in (
            "linear",
            "cyclic",
        ):
            raise ValueError(
                "preparation_mode must be one of {'linear', 'cyclic'}; "
                f"got {self.preparation_mode!r}"
            )
