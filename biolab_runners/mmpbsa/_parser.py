"""Internal helpers for the gmx_MMPBSA runner.

Splitting the parser into its own module keeps ``biolab_runners.mmpbsa``
focused on the runner class while :func:`parse_residue_decomposition``
stays importable for direct unit tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["GmxMMPBSARecord", "parse_residue_decomposition"]


@dataclass(frozen=True)
class GmxMMPBSARecord:
    """Per-energy-component breakdown for a single residue.

    Attributes:
        residue_label: ``"<resname><resid>"`` label, e.g. ``"LEU115"``.
        chain: Chain identifier (or empty string if not parsed).
        vdw_A: van-der-Waals contribution to binding (kcal/mol).
        electrostatic_A: Electrostatic contribution.
        polar_solvation_A: Polar solvation contribution.
        non_polar_solvation_A: Non-polar solvation contribution.
        total_A: Total per-residue contribution.
    """

    residue_label: str
    chain: str
    vdw_A: float  # noqa: N815 — units in field name
    electrostatic_A: float  # noqa: N815 — units in field name
    polar_solvation_A: float  # noqa: N815 — units in field name
    non_polar_solvation_A: float  # noqa: N815 — units in field name
    total_A: float  # noqa: N815 — units in field name

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary (Å² or kcal/mol per term)."""
        return {
            "residue": self.residue_label,
            "chain": self.chain,
            "per_energy_term_A": {
                "van_der_waals": self.vdw_A,
                "electrostatic": self.electrostatic_A,
                "polar_solvation": self.polar_solvation_A,
                "non_polar_solvation": self.non_polar_solvation_A,
                "total": self.total_A,
            },
        }


def _parse_float(token: str) -> float | None:
    """Parse a token as float; return None on malformed input."""
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _split_chain_residue(first_token: str) -> tuple[str, str]:
    """Split the leading ``chain:resname<resid>`` token.

    Returns ``("", token)`` when no chain prefix is present.
    """
    if ":" in first_token:
        chain_token, residue_token = first_token.split(":", 1)
        return chain_token, residue_token
    return "", first_token


def _parse_energy_tokens(energy_tokens: list[str]) -> tuple[float, ...] | None:
    """Parse the first five energy tokens; return None on malformed input.

    Tokens with parse failures, or fewer than 5 tokens, yield ``None``
    so the caller can drop the record (mirroring the malformed-line
    behavior of the upstream gmx_MMPBSA output consumer).
    """
    if len(energy_tokens) < 5:
        return None
    values = tuple(_parse_float(t) for t in energy_tokens[:5])
    if any(v is None for v in values):
        return None
    return tuple(v or 0.0 for v in values)


def _is_skippable_line(line: str) -> bool:
    """Return True for blank or comment-prefixed lines."""
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _build_record(
    chain_token: str,
    residue_token: str,
    values: tuple[float, ...],
) -> GmxMMPBSARecord:
    """Build a :class:`GmxMMPBSARecord` from chain + residue + energy values.

    The five values are interpreted in order as van-der-Waals,
    electrostatic, polar solvation, non-polar solvation, and total
    (kcal/mol each).
    """
    return GmxMMPBSARecord(
        residue_label=residue_token,
        chain=chain_token,
        vdw_A=values[0],
        electrostatic_A=values[1],
        polar_solvation_A=values[2],
        non_polar_solvation_A=values[3],
        total_A=values[4],
    )


def _read_lines(path: Path) -> list[str] | None:
    """Read the file's lines; return ``None`` on read failure or absent file."""
    if not path.exists():
        return None
    try:
        return path.read_text().splitlines()
    except OSError as exc:
        logger.debug("could not read %s: %s", path, exc)
        return None


def parse_residue_decomposition(path: Path) -> tuple[GmxMMPBSARecord, ...]:
    r"""Parse a gmx_MMPBSA per-residue decomposition file.

    Standard output format:
        ``<chain>:<resname><resid>\t<vdw>\t<ele>\t<pol>\t<npl>\t<total>``

    The first column is the chain:residue identifier; the next five
    columns are per-energy-term float values.

    Records with malformed numbers are dropped (the consumer
    treats empty results the same as ``unsupported``).
    """
    lines = _read_lines(path)
    if lines is None:
        return ()
    records: list[GmxMMPBSARecord] = []
    for line in lines:
        if _is_skippable_line(line):
            continue
        tokens = line.split()
        if len(tokens) < 6:
            continue
        chain_token, residue_token = _split_chain_residue(tokens[0])
        values = _parse_energy_tokens(tokens[1:])
        if values is None:
            continue
        records.append(_build_record(chain_token, residue_token, values))
    return tuple(records)
