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


def parse_residue_decomposition(path: Path) -> tuple[GmxMMPBSARecord, ...]:
    r"""Parse a gmx_MMPBSA per-residue decomposition file.

    Standard output format:
        ``<chain>:<resname><resid>\t<vdw>\t<ele>\t<pol>\t<npl>\t<total>``

    The first column is the chain:residue identifier; the next five
    columns are per-energy-term float values.

    Records with malformed numbers are dropped (the consumer
    treats empty results the same as ``unsupported``).
    """
    if not path.exists():
        return ()
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        logger.debug("could not read %s: %s", path, exc)
        return ()
    records: list[GmxMMPBSARecord] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 6:
            continue
        first = tokens[0]
        if ":" in first:
            chain_token, residue_token = first.split(":", 1)
            energy_tokens = tokens[1:]
        else:
            chain_token = ""
            residue_token = first
            energy_tokens = tokens[1:]
        if len(energy_tokens) < 5:
            continue
        values = tuple(_parse_float(t) for t in energy_tokens[:5])
        if any(v is None for v in values):
            continue
        vdw = values[0] or 0.0
        ele = values[1] or 0.0
        pol = values[2] or 0.0
        npl = values[3] or 0.0
        tot = values[4] or 0.0
        records.append(
            GmxMMPBSARecord(
                residue_label=residue_token,
                chain=chain_token,
                vdw_A=vdw,
                electrostatic_A=ele,
                polar_solvation_A=pol,
                non_polar_solvation_A=npl,
                total_A=tot,
            )
        )
    return tuple(records)
