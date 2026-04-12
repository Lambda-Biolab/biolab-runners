"""Utility functions for Boltz-2 structure prediction."""

from __future__ import annotations

import json
import logging
import subprocess
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

from biolab_runners.boltz2.config import ConfidenceScores

logger = logging.getLogger(__name__)


def boltz_available(binary: str = "boltz") -> bool:
    """Check if the Boltz-2 CLI is installed and accessible.

    Args:
        binary: Name or path of the boltz CLI binary.

    Returns:
        True if the boltz CLI responds to --help.
    """
    try:
        result = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def write_boltz_yaml(
    sequences: dict[str, str],
    output_path: Path,
    *,
    msa_paths: dict[str, str] | None = None,
    pocket_contacts: list[tuple[str, int]] | None = None,
    binder_chain: str = "B",
) -> Path:
    """Write a Boltz-2 v2 YAML input file.

    Boltz-2 v2 uses YAML as the primary input format, supporting multi-chain
    complexes, pre-computed MSAs, and pocket constraints.

    Args:
        sequences: Dict mapping chain_id to amino acid sequence.
        output_path: Output YAML path.
        msa_paths: Optional dict mapping chain_id to pre-computed MSA CSV path.
            When provided, Boltz-2 skips the ColabFold server call for that chain.
        pocket_contacts: Optional list of (chain_id, residue_number) tuples
            specifying receptor residues the binder must contact.
        binder_chain: Chain ID of the binder/peptide (default "B").

    Returns:
        Path to the written YAML file.
    """
    msa_paths = msa_paths or {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["version: 1", "sequences:"]
    for chain_id, seq in sequences.items():
        lines.append("  - protein:")
        lines.append(f"      id: {chain_id}")
        lines.append(f"      sequence: {seq}")
        if chain_id in msa_paths:
            lines.append(f"      msa: {msa_paths[chain_id]}")

    if pocket_contacts:
        lines.append("constraints:")
        lines.append("  - pocket:")
        lines.append(f"      binder: {binder_chain}")
        contacts_str = ", ".join(f"[{c}, {r}]" for c, r in pocket_contacts)
        lines.append(f"      contacts: [{contacts_str}]")

    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def is_boltz_output_complete(output_dir: Path) -> bool:
    """Check if a Boltz-2 output directory has actual prediction files.

    A "hollow" directory (e.g. from a failed rsync) may have only
    ``processed/manifest.json`` but no PDB structures or confidence JSONs.

    Args:
        output_dir: Boltz-2 output directory to validate.

    Returns:
        True if at least one structure file and one confidence JSON exist.
    """
    has_structure = bool(
        list(output_dir.glob("**/predictions/**/*.pdb"))
        or list(output_dir.glob("**/*.pdb"))
        or list(output_dir.glob("**/*.cif"))
    )
    has_confidence = bool(list(output_dir.glob("**/confidence_*.json")))
    return has_structure and has_confidence


def parse_boltz_output(output_dir: Path) -> tuple[str, ConfidenceScores]:
    """Parse Boltz-2 output for structure path and confidence scores.

    Boltz-2 v2 outputs:
    - ``boltz_results_<name>/predictions/<name>/<name>_model_0.pdb``
    - ``confidence_<name>_model_0.json``

    Args:
        output_dir: Boltz-2 output directory.

    Returns:
        Tuple of (structure_path, ConfidenceScores).
    """
    structure_path = ""
    confidence = ConfidenceScores()

    for pattern in ["**/predictions/**/*.pdb", "**/*.pdb", "**/*.cif"]:
        matches = sorted(output_dir.glob(pattern))
        if matches:
            structure_path = str(matches[0])
            break

    for conf_file in sorted(output_dir.glob("**/confidence_*.json")):
        try:
            data = json.loads(conf_file.read_text())

            confidence.ptm = float(data.get("ptm", 0))
            confidence.iptm = float(data.get("iptm", data.get("protein_iptm", 0)))
            confidence.ranking_score = float(data.get("confidence_score", confidence.iptm))

            # Boltz-2 v2 reports complex_plddt on 0-1 scale; quality gates
            # use 0-100. Auto-detect and rescale. Zero means missing data.
            raw_plddt = float(data.get("complex_plddt", 0))
            if 0 < raw_plddt <= 1.0:
                confidence.plddt_mean = raw_plddt * 100.0
            else:
                confidence.plddt_mean = raw_plddt

            if "binding_affinity" in data:
                confidence.binding_affinity = float(data["binding_affinity"])
            if "complex_iplddt" in data:
                confidence.complex_iplddt = float(data["complex_iplddt"])
            if "complex_ipde" in data:
                confidence.complex_ipde = float(data["complex_ipde"])
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse Boltz-2 confidence: %s", exc)
        break

    return structure_path, confidence
