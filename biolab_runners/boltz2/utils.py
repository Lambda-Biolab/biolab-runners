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
    modifications: dict[str, list[dict[str, object]]] | None = None,
) -> Path:
    """Write a Boltz-2 v2 YAML input file.

    Boltz-2 v2 uses YAML as the primary input format, supporting multi-chain
    complexes, pre-computed MSAs, pocket constraints, and non-canonical
    residue modifications via the PDB CCD dictionary.

    Args:
        sequences: Dict mapping chain_id to amino acid sequence.
        output_path: Output YAML path.
        msa_paths: Optional dict mapping chain_id to pre-computed MSA CSV path.
            When provided, Boltz-2 skips the ColabFold server call for that chain.
        pocket_contacts: Optional list of (chain_id, residue_number) tuples
            specifying receptor residues the binder must contact.
        binder_chain: Chain ID of the binder/peptide (default "B").
        modifications: Optional dict mapping chain_id to a list of residue
            modifications. Each modification is a dict with keys
            ``position`` (1-indexed residue number) and ``ccd`` (PDB CCD
            three/four-letter code, e.g. ``"AIB"`` for α-aminoisobutyric
            acid). Boltz-2 replaces the sequence token at ``position-1``
            with the CCD residue during parsing, yielding the correct
            non-canonical heavy-atom set in the output structure. See
            ``boltz.data.parse.schema`` for the upstream contract.

    Returns:
        Path to the written YAML file.
    """
    msa_paths = msa_paths or {}
    modifications = modifications or {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["version: 1", "sequences:"]
    for chain_id, seq in sequences.items():
        lines.append("  - protein:")
        lines.append(f"      id: {chain_id}")
        lines.append(f"      sequence: {seq}")
        if chain_id in msa_paths:
            lines.append(f"      msa: {msa_paths[chain_id]}")
        chain_mods = modifications.get(chain_id)
        if chain_mods:
            lines.append("      modifications:")
            for mod in chain_mods:
                lines.append(f"        - position: {mod['position']}")
                lines.append(f"          ccd: {mod['ccd']}")

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


def _find_structure_file(output_dir: Path) -> str:
    """Return the first structure file under output_dir, or ""."""
    for pattern in ["**/predictions/**/*.pdb", "**/*.pdb", "**/*.cif"]:
        matches = sorted(output_dir.glob(pattern))
        if matches:
            return str(matches[0])
    return ""


def _populate_confidence_from_data(confidence: ConfidenceScores, data: dict) -> None:
    """Populate a ConfidenceScores from a parsed confidence JSON dict."""
    confidence.ptm = float(data.get("ptm", 0))
    confidence.iptm = float(data.get("iptm", data.get("protein_iptm", 0)))
    confidence.ranking_score = float(data.get("confidence_score", confidence.iptm))

    # Boltz-2 v2 reports complex_plddt on 0-1 scale; quality gates use 0-100.
    raw_plddt = float(data.get("complex_plddt", 0))
    confidence.plddt_mean = raw_plddt * 100.0 if 0 < raw_plddt <= 1.0 else raw_plddt

    if "binding_affinity" in data:
        confidence.binding_affinity = float(data["binding_affinity"])
    if "complex_iplddt" in data:
        confidence.complex_iplddt = float(data["complex_iplddt"])
    if "complex_ipde" in data:
        confidence.complex_ipde = float(data["complex_ipde"])


def _parse_confidence_file(output_dir: Path, confidence: ConfidenceScores) -> None:
    """Load the first confidence JSON under output_dir into confidence."""
    for conf_file in sorted(output_dir.glob("**/confidence_*.json")):
        try:
            data = json.loads(conf_file.read_text())
            _populate_confidence_from_data(confidence, data)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse Boltz-2 confidence: %s", exc)
        return


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
    confidence = ConfidenceScores()
    structure_path = _find_structure_file(output_dir)
    _parse_confidence_file(output_dir, confidence)
    return structure_path, confidence
