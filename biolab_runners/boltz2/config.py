"""Configuration models for Boltz-2 structure prediction."""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from enum import StrEnum

if typing.TYPE_CHECKING:
    from pathlib import Path


class QualityGate(StrEnum):
    """Quality gate verdicts for predicted structures."""

    PASS = "PASS"  # noqa: S105
    CONDITIONAL = "CONDITIONAL"
    FAIL = "FAIL"
    PENDING = "PENDING"


@dataclass(frozen=True)
class PocketConstraint:
    """A receptor residue the binder must contact.

    Used for spatially constrained predictions (Boltz-2 pocket constraints).
    """

    chain_id: str
    residue_number: int


@dataclass
class Boltz2Config:
    """Configuration for a Boltz-2 structure prediction run.

    Attributes:
        accelerator: Device to use (gpu, cpu, tpu).
        num_workers: Dataloader workers for Boltz-2.
        recycling_steps: Number of recycling iterations in the diffusion model.
        diffusion_samples: Number of diffusion samples (seeds) per prediction.
        use_msa_server: Use ColabFold MSA server for alignment.
        no_kernels: Disable cuequivariance custom kernels (use PyTorch fallback).
            Set to True if cuequivariance is not installed.
        use_potentials: Enable steering potentials to reduce steric clashes.
            Without this, Boltz-2 frequently produces structures with severe
            clashes. See the Boltz-2 docs for the underlying discussion.
        output_format: Output structure format (pdb or cif).
        timeout_seconds: Maximum time per prediction before abort.
        boltz_binary: Name or path of the boltz CLI binary.
    """

    accelerator: str = "gpu"
    num_workers: int = 2
    recycling_steps: int = 3
    diffusion_samples: int = 1
    use_msa_server: bool = True
    no_kernels: bool = True
    use_potentials: bool = True
    output_format: str = "pdb"
    timeout_seconds: int = 1800
    boltz_binary: str = "boltz"


# Quality gate thresholds (literature-standard ranges for ipTM / pLDDT)
IPTM_PASS = 0.7
IPTM_CONDITIONAL = 0.5
PLDDT_PASS = 70.0
PLDDT_CONDITIONAL = 50.0
MAX_CLASHES_PASS = 0
MAX_CLASHES_CONDITIONAL = 3
CONFIDENCE_SCORE_PASS = 0.7
CONFIDENCE_SCORE_CONDITIONAL = 0.5


@dataclass
class ConfidenceScores:
    """Unified confidence metrics from Boltz-2 prediction output.

    Attributes:
        ptm: Predicted TM-score (overall fold quality, 0-1).
        iptm: Interface pTM (complex interface quality, 0-1).
        ranking_score: Model's own ranking metric (confidence_score).
        plddt_mean: Mean predicted lDDT (0-100 scale).
        plddt_min: Minimum pLDDT (worst region).
        binding_affinity: Boltz-2 only: predicted dG (kcal/mol).
        clash_count: Total steric clashes in predicted structure.
        clash_severe_count: SEVERE clashes (<2.0 A) — physically impossible.
        complex_iplddt: Boltz-2 interface pLDDT (0-1 scale).
        complex_ipde: Interface predicted distance error (A, lower=better).
    """

    ptm: float = 0.0
    iptm: float = 0.0
    ranking_score: float | None = None
    plddt_mean: float = 0.0
    plddt_min: float = 0.0
    binding_affinity: float | None = None
    clash_count: int = 0
    clash_severe_count: int = 0
    complex_iplddt: float | None = None
    complex_ipde: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        d: dict[str, object] = {
            "ptm": round(self.ptm, 3),
            "iptm": round(self.iptm, 3),
            "ranking_score": round(self.ranking_score, 3)
            if self.ranking_score is not None
            else None,
            "plddt_mean": round(self.plddt_mean, 1),
            "plddt_min": round(self.plddt_min, 1),
            "binding_affinity_kcal_mol": self.binding_affinity,
            "clash_count": self.clash_count,
            "clash_severe_count": self.clash_severe_count,
        }
        if self.complex_iplddt is not None:
            d["complex_iplddt"] = round(self.complex_iplddt, 4)
        if self.complex_ipde is not None:
            d["complex_ipde"] = round(self.complex_ipde, 2)
        return d


@dataclass
class PredictionResult:
    """Output from a Boltz-2 structure prediction.

    Attributes:
        name: Job name / identifier.
        receptor_sequence: Receptor protein sequence.
        peptide_sequence: Peptide sequence (empty for monomer predictions).
        structure_path: Path to predicted PDB/CIF file.
        confidence_json_path: Path to per-residue confidence JSON.
        confidence: Parsed confidence scores.
        quality_gate: Quality gate verdict.
        gate_reasons: Human-readable reasons for the quality gate.
        runtime_seconds: Wall-clock time for the prediction.
        num_seeds: Number of diffusion samples used.
        error: Error message if prediction failed.
    """

    name: str = ""
    receptor_sequence: str = ""
    peptide_sequence: str = ""
    structure_path: str = ""
    confidence_json_path: str = ""
    confidence: ConfidenceScores = field(default_factory=ConfidenceScores)
    quality_gate: QualityGate = QualityGate.PENDING
    gate_reasons: list[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    num_seeds: int = 1
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "receptor_sequence_length": len(self.receptor_sequence),
            "peptide_sequence_length": len(self.peptide_sequence),
            "structure_path": self.structure_path,
            "confidence": self.confidence.to_dict(),
            "quality_gate": self.quality_gate.value,
            "gate_reasons": self.gate_reasons,
            "runtime_seconds": round(self.runtime_seconds, 1),
            "error": self.error,
        }

    def save(self, output_dir: Path) -> Path:
        """Save prediction result to JSON.

        Args:
            output_dir: Directory to write the result file.

        Returns:
            Path to the saved JSON file.
        """
        import json

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.name}_result.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path
