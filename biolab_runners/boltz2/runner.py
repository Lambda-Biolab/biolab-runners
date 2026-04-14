"""Boltz-2 structure prediction runner.

Boltz-2 (MIT/Recursion) runs locally on RTX 4090 (24 GB VRAM).
It produces structure predictions with confidence scores AND binding
affinity estimates (unique among structure predictors).

Key advantages:
- No job limits, no queue wait
- Commercially usable (Apache 2.0 license)
- Lower clash rate with steering potentials (~0% vs 30-60% without)
- Binding affinity prediction (Pearson r=0.62)
- ~4 minutes for 100-500 residue complexes on RTX 4090

Requires: pip install boltz

Usage::

    from biolab_runners.boltz2 import Boltz2Runner, Boltz2Config

    runner = Boltz2Runner(Boltz2Config(accelerator="gpu"))
    result = runner.predict_complex(
        receptor_sequence="MVKLTAEG...",
        peptide_sequence="RWKLFKKIEK",
        name="GtfB_PEP001",
        output_dir=Path("results/predictions"),
    )
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from biolab_runners.boltz2.config import (
    CONFIDENCE_SCORE_CONDITIONAL,
    CONFIDENCE_SCORE_PASS,
    IPTM_CONDITIONAL,
    IPTM_PASS,
    MAX_CLASHES_CONDITIONAL,
    MAX_CLASHES_PASS,
    PLDDT_CONDITIONAL,
    PLDDT_PASS,
    Boltz2Config,
    ConfidenceScores,
    PredictionResult,
    QualityGate,
)
from biolab_runners.boltz2.utils import (
    boltz_available,
    is_boltz_output_complete,
    parse_boltz_output,
    write_boltz_yaml,
)

logger = logging.getLogger(__name__)


_GATE_SEVERITY = {
    QualityGate.PASS: 0,
    QualityGate.CONDITIONAL: 1,
    QualityGate.FAIL: 2,
    QualityGate.PENDING: 0,
}


def _max_gate(a: QualityGate, b: QualityGate) -> QualityGate:
    """Return the more severe of two gate verdicts."""
    return a if _GATE_SEVERITY[a] >= _GATE_SEVERITY[b] else b


def _evaluate_clashes(conf: ConfidenceScores) -> tuple[QualityGate, list[str]]:
    """Clash check — SEVERE clashes are physically impossible."""
    if conf.clash_severe_count > 0:
        return QualityGate.FAIL, [
            f"SEVERE CLASH: {conf.clash_severe_count} atom pairs with sub-2.0A overlap "
            f"— structure physically impossible"
        ]
    if conf.clash_count > MAX_CLASHES_CONDITIONAL:
        return QualityGate.FAIL, [
            f"CLASH: {conf.clash_count} steric clashes in predicted structure "
            f"(max {MAX_CLASHES_CONDITIONAL} for CONDITIONAL)"
        ]
    if conf.clash_count > MAX_CLASHES_PASS:
        return QualityGate.CONDITIONAL, [
            f"CLASH WARNING: {conf.clash_count} mild clashes (may resolve during MD)"
        ]
    return QualityGate.PASS, []


def _evaluate_iptm(conf: ConfidenceScores) -> tuple[QualityGate, list[str]]:
    """Interface confidence check (complexes only)."""
    if conf.iptm <= 0:
        return QualityGate.PASS, []
    if conf.iptm < IPTM_CONDITIONAL:
        return QualityGate.FAIL, [
            f"LOW INTERFACE CONFIDENCE: ipTM={conf.iptm:.3f} < {IPTM_CONDITIONAL} threshold"
        ]
    if conf.iptm < IPTM_PASS:
        return QualityGate.CONDITIONAL, [
            f"MODERATE INTERFACE: ipTM={conf.iptm:.3f} (threshold for PASS: {IPTM_PASS})"
        ]
    return QualityGate.PASS, []


def _evaluate_plddt(conf: ConfidenceScores) -> tuple[QualityGate, list[str]]:
    """Per-residue confidence check."""
    if conf.plddt_mean <= 0:
        return QualityGate.PASS, []
    if conf.plddt_mean < PLDDT_CONDITIONAL:
        return QualityGate.FAIL, [
            f"LOW CONFIDENCE: mean pLDDT={conf.plddt_mean:.1f} < {PLDDT_CONDITIONAL}"
        ]
    if conf.plddt_mean < PLDDT_PASS:
        return QualityGate.CONDITIONAL, [
            f"MODERATE CONFIDENCE: mean pLDDT={conf.plddt_mean:.1f} (PASS: >={PLDDT_PASS})"
        ]
    return QualityGate.PASS, []


def _evaluate_ranking_score(conf: ConfidenceScores) -> tuple[QualityGate, list[str]]:
    """Boltz-2 composite confidence score check."""
    if conf.ranking_score is None:
        return QualityGate.PASS, []
    if conf.ranking_score < CONFIDENCE_SCORE_CONDITIONAL:
        return QualityGate.FAIL, [
            f"LOW CONFIDENCE SCORE: {conf.ranking_score:.3f} < {CONFIDENCE_SCORE_CONDITIONAL}"
        ]
    if conf.ranking_score < CONFIDENCE_SCORE_PASS:
        return QualityGate.CONDITIONAL, [
            f"MODERATE CONFIDENCE SCORE: {conf.ranking_score:.3f} (PASS: >={CONFIDENCE_SCORE_PASS})"
        ]
    return QualityGate.PASS, []


def apply_quality_gate(result: PredictionResult) -> PredictionResult:
    """Apply quality gates to a prediction result.

    Gates (in order of severity):
    1. Error -> FAIL
    2. No structure file -> FAIL
    3. SEVERE clashes (>0) -> FAIL
    4. Steric clashes > 3 -> FAIL
    5. Interface confidence (ipTM) < 0.5 -> FAIL
    6. Mean pLDDT < 50 -> FAIL
    7. Clashes 1-3 -> CONDITIONAL
    8. ipTM 0.5-0.7 -> CONDITIONAL
    9. pLDDT 50-70 -> CONDITIONAL
    10. All clear -> PASS

    Args:
        result: PredictionResult to evaluate.

    Returns:
        PredictionResult with quality_gate and gate_reasons populated.
    """
    if result.error:
        result.quality_gate = QualityGate.FAIL
        result.gate_reasons = [f"Prediction error: {result.error}"]
        return result

    if not result.structure_path:
        result.quality_gate = QualityGate.FAIL
        result.gate_reasons = ["No structure file produced"]
        return result

    conf = result.confidence
    gate = QualityGate.PASS
    reasons: list[str] = []
    for evaluator in (
        _evaluate_clashes,
        _evaluate_iptm,
        _evaluate_plddt,
        _evaluate_ranking_score,
    ):
        sub_gate, sub_reasons = evaluator(conf)
        gate = _max_gate(gate, sub_gate)
        reasons.extend(sub_reasons)

    if not reasons:
        reasons.append("All quality checks passed")

    result.quality_gate = gate
    result.gate_reasons = reasons
    return result


class Boltz2Runner:
    """Boltz-2 local structure prediction runner.

    Runs the ``boltz predict`` CLI on a local GPU. Handles input file
    generation, subprocess management, output parsing, quality gating,
    MSA caching, and idempotent re-runs.

    Args:
        config: Runner configuration. Defaults to sensible GPU settings.

    Example::

        runner = Boltz2Runner()
        result = runner.predict_complex(
            receptor_sequence="MVKLTAEG...",
            peptide_sequence="RWKLFKKIEK",
            name="complex_01",
        )
        print(result.quality_gate)  # PASS / CONDITIONAL / FAIL
    """

    def __init__(self, config: Boltz2Config | None = None) -> None:
        self.config = config or Boltz2Config()
        self._msa_cache: dict[str, Path] = {}

    def is_available(self) -> bool:
        """Check if the Boltz-2 CLI is installed and accessible."""
        return boltz_available(self.config.boltz_binary)

    def _cache_receptor_msa(self, receptor_sequence: str, boltz_output: Path) -> None:
        """Cache receptor MSA CSV from a completed prediction for reuse.

        Boltz-2 writes receptor MSA to
        ``{out}/boltz_results_*/msa/{target_id}_0.csv``. This is constant
        for all predictions against the same receptor — caching avoids
        redundant ColabFold server calls (~2-5 min each).
        """
        if receptor_sequence in self._msa_cache:
            return
        for csv in sorted(boltz_output.glob("boltz_results_*/msa/*_0.csv")):
            if csv.stat().st_size > 100:
                cache_dir = boltz_output.parent.parent.parent / ".msa_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                import shutil

                cached = cache_dir / "receptor_msa.csv"
                shutil.copy2(csv, cached)
                self._msa_cache[receptor_sequence] = cached
                logger.info(
                    "Cached receptor MSA (%d KB) for reuse",
                    csv.stat().st_size // 1024,
                )
                return

    def _load_cached_prediction(
        self, boltz_output: Path, name: str, result: PredictionResult, label: str
    ) -> bool:
        """Populate result from a completed prediction on disk. Returns True on cache hit."""
        if not (boltz_output.exists() and is_boltz_output_complete(boltz_output)):
            return False
        existing_structure, existing_confidence = parse_boltz_output(boltz_output)
        if not existing_structure:
            return False
        logger.info(
            "Skipping Boltz-2 %s for %s — prediction already exists at %s. "
            "Use force=True to re-run.",
            label,
            name,
            existing_structure,
        )
        result.structure_path = existing_structure
        result.confidence = existing_confidence
        return True

    def _resolve_msa_paths(self, receptor_sequence: str, output_dir: Path) -> dict[str, str]:
        """Resolve MSA paths for a receptor, consulting in-memory and on-disk caches."""
        msa_paths: dict[str, str] = {}
        cached_msa = self._msa_cache.get(receptor_sequence)
        if cached_msa is None:
            disk_cache = output_dir / ".msa_cache" / "receptor_msa.csv"
            if disk_cache.exists() and disk_cache.stat().st_size > 100:
                self._msa_cache[receptor_sequence] = disk_cache
                cached_msa = disk_cache
        if cached_msa and cached_msa.exists():
            msa_paths["A"] = str(cached_msa)
            msa_paths["B"] = "empty"
            logger.info("Using cached receptor MSA: %s", cached_msa)
        return msa_paths

    def _run_boltz_subprocess(
        self, cmd: list[str], result: PredictionResult, stderr_limit: int = 500
    ) -> bool:
        """Invoke the boltz CLI, populating result.error/runtime. Returns True on success."""
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
            result.runtime_seconds = time.time() - start
        except subprocess.TimeoutExpired:
            result.error = f"Boltz-2 timed out after {self.config.timeout_seconds} seconds"
            result.runtime_seconds = time.time() - start
            return False

        if proc.returncode != 0:
            result.error = (
                f"Boltz-2 exited with code {proc.returncode}: {proc.stderr[:stderr_limit]}"
            )
            logger.error(result.error)
            return False
        return True

    def predict_complex(
        self,
        receptor_sequence: str,
        peptide_sequence: str,
        *,
        name: str = "",
        output_dir: Path = Path("results/predictions"),
        num_seeds: int | None = None,
        force: bool = False,
        pocket_contacts: list[tuple[str, int]] | None = None,
        seed: int | None = None,
        dry_run: bool = False,
    ) -> PredictionResult:
        """Predict a peptide-protein complex structure with Boltz-2.

        Args:
            receptor_sequence: Receptor protein amino acid sequence.
            peptide_sequence: Peptide amino acid sequence.
            name: Job name (used for output filenames).
            output_dir: Output directory for prediction files.
            num_seeds: Number of diffusion samples. Defaults to
                ``config.diffusion_samples``.
            force: Re-run even if prediction output already exists.
            pocket_contacts: Optional list of (chain_id, residue_number) tuples
                for spatially constrained prediction.
            seed: Random seed for reproducible predictions.
            dry_run: Validate inputs and log the command without executing.

        Returns:
            PredictionResult with quality gate applied.
        """
        if not name:
            name = f"complex_{peptide_sequence[:6]}"

        seeds = num_seeds or self.config.diffusion_samples

        result = PredictionResult(
            name=name,
            receptor_sequence=receptor_sequence,
            peptide_sequence=peptide_sequence,
            num_seeds=seeds,
        )

        if dry_run:
            cmd = self._build_command(
                yaml_path=output_dir / name / "boltz2" / "input.yaml",
                boltz_output=output_dir / name / "boltz2" / "output",
                num_seeds=seeds,
                seed=seed,
            )
            logger.info(
                "[DRY-RUN] Would run: %s (receptor=%d aa, peptide=%d aa)",
                " ".join(cmd),
                len(receptor_sequence),
                len(peptide_sequence),
            )
            result.quality_gate = QualityGate.PENDING
            result.gate_reasons = ["Dry run — no prediction executed"]
            return result

        if not self.is_available():
            result.error = (
                f"Boltz-2 not installed. Install with: pip install boltz "
                f"(binary={self.config.boltz_binary!r})"
            )
            result.quality_gate = QualityGate.FAIL
            result.gate_reasons = [result.error]
            return result

        run_dir = output_dir / name / "boltz2"
        run_dir.mkdir(parents=True, exist_ok=True)
        boltz_output = run_dir / "output"

        if not force and self._load_cached_prediction(boltz_output, name, result, "complex"):
            return apply_quality_gate(result)

        yaml_path = run_dir / "input.yaml"
        msa_paths = self._resolve_msa_paths(receptor_sequence, output_dir)
        write_boltz_yaml(
            {"A": receptor_sequence, "B": peptide_sequence},
            yaml_path,
            msa_paths=msa_paths,
            pocket_contacts=pocket_contacts,
        )

        cmd = self._build_command(
            yaml_path=yaml_path,
            boltz_output=boltz_output,
            num_seeds=seeds,
            seed=seed,
        )
        logger.info("Running Boltz-2: %s", " ".join(cmd))

        if not self._run_boltz_subprocess(cmd, result):
            return apply_quality_gate(result)

        structure_path, confidence = parse_boltz_output(boltz_output)
        result.structure_path = structure_path
        result.confidence = confidence

        if structure_path:
            logger.info(
                "Boltz-2 prediction complete: %s (ipTM=%.3f, pTM=%.3f, %.1fs)",
                name,
                confidence.iptm,
                confidence.ptm,
                result.runtime_seconds,
            )
            self._cache_receptor_msa(receptor_sequence, boltz_output)
        else:
            result.error = "No structure file in Boltz-2 output"

        return apply_quality_gate(result)

    def _build_monomer_command(self, yaml_path: Path, boltz_output: Path) -> list[str]:
        """Build the boltz predict CLI command for a monomer run."""
        cfg = self.config
        cmd = [
            cfg.boltz_binary,
            "predict",
            str(yaml_path),
            "--out_dir",
            str(boltz_output),
            "--accelerator",
            cfg.accelerator,
            "--model",
            "boltz2",
            "--output_format",
            cfg.output_format,
        ]
        if cfg.use_msa_server:
            cmd.append("--use_msa_server")
        if cfg.no_kernels:
            cmd.append("--no_kernels")
        if cfg.use_potentials:
            cmd.append("--use_potentials")
        return cmd

    def predict_monomer(
        self,
        sequence: str,
        *,
        name: str = "",
        output_dir: Path = Path("results/predictions"),
        force: bool = False,
        dry_run: bool = False,
    ) -> PredictionResult:
        """Predict a single protein structure.

        Args:
            sequence: Protein amino acid sequence.
            name: Job name.
            output_dir: Output directory.
            force: Re-run even if prediction output already exists.
            dry_run: Validate inputs without executing.

        Returns:
            PredictionResult with quality gate applied.
        """
        if not name:
            name = f"monomer_{sequence[:6]}"

        result = PredictionResult(
            name=name,
            receptor_sequence=sequence,
        )

        if dry_run:
            logger.info(
                "[DRY-RUN] Would predict monomer %s (%d aa)",
                name,
                len(sequence),
            )
            result.quality_gate = QualityGate.PENDING
            result.gate_reasons = ["Dry run — no prediction executed"]
            return result

        if not self.is_available():
            result.error = "Boltz-2 not installed"
            return apply_quality_gate(result)

        run_dir = output_dir / name / "boltz2"
        run_dir.mkdir(parents=True, exist_ok=True)
        boltz_output = run_dir / "output"

        if not force and self._load_cached_prediction(boltz_output, name, result, "monomer"):
            return apply_quality_gate(result)

        yaml_path = write_boltz_yaml({"A": sequence}, run_dir / "input.yaml")
        cmd = self._build_monomer_command(yaml_path, boltz_output)

        if not self._run_boltz_subprocess(cmd, result, stderr_limit=300):
            return apply_quality_gate(result)

        structure_path, confidence = parse_boltz_output(boltz_output)
        result.structure_path = structure_path
        result.confidence = confidence
        return apply_quality_gate(result)

    def _build_command(
        self,
        yaml_path: Path,
        boltz_output: Path,
        num_seeds: int,
        seed: int | None,
    ) -> list[str]:
        """Build the boltz predict CLI command."""
        cfg = self.config
        cmd = [
            cfg.boltz_binary,
            "predict",
            str(yaml_path),
            "--out_dir",
            str(boltz_output),
            "--accelerator",
            cfg.accelerator,
            "--num_workers",
            str(cfg.num_workers),
            "--recycling_steps",
            str(cfg.recycling_steps),
            "--diffusion_samples",
            str(num_seeds),
            "--model",
            "boltz2",
            "--output_format",
            cfg.output_format,
        ]
        if cfg.use_msa_server:
            cmd.append("--use_msa_server")
        if cfg.no_kernels:
            cmd.append("--no_kernels")
        if cfg.use_potentials:
            cmd.append("--use_potentials")
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        return cmd
