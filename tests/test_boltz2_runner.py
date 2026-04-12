"""Tests for Boltz2Runner and related utilities."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from biolab_runners.boltz2.config import (
    Boltz2Config,
    ConfidenceScores,
    PredictionResult,
    QualityGate,
)
from biolab_runners.boltz2.runner import Boltz2Runner, apply_quality_gate
from biolab_runners.boltz2.utils import (
    is_boltz_output_complete,
    parse_boltz_output,
    write_boltz_yaml,
)

# ---------------------------------------------------------------------------
# Quality gate tests
# ---------------------------------------------------------------------------


class TestQualityGate:
    """Tests for the quality gate logic."""

    def test_clean_prediction_passes(self) -> None:
        result = PredictionResult(
            name="test",
            structure_path="/tmp/test.pdb",
            confidence=ConfidenceScores(iptm=0.85, ptm=0.80, plddt_mean=82.0, clash_count=0),
        )
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.PASS
        assert "All quality checks passed" in gated.gate_reasons

    def test_error_fails(self) -> None:
        result = PredictionResult(name="test", error="Boltz-2 crashed")
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.FAIL
        assert "Prediction error" in gated.gate_reasons[0]

    def test_no_structure_fails(self) -> None:
        result = PredictionResult(name="test", structure_path="")
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.FAIL

    def test_severe_clashes_fail(self) -> None:
        result = PredictionResult(
            name="test",
            structure_path="/tmp/test.pdb",
            confidence=ConfidenceScores(iptm=0.85, plddt_mean=80.0, clash_severe_count=1),
        )
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.FAIL
        assert any("SEVERE" in r for r in gated.gate_reasons)

    def test_many_clashes_fail(self) -> None:
        result = PredictionResult(
            name="test",
            structure_path="/tmp/test.pdb",
            confidence=ConfidenceScores(iptm=0.85, plddt_mean=80.0, clash_count=5),
        )
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.FAIL

    def test_mild_clashes_conditional(self) -> None:
        result = PredictionResult(
            name="test",
            structure_path="/tmp/test.pdb",
            confidence=ConfidenceScores(iptm=0.85, plddt_mean=80.0, clash_count=2),
        )
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.CONDITIONAL

    def test_low_iptm_fails(self) -> None:
        result = PredictionResult(
            name="test",
            structure_path="/tmp/test.pdb",
            confidence=ConfidenceScores(iptm=0.3, plddt_mean=80.0),
        )
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.FAIL

    def test_moderate_iptm_conditional(self) -> None:
        result = PredictionResult(
            name="test",
            structure_path="/tmp/test.pdb",
            confidence=ConfidenceScores(iptm=0.6, plddt_mean=80.0),
        )
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.CONDITIONAL

    def test_low_plddt_fails(self) -> None:
        result = PredictionResult(
            name="test",
            structure_path="/tmp/test.pdb",
            confidence=ConfidenceScores(iptm=0.85, plddt_mean=40.0),
        )
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.FAIL

    def test_moderate_plddt_conditional(self) -> None:
        result = PredictionResult(
            name="test",
            structure_path="/tmp/test.pdb",
            confidence=ConfidenceScores(iptm=0.85, plddt_mean=60.0),
        )
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.CONDITIONAL

    def test_low_ranking_score_fails(self) -> None:
        result = PredictionResult(
            name="test",
            structure_path="/tmp/test.pdb",
            confidence=ConfidenceScores(iptm=0.85, plddt_mean=80.0, ranking_score=0.3),
        )
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.FAIL

    def test_moderate_ranking_score_conditional(self) -> None:
        result = PredictionResult(
            name="test",
            structure_path="/tmp/test.pdb",
            confidence=ConfidenceScores(iptm=0.85, plddt_mean=80.0, ranking_score=0.6),
        )
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.CONDITIONAL

    def test_zero_iptm_not_checked(self) -> None:
        """ipTM=0 means not populated — should not trigger a gate."""
        result = PredictionResult(
            name="test",
            structure_path="/tmp/test.pdb",
            confidence=ConfidenceScores(iptm=0.0, plddt_mean=80.0),
        )
        gated = apply_quality_gate(result)
        assert gated.quality_gate == QualityGate.PASS


# ---------------------------------------------------------------------------
# YAML writer tests
# ---------------------------------------------------------------------------


class TestWriteBoltzYaml:
    """Tests for YAML input file generation."""

    def test_basic_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "input.yaml"
        result = write_boltz_yaml(
            {"A": "MVKLTAEG", "B": "RWKLFKK"},
            yaml_path,
        )
        assert result == yaml_path
        content = yaml_path.read_text()
        assert "version: 1" in content
        assert "MVKLTAEG" in content
        assert "RWKLFKK" in content
        assert "constraints" not in content

    def test_yaml_with_msa(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "input.yaml"
        write_boltz_yaml(
            {"A": "MVKL", "B": "RWKL"},
            yaml_path,
            msa_paths={"A": "/data/msa.csv", "B": "empty"},
        )
        content = yaml_path.read_text()
        assert "msa: /data/msa.csv" in content
        assert "msa: empty" in content

    def test_yaml_with_pocket_contacts(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "input.yaml"
        write_boltz_yaml(
            {"A": "MVKL", "B": "RWKL"},
            yaml_path,
            pocket_contacts=[("A", 123), ("A", 456)],
        )
        content = yaml_path.read_text()
        assert "constraints:" in content
        assert "pocket:" in content
        assert "binder: B" in content
        assert "[A, 123]" in content
        assert "[A, 456]" in content


# ---------------------------------------------------------------------------
# Output validation tests
# ---------------------------------------------------------------------------


class TestOutputValidation:
    """Tests for Boltz-2 output parsing and validation."""

    def test_hollow_output_not_complete(self, tmp_path: Path) -> None:
        """Empty directory should not be considered complete."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        assert not is_boltz_output_complete(output_dir)

    def test_complete_output(self, tmp_path: Path) -> None:
        """Directory with PDB and confidence JSON is complete."""
        output_dir = tmp_path / "output"
        pred_dir = output_dir / "boltz_results_test" / "predictions" / "test"
        pred_dir.mkdir(parents=True)
        (pred_dir / "test_model_0.pdb").write_text("ATOM  1  CA  ALA A   1")
        (output_dir / "confidence_test_model_0.json").write_text("{}")
        assert is_boltz_output_complete(output_dir)

    def test_parse_boltz_output_confidence(self, tmp_path: Path) -> None:
        """Parse confidence scores from Boltz-2 output."""
        output_dir = tmp_path / "output"
        pred_dir = output_dir / "boltz_results_test" / "predictions" / "test"
        pred_dir.mkdir(parents=True)
        (pred_dir / "test_model_0.pdb").write_text("ATOM  1  CA  ALA A   1")

        conf_data = {
            "ptm": 0.82,
            "iptm": 0.75,
            "confidence_score": 0.78,
            "complex_plddt": 0.85,
            "binding_affinity": -12.3,
            "complex_iplddt": 0.72,
            "complex_ipde": 3.5,
        }
        (output_dir / "confidence_test_model_0.json").write_text(json.dumps(conf_data))

        structure_path, confidence = parse_boltz_output(output_dir)
        assert "test_model_0.pdb" in structure_path
        assert confidence.ptm == 0.82
        assert confidence.iptm == 0.75
        assert confidence.ranking_score == 0.78
        assert confidence.plddt_mean == 85.0  # Rescaled from 0.85
        assert confidence.binding_affinity == -12.3
        assert confidence.complex_iplddt == 0.72
        assert confidence.complex_ipde == 3.5

    def test_parse_boltz_output_plddt_already_scaled(self, tmp_path: Path) -> None:
        """pLDDT already on 0-100 scale should not be rescaled."""
        output_dir = tmp_path / "output"
        pred_dir = output_dir / "boltz_results_test" / "predictions" / "test"
        pred_dir.mkdir(parents=True)
        (pred_dir / "test_model_0.pdb").write_text("ATOM")

        conf_data = {"complex_plddt": 85.0}
        (output_dir / "confidence_test_model_0.json").write_text(json.dumps(conf_data))

        _, confidence = parse_boltz_output(output_dir)
        assert confidence.plddt_mean == 85.0

    def test_parse_empty_output(self, tmp_path: Path) -> None:
        """Empty output directory returns empty results."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        structure_path, confidence = parse_boltz_output(output_dir)
        assert structure_path == ""
        assert confidence.ptm == 0.0


# ---------------------------------------------------------------------------
# Runner tests (mocked subprocess)
# ---------------------------------------------------------------------------


class TestBoltz2Runner:
    """Tests for Boltz2Runner with mocked subprocess calls."""

    def test_is_available_true(self) -> None:
        with patch("biolab_runners.boltz2.utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            runner = Boltz2Runner()
            assert runner.is_available()

    def test_is_available_false(self) -> None:
        with patch(
            "biolab_runners.boltz2.utils.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            runner = Boltz2Runner()
            assert not runner.is_available()

    def test_dry_run_complex(self) -> None:
        runner = Boltz2Runner()
        result = runner.predict_complex(
            receptor_sequence="MVKLTAEG",
            peptide_sequence="RWKLFKK",
            name="test_complex",
            dry_run=True,
        )
        assert result.quality_gate == QualityGate.PENDING
        assert "Dry run" in result.gate_reasons[0]

    def test_dry_run_monomer(self) -> None:
        runner = Boltz2Runner()
        result = runner.predict_monomer(
            sequence="MVKLTAEG",
            name="test_monomer",
            dry_run=True,
        )
        assert result.quality_gate == QualityGate.PENDING

    def test_not_installed_returns_fail(self) -> None:
        with patch(
            "biolab_runners.boltz2.utils.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            runner = Boltz2Runner()
            result = runner.predict_complex(
                receptor_sequence="MVKLTAEG",
                peptide_sequence="RWKLFKK",
            )
            assert result.quality_gate == QualityGate.FAIL
            assert "not installed" in result.error

    def test_idempotent_skip(self, tmp_path: Path) -> None:
        """Existing complete output should be reused."""
        # Set up fake output
        run_dir = tmp_path / "test_complex" / "boltz2" / "output"
        pred_dir = run_dir / "boltz_results_test" / "predictions" / "test"
        pred_dir.mkdir(parents=True)
        (pred_dir / "test_model_0.pdb").write_text("ATOM  1  CA  ALA A   1")

        conf_data = {"ptm": 0.82, "iptm": 0.75, "complex_plddt": 0.85}
        (run_dir / "confidence_test_model_0.json").write_text(json.dumps(conf_data))

        # Mock boltz_available to avoid subprocess check
        with patch("biolab_runners.boltz2.runner.boltz_available", return_value=True):
            runner = Boltz2Runner()
            result = runner.predict_complex(
                receptor_sequence="MVKLTAEG",
                peptide_sequence="RWKLFKK",
                name="test_complex",
                output_dir=tmp_path,
            )

        assert result.structure_path != ""
        assert result.confidence.ptm == 0.82

    def test_subprocess_failure(self, tmp_path: Path) -> None:
        """Non-zero exit code should produce FAIL."""
        with patch("biolab_runners.boltz2.runner.boltz_available", return_value=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stderr="CUDA OOM")
                runner = Boltz2Runner()
                result = runner.predict_complex(
                    receptor_sequence="MVKLTAEG",
                    peptide_sequence="RWKLFKK",
                    name="fail_test",
                    output_dir=tmp_path,
                )
        assert result.quality_gate == QualityGate.FAIL
        assert "CUDA OOM" in result.error


# ---------------------------------------------------------------------------
# Config / result serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for config and result serialization."""

    def test_confidence_scores_to_dict(self) -> None:
        scores = ConfidenceScores(
            ptm=0.82,
            iptm=0.75,
            ranking_score=0.78,
            plddt_mean=85.0,
            binding_affinity=-12.3,
            complex_iplddt=0.72,
            complex_ipde=3.5,
        )
        d = scores.to_dict()
        assert d["ptm"] == 0.82
        assert d["binding_affinity_kcal_mol"] == -12.3
        assert d["complex_iplddt"] == 0.72

    def test_prediction_result_to_dict(self) -> None:
        result = PredictionResult(
            name="test",
            receptor_sequence="MVKL",
            peptide_sequence="RWK",
            structure_path="/tmp/test.pdb",
            quality_gate=QualityGate.PASS,
            runtime_seconds=123.456,
        )
        d = result.to_dict()
        assert d["name"] == "test"
        assert d["receptor_sequence_length"] == 4
        assert d["peptide_sequence_length"] == 3
        assert d["quality_gate"] == "PASS"
        assert d["runtime_seconds"] == 123.5

    def test_prediction_result_save(self, tmp_path: Path) -> None:
        result = PredictionResult(name="test_save")
        path = result.save(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "test_save"

    def test_boltz2_config_defaults(self) -> None:
        config = Boltz2Config()
        assert config.accelerator == "gpu"
        assert config.use_potentials is True
        assert config.timeout_seconds == 1800
        assert config.boltz_binary == "boltz"
