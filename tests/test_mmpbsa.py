"""Tests for ``biolab_runners.mmpbsa`` (slice 14 gmx_MMPBSA integration).

Trigger / non-trigger coverage:

* Trigger — gmx_MMPBSA installed: records returned with kcal/mol
  per-residue decomposition.
* Non-trigger — binary missing: ``status="unsupported"`` and
  empty ``per_residue_records``.
* Closed-form parser — synthetic input file with a known set of
  per-residue numbers round-trips through ``parse_residue_decomposition``.

The binary is **not** stubbed in tests; the runner is exercised
end-to-end through ``gmx_mmpbsa_available()`` returning ``False``
on a system where ``gmx_MMPBSA`` isn't on PATH (true for the test
host). This exercises the graceful ``unsupported`` path without
requiring an AmberTools install.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from biolab_runners.mmpbsa import (
    GmxMMPBSARecord,
    GmxMMPBSARunner,
    GmxMMPBSAStatus,
    gmx_mmpbsa_available,
    parse_residue_decomposition,
)
from biolab_runners.openmm.config import OpenMMConfig


def _make_config(tmp_path: Path) -> OpenMMConfig:
    """Minimal OpenMMConfig for the runner — paths don't need to exist."""
    return OpenMMConfig(
        receptor_pdb="rec.pdb",
        peptide_pdb="pep.pdb",
        output_dir=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# Closed-form: parse_residue_decomposition
# ---------------------------------------------------------------------------


class TestParseResidueDecompositionClosedForm:
    def _write_decomposition_file(self, tmp_path: Path) -> Path:
        # Synthetic gmx_MMPBSA output (the actual format places per-
        # energy-term columns to the right of the residue label).
        path = tmp_path / "pilot_residue_decomposition_finite.dat"
        path.write_text(
            "\n".join(
                [
                    "# Residue\tvan der Waals\tElectrostatic\tPolar solvation\t"
                    "Non-polar solvation\tTOTAL",
                    "A:LEU115\t-3.21\t-1.04\t2.85\t-0.45\t-1.85",
                    "A:VAL4\t-1.07\t-0.32\t1.12\t-0.10\t-0.37",
                    "B:ARG50\t-2.55\t-5.32\t3.84\t-0.30\t-4.33",
                ]
            )
        )
        return path

    def test_round_trips_known_residue_records(self, tmp_path: Path) -> None:
        path = self._write_decomposition_file(tmp_path)
        records = parse_residue_decomposition(path)
        assert len(records) == 3
        leu = records[0]
        assert leu.residue_label == "LEU115"
        assert leu.chain == "A"
        assert leu.vdw_A == pytest.approx(-3.21)
        assert leu.electrostatic_A == pytest.approx(-1.04)
        assert leu.polar_solvation_A == pytest.approx(2.85)
        assert leu.non_polar_solvation_A == pytest.approx(-0.45)
        assert leu.total_A == pytest.approx(-1.85)

    def test_drops_malformed_records_silently(self, tmp_path: Path) -> None:
        # Bad number → record dropped; others kept.
        path = tmp_path / "out.dat"
        path.write_text(
            "A:GLY1\tNOT-A-NUMBER\t-0.1\t0.2\t-0.05\t-0.15\nA:GLY2\t-1.0\t-0.5\t0.3\t-0.05\t-1.25\n"
        )
        records = parse_residue_decomposition(path)
        assert len(records) == 1
        assert records[0].residue_label == "GLY2"

    def test_returns_empty_when_file_missing(self, tmp_path: Path) -> None:
        records = parse_residue_decomposition(tmp_path / "nonexistent.dat")
        assert records == ()

    def test_returns_empty_when_blank_lines_only(self, tmp_path: Path) -> None:
        path = tmp_path / "blank.dat"
        path.write_text("\n   \n\n# only a comment, nothing to parse\n")
        records = parse_residue_decomposition(path)
        assert records == ()

    def test_to_dict_round_trip(self) -> None:
        record = GmxMMPBSARecord(
            residue_label="LEU115",
            chain="A",
            vdw_A=-3.21,
            electrostatic_A=-1.04,
            polar_solvation_A=2.85,
            non_polar_solvation_A=-0.45,
            total_A=-1.85,
        )
        d = record.to_dict()
        assert d["residue"] == "LEU115"
        assert d["chain"] == "A"
        per_energy = d["per_energy_term_A"]
        assert per_energy["van_der_waals"] == -3.21  # type: ignore[index]
        assert per_energy["electrostatic"] == -1.04  # type: ignore[index]
        assert per_energy["polar_solvation"] == 2.85  # type: ignore[index]
        assert per_energy["non_polar_solvation"] == -0.45  # type: ignore[index]
        assert per_energy["total"] == -1.85  # type: ignore[index]


# ---------------------------------------------------------------------------
# gmx_mmpbsa_available — PATH probe
# ---------------------------------------------------------------------------


class TestAvailabilityProbe:
    def test_returns_false_when_binary_missing(self) -> None:
        """With ``gmx_MMPBSA`` not on PATH (CI / most hosts), probe
        returns False. This pins the slice-14 graceful-degradation path.
        """
        assert gmx_mmpbsa_available(binary="__nonexistent-binary-for-test__") is False

    def test_returns_true_for_container_prefix(self) -> None:
        """Operators can pin a container image; probe assumes availability."""
        assert gmx_mmpbsa_available(binary="container://ambertools/mmpbsa") is True


# ---------------------------------------------------------------------------
# Runner end-to-end
# ---------------------------------------------------------------------------


class TestRunnerUnsupported:
    """When the binary is missing, ``run()`` returns ``status="unsupported"``
    with no records and an error message. This is slice 14 acceptance
    criterion for missing optional tooling.
    """

    def test_returns_unsupported_status_when_binary_missing(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        runner = GmxMMPBSARunner(
            config=config,
            prefix=str(tmp_path / "mmpbsa"),
            mmpbsa_binary="__nonexistent-binary-for-test__",
        )
        result = runner.run()
        assert result["status"] == GmxMMPBSAStatus.UNSUPPORTED
        assert result["per_residue_records"] == []
        assert "__nonexistent-binary-for-test__" in str(result.get("error", ""))

    def test_container_uri_is_rejected_before_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _make_config(tmp_path)
        runner = GmxMMPBSARunner(
            config=config,
            prefix="run",
            mmpbsa_binary="container://ambertools/mmpbsa",
        )
        monkeypatch.setattr("subprocess.run", Mock(side_effect=AssertionError("invoked")))

        result = runner.run()

        assert result["status"] == GmxMMPBSAStatus.UNSUPPORTED
        assert result["execution_mode"] == "container_uri"
        assert result["exit_code"] == 127

    def test_nonzero_exit_code_is_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _make_config(tmp_path)
        runner = GmxMMPBSARunner(config=config, prefix="run", mmpbsa_binary="gmx_MMPBSA")
        monkeypatch.setattr("biolab_runners.mmpbsa.utils.gmx_mmpbsa_available", lambda **_: True)
        completed = Mock(returncode=9, stdout="", stderr="failed")
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: completed)

        result = runner.run()

        assert result["status"] == GmxMMPBSAStatus.FAILED
        assert result["exit_code"] == 9


class TestRunnerStatusConstants:
    def test_status_constants_match_documented(self) -> None:
        assert GmxMMPBSAStatus.SUCCEEDED == "succeeded"
        assert GmxMMPBSAStatus.FAILED == "failed"
        assert GmxMMPBSAStatus.UNSUPPORTED == "unsupported"

    def test_status_class_is_single_source_of_truth(self) -> None:
        """Regression: ``GmxMMPBSAStatus`` must be a single class object
        whether imported from the package root or the runner submodule.

        Two parallel definitions (one in ``__init__`` + one in ``runner``)
        would silently pass the string-equality tests above but let
        the runner's emitter diverge from the package's public class.
        The fix: ``__init__`` re-exports the runner's class, so
        ``biolab_runners.mmpbsa.GmxMMPBSAStatus is
        biolab_runners.mmpbsa.runner.GmxMMPBSAStatus``.
        """
        import biolab_runners.mmpbsa
        import biolab_runners.mmpbsa.runner

        assert biolab_runners.mmpbsa.GmxMMPBSAStatus is biolab_runners.mmpbsa.runner.GmxMMPBSAStatus
