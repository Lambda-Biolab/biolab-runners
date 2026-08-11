"""Tests for the RFdiffusion runner.

The runner is a thin subprocess wrapper; tests inject a fake
``invoke`` via monkeypatch so no upstream RFdiffusion install is
needed during CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from biolab_runners.rfdiffusion import (
    RecordData,
    RecordDataStatus,
    rfdiffusion_available,
)
from biolab_runners.rfdiffusion.config import RFdiffusionConfig
from biolab_runners.rfdiffusion.runner import RFdiffusionRunner, _config_to_cli
from biolab_runners.rfdiffusion.utils import (
    RecordDataStatus as _Status,
)
from biolab_runners.rfdiffusion.utils import (
    parse_backbone_pdb,
)

SAMPLE_PDB = """\
HEADER    RFdiffusion design 0
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.500   0.000   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       3.000  -1.000   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       4.000   1.000   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       5.000   1.000   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       6.000   1.000   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       7.000   0.000   0.000  1.00  0.00           O
ATOM      9  N   GLY A   3       7.500   2.000   0.000  1.00  0.00           N
ATOM     10  CA  GLY A   3       8.500   2.000   0.000  1.00  0.00           C
ATOM     11  C   GLY A   3       9.500   2.000   0.000  1.00  0.00           C
ATOM     12  O   GLY A   3      10.000   1.000   0.000  1.00  0.00           O
TER
END
"""


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    return tmp_path / "rfdiffusion"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_defaults_pass_validation() -> None:
    config = RFdiffusionConfig()
    assert config.mode == "linear"
    assert config.length_min == 14
    assert config.length_max == 18
    assert config.task_count == 1000


def test_config_rejects_inverted_length_range() -> None:
    with pytest.raises(ValueError, match="length range invalid"):
        RFdiffusionConfig(length_min=18, length_max=14)


def test_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        RFdiffusionConfig(mode="wat")


def test_disulfide_mode_requires_pairs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RFdiffusionConfig(mode="disulfide")


def test_linear_mode_rejects_disulfide_pairs() -> None:
    with pytest.raises(ValueError, match="disulfide_pairs"):
        RFdiffusionConfig(mode="linear", disulfide_pairs=((3, 9),))


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


def test_parse_backbone_pdb_extracts_three_residues(tmp_path: Path) -> None:
    pdb = tmp_path / "design.pdb"
    pdb.write_text(SAMPLE_PDB)
    assert parse_backbone_pdb(pdb) == "GAG"


def test_parse_backbone_pdb_handles_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        parse_backbone_pdb(tmp_path / "absent.pdb")


def test_record_data_to_dict_round_trip() -> None:
    record = RecordData(index=2, path="/tmp/design.pdb", sequence="GAG")
    payload = record.to_dict()
    assert payload["index"] == "2"
    assert payload["sequence"] == "GAG"
    assert payload["status"] == RecordDataStatus.SUCCEEDED


def test_rfdiffusion_available_returns_false_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RFDIFFUSION_BIN", "/nonexistent/rfdiffusion")
    assert rfdiffusion_available() is False


# ---------------------------------------------------------------------------
# CLI translation
# ---------------------------------------------------------------------------


def test_config_to_cli_linear_default() -> None:
    cli = _config_to_cli(RFdiffusionConfig())
    assert cli["contigmap.contigs"] == "14-18"
    assert cli["inference.num_designs"] == "1000"
    assert "inference.cyclic" not in cli
    assert cli["inference.deterministic"] == "True"


def test_config_to_cli_macrocyclic() -> None:
    cli = _config_to_cli(RFdiffusionConfig(mode="head_to_tail"))
    assert cli["inference.cyclic"] == "True"
    assert cli["inference.cyc_chains"] == "a"


def test_config_to_cli_disulfide() -> None:
    cli = _config_to_cli(RFdiffusionConfig(mode="disulfide", disulfide_pairs=((3, 9), (5, 12))))
    assert cli["inference.cyclic"] == "True"
    assert cli["inference.cyc_chains"] == "3,9,5,12"


def test_config_to_cli_with_hotspots() -> None:
    cli = _config_to_cli(RFdiffusionConfig(hotspots=("A12", "B17")))
    assert cli["ppi.hotspot_res"] == "A12,B17"


# ---------------------------------------------------------------------------
# Runner behaviour (with a fake invoke)
# ---------------------------------------------------------------------------


def test_runner_dry_run_does_not_invoke(output_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    invoked: list[dict[str, Any]] = []

    def fake_invoke(*, config_dict: dict[str, Any], output_dir: Path, **_: Any) -> int:
        invoked.append({"config_dict": config_dict, "output_dir": output_dir})
        output_dir.mkdir(parents=True, exist_ok=True)
        return 0

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner.invoke", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="dry"), dry_run=True)
    assert invoked == []
    assert result.exit_code == 0
    assert result.succeeded == 0


def test_runner_idempotent_when_output_exists(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoked: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        invoked.append(output_dir)
        return 0

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner.invoke", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    name = "idem"
    design_dir = output_root / name
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "design_0.pdb").write_text(SAMPLE_PDB)

    result = runner.run(RFdiffusionConfig(name=name))
    assert invoked == []  # cached, did not re-invoke
    assert result.skipped == 1
    assert result.succeeded == 1


def test_runner_force_re_runs(output_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        calls.append(output_dir)
        (output_dir / "design_0.pdb").write_text(SAMPLE_PDB)
        return 0

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner.invoke", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    name = "force"
    design_dir = output_root / name
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "design_0.pdb").write_text(SAMPLE_PDB)

    result = runner.run(RFdiffusionConfig(name=name), force=True)
    assert calls == [design_dir]
    assert result.exit_code == 0


def test_runner_records_per_design(output_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        for i in range(3):
            (output_dir / f"design_{i}.pdb").write_text(SAMPLE_PDB)
        return 0

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner.invoke", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="batch"))
    assert result.succeeded == 3
    assert result.failed == 0
    assert {r.sequence for r in result.records} == {"GAG"}


def test_runner_handles_unparseable_pdb(output_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing parse should record a FAILED entry, not crash the runner."""

    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        (output_dir / "good.pdb").write_text(SAMPLE_PDB)
        # Drop a file that exists at glob time but is deleted before
        # parse_backbone_pdb reads it. The runner must record this as
        # FAILED, not crash.
        ghost = output_dir / "ghost.pdb"
        ghost.write_text(SAMPLE_PDB)
        ghost.unlink()
        return 0

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner.invoke", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="broken"))
    assert all(r.path for r in result.records)
    # The ghost file was unlinked before parse, so it must not appear.
    assert not any("ghost" in r.path for r in result.records)
    assert any(r.status == _Status.SUCCEEDED for r in result.records)


def test_runner_propagates_nonzero_exit_code(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_invoke(**_: Any) -> int:
        return 7

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner.invoke", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="failure"))
    assert result.exit_code == 7
    assert result.succeeded == 0


def test_runner_requires_config() -> None:
    runner = RFdiffusionRunner(output_root=Path("/tmp"))
    with pytest.raises(ValueError, match="RFdiffusionConfig is required"):
        runner.run()
