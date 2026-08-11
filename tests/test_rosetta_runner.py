"""Tests for the Rosetta runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from biolab_runners.rosetta import (
    RelaxRecord,
    RelaxRecordStatus,
    rosetta_available,
)
from biolab_runners.rosetta.config import RosettaConfig
from biolab_runners.rosetta.runner import RosettaRunner, _config_to_cli
from biolab_runners.rosetta.utils import parse_score_file

SAMPLE_SCORE = """\
SEQUENCE:
SCORE: total_score       fa_atr       fa_rep       fa_sol
SCORE:  -123.456      -854.231      72.123      410.123
"""


def _valid_config(**overrides: Any) -> RosettaConfig:
    base: dict[str, Any] = {
        "name": "relax-1",
        "script_file": "/opt/rosetta/scripts/relax.xml",
        "input_pdb": "/tmp/input.pdb",
        "output_dir": "/tmp/output",
        "nstruct": 1,
        "license_acknowledged": True,
    }
    base.update(overrides)
    return RosettaConfig(**base)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_rejects_unacknowledged_license() -> None:
    with pytest.raises(ValueError, match="license acknowledgement"):
        RosettaConfig(
            script_file="/tmp/x.xml",
            input_pdb="/tmp/x.pdb",
            license_acknowledged=False,
        )


def test_config_rejects_missing_script() -> None:
    with pytest.raises(ValueError, match="script_file is required"):
        RosettaConfig(
            script_file="",
            input_pdb="/tmp/x.pdb",
            license_acknowledged=True,
        )


def test_config_rejects_missing_input() -> None:
    with pytest.raises(ValueError, match="input_pdb is required"):
        RosettaConfig(
            script_file="/tmp/x.xml",
            input_pdb="",
            license_acknowledged=True,
        )


def test_config_rejects_bad_nstruct() -> None:
    with pytest.raises(ValueError, match="nstruct"):
        RosettaConfig(
            script_file="/tmp/x.xml",
            input_pdb="/tmp/x.pdb",
            nstruct=0,
            license_acknowledged=True,
        )


def test_config_to_cli_includes_required_flags() -> None:
    cli = _config_to_cli(_valid_config())
    assert cli["s"] == "/opt/rosetta/scripts/relax.xml"
    assert cli["in:file:s"] == "/tmp/input.pdb"
    assert cli["nstruct"] == "1"


def test_config_to_cli_includes_extra_flags() -> None:
    cfg = _valid_config(
        extra_flags=("beta=1.0", "no_output"),
        extra={"gamma": 2.0},
    )
    cli = _config_to_cli(cfg)
    assert cli["beta"] == "1.0"
    assert "no_output" in cli
    assert cli["gamma"] == "2.0"


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


def test_parse_score_file_extracts_first_float(tmp_path: Path) -> None:
    p = tmp_path / "score.sc"
    p.write_text(SAMPLE_SCORE)
    assert parse_score_file(p) == -123.456


def test_parse_score_file_handles_garbage(tmp_path: Path) -> None:
    p = tmp_path / "score.sc"
    p.write_text("not a score\n")
    assert parse_score_file(p) == 0.0


def test_parse_score_file_handles_empty(tmp_path: Path) -> None:
    p = tmp_path / "score.sc"
    p.write_text("")
    assert parse_score_file(p) == 0.0


def test_rosetta_available_returns_false_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROSETTA_BIN", "/nonexistent/rosetta_scripts")
    assert rosetta_available() is False


def test_relax_record_to_dict_round_trip() -> None:
    record = RelaxRecord(
        index=2,
        path="/tmp/score.sc",
        total_score=-99.5,
        status=RelaxRecordStatus.SUCCEEDED,
    )
    payload = record.to_dict()
    assert payload["index"] == "2"
    assert payload["total_score"] == "-99.5"


# ---------------------------------------------------------------------------
# runner behaviour
# ---------------------------------------------------------------------------


def test_runner_dry_run_does_not_invoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    invoked: list[dict[str, Any]] = []

    def fake_invoke(**_: Any) -> int:
        invoked.append({})
        return 0

    monkeypatch.setattr("biolab_runners.rosetta.runner.invoke", fake_invoke)
    runner = RosettaRunner(
        output_root=tmp_path,
        config=_valid_config(name="dry"),
    )
    result = runner.run(dry_run=True)
    assert invoked == []
    assert result.exit_code == 0


def test_runner_idempotent_when_score_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoked: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        invoked.append(output_dir)
        return 0

    monkeypatch.setattr("biolab_runners.rosetta.runner.invoke", fake_invoke)
    name = "idem"
    design_dir = tmp_path / name
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "score.sc").write_text(SAMPLE_SCORE)

    config = _valid_config(name=name, output_dir=str(design_dir))
    runner = RosettaRunner(output_root=tmp_path, config=config)
    result = runner.run(config)
    assert invoked == []
    assert result.skipped == 1


def test_runner_force_re_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        calls.append(output_dir)
        (output_dir / "score.sc").write_text(SAMPLE_SCORE)
        return 0

    monkeypatch.setattr("biolab_runners.rosetta.runner.invoke", fake_invoke)
    name = "force"
    design_dir = tmp_path / name
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "score.sc").write_text(SAMPLE_SCORE)

    config = _valid_config(name=name, output_dir=str(design_dir))
    runner = RosettaRunner(output_root=tmp_path, config=config)
    result = runner.run(config, force=True)
    assert calls == [design_dir]
    assert result.exit_code == 0


def test_runner_records_per_design(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        # All relax runs produce a single ``score.sc``; the runner
        # consumes the file and records one result per directory.
        (output_dir / "score.sc").write_text(SAMPLE_SCORE)
        return 0

    monkeypatch.setattr("biolab_runners.rosetta.runner.invoke", fake_invoke)
    runner = RosettaRunner(output_root=tmp_path, config=_valid_config(name="batch"))
    result = runner.run()
    assert result.succeeded == 1
    assert result.records[0].total_score == -123.456


def test_runner_propagates_nonzero_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_invoke(**_: Any) -> int:
        return 7

    monkeypatch.setattr("biolab_runners.rosetta.runner.invoke", fake_invoke)
    runner = RosettaRunner(output_root=tmp_path, config=_valid_config(name="failure"))
    result = runner.run()
    assert result.exit_code == 7


def test_runner_requires_config() -> None:
    runner = RosettaRunner(output_root=Path("/tmp"))
    with pytest.raises(ValueError, match="RosettaConfig is required"):
        runner.run()
