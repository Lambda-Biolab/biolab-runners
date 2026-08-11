"""Tests for the GROMACS runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from biolab_runners.gromacs import (
    GromacsRecord,
    GromacsRecordStatus,
    gromacs_available,
)
from biolab_runners.gromacs.config import GromacsConfig
from biolab_runners.gromacs.runner import GromacsRunner, _config_to_cli
from biolab_runners.gromacs.utils import parse_nthcol_energy


def _valid_config(**overrides: Any) -> GromacsConfig:
    base: dict[str, Any] = {
        "name": "gmx-1",
        "structure_file": "/tmp/run.tpr",
        "topology_file": "/tmp/topol.top",
        "output_dir": "/tmp/output",
        "nsteps": 5000,
    }
    base.update(overrides)
    return GromacsConfig(**base)


def test_config_rejects_missing_structure_file() -> None:
    with pytest.raises(ValueError, match="structure_file is required"):
        GromacsConfig(
            structure_file="",
            topology_file="/tmp/topol.top",
        )


def test_config_rejects_missing_topology_file() -> None:
    with pytest.raises(ValueError, match="topology_file is required"):
        GromacsConfig(
            structure_file="/tmp/run.tpr",
            topology_file="",
        )


def test_config_rejects_zero_nsteps() -> None:
    with pytest.raises(ValueError, match="nsteps"):
        GromacsConfig(
            structure_file="/tmp/run.tpr",
            topology_file="/tmp/topol.top",
            nsteps=0,
        )


def test_config_rejects_negative_timestep() -> None:
    with pytest.raises(ValueError, match="timestep_fs"):
        GromacsConfig(
            structure_file="/tmp/run.tpr",
            topology_file="/tmp/topol.top",
            timestep_fs=0.0,
        )


def test_config_to_cli_includes_required_keys() -> None:
    cli = _config_to_cli(_valid_config())
    assert cli["-deffnm"] == "topol"
    assert cli["-s"] == "/tmp/run.tpr"
    assert cli["-nsteps"] == "5000"


def test_config_to_cli_includes_extra() -> None:
    cli = _config_to_cli(_valid_config(extra={"foo": 1.0}))
    assert cli["foo"] == "1.0"


def test_parse_nthcol_energy_skips_metadata_lines(tmp_path: Path) -> None:
    p = tmp_path / "energy.xvg"
    p.write_text('# This is a comment\n@    title "Energy"\n0.0  -123.456\n')
    assert parse_nthcol_energy(p, column=1) == -123.456


def test_parse_nthcol_energy_handles_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.xvg"
    p.write_text("")
    assert parse_nthcol_energy(p) == 0.0


def test_parse_nthcol_energy_returns_zero_for_garbage(tmp_path: Path) -> None:
    p = tmp_path / "garbage.xvg"
    p.write_text("not a number\n")
    assert parse_nthcol_energy(p) == 0.0


def test_gromacs_available_returns_false_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROMACS_BIN", "/nonexistent/gmx")
    assert gromacs_available() is False


def test_runner_dry_run_does_not_invoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    invoked: list[dict[str, Any]] = []

    def fake_invoke(**_: Any) -> int:
        invoked.append({})
        return 0

    monkeypatch.setattr("biolab_runners.gromacs.runner.invoke", fake_invoke)
    runner = GromacsRunner(
        output_root=tmp_path,
        config=_valid_config(name="dry"),
    )
    result = runner.run(dry_run=True)
    assert invoked == []
    assert result.exit_code == 0


def test_runner_idempotent_when_energy_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoked: list[Path] = []

    def fake_invoke(**_: Any) -> int:
        invoked.append(Path("invoked"))
        return 0

    monkeypatch.setattr("biolab_runners.gromacs.runner.invoke", fake_invoke)
    name = "idem"
    design_dir = tmp_path / name
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "topol.edr").write_text("0.0  -123.456\n")

    config = _valid_config(name=name, output_dir=str(design_dir))
    runner = GromacsRunner(output_root=tmp_path, config=config)
    result = runner.run(config)
    assert invoked == []
    assert result.skipped == 1


def test_runner_force_re_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_invoke(**_: Any) -> int:
        calls.append(Path("invoked"))
        return 0

    monkeypatch.setattr("biolab_runners.gromacs.runner.invoke", fake_invoke)
    name = "force"
    design_dir = tmp_path / name
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "topol.edr").write_text("0.0  -123.456\n")

    config = _valid_config(name=name, output_dir=str(design_dir))
    runner = GromacsRunner(output_root=tmp_path, config=config)
    result = runner.run(config, force=True)
    assert calls == [Path("invoked")]
    assert result.exit_code == 0


def test_runner_records_energy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_invoke(**_: Any) -> int:
        design_dir = tmp_path / "x"
        design_dir.mkdir(parents=True, exist_ok=True)
        (design_dir / "topol.edr").write_text("0.0  -42.0\n")
        return 0

    monkeypatch.setattr("biolab_runners.gromacs.runner.invoke", fake_invoke)
    runner = GromacsRunner(output_root=tmp_path, config=_valid_config(name="x"))
    result = runner.run()
    assert result.succeeded == 1
    assert result.records[0].potential_energy == -42.0


def test_runner_propagates_nonzero_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_invoke(**_: Any) -> int:
        return 7

    monkeypatch.setattr("biolab_runners.gromacs.runner.invoke", fake_invoke)
    runner = GromacsRunner(output_root=tmp_path, config=_valid_config(name="failure"))
    result = runner.run()
    assert result.exit_code == 7


def test_runner_requires_config() -> None:
    runner = GromacsRunner(output_root=Path("/tmp"))
    with pytest.raises(ValueError, match="GromacsConfig is required"):
        runner.run()


def test_gromacs_record_to_dict_round_trip() -> None:
    record = GromacsRecord(
        index=3,
        path="/tmp/x.edr",
        potential_energy=-99.5,
        status=GromacsRecordStatus.SUCCEEDED,
    )
    payload = record.to_dict()
    assert payload["index"] == "3"
    assert payload["potential_energy"] == "-99.5"
