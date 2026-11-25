"""Behavioral tests for bounded production-MD segment continuation.

``run_protocol(config, segment_ns=...)`` must run one full-target production
as several ordered invocations without changing the approved protocol target.
These tests drive the public result and the stage manifest, and replace
``_run_subprocess`` with a fake that only materialises the artifacts the
runner inspects. The real-GROMACS two-segment path is covered by
``test_gromacs_segment_cpu_integration.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from biolab_runners.gromacs import GromacsProtocolConfig
from biolab_runners.gromacs.protocol import StageKind
from biolab_runners.gromacs.runner import GromacsProtocolRunner as Runner
from biolab_runners.gromacs.utils import load_stage_manifest

_PROD_TPR = "full-target-production-tpr"


def _steps(ns: float) -> int:
    return round(ns * 1_000_000 / 2)


def _config(tmp_path: Path, name: str, **overrides: Any) -> GromacsProtocolConfig:
    payload: dict[str, Any] = {
        "name": name,
        "input_pdb": "/tmp/in.pdb",
        "output_root": str(tmp_path),
        "production_ns": 0.02,
    }
    payload.update(overrides)
    return GromacsProtocolConfig(**payload)


def _fake_mdrun(
    commands: list[list[str]],
    *,
    tpr_bytes: bytes = _PROD_TPR.encode(),
) -> Any:
    """Return a subprocess replacement that records commands and writes prod artifacts."""

    def _run(cmd: list[str], work_dir: Path, _timeout: int) -> int:
        commands.append(list(cmd))
        if cmd[1] == "grompp" and cmd[cmd.index("-o") + 1] == "prod.tpr":
            (work_dir / "prod.tpr").write_bytes(tpr_bytes)
        if cmd[1] == "mdrun" and cmd[cmd.index("-deffnm") + 1] == "prod":
            for suffix in ("cpt", "gro", "edr", "log", "xtc"):
                path = work_dir / f"prod.{suffix}"
                with path.open("ab" if suffix == "xtc" else "wb") as handle:
                    handle.write(b"frame\n" if suffix == "xtc" else b"artifact")
        return 0

    return _run


def _run_segment(
    config: GromacsProtocolConfig,
    segment_ns: float,
    monkeypatch: pytest.MonkeyPatch,
    commands: list[list[str]],
) -> Any:
    runner = Runner(binary_prefix=["gmx"])
    monkeypatch.setattr(runner, "_run_subprocess", _fake_mdrun(commands))
    return runner.run_protocol(config, segment_ns=segment_ns)


def test_segment_result_reports_bounded_steps_without_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    result = _run_segment(_config(tmp_path, "bounded"), 0.01, monkeypatch, commands)

    assert result.production_complete is False
    assert result.production_steps == _steps(0.01)
    assert result.production_target_steps == _steps(0.02)
    manifest = load_stage_manifest(Path(result.output_dir))
    assert manifest["stages"][StageKind.PRODUCTION.value]["status"] == "running"


def test_segment_setup_and_equilibration_are_skipped_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "skip")
    first_commands: list[list[str]] = []
    _run_segment(config, 0.01, monkeypatch, first_commands)

    second_commands: list[list[str]] = []
    result = _run_segment(config, 0.01, monkeypatch, second_commands)

    assert result.skipped == 7
    assert result.succeeded == 1
    assert all(command[1] != "grompp" for command in second_commands)


def test_segment_resume_carries_cpi_append_and_relative_nsteps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "resume-flags")
    first_commands: list[list[str]] = []
    _run_segment(config, 0.01, monkeypatch, first_commands)

    second_commands: list[list[str]] = []
    _run_segment(config, 0.01, monkeypatch, second_commands)

    mdrun = [cmd for cmd in second_commands if cmd[1] == "mdrun"]
    assert len(mdrun) == 1
    cpt = str(tmp_path / config.name / "prod.cpt")
    assert "-cpi" in mdrun[0] and cpt in mdrun[0]
    assert "-append" in mdrun[0]
    assert "-nsteps" in mdrun[0]
    assert mdrun[0][mdrun[0].index("-nsteps") + 1] == str(_steps(0.01))


def test_segment_keeps_full_target_tpr_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "stable-tpr")
    first_commands: list[list[str]] = []
    _run_segment(config, 0.01, monkeypatch, first_commands)
    tpr = tmp_path / config.name / "prod.tpr"
    before = hashlib.sha256(tpr.read_bytes()).hexdigest()

    second_commands: list[list[str]] = []
    _run_segment(config, 0.01, monkeypatch, second_commands)

    assert hashlib.sha256(tpr.read_bytes()).hexdigest() == before
    assert before == hashlib.sha256(_PROD_TPR.encode()).hexdigest()
    assert not any(cmd[1] == "grompp" for cmd in second_commands)


def test_segment_reaches_completion_only_at_full_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "completion")
    first = _run_segment(config, 0.01, monkeypatch, [])
    assert first.production_complete is False
    assert first.production_steps == _steps(0.01)

    second = _run_segment(config, 0.01, monkeypatch, [])
    assert second.production_complete is True
    assert second.production_steps == _steps(0.02)
    manifest = load_stage_manifest(tmp_path / config.name)
    assert manifest["stages"][StageKind.PRODUCTION.value]["status"] == "completed"

    third = _run_segment(config, 0.01, monkeypatch, [])
    assert third.production_complete is True
    assert third.skipped == 8


def test_segment_none_preserves_full_run_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "full")
    runner = Runner(binary_prefix=["gmx"])
    monkeypatch.setattr(runner, "_run_subprocess", _fake_mdrun([]))

    result = runner.run_protocol(config)

    assert result.production_complete is True
    assert result.production_steps == _steps(0.02)
    assert result.succeeded == 8


@pytest.mark.parametrize("segment_ns", [0.0, -0.1])
def test_segment_rejects_nonpositive_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, segment_ns: float
) -> None:
    with pytest.raises(ValueError, match="segment_ns must be positive"):
        _run_segment(_config(tmp_path, f"nonpositive-{segment_ns}"), segment_ns, monkeypatch, [])


def test_segment_rejects_duration_beyond_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        _run_segment(_config(tmp_path, "overshoot"), 0.03, monkeypatch, [])

    config = _config(tmp_path, "overshoot-cumulative")
    _run_segment(config, 0.01, monkeypatch, [])
    with pytest.raises(ValueError, match="exceeds"):
        _run_segment(config, 0.02, monkeypatch, [])


def test_segment_changed_production_ns_quarantines_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "quarantine")
    _run_segment(config, 0.01, monkeypatch, [])
    cpt = tmp_path / config.name / "prod.cpt"
    cpt.write_bytes(b"bound-checkpoint")
    assert cpt.exists()

    changed = _config(tmp_path, config.name, production_ns=0.04)
    commands: list[list[str]] = []
    result = _run_segment(changed, 0.01, monkeypatch, commands)

    assert result.production_target_steps == _steps(0.04)
    assert cpt.read_bytes() != b"bound-checkpoint"
    assert any(cmd[1] == "grompp" for cmd in commands)
    stale = [path for path in (tmp_path / config.name / ".stale").rglob("*") if path.is_file()]
    assert "prod.cpt" in {path.name for path in stale}


def test_segment_result_serializes_completion_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_segment(_config(tmp_path, "serialize"), 0.01, monkeypatch, [])
    payload = result.to_dict()
    assert payload["production_complete"] is False
    assert payload["production_steps"] == _steps(0.01)


def test_segment_preserves_checkpoint_identity_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from dataclasses import replace

    from tests.test_gromacs_selection_sidecars import (
        _protocol_invoker,
        _sha256,
        _strict_config,
        _write_bundle,
    )

    bundle = _write_bundle(tmp_path / "bundle")
    config = replace(_strict_config(tmp_path, bundle), production_ns=0.02)
    first_commands: list[list[str]] = []
    first_runner = Runner(binary_prefix=["gmx"])
    monkeypatch.setattr(first_runner, "_run_subprocess", _protocol_invoker(first_commands, []))

    first = first_runner.run_protocol(config, segment_ns=0.01)
    work_dir = Path(first.output_dir)
    assert first.production_complete is False
    state = json.loads((work_dir / "selection-map.json").read_text())["gromacs_state"]
    assert state["stage"] == "production"
    assert state["checkpoint_sha256"] == _sha256(work_dir / "prod.cpt")

    (work_dir / "prod.cpt").write_bytes(b"replacement-checkpoint")
    second_commands: list[list[str]] = []
    second_runner = Runner(binary_prefix=["gmx"])
    monkeypatch.setattr(second_runner, "_run_subprocess", _protocol_invoker(second_commands, []))

    second = second_runner.run_protocol(config, segment_ns=0.01)

    assert second.failed == 1
    assert "checkpoint digest mismatch" in second.error
    assert second_commands == []
