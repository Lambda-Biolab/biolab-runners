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
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from biolab_runners.gromacs import GromacsProtocolConfig
from biolab_runners.gromacs.protocol import StageKind
from biolab_runners.gromacs.runner import GromacsProtocolRunner as Runner
from biolab_runners.gromacs.segment import read_cumulative_steps
from biolab_runners.gromacs.utils import load_stage_manifest, save_stage_manifest

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


def _checkpoint_step(checkpoint: Path) -> int:
    if not checkpoint.is_file():
        return 0
    match = re.search(r"step = (\d+)", checkpoint.read_text())
    return int(match.group(1)) if match else 0


def _read_fake_step(checkpoint: Path) -> int | None:
    return _checkpoint_step(checkpoint)


def _fake_mdrun(
    commands: list[list[str]],
    *,
    tpr_bytes: bytes = _PROD_TPR.encode(),
    target_steps: int = _steps(0.02),
) -> Any:
    """Return a subprocess replacement that records commands and writes prod artifacts."""

    def _run(cmd: list[str], work_dir: Path, _timeout: int) -> int:
        commands.append(list(cmd))
        if cmd[1] == "grompp" and cmd[cmd.index("-o") + 1] == "prod.tpr":
            (work_dir / "prod.tpr").write_bytes(tpr_bytes)
        if cmd[1] == "mdrun" and cmd[cmd.index("-deffnm") + 1] == "prod":
            step = target_steps
            if "-nsteps" in cmd:
                step = _checkpoint_step(work_dir / "prod.cpt") + int(cmd[cmd.index("-nsteps") + 1])
            for suffix in ("gro", "edr", "log"):
                (work_dir / f"prod.{suffix}").write_text("artifact")
            (work_dir / "prod.cpt").write_text(f"step = {step}\n")
            with (work_dir / "prod.xtc").open("ab") as handle:
                handle.write(b"frame\n")
        return 0

    return _run


def _run_segment(
    config: GromacsProtocolConfig,
    segment_ns: float,
    monkeypatch: pytest.MonkeyPatch,
    commands: list[list[str]],
) -> Any:
    runner = Runner(binary_prefix=["gmx"])
    monkeypatch.setattr(
        runner,
        "_run_subprocess",
        _fake_mdrun(commands, target_steps=_steps(config.effective_production_ns())),
    )
    monkeypatch.setattr(runner, "_read_checkpoint_step", _read_fake_step)
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


def test_segment_retry_after_unrecorded_success_does_not_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "lost-write")
    _run_segment(config, 0.01, monkeypatch, [])
    work_dir = tmp_path / config.name
    manifest = load_stage_manifest(work_dir)
    record = manifest["stages"][StageKind.PRODUCTION.value]
    record["status"] = "running"
    for key in ("production_steps", "production_target_steps", "production_complete"):
        record.pop(key, None)
    save_stage_manifest(work_dir, manifest)

    retry_commands: list[list[str]] = []
    retry = _run_segment(config, 0.01, monkeypatch, retry_commands)

    assert retry.production_complete is False
    assert retry.production_steps == _steps(0.01)
    assert _checkpoint_step(work_dir / "prod.cpt") == _steps(0.01)
    assert not any(command[1] == "mdrun" for command in retry_commands)

    third = _run_segment(config, 0.01, monkeypatch, [])
    assert third.production_complete is True
    assert third.production_steps == _steps(0.02)


def test_segment_retry_after_interruption_runs_only_remaining_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "partial")
    _run_segment(config, 0.01, monkeypatch, [])
    work_dir = tmp_path / config.name
    segment_steps = _steps(0.01)
    (work_dir / "prod.cpt").write_text(f"step = {segment_steps + 1000}\n")
    manifest = load_stage_manifest(work_dir)
    manifest["stages"][StageKind.PRODUCTION.value]["status"] = "running"
    save_stage_manifest(work_dir, manifest)

    commands: list[list[str]] = []
    result = _run_segment(config, 0.01, monkeypatch, commands)

    mdrun = [command for command in commands if command[1] == "mdrun"]
    assert len(mdrun) == 1
    assert mdrun[0][mdrun[0].index("-nsteps") + 1] == str(segment_steps - 1000)
    assert result.production_complete is True
    assert result.production_steps == _steps(0.02)


def test_segment_after_completed_full_run_reconciles_without_advancing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "legacy-full")
    runner = Runner(binary_prefix=["gmx"])
    monkeypatch.setattr(runner, "_run_subprocess", _fake_mdrun([]))
    full = runner.run_protocol(config)
    assert full.production_complete is True
    work_dir = tmp_path / config.name

    manifest = load_stage_manifest(work_dir)
    record = manifest["stages"][StageKind.PRODUCTION.value]
    for key in ("production_steps", "production_target_steps", "production_complete"):
        record.pop(key, None)
    save_stage_manifest(work_dir, manifest)

    commands: list[list[str]] = []
    segment = _run_segment(config, 0.01, monkeypatch, commands)

    assert segment.production_complete is True
    assert segment.production_steps == _steps(0.02)
    assert not any(command[1] == "mdrun" for command in commands)
    assert _checkpoint_step(work_dir / "prod.cpt") == _steps(0.02)


def test_segment_after_completed_full_run_without_checkpoint_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "legacy-pruned")
    runner = Runner(binary_prefix=["gmx"])
    monkeypatch.setattr(runner, "_run_subprocess", _fake_mdrun([]))
    assert runner.run_protocol(config).production_complete is True
    work_dir = tmp_path / config.name
    (work_dir / "prod.cpt").unlink()

    manifest = load_stage_manifest(work_dir)
    record = manifest["stages"][StageKind.PRODUCTION.value]
    for key in ("production_steps", "production_target_steps", "production_complete"):
        record.pop(key, None)
    save_stage_manifest(work_dir, manifest)

    commands: list[list[str]] = []
    segment = _run_segment(config, 0.01, monkeypatch, commands)

    assert segment.production_complete is True
    assert segment.production_steps == _steps(0.02)
    assert commands == []


def test_segment_missing_checkpoint_restarts_within_one_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "pruned-checkpoint")
    _run_segment(config, 0.01, monkeypatch, [])
    work_dir = tmp_path / config.name
    (work_dir / "prod.cpt").unlink()

    commands: list[list[str]] = []
    result = _run_segment(config, 0.01, monkeypatch, commands)

    mdrun = [command for command in commands if command[1] == "mdrun"]
    assert len(mdrun) == 1
    assert mdrun[0][mdrun[0].index("-nsteps") + 1] == str(_steps(0.01))
    assert result.production_complete is False
    assert result.production_steps == _steps(0.01)


def test_full_run_after_partial_segment_reports_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "mixed")
    _run_segment(config, 0.01, monkeypatch, [])

    runner = Runner(binary_prefix=["gmx"])
    commands: list[list[str]] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_mdrun(commands))
    result = runner.run_protocol(config)

    assert result.production_complete is True
    assert result.production_steps == _steps(0.02)
    record = load_stage_manifest(tmp_path / config.name)["stages"][StageKind.PRODUCTION.value]
    assert record["status"] == "completed"
    assert record["production_complete"] is True
    assert record["production_steps"] == _steps(0.02)


def test_segment_refuses_when_checkpoint_step_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "unreadable")
    _run_segment(config, 0.01, monkeypatch, [])

    runner = Runner(binary_prefix=["gmx"])
    commands: list[list[str]] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_mdrun(commands))
    monkeypatch.setattr(runner, "_read_checkpoint_step", lambda _checkpoint: None)

    result = runner.run_protocol(config, segment_ns=0.01)

    assert result.failed == 1
    assert result.exit_code == 2
    assert not any(command[1] == "mdrun" for command in commands)
    record = load_stage_manifest(tmp_path / config.name)["stages"][StageKind.PRODUCTION.value]
    assert record["status"] == "failed"
    assert "cannot read the integration step" in record["error"]


def test_segment_dry_run_does_not_read_or_record_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, "dry")
    runner = Runner(binary_prefix=["gmx"], dry_run=True)
    monkeypatch.setattr(
        runner, "_read_checkpoint_step", lambda _checkpoint: pytest.fail("dry run read")
    )

    result = runner.run_protocol(config, segment_ns=0.01)

    assert result.validated == 8
    assert result.production_complete is False
    assert result.production_steps == 0
    assert result.production_target_steps == _steps(0.02)
    assert not (tmp_path / config.name / "gromacs_stages.json").exists()


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
    assert result.production_steps == _steps(0.01)
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


def test_segment_interruption_rebinds_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    def _interrupt_production(command: list[str], work_dir: Path, timeout: int) -> int:
        rc = _protocol_invoker(first_commands, [])(command, work_dir, timeout)
        return -15 if command[1] == "mdrun" and "prod" in command else rc

    monkeypatch.setattr(first_runner, "_run_subprocess", _interrupt_production)
    first = first_runner.run_protocol(config, segment_ns=0.01)

    assert first.interrupted == 1
    work_dir = Path(first.output_dir)
    state = json.loads((work_dir / "selection-map.json").read_text())["gromacs_state"]
    assert state["stage"] == "production"
    assert state["checkpoint_sha256"] == _sha256(work_dir / "prod.cpt")

    second_commands: list[list[str]] = []
    second_runner = Runner(binary_prefix=["gmx"])
    monkeypatch.setattr(second_runner, "_run_subprocess", _protocol_invoker(second_commands, []))
    monkeypatch.setattr(second_runner, "_read_checkpoint_step", lambda _checkpoint: 0)

    second = second_runner.run_protocol(config, segment_ns=0.01)

    assert second.exit_code == 0
    assert second.production_complete is False
    assert second.production_steps == _steps(0.01)

    third_commands: list[list[str]] = []
    third_runner = Runner(binary_prefix=["gmx"])
    monkeypatch.setattr(third_runner, "_run_subprocess", _protocol_invoker(third_commands, []))
    monkeypatch.setattr(
        third_runner,
        "_read_checkpoint_step",
        lambda _checkpoint: read_cumulative_steps(work_dir),
    )

    third = third_runner.run_protocol(config, segment_ns=0.01)

    assert third.exit_code == 0
    assert third.production_complete is True
