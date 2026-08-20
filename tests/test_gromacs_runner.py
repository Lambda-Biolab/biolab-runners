"""Tests for the GROMACS runner."""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from biolab_runners.contracts import ExecutionStatus, RunnerTimeoutError
from biolab_runners.gromacs import (
    GromacsProtocolConfig,
    GromacsProtocolRunner,
    GromacsRecord,
    GromacsRecordStatus,
    StageStatus,
    gromacs_available,
)
from biolab_runners.gromacs.config import GromacsConfig
from biolab_runners.gromacs.paths import GromacsFiles
from biolab_runners.gromacs.protocol import StageKind, build_stage_plan, stage_minimum_outputs
from biolab_runners.gromacs.runner import GromacsProtocolResult, GromacsRunner, _config_to_cli
from biolab_runners.gromacs.utils import (
    invoke,
    load_stage_manifest,
    now_utc_iso,
    parse_nthcol_energy,
    record_stage_status,
)


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


def test_runner_rejects_unparseable_energy_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_invoke(**_: Any) -> int:
        output = tmp_path / "bad"
        output.mkdir(exist_ok=True)
        (output / "topol.edr").write_text("not energy\n")
        return 0

    monkeypatch.setattr("biolab_runners.gromacs.runner.invoke", fake_invoke)
    result = GromacsRunner(output_root=tmp_path, config=_valid_config(name="bad")).run()

    assert result.status == ExecutionStatus.MALFORMED
    assert result.succeeded == 0
    assert result.failed == 1


def test_invoke_maps_timeout_to_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("subprocess.run", Mock(side_effect=subprocess.TimeoutExpired("gmx", 1)))
    with pytest.raises(RunnerTimeoutError):
        invoke(
            config_dict={"-deffnm": "topol", "-s": "run.tpr", "-nsteps": "1"},
            output_dir=tmp_path,
            mdrun_extra=(),
            binary_prefix=["gmx"],
            timeout_seconds=1,
        )


def test_protocol_artifacts_ignore_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    output = tmp_path / "output"
    output.mkdir()
    (output / "escape.txt").symlink_to(outside)

    result = GromacsProtocolResult(
        name="escape",
        output_dir=str(output),
        replica_index=0,
        replicas_total=1,
    )

    assert result.artifacts == ()


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


# ---------------------------------------------------------------------------
# Post-merge hotfix regression: interrupted-stage authority (S4)
#
# Bug 1 (truth-against-disk): a Spot reclaim leaves the manifest in
# RUNNING with .tpr/.gro/.edr/.log (and a flushed .cpt) on disk. The OLD
# runner's disk-output fallback promoted that state to COMPLETED and
# silently dropped the -cpi resume.
# Bug 2 (overwrite-on-interrupt): even when the SIGTERM path preserved
# the manifest, the runner continued past the interrupted stage so a
# later missing-input FAILED overwrote the truthful exit_code/error.
# ---------------------------------------------------------------------------


def _setup_interrupted(work_dir: Path, kind: StageKind, prefix: str, *, with_cpt: bool) -> None:
    """Mirror the post-SIGTERM disk state (RUNNING manifest + minimum outputs [+ .cpt])."""
    work_dir.mkdir(parents=True, exist_ok=True)
    for name in stage_minimum_outputs(kind, prefix):
        (work_dir / name).write_text("placeholder")
    if with_cpt:
        (work_dir / GromacsFiles.checkpoint(prefix)).write_text("cpt")
    record_stage_status(
        work_dir,
        kind.value,
        StageStatus.RUNNING,
        outputs=stage_minimum_outputs(kind, prefix),
        started_at=now_utc_iso(),
        error="interrupted by SIGTERM; resumable on next invocation",
    )


def _make_config(name: str, output_root: str) -> GromacsProtocolConfig:
    return GromacsProtocolConfig(name=name, input_pdb="/tmp/in.pdb", output_root=output_root)


class TestRunnerInterruptedResume:
    """SIGTERM recovery: resume, don't silently promote-to-COMPLETED; halt, don't overwrite."""

    def test_running_with_checkpoint_resumes_via_cpi(self, tmp_path: Path) -> None:
        """RUNNING + minimum outputs + .cpt -> runner executes mdrun with -cpi/-append.

        Old bug: disk-output fallback promoted state to COMPLETED,
        skipped the subprocess, and silently dropped -cpi.
        """
        cfg = _make_config("cpi-resume", str(tmp_path))
        work_dir = tmp_path / cfg.name
        stage = next(s for s in build_stage_plan() if s.kind == StageKind.PRODUCTION)
        _setup_interrupted(work_dir, StageKind.PRODUCTION, stage.prefix, with_cpt=True)
        cpt_path = work_dir / GromacsFiles.checkpoint(stage.prefix)

        captured: list[list[str]] = []

        def _capture(cmd: list[str], _wd: Path, _timeout: int) -> int:
            captured.append(cmd)
            return 0

        with patch.object(GromacsProtocolRunner, "_run_subprocess", side_effect=_capture):
            status, rc, was_skipped = GromacsProtocolRunner()._run_single_stage(
                work_dir, stage, cfg
            )

        # Execution path, not silent skip.
        assert was_skipped is False, (
            "interrupted stage with .cpt was silently promoted to COMPLETED "
            "(disk-output fallback fired despite RUNNING manifest + .cpt)"
        )
        assert status == StageStatus.COMPLETED and rc == 0
        # Last subprocess call is gmx mdrun; it must carry -cpi/-append.
        mdrun = captured[-1]
        assert "-cpi" in mdrun and str(cpt_path) in mdrun and "-append" in mdrun, (
            f"expected -cpi {cpt_path} -append in mdrun; got {mdrun}"
        )
        # Manifest now COMPLETED via real execution (not disk-fallback).
        record = load_stage_manifest(work_dir)["stages"][StageKind.PRODUCTION.value]
        assert record["status"] == StageStatus.COMPLETED

    def test_manifest_silent_with_checkpoint_resumes_via_cpi(self, tmp_path: Path) -> None:
        """Manifest silent + minimum outputs + .cpt -> runner resumes via -cpi.

        The disk-output fallback's first conjunct (``manifest_status
        is None``) alone is NOT sufficient; the second conjunct
        (``_checkpoint_for(...) is None``) blocks the fallback when
        a ``.cpt`` is on disk. This complements
        ``test_running_with_checkpoint_resumes_via_cpi`` (which
        exercises the RUNNING-manifest branch) by exercising the
        silent-manifest branch — a regression that dropped the
        second conjunct would re-introduce the bug.
        """
        cfg = _make_config("silent-resume", str(tmp_path))
        work_dir = tmp_path / cfg.name
        stage = next(s for s in build_stage_plan() if s.kind == StageKind.PRODUCTION)
        # Pre-create minimum outputs + .cpt, but no manifest record.
        _setup_interrupted(work_dir, StageKind.PRODUCTION, stage.prefix, with_cpt=True)
        (work_dir / GromacsFiles.STAGE_MANIFEST).unlink()  # make manifest silent
        cpt_path = work_dir / GromacsFiles.checkpoint(stage.prefix)

        captured: list[list[str]] = []

        def _capture(cmd: list[str], _wd: Path, _timeout: int) -> int:
            captured.append(cmd)
            return 0

        with patch.object(GromacsProtocolRunner, "_run_subprocess", side_effect=_capture):
            status, rc, was_skipped = GromacsProtocolRunner()._run_single_stage(
                work_dir, stage, cfg
            )

        assert was_skipped is False, (
            "manifest-silent + .cpt promoted to COMPLETED via disk fallback; "
            "the second guard conjunct (no .cpt) is missing or broken"
        )
        assert status == StageStatus.COMPLETED and rc == 0
        mdrun = captured[-1]
        assert "-cpi" in mdrun and str(cpt_path) in mdrun and "-append" in mdrun, (
            f"expected -cpi {cpt_path} -append in mdrun; got {mdrun}"
        )

    def test_interruption_halts_loop_with_truthful_exit(self, tmp_path: Path) -> None:
        """SIGTERM in stage N halts the loop; stage N+1 is not attempted; exit_code is the sentinel.

        Old bug: the runner continued past interruption; the next
        stage's missing-input FAILED overwrote exit_code to 7 and
        bumped failed++.
        """
        cfg = _make_config("halt-test", str(tmp_path))
        # Setup = 5 calls (topology, box, solvate, ions×2);
        # MINIMIZE grompp = call 6; MINIMIZE mdrun = call 7 (the SIGTERM);
        # anything past the SIGTERM would be a missing-input failure.
        signal_at = 7
        call_count = [0]

        def _track(cmd: list[str], _wd: Path, _timeout: int) -> int:
            call_count[0] += 1
            return (
                -signal.SIGTERM
                if call_count[0] == signal_at
                else 7
                if call_count[0] > signal_at
                else 0
            )

        with patch.object(GromacsProtocolRunner, "_run_subprocess", side_effect=_track):
            result = GromacsProtocolRunner().run_protocol(cfg)

        # Truthful interruption accounting (OLD code wrote failed=1, exit_code=7 here).
        assert result.interrupted == 1
        assert result.failed == 0
        assert result.exit_code == -signal.SIGTERM
        assert result.stage_statuses[StageKind.MINIMIZE.value] == "interrupted"
        # No later stages attempted; loop halted at MINIMIZE.
        for later in (StageKind.EQUIL_NVT, StageKind.EQUIL_NPT, StageKind.PRODUCTION):
            assert later.value not in result.stage_statuses, (
                f"stage {later.value} should be absent from stage_statuses after interruption; "
                f"got {result.stage_statuses.get(later.value)!r}"
            )
        # Exactly 7 subprocess calls — the 8th (would-be fail) never happened.
        assert call_count[0] == signal_at, (
            f"loop continued past interruption ({call_count[0]} calls > {signal_at}); "
            "a would-be missing-input FAILED was attempted"
        )
