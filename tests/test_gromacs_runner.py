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
from biolab_runners.gromacs.runner import (
    GromacsProtocolResult,
    GromacsRunner,
    _config_to_cli,
    _protocol_stage_identity,
)
from biolab_runners.gromacs.utils import (
    invoke,
    load_stage_manifest,
    now_utc_iso,
    parse_nthcol_energy,
    record_stage_status,
    save_stage_manifest,
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


def test_custom_binary_prefix_overrides_container_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROMACS_BIN", "container://gromacs:latest")
    result = GromacsRunner(
        output_root=tmp_path,
        config=_valid_config(name="custom-prefix"),
        binary_prefix=["gmx-custom"],
    ).run(dry_run=True)

    assert result.execution_mode.value == "subprocess"
    assert result.provenance.command[0] == "gmx-custom"


def test_container_uri_is_rejected_before_gromacs_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROMACS_BIN", "container://gromacs:latest")

    with pytest.raises(ValueError, match="container://"):
        GromacsRunner(output_root=tmp_path, config=_valid_config()).run(dry_run=True)


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


def test_runner_cache_counters_reflect_malformed_energy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _valid_config(name="malformed-cache")
    output_dir = tmp_path / config.name
    output_dir.mkdir()
    (output_dir / "topol.edr").write_text("not energy\n")

    result = GromacsRunner(output_root=tmp_path).run(config)

    assert result.status == ExecutionStatus.MALFORMED
    assert result.succeeded == 0
    assert result.failed == 1
    assert result.skipped == 1


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


def _setup_interrupted(
    work_dir: Path,
    kind: StageKind,
    prefix: str,
    *,
    with_cpt: bool,
    config: GromacsProtocolConfig | None = None,
) -> None:
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
    if config is not None:
        manifest = load_stage_manifest(work_dir)
        stage = next(item for item in build_stage_plan() if item.kind == kind)
        manifest["stages"][kind.value]["protocol_identity"] = _protocol_stage_identity(
            stage, config
        )
        save_stage_manifest(work_dir, manifest)


def _make_config(name: str, output_root: str, **overrides: Any) -> GromacsProtocolConfig:
    values: dict[str, Any] = {
        "name": name,
        "input_pdb": "/tmp/in.pdb",
        "output_root": output_root,
    }
    values.update(overrides)
    return GromacsProtocolConfig(**values)


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
        _setup_interrupted(work_dir, StageKind.PRODUCTION, stage.prefix, with_cpt=True, config=cfg)
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

    def test_running_npt_with_matching_identity_resumes_via_t_and_cpi(self, tmp_path: Path) -> None:
        cfg = _make_config("npt-cpi-resume", str(tmp_path))
        work_dir = tmp_path / cfg.name
        stage = next(s for s in build_stage_plan() if s.kind == StageKind.EQUIL_NPT)
        _setup_interrupted(work_dir, stage.kind, stage.prefix, with_cpt=True, config=cfg)
        cpt_path = work_dir / GromacsFiles.checkpoint(stage.prefix)

        captured: list[list[str]] = []

        def _capture(cmd: list[str], _wd: Path, _timeout: int) -> int:
            captured.append(cmd)
            return 0

        with patch.object(GromacsProtocolRunner, "_run_subprocess", side_effect=_capture):
            status, rc, was_skipped = GromacsProtocolRunner()._run_single_stage(
                work_dir, stage, cfg
            )

        assert (status, rc, was_skipped) == (StageStatus.COMPLETED, 0, False)
        assert "-t" in captured[0] and str(cpt_path) in captured[0]
        assert "-cpi" in captured[1] and str(cpt_path) in captured[1]
        assert "-append" in captured[1]

    @pytest.mark.parametrize("change", [{"npt_ps": 150}, {"pressure_bar": 2.0}])
    def test_running_npt_identity_change_quarantines_checkpoint_and_downstream(
        self, tmp_path: Path, change: dict[str, Any]
    ) -> None:
        original = _make_config("npt-identity-change", str(tmp_path))
        with patch.object(GromacsProtocolRunner, "_run_subprocess", return_value=0):
            first = GromacsProtocolRunner().run_protocol(original)
        assert first.succeeded == 8

        work_dir = tmp_path / original.name
        npt = next(s for s in build_stage_plan() if s.kind == StageKind.EQUIL_NPT)
        production = next(s for s in build_stage_plan() if s.kind == StageKind.PRODUCTION)
        npt_cpt = work_dir / GromacsFiles.checkpoint(npt.prefix)
        production_cpt = work_dir / GromacsFiles.checkpoint(production.prefix)
        npt_cpt.write_text("stale npt checkpoint")
        production_cpt.write_text("stale production checkpoint")
        manifest = load_stage_manifest(work_dir)
        manifest["stages"][npt.kind.value]["status"] = StageStatus.RUNNING
        manifest["stages"][npt.kind.value]["protocol_identity"] = _protocol_stage_identity(
            npt, original
        )
        save_stage_manifest(work_dir, manifest)

        changed = _make_config(original.name, str(tmp_path), **change)
        captured: list[list[str]] = []

        def _capture(cmd: list[str], _wd: Path, _timeout: int) -> int:
            captured.append(cmd)
            return 0

        with patch.object(GromacsProtocolRunner, "_run_subprocess", side_effect=_capture):
            result = GromacsProtocolRunner().run_protocol(changed)

        assert result.skipped == 6
        assert result.succeeded == 2
        assert len(captured) == 4
        assert all("-t" not in command and "-cpi" not in command for command in captured)
        assert not npt_cpt.exists()
        assert not production_cpt.exists()
        stale_files = [path for path in (work_dir / ".stale").rglob("*") if path.is_file()]
        assert npt_cpt.name in {path.name for path in stale_files}
        assert production_cpt.name in {path.name for path in stale_files}

    def test_legacy_running_without_identity_starts_fresh(self, tmp_path: Path) -> None:
        cfg = _make_config("legacy-running", str(tmp_path))
        work_dir = tmp_path / cfg.name
        stage = next(s for s in build_stage_plan() if s.kind == StageKind.EQUIL_NPT)
        _setup_interrupted(work_dir, stage.kind, stage.prefix, with_cpt=True)
        cpt_path = work_dir / GromacsFiles.checkpoint(stage.prefix)
        captured: list[list[str]] = []

        def _capture(cmd: list[str], _wd: Path, _timeout: int) -> int:
            captured.append(cmd)
            return 0

        with patch.object(GromacsProtocolRunner, "_run_subprocess", side_effect=_capture):
            status, rc, was_skipped = GromacsProtocolRunner()._run_single_stage(
                work_dir, stage, cfg
            )

        assert (status, rc, was_skipped) == (StageStatus.COMPLETED, 0, False)
        assert "-t" not in captured[0] and "-cpi" not in captured[1]
        assert not cpt_path.exists()
        stale_files = [path for path in (work_dir / ".stale").rglob("*") if path.is_file()]
        assert cpt_path.name in {path.name for path in stale_files}

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


def test_protocol_timeout_is_reported_as_timeout_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config("timeout", str(tmp_path))
    monkeypatch.setattr(GromacsProtocolRunner, "_run_subprocess", lambda *_: 124)

    result = GromacsProtocolRunner().run_protocol(config)

    assert result.exit_code == 124
    assert result.failed == 1
    assert result.status == ExecutionStatus.TIMEOUT


def test_protocol_prebuilt_staging_failure_preserves_container_execution_mode(
    tmp_path: Path,
) -> None:
    topology = tmp_path / "prepared.top"
    topology.write_text("topology")
    config = GromacsProtocolConfig(
        name="prebuilt-staging-failure",
        input_pdb=str(tmp_path / "ignored.pdb"),
        output_root=str(tmp_path / "runs"),
        prebuilt_topology=str(topology),
        prebuilt_coordinates=str(tmp_path / "missing.gro"),
    )

    result = GromacsProtocolRunner(
        binary_prefix=["docker", "run", "--rm", "gromacs:latest", "gmx"]
    ).run_protocol(config)

    assert result.error.startswith("prebuilt topology staging failed:")
    assert result.execution_mode.value == "container_uri"


def test_protocol_prebuilt_provenance_binds_topology_and_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    topology = tmp_path / "prepared.top"
    coordinates = tmp_path / "prepared.gro"
    topology.write_text("topology-v1")
    coordinates.write_text("coordinates-v1")
    config = GromacsProtocolConfig(
        name="prebuilt-digest",
        input_pdb=str(tmp_path / "ignored.pdb"),
        output_root=str(tmp_path / "runs"),
        prebuilt_topology=str(topology),
        prebuilt_coordinates=str(coordinates),
        force=True,
    )
    monkeypatch.setattr(GromacsProtocolRunner, "_run_subprocess", lambda *_: 1)

    first = GromacsProtocolRunner().run_protocol(config)
    topology.write_text("topology-v2")
    second = GromacsProtocolRunner().run_protocol(config)

    assert first.executed is True
    assert first.source_backbone_digest is None
    assert first.executed_config_digest != second.executed_config_digest
