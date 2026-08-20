"""Consumer-facing tests for the shared runner contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from biolab_runners import (
    ExecutionMode,
    ExecutionStatus,
    IncompleteOutputError,
    ProvenanceMetadata,
    RunnerInvocationError,
    RunnerOutputError,
    RunnerUnavailableError,
    require_artifact,
)
from biolab_runners.proteinmpnn.cli import build_command, main
from biolab_runners.proteinmpnn.runner import ProteinMPNNResult


def test_public_contracts_round_trip_a_real_artifact(tmp_path: Path) -> None:
    # ARRANGE
    output = tmp_path / "design.fa"
    output.write_text(">design_0\nACDE\n")

    # ACT
    artifact = require_artifact(output, kind="sequence")
    provenance = ProvenanceMetadata(
        model_identifier="proteinmpnn",
        temperature=None,
        image_digest=None,
        source_backbone_digest=None,
        exit_code=0,
        failure_reason="",
        stderr_tail="",
        base_seed=None,
        requested_seed=None,
        task_count=1,
        rng_intent="single-stream",
        runner_name="proteinmpnn",
        execution_mode=ExecutionMode.SUBPROCESS,
        status=ExecutionStatus.SUCCEEDED,
        artifacts=(artifact,),
    )

    # ASSERT
    payload = provenance.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    artifact_payload = payload["artifacts"]
    assert isinstance(artifact_payload, list)
    assert isinstance(artifact_payload[0], dict)
    assert str(artifact_payload[0]["digest"]).startswith("sha256:")
    assert payload["execution_mode"] == "subprocess"


def test_required_artifact_missing_fails_closed(tmp_path: Path) -> None:
    # ARRANGE
    missing = tmp_path / "missing.pdb"

    # ACT / ASSERT
    with pytest.raises(IncompleteOutputError):
        require_artifact(missing, kind="structure")


def test_error_hierarchy_has_distinct_consumer_catch_points() -> None:
    assert issubclass(RunnerUnavailableError, RuntimeError)
    assert issubclass(RunnerInvocationError, RuntimeError)
    assert issubclass(RunnerOutputError, RuntimeError)


def test_proteinmpnn_entrypoint_builds_direct_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    # ARRANGE
    monkeypatch.setenv("PROTEINMPNN_SCRIPT", "/opt/ProteinMPNN/protein_mpnn_run.py")
    monkeypatch.setenv("PROTEINMPNN_PYTHON", "/usr/bin/python3")

    # ACT
    command = build_command(
        (
            "--input_path",
            "/tmp/input",
            "--output_path",
            "/tmp/output",
            "--pdb_path",
            "backbone.pdb",
            "--seed",
            "7",
            "--ca_only",
            "True",
            "--omit_AA",
            "C",
            "--extra_upstream_flag",
            "value",
        )
    )

    # ASSERT
    assert command == [
        "/usr/bin/python3",
        "/opt/ProteinMPNN/protein_mpnn_run.py",
        "--pdb_path",
        "/tmp/input/backbone.pdb",
        "--out_folder",
        "/tmp/output",
        "--batch_size",
        "1",
        "--model_name",
        "v_48_020",
        "--num_seq_per_target",
        "4",
        "--sampling_temp",
        "0.1",
        "--seed",
        "7",
        "--ca_only",
        "--omit_AAs",
        "C",
        "--extra_upstream_flag",
        "value",
    ]
    assert "shell" not in command


def test_proteinmpnn_entrypoint_rejects_fixed_positions() -> None:
    with pytest.raises(ValueError, match="fixed_positions is unsupported"):
        build_command(
            [
                "--input_path",
                "/tmp/input",
                "--output_path",
                "/tmp/output",
                "--pdb_path",
                "backbone.pdb",
                "--fixed_positions",
                "0,1",
            ]
        )


def test_proteinmpnn_entrypoint_help_does_not_need_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROTEINMPNN_SCRIPT", "/does/not/exist/protein_mpnn_run.py")
    assert main(["--help"]) == 0


def test_proteinmpnn_entrypoint_missing_runtime_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROTEINMPNN_SCRIPT", "/does/not/exist/protein_mpnn_run.py")
    assert main(["--input_path", "/tmp/input"]) == 2


def test_legacy_result_fields_remain_available() -> None:
    result = ProteinMPNNResult(name="design", output_dir="out")
    assert result.name == "design"
    assert result.output_dir == "out"
    assert result.records == ()
    assert result.status == ExecutionStatus.INCOMPLETE
