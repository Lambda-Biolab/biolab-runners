"""Tests for the AfCycDesign feasibility spike."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from biolab_runners.afcycdesign import (
    AfCycDesignResult,
    AfCycDesignRunner,
    AfCycDesignStatus,
)


def _runner(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> AfCycDesignRunner:
    monkeypatch.setenv("AFCYCDESIGN_SPIKE", "1" if enabled else "0")
    return AfCycDesignRunner(container_image="ghcr.io/example/afcyc:0.0.1")


def test_runner_is_feasible_matches_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _runner(monkeypatch, enabled=True).is_feasible() is True
    assert _runner(monkeypatch, enabled=False).is_feasible() is False


def test_runner_skipped_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner(monkeypatch, enabled=False)
    result = runner.run("ACDEF", output_dir=tmp_path, name="off")
    assert result.status == AfCycDesignStatus.FAILED
    assert "AFCYCDESIGN_SPIKE" in result.error
    assert result.sequence_length == 5


def test_runner_enabled_returns_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner(monkeypatch, enabled=True)
    result = runner.run("ACDEF", output_dir=tmp_path, name="on")
    # When ``jax`` is not installed, the enabled path returns a FAILED
    # result with a clear error. The slice 11 runbook documents the
    # container image that carries jax + ColabDesign.
    if result.status == AfCycDesignStatus.FAILED:
        assert "jax" in result.error or "ColabDesign" in result.error
        return
    assert result.output_path == str(tmp_path)
    summary = json.loads((tmp_path / "spike-summary.json").read_text())
    assert summary["name"] == "on"
    assert summary["sequence_length"] == 5


def test_runner_enabled_handles_missing_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``jax`` is unavailable, the runner returns a FAILED result."""

    monkeypatch.setenv("AFCYCDESIGN_SPIKE", "1")
    runner = AfCycDesignRunner()

    # Force the import to fail
    import builtins

    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "jax":
            raise ImportError("jax unavailable for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = runner.run("ACDEF", output_dir=tmp_path, name="miss")
    assert result.status == AfCycDesignStatus.FAILED
    assert "jax" in result.error


def test_result_to_dict_round_trip() -> None:
    result = AfCycDesignResult(
        name="x",
        sequence_length=7,
        mean_pLDDT=0.85,
        metrics={"ipTM": 0.7},
        status=AfCycDesignStatus.SUCCEEDED,
    )
    payload = result.to_dict()
    assert payload["name"] == "x"
    assert payload["mean_pLDDT"] == 0.85
    assert payload["metrics"]["ipTM"] == 0.7
