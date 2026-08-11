"""Tests for the ProteinMPNN runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from biolab_runners.proteinmpnn import (
    DesignRecordStatus,
    proteinmpnn_available,
)
from biolab_runners.proteinmpnn.config import ProteinMPNNConfig
from biolab_runners.proteinmpnn.runner import ProteinMPNNRunner, _config_to_cli
from biolab_runners.proteinmpnn.utils import parse_fasta_sequences

SAMPLE_FASTA = """\
>design_1, score=0.752
ACDEFGHIKLPQRSTVWY
>design_2, score=0.701
ACDEFGHIKLPQRSTVWY
>design_3, score=0.689
ACDEFGHIKLPQRSTVWY
>design_4, score=0.650
ACDEFGHIKLPQRSTVWY
"""


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    return tmp_path / "proteinmpnn"


@pytest.fixture
def pdb_input(tmp_path: Path) -> Path:
    pdb = tmp_path / "backbone.pdb"
    pdb.write_text(
        "HEADER    test\n"
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    return pdb


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_defaults_pass_validation() -> None:
    config = ProteinMPNNConfig()
    assert config.task_count == 4
    assert config.temperature == 0.1


def test_config_rejects_non_positive_task_count() -> None:
    with pytest.raises(ValueError, match="task_count"):
        ProteinMPNNConfig(task_count=0)


def test_config_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ProteinMPNNConfig(temperature=0.0)


def test_config_rejects_zero_indexed_fixed_positions() -> None:
    with pytest.raises(ValueError, match="1-indexed"):
        ProteinMPNNConfig(fixed_positions=(0, 1, 2))


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


def test_parse_fasta_sequences(tmp_path: Path) -> None:
    p = tmp_path / "out.fa"
    p.write_text(SAMPLE_FASTA)
    records = parse_fasta_sequences(p)
    assert len(records) == 4
    names = [name for name, _ in records]
    assert all(name.startswith("design_") for name in names)
    assert all(seq == "ACDEFGHIKLPQRSTVWY" for _, seq in records)


def test_parse_fasta_sequences_handles_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.fa"
    p.write_text("")
    assert parse_fasta_sequences(p) == []


def test_proteinmpnn_available_returns_false_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROTEINMPNN_BIN", "/nonexistent/proteinmpnn")
    assert proteinmpnn_available() is False


# ---------------------------------------------------------------------------
# CLI translation
# ---------------------------------------------------------------------------


def _stub_input(pdb_input: Path) -> dict[str, str]:
    return _config_to_cli(
        ProteinMPNNConfig(name="x", temperature=0.2, seed=42),
        pdb_input,
    )


def test_config_to_cli_default_includes_four_sequences(pdb_input: Path) -> None:
    cli = _stub_input(pdb_input)
    assert cli["num_seq_per_target"] == "4"
    assert cli["sampling_temp"] == "0.2"
    assert cli["seed"] == "42"
    # Upstream ProteinMPNN expects `model_name` to be a *checkpoint
    # prefix* (joined with ".pt" to form the file name), not the
    # weight-folder name. The historical default of "vanilla_model_weights"
    # was a folder name and silently broke checkpoint loading. See
    # protein_mpnn_run.py:57 (`checkpoint_path = model_folder_path +
    # f'{args.model_name}.pt'`) and protein_mpnn_run.py:426 (the
    # --model_name help text lists the valid prefixes).
    assert cli["model_name"] == "v_48_020"
    assert cli["pdb_path"] == pdb_input.name


@pytest.mark.parametrize("checkpoint", ["v_48_002", "v_48_010", "v_48_020", "v_48_030"])
def test_config_to_cli_supports_all_upstream_checkpoints(pdb_input: Path, checkpoint: str) -> None:
    """Lock the runner-side model_name contract to upstream's choices.

    If a future PR widens or narrows this contract, this test forces the
    author to update both the config default and this parametrize list
    in lockstep.
    """
    cli = _config_to_cli(ProteinMPNNConfig(model_name=checkpoint), pdb_input)
    assert cli["model_name"] == checkpoint


def test_proteinmpnn_config_default_is_an_upstream_checkpoint_prefix() -> None:
    """Guard against re-introducing a folder-name default by mistake.

    Pre-existing bug: the dataclass default was the *folder* name
    (``vanilla_model_weights``) instead of the *checkpoint* prefix
    (e.g. ``v_48_020``). Upstream ``--model_name`` expects the latter
    (see protein_mpnn_run.py:57; the script concatenates
    ``<weights_dir>/<model_name>.pt`` to find the checkpoint file).
    """
    default = ProteinMPNNConfig().model_name
    upstream_checkpoints = {"v_48_002", "v_48_010", "v_48_020", "v_48_030"}
    assert default in upstream_checkpoints, (
        f"ProteinMPNNConfig().model_name defaults to {default!r}; "
        f"must be one of upstream's checkpoint prefixes "
        f"({sorted(upstream_checkpoints)}). A folder name silently "
        f"breaks checkpoint loading upstream."
    )


def test_config_to_cli_handles_fixed_positions(pdb_input: Path) -> None:
    cli = _config_to_cli(
        ProteinMPNNConfig(fixed_positions=(3, 7)),
        pdb_input,
    )
    assert cli["fixed_positions"] == "2,6"


def test_config_to_cli_handles_omit_aa(pdb_input: Path) -> None:
    cli = _config_to_cli(ProteinMPNNConfig(omit_aa="CDF"), pdb_input)
    assert cli["omit_AA"] == "CDF"


def test_config_to_cli_with_ca_only(pdb_input: Path) -> None:
    cli = _config_to_cli(ProteinMPNNConfig(ca_only=True), pdb_input)
    assert cli["ca_only"] == "True"


# ---------------------------------------------------------------------------
# Runner behaviour
# ---------------------------------------------------------------------------


def test_runner_dry_run_does_not_invoke(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoked: list[dict[str, Any]] = []

    def fake_invoke(**_: Any) -> int:
        invoked.append({})
        return 0

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner.invoke", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    result = runner.run(pdb_input, dry_run=True)
    assert invoked == []
    assert result.records == ()


def test_runner_idempotent_when_fasta_exists(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoked: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        invoked.append(output_dir)
        return 0

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner.invoke", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    name = "idem"
    design_dir = output_root / name / pdb_input.stem
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "out.fa").write_text(SAMPLE_FASTA)

    result = runner.run(pdb_input, ProteinMPNNConfig(name=name))
    assert invoked == []
    assert result.skipped == 4
    assert result.succeeded == 4


def test_runner_force_re_runs(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        calls.append(output_dir)
        (output_dir / "out.fa").write_text(SAMPLE_FASTA)
        return 0

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner.invoke", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    name = "force"
    design_dir = output_root / name / pdb_input.stem
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "out.fa").write_text(SAMPLE_FASTA)

    result = runner.run(pdb_input, ProteinMPNNConfig(name=name), force=True)
    assert calls == [design_dir]
    assert result.exit_code == 0


def test_runner_records_per_design(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        (output_dir / "out.fa").write_text(SAMPLE_FASTA)
        return 0

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner.invoke", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig(task_count=4))
    result = runner.run(pdb_input)
    assert result.succeeded == 4
    assert result.failed == 0
    assert all(r.sequence == "ACDEFGHIKLPQRSTVWY" for r in result.records)


def test_runner_records_failed_for_unparseable_fasta(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        # Force an OSError on parse by writing then unlinking.
        target = output_dir / "out.fa"
        target.write_text("placeholder")
        target.unlink()
        return 0

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner.invoke", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    result = runner.run(pdb_input)
    assert all(r.status == DesignRecordStatus.FAILED for r in result.records)


def test_runner_propagates_nonzero_exit_code(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_invoke(**_: Any) -> int:
        return 9

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner.invoke", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    result = runner.run(pdb_input)
    assert result.exit_code == 9
    assert result.succeeded == 0


def test_runner_run_batch_processes_each_input(
    output_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = []
    for i in range(3):
        pdb = tmp_path / f"input_{i}.pdb"
        pdb.write_text("HEADER\nATOM      1  CA  GLY A   1       0.000   0.000   0.000\nEND\n")
        inputs.append(pdb)

    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        (output_dir / "out.fa").write_text(SAMPLE_FASTA)
        return 0

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner.invoke", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    results = runner.run_batch(inputs)
    assert len(results) == 3
    assert all(r.succeeded == 4 for r in results)


def test_runner_requires_config() -> None:
    runner = ProteinMPNNRunner(output_root=Path("/tmp"))
    with pytest.raises(ValueError, match="ProteinMPNNConfig is required"):
        runner.run(Path("/tmp/x.pdb"))
