"""Tests for the ProteinMPNN runner."""

from __future__ import annotations

import json
import sys
import unittest.mock as mock_mod
from pathlib import Path
from typing import Any

import pytest
from biolab_runners.proteinmpnn import (
    DesignRecordStatus,
    proteinmpnn_available,
)
from biolab_runners.proteinmpnn.config import ProteinMPNNConfig
from biolab_runners.proteinmpnn.runner import (
    EXCLUDED_FROM_EXECUTED_DIGEST,
    ProteinMPNNRunner,
    _config_to_cli,
)
from biolab_runners.proteinmpnn.utils import (
    InvokeResult,
    invoke,
    parse_fasta_sequences,
)

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

NATIVE_AND_SAMPLE_FASTA = (
    ">design_8, score=0.7520, global_score=0.8010, fixed_chains=[], "
    "designed_chains=['A','B','C'], model_name=v_48_020, git_hash=deadbeef, seed=37\n"
    "MKTAYIAKQRQISFVKSHFSRQDILDLI\n"
    ">T=0.1, sample=1, score=0.6214, global_score=0.6501, seq_recovery=0.4828\n"
    "VKTAYIAKQRQISFVKSHFSRQDILDLI\n"
    ">T=0.1, sample=2, score=0.6382, global_score=0.6610, seq_recovery=0.5172\n"
    "AKTAYIAKQRQISFVKSHFSRQDILDLI\n"
)

SAMPLE_ONLY_STOCK_FASTA = "\n".join(NATIVE_AND_SAMPLE_FASTA.splitlines()[2:]) + "\n"

#: Canonical OCI form.
VALID_OCI_DIGEST = "sha256:" + "ab" * 32  # 64 hex chars
#: Same digest in bare-hex form — accepted and normalised to OCI form.
VALID_BARE_DIGEST = "ab" * 32


def _fake_invoke_ok(**_: Any) -> InvokeResult:
    """Stub: pretend the upstream invocation returned exit_code=0."""
    return InvokeResult(exit_code=0)


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


@pytest.fixture
def two_chain_pdb_input(tmp_path: Path) -> Path:
    pdb = tmp_path / "two_chain_backbone.pdb"
    pdb.write_text(
        "HEADER    test\n"
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  GLY B   1       0.000   1.000   0.000  1.00  0.00           C\n"
        "ATOM      4  CA  ALA B   2       1.000   1.000   0.000  1.00  0.00           C\n"
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
    assert config.seed == 0


def test_config_rejects_non_positive_task_count() -> None:
    with pytest.raises(ValueError, match="task_count"):
        ProteinMPNNConfig(task_count=0)


def test_config_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ProteinMPNNConfig(temperature=0.0)


def test_config_rejects_zero_indexed_fixed_positions() -> None:
    with pytest.raises(ValueError, match="1-indexed"):
        ProteinMPNNConfig(fixed_positions=(0, 1, 2))


def test_excluded_from_executed_digest_constant_is_empty() -> None:
    """ProteinMPNN forwards every config field, so nothing is excluded."""
    assert EXCLUDED_FROM_EXECUTED_DIGEST == ()


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


def test_parse_fasta_sequences(tmp_path: Path) -> None:
    p = tmp_path / "out.fa"
    p.write_text(SAMPLE_FASTA)
    records = parse_fasta_sequences(p)
    assert len(records) == 4
    names = [name for name, _ in records]
    assert names == ["design_1,", "design_2,", "design_3,", "design_4,"]
    assert all(seq == "ACDEFGHIKLPQRSTVWY" for _, seq in records)


def test_parse_fasta_sequences_preserves_full_headers(tmp_path: Path) -> None:
    p = tmp_path / "out.fa"
    p.write_text(SAMPLE_FASTA)

    records = parse_fasta_sequences(p, full_header=True)

    assert [name for name, _ in records] == [
        "design_1, score=0.752",
        "design_2, score=0.701",
        "design_3, score=0.689",
        "design_4, score=0.650",
    ]


def test_parse_fasta_sequences_handles_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.fa"
    p.write_text("")
    assert parse_fasta_sequences(p) == []


def test_proteinmpnn_available_returns_false_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROTEINMPNN_BIN", "/nonexistent/proteinmpnn")
    assert proteinmpnn_available() is False


def test_container_uri_is_rejected_before_proteinmpnn_dispatch(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROTEINMPNN_BIN", "container://proteinmpnn:latest")
    runner = ProteinMPNNRunner(output_root=output_root)

    with pytest.raises(ValueError, match="container://"):
        runner.run(pdb_input, ProteinMPNNConfig())


def test_custom_binary_prefix_overrides_container_environment(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str] | None] = []

    def fake_invoke(**kwargs: Any) -> InvokeResult:
        captured.append(kwargs["binary_prefix"])
        return InvokeResult(exit_code=0, command=("proteinmpnn-local",))

    monkeypatch.setenv("PROTEINMPNN_BIN", "container://proteinmpnn:latest")
    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)

    result = ProteinMPNNRunner(
        output_root=output_root,
        binary_prefix=["proteinmpnn-local"],
    ).run(pdb_input, ProteinMPNNConfig())

    assert captured == [["proteinmpnn-local"]]
    assert result.execution_mode.value == "subprocess"
    assert result.provenance.command == ("proteinmpnn-local",)


def test_invoke_result_round_trips_through_dataclass() -> None:
    """The structured subprocess result must be JSON-safe."""
    import dataclasses
    import json

    payload = InvokeResult(
        exit_code=124,
        stderr_tail="Killed by signal 9",
        timed_out=True,
        failure_reason="timeout after 3600s",
    )
    assert json.loads(json.dumps(dataclasses.asdict(payload)))


def test_invoke_preserves_wrapper_flag_names_and_paths(tmp_path: Path) -> None:
    captured_argv: list[str] = []
    input_pdb = tmp_path / "backbone.pdb"
    output_dir = tmp_path / "output"

    def fake_run(cmd: list[str], **_: Any) -> mock_mod.Mock:
        captured_argv.extend(cmd)
        result = mock_mod.Mock()
        result.returncode = 0
        result.stderr = ""
        return result

    with mock_mod.patch("subprocess.run", side_effect=fake_run):
        exit_code = invoke(
            config_dict={
                "model_name": "v_48_020",
                "num_seq_per_target": "4",
                "sampling_temp": "0.2",
                "seed": "42",
                "ca_only": "True",
                "omit_AA": "CDF",
                "pdb_path": input_pdb.name,
            },
            input_pdb=input_pdb,
            output_dir=output_dir,
            binary_prefix=["proteinmpnn"],
            timeout_seconds=10,
        )

    assert exit_code == 0
    assert captured_argv == [
        "proteinmpnn",
        "--input_path",
        str(tmp_path),
        "--output_path",
        str(output_dir),
        "--batch_size",
        "1",
        "--model_name",
        "v_48_020",
        "--num_seq_per_target",
        "4",
        "--sampling_temp",
        "0.2",
        "--seed",
        "42",
        "--ca_only",
        "True",
        "--omit_AA",
        "CDF",
        "--pdb_path",
        "backbone.pdb",
    ]


# ---------------------------------------------------------------------------
# CLI translation — trigger / non-trigger pairs
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
    # weight-folder name.
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
    """Guard against re-introducing a folder-name default by mistake."""
    default = ProteinMPNNConfig().model_name
    upstream_checkpoints = {"v_48_002", "v_48_010", "v_48_020", "v_48_030"}
    assert default in upstream_checkpoints, (
        f"ProteinMPNNConfig().model_name defaults to {default!r}; "
        f"must be one of upstream's checkpoint prefixes "
        f"({sorted(upstream_checkpoints)}). A folder name silently "
        f"breaks checkpoint loading upstream."
    )


def test_config_to_cli_default_omits_ca_only(pdb_input: Path) -> None:
    """Default ``ca_only=False`` must NOT forward ``--ca_only True``."""
    cli = _stub_input(pdb_input)
    assert "ca_only" not in cli


def test_config_to_cli_with_ca_only(pdb_input: Path) -> None:
    cli = _config_to_cli(ProteinMPNNConfig(ca_only=True), pdb_input)
    assert cli["ca_only"] == "True"


def test_config_to_cli_default_omits_fixed_positions(pdb_input: Path) -> None:
    """Empty fixed_positions tuple must NOT forward ``--fixed_positions``."""
    cli = _stub_input(pdb_input)
    assert "fixed_positions" not in cli


def test_config_to_cli_handles_fixed_positions(pdb_input: Path) -> None:
    fixed_path = pdb_input.parent / "fixed_positions.jsonl"
    cli = _config_to_cli(
        ProteinMPNNConfig(fixed_positions=(3, 7)),
        pdb_input,
        fixed_positions_jsonl=fixed_path,
    )
    assert cli["fixed_positions_jsonl"] == str(fixed_path)
    assert "fixed_positions" not in cli


def test_runner_materializes_chain_aware_fixed_positions_and_exact_argv(
    output_root: Path,
    two_chain_pdb_input: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_invoke(**kwargs: Any) -> InvokeResult:
        captured.update(kwargs)
        (kwargs["output_dir"] / "out.fa").write_text(SAMPLE_FASTA)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)
    config = ProteinMPNNConfig(
        name="fixed",
        fixed_positions=(2, 1),
        extra={"pdb_path_chains": "B A"},
    )

    result = ProteinMPNNRunner(output_root=output_root, binary_prefix=["proteinmpnn"]).run(
        two_chain_pdb_input, config
    )

    fixed_path = output_root / "fixed" / two_chain_pdb_input.stem / "fixed_positions.jsonl"
    assert json.loads(fixed_path.read_text()) == {
        two_chain_pdb_input.stem: {"A": [1, 2], "B": [1, 2]}
    }
    assert captured["config_dict"]["fixed_positions_jsonl"] == str(fixed_path)
    assert "fixed_positions" not in captured["config_dict"]
    assert result.provenance.command == (
        "proteinmpnn",
        "--input_path",
        str(two_chain_pdb_input.parent),
        "--output_path",
        str(fixed_path.parent),
        "--batch_size",
        "1",
        "--model_name",
        "v_48_020",
        "--num_seq_per_target",
        "4",
        "--sampling_temp",
        "0.1",
        "--seed",
        "0",
        "--fixed_positions_jsonl",
        str(fixed_path),
        "--pdb_path",
        two_chain_pdb_input.name,
        "--pdb_path_chains",
        "B A",
    )
    fixed_artifacts = [a for a in result.provenance.artifacts if a.kind == "fixed_positions"]
    assert len(fixed_artifacts) == 1
    assert fixed_artifacts[0].path == str(fixed_path)
    assert fixed_artifacts[0].digest is not None


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({}, "pdb_path_chains"),
        ({"pdb_path_chains": "AB"}, "one-character"),
        ({"pdb_path_chains": "A A"}, "unique"),
    ],
)
def test_runner_fixed_positions_fails_closed_for_invalid_chains(
    output_root: Path,
    pdb_input: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: dict[str, str],
    message: str,
) -> None:
    invoked = False

    def fake_invoke(**_: Any) -> InvokeResult:
        nonlocal invoked
        invoked = True
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)

    with pytest.raises(ValueError, match=message):
        ProteinMPNNRunner(output_root=output_root).run(
            pdb_input,
            ProteinMPNNConfig(name="invalid-chain", fixed_positions=(1,), extra=extra),
        )

    assert invoked is False
    assert not (output_root / "invalid-chain" / pdb_input.stem / "fixed_positions.jsonl").exists()


def test_runner_fixed_positions_rejects_duplicates_and_out_of_range_positions(
    output_root: Path,
    pdb_input: Path,
) -> None:
    config = ProteinMPNNConfig(
        name="invalid-position",
        fixed_positions=(1, 1),
        extra={"pdb_path_chains": "A"},
    )
    with pytest.raises(ValueError, match="duplicate"):
        ProteinMPNNRunner(output_root=output_root).run(pdb_input, config)

    config = ProteinMPNNConfig(
        name="out-of-range",
        fixed_positions=(2,),
        extra={"pdb_path_chains": "A"},
    )
    with pytest.raises(ValueError, match="out of range"):
        ProteinMPNNRunner(output_root=output_root).run(pdb_input, config)


def test_runner_reuses_deterministic_fixed_positions_artifact(
    output_root: Path,
    two_chain_pdb_input: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        (output_dir / "out.fa").write_text(SAMPLE_FASTA)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)
    config = ProteinMPNNConfig(
        name="reuse-fixed",
        fixed_positions=(1,),
        extra={"pdb_path_chains": "A B"},
    )
    runner = ProteinMPNNRunner(output_root=output_root)

    first = runner.run(two_chain_pdb_input, config, force=True)
    fixed_path = Path(first.output_dir) / "fixed_positions.jsonl"
    first_bytes = fixed_path.read_bytes()
    second = runner.run(two_chain_pdb_input, config, force=True)

    assert Path(second.output_dir) / "fixed_positions.jsonl" == fixed_path
    assert fixed_path.read_bytes() == first_bytes
    assert first.provenance.artifacts == second.provenance.artifacts


def test_config_to_cli_default_omits_omit_AA(pdb_input: Path) -> None:
    """Default empty ``omit_aa`` must NOT forward ``--omit_AA``."""
    cli = _stub_input(pdb_input)
    assert "omit_AA" not in cli


def test_config_to_cli_handles_omit_aa(pdb_input: Path) -> None:
    cli = _config_to_cli(ProteinMPNNConfig(omit_aa="CDF"), pdb_input)
    assert cli["omit_AA"] == "CDF"


# ---------------------------------------------------------------------------
# Runner behaviour
# ---------------------------------------------------------------------------


def test_runner_dry_run_does_not_invoke(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoked: list[dict[str, Any]] = []

    def fake_invoke(**_: Any) -> InvokeResult:
        invoked.append({})
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    result = runner.run(pdb_input, dry_run=True)
    assert invoked == []
    assert result.records == ()


def test_runner_idempotent_when_fasta_exists(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    name = "idem"
    design_dir = output_root / name / pdb_input.stem
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "out.fa").write_text(SAMPLE_FASTA)

    result = runner.run(pdb_input, ProteinMPNNConfig(name=name))
    assert result.skipped == 4
    assert result.succeeded == 4


def test_runner_reads_stock_seq_directory_layout(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", _fake_invoke_ok)
    config = ProteinMPNNConfig(name="stock")
    design_dir = output_root / config.name / pdb_input.stem
    (design_dir / "seqs").mkdir(parents=True)
    (design_dir / "seqs" / f"{pdb_input.stem}.fa").write_text(SAMPLE_FASTA)

    result = ProteinMPNNRunner(output_root=output_root).run(pdb_input, config)

    assert result.status.value == "cached"
    assert result.skipped == 4
    assert {Path(record.path).parent.name for record in result.records} == {"seqs"}


@pytest.mark.parametrize("cached", [False, True])
def test_runner_excludes_native_record_from_stock_fasta(
    output_root: Path,
    pdb_input: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cached: bool,
) -> None:
    config = ProteinMPNNConfig(name=f"native-{cached}", task_count=2)
    design_dir = output_root / config.name / pdb_input.stem

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        (output_dir / "seqs").mkdir(parents=True)
        (output_dir / "seqs" / "design_8.fa").write_text(NATIVE_AND_SAMPLE_FASTA)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)
    if cached:
        fake_invoke(output_dir=design_dir)

    result = ProteinMPNNRunner(output_root=output_root).run(pdb_input, config)

    assert result.succeeded == config.task_count
    assert result.failed == 0
    assert result.skipped == (config.task_count if cached else 0)
    assert [record.sequence for record in result.records] == [
        "VKTAYIAKQRQISFVKSHFSRQDILDLI",
        "AKTAYIAKQRQISFVKSHFSRQDILDLI",
    ]
    assert result.provenance.task_count == config.task_count


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "missing_metadata",
    ["fixed_chains=[], ", "designed_chains=['A','B','C'], "],
)
def test_runner_rejects_malformed_native_record_from_stock_fasta(
    output_root: Path,
    pdb_input: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cached: bool,
    missing_metadata: str,
) -> None:
    config = ProteinMPNNConfig(name=f"malformed-native-{cached}", task_count=2)
    design_dir = output_root / config.name / pdb_input.stem

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        (output_dir / "seqs").mkdir(parents=True)
        malformed_fasta = NATIVE_AND_SAMPLE_FASTA.replace(missing_metadata, "")
        (output_dir / "seqs" / "design_8.fa").write_text(malformed_fasta)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)
    if cached:
        fake_invoke(output_dir=design_dir)

    with pytest.raises(ValueError, match="malformed ProteinMPNN FASTA"):
        ProteinMPNNRunner(output_root=output_root).run(pdb_input, config)


def test_runner_accepts_stock_samples_without_native_record(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ProteinMPNNConfig(name="sample-only-stock", task_count=2)

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        (output_dir / "seqs").mkdir(parents=True)
        (output_dir / "seqs" / "design_8.fa").write_text(SAMPLE_ONLY_STOCK_FASTA)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)

    result = ProteinMPNNRunner(output_root=output_root).run(pdb_input, config)

    assert [record.sequence for record in result.records] == [
        "VKTAYIAKQRQISFVKSHFSRQDILDLI",
        "AKTAYIAKQRQISFVKSHFSRQDILDLI",
    ]


def test_runner_cache_counters_reflect_malformed_records(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ProteinMPNNConfig(name="malformed-cache")
    design_dir = output_root / config.name / pdb_input.stem
    design_dir.mkdir(parents=True)
    fasta = design_dir / "cached.fa"
    fasta.write_text(SAMPLE_FASTA)
    monkeypatch.setattr(
        "biolab_runners.proteinmpnn.runner.parse_fasta_sequences",
        mock_mod.Mock(side_effect=OSError("corrupt FASTA")),
    )

    result = ProteinMPNNRunner(output_root=output_root).run(pdb_input, config)

    assert result.status.value == "malformed"
    assert result.succeeded == 0
    assert result.failed == 1
    assert result.skipped == 1


def test_runner_fake_upstream_e2e_reads_stock_layout(output_root: Path, pdb_input: Path) -> None:
    script = output_root.parent / "fake_proteinmpnn.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "args = sys.argv[1:]\n"
        "out = Path(args[args.index('--output_path') + 1]) / 'seqs'\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        f"(out / 'stock.fa').write_text({SAMPLE_FASTA!r})\n"
    )

    result = ProteinMPNNRunner(
        output_root=output_root,
        binary_prefix=[sys.executable, str(script)],
    ).run(pdb_input, ProteinMPNNConfig(name="e2e"))

    assert result.status.value == "succeeded"
    assert result.succeeded == 4
    assert result.provenance.executed is True
    assert result.provenance.command[:2] == (sys.executable, str(script))


def test_proteinmpnn_cli_does_not_abbreviate_flags() -> None:
    from biolab_runners.proteinmpnn.cli import translate_runner_args

    translated = translate_runner_args(
        [
            "--input_path",
            "/tmp/input",
            "--output_path",
            "/tmp/output",
            "--pdb_path",
            "backbone.pdb",
            "--sampling_te",
            "0.2",
        ]
    )

    assert translated[-2:] == ["--sampling_te", "0.2"]


def test_proteinmpnn_cli_forwards_fixed_positions_jsonl() -> None:
    from biolab_runners.proteinmpnn.cli import translate_runner_args

    translated = translate_runner_args(
        [
            "--input_path",
            "/tmp/input",
            "--output_path",
            "/tmp/output",
            "--pdb_path",
            "backbone.pdb",
            "--fixed_positions_jsonl",
            "/tmp/output/fixed_positions.jsonl",
            "--pdb_path_chains",
            "A B",
        ]
    )

    assert translated[-4:] == [
        "--fixed_positions_jsonl",
        "/tmp/output/fixed_positions.jsonl",
        "--pdb_path_chains",
        "A B",
    ]


def test_proteinmpnn_cli_rejects_legacy_fixed_positions() -> None:
    from biolab_runners.proteinmpnn.cli import translate_runner_args

    with pytest.raises(ValueError, match="fixed_positions is unsupported"):
        translate_runner_args(
            [
                "--input_path",
                "/tmp/input",
                "--output_path",
                "/tmp/output",
                "--pdb_path",
                "backbone.pdb",
                "--fixed_positions",
                "1,2",
            ]
        )


def test_runner_force_re_runs(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        calls.append(output_dir)
        (output_dir / "out.fa").write_text(SAMPLE_FASTA)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)
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
    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        (output_dir / "out.fa").write_text(SAMPLE_FASTA)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig(task_count=4))
    result = runner.run(pdb_input)
    assert result.succeeded == 4
    assert result.failed == 0
    assert all(r.sequence == "ACDEFGHIKLPQRSTVWY" for r in result.records)


def test_runner_records_failed_for_unparseable_fasta(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        target = output_dir / "out.fa"
        target.write_text("placeholder")
        target.unlink()
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    result = runner.run(pdb_input)
    assert all(r.status == DesignRecordStatus.FAILED for r in result.records)


def test_runner_propagates_nonzero_exit_code(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "biolab_runners.proteinmpnn.runner._invoke_with_metadata",
        lambda **_: InvokeResult(exit_code=9),
    )
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

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        (output_dir / "out.fa").write_text(SAMPLE_FASTA)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    results = runner.run_batch(inputs)
    assert len(results) == 3
    assert all(r.succeeded == 4 for r in results)


def test_runner_requires_config() -> None:
    runner = ProteinMPNNRunner(output_root=Path("/tmp"))
    with pytest.raises(ValueError, match="ProteinMPNNConfig is required"):
        runner.run(Path("/tmp/x.pdb"))


# ---------------------------------------------------------------------------
# S2 provenance (reproducibility)
# ---------------------------------------------------------------------------


def test_runner_records_honest_provenance_with_canonical_output(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real invocation attaches ProvenanceMetadata with the canonical FASTA
    sequences (the S2 contract — captured *before* any downstream D-residue
    substitution).

    For ProteinMPNN the runner DOES forward ``--seed``, so
    ``base_seed == requested_seed``.
    """
    from biolab_runners.provenance import compute_file_digest

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        (output_dir / "out.fa").write_text(SAMPLE_FASTA)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root)
    config = ProteinMPNNConfig(name="prov", seed=11, task_count=4)
    result = runner.run(pdb_input, config, image_digest=VALID_OCI_DIGEST)
    prov = result.provenance

    assert prov.model_identifier == "v_48_020"
    assert prov.temperature == 0.1
    assert prov.image_digest == VALID_OCI_DIGEST
    assert prov.source_backbone_digest == compute_file_digest(pdb_input)
    assert prov.base_seed == 11
    assert prov.requested_seed == 11
    assert prov.task_count == 4
    assert prov.rng_intent == "single-stream"
    assert prov.exit_code == 0
    assert prov.failure_reason == ""
    assert prov.executed is True
    assert prov.cache_hit is False
    # Canonical output is the raw FASTA, *before* D substitutions.
    assert prov.canonical_output == ("ACDEFGHIKLPQRSTVWY",) * 4


def test_runner_canonical_output_records_sequences_only(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canonical output is the SEQUENCE strings (not FASTA names / scores).
    This is the S2 contract — downstream ``chem_001`` operates on sequence
    strings, and the audit must be able to reconstruct them post-substitution."""

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        (output_dir / "out.fa").write_text(SAMPLE_FASTA)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    result = runner.run(pdb_input)
    assert all(isinstance(s, str) and len(s) > 0 for s in result.provenance.canonical_output)


def test_runner_provenance_does_not_contain_per_task_seeds(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2 honesty: the manifest must not fabricate per-task seeds."""
    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    result = runner.run(pdb_input, ProteinMPNNConfig(name="x", seed=42, task_count=8))
    assert not hasattr(result.provenance, "per_task_seeds")
    assert result.provenance.base_seed == 42
    assert result.provenance.task_count == 8
    assert result.provenance.rng_intent == "single-stream"


def test_runner_executed_config_digest_changes_with_seed(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ProteinMPNN DOES forward ``--seed`` so changing seed flips BOTH
    requested and executed digests. (RFdiffusion is the inverse case.)"""
    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = ProteinMPNNRunner(output_root=output_root)

    a = runner.run(pdb_input, ProteinMPNNConfig(name="x", seed=1)).provenance
    b = runner.run(pdb_input, ProteinMPNNConfig(name="x", seed=999)).provenance

    assert a.executed_config_digest != b.executed_config_digest
    assert a.requested_config_digest != b.requested_config_digest
    # base_seed carries the forwarded seed; both differ.
    assert a.base_seed == 1 and b.base_seed == 999


def test_runner_cache_hit_records_honest_cache_provenance(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a cache hit, ``executed=False``, ``cache_hit=True``,
    ``executed_config_digest=None`` — the runner does not know which
    prior call produced the existing files."""
    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", _fake_invoke_ok)
    name = "idem-prov"
    design_dir = output_root / name / pdb_input.stem
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "out.fa").write_text(SAMPLE_FASTA)

    runner = ProteinMPNNRunner(output_root=output_root)
    result = runner.run(pdb_input, ProteinMPNNConfig(name=name, seed=9))
    prov = result.provenance

    assert prov.cache_hit is True
    assert prov.executed is False
    assert prov.executed_config_digest is None
    assert prov.requested_config_digest != ""
    assert prov.requested_seed == 9
    # Canonical output is re-derived from the cached FASTA — the audit
    # can confirm "the file on disk is the canonical output".
    assert prov.canonical_output == ("ACDEFGHIKLPQRSTVWY",) * 4
    assert result.skipped == 4


def test_runner_dry_run_records_requested_digest_only(output_root: Path, pdb_input: Path) -> None:
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    result = runner.run(pdb_input, ProteinMPNNConfig(name="dry", seed=5), dry_run=True)
    prov = result.provenance
    assert prov.executed is False
    assert prov.cache_hit is False
    assert prov.executed_config_digest is None
    assert prov.requested_config_digest != ""
    # ProteinMPNN forwards --seed, so base_seed IS the requested seed
    # even in dry_run (the runner "would have" forwarded it).
    assert prov.base_seed == 5
    assert prov.requested_seed == 5


def test_runner_propagates_nonzero_exit_code_in_provenance(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-zero exit code surfaces in provenance — audit trail."""
    monkeypatch.setattr(
        "biolab_runners.proteinmpnn.runner._invoke_with_metadata",
        lambda **_: InvokeResult(exit_code=9),
    )
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    result = runner.run(pdb_input)
    assert result.provenance.exit_code == 9


def test_runner_records_timeout_in_provenance(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subprocess timeout surfaces as exit_code=124 and a deterministic
    failure_reason in the provenance manifest."""
    fake_result = InvokeResult(
        exit_code=124,
        stderr_tail="",
        timed_out=True,
        failure_reason="timeout after 3600s",
    )

    monkeypatch.setattr(
        "biolab_runners.proteinmpnn.runner._invoke_with_metadata",
        lambda **_: fake_result,
    )
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    result = runner.run(pdb_input)
    assert result.provenance.exit_code == 124
    assert result.provenance.failure_reason == "timeout after 3600s"


def test_runner_captures_stderr_tail_in_provenance(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The upstream stderr tail flows into the manifest for audit."""
    fake_result = InvokeResult(
        exit_code=1,
        stderr_tail="RuntimeError: cuda out of memory",
        timed_out=False,
        failure_reason="RuntimeError: cuda out of memory",
    )

    monkeypatch.setattr(
        "biolab_runners.proteinmpnn.runner._invoke_with_metadata",
        lambda **_: fake_result,
    )
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    result = runner.run(pdb_input)
    assert result.provenance.exit_code == 1
    assert result.provenance.stderr_tail == "RuntimeError: cuda out of memory"
    assert result.provenance.failure_reason == "RuntimeError: cuda out of memory"


def test_runner_run_batch_threads_image_digest(
    output_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_batch threads the caller's image_digest through to every result."""
    inputs = []
    for i in range(2):
        pdb = tmp_path / f"input_{i}.pdb"
        pdb.write_text("HEADER\nATOM      1  CA  GLY A   1       0.000   0.000   0.000\nEND\n")
        inputs.append(pdb)

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        (output_dir / "out.fa").write_text(SAMPLE_FASTA)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", fake_invoke)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    results = runner.run_batch(inputs, image_digest=VALID_OCI_DIGEST)
    assert len(results) == 2
    assert all(r.provenance.image_digest == VALID_OCI_DIGEST for r in results)


def test_runner_normalises_image_digest_to_oci_form(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both bare-hex and OCI-prefixed forms must be normalised to the OCI form
    BEFORE any subprocess work."""
    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    config = ProteinMPNNConfig(name="img-norm")

    oci_result = runner.run(pdb_input, config, image_digest=VALID_OCI_DIGEST)
    bare_result = runner.run(pdb_input, config, image_digest=VALID_BARE_DIGEST)

    assert oci_result.provenance.image_digest == VALID_OCI_DIGEST
    assert bare_result.provenance.image_digest == VALID_OCI_DIGEST  # normalised


def test_runner_equivalent_rerun_produces_equivalent_provenance(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2 equivalence: same inputs → byte-identical provenance.to_dict()."""
    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = ProteinMPNNRunner(output_root=output_root)
    config = ProteinMPNNConfig(name="eq", seed=42, temperature=0.15)
    first = runner.run(pdb_input, config, image_digest=VALID_OCI_DIGEST).provenance.to_dict()
    second = runner.run(pdb_input, config, image_digest=VALID_OCI_DIGEST).provenance.to_dict()
    assert first == second


def test_runner_validates_malformed_image_digest(
    output_root: Path, pdb_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed image digest must raise ValueError, not silently flow through."""
    monkeypatch.setattr("biolab_runners.proteinmpnn.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = ProteinMPNNRunner(output_root=output_root, config=ProteinMPNNConfig())
    with pytest.raises(ValueError, match="image_digest must be"):
        runner.run(pdb_input, image_digest="not-a-digest")
