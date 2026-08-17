"""Tests for the RFdiffusion runner.

The runner is a thin subprocess wrapper; tests inject a fake
``invoke`` via monkeypatch so no real RFdiffusion install is needed
during CI.
"""

from __future__ import annotations

import json
import logging
import unittest.mock as mock_mod
from pathlib import Path
from typing import Any

import pytest
from biolab_runners.provenance import (
    RNG_INTENT_NON_DETERMINISTIC,
    RNG_INTENT_PER_DESIGN_INDEX,
    compute_config_digest,
    compute_file_digest,
)
from biolab_runners.rfdiffusion import (
    RecordData,
    RecordDataStatus,
    rfdiffusion_available,
)
from biolab_runners.rfdiffusion.config import (
    RESERVED_CANONICAL_KEYS,
    RFdiffusionConfig,
)
from biolab_runners.rfdiffusion.runner import (
    RFdiffusionRunner,
    _cache_identity_token,
    _config_to_cli,
    _executed_digest_excluded_fields,
    _parse_output_dir,
)
from biolab_runners.rfdiffusion.utils import (
    InvokeResult,
    _invoke_with_metadata,
    parse_backbone_pdb,
)
from biolab_runners.rfdiffusion.utils import (
    RecordDataStatus as _Status,
)

SAMPLE_PDB = """\
HEADER    RFdiffusion design 0
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.500   0.000   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       3.000  -1.000   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       4.000   1.000   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       5.000   1.000   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       6.000   1.000   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       7.000   0.000   0.000  1.00  0.00           O
ATOM      9  N   GLY A   3       7.500   2.000   0.000  1.00  0.00           N
ATOM     10  CA  GLY A   3       8.500   2.000   0.000  1.00  0.00           C
ATOM     11  C   GLY A   3       9.500   2.000   0.000  1.00  0.00           C
ATOM     12  O   GLY A   3      10.000   1.000   0.000  1.00  0.00           O
TER
END
"""

#: A canonical image digest in OCI form (the form the runner normalises to).
VALID_OCI_DIGEST = "sha256:" + "ab" * 32  # 64 hex chars
#: The same digest in bare-hex form — must be accepted and normalised to the OCI form.
VALID_BARE_DIGEST = "ab" * 32
#: A second, distinct canonical digest — used to prove image-bound cache isolation.
OTHER_OCI_DIGEST = "sha256:" + "cd" * 32


def _fake_invoke_ok(**_: Any) -> InvokeResult:
    """Stub: pretend the upstream invocation returned exit_code=0."""
    return InvokeResult(exit_code=0)


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    return tmp_path / "rfdiffusion"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_defaults_pass_validation() -> None:
    config = RFdiffusionConfig()
    assert config.mode == "linear"
    assert config.length_min == 14
    assert config.length_max == 18
    assert config.task_count == 1000
    assert config.seed == 0
    assert config.deterministic is True
    assert config.checkpoint == "RFdiffusion"


def test_config_rejects_inverted_length_range() -> None:
    with pytest.raises(ValueError, match="length range invalid"):
        RFdiffusionConfig(length_min=18, length_max=14)


def test_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        RFdiffusionConfig(mode="wat")


def test_disulfide_mode_requires_pairs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RFdiffusionConfig(mode="disulfide")


def test_linear_mode_rejects_disulfide_pairs() -> None:
    with pytest.raises(ValueError, match="disulfide_pairs"):
        RFdiffusionConfig(mode="linear", disulfide_pairs=((3, 9),))


def test_config_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed must be"):
        RFdiffusionConfig(seed=-1)


def test_config_rejects_empty_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint must be"):
        RFdiffusionConfig(checkpoint="")


def test_executed_digest_exclusion_is_mode_dependent() -> None:
    """S2 contract: ``seed`` maps to ``inference.design_startnum`` which is
    forwarded ONLY when ``deterministic=True``.

    * deterministic — nothing excluded: a seed-only change flips the
      executed digest (the seed changes what upstream ran).
    * non-deterministic — ``seed`` excluded: the base seed is not
      forwarded, so a seed-only change must NOT flip the executed digest.
    """
    assert _executed_digest_excluded_fields(RFdiffusionConfig()) == ()
    assert _executed_digest_excluded_fields(RFdiffusionConfig(seed=42)) == ()
    assert _executed_digest_excluded_fields(RFdiffusionConfig(deterministic=False)) == ("seed",)


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


def test_parse_backbone_pdb_extracts_three_residues(tmp_path: Path) -> None:
    pdb = tmp_path / "design.pdb"
    pdb.write_text(SAMPLE_PDB)
    assert parse_backbone_pdb(pdb) == "GAG"


def test_parse_backbone_pdb_handles_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        parse_backbone_pdb(tmp_path / "absent.pdb")


def test_record_data_to_dict_round_trip() -> None:
    record = RecordData(index=2, path="/tmp/design.pdb", sequence="GAG")
    payload = record.to_dict()
    assert payload["index"] == "2"
    assert payload["sequence"] == "GAG"
    assert payload["status"] == RecordDataStatus.SUCCEEDED


def test_rfdiffusion_available_returns_false_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RFDIFFUSION_BIN", "/nonexistent/rfdiffusion")
    assert rfdiffusion_available() is False


def test_invoke_with_metadata_returns_structured_result() -> None:
    """The internal helper exposes exit code + stderr tail + timeout flag."""
    result = InvokeResult(
        exit_code=124,
        stderr_tail="Killed by signal 9",
        timed_out=True,
        failure_reason="timeout after 3600s",
    )
    assert result.exit_code == 124
    assert result.timed_out is True
    assert result.failure_reason == "timeout after 3600s"
    assert result.stderr_tail == "Killed by signal 9"


# ---------------------------------------------------------------------------
# CLI translation — trigger / non-trigger pairs
# ---------------------------------------------------------------------------


def test_config_to_cli_linear_default_triggers_no_cyclic() -> None:
    """Default linear mode must NOT include cyclic / cyc_chains flags."""
    cli = _config_to_cli(RFdiffusionConfig())
    assert cli["contigmap.contigs"] == "14-18"
    assert cli["inference.num_designs"] == "1000"
    assert "inference.cyclic" not in cli
    assert "inference.cyc_chains" not in cli


def test_config_to_cli_linear_default_triggers_deterministic() -> None:
    """Default config has ``deterministic=True`` so the flag IS forwarded."""
    cli = _config_to_cli(RFdiffusionConfig())
    assert cli["inference.deterministic"] == "True"


def test_config_to_cli_non_deterministic_omits_deterministic_flag() -> None:
    """``deterministic=False`` must NOT forward ``inference.deterministic``."""
    cli = _config_to_cli(RFdiffusionConfig(deterministic=False))
    assert "inference.deterministic" not in cli


def test_config_to_cli_forwards_design_startnum() -> None:
    """Deterministic mode MUST forward ``inference.design_startnum=<seed>`` —
    the supported external base for upstream's per-design seeding — plus
    ``inference.deterministic=True``, alongside ``inference.num_designs``."""
    cli = _config_to_cli(RFdiffusionConfig(seed=42, task_count=5))
    assert cli["inference.design_startnum"] == "42"
    assert cli["inference.num_designs"] == "5"
    assert cli["inference.deterministic"] == "True"
    assert "inference.seed" not in cli


def test_config_to_cli_non_deterministic_omits_design_startnum_and_deterministic_flags() -> None:
    """``deterministic=False`` must NOT forward ``inference.design_startnum``
    OR ``inference.deterministic`` — upstream uses system entropy, so a
    forwarded base seed would be inert and the manifest records
    ``rng_intent="non-deterministic"`` with ``base_seed=None``."""
    cli = _config_to_cli(RFdiffusionConfig(seed=42, deterministic=False))
    assert "inference.design_startnum" not in cli
    assert "inference.deterministic" not in cli
    assert "inference.seed" not in cli
    assert cli["inference.num_designs"] == "1000"  # still forwarded


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # default: deterministic
        {"seed": 7, "task_count": 3},
        {"deterministic": False},
        {"mode": "head_to_tail"},
        {"mode": "disulfide", "disulfide_pairs": ((3, 9),)},
    ],
)
def test_config_to_cli_never_emits_inference_seed(kwargs: dict[str, Any]) -> None:
    """Stock upstream has no ``inference.seed`` Hydra key — the runner must
    never emit it, in any mode."""
    cli = _config_to_cli(RFdiffusionConfig(**kwargs))
    assert "inference.seed" not in cli
    assert not any(key.startswith("inference.seed") for key in cli)


def test_config_to_cli_head_to_tail_triggers_cyclic_with_chain_a() -> None:
    cli = _config_to_cli(RFdiffusionConfig(mode="head_to_tail"))
    assert cli["inference.cyclic"] == "True"
    assert cli["inference.cyc_chains"] == "a"


def test_config_to_cli_disulfide_triggers_cyclic_with_pairs() -> None:
    cli = _config_to_cli(RFdiffusionConfig(mode="disulfide", disulfide_pairs=((3, 9), (5, 12))))
    assert cli["inference.cyclic"] == "True"
    assert cli["inference.cyc_chains"] == "3,9,5,12"


def test_config_to_cli_hotspots_triggers_ppi_hotspot_res() -> None:
    cli = _config_to_cli(RFdiffusionConfig(hotspots=("A12", "B17")))
    assert cli["ppi.hotspot_res"] == "A12,B17"


def test_config_to_cli_no_hotspots_omits_ppi_hotspot_res() -> None:
    """Empty hotspots tuple must NOT add the ``ppi.hotspot_res`` flag."""
    cli = _config_to_cli(RFdiffusionConfig(hotspots=()))
    assert "ppi.hotspot_res" not in cli


def test_config_to_cli_does_not_forward_temperature_flag() -> None:
    """RFdiffusion does not expose a single ``temperature`` parameter —
    forwarding to ``diffusion.noise_scale_ca`` would silently change
    upstream behaviour. The runner must NOT forward any temperature field."""
    cli = _config_to_cli(RFdiffusionConfig(seed=0))
    assert "diffusion.noise_scale_ca" not in cli
    assert "temperature" not in cli


def test_config_to_cli_extra_kwargs_are_forwarded() -> None:
    """The ``extra`` mapping is forwarded verbatim so callers can add
    upstream-supported kwargs without changing the dataclass."""
    cli = _config_to_cli(RFdiffusionConfig(extra={"inference.noise_scale_ca": "0.5"}))
    assert cli["inference.noise_scale_ca"] == "0.5"


@pytest.mark.parametrize(
    "reserved_key",
    [
        "inference.design_startnum",
        "inference.num_designs",
        "inference.deterministic",
        "inference.cyclic",
        "inference.cyc_chains",
        "contigmap.contigs",
        "ppi.hotspot_res",
    ],
)
def test_config_rejects_extra_override_of_reserved_canonical_keys(reserved_key: str) -> None:
    """Fail closed: ``extra`` must not silently override a canonical Hydra key
    the runner emits itself — the conflict raises ValueError at construction."""
    with pytest.raises(ValueError, match="reserved canonical keys"):
        RFdiffusionConfig(extra={reserved_key: "sneaky"})


def test_config_rejects_unsupported_inference_seed_extra_key() -> None:
    """``inference.seed`` does not exist in stock upstream; passing it via
    ``extra`` must fail with a clear error naming the supported alternative —
    never silently forward a key upstream cannot parse."""
    with pytest.raises(ValueError, match=r"inference\.seed.*not supported.*design_startnum"):
        RFdiffusionConfig(extra={"inference.seed": "42"})


def test_config_rejects_non_string_extra_keys() -> None:
    """Non-string ``extra`` keys would produce unusable CLI flags — reject
    them at construction."""
    with pytest.raises(ValueError, match="extra keys must be strings"):
        RFdiffusionConfig(extra={123: "x"})  # type: ignore[dict-item]


def test_reserved_and_unsupported_key_sets_are_disjoint_and_exhaustive() -> None:
    """Every key the runner emits is reserved; the unsupported set contains
    keys upstream lacks (notably ``inference.seed``). Callers can inspect
    both sets."""
    from biolab_runners.rfdiffusion.config import UNSUPPORTED_UPSTREAM_KEYS

    emitted = _config_to_cli(RFdiffusionConfig()).keys()
    assert all(key in RESERVED_CANONICAL_KEYS for key in emitted)
    assert "inference.design_startnum" in RESERVED_CANONICAL_KEYS
    assert "inference.num_designs" in RESERVED_CANONICAL_KEYS
    assert "inference.deterministic" in RESERVED_CANONICAL_KEYS
    assert "contigmap.contigs" in RESERVED_CANONICAL_KEYS
    assert "inference.seed" not in RESERVED_CANONICAL_KEYS  # not a key we emit
    assert "inference.seed" in UNSUPPORTED_UPSTREAM_KEYS
    assert not set(RESERVED_CANONICAL_KEYS) & set(UNSUPPORTED_UPSTREAM_KEYS)


def test_invoke_with_metadata_receives_forwarded_design_startnum(tmp_path: Path) -> None:
    """End-to-end CLI path: the base seed lands in the argv handed to
    ``subprocess.run`` as ``--inference.design-startnum <value>`` (underscores
    are hyphenated for argv), and ``inference.seed`` is never emitted."""
    captured_argv: list[str] = []

    def fake_run(cmd: list[str], **_: Any) -> mock_mod.Mock:
        captured_argv.extend(cmd)
        result = mock_mod.Mock()
        result.returncode = 0
        result.stderr = ""
        return result

    config_dict = _config_to_cli(RFdiffusionConfig(seed=42))
    with mock_mod.patch("biolab_runners.rfdiffusion.utils.subprocess.run", side_effect=fake_run):
        result = _invoke_with_metadata(
            config_dict=config_dict,
            output_dir=tmp_path,
            binary_prefix=["fake-rfdiffusion"],
        )

    assert result.exit_code == 0
    assert "--inference.design-startnum" in captured_argv
    assert captured_argv[captured_argv.index("--inference.design-startnum") + 1] == "42"
    # Underscores in Hydra keys are hyphenated for argv (num_designs -> num-designs).
    assert "--inference.num-designs" in captured_argv
    assert "--inference.deterministic" in captured_argv
    assert "--inference.seed" not in captured_argv


def test_upstream_deterministic_per_design_seed_semantics() -> None:
    """Encode the stock upstream contract (RosettaCommons/RFdiffusion,
    ``scripts/run_inference.py``):

        if conf.inference.deterministic:
            make_deterministic()
        for i_des in range(design_startnum, design_startnum + num_designs):
            if conf.inference.deterministic:
                make_deterministic(i_des)
            out_prefix = f"{output_prefix}_{i_des}"   # -> <name>_<i_des>.pdb

    The runner maps ``RFdiffusionConfig.seed`` → ``inference.design_startnum``
    and ``task_count`` → ``inference.num_designs``, so the per-design RNG
    seeds are ``seed .. seed + task_count - 1`` and the emitted output
    indices/names start at ``seed``.
    """
    seed, task_count = 42, 3
    cli = _config_to_cli(RFdiffusionConfig(seed=seed, task_count=task_count))
    assert cli["inference.deterministic"] == "True"

    # Mirror upstream's loop verbatim.
    design_startnum = int(cli["inference.design_startnum"])
    num_designs = int(cli["inference.num_designs"])
    per_design_seeds = list(range(design_startnum, design_startnum + num_designs))
    output_names = [f"design_{i_des}.pdb" for i_des in per_design_seeds]

    assert per_design_seeds == [42, 43, 44]
    assert output_names == ["design_42.pdb", "design_43.pdb", "design_44.pdb"]
    # Provenance encodes the range via base_seed + task_count — no list field.
    assert len(per_design_seeds) == task_count
    assert per_design_seeds[-1] == seed + task_count - 1


# ---------------------------------------------------------------------------
# Runner behaviour (with a fake invoke)
# ---------------------------------------------------------------------------


def test_runner_dry_run_does_not_invoke(output_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    invoked: list[dict[str, Any]] = []

    def fake_invoke(*, config_dict: dict[str, Any], output_dir: Path, **_: Any) -> InvokeResult:
        invoked.append({"config_dict": config_dict, "output_dir": output_dir})
        output_dir.mkdir(parents=True, exist_ok=True)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="dry"), dry_run=True)
    assert invoked == []
    assert result.exit_code == 0
    assert result.succeeded == 0


def test_runner_idempotent_when_output_exists(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = RFdiffusionRunner(output_root=output_root)
    name = "idem"
    # The cache is keyed by the canonical identity (config + image + source):
    # outputs must live at <output_root>/<name>/<identity>/ for the idempotent
    # path to hit. No image and no target → identity over the config alone.
    config = RFdiffusionConfig(name=name)
    design_dir = output_root / name / _cache_identity_token(config, image_digest=None)
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "design_0.pdb").write_text(SAMPLE_PDB)

    result = runner.run(config)
    assert result.skipped == 1
    assert result.succeeded == 1
    assert result.provenance.cache_hit is True
    assert result.provenance.executed is False


def test_runner_force_re_runs(output_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        calls.append(output_dir)
        (output_dir / "design_0.pdb").write_text(SAMPLE_PDB)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    name = "force"
    config = RFdiffusionConfig(name=name)
    design_dir = output_root / name / _cache_identity_token(config, image_digest=None)
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "design_0.pdb").write_text(SAMPLE_PDB)

    result = runner.run(config, force=True)
    assert calls == [design_dir]
    assert result.exit_code == 0


def test_runner_records_per_design(output_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        for i in range(3):
            (output_dir / f"design_{i}.pdb").write_text(SAMPLE_PDB)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="batch"))
    assert result.succeeded == 3
    assert result.failed == 0
    assert {r.sequence for r in result.records} == {"GAG"}


def test_runner_handles_unparseable_pdb(output_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing parse should record a FAILED entry, not crash the runner."""

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        (output_dir / "good.pdb").write_text(SAMPLE_PDB)
        ghost = output_dir / "ghost.pdb"
        ghost.write_text(SAMPLE_PDB)
        ghost.unlink()
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="broken"))
    assert all(r.path for r in result.records)
    assert not any("ghost" in r.path for r in result.records)
    assert any(r.status == _Status.SUCCEEDED for r in result.records)


def test_runner_propagates_nonzero_exit_code(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "biolab_runners.rfdiffusion.runner._invoke_with_metadata",
        lambda **_: InvokeResult(exit_code=7),
    )
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="failure"))
    assert result.exit_code == 7
    assert result.succeeded == 0


def test_runner_requires_config() -> None:
    runner = RFdiffusionRunner(output_root=Path("/tmp"))
    with pytest.raises(ValueError, match="RFdiffusionConfig is required"):
        runner.run()


# ---------------------------------------------------------------------------
# Digest-keyed output layout / cache isolation
# ---------------------------------------------------------------------------


def test_runner_design_dir_is_keyed_by_canonical_identity(
    output_root: Path, tmp_path: Path
) -> None:
    """The on-disk layout is ``<output_root>/<name>/<identity>/`` where the
    identity binds the config digest, the normalized image digest (when
    supplied), and the target content digest (when the file exists)."""
    runner = RFdiffusionRunner(output_root=output_root)
    name = "layout"

    a = RFdiffusionConfig(name=name, seed=1)
    b = RFdiffusionConfig(name=name, seed=2)

    assert runner._design_dir(a) == output_root / name / _cache_identity_token(a, image_digest=None)
    assert runner._design_dir(a) != runner._design_dir(b)
    # Image digest binds: bare-hex and OCI forms of the SAME digest agree;
    # a different digest isolates.
    assert runner._design_dir(a, image_digest=VALID_BARE_DIGEST) == runner._design_dir(
        a, image_digest=VALID_OCI_DIGEST
    )
    assert runner._design_dir(a, image_digest=None) != runner._design_dir(
        a, image_digest=VALID_OCI_DIGEST
    )
    assert runner._design_dir(a, image_digest=VALID_OCI_DIGEST) != runner._design_dir(
        a, image_digest=OTHER_OCI_DIGEST
    )
    # Target content binds: same path, different bytes → different identity.
    target = tmp_path / "t.pdb"
    target.write_text(SAMPLE_PDB)
    c1 = RFdiffusionConfig(name=name, target_pdb=str(target))
    identity_before = _cache_identity_token(c1, image_digest=None)
    target.write_text(SAMPLE_PDB.replace("0.000", "1.234"))
    assert _cache_identity_token(c1, image_digest=None) != identity_before
    # result.name is preserved even though the on-disk path is nested.
    result = runner.run(a, dry_run=True)
    assert result.name == name


def test_runner_same_name_different_seed_does_not_hit_cache(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """seed/task_count/contigs/mode/hotspots/extra variants must not cross-hit:
    same name + different seed → the second run re-invokes upstream and its
    outputs are isolated in its own digest directory."""
    invoked_dirs: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        invoked_dirs.append(output_dir)
        (output_dir / "design_1.pdb").write_text(SAMPLE_PDB)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    name = "seed-isolation"

    first = runner.run(RFdiffusionConfig(name=name, seed=1))
    second = runner.run(RFdiffusionConfig(name=name, seed=2))

    assert len(invoked_dirs) == 2  # both seeds executed — no cross-hit
    assert invoked_dirs[0] != invoked_dirs[1]
    assert first.provenance.executed is True
    assert second.provenance.executed is True
    assert second.provenance.cache_hit is False
    # Each run's records come only from its own digest directory.
    assert Path(first.records[0].path).parent == invoked_dirs[0]
    assert Path(second.records[0].path).parent == invoked_dirs[1]


@pytest.mark.parametrize(
    "variant_kwargs",
    [
        {"task_count": 2},
        {"contigs": "20-24"},
        {"mode": "head_to_tail"},
        {"hotspots": ("A12",)},
        {"deterministic": False},
        {"checkpoint": "custom-ckpt"},
        {"extra": {"inference.noise_scale_ca": "0.5"}},
    ],
)
def test_runner_config_variants_do_not_cross_hit(
    output_root: Path, monkeypatch: pytest.MonkeyPatch, variant_kwargs: dict[str, Any]
) -> None:
    """Every config dimension that changes upstream behaviour gets its own
    digest directory — a variant must never satisfy another variant's cache."""
    invoked_dirs: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        invoked_dirs.append(output_dir)
        (output_dir / "design_0.pdb").write_text(SAMPLE_PDB)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    name = "variant"

    base = RFdiffusionConfig(name=name, seed=0)
    variant = RFdiffusionConfig(name=name, seed=0, **variant_kwargs)
    assert compute_config_digest(base) != compute_config_digest(variant)

    runner.run(base)
    variant_result = runner.run(variant)

    assert len(invoked_dirs) == 2  # the variant was NOT served from base's cache
    assert variant_result.provenance.cache_hit is False
    assert variant_result.provenance.executed is True


def test_runner_same_full_config_hits_cache(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same name AND same full config → the second run is a cache hit: no
    invocation, ``executed=False``, ``cache_hit=True``."""
    invoked_dirs: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        invoked_dirs.append(output_dir)
        (output_dir / "design_0.pdb").write_text(SAMPLE_PDB)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    config = RFdiffusionConfig(name="same-cfg", seed=3, task_count=1)

    first = runner.run(config)
    second = runner.run(config)

    assert len(invoked_dirs) == 1  # one invocation total
    assert first.provenance.executed is True
    assert second.provenance.executed is False
    assert second.provenance.cache_hit is True
    assert second.provenance.executed_config_digest is None
    assert second.provenance.base_seed == 3  # digest-bound cache: seed is provable
    assert Path(second.records[0].path).parent == invoked_dirs[0]


def test_runner_legacy_name_only_outputs_are_not_cache_hits(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-digest-layout outputs at ``<output_root>/<name>/*.pdb`` carry no
    proof of which config produced them — the runner must NOT treat them as a
    cache hit and must never mix them into results. They are left untouched."""
    invoked_dirs: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        invoked_dirs.append(output_dir)
        (output_dir / "design_0.pdb").write_text(SAMPLE_PDB)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    name = "legacy"
    legacy_dir = output_root / name
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "design_0.pdb").write_text(SAMPLE_PDB)

    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name=name))

    assert invoked_dirs  # NOT served from the legacy name-only outputs
    assert result.provenance.cache_hit is False
    # The record comes from the identity-keyed dir, not the legacy flat dir.
    assert len(result.records) == 1
    assert Path(result.records[0].path).parent == invoked_dirs[0]
    assert (legacy_dir / "design_0.pdb").exists()  # legacy files left untouched


def test_runner_image_digest_change_isolates_outputs(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same config, different (normalized) image digest → no cross-hit: the
    second run re-invokes upstream into its own identity directory."""
    invoked_dirs: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        invoked_dirs.append(output_dir)
        (output_dir / "design_0.pdb").write_text(SAMPLE_PDB)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    config = RFdiffusionConfig(name="img-iso")

    first = runner.run(config, image_digest=VALID_OCI_DIGEST)
    second = runner.run(config, image_digest=OTHER_OCI_DIGEST)

    assert len(invoked_dirs) == 2  # image change → not a cache hit
    assert invoked_dirs[0] != invoked_dirs[1]
    assert first.provenance.executed is True
    assert second.provenance.executed is True
    assert second.provenance.cache_hit is False
    assert second.provenance.image_digest == OTHER_OCI_DIGEST
    assert first.provenance.image_digest == VALID_OCI_DIGEST


def test_runner_target_content_change_isolates_outputs(
    output_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same config + same target PATH but changed bytes → no cross-hit: the
    source-backbone content digest is part of the cache identity."""
    target = tmp_path / "t.pdb"
    target.write_text(SAMPLE_PDB)
    invoked_dirs: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        invoked_dirs.append(output_dir)
        (output_dir / "design_0.pdb").write_text(SAMPLE_PDB)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    config = RFdiffusionConfig(name="src-iso", target_pdb=str(target))

    first = runner.run(config)
    first_source = first.provenance.source_backbone_digest
    assert first_source is not None

    # Change the bytes at the same path — the config digest is unchanged.
    target.write_text(SAMPLE_PDB.replace("0.000", "9.999"))
    second = runner.run(config)

    assert len(invoked_dirs) == 2  # content change → not a cache hit
    assert invoked_dirs[0] != invoked_dirs[1]
    assert second.provenance.executed is True
    assert second.provenance.cache_hit is False
    assert second.provenance.source_backbone_digest is not None
    assert second.provenance.source_backbone_digest != first_source


def test_runner_same_config_image_source_hits(
    output_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same config + same image digest + same source bytes → the second run is
    a cache hit (identity fully bound, no re-invocation)."""
    target = tmp_path / "t.pdb"
    target.write_text(SAMPLE_PDB)
    invoked_dirs: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        invoked_dirs.append(output_dir)
        (output_dir / "design_0.pdb").write_text(SAMPLE_PDB)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    config = RFdiffusionConfig(name="full-identity", seed=5, target_pdb=str(target))

    first = runner.run(config, image_digest=VALID_OCI_DIGEST)
    second = runner.run(config, image_digest=VALID_BARE_DIGEST)  # same normalized form

    assert len(invoked_dirs) == 1  # one invocation total
    assert first.provenance.executed is True
    assert second.provenance.executed is False
    assert second.provenance.cache_hit is True
    assert second.provenance.executed_config_digest is None
    # The hit's provenance corresponds to the exact bound identity.
    assert second.provenance.image_digest == VALID_OCI_DIGEST
    assert second.provenance.source_backbone_digest == compute_file_digest(target)
    assert second.provenance.base_seed == 5
    assert Path(second.records[0].path).parent == invoked_dirs[0]


def test_runner_missing_then_present_source_does_not_cross_hit(
    output_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing source stays soft (warning + None digest + run proceeds), and
    a later run once the file exists gets a different identity — the
    no-binding run can never satisfy the bound run's cache."""
    missing = tmp_path / "late.pdb"
    invoked_dirs: list[Path] = []

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        invoked_dirs.append(output_dir)
        (output_dir / "design_0.pdb").write_text(SAMPLE_PDB)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    config = RFdiffusionConfig(name="late-src", target_pdb=str(missing))

    first = runner.run(config)
    assert first.provenance.executed is True
    assert first.provenance.source_backbone_digest is None  # missing → soft signal

    missing.write_text(SAMPLE_PDB)
    second = runner.run(config)

    assert len(invoked_dirs) == 2  # present source → different identity → re-run
    assert second.provenance.executed is True
    assert second.provenance.cache_hit is False
    assert second.provenance.source_backbone_digest is not None


# ---------------------------------------------------------------------------
# RecordData.index — parsed from the design filename
# ---------------------------------------------------------------------------


def test_parse_output_dir_extracts_design_index_from_filename(tmp_path: Path) -> None:
    """``RecordData.index`` is the design's numeric index parsed from the
    filename's final ``_<digits>`` segment (``design_42.pdb`` -> ``42``),
    matching upstream's ``<prefix>_<i_des>.pdb`` naming."""
    design_dir = tmp_path / "designs"
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "design_42.pdb").write_text(SAMPLE_PDB)
    (design_dir / "design_0.pdb").write_text(SAMPLE_PDB)

    records = _parse_output_dir(design_dir)
    by_name = {Path(r.path).name: r.index for r in records}
    assert by_name["design_42.pdb"] == 42
    assert by_name["design_0.pdb"] == 0


def test_parse_output_dir_falls_back_honestly_for_nonstandard_names(tmp_path: Path) -> None:
    """Nonstandard filenames (no final ``_<digits>``) get the smallest
    non-negative index not already used — never a collision with a parsed
    design index."""
    design_dir = tmp_path / "designs"
    design_dir.mkdir(parents=True, exist_ok=True)
    # Parsed indices 0 and 1 are taken; the nonstandard file must not collide.
    (design_dir / "design_0.pdb").write_text(SAMPLE_PDB)
    (design_dir / "design_1.pdb").write_text(SAMPLE_PDB)
    (design_dir / "weird.pdb").write_text(SAMPLE_PDB)

    records = _parse_output_dir(design_dir)
    indices = sorted(r.index for r in records)
    assert indices == [0, 1, 2]
    weird = next(r for r in records if r.path.endswith("weird.pdb"))
    assert weird.index == 2

    # With only a high parsed index, the fallback fills the low hole.
    other = tmp_path / "other"
    other.mkdir(parents=True, exist_ok=True)
    (other / "design_42.pdb").write_text(SAMPLE_PDB)
    (other / "weird.pdb").write_text(SAMPLE_PDB)
    other_records = _parse_output_dir(other)
    indices2 = sorted(r.index for r in other_records)
    assert indices2 == [0, 42]


def test_runner_records_seed_offset_design_indices(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With seed mapped to ``inference.design_startnum``, upstream emits
    ``<prefix>_<i_des>.pdb`` for i_des in seed..seed+task_count-1; the parsed
    RecordData indices equal the per-design seeds."""
    seed, task_count = 7, 2

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        for i_des in range(seed, seed + task_count):
            (output_dir / f"design_{i_des}.pdb").write_text(SAMPLE_PDB)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="offset", seed=seed, task_count=task_count))

    assert sorted(r.index for r in result.records) == [seed, seed + 1]
    assert result.provenance.base_seed == seed
    assert result.provenance.task_count == task_count


@pytest.mark.parametrize(
    "unsafe_name",
    ["", ".", "..", "a/b", "a\\b", "a\x00b", "a\nb"],
)
def test_config_rejects_unsafe_name(unsafe_name: str) -> None:
    """Names that would escape the per-name output directory are rejected at
    construction (fail closed) — the digest layout cannot be subverted."""
    with pytest.raises(ValueError, match="safe path component"):
        RFdiffusionConfig(name=unsafe_name)


# ---------------------------------------------------------------------------
# S2 provenance (reproducibility)
# ---------------------------------------------------------------------------


def test_runner_records_honest_provenance_on_real_run(
    output_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real (non-dry-run, non-cached) invocation attaches a ProvenanceMetadata
    carrying the honest RFdiffusion contract:

    * ``base_seed`` == ``requested_seed`` == the forwarded base seed.
    * ``rng_intent`` is ``"per-design-index"`` for the default
      deterministic mode — upstream seeds design ``i`` with
      ``design_startnum + i``, so per-design seeds are
      ``base_seed .. base_seed + task_count - 1``.
    * ``executed_config_digest`` includes ``seed`` (nothing excluded
      in deterministic mode), so it equals the requested digest.
    """
    target = tmp_path / "target.pdb"
    target.write_text(SAMPLE_PDB)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = RFdiffusionRunner(output_root=output_root)
    config = RFdiffusionConfig(name="prov", seed=7, task_count=3, target_pdb=str(target))
    result = runner.run(config, image_digest=VALID_OCI_DIGEST)
    prov = result.provenance

    assert prov.model_identifier == "RFdiffusion"
    assert prov.temperature is None  # RFdiffusion does not expose temperature
    assert prov.image_digest == VALID_OCI_DIGEST
    assert prov.source_backbone_digest == compute_file_digest(target)
    # S2 honesty: the runner forwarded the base seed, so base_seed == requested_seed.
    assert prov.base_seed == 7
    assert prov.requested_seed == 7
    assert prov.task_count == 3
    assert prov.rng_intent == RNG_INTENT_PER_DESIGN_INDEX
    assert prov.exit_code == 0
    assert prov.failure_reason == ""
    assert prov.executed is True
    assert prov.cache_hit is False
    assert prov.executed_config_digest is not None
    assert prov.requested_config_digest is not None
    # The two digests are equal — seed is part of the executed config.
    assert prov.requested_config_digest == prov.executed_config_digest


def test_runner_records_non_deterministic_rng_intent(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deterministic=False`` → ``rng_intent="non-deterministic"`` and
    ``base_seed=None`` — the runner forwards neither
    ``inference.design_startnum`` nor ``inference.deterministic``, so no
    pinned seed may be claimed even though the caller supplied one."""
    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="nd", seed=42, deterministic=False))
    assert result.provenance.rng_intent == RNG_INTENT_NON_DETERMINISTIC
    assert result.provenance.base_seed is None
    assert result.provenance.requested_seed == 42  # the caller's intent is still audited
    assert result.provenance.executed is True
    assert result.provenance.executed_config_digest is not None


def test_runner_executed_config_digest_stable_for_non_deterministic_seed_changes(
    output_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2 honesty: with ``deterministic=False`` the base seed is NOT forwarded,
    so a seed-only change must flip ONLY the requested digest — the executed
    digest stays stable (upstream's RNG did not depend on it)."""
    target = tmp_path / "t.pdb"
    target.write_text(SAMPLE_PDB)
    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = RFdiffusionRunner(output_root=output_root)

    a = runner.run(
        RFdiffusionConfig(name="nd-seed-test", seed=1, target_pdb=str(target), deterministic=False)
    ).provenance
    b = runner.run(
        RFdiffusionConfig(
            name="nd-seed-test", seed=999, target_pdb=str(target), deterministic=False
        )
    ).provenance

    assert a.executed is True and b.executed is True
    assert a.executed_config_digest is not None and b.executed_config_digest is not None
    # Executed digest stable across non-deterministic seed-only changes.
    assert a.executed_config_digest == b.executed_config_digest
    # Requested digest flips — the caller did ask for different seeds.
    assert a.requested_config_digest != b.requested_config_digest
    # No pinned seed claimed in either run.
    assert a.base_seed is None and b.base_seed is None
    assert a.rng_intent == b.rng_intent == RNG_INTENT_NON_DETERMINISTIC


def test_runner_provenance_does_not_contain_per_task_seeds(
    output_root: Path,
) -> None:
    """S2 honesty: the manifest must not fabricate per-task seeds that the
    runner never actually used."""
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="x", seed=42, task_count=8), dry_run=True)
    assert not hasattr(result.provenance, "per_task_seeds")
    assert result.provenance.base_seed == 42  # the seed this call would forward
    assert result.provenance.requested_seed == 42
    assert result.provenance.task_count == 8
    # The per-design seed range (42..49) is encoded by base_seed + task_count.
    assert result.provenance.rng_intent == RNG_INTENT_PER_DESIGN_INDEX
    assert list(
        range(
            result.provenance.base_seed,
            result.provenance.base_seed + result.provenance.task_count,
        )
    ) == list(range(42, 50))


def test_runner_executed_config_digest_differs_across_seed_only_changes(
    output_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2 honesty: changing only ``seed`` MUST flip the executed config digest
    in deterministic mode because the runner forwards it as
    ``inference.design_startnum`` — the per-design seeds start there, so the
    seed changes what upstream actually ran.

    The ``requested_config_digest`` (full config) also flips, and
    ``base_seed`` differs between the two runs.

    Both calls use the same ``name`` and ``target_pdb`` so the only
    field that varies is ``seed`` — which is the field under test.
    """
    target = tmp_path / "t.pdb"
    target.write_text(SAMPLE_PDB)
    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = RFdiffusionRunner(output_root=output_root)

    a = runner.run(RFdiffusionConfig(name="seed-test", seed=1, target_pdb=str(target))).provenance
    b = runner.run(RFdiffusionConfig(name="seed-test", seed=999, target_pdb=str(target))).provenance

    # Both ran for real (cache miss), so executed_config_digest is set.
    assert a.executed is True and b.executed is True
    assert a.executed_config_digest is not None
    assert b.executed_config_digest is not None
    # Executed digest flips across seed-only changes — the base seed was forwarded.
    assert a.executed_config_digest != b.executed_config_digest
    # Requested digest flips too — the caller asked for different seeds.
    assert a.requested_config_digest != b.requested_config_digest
    # base_seed equals the forwarded base seed for both runs.
    assert a.base_seed == 1 and b.base_seed == 999
    assert a.rng_intent == b.rng_intent == RNG_INTENT_PER_DESIGN_INDEX
    assert a.requested_seed != b.requested_seed


def test_runner_cache_hit_records_honest_cache_provenance(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a cache hit, ``executed=False``, ``cache_hit=True``,
    ``executed_config_digest=None`` — the runner does not know which
    prior call produced the existing files, so it does not fabricate
    an executed digest. ``base_seed`` / ``requested_seed`` /
    ``rng_intent`` ARE reported because the cache is digest-bound:
    the cache key is the full requested-config digest (which includes
    ``seed``), so the cached outputs provably correspond to exactly
    this config and the per-design seed range describes them.
    """
    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", _fake_invoke_ok)
    name = "cache-hit"
    config = RFdiffusionConfig(name=name, seed=11)
    # Identity-bound cache: outputs must live under <name>/<identity(config,
    # VALID_OCI_DIGEST)>/ — the run below supplies exactly this image digest.
    design_dir = output_root / name / _cache_identity_token(config, image_digest=VALID_OCI_DIGEST)
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "design_11.pdb").write_text(SAMPLE_PDB)

    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(config, image_digest=VALID_OCI_DIGEST)
    prov = result.provenance

    assert prov.cache_hit is True
    assert prov.executed is False
    assert prov.executed_config_digest is None
    # Identity-bound cache: the seed semantics describe the cached outputs.
    assert prov.requested_config_digest != ""
    assert prov.requested_seed == 11
    assert prov.base_seed == 11
    assert prov.rng_intent == RNG_INTENT_PER_DESIGN_INDEX
    assert prov.image_digest == VALID_OCI_DIGEST
    # The record's index is parsed from the design filename (design_11.pdb -> 11).
    assert [r.index for r in result.records] == [11]
    # The records on disk are the cached records — the runner counts
    # them as "skipped" (we did NOT re-invoke) but also as "succeeded"
    # (they are usable outputs).
    assert result.skipped == 1
    assert result.succeeded == 1


def test_runner_dry_run_records_requested_digest_only(output_root: Path) -> None:
    """dry_run: ``executed=False``, ``cache_hit=False``,
    ``executed_config_digest=None``. The intended forwarded ``base_seed``
    IS recorded (what would have been executed); the executed digest stays
    ``None`` because nothing ran."""
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="dry", seed=5, task_count=3), dry_run=True)
    prov = result.provenance
    assert prov.executed is False
    assert prov.cache_hit is False
    assert prov.executed_config_digest is None
    assert prov.requested_config_digest != ""
    # ``base_seed`` records the intended forwarded seed for a deterministic
    # dry run — the manifest describes what WOULD have been executed.
    assert prov.base_seed == 5
    assert prov.requested_seed == 5
    assert prov.task_count == 3


def test_runner_propagates_exit_code_into_provenance(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero exit code from the subprocess must surface in the provenance
    record's exit_code, failure_reason (from stderr), and stderr_tail."""
    fake_result = InvokeResult(
        exit_code=7,
        stderr_tail="Traceback (most recent call last):\n  File ... RuntimeError: oops",
        timed_out=False,
        failure_reason="RuntimeError: oops",
    )

    def fake_invoke_with_metadata(**_: Any) -> InvokeResult:
        return fake_result

    monkeypatch.setattr(
        "biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke_with_metadata
    )
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="failure", seed=0))
    assert result.provenance.exit_code == 7
    assert result.provenance.failure_reason == "RuntimeError: oops"
    assert "RuntimeError: oops" in result.provenance.stderr_tail
    assert result.provenance.executed is True


def test_runner_records_timeout_in_provenance(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subprocess timeout must surface as exit_code=124, timed_out=True,
    and a deterministic failure_reason."""
    fake_result = InvokeResult(
        exit_code=124,
        stderr_tail="",
        timed_out=True,
        failure_reason="timeout after 3600s",
    )

    monkeypatch.setattr(
        "biolab_runners.rfdiffusion.runner._invoke_with_metadata",
        lambda **_: fake_result,
    )
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="slow"))
    assert result.provenance.exit_code == 124
    assert result.provenance.failure_reason == "timeout after 3600s"
    assert result.provenance.executed is True


def test_runner_equivalent_rerun_produces_equivalent_provenance(
    output_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2 equivalence: same config + same image digest + same backbone →
    byte-identical provenance.to_dict() for the same execution path."""
    target = tmp_path / "target.pdb"
    target.write_text(SAMPLE_PDB)
    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = RFdiffusionRunner(output_root=output_root)
    config = RFdiffusionConfig(name="equiv", seed=42, target_pdb=str(target))
    first = runner.run(config, image_digest=VALID_OCI_DIGEST).provenance.to_dict()
    second = runner.run(config, image_digest=VALID_OCI_DIGEST).provenance.to_dict()
    assert first == second


def test_runner_provenance_json_roundtrip(
    output_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provenance record serialises to JSON and back without loss — the
    seed fields and both digests survive the roundtrip."""
    target = tmp_path / "target.pdb"
    target.write_text(SAMPLE_PDB)
    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = RFdiffusionRunner(output_root=output_root)
    config = RFdiffusionConfig(name="roundtrip", seed=123, task_count=2, target_pdb=str(target))
    payload = runner.run(config, image_digest=VALID_OCI_DIGEST).provenance.to_dict()

    roundtripped = json.loads(json.dumps(payload))
    assert roundtripped == payload
    # The seed semantics are JSON-visible, not Python-object-only.
    assert roundtripped["base_seed"] == 123
    assert roundtripped["requested_seed"] == 123
    assert roundtripped["rng_intent"] == RNG_INTENT_PER_DESIGN_INDEX
    assert roundtripped["executed"] is True
    assert roundtripped["executed_config_digest"] == roundtripped["requested_config_digest"]
    # The per-design seed range is encoded by base_seed + task_count, not a list.
    assert set(roundtripped) == set(payload)
    assert "per_task_seeds" not in roundtripped


def test_runner_output_indices_start_at_seed(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output-index side effect of ``inference.design_startnum``: upstream names
    each output ``<prefix>_<i_des>.pdb`` with ``i_des`` starting at
    ``design_startnum`` (the configured seed). The runner parses every PDB in
    the design dir regardless of name, so with seed=42 / task_count=3 the
    parsed records are ``design_42.pdb`` .. ``design_44.pdb`` — not
    ``design_0.pdb``..``design_2.pdb``."""
    seed, task_count = 42, 3

    def fake_invoke(*, output_dir: Path, **_: Any) -> InvokeResult:
        # Mirror upstream: out_prefix = f"{output_prefix}_{i_des}" for
        # i_des in range(design_startnum, design_startnum + num_designs).
        for i_des in range(seed, seed + task_count):
            (output_dir / f"design_{i_des}.pdb").write_text(SAMPLE_PDB)
        return InvokeResult(exit_code=0)

    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", fake_invoke)
    runner = RFdiffusionRunner(output_root=output_root)
    result = runner.run(RFdiffusionConfig(name="indexed", seed=seed, task_count=task_count))

    assert result.succeeded == task_count
    names = [Path(r.path).name for r in result.records]
    assert names == [f"design_{i_des}.pdb" for i_des in range(seed, seed + task_count)]
    assert result.provenance.base_seed == seed
    assert result.provenance.task_count == task_count


def test_runner_normalises_image_digest_to_oci_form(
    output_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both bare-hex and OCI-prefixed forms must be normalised to the OCI form
    BEFORE any subprocess work, so downstream comparison sees a single form."""
    target = tmp_path / "target.pdb"
    target.write_text(SAMPLE_PDB)
    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = RFdiffusionRunner(output_root=output_root)
    config = RFdiffusionConfig(name="img-norm", target_pdb=str(target))

    oci_result = runner.run(config, image_digest=VALID_OCI_DIGEST)
    bare_result = runner.run(config, image_digest=VALID_BARE_DIGEST)

    assert oci_result.provenance.image_digest == VALID_OCI_DIGEST
    assert bare_result.provenance.image_digest == VALID_OCI_DIGEST  # normalised


def test_runner_validates_malformed_image_digest(
    output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed image digest must raise ValueError at run() time, not
    silently flow into the manifest."""
    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", _fake_invoke_ok)
    runner = RFdiffusionRunner(output_root=output_root)
    with pytest.raises(ValueError, match="image_digest must be"):
        runner.run(RFdiffusionConfig(name="bad"), image_digest="not-a-digest")


def test_runner_warns_when_target_pdb_missing(
    output_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ``target_pdb`` that does not exist on disk must produce a warning
    AND still record ``source_backbone_digest=None`` in the manifest.
    Missing path is a signal, not a hard error."""
    monkeypatch.setattr("biolab_runners.rfdiffusion.runner._invoke_with_metadata", _fake_invoke_ok)
    missing = tmp_path / "absent.pdb"
    runner = RFdiffusionRunner(output_root=output_root)
    with caplog.at_level(logging.WARNING, logger="biolab_runners.rfdiffusion.runner"):
        result = runner.run(
            RFdiffusionConfig(name="missing-target", target_pdb=str(missing)),
            dry_run=True,
        )
    assert result.provenance.source_backbone_digest is None
    assert any(
        "target_pdb" in record.message and "does not exist" in record.message
        for record in caplog.records
    ), f"expected target_pdb warning, got: {[r.message for r in caplog.records]}"
