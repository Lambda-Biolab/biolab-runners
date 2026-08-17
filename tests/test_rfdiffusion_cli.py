"""Hermetic tests for the in-package ``rfdiffusion`` console script.

A fake stock ``scripts/run_inference.py`` captures argv and mimics
output / exit behaviour, so no real RFdiffusion clone or model
weights are needed. Together they prove the runner→console-script
contract: target / contigs / hotspot / seed / count / cyclic flags
and the owned ``inference.output_prefix`` all reach the upstream
script as correct Hydra positional overrides.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from biolab_runners.rfdiffusion.cli import (
    EXECUTION_CONTRACT_VERSION as CLI_CONTRACT_VERSION,
)
from biolab_runners.rfdiffusion.cli import (
    INFERENCE_OUTPUT_PREFIX_FIELD,
    main,
)
from biolab_runners.rfdiffusion.runner import (
    EXECUTION_CONTRACT_VERSION as RUNNER_CONTRACT_VERSION,
)

FAKE_RUN_INFERENCE = """\
import json, os, sys

capture = os.environ.get("FAKE_CAPTURE_PATH")
if capture:
    payload = {"argv": sys.argv[1:], "pythonpath": os.environ.get("PYTHONPATH")}
    marker = os.environ.get("FAKE_IMPORT_MODULE")
    if marker:
        try:
            mod = __import__(marker)
            payload["imported"] = getattr(mod, "MAGIC", None)
        except Exception as exc:  # noqa: BLE001 - record the failure, exit non-zero
            payload["import_error"] = repr(exc)
    with open(capture, "w") as fh:
        json.dump(payload, fh)
print("fake run_inference stdout")
print("fake run_inference stderr", file=sys.stderr)
sys.exit(int(os.environ.get("FAKE_EXIT_CODE", "0")))
"""

SAMPLE_PDB = """\
HEADER    RFdiffusion design 0
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.500   0.000   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       3.000  -1.000   0.000  1.00  0.00           O
TER
END
"""


@pytest.fixture
def fake_upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake stock clone: ``<home>/scripts/run_inference.py`` captures argv.

    Sets ``RFDIFFUSION_HOME`` and the fake's capture/exit env vars.
    Returns the capture file path.
    """
    home = tmp_path / "RFdiffusion"
    (home / "scripts").mkdir(parents=True)
    (home / "scripts" / "run_inference.py").write_text(FAKE_RUN_INFERENCE)
    capture = tmp_path / "argv.json"
    monkeypatch.setenv("RFDIFFUSION_HOME", str(home))
    monkeypatch.setenv("FAKE_CAPTURE_PATH", str(capture))
    monkeypatch.setenv("FAKE_EXIT_CODE", "0")
    return capture


def _captured_overrides(capture: Path) -> dict[str, str]:
    """The ``key=value`` overrides the fake script recorded, as a dict."""
    payload = json.loads(capture.read_text())
    return dict(token.split("=", 1) for token in payload["argv"] if "=" in token)


# ---------------------------------------------------------------------------
# --help / availability
# ---------------------------------------------------------------------------


def test_help_is_cheap_and_requires_no_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--help`` exits 0 and prints usage WITHOUT ``RFDIFFUSION_HOME`` or
    any model files — the availability probe stays cheap."""
    monkeypatch.setenv("RFDIFFUSION_HOME", str(tmp_path / "no-such-clone"))
    completed = subprocess.run(
        [sys.executable, "-m", "biolab_runners.rfdiffusion.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "rfdiffusion" in completed.stdout
    assert "RFDIFFUSION_HOME" in completed.stdout
    # In-process form returns 0 too.
    assert main(["--help"]) == 0
    assert main(["-h"]) == 0
    assert main([]) == 0


def test_help_does_not_invoke_upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--help must not touch the upstream script at all."""
    invoked = tmp_path / "invoked"
    monkeypatch.setenv("RFDIFFUSION_HOME", str(tmp_path))
    monkeypatch.setattr("biolab_runners.rfdiffusion.cli.subprocess.run", _fail_if_called(invoked))
    assert main(["--help"]) == 0
    assert not invoked.exists()


def _fail_if_called(marker: Path) -> Any:
    def _run(*_: Any, **__: Any) -> Any:
        marker.write_text("called")
        raise AssertionError("subprocess.run must not be called here")

    return _run


# ---------------------------------------------------------------------------
# Contract translation
# ---------------------------------------------------------------------------


def test_cli_translates_runner_contract_to_hydra_overrides(
    fake_upstream: Path, tmp_path: Path
) -> None:
    """Target + binder contigs + hotspot + seed + count + cyclic + owned
    output prefix all land in ONE upstream argv as correct Hydra
    positional overrides; contigs spaces stay inside one element."""
    target = tmp_path / "target.pdb"
    target.write_text(SAMPLE_PDB)
    outdir = tmp_path / "designs"

    rc = main(
        [
            "--output_dir",
            str(outdir),
            "--inference.num-designs",
            "3",
            "--contigmap.contigs",
            "A1-110/0 B1-110/0 14-18",
            "--inference.input-pdb",
            str(target),
            "--inference.design-startnum",
            "42",
            "--inference.deterministic",
            "True",
            "--inference.cyclic",
            "True",
            "--inference.cyc-chains",
            "a",
            "--ppi.hotspot-res",
            "A51,A52",
        ]
    )

    assert rc == 0
    argv = json.loads(fake_upstream.read_text())["argv"]
    assert argv[:2] == ["--config-name", "base"]
    overrides = _captured_overrides(fake_upstream)
    assert overrides["inference.num_designs"] == "3"
    assert overrides["contigmap.contigs"] == "[A1-110/0 B1-110/0 14-18]"
    assert overrides["inference.input_pdb"] == str(target)
    assert overrides["inference.design_startnum"] == "42"
    assert overrides["inference.deterministic"] == "True"
    assert overrides["inference.cyclic"] == "True"
    assert overrides["inference.cyc_chains"] == "a"
    assert overrides["ppi.hotspot_res"] == "['A51','A52']"
    # Owned output prefix: under output_dir, absolutized.
    assert overrides[INFERENCE_OUTPUT_PREFIX_FIELD] == str(Path(outdir).absolute() / "design")
    # The spaces in the contigs value survive as ONE argv element.
    assert "contigmap.contigs=[A1-110/0 B1-110/0 14-18]" in argv


def test_cli_hotspot_res_translation_is_stock_form(fake_upstream: Path, tmp_path: Path) -> None:
    """``ppi.hotspot_res`` becomes the stock binder example's quoted-string
    Hydra list; a single hotspot stays a single-element list."""
    outdir = tmp_path / "d"
    rc = main(
        [
            "--output_dir",
            str(outdir),
            "--ppi.hotspot-res",
            "A51",
        ]
    )
    assert rc == 0
    overrides = _captured_overrides(fake_upstream)
    assert overrides["ppi.hotspot_res"] == "['A51']"


def test_cli_extra_dotted_keys_pass_through(fake_upstream: Path, tmp_path: Path) -> None:
    """Non-list keys (incl. ``extra`` passthrough like noise scales) are
    forwarded verbatim as ``key=value``."""
    outdir = tmp_path / "d"
    rc = main(
        [
            "--output_dir",
            str(outdir),
            "--inference.noise-scale-ca",
            "0.5",
            "--diffuser.T",
            "50",
        ]
    )
    assert rc == 0
    overrides = _captured_overrides(fake_upstream)
    assert overrides["inference.noise_scale_ca"] == "0.5"
    assert overrides["diffuser.T"] == "50"


def test_cli_prepends_clone_root_to_pythonpath(
    fake_upstream: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clone-only deployment: the resolved ``RFDIFFUSION_HOME`` is prepended
    to PYTHONPATH (existing value preserved) so ``run_inference.py`` can
    ``import rfdiffusion`` from the clone root without pip-installing the
    upstream package."""
    clone_home = Path(os.environ["RFDIFFUSION_HOME"])  # set by the fake_upstream fixture
    (clone_home / "rfdiffusion").mkdir(parents=True)
    (clone_home / "rfdiffusion" / "__init__.py").write_text("MAGIC = 42\n")
    monkeypatch.setenv("PYTHONPATH", "/some/existing/path")
    monkeypatch.setenv("FAKE_IMPORT_MODULE", "rfdiffusion")

    rc = main(["--output_dir", str(tmp_path / "d")])
    assert rc == 0

    payload = json.loads(fake_upstream.read_text())
    assert payload["imported"] == 42  # the clone-root module was importable
    pythonpath = payload["pythonpath"]
    assert pythonpath is not None and pythonpath.startswith(str(clone_home))
    assert pythonpath.endswith(os.pathsep + "/some/existing/path")  # preserved


def test_cli_quotes_scalar_paths_and_preserves_types(fake_upstream: Path, tmp_path: Path) -> None:
    """Hydra-quoting: paths with whitespace / quote characters are wrapped
    in quotes with deterministic escaping (exact argv bytes), while
    numeric / bool / plain-path tokens stay unquoted so OmegaConf keeps
    their types."""
    outdir = tmp_path / "out dir"  # space in the output dir → owned prefix quoted
    spaced = tmp_path / "my target.pdb"  # space, no quote → single-quoted
    spaced.write_text(SAMPLE_PDB)
    quoted = tmp_path / "it's target.pdb"  # single quote → double-quoted
    quoted.write_text(SAMPLE_PDB)
    rc = main(
        [
            "--output_dir",
            str(outdir),
            "--inference.input-pdb",
            str(spaced),
            "--inference.num-designs",
            "3",
            "--inference.deterministic",
            "True",
            "--inference.cyc-chains",
            "a",
        ]
    )
    assert rc == 0
    argv = json.loads(fake_upstream.read_text())["argv"]
    assert f"inference.input_pdb='{spaced}'" in argv  # single-quoted (no quote in value)
    assert f"inference.output_prefix='{Path(outdir).absolute() / 'design'}'" in argv
    assert "inference.num_designs=3" in argv  # int token unquoted
    assert "inference.deterministic=True" in argv  # bool token unquoted
    assert "inference.cyc_chains=a" in argv  # plain token unquoted

    rc = main(
        [
            "--output_dir",
            str(tmp_path / "d"),
            "--inference.input-pdb",
            str(quoted),
        ]
    )
    assert rc == 0
    argv = json.loads(fake_upstream.read_text())["argv"]
    # value contains a single quote → double-quoted with the quote literal
    assert f'inference.input_pdb="{quoted}"' in argv


def test_cli_escapes_quotes_deterministically(fake_upstream: Path, tmp_path: Path) -> None:
    """Quote escaping round-trips Hydra's own unescaper (deterministic, not
    shell quoting): a path with a double quote (no single quote) is
    single-quoted with the quote literal; a path containing BOTH quote
    characters is double-quoted with ``\\"`` escaping."""
    dq = tmp_path / 'a"b.pdb'
    dq.write_text(SAMPLE_PDB)
    rc = main(
        [
            "--output_dir",
            str(tmp_path / "d1"),
            "--inference.input-pdb",
            str(dq),
        ]
    )
    assert rc == 0
    argv = json.loads(fake_upstream.read_text())["argv"]
    assert f"inference.input_pdb='{dq}'" in argv

    both = tmp_path / "a'b\"c.pdb"
    both.write_text(SAMPLE_PDB)
    rc = main(
        [
            "--output_dir",
            str(tmp_path / "d2"),
            "--inference.input-pdb",
            str(both),
        ]
    )
    assert rc == 0
    argv = json.loads(fake_upstream.read_text())["argv"]
    # single quote present → double-quoted, with every " escaped as \"
    expected = 'inference.input_pdb="' + str(both).replace('"', '\\"') + '"'
    assert expected in argv

    # backslash immediately before a double quote (plus a single quote) →
    # the backslash is doubled so Hydra's (\\\\)+" unescape recovers one.
    backslash = tmp_path / "a'b\\\"c.pdb"
    backslash.write_text(SAMPLE_PDB)
    rc = main(
        [
            "--output_dir",
            str(tmp_path / "d3"),
            "--inference.input-pdb",
            str(backslash),
        ]
    )
    assert rc == 0
    argv = json.loads(fake_upstream.read_text())["argv"]
    expected = (
        'inference.input_pdb="' + str(backslash).replace("\\", "\\\\").replace('"', '\\"') + '"'
    )
    assert expected in argv


def test_cli_confines_hydra_metadata_under_output_dir(
    fake_upstream: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The console script owns the Hydra confinement overrides — exact argv:
    ``hydra.run.dir=<output_dir>/hydra``, ``hydra.output_subdir=null``,
    ``hydra.job.chdir=False`` — so Hydra's metadata/logs stay under the
    identity output directory and nothing is created in the caller's CWD
    (the runner's PDB parsing is untouched)."""
    outdir = tmp_path / "designs"
    rc = main(["--output_dir", str(outdir), "--inference.num-designs", "1"])
    assert rc == 0
    argv = json.loads(fake_upstream.read_text())["argv"]
    assert f"hydra.run.dir={Path(outdir).absolute() / 'hydra'}" in argv
    assert "hydra.output_subdir=null" in argv
    assert "hydra.job.chdir=False" in argv
    # The CLI itself never creates anything in the caller's CWD.
    assert not (Path.cwd() / "outputs").exists()
    assert not (Path.cwd() / ".hydra").exists()


def test_cli_propagates_exit_code_stdout_stderr(
    fake_upstream: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """The upstream exit code, stdout, and stderr propagate unchanged.

    ``capfd`` (fd-level) is used because the upstream script inherits
    the process fds — sys-level ``capsys`` would see nothing."""
    monkeypatch.setenv("FAKE_EXIT_CODE", "7")
    outdir = tmp_path / "d"
    rc = main(["--output_dir", str(outdir), "--inference.num-designs", "1"])
    assert rc == 7
    captured = capfd.readouterr()
    assert "fake run_inference stdout" in captured.out
    assert "fake run_inference stderr" in captured.err


# ---------------------------------------------------------------------------
# Fail-closed / validation
# ---------------------------------------------------------------------------


def test_cli_fails_clearly_when_upstream_script_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing ``scripts/run_inference.py`` yields a clear error naming
    ``RFDIFFUSION_HOME`` — no traceback, no arbitrary command."""
    monkeypatch.setenv("RFDIFFUSION_HOME", str(tmp_path))
    rc = main(["--output_dir", str(tmp_path / "d")])
    assert rc == 2
    assert "run_inference.py not found" in capsys.readouterr().err


def test_cli_fails_when_target_pdb_missing(
    fake_upstream: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``inference.input_pdb`` pointing at a missing file fails clearly."""
    rc = main(
        [
            "--output_dir",
            str(tmp_path / "d"),
            "--inference.input-pdb",
            str(tmp_path / "absent.pdb"),
        ]
    )
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--inference.num-designs", "5"],  # no --output_dir
        ["--output_dir"],  # --output_dir without a value
        ["--output_dir", "/tmp/x", "--inference.num-designs"],  # flag w/o value
        ["--output_dir", "/tmp/x", "--foo", "1"],  # undotted flag
        ["--output_dir", "/tmp/x", "bare-positional"],  # positional junk
        ["--output_dir", "/tmp/x", "--output_dir", "/tmp/y"],  # duplicate
        ["--output_dir", "/tmp/x", "--inference.seed", "42"],  # unsupported key
        ["--output_dir", "/tmp/x", "--inference.output-prefix", "p"],  # managed key
    ],
)
def test_cli_rejects_malformed_or_disallowed_contract(
    bad_args: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Malformed pairs, undotted/unsupported/managed flags, and a missing
    --output_dir are rejected with a clear error (no upstream launch)."""
    rc = main(bad_args)
    assert rc == 2
    assert "rfdiffusion:" in capsys.readouterr().err


def test_cli_fails_when_output_dir_uncreatable(
    fake_upstream: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An uncreatable ``--output_dir`` (an existing FILE) fails clearly
    before any upstream launch."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file in the way")
    rc = main(["--output_dir", str(blocker)])
    assert rc == 2
    assert "cannot create --output_dir" in capsys.readouterr().err


def test_cli_fails_when_upstream_launch_fails(
    fake_upstream: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed ``subprocess.run`` (OSError) surfaces as a clear 127 error —
    the console script never swallows or fabricates an exit code."""

    def _raise_oserror(*_: Any, **__: Any) -> Any:
        raise OSError("boom")

    monkeypatch.setattr("biolab_runners.rfdiffusion.cli.subprocess.run", _raise_oserror)
    rc = main(["--output_dir", str(tmp_path / "d")])
    assert rc == 127
    assert "failed to launch" in capsys.readouterr().err


def test_cli_never_launches_on_invalid_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid input must not reach subprocess.run at all."""
    invoked = tmp_path / "invoked"
    monkeypatch.setenv("RFDIFFUSION_HOME", str(tmp_path))
    monkeypatch.setattr("biolab_runners.rfdiffusion.cli.subprocess.run", _fail_if_called(invoked))
    rc = main(["--output_dir", "/tmp/x", "--inference.seed", "42"])
    assert rc == 2
    assert not invoked.exists()


def test_contract_version_has_single_authoritative_source() -> None:
    """``EXECUTION_CONTRACT_VERSION`` lives in the translation-owning CLI
    module and the runner imports it — one bump location covers both the
    config→flag mapping and the flag→Hydra translation."""
    assert CLI_CONTRACT_VERSION == RUNNER_CONTRACT_VERSION == 1


# ---------------------------------------------------------------------------
# Runner → console script end-to-end (hermetic)
# ---------------------------------------------------------------------------


def test_runner_invokes_console_script_and_parses_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full chain: RFdiffusionRunner → console script → fake upstream.

    The fake upstream mirrors stock naming (``<prefix>_<i_des>.pdb``
    from ``inference.output_prefix``), and the runner parses the
    resulting record — proving the whole execution contract.
    """
    from biolab_runners.rfdiffusion import RFdiffusionConfig, RFdiffusionRunner

    home = tmp_path / "RFdiffusion"
    (home / "scripts").mkdir(parents=True)
    # Fake upstream: parse output_prefix, write design_42.pdb, exit 0.
    fake_script = (
        "import os, sys\n"
        "prefix = next(t.split('=', 1)[1] for t in sys.argv[1:] "
        "if t.startswith('inference.output_prefix='))\n"
        "out = os.path.dirname(prefix)\n"
        "os.makedirs(out, exist_ok=True)\n"
        "with open(os.path.join(out, os.path.basename(prefix) + '_42.pdb'), 'w') as fh:\n"
        f"    fh.write({SAMPLE_PDB!r})\n"
    )
    (home / "scripts" / "run_inference.py").write_text(fake_script)
    monkeypatch.setenv("RFDIFFUSION_HOME", str(home))
    target = tmp_path / "target.pdb"
    target.write_text(SAMPLE_PDB)

    runner = RFdiffusionRunner(
        output_root=tmp_path / "out",
        binary_prefix=[sys.executable, "-m", "biolab_runners.rfdiffusion.cli"],
    )
    result = runner.run(
        RFdiffusionConfig(name="chain", seed=42, task_count=1, target_pdb=str(target))
    )
    assert result.exit_code == 0
    assert result.succeeded == 1
    assert [r.index for r in result.records] == [42]
    assert result.provenance.executed is True
    assert result.provenance.source_backbone_digest is not None
