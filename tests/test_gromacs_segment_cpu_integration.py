"""Real-CPU integration test for bounded production-MD segment continuation.

Two segments whose sum is the full production target are run with the local
``gmx`` 2026.1 CPU binary. The test settles the empirically verified
``-nsteps`` semantics (relative to the checkpoint step) and asserts that the
full-target ``prod.tpr`` does not change, cumulative steps reach the target,
the appended trajectory has no duplicated boundary frame, and the reported
completion moves from False to True. A retry of the first segment after its
progress record was lost must leave the checkpoint at the same step.

Skips when no GROMACS binary or no ``pdb2gmx`` force field is available. Set
``GROMACS_BIN`` to the binary under test (the consumer's CI uses the installed
2026.1 build).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest
from biolab_runners.gromacs import GromacsProtocolConfig, GromacsProtocolRunner

_ALANINE_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.145   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       0.200   0.140   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       0.150   0.250   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1      -0.070   0.100   0.100  1.00  0.00           C
TER
END
"""

_PRODUCTION_NS = 0.2
_SEGMENT_NS = 0.1
_STEPS_PER_SEGMENT = 50_000
_TARGET_STEPS = 100_000


def _gmx_binary() -> str | None:
    configured = os.environ.get("GROMACS_BIN", "")
    if (
        configured
        and not configured.startswith("container://")
        and (shutil.which(configured) or Path(configured).is_file())
    ):
        return configured
    return shutil.which("gmx")


def _force_field(gmx: str) -> str | None:
    completed = subprocess.run([gmx, "--version"], capture_output=True, text=True, check=False)
    prefix: Path | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("Data prefix:"):
            prefix = Path(line.split(":", 1)[1].strip())
            break
    if prefix is None:
        return None
    top = prefix / "share" / "gromacs" / "top"
    if not top.is_dir():
        return None
    for preferred in ("charmm27", "charmm36m", "amber99sb", "oplsaa"):
        if (top / f"{preferred}.ff").is_dir():
            return preferred
    installed = sorted(path.stem for path in top.glob("*.ff"))
    return installed[0] if installed else None


def _checkpoint_step(gmx: str, checkpoint: Path) -> int:
    dumped = subprocess.run(
        [gmx, "dump", "-cp", str(checkpoint)], capture_output=True, text=True, check=False
    )
    match = re.search(r"step\s*=\s*(\d+)", dumped.stdout)
    assert match is not None, f"no step in checkpoint dump: {dumped.stderr[-400:]}"
    return int(match.group(1))


def _xtc_frame_times(gmx: str, trajectory: Path) -> list[float]:
    checked = subprocess.run(
        [gmx, "check", "-f", str(trajectory)], capture_output=True, text=True, check=False
    )
    return [
        float(value)
        for _index, value in re.findall(r"Reading frame\s+(\d+)\s+time\s+([\d.]+)", checked.stderr)
    ]


def test_two_segments_complete_one_full_target_without_duplicate_frames(tmp_path: Path) -> None:
    gmx = _gmx_binary()
    if gmx is None:
        pytest.skip("no GROMACS binary available (set GROMACS_BIN)")
    force_field = _force_field(gmx)
    if force_field is None:
        pytest.skip("no pdb2gmx force field installed in the GROMACS data directory")

    input_pdb = tmp_path / "alanine.pdb"
    input_pdb.write_text(_ALANINE_PDB)
    config = GromacsProtocolConfig(
        name="segment-cpu",
        input_pdb=str(input_pdb),
        output_root=str(tmp_path / "runs"),
        force_field=force_field,
        nvt_ps=1,
        npt_ps=1,
        production_ns=_PRODUCTION_NS,
        minimization_max_iterations=200,
        nt_threads=2,
        timeout_seconds=600,
    )
    runner = GromacsProtocolRunner(binary_prefix=[gmx])

    first = runner.run_protocol(config, segment_ns=_SEGMENT_NS)
    assert first.exit_code == 0, f"first segment failed: {first.error}"
    work_dir = tmp_path / "runs" / config.name
    tpr = work_dir / "prod.tpr"
    tpr_digest = hashlib.sha256(tpr.read_bytes()).hexdigest()
    assert first.production_complete is False
    assert first.production_steps == _STEPS_PER_SEGMENT
    assert first.production_target_steps == _TARGET_STEPS
    assert _checkpoint_step(gmx, work_dir / "prod.cpt") == _STEPS_PER_SEGMENT

    # A retry of the same segment after its progress write was lost must not
    # advance the checkpoint: -nsteps is relative to the checkpoint step.
    manifest_path = work_dir / "gromacs_stages.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["stages"]["production"]
    record["status"] = "running"
    for key in ("production_steps", "production_target_steps", "production_complete"):
        record.pop(key, None)
    manifest_path.write_text(json.dumps(manifest))

    retry = runner.run_protocol(config, segment_ns=_SEGMENT_NS)
    assert retry.exit_code == 0, f"segment retry failed: {retry.error}"
    assert retry.production_complete is False
    assert retry.production_steps == _STEPS_PER_SEGMENT
    assert _checkpoint_step(gmx, work_dir / "prod.cpt") == _STEPS_PER_SEGMENT

    second = runner.run_protocol(config, segment_ns=_SEGMENT_NS)
    assert second.exit_code == 0, f"second segment failed: {second.error}"
    assert hashlib.sha256(tpr.read_bytes()).hexdigest() == tpr_digest
    assert second.production_complete is True
    assert second.production_steps == _TARGET_STEPS
    assert _checkpoint_step(gmx, work_dir / "prod.cpt") == _TARGET_STEPS

    manifest = json.loads((work_dir / "gromacs_stages.json").read_text())
    production = manifest["stages"]["production"]
    assert production["status"] == "completed"
    assert production["production_steps"] == _TARGET_STEPS

    times = _xtc_frame_times(gmx, work_dir / "prod.xtc")
    assert len(times) >= 3, f"expected appended trajectory frames, got {times}"
    assert all(later > earlier for earlier, later in pairwise(times)), (
        f"duplicated or non-monotonic boundary frame in prod.xtc: {times}"
    )
    assert (work_dir / "prod.segment.cpt").is_file()
