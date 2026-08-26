"""Tests for the GROMACS protocol's prebuilt-topology support.

The Slice-14 (E3 prerequisite, CHEM-001 / peptide-prep) feature
adds ``prebuilt_topology`` and ``prebuilt_coordinates`` paths to
:class:`GromacsProtocolConfig`. When both are set, the protocol
skips the ``gmx pdb2gmx`` topology stage and stages the
caller-supplied ``.top`` / ``.gro`` into the canonical
``topol.top`` / ``processed.gro`` filenames. Downstream
``box_buffer_nm`` / solvation / ions / equilibration / production
stages are unchanged.

Tests:

* :class:`TestConfigPrebuilt` — config validation rules
  (``both or neither``, ``input_pdb`` becomes optional).
* :class:`TestPrebuiltProtocolCommands` — TOPOLOGY stage's
  command list is empty in prebuilt mode.
* :class:`TestPrebuiltStaging` — :func:`stage_prebuilt_topology`
  copies files to canonical names, returns source digests.
* :class:`TestPrebuiltLegacyPath` — legacy fixture-style config
  (``input_pdb`` only) keeps the ``pdb2gmx`` invocation byte-identical.
* :class:`TestPrebuiltSourceInvalidation` — different prebuilt
  source digests invalidate the cached TOPOLOGY stage.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from biolab_runners.gromacs.config import GromacsProtocolConfig
from biolab_runners.gromacs.protocol import (
    build_commands,
    build_stage_plan,
    stage_outputs_for,
    stage_prebuilt_topology,
)
from biolab_runners.gromacs.utils import (
    record_stage_status,
)


def _valid_protocol_config(**overrides: Any) -> GromacsProtocolConfig:
    """Build a minimal protocol config with sensible defaults."""
    base: dict[str, Any] = {
        "name": "prebuilt-test",
        "input_pdb": "/tmp/input.pdb",
        "output_root": "/tmp/output",
    }
    base.update(overrides)
    return GromacsProtocolConfig(**base)


def _sha256(path: Path) -> str:
    """Return the exact SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigPrebuilt:
    """``both or neither`` rule + ``input_pdb`` becomes optional."""

    def test_rejects_only_prebuilt_topology(self) -> None:
        with pytest.raises(ValueError, match="both be set or both be empty"):
            _valid_protocol_config(
                prebuilt_topology="/tmp/x.top",
                prebuilt_coordinates="",
            )

    def test_rejects_only_prebuilt_coordinates(self) -> None:
        with pytest.raises(ValueError, match="both be set or both be empty"):
            _valid_protocol_config(
                prebuilt_topology="",
                prebuilt_coordinates="/tmp/x.gro",
            )

    def test_accepts_complete_prebuilt_pair(self) -> None:
        cfg = _valid_protocol_config(
            prebuilt_topology="/tmp/x.top",
            prebuilt_coordinates="/tmp/x.gro",
        )
        assert cfg.prebuilt_topology == "/tmp/x.top"
        assert cfg.prebuilt_coordinates == "/tmp/x.gro"

    def test_accepts_empty_prebuilt_pair(self) -> None:
        cfg = _valid_protocol_config()
        assert cfg.prebuilt_topology == ""
        assert cfg.prebuilt_coordinates == ""

    def test_input_pdb_optional_when_prebuilt_supplied(self) -> None:
        cfg = GromacsProtocolConfig(
            name="x",
            input_pdb="",
            output_root="/tmp/x",
            prebuilt_topology="/tmp/x.top",
            prebuilt_coordinates="/tmp/x.gro",
        )
        assert cfg.input_pdb == ""

    def test_input_pdb_required_when_no_prebuilt(self) -> None:
        with pytest.raises(ValueError, match="input_pdb is required"):
            GromacsProtocolConfig(
                name="x",
                input_pdb="",
                output_root="/tmp/x",
            )


# ---------------------------------------------------------------------------
# Protocol command construction
# ---------------------------------------------------------------------------


class TestPrebuiltProtocolCommands:
    """TOPOLOGY stage's command list is empty in prebuilt mode."""

    def test_topology_commands_empty_when_prebuilt(self) -> None:
        cfg = _valid_protocol_config(
            prebuilt_topology="/tmp/x.top",
            prebuilt_coordinates="/tmp/x.gro",
        )
        topo_stage = build_stage_plan()[0]
        cmds = build_commands(topo_stage, checkpoint_path=None, config=cfg)
        assert cmds == [], f"TOPOLOGY stage should be a no-op when prebuilt is set; got {cmds}"

    def test_box_solvate_ions_unchanged_in_prebuilt_mode(self) -> None:
        """The downstream stages keep their gmx editconf / solvate / genion invocations."""
        cfg = _valid_protocol_config(
            prebuilt_topology="/tmp/x.top",
            prebuilt_coordinates="/tmp/x.gro",
            ion_concentration_m=0.20,
            box_buffer_nm=1.2,
        )
        stages = build_stage_plan()
        box_cmds = build_commands(stages[1], checkpoint_path=None, config=cfg)
        assert "editconf" in box_cmds[0]
        assert "1.2" in box_cmds[0]
        solvate_cmds = build_commands(stages[2], checkpoint_path=None, config=cfg)
        assert "solvate" in solvate_cmds[0]
        ions_cmds = build_commands(stages[3], checkpoint_path=None, config=cfg)
        assert "grompp" in ions_cmds[0]
        assert "genion" in ions_cmds[1]
        assert "0.2" in ions_cmds[1]

    def test_md_stages_use_prebuilt_topology(self) -> None:
        """MD-stage grompp must reference ``topol.top`` regardless of how it was produced."""
        cfg = _valid_protocol_config(
            prebuilt_topology="/tmp/x.top",
            prebuilt_coordinates="/tmp/x.gro",
        )
        stages = build_stage_plan()
        for stage in stages[4:]:  # MD stages
            grompp = build_commands(stage, checkpoint_path=None, config=cfg)[0]
            assert "topol.top" in grompp


# ---------------------------------------------------------------------------
# Staging helper
# ---------------------------------------------------------------------------


class TestPrebuiltStaging:
    """:func:`stage_prebuilt_topology` writes canonical files + binds digests."""

    def test_copies_prebuilt_to_canonical_names(self, tmp_path: Path) -> None:
        # Set up source files.
        src_top = tmp_path / "src.top"
        src_gro = tmp_path / "src.gro"
        src_top.write_text("; prebuilt top content\n")
        src_gro.write_text("GRO\n   1\n   1A    C    1   0.0   0.0   0.0\n")

        work_dir = tmp_path / "work"
        work_dir.mkdir()

        cfg = _valid_protocol_config(
            prebuilt_topology=str(src_top),
            prebuilt_coordinates=str(src_gro),
            output_root=str(work_dir),
        )
        meta = stage_prebuilt_topology(cfg, work_dir)

        # Canonical names are populated.
        assert (work_dir / "topol.top").is_file()
        assert (work_dir / "processed.gro").is_file()

        # Content is byte-identical to the source.
        assert (work_dir / "topol.top").read_text() == src_top.read_text()
        assert (work_dir / "processed.gro").read_text() == src_gro.read_text()

        # The returned meta carries the SOURCE paths + digests (not
        # the staged copies').
        assert meta["topology"] == str(src_top)
        assert meta["coordinates"] == str(src_gro)
        assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256_topology"])
        assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256_coordinates"])
        assert meta["sha256_topology"] == _sha256(src_top)
        assert meta["sha256_coordinates"] == _sha256(src_gro)

    def test_raises_on_missing_prebuilt_file(self, tmp_path: Path) -> None:
        cfg = _valid_protocol_config(
            prebuilt_topology="/nonexistent/x.top",
            prebuilt_coordinates="/nonexistent/x.gro",
            output_root=str(tmp_path / "work"),
        )
        with pytest.raises(FileNotFoundError):
            stage_prebuilt_topology(cfg, tmp_path / "work")

    def test_raises_on_incomplete_prebuilt_pair(self, tmp_path: Path) -> None:
        # Config validator already rejects this, but the helper
        # double-checks so a caller bypassing the validator
        # can't silently corrupt the work dir.
        cfg = _valid_protocol_config(
            prebuilt_topology="",
            prebuilt_coordinates="",
            output_root=str(tmp_path / "work"),
        )
        with pytest.raises(ValueError, match="incomplete prebuilt pair"):
            stage_prebuilt_topology(cfg, tmp_path / "work")


# ---------------------------------------------------------------------------
# Legacy path is unchanged
# ---------------------------------------------------------------------------


class TestPrebuiltLegacyPath:
    """When prebuilt is empty, the protocol behaves exactly as before."""

    def test_legacy_pdb2gmx_invocation_unchanged(self) -> None:
        cfg = _valid_protocol_config()  # no prebuilt
        topo_stage = build_stage_plan()[0]
        cmds = build_commands(topo_stage, checkpoint_path=None, config=cfg)
        assert len(cmds) == 1
        cmd = cmds[0]
        assert cmd[0] == "gmx"
        assert cmd[1] == "pdb2gmx"
        assert cfg.input_pdb in cmd
        # The legacy command still uses the canonical -ff / -water / -ignh flags.
        idx_ff = cmd.index("-ff")
        idx_water = cmd.index("-water")
        assert cmd[idx_ff + 1] == cfg.force_field
        assert cmd[idx_water + 1] == cfg.water_model
        assert "-ignh" in cmd

    def test_legacy_config_digests_byte_identical(self) -> None:
        """A prebuilt=False config must produce identical command bytes
        to a pre-Slice-14 config (no regression)."""
        baseline = _valid_protocol_config(
            input_pdb="/tmp/input.pdb",
            output_root="/tmp/output",
            force_field="charmm36m",
            water_model="tip3p",
        )
        # ``prebuilt_topology`` and ``prebuilt_coordinates`` are the
        # only added fields; their defaults are empty strings, so
        # command construction is unchanged.
        topo_stage = build_stage_plan()[0]
        baseline_cmds = build_commands(topo_stage, checkpoint_path=None, config=baseline)

        cfg_with_empty_prebuilt = _valid_protocol_config(
            input_pdb="/tmp/input.pdb",
            output_root="/tmp/output",
            force_field="charmm36m",
            water_model="tip3p",
            prebuilt_topology="",
            prebuilt_coordinates="",
        )
        new_cmds = build_commands(topo_stage, checkpoint_path=None, config=cfg_with_empty_prebuilt)
        assert baseline_cmds == new_cmds


# ---------------------------------------------------------------------------
# Source-digest invalidation
# ---------------------------------------------------------------------------


class TestPrebuiltSourceInvalidation:
    """Different prebuilt source digests invalidate the cached TOPOLOGY stage."""

    def test_prebuilt_source_changed_invalidation(self, tmp_path: Path) -> None:
        """Cached TOPOLOGY stage is invalidated when the supplied
        prebuilt source differs from the recorded one.

        Mocks ``_prebuilt_source_changed`` indirectly by setting
        up a manifest with a different ``prebuilt_source`` block
        and asserting the runner's :func:`_prebuilt_source_changed`
        returns ``True`` for the new input.
        """
        from biolab_runners.gromacs.runner import _prebuilt_source_changed

        # Set up a fake cached manifest with one prebuilt-source block.
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        cached_top = tmp_path / "old.top"
        cached_gro = tmp_path / "old.gro"
        cached_top.write_text("old top\n")
        cached_gro.write_text("old gro\n")

        cfg = _valid_protocol_config(
            output_root=str(work_dir),
            prebuilt_topology=str(cached_top),
            prebuilt_coordinates=str(cached_gro),
        )
        # Pre-populate the manifest with the SAME source digests.
        meta = _prebuilt_meta_inline(cfg)
        record_stage_status(
            work_dir,
            "topology",
            "completed",
            outputs=("topol.top", "processed.gro"),
            prebuilt_source=meta,
        )

        # The cached source is identical → no invalidation.
        assert _prebuilt_source_changed(work_dir, cfg) is False

        # Change the source files: digest differs → invalidation.
        new_top = tmp_path / "new.top"
        new_gro = tmp_path / "new.gro"
        new_top.write_text("different top content\n")
        new_gro.write_text("different gro content\n")
        cfg_new = _valid_protocol_config(
            output_root=str(work_dir),
            prebuilt_topology=str(new_top),
            prebuilt_coordinates=str(new_gro),
        )
        assert _prebuilt_source_changed(work_dir, cfg_new) is True


class TestPrebuiltRunnerIntegration:
    """Dry-run the full runner in prebuilt mode (no ``gmx`` subprocess).

    The runner's prebuilt-mode wiring is exercised end-to-end
    without invoking any ``gmx`` binary: the dry-run path
    skips subprocesses (each stage records itself as
    ``validated``), and the prebuilt stage stages the files
    before the stage loop. This is the canonical
    CI-friendly integration check for the new prebuilt
    mode (the heavy ``gmx`` invocation is gated on
    ``gromacs_available()`` in :file:`tests/integration/`).
    """

    def test_dry_run_prebuilt_stages_files_and_records_manifest(self, tmp_path: Path) -> None:
        from biolab_runners.gromacs.runner import GromacsProtocolRunner

        # Source files with non-trivial content so digests
        # are reproducible bytewise.
        src_top = tmp_path / "src.top"
        src_gro = tmp_path / "src.gro"
        src_top.write_text("; prebuilt top content\n")
        src_gro.write_text("GRO\n   1\n   1A    C    1   0.0   0.0   0.0\n")

        work_dir = tmp_path / "work"
        cfg = _valid_protocol_config(
            name="prebuilt-dryrun",
            output_root=str(work_dir),
            prebuilt_topology=str(src_top),
            prebuilt_coordinates=str(src_gro),
        )

        runner = GromacsProtocolRunner(dry_run=True)
        result = runner.run_protocol(cfg)

        # The prebuilt stage is recorded as COMPLETED with the
        # ``prebuilt_source`` digest block; every downstream
        # stage is recorded as ``validated`` (dry-run only).
        assert result.error == ""
        assert "topology" in result.stage_statuses
        assert result.stage_statuses["topology"] == "completed"
        for stage in ("box", "solvate", "ions", "minimize", "equil_nvt", "equil_npt", "production"):
            assert stage in result.stage_statuses
            assert result.stage_statuses[stage] == "validated"

        # Staging wrote the canonical files in the runner's
        # work directory (under the ``name`` subdirectory).
        runner_work_dir = Path(result.output_dir)
        assert (runner_work_dir / "topol.top").is_file()
        assert (runner_work_dir / "processed.gro").is_file()

        # Stage manifest binds the prebuilt source digests.
        from biolab_runners.gromacs.utils import load_stage_manifest

        manifest = load_stage_manifest(runner_work_dir)
        topo_record = manifest["stages"]["topology"]
        assert "prebuilt_source" in topo_record
        meta = topo_record["prebuilt_source"]
        assert meta["prebuilt_topology"] == str(src_top)
        assert meta["prebuilt_coordinates"] == str(src_gro)
        assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256_topology"])
        assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256_coordinates"])
        assert meta["sha256_topology"] == _sha256(src_top)
        assert meta["sha256_coordinates"] == _sha256(src_gro)

    def test_staging_reuses_returned_digest_metadata(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """The staging block must reuse ``stage_prebuilt_topology``'s digests.

        Previously ``_stage_prebuilt`` re-read the source files a
        second time via ``_prebuilt_meta`` after staging. The fix
        reuses the digest metadata returned by
        ``stage_prebuilt_topology`` (single source-file read). The
        manifest binding must still be byte-identical.
        """
        from biolab_runners.gromacs import runner as runner_mod
        from biolab_runners.gromacs.runner import GromacsProtocolRunner

        src_top = tmp_path / "src.top"
        src_gro = tmp_path / "src.gro"
        src_top.write_text("; prebuilt top content\n")
        src_gro.write_text("GRO\n   1\n   1A    C    1   0.0   0.0   0.0\n")

        work_dir = tmp_path / "work"
        cfg = _valid_protocol_config(
            name="prebuilt-single-read",
            output_root=str(work_dir),
            prebuilt_topology=str(src_top),
            prebuilt_coordinates=str(src_gro),
        )

        # Spy on _prebuilt_meta: after the fix, the staging block
        # must NOT call it (the returned staged digests are reused).
        original = runner_mod._prebuilt_meta
        calls: list[Any] = []

        def _spy(config: Any) -> dict[str, str]:
            calls.append(config)
            return original(config)

        monkeypatch.setattr(runner_mod, "_prebuilt_meta", _spy)
        result = GromacsProtocolRunner(dry_run=True).run_protocol(cfg)
        assert result.error == ""

        # The manifest still binds the correct source digests.
        from biolab_runners.gromacs.utils import load_stage_manifest

        manifest = load_stage_manifest(Path(result.output_dir))
        meta = manifest["stages"]["topology"]["prebuilt_source"]
        assert meta["sha256_topology"] == _sha256(src_top)
        assert meta["sha256_coordinates"] == _sha256(src_gro)

        # _prebuilt_meta is still invoked (the invalidation
        # comparison before staging) — but the staging record uses
        # the returned metadata, so there is exactly ONE call per
        # invocation, not two (read-after-stage re-read removed).
        assert len(calls) == 1, (
            f"expected exactly one _prebuilt_meta call (invalidation compare); "
            f"got {len(calls)} — the staging block re-read the source files"
        )

    def test_same_source_resume_preserves_solvated_topology(self, tmp_path: Path) -> None:
        """A resume must not replace the working topology with its prebuilt source."""
        from biolab_runners.gromacs.runner import GromacsProtocolRunner

        src_top = tmp_path / "src.top"
        src_gro = tmp_path / "src.gro"
        src_top.write_text("; prebuilt topology\n")
        src_gro.write_text("GRO\n   1\n   1A    C    1   0.0   0.0   0.0\n")
        cfg = _valid_protocol_config(
            name="prebuilt-resume",
            output_root=str(tmp_path / "work"),
            prebuilt_topology=str(src_top),
            prebuilt_coordinates=str(src_gro),
        )

        first = GromacsProtocolRunner(dry_run=True).run_protocol(cfg)
        working_topology = Path(first.output_dir) / "topol.top"
        working_topology.write_text("; solvated topology with ions\n")

        GromacsProtocolRunner(dry_run=True).run_protocol(cfg)

        assert working_topology.read_text() == "; solvated topology with ions\n"


class TestPrebuiltSourceCascadeInvalidation:
    """B4 — prebuilt source change must invalidate ALL dependent stages.

    When the supplied prebuilt ``.top`` / ``.gro`` differs from
    the cached digests, the TOPOLOGY stage is re-staged AND every
    downstream stage (BOX, SOLVATE, IONS, MINIMIZE, EQUIL_NVT,
    EQUIL_NPT, PRODUCTION) must have its on-disk outputs
    quarantined to ``.stale/<UTC>/`` AND its manifest entry
    reset to PENDING so the per-stage loop re-runs them from
    scratch. The previous bug surfaced by probe 5 was that
    staging ran BEFORE the loop and re-recorded the manifest,
    making the invalidation check unreachable.
    """

    def test_first_run_with_source_v1(self, tmp_path: Path) -> None:
        """First run: prebuilt source v1 — every stage is initialised."""
        from biolab_runners.gromacs.runner import GromacsProtocolRunner, _work_dir
        from biolab_runners.gromacs.utils import load_stage_manifest

        src_top = tmp_path / "v1.top"
        src_gro = tmp_path / "v1.gro"
        src_top.write_text("; top v1\n")
        src_gro.write_text("GRO v1\n   1\n")

        work = tmp_path / "work"
        cfg = _valid_protocol_config(
            name="cascade",
            output_root=str(work),
            prebuilt_topology=str(src_top),
            prebuilt_coordinates=str(src_gro),
        )

        runner = GromacsProtocolRunner(dry_run=True)
        result = runner.run_protocol(cfg)
        assert result.error == ""

        work_dir = _work_dir(cfg)
        # Topology is COMPLETED with the v1 digests.
        manifest = load_stage_manifest(work_dir)
        topo = manifest["stages"]["topology"]
        assert topo["status"] == "completed"
        assert (
            topo["prebuilt_source"]["sha256_topology"]
            == (_prebuilt_meta_inline(cfg)["sha256_topology"])
        )

    def test_source_change_invalidates_downstream(self, tmp_path: Path) -> None:
        """A different prebuilt source MUST cascade the invalidation."""
        from biolab_runners.gromacs.paths import GromacsFiles
        from biolab_runners.gromacs.protocol import StageKind
        from biolab_runners.gromacs.runner import GromacsProtocolRunner, _work_dir
        from biolab_runners.gromacs.utils import (
            load_stage_manifest,
            record_stage_status,
        )

        # Run 1 — prebuilt source v1.
        src_top_v1 = tmp_path / "v1.top"
        src_gro_v1 = tmp_path / "v1.gro"
        src_top_v1.write_text("; top v1\n")
        src_gro_v1.write_text("GRO v1\n   1\n")

        work = tmp_path / "work"
        cfg_v1 = _valid_protocol_config(
            name="cascade",
            output_root=str(work),
            prebuilt_topology=str(src_top_v1),
            prebuilt_coordinates=str(src_gro_v1),
        )
        GromacsProtocolRunner(dry_run=True).run_protocol(cfg_v1)

        work_dir = _work_dir(cfg_v1)
        # Pre-create fake downstream outputs (the kind of stale
        # artifacts a real run leaves behind).
        (work_dir / "boxed.gro").write_text("stale box")
        (work_dir / "solvated.gro").write_text("stale solvate")
        (work_dir / "ions.gro").write_text("stale ions")
        (work_dir / "min.gro").write_text("stale min")
        (work_dir / "nvt.gro").write_text("stale nvt")
        (work_dir / "npt.gro").write_text("stale npt")
        (work_dir / "prod.gro").write_text("stale prod")

        # And mark each downstream stage as COMPLETED in the manifest.
        for kind, outputs in (
            (StageKind.BOX, (GromacsFiles.BOX_GRO,)),
            (StageKind.SOLVATE, (GromacsFiles.SOLVATED_GRO,)),
            (StageKind.IONS, (GromacsFiles.IONIZED_GRO,)),
            (StageKind.MINIMIZE, ("min.tpr", "min.gro")),
            (StageKind.EQUIL_NVT, ("nvt.tpr", "nvt.gro")),
            (StageKind.EQUIL_NPT, ("npt.tpr", "npt.gro")),
            (StageKind.PRODUCTION, ("prod.tpr", "prod.gro")),
        ):
            record_stage_status(
                work_dir,
                kind.value,
                "completed",
                outputs=outputs,
            )

        # Run 2 — prebuilt source v2.
        src_top_v2 = tmp_path / "v2.top"
        src_gro_v2 = tmp_path / "v2.gro"
        src_top_v2.write_text("; top v2 DIFFERENT\n")
        src_gro_v2.write_text("GRO v2 DIFFERENT\n   1\n")
        cfg_v2 = _valid_protocol_config(
            name="cascade",
            output_root=str(work),
            prebuilt_topology=str(src_top_v2),
            prebuilt_coordinates=str(src_gro_v2),
        )
        GromacsProtocolRunner(dry_run=True).run_protocol(cfg_v2)

        # Every downstream stage's manifest entry must be reset to
        # pending AND the on-disk outputs must have been quarantined
        # to .stale/<UTC>/ (the canonical invalidation surface).
        manifest = load_stage_manifest(work_dir)
        for kind in (
            StageKind.BOX,
            StageKind.SOLVATE,
            StageKind.IONS,
            StageKind.MINIMIZE,
            StageKind.EQUIL_NVT,
            StageKind.EQUIL_NPT,
            StageKind.PRODUCTION,
        ):
            record = manifest["stages"][kind.value]
            assert record["status"] == "pending", (
                f"{kind.value} not invalidated after source change: status={record['status']}"
            )
            assert "invalidated_by_prebuilt_source_change" in record

        # Every downstream output was moved out of work_dir into
        # .stale/<UTC>/.
        for name in (
            "boxed.gro",
            "solvated.gro",
            "ions.gro",
            "min.gro",
            "nvt.gro",
            "npt.gro",
            "prod.gro",
        ):
            assert not (work_dir / name).exists(), (
                f"stale {name} not quarantined after source change"
            )

        # The .stale/<UTC>/ directory has the quarantined files.
        stale_dirs = list((work_dir / ".stale").iterdir())
        assert stale_dirs
        for stale in stale_dirs:
            if stale.is_dir():
                stale_files = {p.name for p in stale.iterdir()}
                for name in (
                    "boxed.gro",
                    "solvated.gro",
                    "ions.gro",
                    "min.gro",
                    "nvt.gro",
                    "npt.gro",
                    "prod.gro",
                ):
                    assert name in stale_files, f"{name} not in stale dir {stale.name}"

    def test_source_change_quarantines_full_outputs_and_starts_fresh(self, tmp_path: Path) -> None:
        from biolab_runners.gromacs.runner import GromacsProtocolRunner, _work_dir
        from biolab_runners.gromacs.utils import load_stage_manifest

        src_top_v1 = tmp_path / "v1.top"
        src_gro_v1 = tmp_path / "v1.gro"
        src_top_v1.write_text("; top v1\n")
        src_gro_v1.write_text("GRO v1\n")
        work = tmp_path / "work"
        cfg_v1 = _valid_protocol_config(
            name="full-quarantine",
            output_root=str(work),
            prebuilt_topology=str(src_top_v1),
            prebuilt_coordinates=str(src_gro_v1),
        )
        with patch.object(GromacsProtocolRunner, "_run_subprocess", return_value=0):
            GromacsProtocolRunner().run_protocol(cfg_v1)

        work_dir = _work_dir(cfg_v1)
        for stage in build_stage_plan()[1:]:
            for name in stage_outputs_for(stage.kind, stage.prefix):
                (work_dir / name).write_text(f"stale {name}")

        src_top_v2 = tmp_path / "v2.top"
        src_gro_v2 = tmp_path / "v2.gro"
        src_top_v2.write_text("; top v2\n")
        src_gro_v2.write_text("GRO v2\n")
        cfg_v2 = _valid_protocol_config(
            name="full-quarantine",
            output_root=str(work),
            prebuilt_topology=str(src_top_v2),
            prebuilt_coordinates=str(src_gro_v2),
        )
        captured: list[list[str]] = []

        def _capture(command: list[str], _work_dir: Path, _timeout: int) -> int:
            captured.append(command)
            return 0

        with patch.object(GromacsProtocolRunner, "_run_subprocess", side_effect=_capture):
            result = GromacsProtocolRunner().run_protocol(cfg_v2)

        assert result.succeeded == 7
        assert all("-t" not in command and "-cpi" not in command for command in captured)
        stale_files = {
            path.name
            for stale_dir in (work_dir / ".stale").iterdir()
            if stale_dir.is_dir()
            for path in stale_dir.iterdir()
        }
        assert {"prod.cpt", "prod.xtc", "prod.trr"}.issubset(stale_files)
        assert load_stage_manifest(work_dir)["stages"]["production"]["status"] == "completed"


def _prebuilt_meta_inline(cfg: GromacsProtocolConfig) -> dict[str, str]:
    """Return the runner's ``_prebuilt_meta`` digest block for ``cfg``.

    Delegates to the production helper (single source of truth) —
    the tests must not reimplement the digest computation, or a
    divergence between the test and the runner would go unnoticed.
    """
    from biolab_runners.gromacs.runner import _prebuilt_meta

    return _prebuilt_meta(cfg)
