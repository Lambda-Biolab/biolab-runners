"""Tests for the GROMACS production-grade protocol.

The S4 protocol module owns several contracts; each contract gets
a small focused test class:

- **TestMdpConstraints** — all MD .mdp files use
  ``constraints = h-bonds`` + ``constraint-algorithm = lincs`` at
  ``dt = 0.002`` (the canonical recipe for 2 fs h-bond-constrained MD).
- **TestMdpPositionRestraints** — NVT/NPT carry ``define = -DPOSRES``
  but do **NOT** emit a ``[ position_restraints ]`` block (that block
  belongs in the topology, not the .mdp).
- **TestMdpThermostat** — NVT/NPT use v-rescale; production uses
  nose-hoover.
- **TestMdpReplicaSeed** — ``gen-seed`` is ``replica_index + 1``,
  not ``-1``; different replicas produce different seeds.
- **TestMdpValidation** — zero/negative inputs raise.
- **TestMdpDeterminism** — same inputs → byte-identical output.
- **TestBuildCommands** — grompp precedes mdrun for MD stages;
  ``-cpi`` + ``-append`` are present only when ``.cpt`` exists;
  genion's group selection comes from a known constant (NOT
  ``sh -c`` shell injection); ``-nt`` is omitted when
  ``nt_threads=0``.
- **TestBoxBufferCompat** — ``box_size_nm`` is a deprecated alias
  for ``box_buffer_nm``.
- **TestForceFieldDefault** — default is documented but NOT
  claimed as bundled with GROMACS.
- **TestStageManifest** — load/save round-trip; corrupt manifest
  is tolerated; atomic save leaves no .tmp.
- **TestRunnerSkipAndResume** — manifest authority for skip;
  disk-output fallback when manifest silent; ``-cpi`` + ``-append``
  emitted iff ``.cpt`` exists.
- **TestRunnerAccounting** — counters distinguish skipped /
  succeeded / failed / interrupted.
- **TestRunnerSigtermPreservation** — SIGTERM does NOT mark the
  stage as FAILED; the manifest is left in RUNNING and the
  result reports ``interrupted`` (not ``failed``).
- **TestLegacyApiPreserved** — the S3 public API is unchanged.

The tests deliberately do NOT assert exact .mdp string content
beyond the contractually-required keys (constraints, lincs, define,
tcoupl, gen-seed) — those string-presence checks would be
tautological (we wrote the strings; a regression would have to
delete them, which the type checker already catches).
"""

from __future__ import annotations

import signal
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from biolab_runners.gromacs import (
    GENION_INPUT,
    GromacsProtocolConfig,
    GromacsProtocolRunner,
    GromacsRunner,
    StageStatus,
    gromacs_available,
)
from biolab_runners.gromacs.config import GromacsConfig
from biolab_runners.gromacs.paths import GromacsFiles
from biolab_runners.gromacs.protocol import (
    StageKind,
    build_commands,
    build_stage_plan,
    generate_equil_npt_mdp,
    generate_equil_nvt_mdp,
    generate_minimization_mdp,
    generate_production_mdp,
    ions_mdp_content,
    stage_minimum_outputs,
    stage_outputs_for,
)
from biolab_runners.gromacs.utils import (
    load_stage_manifest,
    record_stage_status,
    save_stage_manifest,
)


def _valid_protocol_config(**overrides: Any) -> GromacsProtocolConfig:
    """Return a protocol config with sensible defaults; callers override specific fields."""
    base: dict[str, Any] = {
        "name": "protocol-test",
        "input_pdb": "/tmp/input.pdb",
        "output_root": "/tmp/output",
    }
    base.update(overrides)
    return GromacsProtocolConfig(**base)


def _code_only(module_name: str) -> str:
    """Return the source of ``module_name`` with all docstring lines stripped."""
    import ast
    import importlib

    mod = importlib.import_module(module_name)
    src_path = mod.__file__
    assert src_path is not None, f"module {module_name!r} has no __file__"
    src = Path(src_path).read_text()
    tree = ast.parse(src)
    code_lines: list[str] = []
    raw = src.splitlines()
    docstring_ranges: list[tuple[int, int]] = []

    def _record_docstring(node: ast.AST) -> None:
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        body_first = node.body[0] if node.body else None
        if body_first is None:
            return
        if not (isinstance(body_first, ast.Expr) and isinstance(body_first.value, ast.Constant)):
            return
        if not isinstance(body_first.value.value, str):
            return
        docstring_ranges.append((body_first.lineno, body_first.end_lineno or body_first.lineno))

    for node in ast.walk(tree):
        _record_docstring(node)

    for i, line in enumerate(raw, start=1):
        if any(start <= i <= end for start, end in docstring_ranges):
            continue
        if line.strip().startswith("#"):
            continue
        code_lines.append(line)
    return "\n".join(code_lines)


def _protocol_code_only() -> str:
    """Return the protocol.py source with all docstring lines stripped."""
    return _code_only("biolab_runners.gromacs.protocol")


def _runner_code_only() -> str:
    """Return the runner.py source with all docstring lines stripped."""
    return _code_only("biolab_runners.gromacs.runner")


# ---------------------------------------------------------------------------
# .mdp contract — LINCS h-bond constraints at dt=0.002
# ---------------------------------------------------------------------------


class TestMdpConstraints:
    """Dynamics .mdp files use LINCS h-bond constraints at 2 fs.

    Steepest-descent minimisation does NOT carry ``constraints`` /
    ``dt`` / LINCS — it's a non-dynamics integrator with no
    thermostat / barostat. Only the three MD dynamics stages
    (NVT / NPT / production) carry constraints.
    """

    @pytest.mark.parametrize(
        "generator",
        [
            generate_equil_nvt_mdp,
            generate_equil_npt_mdp,
            generate_production_mdp,
        ],
    )
    def test_dynamics_mdp_has_hbond_constraints(self, generator: Any) -> None:
        mdp = generator()
        assert "constraints      = h-bonds" in mdp

    @pytest.mark.parametrize(
        "generator",
        [
            generate_equil_nvt_mdp,
            generate_equil_npt_mdp,
            generate_production_mdp,
        ],
    )
    def test_dynamics_mdp_has_lincs_algorithm(self, generator: Any) -> None:
        mdp = generator()
        assert "constraint-algorithm = lincs" in mdp

    @pytest.mark.parametrize(
        "generator",
        [
            generate_equil_nvt_mdp,
            generate_equil_npt_mdp,
            generate_production_mdp,
        ],
    )
    def test_dynamics_mdp_has_two_fs_timestep(self, generator: Any) -> None:
        mdp = generator()
        assert "dt               = 0.002" in mdp


# ---------------------------------------------------------------------------
# .mdp contract — no invalid [position_restraints] block; POSRES define retained
# ---------------------------------------------------------------------------


class TestMdpPositionRestraints:
    """NVT/NPT carry ``define = -DPOSRES`` but emit NO ``[ position_restraints ]`` block.

    The ``[ position_restraints ]`` topology directive belongs in
    the topology file (where ``#include "posre.itp"`` activates it
    when ``-DPOSRES`` is defined). Putting it in the .mdp is an
    error — GROMACS would ignore it (the section is parsed by the
    topology reader, not the .mdp reader) and the operator might
    think their restraints are active when they are not.
    """

    def test_nvt_has_posres_define(self) -> None:
        mdp = generate_equil_nvt_mdp()
        assert "define           = -DPOSRES" in mdp

    def test_nvt_does_not_emit_position_restraints_block(self) -> None:
        mdp = generate_equil_nvt_mdp()
        assert "[ position_restraints ]" not in mdp

    def test_npt_has_posres_define(self) -> None:
        mdp = generate_equil_npt_mdp()
        assert "define           = -DPOSRES" in mdp

    def test_npt_does_not_emit_position_restraints_block(self) -> None:
        mdp = generate_equil_npt_mdp()
        assert "[ position_restraints ]" not in mdp

    def test_production_has_no_posres(self) -> None:
        mdp = generate_production_mdp()
        assert "DPOSRES" not in mdp
        assert "[ position_restraints ]" not in mdp


# ---------------------------------------------------------------------------
# .mdp contract — v-rescale equilibration, nose-hoover production
# ---------------------------------------------------------------------------


class TestMdpThermostat:
    """NVT/NPT use v-rescale (not berendsen); production uses nose-hoover."""

    def test_nvt_uses_v_rescale(self) -> None:
        mdp = generate_equil_nvt_mdp()
        assert "tcoupl           = v-rescale" in mdp
        assert "tcoupl           = berendsen" not in mdp

    def test_npt_uses_v_rescale(self) -> None:
        mdp = generate_equil_npt_mdp()
        assert "tcoupl           = v-rescale" in mdp

    def test_production_uses_nose_hoover(self) -> None:
        mdp = generate_production_mdp()
        assert "tcoupl           = nose-hoover" in mdp

    def test_npt_uses_parrinello_rahman_for_production_pressure(self) -> None:
        # Production carries Parrinello-Rahman; NPT equilibration
        # uses Berendsen for fast pressure equilibration.
        prod = generate_production_mdp()
        equil = generate_equil_npt_mdp()
        assert "pcoupl           = parrinello-rahman" in prod
        assert "pcoupl           = berendsen" in equil

    @pytest.mark.parametrize(
        "generator",
        [generate_equil_nvt_mdp, generate_equil_npt_mdp, generate_production_mdp],
    )
    def test_tc_grps_defaults_to_system(self, generator: Any) -> None:
        """``tc-grps = System`` (single thermostat group).

        The default is a single thermostat group over the whole
        system — robust across force fields and force-field-specific
        group naming (the ``Protein`` / ``Non-Protein`` split
        assumes the operator's topology has those groups, which
        is force-field-specific and not always reliable). Callers
        that need a split group can override the .mdp file
        directly (the protocol writes the .mdp on disk; the
        runner does not regenerate it).
        """
        mdp = generator()
        assert "tc-grps          = System" in mdp
        # The legacy split-group spelling must NOT appear.
        assert "Protein Non-Protein" not in mdp


# ---------------------------------------------------------------------------
# .mdp contract — deterministic per-replica seeds
# ---------------------------------------------------------------------------


class TestMdpReplicaSeed:
    """``gen-seed`` is ``replica_index + 1`` (NOT ``-1``)."""

    @pytest.mark.parametrize(
        "generator",
        [
            generate_minimization_mdp,
            generate_equil_nvt_mdp,
            generate_equil_npt_mdp,
            generate_production_mdp,
        ],
    )
    def test_default_seed_is_one(self, generator: Any) -> None:
        mdp = generator()
        assert "gen-seed         = 1" in mdp
        assert "gen-seed         = -1" not in mdp

    def test_nvt_seed_scales_with_replica(self) -> None:
        mdp0 = generate_equil_nvt_mdp(replica_index=0)
        mdp1 = generate_equil_nvt_mdp(replica_index=1)
        mdp7 = generate_equil_nvt_mdp(replica_index=7)
        assert "gen-seed         = 1" in mdp0
        assert "gen-seed         = 2" in mdp1
        assert "gen-seed         = 8" in mdp7

    def test_production_seed_scales_with_replica(self) -> None:
        mdp0 = generate_production_mdp(replica_index=0)
        mdp3 = generate_production_mdp(replica_index=3)
        assert "gen-seed         = 1" in mdp0
        assert "gen-seed         = 4" in mdp3

    def test_replica_seed_is_deterministic_across_calls(self) -> None:
        # Same replica → same seed (reproducibility within a replica).
        mdp_a = generate_production_mdp(replica_index=2)
        mdp_b = generate_production_mdp(replica_index=2)
        assert mdp_a == mdp_b

    def test_rejects_negative_replica(self) -> None:
        with pytest.raises(ValueError, match="replica_index"):
            generate_equil_nvt_mdp(replica_index=-1)


# ---------------------------------------------------------------------------
# .mdp contract — input validation + determinism
# ---------------------------------------------------------------------------


class TestMdpValidation:
    """Inputs must satisfy positivity constraints; determinism holds."""

    def test_minimization_rejects_zero_iterations(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            generate_minimization_mdp(max_iterations=0)

    def test_minimization_rejects_negative_iterations(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            generate_minimization_mdp(max_iterations=-1)

    def test_nvt_rejects_zero_duration(self) -> None:
        with pytest.raises(ValueError, match="duration_ps"):
            generate_equil_nvt_mdp(duration_ps=0)

    def test_nvt_rejects_zero_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature_k"):
            generate_equil_nvt_mdp(temperature_k=0.0)

    def test_npt_rejects_zero_pressure(self) -> None:
        with pytest.raises(ValueError, match="pressure_bar"):
            generate_equil_npt_mdp(pressure_bar=0.0)

    def test_production_rejects_zero_duration(self) -> None:
        with pytest.raises(ValueError, match="duration_ns"):
            generate_production_mdp(duration_ns=0.0)

    def test_production_rejects_negative_duration(self) -> None:
        with pytest.raises(ValueError, match="duration_ns"):
            generate_production_mdp(duration_ns=-1.0)

    @pytest.mark.parametrize(
        "generator",
        [
            generate_minimization_mdp,
            generate_equil_nvt_mdp,
            generate_equil_npt_mdp,
            generate_production_mdp,
        ],
    )
    def test_byte_deterministic_for_same_input(self, generator: Any) -> None:
        # Catches a class of bug: nondeterministic .mdp content
        # (e.g. a timestamp or random number sneaking in).
        first = generator()
        second = generator()
        assert first == second

    def test_ions_mdp_is_zero_step(self) -> None:
        mdp = ions_mdp_content()
        assert "integrator       = steep" in mdp
        assert "nsteps           = 0" in mdp


# ---------------------------------------------------------------------------
# build_commands — grompp before mdrun, genion stdin metadata, nt=0 honor
# ---------------------------------------------------------------------------


class TestBuildCommands:
    """Command construction: grompp → mdrun, genion group selection, -nt gating."""

    def test_topology_emits_pdb2gmx(self) -> None:
        cfg = _valid_protocol_config()
        stages = build_stage_plan()
        cmds = build_commands(stages[0], checkpoint_path=None, config=cfg)
        assert len(cmds) == 1
        assert cmds[0][0] == "gmx"
        assert cmds[0][1] == "pdb2gmx"
        assert cfg.input_pdb in cmds[0]

    def test_box_uses_cubic_and_buffer_nm(self) -> None:
        cfg = _valid_protocol_config(box_buffer_nm=1.2)
        stages = build_stage_plan()
        cmds = build_commands(stages[1], checkpoint_path=None, config=cfg)
        assert "editconf" in cmds[0]
        assert "cubic" in cmds[0]
        assert "1.2" in cmds[0]

    def test_ions_emits_grompp_then_genion(self) -> None:
        cfg = _valid_protocol_config(ion_concentration_m=0.15)
        stages = build_stage_plan()
        cmds = build_commands(stages[3], checkpoint_path=None, config=cfg)
        assert len(cmds) == 2
        assert "grompp" in cmds[0]
        assert "genion" in cmds[1]
        assert "0.15" in cmds[1]
        assert "-neutral" in cmds[1]

    def test_md_stage_emits_grompp_then_mdrun(self) -> None:
        """The runner must call gmx grompp BEFORE gmx mdrun for every MD stage."""
        cfg = _valid_protocol_config()
        stages = build_stage_plan()
        for stage in stages[4:]:  # all four MD stages
            cmds = build_commands(
                stage,
                checkpoint_path=None,
                config=cfg,
            )
            assert len(cmds) == 2, f"{stage.kind.value}: expected grompp + mdrun"
            assert "grompp" in cmds[0], f"{stage.kind.value}: first cmd is not grompp"
            assert "mdrun" in cmds[1], f"{stage.kind.value}: second cmd is not mdrun"

    def test_grompp_passes_correct_previous_coordinates(self) -> None:
        """grompp -c points at the previous stage's .gro (RELATIVE path)."""
        cfg = _valid_protocol_config()
        stages = build_stage_plan()
        cmds = [
            build_commands(s, checkpoint_path=None, config=cfg)[0]
            for s in stages[4:]  # MD stages only (indices 4..7)
        ]
        grompp_min, grompp_nvt, grompp_npt, grompp_prod = cmds

        # Find the -c argument (always immediately after -c in the cmd).
        def _arg_after(cmd: list[str], flag: str) -> str:
            return cmd[cmd.index(flag) + 1]

        # All -c paths are RELATIVE (gmx resolves them against the
        # process cwd which the runner sets to work_dir). This keeps
        # the protocol container-mount-agnostic.
        assert _arg_after(grompp_min, "-c") == "ions.gro"
        assert _arg_after(grompp_nvt, "-c") == "min.gro"
        assert _arg_after(grompp_npt, "-c") == "nvt.gro"
        assert _arg_after(grompp_prod, "-c") == "npt.gro"

    def test_grompp_uses_relative_paths_for_container_compatibility(self) -> None:
        """All grompp flags that reference work-dir artifacts use relative paths.

        Container runtimes (Singularity, Apptainer, Docker) often
        mount the working directory at a path inside the container
        that differs from the host path. Relative paths avoid the
        mismatch — gmx resolves them against the process cwd.
        """
        cfg = _valid_protocol_config()
        stages = build_stage_plan()
        for stage in stages[4:]:  # MD stages
            grompp = build_commands(stage, checkpoint_path=None, config=cfg)[0]
            # The flags that reference work-dir artifacts MUST be relative.
            for flag in ("-c", "-p", "-o"):
                idx = grompp.index(flag)
                path = grompp[idx + 1]
                assert not path.startswith("/"), (
                    f"{stage.kind.value} grompp {flag}={path!r} is absolute; "
                    "should be relative for container compatibility"
                )

    def test_grompp_passes_topology(self) -> None:
        cfg = _valid_protocol_config()
        stages = build_stage_plan()
        grompp = build_commands(stages[4], checkpoint_path=None, config=cfg)[0]
        assert "topol.top" in grompp

    def test_mdrun_omits_cpi_when_checkpoint_absent(self) -> None:
        """The 'no duplicate start path' rule: -cpi is NOT emitted when no .cpt exists."""
        cfg = _valid_protocol_config()
        stages = build_stage_plan()
        for stage in stages[4:]:
            mdrun = build_commands(
                stage,
                checkpoint_path=None,
                config=cfg,
            )[1]
            assert "-cpi" not in mdrun, f"{stage.kind.value} emitted -cpi without a .cpt"

    def test_mdrun_emits_cpi_and_append_when_checkpoint_present(self) -> None:
        """When -cpi is emitted, -append is also emitted (idempotent on first resume)."""
        cfg = _valid_protocol_config()
        stages = build_stage_plan()
        cpi_path = "/tmp/wd/prod.cpt"
        mdrun = build_commands(
            stages[7],
            checkpoint_path=cpi_path,
            config=cfg,
        )[1]
        assert "-cpi" in mdrun
        idx = mdrun.index("-cpi")
        assert mdrun[idx + 1] == cpi_path
        assert "-append" in mdrun, "-append must accompany -cpi to be safe across GROMACS versions"

    def test_grompp_passes_cpt_via_t_flag_when_resuming(self) -> None:
        """On resume, grompp -t <cpt> preserves velocities/state from the checkpoint.

        Per GROMACS manual §Table 5.4, when -t is provided, grompp
        IGNORES the .mdp's ``gen-vel`` setting — so the
        equilibrated velocities in the .cpt are preserved across
        restart. This test asserts the runner emits -t <cpt> for
        grompp when resuming (the matching mdrun test asserts
        -cpi <cpt> for mdrun).
        """
        cfg = _valid_protocol_config()
        stages = build_stage_plan()
        cpi_path = "/tmp/wd/prod.cpt"
        grompp = build_commands(
            stages[7],
            checkpoint_path=cpi_path,
            config=cfg,
        )[0]
        assert "-t" in grompp
        idx = grompp.index("-t")
        assert grompp[idx + 1] == cpi_path

    def test_genvel_is_set_only_in_nvt_mdp(self) -> None:
        """``gen-vel = yes`` bootstraps velocities in NVT only.

        NPT and production inherit velocities from the previous
        stage's .gro/.cpt. On resume, ``gmx grompp -t <cpt>``
        IGNORES ``gen-vel`` per GROMACS manual §Table 5.4 — so the
        same NVT .mdp with ``gen-vel = yes`` is correct for both
        fresh start and resume.
        """
        nvt = generate_equil_nvt_mdp()
        npt = generate_equil_npt_mdp()
        prod = generate_production_mdp()
        assert "gen-vel          = yes" in nvt
        assert "gen-temp         = " in nvt  # Maxwell-Boltzmann target
        assert "gen-vel" not in npt
        assert "gen-vel" not in prod

    def test_genvel_is_ignored_on_resume_grompp_semantics(self) -> None:
        """GROMACS semantic: grompp -t <cpt> IGNORES gen-vel.

        We document this in the .mdp generation and the runner
        emits ``-t <cpt>`` for resume. A regression that removes
        ``-t`` from the grompp invocation on resume would let
        grompp regenerate velocities from gen-vel, silently
        replacing the equilibrated velocities with a fresh
        Maxwell-Boltzmann draw. This test asserts the runner
        still passes -t on resume (paired with the gen-vel=yes
        in NVT, the combination is the documented behaviour).
        """
        cfg = _valid_protocol_config()
        stages = build_stage_plan()
        cpi_path = "/tmp/wd/nvt.cpt"
        grompp_nvt = build_commands(
            stages[5],  # EQUIL_NVT
            checkpoint_path=cpi_path,
            config=cfg,
        )[0]
        # -t <cpt> is emitted on resume — grompp will ignore the
        # NVT .mdp's gen-vel=yes in favour of the .cpt's velocities.
        assert "-t" in grompp_nvt
        idx = grompp_nvt.index("-t")
        assert grompp_nvt[idx + 1] == cpi_path

    def test_nt_threads_zero_omits_nt_flag(self) -> None:
        """nt_threads=0 → -nt is NOT emitted (GROMACS auto-detects)."""
        cfg = _valid_protocol_config(nt_threads=0)
        stages = build_stage_plan()
        mdrun = build_commands(stages[7], checkpoint_path=None, config=cfg)[1]
        assert "-nt" not in mdrun

    def test_nt_threads_positive_emits_nt_flag(self) -> None:
        cfg = _valid_protocol_config(nt_threads=4)
        stages = build_stage_plan()
        mdrun = build_commands(stages[7], checkpoint_path=None, config=cfg)[1]
        idx = mdrun.index("-nt")
        assert mdrun[idx + 1] == "4"

    def test_extra_mdrun_flags_are_appended(self) -> None:
        cfg = _valid_protocol_config(extra_mdrun_flags=("-nb", "gpu"))
        stages = build_stage_plan()
        mdrun = build_commands(stages[7], checkpoint_path=None, config=cfg)[1]
        for flag in ("-nb", "gpu"):
            assert flag in mdrun

    def test_genion_group_selection_is_a_known_constant(self) -> None:
        """The genion stdin payload must be a module-level constant (not built via shell)."""
        assert GENION_INPUT == "SOL\n"
        # The protocol module's CODE (docstrings excluded) must NOT
        # use shell=True or execute ``sh -c ...`` as a subprocess
        # command. A regression here would silently introduce a
        # shell-injection vector for the SOL group selection.
        code = _protocol_code_only()
        assert "shell=True" not in code, "shell=True found in protocol.py code"
        assert "sh -c" not in code, "shell invocation found in protocol.py code"


# ---------------------------------------------------------------------------
# Config — box_buffer_nm / box_size_nm compat, force-field docs, replica bound
# ---------------------------------------------------------------------------


class TestBoxBufferCompat:
    """``box_size_nm`` is a deprecated alias for ``box_buffer_nm``."""

    def test_default_buffer_is_one_nm(self) -> None:
        cfg = _valid_protocol_config()
        assert cfg.box_buffer_nm == 1.0
        assert cfg.box_size_nm is None

    def test_box_buffer_nm_is_used_directly(self) -> None:
        cfg = _valid_protocol_config(box_buffer_nm=1.5)
        assert cfg.box_buffer_nm == 1.5
        assert cfg.box_size_nm is None

    def test_box_size_nm_alias_overrides_buffer(self) -> None:
        """The deprecated name wins when explicitly set (backward compat)."""
        cfg = _valid_protocol_config(box_size_nm=0.8)
        assert cfg.box_buffer_nm == 0.8
        assert cfg.box_size_nm == 0.8

    def test_validation_runs_on_resolved_buffer(self) -> None:
        with pytest.raises(ValueError, match="box_buffer_nm"):
            GromacsProtocolConfig(
                name="x",
                input_pdb="/tmp/x.pdb",
                output_root="/tmp/out",
                box_buffer_nm=0.0,
            )


class TestConfigValidation:
    """Required fields and ranges are enforced."""

    def test_rejects_missing_name(self) -> None:
        with pytest.raises(ValueError, match="name is required"):
            GromacsProtocolConfig(name="", input_pdb="/tmp/x.pdb", output_root="/tmp/out")

    def test_rejects_missing_input_pdb(self) -> None:
        with pytest.raises(ValueError, match="input_pdb is required"):
            GromacsProtocolConfig(name="x", input_pdb="", output_root="/tmp/out")

    def test_rejects_missing_output_root(self) -> None:
        with pytest.raises(ValueError, match="output_root is required"):
            GromacsProtocolConfig(name="x", input_pdb="/tmp/x.pdb", output_root="")

    def test_rejects_negative_ion_concentration(self) -> None:
        with pytest.raises(ValueError, match="ion_concentration_m"):
            GromacsProtocolConfig(
                name="x",
                input_pdb="/tmp/x.pdb",
                output_root="/tmp/out",
                ion_concentration_m=-0.1,
            )

    def test_rejects_replica_index_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="replica_index"):
            GromacsProtocolConfig(
                name="x",
                input_pdb="/tmp/x.pdb",
                output_root="/tmp/out",
                replica_index=5,
                replicas_total=3,
            )

    def test_effective_production_ns_prefers_screening(self) -> None:
        cfg = GromacsProtocolConfig(
            name="x",
            input_pdb="/tmp/x.pdb",
            output_root="/tmp/out",
            production_ns=200.0,
            screening_ns=5.0,
        )
        assert cfg.effective_production_ns() == 5.0

    def test_effective_production_ns_falls_back_to_production_ns(self) -> None:
        cfg = GromacsProtocolConfig(
            name="x",
            input_pdb="/tmp/x.pdb",
            output_root="/tmp/out",
            production_ns=50.0,
        )
        assert cfg.effective_production_ns() == 50.0


class TestForceFieldDefault:
    """The default force field is documented as a convention, not a bundled availability claim."""

    def test_documented_default_is_charmm36m(self) -> None:
        cfg = _valid_protocol_config()
        assert cfg.force_field == "charmm36m"

    def test_docstring_does_not_claim_bundled_availability(self) -> None:
        # The docstring must explicitly say GROMACS does not bundle
        # force fields. It must NOT use phrases that imply bundling.
        import biolab_runners.gromacs.config as c

        src = Path(c.__file__).read_text()
        # Extract the GromacsProtocolConfig docstring only.
        # Find the class definition and read until the next class/def.
        lines = src.splitlines()
        in_docstring = False
        docstring_lines: list[str] = []
        for line in lines:
            if "class GromacsProtocolConfig" in line:
                in_docstring = True
                continue
            if in_docstring:
                if line.strip().startswith('"""') and docstring_lines:
                    break
                if line.strip().startswith('"""') and not docstring_lines:
                    continue
                docstring_lines.append(line)
        docstring = "\n".join(docstring_lines)
        # Must explicitly state GROMACS does not include force fields
        # out of the box (the docstring uses "does not bundle" OR
        # "does not include" — either phrasing is acceptable as long
        # as the operator knows they must install the FF locally).
        lowered = docstring.lower()
        assert "does not bundle" in lowered or "does not include" in lowered, (
            "GromacsProtocolConfig docstring must clarify that GROMACS does not "
            "include force fields out of the box"
        )
        # Must NOT use forbidden bundling language (other than in
        # negated contexts like "does not bundle").
        for forbidden in ("ships with", "comes with"):
            assert forbidden not in lowered, (
                f"GromacsProtocolConfig docstring must not claim force field is {forbidden} GROMACS"
            )
        # "bundled" is only allowed inside a negation.
        if "bundled" in lowered:
            # Must be in a negated context — e.g. "does not bundle" /
            # "is not bundled" / "is not a bundled claim".
            assert any(
                phrase in lowered
                for phrase in (
                    "does not bundle",
                    "not bundled",
                    "not a bundled",
                )
            ), "the word 'bundled' must be in a negated context"


# ---------------------------------------------------------------------------
# Stage manifest — round-trip, atomic save, corrupt tolerance
# ---------------------------------------------------------------------------


class TestStageMinimumOutputs:
    """Stage-specific minimum outputs — what the disk fallback actually checks.

    These tests exist to catch a class of bug where the fallback
    requires IMPOSSIBLE artifacts (e.g. ``.xtc`` on a 1 ps
    minimisation that never reaches ``nstxtcout``) and forces the
    runner to re-run every short simulation.

    The minimum set MUST be a strict subset of the full set
    (``stage_outputs_for``) and MUST contain only artifacts that
    ``gmx`` writes UNCONDITIONALLY when the stage finishes.
    """

    def test_minimum_is_strict_subset_of_full(self) -> None:
        for stage in build_stage_plan():
            minimum = stage_minimum_outputs(stage.kind, stage.prefix)
            full = stage_outputs_for(stage.kind, stage.prefix)
            assert set(minimum).issubset(set(full)), (
                f"{stage.kind.value}: minimum {minimum} is not a subset of full {full}"
            )

    def test_minimum_does_not_require_xtc_for_short_md_stages(self) -> None:
        """``.xtc`` is only written at ``nstxtcout`` — short runs won't have it.

        NVT is 100 ps × 0.002 fs = 50 000 steps; with nstxtcout=5000
        that DOES fire (10 frames). NPT same. Production 200 ns
        with nstxtcout=50000 also fires. BUT a 1 ps minimisation
        (500 steps) does NOT reach nstxtcout=5000 — so ``.xtc`` is
        not in the minimum for MINIMIZE.
        """
        min_outputs = stage_minimum_outputs(StageKind.MINIMIZE, "min")
        assert "min.xtc" not in min_outputs
        assert "min.trr" not in min_outputs
        assert "min.cpt" not in min_outputs  # intermittent

    def test_minimum_requires_gro_edr_log_for_md_stages(self) -> None:
        """Every MD stage must include ``.gro`` / ``.edr`` / ``.log``.

        ``.tpr`` is also guaranteed (it's the compiled input
        grompp writes regardless of whether mdrun runs).
        """
        for kind in (
            StageKind.MINIMIZE,
            StageKind.EQUIL_NVT,
            StageKind.EQUIL_NPT,
            StageKind.PRODUCTION,
        ):
            minimum = stage_minimum_outputs(kind, "min")
            for required in (".tpr", ".gro", ".edr", ".log"):
                assert any(name.endswith(required) for name in minimum), (
                    f"{kind.value}: minimum {minimum} missing required {required}"
                )

    def test_setup_stages_have_minimum_outputs(self) -> None:
        """TOPOLOGY / BOX / SOLVATE / IONS each have stage-specific minimums."""
        assert stage_minimum_outputs(StageKind.TOPOLOGY, "topol") == ("topol.top", "processed.gro")
        assert stage_minimum_outputs(StageKind.BOX, "box") == ("boxed.gro",)
        assert stage_minimum_outputs(StageKind.SOLVATE, "solvate") == ("solvated.gro",)
        assert stage_minimum_outputs(StageKind.IONS, "ions") == ("ions.tpr", "ions.gro")


class TestStageManifest:
    """Tests for the structured stage manifest round-trip."""

    def test_load_returns_empty_when_absent(self, tmp_path: Path) -> None:
        manifest = load_stage_manifest(tmp_path)
        assert manifest == {"schema_version": 1, "stages": {}}

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        stages = {StageKind.MINIMIZE.value: {"status": "completed"}}
        manifest = {"schema_version": 1, "stages": stages}
        save_stage_manifest(tmp_path, manifest)
        loaded = load_stage_manifest(tmp_path)
        assert loaded["stages"][StageKind.MINIMIZE.value]["status"] == "completed"

    def test_record_stage_status_persists(self, tmp_path: Path) -> None:
        record_stage_status(
            tmp_path,
            StageKind.MINIMIZE.value,
            StageStatus.COMPLETED,
            outputs=("min.cpt", "min.edr"),
        )
        loaded = load_stage_manifest(tmp_path)
        record = loaded["stages"][StageKind.MINIMIZE.value]
        assert record["status"] == StageStatus.COMPLETED
        assert "min.cpt" in record["outputs"]

    def test_record_stage_status_rejects_invalid_status(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid status"):
            record_stage_status(tmp_path, StageKind.MINIMIZE.value, "bogus")

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        """No .tmp file is left on disk after save."""
        save_stage_manifest(tmp_path, {"schema_version": 1, "stages": {}})
        assert list(tmp_path.glob("*.tmp")) == []

    def test_load_tolerates_corrupt_manifest(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / GromacsFiles.STAGE_MANIFEST
        manifest_path.write_text("{not json")
        manifest = load_stage_manifest(tmp_path)
        assert manifest["stages"] == {}


# ---------------------------------------------------------------------------
# Runner — skip / resume / accounting / SIGTERM preservation
# ---------------------------------------------------------------------------


class TestRunnerSkipAndResume:
    """Skip semantics (manifest authority) + resume (``-cpi`` iff .cpt)."""

    def test_skip_when_manifest_already_completed(self, tmp_path: Path) -> None:
        cfg = _valid_protocol_config(output_root=str(tmp_path))
        work_dir = tmp_path / cfg.name
        # Pre-mark EVERY stage as COMPLETED in the manifest.
        for stage in build_stage_plan():
            record_stage_status(
                work_dir,
                stage.kind.value,
                StageStatus.COMPLETED,
                outputs=stage_minimum_outputs(stage.kind, stage.prefix),
            )
        runner = GromacsProtocolRunner()
        with patch.object(GromacsProtocolRunner, "_run_subprocess") as mock_run:
            result = runner.run_protocol(cfg)
        mock_run.assert_not_called()
        # Every stage was marked COMPLETED in the result.
        for stage in build_stage_plan():
            assert result.stage_statuses[stage.kind.value] == StageStatus.COMPLETED
        # All 8 stages were "skipped" (manifest authority).
        assert result.skipped == 8
        assert result.succeeded == 0

    def test_disk_output_fallback_succeeds_with_minimum_outputs(self, tmp_path: Path) -> None:
        """Spot reclaim recovery: manifest lost, only minimum outputs on disk.

        The disk-fallback check uses ``stage_minimum_outputs`` (NOT
        the full set), so a Spot reclaim that left ``.tpr``, ``.gro``,
        ``.edr``, ``.log`` on disk — but lost the intermittent
        ``.cpt`` / ``.xtc`` / ``.trr`` — is correctly recognised as
        complete. Without this fallback, the runner would re-run
        every short simulation that didn't reach ``nstxtcout``.
        """
        cfg = _valid_protocol_config(output_root=str(tmp_path))
        work_dir = tmp_path / cfg.name
        # Manifest is empty (Spot reclaim simulated). Pre-create ONLY
        # the minimum outputs.
        for stage in build_stage_plan():
            for name in stage_minimum_outputs(stage.kind, stage.prefix):
                p = work_dir / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("placeholder")

        runner = GromacsProtocolRunner()
        with patch.object(GromacsProtocolRunner, "_run_subprocess") as mock_run:
            result = runner.run_protocol(cfg)
        # Every stage is recognised as complete via disk fallback;
        # NO subprocess was launched.
        mock_run.assert_not_called()
        assert result.exit_code == 0
        # All 8 stages recorded as COMPLETED.
        for stage in build_stage_plan():
            assert result.stage_statuses[stage.kind.value] == StageStatus.COMPLETED
        assert result.skipped == 8

    def test_disk_output_fallback_fails_without_force_when_outputs_missing(
        self, tmp_path: Path
    ) -> None:
        """Without ``force=True`` and without minimum outputs, the runner runs."""
        cfg = _valid_protocol_config(output_root=str(tmp_path))
        # No outputs, no manifest entries. Runner should run.
        runner = GromacsProtocolRunner()
        with patch.object(GromacsProtocolRunner, "_run_subprocess", return_value=0) as mock_run:
            runner.run_protocol(cfg)
        # At least one subprocess call (setup stages don't have .cpt
        # so they're always fresh; the runner does launch them).
        assert mock_run.call_count >= 1

    def test_disk_output_fallback_force_true_overrides(self, tmp_path: Path) -> None:
        """force=True bypasses the disk fallback even when outputs exist."""
        cfg = _valid_protocol_config(output_root=str(tmp_path), force=True)
        work_dir = tmp_path / cfg.name
        # Pre-create ONLY the minimum outputs.
        for stage in build_stage_plan():
            for name in stage_minimum_outputs(stage.kind, stage.prefix):
                p = work_dir / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("placeholder")

        runner = GromacsProtocolRunner()
        with patch.object(GromacsProtocolRunner, "_run_subprocess") as mock_run:
            runner.run_protocol(cfg)
        # force=True bypasses the disk fallback; every stage re-runs.
        assert mock_run.call_count > 0

    def test_force_re_runs_even_when_manifest_complete(self, tmp_path: Path) -> None:
        cfg = _valid_protocol_config(output_root=str(tmp_path), force=True)
        work_dir = tmp_path / cfg.name
        # Pre-populate every output AND mark every stage COMPLETED.
        for stage in build_stage_plan():
            for name in stage_outputs_for(stage.kind, stage.prefix):
                p = work_dir / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("placeholder")
            record_stage_status(
                work_dir,
                stage.kind.value,
                StageStatus.COMPLETED,
                outputs=stage_outputs_for(stage.kind, stage.prefix),
            )

        runner = GromacsProtocolRunner()
        with patch.object(GromacsProtocolRunner, "_run_subprocess", return_value=0) as mock_run:
            result = runner.run_protocol(cfg)
        # 8 stages × ≥2 commands (MD=2; IONS=2; setup=1) → at least 11 calls.
        assert mock_run.call_count >= 11
        assert result.exit_code == 0


class TestRunnerAccounting:
    """Per-stage counters: skipped, succeeded, failed, interrupted are distinct."""

    def test_dry_run_emits_mdps_without_writing_manifest_record(self, tmp_path: Path) -> None:
        """Dry-run emits the .mdp preview files but DOES NOT mark any stage COMPLETED.

        The previous implementation wrote COMPLETED to the
        manifest on every dry-run stage, which silently turned a
        dry-run into a "skip every stage on the next real run"
        foot-gun. The fix is to leave the manifest empty so a
        subsequent real run exercises every stage from scratch.
        """
        cfg = _valid_protocol_config(output_root=str(tmp_path))
        work_dir = tmp_path / cfg.name
        runner = GromacsProtocolRunner(dry_run=True)
        with patch.object(GromacsProtocolRunner, "_run_subprocess") as mock_run:
            result = runner.run_protocol(cfg)
        mock_run.assert_not_called()
        assert result.exit_code == 0
        assert result.dry_run is True

        # The .mdp files ARE written (the operator may want to
        # inspect them before committing to a real run).
        from biolab_runners.gromacs.paths import GromacsFiles

        for mdp in (
            GromacsFiles.MIN_MDP,
            GromacsFiles.NVT_MDP,
            GromacsFiles.NPT_MDP,
            GromacsFiles.PROD_MDP,
        ):
            assert (work_dir / mdp).is_file(), f"dry-run must emit {mdp}"

        # The manifest stays EMPTY — a subsequent real run will
        # run every stage from scratch instead of skipping them all.
        manifest = load_stage_manifest(work_dir)
        assert manifest["stages"] == {}, (
            f"dry-run must NOT write a terminal manifest record; got {manifest}"
        )

        # Every stage is reported as ``validated`` (not succeeded).
        assert result.validated == 8
        assert result.succeeded == 0
        assert result.skipped == 0
        assert result.failed == 0
        assert all(s == "validated" for s in result.stage_statuses.values())

    def test_dry_run_then_real_run_invokes_every_stage(self, tmp_path: Path) -> None:
        """A real run on a fresh ``work_dir`` that was just dry-run-validated
        MUST invoke every stage — the dry-run must NOT have made the
        stages skip-eligible.

        This is the regression test for the foot-gun where dry-run
        wrote COMPLETED to the manifest and a subsequent real run
        on the same ``work_dir`` silently skipped every stage.
        """
        cfg = _valid_protocol_config(output_root=str(tmp_path))
        work_dir = tmp_path / cfg.name

        # 1. Dry run on the same work_dir.
        dry_runner = GromacsProtocolRunner(dry_run=True)
        with patch.object(GromacsProtocolRunner, "_run_subprocess") as dry_mock:
            dry_result = dry_runner.run_protocol(cfg)
        dry_mock.assert_not_called()
        assert dry_result.validated == 8

        # 2. Real run on the same work_dir.
        real_runner = GromacsProtocolRunner()
        with patch.object(GromacsProtocolRunner, "_run_subprocess", return_value=0) as real_mock:
            real_result = real_runner.run_protocol(cfg)

        # The real run must invoke subprocess for every stage —
        # the dry-run must not have made any of them skip-eligible.
        assert real_mock.call_count >= 8, (
            f"real run only invoked {real_mock.call_count} subprocesses "
            f"after a dry-run; expected ≥ 8 (one per stage)"
        )
        assert real_result.skipped == 0, (
            f"real run skipped {real_result.skipped} stages after a "
            f"dry-run; dry-run must not make stages skip-eligible"
        )
        assert real_result.succeeded == 8
        assert real_result.dry_run is False

        # The manifest now correctly records COMPLETED for each
        # stage (from the real run, not from the dry-run).
        manifest = load_stage_manifest(work_dir)
        for stage in build_stage_plan():
            assert manifest["stages"][stage.kind.value]["status"] == StageStatus.COMPLETED, (
                f"{stage.kind.value} must be COMPLETED after the real run; "
                f"got {manifest['stages'][stage.kind.value]}"
            )

    def test_dry_run_does_not_count_as_skipped(self, tmp_path: Path) -> None:
        """The ``skipped`` counter must NOT include dry-run stages.

        The skipped counter is reserved for stages that were
        COMPLETED in a previous invocation. Dry-run stages are
        ``validated`` — a separate counter that the operator can
        inspect to confirm the dry-run actually ran.
        """
        cfg = _valid_protocol_config(output_root=str(tmp_path))
        runner = GromacsProtocolRunner(dry_run=True)
        with patch.object(GromacsProtocolRunner, "_run_subprocess") as mock_run:
            result = runner.run_protocol(cfg)
        mock_run.assert_not_called()
        assert result.skipped == 0
        assert result.succeeded == 0
        assert result.validated == 8

    def test_dry_run_does_not_count_as_succeeded(self, tmp_path: Path) -> None:
        """The ``succeeded`` counter must NOT include dry-run stages.

        The succeeded counter is reserved for stages that ran
        fresh and exited zero. A dry-run never invokes subprocess,
        so it cannot have "succeeded" in the operational sense.
        """
        cfg = _valid_protocol_config(output_root=str(tmp_path))
        runner = GromacsProtocolRunner(dry_run=True)
        with patch.object(GromacsProtocolRunner, "_run_subprocess") as mock_run:
            result = runner.run_protocol(cfg)
        mock_run.assert_not_called()
        assert result.succeeded == 0
        assert result.validated == 8

    def test_subprocess_failure_increments_failed_and_short_circuits(self, tmp_path: Path) -> None:
        cfg = _valid_protocol_config(output_root=str(tmp_path))
        runner = GromacsProtocolRunner()
        with patch.object(GromacsProtocolRunner, "_run_subprocess", return_value=1):
            result = runner.run_protocol(cfg)
        assert result.exit_code == 1
        assert result.failed == 1
        assert result.succeeded == 0
        assert result.error  # populated with stage name + rc

    def test_subprocess_success_increments_succeeded(self, tmp_path: Path) -> None:
        cfg = _valid_protocol_config(output_root=str(tmp_path))
        runner = GromacsProtocolRunner()
        with patch.object(GromacsProtocolRunner, "_run_subprocess", return_value=0):
            result = runner.run_protocol(cfg)
        assert result.exit_code == 0
        assert result.failed == 0
        assert result.succeeded == 8

    def test_interruption_increments_interrupted_not_failed(self, tmp_path: Path) -> None:
        """SIGTERM halts at the first interrupted stage; only that stage counts.

        The runner MUST NOT continue past an interrupted stage —
        a missing-input FAILED on the next stage would overwrite
        the truthful interruption result. ``interrupted=1``,
        ``failed=0``, ``exit_code=-SIGTERM``.
        """
        cfg = _valid_protocol_config(output_root=str(tmp_path))
        runner = GromacsProtocolRunner()
        with patch.object(GromacsProtocolRunner, "_run_subprocess", return_value=-15):
            result = runner.run_protocol(cfg)
        # Only the first stage (topology) is recorded as interrupted;
        # the loop halted there so the truthful exit_code is preserved.
        assert result.interrupted == 1
        assert result.failed == 0
        assert result.exit_code == -15
        assert StageKind.TOPOLOGY.value in result.stage_statuses

    def test_per_replica_subdirectory_under_replicas_total(self, tmp_path: Path) -> None:
        cfg = _valid_protocol_config(
            output_root=str(tmp_path),
            replica_index=2,
            replicas_total=3,
        )
        runner = GromacsProtocolRunner(dry_run=True)
        with patch.object(GromacsProtocolRunner, "_run_subprocess") as mock_run:
            result = runner.run_protocol(cfg)
        mock_run.assert_not_called()
        assert "rep002" in result.output_dir
        assert result.replica_index == 2
        assert result.replicas_total == 3


class TestRunnerSigtermPreservation:
    """SIGTERM in the parent is forwarded to the child; the stage is left resumable."""

    def test_sigterm_does_not_mark_stage_failed(self, tmp_path: Path) -> None:
        cfg = _valid_protocol_config(output_root=str(tmp_path))
        runner = GromacsProtocolRunner()
        work_dir = tmp_path / cfg.name
        work_dir.mkdir(parents=True, exist_ok=True)

        # Simulate the parent receiving SIGTERM after launching
        # subprocess: the mock _run_subprocess returns -SIGTERM,
        # which the runner treats as "interrupted, resumable".
        with patch.object(GromacsProtocolRunner, "_run_subprocess", return_value=-15):
            runner._run_single_stage(
                work_dir,
                build_stage_plan()[4],  # MINIMIZE
                cfg,
            )
        # Manifest must show RUNNING (NOT failed) so the next
        # invocation resumes from the .cpt.
        manifest = load_stage_manifest(work_dir)
        record = manifest["stages"][StageKind.MINIMIZE.value]
        assert record["status"] == StageStatus.RUNNING
        assert "interrupted" in record["error"]
        assert record["status"] != StageStatus.FAILED

    def test_subprocess_installs_sigterm_handler(self, tmp_path: Path) -> None:
        """The runner installs a SIGTERM handler in the parent so the cloud
        scheduler's reclaim signal is forwarded to the child gmx process."""
        runner = GromacsProtocolRunner()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = ("", "")

        with (
            patch("biolab_runners.gromacs.runner.subprocess.Popen", return_value=mock_proc),
            patch("biolab_runners.gromacs.runner.signal.signal") as mock_signal,
        ):
            runner._run_subprocess(["gmx", "mdrun"], tmp_path, timeout_seconds=60)

        # The signal.signal call registered a SIGTERM handler.
        sigterm_calls = [
            call for call in mock_signal.call_args_list if call.args[0] == signal.SIGTERM
        ]
        assert len(sigterm_calls) >= 1, (
            "expected signal.signal to register a SIGTERM handler; "
            f"got calls: {mock_signal.call_args_list}"
        )

    def test_genion_receives_stdin_input_not_shell(self, tmp_path: Path) -> None:
        """gmx genion is invoked with stdin=GENION_INPUT — NEVER sh -c.

        We mock ``subprocess.Popen`` and assert that ``communicate``
        was called with the expected ``input=GENION_INPUT.encode()``
        payload, AND that no shell invocation (``sh -c``,
        ``shell=True``) appears in the runner source.
        """
        runner = GromacsProtocolRunner()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = ("", "")

        with patch(
            "biolab_runners.gromacs.runner.subprocess.Popen", return_value=mock_proc
        ) as mock_popen:
            genion_cmd = ["gmx", "genion", "-s", "ions.tpr", "-o", "ions.gro"]
            runner._run_subprocess(genion_cmd, tmp_path, timeout_seconds=60)

        # Popen was called with stdin=PIPE (so communicate can feed input).
        kwargs = mock_popen.call_args.kwargs
        assert kwargs.get("stdin") is not None, (
            "genion invocation must have stdin=PIPE so the SOL group selection can be sent"
        )

        # communicate() was called with the GENION_INPUT payload. The
        # runner uses ``text=True`` on Popen, so the payload is a str
        # (not bytes).
        call_kwargs = mock_proc.communicate.call_args.kwargs
        assert call_kwargs.get("input") == GENION_INPUT, (
            f"expected input={GENION_INPUT!r}; got {call_kwargs.get('input')!r}"
        )

        # The runner source's CODE (docstrings excluded) must NOT
        # use shell invocation.
        code = _runner_code_only()
        assert "shell=True" not in code, "shell=True found in runner.py code"
        assert "sh -c" not in code, "shell invocation found in runner.py code"

    def test_sigterm_grace_does_not_block_24h(self, tmp_path: Path) -> None:
        """The SIGTERM grace timer runs in a background thread, NOT on the communicate timeout.

        The 24 h ``timeout_seconds`` cap on ``proc.communicate`` is
        the OVERALL cap for any one ``gmx`` invocation. The
        ``sigterm_grace_seconds`` cap on the SIGTERM→SIGKILL
        escalation runs in a separate daemon ``threading.Timer`` —
        the two are INDEPENDENT. A regression that moved the
        grace logic onto the communicate timeout would either
        block for 24 h on a stuck child or never escalate.
        """
        runner = GromacsProtocolRunner(sigterm_grace_seconds=0.1)

        # Mock a child whose communicate() blocks long enough for the
        # SIGTERM handler + grace timer to fire, then returns. This
        # matches a real gmx mdrun that ignores SIGTERM (the kill
        # escalation is the contract for that case).
        mock_proc = MagicMock()
        mock_proc.returncode = -signal.SIGTERM

        communicate_started = threading.Event()
        proceed_with_communicate = threading.Event()
        kill_called = threading.Event()

        def _slow_communicate(*args: Any, **kwargs: Any) -> tuple[str, str]:
            communicate_started.set()
            proceed_with_communicate.wait(timeout=2.0)
            return ("", "")

        def _kill() -> None:
            kill_called.set()

        mock_proc.communicate.side_effect = _slow_communicate
        mock_proc.kill.side_effect = _kill

        captured_handler: list[Any] = []

        def _capture_signal(signum: int, handler: Any) -> Any:
            if signum == signal.SIGTERM:
                captured_handler.append(handler)
            return lambda *_: None

        with (
            patch("biolab_runners.gromacs.runner.subprocess.Popen", return_value=mock_proc),
            patch("biolab_runners.gromacs.runner.signal.signal", side_effect=_capture_signal),
        ):
            result_holder: dict[str, Any] = {}

            def _runner_thread() -> None:
                try:
                    rc = runner._run_subprocess(["gmx", "mdrun"], tmp_path, timeout_seconds=600)
                    result_holder["rc"] = rc
                except Exception as exc:
                    result_holder["exc"] = exc

            t = threading.Thread(target=_runner_thread)
            t.start()
            # Wait for the runner to enter communicate().
            assert communicate_started.wait(timeout=1.0)
            # Wait for the runner to install its SIGTERM handler
            # (it does so BEFORE calling communicate()).
            for _ in range(20):
                if captured_handler:
                    break
                time.sleep(0.01)
            assert captured_handler, "SIGTERM handler was not installed by _run_subprocess"
            # Trigger the handler — simulating the parent
            # receiving SIGTERM from the cloud scheduler.
            captured_handler[0](signal.SIGTERM, None)
            # The grace timer (100 ms) must fire kill() before
            # communicate() returns (we hold the gate).
            assert kill_called.wait(timeout=2.0), (
                "expected SIGKILL to be dispatched within the grace window"
            )
            # Let communicate() return so the runner thread winds down.
            proceed_with_communicate.set()
            t.join(timeout=3.0)
        # The runner returned the SIGTERM sentinel — the parent
        # honoured the signal without blocking on the overall cap.
        assert result_holder.get("rc") == -signal.SIGTERM


# ---------------------------------------------------------------------------
# Backward compatibility — the S3 public API is untouched
# ---------------------------------------------------------------------------


class TestLegacyApiPreserved:
    """The S3 public API must remain importable and usable."""

    def test_gromacs_config_still_works(self) -> None:
        cfg = GromacsConfig(
            name="legacy",
            structure_file="/tmp/run.tpr",
            topology_file="/tmp/topol.top",
        )
        assert cfg.name == "legacy"

    def test_gromacs_runner_still_constructible(self, tmp_path: Path) -> None:
        runner = GromacsRunner(
            output_root=tmp_path,
            config=GromacsConfig(
                name="x",
                structure_file="/tmp/run.tpr",
                topology_file="/tmp/topol.top",
            ),
        )
        assert runner.output_root == tmp_path

    def test_gromacs_available_returns_false_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROMACS_BIN", "/nonexistent/gmx")
        assert gromacs_available() is False

    def test_parse_nthcol_energy_still_works(self, tmp_path: Path) -> None:
        from biolab_runners.gromacs.utils import parse_nthcol_energy

        p = tmp_path / "energy.xvg"
        p.write_text("0.0  -123.456\n")
        assert parse_nthcol_energy(p, column=1) == -123.456
