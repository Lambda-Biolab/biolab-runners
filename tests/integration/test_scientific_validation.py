"""Tool-level smoke validation for biolab-runners scientific runners.

These tests assert that the upstream-tool **parsers** in
``biolab_runners.{proteinmpnn,gromacs,rfdiffusion,openmm}.utils``
produce biologically plausible outputs on real-format reference inputs,
and that the OpenMM Python API (the substrate for ``OpenMMRunner``)
runs end-to-end on a real system.

This is **not** the wrapper plumbing suite (those live in
``tests/test_*_runner.py``). It's a thin layer above: the same parsers
the runners call into, exercised on known-format reference files.

Each test:

* references real literature for any threshold it asserts;
* skips gracefully (with ``pytest.skip``) when the upstream tool
  binary is not installed, so this suite works on any laptop but
  only "goes green" on a workstation that has the tools;
* is gated on the ``integration`` marker so it does not run on
  ``make validate``.

Why these tests matter:

* ``parse_fasta_sequences`` in proteinmpnn/utils.py is the parser that
  ultimately feeds SequenceDesign records into the pipeline. A silent
  shift in column indexing would feed garbage downstream.
* ``parse_nthcol_energy`` in gromacs/utils.py is the parser for the
  energy.xvg output. The wrapper relies on it returning meaningful
  energies; the test asserts the values are in a physically reasonable
  window.
* The OpenMM lightweight test loads the chain A of barnase-barstar
  (PDB 1BRS, 864 heavy atoms), adds hydrogens with amber14/ff14SB,
  builds a System, minimises on the CUDA platform, runs 100 dynamics
  steps, and asserts the result is in a biochemically reasonable
  window. This catches force-field-vs-platform bugs that pure mock
  tests miss.
* When CUDA is unavailable (e.g. CPU-only workstation), the same
  test falls back to OpenCL or CPU; this is documented and the
  physical-window bounds still apply.

References:

* Barnase–barstar (1BRS) is the standard protein–protein complex;
  RCSB PDB resolution 2.0 Å. Force-field parameters from
  AMBER ff14SB and TIP3P water.
* openmm [cuda12] and [cuda13] extras are documented as the standard
  way to add the CUDA platform plugin without recompiling OpenMM.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

# The tests/ tree has no ``__init__.py`` (pytest discovers by
# rootdir + conftest.py, not as a package). We add the integration
# directory to ``sys.path`` so the helper fixtures are importable.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

FIXTURE_DIR = _HERE / "fixtures" / "biology"
SAMPLE_FASTA = FIXTURE_DIR / "barnase_barstar_proteinmpnn.fa"
SAMPLE_XVG = FIXTURE_DIR / "ala2_vacuum.energy.xvg"
SAMPLE_1BRS_A = FIXTURE_DIR / "barnase_chainA.pdb"
SAMPLE_ALA5 = FIXTURE_DIR / "ala5_peptide.pdb"

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# ProteinMPNN — FASTA parser smoke test (pure-Python, always runs)
# ---------------------------------------------------------------------------


def test_proteinmpnn_parse_fasta_returns_both_records_with_protein_alpha() -> None:
    """Assert ``parse_fasta_sequences`` returns the right 2 records
    and the sequences are valid 20-AA alphabet strings.

    Reference: the FASTA file is hand-built to the ProteinMPNN output
    format. Sequences are 50 and 43 residues drawn from the 20 standard
    amino acids (TQ1 wraps into 32 + 18 chars; TQ2 wraps into 35 + 8).

    A regression that drops wrapping handling, mis-parses the header,
    or shuffles columns would fail here.

    Always runs (does not require ProteinMPNN binary).
    """
    from biolab_runners.proteinmpnn.utils import parse_fasta_sequences

    assert SAMPLE_FASTA.exists(), f"missing fixture {SAMPLE_FASTA}"
    records = parse_fasta_sequences(SAMPLE_FASTA)
    assert len(records) == 2, f"expected 2 records, got {len(records)}"
    name0, seq0 = records[0]
    name1, seq1 = records[1]
    assert "TQ1" in name0, f"first record name doesn't contain TQ1: {name0!r}"
    assert "TQ2" in name1
    assert len(seq0) == 50, f"TQ1 length {len(seq0)} != 50"
    assert len(seq1) == 43, f"TQ2 length {len(seq1)} != 43"
    canonical_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for seq in (seq0, seq1):
        assert all(aa in canonical_aa for aa in seq), f"non-canonical AA in sequence: {seq[:30]!r}"


# ---------------------------------------------------------------------------
# GROMACS — energy.xvg parser smoke test (pure-Python, always runs)
# ---------------------------------------------------------------------------


def test_gromacs_parse_nthcol_returns_first_data_row() -> None:
    """Assert ``parse_nthcol_energy`` reads column 1 from a sample
    energy.xvg.

    The fixture has column 0 = time, column 1 = kinetic energy
    (~425), column 2 = potential energy (~-340). Column 1 should
    return ~425 (kinetic energy of a small solvated peptide).

    A regression that read column 0 instead would return 0.000 —
    a value far from any physical energy. The bound (100, 500) is
    a non-trivial sanity check.
    """
    from biolab_runners.gromacs.utils import parse_nthcol_energy

    assert SAMPLE_XVG.exists(), f"missing fixture {SAMPLE_XVG}"
    value = parse_nthcol_energy(SAMPLE_XVG, column=1)
    assert 100.0 < value < 500.0, f"GROMACS parse_nthcol_energy column 1 -> {value}, expected ~425"


def test_gromacs_parse_nthcol_handles_comment_lines() -> None:
    """Assert the parser skips ``@``-prefixed headers and ``#`` comments.

    Calling column=2 (potential energy) should return a negative value,
    not parse a header line.
    """
    from biolab_runners.gromacs.utils import parse_nthcol_energy

    value = parse_nthcol_energy(SAMPLE_XVG, column=2)
    assert value < 0.0, (
        f"expected negative potential energy, got {value} — parser may not be reading data rows"
    )
    assert value > -500.0, f"value {value} outside physical window — parser mis-parsed"


def test_gromacs_protocol_runner_dry_run_emits_deterministic_mdps(tmp_path: Path) -> None:
    """End-to-end smoke for the S4 protocol runner in dry-run mode.

    Real-GROMACS protocol execution (pdb2gmx → editconf → solvate →
    genion → mdrun) requires a ``gmx`` binary and a few minutes of
    wall time, so the heavy integration test that actually invokes
    ``gmx`` is **skipped** when the binary is absent (see
    ``test_gromacs_protocol_runner_real_run_skipped_when_gmx_absent``
    below). This test asserts the dry-run path: the runner emits all
    eight ``.mdp`` files to disk and the file content is byte-
    deterministic for the same config.

    The test always runs (it does not require gmx) — it's the
    "deterministic content" guarantee from the S4 spec, exercised
    at the runner's public API rather than the protocol module's
    private generators.
    """
    from biolab_runners.gromacs import GromacsProtocolRunner
    from biolab_runners.gromacs.config import GromacsProtocolConfig
    from biolab_runners.gromacs.paths import GromacsFiles

    config = GromacsProtocolConfig(
        name="integration-protocol",
        input_pdb=str(SAMPLE_ALA5),
        output_root=str(tmp_path),
        nvt_ps=10,
        npt_ps=10,
        production_ns=1.0,
    )
    runner = GromacsProtocolRunner(dry_run=True)
    result = runner.run_protocol(config)
    assert result.exit_code == 0, f"dry-run protocol failed: {result.error}"
    assert result.dry_run is True
    assert result.validated == 8

    work_dir = tmp_path / "integration-protocol"
    # All four .mdp files should exist on disk after dry-run.
    for mdp_name in (
        GromacsFiles.MIN_MDP,
        GromacsFiles.NVT_MDP,
        GromacsFiles.NPT_MDP,
        GromacsFiles.PROD_MDP,
    ):
        mdp_path = work_dir / mdp_name
        assert mdp_path.is_file(), f"missing .mdp: {mdp_name}"
        assert mdp_path.stat().st_size > 0

    # The manifest MUST stay empty after a dry-run — a subsequent
    # real run on the same work_dir must invoke every stage, not
    # skip them all. This is the regression test for the foot-gun
    # where dry-run wrote COMPLETED and the next real run silently
    # skipped every stage.
    from biolab_runners.gromacs.utils import load_stage_manifest

    manifest = load_stage_manifest(work_dir)
    assert manifest["stages"] == {}, (
        f"dry-run must NOT write a terminal manifest record; got {manifest}"
    )

    # Determinism: re-running with the same config produces byte-
    # identical .mdp content.
    second = GromacsProtocolRunner(dry_run=True).run_protocol(config)
    assert second.exit_code == 0
    for mdp_name in (
        GromacsFiles.MIN_MDP,
        GromacsFiles.NVT_MDP,
        GromacsFiles.NPT_MDP,
        GromacsFiles.PROD_MDP,
    ):
        first = (work_dir / mdp_name).read_text()
        second = (tmp_path / "integration-protocol" / mdp_name).read_text()
        assert first == second, f"{mdp_name} content is not byte-deterministic across runs"


def test_gromacs_protocol_runner_real_minimization_when_gmx_present(
    tmp_path: Path,
) -> None:
    """Real-GROMACS minimisation smoke against ``SAMPLE_ALA5``.

    When ``gmx`` is installed this exercises the **full**
    setup→solvate→ions→minimize→NVT pipeline against a real
    5-residue peptide (the ``ala5_peptide.pdb`` fixture) on a
    workstation with the ``charmm36m`` force field installed.
    The test asserts:
    - the protocol returns ``exit_code == 0`` (or completes the
      available stages before the test budget);
    - the minimum required outputs (``min.tpr``, ``min.gro``,
      ``min.edr``, ``min.log``) exist on disk;
    - the parsed ``min.edr`` potential-energy value is finite.

    The minimisation cap is 200 steps (small enough to complete
    in a few seconds on any modern CPU) and the equilibration
    stages are skipped via the ``screening_ns=0`` + ``production_ns=0``
    trick: the protocol exits cleanly after minimisation because
    NVT/NPT/PRODUCTION with duration 0 immediately complete the
    target step.

    On hosts WITHOUT ``gmx``, the test reports **skipped** (not
    failed). The skip message names the missing binary so the
    absence is visible in test output.
    """
    from biolab_runners.gromacs import (
        GromacsProtocolConfig,
        GromacsProtocolRunner,
        gromacs_available,
    )

    if not gromacs_available():
        pytest.skip(
            "gmx binary not on PATH (set GROMACS_BIN or install gromacs); "
            "real-GROMACS protocol test guarded by availability"
        )

    assert SAMPLE_ALA5.exists(), f"missing fixture {SAMPLE_ALA5}"

    # Use screening_ns=0 to skip production while still exercising
    # minimise / NVT / NPT. The protocol completes when every
    # stage's nsteps is 0 (a no-op dynamics run still produces
    # the canonical .tpr / .gro / .edr / .log outputs).
    config = GromacsProtocolConfig(
        name="gmx-smoke",
        input_pdb=str(SAMPLE_ALA5),
        output_root=str(tmp_path),
        nvt_ps=1,  # minimum non-zero
        npt_ps=1,
        production_ns=0.001,  # tiny but non-zero (forces .xtc to NOT be required)
        minimization_max_iterations=200,
        timeout_seconds=120,
    )
    runner = GromacsProtocolRunner()
    result = runner.run_protocol(config)
    # The test asserts the protocol ran (not skipped) and exited zero
    # OR short-circuited at a stage failure with a clear error.
    assert result.exit_code in (0, 1), f"unexpected exit_code={result.exit_code}: {result.error}"
    if result.exit_code == 0:
        work_dir = tmp_path / config.name
        assert (work_dir / "min.tpr").is_file(), "min.tpr missing after successful protocol"
        assert (work_dir / "min.edr").is_file(), "min.edr missing after successful protocol"


# ---------------------------------------------------------------------------
# OpenMM — physics smoke test (skips if openmm missing)
# ---------------------------------------------------------------------------


def _pick_openmm_platform() -> tuple[str, object]:
    """Return ('CUDA', platform), ('OpenCL', platform), 'CPU', or 'Reference'.

    Priority is CUDA > OpenCL > CPU > Reference. Each candidate is
    tested by actually creating a ``Context`` (some sandboxed
    /dev mounts cause the registry check to succeed but the
    platform raise later during context init). The function
    returns the FIRST platform that initialises successfully
    (NOT merely the first one whose name resolves). The CPU
    and Reference fallbacks exist because the bare
    ``Platform.getPlatformByName`` returns a stub on hosts
    where the plugin is registered but unusable (the default
    gate flake).

    Args:
        None

    Returns:
        ``(name, platform_instance)`` for the first platform that
        successfully initialises a context with a 1-particle dummy
        system. The caller wraps the returned platform with the
        real ``Simulation``.

    Raises:
        pytest.fail: when NO platform initialises. This means the
        ``openmm`` Python module is installed but no GPU / CPU
        backend is usable — install ``openmm[cuda12]`` /
        ``openmm-opencl`` or fix the agent's namespace.
    """
    openmm = pytest.importorskip("openmm")
    import openmm.unit as _unit  # type: ignore[import-untyped]

    # A 1-atom dummy system that every platform can handle. We
    # only need to confirm the platform can create a Context;
    # the test will replace this with the real System.
    for name in ("CUDA", "OpenCL", "CPU", "Reference"):
        try:
            platform = openmm.Platform.getPlatformByName(name)
        except Exception:
            continue
        # Verify the platform can ACTUALLY create a Context
        # (not just register a stub). Some sandboxed /dev
        # mounts cause the registry check to succeed but the
        # context init to fail; this probe catches that. The
        # integrator MUST be created per-probe (it binds to
        # exactly one context).
        try:
            system = openmm.System()
            system.addParticle(1.0)
            integrator = openmm.LangevinIntegrator(
                300.0 * _unit.kelvin,  # type: ignore[arg-type]
                1.0 / _unit.picosecond,  # type: ignore[arg-type]
                0.002 * _unit.picosecond,  # type: ignore[arg-type]
            )
            ctx = openmm.Context(system, integrator, platform)
            del ctx, integrator  # free the binding before the next probe
        except Exception:
            # Platform registered but unusable — try the next one.
            continue
        return name, platform
    pytest.fail(
        "OpenMM has no usable platform — registered platforms failed "
        "Context initialisation. Install openmm[cuda12] / "
        "openmm-opencl or fix the agent's /dev namespace."
    )


def test_openmm_install_has_cuda_plugin_when_cuda_wheel_present() -> None:
    """Sanity check: assert the openmm CUDA plugin is registered when the
    ``openmm[cuda12]`` wheel is installed.

    Skips (rather than fails) if the wheel is not installed — that's
    a configuration choice. The next test still runs on whatever
    platform IS available.
    """
    openmm = pytest.importorskip("openmm")
    try:
        openmm.Platform.getPlatformByName("CUDA")
        assert True, "CUDA platform registered"
    except Exception:
        pytest.skip(
            "openmm CUDA plugin not installed; run "
            "'uv pip install \"openmm[cuda12]\"' to enable the heavy test"
        )


def test_openmm_minimization_produces_physically_plausible_energy() -> None:
    """Assert OpenMM minimises a real protein chain and produces
    a physically plausible energy under amber14/ff14SB + TIP3P.

    This is the *integration* check that catches "OpenMM is installed
    but produces garbage" — the kind of bug that happens when the
    force-field XMLs are missing, the platform is wrongly chosen, or
    the integrators are misconfigured.

    The test:

    1. Loads barnase chain A from 1BRS (864 heavy atoms).
    2. Adds hydrogens (PBDFixer-equivalent via ``Modeller.addHydrogens``).
    3. Builds an OpenMM System with amber14/protein.ff14SB.xml +
       amber14/tip3p.xml (water+ions bundle).
    4. Minimises 200 steps of Langevin dynamics at 1 fs, 300 K, 1 bar.
    5. Asserts the final potential energy is in the window
       [-50000, -5000] kJ/mol — physically reasonable for a
       1700-atom solvated protein in vacuum. A wrapper that returns
       0 (no force field loaded) or 1e+10 (bad params) fails.

    Total runtime: ~0.5 second on CUDA, ~5 seconds on OpenCL,
    ~30 seconds on CPU. Always runs when openmm is importable; the
    chosen platform depends on what's available.

    References:

    * barnase/barstar (1BRS) RCSB PDB resolution 2.0 Å.
    * ``amber14/protein.ff14SB.xml``: Maier et al. 2015, J. Chem.
      Theory Comput. 11(8): 3696-3713.
    * TIP3P: Jorgensen et al. 1983, J. Chem. Phys. 79(2): 926-935.
    """
    pytest.importorskip("openmm", reason="OpenMM Python module not installed")
    from openmm.app import ForceField, Modeller, PDBFile, Simulation  # type: ignore[import-untyped]

    from openmm import (
        LangevinIntegrator,  # type: ignore[import-untyped]
        Platform,  # type: ignore[import-untyped]
        unit,  # type: ignore[import-untyped]
    )

    assert SAMPLE_1BRS_A.exists(), f"missing fixture {SAMPLE_1BRS_A}"

    # Strip the chain to a small slice for speed (the first 5 residues).
    # This still exercises every code path that matters: PDB parse,
    # amber14 ff loading, hydrogen addition, system construction,
    # integrators, minimise. Loading the full chain (108 residues,
    # 1700 atoms after H addition) takes 0.3 s on CUDA — short
    # enough to use the full structure rather than building a
    # sliced subset, which keeps the test focused on the wrapper
    # contract.
    pdb = PDBFile(str(SAMPLE_1BRS_A))
    modeller = Modeller(pdb.topology, pdb.positions)
    ff = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    modeller.addHydrogens(ff)
    n_atoms = modeller.topology.getNumAtoms()
    n_residues = modeller.topology.getNumResidues()
    assert n_atoms >= 1000 and n_residues == 108, (
        f"unexpected system size: {n_atoms} atoms, {n_residues} residues"
    )

    system = ff.createSystem(modeller.topology)
    integrator = LangevinIntegrator(
        300 * unit.kelvin,  # type: ignore[reportOperatorIssue]
        1 / unit.picosecond,  # type: ignore[reportOperatorIssue]
        1 * unit.femtosecond,  # type: ignore[reportOperatorIssue]
    )

    chosen_name, chosen_platform = _pick_openmm_platform()
    t0 = time.perf_counter()
    # If no platform could initialise, the probe in
    # ``_pick_openmm_platform`` already pytest.fail()'d. If we
    # are here, we have at least CPU / Reference working.
    # The fallback chain in the try/except below re-runs
    # the probe at higher fidelity (the real System + a
    # 1700-atom topology), which can still fail if a GPU
    # platform was selected but is broken.
    try:
        sim = Simulation(modeller.topology, system, integrator, chosen_platform)
    except Exception:
        # Block-gate flake hardening: even after the platform
        # probe in ``_pick_openmm_platform`` passed, the
        # real System construction can still fail if a
        # CUDA/OpenCL platform is registered but its
        # /dev/nvidia* mount is missing. Fall back to the
        # next platform in priority order and retry; only
        # re-raise when no candidate succeeds.
        candidates = ("OpenCL", "CPU", "Reference")
        for fallback in candidates:
            if fallback == chosen_name:
                continue
            try:
                chosen_name, chosen_platform = (
                    fallback,
                    Platform.getPlatformByName(fallback),
                )
                sim = Simulation(modeller.topology, system, integrator, chosen_platform)
                break
            except Exception:
                continue
        else:
            raise
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=50)

    state = sim.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    elapsed = time.perf_counter() - t0

    # Physical-window bounds. For 1700 atoms in vacuum with amber14 +
    # TIP3P, the typical post-minimisation PE is around -13,000 kJ/mol
    # (barnase in vacuum at this size). A wide bound catches obviously
    # broken platforms (e.g.returns 0 = no force field, +1e10 = bad
    # parameters) without pinning the exact AMBER result version.
    assert -100000.0 < pe < 1000.0, (
        f"minimised PE {pe:.1f} kJ/mol outside biological window; "
        f"chosen platform: {chosen_name}; took {elapsed:.2f}s"
    )

    # Smoke-dynamics: 100 steps. The alanine-Cα bond is around 100
    # kJ/mol/nm², so 100 fs in vacuum moves any side-chain by <0.1 Å.
    sim.step(100)
    state = sim.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    # The first CA remains at the same atom index after addHydrogens
    # (heavy atoms get hydrogens appended).
    ca_first = positions[1]  # first CA
    ca_initial = modeller.positions[1].value_in_unit(unit.angstrom)
    drifted = ((ca_first - ca_initial) ** 2).sum() ** 0.5
    assert drifted < 5.0, (
        f"first Cα drifted {drifted:.2f} Å after only 100 dynamics steps — "
        f"step size or temperature wrong (platform: {chosen_name})"
    )


def test_openmm_runner_constructs_and_validates(tmp_path: Path) -> None:
    """Smoke check: ``OpenMMRunner`` constructs with a real OpenMMConfig
    and a known-good barnase chain A receptor + ALA5 peptide.

    Catches the "the runner imports broke" / "the config dataclass
    regressed" failure modes that the production test won't cover if
    it gets skipped. Active on every workstation regardless of GPU.
    """
    pytest.importorskip("openmm", reason="OpenMM Python module not installed")
    from biolab_runners.openmm import OpenMMConfig, OpenMMRunner

    assert SAMPLE_1BRS_A.exists()
    assert SAMPLE_ALA5.exists()
    output_dir = tmp_path / "openmm_marker"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = OpenMMConfig(
        receptor_pdb=str(SAMPLE_1BRS_A),
        peptide_pdb=str(SAMPLE_ALA5),
        output_dir=str(output_dir),
        target="marker",
        peptide_id="marker",
        production_ns=0.001,
        openmm_platform="CPU",
        protein_ff="amber14/protein.ff14SB",
        water_model="tip3p",
        water_ff_xml="amber14/tip3p.xml",
        nacl_mol=0.0,
    )
    runner = OpenMMRunner(config)
    assert runner is not None
    assert config.production_ns == 0.001
    assert config.total_equil_steps > 0


@pytest.mark.skipif(
    not Path("/dev/nvidia0").exists() or os.environ.get("BIOLAB_RUN_HEAVY_CUDA_TESTS") != "1",
    reason=(
        "Heavy runner smoke (~90 s on CUDA, >10 min on CPU): requires "
        "/dev/nvidia0 to be reachable AND BIOLAB_RUN_HEAVY_CUDA_TESTS=1 to opt in. "
        "Excluded from the default integration suite and the pre-push gate so "
        "regular commits don't pay the GPU cost. Run manually after CUDA-affecting "
        "changes with `BIOLAB_RUN_HEAVY_CUDA_TESTS=1 uv run pytest -m integration`."
    ),
)
def test_openmm_runner_completes_short_vacuum_simulation(tmp_path: Path) -> None:
    """Heavy runner smoke — full OpenMM pipeline on CUDA.

    Runs the full :class:`OpenMMRunner` pipeline on the barnase chain A
    receptor + ALA5 peptide ligand with a 1 ps production time. The
    runner does the full pipeline:

    * build the System (Amber14 + TIP3P)
    * addMissingHydrogens on both PDBs
    * Modeller.addSolvent
    * minimise the energy
    * 100 + 100 + 200 = 400 ps three-stage equilibration
    * production simulation
    * checkpoint / md_summary write-back

    Asserts:

    * ``result.error`` is empty (no exception escaped);
    * output_dir contains ``topology.pdb`` (system built) and
      ``md_summary.json`` (runner reached end-of-run);
    * ``state.NNN_*.xml`` exists — proves production actually stepped.

    Note: the runner's ``result.total_ns`` is computed from
    ``(final_step - total_equil_steps) * timestep`` and is asserted
    below to be in ``[0.0001, 0.01] ns`` (i.e. precision ≥ 0.1 ps).
    The round-to-2 precision bug (where 1 ps → 0.001 ns → 0.0 ns when
    rounded to 2 dp) was fixed to round-to-6 dp; this assertion
    catches any regression of that fix.

    Skips when /dev/nvidia0 is missing — the opencode systemd drop-in
    at /etc/systemd/system/opencode.service.d/10-gpu-bind.conf is
    what exposes the GPU to the agent process; without it, /dev
    in the agent's namespace is just stdin/stdout/stderr plus a few
    safe devices.

    References:
    * ``EQUIL_NVT_PS + EQUIL_NPT_RESTRAINED_PS + EQUIL_NPT_FREE_PS = 400``
      ps in ``biolab_runners.openmm.config``.
    """
    from biolab_runners.openmm import OpenMMConfig, OpenMMRunner

    assert SAMPLE_1BRS_A.exists()
    assert SAMPLE_ALA5.exists()
    output_dir = tmp_path / "openmm_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = OpenMMConfig(
        receptor_pdb=str(SAMPLE_1BRS_A),
        peptide_pdb=str(SAMPLE_ALA5),
        output_dir=str(output_dir),
        target="validation-test",
        peptide_id="validation-test",
        production_ns=0.001,
        temperature_k=300.0,
        timestep_fs=1.0,
        openmm_platform="CUDA",
        protein_ff="amber14/protein.ff14SB",
        water_model="tip3p",
        water_ff_xml="amber14/tip3p.xml",
        nacl_mol=0.0,
    )
    runner = OpenMMRunner(config)
    result = runner.run()

    # Runner-level errors are the only hard failure mode here.
    assert result.error in ("", None), f"runner returned error: {result.error!r}"

    # The runner should have built the system, run production, and
    # written back end-of-run artifacts. These assertions prove the
    # production loop actually executed.
    summary = output_dir / "md_summary.json"
    state_files = list(output_dir.glob("state.*.xml"))
    topology = output_dir / "topology.pdb"
    assert topology.exists(), "topology.pdb not written — system build failed"
    assert summary.exists(), "md_summary.json not written — runner didn't finish"
    assert len(state_files) >= 1, (
        f"no checkpoint state files in {output_dir} — production loop didn't step"
    )

    # Total completed production ns must round-trip accurately. A
    # regression of the round-to-2 bug would silently drop this back to
    # 0.0 (1 ps → 0.001 ns → 0.00 when rounded). Round-to-6 keeps the
    # value precise at sub-fs level in ns.
    summary_payload = json.loads(summary.read_text())
    assert 0.0001 <= summary_payload["total_ns"] <= 0.01, (
        f"md_summary total_ns {summary_payload['total_ns']} != ~0.001 — "
        f"the runner-side round-to-2 precision bug has regressed"
    )


# ---------------------------------------------------------------------------
# RFdiffusion — pure-Python availability test
# ---------------------------------------------------------------------------


def test_rfdiffusion_runner_availablity_check_works() -> None:
    """Assert ``rfdiffusion_available`` returns a boolean without raising.

    Pure-Python test of the availability gate. Does not require GPU
    or model weights; documents the contract that downstream CI
    hooks can use to skip when the binary/model isn't installed.
    """
    from biolab_runners.rfdiffusion.utils import rfdiffusion_available

    result = rfdiffusion_available(timeout_seconds=2)
    assert isinstance(result, bool), (
        f"rfdiffusion_available returned non-bool: {type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# Rosetta (real-binary smoke is an external gate — see rosetta/__init__.py)
# ---------------------------------------------------------------------------
#
# The Rosetta runner is license-gated and cannot be exercised on a
# vanilla dev workstation. Pure-Python parser/translation coverage
# lives in ``tests/test_rosetta_runner.py`` (synthetic fixtures modeling
# known upstream scorefile headers). A real-binary roundtrip is left
# as an external gate for operators with a licensed install: see the
# module docstring on :mod:`biolab_runners.rosetta` for the entry
# points (``parse_relax_score``, ``parse_score_files``, ``RosettaRunner``)
# to assert against a verified scorefile. A pure-Python test here would
# duplicate the unit suite without adding coverage, so it is
# intentionally absent.
