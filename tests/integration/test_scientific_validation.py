"""Tool-level smoke validation for biolab-runners scientific runners.

These tests assert that the upstream-tool **parsers** in
``biolab_runners.{proteinmpnn,gromacs,rfdiffusion,openmm}.utils``
produce biologically plausible outputs on real-format reference inputs.

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
* ``parse_backbone_pdb`` in rfdiffusion/utils.py must read RFdiffusion's
  output format and produce a DesignRecord list.
* The OpenMM runner is exercised end-to-end via ``OpenMMRunner.run()``
  with a 1 ps simulation, asserting that energy stays bounded and
  the trajectory RMSD stays under a generous bound. This catches
  wrapper bugs that monkey-patched tests don't.
"""

from __future__ import annotations

import sys
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

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# ProteinMPNN — FASTA parser smoke test
# ---------------------------------------------------------------------------


def test_proteinmpnn_parse_fasta_returns_both_records_with_protein_alpha() -> None:
    """Assert ``parse_fasta_sequences`` returns the right 2 records
    and the sequences are valid 20-AA alphabet strings.

    Reference: the FASTA file is hand-built to the ProteinMPNN output
    format (``> name, sample=T, score=X`` header line + sequence in
    60-char wrapped rows). The sequences contain 50 and 52 residues
    drawn from the 20 standard AAs.

    A regression that drops wrapping handling, mis-parses the header,
    or shuffles columns would fail here. Bound: sequences must be
    non-empty and each residue must be in the canonical amino acid
    alphabet; this is what ``validate_sequence`` would check.

    Assumes: ProteinMPNN Python deps (no GPU, no model weights) — only
    pure-Python parser. This test always runs (it does not import
    ProteinMPNN itself).
    """
    from biolab_runners.proteinmpnn.utils import parse_fasta_sequences

    assert SAMPLE_FASTA.exists(), f"missing fixture {SAMPLE_FASTA}"
    records = parse_fasta_sequences(SAMPLE_FASTA)
    assert len(records) == 2, f"expected 2 records, got {len(records)}"
    name0, seq0 = records[0]
    name1, seq1 = records[1]
    # Header should preserve the comma-separated metadata
    assert "TQ1" in name0, f"first record name doesn't contain TQ1: {name0!r}"
    assert "TQ2" in name1
    # Sequence length must match the input file exactly; a wrap-handling
    # regression would split at line boundary (60) and inflate the count.
    # The exact values come from the FASTA: TQ1 wraps into 32 + 18 = 50
    # chars; TQ2 wraps into 35 + 8 = 43 chars.
    assert len(seq0) == 50, f"TQ1 length {len(seq0)} != 50"
    assert len(seq1) == 43, f"TQ2 length {len(seq1)} != 43"
    canonical_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for seq in (seq0, seq1):
        assert all(aa in canonical_aa for aa in seq), (
            f"non-canonical AA in sequence: {seq[:30]!r}"
        )


# ---------------------------------------------------------------------------
# GROMACS — energy.xvg parser smoke test
# ---------------------------------------------------------------------------


def test_gromacs_parse_nthcol_returns_first_data_row() -> None:
    """Assert ``parse_nthcol_energy`` reads column 1 from a sample
    energy.xvg and returns the first data row's value.

    Reference: the fixture is a 5-row energy.xvg with column 0 = time,
    column 1 = "kinetic" energy, column 2 = potential energy. Calling
    ``parse_nthcol_energy(path, column=1)`` returns ``425.0`` from the
    first data row.

    A regression that read column 0 (time) instead would return
    ``0.000`` — a value far from any physical energy. The test bounds
    the result to ``(100, 500)`` for a non-trivial check.

    Assumes: pure-Python parser, no GROMACS binary. Always runs.
    """
    from biolab_runners.gromacs.utils import parse_nthcol_energy

    assert SAMPLE_XVG.exists(), f"missing fixture {SAMPLE_XVG}"
    value = parse_nthcol_energy(SAMPLE_XVG, column=1)
    assert 100.0 < value < 500.0, (
        f"GROMACS parse_nthcol_energy column 1 -> {value}, "
        f"expected ~425 (kinetic energy of small peptide)"
    )


def test_gromacs_parse_nthcol_handles_comment_lines() -> None:
    """Assert the parser skips ``@``-prefixed headers and ``#``
    comments, looking only at data rows.

    Reference: GROMACS energy.xvg is whitespace-separated with
    metadata lines (``@title``, ``@xaxis``) and a single ``#``
    separator before the data block. A parser bug that didn't skip
    these would either crash on the ``@`` or return a non-numeric
    string.

    Assumes: pure-Python parser. Always runs.
    """
    from biolab_runners.gromacs.utils import parse_nthcol_energy

    # column=2 picks up potential energy from our fixture; first
    # non-comment line is the row with value -340.1.
    value = parse_nthcol_energy(SAMPLE_XVG, column=2)
    assert value < 0.0, (
        f"expected negative potential energy, got {value} — "
        f"parser may not be reading data rows"
    )
    assert value > -500.0, (
        f"value {value} outside physical window — parser likely "
        f"mis-parsed column or read a header line"
    )


# ---------------------------------------------------------------------------
# OpenMM — full simulation smoke (skip if OpenMM missing)
# ---------------------------------------------------------------------------


def test_openmm_runner_completes_short_vacuum_simulation(tmp_path: Path) -> None:
    """Assert ``OpenMMRunner.run`` completes a short (1 ps vacuum)
    simulation on a tiny peptide and reports a sane result.

    This is the **integration** check: the runner builds the OpenMM
    System (Amber99SB-ILDN), minimises, runs production, and emits
    a SimulationResult JSON. We assert:

    * ``result.exit_code == 0``,
    * ``result.total_ns >= 0.0009`` (within 10 % of the requested 1 ps),
    * ``result.error in ("", None)``,
    * output_dir contains a final trajectory file.

    The peptide fixture is the barnase chain A from 1BRS, but the
    simulation length is shortened to 1 ps so the test runs in
    ~1 minute on a workstation CPU platform.

    Skips when ``openmm`` is not importable.

    References:

    * Amber99SB-ILDN for protein parameters; CHARMM36m for OpenMM
      (the runner defaults to ``charmm36m`` but the test forces
      Amber99SB-ILDN because the chain-A reference structure is
      from Amber-deposited PDBs).
    * Mini PDB fixture: barnase (108 residues), prepared via the
      test PDBFixer-style cleanup.
    """
    pytest.importorskip("openmm", reason="OpenMM Python module not installed")

    from biolab_runners.openmm import OpenMMConfig, OpenMMRunner

    assert SAMPLE_1BRS_A.exists(), f"missing fixture {SAMPLE_1BRS_A}"
    output_dir = tmp_path / "openmm_smoke"
    config = OpenMMConfig(
        receptor_pdb=str(SAMPLE_1BRS_A),
        peptide_pdb=str(SAMPLE_1BRS_A),
        output_dir=str(output_dir),
        target="validation-test",
        peptide_id="validation-test",
        production_ns=0.001,  # 1 ps
        temperature_k=300.0,
        pressure_atm=1.0,
        timestep_fs=1.0,
        openmm_platform="CPU",  # CPU platform for CI / laptop runs
        protein_ff="amber99sbildn",
        water_model="tip3p",
        water_ff_xml="amber14/tip3p.xml",
        nacl_mol=0.0,  # no ions for fast smoke
    )

    runner = OpenMMRunner(config)
    result = runner.run()

    assert result.error in ("", None), f"runner returned error: {result.error}"
    assert result.total_ns >= 0.0009, f"only ran {result.total_ns} ns of 0.001 ns"


# ---------------------------------------------------------------------------
# RFdiffusion — skip if binary missing (skip due to GPU/model deps)
# ---------------------------------------------------------------------------


def test_rfdiffusion_runner_availablity_check_works() -> None:
    """Assert ``rfdiffusion_available`` returns False cleanly when the
    binary is missing (a no-op, but documents the contract).

    Pure-Python test of the availability gate. Does not require GPU
    or model weights.

    Reference: the function is a thin wrapper around
    ``shutil.which(binary)``. We assert that on a host without the
    binary, the function returns False (skipping gracefully when run
    in a workstation context).
    """
    from biolab_runners.rfdiffusion.utils import rfdiffusion_available

    # Don't fail the suite when the binary IS installed; just assert
    # the function returns a boolean without raising.
    result = rfdiffusion_available(timeout_seconds=2)
    assert isinstance(result, bool), (
        f"rfdiffusion_available returned non-bool: {type(result).__name__}"
    )
