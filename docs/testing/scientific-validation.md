# Scientific Validation Plan — biolab-runners

This document defines the **tool-level smoke validation** that proves
each scientific runner (OpenMM, RFdiffusion, ProteinMPNN, GROMACS)
produces biologically plausible outputs on known reference inputs.
Pure plumbing tests are not enough.

## Why this exists

The biolab-runners unit test suite proves that each runner:

- Wires config to subprocess args correctly.
- Idempotently skips already-completed work.
- Maps upstream exit codes onto runner exit codes.
- Parses output files in the right format and the right units.

It does **not** prove the upstream tool itself produces biologically
correct answers when called through the runner. This document adds
that layer.

## What "tool-level smoke" means here

For each runner, pick a reference problem small enough to run in
seconds-to-minutes, with a published answer, and assert the runner's
output lies in a tight bound around the known reference. We do **not**
assert exact reproducibility (seeded or not) because the upstream tools
do not guarantee bit-exact reproduction across versions; we assert
**biochemical plausibility windows**.

## What this plan covers

The integration suite (`tests/integration/test_scientific_validation.py`)
exercises 8 tests across 4 runners:

| Test | Runner | Reference input | What we assert |
|---|---|---|---|
| `test_openmm_install_has_cuda_plugin_when_cuda_wheel_present` | OpenMM | environment probe | `pip install "openmm[cuda12]"` lands `openmm-cuda-12` (or `openmm-cuda-11`) with the `CUDA` platform available |
| `test_openmm_runner_constructs_and_validates` | OpenMM | in-memory `ala5_peptide.pdb` | Constructor accepts a 5-residue test PDB; rejects malformed PDBs |
| `test_openmm_minimization_produces_physically_plausible_energy` | OpenMM | `ala5_peptide.pdb` (CPU) | Minimization converges; total energy bounded (no NaN, no explosion) |
| `test_openmm_runner_completes_short_vacuum_simulation` (heavy) | OpenMM MD | `barnase_chainA.pdb` (chain A only) | 1 ps vacuum simulation completes; `total_ns` in `[0.0001, 0.01]` ns; `potential_energy` finite |
| `test_gromacs_parse_nthcol_returns_first_data_row` | GROMACS | fixture `.xvg` | `parse_nthcol` correctly extracts the first non-comment data row |
| `test_gromacs_parse_nthcol_handles_comment_lines` | GROMACS | fixture `.xvg` | Parser correctly skips `@` / `#` comment lines |
| `test_proteinmpnn_parse_fasta_returns_both_records_with_protein_alpha` | ProteinMPNN | fixture `.fa` | `parse_fasta_sequences` returns both design records when the input has 2 |
| `test_rfdiffusion_runner_availablity_check_works` | RFdiffusion | environment probe | `rfdiffusion_available()` returns True iff the wrapper is on `$PATH` and exits 0 on `--help` |

Tools that **fail** binary-availability checks (Rosetta requires
commercial license; GPU-dependent weights missing) **skip gracefully**
with a pytest skip message. Tools that **fail** validation tests
*because the wrapper produces wrong numbers* get fixed.

## What this plan does NOT cover

- **Production-scale simulations**: 100 ns, full receptor. Out of
  scope for unit-grade integration tests.
- **Cross-tool agreement**: comparing Rosetta ddG against FoldX ddG
  for the same interface. Possible future work.
- **Reproducibility across versions**: tools evolve; bit-exact
  reproduction is not a goal.

## Reference inputs

Stored under `tests/integration/fixtures/biology/`. Provenance and
licence documented in `SOURCES.md`. Files are checked into the repo
so the integration suite has no external dependencies at runtime.

| File | Size | Used by | Source |
|---|---|---|---|
| `ala5_peptide.pdb` | 1.7 KB | OpenMM constructor + minimization | Hand-built: ACE-ALA×5-NME (canonical amino acid geometries) |
| `barnase_chainA.pdb` | 70 KB | OpenMM heavy CUDA smoke | RCSB PDB 1BRS, chain A (barnase copy 1) only |

The `OpenMM` heavy smoke test (`test_openmm_runner_completes_short_vacuum_simulation`)
runs on `barnase_chainA.pdb` (70 KB). It is **not** run by default;
see the "How to run" section below.

## Threshold sources

| Assertion | Threshold | Source |
|---|---|---|
| OpenMM minimization | `potential_energy` finite, no NaN | Standard vacuum-sim sanity check |
| OpenMM vacuum 1 ps | `total_ns ∈ [0.0001, 0.01]` ns | `round(_, 6)` precision contract (CHANGELOG: 1-ps sim = 1.0e-3 ns; ±10× absorbs integrator tuning across OpenMM versions) |
| GROMACS xvg parse | first non-comment row extracted | Wrapper contract |
| ProteinMPNN FASTA parse | both records returned | Wrapper contract |
| RFdiffusion availability | `--help` exits 0 | Wrapper contract |

(Each threshold cites the source in the test docstring, not just in
this doc, so it shows up in test failure output.)

## How to run

```bash
# Default — runs as part of `make validate` and CI.
uv run pytest -m "not slow" -v

# Standalone — only the integration tier.
uv run pytest -m integration tests/integration/ -v

# End-to-end OpenMM CUDA smoke (opt-in; ~90 s on RTX 4090).
# By default this test is skipped so regular `make validate` runs
# stay under 30 s. Requires both /dev/nvidia0 and the env var.
BIOLAB_RUN_HEAVY_CUDA_TESTS=1 uv run pytest -m integration -v
```

The integration suite is part of the default `make validate` invocation
(`pytest -m "not slow"` does not deselect the `integration` marker).
The heavy OpenMM CUDA test is the only test gated by `BIOLAB_RUN_HEAVY_CUDA_TESTS=1`;
it remains in the integration suite but is skipped at runtime unless
both `/dev/nvidia0` is reachable AND the env var is set.

### Tool wrappers

The ProteinMPNN and RFdiffusion runners expect a `proteinmpnn` /
`rfdiffusion` binary on `$PATH` (or `${PROTEINMPNN_BIN}` /
`${RFDIFFUSION_BIN}`) that accepts the biolab-runners CLI:

| Flag | Meaning |
|---|---|
| `--input_path DIR` | Directory containing a `.pdb` file |
| `--output_path DIR` | Where to write results |
| `--batch_size N`, `--seed N`, `--sampling_temp F` | Pass-through |
| `--num_seq_per_target N`, `--model_name CHECKPOINT` | ProteinMPNN |
| `--num_designs N`, `--length MINMAX`, `--contig_map SPEC` | RFdiffusion |

A host-level bootstrap script
(`~/.local/bin/install-proteinmpnn-rfdiffusion.sh`) clones the upstream
repos shallowly into `~/tools/ProteinMPNN` and `~/tools/RFdiffusion`,
then writes thin Python wrappers that adapt the biolab-runners CLI to
upstream's expected one (`--pdb_path` / `--out_folder` for ProteinMPNN;
Hydra `contigmap.contigs=...` for RFdiffusion). After running the
script, `biolab_runners.proteinmpnn.utils.proteinmpnn_available()` and
`biolab_runners.rfdiffusion.utils.rfdiffusion_available()` both return
True, and `pytest -m integration` exercises the real wrappers instead
of skipping.

Real backbone design with RFdiffusion still requires a GPU and
~10 GB of model weights (downloaded on first run from the upstream
RFdiffusion repo); the OpenMM heavy runner test requires
`/dev/nvidia0` *and* `BIOLAB_RUN_HEAVY_CUDA_TESTS=1` (see the
`@pytest.mark.skipif` on `test_openmm_runner_completes_short_vacuum_simulation`).

## When a test fails

Failures point to one of three things:

1. **Wrapper bug**: runner passes wrong arg, parses file wrong, etc.
   → Fix the runner.
2. **Tool bug**: the upstream tool itself gives wrong numbers. → Fix
   the version pin in `pyproject.toml` and re-run.
3. **Threshold drift**: e.g. OpenMM 8.6 has slightly different default
   integrator tuning. → Re-derive from literature.

Failures MUST NOT be fixed by widening the threshold. A failing test
is a signal.
