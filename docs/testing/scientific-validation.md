# Scientific Validation Plan — biolab-runners

This document defines the **tool-level smoke validation** available in this
repository. It covers real-format parsers, lightweight GROMACS/OpenMM paths,
and environment probes. It does not claim that every runner has a live
external-tool smoke test: Rosetta is license-gated, peptide preparation has
optional scientific dependencies, and gmx_MMPBSA is optional.

## Why this exists

The biolab-runners unit test suite proves the covered runner plumbing:

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
exercises the following available checks:

| Test | Runner | Reference input | What we assert |
|---|---|---|---|
| `test_openmm_install_has_cuda_plugin_when_cuda_wheel_present` | OpenMM | environment probe | `openmm[cuda12]` or `openmm[cuda13]` exposes a CUDA platform when the matching plugin is installed |
| `test_openmm_runner_constructs_and_validates` | OpenMM | in-memory `ala5_peptide.pdb` | Constructor accepts a 5-residue test PDB and rejects malformed PDBs |
| `test_openmm_minimization_produces_physically_plausible_energy` | OpenMM | `ala5_peptide.pdb` (CPU) | Minimization converges; total energy is finite and bounded |
| `test_openmm_runner_completes_short_vacuum_simulation` (heavy) | OpenMM MD | `barnase_chainA.pdb` (chain A only) | 1 ps vacuum simulation completes; `total_ns` is in `[0.0001, 0.01]` ns and energy is finite |
| `test_gromacs_parse_nthcol_returns_first_data_row` | GROMACS | fixture `.xvg` | `parse_nthcol_energy` extracts the first requested data value |
| `test_gromacs_parse_nthcol_handles_comment_lines` | GROMACS | fixture `.xvg` | Parser skips `@` / `#` comment lines |
| `test_gromacs_protocol_runner_dry_run_emits_deterministic_mdps` | GROMACS protocol | `ala5_peptide.pdb` | Dry-run emits deterministic stage `.mdp` files without writing a terminal manifest |
| `test_gromacs_protocol_runner_real_minimization_when_gmx_present` | GROMACS protocol | `ala5_peptide.pdb` | When `gmx` is available, minimisation produces the required `.tpr`, `.gro`, `.edr`, and `.log` outputs |
| `test_proteinmpnn_parse_fasta_returns_both_records_with_protein_alpha` | ProteinMPNN | fixture `.fa` | `parse_fasta_sequences` returns both design records with canonical amino-acid sequences |
| `test_rfdiffusion_runner_availablity_check_works` | RFdiffusion | environment probe | `rfdiffusion_available()` checks the package adapter or `${RFDIFFUSION_BIN}` and its `--help` result |
| `test_rfdiffusion_wrapper_accepts_design_startnum_and_two_seeds_differ` | RFdiffusion adapter | fake upstream script | Adapter accepts the runner contract and forwards deterministic design-start values |

Tools that **fail** binary-availability checks (Rosetta requires a commercial
license; GPU-dependent weights or optional executables may be missing) **skip
gracefully** with a pytest skip message. Peptide preparation and gmx_MMPBSA
are covered by focused tests rather than this integration module:

| Runner | Focused coverage | Contract exercised |
|---|---|---|
| Peptide preparation | `tests/test_peptide_prep.py` | Required `prepared.pdb`, `prepared.top`, and `prepared.gro` artifacts; optional dependency failures are structured and fail closed; in-process execution mode |
| gmx_MMPBSA | `tests/test_mmpbsa.py` | Parser, availability probe, `unsupported` result for a missing optional binary, and shared metadata on the legacy dict result |
| Rosetta | `tests/test_rosetta_runner.py` | Parser/CLI translation and license acknowledgement without invoking the licensed binary |

Tools that **fail** validation tests *because the wrapper produces wrong
numbers or output* get fixed; a missing optional tool is not a scientific
success.

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

The ProteinMPNN runner uses the package-provided `proteinmpnn` console adapter
(or `${PROTEINMPNN_BIN}`). The adapter translates the stable runner argv
contract to `protein_mpnn_run.py` and can be tested with a fake script; it does
not use a shell:

| Runner flag | Upstream translation |
|---|---|
| `--input_path DIR` + `--pdb_path NAME` | `--pdb_path DIR/NAME` |
| `--output_path DIR` | `--out_folder DIR` |
| `--batch_size`, `--model_name`, `--num_seq_per_target`, `--sampling_temp`, `--seed` | Forwarded with the same names and values |
| `--ca_only` | Emits upstream `--ca_only` for true-like values |
| `--omit_AA VALUE` | `--omit_AAs VALUE` |
| non-empty `--fixed_positions` | Rejected: the adapter has no chain-aware JSONL contract |

Unknown flags are appended unchanged for upstream forward compatibility. Set
`PROTEINMPNN_HOME` (default `~/tools/ProteinMPNN`), `PROTEINMPNN_SCRIPT`, or
`PROTEINMPNN_PYTHON` to select the checkout, script, or interpreter. The
adapter's `--help` needs no upstream installation. `${PROTEINMPNN_BIN}` may
be a local executable; ProteinMPNN rejects `container://` values before
subprocess dispatch. The RFdiffusion runner ships its adapter **in
the package**: the installed wheel provides a `rfdiffusion` console
script (`biolab_runners.rfdiffusion.cli`) that accepts the runner's
fixed flag contract:

| Flag | Meaning |
|---|---|
| `--output_dir DIR` | Where to write the design PDBs (owns `inference.output_prefix=DIR/design`) |
| `--<dotted.hydra.key> <value>` | Hydra overrides with underscores hyphenated, e.g. `--inference.num-designs 10`, `--contigmap.contigs 'A1-110/0 B1-110/0 14-18'`, `--inference.input-pdb target.pdb`, `--ppi.hotspot-res A51,A52` |

The console script validates the pairs, translates list-typed keys to
Hydra list syntax (`contigmap.contigs=[...]`, `ppi.hotspot_res=['A51','A52']`),
quotes string scalars only when needed (numeric/bool types preserved),
prepends the clone root to `PYTHONPATH` (existing value preserved), and
executes the stock `scripts/run_inference.py` with positional
`key=value` overrides (no shell; Hydra's own metadata is confined under
the output directory). Runtime requirements:

* `RFDIFFUSION_HOME` — path to the upstream clone root
  (`~/tools/RFdiffusion` by default), containing
  `scripts/run_inference.py` and the model weights.
* A Python with **PyTorch + CUDA** in the interpreter that runs the
  console script (upstream's `run_inference.py` imports torch and uses
  a GPU when available).
* `--help` needs none of these and stays cheap for the availability
  probe.
* The legacy `container://` `RFDIFFUSION_BIN` form is **removed**
  (it passed `--key value` flags straight to Hydra and hardcoded an
  image-internal path); inside a container, install the wheel and set
  `RFDIFFUSION_HOME` to the mounted clone.

A host-level bootstrap script
(`~/.local/bin/install-proteinmpnn-rfdiffusion.sh`) clones the upstream
repos shallowly into `~/tools/ProteinMPNN` and `~/tools/RFdiffusion`
(the default `RFDIFFUSION_HOME`). After the clone + weights are
present, `biolab_runners.rfdiffusion.utils.rfdiffusion_available()`
returns True. The real-wrapper integration test still requires
`BIOLAB_RUN_RFDIFFUSION_INTEGRATION=1`; otherwise it is skipped.

Real backbone design with RFdiffusion still requires a GPU and
~10 GB of model weights (downloaded on first run from the upstream
RFdiffusion repo); the OpenMM heavy runner test requires
`/dev/nvidia0` *and* `BIOLAB_RUN_HEAVY_CUDA_TESTS=1` (see the
`@pytest.mark.skipif` on `test_openmm_runner_completes_short_vacuum_simulation`).

### Execution modes and contract checks

Boltz2, RFdiffusion, ProteinMPNN, Rosetta, GROMACS, and gmx_MMPBSA use
subprocess execution; OpenMM and peptide preparation are in-process.
ProteinMPNN and GROMACS reject `container://` values before subprocess
dispatch; Rosetta resolves its optional value through its tool utility.
RFdiffusion rejects that legacy form and uses its package adapter. gmx_MMPBSA
records `container_uri` for rejected URI configurations, but its current
command builder does not launch the container.
No runner shares a universal container-launch behavior. The shared contract tests verify
normalized statuses, typed errors, artifact digests, provenance serialization,
the ProteinMPNN adapter, and legacy result fields.

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
