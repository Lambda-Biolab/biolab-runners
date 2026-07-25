# AGENTS.md — biolab-runners

This is the primary instruction file for all AI coding agents working on this project.
Read this file first. It supersedes any default behavior.

## Project Purpose

Standalone, modular Python library containing two computational biology runners extracted from the [OralBiome-AMP](https://github.com/Lambda-Biolab/OralBiome-AMP) pipeline:

1. **Boltz2Runner** — Runs Boltz-2 structure predictions for peptide-protein complexes
2. **OpenMMRunner** — Runs OpenMM molecular dynamics simulations with multi-stage equilibration

The runners are designed for researchers who want to use these tools in their own pipelines without importing the full OralBiome-AMP codebase.

## Architecture

```
biolab_runners/
├── boltz2/
│   ├── config.py     # Boltz2Config, ConfidenceScores, PredictionResult, QualityGate
│   ├── runner.py     # Boltz2Runner class + apply_quality_gate()
│   └── utils.py      # YAML writer, output parser, availability check
└── openmm/
    ├── config.py         # OpenMMConfig, SimulationResult
    ├── runner.py         # OpenMMRunner class (orchestrate pipeline, run production)
    ├── system_builder.py # ForceField, Modeller, System, Integrator, Cα restraint, prepare_simulation
    ├── geometry.py       # Pure-numpy PBC math (pbc_correct, min_pbc_distance)
    ├── offline_gate.py   # Offline mdtraj gate + verdict I/O
    ├── paths.py          # Centralized filenames (FileNames)
    └── utils.py          # Output verification, checkpoint loading, availability check
```

## Domain Rules

### Boltz2Runner

- **Input:** Receptor sequence + peptide sequence (strings), optional pocket constraints
- **Output:** `PredictionResult` with structure path, confidence scores, quality gate
- **Quality gate:** PASS / CONDITIONAL / FAIL based on ipTM, pLDDT, clash thresholds
- **Steering potentials:** Always enabled by default — without them a substantial fraction of predictions carry severe steric clashes (see the Boltz-2 paper / docs on `use_potentials`)
- **pLDDT rescaling:** Boltz-2 v2 reports 0-1 scale; parser auto-detects and rescales to 0-100
- **MSA caching:** Receptor MSA CSV reused across predictions against the same target

### OpenMMRunner

- **Input:** `OpenMMConfig` with receptor/peptide PDB paths, simulation parameters
- **Output:** `SimulationResult` with trajectory DCD, energy CSV, state XML
- **Force fields:** CHARMM36m protein, TIP3P water, dodecahedral solvent box
- **Buffer environment:** Configurable via `OpenMMConfig` fields or the
  `physiological` / `saliva` / `gastric` / `intestinal` preset classmethods
  (see `biolab_runners.openmm.config.OpenMMConfig` docstrings for the
  ionic concentrations, pH, and temperature of each preset). Field
  defaults are physiological PBS-like (150 mM NaCl, pH 7.4, 310 K).
- **Early abort:** 5 ns / 10 ns checkpoint — if peptide dissociates (PBC-corrected Cα RMSD > 2 × `config.target_irmsd_threshold_a`), abort. Threshold is per-system and defaults to 3.5 Å — override for tighter/looser gating
- **Resume safety:** Always load original topology.pdb — re-solvating produces different water counts
- **Force=True quarantine:** `runner.run(force=True)` moves the resumable files (`state.xml`, `checkpoint.json`, `energy.csv`) into a timestamped `output_dir/.stale/<UTC>/` subdirectory BEFORE any fresh build. This ensures an interrupted forced run cannot leave the directory with a stale `state.xml` that a subsequent non-forced run would pair with a freshly-built topology. Discarding the checkpoint is opt-in via `force=True` and must be atomic w.r.t. the fresh build.
- **Atomic checkpoint:** State files are generation-versioned (`state.<absolute_step>_<pid>_<nanos>.xml`) referenced by `checkpoint.json` (the manifest). The manifest's `os.replace` is the **single atomic commit point** — a crash before the rename leaves the previous (coherent) checkpoint active; a crash after leaves the new (coherent) checkpoint active. The manifest's step is the **absolute** OpenMM step (the simulation's current step at save time, computed as `start_step + steps_done` where `start_step` is the absolute step the loop started at: `total_equil_steps` for fresh runs, the saved step for resumed runs). `energy.csv` is NEVER consulted to determine the saved step. Any `state.*.xml` file not referenced by the manifest is an orphan and is garbage-collected at the next save. A non-empty state file without a valid manifest fails fast and requires `force=True` to discard — the orphaned state cannot be paired with a freshly-built System.
- **Run completion:** The runner decides whether a prior run is terminal from the manifest's absolute step + explicit early-abort metadata, NOT from file existence or size. A mid-production checkpoint can produce a 50 MB trajectory and tens of thousands of energy rows while the run is still in progress — file presence does not imply completion. A run is terminal when EITHER `manifest_step >= total_equil_steps + total_steps` (normal completion) OR `early_abort.json` exists with `aborted=True` (intentional early termination). Tests that pre-populate a "complete" output directory must therefore cross the absolute-step threshold (or write a valid `early_abort.json`), not merely create files that look complete-shaped.
- **Manifest state-file validation:** `load_checkpoint` validates the state file referenced by the manifest before returning a valid resume tuple. The reference must be a basename (no path separators), match the expected pattern (`state.xml` or `state.<step>_<pid>_<nanos>.xml`), exist on disk, and be non-empty. A dangling or unsafe reference raises `InvalidCheckpointError` and forces the runner to fail fast with `force=True` guidance — it cannot degrade into a fresh build because `prepare_simulation` would silently build a new System with a different water count and pair it with the stale state.
- **Restraint force on resume:** Must add restraint force (k=0) to system even on resume, or loadState() fails
- **PBC correction:** All RMSD checks use minimum image convention — without it, RMSD can be ~100A
- **SIGTERM handler:** Clean shutdown on cloud preemption (writes checkpoint before exit)

### Dependencies

- **Boltz-2:** `boltz` CLI on PATH, GPU with 24 GB VRAM (RTX 4090)
- **OpenMM:** Best via conda (`conda install -c conda-forge openmm pdbfixer`); pip only provides OpenCL

## How to Add a New Runner

1. Create `biolab_runners/new_runner/` with `__init__.py`, `config.py`, `runner.py`, `utils.py`
2. Define config + result dataclasses
3. Implement runner class with `run()`, `dry_run`, idempotency, logging
4. Add tests in `tests/` using mocks (no real GPU/CLI deps)
5. Add optional extras in `pyproject.toml`
6. Export from `__init__.py`

## Quality Assurance

```bash
make validate       # Full gate: ruff → pyright → complexity → pytest (read-only, CI-safe)
make quick_validate # Fast gate: ruff + pyright
make lint_fix       # Auto-fix formatting + linting
make test           # Run tests only
make check_links    # Check links with lychee
make check_docs     # Lint markdown files
```

## Quick Reference

```
Package:     biolab_runners (hatchling build)
Python:      >=3.11 (3.11, 3.12 tested)
Lint:        ruff (line-length=100, Google docstrings, C90 ≤10, RUF/ANN/PT enabled)
Types:       pyright standard
Complexity:  complexipy cognitive ≤15
Tests:       pytest with mocks (no GPU/CLI needed)
CI:          .github/workflows/ci.yml (lint → type → complexity → test, Python 3.11+3.12)
Coverage:    70% floor (ratcheting to 80%)
```
