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
    ├── config.py     # OpenMMConfig, SimulationResult, EquilibrationStage
    ├── runner.py     # OpenMMRunner class (build, equilibrate, produce)
    └── utils.py      # Output verification, checkpoint loading, availability check
```

## Domain Rules

### Boltz2Runner

- **Input:** Receptor sequence + peptide sequence (strings), optional pocket constraints
- **Output:** `PredictionResult` with structure path, confidence scores, quality gate
- **Quality gate:** PASS / CONDITIONAL / FAIL based on ipTM, pLDDT, clash thresholds
- **Steering potentials:** Always enabled by default — without them, 30-60% of predictions have physically impossible structures
- **pLDDT rescaling:** Boltz-2 v2 reports 0-1 scale; parser auto-detects and rescales to 0-100
- **MSA caching:** Receptor MSA CSV reused across predictions against the same target

### OpenMMRunner

- **Input:** `OpenMMConfig` with receptor/peptide PDB paths, simulation parameters
- **Output:** `SimulationResult` with trajectory DCD, energy CSV, state XML
- **Force fields:** CHARMM36m protein, TIP3P water, 140mM NaCl, 310K
- **Early abort:** 5ns/10ns checkpoint — if peptide dissociates (PBC-corrected RMSD > 2x threshold), abort
- **Resume safety:** Always load original topology.pdb — re-solvating produces different water counts
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
Lint:        ruff (line-length=100, Google docstrings, C90 ≤10)
Types:       pyright basic
Complexity:  complexipy cognitive ≤15
Tests:       pytest with mocks (no GPU/CLI needed)
CI:          .github/workflows/ci.yml (lint → type → test, Python 3.11+3.12)
```
