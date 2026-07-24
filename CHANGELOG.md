# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Architecture (god-module split)**: Extracted PBC geometry helpers
  (`pbc_correct`, `min_pbc_distance`, `collect_chain_ca_positions`) from
  `OpenMMRunner` into a new `biolab_runners.openmm.geometry` module.
  Extracted the system/forcefield/topology/integrator builder family
  (8 methods) into a new `biolab_runners.openmm.system_builder` module.
  `OpenMMRunner` shrank from 1074 → 763 LOC.
- **Lint (ruff)**: Enabled `RUF`, `ANN`, `PT` rule families; moved `S101`
  to per-file ignores for `tests/**`. ~42 new violations surfaced for
  follow-up.
- **Type checker (pyright)**: Bumped `typeCheckingMode` from `basic` to
  `standard`. Added `tests/` to the include list. Tightened
  `reportMissingImports` from `false` to `warning` (the `mdtraj` optional
  dep is the only "missing" import).
- **Coverage**: Added `pytest-cov` configuration with branch coverage
  and a soft 55% floor (will ratchet to 80% in a follow-up).
- **CI**: Switched from `pip install` to `uv sync --frozen --all-extras`
  via `astral-sh/setup-uv@v6` with built-in caching. Added a complexity
  step (`complexipy`).

### Added
- `.pre-commit-config.yaml` with `ruff-format` and `ruff-check` hooks.
- `make smoke_test` recipe that wraps `smoke_test/run_smoke.py` with a
  guard for missing OpenMM.
- Pytest `addopts` (`-ra --strict-markers --strict-config --tb=short`).
- `uv.lock` is now version-controlled for reproducible CI builds.

## [0.1.0] — 2024

Initial release. Two runners extracted from the OralBiome-AMP pipeline:

- `Boltz2Runner` — Boltz-2 structure predictions for peptide-protein
  complexes, with pocket constraints, steering potentials, MSA caching,
  and a quality gate (PASS / CONDITIONAL / FAIL).
- `OpenMMRunner` — OpenMM MD simulations with multi-stage equilibration,
  CHARMM36m / TIP3P force fields, physiological buffer presets
  (`physiological`, `saliva`, `gastric`, `intestinal`), early-abort
  via offline mdtraj gate, SIGTERM-safe checkpointing, and PBC-aware
  RMSD checks.

[Unreleased]: https://github.com/Lambda-Biolab/biolab-runners/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Lambda-Biolab/biolab-runners/releases/tag/v0.1.0
