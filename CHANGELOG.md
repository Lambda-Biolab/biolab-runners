# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2026-07-25

### Changed
- **Architecture (checkpoint extraction, v14)**: Extracted the entire
  checkpoint domain — manifest read/validate, atomic save, orphan GC,
  quarantine, terminal classification, production step math — from
  `system_builder.py` and `utils.py` into a new
  `biolab_runners.openmm.checkpoint` module. The `_atomic_save_checkpoint`
  function (private by name only — 4 production call sites + 10 test
  sites imported it) is now public `atomic_save_checkpoint` and lives
  alongside the manifest parser, terminal schema validator, and
  quarantine. `load_checkpoint` returns a structured `LoadedCheckpoint`
  dataclass instead of a `(step, file)` tuple, collapsing the previous
  trio of `load_checkpoint` / `load_checkpoint_step` /
  `load_checkpoint_full` into one entry point. `system_builder.py`
  shrinks to a single domain (system / force field / solvation) with
  69% coverage; `utils.py` shrinks to availability checks + a
  diagnostic reporter (re-export of `InvalidCheckpointError` preserved
  for back-compat). `runner.py` imports drop from 14 names across two
  private modules to 7 names from one public module.
- **Architecture (god-module split)**: Extracted PBC geometry helpers
  (`pbc_correct`, `min_pbc_distance`, `collect_chain_ca_positions`) from
  `OpenMMRunner` into a new `biolab_runners.openmm.geometry` module.
  Extracted the system/forcefield/topology/integrator builder family
  (8 methods) into a new `biolab_runners.openmm.system_builder` module.
  `OpenMMRunner` shrank from 1074 → 763 LOC at v8; v9–v13 then added
  back ~540 LOC of new helpers (skip/resume/manifest/terminal/
  orphan-checks) bringing it to 1304 LOC. The v14 extraction does NOT
  shrink the runner further — the helpers are tightly coupled to the
  orchestration context and belong on the runner; what changes is that
  the **seam** the runner crosses for checkpoint operations is now
  public (one module, named functions) instead of private (two modules,
  underscore-prefixed imports).
- **Lint (ruff)**: Enabled `RUF`, `ANN`, `PT` rule families; moved `S101`
  to per-file ignores for `tests/**`. RUF001/002/003 (ambiguous unicode)
  are globally ignored because Greek letters, ×, − are correct domain
  notation in this comp-bio library (ASCII transliteration hurts
  readability).
- **Type checker (pyright)**: Bumped `typeCheckingMode` from `basic` to
  `standard`. Added `tests/` to the include list. `reportMissingImports`
  set to `warning` (mdtraj is an optional dep under the `[openmm]`
  extra; the warning is informational, not an error).
- **Coverage**: 70% floor (ratcheting to 80% in a follow-up). Branch
  coverage enabled.
- **CI**: Switched from `pip install` to `uv sync --frozen --extra
  openmm --group dev` for unit tests (was `--all-extras`; saves ~30 s/run
  by skipping the boltz2 extra since the unit-test suite mocks the CLI).
  Smoke-test job keeps `--all-extras`. `astral-sh/setup-uv@v6` with
  built-in caching. Added a complexity step (`complexipy`) to CI.
- **Removed dead config** (`OpenMMConfig`): `ligand_ff`, `cacl2_mol`,
  `kh2po4_mol`, `solvated_atoms`, `equilibration`. Also removed the
  `EquilibrationStage` frozen dataclass (runner hardcodes the 3-stage
  protocol). Saliva preset documented as NaCl-only until multi-ion
  modeling is added.
- **CHARMM detection**: substring match → strict prefix match
  (`"charmm" in name` → `name.startswith("charmm")`). The substring
  form mis-classified names like `non-charmm-test` as CHARMM.
- **`_resolve_pdb` → `resolve_pdb`**: promoted from private to public
  so callers and tests can reuse the same fallback semantics.

### Added
- `.pre-commit-config.yaml` with `ruff-format` and `ruff-check` hooks.
- `make smoke_test` recipe that wraps `smoke_test/run_smoke.py` with a
  guard for missing OpenMM.
- Pytest `addopts` (`-ra --strict-markers --strict-config --tb=short`).
- `uv.lock` is now version-controlled for reproducible CI builds.
- `biolab_runners/openmm/paths.py`: centralized `FileNames` class
  (production filenames). Replaces ~30 hardcoded string literals
  across 5 files.
- `tests/test_geometry.py` (17 tests): pure-numpy PBC helpers +
  3 Hypothesis property tests (lattice wrap, idempotence, half-box
  bound).
- `tests/test_system_builder.py` (16 tests): mockable entry points
  via `FakeApp` / `FakeQuantity` mirrors of OpenMM's API.
- `tests/_helpers.py`: shared `FakeAtom`, `FakeChain`, `FakeApp`,
  `RecordingForceField`, `dodecahedron_box` (deduped across 3 test
  files).
- `tests/test_checkpoint.py` (40 tests): public-interface tests for the
  new `checkpoint` module — atomic save with crash semantics,
  quarantine with timestamp uniqueness, manifest parsing with all
  malformed variants (string step, negative step, missing records,
  dangling state file, empty state file, path traversal, invalid
  filename pattern), the tri-state terminal classification (absent /
  valid / invalid), terminal payload reconstruction, and production_ns
  math. Replaces the previous 6 test classes in `test_openmm_runner.py`
  and 1 test class in `test_system_builder.py` that imported the
  private `_atomic_save_checkpoint` and the tuple-unpacking
  `load_checkpoint`/`load_checkpoint_step`/`load_checkpoint_full`
  trio; per DEEPENING.md, old unit tests on shallow modules become
  waste once tests at the deepened module's interface exist.
- `biolab_runners/openmm/checkpoint.py`: deep module owning the
  entire checkpoint lifecycle. Public interface —
  `atomic_save_checkpoint`, `quarantine_stale_checkpoint`,
  `load_checkpoint` (returning `LoadedCheckpoint`), `is_run_complete`,
  `load_terminal_payload`, `production_ns`, `InvalidCheckpointError`.
  Internal seam (private helpers for the module's own tests):
  `_parse_manifest`, `_validate_state_file_reference`,
  `_parse_state_filename_step`, `_gc_orphan_states`,
  `_check_normal_completion`, `_validate_terminal_payload`,
  `_classify_invalid_terminal`, `_production_steps`. The AGENTS.md
  invariants around checkpoint / manifest / terminal / quarantine /
  orphan GC / resume safety all describe this module.
- `CHANGELOG.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`.

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
