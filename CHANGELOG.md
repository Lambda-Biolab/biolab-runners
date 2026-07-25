# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

- **Architecture (run-state machine extraction, v15, revised)**: Extracted the
  pre-run decision tree and skip-population logic from `OpenMMRunner`
  into a new `biolab_runners.openmm.run_state` deep module. The nine
  private decision-tree methods (`_resolve_skip_or_resume`,
  `_handle_manifest_branch`, `_populate_skip_result`, etc.) are gone.
  The question "given the on-disk checkpoint state, what should the
  runner do?" now has a single public interface:
  `decide(output_dir, config, force) -> RunPlan` returns a tagged-union
  `RunPlan` (`FreshPlan` | `ResumePlan` | `SkipPlan` | `FailurePlan`)
  so invalid constructions (a `FreshPlan` carrying a `resume_xml`, a
  `FailurePlan` carrying a `start_step`) are unrepresentable. The
  `SkipPlan` carries the fully-populated artifact paths, `total_ns`,
  and early-abort fields — the runner just copies them into
  `SimulationResult`. There is no second public call: `populate_skip_result`
  is gone. The `CompletionStatus` enum (`IN_PROGRESS`,
  `NORMAL_COMPLETE`, `EARLY_ABORT`, `INVALID_TERMINAL`) is the
  cross-module protocol for terminal classification, replacing the
  previous string-prefix leak. `biolab_runners.openmm.checkpoint.inspect_checkpoint(output_dir, config)`
  reads the manifest once and returns a fully-classified
  `CheckpointSnapshot`; the previous multi-call pattern
  (`load_checkpoint` + `is_run_complete` + `load_terminal_payload`)
  could combine generation A's state metadata with generation B's
  terminal classification if a concurrent commit landed between
  reads — `inspect_checkpoint` fixes that race.

  Runner shrinks from 1304 → ~933 LOC. Tests rewritten at the new
  public interface (per DEEPENING.md) — `tests/test_run_state.py`
  (43 tests) covers all four plan types, all `CompletionStatus`
  variants, and the invalid-terminal schema variants; the 19
  private-API tests in `test_openmm_runner.py` are deleted. New
  `tests/test_openmm_runner.py::TestRunnerDispatch` (5 tests)
  verifies at the public `run()` interface that `SKIP` and
  `FAIL_FAST` never call `_prepare_simulation`, that artifact
  errors propagate to `result.error`, and that invalid terminal
  payloads always fail fast.

  Behavior change: `load_terminal_payload` is now strict (returns
  `None` for empty reason, wrong type, step mismatch, etc.) — the
  previous lenient behavior was a documented seam leak. Callers
  that need raw access to a possibly-malformed payload should use
  `inspect_checkpoint` directly and read `snapshot.terminal_payload`.

  Behavior change: when `decide` returns `FailurePlan` for an
  `InvalidCheckpointError`, the error message now appends
  "The checkpoint is in an unrecoverable state; re-run with force=True
  to discard it." to the original `InvalidCheckpointError` text. The
  previous implementation copied `str(exc)` directly.

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
- **Architecture (god-module split, v8)**: Extracted PBC geometry helpers
  (`pbc_correct`, `min_pbc_distance`, `collect_chain_ca_positions`) from
  `OpenMMRunner` into a new `biolab_runners.openmm.geometry` module.
  Extracted the system/forcefield/topology/integrator builder family
  (8 methods) into a new `biolab_runners.openmm.system_builder` module.
  At v8 this shrank `OpenMMRunner` from 1074 → 763 LOC.
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
