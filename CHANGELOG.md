# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- **GROMACS interrupted-stage handling (slice S4, post-merge
  hotfix)**: a stage with a RUNNING manifest or an on-disk
  `.cpt` is never silently promoted to `COMPLETED` from disk
  output — it is re-entered via the execution path so the
  `-cpi`/`-append` resume is preserved. An interrupted run now
  halts at the first interrupted stage and reports
  `interrupted=1`, `exit_code=-SIGTERM`, and an error naming
  the stage; the next invocation resumes from that stage.

- **Rosetta `parser:script_vars` argv-flattening regression fix
  (slice S3, review-fix pass)**:

  - The internal CLI representation of ``parser:script_vars`` is
    now a ``list[str]`` (one element per ``%%variable%%`` token),
    not a single space-joined string. ``invoke`` repeats the flag
    once per element so each token lands as its own argv value:
    ``-parser:script_vars k1=v1 -parser:script_vars k2=v2``.
    Pre-joining them as a single argument would have silently
    fed Rosetta the wrong shape (its parser would treat the
    trailing ``"k1=v1 k2=v2"`` blob as one variable name). The
    argv behavior is locked by
    ``tests/test_rosetta_runner.py::test_invoke_repeats_parser_script_vars_per_token``,
    which captures ``subprocess.run``'s actual argv list.
  - ``_config_to_cli`` accepts the dict-channel ``parser:script_vars``
    value as either ``str`` (whitespace-split into tokens — the
    user-friendly form) or ``Sequence[str]`` (each element
    preserved verbatim — the explicit form). The
    ``extra_flags`` channel accepts the canonical
    ``"parser:script_vars k=v ..."`` trailing-tokens form and
    splits on whitespace so each ``k=v`` becomes a separate
    token. Net effect: callers can pass script-var tokens via
    any of three channels and the runner formats them
    identically for the upstream binary.
  - **Additive field split**: ``RelaxScore`` exposes
    ``packstat`` (Lawrence & Coleman packing statistic) and
    ``hbond_energy_fraction`` (fractional H-bond contribution) as
    DISTINCT optional fields, no longer collapsed into
    ``shape_complementarity`` / ``hbond_energy``. The bare
    aliases ``dG`` and ``sc`` are intentionally NOT registered —
    both are too ambiguous (could be interface or whole-complex
    energy, shape complementarity or supercharge). Callers should
    pass the scoped alias (``dG_separated``, ``interface_sc``,
    etc.) for explicit, unambiguous metric assignment. See the
    docstring above :data:`biolab_runners.rosetta.utils.METRIC_ALIASES`
    for the full list of accepted aliases.
  - **Positional ``RelaxRecord`` compat**: the additive ``score``
    field is appended AFTER the legacy field set
    (``index, path, total_score, status, error``), so the
    v0.5 positional-args form ``RelaxRecord(0, "x.sc", -99.5,
    "succeeded", "")`` continues to bind to those fields
    verbatim. The new ``score`` field defaults to an empty
    :class:`RelaxScore` when not supplied (positional or
    keyword). Locked by
    ``tests/test_rosetta_runner.py::test_relax_record_legacy_positional_constructor_preserved``.
  - **JSON serialization**: the legacy ``payload["total_score"]``
    keeps its v0.5 float-as-string repr for source compat, but
    the new ``payload["metrics"]`` dict emits native Python
    ``float`` / ``None`` so downstream JSON consumers don't need
    a float-parsing detour for the structured form. See
    ``RelaxScore.to_dict`` and
    ``tests/test_rosetta_runner.py::test_relax_score_to_dict_emits_native_numbers``.
  - **Accept / reject edge cases** in ``RosettaConfig.__post_init__``:
    rejects empty-string ``preparation_mode`` (``""`` is not
    ``None`` but is not a recognized Literal value); an empty
    ``ConstrainedRelaxOptions()`` contributes no script-var
    tokens at all. Locked by
    ``tests/test_rosetta_runner.py::test_config_rejects_empty_preparation_mode``
    and
    ``tests/test_rosetta_runner.py::test_config_to_cli_empty_constrained_relax_emits_no_script_vars``.

  **Backward-compat summary (unchanged from v0.6)**: every public
  symbol a v0.5 consumer could have touched continues to work
  byte-for-byte. New symbols are strictly additive.

- **RosettaRunner scorefile parser + structured CLI translation
  (slice S3, hardening pass)**:

  - `biolab_runners.rosetta.utils.parse_score_file` remains the
    legacy entry point: it returns the first-row `total_score` as
    a `float`, with `0.0` for empty / garbage / unparseable files.
    **No consumer change required** to keep working.
  - New `biolab_runners.rosetta.utils.parse_relax_score` returns
    the structured `RelaxScore` form (every named metric, with
    `None` for absent columns). New code should prefer this.
  - New `biolab_runners.rosetta.utils.RelaxScore` dataclass
    exposes every InterfaceAnalyzer-style metric by name:
    `total_score`, `total_sasa`, `delta_sasa`, `hydrophobic_sasa`,
    `polar_sasa`, **`interface_polar_sasa`** (new), **`interface_hydrophobic_sasa`** (new),
    `interface_dG`, `interface_dSASA`,
    `buried_unsatisfied_hbonds`, `cross_interface_hbonds`,
    `hbond_energy`, `shape_complementarity`. Each field is
    `float | None` — never defaulted to `0.0` for absent columns;
    upstream `inf` / `-inf` / `nan` sentinel tokens are also
    surfaced as `None` to preserve the missing-vs-present contract.
  - New canonical InterfaceAnalyzer column-name aliases added to
    `METRIC_ALIASES`: **`dSASA_int`** (interface dSASA),
    **`dSASA_polar`** / `polar_int_sasa` (buried polar SASA at the
    interface — distinct from total `polar_sasa`),
    **`dSASA_hphobic`** / `hphobic_int_sasa` (buried hydrophobic
    SASA at the interface — distinct from total `hydrophobic_sasa`),
    **`delta_unsatHbonds`** (delta unsatisfied H-bonds),
    **`hbond_E_fraction`** (fractional hbond energy — collapses
    into `hbond_energy` for historical compatibility; the
    fraction-vs-absolute distinction is the consumer's
    responsibility), and **`packstat`** (Lawrence & Coleman
    packing statistic — collapses into `shape_complementarity`).
  - `RelaxRecord` keeps its legacy `total_score: float` field
    *and* gains a structured `score: RelaxScore` field; the two are
    populated independently. `RelaxRecord.to_dict()` retains the
    legacy v0.5 keys (`index` / `path` / `total_score` / `status` /
    `error`) byte-for-byte and adds a new `metrics` key carrying
    the structured form. Consumers that only read the legacy keys
    see no behavior change.
  - `parse_score_files` now marks all-None / garbage rows as
    `FAILED` (with a synthetic `error` string) so the runner's
    `RosettaResult.succeeded` / `failed` counters reflect "this row
    meant something", not "this row was syntactically readable".
    Counts are taken by per-record `status`, so OSError-on-read and
    unrecognized-header rows both decrement `succeeded`.
  - `RosettaConfig` gains structured options for the upstream
    `%%prep_mode%%` and constrained-relax protocol variables:
    `preparation_mode: Literal["linear", "cyclic"]` and
    `constrained_relax: ConstrainedRelaxOptions | None`. Both are
    optional and translate into `-parser:script_vars key=value`
    pairs.
  - `_config_to_cli` now applies a documented **precedence order**
    for `parser:script_vars` tokens:

    1. Structured `preparation_mode` / `constrained_relax` tokens.
    2. `config.extra["parser:script_vars"]` (the dict-channel
       escape hatch).
    3. `config.extra_flags` entries of the form
       `"parser:script_vars k=v ..."`.

    Each step appends to the same accumulated list; no step
    silently clobbers tokens from earlier steps (the v0.5
    `--no-clobber` regression this fixed is exercised in
    `tests/test_rosetta_runner.py::test_config_to_cli_extra_flags_with_script_vars_does_not_clobber`).

  **Backward-compatibility summary**: every public symbol a v0.5
  consumer could have touched (`parse_score_file` returning
  `float`, `RelaxRecord.total_score`, `RelaxRecord.to_dict`'s
  legacy keys) is preserved. New symbols (`parse_relax_score`,
  `RelaxScore`, `RelaxRecord.score`, `to_dict()["metrics"]`,
  `preparation_mode`, `constrained_relax`) are strictly additive.

  **Real-binary smoke gate**: there is no pytest that invokes
  `rosetta_scripts` — the binary is licensed and not installed in
  CI. Operators with a licensed install should write a regression
  test against a captured scorefile using `parse_relax_score` /
  `parse_score_files`; the entry points are documented in the
  module docstring on `biolab_runners.rosetta`.

- **Slice 12 (MD-OPENMM-001) — `OpenMMConfig.from_md_spec` classmethod**:
  the canonical construction path going forward. Projects every
  engine-neutral field on `bioml_tools.md.system_spec.MDSpec` (added
  in bioml-tools 1.9.0) onto the matching `OpenMMConfig` slot, and
  accepts OpenMM-only overlays (`openmm_platform`, `water_ff_xml`,
  `extra_forcefields`, `target_irmsd_threshold_a`) via
  `**engine_overrides`. The allowlist is fail-closed — unknown
  engine-specific keys raise `TypeError` at the construction boundary
  (catches the "production_NS vs production_ns" typo class).

  **Slice 16 wire-up (biolab-runners#189)** — `from_md_spec` now
  projects the four fields that were previously deferred:
  `equilibration_ps` (reduced from `spec.equilibration`, a tuple
  of stage dicts, via `_extract_equilibration_ps`), `pme`,
  `minimization_max_iterations`, `constraints`. The runner now
  honors them:
  - `_run_equilibration` reads `config.equilibration_ps` for stage
    durations and `config.minimization_max_iterations` for the
    steepest-descent cap; stage 3 uses a fixed 100 ps ramp +
    `npt_free − 100` ps unrestrained remainder (with a warning
    when the configured `npt_free` is shorter than the ramp).
  - `system_builder._create_system` reads `config.pme`
    (`app.PME` ↔ `app.CutoffPeriodic`) and `config.constraints`
    (string mapped to `app.<NAME>`, with `"None"` → Python `None`
    for OpenMM's "no constraints" sentinel).

  **Behavior change** (spec-driven paths only): the runner's legacy
  hardcoded minimization cap was 1,000 iterations. The
  `ACTIVIN_E_PRODUCTION_PROFILE` carries `minimization_max_iterations=50_000`,
  which now reaches the runner — spec-driven runs minimize up to 50×
  longer than before. Steepest descent usually converges early, but
  systems that previously hit the 1,000-iteration cap will reach a
  genuinely different minimized structure. `OpenMMConfig()` direct
  callers are unaffected (the dataclass default is 1,000, matching
  legacy).

  `to_dict` includes the 4 fields under `simulation`; `from_json`
  parses them back. Legacy `system_config.json` files (without the
  new keys) default to the legacy values so in-flight simulations
  continue unchanged. Closes biolab-runners#189.

- **Slice 14 (BMT-MD-001) — `biolab_runners.mmpbsa` package**:
  optional gmx_MMPBSA integration. The runner gracefully degrades
  to `status="unsupported"` when the binary is missing — slice 14
  acceptance: missing optional tooling yields `unsupported`, not a
  fabricated value. Public surface:
  `GmxMMPBSARunner` (subprocess wrapper around gmx_MMPBSA's
  per-residue decomposition), `GmxMMPBSAStatus` (single class
  object, defined in `runner.py` next to the only emitter, and
  re-exported from the package root), `GmxMMPBSARecord`
  (per-energy-component breakdown), `gmx_mmpbsa_available`
  (PATH probe), `parse_residue_decomposition` (file parser for
  the `<prefix>_residue_decomposition_*.dat` output). The
  `parse_residue_decomposition` function was refactored from
  cognitive complexity 24 to 8 via five single-purpose helpers
  (`_read_lines`, `_is_skippable_line`, `_split_chain_residue`,
  `_parse_energy_tokens`, `_build_record`) to satisfy the
  biolab-runners complexity gate (15).

- **CI: private-repo dep resolution**: the `[openmm]` extra pins
  `bioml-tools @ git+https://github.com/Lambda-Biolab/bioml-tools.git@v1.9.0`
  (a private repo on the same GitHub org). The default `GITHUB_TOKEN`
  on a public repo's CI runner can't read cross-org private deps, so
  `uv sync` failed with "could not read Username for 'github.com'".
  Add a `GH_BIOML_TOOLS_TOKEN` PAT as a repo secret and pipe it
  through `gh auth login --with-token` + `gh auth setup-git` in
  `ci.yml` and `boltz-deps-resolve.yml` so the credential helper
  is registered before `uv sync` runs.

  Follow-up: publish bioml-tools to PyPI (Lambda-Biolab/bioml-tools#43)
  and drop the git URL + the `GH_BIOML_TOOLS_TOKEN` secret.

- **CI workflow fix (`boltz-deps-resolve.yml`)**: `uv lock --upgrade --all-extras`
  was the nightly dep-resolution job failing every run. The `--all-extras`
  flag is invalid for `uv lock` (it is only valid for `uv sync`). Fix:
  drop the flag — `uv lock --upgrade` is sufficient. Also added
  `.opencode/` and `mutants/` to `.gitignore`.

- **Heavy CUDA smoke env-gate restored**: `BIOLAB_RUN_HEAVY_CUDA_TESTS=1`
  opt-in on `test_openmm_runner_completes_short_vacuum_simulation`
  was accidentally dropped by the squash-merge of PR #181. The
  1-ps vacuum simulation (~90 s on RTX 4090, >10 min on CPU) now
  skips by default for both the pre-push gate and CI; only operators
  who set `BIOLAB_RUN_HEAVY_CUDA_TESTS=1` pay the GPU cost. Reset
  the default `--skipif` condition that was lost in the squash.

- **Bug fix (runner ProteinMPNN model_name default)**:
  `ProteinMPNNConfig.model_name` defaulted to `"vanilla_model_weights"`
  — a *folder* name. Upstream `protein_mpnn_run.py:57` joins
  `<model_folder_path>/<model_name>.pt`, so passing a folder name
  silently broke checkpoint loading on every default invocation.
  Switch the default to `"v_48_020"` (one of the four upstream
  `--model_name` choices: `v_48_002`, `v_48_010`, `v_48_020`,
  `v_48_030` — each is a checkpoint *prefix*, joined with `.pt`
  upstream). The runner's contract is unchanged: callers may pass any
  of the four prefixes via `ProteinMPNNConfig(model_name=...)`. The
  companion
  `test_config_to_cli_default_includes_four_sequences` is updated to
  assert the new default; a parametrised
  `test_config_to_cli_supports_all_upstream_checkpoints` and a guard
  `test_proteinmpnn_config_default_is_an_upstream_checkpoint_prefix`
  lock the contract against re-introducing a folder name.
  See [protein_mpnn_run.py:57](https://github.com/dauparas/ProteinMPNN/blob/main/protein_mpnn_run.py#L57)
  and the upstream `--model_name` argparse help text.

- **Bug fix (runner total_ns precision, v15 follow-up)**: `total_ns` in
  `SimulationResult` was rounded to 2 decimal places, which silently
  collapsed sub-100 ps simulations (``round(0.001, 2) == 0.0``). The
  1-ps smoke test in ``tests/integration/test_scientific_validation.py``
  used to see ``total_ns == 0.0`` even though the production loop ran
  1,000 steps. Three call sites changed from ``round(_, 2)`` to
  ``round(_, 6)``, giving sub-femtosecond precision in ns while still
  hiding floating-point artefacts from non-integer timesteps:

    * ``biolab_runners.openmm.runner._finalize_result`` — the post-run
      assignment to ``result.total_ns``.
    * ``biolab_runners.openmm.config.SimulationResult.to_dict`` — the
      JSON serialisation.
    * ``biolab_runners.openmm.run_state._normal_completion_total_ns``
      — the skip-population reconstruction path.

  Updated ``tests/test_run_state.py::test_normal_completion_total_ns_rounded_for_float_timestep``
  to assert ``plan.total_ns == 1.5678`` (the 6-dp-rounded value) and
  updated ``tests/test_run_state.py::test_skip_path_populates_full_skip_plan``
  to round against the new contract. Added a new regression assertion
  in the integration suite that ``md_summary["total_ns"]`` is in
  ``[0.0001, 0.01]`` for a 1-ps production, so the round-to-2
  regression is now caught by CI.

- **Architecture (run-state machine extraction, v15, revised twice)**: Extracted the
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
  (~60 tests) covers all four plan types, all `CompletionStatus`
  variants, all plan invariant checks (action non-overridable,
  required fields, cross-field invariants), the invalid-terminal
  schema variants, and the artifact-corruption matrix; the 19
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

  Plan type contract tightened: `action` is now a `ClassVar` (cannot
  be overridden via constructor — `FreshPlan(action=Action.SKIP)` is
  a `TypeError`), required fields have no defaults (`ResumePlan()` is
  a `TypeError`), and `__post_init__` validates domain invariants
  (negative steps, empty paths, early-abort-without-reason).
  Invalid constructions are now unrepresentable.

  Bug fix: `inspect_checkpoint` now distinguishes "key absent"
  from `"terminal": null`. Previously, `last_record.get("terminal")`
  returned `None` for both, so a `terminal: null` manifest at the
  target step was silently accepted as normal completion — exactly
  the failure mode the invalid-terminal rule is intended to prevent.
  The new behavior classifies `terminal: null` as `INVALID_TERMINAL`
  with reason `invalid_terminal_null`, which `run_state.decide()`
  converts to `FailurePlan`.

  Behavior restored: normal-completion `total_ns` is now `round(_, 2)`
  — matches the original skip-population path. The intermediate
  implementation dropped the rounding, which would have leaked
  floating-point artifacts for float `timestep_fs` values. The new
  behavior preserves the exact-value contract the runner had before
  this refactor. Test `test_normal_completion_total_ns_rounded_for_float_timestep`
  uses `production_ns=1.5678` so the unrounded result (`1.5678`)
  differs from the rounded (`1.57`) — the test would fail if
  `round()` were removed again.

  Bug fix: `SkipPlan.abort_reason` is now typed `str` (not
  `Optional[str]`). The previous `None` value silently serialised
  as `"abort_reason": null` in `md_result.json`, breaking the
  long-standing contract that downstream consumers
  (`oral_amp.cloud`) rely on. Normal completions now serialise as
  `"abort_reason": ""`. New runner-level test
  `test_normal_skip_serializes_abort_reason_as_empty_string`
  asserts both `result.abort_reason == ""` AND
  `result.to_dict()["abort_reason"] == ""`.

  Cross-field invariants tightened: the reviewer flagged that
  semantically-contradictory plans still constructed. Added:
  - `ResumePlan.start_step == manifest_step` (refuse mismatched
    step accounting).
  - `Path(resume_xml).name == state_file_basename` (refuse
    manifest/loader path disagreement).
  - `SkipPlan.completion` ∈ {`NORMAL_COMPLETE`, `EARLY_ABORT`}
    (refuse `IN_PROGRESS` or `INVALID_TERMINAL` — those flow
    through `FreshPlan` / `ResumePlan` / `FailurePlan` upstream).
  - `EARLY_ABORT` requires `early_abort=True` and a non-empty
    `abort_reason`.
  - `NORMAL_COMPLETE` requires `early_abort=False` and
    `abort_reason=""`.

  Bug fix: `_artifact_validation_error` now treats directory-
  disguised-as-file (`trajectory.dcd/`), binary-corrupted
  `energy.csv`, and `OSError` from `stat()` as `FailurePlan`
  causes rather than letting them leak out of `decide()` as
  unhandled exceptions. The previous behaviour inherited from
  the runner's `_validate_terminal_artifacts` would crash the
  process on a partial disk failure during the original run.
  New tests `test_directory_disguised_as_trajectory_yields_failure`,
  `test_directory_disguised_as_energy_yields_failure`, and
  `test_binary_corrupted_energy_yields_failure` exercise the
  hardening.

- **Validation (post-v15 smoke runs, RTX 4090 + OpenCL)**: All three
  smoke scripts (`smoke_test/run_smoke.py`, `run_resume.py`,
  `run_true_resume.py`) ran cleanly on OpenMM 8.5.1 via OpenCL
  against the same 50-ps peptide-protein system as the Apr-14
  reference outputs. The post-refactor `trajectory.dcd` is
  byte-identical to the reference (184636 bytes). Bug-fix evidence
  worth pulling out: the pre-v15 resume was a no-op for extended
  configs — phase 2 with `total_steps > saved_step` did not run any
  new steps (energy.csv stayed at step 210000). The post-v15 runner
  correctly extends 210000 → 225000 (3 new energy checkpoints); the
  v15 `run_state.decide()` decision tree tracks `remaining_steps`
  correctly between `ResumePlan` invocations, where the pre-v15
  runner silently treated "manifest step exists" as terminal. The
  true-resume scenario similarly shows the post-v15 runner respects
  the `total_steps` override (extends to 650000); the pre-v15
  reference ignored it (stuck at 450000). See `AGENT_LEARNINGS.md`
  for the full validation report (DCD byte-identity signal, OpenCL
  vs CUDA throughput gap, platform-hardcode caveat).

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

## [0.2.0] — 2026-08

Adds the RFdiffusion runner (Slice 5).

- **`RFdiffusionRunner`** — `biolab_runners.rfdiffusion.runner.RFdiffusionRunner`
  drives the upstream RFdiffusion Hydra CLI for unconditional / motif-
  scaffolding backbone generation. Subprocess wrapper with Hydra
  `contigmap.contigs=...` translation, JSON config passthrough, and
  hook-based result collection. Companion availability probe
  `rfdiffusion_available()` returns `True` iff the wrapper is on
  `$PATH` and exits 0 on `--help`.
- `biolab_runners/rfdiffusion/config.py` — `RFdiffusionConfig` dataclass
  with `length`, `contig_map`, `num_designs`, and JSON config passthrough.
- `tests/integration/test_scientific_validation.py` — early stub for
  `test_rfdiffusion_runner_availablity_check_works` (later re-named
  by the v0.4.0 doc-sweep PR).
- Cleanup: removed dead `.markdownlint.json` and empty `MEMORY.md`;
  declared the documentation hierarchy and refreshed `llms.txt`.

## [0.3.0] — 2026-08

Adds the ProteinMPNN runner (Slice 6).

- **`ProteinMPNNRunner`** — `biolab_runners.proteinmpnn.runner.ProteinMPNNRunner`
  drives the upstream ProteinMPNN `protein_mpnn_run.py` for fixed-backbone
  sequence design. CLI translation: biolab-runners contract
  (`--input_path` / `--output_path` / `--batch_size` / `--seed` /
  `--sampling_temp` / `--num_seq_per_target` /
  `--model_name` / `--ca_only` / `--omit_AA` / `--fixed_positions`)
  to upstream's expected `--pdb_path` / `--out_folder` / `--model_name`
  / `--sampling_temp` etc. Idempotent skip on existing FASTA output.
  Companion `proteinmpnn_available()`.
- `biolab_runners/proteinmpnn/config.py` — `ProteinMPNNConfig` dataclass
  with `task_count`, `temperature`, `seed`, `model_name`, `ca_only`,
  `fixed_positions`, `omit_aa`, `extra` passthrough.
- `tests/integration/test_scientific_validation.py` —
  `test_proteinmpnn_parse_fasta_returns_both_records_with_protein_alpha`.

## [0.4.0] — 2026-08

Adds the Rosetta runner (license-gated).

- **`RosettaRunner`** — `biolab_runners.rosetta.runner.RosettaRunner` is
  the fourth runner. Subprocess wrapper for the Rosetta InterfaceAnalyzer
  binary. Skips at the availability probe when the Rosetta license is
  absent (Rosetta is closed-source / commercial). Companion
  `rosetta_available()`.
- `biolab_runners/rosetta/config.py` — `RosettaConfig` dataclass.

The release commit also touched the Pre-PR gate machinery (the
`hashFiles()` job-level `if:` fix in PR #157) and rolled GitHub Actions
versions forward (PRs #115, #117, #121, #122, #127, #137 — all
through `b34f41a`).


[Unreleased]: https://github.com/Lambda-Biolab/biolab-runners/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Lambda-Biolab/biolab-runners/releases/tag/v0.4.0
[0.3.0]: https://github.com/Lambda-Biolab/biolab-runners/releases/tag/v0.3.0
[0.2.0]: https://github.com/Lambda-Biolab/biolab-runners/releases/tag/v0.2.0
[0.1.0]: https://github.com/Lambda-Biolab/biolab-runners/releases/tag/v0.1.0
