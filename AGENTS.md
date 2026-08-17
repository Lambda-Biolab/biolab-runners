# AGENTS.md — biolab-runners

This is the primary instruction file for all AI coding agents working on this project.
Read this file first. It supersedes any default behavior.

## Project Purpose

Standalone, modular Python library containing the computational biology runners extracted from the [OralBiome-AMP](https://github.com/Lambda-Biolab/OralBiome-AMP) pipeline:

1. **Boltz2Runner** — Runs Boltz-2 structure predictions for peptide-protein complexes
2. **OpenMMRunner** — Runs OpenMM molecular dynamics simulations with multi-stage equilibration
3. **RFdiffusionRunner** — Runs RFdiffusion for target-conditioned peptide binder / unconditional backbone generation
4. **ProteinMPNNRunner** — Runs ProteinMPNN for fixed-backbone sequence design
5. **RosettaRunner** — Runs Rosetta InterfaceAnalyzer (license-gated)
6. **GROMACSRunner** — GROMACS integration via subprocess for production-scale MD

The runners are designed for researchers who want to use these tools in their own pipelines without importing the full OralBiome-AMP codebase.

## Documentation Hierarchy

This file is the primary instruction document for AI coding agents. For
content outside its scope, follow the chain below — do not duplicate
content from an authority doc into a dependent one.

| Topic | Authority doc | Audience |
|---|---|---|
| Project overview, features, install, quick start | [`README.md`](README.md) | Humans |
| Project purpose, architecture, domain rules, QA | [`AGENTS.md`](AGENTS.md) | AI agents (this file) |
| Contributor workflow, PR guidelines, code style | [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contributors |
| Version history | [`CHANGELOG.md`](CHANGELOG.md) | All |
| Security policy, vulnerability reporting | [`SECURITY.md`](SECURITY.md) | Researchers |
| Community standards | [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | Contributors |
| Agent patterns, incident writeups | [`AGENT_LEARNINGS.md`](AGENT_LEARNINGS.md) | AI agents |
| Escalation to humans | [`AGENT_REQUESTS.md`](AGENT_REQUESTS.md) | AI agents |
| Always-loaded domain constraints | [`.claude/rules/`](.claude/rules/) | Claude Code |
| Claude Code entry point (redirects here) | [`CLAUDE.md`](CLAUDE.md) | Claude Code |

## Architecture

```text
biolab_runners/
├── boltz2/
│   ├── config.py     # Boltz2Config, ConfidenceScores, PredictionResult, QualityGate
│   ├── runner.py     # Boltz2Runner class + apply_quality_gate()
│   └── utils.py      # YAML writer, output parser, availability check
├── openmm/
│   ├── config.py         # OpenMMConfig, SimulationResult
│   ├── runner.py         # OpenMMRunner class (thin dispatcher — MD mechanics only)
│   ├── run_state.py      # Decide fresh/resume/skip/fail_fast; tagged-union RunPlan (deep module — owns the run-state decision)
│   ├── system_builder.py # ForceField, Modeller, System, Integrator, Cα restraint, prepare_simulation
│   ├── checkpoint.py     # Manifest I/O, atomic save, orphan GC, quarantine, inspect_checkpoint (single coherent read), CompletionStatus enum (deep module — owns the entire checkpoint lifecycle)
│   ├── geometry.py       # Pure-numpy PBC math (pbc_correct, min_pbc_distance)
│   ├── offline_gate.py   # Offline mdtraj gate + verdict I/O
│   ├── paths.py          # Centralized filenames (FileNames)
│   └── utils.py          # Availability checks (openmm_available, pdbfixer_available) + diagnostic reporter (verify_production_outputs)
├── rfdiffusion/
│   ├── config.py     # RFdiffusionConfig
│   ├── runner.py     # RFdiffusionRunner class + Hydra CLI translation
│   └── utils.py      # rfdiffusion_available() probe
├── proteinmpnn/
│   ├── config.py     # ProteinMPNNConfig (model_name is a checkpoint prefix)
│   ├── runner.py     # ProteinMPNNRunner + _config_to_cli()
│   └── utils.py      # proteinmpnn_available(), parse_fasta_sequences(), invoke()
├── rosetta/
│   ├── config.py     # RosettaConfig
│   ├── runner.py     # RosettaRunner (license-gated)
│   └── utils.py      # rosetta_available() probe
└── gromacs/
    ├── config.py     # GROMACSConfig
    ├── runner.py     # GROMACSRunner
    └── utils.py      # parse_nthcol() for `.xvg` files
```text

The checkpoint invariants (atomic save, manifest binding, terminal schema, force=True quarantine) all describe `biolab_runners.openmm.checkpoint` — that module is the single source of truth for the checkpoint protocol. `system_builder.py` does the system construction only; it does NOT touch the manifest. `biolab_runners.openmm.checkpoint.inspect_checkpoint(output_dir, config)` is the single canonical entry point — it reads the manifest once and returns a fully-classified `CheckpointSnapshot` carrying the absolute step, state filename, last record, structured `CompletionStatus`, completion reason, and validated terminal payload. The previous multi-call pattern (`load_checkpoint` + `is_run_complete` + `load_terminal_payload`) is now a sequence of thin wrappers that delegate to `inspect_checkpoint`.

`biolab_runners.openmm.run_state` owns the pre-run decision tree (manifest present? terminal? corrupted? orphan? force?) and the skip-population logic (artifact validation, terminal reconstruction). `decide(output_dir, config, force)` is the single public entry point — it reads the snapshot once via `inspect_checkpoint` and returns a tagged-union `RunPlan` (`FreshPlan` | `ResumePlan` | `SkipPlan` | `FailurePlan`). The runner matches on the plan type: `FailurePlan` and `SkipPlan` shortcuts return without MD; `FreshPlan` and `ResumePlan` proceed to `_prepare_simulation`. The runner does NOT know about manifest internals, terminal schemas, quarantine, or artifact validation — those are decisions owned by the modules that own the data.

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
- **Force=True quarantine:** `runner.run(force=True)` moves the resumable files (`state*.xml`, `checkpoint.json`, `energy.csv`, `early_abort.json`) into a timestamped `output_dir/.stale/<UTC>/` subdirectory BEFORE any fresh build. This ensures an interrupted forced run cannot leave the directory with a stale `state.xml` that a subsequent non-forced run would pair with a freshly-built topology, AND cannot leave a stale `early_abort.json` that would mis-classify the next run's intermediate checkpoint as terminal. Discarding the checkpoint is opt-in via `force=True` and must be atomic w.r.t. the fresh build.
- **Atomic checkpoint:** State files are generation-versioned (`state.<absolute_step>_<pid>_<nanos>.xml`) referenced by `checkpoint.json` (the manifest). The manifest's `os.replace` is the **single atomic commit point** — a crash before the rename leaves the previous (coherent) checkpoint active; a crash after leaves the new (coherent) checkpoint active. The manifest's step is the **absolute** OpenMM step (the simulation's current step at save time, computed as `start_step + steps_done` where `start_step` is the absolute step the loop started at: `total_equil_steps` for fresh runs, the saved step for resumed runs). `energy.csv` is NEVER consulted to determine the saved step. Any `state.*.xml` file not referenced by the manifest is an orphan and is garbage-collected at the next save. A non-empty state file without a valid manifest fails fast and requires `force=True` to discard — the orphaned state cannot be paired with a freshly-built System.
- **Manifest ↔ state filename binding:** The v7 generation-versioned state filename embeds the absolute step (`state.<absolute_step>_<pid>_<nanos>.xml`). The runner parses this embedded step and requires it to equal the manifest's `step` field. A mismatch indicates a corrupt or forged checkpoint and raises `InvalidCheckpointError` — the runner cannot resume with a state saved at one step and accounting at another. Legacy `state.xml` has no embedded step and is accepted as a compatibility shim (logged at INFO level so the user can decide to migrate).
- **Terminal status is part of the manifest:** The optional `terminal` field on the manifest's last record carries the early-abort classification (`{"type": "early_abort", "step": <absolute_step>, "reason": <str>, "production_ns": <float>}`). The terminal payload commits in the same `os.replace` as the checkpoint step — there is no separate marker write between the state file and the manifest, so a crash cannot leave a resumable-but-terminal decision unrecorded. The `step` of the terminal payload MUST equal the manifest's `step`; a binding mismatch is classified as an invalid-terminal payload and converted by `run_state.decide()` to `FailurePlan` — the run is in an ambiguous state and the user must invoke `force=True` to discard the manifest. The legacy `early_abort.json` file is written AFTER the atomic save as a derived compat file for downstream consumers (`oral_amp.cloud.openmm_cloud`); it is moved by `force=True` quarantine together with the manifest.
- **Derived marker is best-effort:** The derived `early_abort.json` write is NOT authoritative — the manifest has already committed the terminal decision. A failure to write the derived marker (`OSError` from a full disk, permission denied, etc.) MUST be logged + suppressed — the runner must continue normally and return `early_abort=True`. The crash must NOT propagate out of `_poll_offline_gate()` because the manifest's terminal classification is already durable on disk.
- **Terminal schema validation:** The manifest's optional `terminal` payload is the authoritative terminal marker. A valid payload MUST have `step` as a strict positive `int` (no string coercion), `type == "early_abort"` (the only supported non-normal terminal type — unknown or missing types cause the validator to fail the run), and `reason` as a non-empty string. The `terminal.step` MUST equal the manifest's `step`; a mismatch indicates corruption and causes the validator to fail the run. The terminal check is tri-state: ABSENT (no `terminal` key) — the run is in-progress (or normal-completed, which the snapshot determines separately). INVALID (present but failing any schema check) — the validator returns an explicit `invalid_terminal_<reason>` classification which `run_state.decide()` converts to FAIL_FAST; the run is in an ambiguous state and the runner fails fast with `result.error` and `force=True` guidance. The runner MUST NEVER fall back to the inferred normal-completion heuristic for a present-but-invalid payload. VALID — the run is terminal via manifest payload. The structured `CompletionStatus` enum (`IN_PROGRESS`, `NORMAL_COMPLETE`, `EARLY_ABORT`, `INVALID_TERMINAL`) is the cross-module protocol — downstream code branches on the enum value, not on string-prefixes of the `reason` field.
- **Production-time semantics:** Every ns reported to downstream consumers (`abort_ns`, `result.total_ns`, `md_summary.json`, the reconstructed result) is computed from the COMPLETED PRODUCTION steps (`max(0, absolute_step - total_equil_steps) * timestep_fs / 1e6`). Equilibration steps are protocol setup, not scientific progress — they must never appear in `total_ns`. The checkpoint step stays absolute (`start_step + steps_done`) for resume accounting; only the reported time uses the production invariant. `result.ns_per_day` is INVOCATION-LOCAL throughput (`steps_done * timestep_fs / 1e6 / elapsed * 86400`) — mixing cumulative production with invocation-local wall time would inflate throughput on every resumed run.
- **Run completion:** The runner decides whether a prior run is terminal from the manifest's absolute step + the manifest's terminal payload, NOT from file existence or size. A run is terminal when EITHER `manifest_step >= total_equil_steps + total_steps` (normal completion) OR the manifest's last record has a `terminal` payload with the validated schema (early_abort type, strict-int step equal to manifest.step, non-empty reason). When both signals fire on the same absolute step — the offline-mdtraj gate can land the manifest at exactly the target step on its final chunk — the EXPLICIT terminal payload takes precedence over the inferred normal completion. The explicit payload is the user's stated intent; the step threshold is a heuristic. Tests that pre-populate a "complete" output directory must therefore cross the absolute-step threshold (or write a valid manifest `terminal` payload), not merely create files that look complete-shaped.
- **Terminal artifact validation:** A terminal run is reusable only if its scientific outputs are present and usable. After terminal classification, the runner validates that `trajectory.dcd`, `energy.csv` (with ≥1 data row), and `topology.pdb` exist and are non-empty. A missing or empty artifact produces a clear `result.error` rather than returning success with a nonexistent path. Terminality and artifact validity are separate questions — the manifest records the decision, but the user-facing artefacts must also be there. Note: OpenMM reporters cannot be cleanly flushed mid-simulation; an abrupt failure right after the terminal commit may leave buffered reporter state on disk, requiring artifact repair on the next invocation.
- **Manifest state-file validation:** `load_checkpoint` validates the state file referenced by the manifest before returning a valid resume tuple. The reference must be a basename (no path separators), match the expected pattern (`state.xml` or `state.<step>_<pid>_<nanos>.xml`), exist on disk, and be non-empty. The embedded step in the v7 filename must equal the manifest's `step`. A dangling, unsafe, or step-mismatched reference raises `InvalidCheckpointError` and forces the runner to fail fast with `force=True` guidance — it cannot degrade into a fresh build because `prepare_simulation` would silently build a new System with a different water count and pair it with the stale state.
- **Restraint force on resume:** Must add restraint force (k=0) to system even on resume, or loadState() fails
- **PBC correction:** All RMSD checks use minimum image convention — without it, RMSD can be ~100A
- **SIGTERM handler:** Clean shutdown on cloud preemption (writes checkpoint before exit)

### Dependencies

- **Boltz-2:** `boltz` CLI on PATH, GPU with 24 GB VRAM (RTX 4090)
- **OpenMM:** Best via conda (`conda install -c conda-forge openmm pdbfixer`); pip only provides OpenCL
- **RFdiffusion:** upstream clone at `~/tools/RFdiffusion` (the default `RFDIFFUSION_HOME`) + the in-package `rfdiffusion` console script from the installed wheel (or a `${RFDIFFUSION_BIN}` custom binary); Python with PyTorch + CUDA in the runner interpreter, GPU with ~10 GB VRAM for design. The bootstrap (`~/.local/bin/install-proteinmpnn-rfdiffusion.sh`) clones the repo; model weights are downloaded on first run
- **ProteinMPNN:** upstream clone at `~/tools/ProteinMPNN` + wrapper at `~/.local/bin/proteinmpnn` (same bootstrap script)
- **Rosetta:** license-gated; the runner skips when the license is absent
- **GROMACS:** heavy install; parse_nthcol tested via fixture `.xvg` files

### Integration / Scientific-Validation Tests

The `tests/integration/test_scientific_validation.py` suite is the floor
under which the runner plumbing cannot claim scientific correctness. The
suite is marked `@pytest.mark.integration` and runs as part of the
default `make validate` (the `not slow` filter does not deselect the
integration marker). One heavy test,
`test_openmm_runner_completes_short_vacuum_simulation`, is gated
behind `BIOLAB_RUN_HEAVY_CUDA_TESTS=1` *and* `/dev/nvidia0` reachability
— both must be true for the 1-ps vacuum simulation (~90 s on RTX 4090)
to run. The validation plan is `docs/testing/scientific-validation.md`.

## How to Add a New Runner

1. Create `biolab_runners/new_runner/` with `__init__.py`, `config.py`, `runner.py`, `utils.py`
2. Define config + result dataclasses
3. Implement runner class with `run()`, `dry_run`, idempotency, logging
4. Add tests in `tests/` using mocks (no real GPU/CLI deps)
5. Add an integration test in `tests/integration/test_scientific_validation.py`
   that asserts the runner produces a plausible result on a real
   reference input (cite the literature source in the docstring)
6. Add optional extras in `pyproject.toml`
7. Export from `__init__.py`

## Quality Assurance

```bash
make validate       # Full gate: ruff → pyright → complexity → pytest (read-only, CI-safe)
make quick_validate # Fast gate: ruff + pyright
make lint           # Check linting and formatting
make test           # Run tests only
# Heavy OpenMM CUDA smoke (opt-in; ~90 s on RTX 4090):
BIOLAB_RUN_HEAVY_CUDA_TESTS=1 uv run pytest -m integration -v
```text

## Quick Reference

```text
Package:     biolab_runners (hatchling build)
Python:      >=3.11 (3.11, 3.12 tested)
Lint:        ruff (line-length=100, Google docstrings, C90 ≤10, RUF/ANN/PT enabled)
Types:       pyright standard (see "Pyright mode" section below)
Complexity:  complexipy cognitive ≤15
Tests:       pytest with mocks (no GPU/CLI needed)

## Pyright mode

**Current state:** `typeCheckingMode = "standard"` with tests included
(via `[tool.pyright].include = ["biolab_runners", "tests"]`).

**Target state:** `typeCheckingMode = "strict"` once a follow-up
hardening pass fixes the residual errors.

**Why standard for now:** strict mode is a hard cliff — turning it on
will surface several hundred latent errors that the 13-phase hardening
pass explicitly chose not to fix in a single session. The realistic
upgrade path is:

1. **Today:** stay in standard, but with `[tool.pyright].include` set
   to include both `biolab_runners/` and `tests/` (already done).
   `reportXxx = "error"` overrides elevate the checks that catch
   real bugs while keeping the standard-mode leniency for
   third-party stub noise (MDAnalysis, mdtraj, OpenMM, etc.).
2. **Next session / dedicated hardening pass:** switch to `strict`,
   fix the residual errors, document the strict-mode-required
   patterns (no untyped `def`, no implicit `Any`, etc.) in this
   file's "Domain Rules" section.

**Don't widen pyright suppression** (`reportUnknown* = false`,
`reportAttributeAccessIssue = "warning"`, etc.) without a written
justification. Each suppression hides real bugs from a gate that
should be catching them — the biolab-runners audit found 20 latent
errors that were hidden because tests/ was excluded from pyright.
CI:          .github/workflows/ci.yml (lint → type → complexity → test, Python 3.11+3.12)
Coverage:    70% floor (ratcheting to 80%)
```text
