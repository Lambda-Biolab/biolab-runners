# Agent Learnings — biolab-runners

Patterns and incidents discovered during development. Use this as a lookup before solving problems.

## Boltz-2 Steering Potentials — Physically Impossible Structures

**Context:** Boltz2Runner structure predictions.

**Problem:** Without steering potentials, a substantial fraction of predictions carry severe steric clashes (see the Boltz-2 paper's `use_potentials` discussion).

**Solution:** Steering potentials always enabled by default. Promoted to `.claude/rules/steering-potentials.md`.

## OpenMM PBC Correction — RMSD Artifacts

**Context:** OpenMMRunner RMSD stability checks during MD.

**Problem:** Without minimum image convention, PBC artifacts produce RMSD values of ~100 A, causing false binding-failure verdicts.

**Solution:** All RMSD checks use minimum image convention (PBC-corrected). Promoted to `.claude/rules/pbc-correction.md`.

## OpenMM Restraint Force on Resume — loadState Failure

**Context:** OpenMMRunner checkpoint resume.

**Problem:** `loadState()` fails if the restraint force is not added to the system before loading, even when restraint strength (k) is zero.

**Solution:** Always add restraint force (k=0) to system before `loadState()`, even on resume. Promoted to `.claude/rules/restraint-force-resume.md`.

## Checkpoint Domain Extraction — Public Seam Aligned with Reality

**Context:** v14 refactor (`biolab_runners/openmm/checkpoint.py` extraction).

**Problem:** The checkpoint domain (manifest read, atomic save, orphan GC, quarantine, terminal classification, production step math) was scattered across `system_builder.py` (the `_atomic_save_checkpoint` private function — 4 production call sites + 10 test sites imported it despite the underscore prefix) and `utils.py` (manifest parsing, terminal schema validation, `load_checkpoint` trio). The seam the underscore prefix advertised did not match the seam the codebase actually used.

**Solution:** Extracted into a new `biolab_runners.openmm.checkpoint` deep module. The public interface (`atomic_save_checkpoint`, `quarantine_stale_checkpoint`, `load_checkpoint` returning `LoadedCheckpoint`, `is_run_complete`, `load_terminal_payload`, `production_ns`, `InvalidCheckpointError`) matches what callers actually use. Tests were rewritten at the public interface (per DEEPENING.md "delete old tests on shallow modules") — the 10 private-import sites in `test_system_builder.py` and `test_openmm_runner.py` are gone, replaced by `tests/test_checkpoint.py` covering the same behavior through one public seam.

**Pattern:** When a function is prefixed with `_` but imported by multiple modules / test files, the visibility annotation is lying — the function is de-facto public. Promote it, give it a module that owns the surrounding domain, and align the seam with reality. Don't leave the underscore as a hint that was wrong.

## OpenMM Smoke Validation — DCD Byte-Identity + Pre-v15 Resume No-Op

**Context:** Post-v15 refactor validation on RTX 4090 + OpenCL (Jul 26).
Ran all three smoke scripts (`smoke_test/run_smoke.py`, `run_resume.py`,
`run_true_resume.py`) against the Apr-14 pre-refactor CUDA references
(`smoke_test/out_main/`, `out_resume_main/`, `out_true_main/`) and
compared every field in `smoke_verify.json` / `resume_verify.json` /
`true_resume_verify.json`.

**Problem:** Unit tests are 82% mock-based — they cover the decision
tree and manifest schema at the new public interface (`run_state.decide()`,
`checkpoint.inspect_checkpoint`) but not the end-to-end MD roundtrip
(DCDReporter, PDBReporter, statexml, loadState with restraint force).

**Solution:** Real-MD smoke runs add two kinds of evidence the mocks
can't: (1) the file-format roundtrip, and (2) deterministic fields
that survive platform noise. The strongest single signal is
`trajectory.dcd` size — the post-refactor run produced 184636 bytes,
**byte-identical** to the Apr-14 reference. DCD frame count is
deterministic per OpenMM build; coordinate-precision differences ride
inside the frame payload but don't pad the file. Energy `step` column,
`time/ps` column, and `topology.pdb` line count (3073) are also
deterministic. PE/KE values drift ~0.3% between CUDA and OpenCL on
the same 4090 (platform FFT/sum reorder noise); `ns_per_day` is ~30%
lower on OpenCL (1108 vs 1596).

**Bug-fix evidence (worth flagging in the v15 CHANGELOG):** the pre-v15
resume was a no-op for extended configs — phase 2 with
`total_steps > saved_step` did not run any new steps (energy.csv stayed
at step 210000). The post-v15 runner correctly extends 210000 → 225000
(3 new energy checkpoints). The v15 `run_state.decide()` decision tree
tracks `remaining_steps` correctly between `ResumePlan` invocations;
the pre-v15 runner silently treated "manifest step exists" as terminal.
The true-resume scenario shows the same pattern: post-v15 respects the
`total_steps` override (extends to 650000); pre-v15 ignored it (stuck
at 450000).

**Pattern:** When a refactor is validated by mock-based unit tests,
real-MD smoke runs are cheap insurance (~30 s per scenario on a 4090)
and surface a class of bugs the mocks can't: end-to-end file-format
roundtrip, deterministic reproducibility signals across platforms, and
silent regressions in the resume code path. The byte-identical DCD
file is the strongest single comparator — if the size matches, the
structure is structurally identical to the reference, coordinate
noise lives inside the frames.

## Smoke Test Platform Hardcode — Pip vs Conda OpenMM

**Context:** The smoke test scripts (`smoke_test/run_smoke.py`,
`run_resume.py`, `run_true_resume.py`) hardcode
`openmm_platform="CUDA"`. The library default is `OpenCL`
(`biolab_runners/openmm/config.py:36`).

**Problem:** Pip-OpenMM installs only ship `Reference`, `CPU`, and
`OpenCL` platforms (no CUDA). `AGENTS.md` notes this explicitly:
"OpenMM: Best via conda (`conda install -c conda-forge openmm pdbfixer`);
pip only provides OpenCL." Running the smoke scripts on a pip-OpenMM
install fails with `Platform CUDA not available` before any MD runs.
On this box (RTX 4090, pip-OpenMM 8.5.1) the available platforms are
exactly `Reference`, `CPU`, `OpenCL` — matching the AGENTS.md caveat.

**Solution:** Either install OpenMM via conda (which ships CUDA) or
edit the platform string in the smoke scripts before running local
validation. The library itself defaults to OpenCL for this reason — the
hardcoded CUDA in the smoke scripts is for the conda-target CI
environment. The `make smoke_test` recipe doesn't override the
platform, so a pip-OpenMM local run will fail.

