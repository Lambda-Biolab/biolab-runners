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
