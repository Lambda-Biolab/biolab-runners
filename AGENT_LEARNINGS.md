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
