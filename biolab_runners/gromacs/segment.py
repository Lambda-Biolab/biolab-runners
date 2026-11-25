"""Bounded production-MD segment planning and progress state.

One production run may be executed across several ordered invocations. The
approved target (the full ``production_ns`` compiled into ``prod.mdp`` /
``prod.tpr``) never changes; each invocation advances a bounded number of
integration steps and records cumulative progress in the stage manifest.

GROMACS 2026.1 treats ``gmx mdrun -nsteps N`` as RELATIVE to the step stored
in the resume checkpoint: resuming at step 100 with ``-nsteps 200`` finishes
at step 300. The runner therefore passes the number of steps for *this*
segment, not an absolute final step. The full-target step count comes from
the mdp's ``nsteps`` (``_ns_to_steps`` in :mod:`protocol`), so segment and
target always share the same 2 fs tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from biolab_runners.gromacs.protocol import StageKind, _ns_to_steps
from biolab_runners.gromacs.utils import StageStatus, load_stage_manifest

if TYPE_CHECKING:
    from pathlib import Path

    from biolab_runners.gromacs.config import GromacsProtocolConfig

__all__ = [
    "SEGMENT_STATUS",
    "ProductionState",
    "read_cumulative_steps",
    "read_production_state",
    "validate_segment_cumulative",
    "validate_segment_duration",
]

# Per-invocation stage status for a bounded stop that has NOT reached the
# full target. It counts as successful work for the calling runner but is
# deliberately not COMPLETED (the manifest authority must keep the
# production stage resumable).
SEGMENT_STATUS = "segment"


@dataclass(frozen=True)
class ProductionState:
    """Cumulative production progress reported by a protocol invocation."""

    complete: bool
    steps: int
    target_steps: int


def _production_record(work_dir: Path) -> dict[str, object]:
    record = load_stage_manifest(work_dir).get("stages", {}).get(StageKind.PRODUCTION.value)
    return record if isinstance(record, dict) else {}


def read_cumulative_steps(work_dir: Path) -> int:
    """Return production steps completed so far, or 0 when unrecorded."""
    steps = _production_record(work_dir).get("production_steps")
    return steps if isinstance(steps, int) and steps > 0 else 0


def read_production_state(work_dir: Path, config: GromacsProtocolConfig) -> ProductionState:
    """Read truthful completion/steps from the manifest for the caller's result."""
    target_steps = _ns_to_steps(config.effective_production_ns())
    record = _production_record(work_dir)
    steps = record.get("production_steps")
    cumulative = steps if isinstance(steps, int) and steps > 0 else 0
    recorded_complete = record.get("production_complete")
    complete = (
        recorded_complete
        if isinstance(recorded_complete, bool)
        else record.get("status") == StageStatus.COMPLETED
    )
    if complete and cumulative == 0:
        # A pre-segment full run completed without a step counter.
        cumulative = target_steps
    return ProductionState(complete=complete, steps=cumulative, target_steps=target_steps)


def validate_segment_duration(segment_ns: float, config: GromacsProtocolConfig) -> int:
    """Validate one bounded duration and return its integration-step count."""
    if segment_ns <= 0:
        raise ValueError(f"segment_ns must be positive; got {segment_ns}")
    segment_steps = _ns_to_steps(segment_ns)
    if segment_steps <= 0:
        raise ValueError(f"segment_ns must be positive; got {segment_ns}")
    target_steps = _ns_to_steps(config.effective_production_ns())
    if segment_steps > target_steps:
        raise ValueError(
            f"segment ({segment_steps} steps) exceeds the production target ({target_steps} steps)"
        )
    return segment_steps


def validate_segment_cumulative(
    segment_steps: int, cumulative_steps: int, target_steps: int
) -> None:
    """Refuse a segment that would overshoot the full production target."""
    projected = cumulative_steps + segment_steps
    if projected > target_steps:
        raise ValueError(
            f"segment exceeds the production target ({projected} > {target_steps} steps)"
        )
