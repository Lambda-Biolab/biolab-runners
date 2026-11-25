"""Bounded production-MD segment planning and progress state.

One production run may be executed across several ordered invocations. The
approved target (the full ``production_ns`` compiled into ``prod.mdp`` /
``prod.tpr``) never changes; each invocation advances a bounded number of
integration steps and records cumulative progress in the stage manifest.

GROMACS 2026.1 treats ``gmx mdrun -nsteps N`` as RELATIVE to the step stored
in the resume checkpoint: resuming at step 100 with ``-nsteps 200`` finishes
at step 300. The runner therefore passes the steps remaining to this segment's
boundary, derived from the checkpoint's own absolute step, so a retry after an
interrupted or unrecorded segment cannot advance past the approved target. The
full-target step count comes from the mdp's ``nsteps`` (``_ns_to_steps`` in
:mod:`protocol`), so segment and target always share the same 2 fs tick.

GROMACS 2026.1 deprecates the command-line ``-nsteps`` flag in favour of
``gmx convert-tpr``, but still honours the relative count. The CPU
integration test pins that behaviour; a GROMACS that removes the flag would
need a converted per-segment tpr, never a change to the approved ``prod.tpr``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from biolab_runners.gromacs.protocol import StageKind, _ns_to_steps
from biolab_runners.gromacs.utils import StageStatus, load_stage_manifest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from biolab_runners.gromacs.config import GromacsProtocolConfig

__all__ = [
    "SEGMENT_STATUS",
    "ProductionState",
    "parse_checkpoint_step",
    "read_cumulative_steps",
    "read_production_state",
    "validate_segment_cumulative",
    "validate_segment_duration",
]

_CHECKPOINT_STEP_LINE = re.compile(r"^step\s*=\s*(\d+)\s*$")

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


def parse_checkpoint_step(lines: Iterable[str]) -> int | None:
    """Return the absolute ``step = N`` from a ``gmx dump -cp`` header."""
    for line in lines:
        match = _CHECKPOINT_STEP_LINE.match(line.strip())
        if match:
            return int(match.group(1))
    return None


def read_production_state(work_dir: Path, config: GromacsProtocolConfig) -> ProductionState:
    """Read truthful completion/steps from the manifest for the caller's result.

    The stage status is the authority: production is COMPLETED only when the
    full target was reached, or when a pre-segment full run finished. A bounded
    stop is recorded as RUNNING and must never read as complete, and a
    completion recorded for a different target is stale.
    """
    target_steps = _ns_to_steps(config.effective_production_ns())
    record = _production_record(work_dir)
    steps = record.get("production_steps")
    cumulative = steps if isinstance(steps, int) and steps > 0 else 0
    recorded_target = record.get("production_target_steps")
    target_matches = not isinstance(recorded_target, int) or recorded_target == target_steps
    complete = record.get("status") == StageStatus.COMPLETED and target_matches
    if complete:
        cumulative = max(cumulative, target_steps)
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
