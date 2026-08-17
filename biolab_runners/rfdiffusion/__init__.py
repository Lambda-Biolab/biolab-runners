"""RFdiffusion backbone-generation runner.

Thin subprocess wrapper around the upstream
``RosettaCommons/RFdiffusion`` CLI. The runner follows the
``boltz2/`` shallow pattern: a config dataclass, a runner class
with ``submit`` / ``dry-run`` / idempotency semantics, and a utils
module with availability probes and result parsing.

RFdiffusion is not pip-installable. The installed wheel ships the
``rfdiffusion`` console script (``biolab_runners.rfdiffusion.cli``),
which adapts the runner's flag contract to the stock upstream
``scripts/run_inference.py`` located under ``RFDIFFUSION_HOME``
(default ``~/tools/RFdiffusion``; model weights required). Custom
binaries implementing the same contract can be supplied via the
``RFDIFFUSION_BIN`` env var.

Backbone generation emits one PDB file per design; the runner parses
each into a structured :class:`RecordData` result. Failures are
recorded per-task so downstream consumers can short-circuit.
"""

from biolab_runners.rfdiffusion.config import RFdiffusionConfig
from biolab_runners.rfdiffusion.runner import RFdiffusionRunner
from biolab_runners.rfdiffusion.utils import (
    RecordData,
    RecordDataStatus,
    parse_backbone_pdb,
    rfdiffusion_available,
)

__all__ = [
    "RFdiffusionConfig",
    "RFdiffusionRunner",
    "RecordData",
    "RecordDataStatus",
    "parse_backbone_pdb",
    "rfdiffusion_available",
]
