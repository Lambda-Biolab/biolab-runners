"""RFdiffusion backbone-generation runner.

Thin subprocess wrapper around the upstream
``RosettaCommons/RFdiffusion`` CLI. The runner follows the
``boltz2/`` shallow pattern: a config dataclass, a runner class
with ``submit`` / ``dry-run`` / idempotency semantics, and a utils
module with availability probes and result parsing.

RFdiffusion is not pip-installable. Consumers are expected to mount
the upstream Docker image (or build a pinned one) and point the
runner at the container via the ``RFDIFFUSION_BIN`` env var or a
``container://`` URI.

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
