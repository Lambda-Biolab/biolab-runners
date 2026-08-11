"""ProteinMPNN sequence-design runner.

Thin subprocess wrapper around the upstream ProteinMPNN
``protein_mpnn_run.py`` script. The runner follows the
``biolab_runners/boltz2/`` shallow pattern: a config dataclass, a
runner class with ``submit`` / ``dry-run`` / idempotency semantics,
and a utils module with availability probes and result parsing.

ProteinMPNN is published as a git repository, not a pip package.
The runner resolves the executable through ``PROTEINMPNN_BIN`` (or
falls back to a ``proteinmpnn`` binary on the PATH) and supports
the ``container://`` URI form for GCP Batch workers.
"""

from biolab_runners.proteinmpnn.config import ProteinMPNNConfig
from biolab_runners.proteinmpnn.runner import ProteinMPNNRunner
from biolab_runners.proteinmpnn.utils import (
    DesignRecord,
    DesignRecordStatus,
    parse_fasta_sequences,
    proteinmpnn_available,
)

__all__ = [
    "DesignRecord",
    "DesignRecordStatus",
    "ProteinMPNNConfig",
    "ProteinMPNNRunner",
    "parse_fasta_sequences",
    "proteinmpnn_available",
]
