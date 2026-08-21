"""ProteinMPNN sequence-design runner.

Thin subprocess wrapper around the upstream ProteinMPNN
``protein_mpnn_run.py`` script. The runner follows the
``biolab_runners/boltz2/`` shallow pattern: a config dataclass, a
runner class with ``submit`` / ``dry-run`` / idempotency semantics,
and a utils module with availability probes and result parsing.

ProteinMPNN is published as a git repository, not a pip package. The
installed ``proteinmpnn`` console adapter forwards the runner's argv
contract directly to ``protein_mpnn_run.py``. A custom executable can
still be supplied through ``PROTEINMPNN_BIN``.
The runner rejects ``container://`` binary settings before subprocess dispatch;
use a local executable or an explicit executable command prefix.
"""

from biolab_runners.proteinmpnn.cli import build_command, main, resolve_script
from biolab_runners.proteinmpnn.config import ProteinMPNNConfig
from biolab_runners.proteinmpnn.runner import ProteinMPNNResult, ProteinMPNNRunner
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
    "ProteinMPNNResult",
    "ProteinMPNNRunner",
    "build_command",
    "main",
    "parse_fasta_sequences",
    "proteinmpnn_available",
    "resolve_script",
]
