"""Rosetta constrained-relax and scoring runner.

Thin subprocess wrapper around the upstream ``rosetta_scripts`` CLI.
Rosetta is published as a binary download (not a pip package) and
requires a commercial license for commercial use. The runner expects
the operator to provide the binary via the ``ROSETTA_BIN`` env var
(or accept the default ``rosetta_scripts`` on PATH) and to confirm
the license via the captured :class:`RosettaConfig.license_acknowledged`
flag.

The runner is intentionally opt-in: it does not ship a binary, does
not import Rosetta, and does not auto-acknowledge the license. The
``license_acknowledged=False`` default makes the bad path loud.
"""

from biolab_runners.rosetta.config import RosettaConfig
from biolab_runners.rosetta.runner import RosettaRunner
from biolab_runners.rosetta.utils import (
    RelaxRecord,
    RelaxRecordStatus,
    parse_score_file,
    rosetta_available,
)

__all__ = [
    "RelaxRecord",
    "RelaxRecordStatus",
    "RosettaConfig",
    "RosettaRunner",
    "parse_score_file",
    "rosetta_available",
]
