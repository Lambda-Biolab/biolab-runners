"""Rosetta constrained-relax and scoring runner.

Thin subprocess wrapper around the upstream ``rosetta_scripts`` CLI.
Rosetta is published as a binary download (not a pip package) and
requires a commercial license for commercial use. The runner expects
the operator to provide the binary via the ``ROSETTA_BIN`` env var
(or accept the default ``rosetta_scripts`` on PATH) and to confirm
the license via the captured :attr:`RosettaConfig.license_acknowledged`
flag.

The runner is intentionally opt-in: it does not ship a binary, does
not import Rosetta, and does not auto-acknowledge the license. The
``license_acknowledged=False`` default makes the bad path loud.

External (real-binary) smoke gate
-----------------------------------

``tests/test_rosetta_runner.py`` contains the unit + synthetic-fixture
coverage for this runner (parser, config, CLI translation, runner
behaviour). A real-binary end-to-end run requires:

1. A licensed Rosetta install reachable via ``$ROSETTA_BIN`` (or a
   ``container://<engine>://<image>`` URL); verified by
   :func:`biolab_runners.rosetta.utils.rosetta_available`.
2. A sample scorefile from a licensed run (synthetic fixtures
   ``never`` substitute for evidence — see the docstring on
   :class:`RelaxScore`).

Because the gate is licensed + binary-dependent and cannot run on a
vanilla dev workstation, the unittest suite stays the source of
truth for parser semantics; the real-binary roundtrip is a manual
check, not a CI gate. Operators running the licensed path should
write their own regression test against their verified fixtures;
the parser API :func:`parse_relax_score` + :func:`parse_score_files`
is the entry point for those tests.
"""

from biolab_runners.rosetta.artifact import ChainAudit, PDBIdentity, RosettaDecoyArtifact
from biolab_runners.rosetta.config import (
    ConstrainedRelaxOptions,
    PreparationMode,
    RosettaConfig,
)
from biolab_runners.rosetta.resolver import RosettaDecoyResolutionRequest, resolve_decoy
from biolab_runners.rosetta.runner import RosettaResult, RosettaRunner
from biolab_runners.rosetta.utils import (
    METRIC_ALIASES,
    RelaxRecord,
    RelaxRecordStatus,
    RelaxScore,
    RelaxScoreRow,
    parse_relax_score,
    parse_relax_score_rows,
    parse_relax_score_rows_text,
    parse_score_file,
    parse_score_files,
    rosetta_available,
)

__all__ = [
    "METRIC_ALIASES",
    "ChainAudit",
    "ConstrainedRelaxOptions",
    "PDBIdentity",
    "PreparationMode",
    "RelaxRecord",
    "RelaxRecordStatus",
    "RelaxScore",
    "RelaxScoreRow",
    "RosettaConfig",
    "RosettaDecoyArtifact",
    "RosettaDecoyResolutionRequest",
    "RosettaResult",
    "RosettaRunner",
    "parse_relax_score",
    "parse_relax_score_rows",
    "parse_relax_score_rows_text",
    "parse_score_file",
    "parse_score_files",
    "resolve_decoy",
    "rosetta_available",
]
