"""Peptide preparation runner.

The ``biolab_runners.peptide_prep`` package owns the filesystem /
OpenMM / PDBFixer / ParmEd execution that turns a candidate
  peptide backbone + designed sequence into local artifacts:

* ``prepared.pdb`` — minimized, hydrogen-complete structure.
* ``prepared.top`` / ``prepared.gro`` — GROMACS export of the SAME
  OpenMM system/bond graph/net charge.

Public surface:

* :class:`PeptidePrepConfig` — the dataclass the runner consumes.
* :class:`PeptidePrepTopologyDescriptor` (alias of
  ``PeptideTopologyDescriptor``) — the topology modifications.
* :class:`PeptidePrepResult` — the result dataclass.
* :class:`PeptidePrepRunner` — the orchestrator.
* :class:`CoordinateTransformer`, :class:`ChiralityValidator`,
  :class:`ChiralityReport` — the callback Protocols (linear
  all-L preparation needs neither; D-substitution requires both).
* :class:`CoordinateTransformResult` — the typed wrapper accepted
  from a :class:`CoordinateTransformer` (H4 bioml-tools adapter
  compatibility).

The package MUST NOT runtime-import ``bioml_tools``. The
:class:`PeptideTopologyDescriptor` field types are loose enough to
accept the upstream ``bioml_tools.chem.cyclic_topology`` dataclasses
WITHOUT importing them — callers construct those instances
itself and passes them in.
"""

from __future__ import annotations

from biolab_runners.peptide_prep.config import (
    PeptidePrepConfig,
    PeptideTopologyDescriptor,
)
from biolab_runners.peptide_prep.protocols import (
    ChiralityReport,
    ChiralityValidator,
    CoordinateTransformer,
    CoordinateTransformResult,
    extract_coordinate_mapping,
)
from biolab_runners.peptide_prep.runner import PeptidePrepResult, PeptidePrepRunner

# Compatibility alias retained for callers using the longer name.
PeptidePrepTopologyDescriptor = PeptideTopologyDescriptor

__all__ = [
    "ChiralityReport",
    "ChiralityValidator",
    "CoordinateTransformResult",
    "CoordinateTransformer",
    "PeptidePrepConfig",
    "PeptidePrepResult",
    "PeptidePrepRunner",
    "PeptidePrepTopologyDescriptor",
    "PeptideTopologyDescriptor",
    "extract_coordinate_mapping",
]
