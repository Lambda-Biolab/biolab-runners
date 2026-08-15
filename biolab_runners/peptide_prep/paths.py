"""Canonical filenames for the peptide-prep runner.

Single source of truth for the on-disk filenames the runner writes,
so the protocol, the manifest, and the GROMACS prebuilt-loading code
all agree on what to look for. The default GROMACS protocol expects
its own ``topol.top`` / ``processed.gro`` filenames; the prebuilt
loading path in :mod:`biolab_runners.gromacs.protocol` stages the
prep-prepared ``prepared.top`` / ``prepared.gro`` into those
canonical names so the downstream ``gmx grompp`` / ``gmx mdrun``
chain remains unchanged.
"""

from __future__ import annotations


class PeptidePrepFiles:
    """Canonical filenames the peptide-prep runner writes."""

    # Minimized, hydrogen-complete structure (the per-candidate PDB).
    PREPARED_PDB = "prepared.pdb"

    # GROMACS export of the same OpenMM system / bond graph / net charge.
    PREPARED_TOP = "prepared.top"
    PREPARED_GRO = "prepared.gro"

    # Provenance manifest — JSON; bind digests of every output + the
    # source backbone + the requested config. See
    # :mod:`biolab_runners.peptide_prep.runner` for the schema.
    MANIFEST = "peptide_prep_manifest.json"
