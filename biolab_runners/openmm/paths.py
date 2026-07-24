"""Centralized filenames for OpenMM MD outputs.

Single source of truth for the production filenames that the runner,
the offline gate, and the verification utilities all read or write.
Before changing a filename here, grep the codebase to confirm every
caller has been updated.
"""

from __future__ import annotations


class FileNames:
    """Production filenames for OpenMM MD outputs.

    Use as ``output_dir / FileNames.TRAJECTORY``, etc. The verdict
    filename uses ``{ns}ns`` formatting for the sub-second resolution
    the offline gate records (e.g. ``gate_verdict_5.2ns.json``).
    """

    TRAJECTORY = "trajectory.dcd"
    ENERGY = "energy.csv"
    STATE_XML = "state.xml"
    TOPOLOGY = "topology.pdb"
    CHECKPOINT_JSON = "checkpoint.json"
    SYSTEM_CONFIG_JSON = "system_config.json"
    MD_SUMMARY_JSON = "md_summary.json"
    EARLY_ABORT_JSON = "early_abort.json"
    EQUILIBRATION_METADATA_JSON = "equilibration_metadata.json"
    MD_RESULT_JSON = "md_result.json"
    GATE_VERDICT_GLOB = "gate_verdict_*ns.json"
    GATE_VERDICT_PREFIX = "gate_verdict_"
    GATE_VERDICT_SUFFIX = "ns.json"

    # The set of files that together indicate a complete production run.
    # verify_production_outputs() checks each of these in utils.py.
    PRODUCTION_OUTPUT_FILES: tuple[str, ...] = (
        TRAJECTORY,
        ENERGY,
        STATE_XML,
    )

    # Default fallback names used by _resolve_pdb when the explicit
    # config path doesn't exist on disk.
    RECEPTOR_PDB = "receptor.pdb"
    PEPTIDE_PDB = "peptide.pdb"
