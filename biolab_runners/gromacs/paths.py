"""Centralized filenames for the GROMACS protocol.

Single source of truth for the production filenames that the protocol,
the runner, and the manifest I/O all read or write. Before changing a
filename here, grep the codebase to confirm every caller has been
updated.

Stage conventions follow the upstream GROMACS deffnm pattern:
each stage writes ``<prefix>.<ext>`` files into the working
directory, where the prefix is the canonical stage name
(``min``, ``nvt``, ``npt``, ``prod``). The deffnm-based names keep
``gmx mdrun -deffnm <prefix>`` output contiguous across resumes —
a partial run writes ``<prefix>.cpt`` to the same directory and
the next invocation reads it via ``-cpi <prefix>.cpt``.
"""

from __future__ import annotations


class GromacsFiles:
    """Production filenames for the GROMACS protocol.

    Use as ``work_dir / GromacsFiles.STAGE_MANIFEST``, etc. The
    stage prefixes (``MIN_PREFIX``, ``NVT_PREFIX`` ...) are the
    canonical ``-deffnm`` values the runner passes to ``gmx mdrun``.
    """

    # Stage prefixes (canonical -deffnm values for gmx mdrun).
    MIN_PREFIX = "min"
    NVT_PREFIX = "nvt"
    NPT_PREFIX = "npt"
    PROD_PREFIX = "prod"

    # Stage manifest — the structured record of which stages have
    # completed. The runner reads this to decide whether to skip or
    # resume a stage. Lives at the root of the work directory so it
    # survives a Spot interruption alongside the .cpt files.
    STAGE_MANIFEST = "gromacs_stages.json"

    # Topology artifacts written by ``gmx pdb2gmx`` and ``gmx editconf``.
    TOPOLOGY_TOP = "topol.top"
    TOPOLOGY_GRO = "processed.gro"
    BOX_GRO = "boxed.gro"
    SOLVATED_GRO = "solvated.gro"
    IONIZED_TPR = "ions.tpr"
    IONIZED_GRO = "ions.gro"

    # .mdp filename pattern for each stage (input parameters). The
    # runner writes these to disk before invoking grompp; deterministic
    # content is asserted by tests/test_gromacs_protocol.py.
    IONS_MDP = "ions.mdp"
    MIN_MDP = "min.mdp"
    NVT_MDP = "nvt.mdp"
    NPT_MDP = "npt.mdp"
    PROD_MDP = "prod.mdp"

    # Per-stage canonical filenames. All start with the stage prefix
    # so a ``ls work_dir/min*`` reveals the full minimisation state.
    # The ``_out`` suffix matches the GROMACS convention
    # (``-deffnm min`` writes ``min.gro``, ``min.cpt``, etc.).
    @staticmethod
    def stage_outputs(prefix: str) -> tuple[str, ...]:
        """Return the canonical output suffixes for a stage ``-deffnm``."""
        return (
            f"{prefix}.tpr",
            f"{prefix}.gro",
            f"{prefix}.cpt",
            f"{prefix}.edr",
            f"{prefix}.log",
            f"{prefix}.xtc",
            f"{prefix}.trr",
        )

    # GROMACS .cpt filename passed to ``-cpi``. Same name as the
    # ``-deffnm`` writes — the runner reads ``<prefix>.cpt`` if it
    # exists. The runner only adds ``-cpi <prefix>.cpt`` when this
    # file is present on disk (no duplicate start path).
    @staticmethod
    def checkpoint(prefix: str) -> str:
        """Return the checkpoint filename for a stage prefix."""
        return f"{prefix}.cpt"

    @staticmethod
    def mdp_path(prefix: str) -> str:
        """Return the .mdp filename for a stage prefix."""
        return f"{prefix}.mdp"
