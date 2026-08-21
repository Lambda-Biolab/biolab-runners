"""Configuration for GROMACS MD runs.

Two configurations live here:

- :class:`GromacsConfig` — the **legacy one-shot mdrun** config. Used
  by callers that already have a pre-built ``.tpr`` and just want to
  run ``gmx mdrun`` once. Preserved verbatim from the S3 implementation
  so existing callers (and the existing test suite) continue to work.

- :class:`GromacsProtocolConfig` — the **S4 production-grade protocol**
  config. Drives a full pipeline: topology → box → solvate → ions →
  minimization → NVT → NPT → production. Each MD stage is checkpoint-
  resumable via ``-cpi`` (and appends to existing outputs via
  ``-append`` when resuming). Replicas are supported by varying
  ``replica_index`` (the runner writes outputs into
  ``work_dir / f"{name}_rep{replica_index}"``).

The dataclass fields are validated in ``__post_init__`` — both
configs raise ``ValueError`` with a specific message on bad input
so the runner fails fast at the construction boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["GromacsConfig", "GromacsProtocolConfig"]


def _empty_str_tuple() -> tuple[str, ...]:
    return ()


def _empty_dict() -> Mapping[str, Any]:
    return {}


@dataclass(frozen=True)
class GromacsConfig:
    """Per-invocation configuration for a legacy one-shot GROMACS MD run."""

    name: str = "gromacs-md"
    structure_file: str = ""
    topology_file: str = ""
    output_dir: str = ""
    tpr_basename: str = "topol"
    nsteps: int = 1000000  # 2 ns at 2 fs
    integrator: str = "md"
    temperature: float = 310.0
    pressure: float = 1.0
    timestep_fs: float = 2.0
    extra_mdrun_flags: tuple[str, ...] = field(default_factory=_empty_str_tuple)
    extra: Mapping[str, Any] = field(default_factory=_empty_dict)

    def __post_init__(self) -> None:
        """Validate required paths and parameter ranges."""
        if not self.structure_file:
            raise ValueError("GromacsConfig.structure_file is required")
        if not self.topology_file:
            raise ValueError("GromacsConfig.topology_file is required")
        if self.nsteps < 1:
            raise ValueError(f"nsteps must be >= 1; got {self.nsteps}")
        if self.timestep_fs <= 0:
            raise ValueError(f"timestep_fs must be positive; got {self.timestep_fs}")


@dataclass(frozen=True)
class GromacsProtocolConfig:
    """Configuration for the production-grade GROMACS protocol.

    Drives the full pipeline from a single PDB input to a
    checkpoint-resumable production run. All defaults match the
    S4 spec (1.0 nm solute-to-edge buffer, TIP3P, 0.15 M ions,
    100 ps NVT, 100 ps NPT, 200 ns production, steepest-descent
    cap 50 000).

    Attributes:
        name: Logical run name. Used as the subdirectory under
            ``output_root`` (so multiple named runs can coexist).
        input_pdb: Legacy ``gmx pdb2gmx`` source PDB. Required when
            ``prebuilt_topology`` and ``prebuilt_coordinates`` are
            not both set; ignored in prebuilt mode.
        output_root: Root directory for run outputs. Each run
            writes to ``output_root / name / f"rep{replica_index}"``
            when ``replicas_total > 1``, otherwise
            ``output_root / name``.
        force_field: Force field passed to ``gmx pdb2gmx -ff``.

            **The default (``"charmm36m"``) does NOT imply that
            charmm36m is available locally.** GROMACS does not bundle
            force field parameters; the operator must install the
            desired force field via ``GMXDATA`` (or the upstream
            ``gmx pdb2gmx -download`` workflow for the AMBER/CHARMM/
            OPLS references). The default is a sensible convention,
            not an availability claim; if the operator has not
            installed charmm36m, ``gmx pdb2gmx`` will fail with a
            "Force field … not found" error.

            Callers can verify a force field is available via
            ``gmx pdb2gmx -h | grep <force_field>`` before
            constructing the config, or by passing a force field
            the operator has explicitly installed.
        water_model: Water model passed to ``gmx pdb2gmx -water``
            (must be consistent with ``force_field`` and with the
            ``spc216.gro`` coordinate set used by ``gmx solvate``;
            ``"tip3p"`` is the canonical match).
        box_buffer_nm: **Distance from the solute to the cubic
            box edge** (the buffer), in nm. This is the value
            passed to ``gmx editconf -d``. Default 1.0 nm — the
            canonical GROMACS convention for solvated protein
            systems. Note: this is NOT the box edge length itself
            (which would be ``protein_extent + 2 * buffer``).
        box_size_nm: **DEPRECATED alias** for ``box_buffer_nm``,
            kept for backward compatibility with S3-era callers
            that used the misleading name. New code should use
            ``box_buffer_nm`` directly. If both are set,
            ``box_size_nm`` wins (the deprecated name takes
            precedence to honor any caller that explicitly set
            it).
        ion_concentration_m: NaCl concentration in mol/L
            (default 0.15, the physiological 150 mM value).
        minimization_max_iterations: Steepest-descent cap
            (default 50 000 per S4).
        nvt_ps: NVT equilibration length in ps (default 100).
        npt_ps: NPT equilibration length in ps (default 100).
        production_ns: Production length in ns (default 200).
            Override for screening (``production_ns=0.5`` is a
            common fast-screen value).
        screening_ns: Optional override for ``production_ns``
            that takes precedence when set (e.g. ``5.0`` ns
            for a smoke screening run). When ``None`` the
            ``production_ns`` field is used.
        temperature_k: Target temperature in K (default 310,
            physiological).
        pressure_bar: Target pressure in bar (default 1.0,
            atmospheric).
        nt_threads: OpenMP threads for ``gmx mdrun -nt``.
            **Default 0 → the runner OMITS ``-nt`` entirely** so
            GROMACS auto-detects the host's thread count. Set
            to a positive integer (e.g. 1) to force a specific
            thread count for single-threaded reproducibility.
        replica_index: 0-indexed replica number when running
            replicates (default 0). The runner writes outputs
            under a per-replica subdirectory when
            ``replicas_total > 1`` AND drives the ``gen-seed``
            in each .mdp to ``replica_index + 1`` so each
            replica produces a deterministic-but-different
            trajectory.
        replicas_total: Total number of replicas for this name
            (default 1, no per-replica subdirectory). Used by
            callers that want deterministic replica addressing.
        extra_mdrun_flags: Extra flags appended to every
            ``gmx mdrun`` invocation (e.g. ``("-nb", "gpu")``).
        timeout_seconds: Per-stage subprocess timeout (default
            86 400 s = 24 h, matching the legacy one-shot runner).
        force: Re-run stages that already completed (default
            ``False``). Equivalent to ``runner.run(force=True)``
            in the legacy path.
    """

    name: str = "gromacs-protocol"
    input_pdb: str = ""
    output_root: str = ""
    force_field: str = "charmm36m"
    water_model: str = "tip3p"
    box_buffer_nm: float = 1.0
    box_size_nm: float | None = None
    ion_concentration_m: float = 0.15
    minimization_max_iterations: int = 50_000
    nvt_ps: int = 100
    npt_ps: int = 100
    production_ns: float = 200.0
    screening_ns: float | None = None
    temperature_k: float = 310.0
    pressure_bar: float = 1.0
    nt_threads: int = 0
    replica_index: int = 0
    replicas_total: int = 1
    extra_mdrun_flags: tuple[str, ...] = field(default_factory=_empty_str_tuple)
    timeout_seconds: int = 86_400
    force: bool = False
    # When a structure was prepared upstream (biolab_runners.peptide_prep)
    # and a GROMACS ``.top`` / ``.gro`` were produced from the
    # SAME OpenMM system, the GROMACS protocol can consume the
    # prebuilt topology + coordinates and skip the
    # ``pdb2gmx`` / force-field-assignment step. The
    # ``box_buffer_nm`` / solvation / ions / equilibration /
    # production stages are unchanged; only the topology stage
    # is bypassed (the prebuilt topology carries the bonded +
    # non-bonded parameters + atom charges from the upstream
    # OpenMM ForceField).
    #
    # Both ``prebuilt_topology`` and ``prebuilt_coordinates``
    # must be supplied, OR neither (the legacy ``input_pdb``
    # path is unchanged). The runner validates this constraint
    # and fails fast at construction.
    prebuilt_topology: str = ""
    prebuilt_coordinates: str = ""

    def __post_init__(self) -> None:
        """Validate required paths and parameter ranges."""
        # Backward-compat: ``box_size_nm`` is a deprecated alias for
        # ``box_buffer_nm``. If both are explicitly set, ``box_size_nm``
        # wins (so a caller that explicitly set the old name sees
        # the value they passed). We must use ``object.__setattr__``
        # because the dataclass is frozen.
        if self.box_size_nm is not None:
            object.__setattr__(self, "box_buffer_nm", self.box_size_nm)
        self._validate_paths()
        self._validate_geometry()
        self._validate_chemistry()
        self._validate_protocol_durations()
        self._validate_thermodynamics()
        self._validate_replicas()
        self._validate_prebuilt()

    def _validate_paths(self) -> None:
        """Validate the path-bearing required fields.

        ``input_pdb`` is required only in legacy mode, when both
        prebuilt paths are empty. In prebuilt mode it is ignored:
        the runner skips ``pdb2gmx`` and stages
        ``prebuilt_coordinates`` as ``processed.gro``. ``BOX`` uses
        that file, and the resulting ``boxed.gro`` feeds ``SOLVATE``.
        The prebuilt ``.top`` remains the topology source for
        downstream GROMACS stages.
        """
        if not self.name:
            raise ValueError("GromacsProtocolConfig.name is required")
        if not self.output_root:
            raise ValueError("GromacsProtocolConfig.output_root is required")
        if not self.input_pdb and not self.prebuilt_topology:
            raise ValueError(
                "GromacsProtocolConfig.input_pdb is required "
                "(or supply prebuilt_topology + prebuilt_coordinates)"
            )

    def _validate_prebuilt(self) -> None:
        """Validate the prebuilt-topology path pair (``both or neither``).

        The prebuilt path is additive / backward-compat: when
        BOTH ``prebuilt_topology`` and ``prebuilt_coordinates``
        are empty, the legacy ``input_pdb`` path is used
        unchanged (legacy tests / fixtures still pass). When
        both are non-empty, the runner skips the
        ``pdb2gmx`` topology stage and stages the prebuilt
        files into the canonical ``topol.top`` /
        ``processed.gro`` filenames at the start of the
        protocol. ``input_pdb`` is ignored in prebuilt mode;
        ``processed.gro`` feeds the BOX stage, and its
        ``boxed.gro`` output feeds SOLVATE, while ``topol.top``
        supplies the topology.
        """
        has_top = bool(self.prebuilt_topology)
        has_xyz = bool(self.prebuilt_coordinates)
        if has_top != has_xyz:
            raise ValueError(
                "prebuilt_topology and prebuilt_coordinates must both be set "
                "or both be empty; got "
                f"prebuilt_topology={self.prebuilt_topology!r}, "
                f"prebuilt_coordinates={self.prebuilt_coordinates!r}"
            )

    def _validate_geometry(self) -> None:
        """Validate the solvation-box geometry (buffer distance)."""
        if self.box_buffer_nm <= 0:
            raise ValueError(f"box_buffer_nm must be positive; got {self.box_buffer_nm}")

    def _validate_chemistry(self) -> None:
        """Validate the chemistry knobs (ions, minimization cap)."""
        if self.ion_concentration_m < 0:
            raise ValueError(
                f"ion_concentration_m must be non-negative; got {self.ion_concentration_m}"
            )
        if self.minimization_max_iterations < 1:
            err = (
                f"minimization_max_iterations must be positive; "
                f"got {self.minimization_max_iterations}"
            )
            raise ValueError(err)

    def _validate_protocol_durations(self) -> None:
        """Validate the per-stage durations (NVT, NPT, production, screening)."""
        if self.nvt_ps < 1:
            raise ValueError(f"nvt_ps must be positive; got {self.nvt_ps}")
        if self.npt_ps < 1:
            raise ValueError(f"npt_ps must be positive; got {self.npt_ps}")
        if self.production_ns <= 0:
            raise ValueError(f"production_ns must be positive; got {self.production_ns}")
        if self.screening_ns is not None and self.screening_ns <= 0:
            raise ValueError(f"screening_ns must be positive; got {self.screening_ns}")

    def _validate_thermodynamics(self) -> None:
        """Validate temperature and pressure."""
        if self.temperature_k <= 0:
            raise ValueError(f"temperature_k must be positive; got {self.temperature_k}")
        if self.pressure_bar <= 0:
            raise ValueError(f"pressure_bar must be positive; got {self.pressure_bar}")

    def _validate_replicas(self) -> None:
        """Validate the replica-index / replicas-total pair."""
        if self.replica_index < 0:
            raise ValueError(f"replica_index must be non-negative; got {self.replica_index}")
        if self.replicas_total < 1:
            raise ValueError(f"replicas_total must be >= 1; got {self.replicas_total}")
        if self.replica_index >= self.replicas_total:
            raise ValueError(
                f"replica_index ({self.replica_index}) must be < "
                f"replicas_total ({self.replicas_total})"
            )

    def effective_production_ns(self) -> float:
        """Return the production length actually used.

        ``screening_ns`` takes precedence when set; the runner
        uses this to switch between a fast-screen protocol and
        the full 200 ns production without rebuilding the config.
        """
        if self.screening_ns is not None:
            return self.screening_ns
        return self.production_ns
