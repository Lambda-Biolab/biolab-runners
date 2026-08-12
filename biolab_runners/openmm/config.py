"""Configuration models for OpenMM MD simulations.

This module owns the engine-specific dataclass that the OpenMM
runner consumes. The canonical, engine-neutral producer is
:mod:`bioml_tools.md.system_spec` (``MDSpec``, slice 12). The
:class:`OpenMMConfig` here is the **engine-specific runtime**
view: every field in :class:`OpenMMConfig` is either (a) a
projection of an :class:`MDSpec` field onto the dataclass or
(b) an OpenMM-only overlay (``platform``, ``extra_forcefields``,
``water_ff_xml``, ``target_irmsd_threshold_a``).

The :meth:`OpenMMConfig.from_md_spec` classmethod is the canonical
construction path going forward; the legacy ``__init__`` and the
physiological / saliva / gastric / intestinal preset classmethods
remain in place for backward compatibility with serialised
``system_config.json`` files written before slice 12.

The :meth:`OpenMMConfig.from_json` loader still parses the legacy
flat wire format so existing in-flight simulations and golden
fixtures keep working.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.openmm.paths import FileNames

if TYPE_CHECKING:
    from bioml_tools.md.system_spec import MDSpec

logger = logging.getLogger(__name__)

# Default ionic conditions (physiological PBS-like; override via presets or explicit args)
DEFAULT_NACL_M = 0.150
DEFAULT_PH = 7.4

# Default early-abort iRMSD threshold (Å). Override per-system via
# OpenMMConfig.target_irmsd_threshold_a. 3.5 Å is a common mid-range value
# used in peptide-protein binding stability assessment.
DEFAULT_IRMSD_THRESHOLD_A = 3.5

# Simulation parameters
TEMPERATURE_K = 310.0  # Body temperature (37 C)
PRESSURE_ATM = 1.0
TIMESTEP_FS = 2.0
BOX_PADDING_NM = 0.8  # 8 A padding
BOX_SHAPE = "dodecahedron"  # ~29% less solvent than cubic

# Force fields
PROTEIN_FF = "charmm36m"
WATER_MODEL = "tip3p"

# GPU platform — pip OpenMM has OpenCL (not CUDA); conda OpenMM has both
OPENMM_PLATFORM = "OpenCL"

# Equilibration protocol (3 stages, 100 + 100 + 200 = 400 ps total).
# The runner hardcodes the k-values for each stage; only the durations
# (in ps) are exposed as constants so the total-equil-steps calculation
# in __post_init__ stays in sync with the runner's behavior.
EQUIL_NVT_PS = 100.0
EQUIL_NPT_RESTRAINED_PS = 100.0
EQUIL_NPT_FREE_PS = 200.0
EQUIL_TOTAL_PS = EQUIL_NVT_PS + EQUIL_NPT_RESTRAINED_PS + EQUIL_NPT_FREE_PS


@dataclass
class OpenMMConfig:
    """Complete configuration for an OpenMM MD simulation.

    Defaults are physiological PBS-like (150 mM NaCl, pH 7.4, 310 K). For
    other environments use the preset classmethods (``physiological``,
    ``saliva``, ``gastric``, ``intestinal``) or override fields directly.

    Attributes:
        receptor_pdb: Path to receptor PDB file.
        peptide_pdb: Path to peptide PDB file.
        output_dir: Output directory for simulation files.
        target: Optional free-form tag used to group jobs and appear in
            result metadata. Not interpreted by the runner.
        peptide_id: Peptide identifier.
        target_irmsd_threshold_a: Reference iRMSD (Å) for the peptide used
            by the early-abort logic. Runner aborts at 5 ns if peptide Cα
            RMSD > 2× this value. Override per-system for tighter/looser
            gating.
        nacl_mol: NaCl concentration in mol/L. This is the only ionic
            concentration the runner actually passes to OpenMM (the
            ``addSolvent(ionicStrength=…)`` call takes a single value).
            Other ion species (Ca2+, K+, etc.) are not yet modeled.
        temperature_k: Temperature in Kelvin.
        pressure_atm: Pressure in atmospheres.
        timestep_fs: Integration timestep in femtoseconds.
        box_padding_nm: Solvent box padding in nanometers.
        box_shape: Solvent box shape ("dodecahedron" or "cubic").
        protein_ff: Protein force field name.
        water_model: Water model name.
        extra_forcefields: Additional OpenMM force-field XML files to load
            alongside ``protein_ff`` and ``water_model``. Each entry is the
            path (str) to an XML file accepted by ``openmm.app.ForceField``.
            Use for non-canonical residues, small molecules, or user-supplied
            parameter overrides. Entries are appended in order; later files
            take precedence for overlapping atom types. Defaults to the empty
            list.
        openmm_platform: OpenMM platform ("OpenCL", "CUDA", "CPU").
        production_ns: Production simulation length in nanoseconds.
        save_interval_ps: Trajectory save interval in picoseconds.
        checkpoint_interval_hours: Checkpoint save interval in hours.
        protonation_ph: pH for hydrogen addition.
        total_steps: Computed total production steps.
        total_equil_steps: Computed total equilibration steps (constant
            derived from the 3-stage protocol, 400 ps).
        save_every_steps: Computed trajectory save step interval.
        checkpoint_every_steps: Computed checkpoint step interval.
    """

    receptor_pdb: str = ""
    peptide_pdb: str = ""
    output_dir: str = ""
    target: str = ""
    peptide_id: str = ""

    # Ionic conditions (NaCl only — see class docstring)
    nacl_mol: float = DEFAULT_NACL_M

    # Simulation parameters
    temperature_k: float = TEMPERATURE_K
    pressure_atm: float = PRESSURE_ATM
    timestep_fs: float = TIMESTEP_FS
    box_padding_nm: float = BOX_PADDING_NM
    box_shape: str = BOX_SHAPE

    # Force fields
    protein_ff: str = PROTEIN_FF
    water_model: str = WATER_MODEL
    # Water + ions XML path for ``app.ForceField``. Separate from
    # ``water_model`` because ``Modeller.addSolvent(model=…)`` takes a
    # SHORT key like ``"tip3p"`` / ``"tip4pew"`` whereas ``ForceField``
    # needs an XML filename. Bare ``tip3p.xml`` ships water-only
    # parameters and ``addSolvent`` raises "No template found for
    # residue N (NA)" once Na+/Cl- ions are inserted at the
    # configured ionic strength. Point this at e.g.
    # ``"amber14/tip3p.xml"`` for a water+ions bundle, or leave empty
    # and biolab-runners will fall back to ``{water_model}.xml``
    # (appropriate for CHARMM where the ion templates ship with the
    # protein XML). OralBiome-AMP's Aib preprocessing sets this via
    # ``force_fields.water_ff_xml`` — see ``augment_system_config_for_aib``.
    water_ff_xml: str = ""
    extra_forcefields: list[str] = field(default_factory=list)

    # GPU platform
    openmm_platform: str = OPENMM_PLATFORM

    # Production
    production_ns: float = 100.0
    save_interval_ps: float = 10.0
    checkpoint_interval_hours: float = 2.0

    # Protonation pH
    protonation_ph: float = DEFAULT_PH

    # Early-abort reference threshold (Å). See DEFAULT_IRMSD_THRESHOLD_A docstring.
    target_irmsd_threshold_a: float = DEFAULT_IRMSD_THRESHOLD_A

    # Computed fields
    total_steps: int = 0
    total_equil_steps: int = 0
    save_every_steps: int = 0
    checkpoint_every_steps: int = 0

    def __post_init__(self) -> None:
        """Compute derived step counts."""
        steps_per_ps = 1000.0 / self.timestep_fs
        self.total_steps = int(self.production_ns * 1000.0 * steps_per_ps)
        self.total_equil_steps = int(EQUIL_TOTAL_PS * steps_per_ps)
        self.save_every_steps = int(self.save_interval_ps * steps_per_ps)
        self.checkpoint_every_steps = int(
            self.checkpoint_interval_hours * 3600.0 * 1000.0 / self.timestep_fs
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        return {
            "receptor_pdb": self.receptor_pdb,
            "peptide_pdb": self.peptide_pdb,
            "output_dir": self.output_dir,
            "target": self.target,
            "peptide_id": self.peptide_id,
            "ionic_conditions": {
                "NaCl_M": self.nacl_mol,
            },
            "simulation": {
                "temperature_K": self.temperature_k,
                "pressure_atm": self.pressure_atm,
                "timestep_fs": self.timestep_fs,
                "box_padding_nm": self.box_padding_nm,
                "box_shape": self.box_shape,
                "production_ns": self.production_ns,
                "total_steps": self.total_steps,
                "save_interval_ps": self.save_interval_ps,
                "save_every_steps": self.save_every_steps,
                "checkpoint_interval_hours": self.checkpoint_interval_hours,
                "checkpoint_every_steps": self.checkpoint_every_steps,
            },
            "force_fields": {
                "protein": self.protein_ff,
                "water": self.water_model,
                "water_ff_xml": self.water_ff_xml,
                "extra": list(self.extra_forcefields),
            },
            "openmm_platform": self.openmm_platform,
            "protonation_ph": self.protonation_ph,
            "target_irmsd_threshold_a": self.target_irmsd_threshold_a,
        }

    def save(self, path: Path | None = None) -> Path:
        """Save configuration to JSON.

        Args:
            path: Output path. Defaults to output_dir/system_config.json.

        Returns:
            Path to saved file.
        """
        if path is None:
            path = Path(self.output_dir) / FileNames.SYSTEM_CONFIG_JSON
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("System config saved to %s", path)
        return path

    @classmethod
    def from_json(cls, path: Path) -> OpenMMConfig:
        """Load configuration from JSON.

        Args:
            path: Path to config JSON.

        Returns:
            OpenMMConfig instance.
        """
        data = json.loads(path.read_text())
        sim = data.get("simulation", {})
        ions = data.get("ionic_conditions", {})
        ff = data.get("force_fields", {})

        return cls(
            receptor_pdb=data.get("receptor_pdb", ""),
            peptide_pdb=data.get("peptide_pdb", ""),
            output_dir=data.get("output_dir", ""),
            target=data.get("target", ""),
            peptide_id=data.get("peptide_id", ""),
            nacl_mol=ions.get("NaCl_M", DEFAULT_NACL_M),
            temperature_k=sim.get("temperature_K", TEMPERATURE_K),
            production_ns=sim.get("production_ns", 100.0),
            save_interval_ps=sim.get("save_interval_ps", 10.0),
            checkpoint_interval_hours=sim.get("checkpoint_interval_hours", 2.0),
            protein_ff=ff.get("protein", PROTEIN_FF),
            water_model=ff.get("water", WATER_MODEL),
            water_ff_xml=ff.get("water_ff_xml", ""),
            extra_forcefields=list(ff.get("extra", []) or []),
            protonation_ph=data.get("protonation_ph", DEFAULT_PH),
            target_irmsd_threshold_a=float(
                data.get("target_irmsd_threshold_a", DEFAULT_IRMSD_THRESHOLD_A)
            ),
        )

    @classmethod
    def saliva(cls, **overrides: Any) -> OpenMMConfig:  # noqa: ANN401
        """Saliva-like buffer: 140 mM NaCl, pH 6.2, 310 K.

        Literature reference for unstimulated whole saliva reports
        140 mM NaCl + 1.4 mM CaCl2 + 0.5 mM KH2PO4 at pH 6.2. The
        runner currently models only NaCl ionic strength — the
        Ca²⁺ and KH₂PO₄ contributions are documented here as
        unmodelled context, not applied to
        ``addSolvent(ionicStrength=…)``. Multi-ion modeling is
        future work.
        """
        return cls(
            **_preset(
                nacl_mol=0.140,
                temperature_k=310.0,
                protonation_ph=6.2,
                overrides=overrides,
            )
        )

    @classmethod
    def physiological(cls, **overrides: Any) -> OpenMMConfig:  # noqa: ANN401
        """Physiological buffer (PBS / plasma-like): 150 mM NaCl, pH 7.4, 310 K."""
        return cls(
            **_preset(
                nacl_mol=0.150,
                temperature_k=310.0,
                protonation_ph=7.4,
                overrides=overrides,
            )
        )

    @classmethod
    def gastric(cls, **overrides: Any) -> OpenMMConfig:  # noqa: ANN401
        """Gastric fluid: 150 mM NaCl, pH 2.0, 310 K.

        Note: very low pH affects protonation of His/Asp/Glu/N-termini. Verify
        that the selected protein force field handles this regime.
        """
        return cls(
            **_preset(
                nacl_mol=0.150,
                temperature_k=310.0,
                protonation_ph=2.0,
                overrides=overrides,
            )
        )

    @classmethod
    def intestinal(cls, **overrides: Any) -> OpenMMConfig:  # noqa: ANN401
        """Small-intestinal fluid: 150 mM NaCl, pH 6.8, 310 K."""
        return cls(
            **_preset(
                nacl_mol=0.150,
                temperature_k=310.0,
                protonation_ph=6.8,
                overrides=overrides,
            )
        )

    @classmethod
    def from_md_spec(
        cls,
        spec: MDSpec,
        **engine_overrides: Any,  # noqa: ANN401
    ) -> OpenMMConfig:
        """Build an :class:`OpenMMConfig` from a canonical :class:`MDSpec`.

        Slice 12 (MD-OPENMM-001) makes :class:`MDSpec` (from
        :mod:`bioml_tools.md.system_spec`) the canonical producer of
        MD configuration. This classmethod is the canonical
        ``OpenMMConfig`` construction path going forward — it projects
        every engine-neutral :class:`MDSpec` field onto the matching
        :class:`OpenMMConfig` field and adds the OpenMM-specific
        runtime overlay fields with their defaults.

        Engine-specific overlays (``openmm_platform``,
        ``extra_forcefields``, ``water_ff_xml``,
        ``target_irmsd_threshold_a``) take their default values; pass
        any subset via ``engine_overrides`` to override.

        Args:
            spec: Canonical engine-neutral MD spec from bioml-tools.
            **engine_overrides: OpenMM-only fields (see OpenMMConfig
                docs). Caller overrides win; everything else comes
                from the :class:`MDSpec`.

        Returns:
            An :class:`OpenMMConfig` ready for ``prepare_simulation``.

        Raises:
            TypeError: if an unknown engine-specific field is in
                ``engine_overrides``. Catching this at the construction
                boundary keeps a typo (``production_NS`` vs
                ``production_ns``) from silently falling through.

        Notes:
            ``nacl_mol`` on the dataclass corresponds to
            ``spec.ionic_strength_m`` on the spec. ``box_shape``
            comes through as the enum value (string); the rest are
            identical-name projections.
        """
        allowed_overrides = frozenset(
            {
                "openmm_platform",
                "water_ff_xml",
                "extra_forcefields",
                "target_irmsd_threshold_a",
            }
        )
        unknown = set(engine_overrides) - allowed_overrides
        if unknown:
            raise TypeError(
                f"unknown engine-specific override(s): {sorted(unknown)}. "
                f"Allowed: {sorted(allowed_overrides)}. Engine-neutral "
                "fields live on MDSpec — change them via spec, not via "
                "from_md_spec."
            )

        return cls(
            # Per-instance paths / identifiers
            receptor_pdb=spec.receptor_pdb,
            peptide_pdb=spec.peptide_pdb,
            output_dir=spec.output_dir,
            target=spec.target,
            peptide_id=spec.peptide_id,
            # Ionic
            nacl_mol=spec.ionic_strength_m,
            # Simulation
            temperature_k=spec.temperature_k,
            pressure_atm=spec.pressure_atm,
            timestep_fs=spec.timestep_fs,
            box_padding_nm=spec.box_padding_nm,
            box_shape=spec.box_shape.value,
            # Force fields
            protein_ff=spec.protein_ff,
            water_model=spec.water_model,
            # Production cadence (defaults preserved; caller overrides via engine_overrides)
            production_ns=spec.production_ns,
            save_interval_ps=spec.save_interval_ps,
            checkpoint_interval_hours=spec.checkpoint_interval_hours,
            # Protonation
            protonation_ph=spec.protonation_ph,
            **engine_overrides,
        )


def _preset(
    *,
    nacl_mol: float,
    temperature_k: float,
    protonation_ph: float,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge preset values with caller overrides. Caller overrides win."""
    values: dict[str, Any] = {
        "nacl_mol": nacl_mol,
        "temperature_k": temperature_k,
        "protonation_ph": protonation_ph,
    }
    values.update(overrides)
    return values


@dataclass
class SimulationResult:
    """Output from an OpenMM MD simulation.

    Attributes:
        config: The configuration used for this simulation.
        trajectory_path: Path to the DCD trajectory file.
        energy_path: Path to the energy CSV file.
        state_xml_path: Path to the final state XML checkpoint.
        topology_path: Path to the solvated system topology PDB.
        total_ns: Actual simulation time completed (ns).
        elapsed_seconds: Wall-clock time for the simulation.
        ns_per_day: Performance metric (ns of simulation per wall-clock day).
        num_atoms: Number of atoms in the solvated system.
        early_abort: True if the simulation was terminated early.
        abort_reason: Reason for early termination (if any).
        error: Error message if simulation failed.
    """

    config: OpenMMConfig
    trajectory_path: str = ""
    energy_path: str = ""
    state_xml_path: str = ""
    topology_path: str = ""
    total_ns: float = 0.0
    elapsed_seconds: float = 0.0
    ns_per_day: float = 0.0
    num_atoms: int = 0
    early_abort: bool = False
    abort_reason: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "target": self.config.target,
            "peptide_id": self.config.peptide_id,
            "trajectory_path": self.trajectory_path,
            "energy_path": self.energy_path,
            "state_xml_path": self.state_xml_path,
            "topology_path": self.topology_path,
            # 6 decimals = sub-fs precision in ns, needed for the
            # 1-ps smoke test in tests/integration/. round-to-2
            # silently dropped 0.001 ns (1 ps) to 0.0.
            "total_ns": round(self.total_ns, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "ns_per_day": round(self.ns_per_day, 1),
            "num_atoms": self.num_atoms,
            "early_abort": self.early_abort,
            "abort_reason": self.abort_reason,
            "error": self.error,
        }

    def save(self, path: Path | None = None) -> Path:
        """Save simulation result to JSON.

        Args:
            path: Output path. Defaults to output_dir/md_result.json.

        Returns:
            Path to saved file.
        """
        out_dir = Path(self.config.output_dir)
        if path is None:
            path = out_dir / "md_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path
