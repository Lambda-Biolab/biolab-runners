"""Configuration models for OpenMM MD simulations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default ionic conditions (saliva-like; override with presets below or explicit args)
DEFAULT_NACL_M = 0.140
DEFAULT_CACL2_M = 0.0014
DEFAULT_KH2PO4_M = 0.0005
DEFAULT_PH = 6.2

# Simulation parameters
TEMPERATURE_K = 310.0  # Oral cavity temperature (37 C)
PRESSURE_ATM = 1.0
TIMESTEP_FS = 2.0
BOX_PADDING_NM = 0.8  # 8 A padding
BOX_SHAPE = "dodecahedron"  # ~29% less solvent than cubic

# Force fields
PROTEIN_FF = "charmm36m"
WATER_MODEL = "tip3p"
LIGAND_FF = "cgenff"

# GPU platform — pip OpenMM has OpenCL (not CUDA); conda OpenMM has both
OPENMM_PLATFORM = "OpenCL"

# Default equilibration protocol
DEFAULT_EQUIL_STAGES = [
    {"name": "NVT", "ensemble": "NVT", "duration_ps": 100, "restraint_k": 1000.0},
    {
        "name": "NPT_restrained",
        "ensemble": "NPT",
        "duration_ps": 100,
        "restraint_k": 100.0,
    },
    {"name": "NPT_free", "ensemble": "NPT", "duration_ps": 200, "restraint_k": 0.0},
]


@dataclass(frozen=True)
class EquilibrationStage:
    """One stage of the equilibration protocol.

    Attributes:
        name: Human-readable name (e.g. "NVT", "NPT_restrained").
        ensemble: Thermodynamic ensemble ("NVT" or "NPT").
        duration_ps: Duration in picoseconds.
        restraint_k: Backbone restraint force constant (kJ/mol/nm^2).
            0 means no restraints.
    """

    name: str
    ensemble: str
    duration_ps: float
    restraint_k: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "ensemble": self.ensemble,
            "duration_ps": self.duration_ps,
            "restraint_k": self.restraint_k,
        }


@dataclass
class OpenMMConfig:
    """Complete configuration for an OpenMM MD simulation.

    Defaults are saliva-like (140 mM NaCl, pH 6.2) to preserve backward
    compatibility. For other environments use the preset classmethods
    (``physiological``, ``gastric``, ``intestinal``, ``saliva``) or override
    fields directly.

    Attributes:
        receptor_pdb: Path to receptor PDB file.
        peptide_pdb: Path to peptide PDB file.
        output_dir: Output directory for simulation files.
        target: Target protein name (e.g. "GtfB").
        peptide_id: Peptide identifier.
        nacl_mol: NaCl concentration in mol/L.
        cacl2_mol: CaCl2 concentration in mol/L.
        kh2po4_mol: KH2PO4 concentration in mol/L.
        temperature_k: Temperature in Kelvin.
        pressure_atm: Pressure in atmospheres.
        timestep_fs: Integration timestep in femtoseconds.
        box_padding_nm: Solvent box padding in nanometers.
        box_shape: Solvent box shape ("dodecahedron" or "cubic").
        protein_ff: Protein force field name.
        water_model: Water model name.
        ligand_ff: Ligand force field name.
        openmm_platform: OpenMM platform ("OpenCL", "CUDA", "CPU").
        equilibration: List of equilibration stage dicts.
        production_ns: Production simulation length in nanoseconds.
        save_interval_ps: Trajectory save interval in picoseconds.
        checkpoint_interval_hours: Checkpoint save interval in hours.
        protonation_ph: pH for hydrogen addition.
        total_steps: Computed total production steps.
        save_every_steps: Computed trajectory save step interval.
        checkpoint_every_steps: Computed checkpoint step interval.
        solvated_atoms: Number of atoms in solvated system (0 = unknown).
    """

    receptor_pdb: str = ""
    peptide_pdb: str = ""
    output_dir: str = ""
    target: str = ""
    peptide_id: str = ""

    # Ionic conditions
    nacl_mol: float = DEFAULT_NACL_M
    cacl2_mol: float = DEFAULT_CACL2_M
    kh2po4_mol: float = DEFAULT_KH2PO4_M

    # Simulation parameters
    temperature_k: float = TEMPERATURE_K
    pressure_atm: float = PRESSURE_ATM
    timestep_fs: float = TIMESTEP_FS
    box_padding_nm: float = BOX_PADDING_NM
    box_shape: str = BOX_SHAPE

    # Force fields
    protein_ff: str = PROTEIN_FF
    water_model: str = WATER_MODEL
    ligand_ff: str = LIGAND_FF

    # GPU platform
    openmm_platform: str = OPENMM_PLATFORM

    # Equilibration
    equilibration: list[dict[str, object]] = field(
        default_factory=lambda: list(DEFAULT_EQUIL_STAGES)
    )

    # Production
    production_ns: float = 100.0
    save_interval_ps: float = 10.0
    checkpoint_interval_hours: float = 2.0

    # Protonation pH
    protonation_ph: float = DEFAULT_PH

    # Computed fields
    total_steps: int = 0
    save_every_steps: int = 0
    checkpoint_every_steps: int = 0

    # Solvated system size (set after topology building, 0 = unknown)
    solvated_atoms: int = 0

    def __post_init__(self) -> None:
        """Compute derived step counts."""
        steps_per_ps = 1000.0 / self.timestep_fs
        self.total_steps = int(self.production_ns * 1000.0 * steps_per_ps)
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
                "CaCl2_M": self.cacl2_mol,
                "KH2PO4_M": self.kh2po4_mol,
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
                "ligand": self.ligand_ff,
            },
            "openmm_platform": self.openmm_platform,
            "protonation_ph": self.protonation_ph,
            "equilibration": self.equilibration,
            "solvated_atoms": self.solvated_atoms,
        }

    def save(self, path: Path | None = None) -> Path:
        """Save configuration to JSON.

        Args:
            path: Output path. Defaults to output_dir/system_config.json.

        Returns:
            Path to saved file.
        """
        if path is None:
            path = Path(self.output_dir) / "system_config.json"
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
            cacl2_mol=ions.get("CaCl2_M", DEFAULT_CACL2_M),
            kh2po4_mol=ions.get("KH2PO4_M", DEFAULT_KH2PO4_M),
            temperature_k=sim.get("temperature_K", TEMPERATURE_K),
            production_ns=sim.get("production_ns", 100.0),
            save_interval_ps=sim.get("save_interval_ps", 10.0),
            checkpoint_interval_hours=sim.get("checkpoint_interval_hours", 2.0),
            protein_ff=ff.get("protein", PROTEIN_FF),
            water_model=ff.get("water", WATER_MODEL),
            ligand_ff=ff.get("ligand", LIGAND_FF),
            protonation_ph=data.get("protonation_ph", DEFAULT_PH),
            solvated_atoms=int(data.get("solvated_atoms", 0)),
        )

    @classmethod
    def saliva(cls, **overrides: Any) -> OpenMMConfig:
        """Saliva-like buffer: 140 mM NaCl + 1.4 mM CaCl2 + 0.5 mM KH2PO4, pH 6.2, 310 K.

        Literature reference values for unstimulated whole saliva. Matches the
        OralBiome-AMP pipeline defaults.
        """
        return cls(
            **_preset(
                nacl_mol=0.140,
                cacl2_mol=0.0014,
                kh2po4_mol=0.0005,
                temperature_k=310.0,
                protonation_ph=6.2,
                overrides=overrides,
            )
        )

    @classmethod
    def physiological(cls, **overrides: Any) -> OpenMMConfig:
        """Physiological buffer (PBS / plasma-like): 150 mM NaCl, pH 7.4, 310 K."""
        return cls(
            **_preset(
                nacl_mol=0.150,
                cacl2_mol=0.0,
                kh2po4_mol=0.0,
                temperature_k=310.0,
                protonation_ph=7.4,
                overrides=overrides,
            )
        )

    @classmethod
    def gastric(cls, **overrides: Any) -> OpenMMConfig:
        """Gastric fluid: 150 mM NaCl, pH 2.0, 310 K.

        Note: very low pH affects protonation of His/Asp/Glu/N-termini. Verify
        that the selected protein force field handles this regime.
        """
        return cls(
            **_preset(
                nacl_mol=0.150,
                cacl2_mol=0.0,
                kh2po4_mol=0.0,
                temperature_k=310.0,
                protonation_ph=2.0,
                overrides=overrides,
            )
        )

    @classmethod
    def intestinal(cls, **overrides: Any) -> OpenMMConfig:
        """Small-intestinal fluid: 150 mM NaCl, pH 6.8, 310 K."""
        return cls(
            **_preset(
                nacl_mol=0.150,
                cacl2_mol=0.0,
                kh2po4_mol=0.0,
                temperature_k=310.0,
                protonation_ph=6.8,
                overrides=overrides,
            )
        )


def _preset(
    *,
    nacl_mol: float,
    cacl2_mol: float,
    kh2po4_mol: float,
    temperature_k: float,
    protonation_ph: float,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge preset values with caller overrides. Caller overrides win."""
    values: dict[str, Any] = {
        "nacl_mol": nacl_mol,
        "cacl2_mol": cacl2_mol,
        "kh2po4_mol": kh2po4_mol,
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
            "total_ns": round(self.total_ns, 2),
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
