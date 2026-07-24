"""Tests for ``biolab_runners.openmm.system_builder``.

Focus: pure functions and OpenMM-mockable entry points. The
``prepare_simulation`` orchestrator is integration-tested via the
smoke_test/run_smoke.py driver (requires real OpenMM + GPU).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from biolab_runners.openmm.config import OpenMMConfig
from biolab_runners.openmm.system_builder import (
    _resolve_pdb,
    add_ca_restraint,
    assemble_system,
    build_forcefield,
    write_topology,
)

# ---------------------------------------------------------------------------
# _resolve_pdb — pure filesystem
# ---------------------------------------------------------------------------


class TestResolvePdb:
    """Resolve a PDB path with fallback to output_dir / cwd."""

    def test_returns_explicit_path_when_exists(self, tmp_path: Path) -> None:
        explicit = tmp_path / "rec.pdb"
        explicit.write_text("HEADER\n")
        assert _resolve_pdb(str(explicit), "receptor.pdb", tmp_path) == str(explicit)

    def test_returns_empty_when_neither_exists(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        result = _resolve_pdb("", "receptor.pdb", out)
        assert result == ""

    def test_falls_back_to_output_dir_parent(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        fallback = tmp_path / "receptor.pdb"  # tmp_path is output_dir.parent
        fallback.write_text("HEADER\n")
        assert _resolve_pdb("", "receptor.pdb", out) == str(fallback)

    def test_falls_back_to_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        fallback = out / "receptor.pdb"
        fallback.write_text("HEADER\n")
        assert _resolve_pdb("", "receptor.pdb", out) == str(fallback)

    def test_explicit_path_wins_even_if_fallback_exists(self, tmp_path: Path) -> None:
        """If the explicit path exists, the fallback is never consulted."""
        out = tmp_path / "out"
        out.mkdir()
        (out / "receptor.pdb").write_text("FALLBACK\n")
        explicit = tmp_path / "explicit.pdb"
        explicit.write_text("EXPLICIT\n")
        assert _resolve_pdb(str(explicit), "receptor.pdb", out) == str(explicit)


# ---------------------------------------------------------------------------
# add_ca_restraint — mockable (CustomExternalForce)
# ---------------------------------------------------------------------------


class _FakeAtom:
    def __init__(self, name: str, index: int) -> None:
        self.name = name
        self.index = index


class _FakePos:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _FakeChain:
    def __init__(self, atoms: list[_FakeAtom]) -> None:
        self._atoms = atoms

    def atoms(self) -> list[_FakeAtom]:
        return self._atoms


class _FakeSystem:
    def __init__(self) -> None:
        self.forces: list[MagicMock] = []

    def addForce(self, force: MagicMock) -> None:
        self.forces.append(force)


class _FakeOpenMM:
    """Minimal stand-in for the openmm module — only CustomExternalForce."""

    def __init__(self) -> None:
        self.last_force: MagicMock | None = None

    def CustomExternalForce(self, expr: str) -> MagicMock:
        force = MagicMock()
        force._expr = expr
        self.last_force = force
        return force


class TestAddCaRestraint:
    """Add a Cα CustomExternalForce (k=0) to the system."""

    def test_collects_only_ca_indices(self) -> None:
        """Only Cα atoms are added to the restraint."""
        chains = [
            _FakeChain(
                [
                    _FakeAtom("N", 0),
                    _FakeAtom("CA", 1),
                    _FakeAtom("C", 2),
                    _FakeAtom("CA", 3),
                    _FakeAtom("O", 4),
                ]
            ),
        ]
        modeller = MagicMock()
        modeller.positions = [
            _FakePos(0.0, 0.0, 0.0),
            _FakePos(1.0, 0.0, 0.0),
            _FakePos(2.0, 0.0, 0.0),
            _FakePos(3.0, 0.0, 0.0),
            _FakePos(4.0, 0.0, 0.0),
        ]
        system = _FakeSystem()
        openmm = _FakeOpenMM()

        _force, ca_indices = add_ca_restraint(system, modeller, chains, openmm)  # type: ignore[arg-type]

        assert ca_indices == [1, 3]

    def test_force_added_to_system(self) -> None:
        chains = [_FakeChain([_FakeAtom("CA", 0)])]
        modeller = MagicMock()
        modeller.positions = [_FakePos(0.0, 0.0, 0.0)]
        system = _FakeSystem()
        openmm = _FakeOpenMM()

        _force, _ca = add_ca_restraint(system, modeller, chains, openmm)  # type: ignore[arg-type]

        assert len(system.forces) == 1
        assert system.forces[0] is openmm.last_force

    def test_force_expression_is_k_periodicdistance_squared(self) -> None:
        """The expression must be 'k*periodicdistance(...)^2' for the
        restraint to actually constrain when k > 0 (used in equilibration
        stages 1-3)."""
        chains = [_FakeChain([_FakeAtom("CA", 0)])]
        modeller = MagicMock()
        modeller.positions = [_FakePos(0.0, 0.0, 0.0)]
        system = _FakeSystem()
        openmm = _FakeOpenMM()

        _force, _ca = add_ca_restraint(system, modeller, chains, openmm)  # type: ignore[arg-type]

        assert openmm.last_force is not None
        assert openmm.last_force._expr == "k*periodicdistance(x,y,z,x0,y0,z0)^2"

    def test_initial_k_is_zero(self) -> None:
        """k=0 means the force has no effect; restraint is engaged later
        by simulation.context.setParameter('k', ...)."""
        chains = [_FakeChain([_FakeAtom("CA", 0)])]
        modeller = MagicMock()
        modeller.positions = [_FakePos(0.0, 0.0, 0.0)]
        system = _FakeSystem()
        openmm = _FakeOpenMM()

        _force, _ca = add_ca_restraint(system, modeller, chains, openmm)  # type: ignore[arg-type]

        # addGlobalParameter is called with ("k", 0.0)
        openmm.last_force.addGlobalParameter.assert_called_once_with("k", 0.0)

    def test_particles_added_with_position(self) -> None:
        """Each Cα particle is added with its (x0, y0, z0) reference position."""
        chains = [_FakeChain([_FakeAtom("CA", 0), _FakeAtom("CA", 2)])]
        modeller = MagicMock()
        modeller.positions = [
            _FakePos(1.0, 2.0, 3.0),
            _FakePos(0.0, 0.0, 0.0),  # not a CA
            _FakePos(4.0, 5.0, 6.0),
        ]
        system = _FakeSystem()
        openmm = _FakeOpenMM()

        _force, _ca = add_ca_restraint(system, modeller, chains, openmm)  # type: ignore[arg-type]

        # addPerParticleParameter is called 3 times (x0, y0, z0)
        assert openmm.last_force.addPerParticleParameter.call_count == 3
        # addParticle called once per CA
        assert openmm.last_force.addParticle.call_count == 2
        # First particle: index 0, position (1, 2, 3)
        openmm.last_force.addParticle.assert_any_call(0, [1.0, 2.0, 3.0])
        openmm.last_force.addParticle.assert_any_call(2, [4.0, 5.0, 6.0])

    def test_no_ca_atoms(self) -> None:
        """Edge case: a chain with no Cα atoms → empty ca_indices, no particles added."""
        chains = [_FakeChain([_FakeAtom("N", 0), _FakeAtom("C", 1)])]
        modeller = MagicMock()
        modeller.positions = [_FakePos(0.0, 0.0, 0.0), _FakePos(1.0, 0.0, 0.0)]
        system = _FakeSystem()
        openmm = _FakeOpenMM()

        _force, ca_indices = add_ca_restraint(system, modeller, chains, openmm)  # type: ignore[arg-type]

        assert ca_indices == []
        openmm.last_force.addParticle.assert_not_called()


# ---------------------------------------------------------------------------
# assemble_system — mockable (MonteCarloBarostat + LangevinMiddleIntegrator)
# ---------------------------------------------------------------------------


class _FakeOpenMMForAssemble:
    """Mock that records the barostat + integrator construction calls."""

    def __init__(self) -> None:
        self.barostat = MagicMock()
        self.integrator = MagicMock()

    def MonteCarloBarostat(self, pressure, temperature, interval) -> MagicMock:
        self._barostat_args = (pressure, temperature, interval)
        return self.barostat

    def LangevinMiddleIntegrator(self, temperature, friction, timestep) -> MagicMock:
        self._integrator_args = (temperature, friction, timestep)
        return self.integrator


class _FakeForceField:
    def __init__(self) -> None:
        self.system = MagicMock()
        self.system.addForce = MagicMock()

    def createSystem(self, topology, **kwargs) -> MagicMock:
        self._topology = topology
        self._kwargs = kwargs
        return self.system


class _RecordingForceField:
    """Captures the XML paths passed to app.ForceField(...)."""

    def __init__(self, *paths: str) -> None:
        self.paths = paths


class _FakeApp:
    """Stand-in for openmm.app — exposes only the symbols system_builder touches."""

    def __init__(self) -> None:
        self.PME = "PME"
        self.HBonds = "HBonds"
        self.ForceField = _RecordingForceField
        self.PDBFile: MagicMock = MagicMock()


class _FakeQuantity:
    """A quantity that supports ``float * quantity`` and ``float / quantity``."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __rmul__(self, other: float) -> _FakeQuantity:
        return _FakeQuantity(f"{other} {self.label}")

    def __rtruediv__(self, other: float) -> _FakeQuantity:
        return _FakeQuantity(f"{other}/{self.label}")

    def __str__(self) -> str:
        return self.label

    __repr__ = __str__


class _FakeUnit:
    """Stand-in for openmm.unit — quantities stringify for assertions."""

    def __init__(self) -> None:
        self.nanometers = _FakeQuantity("nm")
        self.atmospheres = _FakeQuantity("atm")
        self.kelvin = _FakeQuantity("K")
        self.picoseconds = _FakeQuantity("ps")
        self.femtoseconds = _FakeQuantity("fs")


class TestAssembleSystem:
    """Build the System (PME barostat) and LangevinMiddleIntegrator."""

    def test_uses_pme_with_1nm_cutoff_and_hbonds_constraint(self) -> None:
        ff = _FakeForceField()
        topology = MagicMock()
        config = OpenMMConfig(
            protein_ff="amber14/protein.ff14SB",
            water_model="tip3p",
            pressure_atm=1.0,
            temperature_k=310.0,
        )
        openmm = _FakeOpenMMForAssemble()
        app = _FakeApp()
        unit = _FakeUnit()

        system, integrator = assemble_system(
            ff,
            topology,
            config,
            openmm,
            app,
            unit,  # type: ignore[arg-type]
        )

        assert system is ff.system
        assert integrator is openmm.integrator
        # PME + 1.0 nm + HBonds
        assert ff._kwargs["nonbondedMethod"] == "PME"
        assert str(ff._kwargs["nonbondedCutoff"]) == "1.0 nm"
        assert ff._kwargs["constraints"] == "HBonds"

    def test_barostat_uses_config_pressure_and_temperature(self) -> None:
        ff = _FakeForceField()
        config = OpenMMConfig(
            protein_ff="amber14/protein.ff14SB",
            pressure_atm=1.5,
            temperature_k=300.0,
        )
        openmm = _FakeOpenMMForAssemble()
        app = _FakeApp()
        unit = _FakeUnit()

        _system, _integrator = assemble_system(
            ff,
            MagicMock(),
            config,
            openmm,
            app,
            unit,  # type: ignore[arg-type]
        )

        # Barostat constructed with config.pressure * unit.atmospheres,
        # config.temperature * unit.kelvin, interval 25.
        pressure, temperature, interval = openmm._barostat_args
        assert str(pressure) == "1.5 atm"
        assert str(temperature) == "300.0 K"
        assert interval == 25

    def test_integrator_uses_langevin_middle(self) -> None:
        ff = _FakeForceField()
        config = OpenMMConfig(
            protein_ff="amber14/protein.ff14SB",
            temperature_k=310.0,
            timestep_fs=2.0,
        )
        openmm = _FakeOpenMMForAssemble()
        app = _FakeApp()
        unit = _FakeUnit()

        _system, _integrator = assemble_system(
            ff,
            MagicMock(),
            config,
            openmm,
            app,
            unit,  # type: ignore[arg-type]
        )

        # friction = 1/ps, timestep = config.timestep_fs * fs
        temperature, friction, timestep = openmm._integrator_args
        assert str(temperature) == "310.0 K"
        assert str(friction) == "1.0/ps"
        assert str(timestep) == "2.0 fs"

    def test_barostat_added_to_system(self) -> None:
        """The barostat must be added to the system so NPT dynamics work."""
        ff = _FakeForceField()
        config = OpenMMConfig(protein_ff="amber14/protein.ff14SB")
        openmm = _FakeOpenMMForAssemble()
        app = _FakeApp()
        unit = _FakeUnit()

        _system, _integrator = assemble_system(
            ff,
            MagicMock(),
            config,
            openmm,
            app,
            unit,  # type: ignore[arg-type]
        )

        ff.system.addForce.assert_called_once_with(openmm.barostat)


# ---------------------------------------------------------------------------
# write_topology — mockable (PDBFile.writeFile)
# ---------------------------------------------------------------------------


class _FakePDBFile:
    """Stand-in for openmm.app.PDBFile — exposes ``writeFile`` as a static-ish call."""

    writeFile = MagicMock()


class _FakeAppForWrite(_FakeApp):
    """_FakeApp subclass that wires PDBFile.writeFile for inspection."""

    def __init__(self) -> None:
        super().__init__()
        self.PDBFile = _FakePDBFile()


class TestWriteTopology:
    """Persist the solvated topology PDB and populate result metadata."""

    def test_writes_pdb_to_output_dir(self, tmp_path: Path) -> None:
        app = _FakeAppForWrite()
        modeller = MagicMock()
        modeller.topology.getNumAtoms.return_value = 12345
        result = MagicMock()
        result.num_atoms = 0
        result.topology_path = ""

        write_topology(modeller, tmp_path, app, result)  # type: ignore[arg-type]

        topo_path = tmp_path / "topology.pdb"
        assert topo_path.exists()
        app.PDBFile.writeFile.assert_called_once()
        assert result.num_atoms == 12345
        assert result.topology_path == str(topo_path)

    def test_topology_path_set_even_if_file_already_exists(self, tmp_path: Path) -> None:
        """Result is updated regardless of pre-existing file."""
        app = _FakeAppForWrite()
        modeller = MagicMock()
        modeller.topology.getNumAtoms.return_value = 100
        result = MagicMock()
        result.num_atoms = 0
        result.topology_path = ""

        write_topology(modeller, tmp_path, app, result)  # type: ignore[arg-type]

        assert result.topology_path == str(tmp_path / "topology.pdb")


# ---------------------------------------------------------------------------
# build_forcefield — re-exported test (already in test_openmm_runner.py)
# ---------------------------------------------------------------------------


class TestBuildForcefieldReExported:
    """build_forcefield from system_builder is the same function that
    was on OpenMMRunner._build_forcefield; the existing TestBuildForcefield
    in test_openmm_runner.py covers it. This test guards against future
    regression: the function must remain importable from system_builder."""

    def test_importable_from_system_builder(self) -> None:
        assert callable(build_forcefield)

    def test_charmm_uses_hardcoded_xmls(self) -> None:
        config = OpenMMConfig(protein_ff="charmm36m")
        ff = build_forcefield(config, _FakeApp())  # type: ignore[arg-type]
        assert ff.paths == ("charmm36.xml", "charmm36/water.xml")

    def test_amber_uses_water_model_or_overrides(self) -> None:
        config = OpenMMConfig(
            protein_ff="amber14/protein.ff14SB",
            water_model="tip3p",
        )
        ff = build_forcefield(config, _FakeApp())  # type: ignore[arg-type]
        assert ff.paths == ("amber14/protein.ff14SB.xml", "tip3p.xml")
