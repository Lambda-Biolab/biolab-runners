"""Tests for ``biolab_runners.openmm.system_builder``.

Focus: pure functions and OpenMM-mockable entry points. The
``prepare_simulation`` orchestrator is integration-tested via the
smoke_test/run_smoke.py driver (requires real OpenMM + GPU).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest  # noqa: TC002  # used in helper annotations only
from biolab_runners.openmm.config import OpenMMConfig, SimulationResult
from biolab_runners.openmm.system_builder import (
    SimulationContext,
    add_ca_restraint,
    assemble_system,
    resolve_pdb,
    write_topology,
)

from tests._helpers import (
    FakeApp,
    FakeAtom,
    FakeChain,
    FakePos,
    FakeSystem,
    FakeUnit,
)

# ---------------------------------------------------------------------------
# resolve_pdb — pure filesystem
# ---------------------------------------------------------------------------


class TestResolvePdb:
    """Resolve a PDB path with fallback to output_dir / cwd."""

    def test_returns_explicit_path_when_exists(self, tmp_path: Path) -> None:
        explicit = tmp_path / "rec.pdb"
        explicit.write_text("HEADER\n")
        assert resolve_pdb(str(explicit), "receptor.pdb", tmp_path) == str(explicit)

    def test_returns_empty_when_neither_exists(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        result = resolve_pdb("", "receptor.pdb", out)
        assert result == ""

    def test_falls_back_to_output_dir_parent(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        fallback = tmp_path / "receptor.pdb"  # tmp_path is output_dir.parent
        fallback.write_text("HEADER\n")
        assert resolve_pdb("", "receptor.pdb", out) == str(fallback)

    def test_falls_back_to_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        fallback = out / "receptor.pdb"
        fallback.write_text("HEADER\n")
        assert resolve_pdb("", "receptor.pdb", out) == str(fallback)

    def test_explicit_path_wins_even_if_fallback_exists(self, tmp_path: Path) -> None:
        """If the explicit path exists, the fallback is never consulted."""
        out = tmp_path / "out"
        out.mkdir()
        (out / "receptor.pdb").write_text("FALLBACK\n")
        explicit = tmp_path / "explicit.pdb"
        explicit.write_text("EXPLICIT\n")
        assert resolve_pdb(str(explicit), "receptor.pdb", out) == str(explicit)


# ---------------------------------------------------------------------------
# add_ca_restraint — mockable (CustomExternalForce)
# ---------------------------------------------------------------------------


class _FakeOpenMM:
    """Minimal stand-in for the openmm module — only CustomExternalForce.

    This is system_builder-specific (the shared tests._helpers module does
    not include it because the system_builder test exercises the
    CustomExternalForce call directly).
    """

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
            FakeChain(
                [
                    FakeAtom("N", 0),
                    FakeAtom("CA", 1),
                    FakeAtom("C", 2),
                    FakeAtom("CA", 3),
                    FakeAtom("O", 4),
                ]
            ),
        ]
        modeller = MagicMock()
        modeller.positions = [
            FakePos(0.0, 0.0, 0.0),
            FakePos(1.0, 0.0, 0.0),
            FakePos(2.0, 0.0, 0.0),
            FakePos(3.0, 0.0, 0.0),
            FakePos(4.0, 0.0, 0.0),
        ]
        system = FakeSystem()
        openmm = _FakeOpenMM()

        _force, ca_indices = add_ca_restraint(system, modeller, chains, openmm)  # type: ignore[arg-type]

        assert ca_indices == [1, 3]

    def test_force_added_to_system(self) -> None:
        chains = [FakeChain([FakeAtom("CA", 0)])]
        modeller = MagicMock()
        modeller.positions = [FakePos(0.0, 0.0, 0.0)]
        system = FakeSystem()
        openmm = _FakeOpenMM()

        _force, _ca = add_ca_restraint(system, modeller, chains, openmm)  # type: ignore[arg-type]

        assert len(system.forces) == 1
        assert system.forces[0] is openmm.last_force

    def test_force_expression_is_k_periodicdistance_squared(self) -> None:
        """The expression must be 'k*periodicdistance(...)^2' for the
        restraint to actually constrain when k > 0 (used in equilibration
        stages 1-3)."""
        chains = [FakeChain([FakeAtom("CA", 0)])]
        modeller = MagicMock()
        modeller.positions = [FakePos(0.0, 0.0, 0.0)]
        system = FakeSystem()
        openmm = _FakeOpenMM()

        _force, _ca = add_ca_restraint(system, modeller, chains, openmm)  # type: ignore[arg-type]

        assert openmm.last_force is not None
        assert openmm.last_force._expr == "k*periodicdistance(x,y,z,x0,y0,z0)^2"

    def test_initial_k_is_zero(self) -> None:
        """k=0 means the force has no effect; restraint is engaged later
        by simulation.context.setParameter('k', ...)."""
        chains = [FakeChain([FakeAtom("CA", 0)])]
        modeller = MagicMock()
        modeller.positions = [FakePos(0.0, 0.0, 0.0)]
        system = FakeSystem()
        openmm = _FakeOpenMM()

        _force, _ca = add_ca_restraint(system, modeller, chains, openmm)  # type: ignore[arg-type]

        # addGlobalParameter is called with ("k", 0.0)
        openmm.last_force.addGlobalParameter.assert_called_once_with("k", 0.0)

    def test_particles_added_with_position(self) -> None:
        """Each Cα particle is added with its (x0, y0, z0) reference position."""
        chains = [FakeChain([FakeAtom("CA", 0), FakeAtom("CA", 2)])]
        modeller = MagicMock()
        modeller.positions = [
            FakePos(1.0, 2.0, 3.0),
            FakePos(0.0, 0.0, 0.0),  # not a CA
            FakePos(4.0, 5.0, 6.0),
        ]
        system = FakeSystem()
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
        chains = [FakeChain([FakeAtom("N", 0), FakeAtom("C", 1)])]
        modeller = MagicMock()
        modeller.positions = [FakePos(0.0, 0.0, 0.0), FakePos(1.0, 0.0, 0.0)]
        system = FakeSystem()
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
    """Stand-in for openmm.app.ForceField — captures createSystem args + addForce."""

    def __init__(self) -> None:
        self.system = MagicMock()
        self.system.addForce = MagicMock()

    def createSystem(self, topology, **kwargs) -> MagicMock:
        self._topology = topology
        self._kwargs = kwargs
        return self.system


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
        app = FakeApp()
        unit = FakeUnit()

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
        app = FakeApp()
        unit = FakeUnit()

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
        app = FakeApp()
        unit = FakeUnit()

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
        app = FakeApp()
        unit = FakeUnit()

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


class FakeAppForWrite(FakeApp):
    """FakeApp subclass that wires PDBFile.writeFile for inspection."""

    def __init__(self) -> None:
        super().__init__()
        # Replace the default FakePDBFile (which is a no-op) with a
        # MagicMock so tests can assert it was called.
        self.PDBFile = MagicMock()


class TestWriteTopology:
    """Persist the solvated topology PDB and populate result metadata."""

    def test_writes_pdb_to_output_dir(self, tmp_path: Path) -> None:
        app = FakeAppForWrite()
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
        app = FakeAppForWrite()
        modeller = MagicMock()
        modeller.topology.getNumAtoms.return_value = 100
        result = MagicMock()
        result.num_atoms = 0
        result.topology_path = ""

        write_topology(modeller, tmp_path, app, result)  # type: ignore[arg-type]

        assert result.topology_path == str(tmp_path / "topology.pdb")


# ---------------------------------------------------------------------------
# SimulationContext — guard against field removal (review finding)
# ---------------------------------------------------------------------------


class TestSimulationContextFields:
    """Pin the public surface of ``SimulationContext``.

    The runner's ``_run_equilibration`` and ``_check_post_equilibration
    _displacement`` consume ``ctx.chains``; removing it from the
    dataclass would silently break every fresh (non-resume) run. The
    smoke test would catch this in integration, but the unit-test
    suite doesn't reach that path. These tests guard the contract
    in isolation so a future cleanup can't remove a consumed field
    again.
    """

    def test_chains_field_present(self) -> None:
        """``chains`` must be a field on SimulationContext (defaulted to [])."""
        from dataclasses import fields

        field_names = {f.name for f in fields(SimulationContext)}
        assert "chains" in field_names

    def test_chains_default_is_empty_list(self) -> None:
        """Default factory is list (each context gets its own list, no shared state)."""
        ctx_a = SimulationContext()
        ctx_b = SimulationContext()
        assert ctx_a.chains == []
        assert ctx_b.chains == []
        # Mutating one must not affect the other (separate list instances).
        ctx_a.chains.append("sentinel")
        assert ctx_b.chains == []


class TestPrepareSimulationPopulatesContext:
    """Drive ``prepare_simulation`` end-to-end with mocks and assert the
    returned ``SimulationContext`` carries the chains list.

    The Phase 8 cleanup accidentally removed ``chains`` from
    ``SimulationContext`` based on a "derivable" recommendation,
    but the runner actually consumes ``ctx.chains`` during
    equilibration. The unit suite never reached that path so the
    regression slipped through. This test makes the production
    contract a unit-test invariant: any future removal of the
    ``chains=chains`` constructor argument in ``prepare_simulation``
    would fail this test.
    """

    def test_fresh_run_populates_chains_from_modeller(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fresh (non-resume) prepare_simulation must populate ctx.chains
        from the modeller topology. A missing ``chains=chains`` constructor
        argument would silently leave ctx.chains == [], which downstream
        _run_equilibration would then fail to walk.
        """
        import biolab_runners.openmm.system_builder as sb

        sentinel_chains = [object(), object(), object()]
        fake_modeller = _make_fake_modeller(sentinel_chains, num_atoms=42)
        _stub_openmm(monkeypatch)
        _stub_system_builder(monkeypatch, sb, fake_modeller)

        config = OpenMMConfig(
            receptor_pdb="receptor.pdb",
            peptide_pdb="peptide.pdb",
            output_dir=str(tmp_path),
            openmm_platform="CPU",
        )
        result = SimulationResult(config=config)
        ctx = sb.prepare_simulation(config, tmp_path, "", result)

        assert ctx is not None, f"prepare_simulation failed: {result.error}"
        # The contract: ctx.chains must contain the modeller topology's
        # chain list (list() in prepare_simulation produces a new list,
        # so we compare by element identity, not the list itself).
        assert ctx.chains == sentinel_chains
        for sentinel, populated in zip(sentinel_chains, ctx.chains, strict=True):
            assert sentinel is populated
        # Also verify the metadata fields the BLOCKER #2 fix addressed.
        assert result.num_atoms == 42
        assert result.topology_path == str(tmp_path / "topology.pdb")


def _make_fake_modeller(chains: list[object], num_atoms: int) -> object:
    """Build a minimal OpenMM Modeller stand-in for the prepare_simulation test."""

    class _FakeTopology:
        def chains(self) -> list[object]:
            return chains

        def getNumAtoms(self) -> int:
            return num_atoms

    class _FakeModeller:
        topology = _FakeTopology()
        positions: tuple[object, ...] = ()

    return _FakeModeller()


def _stub_openmm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate ``sys.modules`` with fake openmm / openmm.app / openmm.unit.

    ``prepare_simulation`` does lazy ``import openmm`` /
    ``import openmm.app as app`` / ``import openmm.unit as unit`` inside
    its body, plus ``import numpy as np``. Pre-populating ``sys.modules``
    with ``types.ModuleType`` instances (which carry a ``__path__``) lets
    the ``from X import Y`` form resolve.
    """
    fake_modules = {
        "openmm": _make_fake_openmm_module(),
        "openmm.app": _make_fake_app_module(),
        "openmm.unit": _make_fake_unit_module(),
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _make_fake_openmm_module() -> types.ModuleType:
    """Stand-in for the ``openmm`` package — Platform, CustomExternalForce, etc."""

    class _FakePlatform:
        @staticmethod
        def getPlatformByName(name: str) -> _FakePlatform:
            p = _FakePlatform()
            p._name = name
            return p

        def getName(self) -> str:
            return self._name  # type: ignore[attr-defined]

        def setPropertyDefaultValue(self, key: str, value: str) -> None:
            pass

    class _CustomExternalForce:
        def __init__(self, expr: str) -> None:
            pass

        def addGlobalParameter(self, *a: object, **kw: object) -> None:
            pass

        def addPerParticleParameter(self, *a: object, **kw: object) -> None:
            pass

        def addParticle(self, *a: object, **kw: object) -> None:
            pass

    class _MonteCarloBarostat:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    class _LangevinMiddleIntegrator:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    class _FakeOpenMMModule(types.ModuleType):
        Platform = _FakePlatform
        CustomExternalForce = _CustomExternalForce
        MonteCarloBarostat = _MonteCarloBarostat
        LangevinMiddleIntegrator = _LangevinMiddleIntegrator

    return _FakeOpenMMModule("openmm")


def _make_fake_app_module() -> types.ModuleType:
    """Stand-in for the ``openmm.app`` module — PDBFile, Simulation, etc."""

    class _FakeForceField:
        def __init__(self, *paths: str, **kwargs: object) -> None:
            self.paths = paths

    class _FakePDBFile:
        @staticmethod
        def writeFile(*args: object, **kwargs: object) -> None:
            pass

    class _FakeSimulation:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.context = type("_Ctx", (), {"setPositions": lambda self, p: None})()

    class _FakeAppModule(types.ModuleType):
        PME = "PME"
        HBonds = "HBonds"
        ForceField = _FakeForceField
        PDBFile = _FakePDBFile
        Simulation = _FakeSimulation

    return _FakeAppModule("openmm.app")


def _make_fake_unit_module() -> types.ModuleType:
    """Stand-in for ``openmm.unit`` — no symbols used by prepare_simulation."""
    return types.ModuleType("openmm.unit")


def _stub_system_builder(
    monkeypatch: pytest.MonkeyPatch, sb: object, fake_modeller: object
) -> None:
    """Stub the heavy collaborators of ``prepare_simulation`` so the
    test doesn't need real OpenMM or pdbfixer."""
    monkeypatch.setattr(sb, "build_forcefield", lambda config, app: object())
    monkeypatch.setattr(sb, "build_or_load_modeller", lambda *args, **kw: fake_modeller)
    monkeypatch.setattr(sb, "assemble_system", lambda *a, **kw: (MagicMock(), MagicMock()))
    monkeypatch.setattr(sb, "add_ca_restraint", lambda *a, **kw: (MagicMock(), []))
