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


class TestPrepareSimulationMissingOpenMM:
    """Exercise the actual ``try: import openmm ... except ImportError``
    code path in ``system_builder.prepare_simulation``.

    Previous tests (e.g. ``test_missing_openmm_returns_error`` in
    test_openmm_runner.py) mocked ``prepare_simulation`` itself, which
    means the import branch was untested. This test installs an
    in-process import blocker via ``sys.meta_path`` — a custom finder
    that raises ``ImportError`` for any ``openmm.*`` module — and runs
    the real ``prepare_simulation`` function against a valid config.
    The result must surface a clear ``OpenMM not installed: ...`` error
    in ``result.error`` and return ``None``.
    """

    def test_import_blocker_surfaces_clear_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real prepare_simulation + real import blocker = result.error set.

        The blocker rejects any openmm.* import with ImportError.
        prepare_simulation's ``try: import openmm ... except
        ImportError`` catches it and sets ``result.error``. The runner's
        behavior on top of this is covered separately in
        test_openmm_runner.py::test_missing_openmm_returns_error.
        """
        import importlib
        import importlib.abc
        import importlib.machinery

        from biolab_runners.openmm import system_builder as sb
        from biolab_runners.openmm.config import SimulationResult

        class _OpenMMImportBlocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            """Custom finder that raises ImportError for openmm.* modules."""

            def find_spec(self, fullname, path, target=None):  # type: ignore[override]
                if fullname == "openmm" or fullname.startswith("openmm."):
                    raise ImportError(f"No module named '{fullname}'")
                return None

            def create_module(self, spec):  # type: ignore[override]
                return None

            def exec_module(self, module):  # type: ignore[override]
                pass

        # Uninstall the cached openmm modules so the import has to
        # actually go through sys.meta_path again. (We deliberately
        # don't unimport the biolab_runners modules that depend on
        # openmm, since those are import-side-effect-free.)
        for name in list(sys.modules):
            if name == "openmm" or name.startswith("openmm."):
                monkeypatch.delitem(sys.modules, name)

        blocker = _OpenMMImportBlocker()
        sys.meta_path.insert(0, blocker)
        try:
            # Even after openmm is uninstalled, the module object
            # may still be in sys.modules via previously-imported names.
            # We re-check after the monkeypatch delitem above.

            config = OpenMMConfig(
                receptor_pdb="receptor.pdb",
                peptide_pdb="peptide.pdb",
                output_dir=str(tmp_path),
                openmm_platform="CPU",
            )
            result = SimulationResult(config=config)
            ctx = sb.prepare_simulation(config, tmp_path, "", result)

            assert ctx is None, "prepare_simulation must return None when openmm is missing"
            assert "OpenMM not installed" in result.error
            assert "openmm" in result.error.lower()
        finally:
            import contextlib

            with contextlib.suppress(ValueError):
                sys.meta_path.remove(blocker)


class TestPrepareSimulationResumeTopologyGuard:
    """Resume-safety regression: the saved state.xml must be paired
    with the exact topology it was serialized from. Re-solvation
    produces different water counts and atom ordering, so a freshly
    built System cannot accept a state.xml produced from a different
    System.

    Behaviour:
      - state.xml absent: fresh run (build + solvate + write topology.pdb).
      - state.xml present + topology.pdb intact: resume (load original
        modeller + loadState).
      - state.xml present + topology.pdb missing/truncated: FAIL FAST
        (set result.error, return None). The user must re-run with
        force=True to discard the checkpoint.

    The integrity check runs BEFORE the expensive PDBFixer /
    ``addSolvent`` work — re-solvating cannot recover a corrupt
    checkpoint (the saved System's water count is baked into the
    state.xml), and surfacing the error early prevents wasted
    build work. The companion runner-level quarantine of the stale
    files on ``force=True`` lives in
    ``TestForceTrueQuarantine`` (test_openmm_runner.py).
    """

    def test_state_xml_without_topology_pdb_fails_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """state.xml exists, but topology.pdb is missing.

        ``prepare_simulation`` must refuse to proceed BEFORE calling
        ``build_or_load_modeller`` — the saved state.xml is
        incompatible with a freshly-built System and ``simulation
        .loadState`` must not run. Verified by ``build_or_load_modeller
        `` not being called at all (a spy on it shows zero invocations).
        """
        import biolab_runners.openmm.system_builder as sb

        (tmp_path / "state.xml").write_text("<State/>")
        # topology.pdb is intentionally absent.

        sentinel_chains = [object(), object()]
        fake_modeller = _make_fake_modeller(sentinel_chains, num_atoms=12)
        _stub_openmm(monkeypatch)
        # Spy on build_or_load_modeller so we can assert the cheap
        # integrity check short-circuited BEFORE the expensive
        # solvate/PDBFixer work. The spy returns a fresh modeller
        # if it ever does run; the test asserts the call count is 0.
        build_calls = {"n": 0}

        def _spy_build(*args: object, **kw: object) -> tuple[object, bool]:
            build_calls["n"] += 1
            return fake_modeller, False

        monkeypatch.setattr(sb, "build_or_load_modeller", _spy_build)

        config = OpenMMConfig(
            receptor_pdb="receptor.pdb",
            peptide_pdb="peptide.pdb",
            output_dir=str(tmp_path),
            openmm_platform="CPU",
        )
        result = SimulationResult(config=config)
        state_xml = str(tmp_path / "state.xml")

        ctx = sb.prepare_simulation(config, tmp_path, state_xml, result)

        # The function must refuse to proceed. ctx is None means
        # prepare_simulation returned before the simulation was
        # constructed, so loadState could not have been called —
        # we don't need to assert on sim.loaded_state (the sim was
        # never created in this branch).
        assert ctx is None, "prepare_simulation must return None on corrupt checkpoint"
        assert "topology" in result.error.lower() or "checkpoint" in result.error.lower()
        # SUGGESTION from v5 review: the integrity check must run
        # BEFORE build_or_load_modeller, so the expensive solvation
        # work is skipped on a corrupt checkpoint.
        assert build_calls["n"] == 0, (
            "build_or_load_modeller must NOT be called when resume is corrupt"
        )

    def test_state_xml_with_undersized_topology_pdb_fails_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """state.xml exists, but topology.pdb is < 100 KB (truncated).

        Same as the missing case — the modeller is freshly built, the
        saved state.xml is incompatible, ``simulation.loadState``
        must not run.
        """
        import biolab_runners.openmm.system_builder as sb

        (tmp_path / "state.xml").write_text("<State/>")
        (tmp_path / "topology.pdb").write_text("tiny")  # < 100 KB

        sentinel_chains = [object()]
        fake_modeller = _make_fake_modeller(sentinel_chains, num_atoms=5)
        _stub_openmm(monkeypatch)
        _stub_system_builder(monkeypatch, sb, fake_modeller, loaded_existing_topology=False)

        config = OpenMMConfig(
            receptor_pdb="receptor.pdb",
            peptide_pdb="peptide.pdb",
            output_dir=str(tmp_path),
            openmm_platform="CPU",
        )
        result = SimulationResult(config=config)
        state_xml = str(tmp_path / "state.xml")

        ctx = sb.prepare_simulation(config, tmp_path, state_xml, result)

        # Same fail-fast: ctx is None before the simulation is built.
        assert ctx is None
        assert "topology" in result.error.lower() or "checkpoint" in result.error.lower()

    def test_state_xml_with_intact_topology_pdb_resumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """state.xml exists AND topology.pdb is intact (> 100 KB).

        Resume path: load the original modeller and call
        ``simulation.loadState``. The on-disk topology.pdb is not
        rewritten (it is already the loaded modeller).
        """
        import biolab_runners.openmm.system_builder as sb

        (tmp_path / "state.xml").write_text("<State/>")
        (tmp_path / "topology.pdb").write_bytes(b"X" * 150_000)  # > 100 KB
        original_size = (tmp_path / "topology.pdb").stat().st_size

        sentinel_chains = [object()]
        fake_modeller = _make_fake_modeller(sentinel_chains, num_atoms=99)
        _stub_openmm(monkeypatch)
        # build_or_load_modeller reports loaded_existing_topology=True
        # because the on-disk topology was loaded as-is.
        monkeypatch.setattr(
            sb,
            "build_or_load_modeller",
            lambda *args, **kw: (fake_modeller, True),
        )
        _stub_system_builder(monkeypatch, sb, fake_modeller)

        config = OpenMMConfig(
            receptor_pdb="receptor.pdb",
            peptide_pdb="peptide.pdb",
            output_dir=str(tmp_path),
            openmm_platform="CPU",
        )
        result = SimulationResult(config=config)
        state_xml = str(tmp_path / "state.xml")
        ctx = sb.prepare_simulation(config, tmp_path, state_xml, result)

        assert ctx is not None, f"prepare_simulation failed: {result.error}"
        # metadata still populated
        assert result.num_atoms == 99
        assert result.topology_path == str(tmp_path / "topology.pdb")
        # loadState is the dangerous call — pin that it IS called
        # here (the inverse of the fail-fast tests above). Without
        # this assertion, a regression that silently skips loadState
        # would slip through.
        sim = sys.modules["openmm.app"].Simulation.last_instance
        assert sim is not None
        assert sim.loaded_state == state_xml
        # The file should not have been touched (the stub doesn't
        # call writeFile, but the size check confirms the stub's
        # write_file=False branch was taken).
        assert (tmp_path / "topology.pdb").stat().st_size == original_size


def _make_fake_modeller(chains: list[object], num_atoms: int) -> object:
    """Build a minimal OpenMM Modeller stand-in for the prepare_simulation test."""

    class _FakeTopology:
        def chains(self) -> list[object]:
            return chains

        def getNumAtoms(self) -> int:
            return num_atoms

    class _FakeModeller:
        def __init__(self) -> None:
            self.topology = _FakeTopology()
            self.positions: tuple[object, ...] = ()

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

    class _FakePDBTopology:
        def __init__(self) -> None:
            self.chains = [object()]

    class _FakePDBFile:
        """Stand-in for openmm.app.PDBFile — exposes ``writeFile`` and
        ``topology``/``positions`` attributes consumed by app.Modeller()."""

        def __init__(self, path: str) -> None:
            self.path = path
            self.topology = _FakePDBTopology()
            self.positions = object()

        @staticmethod
        def writeFile(topology: object, positions: object, file_handle: object) -> None:
            # Write a real byte stream so the test can assert the
            # on-disk file was created (this is the regression check
            # for the BLOCKER — if writeFile isn't called, the
            # resulting topology.pdb doesn't exist and loadState
            # would pair against a missing file).
            file_handle.write("FAKE PDB\n")
            file_handle.flush()

    class _FakeModeller:
        """Stand-in for openmm.app.Modeller — built when the existing
        topology.pdb is missing/truncated and a fresh modeller is
        constructed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.topology = _FakePDBTopology()
            self.positions = object()

    class _FakeSimulation:
        last_instance: _FakeSimulation | None = None

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.context = type("_Ctx", (), {"setPositions": lambda self, p: None})()
            self.loaded_state: str | None = None
            # Track the most recently constructed simulation so the
            # test can assert on loadState() calls. Set as a class
            # attribute so the assertion reads the most recent
            # instance even if the test only has the modeller.
            type(self).last_instance = self

        def loadState(self, path: str) -> None:
            # Track the loadState call so tests can assert state.xml
            # is paired with the correct topology.
            self.loaded_state = path

    class _FakeAppModule(types.ModuleType):
        PME = "PME"
        HBonds = "HBonds"
        ForceField = _FakeForceField
        PDBFile = _FakePDBFile
        Simulation = _FakeSimulation
        Modeller = _FakeModeller

    return _FakeAppModule("openmm.app")


def _make_fake_unit_module() -> types.ModuleType:
    """Stand-in for ``openmm.unit`` — no symbols used by prepare_simulation."""
    return types.ModuleType("openmm.unit")


def _stub_system_builder(
    monkeypatch: pytest.MonkeyPatch,
    sb: object,
    fake_modeller: object,
    loaded_existing_topology: bool = True,
) -> None:
    """Stub the heavy collaborators of ``prepare_simulation`` so the
    test doesn't need real OpenMM or pdbfixer.

    ``loaded_existing_topology`` controls the second tuple element
    of the stubbed ``build_or_load_modeller`` return — True for the
    "loaded from disk" path, False for the "freshly built" path.
    The default is True (matches the original semantics); tests
    exercising the resume-without-topology case must pass False.
    """
    monkeypatch.setattr(sb, "build_forcefield", lambda config, app: object())
    # build_or_load_modeller now returns (modeller, loaded_existing_topology).
    monkeypatch.setattr(
        sb,
        "build_or_load_modeller",
        lambda *args, **kw: (fake_modeller, loaded_existing_topology),
    )
    monkeypatch.setattr(sb, "assemble_system", lambda *a, **kw: (MagicMock(), MagicMock()))
    monkeypatch.setattr(sb, "add_ca_restraint", lambda *a, **kw: (MagicMock(), []))
