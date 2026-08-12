"""Shared pytest fixtures and helper classes for the test suite.

Importable from any test file as a regular module. The stand-in classes
mock just enough of OpenMM's surface for the test to drive the runner
or system_builder without requiring a real OpenMM install (which is
optional under the ``[openmm]`` extra).
"""

from __future__ import annotations

import math

import numpy as np
from biolab_runners.openmm.offline_gate import FloatArray

# ---------------------------------------------------------------------------
# Topology stand-ins (for tests that walk chain.atoms() / atom.name / atom.index)
# ---------------------------------------------------------------------------


class FakeAtom:
    """Minimal stand-in for an OpenMM Topology Atom (just .name + .index)."""

    def __init__(self, name: str, index: int) -> None:
        self.name = name
        self.index = index


class FakeChain:
    """Wraps a list of FakeAtom; exposes .atoms() like an OpenMM Chain."""

    def __init__(self, atoms: list[FakeAtom]) -> None:
        self._atoms = atoms

    def atoms(self) -> list[FakeAtom]:
        return self._atoms


# ---------------------------------------------------------------------------
# System / ForceField stand-ins (for system_builder unit tests)
# ---------------------------------------------------------------------------


class FakePos:
    """Position with .x / .y / .z — for tests that read modeller.positions[idx]."""

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class RecordingForceField:
    """Captures the XML paths passed to ``app.ForceField(*paths)``."""

    def __init__(self, *paths: str) -> None:
        self.paths = paths


class FakeSystem:
    """Records forces added via addForce()."""

    def __init__(self) -> None:
        self.forces: list[object] = []

    def addForce(self, force: object) -> None:
        self.forces.append(force)


class FakeQuantity:
    """Quantity that supports ``float * quantity`` and ``float / quantity``,
    stringifying for assertions (e.g. ``\"1.0 nm\"``)."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __rmul__(self, other: float) -> FakeQuantity:
        return FakeQuantity(f"{other} {self.label}")

    def __rtruediv__(self, other: float) -> FakeQuantity:
        return FakeQuantity(f"{other}/{self.label}")

    def __str__(self) -> str:
        return self.label

    __repr__ = __str__


class FakeUnit:
    """Stand-in for openmm.unit with named quantities."""

    def __init__(self) -> None:
        self.nanometers = FakeQuantity("nm")
        self.atmospheres = FakeQuantity("atm")
        self.kelvin = FakeQuantity("K")
        self.picoseconds = FakeQuantity("ps")
        self.femtoseconds = FakeQuantity("fs")


class FakePDBFile:
    """Stand-in for openmm.app.PDBFile — exposes ``writeFile`` as a callable."""

    writeFile = staticmethod(lambda *args, **kwargs: None)  # type: ignore[assignment]


class FakeApp:
    """Base stand-in for openmm.app. Subclass for richer mocks."""

    def __init__(self) -> None:
        self.PME = "PME"
        self.CutoffPeriodic = "CutoffPeriodic"
        self.HBonds = "HBonds"
        self.AllBonds = "AllBonds"
        self.HAngles = "HAngles"
        self.ForceField = RecordingForceField
        self.PDBFile = FakePDBFile


# ---------------------------------------------------------------------------
# Box-vector factory (shared between test_openmm_runner, test_geometry)
# ---------------------------------------------------------------------------


def dodecahedron_box(d: float = 60.0) -> FloatArray:
    """GROMACS rhombic dodecahedron (xy-square) with edge length ``d``.

    Returns a 3x3 box matrix as a numpy array.
    """
    return np.array(
        [
            [d, 0.0, 0.0],
            [0.0, d, 0.0],
            [0.5 * d, 0.5 * d, d / math.sqrt(2.0)],
        ]
    )
