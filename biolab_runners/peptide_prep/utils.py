"""Helpers shared by the peptide-prep runner.

Owns the cross-module utilities the runner consumes:

* :func:`file_sha256` — lowercase hex digest (matches the
  convention used by :mod:`biolab_runners.provenance`).
* :func:`atomic_write_json` — JSON write with ``os.replace`` so a
  crash mid-write leaves the prior content intact.
* :func:`manifest_load` / :func:`manifest_save` — the structured
  manifest I/O. The manifest binds source-digest, config-digest,
  every output-digest, topology bond graph, net charge, chirality
  reports, closure distances, and energies.
* :func:`collect_atom_mapping` — flatten an OpenMM residue's atoms
  into a ``{atom_name: (x, y, z)}`` dict (Å). Used by the
  chirality validator and the D-coordinate transformer.
* :func:`distance` — numpy-free 3-point distance in Å.
* :func:`is_finite` — NaN/inf check across an iterable of numbers.

The runner never re-implements these helpers — they're the canonical
implementations used by the manifest I/O, the D-injection layer,
and the chirality validation layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "atomic_write_json",
    "collect_atom_mapping",
    "distance",
    "file_sha256",
    "is_finite",
    "manifest_load",
    "manifest_save",
]


# ---------------------------------------------------------------------------
# Digests + atomic JSON
# ---------------------------------------------------------------------------

# Manifest schema version — bump when the on-disk shape changes.
# A future loader MUST gate the schema on this field so older
# payloads can be handled explicitly (the runner is forward-
# compatible by ignoring unknown keys but a backward-incompatible
# change needs an explicit loader branch).
MANIFEST_SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str | None:
    """Return the lowercase-hex ``sha256`` of ``path``, or ``None`` if missing."""
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically (``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(str(tmp), str(path))


def manifest_load(work_dir: Path) -> dict[str, Any]:
    """Load the peptide-prep manifest from ``work_dir`` (or empty default)."""
    from biolab_runners.peptide_prep.paths import PeptidePrepFiles

    path = work_dir / PeptidePrepFiles.MANIFEST
    if not path.exists():
        return {"schema_version": MANIFEST_SCHEMA_VERSION}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Manifest at %s unreadable (%s); treating as empty", path, exc)
        return {"schema_version": MANIFEST_SCHEMA_VERSION}
    if not isinstance(data, dict):
        return {"schema_version": MANIFEST_SCHEMA_VERSION}
    return data


def manifest_save(work_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically save the peptide-prep manifest to ``work_dir``."""
    from biolab_runners.peptide_prep.paths import PeptidePrepFiles

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / PeptidePrepFiles.MANIFEST
    atomic_write_json(path, manifest)


# ---------------------------------------------------------------------------
# Geometry helpers (numpy-free, Å-scale)
# ---------------------------------------------------------------------------


def distance(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    """Return the Euclidean distance between two 3-D points in Å.

    Implemented with ``math.hypot`` chain to avoid pulling numpy
    into modules that may run in minimal environments (the runner's
    coordinate-mapping helper is imported by both the D-injection
    layer and the chirality validation layer).
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def is_finite(values: Iterable[float]) -> bool:
    """Return True iff every value is finite (no NaN, no inf, no -inf)."""
    return all(math.isfinite(value) for value in values)


def collect_atom_mapping(
    topology: object,
    positions_value: object,
    residue_index: int,
) -> dict[str, tuple[float, float, float]]:
    """Build an atom-name → (x, y, z) Å mapping for one residue.

    ``positions_value`` is an iterable of OpenMM ``Vec3`` objects
    (from ``simulation.context.getState(getPositions=True).getPositions()``).
    Coordinates are converted to Å via ``value_in_unit(unit.angstrom)``
    so the caller does not need to worry about the openmm unit
    object import — the OpenMM ``Vec3`` objects implement
    ``value_in_unit`` themselves in OpenMM 8.x.

    Args:
        topology: An OpenMM ``app.Topology``.
        positions_value: An iterable of OpenMM ``Vec3`` (length
            ``topology.getNumAtoms()``).
        residue_index: 0-indexed chain position.

    Returns:
        ``{atom_name: (x_angstrom, y_angstrom, z_angstrom)}`` for
        the residue's atoms. Atoms without a name are skipped
        (shouldn't happen for a protein residue but defensively).
    """
    try:
        import openmm.unit as unit
    except ImportError as exc:  # pragma: no cover - openmm is required
        raise RuntimeError(
            "collect_atom_mapping requires openmm.unit; install the openmm extra"
        ) from exc

    atoms = list(topology.atoms())
    mapping: dict[str, tuple[float, float, float]] = {}
    for atom in atoms:
        if atom.residue.index != residue_index:
            continue
        pos = positions_value[atom.index]  # type: ignore[index]
        # OpenMM Vec3 stores x/y/z as Quantity; convert to Å via
        # the ``in_unit_of`` arithmetic that always works.
        x = (pos[0] / unit.angstrom) if hasattr(pos[0], "__truediv__") else pos[0]  # type: ignore[index]
        y = (pos[1] / unit.angstrom) if hasattr(pos[1], "__truediv__") else pos[1]  # type: ignore[index]
        z = (pos[2] / unit.angstrom) if hasattr(pos[2], "__truediv__") else pos[2]  # type: ignore[index]
        mapping[atom.name] = (float(x), float(y), float(z))
    return mapping


# ---------------------------------------------------------------------------
# Topology descriptor + manifest helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologyBondRecord:
    """A bonded pair in the prepared peptide's bond graph.

    The ``bond_type`` is one of:

    * ``"backbone"`` — standard N-CA, CA-C, C-N peptide bonds (these
      were already in the linear system).
    * ``"head_to_tail"`` — explicit closure bond between the
      head-residue C atom and the tail-residue N atom.
    * ``"disulfide"`` — S-S bond between two cysteine SG atoms.

    The ``head_to_tail`` and ``disulfide`` entries are added by the
    runner; the linear ``backbone`` bonds are taken from the OpenMM
    HarmonicBondForce (one entry per pair, regardless of which
    residue owns which atom).
    """

    atom1_index: int
    atom2_index: int
    bond_type: str
    atom1_name: str = ""
    atom2_name: str = ""
    residue1_index: int = -1
    residue2_index: int = -1
