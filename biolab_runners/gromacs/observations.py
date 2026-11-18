"""Thermodynamic observations read from real GROMACS stage energy files.

The observation surface reports what GROMACS wrote during a stage. It
carries timestamps, explicit units, and the source ``.edr`` digest; it
does not compare a sample to a target, aggregate across stages or
replicas, or label a run as equilibrated or stable.

``gmx energy`` reads one selected term from an ``.edr`` and writes an
``.xvg`` whose header declares both axes' units. Each quantity is read by
its own invocation so a missing term cannot silently shift a column, and
the parsed units come from that header rather than from an assumption.
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from biolab_runners.contracts import MalformedOutputError
from biolab_runners.gromacs.protocol import ProtocolStage, StageKind, build_stage_plan
from biolab_runners.provenance import compute_file_digest

__all__ = [
    "PRESSURE_TERM",
    "TEMPERATURE_TERM",
    "GromacsObservationStatus",
    "GromacsStageObservation",
    "GromacsThermodynamicSample",
    "GromacsThermodynamicSeries",
    "extract_protocol_observations",
    "parse_energy_xvg",
]

TEMPERATURE_TERM = "Temperature"
PRESSURE_TERM = "Pressure"

# The bound only stops a wedged helper from blocking the result; reading an
# existing EDR is not simulation work.
_ENERGY_TIMEOUT_SECONDS = 300

# Selection markers gmx energy prints to stderr when the term is absent.
_NO_TERM_MARKER = "No energy terms selected"

_QUANTITIES = (("temperature", TEMPERATURE_TERM), ("pressure", PRESSURE_TERM))
_OBSERVED_KINDS = (StageKind.EQUIL_NVT, StageKind.EQUIL_NPT, StageKind.PRODUCTION)

_TIME_LABEL_RE = re.compile(r'^@\s+xaxis\s+label\s+"Time\s*\(([^)]*)\)"')
_UNIT_LABEL_RE = re.compile(r'^@\s+yaxis\s+label\s+"\(([^)]*)\)"')
_LEGEND_RE = re.compile(r'^@\s+s0\s+legend\s+"([^"]*)"')


class GromacsObservationStatus:
    """Evidence states for observed thermodynamic data.

    ``AVAILABLE`` means the stage energy file yielded the quantity,
    ``PARTIAL`` means at least one requested quantity is missing, and
    ``UNAVAILABLE`` / ``ERROR`` mean no quantity could be read. None of
    these states is a scientific verdict about the run.
    """

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True)
class GromacsThermodynamicSample:
    """One timestamped value read from a stage energy series."""

    time: float
    value: float

    def to_dict(self) -> dict[str, float]:
        """Serialize the sample into a JSON-safe dictionary."""
        return {"time": self.time, "value": self.value}


@dataclass(frozen=True)
class GromacsThermodynamicSeries:
    """One quantity series with the units declared by its xvg header."""

    quantity: str
    unit: str
    time_unit: str
    status: str
    samples: tuple[GromacsThermodynamicSample, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize the series into a JSON-safe dictionary."""
        return {
            "quantity": self.quantity,
            "unit": self.unit,
            "time_unit": self.time_unit,
            "status": self.status,
            "samples": [sample.to_dict() for sample in self.samples],
            "error": self.error,
        }


@dataclass(frozen=True)
class GromacsStageObservation:
    """Observations for one protocol stage, bound to its source EDR."""

    stage: str
    status: str
    source_edr: str = ""
    source_edr_sha256: str = ""
    temperature: GromacsThermodynamicSeries | None = None
    pressure: GromacsThermodynamicSeries | None = None
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize the observation into a JSON-safe dictionary."""
        return {
            "stage": self.stage,
            "status": self.status,
            "source_edr": self.source_edr,
            "source_edr_sha256": self.source_edr_sha256,
            "temperature": None if self.temperature is None else self.temperature.to_dict(),
            "pressure": None if self.pressure is None else self.pressure.to_dict(),
            "error": self.error,
        }


def _unavailable_series(quantity: str, error: str) -> GromacsThermodynamicSeries:
    return GromacsThermodynamicSeries(
        quantity=quantity,
        unit="",
        time_unit="",
        status=GromacsObservationStatus.UNAVAILABLE,
        error=error,
    )


def _error_series(quantity: str, error: str) -> GromacsThermodynamicSeries:
    return GromacsThermodynamicSeries(
        quantity=quantity,
        unit="",
        time_unit="",
        status=GromacsObservationStatus.ERROR,
        error=error,
    )


def _finite_value(token: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise MalformedOutputError(f"non-numeric energy value: {token!r}") from exc
    if not math.isfinite(value):
        raise MalformedOutputError(f"non-finite energy value: {token!r}")
    return value


def _parse_xvg_header(text: str) -> tuple[str, str, str]:
    """Return ``(time_unit, unit, legend)`` from the first xvg header block."""
    time_unit = ""
    unit = ""
    legend = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("@"):
            break
        if match := _TIME_LABEL_RE.match(line):
            time_unit = match.group(1).strip()
        elif match := _UNIT_LABEL_RE.match(line):
            unit = match.group(1).strip()
        elif match := _LEGEND_RE.match(line):
            legend = match.group(1).strip()
    return time_unit, unit, legend


def _parse_xvg_samples(text: str) -> list[GromacsThermodynamicSample]:
    samples: list[GromacsThermodynamicSample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            if samples:
                raise MalformedOutputError("energy xvg contains more than one data block")
            continue
        tokens = line.split()
        if len(tokens) < 2:
            raise MalformedOutputError(f"malformed energy data row: {line!r}")
        samples.append(
            GromacsThermodynamicSample(
                time=_finite_value(tokens[0]),
                value=_finite_value(tokens[1]),
            )
        )
    return samples


def parse_energy_xvg(
    text: str,
    *,
    quantity: str,
    term: str,
) -> GromacsThermodynamicSeries:
    """Parse one ``gmx energy`` xvg into a timestamped series.

    The units and the legend come from the file header, so an xvg that
    omits them, names another term, contains a non-numeric or non-finite
    value, or holds more than one data block raises
    :class:`MalformedOutputError` instead of producing relabelled values.
    """
    time_unit, unit, legend = _parse_xvg_header(text)
    if legend != term:
        raise MalformedOutputError(f"energy xvg legend {legend!r} does not name {term!r}")
    if not unit:
        raise MalformedOutputError("energy xvg has no y-axis unit label")
    if not time_unit:
        raise MalformedOutputError("energy xvg has no time axis unit label")
    samples = _parse_xvg_samples(text)
    if not samples:
        raise MalformedOutputError("energy xvg contains no data rows")
    return GromacsThermodynamicSeries(
        quantity=quantity,
        unit=unit,
        time_unit=time_unit,
        status=GromacsObservationStatus.AVAILABLE,
        samples=tuple(samples),
    )


def _run_energy_selection(
    *,
    executable: tuple[str, ...],
    work_dir: Path,
    edr_name: str,
    term: str,
    destination: Path,
) -> tuple[int, str]:
    """Run ``gmx energy`` for one term and return ``(returncode, stderr)``.

    ``gmx energy`` reads its term selection from stdin; the trailing ``0``
    ends the selection.
    """
    completed = subprocess.run(
        [*executable, "energy", "-f", edr_name, "-o", str(destination)],
        cwd=str(work_dir),
        input=f"{term}\n0\n",
        capture_output=True,
        text=True,
        timeout=_ENERGY_TIMEOUT_SECONDS,
        check=False,
    )
    return completed.returncode, completed.stderr


def _extract_series(
    work_dir: Path,
    executable: tuple[str, ...],
    edr_name: str,
    quantity: str,
    term: str,
) -> GromacsThermodynamicSeries:
    """Read one quantity from a stage EDR, recording any failure as a state."""
    try:
        temporary = tempfile.TemporaryDirectory(prefix="gromacs-observation-")
    except OSError as exc:
        return _error_series(quantity, f"gmx energy output directory is not available: {exc}")
    with temporary as tmp_dir:
        destination = Path(tmp_dir) / f"{quantity}.xvg"
        try:
            returncode, stderr = _run_energy_selection(
                executable=executable,
                work_dir=work_dir,
                edr_name=edr_name,
                term=term,
                destination=destination,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return _error_series(quantity, f"gmx energy could not run: {exc}")
        if returncode != 0:
            if _NO_TERM_MARKER in stderr:
                return _unavailable_series(quantity, f"{term} is not present in {edr_name}")
            detail = stderr.strip().splitlines()[-1] if stderr.strip() else f"rc={returncode}"
            return _error_series(quantity, f"gmx energy failed for {term}: {detail}")
        try:
            text = destination.read_text()
        except OSError as exc:
            return _error_series(quantity, f"gmx energy output is not readable: {exc}")
        try:
            return parse_energy_xvg(text, quantity=quantity, term=term)
        except MalformedOutputError as exc:
            return _error_series(quantity, str(exc))


def _stage_status(
    series: dict[str, GromacsThermodynamicSeries],
) -> tuple[str, str]:
    available = [
        entry for entry in series.values() if entry.status == GromacsObservationStatus.AVAILABLE
    ]
    failed = [entry for entry in series.values() if entry.status == GromacsObservationStatus.ERROR]
    if len(available) == len(series):
        return GromacsObservationStatus.AVAILABLE, ""
    unmet = [
        entry for entry in series.values() if entry.status != GromacsObservationStatus.AVAILABLE
    ]
    reason = "; ".join(f"{entry.quantity}: {entry.error}" for entry in unmet if entry.error)
    if available:
        return GromacsObservationStatus.PARTIAL, reason
    if failed:
        return GromacsObservationStatus.ERROR, reason
    return GromacsObservationStatus.UNAVAILABLE, reason


def _observed_stages() -> tuple[ProtocolStage, ...]:
    plan = {stage.kind: stage for stage in build_stage_plan()}
    return tuple(plan[kind] for kind in _OBSERVED_KINDS)


def _observe_stage(
    work_dir: Path,
    stage: ProtocolStage,
    executable: tuple[str, ...],
) -> GromacsStageObservation:
    edr = work_dir / f"{stage.prefix}.edr"
    if edr.is_symlink():
        return GromacsStageObservation(
            stage=stage.kind.value,
            status=GromacsObservationStatus.ERROR,
            error=f"stage energy file symlink is outside the output boundary: {edr.name}",
        )
    if not edr.is_file():
        return GromacsStageObservation(
            stage=stage.kind.value,
            status=GromacsObservationStatus.UNAVAILABLE,
            error=f"stage energy file not present: {edr.name}",
        )
    try:
        digest = compute_file_digest(edr) or ""
    except OSError as exc:
        return GromacsStageObservation(
            stage=stage.kind.value,
            status=GromacsObservationStatus.ERROR,
            source_edr=str(edr),
            error=f"stage energy file could not be hashed: {edr.name}: {exc}",
        )
    if not digest:
        return GromacsStageObservation(
            stage=stage.kind.value,
            status=GromacsObservationStatus.UNAVAILABLE,
            source_edr=str(edr),
            error=f"stage energy file could not be hashed: {edr.name}",
        )
    series = {
        quantity: _extract_series(work_dir, executable, edr.name, quantity, term)
        for quantity, term in _QUANTITIES
    }
    status, error = _stage_status(series)
    return GromacsStageObservation(
        stage=stage.kind.value,
        status=status,
        source_edr=str(edr),
        source_edr_sha256=digest,
        temperature=series["temperature"],
        pressure=series["pressure"],
        error=error,
    )


def extract_protocol_observations(
    work_dir: Path,
    executable: tuple[str, ...],
) -> tuple[GromacsStageObservation, ...]:
    """Return one observation record per dynamic protocol stage.

    NVT, NPT, and production stay separate records; nothing is aggregated
    across stages or replicas, and a stage whose EDR is absent or unreadable
    reports that state instead of a value.
    """
    return tuple(_observe_stage(work_dir, stage, executable) for stage in _observed_stages())
