"""Tests for the GROMACS thermodynamic observation surface.

``GromacsProtocolResult.observations`` reports what GROMACS actually wrote
into each dynamic stage's ``.edr`` file. These tests pin three contracts:

- the ``gmx energy`` xvg parser records timestamps, units, and values from
  the file header instead of assuming them;
- absent, partial, malformed, and failed extractions stay explicit
  observation states (never fabricated zeros or a run-level pass);
- every record is bound to its own protocol stage and stage EDR digest.

The parser tests use headers copied from a real GROMACS 2026.1
``gmx energy`` run. The protocol-level tests replace the ``gmx energy``
call, so they document the wiring contract only; the real extraction path
is covered by ``tests/integration/test_scientific_validation.py``.
No test asserts thermal stability, an equilibration verdict, or a bound:
the observation surface carries measurements, not conclusions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from biolab_runners.contracts import ExecutionStatus, MalformedOutputError
from biolab_runners.gromacs import (
    GromacsObservationStatus,
    GromacsProtocolConfig,
    GromacsProtocolResult,
    GromacsProtocolRunner,
)
from biolab_runners.gromacs.observations import (
    extract_protocol_observations,
    parse_energy_xvg,
)
from biolab_runners.gromacs.protocol import StageKind, build_stage_plan
from biolab_runners.gromacs.utils import StageStatus
from biolab_runners.provenance import compute_file_digest

_ENERGY_HEADER = """\
# This file was created by gmx energy
# Created by:
#                      :-) GROMACS - gmx energy, 2026.1 (-:
#
# Executable:   /usr/local/bin/gmx
# Data prefix:  /usr/local/gromacs
# Command line:
#   gmx energy -f nvt.edr -o temperature.xvg
@    title "GROMACS Energies"
@    xaxis  label "Time (ps)"
@    yaxis  label "({unit})"
@TYPE xy
@ view 0.15, 0.15, 0.75, 0.85
@ legend on
@ legend box on
@ legend loctype view
@ legend 0.78, 0.8
@ legend length 2
@ s0 legend "{term}"
"""

_STAGE_EDR_PAYLOADS = {
    "nvt.edr": b"nvt-stage-energy-bytes",
    "npt.edr": b"npt-stage-energy-bytes",
    "prod.edr": b"prod-stage-energy-bytes",
}


def _xvg(term: str, unit: str, rows: str) -> str:
    return _ENERGY_HEADER.format(term=term, unit=unit) + rows


TEMPERATURE_XVG = _xvg(
    "Temperature",
    "K",
    "    0.000000  306.527649\n    1.000000  317.481415\n",
)
PRESSURE_XVG = _xvg(
    "Pressure",
    "bar",
    "    0.000000  1309.078369\n    1.000000  323.859222\n",
)
AVAILABLE_STAGE_XVGS = {
    ("nvt.edr", "Temperature"): TEMPERATURE_XVG,
    ("nvt.edr", "Pressure"): PRESSURE_XVG,
    ("npt.edr", "Temperature"): TEMPERATURE_XVG,
    ("npt.edr", "Pressure"): PRESSURE_XVG,
    ("prod.edr", "Temperature"): TEMPERATURE_XVG,
    ("prod.edr", "Pressure"): PRESSURE_XVG,
}


def _protocol_config(tmp_path: Path, **overrides: Any) -> GromacsProtocolConfig:
    payload: dict[str, Any] = {
        "name": "observations",
        "input_pdb": "/tmp/input.pdb",
        "output_root": str(tmp_path),
    }
    payload.update(overrides)
    return GromacsProtocolConfig(**payload)


def _energy_stub(by_edr_and_term: dict[tuple[str, str], str]) -> Any:
    """Return a ``gmx energy`` replacement that writes fixture xvg text."""

    def fake(
        *,
        executable: tuple[str, ...],
        work_dir: Path,
        edr_name: str,
        term: str,
        destination: Path,
    ) -> tuple[int, str]:
        text = by_edr_and_term.get((edr_name, term))
        if text is None:
            return 1, "Fatal error:\nNo energy terms selected\n"
        destination.write_text(text)
        return 0, ""

    return fake


def _seed_stage_edrs(work_dir: Path) -> None:
    """Write one distinct EDR payload per dynamic stage."""
    work_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in _STAGE_EDR_PAYLOADS.items():
        (work_dir / name).write_bytes(payload)


class TestParseEnergyXvg:
    def test_parses_timestamped_temperature_samples_in_kelvin(self) -> None:
        series = parse_energy_xvg(TEMPERATURE_XVG, quantity="temperature", term="Temperature")

        assert series.quantity == "temperature"
        assert series.unit == "K"
        assert series.time_unit == "ps"
        assert series.status == GromacsObservationStatus.AVAILABLE
        assert series.error == ""
        assert [(sample.time, sample.value) for sample in series.samples] == [
            (0.0, pytest.approx(306.527649)),
            (1.0, pytest.approx(317.481415)),
        ]

    def test_parses_timestamped_pressure_samples_in_bar(self) -> None:
        series = parse_energy_xvg(PRESSURE_XVG, quantity="pressure", term="Pressure")

        assert series.unit == "bar"
        assert series.time_unit == "ps"
        assert [sample.value for sample in series.samples] == [
            pytest.approx(1309.078369),
            pytest.approx(323.859222),
        ]

    def test_records_time_unit_from_the_header_without_conversion(self) -> None:
        text = TEMPERATURE_XVG.replace('"Time (ps)"', '"Time (ns)"')

        series = parse_energy_xvg(text, quantity="temperature", term="Temperature")

        assert series.time_unit == "ns"
        assert [sample.time for sample in series.samples] == [0.0, 1.0]

    def test_rejects_missing_y_unit_header(self) -> None:
        text = TEMPERATURE_XVG.replace('@    yaxis  label "(K)"\n', "")

        with pytest.raises(MalformedOutputError, match="y-axis"):
            parse_energy_xvg(text, quantity="temperature", term="Temperature")

    def test_rejects_missing_time_unit_header(self) -> None:
        text = TEMPERATURE_XVG.replace('@    xaxis  label "Time (ps)"\n', "")

        with pytest.raises(MalformedOutputError, match="time"):
            parse_energy_xvg(text, quantity="temperature", term="Temperature")

    def test_rejects_legend_for_a_different_term(self) -> None:
        with pytest.raises(MalformedOutputError, match="legend"):
            parse_energy_xvg(PRESSURE_XVG, quantity="temperature", term="Temperature")

    def test_rejects_non_numeric_value(self) -> None:
        text = _xvg("Temperature", "K", "    0.000000  not-a-number\n")

        with pytest.raises(MalformedOutputError, match="non-numeric"):
            parse_energy_xvg(text, quantity="temperature", term="Temperature")

    @pytest.mark.parametrize("token", ["nan", "inf", "-inf"])
    def test_rejects_non_finite_value(self, token: str) -> None:
        text = _xvg("Temperature", "K", f"    0.000000  {token}\n")

        with pytest.raises(MalformedOutputError, match="non-finite"):
            parse_energy_xvg(text, quantity="temperature", term="Temperature")

    def test_rejects_malformed_data_row(self) -> None:
        text = _xvg("Temperature", "K", "    0.000000\n")

        with pytest.raises(MalformedOutputError, match="malformed"):
            parse_energy_xvg(text, quantity="temperature", term="Temperature")

    def test_rejects_second_data_block(self) -> None:
        text = (
            TEMPERATURE_XVG
            + _ENERGY_HEADER.format(term="Pressure", unit="bar")
            + "    0.000000  1309.078369\n"
        )

        with pytest.raises(MalformedOutputError, match="data block"):
            parse_energy_xvg(text, quantity="temperature", term="Temperature")

    def test_rejects_xvg_without_data_rows(self) -> None:
        text = _xvg("Temperature", "K", "")

        with pytest.raises(MalformedOutputError, match="no data"):
            parse_energy_xvg(text, quantity="temperature", term="Temperature")


class TestStageObservations:
    def test_observes_only_nvt_npt_and_production(self, tmp_path: Path) -> None:
        (tmp_path / "min.edr").write_bytes(b"minimisation-energy-bytes")

        observations = extract_protocol_observations(tmp_path, ("gmx",))

        assert [observation.stage for observation in observations] == [
            StageKind.EQUIL_NVT.value,
            StageKind.EQUIL_NPT.value,
            StageKind.PRODUCTION.value,
        ]
        for observation in observations:
            assert observation.status == GromacsObservationStatus.UNAVAILABLE
            assert observation.source_edr == ""
            assert observation.source_edr_sha256 == ""
            assert observation.temperature is None
            assert observation.pressure is None
            assert "not present" in observation.error

    def test_symlinked_edr_is_refused_without_extraction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        outside = tmp_path / "outside.edr"
        outside.write_bytes(b"outside-stage-energy-bytes")
        outside.chmod(0o000)
        (work_dir / "nvt.edr").symlink_to(outside)

        def unexpected(**_: Any) -> tuple[int, str]:
            raise AssertionError("gmx energy must not run for a symlinked EDR")

        monkeypatch.setattr("biolab_runners.gromacs.observations._run_energy_selection", unexpected)

        observations = extract_protocol_observations(work_dir, ("gmx",))
        nvt = observations[0]

        assert nvt.stage == StageKind.EQUIL_NVT.value
        assert nvt.status == GromacsObservationStatus.ERROR
        assert "symlink" in nvt.error
        assert nvt.source_edr == ""
        assert nvt.source_edr_sha256 == ""
        assert nvt.temperature is None
        assert nvt.pressure is None
        assert [observation.status for observation in observations[1:]] == [
            GromacsObservationStatus.UNAVAILABLE
        ] * 2

    def test_unreadable_stage_edr_marks_stage_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = tmp_path / "run"
        work_dir.mkdir()
        edr = work_dir / "nvt.edr"
        edr.write_bytes(b"stage-energy-bytes")

        def unreadable(_: Path) -> str | None:
            raise PermissionError("permission denied")

        monkeypatch.setattr("biolab_runners.gromacs.observations.compute_file_digest", unreadable)

        nvt = extract_protocol_observations(work_dir, ("gmx",))[0]

        assert nvt.stage == StageKind.EQUIL_NVT.value
        assert nvt.status == GromacsObservationStatus.ERROR
        assert nvt.source_edr == str(edr)
        assert nvt.source_edr_sha256 == ""
        assert nvt.temperature is None
        assert nvt.pressure is None
        assert "could not be hashed" in nvt.error
        assert "permission denied" in nvt.error

    def test_binds_each_observation_to_its_stage_edr_digest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_stage_edrs(tmp_path)
        stub = {
            ("nvt.edr", "Temperature"): _xvg("Temperature", "K", "    0.000000  308.0\n"),
            ("nvt.edr", "Pressure"): _xvg("Pressure", "bar", "    0.000000  12.0\n"),
            ("npt.edr", "Temperature"): _xvg("Temperature", "K", "    0.000000  309.5\n"),
            ("npt.edr", "Pressure"): _xvg("Pressure", "bar", "    0.000000   1.5\n"),
            ("prod.edr", "Temperature"): _xvg(
                "Temperature", "K", "    0.000000  310.0\n   10.000000  311.0\n"
            ),
            ("prod.edr", "Pressure"): _xvg(
                "Pressure", "bar", "    0.000000  2.0\n   10.000000  3.0\n"
            ),
        }
        monkeypatch.setattr(
            "biolab_runners.gromacs.observations._run_energy_selection", _energy_stub(stub)
        )

        observations = extract_protocol_observations(tmp_path, ("gmx",))

        for observation, edr_name in zip(observations, _STAGE_EDR_PAYLOADS, strict=True):
            edr = tmp_path / edr_name
            assert observation.status == GromacsObservationStatus.AVAILABLE
            assert observation.source_edr == str(edr)
            assert observation.source_edr_sha256 == compute_file_digest(edr)
            assert observation.source_edr_sha256 != ""
        nvt_temperature = observations[0].temperature
        nvt_pressure = observations[0].pressure
        prod_temperature = observations[2].temperature
        assert nvt_temperature is not None
        assert nvt_pressure is not None
        assert prod_temperature is not None
        assert nvt_temperature.unit == "K"
        assert nvt_temperature.time_unit == "ps"
        assert nvt_pressure.unit == "bar"
        # Stage values stay separate: no aggregation across stages.
        assert [sample.value for sample in nvt_temperature.samples] == [pytest.approx(308.0)]
        assert [sample.value for sample in prod_temperature.samples] == [
            pytest.approx(310.0),
            pytest.approx(311.0),
        ]

    def test_missing_term_marks_stage_partial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_stage_edrs(tmp_path)
        temperature_only = {
            (edr_name, "Temperature"): text
            for (edr_name, term), text in AVAILABLE_STAGE_XVGS.items()
            if term == "Temperature"
        }
        monkeypatch.setattr(
            "biolab_runners.gromacs.observations._run_energy_selection",
            _energy_stub(temperature_only),
        )

        observations = extract_protocol_observations(tmp_path, ("gmx",))

        for observation in observations:
            assert observation.status == GromacsObservationStatus.PARTIAL
            assert observation.temperature is not None
            assert observation.temperature.status == GromacsObservationStatus.AVAILABLE
            assert observation.pressure is not None
            assert observation.pressure.status == GromacsObservationStatus.UNAVAILABLE
            assert observation.pressure.samples == ()
            assert "pressure" in observation.error

    def test_malformed_series_marks_series_error_without_samples(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_stage_edrs(tmp_path)
        stub = {
            ("nvt.edr", "Temperature"): _xvg("Temperature", "K", "    0.000000  nan\n"),
            ("nvt.edr", "Pressure"): _xvg("Pressure", "bar", "    0.000000  1.0\n"),
        }
        monkeypatch.setattr(
            "biolab_runners.gromacs.observations._run_energy_selection", _energy_stub(stub)
        )

        observation = extract_protocol_observations(tmp_path, ("gmx",))[0]

        assert observation.stage == StageKind.EQUIL_NVT.value
        assert observation.status == GromacsObservationStatus.PARTIAL
        assert observation.temperature is not None
        assert observation.temperature.status == GromacsObservationStatus.ERROR
        assert "non-finite" in observation.temperature.error
        assert observation.temperature.samples == ()
        assert observation.pressure is not None
        assert observation.pressure.status == GromacsObservationStatus.AVAILABLE

    def test_temporary_output_failure_marks_series_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_stage_edrs(tmp_path)

        def failing(*_: Any, **__: Any) -> Any:
            raise OSError("no space left on device")

        monkeypatch.setattr(
            "biolab_runners.gromacs.observations.tempfile.TemporaryDirectory", failing
        )

        observation = extract_protocol_observations(tmp_path, ("gmx",))[0]

        assert observation.status == GromacsObservationStatus.ERROR
        assert observation.temperature is not None
        assert observation.temperature.status == GromacsObservationStatus.ERROR
        assert "not available" in observation.temperature.error
        assert observation.pressure is not None
        assert observation.pressure.status == GromacsObservationStatus.ERROR

    def test_extractor_failure_marks_stage_error_not_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_stage_edrs(tmp_path)

        def failing(**_: Any) -> tuple[int, str]:
            raise FileNotFoundError("gmx")

        monkeypatch.setattr("biolab_runners.gromacs.observations._run_energy_selection", failing)

        observation = extract_protocol_observations(tmp_path, ("gmx",))[0]

        assert observation.status == GromacsObservationStatus.ERROR
        assert observation.temperature is not None
        assert observation.temperature.status == GromacsObservationStatus.ERROR
        assert observation.temperature.samples == ()
        assert "gmx" in observation.temperature.error

    def test_unreadable_term_output_marks_series_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_stage_edrs(tmp_path)

        def no_output(**_: Any) -> tuple[int, str]:
            return 0, ""

        monkeypatch.setattr("biolab_runners.gromacs.observations._run_energy_selection", no_output)

        observation = extract_protocol_observations(tmp_path, ("gmx",))[0]

        assert observation.status == GromacsObservationStatus.ERROR
        assert observation.temperature is not None
        assert observation.temperature.status == GromacsObservationStatus.ERROR
        assert "not readable" in observation.temperature.error


class TestProtocolResultObservations:
    def test_default_result_has_no_observations(self, tmp_path: Path) -> None:
        # The pre-observation constructor call shape stays valid.
        result = GromacsProtocolResult(
            name="legacy",
            output_dir=str(tmp_path),
            replica_index=0,
            replicas_total=1,
        )

        assert result.observations == ()
        assert result.to_dict()["observations"] == []
        assert result.status == ExecutionStatus.INCOMPLETE

    def test_protocol_result_reports_stage_edr_observations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _protocol_config(tmp_path)
        _seed_stage_edrs(tmp_path / config.name)
        monkeypatch.setattr(
            "biolab_runners.gromacs.observations._run_energy_selection",
            _energy_stub(AVAILABLE_STAGE_XVGS),
        )
        monkeypatch.setattr(GromacsProtocolRunner, "_run_subprocess", lambda *args: 0)

        result = GromacsProtocolRunner().run_protocol(config)

        assert result.exit_code == 0
        assert result.error == ""
        assert result.status == ExecutionStatus.SUCCEEDED
        assert [observation.stage for observation in result.observations] == [
            StageKind.EQUIL_NVT.value,
            StageKind.EQUIL_NPT.value,
            StageKind.PRODUCTION.value,
        ]
        for observation in result.observations:
            edr = Path(observation.source_edr)
            assert edr.name in _STAGE_EDR_PAYLOADS
            assert observation.source_edr_sha256 == compute_file_digest(edr)
        serialized = result.to_dict()["observations"]
        assert [record["stage"] for record in serialized] == [
            StageKind.EQUIL_NVT.value,
            StageKind.EQUIL_NPT.value,
            StageKind.PRODUCTION.value,
        ]
        nvt = serialized[0]
        assert nvt["status"] == GromacsObservationStatus.AVAILABLE
        assert nvt["temperature"]["unit"] == "K"
        assert nvt["temperature"]["time_unit"] == "ps"
        assert nvt["temperature"]["samples"] == [
            {"time": 0.0, "value": pytest.approx(306.527649)},
            {"time": 1.0, "value": pytest.approx(317.481415)},
        ]
        assert nvt["pressure"]["unit"] == "bar"

    def test_cached_protocol_result_still_reports_observations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _protocol_config(tmp_path)
        _seed_stage_edrs(tmp_path / config.name)
        monkeypatch.setattr(
            "biolab_runners.gromacs.observations._run_energy_selection",
            _energy_stub(AVAILABLE_STAGE_XVGS),
        )
        monkeypatch.setattr(GromacsProtocolRunner, "_run_subprocess", lambda *args: 0)

        GromacsProtocolRunner().run_protocol(config)
        cached = GromacsProtocolRunner().run_protocol(config)

        assert cached.skipped == len(build_stage_plan())
        assert cached.executed is False
        assert [observation.status for observation in cached.observations] == [
            GromacsObservationStatus.AVAILABLE
        ] * 3

    def test_observation_failure_leaves_process_status_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _protocol_config(tmp_path)
        _seed_stage_edrs(tmp_path / config.name)

        def failing(**_: Any) -> tuple[int, str]:
            raise OSError("energy extraction unavailable")

        monkeypatch.setattr("biolab_runners.gromacs.observations._run_energy_selection", failing)
        monkeypatch.setattr(GromacsProtocolRunner, "_run_subprocess", lambda *args: 0)

        result = GromacsProtocolRunner().run_protocol(config)

        assert result.exit_code == 0
        assert result.error == ""
        assert result.status == ExecutionStatus.SUCCEEDED
        assert result.stage_statuses[StageKind.PRODUCTION.value] == StageStatus.COMPLETED
        assert [observation.status for observation in result.observations] == [
            GromacsObservationStatus.ERROR
        ] * 3

    def test_dry_run_result_reports_no_observations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _protocol_config(tmp_path)
        _seed_stage_edrs(tmp_path / config.name)

        def unexpected(**_: Any) -> tuple[int, str]:
            raise AssertionError("dry run must not read stage energy files")

        monkeypatch.setattr("biolab_runners.gromacs.observations._run_energy_selection", unexpected)

        result = GromacsProtocolRunner(dry_run=True).run_protocol(config)

        assert result.observations == ()
        assert result.validated > 0
