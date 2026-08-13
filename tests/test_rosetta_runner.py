"""Tests for the Rosetta runner.

The scorefile fixtures in this module are **synthetic** — they were
hand-assembled to model known upstream scorefile headers produced by
``InterfaceAnalyzer`` (default and relaxed ``-score:weights``) and
the per-protocol FastRelax variants used by the runner's consumer
XML. No score in this file was captured from a licensed Rosetta
run; the real-binary smoke is documented as an external gate on
:mod:`biolab_runners.rosetta` (licensed + binary-dependent).
"""

from __future__ import annotations

import unittest.mock as mock_mod
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
from biolab_runners.rosetta import (
    METRIC_ALIASES,
    ConstrainedRelaxOptions,
    RelaxRecord,
    RelaxRecordStatus,
    RelaxScore,
    rosetta_available,
)
from biolab_runners.rosetta.config import RosettaConfig
from biolab_runners.rosetta.runner import RosettaRunner, _config_to_cli
from biolab_runners.rosetta.utils import (
    invoke,
    parse_relax_score,
    parse_score_file,
    parse_score_files,
)

# ---------------------------------------------------------------------------
# Synthetic scorefile fixtures
# ---------------------------------------------------------------------------
#
# Each fixture models a known Rosetta scorefile layout. The token counts
# in the data rows are *intentionally* distinct for each metric so a
# shifted-column mutation cannot satisfy two assertions at once. The
# headers use canonical + non-canonical aliases to exercise both the
# alias map and the up-front tokenizer.

# Minimal scorefile — only ``total_score`` is present (legacy baseline).
SCORE_MINIMAL = "SEQUENCE:\nSCORE: total_score     description\nSCORE:    -123.456    input.pdb\n"

# Canonical InterfaceAnalyzer layout — every tracked metric present.
# Uses ``interface_sc`` (the scoped alias for shape complementarity)
# — the bare ``sc`` alias was deliberately omitted from the alias
# table to avoid ambiguity.
SCORE_FULL_CANONICAL = (
    "SEQUENCE:\n"
    "SCORE: total_score sasa dsasa sasa_hphobic sasa_polar"
    " dSASA_polar dSASA_hphobic dG_separated dSASA_iface"
    " unsat_hbonds hbonds_int hbond_E_int interface_sc packstat"
    " description\n"
    "SCORE: -123.456 12450.7 -850.4 7320.1 5130.6"
    " 1000.0 6320.1 -45.231 -212.004 3.000 12.000 -8.450"
    " 0.712 0.65 input.pdb\n"
)

# Alias-driven scorefile — non-canonical column names for every
# metric (e.g. ``Total_Score`` instead of ``total_score``,
# ``dG_cross`` instead of ``dG_separated``). Same numerical
# values as the canonical fixture so a single parametrized pin
# table works against both — proves the alias map is
# semantically transparent (no value shift, no metric swap).
#
# The v0.7 disambiguated metrics (``hbond_energy_fraction``,
# ``packstat``) still appear here via their scoped aliases
# (``hbond_E_fraction`` and ``packstat``) and parse into their
# OWN fields, NOT into ``hbond_energy`` / ``shape_complementarity``
# — the v0.7 fix is that those metrics are no longer conflated.
# Their pin values for the alias test come from this fixture
# only.
SCORE_FULL_ALIASES = (
    "SEQUENCE:\n"
    "SCORE: Total_Score fa_sasa sasa_delta sasa_hydrophobic polar_sasa"
    " dSASA_polar dSASA_hphobic dG_cross dSASA_int"
    " delta_unsatHbonds cross_interface_hbonds hbond_E_int"
    " hbond_E_fraction sc_value packstat description\n"
    "SCORE: -123.456 12450.7 -850.4 7320.1 5130.6"
    " 1000.0 6320.1 -45.231 -212.004 3.000 12.000 -8.450"
    " 0.45 0.712 0.65 input.pdb\n"
)

# Newly-added-alias scorefile — every v0.6 / v0.7 canonical
# InterfaceAnalyzer alias (``dSASA_int`` / ``dSASA_polar`` /
# ``dSASA_hphobic`` / ``delta_unsatHbonds`` / ``hbond_E_fraction`` /
# ``packstat``) resolves to its DISTINCT canonical metric.
SCORE_NEW_ALIASES = (
    "SEQUENCE:\n"
    "SCORE: total_score dSASA_int dSASA_polar dSASA_hphobic"
    " delta_unsatHbonds hbond_E_fraction packstat"
    " description\n"
    "SCORE: -50.0 -100.0 200.0 800.0 2.0 -5.0 0.65\n"
)

# Subset scorefile — header lists every column the protocol writes,
# but the data row has too few tokens (real upstream behavior when
# the scorefunction omits disabled terms).
SCORE_TRUNCATED_DATA = (
    "SEQUENCE:\nSCORE: total_score sasa dsasa description\nSCORE: -7.500 9000.0\n"
)

# All-empty scorefile — header lists only the description placeholder.
SCORE_EMPTY_HEADERS = "SEQUENCE:\nSCORE: description\nSCORE: input.pdb\n"

# Garbage — looks nothing like a scorefile.
SCORE_GARBAGE = "not a score\n# maybe a comment\n"


def _valid_config(**overrides: Any) -> RosettaConfig:
    base: dict[str, Any] = {
        "name": "relax-1",
        "script_file": "/opt/rosetta/scripts/relax.xml",
        "input_pdb": "/tmp/input.pdb",
        "output_dir": "/tmp/output",
        "nstruct": 1,
        "license_acknowledged": True,
    }
    base.update(overrides)
    return RosettaConfig(**base)


def _write_scorefile(tmp_path: Path, content: str, name: str = "score.sc") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_rejects_unacknowledged_license() -> None:
    with pytest.raises(ValueError, match="license acknowledgement"):
        RosettaConfig(
            script_file="/tmp/x.xml",
            input_pdb="/tmp/x.pdb",
            license_acknowledged=False,
        )


def test_config_rejects_missing_script() -> None:
    with pytest.raises(ValueError, match="script_file is required"):
        RosettaConfig(
            script_file="",
            input_pdb="/tmp/x.pdb",
            license_acknowledged=True,
        )


def test_config_rejects_missing_input() -> None:
    with pytest.raises(ValueError, match="input_pdb is required"):
        RosettaConfig(
            script_file="/tmp/x.xml",
            input_pdb="",
            license_acknowledged=True,
        )


def test_config_rejects_bad_nstruct() -> None:
    with pytest.raises(ValueError, match="nstruct"):
        RosettaConfig(
            script_file="/tmp/x.xml",
            input_pdb="/tmp/x.pdb",
            nstruct=0,
            license_acknowledged=True,
        )


def test_config_rejects_invalid_preparation_mode() -> None:
    with pytest.raises(ValueError, match="preparation_mode"):
        # The ``# type: ignore`` flags below acknowledge the literal
        # bypass — the test's purpose is to lock the runtime guard,
        # not the type system.
        RosettaConfig(
            script_file="/tmp/x.xml",
            input_pdb="/tmp/x.pdb",
            license_acknowledged=True,
            preparation_mode="triangular",  # type: ignore[arg-type]
        )


def test_config_rejects_empty_preparation_mode() -> None:
    """Edge case: empty string is non-None but not a recognized
    Literal value — must be rejected at construction so a typo
    doesn't silently route a script-var to ``prep_mode=``.
    """
    with pytest.raises(ValueError, match="preparation_mode"):
        RosettaConfig(
            script_file="/tmp/x.xml",
            input_pdb="/tmp/x.pdb",
            license_acknowledged=True,
            preparation_mode="",  # type: ignore[arg-type]
        )


def test_config_accepts_linear_preparation_mode() -> None:
    cfg = _valid_config(preparation_mode="linear")
    assert cfg.preparation_mode == "linear"


def test_config_accepts_cyclic_preparation_mode() -> None:
    cfg = _valid_config(preparation_mode="cyclic")
    assert cfg.preparation_mode == "cyclic"


def test_config_default_constrained_relax_is_none() -> None:
    cfg = _valid_config()
    assert cfg.constrained_relax is None


def test_constrained_relax_options_accept_partial_state() -> None:
    """A populated subset is allowed; only fields the caller set appear."""
    opts = ConstrainedRelaxOptions(relax_cycles=5, constrain_to_start_coords=True)
    assert opts.relax_cycles == 5
    assert opts.constrain_to_start_coords is True
    # The other fields stay None so the runner only emits the set ones.
    assert opts.ramp_constraints is None
    assert opts.coord_constrain_sidechains is None
    assert opts.bb_min_only is None


def test_constrained_relax_options_default_all_none() -> None:
    """Empty ConstrainedRelaxOptions emits no script-var tokens at all
    (covers the edge case where the caller instantiates an
    empty-typed dataclass and inadvertently adds no behavior).
    """
    opts = ConstrainedRelaxOptions()
    assert all(getattr(opts, f.name) is None for f in fields(opts))


# ---------------------------------------------------------------------------
# utils — scorefile parser
# ---------------------------------------------------------------------------


def test_parse_score_file_legacy_returns_total_score_float(tmp_path: Path) -> None:
    """Backward-compat: ``parse_score_file`` returns the first-row
    total score as a ``float`` (matches the v0.5 contract), with
    ``0.0`` for empty/garbage/unparseable files.
    """
    minimal = _write_scorefile(tmp_path, SCORE_MINIMAL, name="min.sc")
    assert parse_score_file(minimal) == pytest.approx(-123.456)
    empty = _write_scorefile(tmp_path, "", name="e.sc")
    assert parse_score_file(empty) == 0.0
    garbage = _write_scorefile(tmp_path, SCORE_GARBAGE, name="g.sc")
    assert parse_score_file(garbage) == 0.0
    only_desc = _write_scorefile(tmp_path, SCORE_EMPTY_HEADERS, name="o.sc")
    assert parse_score_file(only_desc) == 0.0


def test_parse_relax_score_structured_form(tmp_path: Path) -> None:
    """The new ``parse_relax_score`` returns the full structured
    score with every named metric; absent columns are ``None``.
    """
    p = _write_scorefile(tmp_path, SCORE_MINIMAL)
    score = parse_relax_score(p)
    assert score.total_score == pytest.approx(-123.456)
    for field_name in (
        "total_sasa",
        "delta_sasa",
        "hydrophobic_sasa",
        "polar_sasa",
        "interface_polar_sasa",
        "interface_hydrophobic_sasa",
        "interface_dG",
        "interface_dSASA",
        "buried_unsatisfied_hbonds",
        "cross_interface_hbonds",
        "hbond_energy",
        "hbond_energy_fraction",
        "shape_complementarity",
        "packstat",
    ):
        assert getattr(score, field_name) is None, (
            f"{field_name} should be None on minimal fixture; got {getattr(score, field_name)!r}"
        )


# Deduplicated: one parametrized pin per metric, covering both the
# canonical-header and alias-header fixtures with a single
# parametrized table (same numerical values, just different header
# names). A shifted-column mutation cannot satisfy two assertions
# because every value is distinct.
_CANONICAL_FIXTURE_PINS = [
    ("total_score", -123.456),
    ("total_sasa", 12450.7),
    ("delta_sasa", -850.4),
    ("hydrophobic_sasa", 7320.1),
    ("polar_sasa", 5130.6),
    # Interface-polar / interface-hydrophobic SASA populate the
    # dedicated ``interface_*_sasa`` metrics, NOT the total
    # polar/hydrophobic SASAs (distinct science).
    ("interface_polar_sasa", 1000.0),
    ("interface_hydrophobic_sasa", 6320.1),
    ("interface_dG", -45.231),
    ("interface_dSASA", -212.004),
    ("buried_unsatisfied_hbonds", 3.0),
    ("cross_interface_hbonds", 12.0),
    ("hbond_energy", -8.450),
    ("shape_complementarity", 0.712),
    ("packstat", 0.65),
]


@pytest.mark.parametrize(("metric", "expected"), _CANONICAL_FIXTURE_PINS)
def test_parse_relax_score_full_canonical_pins_every_metric(
    tmp_path: Path, metric: str, expected: float
) -> None:
    """Every metric the runner can expose is pinned to a specific
    token value (canonical fixture, full data row).
    """
    p = _write_scorefile(tmp_path, SCORE_FULL_CANONICAL)
    assert getattr(parse_relax_score(p), metric) == pytest.approx(expected)


@pytest.mark.parametrize(("metric", "expected"), _CANONICAL_FIXTURE_PINS)
def test_parse_relax_score_full_alias_pins_every_metric(
    tmp_path: Path, metric: str, expected: float
) -> None:
    """Same metric values as the canonical fixture but every
    column uses a non-canonical alias. A regression that
    mis-routes ``packstat`` to ``shape_complementarity`` or
    ``hbond_E_fraction`` to ``hbond_energy`` would break here
    for that exact metric.
    """
    p = _write_scorefile(tmp_path, SCORE_FULL_ALIASES)
    assert getattr(parse_relax_score(p), metric) == pytest.approx(expected)


def test_parse_relax_score_hbond_energy_fraction_is_distinct(tmp_path: Path) -> None:
    """The alias fixture emits BOTH ``hbond_E_int`` (absolute
    energy) and ``hbond_E_fraction`` (fractional). They map to
    DISTINCT fields — ``v0.7`` disambiguation — so the parser
    surfaces them as two separate floats with two different
    values. A v0.6 conflation regression would drop the
    ``hbond_energy_fraction`` value into ``hbond_energy`` and
    leave one of the two tests below reading ``-8.450``.
    """
    p = _write_scorefile(tmp_path, SCORE_FULL_ALIASES)
    score = parse_relax_score(p)
    assert score.hbond_energy == pytest.approx(-8.450)
    assert score.hbond_energy_fraction == pytest.approx(0.45)
    # And both can be populated simultaneously.
    assert score.hbond_energy != score.hbond_energy_fraction


def test_parse_relax_score_handles_garbage(tmp_path: Path) -> None:
    p = _write_scorefile(tmp_path, SCORE_GARBAGE)
    assert parse_relax_score(p) == RelaxScore()


def test_parse_relax_score_handles_empty(tmp_path: Path) -> None:
    p = _write_scorefile(tmp_path, "")
    assert parse_relax_score(p) == RelaxScore()


def test_parse_relax_score_preserves_missing_columns_as_none(tmp_path: Path) -> None:
    """Lock the missing-column contract end-to-end: present columns
    get values, absent columns stay ``None`` (no ``0.0`` default).
    """
    p = _write_scorefile(tmp_path, SCORE_MINIMAL)
    score = parse_relax_score(p)
    assert score.total_score == pytest.approx(-123.456)
    for metric in (
        "total_sasa",
        "delta_sasa",
        "hydrophobic_sasa",
        "polar_sasa",
        "interface_polar_sasa",
        "interface_hydrophobic_sasa",
        "interface_dG",
        "interface_dSASA",
        "buried_unsatisfied_hbonds",
        "cross_interface_hbonds",
        "hbond_energy",
        "hbond_energy_fraction",
        "shape_complementarity",
        "packstat",
    ):
        assert getattr(score, metric) is None, (
            f"{metric} should be None when the column is absent; got {getattr(score, metric)!r}"
        )


def test_parse_relax_score_handles_truncated_data_row(tmp_path: Path) -> None:
    """Header lists four columns but the data row carries only two.
    Parsed columns get values; the rest stay ``None``.
    """
    p = _write_scorefile(tmp_path, SCORE_TRUNCATED_DATA)
    score = parse_relax_score(p)
    assert score.total_score == pytest.approx(-7.5)
    assert score.total_sasa == pytest.approx(9000.0)
    assert score.delta_sasa is None
    assert score.interface_dG is None


def test_parse_relax_score_empty_header_row_yields_all_none(tmp_path: Path) -> None:
    p = _write_scorefile(tmp_path, SCORE_EMPTY_HEADERS)
    assert parse_relax_score(p) == RelaxScore()


def test_parse_relax_score_rejects_nonfinite_tokens(tmp_path: Path) -> None:
    """``inf`` / ``-inf`` / ``nan`` sentinel tokens are rejected
    and surfaced as ``None``.
    """
    fixture = (
        "SEQUENCE:\nSCORE: total_score sasa delta_sasa description\nSCORE: 5.0 inf nan input.pdb\n"
    )
    p = _write_scorefile(tmp_path, fixture)
    score = parse_relax_score(p)
    assert score.total_score == pytest.approx(5.0)
    assert score.total_sasa is None
    assert score.delta_sasa is None


# ---------------------------------------------------------------------------
# Canonical InterfaceAnalyzer aliases — distinct semantic mapping.
# ---------------------------------------------------------------------------


_V06_PIN_PAIRS = [
    ("interface_dSASA", -100.0),  # dSASA_int
    ("interface_polar_sasa", 200.0),  # dSASA_polar
    ("interface_hydrophobic_sasa", 800.0),  # dSASA_hphobic
    ("buried_unsatisfied_hbonds", 2.0),  # delta_unsatHbonds
    ("hbond_energy_fraction", -5.0),  # hbond_E_fraction
    ("packstat", 0.65),  # packstat
]


@pytest.mark.parametrize(("metric", "expected"), _V06_PIN_PAIRS)
def test_parse_relax_score_resolves_v06_aliases(
    tmp_path: Path, metric: str, expected: float
) -> None:
    """Verify each v0.6/v0.7 InterfaceAnalyzer alias resolves to a
    DISTINCT canonical metric (no further semantic conflation).
    """
    p = _write_scorefile(tmp_path, SCORE_NEW_ALIASES)
    assert getattr(parse_relax_score(p), metric) == pytest.approx(expected)


def test_metric_aliases_table_covers_every_canonical_metric() -> None:
    """Sanity-check the alias table: every canonical metric has at
    least one alias, no alias is empty, and no two metrics share an
    alias (collisions are a silent ambiguity bug).
    """
    seen: set[str] = set()
    for metric, aliases in METRIC_ALIASES.items():
        assert aliases, f"METRIC_ALIASES[{metric!r}] has no aliases"
        for alias in aliases:
            assert alias, f"METRIC_ALIASES[{metric!r}] contains empty alias"
            assert alias.lower() not in seen, (
                f"alias {alias!r} is registered under multiple metrics — ambiguous header lookup"
            )
            seen.add(alias.lower())


def test_metric_aliases_includes_all_required_metrics() -> None:
    """Lock the metric set the runner exposes downstream — adding
    or removing a metric requires updating this test on purpose.
    """
    expected = {
        "total_score",
        "total_sasa",
        "delta_sasa",
        "hydrophobic_sasa",
        "polar_sasa",
        "interface_polar_sasa",
        "interface_hydrophobic_sasa",
        "interface_dG",
        "interface_dSASA",
        "buried_unsatisfied_hbonds",
        "cross_interface_hbonds",
        "hbond_energy",
        "hbond_energy_fraction",
        "shape_complementarity",
        "packstat",
    }
    assert set(METRIC_ALIASES) == expected


def test_metric_aliases_drops_bare_dG_and_sc() -> None:
    """Bare ``dG`` and ``sc`` are too ambiguous to accept (could be
    interface or whole-complex, shape complementarity or supercharge).
    Explicitly check that neither alias sneaks back into the table.
    """
    flat = {alias for aliases in METRIC_ALIASES.values() for alias in aliases}
    assert "dG" not in flat, (
        "bare 'dG' alias risks ambiguous metric assignment; "
        "use dG_separated / dG_cross / dG_int for an explicit binding"
    )
    assert "sc" not in flat, (
        "bare 'sc' alias risks ambiguous metric assignment; "
        "use interface_sc / sc_value for an explicit shape complementarity"
    )


def test_metric_aliases_exposes_v06_aliases() -> None:
    """Pin the v0.6 InterfaceAnalyzer aliases explicitly so a
    future refactor can't quietly drop them.
    """
    flat = {alias for aliases in METRIC_ALIASES.values() for alias in aliases}
    for alias in (
        "dSASA_int",
        "dSASA_polar",
        "dSASA_hphobic",
        "delta_unsatHbonds",
        "hbond_E_fraction",
        "packstat",
    ):
        assert alias in flat, f"{alias!r} should be registered in METRIC_ALIASES"


# ---------------------------------------------------------------------------
# utils — RelaxScore / RelaxRecord
# ---------------------------------------------------------------------------


def test_relax_score_to_dict_emits_native_numbers() -> None:
    """Structured :class:`RelaxScore` serialization uses native
    ``float`` / ``None`` — NOT string reprs — so downstream JSON
    consumers don't need a float-parsing detour.
    """
    score = RelaxScore(total_score=-99.5, packstat=0.65)
    payload = score.to_dict()
    assert payload["total_score"] == -99.5
    assert isinstance(payload["total_score"], float)
    assert payload["packstat"] == 0.65
    assert payload["interface_dG"] is None
    # All None-vs-present semantics are preserved.
    none_count = sum(1 for v in payload.values() if v is None)
    assert none_count == len(payload) - 2


def test_relax_record_defaults_to_empty_score_legacy_compat() -> None:
    """Backward-compat: default-constructed record carries the v0.5
    positional-args defaults (``total_score=0.0``, no ``score``).
    """
    record = RelaxRecord(index=0, path="/tmp/score.sc")
    assert record.total_score == 0.0
    assert record.score == RelaxScore()
    assert record.status == RelaxRecordStatus.SUCCEEDED
    assert record.error == ""


def test_relax_record_legacy_positional_constructor_preserved() -> None:
    """Source-compat: pre-v0.6 positional construction with
    ``(index, path, total_score, status, error)`` continues to
    bind to those fields verbatim, with the additive ``score`` at
    the end (matches the v0.5 signature order).
    """
    record = RelaxRecord(2, "/tmp/score.sc", -99.5, "succeeded", "")
    assert record.index == 2
    assert record.path == "/tmp/score.sc"
    assert record.total_score == pytest.approx(-99.5)
    assert record.status == RelaxRecordStatus.SUCCEEDED
    assert record.error == ""
    # Additive struct defaults to empty when not passed positionally.
    assert record.score == RelaxScore()
    assert record.score.total_score is None


def test_relax_record_legacy_kwargs_still_accepted() -> None:
    """Source-compat: the legacy kwargs form keeps working too."""
    record = RelaxRecord(
        index=2,
        path="/tmp/score.sc",
        total_score=-99.5,
        status="succeeded",
    )
    assert record.total_score == pytest.approx(-99.5)
    assert record.score == RelaxScore()


def test_relax_record_to_dict_legacy_top_level_total_score_stays_string() -> None:
    """The v0.5 ``payload["total_score"]`` form is preserved as a
    string repr (legacy source compat) even when the structured
    form below uses a native number.
    """
    record = RelaxRecord(
        index=2,
        path="/tmp/score.sc",
        total_score=-99.5,
        score=RelaxScore(total_score=-99.5),
    )
    payload = record.to_dict()
    # Legacy key: float-as-string repr (matches v0.5).
    assert payload["index"] == "2"
    assert payload["path"] == "/tmp/score.sc"
    assert payload["total_score"] == "-99.5"
    assert isinstance(payload["total_score"], str)
    assert payload["status"] == RelaxRecordStatus.SUCCEEDED
    assert payload["error"] == ""
    # New structured key: native numbers / None.
    assert payload["metrics"]["total_score"] == -99.5
    assert isinstance(payload["metrics"]["total_score"], float)
    assert payload["metrics"]["interface_dG"] is None


def test_relax_record_failed_status_round_trip() -> None:
    record = RelaxRecord(
        index=0,
        path="/tmp/score.sc",
        score=RelaxScore(),
        status=RelaxRecordStatus.FAILED,
        error="bad score file",
    )
    payload = record.to_dict()
    assert payload["status"] == RelaxRecordStatus.FAILED
    assert payload["error"] == "bad score file"
    assert payload["metrics"]["total_score"] is None
    # FAILED records report ``total_score`` as the legacy "0.0"
    # string by convention; the structured field is ``None``.
    assert payload["total_score"] == "0.0"


def test_rosetta_available_returns_false_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROSETTA_BIN", "/nonexistent/rosetta_scripts")
    assert rosetta_available() is False


# ---------------------------------------------------------------------------
# utils — invoke() argv flattening (THE regression)
# ---------------------------------------------------------------------------


def test_invoke_repeats_parser_script_vars_per_token(tmp_path: Path) -> None:
    """Lock the argv shape Rosetta expects: each ``parser:script_vars``
    token is its own argv element, not a single space-joined
    argument. A regression that re-introduces space-joined
    arguments would let Rosetta's parser mis-interpret the
    trailing ``"k1=v1 k2=v2"`` as one variable.
    """
    captured_argv: list[str] = []

    def fake_run(cmd: list[str], **_: Any) -> mock_mod.Mock:
        # Capture exactly what invoke() handed to subprocess.run.
        captured_argv.extend(cmd)
        result = mock_mod.Mock()
        result.returncode = 0
        return result

    with mock_mod.patch("subprocess.run", side_effect=fake_run):
        exit_code = invoke(
            config={
                "s": "/x.xml",
                "in:file:s": "/x.pdb",
                "out:path:all": str(tmp_path),
                "nstruct": "1",
                # Three tokens, sequence value — each must become
                # a separate argv element after a repeated flag.
                "parser:script_vars": [
                    "prep_mode=cyclic",
                    "constrain_to_start_coords=1",
                    "relax_cycles=4",
                ],
            },
            output_dir=tmp_path,
            binary_prefix=["rosetta_scripts"],
            timeout_seconds=10,
        )
    assert exit_code == 0

    assert captured_argv[0] == "rosetta_scripts"
    assert "--parser" in captured_argv
    assert "protocol" in captured_argv

    # Three occurrences of -parser:script_vars, one per token.
    occurrences = [i for i, tok in enumerate(captured_argv) if tok == "-parser:script_vars"]
    assert len(occurrences) == 3, captured_argv
    # Each token immediately follows its flag.
    assert captured_argv[occurrences[0] + 1] == "prep_mode=cyclic"
    assert captured_argv[occurrences[1] + 1] == "constrain_to_start_coords=1"
    assert captured_argv[occurrences[2] + 1] == "relax_cycles=4"
    # No element contains whitespace (no space-joined args).
    for tok in captured_argv:
        assert " " not in tok, (
            f"argv element {tok!r} contains whitespace — the "
            "script_vars tokens were space-joined instead of "
            "flattened into separate argv elements"
        )


def test_invoke_string_value_emits_single_argv_token(tmp_path: Path) -> None:
    """Verify the non-special case is still single-token: an
    ordinary flag with a string value produces ``-flag value``,
    not the multi-value sequence form.
    """
    captured_argv: list[str] = []

    def fake_run(cmd: list[str], **_: Any) -> mock_mod.Mock:
        captured_argv.extend(cmd)
        result = mock_mod.Mock()
        result.returncode = 0
        return result

    with mock_mod.patch("subprocess.run", side_effect=fake_run):
        invoke(
            config={
                "in:file:s": "/x.pdb",
                "nstruct": "3",
            },
            output_dir=tmp_path,
            binary_prefix=["rosetta_scripts"],
            timeout_seconds=10,
        )

    # ``-in:file:s /x.pdb`` (single value).
    idx = captured_argv.index("-in:file:s")
    assert captured_argv[idx + 1] == "/x.pdb"
    # ``-nstruct 3`` (single value).
    nidx = captured_argv.index("-nstruct")
    assert captured_argv[nidx + 1] == "3"


# ---------------------------------------------------------------------------
# config -> CLI translation
# ---------------------------------------------------------------------------


def test_config_to_cli_includes_required_flags() -> None:
    cli = _config_to_cli(_valid_config())
    assert cli["s"] == "/opt/rosetta/scripts/relax.xml"
    assert cli["in:file:s"] == "/tmp/input.pdb"
    assert cli["nstruct"] == "1"


def test_config_to_cli_includes_extra_flags() -> None:
    cfg = _valid_config(
        extra_flags=("beta=1.0", "no_output"),
        extra={"gamma": 2.0},
    )
    cli = _config_to_cli(cfg)
    assert cli["beta"] == "1.0"
    assert "no_output" in cli
    assert cli["gamma"] == "2.0"


def test_config_to_cli_emits_script_vars_as_list_for_linear() -> None:
    """The internal CLI representation stores ``parser:script_vars``
    as a list so :func:`invoke` can flatten each entry into its
    own argv element.
    """
    cli = _config_to_cli(_valid_config(preparation_mode="linear"))
    assert cli["parser:script_vars"] == ["prep_mode=linear"]


def test_config_to_cli_emits_script_vars_as_list_for_cyclic() -> None:
    cli = _config_to_cli(_valid_config(preparation_mode="cyclic"))
    assert cli["parser:script_vars"] == ["prep_mode=cyclic"]


def test_config_to_cli_omits_script_vars_when_no_structured_options() -> None:
    """Without ``preparation_mode`` or ``constrained_relax``, no
    ``parser:script_vars`` key is emitted at all.
    """
    cfg = _valid_config()
    cli = _config_to_cli(cfg)
    assert "parser:script_vars" not in cli


def test_config_to_cli_translates_constrained_relax_options() -> None:
    opts = ConstrainedRelaxOptions(
        constrain_to_start_coords=True,
        ramp_constraints=False,
        coord_constrain_sidechains=True,
        relax_cycles=5,
        bb_min_only=False,
    )
    cfg = _valid_config(constrained_relax=opts)
    cli = _config_to_cli(cfg)
    assert cli["parser:script_vars"] == [
        "constrain_to_start_coords=1",
        "ramp_constraints=0",
        "coord_constrain_sidechains=1",
        "relax_cycles=5",
        "bb_min_only=0",
    ]


def test_config_to_cli_emits_only_set_constrained_relax_fields() -> None:
    """Partial ``ConstrainedRelaxOptions`` produce only the set
    keys — no spurious ``None`` translations.
    """
    opts = ConstrainedRelaxOptions(relax_cycles=3, ramp_constraints=True)
    cfg = _valid_config(constrained_relax=opts)
    cli = _config_to_cli(cfg)
    assert cli["parser:script_vars"] == ["ramp_constraints=1", "relax_cycles=3"]


def test_config_to_cli_merges_prep_mode_and_constrained_relax() -> None:
    """Both structured knobs contribute to the same list (in order)."""
    cfg = _valid_config(
        preparation_mode="cyclic",
        constrained_relax=ConstrainedRelaxOptions(relax_cycles=2),
    )
    cli = _config_to_cli(cfg)
    assert cli["parser:script_vars"] == ["prep_mode=cyclic", "relax_cycles=2"]


def test_config_to_cli_extra_str_value_whitespace_splits_into_tokens() -> None:
    """Dict-channel ``parser:script_vars`` accepts the user-friendly
    space-separated string form AND the explicit list form
    interchangeably; both produce separate script-var tokens.
    """
    cfg = _valid_config(extra={"parser:script_vars": "k1=v1 k2=v2"})
    cli = _config_to_cli(cfg)
    assert cli["parser:script_vars"] == ["k1=v1", "k2=v2"]


def test_config_to_cli_extra_list_value_passes_through() -> None:
    """Dict-channel list values are preserved verbatim — each list
    element becomes one script-var token.
    """
    cfg = _valid_config(extra={"parser:script_vars": ["a=1", "b=2", "c=3"]})
    cli = _config_to_cli(cfg)
    assert cli["parser:script_vars"] == ["a=1", "b=2", "c=3"]


def test_config_to_cli_precedence_structured_then_extra_then_extra_flags() -> None:
    """Pin the precedence order for ``parser:script_vars``:
    structured → ``extra`` dict-channel → ``extra_flags`` trailing-
    tokens form, in that order. None clobbers another.
    """
    cfg = _valid_config(
        preparation_mode="linear",
        extra={"parser:script_vars": "from_extra=e"},
        extra_flags=("parser:script_vars from_flag=f",),
    )
    cli = _config_to_cli(cfg)
    assert cli["parser:script_vars"] == [
        "prep_mode=linear",
        "from_extra=e",
        "from_flag=f",
    ]


def test_config_to_cli_extra_flags_with_script_vars_does_not_clobber() -> None:
    """Regression: an ``extra_flags`` entry carrying the
    ``parser:script_vars`` prefix used to silently drop
    structured-config tokens. Now: merge, not clobber.
    """
    cfg = _valid_config(
        constrained_relax=ConstrainedRelaxOptions(relax_cycles=2),
        extra_flags=("beta=1.0", "parser:script_vars custom=c"),
    )
    cli = _config_to_cli(cfg)
    assert cli["beta"] == "1.0"
    assert cli["parser:script_vars"] == ["relax_cycles=2", "custom=c"]


def test_config_to_cli_empty_constrained_relax_emits_no_script_vars() -> None:
    """Edge case: an explicit empty ``ConstrainedRelaxOptions``
    contributes no tokens. The protocol XML inherits the
    upstream defaults.
    """
    cfg = _valid_config(constrained_relax=ConstrainedRelaxOptions())
    cli = _config_to_cli(cfg)
    assert "parser:script_vars" not in cli


# ---------------------------------------------------------------------------
# runner behaviour
# ---------------------------------------------------------------------------


def test_runner_dry_run_does_not_invoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    invoked: list[dict[str, Any]] = []

    def fake_invoke(**_: Any) -> int:
        invoked.append({})
        return 0

    monkeypatch.setattr("biolab_runners.rosetta.runner.invoke", fake_invoke)
    runner = RosettaRunner(
        output_root=tmp_path,
        config=_valid_config(name="dry"),
    )
    result = runner.run(dry_run=True)
    assert invoked == []
    assert result.exit_code == 0


def test_runner_idempotent_when_score_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a ``score.sc`` already exists, the runner parses it
    rather than re-invoking Rosetta, and the cached records carry
    the structured ``score`` form too.
    """
    monkeypatch.setattr("biolab_runners.rosetta.runner.invoke", lambda **_: 0)
    name = "idem"
    design_dir = tmp_path / name
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "score.sc").write_text(SCORE_FULL_CANONICAL)

    config = _valid_config(name=name, output_dir=str(design_dir))
    runner = RosettaRunner(output_root=tmp_path, config=config)
    result = runner.run(config)
    assert result.skipped == 1
    assert result.records[0].total_score == pytest.approx(-123.456)
    assert result.records[0].score.total_score == pytest.approx(-123.456)


def test_runner_records_per_design(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_invoke(*, output_dir: Path, **_: Any) -> int:
        (output_dir / "score.sc").write_text(SCORE_FULL_CANONICAL)
        return 0

    monkeypatch.setattr("biolab_runners.rosetta.runner.invoke", fake_invoke)
    runner = RosettaRunner(output_root=tmp_path, config=_valid_config(name="batch"))
    result = runner.run()
    assert result.succeeded == 1
    assert result.records[0].total_score == pytest.approx(-123.456)
    assert result.records[0].score.interface_dSASA == pytest.approx(-212.004)


def test_runner_records_count_failed_by_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached ``score.sc`` files that fail to parse are counted in
    ``result.failed`` (per status, not per file existence).
    """
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "score.sc").write_text("not a score\n")
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    (good_dir / "score.sc").write_text(SCORE_FULL_CANONICAL)

    monkeypatch.setattr("biolab_runners.rosetta.runner.invoke", lambda **_: 0)
    runner = RosettaRunner(output_root=tmp_path)
    bad_result = runner.run(_valid_config(name="bad", output_dir=str(bad_dir)))
    assert bad_result.succeeded == 0
    assert bad_result.failed == 1
    assert bad_result.records[0].status == RelaxRecordStatus.FAILED

    good_result = runner.run(_valid_config(name="good", output_dir=str(good_dir)))
    assert good_result.succeeded == 1
    assert good_result.failed == 0


def test_runner_propagates_nonzero_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("biolab_runners.rosetta.runner.invoke", lambda **_: 7)
    runner = RosettaRunner(output_root=tmp_path, config=_valid_config(name="failure"))
    assert runner.run().exit_code == 7


def test_runner_requires_config() -> None:
    runner = RosettaRunner(output_root=Path("/tmp"))
    with pytest.raises(ValueError, match="RosettaConfig is required"):
        runner.run()


# ---------------------------------------------------------------------------
# parse_score_files (the orchestrator)
# ---------------------------------------------------------------------------


def test_parse_score_files_picks_first_data_row_per_file(tmp_path: Path) -> None:
    paths = [
        _write_scorefile(tmp_path, SCORE_FULL_CANONICAL, name="a.sc"),
        _write_scorefile(tmp_path, SCORE_FULL_ALIASES, name="b.sc"),
        _write_scorefile(tmp_path, SCORE_MINIMAL, name="c.sc"),
    ]
    records = parse_score_files(paths)
    assert len(records) == 3
    # Both fixtures share the same numerical values; the alias test
    # proves the alias map is transparent, the canonical test pins
    # the columns themselves.
    assert records[0].total_score == pytest.approx(-123.456)
    assert records[1].total_score == pytest.approx(-123.456)
    # Minimal fixture has no interface_dG column.
    assert records[2].score.interface_dG is None


def test_parse_score_files_marks_all_none_as_failed(tmp_path: Path) -> None:
    """Lock the all-None / garbage contract: ``parse_score_files``
    surfaces an unparseable row as a FAILED record with an empty
    :class:`RelaxScore` and a non-empty ``error`` string.
    """
    paths = [
        _write_scorefile(tmp_path, SCORE_FULL_CANONICAL, name="good.sc"),
        _write_scorefile(tmp_path, SCORE_GARBAGE, name="garbage.sc"),
        _write_scorefile(tmp_path, SCORE_EMPTY_HEADERS, name="empty.sc"),
    ]
    records = parse_score_files(paths)
    assert len(records) == 3
    assert records[0].status == RelaxRecordStatus.SUCCEEDED
    assert records[0].score.total_score == pytest.approx(-123.456)
    assert records[1].status == RelaxRecordStatus.FAILED
    assert records[1].score == RelaxScore()
    assert records[1].total_score == 0.0
    assert records[1].error
    assert records[2].status == RelaxRecordStatus.FAILED
    assert records[2].score == RelaxScore()


def test_parse_score_files_handles_missing_file(tmp_path: Path) -> None:
    records = parse_score_files([tmp_path / "does-not-exist.sc"])
    assert len(records) == 1
    assert records[0].status == RelaxRecordStatus.FAILED
    assert records[0].score == RelaxScore()
    assert records[0].error
