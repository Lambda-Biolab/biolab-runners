"""CLI + availability helpers for the Rosetta runner."""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "METRIC_ALIASES",
    "RelaxRecord",
    "RelaxRecordStatus",
    "RelaxScore",
    "parse_relax_score",
    "parse_score_file",
    "parse_score_files",
    "rosetta_available",
]


# ---------------------------------------------------------------------------
# Scorefile parser
# ---------------------------------------------------------------------------
#
# A Rosetta ``score.sc`` file is whitespace-delimited with two header
# lines (``SEQUENCE:`` and ``SCORE: <column names>``) followed by data
# rows prefixed with ``SCORE:``. Column names vary across score functions
# and apps (InterfaceAnalyzer, FastRelax, RBSscreener), so the parser
# maps a curated set of canonical scientific metrics to all of the
# recognized upstream column names. Missing columns are preserved as
# ``None`` so the caller can distinguish "column absent" from "column
# present and zero."
#
# Aliases are lowercased before lookup; comparison is case-insensitive
# so a scorefile written as ``Total_Score`` matches ``total_score``.
# Within an alias tuple the canonical metric is the source of truth —
# every alias is just a column-name variant the upstream score-writer
# might emit.
#
# Semantic-mapping notes (per the upstream ``InterfaceAnalyzer`` default
# report):
#
#   - ``dSASA_int`` / ``dSASA_iface`` / ``interface_dSASA`` are all the
#     same metric: buried SASA per interface residue. (Aliases for
#     ``interface_dSASA``.)
#   - ``delta_sasa`` / ``dsasa`` / ``dSASA`` are the COMPLEX-MINUS-
#     PARTNERS total delta SASA, NOT the per-residue interface value.
#     (Aliases for ``delta_sasa``.)
#   - ``dSASA_polar`` is the buried POLAR SASA at the interface —
#     distinct from ``polar_sasa`` (which is total polar SASA). Maps
#     to the dedicated ``interface_polar_sasa`` metric.
#   - ``dSASA_hphobic`` is the buried HYDROPHOBIC SASA at the
#     interface — distinct from ``hydrophobic_sasa``. Maps to the
#     dedicated ``interface_hydrophobic_sasa`` metric.
#   - ``delta_unsatHbonds`` is the change-in-unsatisfied-polar count
#     upon complexation. (Alias for ``buried_unsatisfied_hbonds``.)
#   - ``hbond_E_fraction`` is the FRACTIONAL H-bond energy contribution
#     to total score, distinct from ``hbond_E_int``/``hbond_energy``
#     which carry absolute energy. Maps to the dedicated
#     ``hbond_energy_fraction`` metric.
#   - ``packstat`` is the Lawrence & Coleman packing statistic,
#     distinct from shape complementarity. Maps to the dedicated
#     ``packstat`` metric.
#
# Bare / ambiguous aliases that are intentionally NOT accepted:
#
#   - ``dG`` — too broad (could be interface or whole-complex dG).
#   - ``sc`` — too broad (could be shape complementarity, supercharge,
#     score column, etc.).
#
# Callers should pass one of the scoped aliases (e.g. ``dG_separated``,
# ``interface_sc``) instead. Adding a bare alias would risk silent
# metric mis-assignment on consumer-side parsing.


# Canonical metric -> tuple of recognized column aliases (any case).
# The first alias that resolves wins; downstream code keeps the
# canonical key, not the alias.
METRIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    "total_score": ("total_score", "score", "total"),
    "total_sasa": ("total_sasa", "sasa", "fa_sasa"),
    "delta_sasa": (
        "delta_sasa",
        "dsasa",
        "delta_sasa_int",
        "sasa_delta",
        "sasa_int",
        "complex_delta_sasa",
    ),
    "hydrophobic_sasa": (
        "hydrophobic_sasa",
        "sasa_hydrophobic",
        "sasa_hphobic",
        "sasa_hphob",
    ),
    "polar_sasa": ("polar_sasa", "sasa_polar", "sasa_pol"),
    # Distinct from ``polar_sasa`` (total polar SASA) — this is buried
    # polar SASA at the interface.
    "interface_polar_sasa": (
        "interface_polar_sasa",
        "dSASA_polar",
        "polar_int_sasa",
    ),
    # Distinct from ``hydrophobic_sasa`` (total hydrophobic SASA) —
    # this is buried hydrophobic SASA at the interface.
    "interface_hydrophobic_sasa": (
        "interface_hydrophobic_sasa",
        "dSASA_hphobic",
        "hphobic_int_sasa",
    ),
    "interface_dG": (
        "interface_dG",
        "dG_separated",
        "sc_dG_separated",
        "dG_cross",
        "dG_int",
        "binding_dG",
        # Bare ``dG`` deliberately omitted (too ambiguous between
        # interface and whole-complex free energy); use
        # ``dG_separated`` or another scoped alias.
    ),
    "interface_dSASA": (
        "interface_dSASA",
        "dSASA_iface",
        "delta_sasa_iface",
        "sc_dSASA",
        "dSASA_int",
    ),
    "buried_unsatisfied_hbonds": (
        "buried_unsatisfied_hbonds",
        "n_unsat_hbonds",
        "unsat_hbonds",
        "unsat_hbond",
        "buried_unsatisfied_polars",
        "buried_unsat",
        "delta_unsatHbonds",
    ),
    "cross_interface_hbonds": (
        "cross_interface_hbonds",
        "hbonds_int",
        "interface_hbonds",
        "hbond_int",
        "interface_hbond_count",
    ),
    "hbond_energy": (
        "hbond_energy",
        "hbond_E_int",
        "env_hbond",
        "interface_hbond_energy",
    ),
    # Distinct from ``hbond_energy`` (absolute energy) — the
    # fractional H-bond contribution to total score. Range
    # typically ``[0, 1]``.
    "hbond_energy_fraction": (
        "hbond_energy_fraction",
        "hbond_E_fraction",
    ),
    "shape_complementarity": (
        "shape_complementarity",
        "interface_sc",
        "sc_value",
        # Bare ``sc`` deliberately omitted (too ambiguous — could be
        # shape complementarity, ``score`` column, supercharge, etc.);
        # use ``interface_sc`` or ``sc_value``.
    ),
    # Lawrence & Coleman packing statistic — a separate interface
    # quality metric from shape complementarity, despite the
    # related subject matter. Range typically ``[0, 1]``.
    "packstat": ("packstat",),
}


class RelaxRecordStatus:
    """Normalized outcome values for per-structure relax records."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


# Integer-typed counts come back as floats in upstream scorefiles; the
# dataclass keeps them as ``float | None`` to preserve the raw value
# the writer emitted. Downstream code that wants an int can call
# ``int(...)``; see tests for the expected behavior.
@dataclass(frozen=True)
class RelaxScore:
    """Structured scientific metrics parsed from a Rosetta ``score.sc``.

    Every field is the parsed value from the first data row, or
    ``None`` when the scorefile did not include a recognized column
    for that metric. Callers must check ``is not None`` before
    treating any field as a number — the parser never fabricates a
    value for a missing column.

    Fields are populated from the canonical ``METRIC_ALIASES`` table;
    alias column names collapse into the canonical metric they share
    semantics with (see the docstring above). New metrics can be
    added by extending :data:`METRIC_ALIASES` — there is no separate
    registry to keep in sync.
    """

    total_score: float | None = None
    total_sasa: float | None = None
    delta_sasa: float | None = None
    hydrophobic_sasa: float | None = None
    polar_sasa: float | None = None
    interface_polar_sasa: float | None = None
    interface_hydrophobic_sasa: float | None = None
    # ``interface_dG`` / ``interface_dSASA`` keep the upstream
    # InterfaceAnalyzer case convention ("G" for Gibbs, "dSASA" for
    # delta-SASA). N815 (mixedCase) noqa: the names are fixed by
    # upstream metric convention; renaming to lowercase would break
    # the alias table reader.
    interface_dG: float | None = None  # noqa: N815
    interface_dSASA: float | None = None  # noqa: N815
    buried_unsatisfied_hbonds: float | None = None
    cross_interface_hbonds: float | None = None
    hbond_energy: float | None = None
    hbond_energy_fraction: float | None = None
    shape_complementarity: float | None = None
    packstat: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the score into a JSON-safe dictionary.

        Numeric fields are emitted as native Python ``float`` /
        ``None`` so downstream consumers receive real numbers (not
        string reprs) when the value is present. ``None`` is
        preserved as ``None`` for absent columns.
        """
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            result[f.name] = None if value is None else float(value)
        return result


@dataclass(frozen=True)
class RelaxRecord:
    """One relax output produced by ``rosetta_scripts``.

    Field order is the v0.5 backward-compat ordering:
    ``index, path, total_score, status, error`` followed by the new
    additive ``score`` field at the end. Legacy callers using
    positional construction (e.g. ``RelaxRecord(0, "x.sc", -99.5,
    "succeeded", "")``) continue to work byte-for-byte.

    ``total_score`` and ``score`` are populated independently by
    :func:`parse_score_files`; legacy callers that read
    ``record.total_score`` continue to get the same float the
    pre-v0.6 parser would have returned.
    """

    index: int
    path: str
    total_score: float = 0.0
    status: str = RelaxRecordStatus.SUCCEEDED
    error: str = ""
    # Additive: the structured :class:`RelaxScore` form appended to
    # the end of the legacy field set so positional construction
    # remains source-compatible.
    score: RelaxScore = field(default_factory=RelaxScore)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record into a JSON-safe dictionary.

        Retains the legacy v0.5 keys (``index`` / ``path`` /
        ``total_score`` / ``status`` / ``error``) byte-for-byte and
        adds the new ``metrics`` key with the structured
        :class:`RelaxScore` form (v0.6 additive change). Consumers
        that only read the legacy keys see no behavior change.

        The legacy ``total_score`` keeps its float-as-string repr
        for source compat with v0.5 consumers that JSON-decode the
        field; the new ``metrics`` keys are emitted as native
        numbers / ``None`` so downstream JSON consumers don't need a
        float-parsing detour for the structured form.
        """
        return {
            "index": str(self.index),
            "path": self.path,
            # Legacy float-as-string form; preserved for callers that
            # read it as ``payload["total_score"]``.
            "total_score": repr(self.total_score),
            # New structured form (v0.6). Additive: existing callers
            # that don't read ``metrics`` see no change. The values
            # are native ``float`` / ``None`` — string reprs only
            # live on the legacy ``total_score`` key above.
            "metrics": self.score.to_dict(),
            "status": self.status,
            "error": self.error,
        }


def rosetta_available(timeout_seconds: int = 30) -> bool:
    """Return True when the upstream Rosetta CLI is callable."""
    import os

    binary = os.environ.get("ROSETTA_BIN", "rosetta_scripts")
    if binary.startswith("container://"):
        return True
    if shutil.which(binary) is None:
        return False
    try:
        completed = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _resolved_binary() -> list[str]:
    """Return the command prefix used to invoke ``rosetta_scripts``."""
    import os

    binary = os.environ.get("ROSETTA_BIN", "rosetta_scripts")
    if binary.startswith("container://"):
        spec = binary[len("container://") :]
        runtime = os.environ.get("CONTAINER_RUNTIME", "docker")
        return [runtime, "run", "--rm", spec, "rosetta_scripts"]
    return [binary]


# ---------------------------------------------------------------------------
# Scorefile parsing
# ---------------------------------------------------------------------------


def parse_score_file(path: Path) -> float:
    """Return the first data row's ``total_score`` as a ``float``.

    Backward-compatible entry point retained from the v0.5 API: the
    legacy parser returned a bare ``float`` representing the
    first-row total score (or ``0.0`` when the scorefile contained
    no recognizable value). New code should prefer
    :func:`parse_relax_score` for the structured form that exposes
    every metric by name.

    The wrapper delegates to :func:`parse_relax_score` and reads
    ``RelaxScore.total_score``; a missing total column yields
    ``0.0``, matching the pre-v0.6 default for unparseable files.
    """
    score = parse_relax_score(path)
    return score.total_score if score.total_score is not None else 0.0


def parse_relax_score(path: Path) -> RelaxScore:
    """Parse the first data row of a Rosetta ``score.sc`` into a :class:`RelaxScore`.

    The function walks the scorefile line by line: it ignores blanks
    and ``#``-prefixed comments, treats the first ``SCORE:`` row as
    the column-header line, and returns the first subsequent
    ``SCORE:`` data row as a structured score. Each canonical
    metric is looked up by any of its recognized aliases
    (case-insensitive). Columns that are not present in the
    header, data rows with too few tokens, or non-finite tokens
    (``inf`` / ``-inf`` / ``nan``) are preserved as ``None`` — the
    parser never defaults to ``0.0`` for an absent metric and
    never lets an upstream sentinel value sneak through.

    A scorefile that lists ``total_score`` but no ``interface_dG``
    yields ``RelaxScore(total_score=..., interface_dG=None, ...)``.
    An empty or unparseable scorefile yields ``RelaxScore()`` (all
    fields ``None``); see :func:`parse_score_files` for how that's
    surfaced to callers as a FAILED record.
    """
    text = path.read_text()
    header_to_idx, first_data_row = _extract_first_score_row(text)
    return _build_relax_score(header_to_idx, first_data_row)


def _extract_first_score_row(
    text: str,
) -> tuple[dict[str, int], list[float | None] | None]:
    """Walk ``text``; return ``(header→index map, first data row)``.

    The first ``SCORE:`` line defines the column names; the first
    subsequent ``SCORE:`` line is parsed as a data row.
    Non-numeric tokens in the data row (typically the trailing
    ``description`` column) are recorded as ``None`` so the column
    index is still addressable. ``SEQUENCE:`` and ``#``-prefixed
    lines are ignored.
    """
    header_to_idx: dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith("SCORE:"):
            continue
        tokens = stripped[len("SCORE:") :].split()
        if not tokens:
            continue
        if not header_to_idx:
            for idx, name in enumerate(tokens):
                header_to_idx[name.lower()] = idx
            continue
        # First non-header ``SCORE:`` line: convert each token,
        # tolerating non-numeric columns (e.g. the ``description``
        # slot) by recording them as ``None`` so downstream lookups
        # still resolve cleanly to a missing value.
        return header_to_idx, [_parse_token(t) for t in tokens]
    return header_to_idx, None


def _parse_token(token: str) -> float | None:
    """Parse a single scorefile token; ``None`` for non-numeric or non-finite values.

    The non-finite check rejects ``inf``, ``-inf``, and ``nan``
    sentinel tokens that some upstream writers emit when a term is
    disabled or the scorefunction collapses a degenerate case.
    Allowing those through would silently inject ``+inf`` / ``nan``
    into downstream consumers; rejecting preserves the
    missing-vs-present contract that the alias table relies on.
    """
    try:
        value = float(token)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _build_relax_score(
    header_to_idx: Mapping[str, int],
    first_data_row: list[float | None] | None,
) -> RelaxScore:
    """Construct a :class:`RelaxScore` from a parsed header + data row.

    Each canonical metric looks up the first matching alias in the
    header. An unknown column, an out-of-range index, or a
    non-numeric / non-finite token at the matched position all
    yield ``None`` for the metric — this is the contract that lets
    the rest of the pipeline distinguish "the scorefile did not
    include this metric" from "the metric is zero."
    """
    resolved: dict[str, float | None] = {}
    for metric, aliases in METRIC_ALIASES.items():
        resolved[metric] = _lookup_metric(header_to_idx, first_data_row, aliases)
    return RelaxScore(**resolved)


def _lookup_metric(
    header_to_idx: Mapping[str, int],
    first_data_row: list[float | None] | None,
    aliases: tuple[str, ...],
) -> float | None:
    """Return the first float that matches any of the metric's aliases.

    Returns ``None`` if the scorefile did not include the column,
    the matched column is past the end of the (possibly truncated)
    data row, or the token at that position is non-numeric or
    non-finite. Aliases are checked in the order listed in
    :data:`METRIC_ALIASES`, so the first successful match wins.
    """
    for alias in aliases:
        idx = header_to_idx.get(alias.lower())
        if idx is None or first_data_row is None:
            continue
        if idx >= len(first_data_row):
            continue
        return first_data_row[idx]
    return None


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------


# A value in the config dict passed to :func:`invoke` may be either a
# single string (one argv token after the flag) or a sequence of strings
# (one argv token per element, with the flag repeated). The sequence
# form exists for the special-case key ``parser:script_vars``, which
# the upstream CLI accepts multiple times:
#
#   rosetta_scripts ... -parser:script_vars k1=v1 \
#                           -parser:script_vars k2=v2 ...
#
# Single-string values still produce a single argv token after the
# flag (``-flag value``), the canonical ``key=value`` form.
FlagArgValue = Any  # str | list[str] | tuple[str, ...]


def _config_value_to_argv(key: str, value: FlagArgValue) -> list[str]:
    """Render one ``(key, value)`` pair as a flat argv fragment.

    - ``str`` value → ``["-key", value]``.
    - Sequence value (``list`` / ``tuple``) → ``["-key", v1, "-key",
      v2, ...]`` (the flag repeats so each element is its own argv
      token). Empty sequence contributes nothing — the ``if script_vars``
      gate in :func:`biolab_runners.rosetta.runner._config_to_cli`
      ensures we never emit a bare ``-parser:script_vars`` with no
      payload.
    - Anything else → stringified via ``str()`` (the v0.5 fallback).
    """
    flag = f"-{key}" if not key.startswith("-") else key
    if isinstance(value, str):
        return [flag, value]
    if isinstance(value, (list, tuple)):
        # Sequence: each element becomes its own argv token after
        # the repeated flag. Empty sequences emit nothing.
        argv: list[str] = []
        for v in value:
            argv.append(flag)
            argv.append(str(v))
        return argv
    return [flag, str(value)]


def invoke(
    *,
    config: dict[str, FlagArgValue],
    output_dir: Path,
    binary_prefix: list[str] | None = None,
    timeout_seconds: int = 3600,
) -> int:
    """Run ``rosetta_scripts`` once; returns the process exit code.

    The ``config`` mapping may carry either string values (each one
    becomes a single argv token after its flag) or sequence values
    (the flag repeats so each element becomes its own argv token).
    The sequence form is required for ``parser:script_vars``, which
    upstream expects to receive multiple times — one token per
    variable — not as a single space-joined argument.
    """
    prefix = binary_prefix if binary_prefix is not None else _resolved_binary()
    output_dir.mkdir(parents=True, exist_ok=True)
    args: list[str] = [
        *prefix,
        "--parser",
        "protocol",
    ]
    for key, value in config.items():
        args.extend(_config_value_to_argv(key, value))
    started = time.monotonic()
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error("Rosetta timed out after %ds", timeout_seconds)
        return 124
    logger.info(
        "Rosetta run finished rc=%d in %.1fs",
        completed.returncode,
        time.monotonic() - started,
    )
    return completed.returncode


def parse_score_files(score_files: Iterable[Path]) -> list[RelaxRecord]:
    """Convert a list of ``score.sc`` paths into :class:`RelaxRecord`.

    Files that parse into a structured :class:`RelaxScore` with at
    least one populated metric become SUCCEEDED records carrying
    both the legacy ``total_score`` float and the structured
    ``score`` form (additive — pre-v0.6 callers that only read
    ``record.total_score`` see the same float they always did).

    Files that fall into any of these buckets become FAILED records
    with an empty :class:`RelaxScore`, ``total_score = 0.0``, and
    a non-empty ``error`` string:

    - **Unreadable** — ``OSError`` / ``UnicodeDecodeError`` on read.
    - **All-None score** — the parser found a valid scorefile
      header row but no recognized metric column was populated
      (e.g. a descriptor-only or empty-data scorefile). Surfacing
      this as FAILED rather than SUCCEEDED-with-zeros ensures the
      runner's ``succeeded`` / ``failed`` counters reflect "this row
      meant something", not just "this row was syntactically
      valid".
    """
    records: list[RelaxRecord] = []
    for path in score_files:
        path_str = str(path)
        try:
            score = parse_relax_score(path)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("failed to parse %s: %s", path_str, exc)
            records.append(
                RelaxRecord(
                    index=len(records),
                    path=path_str,
                    total_score=0.0,
                    score=RelaxScore(),
                    status=RelaxRecordStatus.FAILED,
                    error=str(exc),
                )
            )
            continue
        # all-None score → FAILED with a synthetic error. The parser
        # ran without raising, but the row carries no usable value.
        if _is_empty_score(score):
            records.append(
                RelaxRecord(
                    index=len(records),
                    path=path_str,
                    total_score=0.0,
                    status=RelaxRecordStatus.FAILED,
                    error=(
                        "scorefile contained no recognized metric columns "
                        "(or only non-finite values)"
                    ),
                )
            )
            continue
        records.append(
            RelaxRecord(
                index=len(records),
                path=path_str,
                total_score=score.total_score if score.total_score is not None else 0.0,
                score=score,
                status=RelaxRecordStatus.SUCCEEDED,
            )
        )
    return records


def _is_empty_score(score: RelaxScore) -> bool:
    """True when every metric field is ``None`` (no usable value)."""
    return all(getattr(score, f.name) is None for f in fields(score))
