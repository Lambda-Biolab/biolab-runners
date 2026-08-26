"""Behavioral tests for pure Rosetta decoy resolution."""

from __future__ import annotations

import hashlib
import os
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest
from biolab_runners.contracts import ExecutionStatus, IncompleteOutputError, MalformedOutputError
from biolab_runners.rosetta import (
    ChainAudit,
    PDBIdentity,
    RelaxScore,
    RelaxScoreRow,
    RosettaDecoyResolutionRequest,
    parse_relax_score,
    parse_relax_score_rows,
    resolve_decoy,
)


def _atom(
    serial: int,
    chain: str,
    residue: int,
    x: float = 1.0,
    y: float = 2.0,
    z: float = 3.0,
    insertion: str = " ",
    atom_name: str = "CA",
    residue_name: str = "ALA",
    record: str = "ATOM",
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom_name:^4s} {residue_name:>3s} "
        f"{chain}{residue:4d}{insertion}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C  \n"
    )


def _pdb(*records: str) -> bytes:
    return "".join(records).encode("ascii")


def _crlf(record: str) -> bytes:
    return record.encode("ascii").replace(b"\n", b"\r\n")


def _score(description: str, total_score: str = "-12.5", extra: str = "") -> str:
    return (
        "SEQUENCE:\n"
        "SCORE: total_score packstat description\n"
        f"SCORE: {total_score} {extra or '0.7'} {description}\n"
    )


def _request(tmp_path: Path, **overrides: object) -> RosettaDecoyResolutionRequest:
    input_pdb = tmp_path / "input.pdb"
    output_pdb = tmp_path / "2wpt.ppk_0001.pdb"
    score_file = tmp_path / "score.sc"
    input_pdb.write_bytes(_pdb(_atom(1, "A", 1)))
    output_pdb.write_bytes(
        _pdb(
            _atom(1, "A", 1),
            _atom(2, "A", 1, 2.0),
            _atom(3, "B", 1),
            _atom(4, "B", 2),
            _atom(5, "B", 2, 2.0, insertion="A"),
            _atom(6, "C", 1, record="HETATM"),
        ).replace(b"\n", b"\r\n")
    )
    score_file.write_text(_score("2wpt.ppk_0001"))
    values: dict[str, object] = {
        "score_file": score_file,
        "input_pdb": input_pdb,
        "output_pdb": output_pdb,
        "input_pdb_identity": PDBIdentity(
            "gs://bucket/input.pdb", hashlib.sha256(input_pdb.read_bytes()).hexdigest()
        ),
        "output_pdb_identity": PDBIdentity(
            "gs://bucket/2wpt.ppk_0001.pdb", hashlib.sha256(output_pdb.read_bytes()).hexdigest()
        ),
        "candidate_identity": "candidate-7",
        "parent_input_identity": "parent-3",
        "protocol_identity": "protocol-1",
        "config_identity": "config-2",
        "runtime_identity": "runtime-4",
        "expected_chain_roles": (
            ("A", "receptor-alpha"),
            ("B", "receptor-beta"),
            ("C", "binder"),
        ),
        "status": ExecutionStatus.SUCCEEDED,
    }
    values.update(overrides)
    return RosettaDecoyResolutionRequest(**values)  # type: ignore[arg-type]


def test_resolver_returns_exact_identities_score_audits_and_status(tmp_path: Path) -> None:
    request = _request(tmp_path)
    input_bytes = request.input_pdb.read_bytes()
    output_bytes = request.output_pdb.read_bytes()

    artifact = resolve_decoy(request)

    assert artifact.input_pdb_identity.sha256 == hashlib.sha256(input_bytes).hexdigest()
    assert artifact.output_pdb_identity.sha256 == hashlib.sha256(output_bytes).hexdigest()
    assert artifact.input_pdb_identity == request.input_pdb_identity
    assert artifact.output_pdb_identity == request.output_pdb_identity
    assert artifact.candidate_identity == "candidate-7"
    assert artifact.parent_input_identity == "parent-3"
    assert artifact.protocol_identity == "protocol-1"
    assert artifact.config_identity == "config-2"
    assert artifact.runtime_identity == "runtime-4"
    assert artifact.relax_score.total_score == -12.5
    assert artifact.chain_audits == (
        ChainAudit("A", "receptor-alpha", 1, 2),
        ChainAudit("B", "receptor-beta", 3, 3),
        ChainAudit("C", "binder", 1, 1),
    )
    assert artifact.status is ExecutionStatus.SUCCEEDED


@pytest.mark.parametrize("description", ["2wpt.ppk_0001", "2wpt.ppk_0001.pdb"])
def test_resolver_accepts_extensionless_and_exact_filename_descriptions(
    tmp_path: Path, description: str
) -> None:
    request = _request(tmp_path)
    request.score_file.write_text(_score(description))

    artifact = resolve_decoy(request)

    assert artifact.relax_score == RelaxScore(total_score=-12.5, packstat=0.7)


def test_score_rows_preserve_descriptions_and_first_row_parser_is_unchanged(tmp_path: Path) -> None:
    score_file = tmp_path / "score.sc"
    content = (
        "SEQUENCE:\n"
        "SCORE: total_score packstat description\n"
        "SCORE: total_score packstat description\n"
        "SCORE: -1.0 0.1 first.raw\n"
        "SCORE: -2.0 0.2 second.raw\n"
    )
    score_file.write_text(content)

    rows = parse_relax_score_rows(score_file)

    assert rows == (
        RelaxScoreRow("first.raw", RelaxScore(total_score=-1.0, packstat=0.1)),
        RelaxScoreRow("second.raw", RelaxScore(total_score=-2.0, packstat=0.2)),
    )
    legacy_score_file = tmp_path / "legacy-score.sc"
    legacy_score_file.write_text(
        "SCORE: total_score packstat description\nSCORE: -1.0 0.1 first.raw\n"
    )
    assert parse_relax_score(legacy_score_file) == RelaxScore(total_score=-1.0, packstat=0.1)


def test_multiple_rows_require_unique_exact_selector(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.score_file.write_text(
        "SCORE: total_score description\n"
        "SCORE: -1.0 2wpt.ppk_0001.pdb\n"
        "SCORE: -2.0 2wpt.ppk_0001.pdb\n"
    )

    with pytest.raises(MalformedOutputError, match="multiple"):
        resolve_decoy(request)
    with pytest.raises(MalformedOutputError, match="exactly one"):
        resolve_decoy(replace(request, decoy_description="2wpt.ppk_0001.pdb"))
    with pytest.raises(MalformedOutputError, match="exactly one"):
        resolve_decoy(replace(request, decoy_description="not-present"))

    request.score_file.write_text(
        "SCORE: total_score description\nSCORE: -1.0 other.pdb\nSCORE: -2.0 2wpt.ppk_0001.pdb\n"
    )
    selected = resolve_decoy(replace(request, decoy_description="2wpt.ppk_0001.pdb"))
    assert selected.relax_score.total_score == -2.0


@pytest.mark.parametrize("field", ["input_pdb", "output_pdb"])
def test_resolver_rejects_replaced_pdb_snapshot(tmp_path: Path, field: str) -> None:
    request = _request(tmp_path)
    path = getattr(request, field)
    replacement = (
        _pdb(_atom(99, "A", 1, x=9.0))
        if field == "input_pdb"
        else _pdb(
            _atom(1, "A", 1, x=9.0),
            _atom(2, "A", 1, 2.0),
            _atom(3, "B", 1),
            _atom(4, "B", 2),
            _atom(5, "B", 2, 2.0, insertion="A"),
            _atom(6, "C", 1, record="HETATM"),
        )
    )
    path.write_bytes(replacement)

    with pytest.raises(MalformedOutputError, match="expected PDB identity"):
        resolve_decoy(request)


@pytest.mark.parametrize("field", ["input_pdb_identity", "output_pdb_identity"])
def test_request_requires_typed_pdb_identities(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValueError, match="PDBIdentity"):
        _request(tmp_path, **{field: "gs://bucket/replaced.pdb"})


def test_fifo_required_file_is_rejected_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("os.mkfifo is unavailable")
    request = _request(tmp_path)
    request.input_pdb.unlink()
    os.mkfifo(request.input_pdb)

    with pytest.raises(IncompleteOutputError, match="input PDB"):
        resolve_decoy(request)


@pytest.mark.parametrize("field", ["input_pdb", "output_pdb", "score_file"])
@pytest.mark.parametrize("kind", ["missing", "empty", "symlink"])
def test_required_files_must_be_nonempty_regular_non_symlinks(
    tmp_path: Path, field: str, kind: str
) -> None:
    request = _request(tmp_path)
    path = getattr(request, field)
    if kind == "missing":
        path.unlink()
    elif kind == "empty":
        path.write_bytes(b"")
    else:
        real = tmp_path / f"real-{field}"
        real.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(real)

    with pytest.raises(IncompleteOutputError, match=field.replace("_", " ")[:5]):
        resolve_decoy(request)


@pytest.mark.parametrize(
    "description",
    ["", "../2wpt.ppk_0001.pdb", "other.pdb"],
)
def test_description_must_be_safe_and_map_to_output(tmp_path: Path, description: str) -> None:
    request = _request(tmp_path)
    request.score_file.write_text(_score(description))

    with pytest.raises(MalformedOutputError):
        resolve_decoy(request)


def test_required_metrics_must_be_present_finite_and_known(tmp_path: Path) -> None:
    request = _request(tmp_path, required_metrics=("packstat",))
    request.score_file.write_text("SCORE: total_score description\nSCORE: -12.5 2wpt.ppk_0001\n")
    with pytest.raises(MalformedOutputError, match="packstat"):
        resolve_decoy(request)

    request = replace(request, required_metrics=("total_score",))
    request.score_file.write_text(_score("2wpt.ppk_0001", total_score="nan"))
    with pytest.raises(MalformedOutputError, match="total_score"):
        resolve_decoy(request)

    with pytest.raises(ValueError, match="unknown required metric"):
        _request(tmp_path, required_metrics=("not_a_metric",))


@pytest.mark.parametrize(
    ("output_mutation", "expected_error"),
    [
        (
            lambda output: output.replace(_crlf(_atom(6, "C", 1, record="HETATM")), b""),
            "output PDB chain set does not match expected chain roles",
        ),
        (
            lambda output: output + _atom(7, "D", 1).encode(),
            "output PDB chain set does not match expected chain roles",
        ),
        (
            lambda output: output.replace(b"   1.000", b"     bad", 1),
            "ATOM/HETATM record has malformed serial, residue number, or coordinates",
        ),
        (lambda output: b"HEADER\n", "output PDB contains no atoms"),
        (
            lambda output: b"MODEL        1\n" + output + b"MODEL        2\n",
            "output PDB contains multiple or nested MODEL records",
        ),
        (
            lambda output: b"MODEL        1\n" + output,
            "output PDB contains an unclosed MODEL record",
        ),
        (
            lambda output: output + b"ENDMDL\n",
            "output PDB contains an unmatched ENDMDL record",
        ),
        (
            lambda output: b"MODEL        1\n" + output + b"ENDMDL\n" + _atom(7, "A", 1).encode(),
            "output PDB contains atoms outside its explicit model",
        ),
        (
            lambda output: output + b"MODEL        1\nENDMDL\n",
            "output PDB contains atoms outside its explicit model",
        ),
        (
            lambda output: output.replace(_crlf(_atom(1, "A", 1)), _crlf(_atom(1, " ", 1))),
            "ATOM/HETATM record has a blank chain",
        ),
        (
            lambda output: output.replace(b"ATOM      1", b"ATOM  bad 1", 1),
            "ATOM/HETATM record has malformed serial, residue number, or coordinates",
        ),
        (
            lambda output: output.replace(
                _crlf(_atom(1, "A", 1)), _crlf(_atom(1, "A", 1, atom_name=""))
            ),
            "ATOM/HETATM record has a blank atom or residue name",
        ),
        (
            lambda output: output.replace(
                _crlf(_atom(1, "A", 1)), _crlf(_atom(1, "A", 1, residue_name=""))
            ),
            "ATOM/HETATM record has a blank atom or residue name",
        ),
    ],
)
def test_output_pdb_must_have_valid_expected_single_model_chains(
    tmp_path: Path, output_mutation: object, expected_error: str
) -> None:
    request = _request(tmp_path)
    output = request.output_pdb.read_bytes()
    mutated_output = output_mutation(output)  # type: ignore[operator]
    request.output_pdb.write_bytes(mutated_output)
    request = replace(
        request,
        output_pdb_identity=PDBIdentity(
            request.output_pdb_identity.uri, hashlib.sha256(mutated_output).hexdigest()
        ),
    )

    with pytest.raises(MalformedOutputError, match=expected_error):
        resolve_decoy(request)


def test_output_pdb_accepts_one_balanced_explicit_model(tmp_path: Path) -> None:
    request = _request(tmp_path)
    output = request.output_pdb.read_bytes()
    explicit_output = b"MODEL        1\n" + output + b"ENDMDL\n"
    request.output_pdb.write_bytes(explicit_output)
    request = replace(
        request,
        output_pdb_identity=PDBIdentity(
            request.output_pdb_identity.uri, hashlib.sha256(explicit_output).hexdigest()
        ),
    )

    artifact = resolve_decoy(request)

    assert artifact.chain_audits[0].atom_count == 2


@pytest.mark.parametrize(
    "score_text",
    [
        "SCORE: total_score description packstat\nSCORE: -12.5 2wpt.ppk_0001\n",
        "SCORE: total_score packstat description\nSCORE: -12.5 0.7 2wpt.ppk_0001 extra\n",
    ],
)
def test_resolver_rejects_score_rows_with_wrong_arity(tmp_path: Path, score_text: str) -> None:
    request = _request(tmp_path)
    request.score_file.write_text(score_text)

    with pytest.raises(MalformedOutputError, match="schema"):
        resolve_decoy(request)


def test_resolver_rejects_duplicate_score_description_columns(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.score_file.write_text(
        "SCORE: total_score DESCRIPTION description\nSCORE: -12.5 2wpt.ppk_0001 2wpt.ppk_0001\n"
    )

    with pytest.raises(MalformedOutputError, match="schema"):
        resolve_decoy(request)


def test_request_rejects_duplicate_or_blank_chain_expectations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        _request(tmp_path, expected_chain_roles=(("A", "one"), ("A", "two")))
    with pytest.raises(ValueError, match="chain IDs"):
        _request(tmp_path, expected_chain_roles=(("", "one"),))
    with pytest.raises(ValueError, match="chain roles"):
        _request(tmp_path, expected_chain_roles=(("A", ""),))
    with pytest.raises(ValueError, match="exactly one"):
        _request(tmp_path, expected_chain_roles=(("AB", "role"),))


def test_request_requires_nonempty_required_metrics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _request(tmp_path, required_metrics=())


@pytest.mark.parametrize("selector", ["", 42])
def test_request_requires_nonempty_string_selector(tmp_path: Path, selector: object) -> None:
    with pytest.raises(ValueError, match="decoy_description"):
        _request(tmp_path, decoy_description=selector)


def test_request_requires_explicit_terminal_status(tmp_path: Path) -> None:
    request = _request(tmp_path)
    values = {
        field.name: getattr(request, field.name)
        for field in fields(request)
        if field.name != "status"
    }

    with pytest.raises(TypeError, match="status"):
        RosettaDecoyResolutionRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "status", [status for status in ExecutionStatus if status is not ExecutionStatus.SUCCEEDED]
)
def test_every_non_success_status_is_rejected(tmp_path: Path, status: ExecutionStatus) -> None:
    request = _request(tmp_path, status=status)

    with pytest.raises(MalformedOutputError, match="scientific terminal success"):
        resolve_decoy(request)


def test_resolver_is_read_only_and_request_is_frozen(tmp_path: Path) -> None:
    request = _request(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    entries_before = sorted(path.name for path in tmp_path.iterdir())

    resolve_decoy(request)

    assert sorted(path.name for path in tmp_path.iterdir()) == entries_before
    assert {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before
    with pytest.raises(FrozenInstanceError):
        request.status = ExecutionStatus.FAILED  # type: ignore[misc]
