"""Console adapter for the upstream ProteinMPNN script.

The runner speaks a stable ``--key value`` contract. This adapter resolves
the operator's upstream checkout and forwards the arguments as an argv list;
it never invokes a shell or fabricates scientific output.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["build_command", "main", "resolve_script", "translate_runner_args"]

DEFAULT_HOME = "~/tools/ProteinMPNN"
UPSTREAM_SCRIPT = "protein_mpnn_run.py"
_MANAGED_FLAGS = frozenset(
    {
        "input_path",
        "output_path",
        "batch_size",
        "model_name",
        "num_seq_per_target",
        "sampling_temp",
        "seed",
        "ca_only",
        "omit_AA",
        "fixed_positions_jsonl",
        "pdb_path",
        "pdb_path_chains",
    }
)
_OVERRIDE_FLAGS = frozenset({"out_folder", "omit_AAs"})


def resolve_script() -> Path:
    """Resolve the upstream ``protein_mpnn_run.py`` script."""
    configured = os.environ.get("PROTEINMPNN_SCRIPT")
    if configured:
        return Path(configured).expanduser()
    home = Path(os.environ.get("PROTEINMPNN_HOME", DEFAULT_HOME)).expanduser()
    return home / UPSTREAM_SCRIPT


class _ArgumentParser(argparse.ArgumentParser):
    """Argument parser that reports contract errors without exiting."""

    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


def translate_runner_args(args: list[str]) -> list[str]:
    """Translate wrapper flags into the upstream ProteinMPNN contract.

    Unknown flags are intentionally preserved and appended after the known
    translation. This keeps forward-compatible upstream options available
    without allowing the adapter to interpret them.
    """
    parser = _ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", default="1")
    parser.add_argument("--model_name", default="v_48_020")
    parser.add_argument("--num_seq_per_target", default="4")
    parser.add_argument("--sampling_temp", default="0.1")
    parser.add_argument("--seed", default="0")
    parser.add_argument("--ca_only", default="False")
    parser.add_argument("--omit_AA", default="")
    parser.add_argument("--fixed_positions_jsonl", default="")
    parser.add_argument("--pdb_path", required=True)
    _reject_ambiguous_flags(args)
    try:
        parsed, extra = parser.parse_known_args(args)
    except SystemExit as exc:  # pragma: no cover - defensive for argparse versions
        raise ValueError("invalid ProteinMPNN runner arguments") from exc
    if any(arg == "--fixed_positions" or arg.startswith("--fixed_positions=") for arg in args):
        raise ValueError("fixed_positions is unsupported without a chain-aware JSONL contract")

    upstream = [
        "--pdb_path",
        str(Path(parsed.input_path) / parsed.pdb_path),
        "--out_folder",
        parsed.output_path,
        "--batch_size",
        parsed.batch_size,
        "--model_name",
        parsed.model_name,
        "--num_seq_per_target",
        parsed.num_seq_per_target,
        "--sampling_temp",
        parsed.sampling_temp,
        "--seed",
        parsed.seed,
    ]
    if parsed.ca_only.lower() in {"1", "true", "yes"}:
        upstream.append("--ca_only")
    if parsed.omit_AA:
        upstream.extend(["--omit_AAs", parsed.omit_AA])
    if parsed.fixed_positions_jsonl:
        upstream.extend(["--fixed_positions_jsonl", parsed.fixed_positions_jsonl])
    return [*upstream, *extra]


def _reject_ambiguous_flags(args: list[str]) -> None:
    """Reject legacy overrides and duplicate wrapper-owned arguments."""
    seen: set[str] = set()
    for arg in args:
        if not arg.startswith("--"):
            continue
        flag = arg[2:].split("=", 1)[0]
        if flag == "fixed_positions":
            raise ValueError("fixed_positions is unsupported without a chain-aware JSONL contract")
        if flag in _OVERRIDE_FLAGS:
            raise ValueError(f"{flag} is managed by the runner and cannot be overridden")
        if flag in _MANAGED_FLAGS:
            if flag in seen:
                raise ValueError(f"duplicate managed flag --{flag}")
            seen.add(flag)


def build_command(argv: Sequence[str]) -> list[str]:
    """Build the direct Python subprocess command for upstream."""
    python = os.environ.get("PROTEINMPNN_PYTHON", sys.executable)
    return [python, str(resolve_script()), *translate_runner_args(list(argv))]


def _usage() -> str:
    return (
        "usage: proteinmpnn [ProteinMPNN flags]\n\n"
        "Forwards the runner contract to protein_mpnn_run.py without a shell.\n"
        "Set PROTEINMPNN_HOME, PROTEINMPNN_SCRIPT, or PROTEINMPNN_PYTHON "
        "to select the upstream runtime."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Forward runner arguments to the upstream script and return its exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or "--help" in args or "-h" in args:
        print(_usage())
        return 0
    script = resolve_script()
    if not script.is_file():
        print(
            f"ProteinMPNN runtime missing: {script} (set PROTEINMPNN_HOME or PROTEINMPNN_SCRIPT)",
            file=sys.stderr,
        )
        return 2
    try:
        command = build_command(args)
    except ValueError as exc:
        print(f"proteinmpnn: {exc}", file=sys.stderr)
        return 2
    completed = subprocess.run(command, check=False)
    return completed.returncode
