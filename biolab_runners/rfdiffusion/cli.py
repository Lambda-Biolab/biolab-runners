"""In-package ``rfdiffusion`` console script (stock RFdiffusion adapter).

The biolab-runners
:class:`~biolab_runners.rfdiffusion.runner.RFdiffusionRunner` invokes
``rfdiffusion`` (or ``${RFDIFFUSION_BIN}``) with a fixed subprocess
contract:

    rfdiffusion --output_dir <dir> --<dotted.hydra.key> <value> ...

This module is that binary. It validates the contract, translates
each flag into a stock Hydra positional override (``key=value``), and
executes the upstream ``scripts/run_inference.py`` located under
``RFDIFFUSION_HOME`` (default ``~/tools/RFdiffusion`` — the convention
used by the org's local install script). The upstream clone must be
present with the model weights downloaded; ``--help`` needs neither.

Contract rules:

* ``--output_dir`` is required and owns ``inference.output_prefix``
  (``<output_dir>/design``, absolutized), so upstream emits
  ``design_<i_des>.pdb`` in that directory — exactly what the runner
  parses back. A caller-supplied ``inference.output_prefix`` is
  rejected: the script owns it.
* Every other flag is ``--<dotted.hydra.key> <value>`` with
  underscores hyphenated exactly as the runner emits them
  (``--inference.num-designs 5`` → ``inference.num_designs=5``).
* List-typed keys are translated to Hydra list syntax:
  ``contigmap.contigs`` → ``contigmap.contigs=[<value>]`` (the stock
  README's space-separated form, preserved as one argv element) and
  ``ppi.hotspot_res`` → ``ppi.hotspot_res=['A51','B52']`` (the stock
  binder example's quoted-string list).
* String scalars are Hydra-quoted only when needed: plain tokens
  (numbers, booleans, simple paths) stay unquoted so their types are
  preserved; values with whitespace / quotes / commas / ``#`` / ``~``
  etc. are wrapped in quotes with deterministic escaping that Hydra's
  own unescaper round-trips (no shell quoting).
* ``inference.seed`` is rejected — the key is inert upstream; the
  runner maps ``seed`` to ``inference.design_startnum``.
* No shell is involved. ``RFDIFFUSION_HOME`` is a directory path
  (never a command), and ``RFDIFFUSION_PYTHON`` is one executable
  filesystem path (never a command plus arguments). The upstream script is
  resolved as a path under ``RFDIFFUSION_HOME`` and executed with
  ``RFDIFFUSION_PYTHON`` when set, otherwise the current interpreter, so
  arbitrary wrapper commands cannot be injected. Exit code / stdout / stderr
  propagate to the caller.
* The resolved clone root is prepended to ``PYTHONPATH`` (the existing
  value is preserved), so a clone-only deployment can ``import
  rfdiffusion`` inside ``run_inference.py`` without installing the
  upstream package.
* Hydra's own metadata / logs are confined under the output
  directory: the script owns ``hydra.run.dir=<output_dir>/hydra``,
  ``hydra.output_subdir=null``, and ``hydra.job.chdir=False`` so
  nothing leaks into the caller's CWD and PDB outputs are unaffected.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

__all__ = ["EXECUTION_CONTRACT_VERSION", "main"]

#: Version of the runner→CLI execution contract. **Bump** whenever any
#: part of the translation changes — the runner's config→flag mapping
#: (``runner._config_to_cli``), the flag→Hydra translation in this
#: module (``_hydra_override`` / ``_hydra_scalar``), or the owned
#: overrides (``inference.output_prefix``, ``hydra.run.dir``,
#: ``hydra.output_subdir``, ``hydra.job.chdir``) — because the derived
#: execution payload is bound into the cache identity and the
#: executed-config digest (``runner._execution_payload`` /
#: ``runner._executed_digest``). A bump invalidates cached outputs
#: once; ``runner.py`` imports this constant so there is a single
#: authoritative bump location for the whole translation.
EXECUTION_CONTRACT_VERSION = 1

#: Hydra key the console script owns (derived from ``--output_dir``).
INFERENCE_OUTPUT_PREFIX_FIELD = "inference.output_prefix"
#: Output prefix name — upstream writes ``<prefix>_<i_des>.pdb``, so
#: ``<output_dir>/design`` yields ``design_<i_des>.pdb``.
OUTPUT_PREFIX_NAME = "design"
#: Default upstream clone location (org install convention — the
#: ``install-proteinmpnn-rfdiffusion.sh`` bootstrap clones into
#: ``~/tools/RFdiffusion``). Override with ``RFDIFFUSION_HOME``.
DEFAULT_RFDIFFUSION_HOME = Path.home() / "tools" / "RFdiffusion"
#: Keys that do not exist in stock RFdiffusion and are rejected.
UNSUPPORTED_KEYS = frozenset({"inference.seed"})

USAGE = """\
rfdiffusion — biolab-runners adapter for stock RFdiffusion.

Usage:
    rfdiffusion --output_dir <dir> [--<dotted.hydra.key> <value> ...]

Required:
    --output_dir <dir>   output directory for design PDBs
                         (inference.output_prefix=<dir>/design)

Flags are dotted Hydra keys with underscores hyphenated, exactly as
RFdiffusionRunner emits them (each takes one value argument):
    --inference.num-designs 10
    --contigmap.contigs 'A1-110/0 B1-110/0 14-18'
    --inference.input-pdb target.pdb
    --inference.design-startnum 42
    --inference.deterministic True
    --inference.cyclic True
    --inference.cyc-chains a
    --ppi.hotspot-res A51,A52

Runtime requirements:
    RFDIFFUSION_HOME   path to the upstream RFdiffusion clone
                       (default: ~/tools/RFdiffusion) containing
                       scripts/run_inference.py and the model weights.
    RFDIFFUSION_PYTHON optional path to an executable Python runtime for
                       stock RFdiffusion (default: current interpreter).
                       It must be one filesystem path, not shell text or
                       a command with arguments.
    --help needs neither.
"""


def _hydra_key(flag: str) -> str:
    """``--inference.num-designs`` -> ``inference.num_designs``.

    The runner hyphenates every underscore for argv; stock Hydra keys
    never contain hyphens, so the reversal is a bijection on the
    runner's emitted flags.
    """
    return flag[2:].replace("-", "_")


def _parse_flag(argv: list[str], index: int, overrides: dict[str, str]) -> int:
    """Validate one ``--<dotted.key>`` flag and record its override.

    Rejects undotted flags, unsupported keys (``inference.seed``), the
    script-managed ``inference.output_prefix``, and flags without a
    value. Returns the next index to consume.
    """
    arg = argv[index]
    key = _hydra_key(arg)
    if "." not in key:
        raise ValueError(
            f"unsupported flag {arg!r}: expected a dotted Hydra key (e.g. --inference.num-designs)"
        )
    if key in UNSUPPORTED_KEYS:
        raise ValueError(
            f"key {key!r} is not supported by stock RFdiffusion; set "
            "RFdiffusionConfig.seed instead (forwarded as "
            "inference.design_startnum)"
        )
    if key == INFERENCE_OUTPUT_PREFIX_FIELD:
        raise ValueError(
            f"{INFERENCE_OUTPUT_PREFIX_FIELD} is managed by the rfdiffusion console "
            "script (derived from --output_dir); it cannot be overridden"
        )
    if index + 1 >= len(argv):
        raise ValueError(f"{arg} requires a value")
    overrides[key] = argv[index + 1]
    return index + 2


def _parse_args(argv: list[str]) -> tuple[str, dict[str, str]]:
    """Parse the runner contract into ``(output_dir, overrides)``.

    Raises ``ValueError`` on malformed input: bare positionals, flags
    without values, undotted flags, unsupported keys, duplicate
    ``--output_dir``, a caller-managed ``inference.output_prefix``, or
    a missing ``--output_dir``.
    """
    output_dir: str | None = None
    overrides: dict[str, str] = {}
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--output_dir":
            if output_dir is not None:
                raise ValueError("duplicate --output_dir")
            if index + 1 >= len(argv):
                raise ValueError("--output_dir requires a value")
            output_dir = argv[index + 1]
            index += 2
        elif arg.startswith("--"):
            index = _parse_flag(argv, index, overrides)
        else:
            raise ValueError(f"unexpected argument {arg!r}: expected --<dotted.key> <value> pairs")
    if output_dir is None:
        raise ValueError("--output_dir is required")
    return output_dir, overrides


#: Characters that parse safely UNQUOTED as a Hydra scalar override
#: value (verified against Hydra 1.3): letters/digits and path
#: punctuation. Everything else — whitespace, quotes, commas, ``#``,
#: ``;``, ``~`` (YAML null!), brackets, … — forces quoting so the
#: value stays exactly one string scalar.
_SAFE_UNQUOTED_RE = re.compile(r"^[A-Za-z0-9_./:%+-]+$")


def _hydra_scalar(value: str) -> str:
    r"""Render a string scalar for a Hydra ``key=value`` override.

    Plain tokens (numbers, booleans, ``null``, simple paths) are
    returned unquoted so OmegaConf preserves their types; anything
    else is wrapped in quotes with deterministic escaping that Hydra's
    own unescaper (``overrides_visitor._unescape_quoted_string``)
    round-trips:

    * no single quote in the value → single-quoted; backslashes are
      literal there, except a trailing run (before the added closing
      quote) which is doubled so the closing quote is not consumed by
      the ``(\\\\)+'`` unescape;
    * single quote present → double-quoted; every ``"`` is escaped as
      ``\\"`` and every backslash immediately preceding a ``"`` (or
      the added closing quote) is doubled.

    This is a deterministic escape-safe implementation — NOT shell
    quoting (no ``shlex``), so the exact bytes survive argv.
    """
    if _SAFE_UNQUOTED_RE.fullmatch(value):
        return value
    if "'" not in value:
        trailing = len(value) - len(value.rstrip("\\"))
        return "'" + value + "\\" * trailing + "'"
    escaped: list[str] = []
    index = 0
    length = len(value)
    while index < length:
        char = value[index]
        if char == '"':
            escaped.append('\\"')
        elif char == "\\" and (index + 1 == length or value[index + 1] == '"'):
            escaped.append("\\\\")
        else:
            escaped.append(char)
        index += 1
    return '"' + "".join(escaped) + '"'


def _hydra_override(key: str, value: str) -> str:
    """Translate one runner-contract value into a stock Hydra override.

    List-typed keys get Hydra list syntax: ``contigmap.contigs`` is
    wrapped in brackets with the value verbatim (the stock README's
    space-separated list), and ``ppi.hotspot_res`` becomes a
    comma-separated list of quoted strings (the stock binder example's
    form). All other keys pass through as ``key=value`` with the value
    Hydra-quoted via :func:`_hydra_scalar` (types preserved).
    """
    if key == "contigmap.contigs":
        return f"contigmap.contigs=[{value}]"
    if key == "ppi.hotspot_res":
        items = [item for item in value.split(",") if item]
        return "ppi.hotspot_res=[" + ",".join(f"'{item}'" for item in items) + "]"
    return f"{key}={_hydra_scalar(value)}"


def _resolve_run_inference() -> Path:
    """Locate stock ``scripts/run_inference.py`` under ``RFDIFFUSION_HOME``.

    ``RFDIFFUSION_HOME`` is a directory path (never a command); the
    default follows the org bootstrap convention (``~/tools/RFdiffusion``).
    """
    home = os.environ.get("RFDIFFUSION_HOME", str(DEFAULT_RFDIFFUSION_HOME))
    script = Path(home) / "scripts" / "run_inference.py"
    if not script.is_file():
        raise RuntimeError(
            f"stock RFdiffusion run_inference.py not found at {script}; set "
            "RFDIFFUSION_HOME to the upstream clone root "
            "(https://github.com/RosettaCommons/RFdiffusion) with the model "
            "weights downloaded"
        )
    return script


def _has_unsupported_command_shape(value: str) -> bool:
    """Return whether an invalid runtime value looks like shell text."""
    if any(char in value for char in ";&|<>$`'\"~\r\n\t"):
        return True
    try:
        parts = shlex.split(value)
    except ValueError:
        return True
    return len(parts) != 1 or parts[0] != value


def _resolve_python() -> str:
    """Resolve and validate the optional stock RFdiffusion Python runtime."""
    configured = os.environ.get("RFDIFFUSION_PYTHON")
    if configured is None:
        return sys.executable
    if not configured:
        raise RuntimeError(
            "RFDIFFUSION_PYTHON must be a single executable filesystem path, not shell text"
        )

    runtime = Path(configured)
    if not runtime.exists():
        if _has_unsupported_command_shape(configured):
            raise RuntimeError(
                "RFDIFFUSION_PYTHON must be a single executable filesystem path, "
                f"not shell text or a command with arguments: {configured!r}"
            )
        raise RuntimeError(f"RFDIFFUSION_PYTHON path does not exist: {configured!r}")
    if not runtime.is_file():
        raise RuntimeError(
            f"RFDIFFUSION_PYTHON must point to a regular executable file: {configured!r}"
        )
    if not runtime.stat().st_mode & 0o111 or not os.access(runtime, os.X_OK):
        raise RuntimeError(f"RFDIFFUSION_PYTHON is not executable: {configured!r}")
    return str(runtime.absolute())


def main(argv: list[str] | None = None) -> int:
    """Console entry point; returns the process exit code.

    ``--help``/``-h`` prints usage and returns ``0`` without validating
    ``RFDIFFUSION_HOME`` or ``RFDIFFUSION_PYTHON`` or touching model files,
    so the availability probe stays cheap. On success the upstream exit code
    / stdout / stderr propagate to the caller unchanged.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or "-h" in args or "--help" in args:
        print(USAGE)
        return 0
    try:
        output_dir, overrides = _parse_args(args)
    except ValueError as exc:
        print(f"rfdiffusion: {exc}", file=sys.stderr)
        print("run 'rfdiffusion --help' for usage", file=sys.stderr)
        return 2
    try:
        python_runtime = _resolve_python()
    except (OSError, RuntimeError) as exc:
        print(f"rfdiffusion: {exc}", file=sys.stderr)
        return 2
    if "inference.input_pdb" in overrides:
        target = Path(overrides["inference.input_pdb"])
        if not target.is_file():
            print(
                f"rfdiffusion: inference.input_pdb {str(target)!r} does not exist",
                file=sys.stderr,
            )
            return 2
    try:
        script = _resolve_run_inference()
    except RuntimeError as exc:
        print(f"rfdiffusion: {exc}", file=sys.stderr)
        return 2
    try:
        output_dir_path = Path(output_dir).absolute()
        output_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"rfdiffusion: cannot create --output_dir {output_dir!r}: {exc}", file=sys.stderr)
        return 2
    # Own the translation values (all covered by EXECUTION_CONTRACT_VERSION):
    #  * inference.output_prefix — absolutized so Hydra's job CWD (see below)
    #    can never relocate the designs: upstream writes <prefix>_<i_des>.pdb.
    #  * hydra.run.dir / output_subdir / job.chdir — confine Hydra's own
    #    metadata + logs under the identity output directory; the job keeps
    #    the caller's CWD (chdir=False), so nothing leaks into the caller's
    #    working tree and PDB output parsing is unaffected.
    overrides[INFERENCE_OUTPUT_PREFIX_FIELD] = str(output_dir_path / OUTPUT_PREFIX_NAME)
    overrides["hydra.run.dir"] = str(output_dir_path / "hydra")
    overrides["hydra.output_subdir"] = "null"
    overrides["hydra.job.chdir"] = "False"
    command = [
        python_runtime,
        str(script),
        "--config-name",
        "base",
        *[_hydra_override(key, value) for key, value in overrides.items()],
    ]
    # Clone-only deployment: prepend the resolved clone root so
    # run_inference.py can `import rfdiffusion` without the upstream
    # package being pip-installed. Preserve any existing PYTHONPATH.
    run_env = os.environ.copy()
    clone_root = str(script.parents[1])
    existing_pythonpath = run_env.get("PYTHONPATH")
    run_env["PYTHONPATH"] = (
        clone_root if not existing_pythonpath else clone_root + os.pathsep + existing_pythonpath
    )
    try:
        completed = subprocess.run(command, check=False, env=run_env, shell=False)
    except OSError as exc:
        print(f"rfdiffusion: failed to launch {script}: {exc}", file=sys.stderr)
        return 127
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
