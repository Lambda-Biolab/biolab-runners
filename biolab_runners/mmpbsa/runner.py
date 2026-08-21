"""GmxMMPBSARunner (slice 14 / BMT-MD-001 optional gmx_MMPBSA integration).

Located at the biolab_runners.mmpbsa package root for symmetry with
``biolab_runners.gromacs``. The runner class delegates parser
work to ``biolab_runners.mmpbsa._parser``.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from biolab_runners.contracts import (
    ArtifactReference,
    ExecutionMode,
    ExecutionStatus,
    RunnerInvocationError,
    RunnerUnavailableError,
)
from biolab_runners.mmpbsa._parser import (
    GmxMMPBSARecord,
    parse_residue_decomposition,
)
from biolab_runners.openmm.config import OpenMMConfig  # noqa: TC001 — runtime use as field type
from biolab_runners.provenance import (
    build_execution_provenance,
    compute_config_digest,
    compute_executed_config_digest,
    compute_file_digest,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GmxMMPBSARunner",
    "GmxMMPBSAStatus",
]


class GmxMMPBSAStatus:
    """Normalized outcome values for the gmx_MMPBSA runner.

    The class lives in :mod:`biolab_runners.mmpbsa.runner` because
    :meth:`GmxMMPBSARunner.run` is what emits these status values
    — keeping the constants adjacent to the only emitter makes
    future divergence (one side changing a string, the other
    forgetting to mirror it) impossible. The package
    :mod:`biolab_runners.mmpbsa.__init__` re-exports this class so
    consumers always import a single class object regardless of
    which submodule they came in through.
    """

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class GmxMMPBSARunner:
    """Wrapper around the gmx_MMPBSA CLI for per-residue decomposition."""

    config: OpenMMConfig
    prefix: str
    mmpbsa_binary: str = "gmx_MMPBSA"
    timeout_seconds: int = 1800

    def _build_command(self) -> list[str]:
        """Build the gmx_MMPBSA CLI invocation.

        The exact flag set varies by gmx_MMPBSA version; the
        canonical ``complex / protein / ligand`` triple plus
        ``--use-md-decomposition residue`` is the minimum needed
        for per-residue decomposition.
        """
        return [
            self.mmpbsa_binary,
            "--use-md-decomposition",
            "residue",
            "--complex-type",
            "protein-ligand",
            "--complex-md",
            str(self.config.receptor_pdb),
            "--ligand-md",
            str(self.config.peptide_pdb),
            "--prefix",
            str(self.prefix),
        ]

    def _load_records(self) -> tuple[GmxMMPBSARecord, ...]:
        """Find and parse the per-residue decomposition file.

        Different gmx_MMPBSA versions emit different filenames; we
        accept the common variants. If no matching file is found,
        returns an empty tuple — the caller treats this as
        ``unsupported``.
        """
        candidate_names = (
            "residue_decomposition_finite.dat",
            "residue_decomposition.dat",
            "FINAL_RESULTS_MMPBSA.dat",
        )
        for name in candidate_names:
            path = Path(self.config.output_dir) / f"{self.prefix}_{name}"
            if path.exists() and not path.is_symlink():
                return parse_residue_decomposition(path)
        return ()

    def run(self) -> dict[str, object]:
        """Execute gmx_MMPBSA for per-residue decomposition.

        Returns a JSON-stable dict with ``status``, ``records``,
        ``error``. When the binary isn't on PATH (or the --version
        probe fails), returns ``status="unsupported"`` with empty
        records — slice 14 acceptance criterion: missing optional
        tooling yields ``unsupported``, not a fabricated value.
        """
        from biolab_runners.mmpbsa.utils import gmx_mmpbsa_available

        if self.mmpbsa_binary.startswith("container://"):
            return self._result(
                status=ExecutionStatus.UNSUPPORTED,
                binary=self.mmpbsa_binary,
                prefix=self.prefix,
                per_residue_records=[],
                error=(
                    "container:// gmx_MMPBSA execution is unsupported; "
                    "resolve the image to a host executable before invoking"
                ),
                exit_code=127,
            )
        if not gmx_mmpbsa_available(binary=self.mmpbsa_binary):
            return self._result(
                status=GmxMMPBSAStatus.UNSUPPORTED,
                binary=self.mmpbsa_binary,
                prefix=self.prefix,
                per_residue_records=[],
                error=f"{self.mmpbsa_binary} not on PATH",
                exit_code=127,
            )
        cmd = self._build_command()
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
                cwd=self.config.output_dir,
            )
        except subprocess.TimeoutExpired:
            return self._result(
                status=ExecutionStatus.TIMEOUT,
                binary=self.mmpbsa_binary,
                prefix=self.prefix,
                per_residue_records=[],
                error=f"gmx_MMPBSA timed out after {self.timeout_seconds}s",
                exit_code=124,
                command=tuple(cmd),
                executed=True,
            )
        except FileNotFoundError as exc:
            raise RunnerUnavailableError(
                f"gmx_MMPBSA executable unavailable: {self.mmpbsa_binary}",
                runner="gmx_mmpbsa",
            ) from exc
        except OSError as exc:
            raise RunnerInvocationError(
                f"gmx_MMPBSA invocation failed: {exc}", runner="gmx_mmpbsa"
            ) from exc
        if completed.returncode != 0:
            return self._result(
                status=GmxMMPBSAStatus.FAILED,
                binary=self.mmpbsa_binary,
                prefix=self.prefix,
                per_residue_records=[],
                error=(completed.stderr or completed.stdout).strip()[-2000:],
                exit_code=completed.returncode,
                command=tuple(cmd),
                executed=True,
            )
        records = self._load_records()
        status = ExecutionStatus.SUCCEEDED if records else ExecutionStatus.INCOMPLETE
        error = "" if records else "required decomposition output is missing or empty"
        return self._result(
            status=status,
            binary=self.mmpbsa_binary,
            prefix=self.prefix,
            per_residue_records=[r.to_dict() for r in records],
            error=error,
            command=tuple(cmd),
            executed=True,
        )

    def _result(self, **payload: object) -> dict[str, object]:
        """Add shared execution fields while preserving the legacy dict shape."""
        status = ExecutionStatus(payload.get("status", ExecutionStatus.FAILED)).value
        raw_exit_code = payload.get("exit_code", 0)
        exit_code = raw_exit_code if isinstance(raw_exit_code, int) else 0
        raw_command = payload.get("command", ())
        command = (
            tuple(str(item) for item in raw_command)
            if isinstance(raw_command, (list, tuple))
            else ()
        )
        executed = bool(payload.get("executed", False))
        artifacts = self._artifacts()
        mode = (
            ExecutionMode.CONTAINER_URI
            if self.mmpbsa_binary.startswith("container://")
            else ExecutionMode.SUBPROCESS
        )
        result = dict(payload)
        result["execution_mode"] = mode
        result["exit_code"] = exit_code
        result["artifacts"] = [artifact.to_dict() for artifact in artifacts]
        result["provenance"] = build_execution_provenance(
            runner_name="gmx_mmpbsa",
            execution_mode=mode,
            status=status,
            exit_code=exit_code,
            artifacts=artifacts,
            command=command,
            executed=executed,
            cache_hit=bool(payload.get("cache_hit", False)),
            requested_config_digest=compute_config_digest(self.config),
            executed_config_digest=(
                compute_executed_config_digest(
                    {
                        "command": list(command),
                        "prefix": self.prefix,
                        "input_digests": {
                            "receptor_pdb": compute_file_digest(Path(self.config.receptor_pdb)),
                            "peptide_pdb": compute_file_digest(Path(self.config.peptide_pdb)),
                        },
                    }
                )
                if executed
                else None
            ),
            source_backbone_digest=compute_file_digest(Path(self.config.receptor_pdb)),
        ).to_dict()
        return result

    def _artifacts(self) -> tuple[ArtifactReference, ...]:
        """Return references to decomposition files that exist."""
        candidate_names = (
            "residue_decomposition_finite.dat",
            "residue_decomposition.dat",
            "FINAL_RESULTS_MMPBSA.dat",
        )
        return tuple(
            ArtifactReference.from_path(path, kind="decomposition")
            for path in (
                Path(self.config.output_dir) / f"{self.prefix}_{name}" for name in candidate_names
            )
            if path.is_file() and not path.is_symlink()
        )
