"""Small public contracts shared by the tool-specific runners.

The runners intentionally keep their own configuration and result records.
This module only standardises the vocabulary used when those records cross a
pipeline boundary.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = [
    "ArtifactReference",
    "ExecutionMode",
    "ExecutionStatus",
    "IncompleteOutputError",
    "MalformedOutputError",
    "RunnerError",
    "RunnerInterruptedError",
    "RunnerInvocationError",
    "RunnerOutputError",
    "RunnerTimeoutError",
    "RunnerUnavailableError",
    "artifact_from_path",
    "require_artifact",
    "validate_artifact_digest",
]


class ExecutionStatus(StrEnum):
    """Normalized status values emitted by runner result records."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"
    MALFORMED = "malformed"
    INCOMPLETE = "incomplete"
    CACHED = "cached"
    SKIPPED = "skipped"
    DRY_RUN = "dry_run"


class ExecutionMode(StrEnum):
    """How a runner performs work in the current process boundary."""

    IN_PROCESS = "in_process"
    SUBPROCESS = "subprocess"
    CONTAINER_URI = "container_uri"


class RunnerError(RuntimeError):
    """Base class for typed runner failures."""

    def __init__(self, message: str, *, runner: str | None = None) -> None:
        super().__init__(message)
        self.runner = runner


class RunnerUnavailableError(RunnerError):
    """The configured executable, dependency, or runtime is unavailable."""


class RunnerInvocationError(RunnerError):
    """An executable was available but invocation or execution failed."""


class RunnerTimeoutError(RunnerError):
    """A runner exceeded its configured time limit."""


class RunnerInterruptedError(RunnerError):
    """A runner was interrupted and may be resumable."""


class RunnerOutputError(RunnerError):
    """Required output was malformed, incomplete, missing, or unreadable."""


class MalformedOutputError(RunnerOutputError):
    """An output exists but does not satisfy its parser contract."""


class IncompleteOutputError(RunnerOutputError):
    """A required output artifact is absent or empty."""


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_OCI_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_artifact_digest(digest: str | None) -> str | None:
    """Return a canonical OCI digest, rejecting malformed values."""
    if digest is None:
        return None
    if _SHA256_OCI_RE.fullmatch(digest):
        return digest
    if _SHA256_HEX_RE.fullmatch(digest):
        return f"sha256:{digest}"
    raise ValueError(f"artifact digest must be sha256:<64 lowercase hex>; got {digest!r}")


@dataclass(frozen=True)
class ArtifactReference:
    """A serializable reference to one runner-produced artifact."""

    path: str
    digest: str | None = None
    required: bool = True
    kind: str | None = None

    def __post_init__(self) -> None:
        """Validate and canonicalize the digest supplied by a producer."""
        if not self.path:
            raise ValueError("artifact path must be non-empty")
        object.__setattr__(self, "digest", validate_artifact_digest(self.digest))

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        required: bool = True,
        kind: str | None = None,
    ) -> ArtifactReference:
        """Create a reference and digest an existing regular file."""
        candidate = Path(path)
        digest: str | None = None
        if candidate.is_file():
            digest = f"sha256:{_sha256(candidate)}"
        return cls(str(candidate), digest=digest, required=required, kind=kind)

    @property
    def exists(self) -> bool:
        """Whether the referenced path is a regular file."""
        return Path(self.path).is_file()

    def to_dict(self) -> dict[str, object]:
        """Serialize the reference into a JSON-safe dictionary."""
        return {
            "path": self.path,
            "digest": self.digest,
            "required": self.required,
            "kind": self.kind,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_from_path(
    path: str | Path,
    *,
    required: bool = True,
    kind: str | None = None,
) -> ArtifactReference:
    """Compatibility factory for callers that prefer a function API."""
    return ArtifactReference.from_path(path, required=required, kind=kind)


def require_artifact(
    path: str | Path,
    *,
    kind: str | None = None,
) -> ArtifactReference:
    """Return a reference or fail closed when a required file is unusable."""
    reference = ArtifactReference.from_path(path, required=True, kind=kind)
    candidate = Path(reference.path)
    if not candidate.is_file() or candidate.stat().st_size == 0:
        raise IncompleteOutputError(f"required artifact is missing or empty: {candidate}")
    return reference
