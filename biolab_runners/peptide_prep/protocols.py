"""Typed Protocols for the runner's two injection points.

The peptide-prep runner is engine-neutral with respect to the
bioml-tools coordinate-transformation and chirality-validation
vocabularies (per AGENTS.md: ``biolab_runners`` MUST NOT runtime-
import bioml-tools). Two narrowly-scoped Protocols cover what the
runner actually needs from the caller — the cross-repo seam — and
nothing more.

Linear all-L preparation needs NEITHER callback. D-substitution
requires BOTH (the runner fails closed if either is missing when a
D-residue is requested). The Protocols are defined as Python
``runtime_checkable`` Protocols so the runner can ``isinstance``
guard the injected callables without forcing the caller to inherit
from a base class.

Callback compatibility
-----------------------
The runner accepts EITHER the documented bare ``dict`` mapping OR a
:class:`CoordinateTransformResult` wrapper. Both forms are accepted
at the same call site; the runner unwraps the wrapper transparently
so adapters built against ``bioml_tools.chem.cyclic_topology``'s
``CoordinateTransformResult.mapping`` (and the canonical
``construct_d_substitution_coordinates`` / ``validate_ca_chirality``
functions) drop in without forcing a bioml-tools runtime import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "ChiralityReport",
    "ChiralityValidator",
    "CoordinateTransformResult",
    "CoordinateTransformer",
    "extract_coordinate_mapping",
]


# ---------------------------------------------------------------------------
# D-coordinate transformer
# ---------------------------------------------------------------------------


@runtime_checkable
class CoordinateTransformer(Protocol):
    """Apply the D-residue mirror transform for a single residue.

    The runner collects, per non-Gly residue, the side-chain / backbone
    atom NAME → coordinate (Å) mapping in PDBFixer's OpenMM
    ``Topology`` ordering; the callable returns the transformed
    mapping (in the SAME NAME → coordinate form). The implementation
    is expected to detect D-substitutions from the residue name
    itself (DALA, DLEU, etc.) OR from external state — the runner
    supplies the mapping and asks for the mapping back; the callable
    is free to apply D / not based on whatever heuristic the
    Activin-side D-detection uses.

    The runner accepts the returned value as EITHER a bare dict
    (``{atom_name: (x, y, z)}``) OR a :class:`CoordinateTransformResult`
    wrapper; the runner unwraps the wrapper via
    :func:`extract_coordinate_mapping`. Implementations are free
    to choose whichever form matches their internal style.

    Args:
        mapping: ``{atom_name_str: (x_angstrom, y_angstrom, z_angstrom)}``
            keyed by the OpenMM atom name (``"CA"``, ``"CB"``, ...).
        residue_name: 3-letter residue name (e.g. ``"ALA"``).
        residue_index: 0-indexed position in the chain.

    Returns:
        A mapping of the same shape (atom name → 3-tuple of Å),
        mutated to reflect D-residue chirality. May be the SAME
        dict object (in-place) or a new dict; the runner treats
        it as opaque. May be wrapped in a
        :class:`CoordinateTransformResult`.
    """

    def __call__(
        self,
        mapping: dict[str, tuple[float, float, float]],
        residue_name: str,
        residue_index: int,
        **kwargs: object,
    ) -> dict[str, tuple[float, float, float]] | CoordinateTransformResult:
        """Apply the transform and return the (possibly wrapped) mapping."""
        ...


# ---------------------------------------------------------------------------
# Typed wrapper accepted from CoordinateTransformer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoordinateTransformResult:
    """The transformed coordinate mapping (returned by a CoordinateTransformer).

    Thin typed wrapper mirroring
    ``bioml_tools.chem.cyclic_topology.CoordinateTransformResult``.
    The shape is intentionally identical so adapters drop in
    without modification.

    Attributes:
        mapping: Atom name → (x, y, z) in Å.
        residue_name: 3-letter residue name passed through.
        residue_index: 0-indexed chain position passed through.
    """

    mapping: dict[str, tuple[float, float, float]]
    residue_name: str
    residue_index: int


def extract_coordinate_mapping(
    result: object,
) -> dict[str, tuple[float, float, float]]:
    """Unwrap a transformer return value into the bare mapping dict.

    The runner calls the transformer per-residue and accepts EITHER
    form:

    * a bare ``dict`` — returned as-is.
    * a :class:`CoordinateTransformResult` — ``.mapping`` extracted.

    Anything else raises :class:`TypeError` (the runner fails
    closed; the caller's adapter is misconfigured).

    The parameter is typed as ``object`` (not the bare ``dict |
    CoordinateTransformResult`` union) because pyright in strict
    mode flags the runtime isinstance guard as redundant when
    the union is statically known — but the runtime check IS the
    guard, and callers (bioml adapters) may pass loose objects
    when misconfigured.
    """
    if isinstance(result, CoordinateTransformResult):
        return result.mapping
    if isinstance(result, dict):
        return result
    raise TypeError(
        f"coordinate_transformer must return a dict or "
        f"CoordinateTransformResult; got {type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# Chirality validator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChiralityReport:
    """One residue's chirality validation result.

    Attributes:
        residue_index: 0-indexed chain position of the validated residue.
        residue_name: 3-letter residue name at validation time.
        expected: ``"L"`` or ``"D"`` — what the runner asked the
            validator to find.
        observed: ``"L"``, ``"D"``, or ``"ambiguous"`` (the validator
            could not decide).
        valid: True iff ``observed == expected``.
        detail: Optional free-form note from the validator. The
            ``runner`` does not parse this; it is recorded verbatim
            in the manifest for downstream audit.
    """

    residue_index: int
    residue_name: str
    expected: str
    observed: str
    valid: bool
    detail: str = ""


@runtime_checkable
class ChiralityValidator(Protocol):
    """Validate the chirality of every non-Gly residue in a peptide.

    The runner calls the validator THREE times per non-Gly residue
    (the explicit ``stage=`` kwarg names which), per CHEM-001:

    * ``stage="post_h"`` — post-hydrogenation, pre-D-transform.
      Side-chain / head-N-H orientation that the hydrogen-add
      step produced. The D-coord transform has not run; a
      designated D residue is still in its pre-transform L
      geometry and the runner passes ``expected="L"`` here.
    * ``stage="pre"`` — post-D-transform, pre-minimization. The
      D-coordinate transform has run; designated D residues are
      in D geometry and the runner passes ``expected="D"``.
    * ``stage="post"`` — post-minimization. Same expectations as
      ``"pre"`` (minimization is restrained and must not flip the
      D orientation).

    There is no pre-hydrogenation validation stage; the runner
    invokes the validator only after ``addMissingHydrogens`` so
    every call sees a hydrogen-complete residue.

    The runner forwards ``**kwargs`` audit context (``stage``) to
    every validator call. Validators that surface signature
    enforcement (i.e. refuse unknown kwargs) will raise
    ``TypeError``; the runner treats that as a structured failure
    — not as an uncaught exception that escapes the orchestrator.

    Any ``ChiralityReport`` with ``valid=False`` fails the run —
    the runner does NOT waive failures (per the CHEM-001 contract
    that every residue either passes validation or the run is
    reported as invalid).

    Args:
        mapping: ``{atom_name_str: (x_angstrom, y_angstrom, z_angstrom)}``
            for ONE residue. The validator is invoked per-residue.
        residue_name: 3-letter residue name.
        residue_index: 0-indexed chain position.
        expected: ``"L"`` or ``"D"`` — the chirality the topology
            descriptor demands for this position (stage-conditional;
            see class docstring).
        **kwargs: Audit context forwarded by the runner. The
            documented kwargs are ``stage`` (one of ``"post_h"``,
            ``"pre"``, ``"post"``); production validators may ignore
            it.

    Returns:
        A :class:`ChiralityReport` summarising the observation.
    """

    def __call__(
        self,
        mapping: dict[str, tuple[float, float, float]],
        residue_name: str,
        residue_index: int,
        *,
        expected: str,
        **kwargs: object,
    ) -> ChiralityReport:
        """Validate chirality for one residue; return a structured report."""
        ...
