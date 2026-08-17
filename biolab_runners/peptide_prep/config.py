"""Configuration for the peptide preparation runner.

The peptide-prep slice is the ``biolab-runners`` half of the cross-repo
Activin CHEM-001 prerequisite. It owns the filesystem / OpenMM /
PDBFixer / ParmEd execution that turns a candidate peptide backbone +
designed sequence into the per-candidate local artifacts E3 (the
downstream MD step) consumes:

* ``prepared.pdb`` — minimized, hydrogen-complete structure.
* ``prepared.top`` / ``prepared.gro`` — GROMACS export of the SAME
  OpenMM system/bond graph / net charge.

Inputs/outputs are GENERIC runner data; this module deliberately does
NOT reference Activin target policy or any consumer-project types.
The topology descriptor type aliases the upstream
:mod:`bioml_tools.chem.cyclic_topology` dataclasses under
``TYPE_CHECKING`` so pyright narrows correctly without forcing a
runtime dependency on bioml-tools at import time — the actual
DESCRIPTOR VALUES are passed in by the caller (Activin constructs them
from the same data Activin already uses for design validation).

The dataclasses are intentionally frozen and ``__post_init__``
validating: every failure mode is a :class:`ValueError` with a
specific message so the runner fails fast at the construction
boundary rather than deep inside an OpenMM call.

Closure bond-length physics (H5)
--------------------------------
The runner enforces explicit physical bond-length limits for
post-minimization closure distances. The defaults are
**engine-neutral** (they reflect chemical bond physics, not an
OpenMM/ParmEd detail):

* ``max_disulfide_distance_angstrom = 2.5`` — S-S covalent bond
  equilibrium is ~2.05 Å; anything >2.5 Å after restrained
  minimization means the two SG atoms are NOT covalently bound
  and the runner fails closed (a 7.6 Å "disulfide" is a real bond
  on paper only).
* ``max_head_to_tail_distance_angstrom = 2.0`` — peptide C-N
  equilibrium is ~1.33 Å; >2.0 Å means the closure bond is not
  a real bond.

Both are configurable but should NOT be relaxed to "make a test
pass" — that's the silent-bond-recording failure the probes
surfaced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_DISULFIDE_DISTANCE_A",
    "DEFAULT_MAX_HEAD_TO_TAIL_DISTANCE_A",
    "PeptidePrepConfig",
    "PeptideTopologyDescriptor",
]


# Engine-neutral physical bond-length limits used by the runner's
# closure-integrity check (H5). Exposed as module-level constants
# so tests can import them without reaching into the dataclass.
DEFAULT_MAX_DISULFIDE_DISTANCE_A = 2.5
DEFAULT_MAX_HEAD_TO_TAIL_DISTANCE_A = 2.0


# ---------------------------------------------------------------------------
# Topology descriptor (upstream dataclasses re-exported at the type level)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeptideTopologyDescriptor:
    """The aggregated set of modifications for one peptide backbone.

    Mirrors :class:`bioml_tools.chem.cyclic_topology.PreparedTopology`
    minus the ``original_length`` (the runner reads that from the
    designed sequence) and the human ``summary`` string (the runner
    records its own structured manifest). Each field defaults to the
    empty / ``None`` value so the descriptor is a no-op for linear
    all-L preparation (the common case for ``chem_001``-design
    peptides that don't need D-substitution / head-to-tail /
    disulfide closure).

    Fields:
        d_substitutions: 1-indexed positions that get D-residue
            coordinate transforms. ``bio``: :class:`DSubstitution`.
            Empty for all-L preparation.
        head_to_tail: Optional :class:`CyclicTerminus` describing the
            N-term-to-C-term closure bond. ``None`` for linear
            peptides.
        disulfides: 1-indexed cysteine pairs that get S-S bridges.
            Each entry is a :class:`DisulfideBond`. Empty for
            cysteine-free peptides.

    Runtime coupling to bioml-tools is forbidden; the actual VALUES
    are typed loosely so the caller (Activin) can pass instances
    constructed by ``bioml_tools.chem.cyclic_topology`` without a
    package-level import.
    """

    d_substitutions: tuple[Any, ...] = ()
    head_to_tail: Any | None = None
    disulfides: tuple[Any, ...] = ()

    def has_any_modification(self) -> bool:
        """Return True iff the descriptor specifies any non-trivial modification."""
        return bool(self.d_substitutions) or self.head_to_tail is not None or bool(self.disulfides)


# ---------------------------------------------------------------------------
# Prep configuration (the runner's primary dataclass)
# ---------------------------------------------------------------------------


# Canonical amino-acid 1-letter alphabet. ``X`` is rejected (ambiguous);
# ``U`` (selenocysteine) and ``O`` (pyrrolysine) are out of scope for
# ``chem_001``-era design.
_CANONICAL_AA_ALPHABET: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _empty_topology() -> PeptideTopologyDescriptor:
    return PeptideTopologyDescriptor()


@dataclass(frozen=True)
class PeptidePrepConfig:
    """Per-invocation configuration for the peptide-prep runner.

    The dataclass is frozen and every field is validated in
    ``__post_init__``. Required-path fields are empty strings; the
    constructor raises :class:`ValueError` with a specific message
    on missing/empty values so the runner never opens an
    undefined-path bug at runtime.

    Attributes:
        name: Logical run name. Used as the subdirectory under
            ``output_root`` (so multiple candidates can coexist).
        backbone_pdb: Path to the source backbone PDB (the input
            structure, with backbone coordinates only). Must exist
            on disk at construction time; the runner validates this
            and fails fast.
        sequence: Designed 1-letter amino-acid sequence. Must be a
            non-empty string using only the canonical 20-letter
            alphabet (``ACDEFGHIKLMNPQRSTVWY``); ambiguous letters
            (``X``) and exotic residues (``U``, ``O``) are rejected.
        chain_id: Chain identifier in ``backbone_pdb`` that holds
            the peptide chain. Defaults to ``"A"``.
        output_root: Root directory for the run's outputs. The
            runner writes everything under
            ``output_root / name``.
        topology: Topology modifications relative to the linear
            L-amino-acid backbone. Defaults to the empty
            descriptor (linear all-L preparation).
        protein_ff: OpenMM protein force-field XML filename. The
            runner loads this via ``app.ForceField``. Default
            ``"amber99sbildn.xml"`` matches the OpenMM extra and
            is what the Activin upstream pipeline uses.
        water_ff_xml: Water force-field XML for the ForceField
            loader. NOT used to solvate (the runner is unsolvated);
            the file is included so ``createSystem`` has the
            residue templates it expects for cysteine / NME / ACE
            patches. Default ``"tip3p.xml"``.
        minimization_max_iterations: Steepest-descent iteration cap
            for the restrained backbone minimization. Default 1 000
            (canonical for unsolvated peptide preparation).
        restraint_force_k_kjmol_nm2: Strength of the backbone
            N/CA/C positional restraint during minimization.
            Default 1 000 kJ/mol/nm² — strong enough to hold the
            backbone near the threaded coordinates while side-chain
            clashes relax. The restraint is REMOVED after
            minimization (not retained in the exported system).
        minimization_tolerance_kjmol_nm: Convergence tolerance for
            the minimization (passed to ``app.LocalEnergyMinimizer``
            via ``simulation.minimizeEnergy(tolerance=...)``).
            Default 10 kJ/mol/nm — the canonical unsolvated
            peptide value.
        max_disulfide_distance_angstrom: Post-minimization maximum
            SG-SG distance (Å) for a closure to count as covalent.
            Default 2.5 Å (S-S equilibrium ~2.05 Å). A 7.6 Å
            "disulfide" is a real bond on paper only — the runner
            fails closed when this limit is exceeded.
        max_head_to_tail_distance_angstrom: Post-minimization
            maximum C-N distance (Å) for head-to-tail closure.
            Default 2.0 Å (peptide C-N equilibrium ~1.33 Å).
        openmm_platform: OpenMM platform name. Default
            ``"Reference"`` — deterministic, no GPU dependency,
            so the cross-repo gate (CPU-only machines) works.
            The runner validates the platform exists at start;
            ``CPU`` is also accepted.
        coordinate_transformer_identity: Explicit caller-declared
            identity/version of the ``coordinate_transformer``
            callback (e.g. ``"bioml-tools 1.9.0
            construct_d_substitution_coordinates"``). REQUIRED
            (non-empty) when ``topology.d_substitutions`` is
            non-empty; the science config digest binds it so a
            changed transform invalidates the cached run. Optional
            and ignored for all-L preparation (no D-residue needs
            a transform). The runner still requires the callable
            itself at ``run()`` time; this field is the
            cache-binding identity, not the callable.
        chirality_validator_identity: Explicit caller-declared
            identity/version of the ``chirality_validator``
            callback. REQUIRED (non-empty) when
            ``topology.d_substitutions`` is non-empty (the runner
            cannot attest chirality without knowing which
            validator produced the reports); optional for all-L
            preparation.
        force: Re-run even if the manifest already covers this
            config. ``False`` by default; the runner
            QUARANTINES the existing artifacts into
            ``.stale/<UTC>/`` before re-running so a manual
            ``force=True`` cannot silently overwrite provenance.
            ``force`` is an execution control — it is NOT part of
            the science config digest, so a ``force=True`` rebuild
            is reusable by the next normal invocation.
        dry_run: Validate the config + plan, do not write
            outputs. The runner still emits the manifest and the
            prebuilt file digests, so a subsequent real run can
            short-circuit on idempotency. ``dry_run`` is an
            execution control — NOT part of the science config
            digest — and never reuses or overwrites a production
            manifest (a dry-run after a real run leaves the
            production binding intact).

    Side-channels the runner DOES NOT touch:

    * The user-supplied ``coordinate_transformer`` and
      ``chirality_validator`` callbacks (``runner.run(...)``).
      Their absence is enforced at run-time only when the
      descriptor requests D-substitution; an all-L prep needs no
      callback. See the runner docstring. The *identities* of
      those callbacks are config fields (above) so the science
      digest can bind them.
    """

    name: str = "peptide_prep"
    backbone_pdb: str = ""
    sequence: str = ""
    chain_id: str = "A"
    output_root: str = ""
    topology: PeptideTopologyDescriptor = field(default_factory=_empty_topology)

    # Force-field templates
    protein_ff: str = "amber99sbildn.xml"
    water_ff_xml: str = "tip3p.xml"

    # Minimization
    minimization_max_iterations: int = 1_000
    restraint_force_k_kjmol_nm2: float = 1000.0
    minimization_tolerance_kjmol_nm: float = 10.0

    # Closure bond-length limits (H5)
    max_disulfide_distance_angstrom: float = DEFAULT_MAX_DISULFIDE_DISTANCE_A
    max_head_to_tail_distance_angstrom: float = DEFAULT_MAX_HEAD_TO_TAIL_DISTANCE_A

    # Platform
    openmm_platform: str = "Reference"

    # Callback identities (science digest binding; required for D-substitution)
    coordinate_transformer_identity: str = ""
    chirality_validator_identity: str = ""

    # Idempotency
    force: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Validate required paths, sequence alphabet/length, and topology positions."""
        self._validate_paths()
        self._validate_sequence()
        self._validate_minimization()
        self._validate_closure_limits()
        self._validate_topology()
        self._validate_callback_identities()

    # --- validators ---

    def _validate_paths(self) -> None:
        if not self.name:
            raise ValueError("PeptidePrepConfig.name is required")
        if not self.backbone_pdb:
            raise ValueError("PeptidePrepConfig.backbone_pdb is required")
        if not self.output_root:
            raise ValueError("PeptidePrepConfig.output_root is required")
        if not self.chain_id:
            raise ValueError("PeptidePrepConfig.chain_id is required")

    def _validate_sequence(self) -> None:
        if not self.sequence:
            raise ValueError("PeptidePrepConfig.sequence is required (empty string rejected)")
        seq = self.sequence.upper()
        bad = sorted({c for c in seq if c not in _CANONICAL_AA_ALPHABET})
        if bad:
            raise ValueError(
                f"sequence contains invalid characters {bad!r}; "
                f"only the 20 canonical amino-acid 1-letter codes are accepted"
            )
        # Stash the uppercase form so the runner doesn't have to re-uppercase.
        if seq != self.sequence:
            object.__setattr__(self, "sequence", seq)

    def _validate_minimization(self) -> None:
        if self.minimization_max_iterations < 1:
            raise ValueError(
                f"minimization_max_iterations must be positive; "
                f"got {self.minimization_max_iterations}"
            )
        if self.restraint_force_k_kjmol_nm2 < 0:
            raise ValueError(
                f"restraint_force_k_kjmol_nm2 must be non-negative; "
                f"got {self.restraint_force_k_kjmol_nm2}"
            )
        if self.minimization_tolerance_kjmol_nm <= 0:
            raise ValueError(
                f"minimization_tolerance_kjmol_nm must be positive; "
                f"got {self.minimization_tolerance_kjmol_nm}"
            )

    def _validate_closure_limits(self) -> None:
        """Validate the H5 closure-integrity physical bond-length limits."""
        if self.max_disulfide_distance_angstrom <= 0:
            raise ValueError(
                f"max_disulfide_distance_angstrom must be positive; "
                f"got {self.max_disulfide_distance_angstrom}"
            )
        if self.max_head_to_tail_distance_angstrom <= 0:
            raise ValueError(
                f"max_head_to_tail_distance_angstrom must be positive; "
                f"got {self.max_head_to_tail_distance_angstrom}"
            )

    def _validate_topology(self) -> None:
        """Validate topology positions fall within the sequence and are well-formed.

        The runner delegates the ``bioml_tools.chem.cyclic_topology``
        style validation (e.g. ``_mutually_exclusive_termini``) to
        the upstream ``build_prepared_topology`` call site in
        Activin; this validator only enforces the invariants that
        affect the runner's correctness directly — namely, that
        every 1-indexed position falls inside the sequence, and
        for cyclic closures that head/tail are the true terminal
        positions (blocker #3: the runner supports only true
        head-to-tail terminal closure — arbitrary indices are
        rejected at config time so the runner never silently
        ignores them).
        """
        seq_len = len(self.sequence)
        topo = self.topology
        d_positions: list[int] = []

        for d in topo.d_substitutions:
            self._validate_d_substitution_entry(d, seq_len)
            d_positions.append(d.position)

        if d_positions and len(d_positions) != len(set(d_positions)):
            duplicate_positions = sorted(
                position for position in set(d_positions) if d_positions.count(position) > 1
            )
            raise ValueError(
                "topology.d_substitutions contains duplicate position(s) "
                f"{duplicate_positions!r}; each sequence position may be D-substituted exactly once"
            )

        if topo.head_to_tail is not None:
            self._validate_head_to_tail(topo.head_to_tail, seq_len)

        for bond in topo.disulfides:
            self._validate_disulfide_entry(bond, seq_len)

    def _validate_d_substitution_entry(self, d: object, seq_len: int) -> None:
        """Validate one ``DSubstitution``-shaped entry."""
        pos = getattr(d, "position", None)
        residue = getattr(d, "residue", None)
        if not isinstance(pos, int) or pos < 1 or pos > seq_len:
            raise ValueError(
                f"topology.d_substitutions entry has invalid position {pos!r}; "
                f"sequence length is {seq_len}"
            )
        if not isinstance(residue, str) or len(residue) != 3:
            raise ValueError(
                f"topology.d_substitutions entry has invalid residue {residue!r}; "
                f"expected a 3-letter amino-acid code"
            )

    def _validate_head_to_tail(self, ht: object, seq_len: int) -> None:
        """Validate one ``CyclicTerminus``-shaped entry.

        Blocker #3: head-to-tail closure is supported ONLY when
        ``head == 1`` and ``tail == len(sequence)`` (true terminal
        closure). The runner cannot template-match arbitrary
        indices as cyclized residues (amber99sbildn has no
        template for an internal-residue-as-closure; see the
        failure-classification comment in
        ``topology._verify_cyclic_topology_chemistry``), so we
        fail closed at the config boundary rather than silently
        dropping or silently rewriting the indices.
        """
        head = getattr(ht, "head", None)
        tail = getattr(ht, "tail", None)
        if not isinstance(head, int) or head < 1 or head > seq_len:
            raise ValueError(f"topology.head_to_tail.head {head!r} is out of range [1, {seq_len}]")
        if not isinstance(tail, int) or tail < 1 or tail > seq_len:
            raise ValueError(f"topology.head_to_tail.tail {tail!r} is out of range [1, {seq_len}]")
        if head == tail:
            raise ValueError(
                f"topology.head_to_tail head {head} == tail {tail}; head and tail "
                f"must be different (head-to-tail closure requires distinct terminal "
                f"residues). The runner supports only true head-to-tail terminal closure."
            )
        if head != 1 or tail != seq_len:
            raise ValueError(
                f"topology.head_to_tail head={head}, tail={tail}; "
                f"the runner supports only true terminal head-to-tail closure "
                f"(head must be 1 and tail must be {seq_len}). "
                f"Arbitrary mid-chain closure indices are rejected at config time "
                f"rather than silently ignored."
            )

    def _validate_disulfide_entry(self, bond: object, seq_len: int) -> None:
        """Validate one ``DisulfideBond``-shaped entry.

        The sequence character at a disulfide position MUST be CYS —
        non-cysteine disulfide pairs are physically impossible (no
        SG atom for amber99sbildn to template).
        """
        first = getattr(bond, "first", None)
        second = getattr(bond, "second", None)
        if not isinstance(first, int) or first < 1 or first > seq_len:
            raise ValueError(
                f"topology.disulfides entry has invalid first {first!r}; "
                f"sequence length is {seq_len}"
            )
        if not isinstance(second, int) or second < 1 or second > seq_len:
            raise ValueError(
                f"topology.disulfides entry has invalid second {second!r}; "
                f"sequence length is {seq_len}"
            )
        for label, pos in (("first", first), ("second", second)):
            aa = self.sequence[pos - 1]
            if aa != "C":
                raise ValueError(
                    f"disulfide {label} position {pos} has residue {aa!r}; disulfide requires CYS"
                )

    def _validate_callback_identities(self) -> None:
        """Require explicit callback identities when D-substitution is requested.

        D-substitution changes the geometry through a caller-supplied
        ``coordinate_transformer`` and is attested by a
        caller-supplied ``chirality_validator``. The runner cannot
        bind those callables into the science digest by name or
        pointer (non-reproducible), so the caller MUST declare an
        explicit identity/version string for each when any
        D-residue is present. The identity is part of the science
        config digest: changing it invalidates the cached run.

        All-L preparation needs no callback, so both identity
        fields stay optional (empty) and are ignored for the
        science digest in that case — keeping the non-D API
        ergonomic and backward compatible.
        """
        if not self.topology.d_substitutions:
            return
        if not self.coordinate_transformer_identity:
            raise ValueError(
                "coordinate_transformer_identity is required when "
                "topology.d_substitutions is set (D-substitution needs an "
                "explicitly versioned transform so the science digest can "
                "bind it); set it to e.g. 'bioml-tools 1.9.0 "
                "construct_d_substitution_coordinates'"
            )
        if not self.chirality_validator_identity:
            raise ValueError(
                "chirality_validator_identity is required when "
                "topology.d_substitutions is set (D-substitution needs an "
                "explicitly versioned chirality validator so the science "
                "digest can bind it)"
            )
