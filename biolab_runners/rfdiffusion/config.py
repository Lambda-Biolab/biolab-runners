"""Configuration for an RFdiffusion backbone generation.

Mirrors the upstream CLI flags that the Activin-E pipeline actually
uses. Every field has a conservative default so a bare-minimum
campaign still produces a valid result.

The runner accepts the upstream ``contigmap.contigs`` syntax
(e.g. ``"12-18 A3-117/0"``) so callers familiar with
RFdiffusion can use the documentation directly. ``contigs`` is
forwarded **byte-for-byte** to the CLI — the runner never parses or
rewrites it, and it never invents a shorthand chain parser.

Target-conditioned binder design:

* Set ``target_pdb`` to a PDB containing the fixed target chain(s)
  and use stock contig syntax that references them — e.g.
  ``contigs="A1-110/0 B1-110/0 14-18"``: two fixed target chains
  (A, B) followed by a generated 14–18-residue binder segment.
  ``target_pdb`` is forwarded as the canonical stock Hydra key
  ``inference.input_pdb``.
* Binder contigs (those referencing chain IDs) and hotspots
  (``ppi.hotspot_res`` — input-PDB chain residues) **require**
  ``target_pdb``. Stock upstream silently substitutes a bundled
  example PDB when ``inference.input_pdb`` is unset
  (``rfdiffusion/inference/model_runners.py``), so a chain-
  referencing contig or hotspot list without a target would design
  against the wrong structure. The config rejects those
  combinations (fail closed), and a set-but-missing ``target_pdb``
  file is a hard error at run time.
* With an empty ``target_pdb`` the run is generic **unconditional**
  generation (e.g. the default ``contigs="14-18"``) — backward
  compatible, but that is **not** a binder.

Topology modes are truthful about what stock RFdiffusion can do:
``inference.cyclic`` / ``inference.cyc_chains`` only express
head-to-tail cyclization of named chains — they cannot encode
residue-pair disulfides. The runner therefore emits cyclic flags
only for ``mode="head_to_tail"`` and
``mode="head_to_tail_and_disulfide"``; plain ``mode="disulfide"``
is **not** cyclic, and ``disulfide_pairs`` is kept in the config /
provenance as downstream topology intent (closure is applied and
validated downstream, e.g. by ``biolab_runners.peptide_prep``, not
by RFdiffusion).

S2 reproducibility fields (per the Activin-E reproducibility plan):

* ``seed`` — the user-facing non-negative base seed. Stock upstream
  RFdiffusion has **no** ``inference.seed`` key (a wrapper that
  appends it via Hydra's ``+inference.seed=...`` would have it
  silently ignored — it is inert, nothing reads it; a strict
  override is rejected; either way it never affects the RNG). Its
  deterministic mode seeds each design with the design index. The
  supported external base is ``inference.design_startnum``, and the
  runner maps ``seed`` → ``inference.design_startnum`` when
  ``deterministic=True``. Concretely, upstream's
  ``scripts/run_inference.py`` seeds design ``i_des`` with
  ``i_des`` inside ``for i_des in range(design_startnum,
  design_startnum + num_designs)``, so the per-design seeds are
  ``seed, seed+1, ..., seed + task_count - 1`` and output
  indices/names start at ``seed``. The provenance manifest records
  ``base_seed == requested_seed == seed``; the per-design range is
  encoded by ``base_seed`` + ``task_count`` (no per-seed list is
  fabricated). When ``deterministic=False`` the seed is
  deliberately **not** forwarded — upstream uses system entropy, so
  claiming a pinned seed would be dishonest (the manifest records
  ``base_seed=None`` and ``rng_intent="non-deterministic"``).
* **Default ``seed=0`` is reproducible, not diverse.** With
  ``deterministic=True`` (the default) the same config always
  produces the *same* designs (per-design seeds
  ``0..task_count-1``). Callers that want distinct designs across
  runs or replicas must vary ``seed`` explicitly.
* ``checkpoint`` — the model / checkpoint identifier that flows
  into the provenance record. Defaults to ``"RFdiffusion"``
  because the upstream container ships a single model; callers
  pinning a custom checkpoint can override.

Note on temperature: RFdiffusion does not expose a sampling
temperature in the sense ProteinMPNN does — its diffusion process
is parameterised by noise scales (``diffusion.noise_scale_ca``,
``diffusion.noise_scale_frame``) that are tuned per-application
and live upstream-internal. We deliberately do **not** surface a
``temperature`` field here; mapping to ``noise_scale_ca`` would
silently change the upstream behaviour and is not part of the
canonical contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["RESERVED_CANONICAL_KEYS", "UNSUPPORTED_UPSTREAM_KEYS", "ContigMap", "RFdiffusionConfig"]


#: Canonical Hydra keys the runner emits itself (or that the in-package
#: ``rfdiffusion`` console script owns, like ``inference.output_prefix``).
#: ``extra`` may not override them — a conflict raises ``ValueError`` at
#: config construction time (fail closed) so a caller cannot silently
#: change what the runner forwards to upstream.
RESERVED_CANONICAL_KEYS: tuple[str, ...] = (
    "inference.num_designs",
    "inference.design_startnum",
    "inference.deterministic",
    "inference.input_pdb",
    "inference.cyclic",
    "inference.cyc_chains",
    "inference.output_prefix",
    "contigmap.contigs",
    "ppi.hotspot_res",
)

#: Upstream keys that do NOT exist in stock RFdiffusion and must never
#: be forwarded. ``inference.seed`` is not a Hydra key in
#: ``config/inference/base.yaml``: a wrapper that appends it (``+``
#: override) would have it silently ignored — it is inert, nothing in
#: upstream reads it — and a strict override is rejected. Either way
#: it never affects the RNG. The supported base is
#: ``inference.design_startnum``. Passing these via ``extra`` raises a
#: clear ``ValueError`` instead of forwarding a key that can only be
#: inert-or-rejected upstream.
UNSUPPORTED_UPSTREAM_KEYS: tuple[str, ...] = ("inference.seed",)


@dataclass(frozen=True)
class ContigMap:
    """Parsed upstream ``contigmap.contigs`` value.

    The runner passes ``contigs`` through as a string to the CLI;
    this dataclass exists for validation and for callers that prefer
    programmatic construction.
    """

    contigs: str
    length_min: int
    length_max: int

    def __post_init__(self) -> None:
        """Validate contigs + length range."""
        if not self.contigs:
            raise ValueError("contigs must be a non-empty string")
        if self.length_min < 1 or self.length_max < self.length_min:
            raise ValueError(f"length range invalid: min={self.length_min} max={self.length_max}")


@dataclass(frozen=True)
class RFdiffusionConfig:
    """Per-invocation configuration for the RFdiffusion runner.

    The defaults are generic **unconditional** generation: a linear
    14–18-residue chain with no target PDB and no hotspots. That is
    **not** a binder — binder design requires ``target_pdb`` plus
    stock ``contigs`` that reference the target chain IDs (see the
    module docstring). Head-to-tail macrocycles, disulfides, and
    their combination are opt-in via ``mode`` / ``disulfide_pairs``;
    cyclic flags are emitted only for the head-to-tail variants, and
    ``disulfide_pairs`` is downstream closure intent (RFdiffusion
    cannot encode disulfides).
    """

    name: str = "backbone"
    task_count: int = 1000
    target_pdb: str = ""
    contigs: str = "14-18"
    length_min: int = 14
    length_max: int = 18
    mode: str = "linear"  # linear | head_to_tail | disulfide | head_to_tail_and_disulfide
    disulfide_pairs: tuple[tuple[int, int], ...] = ()
    #: Output chain to cyclize head-to-tail. Stock ``inference.cyc_chains``
    #: names output chains; generated (inpainted) chains are emitted first
    #: as ``A``, ``B``, ... so the default ``"a"`` cyclizes the first
    #: generated chain — the binder in every single-segment binder contig.
    #: Callers whose binder is a different generated chain set it explicitly.
    cyc_chains: str = "a"
    hotspots: tuple[str, ...] = ()
    deterministic: bool = True
    seed: int = 0
    checkpoint: str = "RFdiffusion"
    extra: Mapping[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """Validate name + mode + topology + S2 fields + extra keys."""
        _validate_name(self)
        _validate_mode_and_lengths(self)
        _validate_cyc_chains(self)
        _validate_target_intent(self)
        _validate_s2_fields(self)
        _validate_extra_keys(self)


def _validate_name(cfg: RFdiffusionConfig) -> None:
    """Validate ``name`` as a single safe path component.

    The runner places outputs at ``<output_root>/<name>/<digest>/``; a
    name containing path separators, NUL, or control characters (or
    the ``.`` / ``..`` components) would escape the per-name directory
    or produce an unusable path. Rejected at construction (fail
    closed) rather than silently mangled.
    """
    name = cfg.name
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or any(ord(ch) < 32 for ch in name)
    ):
        raise ValueError(
            "name must be a single safe path component (no separators, "
            f"NUL, or control characters); got {name!r}"
        )


_VALID_MODES = frozenset({"linear", "head_to_tail", "disulfide", "head_to_tail_and_disulfide"})


def _validate_mode_and_lengths(cfg: RFdiffusionConfig) -> None:
    """Validate the mode / length / disulfide-pair contract.

    Trigger / non-trigger pairs:

    * ``disulfide`` and ``head_to_tail_and_disulfide`` **require** at
      least one ``disulfide_pairs`` entry (the mode names claim
      disulfide intent; an empty tuple would silently drop it).
    * ``linear`` rejects ``disulfide_pairs`` (no closure intent).
    * ``head_to_tail`` may carry pairs — a combined head-to-tail +
      disulfide config, preserved for existing consumers; the pairs
      are downstream closure intent and are never encoded into
      ``inference.cyc_chains``.
    """
    if cfg.mode not in _VALID_MODES:
        raise ValueError(
            "mode must be one of linear/head_to_tail/disulfide/"
            f"head_to_tail_and_disulfide; got {cfg.mode!r}"
        )
    if cfg.length_min < 1 or cfg.length_max < cfg.length_min:
        raise ValueError(f"length range invalid: min={cfg.length_min} max={cfg.length_max}")
    if cfg.mode in {"disulfide", "head_to_tail_and_disulfide"} and not cfg.disulfide_pairs:
        raise ValueError(f"mode={cfg.mode} requires at least one configured pair")
    if cfg.mode == "linear" and cfg.disulfide_pairs:
        raise ValueError(
            "disulfide_pairs only valid when mode=disulfide, head_to_tail, "
            "or head_to_tail_and_disulfide"
        )


def _validate_cyc_chains(cfg: RFdiffusionConfig) -> None:
    """Validate ``cyc_chains`` as exactly one ASCII chain letter.

    Stock ``inference.cyc_chains`` is a string naming the output
    chains to cyclize head-to-tail; upstream uppercases it internally
    and matches ``contigmap`` chain IDs (``_init_cyclic_reses`` in
    ``rfdiffusion/inference/model_runners.py``). Generated
    (inpainted) chains are emitted first as ``A``, ``B``, ... (see
    ``rfdiffusion/contigs.py``), so the default ``"a"`` cyclizes the
    first generated chain — the binder in every single-segment binder
    contig. A multi-letter value is rejected here: the runner names
    exactly one binder chain, and a caller whose binder is a
    different generated chain must set it explicitly rather than
    have the runner guess.
    """
    value = cfg.cyc_chains
    if len(value) != 1 or not value.isascii() or not value.isalpha():
        raise ValueError(
            "cyc_chains must be exactly one ASCII chain letter (the generated "
            f"chain to cyclize head-to-tail); got {value!r}"
        )


#: Stock contig syntax uses chain letters (``A1-110``) to reference
#: fixed chains from ``inference.input_pdb``; pure length contigs are
#: digits/``-``/``/`` only (e.g. the default ``14-18``).
_CHAIN_REF_RE = re.compile(r"[A-Za-z]")


def _validate_target_intent(cfg: RFdiffusionConfig) -> None:
    """Target-conditioned fields require ``target_pdb`` (fail closed).

    Two triggers, both rooted in stock upstream behaviour:

    * Chain-referencing contigs (e.g. ``A1-110/0 B1-110/0 14-18``)
      are target-conditioned: upstream parses ``input_pdb`` to
      extract those chains.
    * ``hotspots`` (``ppi.hotspot_res``) reference input-PDB chain
      residues (``"A51"`` = chain A residue 51) — they are
      meaningless without an input PDB even when ``contigs`` is a
      pure length spec like the default ``14-18``.

    When ``inference.input_pdb`` is unset, stock
    ``rfdiffusion/inference/model_runners.py`` silently substitutes a
    bundled example PDB — the run would proceed against the wrong
    structure while the caller believes they designed a binder. Fail
    closed at construction instead. Pure length contigs with no
    hotspots are unconditional and remain valid with an empty
    ``target_pdb``.
    """
    if not cfg.target_pdb and _CHAIN_REF_RE.search(cfg.contigs):
        raise ValueError(
            "contigs reference chain IDs (target-conditioned binder design) "
            "but target_pdb is empty; stock RFdiffusion would silently fall "
            "back to a bundled example PDB — set target_pdb to the target "
            "structure"
        )
    if not cfg.target_pdb and cfg.hotspots:
        raise ValueError(
            "hotspots require target_pdb: ppi.hotspot_res references "
            "input-PDB chain residues, but target_pdb is empty; stock "
            "RFdiffusion would silently fall back to a bundled example PDB — "
            "set target_pdb to the target structure"
        )


def _validate_s2_fields(cfg: RFdiffusionConfig) -> None:
    """Validate the S2 reproducibility fields (seed / checkpoint)."""
    if cfg.seed < 0:
        raise ValueError(f"seed must be ≥ 0; got {cfg.seed}")
    if not cfg.checkpoint:
        raise ValueError("checkpoint must be a non-empty model identifier")


def _validate_extra_keys(cfg: RFdiffusionConfig) -> None:
    """Reject ``extra`` entries that would corrupt the upstream contract.

    Fail closed, three ways:

    1. Canonical Hydra keys the runner emits itself (or that the
       console script owns — ``inference.num_designs`` /
       ``inference.design_startnum`` / ``inference.deterministic`` /
       ``inference.input_pdb`` / ``inference.cyclic`` /
       ``inference.cyc_chains`` / ``inference.output_prefix`` /
       ``contigmap.contigs`` / ``ppi.hotspot_res``, ...) may
       not be overridden via ``extra`` — a conflict raises
       ``ValueError`` instead of a last-write-wins dict merge.
    2. Keys that do not exist upstream (``inference.seed``) raise a
       clear ``ValueError`` rather than silently forwarding a key the
       stock Hydra schema cannot parse.
    3. Non-string keys raise ``ValueError`` — the CLI layer coerces
       every ``extra`` value to ``str``, so a non-string *key* would
       produce an unusable flag.
    """
    for key in cfg.extra:
        # Reason: ``extra`` is annotated Mapping[str, Any] but callers can
        # still pass non-str keys at runtime (the annotation is advisory);
        # the CLI layer would crash later with an opaque AttributeError, so
        # reject the config here with a clear error.
        if not isinstance(key, str):  # type: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"extra keys must be strings; got {type(key).__name__}")
        if key in UNSUPPORTED_UPSTREAM_KEYS:
            raise ValueError(
                f"extra key {key!r} is not supported by upstream RFdiffusion; "
                f"set RFdiffusionConfig.seed instead (forwarded as "
                f"inference.design_startnum when deterministic=True)"
            )
        if key in RESERVED_CANONICAL_KEYS:
            raise ValueError(f"extra cannot override reserved canonical keys: {key}")
