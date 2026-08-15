"""Peptide preparation runner — public orchestrator.

This module ties the config, the protocols, the topology layer,
the chemistry / minimization / export modules, and the manifest
I/O together. The single public entry point is
:class:`PeptidePrepRunner` — the result is :class:`PeptidePrepResult`.

The runner is deliberately conservative: every failure mode is
surfaced as a ``PeptidePrepResult(success=False, error=...)``
record so callers (e.g. the Activin orchestrator) can branch on
``result.success`` rather than on whether a ``RuntimeError``
raised. The runner never silently overwrites provenance — a
``force=True`` invocation QUARANTINES the prior artifacts into
``.stale/<UTC>/`` before re-running.

Idempotency:

* No existing manifest → fresh run.
* Existing manifest + matching source-digest + config-digest
  + every output-digest → ``reused=True`` (no work done).
* Existing manifest + any mismatch → ``error`` populated unless
  ``force=True``, in which case the prior artifacts are
  quarantined and a fresh run is performed.

Closure bond integrity (H5):

* After minimization, the runner reads every closure bond's
  end-to-end distance (Å) and fails closed if the distance
  exceeds the configured ``max_disulfide_distance_angstrom``
  (S-S, default 2.5) or ``max_head_to_tail_distance_angstrom``
  (C-N, default 2.0). A 7.6 Å "disulfide" or a 5.0 Å
  "head-to-tail" closure is a real bond on paper only — the
  runner refuses to write ``prepared.pdb`` / ``prepared.top`` /
  ``prepared.gro`` for those inputs.

Callback compatibility (H4):

* The runner accepts :class:`CoordinateTransformer` and
  :class:`ChiralityValidator` callbacks that match the documented
  Protocols OR the upstream bioml-tools surfaces
  (``construct_d_substitution_coordinates`` /
  ``validate_ca_chirality``). Adapter glue is provided by the
  tests; the runner is engine-neutral and does not import
  bioml-tools at runtime.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biolab_runners.peptide_prep.utils import (
    TopologyBondRecord,
    collect_atom_mapping,
    file_sha256,
    manifest_load,
    manifest_save,
)

if TYPE_CHECKING:
    from biolab_runners.peptide_prep.config import PeptidePrepConfig
    from biolab_runners.peptide_prep.protocols import (
        ChiralityReport,
        ChiralityValidator,
        CoordinateTransformer,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "PeptidePrepResult",
    "PeptidePrepRunner",
]


# Sentinel object returned by :meth:`PeptidePrepRunner._check_reused`
# when the on-disk artifact digests do not match the manifest's
# recorded digests (a provenance drift). ``run()`` distinguishes this
# from a clean reuse (``None``) by identity comparison and refuses to
# silently overwrite the broken artifacts.
_PROVENANCE_DRIFT = object()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


def _empty_float_dict() -> dict[str, float]:
    return {}


@dataclass(frozen=True)
class PeptidePrepResult:
    """Outcome of one peptide-prep invocation.

    Attributes:
        name: Logical run name (echoes ``config.name``).
        output_dir: Per-candidate output directory.
        success: ``True`` iff every step (threading, optional
            closure bond, minimization, chirality validation,
            GROMACS export) completed without errors.
        reused: ``True`` iff the result came from the idempotent
            cache path (existing manifest + matching digests).
        dry_run: ``True`` iff the runner was constructed /
            invoked with ``dry_run=True``.
        error: Populated with a clear human-readable error
            message on failure. ``""`` on success.
        prepared_pdb: Absolute path to the minimized PDB.
        prepared_pdb_sha256: Lowercase hex digest of the PDB.
        gromacs_top: Absolute path to the GROMACS ``.top``.
        gromacs_top_sha256: Lowercase hex digest of the ``.top``.
        gromacs_gro: Absolute path to the GROMACS ``.gro``.
        gromacs_gro_sha256: Lowercase hex digest of the ``.gro``.
        manifest_path: Absolute path to the manifest JSON.
        source_backbone_digest: SHA256 of the source PDB.
        source_config_digest: SHA256 of the canonical config.
        topology_bond_graph: Tuple of :class:`TopologyBondRecord`
            describing the closure bonds that were added.
        net_charge: Total nonbonded charge (elementary-charge
            units). The manifest records the OpenMM-computed
            value and the ParmEd-parsed value must match to
            ``1e-6``.
        chirality_reports_before: Tuple of :class:`ChiralityReport`
            from the pre-minimization validation pass.
        chirality_reports_post_hydrogenation: Tuple from the
            post-hydrogenation / pre-D-transform validation pass
            (blocker #5 — the canonical "HA survived the
            hydrogen-add step" check).
        chirality_reports_after: Tuple from the post-minimization
            validation pass.
        closure_distances_before: ``{bond_label: distance_A}`` —
            Å-scale distances of closure pairs BEFORE minimization.
        closure_distances_after: Å-scale distances AFTER
            minimization. None when the run was a cache hit.
        potential_energy_before_kjmol: Restrained-system energy
            before minimization.
        potential_energy_after_kjmol: Restrained-system energy
            after minimization (the restraint is still attached
            at this read; B2).
        no_nan: ``True`` iff every position + energy value the
            runner saw was finite (no NaN, no inf).
    """

    name: str
    output_dir: str
    success: bool = False
    reused: bool = False
    dry_run: bool = False
    error: str = ""

    prepared_pdb: str = ""
    prepared_pdb_sha256: str = ""
    gromacs_top: str = ""
    gromacs_top_sha256: str = ""
    gromacs_gro: str = ""
    gromacs_gro_sha256: str = ""
    manifest_path: str = ""

    source_backbone_digest: str = ""
    source_config_digest: str = ""

    topology_bond_graph: tuple[Any, ...] = ()
    net_charge: float = 0.0

    chirality_reports_before: tuple[Any, ...] = ()
    chirality_reports_post_hydrogenation: tuple[Any, ...] = ()
    chirality_reports_after: tuple[Any, ...] = ()

    closure_distances_before: dict[str, float] = field(default_factory=_empty_float_dict)
    closure_distances_after: dict[str, float] = field(default_factory=_empty_float_dict)

    potential_energy_before_kjmol: float = 0.0
    potential_energy_after_kjmol: object = 0.0

    no_nan: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "output_dir": self.output_dir,
            "success": self.success,
            "reused": self.reused,
            "dry_run": self.dry_run,
            "error": self.error,
            "prepared_pdb": self.prepared_pdb,
            "prepared_pdb_sha256": self.prepared_pdb_sha256,
            "gromacs_top": self.gromacs_top,
            "gromacs_top_sha256": self.gromacs_top_sha256,
            "gromacs_gro": self.gromacs_gro,
            "gromacs_gro_sha256": self.gromacs_gro_sha256,
            "manifest_path": self.manifest_path,
            "source_backbone_digest": self.source_backbone_digest,
            "source_config_digest": self.source_config_digest,
            "topology_bond_graph": [dataclasses.asdict(b) for b in self.topology_bond_graph],
            "net_charge": self.net_charge,
            "chirality_reports_before": [
                dataclasses.asdict(r) for r in self.chirality_reports_before
            ],
            "chirality_reports_post_hydrogenation": [
                dataclasses.asdict(r) for r in self.chirality_reports_post_hydrogenation
            ],
            "chirality_reports_after": [
                dataclasses.asdict(r) for r in self.chirality_reports_after
            ],
            "closure_distances_before": dict(self.closure_distances_before),
            "closure_distances_after": dict(self.closure_distances_after),
            "potential_energy_before_kjmol": self.potential_energy_before_kjmol,
            "potential_energy_after_kjmol": self.potential_energy_after_kjmol,
            "no_nan": self.no_nan,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class PeptidePrepRunner:
    """Run the peptide-prep pipeline end-to-end.

    The single public entry point is :meth:`run`. The runner is a
    thin orchestrator; the heavy lifting lives in
    :mod:`biolab_runners.peptide_prep.topology`,
    :mod:`biolab_runners.peptide_prep.chemistry`,
    :mod:`biolab_runners.peptide_prep.minimization`, and
    :mod:`biolab_runners.peptide_prep.export`.
    """

    def __init__(
        self,
        *,
        platform_name: str | None = None,
        sigterm_grace_seconds: float = 30.0,
    ) -> None:
        self._platform_override = platform_name
        self._sigterm_grace_seconds = sigterm_grace_seconds

    # ------------------------------------------------------------------ public

    def run(
        self,
        config: PeptidePrepConfig,
        *,
        coordinate_transformer: CoordinateTransformer | None = None,
        chirality_validator: ChiralityValidator | None = None,
    ) -> PeptidePrepResult:
        """Execute the pipeline and return the structured result.

        Callback requirement:

        * Linear all-L preparation: NO callbacks required.
        * Any D-substitution: BOTH ``coordinate_transformer`` and
          ``chirality_validator`` are required. The runner fails
          closed BEFORE writing any output if either is missing.

        Atomicity (blocker #10): callback / config validation runs
        BEFORE the on-disk quarantine (which is the destructive
        step on ``force=True``). A missing-callback failure must
        NOT leave the prior run's outputs in ``.stale/<UTC>/``
        while emitting a structured failure — the prior outputs
        must remain in place until a successful replacement run
        quarantines them. The order is:

        1. Dry-run short-circuit (no destructive steps).
        2. Reused-run short-circuit (no destructive steps).
        3. Callback / config validation (atomicity boundary).
        4. ``force=True`` quarantine.
        5. Full pipeline.
        """
        work_dir = Path(config.output_root) / config.name
        manifest_path = work_dir / "peptide_prep_manifest.json"

        if config.dry_run:
            return self._dry_run(config, work_dir, manifest_path)

        if not config.force:
            reused = self._check_reused(config, work_dir)
            if reused is _PROVENANCE_DRIFT:
                return self._fail(
                    config,
                    work_dir,
                    manifest_path,
                    error=(
                        "manifest's recorded digests do not match the "
                        "on-disk artifact digests; on-disk artifacts "
                        "have been modified or corrupted since the "
                        "last run. Re-run with force=True to quarantine "
                        "and rebuild, OR restore the artifacts from the "
                        "manifest digests"
                    ),
                )
            if reused is not None:
                return reused

        # Atomicity boundary (blocker #10): validate callbacks
        # BEFORE the on-disk quarantine. A missing / invalid
        # callback must NOT trigger the destructive quarantine
        # step. The callback check itself fails closed via the
        # same _fail() machinery as the rest of the pipeline;
        # structured failure, no partial outputs.
        needs_callbacks = bool(config.topology.d_substitutions)
        if needs_callbacks and coordinate_transformer is None:
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=(
                    "D-substitution requested but coordinate_transformer "
                    "is missing; peptide-prep refuses to write outputs "
                    "without a callable that applies the D-mirror transform"
                ),
            )
        if needs_callbacks and chirality_validator is None:
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=(
                    "D-substitution requested but chirality_validator "
                    "is missing; peptide-prep refuses to write outputs "
                    "without a callable that validates per-residue chirality"
                ),
            )

        return self._execute(
            config,
            work_dir,
            manifest_path,
            coordinate_transformer=coordinate_transformer,
            chirality_validator=chirality_validator,
        )

    # ------------------------------------------------------------------ private

    def _dry_run(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
    ) -> PeptidePrepResult:
        """Emit a dry-run result that still binds digests."""
        source_digest = file_sha256(Path(config.backbone_pdb))
        config_digest = _compute_config_digest(config)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "name": config.name,
            "output_dir": str(work_dir),
            "dry_run": True,
            "source_backbone_path": config.backbone_pdb,
            "source_backbone_sha256": source_digest,
            "config_digest": config_digest,
            "force": config.force,
        }
        manifest_save(work_dir, manifest)
        return PeptidePrepResult(
            name=config.name,
            output_dir=str(work_dir),
            success=True,
            dry_run=True,
            manifest_path=str(manifest_path),
            source_backbone_digest=source_digest or "",
            source_config_digest=config_digest,
        )

    def _check_reused(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
    ) -> PeptidePrepResult | None:
        """Return a reused :class:`PeptidePrepResult` if every digest matches."""
        manifest_path = work_dir / "peptide_prep_manifest.json"
        manifest = manifest_load(work_dir)
        if not manifest or "outputs" not in manifest:
            return None

        source_digest = file_sha256(Path(config.backbone_pdb)) or ""
        config_digest = _compute_config_digest(config)

        expected_source = manifest.get("source_backbone_sha256") or ""
        expected_config = manifest.get("config_digest") or ""
        if expected_source != source_digest or expected_config != config_digest:
            return None

        outputs = manifest["outputs"]
        prepared_pdb_sha = outputs.get("prepared_pdb_sha256") or ""
        gromacs_top_sha = outputs.get("gromacs_top_sha256") or ""
        gromacs_gro_sha = outputs.get("gromacs_gro_sha256") or ""

        try:
            disk_pdb_sha = file_sha256(Path(outputs["prepared_pdb"]))
            disk_top_sha = file_sha256(Path(outputs["gromacs_top"]))
            disk_gro_sha = file_sha256(Path(outputs["gromacs_gro"]))
        except KeyError:
            return None

        if (
            disk_pdb_sha != prepared_pdb_sha
            or disk_top_sha != gromacs_top_sha
            or disk_gro_sha != gromacs_gro_sha
        ):
            return _PROVENANCE_DRIFT  # type: ignore[return-value]

        return PeptidePrepResult(
            name=config.name,
            output_dir=str(work_dir),
            success=True,
            reused=True,
            prepared_pdb=outputs.get("prepared_pdb", ""),
            prepared_pdb_sha256=prepared_pdb_sha,
            gromacs_top=outputs.get("gromacs_top", ""),
            gromacs_top_sha256=gromacs_top_sha,
            gromacs_gro=outputs.get("gromacs_gro", ""),
            gromacs_gro_sha256=gromacs_gro_sha,
            manifest_path=str(manifest_path),
            source_backbone_digest=source_digest,
            source_config_digest=config_digest,
            net_charge=manifest.get("net_charge", 0.0),
            topology_bond_graph=tuple(manifest.get("topology_bond_graph", [])),
            chirality_reports_before=tuple(manifest.get("chirality_reports_before", [])),
            chirality_reports_post_hydrogenation=tuple(
                manifest.get("chirality_reports_post_hydrogenation", [])
            ),
            chirality_reports_after=tuple(manifest.get("chirality_reports_after", [])),
            closure_distances_before=dict(manifest.get("closure_distances_before", {})),
            closure_distances_after=dict(manifest.get("closure_distances_after", {})),
            potential_energy_before_kjmol=manifest.get("potential_energy_before_kjmol", 0.0),
            potential_energy_after_kjmol=manifest.get("potential_energy_after_kjmol", 0.0),
            no_nan=manifest.get("no_nan", True),
        )

    def _quarantine_stale(self, work_dir: Path) -> None:
        """Move prior artifacts to ``.stale/<UTC>/`` for forensic review."""
        import shutil

        if not work_dir.exists():
            return
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S_%f") + f"_{os.getpid()}"
        stale_dir = work_dir / ".stale" / ts
        stale_dir.mkdir(parents=True, exist_ok=False)

        from biolab_runners.peptide_prep.paths import PeptidePrepFiles

        for name in (
            PeptidePrepFiles.PREPARED_PDB,
            PeptidePrepFiles.PREPARED_TOP,
            PeptidePrepFiles.PREPARED_GRO,
            PeptidePrepFiles.MANIFEST,
        ):
            src = work_dir / name
            if src.exists():
                try:
                    shutil.move(str(src), str(stale_dir / name))
                    logger.info("Quarantined stale %s -> %s", src, stale_dir / name)
                except OSError as exc:
                    logger.warning("Quarantine failed for %s: %s", src, exc)

    def _execute(  # noqa: C901 — orchestrator branches over the union of prep paths
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        *,
        coordinate_transformer: CoordinateTransformer | None,
        chirality_validator: ChiralityValidator | None,
    ) -> PeptidePrepResult:
        """The full pipeline. Returns :class:`PeptidePrepResult` on every code path."""
        work_dir.mkdir(parents=True, exist_ok=True)
        source_digest = file_sha256(Path(config.backbone_pdb)) or ""
        config_digest = _compute_config_digest(config)

        if not source_digest:
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=f"backbone PDB missing or unreadable: {config.backbone_pdb}",
            )

        # Quarantine any prior-run outputs (force=True). The
        # callback / config validation already ran in run() so
        # this destructive step is atomicity-safe (blocker #10).
        if config.force:
            self._quarantine_stale(work_dir)

        # Stage 1 — build the prepared topology + restrained system
        # + closed system + initial energy.
        success, artifacts_or_failure = self._stage_build_topology(
            config, work_dir, manifest_path, source_digest, config_digest
        )
        if not success:
            return artifacts_or_failure  # type: ignore[return-value]
        artifacts = artifacts_or_failure

        # Stage 2 — chirality validation on the hydrogenated but
        # NOT-D-transformed structure (blocker #5). This catches
        # side-chain / HA orientation mistakes introduced by the
        # upstream hydrogenation step (a misoriented HA would be
        # invisible to the heavy-atom validator). When the
        # descriptor requests no D substitutions, the validator
        # is still invoked (every residue still has an "expected"
        # chirality of L and a corresponding L or D observation).
        post_h_chirality = self._stage_post_h_chirality(
            config,
            work_dir,
            manifest_path,
            source_digest,
            config_digest,
            artifacts,
            chirality_validator,
        )
        if isinstance(post_h_chirality, PeptidePrepResult):
            return post_h_chirality

        # Stage 3 — D-coordinate transform.
        failed = self._stage_d_transform(
            config,
            work_dir,
            manifest_path,
            source_digest,
            config_digest,
            artifacts,
            coordinate_transformer,
        )
        if failed is not None:
            return failed

        # Stage 4 — pre-minimization chirality validation (after
        # heavy-atom D construction — if any D substitutions were
        # requested).
        pre_chirality = self._stage_pre_chirality(
            config,
            work_dir,
            manifest_path,
            source_digest,
            config_digest,
            artifacts,
            chirality_validator,
        )
        if isinstance(pre_chirality, PeptidePrepResult):
            return pre_chirality

        # Stage 5 — restrained minimization on the LIVE system.
        minimized = self._stage_minimize(
            config,
            work_dir,
            manifest_path,
            source_digest,
            config_digest,
            artifacts,
            pre_chirality,
        )
        if isinstance(minimized, PeptidePrepResult):
            return minimized
        positions_after, energy_after, no_nan = minimized

        # Stage 6 — post-minimization chirality validation.
        post_chirality = self._stage_post_chirality(
            config,
            work_dir,
            manifest_path,
            source_digest,
            config_digest,
            artifacts,
            pre_chirality,
            positions_after,
            energy_after,
            chirality_validator,
        )
        if isinstance(post_chirality, PeptidePrepResult):
            return post_chirality

        # Stage 7 — closure-integrity check (H5).
        closure_distances_after = self._closure_distances(positions_after, artifacts.bond_graph)
        integrity_failure = self._check_closure_integrity(
            config,
            work_dir,
            manifest_path,
            source_digest,
            config_digest,
            artifacts,
            pre_chirality,
            post_chirality,
            energy_after,
            closure_distances_after,
        )
        if integrity_failure is not None:
            return integrity_failure

        # Stage 8 — chirality failure check.
        failure = self._check_chirality_failure(
            config,
            work_dir,
            manifest_path,
            source_digest,
            config_digest,
            artifacts,
            pre_chirality,
            post_h_chirality,
            post_chirality,
            energy_after,
        )
        if failure is not None:
            return failure

        # Stage 9 — write the prepared PDB + CONECT records.
        prepared_pdb = self._stage_write_pdb(
            config,
            work_dir,
            manifest_path,
            source_digest,
            config_digest,
            artifacts,
            pre_chirality,
            post_chirality,
            energy_after,
            positions_after,
        )
        if isinstance(prepared_pdb, PeptidePrepResult):
            return prepared_pdb
        prepared_pdb_path, prepared_pdb_sha = prepared_pdb

        # Stage 10 — export GROMACS .top/.gro via ParmEd.
        exported = self._stage_export_gromacs(
            config,
            work_dir,
            manifest_path,
            source_digest,
            config_digest,
            artifacts,
            pre_chirality,
            post_chirality,
            energy_after,
            positions_after,
        )
        if isinstance(exported, PeptidePrepResult):
            return exported
        top_path, top_sha, gro_path, gro_sha = exported

        # Stage 11 — verify export parity.
        parity = self._stage_verify_parity(
            config,
            work_dir,
            manifest_path,
            source_digest,
            config_digest,
            artifacts,
            pre_chirality,
            post_chirality,
            energy_after,
            top_path,
            gro_path,
            no_nan=no_nan,
            positions_after=positions_after,
        )
        if isinstance(parity, PeptidePrepResult):
            return parity

        # Stage 12 — write the manifest and return success.
        return self._stage_finalize(
            config=config,
            work_dir=work_dir,
            source_digest=source_digest,
            config_digest=config_digest,
            prepared_pdb_path=prepared_pdb_path,
            prepared_pdb_sha=prepared_pdb_sha,
            top_path=top_path,
            top_sha=top_sha,
            gro_path=gro_path,
            gro_sha=gro_sha,
            artifacts=artifacts,
            pre_chirality=pre_chirality,
            post_h_chirality=post_h_chirality,
            post_chirality=post_chirality,
            closure_distances_after=closure_distances_after,
            energy_after=energy_after,
            no_nan=no_nan,
        )

    # ------------------------------------------------------------------ stage helpers

    def _stage_build_topology(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        source_digest: str,
        config_digest: str,
    ) -> tuple[bool, object | PeptidePrepResult]:
        """Stage 1 — build the prepared topology + restrained system."""
        from biolab_runners.peptide_prep import topology

        try:
            artifacts = topology.build_modeller(config)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            return False, self._fail(
                config,
                work_dir,
                manifest_path,
                error=f"topology preparation failed: {exc}",
                source_digest=source_digest,
                config_digest=config_digest,
            )
        return True, artifacts

    def _stage_d_transform(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        source_digest: str,
        config_digest: str,
        artifacts: object,
        coordinate_transformer: CoordinateTransformer | None,
    ) -> PeptidePrepResult | None:
        """Stage 2 — D-coordinate transform (if requested)."""
        if not config.topology.d_substitutions:
            return None
        try:
            self._apply_d_transform(config, artifacts, coordinate_transformer)  # type: ignore[arg-type]
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=f"D-coordinate transform failed: {exc}",
                source_digest=source_digest,
                config_digest=config_digest,
                bond_graph=artifacts.bond_graph,
                net_charge=artifacts.net_charge,
                potential_energy_before_kjmol=artifacts.energy_before_kjmol,
            )
        return None

    def _stage_post_h_chirality(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        source_digest: str,
        config_digest: str,
        artifacts: object,
        chirality_validator: ChiralityValidator | None,
    ) -> tuple[Any, ...] | PeptidePrepResult:
        """Stage 2 — chirality validation on the post-hydrogenation structure.

        Called AFTER ``apply_sequence_mutation`` (which adds
        hydrogens) and AFTER ``build_modeller`` (which applies
        cap removal / closure bond edits). This is the
        canonical "side-chain + HA orientation survived the
        hydrogen-add step" check (blocker #5). The runner
        surfaces this report in the manifest so downstream
        operators can audit whether HA reflection (when a D
        substitution is requested) actually flipped HA into a
        D-consistent orientation.
        """
        if chirality_validator is None:
            return ()
        try:
            return self._run_chirality_validation(
                artifacts.topology,
                artifacts.positions,
                config.sequence,
                config.topology,
                chirality_validator,
                stage="post_h",
            )
        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
            # ``TypeError`` is caught because the runner forwards an
            # explicit ``stage=`` kwarg (see ``_run_chirality_validation``);
            # a strict validator that rejects unknown kwargs would
            # raise ``TypeError``. The fail-closed contract converts
            # every callback exception into a structured failure so
            # the orchestrator can branch on ``result.success``.
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=f"post-hydrogenation chirality validation failed: {exc}",
                source_digest=source_digest,
                config_digest=config_digest,
                bond_graph=artifacts.bond_graph,
                net_charge=artifacts.net_charge,
                potential_energy_before_kjmol=artifacts.energy_before_kjmol,
            )

    def _stage_pre_chirality(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        source_digest: str,
        config_digest: str,
        artifacts: object,
        chirality_validator: ChiralityValidator | None,
    ) -> tuple[Any, ...] | PeptidePrepResult:
        """Stage 4 — pre-minimization chirality validation."""
        pre_chirality: tuple[Any, ...] = ()
        if chirality_validator is None:
            return pre_chirality
        try:
            return self._run_chirality_validation(
                artifacts.topology,
                artifacts.positions,
                config.sequence,
                config.topology,
                chirality_validator,
                stage="pre",
            )
        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=f"pre-minimization chirality validation failed: {exc}",
                source_digest=source_digest,
                config_digest=config_digest,
                bond_graph=artifacts.bond_graph,
                net_charge=artifacts.net_charge,
                potential_energy_before_kjmol=artifacts.energy_before_kjmol,
            )

    def _stage_minimize(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        source_digest: str,
        config_digest: str,
        artifacts: object,
        pre_chirality: tuple[Any, ...],
    ) -> tuple[Any, object, bool] | PeptidePrepResult:
        """Stage 4 — restrained minimization on the LIVE system (B2)."""
        from biolab_runners.peptide_prep import minimization

        try:
            return minimization.run_minimization(
                artifacts.topology,
                artifacts.system,
                artifacts.positions,
                platform_name=self._platform_override or config.openmm_platform,
                max_iterations=config.minimization_max_iterations,
                tolerance_kjmol_nm=config.minimization_tolerance_kjmol_nm,
            )
        except RuntimeError as exc:
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=f"minimization failed: {exc}",
                source_digest=source_digest,
                config_digest=config_digest,
                bond_graph=artifacts.bond_graph,
                net_charge=artifacts.net_charge,
                chirality_reports_before=pre_chirality,
                potential_energy_before_kjmol=artifacts.energy_before_kjmol,
            )

    def _stage_post_chirality(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        source_digest: str,
        config_digest: str,
        artifacts: object,
        pre_chirality: tuple[Any, ...],
        positions_after: object,
        energy_after: object,
        chirality_validator: ChiralityValidator | None,
    ) -> tuple[Any, ...] | PeptidePrepResult:
        """Stage 5 — post-minimization chirality validation."""
        post_chirality: tuple[Any, ...] = ()
        if chirality_validator is None:
            return post_chirality
        try:
            return self._run_chirality_validation(
                artifacts.topology,
                positions_after,
                config.sequence,
                config.topology,
                chirality_validator,
                stage="post",
            )
        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=f"post-minimization chirality validation failed: {exc}",
                source_digest=source_digest,
                config_digest=config_digest,
                bond_graph=artifacts.bond_graph,
                net_charge=artifacts.net_charge,
                chirality_reports_before=pre_chirality,
                potential_energy_before_kjmol=artifacts.energy_before_kjmol,
                potential_energy_after_kjmol=energy_after,
            )

    def _stage_write_pdb(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        source_digest: str,
        config_digest: str,
        artifacts: object,
        pre_chirality: tuple[Any, ...],
        post_chirality: tuple[Any, ...],
        energy_after: object,
        positions_after: object,
    ) -> tuple[Path, str] | PeptidePrepResult:
        """Stage 8 — write the prepared PDB + CONECT records."""
        try:
            return self._write_prepared_pdb(
                work_dir,
                artifacts.topology,
                positions_after,
                closure_bond_records=artifacts.bond_graph,
            )
        except (OSError, RuntimeError) as exc:
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=f"prepared.pdb write failed: {exc}",
                source_digest=source_digest,
                config_digest=config_digest,
                bond_graph=artifacts.bond_graph,
                net_charge=artifacts.net_charge,
                chirality_reports_before=pre_chirality,
                chirality_reports_after=post_chirality,
                potential_energy_before_kjmol=artifacts.energy_before_kjmol,
                potential_energy_after_kjmol=energy_after,
            )

    def _stage_export_gromacs(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        source_digest: str,
        config_digest: str,
        artifacts: object,
        pre_chirality: tuple[Any, ...],
        post_chirality: tuple[Any, ...],
        energy_after: object,
        positions_after: object,
    ) -> tuple[Path, str, Path, str] | PeptidePrepResult:
        """Stage 9 — export GROMACS .top/.gro via ParmEd."""
        from biolab_runners.peptide_prep import export

        try:
            gromacs_artifacts = export.export_gromacs(
                artifacts.topology,
                artifacts.closed_system,
                positions_after,
                top_path=work_dir / "prepared.top",
                gro_path=work_dir / "prepared.gro",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=f"GROMACS export failed: {exc}",
                source_digest=source_digest,
                config_digest=config_digest,
                bond_graph=artifacts.bond_graph,
                net_charge=artifacts.net_charge,
                chirality_reports_before=pre_chirality,
                chirality_reports_after=post_chirality,
                potential_energy_before_kjmol=artifacts.energy_before_kjmol,
                potential_energy_after_kjmol=energy_after,
            )
        top_path = Path(gromacs_artifacts["top_path"])
        gro_path = Path(gromacs_artifacts["gro_path"])
        return (
            top_path,
            file_sha256(top_path) or "",
            gro_path,
            file_sha256(gro_path) or "",
        )

    def _stage_verify_parity(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        source_digest: str,
        config_digest: str,
        artifacts: object,
        pre_chirality: tuple[Any, ...],
        post_chirality: tuple[Any, ...],
        energy_after: object,
        top_path: Path,
        gro_path: Path,
        no_nan: bool,
        positions_after: object,
    ) -> tuple[bool, str] | PeptidePrepResult:
        """Stage 10 — independent parity check (M1 + blocker #4)."""
        from biolab_runners.peptide_prep import export

        try:
            parity_ok, parity_msg = export.verify_export_parity(
                artifacts.topology,
                artifacts.closed_system,
                positions_after,
                top_path=top_path,
                gro_path=gro_path,
                no_nan=no_nan,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=f"GROMACS parity check raised: {exc}",
                source_digest=source_digest,
                config_digest=config_digest,
                bond_graph=artifacts.bond_graph,
                net_charge=artifacts.net_charge,
                chirality_reports_before=pre_chirality,
                chirality_reports_after=post_chirality,
                potential_energy_before_kjmol=artifacts.energy_before_kjmol,
                potential_energy_after_kjmol=energy_after,
            )
        if not parity_ok:
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=f"GROMACS parity check failed: {parity_msg}",
                source_digest=source_digest,
                config_digest=config_digest,
                bond_graph=artifacts.bond_graph,
                net_charge=artifacts.net_charge,
                chirality_reports_before=pre_chirality,
                chirality_reports_after=post_chirality,
                potential_energy_before_kjmol=artifacts.energy_before_kjmol,
                potential_energy_after_kjmol=energy_after,
            )

        # Optional gmx grompp -pp round-trip (skipped when gmx
        # is not on PATH — see export.gmx_grompp_pp_check).
        grompp_ok, grompp_msg = export.gmx_grompp_pp_check(
            top_path,
            gro_path,
            audit_workdir=work_dir / ".grompp_audit",
        )
        if not grompp_ok:
            return self._fail(
                config,
                work_dir,
                manifest_path,
                error=f"gmx grompp round-trip failed: {grompp_msg}",
                source_digest=source_digest,
                config_digest=config_digest,
                bond_graph=artifacts.bond_graph,
                net_charge=artifacts.net_charge,
                chirality_reports_before=pre_chirality,
                chirality_reports_after=post_chirality,
                potential_energy_before_kjmol=artifacts.energy_before_kjmol,
                potential_energy_after_kjmol=energy_after,
            )
        return True, ""

    def _stage_finalize(
        self,
        *,
        config: PeptidePrepConfig,
        work_dir: Path,
        source_digest: str,
        config_digest: str,
        prepared_pdb_path: Path,
        prepared_pdb_sha: str,
        top_path: Path,
        top_sha: str,
        gro_path: Path,
        gro_sha: str,
        artifacts: object,
        pre_chirality: tuple[Any, ...],
        post_h_chirality: tuple[Any, ...],
        post_chirality: tuple[Any, ...],
        closure_distances_after: dict[str, float],
        energy_after: object,
        no_nan: bool,
    ) -> PeptidePrepResult:
        """Stage 11 — write the manifest and return the success result."""
        manifest_path = work_dir / "peptide_prep_manifest.json"
        manifest = self._build_manifest(
            config=config,
            work_dir=work_dir,
            source_digest=source_digest,
            config_digest=config_digest,
            prepared_pdb_path=prepared_pdb_path,
            prepared_pdb_sha=prepared_pdb_sha,
            top_path=top_path,
            top_sha=top_sha,
            gro_path=gro_path,
            gro_sha=gro_sha,
            bond_graph=artifacts.bond_graph,
            net_charge=artifacts.net_charge,
            chirality_reports_before=pre_chirality,
            chirality_reports_post_hydrogenation=post_h_chirality,
            chirality_reports_after=post_chirality,
            closure_distances_before=artifacts.closure_distances_before,
            closure_distances_after=closure_distances_after,
            potential_energy_before_kjmol=artifacts.energy_before_kjmol,
            potential_energy_after_kjmol=energy_after,
            no_nan=no_nan,
        )
        manifest_save(work_dir, manifest)
        return PeptidePrepResult(
            name=config.name,
            output_dir=str(work_dir),
            success=True,
            prepared_pdb=str(prepared_pdb_path),
            prepared_pdb_sha256=prepared_pdb_sha,
            gromacs_top=str(top_path),
            gromacs_top_sha256=top_sha,
            gromacs_gro=str(gro_path),
            gromacs_gro_sha256=gro_sha,
            manifest_path=str(manifest_path),
            source_backbone_digest=source_digest,
            source_config_digest=config_digest,
            topology_bond_graph=tuple(artifacts.bond_graph),
            net_charge=artifacts.net_charge,
            chirality_reports_before=pre_chirality,
            chirality_reports_post_hydrogenation=post_h_chirality,
            chirality_reports_after=post_chirality,
            closure_distances_before=artifacts.closure_distances_before,
            closure_distances_after=closure_distances_after,
            potential_energy_before_kjmol=artifacts.energy_before_kjmol,
            potential_energy_after_kjmol=energy_after,
            no_nan=no_nan,
        )

    # ------------------------------------------------------------------ helpers

    def _apply_d_transform(
        self,
        config: PeptidePrepConfig,
        artifacts: object,
        transformer: CoordinateTransformer,
    ) -> object:
        """Apply the D-residue mirror transform to configured positions.

        The transformer receives a name→coordinate mapping per
        residue and returns the transformed mapping. The runner
        accepts EITHER a bare dict OR a
        :class:`CoordinateTransformResult` wrapper
        (the wrapper is unwrapped via
        :func:`biolab_runners.peptide_prep.protocols.extract_coordinate_mapping`).

        The transformer may RAISE any exception (mirroring a bug
        in the upstream bioml-tools coordinate math); the runner
        catches it and returns a structured failure so a callback
        exception never escapes the orchestrator.
        """
        from biolab_runners.peptide_prep.protocols import extract_coordinate_mapping

        positions = artifacts.positions
        atoms = list(artifacts.topology.atoms())

        nm_positions: list[tuple[float, float, float]] = []
        for pos in positions:
            nm_positions.append(
                (
                    _to_nm(pos[0]),
                    _to_nm(pos[1]),
                    _to_nm(pos[2]),
                )
            )

        for d in config.topology.d_substitutions:
            pos_idx_raw = getattr(d, "position", None)
            pos_idx = (pos_idx_raw - 1) if pos_idx_raw is not None else 0
            residue_aa = getattr(d, "residue", "ALA")
            mapping = collect_atom_mapping(artifacts.topology, positions, pos_idx)
            raw = transformer(mapping, residue_aa, pos_idx)
            transformed = extract_coordinate_mapping(raw)
            for atom in atoms:
                if atom.residue.index != pos_idx:
                    continue
                atom_name = atom.name
                if atom_name not in transformed:
                    continue
                tx, ty, tz = transformed[atom_name]
                nm_positions[atom.index] = (
                    tx / 10.0,
                    ty / 10.0,
                    tz / 10.0,
                )

        artifacts.positions = _build_positions(artifacts.topology, nm_positions)
        return artifacts

    def _run_chirality_validation(
        self,
        topology: object,
        positions: object,
        sequence: str,
        topology_descriptor: object,
        validator: ChiralityValidator,
        *,
        stage: str,
    ) -> tuple[ChiralityReport, ...]:
        """Run the chirality validator over every non-Gly residue.

        Args:
            topology: OpenMM ``app.Topology`` whose residues are
                validated in chain order.
            positions: Iterable of OpenMM ``Vec3`` (length
                ``topology.getNumAtoms()``) giving the
                coordinates to validate.
            sequence: 1-letter sequence matching the topology
                residues, used to skip Glycine (which is achiral
                and excluded from CHEM-001 validation).
            topology_descriptor: :class:`PeptideTopologyDescriptor`
                whose ``d_substitutions`` declare which residues
                are D. The annotation is only applied at
                post-transform stages — see ``stage``.
            validator: :class:`ChiralityValidator` callable that
                decides ``L``/``D`` for one residue at a time.
            stage: One of ``"post_h"`` (post-hydrogenation,
                BEFORE the D-coordinate transform), ``"pre"``
                (post-D-transform, pre-minimization), or
                ``"post"`` (post-minimization). The stage is the
                explicit seam that decides when the descriptor's D
                annotations apply — the post-hydrogenation stage
                audits the side-chain orientation that the
                hydrogen-add step produced, and at that point a
                designated D residue is still in its pre-transform
                L geometry. Inferring the stage from call order
                breaks the contract silently when the stage order
                changes; therefore the seam is explicit.

        The validator is invoked per-residue (CHEM-001 requires
        per-residue validation, not bulk validation). Any
        exception raised by the validator (a bug in the upstream
        bioml-tools math) is caught by the runner's caller and
        surfaced as a structured failure — never re-raised.
        ``stage`` is forwarded to the validator as a
        ``**kwargs`` audit hint so recording validators can
        attribute calls without inferring from call order.
        """
        # The post-hydrogenation stage runs BEFORE the D
        # transform. Apply the descriptor's D annotations ONLY
        # at the post-transform stages so the validator's
        # ``expected`` value matches the geometry it actually
        # sees.
        apply_d_annotations = stage != "post_h"
        reports: list[ChiralityReport] = []
        for index, aa in enumerate(sequence):
            if aa == "G":
                continue
            mapping = collect_atom_mapping(topology, positions, index)
            is_d_position = apply_d_annotations and any(
                getattr(d, "position", -1) == index + 1 for d in topology_descriptor.d_substitutions
            )
            expected = "D" if is_d_position else "L"
            report = validator(
                mapping,
                three_letter_for(index, aa),
                index,
                expected=expected,
                stage=stage,
            )
            reports.append(report)
        return tuple(reports)

    def _closure_distances(
        self,
        positions: object,
        bond_graph: list[Any],
    ) -> dict[str, float]:
        """Å-scale distances of every closure bond AFTER minimization."""
        from math import sqrt

        distances: dict[str, float] = {}
        for rec in bond_graph:
            pi = positions[rec.atom1_index]  # type: ignore[index]
            pj = positions[rec.atom2_index]  # type: ignore[index]
            dx = _to_nm(pi[0]) - _to_nm(pj[0])
            dy = _to_nm(pi[1]) - _to_nm(pj[1])
            dz = _to_nm(pi[2]) - _to_nm(pj[2])
            d_nm = sqrt(dx * dx + dy * dy + dz * dz)
            distances[f"{rec.bond_type}_{rec.residue1_index}_{rec.residue2_index}"] = d_nm * 10.0
        return distances

    def _check_closure_integrity(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        source_digest: str,
        config_digest: str,
        artifacts: object,
        pre_chirality: tuple[Any, ...],
        post_chirality: tuple[Any, ...],
        energy_after: object,
        closure_distances_after: dict[str, float],
    ) -> PeptidePrepResult | None:
        """H5 — fail closed if any post-minimization closure is not covalent.

        The disulfide S-S equilibrium is ~2.05 Å; anything > the
        configured ``max_disulfide_distance_angstrom`` (default
        2.5 Å) means the two SG atoms are NOT covalently bound and
        the runner refuses to write outputs. Same for head-to-tail
        C-N (equilibrium ~1.33 Å; default limit 2.0 Å).

        Returns a :class:`PeptidePrepResult` failure on bad
        distances, ``None`` otherwise.
        """
        for label, distance in closure_distances_after.items():
            bond_type = label.split("_")[0]
            if bond_type == "disulfide" and distance > config.max_disulfide_distance_angstrom:
                return self._fail(
                    config,
                    work_dir,
                    manifest_path,
                    error=(
                        f"closure-integrity: disulfide {label} is {distance:.3f} Å "
                        f"after minimization; exceeds max_disulfide_distance_angstrom "
                        f"({config.max_disulfide_distance_angstrom:.3f} Å). The S-S "
                        f"bond is not covalent — refusing to write outputs."
                    ),
                    source_digest=source_digest,
                    config_digest=config_digest,
                    bond_graph=artifacts.bond_graph,
                    net_charge=artifacts.net_charge,
                    chirality_reports_before=pre_chirality,
                    chirality_reports_after=post_chirality,
                    closure_distances_before=artifacts.closure_distances_before,
                    closure_distances_after=closure_distances_after,
                    potential_energy_before_kjmol=artifacts.energy_before_kjmol,
                    potential_energy_after_kjmol=energy_after,
                )
            if bond_type == "head_to_tail" and distance > config.max_head_to_tail_distance_angstrom:
                return self._fail(
                    config,
                    work_dir,
                    manifest_path,
                    error=(
                        f"closure-integrity: head_to_tail {label} is {distance:.3f} Å "
                        f"after minimization; exceeds max_head_to_tail_distance_angstrom "
                        f"({config.max_head_to_tail_distance_angstrom:.3f} Å). The C-N "
                        f"closure is not covalent — refusing to write outputs."
                    ),
                    source_digest=source_digest,
                    config_digest=config_digest,
                    bond_graph=artifacts.bond_graph,
                    net_charge=artifacts.net_charge,
                    chirality_reports_before=pre_chirality,
                    chirality_reports_after=post_chirality,
                    closure_distances_before=artifacts.closure_distances_before,
                    closure_distances_after=closure_distances_after,
                    potential_energy_before_kjmol=artifacts.energy_before_kjmol,
                    potential_energy_after_kjmol=energy_after,
                )
        return None

    def _check_chirality_failure(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        source_digest: str,
        config_digest: str,
        artifacts: object,
        pre_chirality: tuple[Any, ...],
        post_h_chirality: tuple[Any, ...],
        post_chirality: tuple[Any, ...],
        energy_after: object,
    ) -> PeptidePrepResult | None:
        """Return a failure when any chirality report is invalid."""
        bad_pre = [r for r in pre_chirality if not r.valid]
        bad_post_h = [r for r in post_h_chirality if not r.valid]
        bad_post = [r for r in post_chirality if not r.valid]
        if not (bad_pre or bad_post_h or bad_post):
            return None
        return self._fail(
            config,
            work_dir,
            manifest_path,
            error=(
                f"chirality validation failed: "
                f"{len(bad_pre)} pre-min reports, "
                f"{len(bad_post_h)} post-hydrogenation reports, "
                f"{len(bad_post)} post-min reports invalid"
            ),
            source_digest=source_digest,
            config_digest=config_digest,
            bond_graph=artifacts.bond_graph,
            net_charge=artifacts.net_charge,
            chirality_reports_before=pre_chirality,
            chirality_reports_post_hydrogenation=post_h_chirality,
            chirality_reports_after=post_chirality,
            potential_energy_before_kjmol=artifacts.energy_before_kjmol,
            potential_energy_after_kjmol=energy_after,
        )

    def _write_prepared_pdb(
        self,
        work_dir: Path,
        topology: object,
        positions: object,
        *,
        closure_bond_records: tuple[Any, ...] = (),
    ) -> tuple[Path, str]:
        """Write ``prepared.pdb`` via OpenMM ``PDBFile.writeFile`` + CONECT."""
        from biolab_runners.peptide_prep.export import write_prepared_pdb
        from biolab_runners.peptide_prep.paths import PeptidePrepFiles

        path = work_dir / PeptidePrepFiles.PREPARED_PDB
        write_prepared_pdb(
            path,
            topology,
            positions,
            closure_bond_records=tuple(closure_bond_records),
        )
        sha = file_sha256(path) or ""
        return path, sha

    def _build_manifest(
        self,
        *,
        config: PeptidePrepConfig,
        work_dir: Path,
        source_digest: str,
        config_digest: str,
        prepared_pdb_path: Path,
        prepared_pdb_sha: str,
        top_path: Path,
        top_sha: str,
        gro_path: Path,
        gro_sha: str,
        bond_graph: list[Any],
        net_charge: float,
        chirality_reports_before: tuple[Any, ...],
        chirality_reports_post_hydrogenation: tuple[Any, ...],
        chirality_reports_after: tuple[Any, ...],
        closure_distances_before: dict[str, float],
        closure_distances_after: dict[str, float],
        potential_energy_before_kjmol: object,
        potential_energy_after_kjmol: object,
        no_nan: bool,
    ) -> dict[str, Any]:
        """Build the manifest payload."""
        return {
            "schema_version": 1,
            "name": config.name,
            "output_dir": str(work_dir),
            "dry_run": False,
            "force": config.force,
            "source_backbone_path": config.backbone_pdb,
            "source_backbone_sha256": source_digest,
            "config_digest": config_digest,
            "outputs": {
                "prepared_pdb": str(prepared_pdb_path),
                "prepared_pdb_sha256": prepared_pdb_sha,
                "gromacs_top": str(top_path),
                "gromacs_top_sha256": top_sha,
                "gromacs_gro": str(gro_path),
                "gromacs_gro_sha256": gro_sha,
            },
            "topology_bond_graph": [dataclasses.asdict(b) for b in bond_graph],
            "net_charge": net_charge,
            "chirality_reports_before": [dataclasses.asdict(r) for r in chirality_reports_before],
            "chirality_reports_post_hydrogenation": [
                dataclasses.asdict(r) for r in chirality_reports_post_hydrogenation
            ],
            "chirality_reports_after": [dataclasses.asdict(r) for r in chirality_reports_after],
            "closure_distances_before": closure_distances_before,
            "closure_distances_after": closure_distances_after,
            "potential_energy_before_kjmol": potential_energy_before_kjmol,
            "potential_energy_after_kjmol": potential_energy_after_kjmol,
            "no_nan": no_nan,
            "openmm_platform": config.openmm_platform,
            "minimization_max_iterations": config.minimization_max_iterations,
            "restraint_force_k_kjmol_nm2": config.restraint_force_k_kjmol_nm2,
        }

    def _fail(
        self,
        config: PeptidePrepConfig,
        work_dir: Path,
        manifest_path: Path,
        *,
        error: str,
        source_digest: str = "",
        config_digest: str = "",
        bond_graph: list[Any] | None = None,
        net_charge: float = 0.0,
        chirality_reports_before: tuple[Any, ...] = (),
        chirality_reports_post_hydrogenation: tuple[Any, ...] = (),
        chirality_reports_after: tuple[Any, ...] = (),
        closure_distances_before: dict[str, float] | None = None,
        closure_distances_after: dict[str, float] | None = None,
        potential_energy_before_kjmol: object = 0.0,
        potential_energy_after_kjmol: object = 0.0,
    ) -> PeptidePrepResult:
        """Build a failure :class:`PeptidePrepResult` and persist a minimal manifest."""
        if source_digest and not config_digest:
            config_digest = _compute_config_digest(config)
        closure_distances_before = closure_distances_before or {}
        closure_distances_after = closure_distances_after or {}
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "name": config.name,
            "output_dir": str(work_dir),
            "dry_run": False,
            "force": config.force,
            "error": error,
            "source_backbone_sha256": source_digest,
            "config_digest": config_digest,
            "topology_bond_graph": [dataclasses.asdict(b) for b in (bond_graph or [])],
            "net_charge": net_charge,
            "chirality_reports_before": [dataclasses.asdict(r) for r in chirality_reports_before],
            "chirality_reports_post_hydrogenation": [
                dataclasses.asdict(r) for r in chirality_reports_post_hydrogenation
            ],
            "chirality_reports_after": [dataclasses.asdict(r) for r in chirality_reports_after],
            "closure_distances_before": dict(closure_distances_before),
            "closure_distances_after": dict(closure_distances_after),
            "potential_energy_before_kjmol": potential_energy_before_kjmol,
            "potential_energy_after_kjmol": potential_energy_after_kjmol,
        }
        manifest_save(work_dir, manifest)
        return PeptidePrepResult(
            name=config.name,
            output_dir=str(work_dir),
            success=False,
            error=error,
            manifest_path=str(manifest_path),
            source_backbone_digest=source_digest,
            source_config_digest=config_digest,
            topology_bond_graph=tuple(bond_graph or []),
            net_charge=net_charge,
            chirality_reports_before=chirality_reports_before,
            chirality_reports_post_hydrogenation=chirality_reports_post_hydrogenation,
            chirality_reports_after=chirality_reports_after,
            closure_distances_before=closure_distances_before,
            closure_distances_after=closure_distances_after,
            potential_energy_before_kjmol=float(potential_energy_before_kjmol),  # type: ignore[arg-type]
            potential_energy_after_kjmol=float(potential_energy_after_kjmol),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


_THREE_LETTER = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def three_letter_for(index: int, one_letter: str) -> str:  # noqa: ARG001
    """Return the 3-letter residue code for a 1-letter ``one_letter``."""
    return _THREE_LETTER[one_letter]


def label_for_bond(rec: TopologyBondRecord) -> str:
    """Stable string label for a bond in the manifest."""
    return f"{rec.bond_type}_{rec.residue1_index}_{rec.residue2_index}"


def _compute_config_digest(config: PeptidePrepConfig) -> str:
    """SHA256 of the canonical config (matches provenance convention)."""
    import hashlib

    from biolab_runners.provenance import _canonical_json

    payload = dataclasses.asdict(config)
    topo = payload.get("topology", {})
    if isinstance(topo, dict):
        head_to_tail_val = topo.get("head_to_tail")
        payload["topology"] = {
            "d_substitutions": [_attr_dict(d) for d in topo.get("d_substitutions", ())],
            "head_to_tail": (
                _attr_dict(head_to_tail_val) if head_to_tail_val is not None else None
            ),
            "disulfides": [_attr_dict(b) for b in topo.get("disulfides", ())],
        }
    encoded = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _attr_dict(obj: object) -> dict[str, Any] | None:
    """Convert a loose typed object to a JSON-native dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)  # type: ignore[arg-type]
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    raise TypeError(
        f"topology descriptor entry of type {type(obj).__name__!r} is not "
        f"dataclass / dict / __dict__-bearing; serialise to a dict before "
        f"passing it to PeptidePrepConfig.topology"
    )


def _build_positions(topology: object, xyz_nm: list[tuple[float, float, float]]) -> object:  # noqa: ARG001
    """Construct an OpenMM positions vector from a flat list of nm coordinates."""
    import openmm.unit as unit

    import openmm

    out = []
    for x, y, z in xyz_nm:
        out.append(openmm.Vec3(x, y, z) * unit.nanometer)
    return out


def _to_nm(component: object) -> float:
    """Convert an OpenMM Vec3 component to plain nm."""
    import openmm.unit as unit

    if hasattr(component, "value_in_unit"):
        return float(component.value_in_unit(unit.nanometer))  # type: ignore[arg-type]
    return float(component)  # type: ignore[arg-type]
