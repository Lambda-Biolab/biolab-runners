"""Tests for ``OpenMMConfig.from_md_spec`` (slice 12 / MD-OPENMM-001).

Covers:

* field-by-field projection of an :class:`MDSpec` onto an
  :class:`OpenMMConfig` (every engine-neutral field lands on the
  matching dataclass field);
* engine-specific overlays (``openmm_platform``, ``extra_forcefields``,
  ``water_ff_xml``, ``target_irmsd_threshold_a``) keep their defaults
  unless the caller overrides via kwargs;
* typos in engine_overrides raise ``TypeError`` so silent fall-throughs
  (e.g. ``production_NS`` vs ``production_ns``) can't pollute the config;
* equivalence with an OpenMMConfig built from the same fields manually
  (legacy ``__init__`` path) — both paths must produce the same wire
  output for the engine-neutral subset;
* round-trip via :class:`MDSpec` ``save`` → ``load`` →
  ``OpenMMConfig.from_md_spec`` works (proves the canonical JSON wire
  format is self-sufficient for downstream consumers).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from biolab_runners.openmm.config import (
    DEFAULT_IRMSD_THRESHOLD_A,
    OPENMM_PLATFORM,
    OpenMMConfig,
)
from bioml_tools.md.system_spec import (
    ACTIVIN_E_PRODUCTION_PROFILE,
    MDSpec,
    TopologyMetadata,
)

# ---------------------------------------------------------------------------
# 1. Field-by-field projection
# ---------------------------------------------------------------------------


class TestFromMdSpecProjection:
    """``from_md_spec`` projects every engine-neutral MDSpec field onto
    the corresponding OpenMMConfig dataclass field.

    Engine-neutral means: any field that an OpenMM cross-port (GROMACS,
    NAMD) would also respect. Engine-specific overlays (platform,
    extra FF XMLs, target_irmsd_threshold_a, water_ff_xml) keep their
    defaults and are tested separately under TestEngineOverlay.
    """

    def _build_spec(self) -> MDSpec:
        """Spec built from the canonical Activin-E production profile.

        ``from_profile`` only takes per-instance data; simulation
        parameters come from the profile itself. To override those,
        construct :class:`MDSpec` directly (see the equivalent
        round-trip tests below for one such path).
        """
        return MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="receptor.pdb",
            peptide_pdb="peptide.pdb",
            output_dir="results/md/FadA/pep_001",
            target="FadA",
            peptide_id="pep_001",
        )

    def test_paths_and_identifiers_propagate(self) -> None:
        spec = self._build_spec()
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.receptor_pdb == "receptor.pdb"
        assert cfg.peptide_pdb == "peptide.pdb"
        assert cfg.output_dir == "results/md/FadA/pep_001"
        assert cfg.target == "FadA"
        assert cfg.peptide_id == "pep_001"

    def test_protein_ff_and_water_model_propagate(self) -> None:
        spec = self._build_spec()
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.protein_ff == "Amber99SB-ILDN"
        assert cfg.water_model == "TIP3P"

    def test_ionic_strength_maps_to_nacl_mol(self) -> None:
        """MDSpec.ionic_strength_m → OpenMMConfig.nacl_mol (NaCl-only).

        See MDSpec docstring: ``ionic_strength_m`` is the only ionic
        concentration the runner currently models; multi-ion extensions
        live in topology_metadata.
        """
        spec = self._build_spec()  # default 0.15 M from ACTIVIN_E_PROFILE
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.nacl_mol == pytest.approx(0.150)

    def test_box_settings_propagate(self) -> None:
        spec = self._build_spec()  # cubic, 1.0 nm
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.box_shape == "cubic"
        assert cfg.box_padding_nm == pytest.approx(1.0)

    def test_simulation_parameters_propagate(self) -> None:
        spec = self._build_spec()
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.temperature_k == pytest.approx(310.0)
        assert cfg.pressure_atm == pytest.approx(1.0)
        assert cfg.timestep_fs == pytest.approx(2.0)

    def test_production_and_cadence_propagate(self) -> None:
        spec = self._build_spec()
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.production_ns == pytest.approx(200.0)
        assert cfg.total_steps == 100_000_000
        assert cfg.save_interval_ps == pytest.approx(10.0)
        assert cfg.checkpoint_interval_hours == pytest.approx(2.0)

    def test_protonation_ph_propagates(self) -> None:
        spec = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
            protonation_ph=5.5,
        )
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.protonation_ph == pytest.approx(5.5)

    def test_topology_metadata_is_ignored_by_engine(self) -> None:
        """Topology metadata (cycle / disulfide / D-residues) lives on the
        MDSpec but is the design-layer concern; the runner reads it via
        MDSpec directly. from_md_spec doesn't project it onto
        OpenMMConfig fields (which don't have those fields).

        This is a regression guard: if someone adds a ``topology``
        parameter to OpenMMConfig without updating the runner, the
        builder still needs to pass — a missing projection is correct.
        """
        spec = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
            topology_metadata=TopologyMetadata(
                head_to_tail=True, disulfides=((3, 8),), d_positions=(5, 7)
            ),
        )
        cfg = OpenMMConfig.from_md_spec(spec)
        # No fields on OpenMMConfig mirror topology_metadata; this is
        # constructed silently and the runner reads it via spec.
        assert not hasattr(cfg, "topology_metadata")


# ---------------------------------------------------------------------------
# 2. Engine-specific overlay defaults
# ---------------------------------------------------------------------------


class TestEngineOverlayDefaults:
    """Engine-specific fields (platform, extra FF XMLs, etc.) keep
    default values unless the caller passes them via kwargs.
    """

    def test_platform_defaults_to_opencl(self) -> None:
        spec = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
        )
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.openmm_platform == OPENMM_PLATFORM

    def test_extra_forcefields_default_to_empty_list(self) -> None:
        spec = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
        )
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.extra_forcefields == []
        assert cfg.water_ff_xml == ""

    def test_irmsd_threshold_defaults_to_module_constant(self) -> None:
        spec = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
        )
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.target_irmsd_threshold_a == DEFAULT_IRMSD_THRESHOLD_A


# ---------------------------------------------------------------------------
# 3. Engine overlay overrides
# ---------------------------------------------------------------------------


class TestEngineOverlayOverrides:
    """The caller can override engine-specific fields via ``engine_overrides``."""

    def test_overrides_apply_per_field(self) -> None:
        spec = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
        )
        cfg = OpenMMConfig.from_md_spec(
            spec,
            openmm_platform="CUDA",
            extra_forcefields=["custom_a.xml", "custom_b.xml"],
            water_ff_xml="amber14/tip3p.xml",
            target_irmsd_threshold_a=2.5,
        )
        assert cfg.openmm_platform == "CUDA"
        assert cfg.extra_forcefields == ["custom_a.xml", "custom_b.xml"]
        assert cfg.water_ff_xml == "amber14/tip3p.xml"
        assert cfg.target_irmsd_threshold_a == pytest.approx(2.5)

    def test_unknown_engine_override_raises_type_error(self) -> None:
        """A typo in an override key (e.g. ``production_NS`` vs
        ``production_ns``) must NOT silently fall through; the user
        almost certainly meant to override a field, and dropping it
        leads to confusing runtime behaviour.
        """
        spec = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
        )
        with pytest.raises(TypeError, match="production_NS"):
            OpenMMConfig.from_md_spec(spec, production_NS=300.0)  # pyright: ignore[reportCallIssue]

    def test_engine_neutral_override_also_raises(self) -> None:
        """Engine-neutral fields live on MDSpec. Allowing an override
        here would create two parallel sources of truth. Reject.
        """
        spec = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
        )
        with pytest.raises(TypeError, match="protein_ff"):
            OpenMMConfig.from_md_spec(spec, protein_ff="charmm36m")

    def test_protonation_ph_override_is_rejected(self) -> None:
        """Same rationale: protonation_pH is on MDSpec, not OpenMMConfig."""
        spec = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
        )
        with pytest.raises(TypeError, match="protonation_pH"):
            OpenMMConfig.from_md_spec(spec, protonation_pH=2.0)  # pyright: ignore[reportCallIssue]


# ---------------------------------------------------------------------------
# 4. Equivalence with the legacy __init__ path
# ---------------------------------------------------------------------------


class TestEquivalenceWithLegacyPath:
    """Building an OpenMMConfig via from_md_spec(spec) must produce the
    same engine-neutral subset as building it via OpenMMConfig(...) with
    matching fields.

    This is a regression guard: if someone changes ``from_md_spec``'s
    field mapping without updating the rest of biolab-runners, the test
    catches it before a CI run hits the worker and produces a different
    system than expected.
    """

    def test_from_md_spec_matches_manual_construction(self) -> None:
        spec = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="receptor.pdb",
            peptide_pdb="peptide.pdb",
            output_dir="results/md",
            target="t",
            peptide_id="p",
        )
        cfg_from_spec = OpenMMConfig.from_md_spec(spec)
        cfg_manual = OpenMMConfig(
            receptor_pdb="receptor.pdb",
            peptide_pdb="peptide.pdb",
            output_dir="results/md",
            target="t",
            peptide_id="p",
            nacl_mol=spec.ionic_strength_m,
            temperature_k=spec.temperature_k,
            pressure_atm=spec.pressure_atm,
            timestep_fs=spec.timestep_fs,
            box_padding_nm=spec.box_padding_nm,
            box_shape=spec.box_shape.value,
            protein_ff=spec.protein_ff,
            water_model=spec.water_model,
            production_ns=spec.production_ns,
            save_interval_ps=spec.save_interval_ps,
            checkpoint_interval_hours=spec.checkpoint_interval_hours,
            protonation_ph=spec.protonation_ph,
        )
        # Same dataclass field → same value.
        for field_name in (
            "receptor_pdb",
            "peptide_pdb",
            "output_dir",
            "target",
            "peptide_id",
            "nacl_mol",
            "temperature_k",
            "pressure_atm",
            "timestep_fs",
            "box_padding_nm",
            "box_shape",
            "protein_ff",
            "water_model",
            "production_ns",
            "save_interval_ps",
            "checkpoint_interval_hours",
            "protonation_ph",
            "total_steps",
            "save_every_steps",
            "checkpoint_every_steps",
        ):
            assert getattr(cfg_from_spec, field_name) == getattr(cfg_manual, field_name), (
                f"field {field_name!r} diverges between from_md_spec and manual"
            )


# ---------------------------------------------------------------------------
# 5. End-to-end: MDSpec save → load → OpenMMConfig.from_md_spec
# ---------------------------------------------------------------------------


class TestWireFormatRoundTrip:
    """Proves the canonical ``MDSpec`` JSON wire format is self-sufficient:
    a consumer can persist it via MDSpec.save(), reload with MDSpec.load(),
    and rebuild the OpenMMConfig with no loss of engine-neutral fields.
    """

    def test_full_round_trip_via_disk(self, tmp_path: Path) -> None:
        original = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="receptor.pdb",
            peptide_pdb="peptide.pdb",
            output_dir=str(tmp_path),
            target="FadA",
            peptide_id="pep_001",
            protonation_ph=6.2,
        )
        path = original.save()
        restored = MDSpec.load(path)
        cfg = OpenMMConfig.from_md_spec(restored)

        assert cfg.protein_ff == "Amber99SB-ILDN"
        assert cfg.water_model == "TIP3P"
        assert cfg.box_shape == "cubic"
        assert cfg.box_padding_nm == pytest.approx(1.0)
        assert cfg.nacl_mol == pytest.approx(0.150)
        assert cfg.production_ns == pytest.approx(200.0)
        assert cfg.total_steps == 100_000_000
        assert cfg.target == "FadA"
        assert cfg.peptide_id == "pep_001"

    def test_round_trip_produces_identical_engineneutral_subset(self, tmp_path: Path) -> None:
        """The OpenMMConfig built from the round-tripped spec must match
        the one built from the original spec — proves the wire format is
        a lossless transport for engine-neutral fields.
        """
        original = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="receptor.pdb",
            peptide_pdb="peptide.pdb",
            output_dir=str(tmp_path / "md"),
            target="t",
            peptide_id="p",
        )
        path = original.save()
        restored = MDSpec.load(path)

        cfg_a = OpenMMConfig.from_md_spec(original)
        cfg_b = OpenMMConfig.from_md_spec(restored)

        # Compare engine-neutral fields directly.
        for field_name in (
            "receptor_pdb",
            "peptide_pdb",
            "output_dir",
            "target",
            "peptide_id",
            "nacl_mol",
            "temperature_k",
            "pressure_atm",
            "timestep_fs",
            "box_padding_nm",
            "box_shape",
            "protein_ff",
            "water_model",
            "production_ns",
            "save_interval_ps",
            "checkpoint_interval_hours",
            "protonation_ph",
            "total_steps",
        ):
            assert getattr(cfg_a, field_name) == getattr(cfg_b, field_name), (
                f"field {field_name!r} diverges after round-trip"
            )
