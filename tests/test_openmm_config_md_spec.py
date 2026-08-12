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

import json
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
            equilibration_ps=_extract_equilibration_ps_from_spec_for_test(spec),
            pme=spec.pme,
            minimization_max_iterations=spec.minimization_max_iterations,
            constraints=spec.constraints,
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
            "equilibration_ps",
            "pme",
            "minimization_max_iterations",
            "constraints",
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


# ---------------------------------------------------------------------------
# 6. Deferred fields now projected (biolab-runners#189 / slice 16)
# ---------------------------------------------------------------------------


def _spec_with_custom_equilibration(
    nvt_ps: float,
    npt_restrained_ps: float,
    npt_free_ps: float,
) -> MDSpec:
    """Build an MDSpec with a custom 3-stage equilibration protocol.

    Bypasses ``MDSpec.from_profile`` (which only accepts the
    profile's hardcoded ``equilibration_ps``) by constructing
    ``MDSpec`` directly with a custom ``equilibration`` stage list.
    """
    return MDSpec(
        receptor_pdb="receptor.pdb",
        peptide_pdb="peptide.pdb",
        output_dir="results/md",
        target="t",
        peptide_id="p",
        equilibration=(
            {"name": "NVT", "ensemble": "NVT", "duration_ps": nvt_ps, "restraint_k": 1000.0},
            {
                "name": "NPT-restrained",
                "ensemble": "NPT",
                "duration_ps": npt_restrained_ps,
                "restraint_k": 100.0,
            },
            {
                "name": "NPT-free",
                "ensemble": "NPT",
                "duration_ps": npt_free_ps,
                "restraint_k": 0.0,
            },
        ),
        production_ns=10.0,  # short production; the equilibration is what we're testing
    )


class TestDeferredFieldsProjected:
    """biolab-runners#189 — slice 16 wire-up.

    ``equilibration_ps``, ``pme``, ``minimization_max_iterations``,
    and ``constraints`` were carried on the spec for round-tripping
    but the runner silently dropped them. This class pins the
    ``from_md_spec`` projection so a reviewer-signed-off profile
    change reaches the runner.
    """

    def test_equilibration_ps_extracted_from_spec(self) -> None:
        """``spec.equilibration`` (list of stage dicts) projects onto
        ``OpenMMConfig.equilibration_ps`` (3-tuple of ps durations).

        The runner is hardcoded to 3-stage equilibration, so the
        projection raises if the spec carries a different stage count.
        """
        spec = _spec_with_custom_equilibration(50.0, 75.0, 150.0)
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.equilibration_ps == (50.0, 75.0, 150.0)

    def test_equilibration_ps_default_matches_canonical_profile(self) -> None:
        """The canonical ``ACTIVIN_E_PRODUCTION_PROFILE`` carries
        100/100/200 ps; ``from_md_spec`` reproduces that.
        """
        spec = MDSpec.from_profile(
            ACTIVIN_E_PRODUCTION_PROFILE,
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
        )
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.equilibration_ps == (100.0, 100.0, 200.0)

    def test_equilibration_ps_with_wrong_stage_count_raises(self) -> None:
        """4-stage or 2-stage specs must raise at the projection boundary,
        not silently truncate. The runner is hardcoded for 3 stages.
        """
        spec = MDSpec(
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
            equilibration=(
                {"name": "NVT", "ensemble": "NVT", "duration_ps": 50.0, "restraint_k": 1000.0},
                {"name": "NPT-r", "ensemble": "NPT", "duration_ps": 100.0, "restraint_k": 100.0},
                {"name": "NPT-f", "ensemble": "NPT", "duration_ps": 150.0, "restraint_k": 0.0},
                {"name": "extra", "ensemble": "NPT", "duration_ps": 50.0, "restraint_k": 0.0},
            ),
        )
        with pytest.raises(ValueError, match="3 equilibration stages"):
            OpenMMConfig.from_md_spec(spec)

    def test_total_equil_steps_uses_spec_durations(self) -> None:
        """``total_equil_steps`` is derived from the sum of
        ``equilibration_ps`` (post-projection). A custom spec with
        non-default durations produces a different step count.
        """
        spec = _spec_with_custom_equilibration(50.0, 75.0, 150.0)
        cfg = OpenMMConfig.from_md_spec(spec)
        # 50 + 75 + 150 = 275 ps total equilibration; timestep 2 fs →
        # 137_500 steps.
        assert cfg.total_equil_steps == 137_500

    def test_pme_propagates(self) -> None:
        """``spec.pme`` projects onto ``OpenMMConfig.pme``. The runner
        uses this to select ``app.PME`` vs ``app.Cutoff``.
        """
        spec = MDSpec(
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
            equilibration=_ACTIVIN_E_3_STAGES,
            pme=False,  # coarse-grained system; PME off
            production_ns=10.0,
        )
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.pme is False

    def test_minimization_max_iterations_propagates(self) -> None:
        """``spec.minimization_max_iterations`` projects onto
        ``OpenMMConfig.minimization_max_iterations``. The runner's
        ``simulation.minimizeEnergy(maxIterations=...)`` reads this.
        """
        spec = MDSpec(
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
            equilibration=_ACTIVIN_E_3_STAGES,
            minimization_max_iterations=20_000,
            production_ns=10.0,
        )
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.minimization_max_iterations == 20_000

    def test_constraints_propagates(self) -> None:
        """``spec.constraints`` projects onto ``OpenMMConfig.constraints``.
        ``system_builder._create_system`` maps the string to
        ``app.<NAME>``.
        """
        spec = MDSpec(
            receptor_pdb="r",
            peptide_pdb="p",
            output_dir="o",
            equilibration=_ACTIVIN_E_3_STAGES,
            constraints="AllBonds",
            production_ns=10.0,
        )
        cfg = OpenMMConfig.from_md_spec(spec)
        assert cfg.constraints == "AllBonds"

    def test_to_dict_includes_deferred_fields(self) -> None:
        """``OpenMMConfig.to_dict`` must carry the 4 deferred fields so
        ``system_config.json`` round-trips through disk without loss.
        """
        spec = _spec_with_custom_equilibration(50.0, 75.0, 150.0)
        cfg = OpenMMConfig.from_md_spec(spec)
        # Override one more field to test pme round-trip too.
        cfg.pme = False
        cfg.minimization_max_iterations = 20_000
        cfg.constraints = "AllBonds"
        sim: dict[str, object] = cfg.to_dict()["simulation"]  # type: ignore[assignment]
        assert sim["equilibration_ps"] == [50.0, 75.0, 150.0]
        assert sim["pme"] is False
        assert sim["minimization_max_iterations"] == 20_000
        assert sim["constraints"] == "AllBonds"


# Helper: the canonical Activin-E 3-stage equilibration block, used
# across multiple test methods above.
_ACTIVIN_E_3_STAGES = (
    {"name": "NVT", "ensemble": "NVT", "duration_ps": 100.0, "restraint_k": 1000.0},
    {"name": "NPT-restrained", "ensemble": "NPT", "duration_ps": 100.0, "restraint_k": 100.0},
    {"name": "NPT-free", "ensemble": "NPT", "duration_ps": 200.0, "restraint_k": 0.0},
)


def _extract_equilibration_ps_from_spec_for_test(spec: MDSpec) -> tuple[float, float, float]:
    """Test-side mirror of ``_extract_equilibration_ps`` in config.py.

    Required because pyright treats ``tuple[float, ...]`` as
    indeterminate-length; we need an explicit 3-tuple annotation
    to match the dataclass field type. Kept private to this test
    module — the canonical extraction lives in ``from_md_spec``.
    """
    return (
        float(spec.equilibration[0]["duration_ps"]),
        float(spec.equilibration[1]["duration_ps"]),
        float(spec.equilibration[2]["duration_ps"]),
    )


# ---------------------------------------------------------------------------
# 7. Wire-format round-trip + backwards compatibility (biolab-runners#189)
# ---------------------------------------------------------------------------


class TestJsonWireRoundTrip:
    """``OpenMMConfig.save`` → ``from_json`` round-trip for the 4
    deferred fields, plus legacy-file backwards compatibility.

    The 4 fields are added under the ``simulation`` block of
    ``to_dict``; ``from_json`` parses them back. Legacy
    ``system_config.json`` files (written before slice 16) omit
    the new keys and must default to the legacy hardcoded values
    so in-flight simulations don't break.
    """

    def test_save_then_load_round_trips_deferred_fields(self, tmp_path: Path) -> None:
        spec = _spec_with_custom_equilibration(50.0, 75.0, 150.0)
        cfg = OpenMMConfig.from_md_spec(spec)
        cfg.pme = False
        cfg.minimization_max_iterations = 20_000
        cfg.constraints = "AllBonds"
        path = cfg.save(tmp_path / "system_config.json")
        loaded = OpenMMConfig.from_json(path)

        assert loaded.equilibration_ps == (50.0, 75.0, 150.0)
        assert loaded.pme is False
        assert loaded.minimization_max_iterations == 20_000
        assert loaded.constraints == "AllBonds"

    def test_legacy_json_file_defaults_deferred_fields(self, tmp_path: Path) -> None:
        """A ``system_config.json`` written before slice 16 has no
        ``simulation.equilibration_ps`` / ``pme`` /
        ``minimization_max_iterations`` / ``constraints`` keys.
        ``from_json`` must default to the legacy hardcoded values
        so the simulation continues exactly as it would have
        before the slice.
        """
        legacy = {
            "receptor_pdb": "r.pdb",
            "peptide_pdb": "p.pdb",
            "output_dir": str(tmp_path),
            "target": "t",
            "peptide_id": "p",
            "ionic_conditions": {"NaCl_M": 0.150},
            "simulation": {
                "temperature_K": 310.0,
                "production_ns": 100.0,
            },
            "force_fields": {"protein": "charmm36m", "water": "tip3p"},
        }
        path = tmp_path / "legacy_system_config.json"
        path.write_text(json.dumps(legacy))

        cfg = OpenMMConfig.from_json(path)

        # Legacy defaults — match the runner's hardcoded values.
        assert cfg.equilibration_ps == (100.0, 100.0, 200.0)
        assert cfg.pme is True
        assert cfg.minimization_max_iterations == 1_000
        assert cfg.constraints == "HBonds"
