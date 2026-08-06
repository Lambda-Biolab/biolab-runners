"""Tests for the PBC geometry helpers in ``biolab_runners.openmm.geometry``.

These exercise the public surface (no OpenMM dependency). Property-based
tests use Hypothesis to assert invariants of ``pbc_correct``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from biolab_runners.openmm.geometry import (
    collect_chain_ca_positions,
    min_pbc_distance,
    pbc_correct,
)
from biolab_runners.openmm.offline_gate import FloatArray

from tests._helpers import FakeAtom, FakeChain, dodecahedron_box

# ---------------------------------------------------------------------------
# collect_chain_ca_positions
# ---------------------------------------------------------------------------


class TestCollectChainCaPositions:
    """Chain 0 → receptor, chain 1+ → peptide."""

    def test_splits_by_chain_index(self) -> None:
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        )
        chain0 = FakeChain([FakeAtom("CA", 0), FakeAtom("N", 1), FakeAtom("CA", 2)])
        chain1 = FakeChain([FakeAtom("CA", 3), FakeAtom("CA", 4)])
        rec, pep = collect_chain_ca_positions([chain0, chain1], positions)
        assert len(rec) == 2
        assert len(pep) == 2
        # rec[i] and pep[i] are typed as object (the public surface accepts
        # any array-like for compatibility with OpenMM Quantity rows). Cast
        # to numpy arrays so assert_array_equal's ArrayLike parameter is
        # satisfied under strict pyright mode.
        np.testing.assert_array_equal(cast("Any", rec[0]), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(cast("Any", rec[1]), [2.0, 0.0, 0.0])
        np.testing.assert_array_equal(cast("Any", pep[0]), [3.0, 0.0, 0.0])
        np.testing.assert_array_equal(cast("Any", pep[1]), [4.0, 0.0, 0.0])

    def test_ignores_non_ca_atoms(self) -> None:
        positions = np.array([[0.0], [1.0], [2.0]])
        chain = FakeChain(
            [
                FakeAtom("N", 0),
                FakeAtom("CA", 1),
                FakeAtom("C", 2),
                FakeAtom("O", 0),
            ]
        )
        rec, pep = collect_chain_ca_positions([chain], positions)
        assert len(rec) == 1
        assert len(pep) == 0

    def test_three_chains_all_peptide_after_first(self) -> None:
        """Chains after index 0 all go into the peptide list (multi-chain peptide)."""
        positions = np.array([[i, 0.0, 0.0] for i in range(6)])
        chain0 = FakeChain([FakeAtom("CA", 0), FakeAtom("CA", 1)])
        chain1 = FakeChain([FakeAtom("CA", 2), FakeAtom("CA", 3)])
        chain2 = FakeChain([FakeAtom("CA", 4), FakeAtom("CA", 5)])
        rec, pep = collect_chain_ca_positions([chain0, chain1, chain2], positions)
        assert len(rec) == 2
        assert len(pep) == 4

    def test_empty_chains(self) -> None:
        positions = np.zeros((0, 3))
        rec, pep = collect_chain_ca_positions([], positions)
        assert rec == []
        assert pep == []


# ---------------------------------------------------------------------------
# pbc_correct (deterministic)
# ---------------------------------------------------------------------------


class TestPbcCorrectDeterministic:
    """Smoke + shape tests for ``pbc_correct``."""

    def test_orthorhombic_zero_diff(self) -> None:
        box = np.diag([10.0, 10.0, 10.0])
        diff = np.array([[0.0, 0.0, 0.0]])
        out = pbc_correct(diff, box, np)
        np.testing.assert_allclose(out, 0.0, atol=1e-10)

    def test_orthorhombic_half_box_wraps(self) -> None:
        box = np.diag([10.0, 10.0, 10.0])
        # Displacement of 6.0 in a 10.0 box → wraps to -4.0
        diff = np.array([[6.0, 0.0, 0.0]])
        out = pbc_correct(diff, box, np)
        np.testing.assert_allclose(out, [[-4.0, 0.0, 0.0]], atol=1e-10)

    def test_orthorhombic_two_atoms_3d(self) -> None:
        box = np.diag([5.0, 5.0, 5.0])
        # Use 2.4 (no wrap) and 3.0 (wraps to -2.0) to avoid the 0.5
        # banker's-rounding boundary case in pbc_correct.
        diff = np.array([[2.4, 0.0, 0.0], [0.0, 3.0, 0.0]])
        out = pbc_correct(diff, box, np)
        np.testing.assert_allclose(out, [[2.4, 0.0, 0.0], [0.0, -2.0, 0.0]], atol=1e-10)

    def test_preserves_shape(self) -> None:
        box = np.diag([5.0, 5.0, 5.0])
        diff = np.random.default_rng(0).uniform(-3.0, 3.0, size=(7, 11, 3))
        out = pbc_correct(diff, box, np)
        assert out.shape == (7, 11, 3)

    def test_dodecahedron_diagonal_face_zero(self) -> None:
        """Displacement of a full lattice vector collapses to zero."""
        box = dodecahedron_box(60.0)
        diff = box[2].reshape(1, 3).copy()
        out = pbc_correct(diff, box, np)
        np.testing.assert_allclose(out, 0.0, atol=1e-10)

        # Regression #163: the pre-fix diagonal-only formula left
        # off-diagonal slop behind. With d = 60 Å, the stale xy
        # displacement is ≈ d/√2 ≈ 42 Å.
        box_diag = np.array([box[0, 0], box[1, 1], box[2, 2]])
        pre_fix = diff - np.round(diff / box_diag) * box_diag
        assert np.linalg.norm(pre_fix) > 30.0

    def test_dodecahedron_face_crossing_minimum_image(self) -> None:
        """Regression #163: displacement across a diagonal face wraps to ~0.5 Å.

        In a bound complex with the receptor near the origin and the
        peptide images diffusing across a non-orthogonal face, the
        diagonal-only code reported min distances of 30–40 Å while
        the true minimum-image distance is < 1 Å.
        """
        box = dodecahedron_box(d=60.0)
        rec = np.array([[0.0, 0.0, 0.0]])
        pep = (box[2] + np.array([0.3, 0.3, 0.1])).reshape(1, 3)
        diff = pep - rec
        out = pbc_correct(diff.copy(), box, np)
        min_dist = float(np.linalg.norm(out, axis=-1).min())
        assert min_dist < 1.0  # true distance ≈ 0.436 Å

        # Pre-fix diagonal code returned ~36 Å here (huge false "dissociation").
        box_diag = np.array([box[0, 0], box[1, 1], box[2, 2]])
        pre_fix = diff - np.round(diff / box_diag) * box_diag
        assert float(np.linalg.norm(pre_fix, axis=-1).min()) > 30.0

    def test_broadcasts_over_leading_axes(self) -> None:
        """``pbc_correct`` must accept (M, N, 3) arrays used by ``min_pbc_distance``."""
        box = dodecahedron_box()
        rec = np.zeros((3, 3))
        pep = np.tile(box[2], (4, 1)) + 0.5  # 4 peptide atoms past one c-image
        diffs = rec[:, None, :] - pep[None, :, :]  # shape (3, 4, 3)
        out = pbc_correct(diffs.copy(), box, np)
        assert out.shape == (3, 4, 3)
        # All pairs wrap to the same ~0.866 Å displacement.
        assert np.allclose(np.linalg.norm(out, axis=-1), np.sqrt(0.75), atol=1e-10)

    def test_orthorhombic_parity_with_diagonal_formula(self) -> None:
        """For rectangular boxes the new formula must agree with the diagonal one."""
        box = np.diag([60.0, 45.0, 80.0]).astype(float)
        rng = np.random.default_rng(seed=0)
        diff = rng.uniform(-100.0, 100.0, size=(8, 3))
        out = pbc_correct(diff.copy(), box, np)
        box_diag = np.array([box[0, 0], box[1, 1], box[2, 2]])
        expected = diff - np.round(diff / box_diag) * box_diag
        assert np.allclose(out, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# pbc_correct (property-based)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("hypothesis"),
    reason="hypothesis not installed",
)
class TestPbcCorrectProperties:
    """Invariants that must hold for any lattice / displacement."""

    @pytest.fixture
    def small_box(self) -> FloatArray:
        return np.diag([7.0, 8.0, 9.0])

    def test_lattice_vector_always_zero(self, small_box: FloatArray) -> None:
        """Any integer combination of lattice vectors wraps to 0."""
        from hypothesis import given
        from hypothesis import strategies as st

        @given(coeffs=st.lists(st.integers(min_value=-3, max_value=3), min_size=3, max_size=3))
        def inner(coeffs: list[int]) -> None:
            disp = sum(c * v for c, v in zip(coeffs, small_box, strict=True))
            out = pbc_correct(disp.reshape(1, 3), small_box, np)
            np.testing.assert_allclose(out, 0.0, atol=1e-10)

        inner()  # type: ignore[reportCallIssue]

    def test_idempotent(self, small_box: FloatArray) -> None:
        """pbc_correct(pbc_correct(x)) == pbc_correct(x) for any x in the box."""
        from hypothesis import given
        from hypothesis import strategies as st

        @given(
            disp=st.lists(
                st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=3,
                max_size=3,
            )
        )
        def inner(disp: list[float]) -> None:
            arr = np.array(disp).reshape(1, 3)
            once = pbc_correct(arr, small_box, np)
            twice = pbc_correct(once, small_box, np)
            np.testing.assert_allclose(once, twice, atol=1e-10)

        inner()  # type: ignore[reportCallIssue]

    def test_output_within_half_box(self, small_box: FloatArray) -> None:
        """For orthorhombic cells, output components lie in [-L/2, L/2)."""
        from hypothesis import given
        from hypothesis import strategies as st

        @given(
            disp=st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=3,
                max_size=3,
            )
        )
        def inner(disp: list[float]) -> None:
            arr = np.array(disp).reshape(1, 3)
            out = pbc_correct(arr, small_box, np)
            half = np.diag(small_box) / 2.0
            assert np.all(np.abs(out) <= half + 1e-10), f"out={out}, half={half}"

        inner()  # type: ignore[reportCallIssue]


# ---------------------------------------------------------------------------
# min_pbc_distance
# ---------------------------------------------------------------------------


class TestMinPbcDistance:
    """``min_pbc_distance`` is a thin wrapper over ``pbc_correct``."""

    def test_zero_for_same_atom(self) -> None:
        box = np.diag([10.0, 10.0, 10.0])
        pos = [np.array([1.0, 2.0, 3.0])]
        d = min_pbc_distance(pos, pos, box, np)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_direct_distance(self) -> None:
        box = np.diag([10.0, 10.0, 10.0])
        rec = [np.array([0.0, 0.0, 0.0])]
        pep = [np.array([3.0, 4.0, 0.0])]
        d = min_pbc_distance(rec, pep, box, np)
        assert d == pytest.approx(5.0, abs=1e-10)

    def test_min_over_all_pairs(self) -> None:
        box = np.diag([10.0, 10.0, 10.0])
        rec = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        # Pairs: (0, 0.5)=0.5, (0, 10)=0 (wrap), (1, 0.5)=0.5, (1, 10)=1
        pep = [np.array([0.5, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])]
        d = min_pbc_distance(rec, pep, box, np)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_dodecahedron_face_crossing(self) -> None:
        """OralBiome-AMP #163: peptide across a non-orthogonal face wraps to ~0.5 A."""
        d = 60.0
        box = np.array(
            [
                [d, 0.0, 0.0],
                [0.0, d, 0.0],
                [0.5 * d, 0.5 * d, d / np.sqrt(2.0)],
            ]
        )
        rec = [np.array([0.0, 0.0, 0.0])]
        pep = [(box[2] + np.array([0.3, 0.3, 0.1])).reshape(3)]
        dist = min_pbc_distance(rec, pep, box, np)
        assert dist < 1.0  # true minimum is ≈ 0.436 A

    def test_empty_input_is_caller_responsibility(self) -> None:
        """min_pbc_distance assumes non-empty inputs (caller must guard).

        The real caller (``_check_post_equilibration_displacement``) checks
        ``if not (rec_ca and pep_ca): return`` before invoking this
        function. We document the precondition here so any future caller
        that forgets the guard fails loudly with IndexError rather than
        silently returning a misleading value.
        """
        box = np.diag([10.0, 10.0, 10.0])
        with pytest.raises(IndexError):
            min_pbc_distance([], [np.array([0.0, 0.0, 0.0])], box, np)


# ---------------------------------------------------------------------------
# Closed-form cross-checks (test-quality.md rule #3)
# ---------------------------------------------------------------------------
#
# Each test asserts a specific value derived from the closed-form
# minimum-image formula. For orthorhombic boxes (diagonal matrix Lx, Ly,
# Lz), the inverse is diag(1/Lx, 1/Ly, 1/Lz), so the fractional coordinate
# is diff[i]/L_i, snap-to-nearest-integer gives the [-0.5, 0.5] image,
# and the corrected displacement is round(diff[i]/L_i - 0.5) * L_i.
# ±1e-9 catches arithmetic mutations (sign flip, off-by-one in the
# round-to-nearest step, transpose of the inverse-lattice multiplication).
# ---------------------------------------------------------------------------


class TestPbcCorrectClosedForm:
    """Hand-derived minimum-image values for orthorhombic and triclinic boxes."""

    def test_orthorhombic_half_box_boundary_stays(self) -> None:
        """Displacement of exactly +5 in a 10-unit box → stays at +5 (boundary).
        frac = 5/10 = 0.5 → round(0.5) = 0 (banker's rounding in numpy).
        corrected = (0.5 - 0) * 10 = +5.0. Displacement of +5.01 wraps to -4.99.
        """
        box = np.diag([10.0, 10.0, 10.0])
        diff = np.array([[5.0, 0.0, 0.0]])
        out = pbc_correct(diff, box, np)
        # Boundary case: numpy's banker's rounding keeps +5.0 as +5.0
        np.testing.assert_allclose(out, [[5.0, 0.0, 0.0]], atol=1e-9)

    def test_orthorhombic_just_past_half_wraps(self) -> None:
        """Displacement of +5.01 in a 10-unit box → wraps to -4.99.
        frac = 5.01/10 = 0.501 → round(0.501) = 1 → corrected = (0.501 - 1) * 10 = -4.99.
        """
        box = np.diag([10.0, 10.0, 10.0])
        diff = np.array([[5.01, 0.0, 0.0]])
        out = pbc_correct(diff, box, np)
        np.testing.assert_allclose(out, [[-4.99, 0.0, 0.0]], atol=1e-9)

    def test_orthorhombic_three_quarter_box_wraps(self) -> None:
        """Displacement of +7.5 in a 10-unit box → wraps to -2.5.
        frac = 7.5/10 = 0.75 → round(0.75) = 1 → corrected = (0.75 - 1) * 10 = -2.5.
        """
        box = np.diag([10.0, 10.0, 10.0])
        diff = np.array([[7.5, 0.0, 0.0]])
        out = pbc_correct(diff, box, np)
        np.testing.assert_allclose(out, [[-2.5, 0.0, 0.0]], atol=1e-9)

    def test_orthorhombic_full_box_wraps_to_zero(self) -> None:
        """Displacement of exactly +10 in a 10-unit box → wraps to 0.
        frac = 10/10 = 1.0 → round(1.0) = 1 → corrected = (1.0 - 1) * 10 = 0.
        """
        box = np.diag([10.0, 10.0, 10.0])
        diff = np.array([[10.0, 0.0, 0.0]])
        out = pbc_correct(diff, box, np)
        np.testing.assert_allclose(out, [[0.0, 0.0, 0.0]], atol=1e-9)

    def test_negative_displacement_wraps_correctly(self) -> None:
        """Displacement of -7 in a 10-unit box → wraps to +3.
        frac = -7/10 = -0.7 → round(-0.7) = -1 → corrected = (-0.7 - (-1)) * 10 = +3.
        """
        box = np.diag([10.0, 10.0, 10.0])
        diff = np.array([[-7.0, 0.0, 0.0]])
        out = pbc_correct(diff, box, np)
        np.testing.assert_allclose(out, [[3.0, 0.0, 0.0]], atol=1e-9)


class TestMinPbcDistanceClosedForm:
    """Hand-derived min PBC distances for orthorhombic boxes."""

    def test_two_atoms_3d_distance(self) -> None:
        """Receptor at (0,0,0), peptide at (3,4,0) in a 10-unit box.
        Minimum-image distance = sqrt(9 + 16 + 0) = 5.0.
        """
        box = np.diag([10.0, 10.0, 10.0])
        rec = [np.array([0.0, 0.0, 0.0])]
        pep = [np.array([3.0, 4.0, 0.0])]
        assert min_pbc_distance(rec, pep, box, np) == pytest.approx(5.0, abs=1e-9)

    def test_atoms_across_box_boundary(self) -> None:
        """Receptor at (1,0,0), peptide at (9,0,0) in a 10-unit box.
        Direct distance = 8; minimum-image distance = 2 (peptides wrap).
        """
        box = np.diag([10.0, 10.0, 10.0])
        rec = [np.array([1.0, 0.0, 0.0])]
        pep = [np.array([9.0, 0.0, 0.0])]
        assert min_pbc_distance(rec, pep, box, np) == pytest.approx(2.0, abs=1e-9)

    def test_atoms_at_box_boundary_minimum_image(self) -> None:
        """Receptor at (0,0,0), peptide at (5,5,5) in a 10-unit box.
        Direct distance = sqrt(75) ≈ 8.660; minimum-image distance is
        min(sqrt(75), sqrt((-5)^2 + (-5)^2 + (-5)^2)) = sqrt(75) — both
        images give the same distance since 5 < 10/2.
        """
        box = np.diag([10.0, 10.0, 10.0])
        rec = [np.array([0.0, 0.0, 0.0])]
        pep = [np.array([5.0, 5.0, 5.0])]
        expected = float(np.sqrt(75))
        assert min_pbc_distance(rec, pep, box, np) == pytest.approx(expected, abs=1e-9)

    def test_min_over_multiple_pairs(self) -> None:
        """Multiple receptor / peptide pairs — function returns the min
        over all pairs.
        Receptor at (0,0,0), (1,0,0).
        Peptide at (8,0,0), (3,4,0).
        Distances: (0,0,0)-(8,0,0) → min=2; (0,0,0)-(3,4,0) → 5;
        (1,0,0)-(8,0,0) → min=3; (1,0,0)-(3,4,0) → sqrt(4+16)=sqrt(20).
        Overall min = 2.0.
        """
        box = np.diag([10.0, 10.0, 10.0])
        rec = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        pep = [np.array([8.0, 0.0, 0.0]), np.array([3.0, 4.0, 0.0])]
        assert min_pbc_distance(rec, pep, box, np) == pytest.approx(2.0, abs=1e-9)
