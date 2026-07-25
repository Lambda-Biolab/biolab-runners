"""Tests for the PBC geometry helpers in ``biolab_runners.openmm.geometry``.

These exercise the public surface (no OpenMM dependency). Property-based
tests use Hypothesis to assert invariants of ``pbc_correct``.
"""

from __future__ import annotations

import numpy as np
import pytest
from biolab_runners.openmm.geometry import (
    collect_chain_ca_positions,
    min_pbc_distance,
    pbc_correct,
)

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
        np.testing.assert_array_equal(rec[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(rec[1], [2.0, 0.0, 0.0])
        np.testing.assert_array_equal(pep[0], [3.0, 0.0, 0.0])
        np.testing.assert_array_equal(pep[1], [4.0, 0.0, 0.0])

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
    def small_box(self) -> np.ndarray:
        return np.diag([7.0, 8.0, 9.0])

    def test_lattice_vector_always_zero(self, small_box: np.ndarray) -> None:
        """Any integer combination of lattice vectors wraps to 0."""
        from hypothesis import given
        from hypothesis import strategies as st

        @given(coeffs=st.lists(st.integers(min_value=-3, max_value=3), min_size=3, max_size=3))
        def inner(coeffs: list[int]) -> None:
            disp = sum(c * v for c, v in zip(coeffs, small_box, strict=True))
            out = pbc_correct(disp.reshape(1, 3), small_box, np)
            np.testing.assert_allclose(out, 0.0, atol=1e-10)

        inner()

    def test_idempotent(self, small_box: np.ndarray) -> None:
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

        inner()

    def test_output_within_half_box(self, small_box: np.ndarray) -> None:
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

        inner()


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
