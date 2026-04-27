import math

import numpy as np
import pandas as pd
import pytest

from metrics import align_states, temporal_disjoint_metrics, variation_of_information


class TestAlignStates:
    def test_identity_returns_target_unchanged(self):
        ref = np.array([0, 1, 2, 0, 1, 2], dtype=int)
        tgt = ref.copy()
        out = align_states(ref, tgt, n_states=3)
        np.testing.assert_array_equal(out, ref)

    def test_cyclic_permutation_is_recovered(self):
        ref = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=int)
        # target labels are ref cycled by (0->1, 1->2, 2->0)
        tgt = np.array([1, 1, 1, 2, 2, 2, 0, 0, 0], dtype=int)
        out = align_states(ref, tgt, n_states=3)
        np.testing.assert_array_equal(out, ref)

    def test_dominant_match_increases_agreement(self):
        ref = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        tgt = np.array([1, 1, 0, 0, 0, 1], dtype=int)
        before = int(np.sum(ref == tgt))
        out = align_states(ref, tgt, n_states=2)
        after = int(np.sum(ref == out))
        assert after >= before
        assert after == 4  # Hungarian swap: 2 matches -> 4 matches

    def test_empty_inputs_return_empty_array(self):
        ref = np.array([], dtype=int)
        tgt = np.array([], dtype=int)
        out = align_states(ref, tgt, n_states=2)
        assert out.shape == (0,)
        assert out.dtype == np.int_ or out.dtype == np.int64 or out.dtype == np.int32


class TestVariationOfInformation:
    def test_identical_partitions_is_zero(self):
        a = np.array([0, 0, 1, 1, 2, 2], dtype=int)
        vi = variation_of_information(a, a)
        assert vi == pytest.approx(0.0, abs=1e-12)

    def test_independent_balanced_binary_equals_two_ln2(self):
        # Construction where MI(a,b) = 0 exactly: joint matches marginal product.
        a = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        b = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        vi = variation_of_information(a, b)
        assert vi == pytest.approx(2.0 * math.log(2.0), rel=1e-10)

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 3, size=50)
        b = rng.integers(0, 3, size=50)
        assert variation_of_information(a, b) == pytest.approx(
            variation_of_information(b, a), rel=1e-12
        )

    def test_empty_input_is_nan(self):
        a = np.array([], dtype=int)
        b = np.array([], dtype=int)
        assert math.isnan(variation_of_information(a, b))


class TestTemporalDisjointMetrics:
    def test_equal_length_disjoint_segments_match_perfectly(self):
        # a has indices 1..8; b has indices 5..12 (step=4 overlap pattern).
        # idx_a_only = {1,2,3,4}; idx_b_only = {9,10,11,12}; both length 4.
        # a[1..4] = [0,0,1,1]; b[9..12] = [0,0,1,1] -> ARI = 1.0 (exact match).
        a = pd.Series([0, 0, 1, 1, 0, 0, 1, 1], index=list(range(1, 9)))
        b = pd.Series([0, 0, 1, 1, 0, 0, 1, 1], index=list(range(5, 13)))
        scores = temporal_disjoint_metrics(a, b)
        assert scores.ari == pytest.approx(1.0)

    def test_center_truncation_on_unequal_lengths(self):
        # Regression test for center-truncation fix.
        #
        # idx_a_only = {1,2,3,4} -> a_only = [0,0,1,1] (length 4)
        # idx_b_only = {13,...,20} -> b_only = [0,0,0,0,1,1,0,0] (length 8)
        #
        # Head-truncation (old behavior, `[:4]`):
        #   a_only vs b_only[:4] = [0,0,1,1] vs [0,0,0,0] -> ARI = 0
        # Center-truncation (new behavior):
        #   a_only vs b_only[2:6] = [0,0,1,1] vs [0,0,1,1] -> ARI = 1.0
        a = pd.Series([0, 0, 1, 1] + [0] * 8, index=list(range(1, 13)))
        b_vals = [0] * 8 + [0, 0, 0, 0, 1, 1, 0, 0]
        b = pd.Series(b_vals, index=list(range(5, 21)))
        scores = temporal_disjoint_metrics(a, b)
        assert scores.ari == pytest.approx(1.0)

    def test_identical_indices_returns_nan(self):
        a = pd.Series([0, 1, 0, 1], index=list(range(1, 5)))
        b = pd.Series([0, 1, 0, 1], index=list(range(1, 5)))
        scores = temporal_disjoint_metrics(a, b)
        assert math.isnan(scores.ari)
        assert math.isnan(scores.ami)
        assert math.isnan(scores.vi)
