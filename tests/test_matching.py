import math

import numpy as np
import pytest

from runner import _matched_ordering_metrics, _matched_wasserstein_cost


def _profile(mean, vol, cvar, downside_vol=None, var_alpha=None, n=100):
    return {
        "n": float(n),
        "mean": float(mean),
        "vol": float(vol),
        "downside_vol": float(downside_vol if downside_vol is not None else vol * 0.8),
        "var_alpha": float(var_alpha if var_alpha is not None else cvar * 0.7),
        "cvar_alpha": float(cvar),
    }


class TestMatchedWassersteinCost:
    def test_identical_samples_per_state_is_zero(self):
        rng = np.random.default_rng(0)
        s0 = rng.normal(-1.0, 0.1, 200)
        s1 = rng.normal(1.0, 0.1, 200)
        cost, n_finite, k = _matched_wasserstein_cost([s0, s1], [s0, s1])
        assert cost == pytest.approx(0.0, abs=1e-12)
        assert (n_finite, k) == (2, 2)

    def test_permuted_states_recover_zero_cost(self):
        rng = np.random.default_rng(1)
        s0 = rng.normal(-1.0, 0.1, 200)
        s1 = rng.normal(1.0, 0.1, 200)
        cost, n_finite, k = _matched_wasserstein_cost([s0, s1], [s1, s0])
        assert cost == pytest.approx(0.0, abs=1e-12)
        assert (n_finite, k) == (2, 2)

    def test_pure_shift_equals_shift_magnitude(self):
        a = np.array([0.0, 1.0, 2.0, 3.0])
        b = a + 1.0
        cost, n_finite, k = _matched_wasserstein_cost([a], [b])
        assert cost == pytest.approx(1.0, rel=1e-12)
        assert (n_finite, k) == (1, 1)

    def test_empty_sample_list_returns_nan(self):
        cost, n_finite, k = _matched_wasserstein_cost([], [])
        assert math.isnan(cost)
        assert (n_finite, k) == (0, 0)

    def test_one_empty_state_is_penalized(self):
        a_good = np.array([1.0, 2.0, 3.0])
        b_good = np.array([1.0, 2.0, 3.0])
        b_shift = np.array([4.0, 5.0, 6.0])
        # W(a_good, b_good) = 0; W(a_good, b_shift) = 3.
        # The empty state on side A cannot match anything, so it gets the penalty.
        cost, n_finite, k = _matched_wasserstein_cost(
            [a_good, np.array([])],
            [b_good, b_shift],
        )
        assert math.isfinite(cost)
        # Penalty is 10 * max_finite + 1 = 31; matched = (0, 31); mean = 15.5.
        assert cost == pytest.approx(15.5, rel=1e-12)
        # Only 1 of k=2 matched pairs came from a finite cost cell —
        # the other was penalty-filled. Downstream callers can use this
        # to filter or flag partially-defined comparisons.
        assert (n_finite, k) == (1, 2)


class TestMatchedOrderingMetrics:
    def _three_state_profiles(self):
        # Distinct (cvar, vol) per state so Hungarian has a unique solution.
        return [
            _profile(mean=0.001, vol=0.01, cvar=-0.03, downside_vol=0.008),
            _profile(mean=-0.001, vol=0.02, cvar=-0.06, downside_vol=0.015),
            _profile(mean=-0.010, vol=0.03, cvar=-0.12, downside_vol=0.025),
        ]

    def test_identical_profiles_are_perfectly_consistent(self):
        profs = self._three_state_profiles()
        out = _matched_ordering_metrics(profs, profs)
        assert out["top1_consistency"] == pytest.approx(1.0)
        assert out["spearman"] == pytest.approx(1.0, rel=1e-12)
        assert out["high_risk_mean_sign_consistency"] == pytest.approx(1.0)
        assert out["high_risk_mean_abs_diff"] == pytest.approx(0.0, abs=1e-12)
        assert out["high_risk_downside_vol_abs_diff"] == pytest.approx(0.0, abs=1e-12)

    def test_permuted_profiles_are_recovered_by_matching(self):
        profs_a = self._three_state_profiles()
        # Cyclic shift of state order in B. Hungarian on (|cvar diff|+|vol diff|)
        # should re-pair them, restoring perfect ordering consistency.
        profs_b = [profs_a[2], profs_a[0], profs_a[1]]
        out = _matched_ordering_metrics(profs_a, profs_b)
        assert out["top1_consistency"] == pytest.approx(1.0)
        assert out["spearman"] == pytest.approx(1.0, rel=1e-12)
        assert out["high_risk_mean_abs_diff"] == pytest.approx(0.0, abs=1e-12)
        assert out["high_risk_downside_vol_abs_diff"] == pytest.approx(0.0, abs=1e-12)

    def test_single_state_returns_all_nan(self):
        profs = [_profile(mean=0.0, vol=0.01, cvar=-0.03)]
        out = _matched_ordering_metrics(profs, profs)
        for key in (
            "top1_consistency",
            "spearman",
            "high_risk_mean_sign_consistency",
            "high_risk_mean_abs_diff",
            "high_risk_downside_vol_abs_diff",
        ):
            assert math.isnan(out[key]), f"{key} should be nan when k<=1"

    def test_top1_disagreement_when_vol_drives_matching(self):
        # A: worst-cvar state is index 0 (cvar=-1, vol=0).
        # B: worst-cvar state is index 1 (cvar=-1, vol=0.5).
        # With cost = |cvar diff| + |vol diff|, the vol term pairs the low-vol states
        # together, mismatching the worst-cvar states.
        profs_a = [
            _profile(mean=-0.01, vol=0.00, cvar=-1.00),
            _profile(mean=-0.01, vol=0.50, cvar=-0.90),
        ]
        profs_b = [
            _profile(mean=-0.01, vol=0.00, cvar=-0.90),
            _profile(mean=-0.01, vol=0.50, cvar=-1.00),
        ]
        out = _matched_ordering_metrics(profs_a, profs_b)
        assert out["top1_consistency"] == pytest.approx(0.0)
