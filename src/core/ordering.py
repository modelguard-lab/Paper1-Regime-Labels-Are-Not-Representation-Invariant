"""
Ordering-consistency analysis (cross-representation and temporal).

Computes how often per-state risk profile orderings agree across
representations (and across temporal windows), with both Hungarian-
matched and rank-permutation-null baselines.

Functions:
  _matched_ordering_metrics                          matched orderings + metrics
  _ordering_null_distribution                        permutation null
  _ordering_one_seed                                 cross-rep ordering per seed
  _compute_ordering_consistency_crossrep_seed_summary
  _ordering_independent_null_distribution            independent rank null
  _compute_ordering_null_baseline                    aggregate null baseline
  _temporal_ordering_one_seed                        temporal ordering per seed
  _compute_ordering_consistency_temporal_seed_summary
"""

from __future__ import annotations

import logging
import math
from itertools import combinations, permutations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.core.parallel import _parallel_or_sequential
from src.core.stability import (
    _matched_wasserstein_cost,
    _risk_profile_from_returns,
    _risk_profiles_by_state,
    _state_return_samples,
)

logger = logging.getLogger(__name__)


def _matched_ordering_metrics(
    prof_a: List[Dict[str, float]],
    prof_b: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Match states using Hungarian on a simple profile distance, then compute:
    - top1_consistency: whether highest-risk state aligns (after matching)
    - spearman: correlation between risk ranks across matched states
    """
    k = int(min(len(prof_a), len(prof_b)))
    if k <= 1:
        return {
            "top1_consistency": float("nan"),
            "spearman": float("nan"),
            "high_risk_mean_sign_consistency": float("nan"),
            "high_risk_mean_abs_diff": float("nan"),
            "high_risk_downside_vol_abs_diff": float("nan"),
        }

    # Cost = |cvar diff| + |vol diff|; missing values incur penalty.
    cost = np.full((k, k), np.nan, dtype=float)
    for i in range(k):
        for j in range(k):
            a_c = float(prof_a[i].get("cvar_alpha", float("nan")))
            b_c = float(prof_b[j].get("cvar_alpha", float("nan")))
            a_v = float(prof_a[i].get("vol", float("nan")))
            b_v = float(prof_b[j].get("vol", float("nan")))
            if not (math.isfinite(a_c) and math.isfinite(b_c) and math.isfinite(a_v) and math.isfinite(b_v)):
                continue
            cost[i, j] = abs(a_c - b_c) + abs(a_v - b_v)

    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        return {
            "top1_consistency": float("nan"),
            "spearman": float("nan"),
            "high_risk_mean_sign_consistency": float("nan"),
            "high_risk_mean_abs_diff": float("nan"),
            "high_risk_downside_vol_abs_diff": float("nan"),
        }

    penalty = float(finite.max()) * 10.0 + 1.0
    cost_filled = np.where(np.isfinite(cost), cost, penalty)
    row_ind, col_ind = linear_sum_assignment(cost_filled)
    mapping = {int(i): int(j) for i, j in zip(row_ind, col_ind)}

    cvar_a = np.array([float(p.get("cvar_alpha", float("nan"))) for p in prof_a[:k]], dtype=float)
    cvar_b = np.array([float(p.get("cvar_alpha", float("nan"))) for p in prof_b[:k]], dtype=float)

    # High-risk = most negative CVaR (smallest value).
    top_a = int(np.nanargmin(cvar_a)) if np.isfinite(cvar_a).any() else None
    top_b = int(np.nanargmin(cvar_b)) if np.isfinite(cvar_b).any() else None
    if top_a is None or top_b is None or top_a not in mapping:
        top1 = float("nan")
    else:
        top1 = 1.0 if int(mapping[top_a]) == int(top_b) else 0.0

    # Directional consistency for the high-risk (worst-CVaR) state:
    # compare the state in B matched to A's high-risk state.
    if top_a is None or top_a not in mapping:
        sign_cons = float("nan")
        mean_abs_diff = float("nan")
        dvol_abs_diff = float("nan")
    else:
        j = int(mapping[top_a])
        mean_a = float(prof_a[top_a].get("mean", float("nan")))
        mean_b = float(prof_b[j].get("mean", float("nan")))
        dv_a = float(prof_a[top_a].get("downside_vol", float("nan")))
        dv_b = float(prof_b[j].get("downside_vol", float("nan")))
        if math.isfinite(mean_a) and math.isfinite(mean_b):
            sa = float(np.sign(mean_a))
            sb = float(np.sign(mean_b))
            sign_cons = 1.0 if sa == sb else 0.0
            mean_abs_diff = float(abs(mean_a - mean_b))
        else:
            sign_cons = float("nan")
            mean_abs_diff = float("nan")
        dvol_abs_diff = float(abs(dv_a - dv_b)) if (math.isfinite(dv_a) and math.isfinite(dv_b)) else float("nan")

    # Rank consistency: Spearman between CVaR ranks after matching.
    ra = pd.Series(cvar_a).rank(ascending=True, method="average")
    rb = pd.Series(cvar_b).rank(ascending=True, method="average")
    rb_m = pd.Series({i: float(rb[mapping.get(i, -1)]) if i in mapping else float("nan") for i in range(k)})
    df = pd.DataFrame({"ra": ra, "rb": rb_m}).dropna()
    if len(df) <= 1:
        sp = float("nan")
    else:
        # Spearman(ra, rb) == Pearson(rank(ra), rank(rb)). ra/rb already ranks.
        sp = float(df["ra"].corr(df["rb"], method="pearson"))

    return {
        "top1_consistency": float(top1),
        "spearman": float(sp),
        "high_risk_mean_sign_consistency": float(sign_cons),
        "high_risk_mean_abs_diff": float(mean_abs_diff),
        "high_risk_downside_vol_abs_diff": float(dvol_abs_diff),
    }


def _ordering_null_distribution(
    returns: pd.Series,
    states: pd.Series,
    n_states: int,
    n_perm: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Compute chance-level ordering metrics under random label permutation.

    Randomly permutes the state labels of *states* B times, recomputes the
    risk profiles, and then compares each permuted profile to the original
    using ``_matched_ordering_metrics``.  Returns mean top-1 and mean
    Spearman under the null (plus 95th-percentile for reference).
    """
    rng = np.random.default_rng(seed)
    samples_orig = _state_return_samples(returns, states, n_states)
    prof_orig = _risk_profiles_by_state(samples_orig, alpha=alpha)

    top1s: List[float] = []
    spearmans: List[float] = []
    for _ in range(n_perm):
        perm = states.copy()
        perm.iloc[:] = rng.permutation(states.values)
        samples_p = _state_return_samples(returns, perm, n_states)
        prof_p = _risk_profiles_by_state(samples_p, alpha=alpha)
        m = _matched_ordering_metrics(prof_orig, prof_p)
        if math.isfinite(m["top1_consistency"]):
            top1s.append(m["top1_consistency"])
        if math.isfinite(m["spearman"]):
            spearmans.append(m["spearman"])

    return {
        "null_top1_mean": float(np.mean(top1s)) if top1s else float("nan"),
        "null_top1_p95": float(np.percentile(top1s, 95)) if top1s else float("nan"),
        "null_spearman_mean": float(np.mean(spearmans)) if spearmans else float("nan"),
        "null_spearman_p95": float(np.percentile(spearmans, 95)) if spearmans else float("nan"),
        "null_n": len(top1s),
    }


def _ordering_one_seed(
    model_name: str,
    seed: int,
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series,
    rep_names: List[str],
    k: int,
    window: int,
    rolls: List[str],
    alpha: float,
) -> Dict:
    """Compute ordering consistency for one (model, seed); parallelisable unit."""
    top_sum = 0.0; top_n = 0
    sp_sum = 0.0; sp_n = 0
    sign_sum = 0.0; sign_n = 0
    mean_abs_sum = 0.0; mean_abs_n = 0
    dvol_abs_sum = 0.0; dvol_abs_n = 0
    n_pairs = 0
    for roll in rolls:
        prof_by_rep: Dict[str, List[Dict[str, float]]] = {}
        for rep in rep_names:
            s = hard_map.get((rep, model_name, int(seed), str(roll)))
            if s is None:
                continue
            samples = _state_return_samples(returns, s, k)
            prof_by_rep[rep] = _risk_profiles_by_state(samples, alpha=alpha)
        for i in range(len(rep_names)):
            for j in range(i + 1, len(rep_names)):
                a, b = rep_names[i], rep_names[j]
                if a not in prof_by_rep or b not in prof_by_rep:
                    continue
                m = _matched_ordering_metrics(prof_by_rep[a], prof_by_rep[b])
                n_pairs += 1
                if math.isfinite(m["top1_consistency"]):
                    top_sum += m["top1_consistency"]; top_n += 1
                if math.isfinite(m["spearman"]):
                    sp_sum += m["spearman"]; sp_n += 1
                if math.isfinite(m["high_risk_mean_sign_consistency"]):
                    sign_sum += m["high_risk_mean_sign_consistency"]; sign_n += 1
                if math.isfinite(m["high_risk_mean_abs_diff"]):
                    mean_abs_sum += m["high_risk_mean_abs_diff"]; mean_abs_n += 1
                if math.isfinite(m["high_risk_downside_vol_abs_diff"]):
                    dvol_abs_sum += m["high_risk_downside_vol_abs_diff"]; dvol_abs_n += 1
    return {
        "kind": "cross_rep", "scope": "all_rep_pairs",
        "model": model_name, "K": int(k), "window": int(window),
        "seed": int(seed), "alpha": float(alpha),
        "top1_high_risk_consistency_mean": (top_sum / top_n) if top_n else float("nan"),
        "spearman_rank_consistency_mean": (sp_sum / sp_n) if sp_n else float("nan"),
        "high_risk_mean_sign_consistency_mean": (sign_sum / sign_n) if sign_n else float("nan"),
        "high_risk_mean_abs_diff_mean": (mean_abs_sum / mean_abs_n) if mean_abs_n else float("nan"),
        "high_risk_downside_vol_abs_diff_mean": (dvol_abs_sum / dvol_abs_n) if dvol_abs_n else float("nan"),
        "n_pairs": int(n_pairs), "n_top1": int(top_n), "n_spearman": int(sp_n),
        "n_sign": int(sign_n), "n_mean_abs": int(mean_abs_n), "n_dvol_abs": int(dvol_abs_n),
    }


def _compute_ordering_consistency_crossrep_seed_summary(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series,
    rep_names: List[str],
    models: List[str],
    k: int,
    window: int,
    seeds: List[int],
    rolls: List[str],
    alpha: float = 0.05,
    n_jobs: int = -1,
) -> List[Dict]:
    """
    Cross-representation ordering consistency (seed-level summaries).

    For each (model, seed), average Top-1 high-risk alignment and Spearman rank
    consistency across all rep pairs and rolls (after Hungarian matching).
    Parallelised over (model, seed) pairs via joblib.
    """
    from joblib import Parallel, delayed
    import time as _time

    tasks = [(m, s) for m in models for s in seeds]
    effective_jobs = min(n_jobs if n_jobs > 0 else 4, len(tasks))

    # Pre-shard hard_map so each worker only receives its own slice
    logger.info("Ordering cross-rep: sharding hard_map for %d tasks, n_jobs=%d", len(tasks), effective_jobs)
    shards = {}
    for m, s in tasks:
        shards[(m, s)] = {k: v for k, v in hard_map.items() if k[1] == m and k[2] == int(s)}
    logger.info("Ordering cross-rep: sharding done; shard sizes: %s",
                {k: len(v) for k, v in list(shards.items())[:3]})

    if effective_jobs <= 1:
        logger.info("Ordering cross-rep: running sequentially (effective_jobs=1)")
        return [_ordering_one_seed(m, s, shards[(m, s)], returns, rep_names, k, window, rolls, alpha)
                for m, s in tasks]

    args = [(m, s, shards[(m, s)], returns, rep_names, k, window, rolls, alpha) for m, s in tasks]
    return list(_parallel_or_sequential(_ordering_one_seed, args, effective_jobs, "Ordering cross-rep"))


def _ordering_independent_null_distribution(
    returns: pd.Series,
    states: pd.Series,
    n_states: int,
    n_perm: int = 500,
    seed: int = 43,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Ordering metrics under two independently generated random K-partitions.

    Empirically this null produces top-1 values of 0.78-0.96 across K=2..5
    rather than the 1/K we naively expected, because `_matched_ordering_metrics`
    matches states via Hungarian on |delta CVaR| + |delta Vol|: the matcher
    essentially forces rank-alignment of the two partitions' CVaR-extreme states
    regardless of whether the underlying partitions share meaningful ordering
    structure. Consequently this null cannot be interpreted as "symmetric chance
    baseline"; the Hungarian + CVaR combination is a constructive bias.

    See posthoc_rank_aligned_ordering.py for a metric with a well-defined 1/K
    null (relabel both sequences by within-partition CVaR rank, then compare
    pointwise) that does not share this defect.
    """
    rng = np.random.default_rng(seed)
    dates = states.dropna().index
    T = len(dates)

    top1s: List[float] = []
    spearmans: List[float] = []
    for _ in range(n_perm):
        labels_a = pd.Series(rng.integers(0, n_states, size=T), index=dates)
        labels_b = pd.Series(rng.integers(0, n_states, size=T), index=dates)
        samples_a = _state_return_samples(returns, labels_a, n_states)
        prof_a = _risk_profiles_by_state(samples_a, alpha=alpha)
        samples_b = _state_return_samples(returns, labels_b, n_states)
        prof_b = _risk_profiles_by_state(samples_b, alpha=alpha)
        m = _matched_ordering_metrics(prof_a, prof_b)
        if math.isfinite(m["top1_consistency"]):
            top1s.append(m["top1_consistency"])
        if math.isfinite(m["spearman"]):
            spearmans.append(m["spearman"])

    return {
        "indep_null_top1_mean": float(np.mean(top1s)) if top1s else float("nan"),
        "indep_null_top1_p95": float(np.percentile(top1s, 95)) if top1s else float("nan"),
        "indep_null_spearman_mean": float(np.mean(spearmans)) if spearmans else float("nan"),
        "indep_null_spearman_p95": float(np.percentile(spearmans, 95)) if spearmans else float("nan"),
        "indep_null_n": len(top1s),
    }


def _compute_ordering_null_baseline(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series,
    rep_names: List[str],
    models: List[str],
    k: int,
    seeds: List[int],
    rolls: List[str],
    n_perm: int = 500,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Compute chance-level ordering metrics (Hungarian-matched random permutation).

    Uses the first available (model, seed, roll, rep) combination to estimate
    the null distribution.  Returns mean and 95th percentile of top-1 agreement
    and Spearman under random label permutation.
    """
    # Find first available state sequence
    for model_name in models:
        for seed in seeds:
            for roll in rolls:
                for rep in rep_names:
                    s = hard_map.get((rep, model_name, int(seed), str(roll)))
                    if s is not None and len(s.dropna()) >= 20:
                        perm = _ordering_null_distribution(
                            returns, s, k, n_perm=n_perm, seed=42, alpha=alpha,
                        )
                        indep = _ordering_independent_null_distribution(
                            returns, s, k, n_perm=n_perm, seed=43, alpha=alpha,
                        )
                        result = {**perm, **indep}
                        logger.info(
                            "Ordering null baseline (K=%d): perm top1=%.3f indep top1=%.3f "
                            "perm sp=%.3f indep sp=%.3f (n_perm=%d)",
                            k,
                            result["null_top1_mean"],
                            result.get("indep_null_top1_mean", float("nan")),
                            result["null_spearman_mean"],
                            result.get("indep_null_spearman_mean", float("nan")),
                            n_perm,
                        )
                        return result
    return {
        "null_top1_mean": float("nan"),
        "null_top1_p95": float("nan"),
        "null_spearman_mean": float("nan"),
        "null_spearman_p95": float("nan"),
        "null_n": 0,
        "indep_null_top1_mean": float("nan"),
        "indep_null_top1_p95": float("nan"),
        "indep_null_spearman_mean": float("nan"),
        "indep_null_spearman_p95": float("nan"),
        "indep_null_n": 0,
    }


def _temporal_ordering_one_seed(
    model_name: str,
    seed: int,
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series,
    rep_names: List[str],
    k: int,
    window: int,
    rolls: List[str],
    alpha: float,
) -> List[Dict]:
    """Compute temporal ordering consistency for one (model, seed); parallelisable."""
    records: List[Dict] = []
    top_sum_all = 0.0; top_n_all = 0
    sp_sum_all = 0.0; sp_n_all = 0
    sign_sum_all = 0.0; sign_n_all = 0
    mean_abs_sum_all = 0.0; mean_abs_n_all = 0
    dvol_abs_sum_all = 0.0; dvol_abs_n_all = 0
    n_pairs_all = 0
    for rep in rep_names:
        top_sum = 0.0; top_n = 0; sp_sum = 0.0; sp_n = 0
        sign_sum = 0.0; sign_n = 0; mean_abs_sum = 0.0; mean_abs_n = 0
        dvol_abs_sum = 0.0; dvol_abs_n = 0; n_pairs = 0
        for i in range(len(rolls) - 1):
            roll_a, roll_b = str(rolls[i]), str(rolls[i + 1])
            a = hard_map.get((rep, model_name, int(seed), roll_a))
            b = hard_map.get((rep, model_name, int(seed), roll_b))
            if a is None or b is None:
                continue
            pa = _risk_profiles_by_state(_state_return_samples(returns, a, k), alpha=alpha)
            pb = _risk_profiles_by_state(_state_return_samples(returns, b, k), alpha=alpha)
            m = _matched_ordering_metrics(pa, pb)
            n_pairs += 1; n_pairs_all += 1
            if math.isfinite(m["top1_consistency"]):
                top_sum += m["top1_consistency"]; top_n += 1
                top_sum_all += m["top1_consistency"]; top_n_all += 1
            if math.isfinite(m["spearman"]):
                sp_sum += m["spearman"]; sp_n += 1
                sp_sum_all += m["spearman"]; sp_n_all += 1
            if math.isfinite(m["high_risk_mean_sign_consistency"]):
                sign_sum += m["high_risk_mean_sign_consistency"]; sign_n += 1
                sign_sum_all += m["high_risk_mean_sign_consistency"]; sign_n_all += 1
            if math.isfinite(m["high_risk_mean_abs_diff"]):
                mean_abs_sum += m["high_risk_mean_abs_diff"]; mean_abs_n += 1
                mean_abs_sum_all += m["high_risk_mean_abs_diff"]; mean_abs_n_all += 1
            if math.isfinite(m["high_risk_downside_vol_abs_diff"]):
                dvol_abs_sum += m["high_risk_downside_vol_abs_diff"]; dvol_abs_n += 1
                dvol_abs_sum_all += m["high_risk_downside_vol_abs_diff"]; dvol_abs_n_all += 1
        records.append({
            "kind": "temporal", "scope": f"rep={rep}", "model": model_name,
            "K": int(k), "window": int(window), "seed": int(seed), "alpha": float(alpha),
            "top1_high_risk_consistency_mean": (top_sum / top_n) if top_n else float("nan"),
            "spearman_rank_consistency_mean": (sp_sum / sp_n) if sp_n else float("nan"),
            "high_risk_mean_sign_consistency_mean": (sign_sum / sign_n) if sign_n else float("nan"),
            "high_risk_mean_abs_diff_mean": (mean_abs_sum / mean_abs_n) if mean_abs_n else float("nan"),
            "high_risk_downside_vol_abs_diff_mean": (dvol_abs_sum / dvol_abs_n) if dvol_abs_n else float("nan"),
            "n_pairs": int(n_pairs), "n_top1": int(top_n), "n_spearman": int(sp_n),
            "n_sign": int(sign_n), "n_mean_abs": int(mean_abs_n), "n_dvol_abs": int(dvol_abs_n),
        })
    records.append({
        "kind": "temporal", "scope": "all_reps", "model": model_name,
        "K": int(k), "window": int(window), "seed": int(seed), "alpha": float(alpha),
        "top1_high_risk_consistency_mean": (top_sum_all / top_n_all) if top_n_all else float("nan"),
        "spearman_rank_consistency_mean": (sp_sum_all / sp_n_all) if sp_n_all else float("nan"),
        "high_risk_mean_sign_consistency_mean": (sign_sum_all / sign_n_all) if sign_n_all else float("nan"),
        "high_risk_mean_abs_diff_mean": (mean_abs_sum_all / mean_abs_n_all) if mean_abs_n_all else float("nan"),
        "high_risk_downside_vol_abs_diff_mean": (dvol_abs_sum_all / dvol_abs_n_all) if dvol_abs_n_all else float("nan"),
        "n_pairs": int(n_pairs_all), "n_top1": int(top_n_all), "n_spearman": int(sp_n_all),
        "n_sign": int(sign_n_all), "n_mean_abs": int(mean_abs_n_all), "n_dvol_abs": int(dvol_abs_n_all),
    })
    return records


def _compute_ordering_consistency_temporal_seed_summary(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series,
    rep_names: List[str],
    models: List[str],
    k: int,
    window: int,
    seeds: List[int],
    rolls: List[str],
    alpha: float = 0.05,
    n_jobs: int = -1,
) -> List[Dict]:
    """
    Temporal ordering consistency (seed-level summaries).

    For each (rep, model, seed) and consecutive (roll_a, roll_b), compute ordering
    metrics after Hungarian matching, then average across roll pairs.
    Parallelised over (model, seed) pairs via joblib.
    """
    from joblib import Parallel, delayed

    tasks = [(m, s) for m in models for s in seeds]
    effective_jobs = min(n_jobs if n_jobs > 0 else 4, len(tasks))

    shards = {}
    for m, s in tasks:
        shards[(m, s)] = {k: v for k, v in hard_map.items() if k[1] == m and k[2] == int(s)}

    if effective_jobs <= 1:
        return [rec for m, s in tasks
                for rec in _temporal_ordering_one_seed(m, s, shards[(m, s)], returns, rep_names, k, window, rolls, alpha)]

    args = [(m, s, shards[(m, s)], returns, rep_names, k, window, rolls, alpha) for m, s in tasks]
    nested = _parallel_or_sequential(_temporal_ordering_one_seed, args, effective_jobs, "Ordering temporal")
    return [rec for batch in nested for rec in batch]

