"""
Cross-representation stability, semantic Wasserstein, and risk profiles.

Functions:
  _load_hard_map_from_rep_csv               load saved hard state CSVs
  _rep_stability_one_seed                   ARI per (rep, model, seed)
  _compute_rep_stability_from_map           cross-rep stability summary
  _window_stability_one_unit                per-window temporal stability
  _compute_window_stability_from_map        window-stability summary
  _state_return_samples                     state-conditional return samples
  _matched_wasserstein_cost                 W_2 cost on matched states
  _semantic_crossrep_one_seed               semantic cross-rep Wasserstein
  _compute_semantic_crossrep_wasserstein_from_map
  _semantic_temporal_one_unit               semantic temporal Wasserstein
  _compute_semantic_temporal_wasserstein_from_map
  _risk_profile_from_returns                per-state risk profile
  _risk_profiles_by_state                   risk profiles for K states
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

from src.core.metrics import semantic_drift, stability_metrics, temporal_disjoint_metrics
from src.core.parallel import _parallel_or_sequential
from src.core.utils import safe_name

logger = logging.getLogger(__name__)




def _load_hard_map_from_rep_csv(
    rep_dir: Path,
    rep_name: str,
) -> Dict[Tuple[str, str, int, str], pd.Series]:
    """
    Load hard state sequences from `windows_states_hard.csv` into a map keyed by
    (rep, model, seed, roll) -> pd.Series(date->state).
    """

    rep_dir = Path(rep_dir)
    files = sorted(rep_dir.glob("windows_states_hard_*.csv"))
    # Backward compatibility (older runs)
    legacy = rep_dir / "windows_states_hard.csv"
    if legacy.exists():
        files.append(legacy)
    if not files:
        return {}

    out: Dict[Tuple[str, str, int, str], pd.Series] = {}
    for p in files:
        df = pd.read_csv(p, parse_dates=["date"])
        if df.empty:
            continue
        for (model, seed, roll), g in df.groupby(["model", "seed", "roll"], sort=False):
            # Guard against duplicate timestamps within a (model, seed, roll) group,
            # which can happen if shard merges were duplicated or interrupted.
            g = g.sort_values("date")
            if g["date"].duplicated().any():
                n_dup = int(g["date"].duplicated().sum())
                logger.warning(
                    "Duplicate dates in hard states CSV; dropping duplicates. rep=%s model=%s seed=%s roll=%s file=%s n_dup=%d",
                    rep_name,
                    model,
                    seed,
                    roll,
                    p,
                    n_dup,
                )
                g = g.drop_duplicates(subset=["date"], keep="last")
            s = pd.Series(g["state"].values, index=g["date"].values, dtype="Int64")
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            out[(rep_name, str(model), int(seed), str(roll))] = s
    return out


def _rep_stability_one_seed(
    model_name: str, seed: int,
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    rep_names: List[str], k: int, window: int, rolls: List[str],
) -> List[Dict]:
    records: List[Dict] = []
    for roll in rolls:
        series: Dict[str, pd.Series] = {}
        for rep in rep_names:
            s = hard_map.get((rep, model_name, int(seed), roll))
            if s is not None:
                series[rep] = s
        for i in range(len(rep_names)):
            for j in range(i + 1, len(rep_names)):
                a, b = rep_names[i], rep_names[j]
                if a not in series or b not in series:
                    continue
                scores = stability_metrics(series[a], series[b], k)
                records.append({
                    "rep_a": a, "rep_b": b, "model": model_name,
                    "K": k, "window": window, "seed": seed, "roll": roll,
                    "ari": scores.ari, "nmi": scores.nmi, "ami": scores.ami, "vi": scores.vi,
                })
    return records


def _compute_rep_stability_from_map(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    rep_names: List[str],
    models: List[str],
    k: int,
    window: int,
    seeds: List[int],
    rolls: List[str],
    n_jobs: int = -1,
) -> List[Dict]:
    from joblib import Parallel, delayed
    import time as _time

    tasks = [(m, s) for m in models for s in seeds]
    effective_jobs = min(n_jobs if n_jobs > 0 else 4, len(tasks))

    logger.info("Rep stability: %d tasks, n_jobs=%d", len(tasks), effective_jobs)
    t0 = _time.perf_counter()

    shards = {}
    for m, s in tasks:
        shards[(m, s)] = {key: v for key, v in hard_map.items() if key[1] == m and key[2] == int(s)}

    if effective_jobs <= 1:
        all_recs = []
        for m, s in tasks:
            all_recs.extend(_rep_stability_one_seed(m, s, shards[(m, s)], rep_names, k, window, rolls))
        logger.info("Rep stability: done in %.1fs (sequential)", _time.perf_counter() - t0)
        return all_recs

    args = [(m, s, shards[(m, s)], rep_names, k, window, rolls) for m, s in tasks]
    nested = _parallel_or_sequential(_rep_stability_one_seed, args, effective_jobs, "Rep stability")
    return [rec for batch in nested for rec in batch]


def _window_stability_one_unit(
    rep: str, model_name: str, seed: int,
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    k: int, window: int, rolls: List[str],
) -> List[Dict]:
    records: List[Dict] = []
    for i in range(len(rolls) - 1):
        roll_a, roll_b = rolls[i], rolls[i + 1]
        a = hard_map.get((rep, model_name, int(seed), roll_a))
        b = hard_map.get((rep, model_name, int(seed), roll_b))
        if a is None or b is None:
            continue
        scores_overlap = stability_metrics(a, b, k)
        scores_disjoint = temporal_disjoint_metrics(a, b)
        idx_a = a.dropna().index
        idx_b = b.dropna().index
        n_overlap = int(len(idx_a.intersection(idx_b)))
        n_union = int(len(idx_a.union(idx_b)))
        overlap_ratio = float(n_overlap / n_union) if n_union > 0 else float("nan")
        scores = scores_disjoint
        records.append({
            "rep": rep, "model": model_name, "K": k, "window": window,
            "seed": seed, "roll_a": roll_a, "roll_b": roll_b,
            "temporal_eval_mode": "disjoint",
            "overlap_n": n_overlap, "overlap_ratio": overlap_ratio,
            "ari": scores.ari, "nmi": scores.nmi, "ami": scores.ami, "vi": scores.vi,
            "ari_overlap": scores_overlap.ari, "nmi_overlap": scores_overlap.nmi,
            "ami_overlap": scores_overlap.ami, "vi_overlap": scores_overlap.vi,
            "ari_disjoint": scores_disjoint.ari, "nmi_disjoint": scores_disjoint.nmi,
            "ami_disjoint": scores_disjoint.ami, "vi_disjoint": scores_disjoint.vi,
        })
    return records


def _compute_window_stability_from_map(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    rep_names: List[str],
    models: List[str],
    k: int,
    window: int,
    seeds: List[int],
    rolls: List[str],
    n_jobs: int = -1,
) -> List[Dict]:
    from joblib import Parallel, delayed
    import time as _time

    tasks = [(rep, m, s) for rep in rep_names for m in models for s in seeds]
    effective_jobs = min(n_jobs if n_jobs > 0 else 4, len(tasks))

    logger.info("Window stability: %d tasks, n_jobs=%d", len(tasks), effective_jobs)
    t0 = _time.perf_counter()

    shards = {}
    for rep, m, s in tasks:
        shards[(rep, m, s)] = {key: v for key, v in hard_map.items()
                               if key[0] == rep and key[1] == m and key[2] == int(s)}

    if effective_jobs <= 1:
        all_recs = []
        for rep, m, s in tasks:
            all_recs.extend(_window_stability_one_unit(rep, m, s, shards[(rep, m, s)], k, window, rolls))
        logger.info("Window stability: done in %.1fs (sequential)", _time.perf_counter() - t0)
        return all_recs

    args = [(rep, m, s, shards[(rep, m, s)], k, window, rolls) for rep, m, s in tasks]
    nested = _parallel_or_sequential(_window_stability_one_unit, args, effective_jobs, "Window stability")
    return [rec for batch in nested for rec in batch]


def _state_return_samples(
    returns: pd.Series, states: pd.Series, n_states: int
) -> List[np.ndarray]:
    """
    Build per-state return samples (1D) on the intersection of indices.

    Returns a list of length n_states; each entry is a float ndarray (possibly empty).
    """
    idx = states.dropna().index.intersection(returns.dropna().index)
    if len(idx) == 0:
        return [np.array([], dtype=float) for _ in range(int(n_states))]
    r = returns.loc[idx].astype(float)
    z = states.loc[idx].astype(int)
    out: List[np.ndarray] = []
    for s in range(int(n_states)):
        out.append(r[z == s].values.astype(float))
    return out


def _matched_wasserstein_cost(
    samples_a: List[np.ndarray], samples_b: List[np.ndarray]
) -> Tuple[float, int, int]:
    """
    Minimum average 1D Wasserstein cost after Hungarian matching.

    Empty-sample cells incur a large penalty so well-defined states match first.
    A record where the penalty participates in the Hungarian assignment produces
    a strongly inflated cost that can dominate downstream averaging (e.g. a K=3
    window where a tail state has zero members).

    Returns:
        (cost, n_matched_finite, k):
            cost: average matched cost including any penalty cells
            n_matched_finite: number of Hungarian-matched pairs whose cost was
                computable (i.e. NOT penalty-filled); callers should filter or
                flag records where n_matched_finite < k
            k: min(len(samples_a), len(samples_b))
    """
    k = int(min(len(samples_a), len(samples_b)))
    if k <= 0:
        return float("nan"), 0, 0
    cost = np.full((k, k), np.nan, dtype=float)
    for i in range(k):
        for j in range(k):
            a = samples_a[i]
            b = samples_b[j]
            if a.size == 0 or b.size == 0:
                continue
            cost[i, j] = float(wasserstein_distance(a, b))

    # If everything is NaN, cost is undefined.
    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        return float("nan"), 0, k

    penalty = float(finite.max()) * 10.0 + 1.0
    cost_filled = np.where(np.isfinite(cost), cost, penalty)
    row_ind, col_ind = linear_sum_assignment(cost_filled)
    matched = cost_filled[row_ind, col_ind]
    matched_from_finite = np.isfinite(cost[row_ind, col_ind])
    n_finite = int(np.sum(matched_from_finite))
    cost_mean = float(np.mean(matched)) if matched.size else float("nan")
    return cost_mean, n_finite, k


def _semantic_crossrep_one_seed(
    model_name: str, seed: int,
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series, rep_names: List[str],
    k: int, window: int, rolls: List[str],
) -> List[Dict]:
    records: List[Dict] = []
    for roll in rolls:
        samples_by_rep: Dict[str, List[np.ndarray]] = {}
        for rep in rep_names:
            s = hard_map.get((rep, model_name, int(seed), roll))
            if s is None:
                continue
            samples_by_rep[rep] = _state_return_samples(returns, s, k)
        for i in range(len(rep_names)):
            for j in range(i + 1, len(rep_names)):
                a, b = rep_names[i], rep_names[j]
                if a not in samples_by_rep or b not in samples_by_rep:
                    continue
                v, n_finite, k_total = _matched_wasserstein_cost(samples_by_rep[a], samples_by_rep[b])
                records.append({
                    "kind": "cross_rep", "rep_a": a, "rep_b": b,
                    "model": model_name, "K": k, "window": window,
                    "seed": int(seed), "roll": str(roll), "wasserstein": v,
                    "wasserstein_n_finite": n_finite, "wasserstein_k": k_total,
                })
    return records


def _compute_semantic_crossrep_wasserstein_from_map(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series,
    rep_names: List[str],
    models: List[str],
    k: int,
    window: int,
    seeds: List[int],
    rolls: List[str],
    n_jobs: int = -1,
) -> List[Dict]:
    from joblib import Parallel, delayed
    import time as _time

    tasks = [(m, s) for m in models for s in seeds]
    effective_jobs = min(n_jobs if n_jobs > 0 else 4, len(tasks))
    logger.info("Semantic cross-rep: %d tasks, n_jobs=%d", len(tasks), effective_jobs)
    t0 = _time.perf_counter()

    shards = {(m, s): {key: v for key, v in hard_map.items() if key[1] == m and key[2] == int(s)}
              for m, s in tasks}

    if effective_jobs <= 1:
        recs = [r for m, s in tasks for r in _semantic_crossrep_one_seed(m, s, shards[(m, s)], returns, rep_names, k, window, rolls)]
        logger.info("Semantic cross-rep: done in %.1fs", _time.perf_counter() - t0)
        return recs

    args = [(m, s, shards[(m, s)], returns, rep_names, k, window, rolls) for m, s in tasks]
    nested = _parallel_or_sequential(_semantic_crossrep_one_seed, args, effective_jobs, "Semantic cross-rep")
    return [r for batch in nested for r in batch]


def _semantic_temporal_one_unit(
    rep: str, model_name: str, seed: int,
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series, k: int, window: int, rolls: List[str],
) -> List[Dict]:
    records: List[Dict] = []
    for i in range(len(rolls) - 1):
        roll_a, roll_b = rolls[i], rolls[i + 1]
        a = hard_map.get((rep, model_name, int(seed), roll_a))
        b = hard_map.get((rep, model_name, int(seed), roll_b))
        if a is None or b is None:
            continue
        sa = _state_return_samples(returns, a, k)
        sb = _state_return_samples(returns, b, k)
        v, n_finite, k_total = _matched_wasserstein_cost(sa, sb)
        records.append({
            "kind": "temporal", "rep": rep, "model": model_name,
            "K": k, "window": window, "seed": int(seed),
            "roll_a": str(roll_a), "roll_b": str(roll_b), "wasserstein": v,
            "wasserstein_n_finite": n_finite, "wasserstein_k": k_total,
        })
    return records


def _compute_semantic_temporal_wasserstein_from_map(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series,
    rep_names: List[str],
    models: List[str],
    k: int,
    window: int,
    seeds: List[int],
    rolls: List[str],
    n_jobs: int = -1,
) -> List[Dict]:
    from joblib import Parallel, delayed
    import time as _time

    tasks = [(rep, m, s) for rep in rep_names for m in models for s in seeds]
    effective_jobs = min(n_jobs if n_jobs > 0 else 4, len(tasks))
    logger.info("Semantic temporal: %d tasks, n_jobs=%d", len(tasks), effective_jobs)
    t0 = _time.perf_counter()

    shards = {(rep, m, s): {key: v for key, v in hard_map.items()
                            if key[0] == rep and key[1] == m and key[2] == int(s)}
              for rep, m, s in tasks}

    if effective_jobs <= 1:
        recs = [r for rep, m, s in tasks for r in _semantic_temporal_one_unit(rep, m, s, shards[(rep, m, s)], returns, k, window, rolls)]
        logger.info("Semantic temporal: done in %.1fs", _time.perf_counter() - t0)
        return recs

    args = [(rep, m, s, shards[(rep, m, s)], returns, k, window, rolls) for rep, m, s in tasks]
    nested = _parallel_or_sequential(_semantic_temporal_one_unit, args, effective_jobs, "Semantic temporal")
    logger.info("Semantic temporal: done in %.1fs", _time.perf_counter() - t0)
    return [r for batch in nested for r in batch]


def _risk_profile_from_returns(x: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """
    Compute a simple state risk profile from 1D return samples.

    Returns are log-returns. Risk metrics are computed on the left tail (loss side).
    - var_alpha: empirical alpha-quantile (typically negative)
    - cvar_alpha: mean of returns in the left tail (<= var_alpha)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "vol": float("nan"),
            "downside_vol": float("nan"),
            "var_alpha": float("nan"),
            "cvar_alpha": float("nan"),
        }
    mean = float(np.mean(x))
    vol = float(np.std(x, ddof=1)) if n > 1 else 0.0
    neg = x[x < 0.0]
    downside_vol = float(np.std(neg, ddof=1)) if neg.size > 1 else (0.0 if neg.size == 1 else float("nan"))
    var_a = float(np.quantile(x, float(alpha)))
    tail = x[x <= var_a]
    cvar_a = float(np.mean(tail)) if tail.size > 0 else var_a
    return {
        "n": float(n),
        "mean": mean,
        "vol": vol,
        "downside_vol": downside_vol,
        "var_alpha": var_a,
        "cvar_alpha": cvar_a,
    }


def _risk_profiles_by_state(
    samples: List[np.ndarray], alpha: float = 0.05
) -> List[Dict[str, float]]:
    """Compute risk profiles for each state (list index = state id)."""
    out: List[Dict[str, float]] = []
    for s in samples:
        out.append(_risk_profile_from_returns(s, alpha=alpha))
    return out

