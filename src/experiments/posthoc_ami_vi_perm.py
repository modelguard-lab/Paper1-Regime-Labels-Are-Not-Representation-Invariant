"""
Post-hoc computation of AMI, VI, and permutation p-values from existing outputs.

Reads windows_states_hard_*.csv files already on disk, recomputes cross-rep and
temporal stability (temporal uses disjoint-window segments to avoid overlap inflation)
with the new metrics (AMI, VI), runs aggregate permutation tests,
and appends the results to each asset/step key_results.csv.

Run from the project root:
    python src/posthoc_ami_vi_perm.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
)
from scipy.stats import entropy as _scipy_entropy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# Project root is two levels above this file (src/experiments/<this>.py).
ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"
# REPS and MODELS are intentionally not defined at module level: they must be
# sourced from config.yaml via _resolve_run_defs(cfg). Hardcoded module-level
# lists drift silently from config (e.g., rep_e was once missing), dropping
# representations from every downstream aggregate.
K_DEFAULT = 3
N_PERM = 99  # min p = 1/100 = 0.01 < 0.05; sufficient for paper's p<0.05 claim
PERM_SEED = 42
PERM_MAX_PAIRS = 2000  # subsample for speed; all pairs have p << 0.05 trivially
# CBB null: Politis-Romano (1992) circular block bootstrap. Block length is
# selected per-pair via Patton-Politis-White (2009) optimal length, with a
# T**(1/3) fallback when the selector is unavailable or fails on degenerate
# (single-state) sequences.
CBB_FALLBACK_EXP = 1.0 / 3.0
BH_Q_LEVELS = (0.05, 0.10)  # Benjamini-Hochberg significance levels reported


def _parallel_or_sequential(func, args_list, n_jobs, label="parallel"):
    """Run func(*args) for each args in args_list via joblib, with sequential fallback.

    Mirrors the same helper in runner.py: loky workers can be killed on
    Windows by MKL segfaults, OS OOM, or BrokenProcessPool under nested spawn.
    On failure we log the traceback and retry sequentially so the post-hoc
    pipeline still produces output instead of crashing out.
    """
    import traceback as _tb
    import time as _t

    from joblib import Parallel, delayed
    from joblib.externals.loky.process_executor import TerminatedWorkerError

    n_tasks = len(args_list)
    if n_jobs is None or n_jobs <= 1 or n_tasks <= 1:
        t0 = _t.perf_counter()
        results = [func(*a) for a in args_list]
        logger.info("%s: done in %.1fs (sequential, %d tasks)", label, _t.perf_counter() - t0, n_tasks)
        return results
    try:
        t0 = _t.perf_counter()
        results = Parallel(n_jobs=min(n_jobs, n_tasks), backend="loky")(
            delayed(func)(*a) for a in args_list
        )
        logger.info("%s: done in %.1fs (loky, n_jobs=%d, %d tasks)", label, _t.perf_counter() - t0, min(n_jobs, n_tasks), n_tasks)
        return results
    except TerminatedWorkerError as e:
        logger.warning("%s: loky worker died (%s); falling back to sequential.", label, e)
    except Exception as e:
        logger.error("%s: parallel error (%s); falling back to sequential.\n%s", label, e, _tb.format_exc())
    t0 = _t.perf_counter()
    results = [func(*a) for a in args_list]
    logger.info("%s: done in %.1fs (sequential fallback, %d tasks)", label, _t.perf_counter() - t0, n_tasks)
    return results


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def variation_of_information(a: np.ndarray, b: np.ndarray) -> float:
    """Variation of Information = H(X|Y) + H(Y|X), in nats."""
    if len(a) == 0:
        return float("nan")
    mi = float(mutual_info_score(a, b))
    _, ca = np.unique(a, return_counts=True)
    _, cb = np.unique(b, return_counts=True)
    h_a = float(_scipy_entropy(ca))
    h_b = float(_scipy_entropy(cb))
    return h_a + h_b - 2.0 * mi


def align_and_ari(a: np.ndarray, b: np.ndarray) -> float:
    return float(adjusted_rand_score(a, b))


def permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
) -> float:
    """
    One-sided permutation p-value: P(ARI_perm >= ARI_obs) under H0 of independence.

    Uses the observed ARI as test statistic and permutes b.
    """
    rng = np.random.default_rng(seed)
    observed = float(adjusted_rand_score(a, b))
    count = 0
    for _ in range(n_perm):
        b_perm = rng.permutation(b)
        if float(adjusted_rand_score(a, b_perm)) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hard_states(
    asset_dir: Path, step: Optional[int], rep: str, model: str
) -> Optional[pd.DataFrame]:
    """Load windows_states_hard_<model>.csv for step-layout or root-layout outputs."""
    candidates: List[Path] = []
    if step is not None:
        candidates.append(
            asset_dir / f"step_{step}" / "results" / rep / f"windows_states_hard_{model}.csv"
        )
    candidates.append(asset_dir / "results" / rep / f"windows_states_hard_{model}.csv")
    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_hard_map(
    asset_dir: Path,
    step: Optional[int],
    reps: List[str],
    models: List[str],
) -> Dict[Tuple[str, str, int, str], pd.Series]:
    """
    Build {(rep, model, seed, roll): state_series} from disk.
    state_series is indexed by date.
    """
    hard_map: Dict[Tuple[str, str, int, str], pd.Series] = {}
    for rep in reps:
        for model in models:
            df = load_hard_states(asset_dir, step, rep, model)
            if df is None:
                continue
            for (seed, roll), grp in df.groupby(["seed", "roll"]):
                s = grp.set_index("date")["state"].astype(int).sort_index()
                # Drop duplicate dates (keep last, consistent with runner)
                s = s[~s.index.duplicated(keep="last")]
                hard_map[(rep, model, int(seed), str(roll))] = s
    return hard_map


# ---------------------------------------------------------------------------
# Cross-rep and temporal metric computation
# ---------------------------------------------------------------------------

def compute_pair_metrics(
    s_a: pd.Series, s_b: pd.Series, mode: str = "intersection"
) -> Dict[str, float]:
    """
    Compute ARI, AMI, VI for a pair of state series.

    mode="intersection": compare on common timestamps.
    mode="disjoint": compare non-overlapping segments only (ordered pairing).
    """
    if mode == "disjoint":
        idx_a_only = s_a.index.difference(s_b.index)
        idx_b_only = s_b.index.difference(s_a.index)
        if len(idx_a_only) < 2 or len(idx_b_only) < 2:
            return {"ari": float("nan"), "ami": float("nan"), "vi": float("nan")}
        a_raw = s_a.loc[idx_a_only].values.astype(int)
        b_raw = s_b.loc[idx_b_only].values.astype(int)
        n = int(min(len(a_raw), len(b_raw)))
        if n < 2:
            return {"ari": float("nan"), "ami": float("nan"), "vi": float("nan")}
        a = a_raw[:n]
        b = b_raw[:n]
    else:
        idx = s_a.index.intersection(s_b.index)
        if len(idx) < 2:
            return {"ari": float("nan"), "ami": float("nan"), "vi": float("nan")}
        a = s_a.loc[idx].values.astype(int)
        b = s_b.loc[idx].values.astype(int)
    return {
        "ari": float(adjusted_rand_score(a, b)),
        "ami": float(adjusted_mutual_info_score(a, b)),
        "vi": float(variation_of_information(a, b)),
    }


def _cross_rep_one_seed(
    model: str, seed: int,
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    reps: List[str], rolls: List[str],
) -> List[Dict]:
    records = []
    for roll in rolls:
        series = {
            rep: hard_map[(rep, model, seed, roll)]
            for rep in reps
            if (rep, model, seed, roll) in hard_map
        }
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                a_name, b_name = reps[i], reps[j]
                if a_name not in series or b_name not in series:
                    continue
                m = compute_pair_metrics(series[a_name], series[b_name])
                records.append({"rep_a": a_name, "rep_b": b_name, "model": model,
                                "seed": seed, "roll": roll, **m})
    return records


def cross_rep_records(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    reps: List[str],
    models: List[str],
    seeds: List[int],
    rolls: List[str],
    n_jobs: int = 18,
) -> List[Dict]:
    tasks = [(m, s) for m in models for s in seeds]
    shards = {(m, s): {k: v for k, v in hard_map.items() if k[1] == m and k[2] == s}
              for m, s in tasks}
    logger.info("  Cross-rep AMI/VI: %d tasks, n_jobs=%d", len(tasks), n_jobs)
    args = [(m, s, shards[(m, s)], reps, rolls) for m, s in tasks]
    nested = _parallel_or_sequential(_cross_rep_one_seed, args, n_jobs, "Cross-rep AMI/VI")
    return [r for batch in nested for r in batch]


def _temporal_one_unit(
    rep: str, model: str, seed: int,
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    rolls: List[str],
) -> List[Dict]:
    records = []
    for i in range(len(rolls) - 1):
        roll_a, roll_b = rolls[i], rolls[i + 1]
        sa = hard_map.get((rep, model, seed, roll_a))
        sb = hard_map.get((rep, model, seed, roll_b))
        if sa is None or sb is None:
            continue
        m = compute_pair_metrics(sa, sb, mode="disjoint")
        records.append({"rep": rep, "model": model, "seed": seed,
                        "roll_a": roll_a, "roll_b": roll_b,
                        "temporal_eval_mode": "disjoint", **m})
    return records


def temporal_records(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    reps: List[str],
    models: List[str],
    seeds: List[int],
    rolls: List[str],
    n_jobs: int = 18,
) -> List[Dict]:
    tasks = [(rep, m, s) for rep in reps for m in models for s in seeds]
    shards = {(rep, m, s): {k: v for k, v in hard_map.items() if k[0] == rep and k[1] == m and k[2] == s}
              for rep, m, s in tasks}
    logger.info("  Temporal AMI/VI: %d tasks, n_jobs=%d", len(tasks), n_jobs)
    args = [(rep, m, s, shards[(rep, m, s)], rolls) for rep, m, s in tasks]
    nested = _parallel_or_sequential(_temporal_one_unit, args, n_jobs, "Temporal AMI/VI")
    return [r for batch in nested for r in batch]


# ---------------------------------------------------------------------------
# Aggregate permutation test
# ---------------------------------------------------------------------------

def _perm_one_pair(
    a: np.ndarray, b: np.ndarray, n_perm: int, seed: int,
) -> Dict[str, float]:
    """Permutation test for one pair — parallelisable unit."""
    rng = np.random.default_rng(seed)
    obs = float(adjusted_rand_score(a, b))
    exceed = sum(
        1 for _ in range(n_perm)
        if float(adjusted_rand_score(a, rng.permutation(b))) >= obs
    )
    pval = (exceed + 1) / (n_perm + 1)
    return {"obs": obs, "exceed": exceed, "pval": pval}


# ---------------------------------------------------------------------------
# Politis-Romano circular block bootstrap null
# ---------------------------------------------------------------------------

def _ppw_block_length(x: np.ndarray) -> int:
    """Patton-Politis-White (2009) optimal circular block length.

    Falls back to ``ceil(T**(1/3))`` when the ``arch`` selector is unavailable
    or fails (e.g., on a constant single-state sequence). Lower-bounded at 2.
    """
    T = int(len(x))
    fallback = max(2, int(np.ceil(T ** CBB_FALLBACK_EXP)))
    if T < 8:
        return fallback
    try:
        from arch.bootstrap import optimal_block_length
        # PPW is undefined on a constant series; short-circuit.
        if np.unique(x).size <= 1:
            return fallback
        df = optimal_block_length(np.asarray(x, dtype=float))
        b = float(df["circular"].iloc[0])
        if not np.isfinite(b) or b < 1.0:
            return fallback
        return max(2, min(int(np.ceil(b)), max(2, T // 2)))
    except Exception:
        return fallback


def _cbb_resample(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Single circular block bootstrap sample of length ``len(x)``.

    Politis-Romano (1992): random block start, fixed block length, circular
    indexing so the sequence wraps around. Preserves within-series
    autocorrelation; randomises alignment with the partner labelling.
    """
    T = int(len(x))
    if T == 0:
        return x.copy()
    block = max(1, min(int(block), T))
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    out = np.empty(T, dtype=x.dtype)
    pos = 0
    for s in starts:
        take = min(block, T - pos)
        idx = (int(s) + np.arange(take)) % T
        out[pos:pos + take] = x[idx]
        pos += take
        if pos >= T:
            break
    return out


def _cbb_one_pair(
    a: np.ndarray, b: np.ndarray, n_perm: int, seed: int,
) -> Dict[str, float]:
    """CBB null for one pair: preserve b's autocorrelation, randomise alignment.

    The block length is chosen per-pair from b's PPW optimum (a is held fixed).
    Returns the observed ARI, exceedance count, p-value, and block length used.
    """
    rng = np.random.default_rng(seed)
    obs = float(adjusted_rand_score(a, b))
    block = _ppw_block_length(b)
    exceed = sum(
        1 for _ in range(n_perm)
        if float(adjusted_rand_score(a, _cbb_resample(b, block, rng))) >= obs
    )
    pval = (exceed + 1) / (n_perm + 1)
    return {"obs": obs, "exceed": exceed, "pval": pval, "block": float(block)}


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

def _bh_fdr_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg (1995) step-up adjusted p-values (q-values).

    Returns the BH-adjusted p-values in the original input order, monotonised
    from largest to smallest and clipped to [0, 1]. Compare against the chosen
    q-level to identify discoveries (e.g., ``q < 0.05``).
    """
    p = np.asarray(pvals, dtype=float).ravel()
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1, dtype=float)
    # enforce monotonicity from largest to smallest (BH step-up)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def aggregate_permutation_pvalue(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    reps: List[str],
    models: List[str],
    seeds: List[int],
    rolls: List[str],
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
    n_sample_pairs: Optional[int] = None,
    n_jobs: int = 18,
) -> Dict[str, float]:
    """
    Efficient one-sided permutation p-value: H0: cross-rep ARI <= 0 (random labelling).

    Parallelised over pairs via joblib.
    """
    rng = np.random.default_rng(seed)

    # Collect all valid (key_a, key_b) pair candidates
    candidates = []
    for model in models:
        for s in seeds:
            for roll in rolls:
                for i in range(len(reps)):
                    for j in range(i + 1, len(reps)):
                        ka = (reps[i], model, s, roll)
                        kb = (reps[j], model, s, roll)
                        if ka in hard_map and kb in hard_map:
                            candidates.append((ka, kb))

    if not candidates:
        return {"pvalue_all": float("nan")}

    if n_sample_pairs is None:
        chosen = list(candidates)
    else:
        chosen_idx = rng.choice(
            len(candidates),
            size=min(int(n_sample_pairs), len(candidates)),
            replace=False,
        )
        chosen = [candidates[i] for i in chosen_idx]

    # Prepare arrays (avoid pickling hard_map)
    pair_data = []
    for ka, kb in chosen:
        sa = hard_map[ka]
        sb = hard_map[kb]
        idx = sa.index.intersection(sb.index)
        if len(idx) < 10:
            continue
        a = sa.loc[idx].values.astype(int)
        b = sb.loc[idx].values.astype(int)
        pair_data.append((a, b))

    if not pair_data:
        return {"pvalue_all": float("nan")}

    logger.info("  Permutation test: %d pairs, n_perm=%d, n_jobs=%d", len(pair_data), n_perm, n_jobs)

    # Parallel permutation tests — each pair gets a unique seed
    perm_args = [(a, b, n_perm, seed + i) for i, (a, b) in enumerate(pair_data)]
    results = _parallel_or_sequential(_perm_one_pair, perm_args, n_jobs, "Permutation test")

    total_exceed = sum(r["exceed"] + 1 for r in results)
    total_trials = len(results) * (n_perm + 1)
    pvalue = total_exceed / total_trials
    obs_aris = [r["obs"] for r in results]
    mean_obs = float(np.mean(obs_aris))
    per_pair_pvals = [r["pval"] for r in results]

    logger.info(
        "  Permutation test: mean_obs_ARI=%.4f, aggregate p-value=%.6f "
        "(n_pairs=%d, n_perm=%d each)",
        mean_obs, pvalue, len(obs_aris), n_perm,
    )
    if per_pair_pvals:
        logger.info(
            "  Per-pair p-values: min=%.4f, median=%.4f, max=%.4f (n=%d)",
            float(np.min(per_pair_pvals)), float(np.median(per_pair_pvals)),
            float(np.max(per_pair_pvals)), len(per_pair_pvals),
        )

    # --- Politis-Romano circular block bootstrap null on the same pairs ---
    # PPW-selected block length per pair preserves within-series autocorrelation,
    # addressing the "structural inflation" of the iid permutation null on
    # rolling-window label sequences.
    cbb_args = [(a, b, n_perm, seed + 100_000 + i) for i, (a, b) in enumerate(pair_data)]
    cbb_results = _parallel_or_sequential(_cbb_one_pair, cbb_args, n_jobs, "CBB null")

    cbb_total_exceed = sum(r["exceed"] + 1 for r in cbb_results)
    cbb_total_trials = len(cbb_results) * (n_perm + 1)
    cbb_pvalue = cbb_total_exceed / cbb_total_trials
    cbb_per_pair_pvals = [r["pval"] for r in cbb_results]
    cbb_blocks = [r["block"] for r in cbb_results]

    if cbb_per_pair_pvals:
        logger.info(
            "  CBB null: mean_obs_ARI=%.4f, aggregate p-value=%.6f "
            "(n_pairs=%d, block median=%.1f, range=[%.0f, %.0f])",
            mean_obs, cbb_pvalue, len(cbb_per_pair_pvals),
            float(np.median(cbb_blocks)), float(np.min(cbb_blocks)), float(np.max(cbb_blocks)),
        )
        logger.info(
            "  CBB per-pair p-values: min=%.4f, median=%.4f, max=%.4f",
            float(np.min(cbb_per_pair_pvals)), float(np.median(cbb_per_pair_pvals)),
            float(np.max(cbb_per_pair_pvals)),
        )

    # --- Benjamini-Hochberg multiple-comparison correction ---
    # Applied independently to perm and CBB per-pair p-value vectors.
    perm_arr = np.asarray(per_pair_pvals, dtype=float)
    cbb_arr = np.asarray(cbb_per_pair_pvals, dtype=float)
    perm_q = _bh_fdr_adjust(perm_arr) if perm_arr.size else perm_arr
    cbb_q = _bh_fdr_adjust(cbb_arr) if cbb_arr.size else cbb_arr
    n_perm_pairs = perm_arr.size
    n_cbb_pairs = cbb_arr.size

    result: Dict[str, float] = {"pvalue_all": pvalue, "obs_ari": mean_obs}
    if per_pair_pvals:
        result["pvalue_perpair_min"] = float(np.min(per_pair_pvals))
        result["pvalue_perpair_median"] = float(np.median(per_pair_pvals))
        result["pvalue_perpair_max"] = float(np.max(per_pair_pvals))
        result["pvalue_perpair_n"] = float(len(per_pair_pvals))
        for q in BH_Q_LEVELS:
            tag = str(int(round(q * 100))).zfill(2)
            result[f"pvalue_perpair_bh_q{tag}_frac"] = float(np.mean(perm_q < q))
        # Bonferroni as conservative upper bound (raw p * n < alpha).
        result["pvalue_perpair_bonf_q05_frac"] = float(np.mean(perm_arr * n_perm_pairs < 0.05))

    if cbb_per_pair_pvals:
        result["cbb_pvalue_all"] = cbb_pvalue
        result["cbb_pvalue_perpair_min"] = float(np.min(cbb_per_pair_pvals))
        result["cbb_pvalue_perpair_median"] = float(np.median(cbb_per_pair_pvals))
        result["cbb_pvalue_perpair_max"] = float(np.max(cbb_per_pair_pvals))
        result["cbb_pvalue_perpair_n"] = float(len(cbb_per_pair_pvals))
        result["cbb_block_median"] = float(np.median(cbb_blocks))
        result["cbb_block_min"] = float(np.min(cbb_blocks))
        result["cbb_block_max"] = float(np.max(cbb_blocks))
        for q in BH_Q_LEVELS:
            tag = str(int(round(q * 100))).zfill(2)
            result[f"cbb_pvalue_perpair_bh_q{tag}_frac"] = float(np.mean(cbb_q < q))
        result["cbb_pvalue_perpair_bonf_q05_frac"] = float(np.mean(cbb_arr * n_cbb_pairs < 0.05))

    return result


# ---------------------------------------------------------------------------
# Key-results update
# ---------------------------------------------------------------------------

def _mean_or_nan(x) -> float:
    try:
        v = float(pd.to_numeric(pd.Series(x), errors="coerce").mean())
        return v
    except Exception:
        return float("nan")


def update_key_results(key_results_path: Path, new_rows: List[Dict]) -> None:
    """Append new metric rows to an existing key_results.csv (no duplicates)."""
    existing = pd.read_csv(key_results_path)
    # Remove any previously written posthoc rows (idempotent re-run)
    posthoc_metrics = {r["metric"] for r in new_rows}
    existing = existing[~existing["metric"].isin(posthoc_metrics)]
    new_df = pd.DataFrame(new_rows)
    out = pd.concat([existing, new_df], ignore_index=True)
    out.to_csv(key_results_path, index=False)
    logger.info("  Updated %s (+%d rows)", key_results_path.name, len(new_rows))


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_asset_step(
    asset_dir: Path,
    step: Optional[int],
    reps: List[str],
    models: List[str],
    key_results_path: Path,
) -> None:
    asset = asset_dir.name
    step_label = "root" if step is None else f"step={step}"
    logger.info("Processing %s / %s", asset, step_label)

    # Infer seeds and rolls from disk
    hard_map = build_hard_map(asset_dir, step, reps=reps, models=models)
    if not hard_map:
        logger.warning("  No hard states found; skipping.")
        return

    present_reps = sorted({k[0] for k in hard_map})
    present_models = sorted({k[1] for k in hard_map})
    seeds = sorted({k[2] for k in hard_map})
    rolls = sorted({k[3] for k in hard_map})

    logger.info("  reps=%s models=%s seeds=%s rolls=%d",
                present_reps, present_models, seeds, len(rolls))

    # --- Cross-rep metrics ---
    cr_records = cross_rep_records(hard_map, present_reps, present_models, seeds, rolls)
    cr_df = pd.DataFrame(cr_records)

    # --- Temporal metrics ---
    tmp_records = temporal_records(hard_map, present_reps, present_models, seeds, rolls)
    tmp_df = pd.DataFrame(tmp_records)

    # --- Permutation test (aggregate, only for baseline step) ---
    perm_result = aggregate_permutation_pvalue(
        hard_map, present_reps, present_models, seeds, rolls,
        n_sample_pairs=PERM_MAX_PAIRS,
    )

    # --- Build new key_results rows ---
    new_rows: List[Dict] = []

    def _add(metric, scope, value, n=0):
        if not np.isnan(float(value if value is not None else float("nan"))):
            new_rows.append({"metric": metric, "scope": scope, "value": value, "n": n})

    if not cr_df.empty:
        _add("cross_rep_ami_mean", "all", _mean_or_nan(cr_df["ami"]), len(cr_df))
        _add("cross_rep_vi_mean",  "all", _mean_or_nan(cr_df["vi"]),  len(cr_df))
        if "model" in cr_df.columns:
            for model_name, g in cr_df.groupby("model"):
                _add(f"cross_rep_ami_mean", f"model={model_name}",
                     _mean_or_nan(g["ami"]), len(g))
                _add(f"cross_rep_vi_mean",  f"model={model_name}",
                     _mean_or_nan(g["vi"]),  len(g))

    if not tmp_df.empty:
        _add("temporal_ami_mean", "all", _mean_or_nan(tmp_df["ami"]), len(tmp_df))
        _add("temporal_vi_mean",  "all", _mean_or_nan(tmp_df["vi"]),  len(tmp_df))
        if "model" in tmp_df.columns:
            for model_name, g in tmp_df.groupby("model"):
                _add(f"temporal_ami_mean", f"model={model_name}",
                     _mean_or_nan(g["ami"]), len(g))
                _add(f"temporal_vi_mean",  f"model={model_name}",
                     _mean_or_nan(g["vi"]),  len(g))

    if not np.isnan(perm_result.get("pvalue_all", float("nan"))):
        _add("crossrep_ari_perm_pvalue", "all", perm_result["pvalue_all"])
    for suffix in ("perpair_min", "perpair_median", "perpair_max", "perpair_n"):
        key = f"pvalue_{suffix}"
        if key in perm_result and not np.isnan(perm_result[key]):
            _add(f"crossrep_ari_perm_{suffix}", "all", perm_result[key])
    # BH-FDR + Bonferroni fractions on the iid-permutation per-pair p-values.
    for suffix in (
        "perpair_bh_q05_frac", "perpair_bh_q10_frac", "perpair_bonf_q05_frac",
    ):
        key = f"pvalue_{suffix}"
        if key in perm_result and not np.isnan(perm_result[key]):
            _add(f"crossrep_ari_perm_{suffix}", "all", perm_result[key])
    # Politis-Romano circular block bootstrap null (preserves autocorrelation).
    if not np.isnan(perm_result.get("cbb_pvalue_all", float("nan"))):
        _add("crossrep_ari_cbb_pvalue", "all", perm_result["cbb_pvalue_all"])
    for suffix in (
        "perpair_min", "perpair_median", "perpair_max", "perpair_n",
        "perpair_bh_q05_frac", "perpair_bh_q10_frac", "perpair_bonf_q05_frac",
    ):
        key = f"cbb_pvalue_{suffix}"
        if key in perm_result and not np.isnan(perm_result[key]):
            _add(f"crossrep_ari_cbb_{suffix}", "all", perm_result[key])
    for suffix in ("median", "min", "max"):
        key = f"cbb_block_{suffix}"
        if key in perm_result and not np.isnan(perm_result[key]):
            _add(f"crossrep_ari_cbb_block_{suffix}", "all", perm_result[key])

    # --- Update key_results.csv ---
    if key_results_path.exists() and new_rows:
        update_key_results(key_results_path, new_rows)
    else:
        logger.warning("  key_results.csv not found at %s", key_results_path)


def update_step_sweep_summary(
    outputs_dir: Path, steps: List[int], allowed_assets: Optional[List[str]] = None
) -> None:
    """Regenerate step_sweep_summary.csv with ami/vi columns added.

    If `allowed_assets` is provided (typically from cfg.assets via utils.safe_name),
    only those asset directories are included. Otherwise every subdirectory of
    outputs_dir is included — which incorrectly pulls in diagnostic dirs like
    synthetic_sanity/ and contaminates the 4-asset aggregates.
    """
    summary_path = outputs_dir / "step_sweep_summary.csv"
    allowed_set = set(allowed_assets) if allowed_assets is not None else None
    rows = []
    for asset_dir in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        asset = asset_dir.name
        if allowed_set is not None and asset not in allowed_set:
            continue
        for step in steps:
            kr_path = asset_dir / f"step_{step}" / "key_results.csv"
            if not kr_path.exists():
                continue
            kr = pd.read_csv(kr_path)
            def _get(metric, scope="all"):
                s = kr[(kr["metric"] == metric) & (kr["scope"] == scope)]
                return float(pd.to_numeric(s["value"].iloc[0], errors="coerce")) if not s.empty else float("nan")
            rows.append({
                "step": step,
                "asset": asset,
                "cross_rep_ari_mean":  _get("cross_rep_ari_mean"),
                "temporal_ari_mean":   _get("temporal_ari_mean"),
                "temporal_overlap_ari_mean": _get("temporal_overlap_ari_mean"),
                "cross_rep_ami_mean":  _get("cross_rep_ami_mean"),
                "temporal_ami_mean":   _get("temporal_ami_mean"),
                "cross_rep_vi_mean":   _get("cross_rep_vi_mean"),
                "temporal_vi_mean":    _get("temporal_vi_mean"),
                "crossrep_ari_perm_pvalue": _get("crossrep_ari_perm_pvalue"),
            })
    if rows:
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        logger.info("Updated step_sweep_summary.csv (%d rows)", len(rows))


def _resolve_run_defs(cfg: Dict) -> Tuple[List[int], List[str], List[str]]:
    """Resolve (steps, reps, models) from cfg.

    Raises on missing/empty cfg keys. Previously this function fell back to
    hardcoded module-level REPS/MODELS when cfg was absent or incomplete,
    which silently dropped any rep defined in config.yaml but not in the
    hardcoded list (e.g., rep_e). Silent fallback is now disabled; callers
    must provide a fully populated cfg loaded from config.yaml.
    """
    if not isinstance(cfg, dict):
        raise TypeError(
            "_resolve_run_defs requires a dict cfg (e.g. from config.yaml); got "
            f"{type(cfg).__name__}"
        )

    grid = cfg.get("grid", {})
    if not isinstance(grid, dict):
        grid = {}
    if isinstance(grid.get("step_sweep"), list) and grid.get("step_sweep"):
        steps = [int(x) for x in grid["step_sweep"]]
    elif "step" in grid:
        steps = [int(grid["step"])]
    else:
        raise KeyError("cfg.grid must define 'step_sweep' (non-empty list) or 'step' (int)")

    reps_cfg = cfg.get("representations")
    if not isinstance(reps_cfg, dict) or not reps_cfg:
        raise KeyError("cfg.representations must be a non-empty dict of rep_name -> spec")
    reps = [str(k) for k in reps_cfg.keys()]

    models_cfg = cfg.get("models")
    if not isinstance(models_cfg, dict) or not models_cfg:
        raise KeyError("cfg.models must be a non-empty dict")
    models = [
        str(m)
        for m in ("gmm", "hmm")
        if isinstance(models_cfg.get(m, {}), dict)
        and bool(models_cfg.get(m, {}).get("enabled", True))
    ]
    if not models:
        raise ValueError("cfg.models has no enabled entries among ('gmm', 'hmm')")

    return steps, reps, models


def main(cfg: Optional[Dict] = None) -> None:
    if cfg is None:
        cfg_path = ROOT / "config.yaml"
        if cfg_path.exists():
            try:
                import yaml

                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = None
    if cfg is None:
        logger.error("posthoc_ami_vi_perm requires config context (cfg or ROOT/config.yaml).")
        sys.exit(1)
    steps, reps, models = _resolve_run_defs(cfg)
    outputs_dir = Path(cfg.get("outputs_dir", str(OUTPUTS_DIR))) if isinstance(cfg, dict) else OUTPUTS_DIR

    # Restrict to cfg.assets (safe_name-normalised) to avoid pulling in diagnostic
    # subdirs like synthetic_sanity/ that contaminate 4-asset aggregates.
    from src.core.utils import assets_from_cfg, safe_name
    allowed_assets = {safe_name(a) for a in assets_from_cfg(cfg)}

    asset_dirs = sorted(
        p for p in outputs_dir.iterdir()
        if p.is_dir() and p.name in allowed_assets
        and any((p / f"step_{s}").exists() for s in steps)
    )
    if not asset_dirs:
        # Fallback root-layout assets (still filtered to cfg.assets).
        asset_dirs = sorted(
            p for p in outputs_dir.iterdir()
            if p.is_dir() and p.name in allowed_assets and (p / "key_results.csv").exists()
        )
    if not asset_dirs:
        logger.error("No asset directories found under %s (cfg.assets=%s)", outputs_dir, sorted(allowed_assets))
        sys.exit(1)

    for asset_dir in asset_dirs:
        step_dirs = [asset_dir / f"step_{step}" for step in steps]
        has_step_layout = any(p.exists() for p in step_dirs)
        if has_step_layout:
            for step in steps:
                step_dir = asset_dir / f"step_{step}"
                if step_dir.exists():
                    process_asset_step(
                        asset_dir,
                        step,
                        reps=reps,
                        models=models,
                        key_results_path=asset_dir / f"step_{step}" / "key_results.csv",
                    )
        else:
            process_asset_step(
                asset_dir,
                None,
                reps=reps,
                models=models,
                key_results_path=asset_dir / "key_results.csv",
            )

    update_step_sweep_summary(outputs_dir, steps, allowed_assets=sorted(allowed_assets))

    # Regenerate the multi-asset summary so newly-appended perm p-values and
    # AMI/VI rows propagate to outputs/key_results_all_assets.csv. The runner's
    # native writer is keyed on baseline step=21 per-asset key_results.csv.
    baseline_step = int(cfg.get("grid", {}).get("step", 21))
    rows: List[pd.DataFrame] = []
    for asset in allowed_assets:
        kr = outputs_dir / asset / f"step_{baseline_step}" / "key_results.csv"
        if not kr.exists():
            continue
        sub = pd.read_csv(kr)
        sub.insert(0, "asset", asset)
        rows.append(sub)
    if rows:
        all_path = outputs_dir / "key_results_all_assets.csv"
        pd.concat(rows, axis=0, ignore_index=True).to_csv(all_path, index=False)
        logger.info("Updated %s (%d rows)", all_path, sum(len(r) for r in rows))

    logger.info("Done.")


if __name__ == "__main__":
    main()
