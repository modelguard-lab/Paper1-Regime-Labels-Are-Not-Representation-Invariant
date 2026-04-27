"""
Post-hoc rank-aligned ordering consistency.

Motivation
----------
The pipeline-native ordering metric pairs states across representations via
Hungarian matching on |delta CVaR| + |delta Vol| and then asks whether the
worst-CVaR state in A is paired with the worst-CVaR state in B. Because the
matching cost is driven by CVaR, the optimiser is effectively forced to
rank-align extremes: the empirical independent-random-partition null is
0.78-0.92 (far above the docstring's claimed 1/K), so "observed top-1 ≈ 0.93"
is not meaningfully above chance under the same matching rule.

Proper test
-----------
Relabel both state sequences by within-partition CVaR rank (0 = worst),
then compare the relabelled sequences pointwise. Under two independent
uniform random K-partitions the expected pointwise agreement is exactly 1/K
and the expected top-1 (fraction of A-worst dates that are also B-worst) is
exactly 1/K, so the null is unambiguous and the constructive bias of the
Hungarian + CVaR rule is removed.

This script reads the existing ``windows_states_hard_{model}.csv`` files that
the pipeline already produced, recomputes rank-aligned metrics for every
admissible (rep_a, rep_b) pair within each (asset, model, K, seed, roll), and
writes a summary CSV alongside the existing ordering outputs. No pipeline
re-fit is required.
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data import load_price_series
from utils import assets_from_cfg, enabled_models_from_cfg, reps_from_cfg, safe_name

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
ALPHA_DEFAULT = 0.05
N_PERM_DEFAULT = 500
PERM_SEED = 4242


def _cvar_per_state(returns: np.ndarray, labels: np.ndarray, K: int, alpha: float) -> np.ndarray:
    out = np.full(K, np.nan)
    for k in range(K):
        r = returns[labels == k]
        if r.size == 0:
            continue
        q = float(np.quantile(r, alpha))
        tail = r[r <= q]
        out[k] = float(np.mean(tail)) if tail.size else q
    return out


def _rank_by_cvar(cv: np.ndarray) -> np.ndarray:
    """Return a permutation array `rank` of length K such that `rank[old_label]`
    is the CVaR rank with 0 = worst (most negative). NaN states are ranked last.
    """
    K = len(cv)
    order = np.argsort(np.where(np.isnan(cv), np.inf, cv))  # most-negative first; NaN last
    rank = np.empty(K, dtype=int)
    for new_rank, old_label in enumerate(order):
        rank[int(old_label)] = int(new_rank)
    return rank


def rank_aligned_metrics(
    returns: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray, K: int, alpha: float
) -> Dict[str, float]:
    """Rank-aligned top-1 overlap, pointwise agreement, and Spearman on rank sequences.

    - pointwise_agreement: mean over t of I[rank(A[t]) == rank(B[t])]. Null = 1/K.
    - top1_overlap: |A_worst ∩ B_worst| / |A_worst ∪ B_worst| (Jaccard). Null = 1/(2K-1).
    - spearman: Spearman rank corr of the rank sequences (ties handled by scipy).
    """
    cv_a = _cvar_per_state(returns, labels_a, K, alpha)
    cv_b = _cvar_per_state(returns, labels_b, K, alpha)
    r_a = _rank_by_cvar(cv_a)
    r_b = _rank_by_cvar(cv_b)
    ra_seq = r_a[labels_a]
    rb_seq = r_b[labels_b]

    agreement = float(np.mean(ra_seq == rb_seq))
    a_worst = ra_seq == 0
    b_worst = rb_seq == 0
    union = np.logical_or(a_worst, b_worst).sum()
    top1 = float(np.logical_and(a_worst, b_worst).sum() / union) if union > 0 else float("nan")

    # Spearman on the pointwise rank sequences.
    sp = float(pd.Series(ra_seq).corr(pd.Series(rb_seq), method="spearman"))

    return {"pointwise_agreement": agreement, "top1_jaccard": top1, "spearman_rank": sp}


def simulate_indep_null(
    returns: np.ndarray, K: int, n_perm: int, alpha: float, seed: int
) -> Dict[str, float]:
    """Monte-Carlo null: two independent uniform random K-partitions on the
    SAME return series. Confirms that the theoretical null (1/K) holds on
    finite samples even with empirical returns driving CVaR computation.
    """
    rng = np.random.default_rng(seed)
    T = returns.size
    agreements: List[float] = []
    top1s: List[float] = []
    sps: List[float] = []
    for _ in range(n_perm):
        la = rng.integers(0, K, size=T)
        lb = rng.integers(0, K, size=T)
        m = rank_aligned_metrics(returns, la, lb, K, alpha)
        agreements.append(m["pointwise_agreement"])
        if math.isfinite(m["top1_jaccard"]):
            top1s.append(m["top1_jaccard"])
        if math.isfinite(m["spearman_rank"]):
            sps.append(m["spearman_rank"])
    return {
        "null_pointwise_mean": float(np.mean(agreements)),
        "null_pointwise_p95": float(np.percentile(agreements, 95)),
        "null_top1_mean": float(np.mean(top1s)) if top1s else float("nan"),
        "null_top1_p95": float(np.percentile(top1s, 95)) if top1s else float("nan"),
        "null_spearman_mean": float(np.mean(sps)) if sps else float("nan"),
        "null_spearman_p95": float(np.percentile(sps, 95)) if sps else float("nan"),
        "null_n": n_perm,
    }


def _load_hard_states_for_asset_k(
    asset_dir: Path, step: int, rep: str, model: str, K: int
) -> Optional[pd.DataFrame]:
    # K=3 lives under the baseline step_<step>/results/ tree; other K live under
    # the robustness sweep at robustness/K_<K>/results/. Try both.
    candidates = [
        asset_dir / f"step_{step}" / "results" / rep / f"windows_states_hard_{model}.csv",
        asset_dir / "robustness" / f"K_{K}" / "results" / rep / f"windows_states_hard_{model}.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["date"])
        if "K" in df.columns:
            df = df[pd.to_numeric(df["K"], errors="coerce") == K]
        if not df.empty:
            return df
    return None


def _load_returns_for_asset(raw_dir: Path, asset: str) -> pd.Series:
    safe = safe_name(asset)
    path = Path(raw_dir) / f"{safe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Price CSV not found: {path}")
    price = load_price_series(path)
    r = np.log(price / price.shift(1)).dropna()
    r.index = pd.to_datetime(r.index)
    return r


def compute_for_asset_k(
    asset_dir: Path, raw_dir: Path, asset: str, K: int, step: int,
    reps: List[str], models: List[str], alpha: float, n_perm_null: int,
) -> Tuple[List[Dict], Dict[str, float]]:
    returns = _load_returns_for_asset(raw_dir, asset)

    # Build hard_map keyed by (rep, model, seed, roll)
    hard_map: Dict[Tuple[str, str, int, str], pd.Series] = {}
    for rep in reps:
        for model in models:
            df = _load_hard_states_for_asset_k(asset_dir, step, rep, model, K)
            if df is None:
                continue
            for (seed, roll), grp in df.groupby(["seed", "roll"]):
                s = grp.set_index("date")["state"].astype(int).sort_index()
                s = s[~s.index.duplicated(keep="last")]
                hard_map[(rep, model, int(seed), str(roll))] = s

    present_reps = sorted({k[0] for k in hard_map})
    present_models = sorted({k[1] for k in hard_map})
    seeds = sorted({k[2] for k in hard_map})
    rolls = sorted({k[3] for k in hard_map})

    pair_records: List[Dict] = []
    for model in present_models:
        for seed in seeds:
            for roll in rolls:
                for i in range(len(present_reps)):
                    for j in range(i + 1, len(present_reps)):
                        ra, rb = present_reps[i], present_reps[j]
                        sa = hard_map.get((ra, model, seed, roll))
                        sb = hard_map.get((rb, model, seed, roll))
                        if sa is None or sb is None:
                            continue
                        idx = sa.index.intersection(sb.index).intersection(returns.index)
                        if len(idx) < 20:
                            continue
                        la = sa.loc[idx].values.astype(int)
                        lb = sb.loc[idx].values.astype(int)
                        r = returns.loc[idx].values.astype(float)
                        m = rank_aligned_metrics(r, la, lb, K, alpha)
                        pair_records.append({
                            "asset": asset, "K": K, "model": model, "seed": seed, "roll": roll,
                            "rep_a": ra, "rep_b": rb, **m, "n": int(len(idx)),
                        })

    # Null: Monte-Carlo on the asset's full return series (length T)
    if len(returns) >= 60:
        null = simulate_indep_null(
            returns.values.astype(float), K, n_perm=n_perm_null, alpha=alpha, seed=PERM_SEED + K
        )
    else:
        null = {"null_pointwise_mean": float("nan")}
    null = {"asset": asset, "K": K, **null}
    return pair_records, null


def main(cfg: Optional[Dict] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    project = Path(__file__).resolve().parent.parent
    if cfg is None:
        cfg_path = project / "config.yaml"
        if cfg_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = None
    if not isinstance(cfg, dict):
        logger.error("posthoc_rank_aligned_ordering requires config context.")
        sys.exit(1)

    raw_dir = project / cfg.get("raw_dir", "data")
    outputs_dir = project / cfg.get("outputs_dir", "outputs")
    assets = assets_from_cfg(cfg)
    reps = reps_from_cfg(cfg)
    models = enabled_models_from_cfg(cfg)

    grid = cfg.get("grid", {}) or {}
    step = int(grid.get("step", 21))
    K_grid = [int(k) for k in ((grid.get("robustness") or {}).get("n_states") or grid.get("n_states") or [3])]

    all_pairs: List[Dict] = []
    all_null: List[Dict] = []
    for asset in assets:
        asset_safe = safe_name(asset)
        asset_dir = outputs_dir / asset_safe
        if not asset_dir.exists():
            logger.warning("Asset dir missing: %s", asset_dir)
            continue
        for K in K_grid:
            logger.info("Processing %s K=%d", asset, K)
            pairs, null = compute_for_asset_k(
                asset_dir, raw_dir, asset, K, step, reps, models,
                alpha=ALPHA_DEFAULT, n_perm_null=N_PERM_DEFAULT,
            )
            all_pairs.extend(pairs)
            all_null.append(null)

    if not all_pairs:
        logger.warning("No pair records computed.")
        return
    pairs_df = pd.DataFrame(all_pairs)
    null_df = pd.DataFrame(all_null)

    # Per-(asset, K) summary: mean across pairs x rolls x seeds x models.
    summary = (
        pairs_df.groupby(["asset", "K"], as_index=False)[
            ["pointwise_agreement", "top1_jaccard", "spearman_rank"]
        ]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [
        "_".join([str(x) for x in c if x]) if isinstance(c, tuple) else str(c) for c in summary.columns
    ]
    summary = summary.reset_index()
    summary = summary.merge(null_df, on=["asset", "K"], how="left")

    pairs_path = outputs_dir / "rank_aligned_ordering_pairs.csv"
    summary_path = outputs_dir / "rank_aligned_ordering_summary.csv"
    pairs_df.to_csv(pairs_path, index=False)
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s (%d pair rows)", pairs_path, len(pairs_df))
    logger.info("Wrote %s (%d summary rows)", summary_path, len(summary))

    # Short human-readable report.
    print()
    print(summary[[
        "asset", "K", "pointwise_agreement_mean", "null_pointwise_mean",
        "top1_jaccard_mean", "null_top1_mean", "spearman_rank_mean", "null_spearman_mean",
    ]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
