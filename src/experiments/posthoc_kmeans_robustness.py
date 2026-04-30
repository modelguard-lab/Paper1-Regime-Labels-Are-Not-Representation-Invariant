"""
Post-hoc non-parametric robustness: k-means and spectral clustering.

Motivation
----------
Reviewer 2 (FRL major revision) asked how non-parametric models -- specifically
those that do not assume discrete states with full Gaussian emissions -- would
affect the cross-representation agreement results. The main pipeline uses
Gaussian HMM and Gaussian Mixture Models, both of which impose parametric
assumptions on within-state emissions. K-means imposes only a centroid-distance
assumption, and spectral clustering uses an eigendecomposition of an affinity
matrix without any parametric emission model.

What this script does
---------------------
For each (asset, rep, roll, seed), re-fit k-means (and optionally spectral)
on the same feature matrix and roll boundaries used by the main pipeline,
producing a hard state sequence with K=3. Then compute cross-representation
ARI within k-means (and within spectral) and compare to the HMM/GMM cross-rep
ARI from the baseline.

Inputs (read-only):
- outputs/<asset>/step_21/results/<rep>/features.csv
- outputs/<asset>/step_21/results/windows_index.csv

Outputs (written):
- outputs/kmeans_robustness_summary.csv: per-(asset, model) mean +/- 95% CI
  cross-rep ARI across seeds.
- outputs/kmeans_robustness_per_pair.csv: per-(asset, model, seed, rep_pair)
  mean ARI across rolls (for response-letter detail).
"""
from __future__ import annotations

import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"


def _safe_name(asset: str) -> str:
    return asset.replace("^", "").replace("/", "_").replace("\\", "_").replace(":", "_").strip()


def _load_features(asset_dir: Path, rep: str) -> pd.DataFrame:
    f = asset_dir / rep / "features.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()
    return df


def _load_windows_index(asset_dir: Path) -> pd.DataFrame:
    f = asset_dir / "windows_index.csv"
    df = pd.read_csv(f, parse_dates=["start_date", "end_date"])
    return df


def _fit_kmeans(X: np.ndarray, K: int, seed: int) -> np.ndarray:
    km = KMeans(n_clusters=K, n_init=10, random_state=int(seed))
    return km.fit_predict(X)


def _fit_spectral(X: np.ndarray, K: int, seed: int) -> np.ndarray:
    # affinity='nearest_neighbors' is more stable than RBF on small windows;
    # n_neighbors=10 is the sklearn default but we make it explicit.
    sc = SpectralClustering(
        n_clusters=K,
        affinity="nearest_neighbors",
        n_neighbors=10,
        assign_labels="kmeans",
        random_state=int(seed),
        n_init=10,
    )
    return sc.fit_predict(X)


def _fit_for_window(
    feats_by_rep: Dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    K: int,
    seed: int,
    method: str,
) -> Dict[str, np.ndarray]:
    """Fit clustering for a single rolling window across all reps; return {rep: hard labels}."""
    out: Dict[str, np.ndarray] = {}
    for rep, feats in feats_by_rep.items():
        # Slice [start, end) inclusive of start, exclusive of end -- matches the pipeline
        # convention end=start+W where W is window length.
        sub = feats.loc[(feats.index >= start) & (feats.index < end)]
        if sub.shape[0] < 2 * K or sub.isna().any(axis=1).all():
            continue
        sub = sub.dropna(axis=0, how="any")
        if sub.shape[0] < 2 * K:
            continue
        X = sub.values
        if method == "kmeans":
            labels = _fit_kmeans(X, K, seed)
        elif method == "spectral":
            try:
                labels = _fit_spectral(X, K, seed)
            except Exception:
                # Spectral can fail on degenerate affinity matrices; fall back to k-means
                labels = _fit_kmeans(X, K, seed)
        else:
            raise ValueError(f"Unknown method: {method}")
        # Re-index to the slice index so we can align across reps later
        out[rep] = pd.Series(labels, index=sub.index, dtype="Int64")
    return out


def _cross_rep_ari(labels_by_rep: Dict[str, pd.Series]) -> List[Tuple[str, str, float]]:
    """Pairwise ARI for all (rep_a, rep_b) on intersected dates."""
    reps = sorted(labels_by_rep.keys())
    rows: List[Tuple[str, str, float]] = []
    for ra, rb in combinations(reps, 2):
        la = labels_by_rep[ra]
        lb = labels_by_rep[rb]
        idx = la.index.intersection(lb.index)
        if len(idx) < 10:
            continue
        ari = float(adjusted_rand_score(la.loc[idx].astype(int).values, lb.loc[idx].astype(int).values))
        rows.append((ra, rb, ari))
    return rows


def run_asset(
    asset: str,
    reps: List[str],
    K: int,
    seeds: List[int],
    methods: List[str],
) -> List[Dict]:
    safe = _safe_name(asset)
    asset_dir = OUTPUTS_DIR / safe / "step_21" / "results"
    if not asset_dir.exists():
        logger.warning("Skip %s: no outputs at %s", asset, asset_dir)
        return []
    windows = _load_windows_index(asset_dir)
    feats_by_rep: Dict[str, pd.DataFrame] = {}
    for rep in reps:
        f = _load_features(asset_dir, rep)
        if not f.empty:
            feats_by_rep[rep] = f
    if len(feats_by_rep) < 2:
        logger.warning("Skip %s: fewer than 2 reps with features", asset)
        return []

    rows: List[Dict] = []
    for method in methods:
        logger.info("Asset=%s method=%s reps=%d rolls=%d seeds=%d",
                    asset, method, len(feats_by_rep), len(windows), len(seeds))
        for seed in seeds:
            for _, w in windows.iterrows():
                labels_by_rep = _fit_for_window(
                    feats_by_rep, w["start_date"], w["end_date"], K, seed, method
                )
                if len(labels_by_rep) < 2:
                    continue
                for ra, rb, ari in _cross_rep_ari(labels_by_rep):
                    rows.append({
                        "asset": asset,
                        "method": method,
                        "K": K,
                        "seed": seed,
                        "roll": w["roll"],
                        "rep_a": ra,
                        "rep_b": rb,
                        "ari": ari,
                    })
    return rows


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assets = list(cfg["assets"])
    reps_all = list(cfg["representations"].keys())

    # Match the main pipeline subset: drop rep_e for non-SPX (asset_filter handled per-asset below).
    K = 3
    seeds = [1, 2, 3, 4, 5]
    # Spectral clustering on 252-day windows is expensive (eigendecomposition of
    # k-NN affinity); k-means alone provides the non-parametric / centroid-based
    # comparison the reviewer asked for. Spectral can be re-enabled by appending
    # "spectral" but expect ~30+ min total runtime for the full sweep.
    methods = ["kmeans"]

    all_rows: List[Dict] = []
    for asset in assets:
        # Determine reps available for this asset (feature dirs already on disk)
        safe = _safe_name(asset)
        asset_dir = OUTPUTS_DIR / safe / "step_21" / "results"
        reps = [r for r in reps_all if (asset_dir / r / "features.csv").exists()]
        rows = run_asset(asset, reps, K, seeds, methods)
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No results produced.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    out_dir = OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-pair detail
    pair_path = out_dir / "kmeans_robustness_per_pair.csv"
    df_pair = (
        df.groupby(["asset", "method", "seed", "rep_a", "rep_b"], as_index=False)["ari"]
          .mean()
          .rename(columns={"ari": "mean_ari"})
    )
    df_pair.to_csv(pair_path, index=False)
    logger.info("Wrote %s (%d rows)", pair_path, len(df_pair))

    # Per-(asset, method) summary across seeds and pairs
    df_seed_mean = (
        df.groupby(["asset", "method", "seed"], as_index=False)["ari"]
          .mean()
    )
    summary_rows: List[Dict] = []
    for (asset, method), grp in df_seed_mean.groupby(["asset", "method"]):
        seed_means = grp["ari"].values
        n = len(seed_means)
        mean = float(np.mean(seed_means))
        # Asymptotic 95% CI using t with n-1 df; for n=5 t_{4,0.975}=2.776.
        from scipy.stats import t as _t
        if n > 1:
            se = float(np.std(seed_means, ddof=1) / np.sqrt(n))
            half = float(_t.ppf(0.975, n - 1)) * se
        else:
            half = 0.0
        summary_rows.append({
            "asset": asset,
            "method": method,
            "n_seeds": n,
            "mean_ari": mean,
            "ci95_half": half,
            "ari_low": mean - half,
            "ari_high": mean + half,
        })
    df_summary = pd.DataFrame(summary_rows).sort_values(["method", "asset"])
    summary_path = out_dir / "kmeans_robustness_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s", summary_path)

    # Asset-average summary (mean of asset means)
    asset_avg_rows: List[Dict] = []
    for method, grp in df_summary.groupby("method"):
        m = float(grp["mean_ari"].mean())
        # Across-asset SE (n=4 assets)
        n_a = len(grp)
        from scipy.stats import t as _t
        if n_a > 1:
            se = float(grp["mean_ari"].std(ddof=1) / np.sqrt(n_a))
            half = float(_t.ppf(0.975, n_a - 1)) * se
        else:
            half = 0.0
        asset_avg_rows.append({
            "method": method,
            "n_assets": n_a,
            "mean_ari": m,
            "ci95_half": half,
        })
    df_avg = pd.DataFrame(asset_avg_rows)
    avg_path = out_dir / "kmeans_robustness_asset_avg.csv"
    df_avg.to_csv(avg_path, index=False)
    logger.info("Wrote %s", avg_path)

    print("\n=== Asset-average cross-rep ARI (K=3, baseline window=252, step=21) ===")
    print(df_avg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Per-asset summary ===")
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
