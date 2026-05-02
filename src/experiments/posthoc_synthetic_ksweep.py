"""State-count sweep for the synthetic ground-truth experiment.

Runs ``run_synthetic_groundtruth`` across K in {2, 3, 4} to address
reviewer concern that a single (p, K, T) configuration is too narrow,
companion to ``posthoc_synthetic_psweep`` which sweeps persistence_p.
Writes a long-format CSV with an added K column and prints a wide
summary table grouped by (K, model).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.posthoc_synthetic_groundtruth import run_synthetic_groundtruth

logger = logging.getLogger(__name__)

K_GRID: tuple[int, ...] = (2, 3, 4)
# Log-spaced volatility levels from 0.006 to 0.024 (matches the SynthParams
# default 3-tuple exactly at K=3; extends to K=2 and K=4 by interpolation
# in log-space). Documented in Supplementary Section S6 footnote.
SIGMA_LO: float = 0.006
SIGMA_HI: float = 0.024


def k_sigmas(K: int) -> tuple[float, ...]:
    return tuple(
        np.exp(np.linspace(np.log(SIGMA_LO), np.log(SIGMA_HI), K)).tolist()
    )


def run_ksweep(
    n_seeds: int = 10,
    persistence_p: float = 0.97,
    W: int = 252,
    T: int = 2200,
    k_grid: tuple[int, ...] = K_GRID,
) -> pd.DataFrame:
    frames = []
    for K in k_grid:
        logger.info("Running K=%d sigmas=%s", K, k_sigmas(K))
        df = run_synthetic_groundtruth(
            n_seeds=n_seeds, K=K, W=W, T=T,
            persistence_p=persistence_p, sigmas=k_sigmas(K),
        )
        df["K"] = K
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["K", "model"])
        .agg(
            ari_truth_mean=("ari_vs_truth", "mean"),
            ari_truth_se=("ari_vs_truth", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
            ari_cross_mean=("ari_cross_rep_mean", "mean"),
            ari_cross_se=("ari_cross_rep_mean", lambda s: s.dropna().std(ddof=1) / np.sqrt(s.dropna().size) if s.dropna().size else np.nan),
            n=("ari_vs_truth", "size"),
        )
        .reset_index()
    )
    return agg


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = run_ksweep(n_seeds=10)
    if df.empty:
        print("No results.")
        return

    out_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "synthetic_groundtruth_ksweep.csv"
    df.to_csv(long_path, index=False)

    summary = _summary(df)
    summary_path = out_dir / "synthetic_groundtruth_ksweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved long-format to {long_path}")
    print(f"Saved summary to     {summary_path}")
    print()
    print("State-count sweep summary (mean ± SE across DGP seeds × fit seeds):")
    print()
    for K in sorted(df["K"].unique()):
        print(f"  K = {K}")
        for model in ["hmm", "gmm"]:
            row = summary[(summary["K"] == K) & (summary["model"] == model)]
            if row.empty:
                continue
            r = row.iloc[0]
            print(
                f"    {model.upper():3s}  "
                f"ARI vs truth = {r['ari_truth_mean']:.3f} ± {r['ari_truth_se']:.3f}   "
                f"cross-rep ARI = {r['ari_cross_mean']:.3f} ± {r['ari_cross_se']:.3f}   "
                f"(n = {int(r['n'])})"
            )
        print()


if __name__ == "__main__":
    main()
