"""Persistence sweep for the synthetic ground-truth experiment.

Runs ``run_synthetic_groundtruth`` across persistence_p in {0.95, 0.97, 0.99}
to address reviewer concern that a single (p, K, T) configuration is too
narrow. Writes a long-format CSV with an added persistence_p column and
prints a wide summary table grouped by (p, model).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.posthoc_synthetic_groundtruth import run_synthetic_groundtruth

logger = logging.getLogger(__name__)

PERSISTENCE_GRID: tuple[float, ...] = (0.95, 0.97, 0.99)


def run_psweep(
    n_seeds: int = 10,
    K: int = 3,
    W: int = 252,
    T: int = 2200,
    persistence_grid: tuple[float, ...] = PERSISTENCE_GRID,
) -> pd.DataFrame:
    frames = []
    for p in persistence_grid:
        logger.info("Running persistence_p=%.2f", p)
        df = run_synthetic_groundtruth(
            n_seeds=n_seeds, K=K, W=W, T=T, persistence_p=p,
        )
        df["persistence_p"] = p
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["persistence_p", "model"])
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
    df = run_psweep(n_seeds=10)
    if df.empty:
        print("No results.")
        return

    out_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "synthetic_groundtruth_psweep.csv"
    df.to_csv(long_path, index=False)

    summary = _summary(df)
    summary_path = out_dir / "synthetic_groundtruth_psweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved long-format to {long_path}")
    print(f"Saved summary to     {summary_path}")
    print()
    print("Persistence sweep summary (mean ± SE across DGP seeds × fit seeds):")
    print()
    for p in sorted(df["persistence_p"].unique()):
        print(f"  p = {p:.2f}")
        for model in ["hmm", "gmm"]:
            row = summary[(summary["persistence_p"] == p) & (summary["model"] == model)]
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
