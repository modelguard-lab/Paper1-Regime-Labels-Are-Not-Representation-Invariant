"""
Post-hoc synthetic experiment: instability vs plurality.

Generates data from a known 3-state regime-switching DGP, runs the
representation pipeline, and compares each representation's inferred
labels against the ground truth.  If ARI(rep, truth) is high across
representations but ARI(rep_i, rep_j) is low, the low cross-rep ARI
reflects plurality.  If ARI(rep, truth) is also low, it reflects
genuine instability in the inference pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.core.features import RepConfig, build_representation_single
from src.core.models import fit_hmm, fit_gmm
from src.experiments.synthetic_sanity import SynthParams, generate_synthetic_price_and_truth

logger = logging.getLogger(__name__)


def _default_reps() -> list[RepConfig]:
    return [
        RepConfig(name="rep_a", features=["volatility", "drawdown", "max_drawdown_window", "var", "cvar"],
                  windows={"vol_window": 20, "drawdown_window": 60, "tail_window": 60, "tail_alpha": 0.05},
                  standardization={"mode": "rolling_zscore", "window": 120}),
        RepConfig(name="rep_a_unscaled", features=["volatility", "drawdown", "max_drawdown_window", "var", "cvar"],
                  windows={"vol_window": 20, "drawdown_window": 60, "tail_window": 60, "tail_alpha": 0.05},
                  standardization={"mode": "none"}),
        RepConfig(name="rep_b", features=["realized_skew", "stability", "var", "cvar"],
                  windows={"skew_window": 60, "stability_window": 60, "tail_window": 60, "tail_alpha": 0.05},
                  standardization={"mode": "rolling_zscore", "window": 120}),
        RepConfig(name="rep_c1", features=["volatility", "drawdown", "var", "cvar"],
                  windows={"vol_window": 30, "drawdown_window": 90, "tail_window": 90, "tail_alpha": 0.05},
                  standardization={"mode": "rolling_zscore", "window": 120}),
    ]


def run_synthetic_groundtruth(
    n_seeds: int = 10,
    K: int = 3,
    W: int = 252,
    T: int = 2200,
    persistence_p: float = 0.97,
    drift_alpha: float = 0.6,
    sigmas: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Run the synthetic ground-truth experiment.

    Returns a DataFrame with columns:
    - dgp_seed: seed for the DGP
    - rep: representation name
    - model: gmm or hmm
    - fit_seed: seed for the model fit
    - ari_vs_truth: ARI of inferred labels vs ground truth
    - ari_cross_rep: mean ARI vs other representations (same fit_seed)
    """
    sp_kwargs: dict = {"T": T, "K": K, "persistence_p": persistence_p, "drift_alpha": drift_alpha}
    if sigmas is not None:
        sp_kwargs["sigmas"] = sigmas
    params = SynthParams(**sp_kwargs)
    reps = _default_reps()
    records = []

    for dgp_seed in range(1, n_seeds + 1):
        prices_df, truth_df = generate_synthetic_price_and_truth(
            params=params, seed=dgp_seed, drift_alpha=params.drift_alpha,
        )
        price = prices_df.set_index("Date")["Close"]
        truth = truth_df.set_index("Date")["state_true"]

        # Build representations
        rep_features = {}
        for rc in reps:
            feat = build_representation_single(price, rc).dropna()
            if len(feat) >= W:
                rep_features[rc.name] = feat

        if len(rep_features) < 2:
            continue

        # Use a single window in the middle of the sample (most representative)
        mid = len(price) // 2
        start = mid - W // 2
        end = start + W

        for fit_seed in [1, 2, 3]:
            inferred = {}
            for rep_name, feat in rep_features.items():
                X = feat.iloc[start:end]
                if len(X) < W:
                    continue
                window_dates = X.index

                for model_name in ["hmm", "gmm"]:
                    try:
                        if model_name == "hmm":
                            result = fit_hmm(X, n_states=K, random_state=fit_seed)
                        else:
                            result = fit_gmm(X, n_states=K, random_state=fit_seed)
                    except Exception:
                        continue

                    s = result.states_hard

                    # ARI vs ground truth
                    common = truth.reindex(s.index).dropna()
                    s_common = s.reindex(common.index).dropna()
                    if len(s_common) < 20:
                        continue
                    ari_truth = float(adjusted_rand_score(
                        common.loc[s_common.index].values,
                        s_common.values,
                    ))

                    key = (rep_name, model_name, fit_seed)
                    inferred[key] = s_common

                    records.append({
                        "dgp_seed": dgp_seed,
                        "rep": rep_name,
                        "model": model_name,
                        "fit_seed": fit_seed,
                        "ari_vs_truth": ari_truth,
                    })

            # Cross-rep ARI
            keys = list(inferred.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    ki, kj = keys[i], keys[j]
                    if ki[1] != kj[1] or ki[2] != kj[2]:
                        continue  # same model and fit_seed
                    si, sj = inferred[ki], inferred[kj]
                    common_idx = si.index.intersection(sj.index)
                    if len(common_idx) < 20:
                        continue
                    ari_cross = float(adjusted_rand_score(
                        si.loc[common_idx].values,
                        sj.loc[common_idx].values,
                    ))
                    # Append to both reps
                    for rec in records:
                        if (rec["dgp_seed"] == dgp_seed and
                            rec["model"] == ki[1] and
                            rec["fit_seed"] == ki[2] and
                            rec["rep"] in (ki[0], kj[0])):
                            rec.setdefault("_cross_aris", []).append(ari_cross)

        logger.info("DGP seed %d done", dgp_seed)

    # Compute mean cross-rep ARI per record
    for rec in records:
        cross = rec.pop("_cross_aris", [])
        rec["ari_cross_rep_mean"] = float(np.mean(cross)) if cross else np.nan

    return pd.DataFrame(records)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline drift_alpha=0.6 (canonical 4-rep synthetic experiment cited in
    # main text Section 3.6 and Supplementary Section S9).
    df_main = run_synthetic_groundtruth(n_seeds=10, drift_alpha=0.6)
    if df_main.empty:
        print("No results.")
        return
    df_main.to_csv(out_dir / "synthetic_groundtruth.csv", index=False)
    print(f"Saved baseline (alpha=0.6) to {out_dir / 'synthetic_groundtruth.csv'}")

    # Drift ablation: alpha=0 (no deterministic drift component, states differ
    # only in volatility). Sourced for the 4-rep alpha-invariance claim in
    # main text Section 3.6 and the alpha-ablation cell of Supplementary
    # Table S6.
    df_alpha0 = run_synthetic_groundtruth(n_seeds=10, drift_alpha=0.0)
    df_alpha0.to_csv(out_dir / "synthetic_groundtruth_alpha0.csv", index=False)
    print(f"Saved alpha=0 ablation to {out_dir / 'synthetic_groundtruth_alpha0.csv'}")
    print()

    # Summary across both runs
    for label, df in [("alpha=0.6 (baseline)", df_main), ("alpha=0   (ablation)", df_alpha0)]:
        print(f"--- {label} ---")
        for model in ["hmm", "gmm"]:
            sub = df[df["model"] == model]
            if sub.empty:
                continue
            mean_truth = sub["ari_vs_truth"].mean()
            mean_cross = sub["ari_cross_rep_mean"].dropna().mean()
            print(f"{model.upper()}: ARI vs truth = {mean_truth:.3f}; ARI cross-rep = {mean_cross:.3f}")
        print()


if __name__ == "__main__":
    main()
