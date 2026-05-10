"""
Representation-dimension decomposition and variance decomposition.

Addresses two questions from Major Critique 2:

(i)  Which variation dimension drives the low cross-representation ARI?
     Pairs are classified into four mutually exclusive categories:
       - dim_std     : standardization only     (rep_a vs rep_a_unscaled)
       - dim_feature : feature-subset, same windows  ({rep_a, rep_c2, rep_c3} pairs)
       - dim_window  : window length only       (rep_c1 vs rep_c3)
       - dim_estimator: estimator family        (rep_d vs rolling-window reps)
       - dim_philosophy: feature philosophy     (rep_b vs vol/dd reps)
       - dim_mixed   : window + feature together ({rep_a, rep_c2} vs rep_c1)
       - dim_external: rep_e (VIX) vs any other rep  [GSPC only]
     Mean ARI per category reveals whether low agreement is driven by
     near-trivial window noise or genuine economic disagreement.

(ii) Variance decomposition: how much of the cross-rep ARI variance is
     explained by which pair is compared (representation effect) vs.
     which seed is used (estimation noise)?
     Computed as eta-squared: SS_between / SS_total, where SS_between is
     the between-pair-type sum-of-squares and SS_within is the within-pair
     seed-level variance.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pair classification
# ---------------------------------------------------------------------------

# Each pair (a, b) with a < b (alphabetical) maps to a dimension label.
_PAIR_DIMENSION: Dict[Tuple[str, str], str] = {
    # Dimension I: standardization only (same features, same windows)
    ("rep_a", "rep_a_unscaled"): "dim_std",

    # Dimension II: feature-subset, same windows (vol=20, dd=60, tail=60)
    ("rep_a", "rep_c2"): "dim_feature",    # rep_a adds VaR  over rep_c2
    ("rep_a", "rep_c3"): "dim_feature",    # rep_a adds MaxDD over rep_c3
    ("rep_c2", "rep_c3"): "dim_feature",   # MaxDD vs VaR toggle

    # Dimension III: window length only (same features [vol,dd,var,cvar])
    ("rep_c1", "rep_c3"): "dim_window",

    # Dimension IV: volatility estimator (GARCH vs rolling)
    ("rep_a", "rep_d"): "dim_estimator",
    ("rep_a_unscaled", "rep_d"): "dim_estimator",
    ("rep_c1", "rep_d"): "dim_estimator",
    ("rep_c2", "rep_d"): "dim_estimator",
    ("rep_c3", "rep_d"): "dim_estimator",

    # Feature philosophy: rep_b (skew/stability) vs vol/dd family
    ("rep_a", "rep_b"): "dim_philosophy",
    ("rep_a_unscaled", "rep_b"): "dim_philosophy",
    ("rep_b", "rep_c1"): "dim_philosophy",
    ("rep_b", "rep_c2"): "dim_philosophy",
    ("rep_b", "rep_c3"): "dim_philosophy",
    ("rep_b", "rep_d"): "dim_philosophy",   # two different non-standard reps

    # Mixed: window + feature change together
    ("rep_a", "rep_c1"): "dim_mixed",       # window change + MaxDD drop
    ("rep_a_unscaled", "rep_c1"): "dim_mixed",
    ("rep_c1", "rep_c2"): "dim_mixed",      # window change + VaR drop
}

# rep_e pairs are classified separately (external/structurally different signal)
def _classify_pair(a: str, b: str) -> str:
    """Return the dimension label for a representation pair."""
    key = tuple(sorted([a, b]))
    if "rep_e" in key:
        return "dim_external"
    return _PAIR_DIMENSION.get(key, "dim_other")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_decomposition(outputs_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute dimension decomposition and variance decomposition.

    Parameters
    ----------
    outputs_dir : Path
        Top-level outputs directory (contains per-asset subdirectories).

    Returns
    -------
    decomp_df : pd.DataFrame
        Mean and 95% CI of ARI per (asset, model, dimension_category).
    var_df : pd.DataFrame
        Variance decomposition per (asset, model):
        between-pair variance, within-pair seed variance, eta-squared.
    """
    decomp_rows: List[Dict] = []
    var_rows: List[Dict] = []

    # Support both flat layout (outputs/ASSET/plots/) and step-sweep layout
    # (outputs/ASSET/step_21/plots/).  Prefer step_21 (baseline) when present.
    asset_dirs = [
        d for d in outputs_dir.iterdir()
        if d.is_dir() and (
            (d / "plots").exists()
            or (d / "step_21" / "plots").exists()
        )
    ]
    for asset_dir in sorted(asset_dirs):
        asset = asset_dir.name
        if (asset_dir / "step_21" / "plots" / "stability_summary.csv").exists():
            stab_path = asset_dir / "step_21" / "plots" / "stability_summary.csv"
        else:
            stab_path = asset_dir / "plots" / "stability_summary.csv"
        if not stab_path.exists():
            logger.warning("Missing stability_summary.csv for %s; skipping.", asset)
            continue

        try:
            df = pd.read_csv(stab_path, low_memory=False)
        except Exception as exc:
            logger.warning("Could not read %s: %s", stab_path, exc)
            continue

        required = {"model", "seed", "rep_a", "rep_b", "ari"}
        if not required.issubset(df.columns):
            logger.warning("stability_summary.csv for %s missing columns %s", asset, required - set(df.columns))
            continue

        # Cross-rep rows only (rep_a != rep_b)
        cross = df.dropna(subset=["rep_a", "rep_b", "ari"]).copy()
        cross = cross[cross["rep_a"].astype(str) != cross["rep_b"].astype(str)]
        cross["ari"] = pd.to_numeric(cross["ari"], errors="coerce")
        cross["seed"] = pd.to_numeric(cross["seed"], errors="coerce")
        cross = cross.dropna(subset=["ari", "seed"])

        if cross.empty:
            continue

        cross["dimension"] = cross.apply(
            lambda r: _classify_pair(str(r["rep_a"]), str(r["rep_b"])), axis=1
        )
        cross["pair_key"] = cross.apply(
            lambda r: "_vs_".join(sorted([str(r["rep_a"]), str(r["rep_b"])])), axis=1
        )

        for model, mdf in cross.groupby("model"):
            # --- Dimension decomposition ---
            for dim, ddf in mdf.groupby("dimension"):
                vals = ddf["ari"].dropna().values
                n = len(vals)
                mean = float(np.mean(vals)) if n > 0 else float("nan")
                sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
                ci95 = 1.96 * sem if np.isfinite(sem) else float("nan")
                decomp_rows.append({
                    "asset": asset,
                    "model": str(model),
                    "dimension": str(dim),
                    "mean_ari": mean,
                    "ci95": ci95,
                    "n_obs": int(n),
                })

            # --- Variance decomposition ---
            # For each (pair_key), compute mean and variance of ARI across seeds.
            pair_stats = (
                mdf.groupby("pair_key")["ari"]
                .agg(["mean", "var", "count"])
                .rename(columns={"mean": "pair_mean", "var": "pair_var", "count": "pair_n"})
                .reset_index()
            )
            pair_stats = pair_stats.dropna(subset=["pair_mean"])
            if pair_stats.empty:
                continue

            overall_mean = float(pair_stats["pair_mean"].mean())
            # Between-pair: variance of pair-level means
            between_var = float(pair_stats["pair_mean"].var(ddof=1)) if len(pair_stats) > 1 else 0.0
            # Within-pair/seed: mean of per-pair seed variances
            within_var = float(pair_stats["pair_var"].mean()) if not pair_stats["pair_var"].isna().all() else 0.0
            total_var = between_var + within_var
            eta_sq = between_var / total_var if total_var > 1e-12 else float("nan")

            var_rows.append({
                "asset": asset,
                "model": str(model),
                "overall_mean_ari": overall_mean,
                "between_pair_var": between_var,
                "within_pair_seed_var": within_var,
                "total_var": total_var,
                "eta_squared": eta_sq,
                "n_pairs": int(len(pair_stats)),
            })

    decomp_df = pd.DataFrame(decomp_rows)
    var_df = pd.DataFrame(var_rows)
    return decomp_df, var_df


def _dim_label(dim: str) -> str:
    return {
        "dim_std": "Standardization",
        "dim_feature": "Feature subset (same windows)",
        "dim_window": "Window length",
        "dim_estimator": "Volatility estimator (GARCH vs rolling)",
        "dim_philosophy": "Feature philosophy (skew/stab vs risk)",
        "dim_mixed": "Window + feature combined",
        "dim_external": "External signal (VIX vs rolling, GSPC only)",
        "dim_other": "Other",
    }.get(dim, dim)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    outputs_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
    if not outputs_dir.exists():
        print(f"outputs_dir not found: {outputs_dir}")
        return

    decomp_df, var_df = run_decomposition(outputs_dir)

    if decomp_df.empty:
        print("No decomposition data found.")
    else:
        out_decomp = outputs_dir / "repr_decomp_summary.csv"
        decomp_df.to_csv(out_decomp, index=False)
        print(f"Dimension decomposition saved to {out_decomp}")
        print()
        # Pretty-print global mean per dimension (averaged across assets and models)
        g = decomp_df.groupby("dimension")["mean_ari"].mean().sort_values()
        print("Mean ARI by variation dimension (across assets and models):")
        for dim, val in g.items():
            print(f"  {_dim_label(str(dim)):<45} {val:.3f}")
        print()

    if var_df.empty:
        print("No variance decomposition data found.")
    else:
        out_var = outputs_dir / "repr_variance_decomp.csv"
        var_df.to_csv(out_var, index=False)
        print(f"Variance decomposition saved to {out_var}")
        print()
        print("Eta-squared (representation effect / total ARI variance):")
        for _, r in var_df.iterrows():
            print(
                f"  {r['asset']:<15} {r['model']:<5} "
                f"eta^2={r['eta_squared']:.3f}  "
                f"(between={r['between_pair_var']:.4f}  "
                f"within={r['within_pair_seed_var']:.4f})"
            )


if __name__ == "__main__":
    main()
