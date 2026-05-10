from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import json
import logging
import sys

import pandas as pd

from src.visualization.plots import (
    plot_ari_gap_distribution_from_key_results,
    plot_disagreement_timeseries,
    plot_pairwise_matrix_heatmap,
    plot_model_split_grouped_bar_from_key_results,
    plot_representation_failure_matrix,
)
from src.core.utils import reps_from_cfg


logger = logging.getLogger(__name__)


def main(cfg: Optional[Dict] = None) -> None:
    project_dir = Path(__file__).resolve().parent.parent.parent
    outputs_dir = project_dir / "outputs"
    raw_dir = project_dir / "data"

    if cfg is None:
        cfg_path = project_dir / "config.yaml"
        if cfg_path.exists():
            try:
                import yaml

                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = None
    if not isinstance(cfg, dict):
        logger.error("posthoc_figs requires config context (cfg or ROOT/config.yaml).")
        sys.exit(1)
    all_reps_from_cfg = reps_from_cfg(cfg)
    # 1) ARI gap distribution (temporal-minus-cross) from key_results_all_assets.csv.
    key_results_path = outputs_dir / "key_results_all_assets.csv"
    if key_results_path.exists():
        key_df = pd.read_csv(key_results_path)
        plot_ari_gap_distribution_from_key_results(
            key_df,
            outputs_dir / "fig_ari_gap_distribution.png",
            title="Temporal-minus-cross ARI across assets and models",
        )
        # Figure 2: grouped bar of cross-rep vs temporal ARI by model class.
        plot_model_split_grouped_bar_from_key_results(
            key_df,
            outputs_dir / "fig_ari_gap_by_model.png",
            title="Cross-representation vs temporal ARI by model class",
        )

    # 2) Pairwise ARI heatmap across representations for a representative asset.
    # Use GSPC (S&P 500) under the baseline configuration: step=21, K=3, model=gmm.
    asset = "GSPC"
    rep_stab_path = outputs_dir / asset / "step_21" / "results" / "rep_stability.json"
    if rep_stab_path.exists():
        payload = json.loads(rep_stab_path.read_text(encoding="utf-8"))
        records = payload.get("rep_stability", [])
        if records:
            df = pd.DataFrame(records)
            # Baseline configuration: K=3, window=252, model=gmm.
            mask = (df.get("K") == 3) & (df.get("window") == 252) & (df.get("model") == "gmm")
            pairs = df[mask].copy()
            if not pairs.empty:
                plot_pairwise_matrix_heatmap(
                    pairs,
                    outputs_dir / "fig_pairwise_ari_heatmap.png",
                    value_col="ari",
                    title="Pairwise ARI across representations (GMM, S&P 500, K=3)",
                )

    # 3) Representation conflict / failure matrix figure for a representative asset.
    try:
        asset = "GSPC"
        model = "hmm"
        k = 3
        w = 252
        target_start = pd.Timestamp("2020-02-01")
        target_end = pd.Timestamp("2020-06-30")
        step_dir = outputs_dir / asset / "step_21"
        logger.info(
            "posthoc_figs: attempting failure-matrix plot; asset=%s model=%s K=%d W=%d",
            asset,
            model,
            k,
            w,
        )
        # Load price series.
        price_path = raw_dir / f"{asset}.csv"
        if not price_path.exists():
            logger.warning("posthoc_figs: price file not found: %s", price_path)
            return
        price_df = pd.read_csv(price_path, parse_dates=["Date"])
        price_df = price_df.sort_values("Date")
        if "Adj Close" in price_df.columns:
            price = price_df.set_index("Date")["Adj Close"]
        else:
            # Fallback: first numeric column other than Date.
            candidates = [c for c in price_df.columns if c != "Date"]
            price = price_df.set_index("Date")[candidates[0]]
        logger.info("posthoc_figs: loaded price series with %d rows", len(price))

        # Helper to load a single representation's hard states.
        def _load_states(rep_name: str) -> pd.Series | None:
            p = step_dir / "results" / rep_name / f"windows_states_hard_{model}.csv"
            if not p.exists():
                logger.warning("posthoc_figs: states file missing for %s: %s", rep_name, p)
                return None
            df = pd.read_csv(p, parse_dates=["date"])
            if not {"K", "W", "seed", "roll", "date", "state"}.issubset(df.columns):
                logger.warning("posthoc_figs: missing required columns in %s", p)
                return None
            mask = (df["K"] == k) & (df["W"] == w)
            sub = df[mask].copy()
            if sub.empty:
                logger.warning(
                    "posthoc_figs: no rows after K/W filter for %s in %s", rep_name, p
                )
                return None
            # Use a single (seed, roll) slice to avoid mixing heterogeneous runs.
            seeds_present = sorted(int(x) for x in pd.to_numeric(sub["seed"], errors="coerce").dropna().unique())
            if not seeds_present:
                logger.warning("posthoc_figs: no valid seed values for %s in %s", rep_name, p)
                return None
            chosen_seed = int(seeds_present[0])
            sub_seed = sub[pd.to_numeric(sub["seed"], errors="coerce") == chosen_seed].copy()
            if sub_seed.empty:
                return None
            roll_ranges = (
                sub_seed.groupby("roll")["date"]
                .agg(["min", "max"])
                .reset_index()
            )
            if roll_ranges.empty:
                logger.warning("posthoc_figs: no valid roll values for %s in %s", rep_name, p)
                return None
            roll_ranges["target_overlap"] = (
                (roll_ranges["max"] >= target_start) & (roll_ranges["min"] <= target_end)
            )
            if bool(roll_ranges["target_overlap"].any()):
                cand = roll_ranges[roll_ranges["target_overlap"]].copy()
                # Prefer the roll whose center is closest to the target interval center.
                center = target_start + (target_end - target_start) / 2
                cand["center"] = cand["min"] + (cand["max"] - cand["min"]) / 2
                cand["dist"] = (cand["center"] - center).abs()
                chosen_roll = str(cand.sort_values(["dist", "roll"]).iloc[0]["roll"])
            else:
                # Fallback: pick latest roll only when target interval is unavailable.
                chosen_roll = str(sorted(sub_seed["roll"].dropna().astype(str).unique())[-1])
            sub = sub_seed[sub_seed["roll"].astype(str) == chosen_roll].copy()
            logger.info(
                "posthoc_figs: selected seed=%d roll=%s for %s", chosen_seed, chosen_roll, rep_name
            )
            s = sub.set_index("date")["state"]
            logger.info(
                "posthoc_figs: loaded states for %s with %d dates", rep_name, len(s)
            )
            return s.sort_index()

        states_a = _load_states("rep_a")
        states_b = _load_states("rep_c1")
        if states_a is None or states_b is None:
            logger.warning(
                "posthoc_figs: states_a or states_b is None; skipping failure-matrix plot."
            )
        else:
            logger.info(
                "posthoc_figs: calling plot_representation_failure_matrix; len_a=%d len_b=%d",
                len(states_a),
                len(states_b),
            )
            plot_representation_failure_matrix(
                price=price,
                states_a=states_a,
                states_b=states_b,
                conflict_path=outputs_dir / "fig_representation_conflict.png",
                matrix_path=outputs_dir / "fig_representation_failure_matrix.png",
                start=str(target_start.date()),
                end=str(target_end.date()),
            )
    except Exception:
        # This figure is optional; failures should not break the posthoc pipeline.
        logger.exception("posthoc_figs: failure-matrix plot failed with exception.")
        import traceback

        traceback.print_exc()

    # 4) Disagreement-rate time series across all representations (full sample).
    try:
        asset = "GSPC"
        model = "hmm"
        k = 3
        w = 252
        step_dir = outputs_dir / asset / "step_21"
        # Source rep list from config.yaml (hardcoded list previously dropped
        # rep_e from this figure). Missing per-rep files are tolerated below.
        all_reps = list(all_reps_from_cfg)

        # Load price
        price_path = raw_dir / f"{asset}.csv"
        if price_path.exists():
            price_df = pd.read_csv(price_path, parse_dates=["Date"]).sort_values("Date")
            if "Adj Close" in price_df.columns:
                price = price_df.set_index("Date")["Adj Close"]
            else:
                price = price_df.set_index("Date")[
                    [c for c in price_df.columns if c != "Date"][0]
                ]

            # Build hard_map for all available reps, using the latest roll per rep
            hard_map: dict[tuple, pd.Series] = {}
            seed = 1
            for rep in all_reps:
                p = step_dir / "results" / rep / f"windows_states_hard_{model}.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p, parse_dates=["date"])
                if not {"K", "W", "seed", "roll", "date", "state"}.issubset(df.columns):
                    continue
                sub = df[(df["K"] == k) & (df["W"] == w)].copy()
                sub = sub[pd.to_numeric(sub["seed"], errors="coerce") == seed]
                if sub.empty:
                    continue
                # Use all rolls; pick the one with the most dates
                for roll_name, g in sub.groupby("roll"):
                    s = g.set_index("date")["state"].sort_index()
                    hard_map[(rep, model, seed, str(roll_name))] = s

            if hard_map:
                plot_disagreement_timeseries(
                    hard_map=hard_map,
                    price=price,
                    out_path=outputs_dir / "fig_disagreement_timeseries.png",
                    reps=all_reps,
                    model=model,
                    seed=seed,
                    smooth_window=63,
                    stress_periods=[
                        ("2020-02-19", "2020-06-08", "COVID-19"),
                        ("2022-01-03", "2022-10-14", "2022 inflation"),
                    ],
                )
            else:
                logger.warning("posthoc_figs: no hard_map entries for disagreement plot.")
        else:
            logger.warning("posthoc_figs: price file not found for disagreement plot: %s", price_path)
    except Exception:
        logger.exception("posthoc_figs: disagreement time-series plot failed.")
        import traceback
        traceback.print_exc()

    # 6) Representation-dimension and variance decomposition.
    _run_repr_decomp(outputs_dir)


def _run_repr_decomp(outputs_dir: Path) -> None:
    """Run representation-dimension and variance decomposition."""
    try:
        from src.experiments.posthoc_repr_decomp import run_decomposition
        decomp_df, var_df = run_decomposition(outputs_dir)
        if not decomp_df.empty:
            decomp_df.to_csv(outputs_dir / "repr_decomp_summary.csv", index=False)
            logger.info("repr_decomp_summary.csv written (%d rows)", len(decomp_df))
        if not var_df.empty:
            var_df.to_csv(outputs_dir / "repr_variance_decomp.csv", index=False)
            logger.info("repr_variance_decomp.csv written (%d rows)", len(var_df))
    except Exception:
        logger.exception("posthoc_figs: representation decomposition failed.")


if __name__ == "__main__":
    main()
    outputs_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
    _run_repr_decomp(outputs_dir)

