from __future__ import annotations

from pathlib import Path
import json
import logging

import pandas as pd

from plots import (
    plot_ari_gap_distribution_from_key_results,
    plot_pairwise_matrix_heatmap,
    plot_model_split_grouped_bar_from_key_results,
    plot_representation_failure_matrix,
)


logger = logging.getLogger(__name__)


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    outputs_dir = project_dir / "outputs"
    raw_dir = project_dir / "data"
    paper_dir = project_dir / "paper"

    # 1) ARI gap distribution (temporal-minus-cross) from key_results_all_assets.csv.
    key_results_path = outputs_dir / "key_results_all_assets.csv"
    if key_results_path.exists():
        key_df = pd.read_csv(key_results_path)
        plot_ari_gap_distribution_from_key_results(
            key_df,
            paper_dir / "fig_ari_gap_distribution.png",
            title="Temporal-minus-cross ARI across assets and models",
        )
        # Figure 2: grouped bar of cross-rep vs temporal ARI by model class.
        plot_model_split_grouped_bar_from_key_results(
            key_df,
            paper_dir / "fig_ari_gap_by_model.png",
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
                    paper_dir / "fig_pairwise_ari_heatmap.png",
                    value_col="ari",
                    title="Pairwise ARI across representations (GMM, S&P 500, K=3)",
                )

    # 3) Representation conflict / failure matrix figure for a representative asset.
    try:
        asset = "GSPC"
        model = "gmm"
        k = 3
        w = 252
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
            mask = (df.get("K") == k) & (df.get("W") == w)
            sub = df[mask].copy()
            if sub.empty:
                logger.warning(
                    "posthoc_figs: no rows after K/W filter for %s in %s", rep_name, p
                )
                return None
            # Prefer the latest roll per date (drop duplicates by date).
            sub = sub.sort_values(["date", "roll"])
            sub = sub.drop_duplicates(subset=["date"], keep="last")
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
                conflict_path=paper_dir / "fig_representation_conflict.png",
                matrix_path=paper_dir / "fig_representation_failure_matrix.png",
                start="2020-02-01",
                end="2020-06-30",
            )
    except Exception:
        # This figure is optional; failures should not break the posthoc pipeline.
        logger.exception("posthoc_figs: failure-matrix plot failed with exception.")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

