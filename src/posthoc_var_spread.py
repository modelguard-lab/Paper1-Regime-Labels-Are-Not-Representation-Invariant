"""
Post-hoc analysis: regime-conditional CVaR spread across representations.

Quantifies the downstream economic impact of representation uncertainty by
computing, for each rolling window and date, the CVaR that a risk manager
would estimate conditional on the regime assignment under each representation,
then measuring how much this estimate varies across representations.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import assets_from_cfg, enabled_models_from_cfg, reps_from_cfg

logger = logging.getLogger(__name__)

# REPS/MODELS/ASSETS intentionally not defined at module level: they must be
# sourced from config.yaml. A hardcoded 7-rep list here previously silently
# dropped rep_e from the S&P-500 CVaR spread analysis.
K_DEFAULT = 3
W_DEFAULT = 252
SEED_DEFAULT = 1
ALPHA = 0.05

STRESS_PERIODS = {
    "COVID": ("2020-02-19", "2020-06-08"),
    "2022_inflation": ("2022-01-03", "2022-10-14"),
}


def _state_cvar(returns: pd.Series, states: pd.Series, alpha: float = ALPHA) -> Dict[int, float]:
    """Compute CVaR for each state from in-state returns."""
    result = {}
    for s in sorted(states.unique()):
        idx = states[states == s].index
        r = returns.reindex(idx).dropna()
        if len(r) < 5:
            result[s] = np.nan
        else:
            q = np.quantile(r.values, alpha)
            tail = r.values[r.values <= q]
            result[s] = float(np.mean(tail)) if len(tail) > 0 else float(q)
    return result


def compute_var_spread(
    outputs_dir: Path,
    raw_dir: Path,
    assets: List[str],
    models: List[str],
    reps: List[str],
    K: int = K_DEFAULT,
    W: int = W_DEFAULT,
    seed: int = SEED_DEFAULT,
    alpha: float = ALPHA,
    sample_every: int = 5,
) -> pd.DataFrame:
    """Compute regime-conditional CVaR spread across representations.

    Returns a DataFrame with one row per (asset, model) containing:
    - mean_spread_pp: mean daily CVaR spread in percentage points
    - overall_cvar_pp: unconditional CVaR in percentage points
    - spread_pct_of_cvar: spread as fraction of |unconditional CVaR|
    - stress-period columns when data is available

    assets/models/reps are required (no silent fallback to hardcoded defaults).
    Callers should source them from config.yaml via utils.assets_from_cfg /
    enabled_models_from_cfg / reps_from_cfg.
    """
    results = []

    for asset in assets:
        safe = asset.replace("^", "")
        price_path = raw_dir / f"{safe}.csv"
        if not price_path.exists():
            logger.warning("Price file not found: %s", price_path)
            continue
        pdf = pd.read_csv(price_path, parse_dates=["Date"]).sort_values("Date")
        col = "Adj Close" if "Adj Close" in pdf.columns else pdf.columns[1]
        price = pdf.set_index("Date")[col]
        returns = np.log(price / price.shift(1)).dropna()
        overall_cvar = abs(float(np.quantile(returns.values, alpha)))

        for model in models:
            # Pre-load all reps
            all_states: Dict[str, pd.DataFrame] = {}
            for rep in reps:
                p = outputs_dir / safe / "step_21" / "results" / rep / f"windows_states_hard_{model}.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p, parse_dates=["date"])
                sub = df[(df["K"] == K) & (df["W"] == W) & (df["seed"] == seed)]
                if not sub.empty:
                    all_states[rep] = sub

            if len(all_states) < 2:
                continue

            rolls = sorted(list(all_states.values())[0]["roll"].unique())
            sampled = rolls[::sample_every]

            roll_spreads: List[float] = []
            stress_spreads: Dict[str, List[float]] = {k: [] for k in STRESS_PERIODS}

            for roll_name in sampled:
                cvar_by_rep: Dict[str, pd.Series] = {}
                for rep, sub_full in all_states.items():
                    sub = sub_full[sub_full["roll"] == roll_name]
                    if sub.empty:
                        continue
                    states = sub.set_index("date")["state"].astype(int)
                    sc = _state_cvar(returns, states, alpha)
                    date_cvar = states.map(lambda x, m=sc: m.get(x, np.nan))
                    cvar_by_rep[rep] = date_cvar

                if len(cvar_by_rep) >= 2:
                    cvar_df = pd.DataFrame(cvar_by_rep).dropna()
                    if cvar_df.empty:
                        continue
                    spread = cvar_df.max(axis=1) - cvar_df.min(axis=1)
                    roll_spreads.append(float(spread.mean()))

                    for sp_name, (sp_start, sp_end) in STRESS_PERIODS.items():
                        mask = (cvar_df.index >= sp_start) & (cvar_df.index <= sp_end)
                        if mask.any():
                            stress_spreads[sp_name].append(float(spread[mask].mean()))

            if roll_spreads:
                mean_sp = float(np.mean(roll_spreads))
                max_sp = float(np.max(roll_spreads))
                row = {
                    "asset": asset,
                    "model": model,
                    "mean_spread_pp": mean_sp * 100,
                    "max_spread_pp": max_sp * 100,
                    "overall_cvar_pp": overall_cvar * 100,
                    "spread_pct_of_cvar": mean_sp / overall_cvar * 100,
                    "max_spread_pct_of_cvar": max_sp / overall_cvar * 100,
                    "n_rolls": len(roll_spreads),
                }
                for sp_name in STRESS_PERIODS:
                    if stress_spreads[sp_name]:
                        sp_val = float(np.mean(stress_spreads[sp_name]))
                        row[f"{sp_name}_spread_pp"] = sp_val * 100
                        row[f"{sp_name}_pct_of_cvar"] = sp_val / overall_cvar * 100
                results.append(row)
                logger.info(
                    "%s %s: spread=%.2fpp (%.0f%% of |CVaR|)",
                    asset, model, mean_sp * 100, mean_sp / overall_cvar * 100,
                )

    return pd.DataFrame(results)


def main(cfg: Optional[Dict] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    project = Path(__file__).resolve().parent.parent
    outputs = project / "outputs"
    raw = project / "data"

    if cfg is None:
        cfg_path = project / "config.yaml"
        if cfg_path.exists():
            try:
                import yaml

                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = None
    if not isinstance(cfg, dict):
        logger.error("posthoc_var_spread requires config context (cfg or ROOT/config.yaml).")
        sys.exit(1)

    assets = assets_from_cfg(cfg)
    reps = reps_from_cfg(cfg)
    models = enabled_models_from_cfg(cfg)

    # Average across the first 10 fitting seeds from cfg.grid.seeds. Single-seed
    # spreads are materially biased for HMM (per-seed CV ~ 13%); seed averaging
    # gives the practitioner-relevant point estimate. Backwards compatibility:
    # if cfg.grid.seeds is missing or short, fall back to seed=1 only.
    grid = cfg.get("grid", {}) or {}
    seeds = list(grid.get("seeds") or [1])
    seeds = [int(s) for s in seeds[:10]]
    if not seeds:
        seeds = [1]

    per_seed: List[pd.DataFrame] = []
    for s in seeds:
        df_s = compute_var_spread(outputs, raw, assets=assets, models=models, reps=reps, seed=s)
        if df_s.empty:
            continue
        df_s = df_s.copy()
        df_s["seed"] = s
        per_seed.append(df_s)

    if not per_seed:
        print("No results computed.")
        return
    per_seed_df = pd.concat(per_seed, ignore_index=True)
    per_seed_df.to_csv(outputs / "var_spread_summary_per_seed.csv", index=False)

    # Seed-averaged summary: mean and SD across seeds for each (asset, model).
    metric_cols = [c for c in per_seed_df.columns if c not in {"asset", "model", "seed", "n_rolls"}]
    grouped = per_seed_df.groupby(["asset", "model"], as_index=False)[metric_cols].mean()
    df = grouped

    out_path = outputs / "var_spread_summary.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s (seed-averaged across %d seeds: %s)", out_path, len(seeds), seeds)
    print(f"\nSaved to {out_path}  (mean across {len(seeds)} seeds)")
    print()
    print(df.to_string(index=False, float_format="%.1f"))
    print()

    # Summary
    for model in models:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        print(f"{model.upper()}: mean spread = {sub['spread_pct_of_cvar'].mean():.0f}% of |CVaR|")
        for sp_name in STRESS_PERIODS:
            col = f"{sp_name}_pct_of_cvar"
            if col in sub.columns:
                vals = sub[col].dropna()
                if not vals.empty:
                    print(f"  {sp_name}: {vals.mean():.0f}% of |CVaR|")


if __name__ == "__main__":
    main()
