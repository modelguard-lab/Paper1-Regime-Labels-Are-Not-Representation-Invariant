from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.core.features import RepConfig
from src.visualization.plots import plot_synth_ari_vs_step_by_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthParams:
    T: int = 2200
    K: int = 3
    persistence_p: float = 0.97
    sigmas: Tuple[float, ...] = (0.006, 0.012, 0.024)  # daily vol levels (log-return scale)
    drift_alpha: float = 0.6  # multiplicative drift in volatility scale
    start_date: str = "2000-01-03"


def _make_transition(K: int, p: float) -> np.ndarray:
    K = int(K)
    p = float(p)
    if K <= 1:
        return np.ones((1, 1), dtype=float)
    off = (1.0 - p) / float(K - 1)
    P = np.full((K, K), off, dtype=float)
    np.fill_diagonal(P, p)
    return P


def _sample_markov_states(rng: np.random.Generator, T: int, P: np.ndarray) -> np.ndarray:
    K = int(P.shape[0])
    s = np.empty(int(T), dtype=int)
    s[0] = int(rng.integers(0, K))
    for t in range(1, int(T)):
        s[t] = int(rng.choice(K, p=P[s[t - 1], :]))
    return s


def generate_synthetic_price_and_truth(
    *,
    params: SynthParams,
    seed: int,
    drift_alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a synthetic price series with known latent states.

    Returns:
    - prices_df with columns: Date, Close
    - truth_df with columns: Date, state_true
    """
    rng = np.random.default_rng(int(seed))
    T = int(params.T)
    K = int(params.K)
    P = _make_transition(K, float(params.persistence_p))
    s_true = _sample_markov_states(rng, T=T, P=P)

    # Drift in volatility scale (slow, deterministic).
    t = np.arange(T, dtype=float)
    drift = 1.0 + float(drift_alpha) * (t / float(max(1, T - 1)))
    base_sigmas = np.array(list(params.sigmas), dtype=float)
    sigma_t = base_sigmas[s_true] * drift

    r = rng.normal(loc=0.0, scale=sigma_t, size=T).astype(float)  # log-returns
    price0 = 100.0
    price = price0 * np.exp(np.cumsum(r))

    idx = pd.bdate_range(start=params.start_date, periods=T)
    prices_df = pd.DataFrame({"Date": idx, "Close": price})
    truth_df = pd.DataFrame({"Date": idx, "state_true": s_true})
    return prices_df, truth_df


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _default_synth_reps() -> List[RepConfig]:
    """
    Two 'reasonable' rolling representations that differ only in smoothing horizons.

    Rep-U: shorter horizons (more reactive).
    Rep-L: longer horizons (more averaged).
    """
    rep_u = RepConfig(
        name="rep_u",
        features=["volatility", "drawdown", "var", "cvar"],
        windows={
            "vol_window": 20,
            "drawdown_window": 60,
            "tail_window": 60,
            "tail_alpha": 0.05,
        },
        standardization={"mode": "rolling_zscore", "window": 120},
    )
    rep_l = RepConfig(
        name="rep_l",
        features=["volatility", "drawdown", "var", "cvar"],
        windows={
            "vol_window": 60,
            "drawdown_window": 120,
            "tail_window": 120,
            "tail_alpha": 0.05,
        },
        standardization={"mode": "rolling_zscore", "window": 240},
    )
    return [rep_u, rep_l]


def _synth_cfg_from_base(cfg: Dict, reps: List[RepConfig], *, K: int, W: int, step: int, seeds: List[int]) -> Dict:
    """
    Create a minimal cfg for running the existing pipeline on SYNTH without touching the main config.
    """
    synth_cfg = dict(cfg)
    synth_cfg["assets"] = ["SYNTH"]
    synth_cfg["representations"] = {
        r.name: {
            "features": list(r.features),
            "windows": dict(r.windows or {}),
            "standardization": dict(r.standardization or {}),
        }
        for r in reps
    }
    synth_cfg["grid"] = dict(synth_cfg.get("grid") or {})
    synth_cfg["grid"]["n_states"] = [int(K)]
    synth_cfg["grid"]["windows"] = [int(W)]
    synth_cfg["grid"]["step"] = int(step)
    synth_cfg["grid"]["seeds"] = [int(s) for s in seeds]
    # Keep runtime bounded (synthetic is just a sanity check).
    synth_cfg["grid"]["n_jobs"] = int(min(int((synth_cfg["grid"].get("n_jobs") or 4)), 6))
    return synth_cfg


def _seed_level_means_from_stability_summary(stability_csv: Path) -> pd.DataFrame:
    """
    Read plots/stability_summary.csv and compute seed-level mean ARI for cross-rep and temporal.
    """
    st = pd.read_csv(stability_csv, low_memory=False)
    if st.empty or "seed" not in st.columns or "model" not in st.columns or "ari" not in st.columns:
        return pd.DataFrame()
    st["ari"] = pd.to_numeric(st["ari"], errors="coerce")
    st["seed"] = pd.to_numeric(st["seed"], errors="coerce")

    out = []
    # Cross-rep: rep_a/rep_b present
    if {"rep_a", "rep_b"}.issubset(st.columns):
        cross = st.dropna(subset=["rep_a", "rep_b", "ari"]).copy()
        if not cross.empty:
            cross = cross[cross["rep_a"] != cross["rep_b"]]
            g = cross.groupby(["model", "seed"], as_index=False)["ari"].mean()
            for _, r in g.iterrows():
                out.append(
                    {
                        "metric": "cross_rep_ari_seed_mean",
                        "model": str(r["model"]),
                        "seed": int(r["seed"]),
                        "value": float(r["ari"]),
                    }
                )

    # Temporal: roll_a/roll_b present
    if {"roll_a", "roll_b"}.issubset(st.columns):
        tmp = st.dropna(subset=["roll_a", "roll_b", "ari"]).copy()
        if not tmp.empty:
            g = tmp.groupby(["model", "seed"], as_index=False)["ari"].mean()
            for _, r in g.iterrows():
                out.append(
                    {
                        "metric": "temporal_ari_seed_mean",
                        "model": str(r["model"]),
                        "seed": int(r["seed"]),
                        "value": float(r["ari"]),
                    }
                )

    return pd.DataFrame(out)


def _mean_ci95(x: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return float("nan"), float("nan")
    mean = float(x.mean())
    if x.shape[0] <= 1:
        return mean, float("nan")
    std = float(x.std(ddof=1))
    ci = 1.96 * (std / math.sqrt(int(x.shape[0])))
    return mean, float(ci)


def run_synthetic_sanity_check(cfg: Dict, outputs_dir: Path) -> None:
    """
    Run a one-page synthetic sanity check and write artifacts under outputs/synthetic_sanity/.
    """
    from src.workflows.pipeline import _run_single_asset  # local import to avoid circular import at module load

    synth_cfg = cfg.get("synthetic_sanity") or {}
    if not bool(synth_cfg.get("enabled", False)):
        return

    out_root = Path(outputs_dir) / "synthetic_sanity"
    out_root.mkdir(parents=True, exist_ok=True)

    # Parameters
    params = SynthParams(
        T=int(synth_cfg.get("T", 2200)),
        K=int(synth_cfg.get("K", 3)),
        persistence_p=float(synth_cfg.get("persistence_p", 0.97)),
        drift_alpha=float(synth_cfg.get("drift_alpha", 0.6)),
    )
    W = int(synth_cfg.get("W", 252))
    steps = [int(x) for x in (synth_cfg.get("steps") or [21, 63, 126, 252])]
    seeds = [int(x) for x in (synth_cfg.get("seeds") or list(range(1, 11)))]

    raw_dir = Path(cfg.get("raw_dir", cfg.get("data", {}).get("raw_dir", "data")))
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Generate drift dataset (main sanity check)
    prices_df, truth_df = generate_synthetic_price_and_truth(
        params=params, seed=int(synth_cfg.get("data_seed", 123)), drift_alpha=float(params.drift_alpha)
    )
    _write_csv(prices_df, raw_dir / "SYNTH.csv")
    _write_csv(truth_df, out_root / "truth_states.csv")

    reps = _default_synth_reps()

    logger.info("Synthetic sanity check: running step sweep (SYNTH). out=%s", out_root)
    seed_metric_rows: List[Dict] = []

    for step in steps:
        cfg_step = _synth_cfg_from_base(cfg, reps, K=params.K, W=W, step=int(step), seeds=seeds)
        out_dir = out_root / f"step_{int(step)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_single_asset("SYNTH", cfg_step, outputs_dir, out_dir_override=out_dir)

        stability_csv = out_dir / "plots" / "stability_summary.csv"
        if stability_csv.exists():
            d = _seed_level_means_from_stability_summary(stability_csv)
            if not d.empty:
                d.insert(0, "step", int(step))
                seed_metric_rows.append(d)

    if not seed_metric_rows:
        logger.warning("Synthetic sanity check produced no seed-level metrics; skipping plots.")
        return

    seed_df = pd.concat(seed_metric_rows, axis=0, ignore_index=True)
    seed_df.to_csv(out_root / "seed_metrics.csv", index=False)

    # CI summary (by step × model × metric)
    rows = []
    for (step, model, metric), g in seed_df.groupby(["step", "model", "metric"], sort=False):
        mean, ci = _mean_ci95(g["value"])
        rows.append(
            {
                "step": int(step),
                "model": str(model),
                "metric": str(metric),
                "mean": mean,
                "ci95": ci,
                "n_seeds": int(g["seed"].nunique()),
            }
        )
    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(out_root / "ci_summary.csv", index=False)

    # Make a compact plot for Appendix (2 panels; models as lines)
    fig_df = (
        ci_df.pivot_table(index=["step", "model"], columns="metric", values=["mean", "ci95"], aggfunc="first")
        .reset_index()
    )
    # Flatten columns
    fig_df.columns = [
        ("_".join([c for c in col if c]).strip("_") if isinstance(col, tuple) else str(col))
        for col in fig_df.columns
    ]
    fig_df = fig_df.rename(
        columns={
            "mean_cross_rep_ari_seed_mean": "cross_mean",
            "ci95_cross_rep_ari_seed_mean": "cross_ci95",
            "mean_temporal_ari_seed_mean": "temporal_mean",
            "ci95_temporal_ari_seed_mean": "temporal_ci95",
        }
    )
    fig_out = out_root / "synth_sanity_ari_vs_step.png"
    plot_synth_ari_vs_step_by_model(fig_df, fig_out)

    # Optional: recovery sanity (single baseline; ARI to ground truth) for alpha=0 vs alpha>0.
    # Keep it lightweight: one step only.
    recovery_rows = []
    for drift_alpha in [0.0, float(params.drift_alpha)]:
        prices_df2, truth_df2 = generate_synthetic_price_and_truth(
            params=params, seed=int(synth_cfg.get("data_seed", 123)), drift_alpha=float(drift_alpha)
        )
        asset_name = "SYNTH_NODRIFT" if drift_alpha == 0.0 else "SYNTH"
        _write_csv(prices_df2, raw_dir / f"{asset_name}.csv")
        truth = truth_df2.set_index("Date")["state_true"]

        cfg_base = _synth_cfg_from_base(cfg, reps[:1], K=params.K, W=W, step=int(steps[0]), seeds=seeds[:5])
        out_dir = out_root / ("recovery_nodrift" if drift_alpha == 0.0 else "recovery_drift")
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_single_asset(asset_name, cfg_base, outputs_dir, out_dir_override=out_dir)

        # Use rep_u hard states for recovery ARI; ARI is permutation-invariant.
        for model in ["gmm", "hmm"]:
            p_hard = out_dir / "results" / "rep_u" / f"windows_states_hard_{model}.csv"
            if not p_hard.exists():
                continue
            df = pd.read_csv(p_hard, parse_dates=["date"])
            df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
            df["state"] = pd.to_numeric(df["state"], errors="coerce")
            if df.empty:
                continue
            # Per-seed mean ARI across rolls.
            per_seed = []
            for seed, g in df.groupby("seed", sort=False):
                if not np.isfinite(seed):
                    continue
                aris = []
                for roll, gg in g.groupby("roll", sort=False):
                    z = pd.Series(gg["state"].values, index=pd.to_datetime(gg["date"]))
                    common = z.dropna().index.intersection(truth.dropna().index)
                    if len(common) == 0:
                        continue
                    aris.append(float(adjusted_rand_score(truth.loc[common].astype(int), z.loc[common].astype(int))))
                if aris:
                    per_seed.append(float(np.mean(aris)))
            if per_seed:
                mean, ci = _mean_ci95(pd.Series(per_seed))
                recovery_rows.append(
                    {
                        "drift_alpha": float(drift_alpha),
                        "model": str(model),
                        "recovery_ari_mean": mean,
                        "recovery_ari_ci95": ci,
                        "n_seeds": int(len(per_seed)),
                    }
                )

    if recovery_rows:
        pd.DataFrame(recovery_rows).to_csv(out_root / "recovery_table.csv", index=False)
    logger.info("Synthetic sanity check finished. Figure=%s", fig_out)

