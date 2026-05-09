"""
AR(1) / GARCH(1,1) stationarity null for cross-representation ARI.

Addresses the reviewer concern that representation-dependence might be an
artefact of clustering any stationary process rather than a structural finding.

Method
------
For each asset we:
  1. Fit AR(1) and GARCH(1,1) DGPs to the observed log-return series.
  2. Simulate N_SIM paths of the same length under each null DGP.
  3. Build two contrasting feature representations on each simulated path
     (rep_a: vol/drawdown/VaR/CVaR with z-score, and rep_c3: same features
     but window 30/90 days; chosen as the tightest same-architecture pair).
  4. Fit a K=3 GMM on each representation (single window = full path).
  5. Compute the pairwise cross-representation ARI.
  6. Compare the resulting null distribution to the observed cross-rep ARI
     read from key_results_all_assets.csv.

Interpretation
--------------
If the null ARI is *also* low (< 0.3), representation sensitivity is a
feature of fitting any finite-sample cluster model to smooth risk signals;
the framing would need to acknowledge this inherent sensitivity.

If the null ARI is *high* (> 0.5), real data specifically produces low
cross-representation agreement, strengthening the empirical claim.

Outputs
-------
outputs/stationarity_null_summary.csv
    One row per (asset, dgp, rep_pair).  Columns: null_mean, null_p5, null_p95,
    null_frac_below_065, obs_mean_ari (from key_results if available).
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSETS = ["BTC-USD", "GLD", "GSPC", "IEF"]
N_SIM = 200
K = 3
RANDOM_SEED = 0
# Minimum path length for a rep with 120-day std window + 90-day tail window
MIN_T = 300

# ---------------------------------------------------------------------------
# Feature helpers (lightweight, single-window, no rolling protocol)
# ---------------------------------------------------------------------------


def _log_ret(price: np.ndarray) -> np.ndarray:
    return np.log(price[1:] / price[:-1])


def _rolling_vol(r: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(r), np.nan)
    for i in range(window - 1, len(r)):
        out[i] = float(np.std(r[i - window + 1 : i + 1], ddof=1)) * np.sqrt(252)
    return out


def _rolling_drawdown(price: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(price), np.nan)
    for i in range(window - 1, len(price)):
        segment = price[i - window + 1 : i + 1]
        out[i] = price[i] / np.max(segment) - 1.0
    return out


def _rolling_cvar(r: np.ndarray, window: int, alpha: float = 0.05) -> np.ndarray:
    out = np.full(len(r), np.nan)
    for i in range(window - 1, len(r)):
        seg = r[i - window + 1 : i + 1]
        q = np.quantile(seg, alpha)
        tail = seg[seg <= q]
        out[i] = float(np.mean(tail)) if len(tail) > 0 else q
    return out


def _rolling_var(r: np.ndarray, window: int, alpha: float = 0.05) -> np.ndarray:
    out = np.full(len(r), np.nan)
    for i in range(window - 1, len(r)):
        out[i] = float(np.quantile(r[i - window + 1 : i + 1], alpha))
    return out


def _rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(window - 1, len(x)):
        seg = x[i - window + 1 : i + 1]
        mu = np.nanmean(seg)
        sd = np.nanstd(seg, ddof=1)
        out[i] = (x[i] - mu) / sd if sd > 1e-12 else 0.0
    return out


def _build_features(price: np.ndarray, vol_w: int, dd_w: int, tail_w: int, zscore_w: int) -> np.ndarray:
    """Return (T, 4) feature matrix: vol, drawdown, VaR, CVaR (z-scored)."""
    r = _log_ret(price)
    # Align lengths: price has T+1 points → returns has T, drawdown on price T+1
    T = len(r)
    price_full = price  # length T+1

    vol = _rolling_vol(r, vol_w)
    dd = _rolling_drawdown(price_full, dd_w)[1:]  # align to returns length
    var_ = _rolling_var(r, tail_w)
    cvar = _rolling_cvar(r, tail_w)

    # Z-score each feature
    feats = np.stack([vol, dd, var_, cvar], axis=1)  # (T, 4)
    for j in range(feats.shape[1]):
        feats[:, j] = _rolling_zscore(feats[:, j], zscore_w)

    return feats


def _fit_gmm_states(feats: np.ndarray, k: int, seed: int) -> Optional[np.ndarray]:
    """Fit GMM and return hard state assignments; None if not enough clean rows."""
    mask = np.all(np.isfinite(feats), axis=1)
    if mask.sum() < k * 10:
        return None
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=1)
    labels = np.full(len(feats), -1, dtype=int)
    labels[mask] = gmm.fit_predict(feats[mask])
    return labels


def _cross_ari(states_a: np.ndarray, states_b: np.ndarray) -> float:
    """Compute ARI on common valid rows."""
    valid = (states_a >= 0) & (states_b >= 0)
    if valid.sum() < 10:
        return np.nan
    return float(adjusted_rand_score(states_a[valid], states_b[valid]))


# ---------------------------------------------------------------------------
# DGP fitting and simulation
# ---------------------------------------------------------------------------


def _fit_ar1(returns: np.ndarray) -> Tuple[float, float, float]:
    """Fit AR(1) via OLS.  Returns (mu, phi, sigma)."""
    y = returns[1:]
    x = returns[:-1]
    n = len(y)
    x_mean, y_mean = x.mean(), y.mean()
    phi = float(np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2))
    mu = float(y_mean - phi * x_mean)
    resid = y - (mu + phi * x)
    sigma = float(np.std(resid, ddof=2))
    return mu, phi, sigma


def _simulate_ar1(mu: float, phi: float, sigma: float, T: int, rng: np.random.Generator) -> np.ndarray:
    r = np.empty(T)
    r[0] = mu / (1 - phi) if abs(phi) < 1 else 0.0
    eps = rng.normal(0.0, sigma, size=T)
    for t in range(1, T):
        r[t] = mu + phi * r[t - 1] + eps[t]
    return r


def _fit_garch11(returns: np.ndarray) -> Optional[Dict]:
    """Fit GARCH(1,1) via arch.  Returns param dict or None on failure."""
    try:
        from arch import arch_model  # type: ignore
    except ImportError:
        logger.warning("arch package not installed; skipping GARCH null.")
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            am = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            params = res.params
            omega = float(params.get("omega", 0.01))
            alpha1 = float(params.get("alpha[1]", 0.05))
            beta1 = float(params.get("beta[1]", 0.9))
            mu = float(params.get("mu", 0.0))
            return {"mu": mu, "omega": omega, "alpha1": alpha1, "beta1": beta1,
                    "scale": 100.0, "init_var": float(np.var(returns * 100))}
        except Exception as exc:
            logger.warning("GARCH fit failed: %s", exc)
            return None


def _simulate_garch11(params: Dict, T: int, rng: np.random.Generator) -> np.ndarray:
    mu = params["mu"]
    omega = params["omega"]
    alpha1 = params["alpha1"]
    beta1 = params["beta1"]
    scale = params["scale"]
    init_var = params["init_var"]

    r = np.empty(T)
    h = init_var
    eps = rng.normal(0, 1, size=T)
    r[0] = (mu + np.sqrt(h) * eps[0]) / scale
    prev_r_sq = r[0] ** 2 * scale ** 2
    for t in range(1, T):
        h = omega + alpha1 * prev_r_sq + beta1 * h
        h = max(h, 1e-10)
        r[t] = (mu + np.sqrt(h) * eps[t]) / scale
        prev_r_sq = (r[t] * scale) ** 2
    return r


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

# Two rep pairs spanning different variation dimensions
_REP_PAIRS: List[Tuple[str, str, int, int, int, int, int, int, int, int]] = [
    # (pair_name, vol_w_a, dd_w_a, tail_w_a, zs_w_a, vol_w_b, dd_w_b, tail_w_b, zs_w_b)
    # pair 1: window only (rep_a vs rep_c1 analog)
    ("rep_a_vs_rep_c1", 20, 60, 60, 120, 30, 90, 90, 120),
    # pair 2: same windows, no standardisation vs z-score (rep_a vs rep_a_unscaled analog)
    # handled separately; just use raw features for rep_b
]


def _compute_pair_aris(
    price_sim: np.ndarray,
    vol_w_a: int, dd_w_a: int, tail_w_a: int, zs_w_a: int,
    vol_w_b: int, dd_w_b: int, tail_w_b: int, zs_w_b: int,
    k: int,
    seed: int,
) -> float:
    fa = _build_features(price_sim, vol_w_a, dd_w_a, tail_w_a, zs_w_a)
    fb = _build_features(price_sim, vol_w_b, dd_w_b, tail_w_b, zs_w_b)
    sa = _fit_gmm_states(fa, k, seed)
    sb = _fit_gmm_states(fb, k, seed + 1)
    if sa is None or sb is None:
        return np.nan
    return _cross_ari(sa, sb)


def run_stationarity_null(
    outputs_dir: Path,
    raw_dir: Path,
    assets: List[str] | None = None,
    n_sim: int = N_SIM,
    k: int = K,
) -> pd.DataFrame:
    """Run the stationarity null and return summary DataFrame."""
    if assets is None:
        assets = ASSETS

    obs_df: Optional[pd.DataFrame] = None
    obs_path = outputs_dir / "key_results_all_assets.csv"
    if obs_path.exists():
        try:
            obs_df = pd.read_csv(obs_path)
        except Exception:
            pass

    rows: List[Dict] = []
    rng_master = np.random.default_rng(RANDOM_SEED)

    for asset in assets:
        safe = asset.replace("^", "")
        price_path = raw_dir / f"{safe}.csv"
        if not price_path.exists():
            logger.warning("Price file not found for %s: %s", asset, price_path)
            continue

        pdf = pd.read_csv(price_path, parse_dates=["Date"]).sort_values("Date")
        col = "Adj Close" if "Adj Close" in pdf.columns else [c for c in pdf.columns if c != "Date"][0]
        price_obs = pdf.set_index("Date")[col].dropna().values.astype(float)
        returns_obs = np.log(price_obs[1:] / price_obs[:-1])
        T = len(returns_obs)

        if T < MIN_T:
            logger.warning("Asset %s too short (%d returns); skipping.", asset, T)
            continue

        logger.info("Asset %s: T=%d returns; fitting DGPs ...", asset, T)

        # Observed mean cross-rep ARI (from key_results if available)
        obs_ari: Optional[float] = None
        if obs_df is not None:
            obs_rows = obs_df[obs_df["asset"].astype(str).str.replace("^", "", regex=False) == safe]
            if not obs_rows.empty and "cross_ari_mean" in obs_rows.columns:
                obs_ari = float(obs_rows["cross_ari_mean"].mean())

        # --- AR(1) null ---
        ar_mu, ar_phi, ar_sigma = _fit_ar1(returns_obs)
        logger.info(
            "  AR(1): mu=%.5f phi=%.3f sigma=%.5f", ar_mu, ar_phi, ar_sigma
        )

        # --- GARCH(1,1) null ---
        garch_params = _fit_garch11(returns_obs)

        for dgp_name in (["ar1"] + (["garch11"] if garch_params is not None else [])):
            pair_aris: List[float] = []
            for sim_idx in range(n_sim):
                rng = np.random.default_rng(rng_master.integers(0, 2**32))
                if dgp_name == "ar1":
                    r_sim = _simulate_ar1(ar_mu, ar_phi, ar_sigma, T, rng)
                else:
                    r_sim = _simulate_garch11(garch_params, T, rng)

                # Convert returns to price (start at 100)
                price_sim = np.empty(T + 1)
                price_sim[0] = 100.0
                price_sim[1:] = 100.0 * np.exp(np.cumsum(r_sim))

                # Pair 1: window variation (rep_a vs rep_c1 analog)
                ari_win = _compute_pair_aris(
                    price_sim,
                    vol_w_a=20, dd_w_a=60, tail_w_a=60, zs_w_a=120,
                    vol_w_b=30, dd_w_b=90, tail_w_b=90, zs_w_b=120,
                    k=k, seed=sim_idx,
                )
                if np.isfinite(ari_win):
                    pair_aris.append(ari_win)

            if not pair_aris:
                continue

            arr = np.array(pair_aris)
            null_mean = float(np.nanmean(arr))
            null_p5 = float(np.nanpercentile(arr, 5))
            null_p95 = float(np.nanpercentile(arr, 95))
            null_frac_below = float(np.mean(arr < 0.65))

            rows.append({
                "asset": asset,
                "dgp": dgp_name,
                "rep_pair": "rep_a_vs_rep_c1",
                "null_mean_ari": null_mean,
                "null_p5": null_p5,
                "null_p95": null_p95,
                "null_frac_below_065": null_frac_below,
                "obs_mean_ari": obs_ari,
                "n_sim": len(arr),
            })
            logger.info(
                "  %s %s: null_mean=%.3f [%.3f, %.3f]  frac<0.65=%.2f  obs=%.3f",
                asset, dgp_name, null_mean, null_p5, null_p95, null_frac_below,
                obs_ari if obs_ari is not None else float("nan"),
            )

    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    project = Path(__file__).resolve().parent.parent
    outputs = project / "outputs"
    raw = project / "data"

    df = run_stationarity_null(outputs, raw)
    if df.empty:
        print("No null results computed.")
        return

    out_path = outputs / "stationarity_null_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}\n")
    print(df.to_string(index=False, float_format="%.3f"))
    print()
    print("Interpretation:")
    print("  null_frac_below_065 = fraction of simulated paths where cross-rep ARI < 0.65")
    print("  If null_mean >> obs_mean_ari → real data has significantly lower agreement")
    print("  than a stationary null, strengthening the main finding.")
    print("  If null_mean ≈ obs_mean_ari → representation sensitivity may be inherent")
    print("  to any cluster model applied to smooth risk features.")


if __name__ == "__main__":
    main()
