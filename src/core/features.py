from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepConfig:
    name: str
    features: List[str]
    windows: Dict[str, int | float]
    drop_features: List[str] | None = None
    standardization: Dict[str, int | str] | None = None
    asset_filter: List[str] | None = None  # None = all assets; non-None = only run for listed tickers


def _compute_log_returns(price: pd.Series) -> pd.Series:
    # Use log returns consistently across all features/metrics.
    return np.log(price / price.shift(1))


def _compute_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)


def _compute_drawdown(price: pd.Series, window: int) -> pd.Series:
    rolling_max = price.rolling(window=window, min_periods=window).max()
    return price / rolling_max - 1.0


def _compute_max_drawdown_window(price: pd.Series, window: int) -> pd.Series:
    def _max_dd(x: np.ndarray) -> float:
        if x.size == 0:
            return np.nan
        running_max = np.maximum.accumulate(x)
        dd = x / running_max - 1.0
        return float(np.min(dd))

    return price.rolling(window=window, min_periods=window).apply(_max_dd, raw=True)


def _compute_tail_risk(returns: pd.Series, window: int, alpha: float) -> tuple[pd.Series, pd.Series]:
    var = returns.rolling(window=window, min_periods=window).quantile(alpha)

    def _cvar(x: np.ndarray) -> float:
        if x.size == 0:
            return np.nan
        cutoff = np.quantile(x, alpha)
        tail = x[x <= cutoff]
        return float(np.mean(tail)) if tail.size else np.nan

    cvar = returns.rolling(window=window, min_periods=window).apply(_cvar, raw=True)
    return var.rename("var"), cvar.rename("cvar")


def _compute_garch_vol(returns: pd.Series) -> pd.Series:
    """Annualised conditional volatility from a GARCH(1,1) model.

    Fits a single GARCH(1,1) to the full return series and extracts the
    one-step-ahead conditional standard deviation, annualised by sqrt(252).
    Falls back to rolling 20-day realised volatility if the GARCH fit fails
    (e.g. too few observations or convergence failure).
    """
    try:
        from arch import arch_model  # optional dependency
    except ImportError as exc:
        raise ImportError(
            "The 'arch' package is required for GARCH features. "
            "Install it with: pip install arch>=5.0"
        ) from exc

    # arch expects returns in percentage points for numerical stability.
    #
    # Observability: fallback events log a "GARCH_FALLBACK" tag so operators can
    # count them per run with `grep -c GARCH_FALLBACK outputs/run.log`. When this
    # fallback fires, rep_d silently degrades to a rolling-vol representation
    # (rep_a-subset), which undermines the "structurally heterogeneous vol
    # estimator" claim; treat non-zero counts as a data-quality signal.
    scaled = returns.dropna() * 100.0
    if len(scaled) < 50:
        logger.warning(
            "GARCH_FALLBACK reason=series_too_short n_obs=%d (rep_d degraded to 20-day rolling vol)",
            len(scaled),
        )
        return (_compute_volatility(returns, 20)).rename("garch_vol")

    try:
        model = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        result = model.fit(disp="off", show_warning=False)
        # conditional_volatility is in the same scale as the input (pct points)
        cond_vol = result.conditional_volatility / 100.0 * np.sqrt(252)
        out = pd.Series(np.nan, index=returns.index, name="garch_vol")
        out.loc[cond_vol.index] = cond_vol.values
        return out
    except Exception:
        logger.warning(
            "GARCH_FALLBACK reason=fit_failed (rep_d degraded to 20-day rolling vol)",
            exc_info=True,
        )
        return (_compute_volatility(returns, 20)).rename("garch_vol")


def _compute_vix_level(vix: pd.Series) -> pd.Series:
    """Raw VIX level, renamed for rep_e consumption.

    Previously this function applied a 120-day rolling z-score internally,
    which (combined with rep_e's `standardization: rolling_zscore(window=120)`
    at the rep level) produced a double-z-score. For matched windows the two
    applications are asymptotically idempotent (z(z(x)) ≈ z(x)), so historical
    numerics are not materially corrupted, but the transformation was
    misleading. We now return the raw level and let the rep-level standardizer
    handle the single z-score pass uniformly with vix_change and vix_percentile.
    """
    return vix.rename("vix_level")


def _compute_vix_change(vix: pd.Series, window: int = 5) -> pd.Series:
    """N-day log-change in VIX (captures direction and speed of fear moves)."""
    return np.log(vix / vix.shift(window)).rename("vix_change")


def _compute_vix_percentile(vix: pd.Series, window: int = 60) -> pd.Series:
    """Rolling empirical CDF of VIX at the current observation.

    For each date t, returns ``sum(x_s <= x_t) / window`` over the trailing
    ``window``-day window. The output is the empirical CDF value at the
    current VIX level within that window: range ``[1/window, 1]`` (the
    minimum value occurs when the current observation is the unique
    minimum of the window). High values indicate extreme stress relative
    to the recent past.
    """
    def _pct(x: np.ndarray) -> float:
        return float(np.sum(x <= x[-1]) / len(x))
    return vix.rolling(window=window, min_periods=window).apply(_pct, raw=True).rename("vix_percentile")


def _compute_realized_skew(log_returns: pd.Series, window: int) -> pd.Series:
    return log_returns.rolling(window=window, min_periods=window).skew().rename("realized_skew")


def _compute_stability(volatility: pd.Series, window: int) -> pd.Series:
    # volatility-of-volatility (lower = more stable)
    return volatility.rolling(window=window, min_periods=window).std().rename("stability")


def _apply_standardization(df: pd.DataFrame, standardization: Dict[str, int | str] | None) -> pd.DataFrame:
    if not standardization:
        return df
    mode = standardization.get("mode", "none")
    if mode == "none":
        return df
    if mode == "rolling_zscore":
        window = int(standardization.get("window", 120))
        mean = df.rolling(window=window, min_periods=window).mean()
        std = df.rolling(window=window, min_periods=window).std()
        return (df - mean) / std
    raise ValueError(f"Unknown standardization mode: {mode}")


def build_representation_single(
    price: pd.Series,
    rep: RepConfig,
    aux: Dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Build a single feature representation.

    Parameters
    ----------
    price : pd.Series
        Adjusted-close price series for the target asset.
    rep : RepConfig
        Representation specification.
    aux : dict, optional
        Auxiliary series keyed by name (e.g. ``{"^VIX": vix_series}``).
        Required when ``rep.features`` contains ``vix_*`` features.
    """
    windows = rep.windows or {}
    vol_window = int(windows.get("vol_window", 20))
    drawdown_window = int(windows.get("drawdown_window", 60))
    tail_window = int(windows.get("tail_window", 60))
    tail_alpha = float(windows.get("tail_alpha", 0.05))
    skew_window = int(windows.get("skew_window", 60))
    stability_window = int(windows.get("stability_window", 60))
    vix_change_window = int(windows.get("vix_change_window", 5))
    vix_pct_window = int(windows.get("vix_pct_window", 60))

    r = _compute_log_returns(price)
    vol = _compute_volatility(r, vol_window).rename("volatility")
    dd = _compute_drawdown(price, drawdown_window).rename("drawdown")
    mdd = _compute_max_drawdown_window(price, drawdown_window).rename("max_drawdown_window")
    var, cvar = _compute_tail_risk(r, tail_window, tail_alpha)
    skew = _compute_realized_skew(r, skew_window)
    stab = _compute_stability(vol, stability_window)

    feat_list = [vol, dd, mdd, var, cvar, skew, stab]

    # GARCH conditional vol; only computed when requested (avoids arch import overhead)
    if "garch_vol" in rep.features:
        gvol = _compute_garch_vol(r)
        feat_list.append(gvol)

    # VIX-based features; require aux["^VIX"] to be provided
    vix_features_needed = {"vix_level", "vix_change", "vix_percentile"}
    if vix_features_needed.intersection(rep.features):
        if aux is None or "^VIX" not in aux:
            raise ValueError(
                f"Rep '{rep.name}' requires VIX features but aux['^VIX'] was not provided."
            )
        vix = aux["^VIX"].reindex(price.index).ffill()
        if "vix_level" in rep.features:
            feat_list.append(_compute_vix_level(vix))
        if "vix_change" in rep.features:
            feat_list.append(_compute_vix_change(vix, vix_change_window))
        if "vix_percentile" in rep.features:
            feat_list.append(_compute_vix_percentile(vix, vix_pct_window))

    all_feats = pd.concat(feat_list, axis=1)

    wanted = [f for f in rep.features if f in all_feats.columns]
    if len(wanted) < len(rep.features):
        missing = [f for f in rep.features if f not in all_feats.columns]
        logger.debug("Rep %s: missing features ignored: %s", rep.name, missing)

    df = all_feats[wanted].copy()
    if rep.drop_features:
        df = df.drop(columns=[c for c in rep.drop_features if c in df.columns])

    df = _apply_standardization(df, rep.standardization)
    return df

