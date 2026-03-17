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


def build_representation_single(price: pd.Series, rep: RepConfig) -> pd.DataFrame:
    windows = rep.windows or {}
    vol_window = int(windows.get("vol_window", 20))
    drawdown_window = int(windows.get("drawdown_window", 60))
    tail_window = int(windows.get("tail_window", 60))
    tail_alpha = float(windows.get("tail_alpha", 0.05))
    skew_window = int(windows.get("skew_window", 60))
    stability_window = int(windows.get("stability_window", 60))

    r = _compute_log_returns(price)
    vol = _compute_volatility(r, vol_window).rename("volatility")
    dd = _compute_drawdown(price, drawdown_window).rename("drawdown")
    mdd = _compute_max_drawdown_window(price, drawdown_window).rename("max_drawdown_window")
    var, cvar = _compute_tail_risk(r, tail_window, tail_alpha)
    skew = _compute_realized_skew(r, skew_window)
    stab = _compute_stability(vol, stability_window)

    all_feats = pd.concat([vol, dd, mdd, var, cvar, skew, stab], axis=1)

    wanted = [f for f in rep.features if f in all_feats.columns]
    if len(wanted) < len(rep.features):
        missing = [f for f in rep.features if f not in all_feats.columns]
        logger.debug("Rep %s: missing features ignored: %s", rep.name, missing)

    df = all_feats[wanted].copy()
    if rep.drop_features:
        df = df.drop(columns=[c for c in rep.drop_features if c in df.columns])

    df = _apply_standardization(df, rep.standardization)
    return df

