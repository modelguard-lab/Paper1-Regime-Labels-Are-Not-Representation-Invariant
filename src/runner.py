from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

from runtime import set_thread_env_defaults
from runtime import configure_console_logging
from runtime import configure_global_file_logging
import csv

# Limit threads per process early (before numpy/scikit-learn imports).
configure_console_logging()
set_thread_env_defaults(1)

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from data import download_tickers, load_prices
from features import RepConfig, build_representation_single
from metrics import semantic_drift, stability_metrics
from models import fit_gmm, fit_hmm
from plots import (
    plot_ari_vs_step,
    plot_cross_rep_box_by_rep,
    plot_line_by_group,
    plot_ordering_consistency_summary,
    plot_pairwise_matrix_heatmap,
)
from utils import ensure_dir, rolling_slices, safe_name, save_json
from paper_autofill import update_empirical_results_md
from synthetic_sanity import run_synthetic_sanity_check

logger = logging.getLogger(__name__)


def _fmt_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS (rounded)."""
    try:
        s = int(round(float(seconds)))
    except Exception:
        return "NA"
    if s < 0:
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _timing_summary_lines(title: str, totals: Dict[str, float], top_k: int = 12) -> List[str]:
    """Return a compact timing summary line (largest first)."""
    if not totals:
        return [f"{title}: (no timings collected)"]
    items = sorted(((str(k), float(v)) for k, v in totals.items()), key=lambda kv: kv[1], reverse=True)
    shown = items[: max(1, int(top_k))]
    parts = [f"{k}={_fmt_hms(v)}" for k, v in shown]
    more = "" if len(items) <= len(shown) else f" (+{len(items) - len(shown)} more)"
    total_all = sum(v for _, v in items)
    return [f"{title}: total={_fmt_hms(total_all)}; " + ", ".join(parts) + more]

try:
    # Raised when a loky worker is killed (OOM / segfault / OS termination).
    from joblib.externals.loky.process_executor import TerminatedWorkerError
except Exception:  # pragma: no cover
    TerminatedWorkerError = None  # type: ignore[assignment]

def _rmtree_with_retries(path: Path, retries: int = 8, base_sleep_s: float = 0.3) -> None:
    """
    Robust rmtree for Windows.

    Handles transient "directory not empty" / permission errors that can happen
    when a previous run crashed and left temporary shard files behind, or when
    the OS is still releasing file handles.
    """

    path = Path(path)
    if not path.exists():
        return

    def _onerror(func, p, exc_info):  # type: ignore[no-untyped-def]
        try:
            os.chmod(p, 0o666)
        except Exception:
            pass
        try:
            func(p)
        except Exception:
            pass

    last_err: Exception | None = None
    for i in range(int(retries)):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except Exception as e:
            last_err = e
            time.sleep(base_sleep_s * (2**i))
    if last_err is not None:
        raise last_err


def _window_roll_name(i: int) -> str:
    return f"roll_{i:03d}"


def _build_rep_configs(cfg: Dict) -> List[RepConfig]:
    reps: List[RepConfig] = []
    for name, rep in (cfg.get("representations", {}) or {}).items():
        reps.append(
            RepConfig(
                name=name,
                features=rep.get("features", []),
                windows=rep.get("windows", {}) or {},
                drop_features=rep.get("drop_features", None),
                standardization=rep.get("standardization", None),
            )
        )
    if not reps:
        raise ValueError("No representations configured.")
    return reps


def _fit_one(
    model_name: str, X: pd.DataFrame, k: int, seed: int, model_cfg: Dict
) -> Tuple[pd.Series, pd.DataFrame, Dict, Dict]:
    if model_name == "hmm":
        mc = model_cfg.get("hmm", {}) if isinstance(model_cfg, dict) else {}
        res = fit_hmm(
            X,
            n_states=k,
            covariance_type=str(mc.get("covariance_type", "full")),
            n_iter=int(mc.get("n_iter", 200)),
            random_state=seed,
        )
    else:
        mc = model_cfg.get("gmm", {}) if isinstance(model_cfg, dict) else {}
        res = fit_gmm(
            X,
            n_states=k,
            covariance_type=str(mc.get("covariance_type", "full")),
            n_init=int(mc.get("n_init", 5)),
            random_state=seed,
        )
    return res.states_hard, res.states_soft, res.model_params, res.scores


def _fit_slice_collect(
    model_name: str,
    X_values: np.ndarray,
    X_index_values: np.ndarray,
    X_columns: List[str],
    start: int,
    end: int,
    k: int,
    seed: int,
    rep_name: str,
    roll: str,
    w: int,
    model_cfg: Dict,
) -> Dict:
    idx = pd.to_datetime(X_index_values[start:end])
    window_X = pd.DataFrame(X_values[start:end], index=idx, columns=X_columns)
    window_X.index.name = "date"
    try:
        hard, soft, params, scores = _fit_one(model_name, window_X, k, seed, model_cfg)
        drift = semantic_drift(window_X, hard, list(window_X.columns))
        scores = {
            **(scores or {}),
            "semantic_drift_mean": float(drift.mean()) if not drift.empty else float("nan"),
            "semantic_drift_std": float(drift.std()) if not drift.empty else float("nan"),
        }
        return {
            "ok": True,
            "rep": rep_name,
            "model": model_name,
            "K": int(k),
            "W": int(w),
            "seed": int(seed),
            "roll": str(roll),
            "hard": hard,
            "soft": soft,
            "scores": scores,
            "semantic_drift": drift,
            "model_params": params,
        }
    except Exception:
        logger.exception(
            "Model fit failed; rep=%s, model=%s, K=%d, seed=%d, roll=%d, n=%d, p=%d",
            rep_name,
            model_name,
            k,
            seed,
            int(roll.split("_")[-1]) if str(roll).startswith("roll_") else -1,
            len(window_X),
            window_X.shape[1],
        )
        return {
            "ok": False,
            "rep": rep_name,
            "model": model_name,
            "K": int(k),
            "W": int(w),
            "seed": int(seed),
            "roll": str(roll),
        }


def _append_csv_row(path: Path, header: List[str], row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in header})


def _fit_slice_write_shard(
    shard_dir: Path,
    log_path: str,
    asset: str,
    model_name: str,
    X_values: np.ndarray,
    X_index_values: np.ndarray,
    X_columns: List[str],
    start: int,
    end: int,
    k: int,
    seed: int,
    rep_name: str,
    roll: str,
    w: int,
    model_cfg: Dict,
) -> bool:
    """
    Fit a window and append results to per-process shard files.

    This avoids returning large pandas objects across processes (which can crash
    loky workers on Windows due to memory/IPC pressure).
    """

    # In loky process mode, workers do not inherit the main process' logging/warnings
    # configuration. Configure them here so Python warnings (e.g., hmmlearn convergence
    # warnings) are written into the single global log file.
    try:
        configure_global_file_logging(Path(log_path))
    except Exception:
        # Never fail the fit due to logging setup.
        pass

    pid = os.getpid()
    idx = pd.to_datetime(X_index_values[start:end])
    window_X = pd.DataFrame(X_values[start:end], index=idx, columns=X_columns)
    window_X.index.name = "date"

    try:
        hard, soft, params, scores = _fit_one(model_name, window_X, k, seed, model_cfg)
        drift = semantic_drift(window_X, hard, list(window_X.columns))
        scores = {
            **(scores or {}),
            "semantic_drift_mean": float(drift.mean()) if not drift.empty else float("nan"),
            "semantic_drift_std": float(drift.std()) if not drift.empty else float("nan"),
        }

        m = str(model_name)

        # States (hard) shard
        hard_path = Path(shard_dir) / f"states_hard_{m}_{pid}.csv"
        hard_df = hard.rename("state").to_frame().reset_index()
        hard_df = hard_df.rename(columns={hard_df.columns[0]: "date"})
        hard_df.insert(0, "model", model_name)
        hard_df.insert(1, "K", int(k))
        hard_df.insert(2, "W", int(w))
        hard_df.insert(3, "seed", int(seed))
        hard_df.insert(4, "roll", str(roll))
        # normalize date to YYYY-MM-DD string for compact CSV
        hard_df["date"] = pd.to_datetime(hard_df["date"]).dt.date.astype(str)
        hard_df.to_csv(hard_path, index=False, mode="a", header=not hard_path.exists())

        # States (soft) shard
        soft_path = Path(shard_dir) / f"states_soft_{m}_{pid}.csv"
        soft_df = soft.reset_index()
        soft_df = soft_df.rename(columns={soft_df.columns[0]: "date"})
        soft_df.insert(0, "model", model_name)
        soft_df.insert(1, "K", int(k))
        soft_df.insert(2, "W", int(w))
        soft_df.insert(3, "seed", int(seed))
        soft_df.insert(4, "roll", str(roll))
        soft_df["date"] = pd.to_datetime(soft_df["date"]).dt.date.astype(str)
        soft_df.to_csv(soft_path, index=False, mode="a", header=not soft_path.exists())

        # Scores shard (one row per window; fixed schema)
        score_path = Path(shard_dir) / f"scores_{m}_{pid}.csv"
        score_header = [
            "rep",
            "model",
            "K",
            "W",
            "seed",
            "roll",
            "loglik",
            "aic",
            "bic",
            "semantic_drift_mean",
            "semantic_drift_std",
        ]
        score_row = {
            "rep": rep_name,
            "model": model_name,
            "K": int(k),
            "W": int(w),
            "seed": int(seed),
            "roll": str(roll),
            "loglik": scores.get("loglik"),
            "aic": scores.get("aic"),
            "bic": scores.get("bic"),
            "semantic_drift_mean": scores.get("semantic_drift_mean"),
            "semantic_drift_std": scores.get("semantic_drift_std"),
        }
        _append_csv_row(score_path, score_header, score_row)

        # Drift shard
        if drift is not None and not drift.empty:
            drift_path = Path(shard_dir) / f"semantic_drift_{m}_{pid}.csv"
            ddf = drift.rename("semantic_drift").to_frame().reset_index()
            ddf = ddf.rename(columns={ddf.columns[0]: "state"})
            ddf.insert(0, "model", model_name)
            ddf.insert(1, "K", int(k))
            ddf.insert(2, "W", int(w))
            ddf.insert(3, "seed", int(seed))
            ddf.insert(4, "roll", str(roll))
            ddf.to_csv(drift_path, index=False, mode="a", header=not drift_path.exists())

        # Params shard (jsonl)
        params_path = Path(shard_dir) / f"model_params_{m}_{pid}.jsonl"
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(params_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "rep": rep_name,
                        "model": model_name,
                        "K": int(k),
                        "W": int(w),
                        "seed": int(seed),
                        "roll": str(roll),
                        "params": params,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        return True
    except Exception:
        logger.exception(
            "Model fit failed (shard); asset=%s rep=%s model=%s K=%d seed=%d roll=%s n=%d p=%d",
            asset,
            rep_name,
            model_name,
            k,
            seed,
            roll,
            len(window_X),
            window_X.shape[1],
        )
        return False


def _run_parallel_sharded_fits(
    *,
    tasks: List[Tuple[str, int, int, int, str]],
    n_jobs: int,
    backend: str,
    shard_dir: Path,
    log_path: Path,
    asset: str,
    X_values: np.ndarray,
    X_index_values: np.ndarray,
    X_columns: List[str],
    k: int,
    rep_name: str,
    w: int,
    model_cfg: Dict,
) -> List[bool]:
    """
    Run sharded fits with a safe fallback.

    On Windows, large loky grids can trigger TerminatedWorkerError (often OOM).
    We retry with safer settings rather than failing the full run.
    """

    def _run_seq() -> List[bool]:
        ok_flags: List[bool] = []
        for (model_name, seed, s, e, roll) in tasks:
            ok_flags.append(
                bool(
                    _fit_slice_write_shard(
                        shard_dir=shard_dir,
                        log_path=str(log_path),
                        asset=str(asset),
                        model_name=model_name,
                        X_values=X_values,
                        X_index_values=X_index_values,
                        X_columns=X_columns,
                        start=int(s),
                        end=int(e),
                        k=int(k),
                        seed=int(seed),
                        rep_name=str(rep_name),
                        roll=str(roll),
                        w=int(w),
                        model_cfg=model_cfg,
                    )
                )
            )
        return ok_flags

    def _reset_shard_dir(reason: str) -> None:
        """
        Ensure shard_dir is empty before a retry.

        If a previous parallel attempt partially wrote shard files, rerunning tasks
        into the same directory would duplicate outputs (e.g., duplicate dates per roll),
        which can later break stability alignment and inflate counts.
        """
        try:
            if shard_dir.exists():
                _rmtree_with_retries(shard_dir)
            shard_dir.mkdir(parents=True, exist_ok=True)
            logger.warning("Reset shard dir before retry (%s). asset=%s rep=%s dir=%s", reason, asset, rep_name, shard_dir)
        except Exception:
            logger.exception("Failed to reset shard dir. asset=%s rep=%s dir=%s", asset, rep_name, shard_dir)

    try:
        return Parallel(
            n_jobs=int(n_jobs),
            backend=str(backend),
            verbose=0,
            max_nbytes="50M",
            mmap_mode="r",
        )(
            delayed(_fit_slice_write_shard)(
                shard_dir=shard_dir,
                log_path=str(log_path),
                asset=str(asset),
                model_name=model_name,
                X_values=X_values,
                X_index_values=X_index_values,
                X_columns=X_columns,
                start=s,
                end=e,
                k=k,
                seed=seed,
                rep_name=rep_name,
                roll=roll,
                w=w,
                model_cfg=model_cfg,
            )
            for (model_name, seed, s, e, roll) in tasks
        )
    except Exception as e:
        is_terminated = TerminatedWorkerError is not None and isinstance(
            e, TerminatedWorkerError
        )
        if not is_terminated:
            logger.exception(
                "Parallel fit failed (non-terminated); asset=%s backend=%s n_jobs=%d tasks=%d rep=%s K=%d W=%d",
                asset,
                backend,
                int(n_jobs),
                len(tasks),
                rep_name,
                int(k),
                int(w),
            )
            raise

        logger.exception(
            "TerminatedWorkerError during parallel fits; retrying with safer settings. "
            "asset=%s backend=%s n_jobs=%d tasks=%d rep=%s K=%d W=%d",
            asset,
            backend,
            int(n_jobs),
            len(tasks),
            rep_name,
            int(k),
            int(w),
        )

        # Retry A: switch to threads and reduce concurrency.
        try:
            _reset_shard_dir("threading")
            safer_jobs = max(1, min(int(n_jobs), 4))
            logger.warning("Retrying with backend=threading; asset=%s n_jobs=%d.", asset, safer_jobs)
            return Parallel(n_jobs=safer_jobs, backend="threading", verbose=0)(
                delayed(_fit_slice_write_shard)(
                    shard_dir=shard_dir,
                    log_path=str(log_path),
                    asset=str(asset),
                    model_name=model_name,
                    X_values=X_values,
                    X_index_values=X_index_values,
                    X_columns=X_columns,
                    start=s,
                    end=e,
                    k=k,
                    seed=seed,
                    rep_name=rep_name,
                    roll=roll,
                    w=w,
                    model_cfg=model_cfg,
                )
                for (model_name, seed, s, e, roll) in tasks
            )
        except Exception:
            logger.exception("Threading retry failed; falling back to sequential. asset=%s", asset)
            _reset_shard_dir("sequential")
            return _run_seq()


def _load_hard_map_from_rep_csv(
    rep_dir: Path,
    rep_name: str,
) -> Dict[Tuple[str, str, int, str], pd.Series]:
    """
    Load hard state sequences from `windows_states_hard.csv` into a map keyed by
    (rep, model, seed, roll) -> pd.Series(date->state).
    """

    rep_dir = Path(rep_dir)
    files = sorted(rep_dir.glob("windows_states_hard_*.csv"))
    # Backward compatibility (older runs)
    legacy = rep_dir / "windows_states_hard.csv"
    if legacy.exists():
        files.append(legacy)
    if not files:
        return {}

    out: Dict[Tuple[str, str, int, str], pd.Series] = {}
    for p in files:
        df = pd.read_csv(p, parse_dates=["date"])
        if df.empty:
            continue
        for (model, seed, roll), g in df.groupby(["model", "seed", "roll"], sort=False):
            # Guard against duplicate timestamps within a (model, seed, roll) group,
            # which can happen if shard merges were duplicated or interrupted.
            g = g.sort_values("date")
            if g["date"].duplicated().any():
                n_dup = int(g["date"].duplicated().sum())
                logger.warning(
                    "Duplicate dates in hard states CSV; dropping duplicates. rep=%s model=%s seed=%s roll=%s file=%s n_dup=%d",
                    rep_name,
                    model,
                    seed,
                    roll,
                    p,
                    n_dup,
                )
                g = g.drop_duplicates(subset=["date"], keep="last")
            s = pd.Series(g["state"].values, index=g["date"].values, dtype="Int64")
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            out[(rep_name, str(model), int(seed), str(roll))] = s
    return out


def _compute_rep_stability_from_map(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    rep_names: List[str],
    models: List[str],
    k: int,
    window: int,
    seeds: List[int],
    rolls: List[str],
) -> List[Dict]:
    records: List[Dict] = []
    for model_name in models:
        for seed in seeds:
            for roll in rolls:
                series: Dict[str, pd.Series] = {}
                for rep in rep_names:
                    s = hard_map.get((rep, model_name, int(seed), roll))
                    if s is not None:
                        series[rep] = s
                for i in range(len(rep_names)):
                    for j in range(i + 1, len(rep_names)):
                        a = rep_names[i]
                        b = rep_names[j]
                        if a not in series or b not in series:
                            continue
                        scores = stability_metrics(series[a], series[b], k)
                        records.append(
                            {
                                "rep_a": a,
                                "rep_b": b,
                                "model": model_name,
                                "K": k,
                                "window": window,
                                "seed": seed,
                                "roll": roll,
                                "ari": scores.ari,
                                "nmi": scores.nmi,
                                "ami": scores.ami,
                                "vi": scores.vi,
                            }
                        )
    return records


def _compute_window_stability_from_map(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    rep_names: List[str],
    models: List[str],
    k: int,
    window: int,
    seeds: List[int],
    rolls: List[str],
) -> List[Dict]:
    records: List[Dict] = []
    for rep in rep_names:
        for model_name in models:
            for seed in seeds:
                for i in range(len(rolls) - 1):
                    roll_a = rolls[i]
                    roll_b = rolls[i + 1]
                    a = hard_map.get((rep, model_name, int(seed), roll_a))
                    b = hard_map.get((rep, model_name, int(seed), roll_b))
                    if a is None or b is None:
                        continue
                    scores = stability_metrics(a, b, k)
                    records.append(
                        {
                            "rep": rep,
                            "model": model_name,
                            "K": k,
                            "window": window,
                            "seed": seed,
                            "roll_a": roll_a,
                            "roll_b": roll_b,
                            "ari": scores.ari,
                            "nmi": scores.nmi,
                            "ami": scores.ami,
                            "vi": scores.vi,
                        }
                    )
    return records


def _state_return_samples(
    returns: pd.Series, states: pd.Series, n_states: int
) -> List[np.ndarray]:
    """
    Build per-state return samples (1D) on the intersection of indices.

    Returns a list of length n_states; each entry is a float ndarray (possibly empty).
    """
    idx = states.dropna().index.intersection(returns.dropna().index)
    if len(idx) == 0:
        return [np.array([], dtype=float) for _ in range(int(n_states))]
    r = returns.loc[idx].astype(float)
    z = states.loc[idx].astype(int)
    out: List[np.ndarray] = []
    for s in range(int(n_states)):
        out.append(r[z == s].values.astype(float))
    return out


def _matched_wasserstein_cost(samples_a: List[np.ndarray], samples_b: List[np.ndarray]) -> float:
    """
    Minimum average 1D Wasserstein cost after Hungarian matching.

    Empty samples incur a large penalty so that well-defined states match first.
    """
    k = int(min(len(samples_a), len(samples_b)))
    if k <= 0:
        return float("nan")
    cost = np.full((k, k), np.nan, dtype=float)
    for i in range(k):
        for j in range(k):
            a = samples_a[i]
            b = samples_b[j]
            if a.size == 0 or b.size == 0:
                continue
            cost[i, j] = float(wasserstein_distance(a, b))

    # If everything is NaN, cost is undefined.
    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        return float("nan")

    penalty = float(finite.max()) * 10.0 + 1.0
    cost_filled = np.where(np.isfinite(cost), cost, penalty)
    row_ind, col_ind = linear_sum_assignment(cost_filled)
    matched = cost_filled[row_ind, col_ind]
    return float(np.mean(matched)) if matched.size else float("nan")


def _compute_semantic_crossrep_wasserstein_from_map(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series,
    rep_names: List[str],
    models: List[str],
    k: int,
    window: int,
    seeds: List[int],
    rolls: List[str],
) -> List[Dict]:
    """
    Cross-representation semantic mismatch within each rolling window.

    For each (model, seed, roll, rep_a, rep_b), we compute per-state return samples and
    take the minimum average 1D Wasserstein cost after matching states (Hungarian).
    """
    records: List[Dict] = []
    for model_name in models:
        for seed in seeds:
            for roll in rolls:
                # Precompute samples for each rep once per (model, seed, roll)
                samples_by_rep: Dict[str, List[np.ndarray]] = {}
                for rep in rep_names:
                    s = hard_map.get((rep, model_name, int(seed), roll))
                    if s is None:
                        continue
                    samples_by_rep[rep] = _state_return_samples(returns, s, k)
                for i in range(len(rep_names)):
                    for j in range(i + 1, len(rep_names)):
                        a = rep_names[i]
                        b = rep_names[j]
                        if a not in samples_by_rep or b not in samples_by_rep:
                            continue
                        v = _matched_wasserstein_cost(samples_by_rep[a], samples_by_rep[b])
                        records.append(
                            {
                                "kind": "cross_rep",
                                "rep_a": a,
                                "rep_b": b,
                                "model": model_name,
                                "K": k,
                                "window": window,
                                "seed": int(seed),
                                "roll": str(roll),
                                "wasserstein": v,
                            }
                        )
    return records


def _compute_semantic_temporal_wasserstein_from_map(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series,
    rep_names: List[str],
    models: List[str],
    k: int,
    window: int,
    seeds: List[int],
    rolls: List[str],
) -> List[Dict]:
    """
    Temporal semantic drift between consecutive rolling windows.

    For each (rep, model, seed, roll_a, roll_b), we compute per-state return samples and
    take the minimum average 1D Wasserstein cost after matching states (Hungarian).
    """
    records: List[Dict] = []
    for rep in rep_names:
        for model_name in models:
            for seed in seeds:
                for i in range(len(rolls) - 1):
                    roll_a = rolls[i]
                    roll_b = rolls[i + 1]
                    a = hard_map.get((rep, model_name, int(seed), roll_a))
                    b = hard_map.get((rep, model_name, int(seed), roll_b))
                    if a is None or b is None:
                        continue
                    sa = _state_return_samples(returns, a, k)
                    sb = _state_return_samples(returns, b, k)
                    v = _matched_wasserstein_cost(sa, sb)
                    records.append(
                        {
                            "kind": "temporal",
                            "rep": rep,
                            "model": model_name,
                            "K": k,
                            "window": window,
                            "seed": int(seed),
                            "roll_a": str(roll_a),
                            "roll_b": str(roll_b),
                            "wasserstein": v,
                        }
                    )
    return records


def _risk_profile_from_returns(x: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """
    Compute a simple state risk profile from 1D return samples.

    Returns are log-returns. Risk metrics are computed on the left tail (loss side).
    - var_alpha: empirical alpha-quantile (typically negative)
    - cvar_alpha: mean of returns in the left tail (<= var_alpha)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "vol": float("nan"),
            "downside_vol": float("nan"),
            "var_alpha": float("nan"),
            "cvar_alpha": float("nan"),
        }
    mean = float(np.mean(x))
    vol = float(np.std(x, ddof=1)) if n > 1 else 0.0
    neg = x[x < 0.0]
    downside_vol = float(np.std(neg, ddof=1)) if neg.size > 1 else (0.0 if neg.size == 1 else float("nan"))
    var_a = float(np.quantile(x, float(alpha)))
    tail = x[x <= var_a]
    cvar_a = float(np.mean(tail)) if tail.size > 0 else var_a
    return {
        "n": float(n),
        "mean": mean,
        "vol": vol,
        "downside_vol": downside_vol,
        "var_alpha": var_a,
        "cvar_alpha": cvar_a,
    }


def _risk_profiles_by_state(
    samples: List[np.ndarray], alpha: float = 0.05
) -> List[Dict[str, float]]:
    """Compute risk profiles for each state (list index = state id)."""
    out: List[Dict[str, float]] = []
    for s in samples:
        out.append(_risk_profile_from_returns(s, alpha=alpha))
    return out


def _matched_ordering_metrics(
    prof_a: List[Dict[str, float]],
    prof_b: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Match states using Hungarian on a simple profile distance, then compute:
    - top1_consistency: whether highest-risk state aligns (after matching)
    - spearman: correlation between risk ranks across matched states
    """
    k = int(min(len(prof_a), len(prof_b)))
    if k <= 1:
        return {
            "top1_consistency": float("nan"),
            "spearman": float("nan"),
            "high_risk_mean_sign_consistency": float("nan"),
            "high_risk_mean_abs_diff": float("nan"),
            "high_risk_downside_vol_abs_diff": float("nan"),
        }

    # Cost = |cvar diff| + |vol diff|; missing values incur penalty.
    cost = np.full((k, k), np.nan, dtype=float)
    for i in range(k):
        for j in range(k):
            a_c = float(prof_a[i].get("cvar_alpha", float("nan")))
            b_c = float(prof_b[j].get("cvar_alpha", float("nan")))
            a_v = float(prof_a[i].get("vol", float("nan")))
            b_v = float(prof_b[j].get("vol", float("nan")))
            if not (math.isfinite(a_c) and math.isfinite(b_c) and math.isfinite(a_v) and math.isfinite(b_v)):
                continue
            cost[i, j] = abs(a_c - b_c) + abs(a_v - b_v)

    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        return {
            "top1_consistency": float("nan"),
            "spearman": float("nan"),
            "high_risk_mean_sign_consistency": float("nan"),
            "high_risk_mean_abs_diff": float("nan"),
            "high_risk_downside_vol_abs_diff": float("nan"),
        }

    penalty = float(finite.max()) * 10.0 + 1.0
    cost_filled = np.where(np.isfinite(cost), cost, penalty)
    row_ind, col_ind = linear_sum_assignment(cost_filled)
    mapping = {int(i): int(j) for i, j in zip(row_ind, col_ind)}

    cvar_a = np.array([float(p.get("cvar_alpha", float("nan"))) for p in prof_a[:k]], dtype=float)
    cvar_b = np.array([float(p.get("cvar_alpha", float("nan"))) for p in prof_b[:k]], dtype=float)

    # High-risk = most negative CVaR (smallest value).
    top_a = int(np.nanargmin(cvar_a)) if np.isfinite(cvar_a).any() else None
    top_b = int(np.nanargmin(cvar_b)) if np.isfinite(cvar_b).any() else None
    if top_a is None or top_b is None or top_a not in mapping:
        top1 = float("nan")
    else:
        top1 = 1.0 if int(mapping[top_a]) == int(top_b) else 0.0

    # Directional consistency for the high-risk (worst-CVaR) state:
    # compare the state in B matched to A's high-risk state.
    if top_a is None or top_a not in mapping:
        sign_cons = float("nan")
        mean_abs_diff = float("nan")
        dvol_abs_diff = float("nan")
    else:
        j = int(mapping[top_a])
        mean_a = float(prof_a[top_a].get("mean", float("nan")))
        mean_b = float(prof_b[j].get("mean", float("nan")))
        dv_a = float(prof_a[top_a].get("downside_vol", float("nan")))
        dv_b = float(prof_b[j].get("downside_vol", float("nan")))
        if math.isfinite(mean_a) and math.isfinite(mean_b):
            sa = float(np.sign(mean_a))
            sb = float(np.sign(mean_b))
            sign_cons = 1.0 if sa == sb else 0.0
            mean_abs_diff = float(abs(mean_a - mean_b))
        else:
            sign_cons = float("nan")
            mean_abs_diff = float("nan")
        dvol_abs_diff = float(abs(dv_a - dv_b)) if (math.isfinite(dv_a) and math.isfinite(dv_b)) else float("nan")

    # Rank consistency: Spearman between CVaR ranks after matching.
    ra = pd.Series(cvar_a).rank(ascending=True, method="average")
    rb = pd.Series(cvar_b).rank(ascending=True, method="average")
    rb_m = pd.Series({i: float(rb[mapping.get(i, -1)]) if i in mapping else float("nan") for i in range(k)})
    df = pd.DataFrame({"ra": ra, "rb": rb_m}).dropna()
    if len(df) <= 1:
        sp = float("nan")
    else:
        # Spearman(ra, rb) == Pearson(rank(ra), rank(rb)). ra/rb already ranks.
        sp = float(df["ra"].corr(df["rb"], method="pearson"))

    return {
        "top1_consistency": float(top1),
        "spearman": float(sp),
        "high_risk_mean_sign_consistency": float(sign_cons),
        "high_risk_mean_abs_diff": float(mean_abs_diff),
        "high_risk_downside_vol_abs_diff": float(dvol_abs_diff),
    }


def _compute_ordering_consistency_crossrep_seed_summary(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series,
    rep_names: List[str],
    models: List[str],
    k: int,
    window: int,
    seeds: List[int],
    rolls: List[str],
    alpha: float = 0.05,
) -> List[Dict]:
    """
    Cross-representation ordering consistency (seed-level summaries).

    For each (model, seed), average Top-1 high-risk alignment and Spearman rank
    consistency across all rep pairs and rolls (after Hungarian matching).
    """
    records: List[Dict] = []
    for model_name in models:
        for seed in seeds:
            top_sum = 0.0
            top_n = 0
            sp_sum = 0.0
            sp_n = 0
            sign_sum = 0.0
            sign_n = 0
            mean_abs_sum = 0.0
            mean_abs_n = 0
            dvol_abs_sum = 0.0
            dvol_abs_n = 0
            n_pairs = 0
            for roll in rolls:
                samples_by_rep: Dict[str, List[np.ndarray]] = {}
                prof_by_rep: Dict[str, List[Dict[str, float]]] = {}
                for rep in rep_names:
                    s = hard_map.get((rep, model_name, int(seed), str(roll)))
                    if s is None:
                        continue
                    samples = _state_return_samples(returns, s, k)
                    samples_by_rep[rep] = samples
                    prof_by_rep[rep] = _risk_profiles_by_state(samples, alpha=alpha)
                for i in range(len(rep_names)):
                    for j in range(i + 1, len(rep_names)):
                        a = rep_names[i]
                        b = rep_names[j]
                        if a not in prof_by_rep or b not in prof_by_rep:
                            continue
                        m = _matched_ordering_metrics(prof_by_rep[a], prof_by_rep[b])
                        n_pairs += 1
                        if math.isfinite(m["top1_consistency"]):
                            top_sum += float(m["top1_consistency"])
                            top_n += 1
                        if math.isfinite(m["spearman"]):
                            sp_sum += float(m["spearman"])
                            sp_n += 1
                        if math.isfinite(m["high_risk_mean_sign_consistency"]):
                            sign_sum += float(m["high_risk_mean_sign_consistency"])
                            sign_n += 1
                        if math.isfinite(m["high_risk_mean_abs_diff"]):
                            mean_abs_sum += float(m["high_risk_mean_abs_diff"])
                            mean_abs_n += 1
                        if math.isfinite(m["high_risk_downside_vol_abs_diff"]):
                            dvol_abs_sum += float(m["high_risk_downside_vol_abs_diff"])
                            dvol_abs_n += 1
            records.append(
                {
                    "kind": "cross_rep",
                    "scope": "all_rep_pairs",
                    "model": model_name,
                    "K": int(k),
                    "window": int(window),
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "top1_high_risk_consistency_mean": (top_sum / top_n) if top_n else float("nan"),
                    "spearman_rank_consistency_mean": (sp_sum / sp_n) if sp_n else float("nan"),
                    "high_risk_mean_sign_consistency_mean": (sign_sum / sign_n) if sign_n else float("nan"),
                    "high_risk_mean_abs_diff_mean": (mean_abs_sum / mean_abs_n) if mean_abs_n else float("nan"),
                    "high_risk_downside_vol_abs_diff_mean": (dvol_abs_sum / dvol_abs_n) if dvol_abs_n else float("nan"),
                    "n_pairs": int(n_pairs),
                    "n_top1": int(top_n),
                    "n_spearman": int(sp_n),
                    "n_sign": int(sign_n),
                    "n_mean_abs": int(mean_abs_n),
                    "n_dvol_abs": int(dvol_abs_n),
                }
            )
    return records


def _compute_ordering_consistency_temporal_seed_summary(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    returns: pd.Series,
    rep_names: List[str],
    models: List[str],
    k: int,
    window: int,
    seeds: List[int],
    rolls: List[str],
    alpha: float = 0.05,
) -> List[Dict]:
    """
    Temporal ordering consistency (seed-level summaries).

    For each (rep, model, seed) and consecutive (roll_a, roll_b), compute ordering
    metrics after Hungarian matching, then average across roll pairs.
    Also emits an all-reps aggregate per (model, seed).
    """
    records: List[Dict] = []
    for model_name in models:
        for seed in seeds:
            # Aggregate across reps
            top_sum_all = 0.0
            top_n_all = 0
            sp_sum_all = 0.0
            sp_n_all = 0
            sign_sum_all = 0.0
            sign_n_all = 0
            mean_abs_sum_all = 0.0
            mean_abs_n_all = 0
            dvol_abs_sum_all = 0.0
            dvol_abs_n_all = 0
            n_pairs_all = 0
            for rep in rep_names:
                top_sum = 0.0
                top_n = 0
                sp_sum = 0.0
                sp_n = 0
                sign_sum = 0.0
                sign_n = 0
                mean_abs_sum = 0.0
                mean_abs_n = 0
                dvol_abs_sum = 0.0
                dvol_abs_n = 0
                n_pairs = 0
                for i in range(len(rolls) - 1):
                    roll_a = str(rolls[i])
                    roll_b = str(rolls[i + 1])
                    a = hard_map.get((rep, model_name, int(seed), roll_a))
                    b = hard_map.get((rep, model_name, int(seed), roll_b))
                    if a is None or b is None:
                        continue
                    pa = _risk_profiles_by_state(_state_return_samples(returns, a, k), alpha=alpha)
                    pb = _risk_profiles_by_state(_state_return_samples(returns, b, k), alpha=alpha)
                    m = _matched_ordering_metrics(pa, pb)
                    n_pairs += 1
                    n_pairs_all += 1
                    if math.isfinite(m["top1_consistency"]):
                        top_sum += float(m["top1_consistency"])
                        top_n += 1
                        top_sum_all += float(m["top1_consistency"])
                        top_n_all += 1
                    if math.isfinite(m["spearman"]):
                        sp_sum += float(m["spearman"])
                        sp_n += 1
                        sp_sum_all += float(m["spearman"])
                        sp_n_all += 1
                    if math.isfinite(m["high_risk_mean_sign_consistency"]):
                        sign_sum += float(m["high_risk_mean_sign_consistency"])
                        sign_n += 1
                        sign_sum_all += float(m["high_risk_mean_sign_consistency"])
                        sign_n_all += 1
                    if math.isfinite(m["high_risk_mean_abs_diff"]):
                        mean_abs_sum += float(m["high_risk_mean_abs_diff"])
                        mean_abs_n += 1
                        mean_abs_sum_all += float(m["high_risk_mean_abs_diff"])
                        mean_abs_n_all += 1
                    if math.isfinite(m["high_risk_downside_vol_abs_diff"]):
                        dvol_abs_sum += float(m["high_risk_downside_vol_abs_diff"])
                        dvol_abs_n += 1
                        dvol_abs_sum_all += float(m["high_risk_downside_vol_abs_diff"])
                        dvol_abs_n_all += 1
                records.append(
                    {
                        "kind": "temporal",
                        "scope": f"rep={rep}",
                        "model": model_name,
                        "K": int(k),
                        "window": int(window),
                        "seed": int(seed),
                        "alpha": float(alpha),
                        "top1_high_risk_consistency_mean": (top_sum / top_n) if top_n else float("nan"),
                        "spearman_rank_consistency_mean": (sp_sum / sp_n) if sp_n else float("nan"),
                        "high_risk_mean_sign_consistency_mean": (sign_sum / sign_n) if sign_n else float("nan"),
                        "high_risk_mean_abs_diff_mean": (mean_abs_sum / mean_abs_n) if mean_abs_n else float("nan"),
                        "high_risk_downside_vol_abs_diff_mean": (dvol_abs_sum / dvol_abs_n) if dvol_abs_n else float("nan"),
                        "n_pairs": int(n_pairs),
                        "n_top1": int(top_n),
                        "n_spearman": int(sp_n),
                        "n_sign": int(sign_n),
                        "n_mean_abs": int(mean_abs_n),
                        "n_dvol_abs": int(dvol_abs_n),
                    }
                )
            records.append(
                {
                    "kind": "temporal",
                    "scope": "all_reps",
                    "model": model_name,
                    "K": int(k),
                    "window": int(window),
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "top1_high_risk_consistency_mean": (top_sum_all / top_n_all) if top_n_all else float("nan"),
                    "spearman_rank_consistency_mean": (sp_sum_all / sp_n_all) if sp_n_all else float("nan"),
                    "high_risk_mean_sign_consistency_mean": (sign_sum_all / sign_n_all) if sign_n_all else float("nan"),
                    "high_risk_mean_abs_diff_mean": (mean_abs_sum_all / mean_abs_n_all) if mean_abs_n_all else float("nan"),
                    "high_risk_downside_vol_abs_diff_mean": (dvol_abs_sum_all / dvol_abs_n_all) if dvol_abs_n_all else float("nan"),
                    "n_pairs": int(n_pairs_all),
                    "n_top1": int(top_n_all),
                    "n_spearman": int(sp_n_all),
                    "n_sign": int(sign_n_all),
                    "n_mean_abs": int(mean_abs_n_all),
                    "n_dvol_abs": int(dvol_abs_n_all),
                }
            )
    return records


def _mean_or_nan(x: pd.Series) -> float:
    try:
        return float(x.mean())
    except Exception:
        return float("nan")


def _write_key_outputs(
    scores: pd.DataFrame,
    stability: pd.DataFrame,
    semantic: pd.DataFrame,
    ordering: pd.DataFrame,
    out_base: Path,
) -> None:
    rows: List[Dict] = []

    # Cross-representation agreement
    if {"rep_a", "rep_b", "ari"}.issubset(stability.columns):
        rep_df = stability.dropna(subset=["rep_a", "rep_b", "ari"]).copy()
        rep_df = rep_df[rep_df["rep_a"] != rep_df["rep_b"]]
        if not rep_df.empty:
            rows.append(
                {
                    "metric": "cross_rep_ari_mean",
                    "scope": "all",
                    "value": _mean_or_nan(rep_df["ari"]),
                    "n": int(len(rep_df)),
                }
            )
            for metric_col in ("nmi", "ami", "vi"):
                if metric_col in rep_df.columns:
                    rows.append(
                        {
                            "metric": f"cross_rep_{metric_col}_mean",
                            "scope": "all",
                            "value": _mean_or_nan(rep_df[metric_col]),
                            "n": int(len(rep_df)),
                        }
                    )
            if "model" in rep_df.columns:
                for model_name, g in rep_df.groupby("model"):
                    rows.append(
                        {
                            "metric": "cross_rep_ari_mean",
                            "scope": f"model={model_name}",
                            "value": _mean_or_nan(g["ari"]),
                            "n": int(len(g)),
                        }
                    )
                    for metric_col in ("nmi", "ami", "vi"):
                        if metric_col in g.columns:
                            rows.append(
                                {
                                    "metric": f"cross_rep_{metric_col}_mean",
                                    "scope": f"model={model_name}",
                                    "value": _mean_or_nan(g[metric_col]),
                                    "n": int(len(g)),
                                }
                            )

            # Ablation: rep_a vs rep_a_unscaled
            if "rep_a_unscaled" in set(rep_df["rep_a"]) | set(rep_df["rep_b"]):
                mask = (
                    (rep_df["rep_a"] == "rep_a") & (rep_df["rep_b"] == "rep_a_unscaled")
                ) | (
                    (rep_df["rep_a"] == "rep_a_unscaled") & (rep_df["rep_b"] == "rep_a")
                )
                ab = rep_df[mask]
                if not ab.empty:
                    rows.append(
                        {
                            "metric": "ablation_rep_a_vs_unscaled_ari_mean",
                            "scope": "rep_a <-> rep_a_unscaled",
                            "value": _mean_or_nan(ab["ari"]),
                            "n": int(len(ab)),
                        }
                    )
                    if "nmi" in ab.columns:
                        rows.append(
                            {
                                "metric": "ablation_rep_a_vs_unscaled_nmi_mean",
                                "scope": "rep_a <-> rep_a_unscaled",
                                "value": _mean_or_nan(ab["nmi"]),
                                "n": int(len(ab)),
                            }
                        )

    # Temporal stability
    if {"roll_a", "roll_b", "ari"}.issubset(stability.columns):
        tmp = stability.dropna(subset=["ari", "roll_a", "roll_b"]).copy()
        if not tmp.empty:
            rows.append(
                {
                    "metric": "temporal_ari_mean",
                    "scope": "all",
                    "value": _mean_or_nan(tmp["ari"]),
                    "n": int(len(tmp)),
                }
            )
            for metric_col in ("ami", "vi"):
                if metric_col in tmp.columns:
                    rows.append(
                        {
                            "metric": f"temporal_{metric_col}_mean",
                            "scope": "all",
                            "value": _mean_or_nan(tmp[metric_col]),
                            "n": int(len(tmp)),
                        }
                    )
            if "model" in tmp.columns:
                for model_name, g in tmp.groupby("model"):
                    rows.append(
                        {
                            "metric": "temporal_ari_mean",
                            "scope": f"model={model_name}",
                            "value": _mean_or_nan(g["ari"]),
                            "n": int(len(g)),
                        }
                    )
                    for metric_col in ("ami", "vi"):
                        if metric_col in g.columns:
                            rows.append(
                                {
                                    "metric": f"temporal_{metric_col}_mean",
                                    "scope": f"model={model_name}",
                                    "value": _mean_or_nan(g[metric_col]),
                                    "n": int(len(g)),
                                }
                            )

    # Semantic consistency (return-distribution profiles; 1D Wasserstein + matching)
    if not semantic.empty and "wasserstein" in semantic.columns and "kind" in semantic.columns:
        cross_s = semantic[(semantic["kind"].astype(str) == "cross_rep")].dropna(
            subset=["wasserstein"]
        )
        if not cross_s.empty:
            rows.append(
                {
                    "metric": "semantic_cross_rep_wasserstein_mean",
                    "scope": "all",
                    "value": _mean_or_nan(cross_s["wasserstein"]),
                    "n": int(len(cross_s)),
                }
            )
            if "model" in cross_s.columns:
                for model_name, g in cross_s.groupby("model"):
                    rows.append(
                        {
                            "metric": "semantic_cross_rep_wasserstein_mean",
                            "scope": f"model={model_name}",
                            "value": _mean_or_nan(g["wasserstein"]),
                            "n": int(len(g)),
                        }
                    )

        temporal_s = semantic[(semantic["kind"].astype(str) == "temporal")].dropna(
            subset=["wasserstein"]
        )
        if not temporal_s.empty:
            rows.append(
                {
                    "metric": "semantic_temporal_wasserstein_mean",
                    "scope": "all",
                    "value": _mean_or_nan(temporal_s["wasserstein"]),
                    "n": int(len(temporal_s)),
                }
            )
            if "model" in temporal_s.columns:
                for model_name, g in temporal_s.groupby("model"):
                    rows.append(
                        {
                            "metric": "semantic_temporal_wasserstein_mean",
                            "scope": f"model={model_name}",
                            "value": _mean_or_nan(g["wasserstein"]),
                            "n": int(len(g)),
                        }
                    )

    # Ordering consistency: high-risk state alignment + rank consistency
    if not ordering.empty and {"kind", "scope", "model", "seed"}.issubset(ordering.columns):
        # Cross-rep (seed-level, already aggregated across rep pairs & rolls)
        cross_o = ordering[
            (ordering["kind"].astype(str) == "cross_rep")
            & (ordering["scope"].astype(str) == "all_rep_pairs")
        ].dropna(subset=["top1_high_risk_consistency_mean"])
        if not cross_o.empty:
            rows.append(
                {
                    "metric": "ordering_cross_rep_top1_mean",
                    "scope": "all",
                    "value": _mean_or_nan(cross_o["top1_high_risk_consistency_mean"]),
                    "n": int(len(cross_o)),
                }
            )
            rows.append(
                {
                    "metric": "ordering_cross_rep_spearman_mean",
                    "scope": "all",
                    "value": _mean_or_nan(cross_o["spearman_rank_consistency_mean"]),
                    "n": int(len(cross_o)),
                }
            )
            for model_name, g in cross_o.groupby("model"):
                rows.append(
                    {
                        "metric": "ordering_cross_rep_top1_mean",
                        "scope": f"model={model_name}",
                        "value": _mean_or_nan(g["top1_high_risk_consistency_mean"]),
                        "n": int(len(g)),
                    }
                )
                rows.append(
                    {
                        "metric": "ordering_cross_rep_spearman_mean",
                        "scope": f"model={model_name}",
                        "value": _mean_or_nan(g["spearman_rank_consistency_mean"]),
                        "n": int(len(g)),
                    }
                )

            if "high_risk_mean_sign_consistency_mean" in cross_o.columns:
                rows.append(
                    {
                        "metric": "ordering_cross_rep_high_risk_mean_sign_mean",
                        "scope": "all",
                        "value": _mean_or_nan(cross_o["high_risk_mean_sign_consistency_mean"]),
                        "n": int(len(cross_o)),
                    }
                )
            if "high_risk_mean_abs_diff_mean" in cross_o.columns:
                rows.append(
                    {
                        "metric": "ordering_cross_rep_high_risk_mean_abs_diff_mean",
                        "scope": "all",
                        "value": _mean_or_nan(cross_o["high_risk_mean_abs_diff_mean"]),
                        "n": int(len(cross_o)),
                    }
                )
            if "high_risk_downside_vol_abs_diff_mean" in cross_o.columns:
                rows.append(
                    {
                        "metric": "ordering_cross_rep_high_risk_downside_vol_abs_diff_mean",
                        "scope": "all",
                        "value": _mean_or_nan(cross_o["high_risk_downside_vol_abs_diff_mean"]),
                        "n": int(len(cross_o)),
                    }
                )

        # Temporal (seed-level, aggregated across reps)
        temp_o = ordering[
            (ordering["kind"].astype(str) == "temporal")
            & (ordering["scope"].astype(str) == "all_reps")
        ].dropna(subset=["top1_high_risk_consistency_mean"])
        if not temp_o.empty:
            rows.append(
                {
                    "metric": "ordering_temporal_top1_mean",
                    "scope": "all",
                    "value": _mean_or_nan(temp_o["top1_high_risk_consistency_mean"]),
                    "n": int(len(temp_o)),
                }
            )
            rows.append(
                {
                    "metric": "ordering_temporal_spearman_mean",
                    "scope": "all",
                    "value": _mean_or_nan(temp_o["spearman_rank_consistency_mean"]),
                    "n": int(len(temp_o)),
                }
            )
            for model_name, g in temp_o.groupby("model"):
                rows.append(
                    {
                        "metric": "ordering_temporal_top1_mean",
                        "scope": f"model={model_name}",
                        "value": _mean_or_nan(g["top1_high_risk_consistency_mean"]),
                        "n": int(len(g)),
                    }
                )
                rows.append(
                    {
                        "metric": "ordering_temporal_spearman_mean",
                        "scope": f"model={model_name}",
                        "value": _mean_or_nan(g["spearman_rank_consistency_mean"]),
                        "n": int(len(g)),
                    }
                )

            if "high_risk_mean_sign_consistency_mean" in temp_o.columns:
                rows.append(
                    {
                        "metric": "ordering_temporal_high_risk_mean_sign_mean",
                        "scope": "all",
                        "value": _mean_or_nan(temp_o["high_risk_mean_sign_consistency_mean"]),
                        "n": int(len(temp_o)),
                    }
                )
            if "high_risk_mean_abs_diff_mean" in temp_o.columns:
                rows.append(
                    {
                        "metric": "ordering_temporal_high_risk_mean_abs_diff_mean",
                        "scope": "all",
                        "value": _mean_or_nan(temp_o["high_risk_mean_abs_diff_mean"]),
                        "n": int(len(temp_o)),
                    }
                )
            if "high_risk_downside_vol_abs_diff_mean" in temp_o.columns:
                rows.append(
                    {
                        "metric": "ordering_temporal_high_risk_downside_vol_abs_diff_mean",
                        "scope": "all",
                        "value": _mean_or_nan(temp_o["high_risk_downside_vol_abs_diff_mean"]),
                        "n": int(len(temp_o)),
                    }
                )

    rows.append(
        {"metric": "scores_rows", "scope": "all", "value": float(len(scores)), "n": 0}
    )
    rows.append(
        {
            "metric": "stability_rows",
            "scope": "all",
            "value": float(len(stability)),
            "n": 0,
        }
    )

    out_base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_base / "key_results.csv", index=False)

    def _get(metric: str) -> float | None:
        m = [r for r in rows if r["metric"] == metric and r["scope"] == "all"]
        return float(m[0]["value"]) if m else None

    cross_rep = _get("cross_rep_ari_mean")
    temporal = _get("temporal_ari_mean")
    sem_cross = _get("semantic_cross_rep_wasserstein_mean")
    sem_temporal = _get("semantic_temporal_wasserstein_mean")
    ablation = None
    for r in rows:
        if r["metric"] == "ablation_rep_a_vs_unscaled_ari_mean":
            ablation = float(r["value"])
            break

    lines = [
        "# Paper 1 — Unified Run Analysis",
        "",
        f"- **scores_summary rows**: {len(scores)}",
        f"- **stability_summary rows**: {len(stability)}",
        "",
        "## Key stability metrics (means)",
        f"- **Cross-representation ARI** (all pairs): {cross_rep if cross_rep is not None else 'NA'}",
        f"- **Temporal ARI** (consecutive windows): {temporal if temporal is not None else 'NA'}",
        f"- **Semantic cross-rep Wasserstein** (mean): {sem_cross if sem_cross is not None else 'NA'}",
        f"- **Semantic temporal Wasserstein** (mean): {sem_temporal if sem_temporal is not None else 'NA'}",
        f"- **Ablation ARI** (`rep_a` vs `rep_a_unscaled`): {ablation if ablation is not None else 'NA'}",
        "",
        "## Interpretation (template)",
        "- Low cross-representation agreement supports representation dependence.",
        "- Temporal ARI below 1.0 supports nonstationary drift of state structure.",
        "- A low `rep_a` vs `rep_a_unscaled` agreement indicates preprocessing (standardization) alone changes inferred states.",
        "",
        "See `plots/stability_summary.csv`, `plots/scores_summary.csv`, and `plots/semantic_summary.csv` for full tables.",
    ]
    (out_base / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(
    scores: pd.DataFrame, stability: pd.DataFrame, plots_dir: Path
) -> None:
    ensure_dir(plots_dir)
    # Cross-representation rows exist only when >=2 representations are configured.
    if (
        stability is not None
        and not stability.empty
        and {"rep_a", "rep_b"}.issubset(stability.columns)
    ):
        cross = stability.dropna(subset=["rep_a", "rep_b"]).copy()
    else:
        cross = pd.DataFrame()

    if not cross.empty and "ari" in cross.columns:
        # Symmetric matrix over (rep_a, rep_b) pairs. Diagonal is set to 1.
        plot_pairwise_matrix_heatmap(
            cross,
            plots_dir / "cross_rep_ari_matrix_all.png",
            value_col="ari",
            title="Cross-representation ARI (mean; all models)",
            vmin=-0.2,
            vmax=1.0,
        )
        if "model" in cross.columns:
            for m in sorted(set(str(x) for x in cross["model"].dropna().unique())):
                sub = cross[cross["model"].astype(str) == m]
                if not sub.empty:
                    plot_pairwise_matrix_heatmap(
                        sub,
                        plots_dir / f"cross_rep_ari_matrix_{m}.png",
                        value_col="ari",
                        title=f"Cross-representation ARI (mean; model={m})",
                        vmin=-0.2,
                        vmax=1.0,
                    )

    if not cross.empty and "ari" in cross.columns:
        plot_cross_rep_box_by_rep(
            cross,
            out_path=plots_dir / "cross_rep_ari_by_rep.png",
            value_col="ari",
            title="Cross-representation ARI by representation (pairs as endpoints)",
        )

    if not scores.empty and "K" in scores.columns and "model" in scores.columns:
        metric_cols = []
        if "loglik" in scores.columns:
            metric_cols.append(("loglik", "model_loglik_by_k.png"))
        for c in ("aic", "bic"):
            if c in scores.columns:
                metric_cols.append((c, f"model_{c}_by_k.png"))
        for col, out_name in metric_cols:
            agg = scores.groupby(["model", "K"], as_index=False)[col].mean()
            plot_line_by_group(
                agg,
                x="K",
                y=col,
                group="model",
                out_path=plots_dir / out_name,
                title=f"{col.upper()} by K",
            )


def _run_single_asset(
    asset: str, cfg: Dict, experiment_dir: Path, out_dir_override: Path | None = None
) -> float:
    t_asset0 = time.perf_counter()
    name = safe_name(asset)
    out_dir = (out_dir_override if out_dir_override is not None else experiment_dir / name)
    results_dir = out_dir / "results"
    plots_dir = out_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)
    logger.info("Run started; asset=%s, out_dir=%s", asset, out_dir)

    reps = _build_rep_configs(cfg)
    grid = cfg.get("grid", {}) or {}
    k = int((grid.get("n_states", [3]) or [3])[0])
    w = int((grid.get("windows", [252]) or [252])[0])
    step = int(grid.get("step", 21))
    seeds = [int(s) for s in (grid.get("seeds", [1, 2]) or [1, 2])]
    n_jobs = int(grid.get("n_jobs", 1))
    ctx = f"asset={asset} step={step} K={k} W={w}"

    model_cfg = cfg.get("models", {}) or {}
    models = [m for m in ("hmm", "gmm") if model_cfg.get(m, {}).get("enabled", True)]
    logger.info(
        "[%s] Config summary; reps=%d models=%s K=%d window=%d step=%d seeds=%s n_jobs=%d",
        ctx,
        len(reps),
        ",".join(models),
        k,
        w,
        step,
        ",".join(str(s) for s in seeds),
        n_jobs,
    )

    raw_dir = Path(cfg.get("raw_dir", cfg.get("data", {}).get("raw_dir", "data")))
    prices = load_prices([asset], raw_dir, price_col=None)
    if asset not in prices:
        raise FileNotFoundError(f"Price series for {asset} not found under {raw_dir}")
    price = prices[asset]
    if not price.empty:
        logger.info(
            "Loaded price series; asset=%s n=%d start=%s end=%s",
            asset,
            int(price.shape[0]),
            str(price.index.min()),
            str(price.index.max()),
        )

    # Build all representations once, then align to a common post-warmup index.
    X_by_rep: Dict[str, pd.DataFrame] = {}
    first_valid: Dict[str, pd.Timestamp] = {}
    for rep in reps:
        X_raw = build_representation_single(price, rep)
        if X_raw.empty:
            raise ValueError(f"Empty features for rep={rep.name}")
        X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
        X_by_rep[rep.name] = X_raw
        idx_valid = X_raw.dropna().index
        if idx_valid.empty:
            raise ValueError(f"No valid (non-NaN) feature rows for rep={rep.name}")
        first_valid[rep.name] = idx_valid.min()

    common_start = max(first_valid.values())
    common_index: pd.Index | None = None
    for rep in reps:
        idx = X_by_rep[rep.name].loc[common_start:].dropna().index
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None:
        raise ValueError("Failed to build common feature index.")
    common_index = common_index.sort_values()
    if len(common_index) < w:
        raise ValueError(
            f"Not enough complete feature rows after warm-up for asset={asset}: "
            f"need window={w}, have={len(common_index)}"
        )

    logger.info(
        "Common feature index; asset=%s start=%s n=%d",
        asset,
        str(common_index.min()),
        int(len(common_index)),
    )

    # Return series (used for semantic consistency metrics).
    # Align to common_index so all representations share the same timestamp universe.
    # Use log-returns; drop NaNs from the initial diff.
    price_common = price.loc[common_index].astype(float)
    returns = np.log(price_common).diff().rename("log_return").dropna()

    slices = rolling_slices(len(common_index), w, step)
    rolls = [_window_roll_name(i) for i in range(len(slices))]
    logger.info(
        "[%s] Rolling slices (post-warmup); n_slices=%d window=%d step=%d",
        ctx,
        len(slices),
        w,
        step,
    )

    # Global window index (shared across reps for this asset)
    slice_records: List[Dict] = []
    for roll_idx, (s, e) in enumerate(slices):
        slice_records.append(
            {
                "roll": _window_roll_name(roll_idx),
                "start_pos": int(s),
                "end_pos": int(e),
                "start_date": str(common_index[s]),
                "end_date": str(common_index[e - 1]),
            }
        )
    pd.DataFrame(slice_records).to_csv(results_dir / "windows_index.csv", index=False)

    # Store hard states for stability metrics across reps/windows.
    hard_map: Dict[Tuple[str, str, int, str], pd.Series] = {}

    score_rows: List[Dict] = []

    for rep in reps:
        t_rep0 = time.perf_counter()
        rep_dir = results_dir / rep.name
        ensure_dir(rep_dir)

        X = X_by_rep[rep.name].loc[common_index].copy()
        X.to_csv(rep_dir / "features.csv")
        X_values = X.values
        X_index_values = X.index.values
        X_columns = [str(c) for c in X.columns]
        nan_rows = int(X.isna().any(axis=1).sum())
        logger.info(
            "[%s rep=%s] Prepared representation; n=%d p=%d nan_rows=%d",
            ctx,
            rep.name,
            int(X.shape[0]),
            int(X.shape[1]),
            nan_rows,
        )

        tasks = []
        for model_name in models:
            for seed in seeds:
                for roll_idx, (s, e) in enumerate(slices):
                    tasks.append(
                        (model_name, int(seed), int(s), int(e), _window_roll_name(roll_idx))
                    )

        parallel_backend = str(grid.get("parallel_backend", "loky"))
        if n_jobs == 1:
            results = []
            for model_name, seed, s, e, roll in tqdm(
                tasks, desc=f"Fitting {name}/{rep.name}", ncols=100
            ):
                results.append(
                    _fit_slice_collect(
                        model_name=model_name,
                        X_values=X_values,
                        X_index_values=X_index_values,
                        X_columns=X_columns,
                        start=s,
                        end=e,
                        k=k,
                        seed=seed,
                        rep_name=rep.name,
                        roll=roll,
                        w=w,
                        model_cfg=model_cfg,
                    )
                )
        else:
            logger.info(
                "[%s rep=%s] Running %d tasks with %d workers (backend=%s)",
                ctx,
                rep.name,
                len(tasks),
                n_jobs,
                parallel_backend,
            )
            shard_dir = rep_dir / "_shards"
            if shard_dir.exists():
                _rmtree_with_retries(shard_dir)
            shard_dir.mkdir(parents=True, exist_ok=True)

            ok_flags = _run_parallel_sharded_fits(
                tasks=tasks,
                n_jobs=n_jobs,
                backend=parallel_backend,
                shard_dir=shard_dir,
                log_path=Path(experiment_dir) / "run.log",
                asset=asset,
                X_values=X_values,
                X_index_values=X_index_values,
                X_columns=X_columns,
                k=k,
                rep_name=rep.name,
                w=w,
                model_cfg=model_cfg,
            )

            ok = int(sum(1 for x in ok_flags if x))
            fail = int(len(ok_flags) - ok)

            # Merge shards into rep-level files (split by model suffix).
            for m in models:
                m = str(m)
                parts_hard = sorted([p for p in shard_dir.glob(f"states_hard_{m}_*.csv")])
                if parts_hard:
                    hard_df = pd.concat([pd.read_csv(p) for p in parts_hard], axis=0, ignore_index=True)
                    # Defensive: if a retry happened without a clean shard reset, we may have duplicates.
                    if set(["model", "seed", "roll", "date"]).issubset(set(hard_df.columns)):
                        hard_df = hard_df.drop_duplicates(subset=["model", "seed", "roll", "date"], keep="last")
                    hard_df.to_csv(rep_dir / f"windows_states_hard_{m}.csv", index=False)

                parts_soft = sorted([p for p in shard_dir.glob(f"states_soft_{m}_*.csv")])
                if parts_soft:
                    pd.concat(
                        [pd.read_csv(p) for p in parts_soft], axis=0, ignore_index=True
                    ).to_csv(rep_dir / f"windows_states_soft_{m}.csv", index=False)

                parts_drift = sorted([p for p in shard_dir.glob(f"semantic_drift_{m}_*.csv")])
                if parts_drift:
                    pd.concat(
                        [pd.read_csv(p) for p in parts_drift], axis=0, ignore_index=True
                    ).to_csv(rep_dir / f"windows_semantic_drift_{m}.csv", index=False)

                parts_scores = sorted([p for p in shard_dir.glob(f"scores_{m}_*.csv")])
                if parts_scores:
                    sc = pd.concat(
                        [pd.read_csv(p) for p in parts_scores], axis=0, ignore_index=True
                    )
                    sc.to_csv(rep_dir / f"windows_scores_{m}.csv", index=False)
                    score_rows.extend(sc.to_dict(orient="records"))

                parts_params = sorted([p for p in shard_dir.glob(f"model_params_{m}_*.jsonl")])
                if parts_params:
                    (rep_dir / f"windows_model_params_{m}.jsonl").write_text(
                        "".join([Path(p).read_text(encoding="utf-8") for p in parts_params]),
                        encoding="utf-8",
                    )

            _rmtree_with_retries(shard_dir, retries=6)

            results = []  # no in-memory results in process mode

        if n_jobs == 1:
            ok = int(sum(1 for r in results if r.get("ok")))
            fail = int(len(results) - ok)

        # Assemble per-rep merged outputs.
        hard_parts_by_model: Dict[str, List[pd.DataFrame]] = {}
        soft_parts_by_model: Dict[str, List[pd.DataFrame]] = {}
        drift_parts_by_model: Dict[str, List[pd.DataFrame]] = {}
        params_lines_by_model: Dict[str, List[str]] = {}
        local_score_rows: List[Dict] = []

        for r in results:
            if not r.get("ok"):
                continue
            model_name = str(r["model"])
            seed = int(r["seed"])
            roll = str(r["roll"])

            hard: pd.Series = r["hard"]
            soft: pd.DataFrame = r["soft"]
            drift: pd.Series = r["semantic_drift"]
            params: Dict = r.get("model_params", {}) or {}
            scores: Dict = r.get("scores", {}) or {}

            # Hard states
            h = hard.rename("state").to_frame().reset_index()
            date_col = hard.index.name if hard.index.name is not None else "index"
            if date_col in h.columns:
                h = h.rename(columns={date_col: "date"})
            elif "index" in h.columns:
                h = h.rename(columns={"index": "date"})
            h.insert(0, "model", model_name)
            h.insert(1, "K", int(k))
            h.insert(2, "W", int(w))
            h.insert(3, "seed", seed)
            h.insert(4, "roll", roll)
            hard_parts_by_model.setdefault(model_name, []).append(h)

            hard_map[(rep.name, model_name, seed, roll)] = hard

            # Soft states
            sft = soft.copy()
            sft = sft.reset_index().rename(columns={sft.index.name or "index": "date"})
            # If reset_index didn't name it "date" (e.g. index had no name), ensure column exists.
            if "date" not in sft.columns:
                sft = sft.rename(columns={sft.columns[0]: "date"})
            sft.insert(0, "model", model_name)
            sft.insert(1, "K", int(k))
            sft.insert(2, "W", int(w))
            sft.insert(3, "seed", seed)
            sft.insert(4, "roll", roll)
            soft_parts_by_model.setdefault(model_name, []).append(sft)

            # Scores (one row per window)
            row = {
                "model": model_name,
                "K": int(k),
                "W": int(w),
                "seed": seed,
                "roll": roll,
            }
            row.update({str(kk): vv for kk, vv in scores.items()})
            score_rows.append({"rep": rep.name, **row})
            local_score_rows.append({"rep": rep.name, **row})

            # Semantic drift (one row per state per window)
            if drift is not None and not drift.empty:
                d = drift.rename("semantic_drift").to_frame().reset_index().rename(
                    columns={"index": "state"}
                )
                d.insert(0, "model", model_name)
                d.insert(1, "K", int(k))
                d.insert(2, "W", int(w))
                d.insert(3, "seed", seed)
                d.insert(4, "roll", roll)
                drift_parts_by_model.setdefault(model_name, []).append(d)

            # Model params (jsonl; one line per window)
            params_line = json.dumps(
                {
                    "model": model_name,
                    "K": int(k),
                    "W": int(w),
                    "seed": seed,
                    "roll": roll,
                    "params": params,
                },
                ensure_ascii=False,
            )
            params_lines_by_model.setdefault(model_name, []).append(params_line)

        if n_jobs == 1:
            for m in models:
                m = str(m)
                hp = hard_parts_by_model.get(m, [])
                if hp:
                    pd.concat(hp, axis=0, ignore_index=True).to_csv(
                        rep_dir / f"windows_states_hard_{m}.csv", index=False
                    )
                sp = soft_parts_by_model.get(m, [])
                if sp:
                    pd.concat(sp, axis=0, ignore_index=True).to_csv(
                        rep_dir / f"windows_states_soft_{m}.csv", index=False
                    )
                dp = drift_parts_by_model.get(m, [])
                if dp:
                    pd.concat(dp, axis=0, ignore_index=True).to_csv(
                        rep_dir / f"windows_semantic_drift_{m}.csv", index=False
                    )
                pl = params_lines_by_model.get(m, [])
                if pl:
                    (rep_dir / f"windows_model_params_{m}.jsonl").write_text(
                        "\n".join(pl) + "\n", encoding="utf-8"
                    )

            # Also write per-model scores CSVs for consistency with process mode.
            if local_score_rows:
                sdf = pd.DataFrame(local_score_rows)
                for m in models:
                    m = str(m)
                    sub = sdf[sdf["model"].astype(str) == m].copy()
                    if not sub.empty:
                        sub.drop(columns=["rep"]).to_csv(
                            rep_dir / f"windows_scores_{m}.csv", index=False
                        )

        logger.info(
            "[%s rep=%s] Finished fits; tasks=%d ok=%d fail=%d elapsed_s=%.1f",
            ctx,
            rep.name,
            len(tasks),
            ok,
            fail,
            time.perf_counter() - t_rep0,
        )

    # In process mode (loky), we avoid returning large pandas objects from workers,
    # so `hard_map` may be empty here. Reconstruct it from merged CSVs.
    if not hard_map:
        for rep in reps:
            rep_dir = results_dir / rep.name
            hard_map.update(_load_hard_map_from_rep_csv(rep_dir, rep.name))

    scores = pd.DataFrame(score_rows)
    # For downstream plotting/analysis, keep a combined scores DataFrame with the same schema.
    if not scores.empty and "rep" in scores.columns:
        scores = scores.drop(columns=["rep"])

    # Stability (computed from hard_map)
    rep_names = [r.name for r in reps]
    rep_stability = _compute_rep_stability_from_map(
        hard_map=hard_map,
        rep_names=rep_names,
        models=models,
        k=k,
        window=w,
        seeds=seeds,
        rolls=rolls,
    )
    window_stability = _compute_window_stability_from_map(
        hard_map=hard_map,
        rep_names=rep_names,
        models=models,
        k=k,
        window=w,
        seeds=seeds,
        rolls=rolls,
    )
    if rep_stability:
        save_json(results_dir / "rep_stability.json", {"rep_stability": rep_stability})
    if window_stability:
        save_json(results_dir / "window_stability.json", {"window_stability": window_stability})

    stability = pd.DataFrame(rep_stability + window_stability)
    # Semantic consistency summary (return distributions by state).
    semantic_cross = _compute_semantic_crossrep_wasserstein_from_map(
        hard_map=hard_map,
        returns=returns,
        rep_names=rep_names,
        models=models,
        k=k,
        window=w,
        seeds=seeds,
        rolls=rolls,
    )
    semantic_temporal = _compute_semantic_temporal_wasserstein_from_map(
        hard_map=hard_map,
        returns=returns,
        rep_names=rep_names,
        models=models,
        k=k,
        window=w,
        seeds=seeds,
        rolls=rolls,
    )
    semantic = pd.DataFrame(semantic_cross + semantic_temporal)

    # Ordering consistency (risk-profile alignment): high-risk state and rank stability.
    ordering_cross = _compute_ordering_consistency_crossrep_seed_summary(
        hard_map=hard_map,
        returns=returns,
        rep_names=rep_names,
        models=models,
        k=k,
        window=w,
        seeds=seeds,
        rolls=rolls,
        alpha=0.05,
    )
    ordering_temporal = _compute_ordering_consistency_temporal_seed_summary(
        hard_map=hard_map,
        returns=returns,
        rep_names=rep_names,
        models=models,
        k=k,
        window=w,
        seeds=seeds,
        rolls=rolls,
        alpha=0.05,
    )
    ordering = pd.DataFrame(ordering_cross + ordering_temporal)
    scores.to_csv(plots_dir / "scores_summary.csv", index=False)
    stability.to_csv(plots_dir / "stability_summary.csv", index=False)
    if not semantic.empty:
        semantic.to_csv(plots_dir / "semantic_summary.csv", index=False)
    if not ordering.empty:
        ordering.to_csv(plots_dir / "ordering_consistency_seed_summary.csv", index=False)
        try:
            plot_ordering_consistency_summary(ordering, plots_dir / "ordering_consistency.png")
        except Exception as e:
            logger.warning("[%s] Could not plot ordering consistency: %s", ctx, e)
    logger.info(
        "[%s] Aggregated summaries; scores_rows=%d stability_rows=%d",
        ctx,
        int(len(scores)),
        int(len(stability)),
    )

    logger.info("[%s] Writing plots to %s", ctx, plots_dir)
    _write_plots(scores, stability, plots_dir)
    pngs = sorted([p.name for p in plots_dir.glob("*.png")])
    if pngs:
        logger.info("[%s] Plots written (%d): %s", ctx, len(pngs), ", ".join(pngs))
    else:
        logger.warning("[%s] No PNG plots found in %s after plotting step", ctx, plots_dir)
    _write_key_outputs(scores, stability, semantic, ordering, out_dir)

    elapsed_s = float(time.perf_counter() - t_asset0)
    logger.info("Run finished; asset=%s elapsed_s=%.1f (%s)", asset, elapsed_s, _fmt_hms(elapsed_s))
    return elapsed_s


def _extract_metrics_from_key_results(p: Path) -> dict | None:
    """Read key_results.csv; return cross_rep_ari_mean, temporal_ari_mean (scope=all) or None."""
    if not p.exists():
        return None
    df = pd.read_csv(p)
    all_rows = df[df["scope"] == "all"]
    cross = all_rows[all_rows["metric"] == "cross_rep_ari_mean"]
    temp = all_rows[all_rows["metric"] == "temporal_ari_mean"]
    if cross.empty:
        return None
    return {
        "cross_rep_ari_mean": float(cross["value"].iloc[0]),
        # Temporal ARI can be undefined at step=W (no overlap): the intersection
        # between consecutive windows is empty, so no temporal_ari_mean row is written.
        "temporal_ari_mean": (
            pd.to_numeric(temp["value"].iloc[0], errors="coerce")
            if not temp.empty
            else float("nan")
        ),
    }


def _write_key_results_all_assets(
    outputs_dir: Path, assets: List[str], candidate_paths_by_asset: Dict[str, Path], label: str
) -> None:
    """
    Write a concise multi-asset key results table.

    This is used for paper tables that need a single CSV (e.g., baseline step=21 run),
    even when the main run was a sweep.
    """
    rows: List[pd.DataFrame] = []
    for asset in assets:
        name = safe_name(asset)
        p = candidate_paths_by_asset.get(name)
        if p is None or not p.exists():
            continue
        df = pd.read_csv(p)
        df.insert(0, "asset", name)
        rows.append(df)
    if not rows:
        logger.warning("No key_results found for multi-asset summary (%s).", label)
        return
    out_csv = Path(outputs_dir) / "key_results_all_assets.csv"
    pd.concat(rows, axis=0, ignore_index=True).to_csv(out_csv, index=False)
    (Path(outputs_dir) / "analysis_all_assets.md").write_text(
        "# Multi-asset summary\n\n"
        + f"- label: {label}\n"
        + f"- assets: {', '.join(safe_name(a) for a in assets)}\n"
        + f"- csv: {out_csv.name}\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s (%s)", out_csv, label)


def run(config_path: Path) -> None:
    t_run0 = time.perf_counter()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    assets = cfg.get("assets", None)
    if not isinstance(assets, list) or not assets:
        raise ValueError("Config requires `assets` as a non-empty list.")
    assets = [str(a) for a in assets]

    outputs_dir = Path(cfg.get("outputs_dir", "outputs"))
    raw_dir = Path(cfg.get("raw_dir", cfg.get("data", {}).get("raw_dir", "data")))
    grid = cfg.get("grid") or {}
    step_sweep = grid.get("step_sweep")
    robustness = grid.get("robustness") or {}
    robustness_enabled = bool(robustness.get("enabled", False))

    # Clear outputs once per invocation (requested).
    if outputs_dir.exists():
        _rmtree_with_retries(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    configure_global_file_logging(outputs_dir / "run.log")

    try:
        logger.info(
            "Starting run; assets=%s raw_dir=%s outputs_dir=%s config=%s",
            ",".join(assets),
            raw_dir,
            outputs_dir,
            Path(config_path),
        )
        missing = [a for a in assets if not (raw_dir / f"{safe_name(a)}.csv").exists()]
        if missing:
            print(f"[runner] Downloading missing tickers to {raw_dir}: {missing}")
            download_tickers(missing, output_dir=raw_dir)

        ran_any = False

        # 1) Step sweep (overlap sensitivity)
        if isinstance(step_sweep, list) and len(step_sweep) > 0:
            _run_step_sweep(cfg, assets, outputs_dir, [int(x) for x in step_sweep])
            # Write baseline summary early (useful even if robustness is long-running).
            if 21 in [int(x) for x in step_sweep]:
                candidates: Dict[str, Path] = {}
                for asset in assets:
                    name = safe_name(asset)
                    candidates[name] = outputs_dir / name / "step_21" / "key_results.csv"
                _write_key_results_all_assets(
                    outputs_dir=outputs_dir,
                    assets=assets,
                    candidate_paths_by_asset=candidates,
                    label="baseline step=21 (from step_sweep)",
                )
            ran_any = True

        # 2) Robustness sweep (K × seeds)
        if robustness_enabled:
            _run_robustness_sweep(cfg, assets, outputs_dir, robustness)
            ran_any = True

        # 3) Fallback: normal single-step run (only if no sweeps were requested)
        if not ran_any:
            experiment_dir = outputs_dir
            for asset in assets:
                _run_single_asset(asset, cfg, experiment_dir)

        # Optional: one-page synthetic sanity check for Appendix (disabled by default).
        try:
            run_synthetic_sanity_check(cfg, outputs_dir)
        except Exception as e:
            logger.warning("Synthetic sanity check failed (continuing without it): %s", e)

        # Update the paper Results numbers from outputs (auto-filled blocks).
        try:
            project_dir = Path(__file__).resolve().parents[1]
            md_path = project_dir / "paper" / "sections" / "04_empirical_results.md"
            if md_path.exists():
                update_empirical_results_md(outputs_dir=outputs_dir, md_path=md_path, cfg=cfg)
                logger.info("Auto-filled Results numbers in %s", md_path)
        except Exception as e:
            logger.warning("Could not auto-fill paper Results numbers: %s", e)

        # Always try to write a baseline multi-asset summary CSV for the paper.
        # Prefer step=21 results if a step sweep ran; otherwise fall back to per-asset root results.
        candidates: Dict[str, Path] = {}
        if isinstance(step_sweep, list) and len(step_sweep) > 0 and 21 in [int(x) for x in step_sweep]:
            for asset in assets:
                name = safe_name(asset)
                candidates[name] = outputs_dir / name / "step_21" / "key_results.csv"
            _write_key_results_all_assets(
                outputs_dir=outputs_dir,
                assets=assets,
                candidate_paths_by_asset=candidates,
                label="baseline step=21 (from step_sweep)",
            )
        else:
            for asset in assets:
                name = safe_name(asset)
                candidates[name] = outputs_dir / name / "key_results.csv"
            _write_key_results_all_assets(
                outputs_dir=outputs_dir,
                assets=assets,
                candidate_paths_by_asset=candidates,
                label="baseline (per-asset root)",
            )
    except Exception:
        # Ensure fatal exceptions are persisted to the single run.log.
        logger.exception("Run failed with an uncaught exception.")
        raise
    finally:
        elapsed_run = float(time.perf_counter() - t_run0)
        logger.info("Run complete; total_elapsed_s=%.1f (%s)", elapsed_run, _fmt_hms(elapsed_run))


def _run_step_sweep(cfg: Dict, assets: List[str], outputs_dir: Path, steps: List[int]) -> None:
    """Run pipeline for each step in steps; asset-first layout; write step_sweep_summary.csv and ari_vs_step.png."""
    t0 = time.perf_counter()
    logger.info("Starting step sweep; assets=%s steps=%s", ",".join(assets), steps)

    cfg["grid"] = cfg.get("grid") or {}
    summary_rows: List[Dict] = []
    totals_by_asset: Dict[str, float] = {}
    totals_by_step: Dict[str, float] = {}

    for step in steps:
        cfg["grid"]["step"] = int(step)
        t_step0 = time.perf_counter()
        for asset in assets:
            out_dir = outputs_dir / safe_name(asset) / f"step_{int(step)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Step sweep: asset=%s step=%d -> %s", asset, int(step), out_dir)
            elapsed = _run_single_asset(asset, cfg, outputs_dir, out_dir_override=out_dir)
            totals_by_asset[safe_name(asset)] = totals_by_asset.get(safe_name(asset), 0.0) + float(elapsed)
        for asset in assets:
            p = outputs_dir / safe_name(asset) / f"step_{int(step)}" / "key_results.csv"
            m = _extract_metrics_from_key_results(p)
            if m is not None:
                summary_rows.append({"step": int(step), "asset": safe_name(asset), **m})
        totals_by_step[f"step_{int(step)}"] = float(time.perf_counter() - t_step0)

    if not summary_rows:
        logger.warning("No key_results found for any step; skipping summary and plot.")
        return

    summary = pd.DataFrame(summary_rows)[["step", "asset", "cross_rep_ari_mean", "temporal_ari_mean"]]
    summary.to_csv(outputs_dir / "step_sweep_summary.csv", index=False)
    logger.info("Wrote %s", outputs_dir / "step_sweep_summary.csv")
    try:
        plot_ari_vs_step(summary, outputs_dir / "ari_vs_step.png")
        logger.info("Wrote %s", outputs_dir / "ari_vs_step.png")
    except Exception as e:
        logger.warning("Could not plot ARI vs step: %s", e)
    (outputs_dir / "analysis_step_sweep.md").write_text(
        "Step-sweep finished. Summary: step_sweep_summary.csv. Figure: ari_vs_step.png. "
        "Layout: <asset>/step_<s>/. See Section 5.5 in the paper for steps, data, and interpretation.\n",
        encoding="utf-8",
    )

    elapsed_all = float(time.perf_counter() - t0)
    for line in _timing_summary_lines("Timing (step_sweep; totals by asset)", totals_by_asset):
        logger.info(line)
    for line in _timing_summary_lines("Timing (step_sweep; totals by step)", totals_by_step):
        logger.info(line)
    logger.info("Step sweep complete; elapsed_s=%.1f (%s)", elapsed_all, _fmt_hms(elapsed_all))


def _run_robustness_sweep(cfg: Dict, assets: List[str], outputs_dir: Path, robustness: Dict) -> None:
    """
    Robustness sweep over random seeds and K.

    Writes:
    - outputs/robustness_seed_metrics.csv (per-seed means; global)
    - outputs/robustness_ci_summary.csv (mean/std/95% CI across seeds; global)
    - outputs/robustness_temporal_ci_by_k.png (global figure)
    - outputs/robustness_crossrep_ci_by_k.png (global figure)

    Per-asset detailed runs are stored under:
    - outputs/<asset>/robustness/K_<K>/ ...
    """

    step = int(robustness.get("step", (cfg.get("grid") or {}).get("step", 21)))
    ks = [int(x) for x in (robustness.get("n_states", []) or [])]
    seeds = [int(x) for x in (robustness.get("seeds", []) or [])]
    if not ks or not seeds:
        logger.warning("Robustness sweep enabled but missing n_states or seeds; skipping.")
        return

    t0 = time.perf_counter()
    logger.info("Starting robustness sweep; step=%d K=%s seeds=%d", step, ks, len(seeds))

    cfg["grid"] = cfg.get("grid") or {}
    cfg["grid"]["step"] = step
    cfg["grid"]["seeds"] = seeds

    # Run K sweep (each K runs all seeds internally)
    totals_by_asset: Dict[str, float] = {}
    totals_by_k: Dict[str, float] = {}
    for k in ks:
        cfg["grid"]["n_states"] = [int(k)]
        t_k0 = time.perf_counter()
        for asset in assets:
            out_dir = outputs_dir / safe_name(asset) / "robustness" / f"K_{int(k)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Robustness: asset=%s K=%d -> %s", asset, int(k), out_dir)
            elapsed = _run_single_asset(asset, cfg, outputs_dir, out_dir_override=out_dir)
            totals_by_asset[safe_name(asset)] = totals_by_asset.get(safe_name(asset), 0.0) + float(elapsed)
        totals_by_k[f"K_{int(k)}"] = float(time.perf_counter() - t_k0)

    # Aggregate per-seed means from stability_summary.csv
    seed_rows: List[Dict] = []
    for asset in assets:
        for k in ks:
            p = (
                outputs_dir
                / safe_name(asset)
                / "robustness"
                / f"K_{int(k)}"
                / "plots"
                / "stability_summary.csv"
            )
            if not p.exists():
                continue
            # Avoid pandas DtypeWarning on large CSVs with mixed-type columns.
            # We only need a few columns for aggregation; read a narrow subset to reduce memory.
            needed_cols = ["seed", "model", "ari", "rep_a", "rep_b", "roll_a", "roll_b"]
            try:
                st = pd.read_csv(p, low_memory=False, usecols=needed_cols)
            except ValueError:
                # Backward/forward compatibility: if columns differ, fall back to full read.
                st = pd.read_csv(p, low_memory=False)
            if "ari" in st.columns:
                st["ari"] = pd.to_numeric(st["ari"], errors="coerce")
            if "seed" in st.columns:
                st["seed"] = pd.to_numeric(st["seed"], errors="coerce")
            if st.empty or "seed" not in st.columns or "model" not in st.columns or "ari" not in st.columns:
                continue

            # Cross-representation rows: have rep_a/rep_b
            cross = st.dropna(subset=["rep_a", "rep_b", "ari"]).copy() if {"rep_a", "rep_b"}.issubset(st.columns) else pd.DataFrame()
            if not cross.empty:
                cross = cross[cross["rep_a"] != cross["rep_b"]]
                g = cross.groupby(["model", "seed"], as_index=False)["ari"].mean()
                for _, r in g.iterrows():
                    seed_rows.append(
                        {
                            "asset": safe_name(asset),
                            "K": int(k),
                            "model": str(r["model"]),
                            "seed": int(r["seed"]),
                            "metric": "cross_rep_ari_seed_mean",
                            "value": float(r["ari"]),
                        }
                    )

            # Temporal rows: have roll_a/roll_b
            temporal = st.dropna(subset=["roll_a", "roll_b", "ari"]).copy() if {"roll_a", "roll_b"}.issubset(st.columns) else pd.DataFrame()
            if not temporal.empty:
                g = temporal.groupby(["model", "seed"], as_index=False)["ari"].mean()
                for _, r in g.iterrows():
                    seed_rows.append(
                        {
                            "asset": safe_name(asset),
                            "K": int(k),
                            "model": str(r["model"]),
                            "seed": int(r["seed"]),
                            "metric": "temporal_ari_seed_mean",
                            "value": float(r["ari"]),
                        }
                    )

    if not seed_rows:
        logger.warning("Robustness sweep produced no seed-level metrics; skipping summaries/plots.")
        return

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(outputs_dir / "robustness_seed_metrics.csv", index=False)

    # CI summary across seeds
    def _ci95(std: float, n: int) -> float:
        return 1.96 * (std / math.sqrt(n)) if n > 1 and math.isfinite(std) else float("nan")

    grouped = (
        seed_df.groupby(["asset", "K", "model", "metric"], as_index=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_seeds"})
    )
    grouped["ci95"] = [
        _ci95(float(s), int(n)) for s, n in zip(grouped["std"].astype(float), grouped["n_seeds"].astype(int))
    ]
    grouped.to_csv(outputs_dir / "robustness_ci_summary.csv", index=False)

    # Plot: per-asset subplots with error bars across K (HMM vs GMM)
    try:
        import matplotlib.pyplot as plt

        assets_sorted = [safe_name(a) for a in assets]
        for metric, fname, ylab in [
            ("temporal_ari_seed_mean", "robustness_temporal_ci_by_k.png", "temporal ARI (mean across seeds)"),
            ("cross_rep_ari_seed_mean", "robustness_crossrep_ci_by_k.png", "cross-rep ARI (mean across seeds)"),
        ]:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
            axes = axes.flatten()
            for ax, asset_name in zip(axes, assets_sorted):
                sub = grouped[(grouped["asset"] == asset_name) & (grouped["metric"] == metric)]
                if sub.empty:
                    ax.set_title(asset_name)
                    ax.axis("off")
                    continue
                for model_name in sorted(set(str(x) for x in sub["model"].dropna().unique())):
                    m = sub[sub["model"].astype(str) == model_name].sort_values("K")
                    ax.errorbar(
                        m["K"].astype(int).values,
                        m["mean"].astype(float).values,
                        yerr=m["ci95"].astype(float).values,
                        marker="o",
                        linewidth=1.5,
                        capsize=3,
                        label=model_name,
                    )
                ax.set_title(asset_name)
                ax.set_xlabel("K")
                ax.set_ylabel(ylab)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=8)
            fig.suptitle(f"Robustness vs K ({metric}; 95% CI across seeds)")
            fig.tight_layout()
            fig.savefig(outputs_dir / fname)
            plt.close(fig)
    except Exception:
        logger.exception("Failed to plot robustness CI figure.")

    # Timing summary (tail-friendly)
    elapsed_all = float(time.perf_counter() - t0)
    for line in _timing_summary_lines("Timing (robustness; totals by asset)", totals_by_asset):
        logger.info(line)
    for line in _timing_summary_lines("Timing (robustness; totals by K)", totals_by_k):
        logger.info(line)
    logger.info("Robustness sweep complete; elapsed_s=%.1f (%s)", elapsed_all, _fmt_hms(elapsed_all))

    # --- Ordering consistency: global seed metrics + CI across seeds ---
    ordering_seed_rows: List[Dict] = []
    for asset in assets:
        for k in ks:
            p = (
                outputs_dir
                / safe_name(asset)
                / "robustness"
                / f"K_{int(k)}"
                / "plots"
                / "ordering_consistency_seed_summary.csv"
            )
            if not p.exists():
                continue
            od = pd.read_csv(p)
            if od.empty:
                continue
            required = {"kind", "scope", "model", "seed", "top1_high_risk_consistency_mean", "spearman_rank_consistency_mean"}
            if not required.issubset(od.columns):
                continue

            # Cross-rep seed summaries (already aggregated across rep pairs & rolls)
            cross = od[(od["kind"].astype(str) == "cross_rep") & (od["scope"].astype(str) == "all_rep_pairs")].copy()
            if not cross.empty:
                for _, r in cross.iterrows():
                    ordering_seed_rows.append(
                        {
                            "asset": safe_name(asset),
                            "K": int(k),
                            "model": str(r["model"]),
                            "seed": int(r["seed"]),
                            "metric": "ordering_cross_rep_top1_seed_mean",
                            "value": float(r["top1_high_risk_consistency_mean"]),
                        }
                    )
                    ordering_seed_rows.append(
                        {
                            "asset": safe_name(asset),
                            "K": int(k),
                            "model": str(r["model"]),
                            "seed": int(r["seed"]),
                            "metric": "ordering_cross_rep_spearman_seed_mean",
                            "value": float(r["spearman_rank_consistency_mean"]),
                        }
                    )
                    if "high_risk_mean_sign_consistency_mean" in cross.columns:
                        ordering_seed_rows.append(
                            {
                                "asset": safe_name(asset),
                                "K": int(k),
                                "model": str(r["model"]),
                                "seed": int(r["seed"]),
                                "metric": "ordering_cross_rep_high_risk_mean_sign_seed_mean",
                                "value": float(r["high_risk_mean_sign_consistency_mean"]),
                            }
                        )
                    if "high_risk_mean_abs_diff_mean" in cross.columns:
                        ordering_seed_rows.append(
                            {
                                "asset": safe_name(asset),
                                "K": int(k),
                                "model": str(r["model"]),
                                "seed": int(r["seed"]),
                                "metric": "ordering_cross_rep_high_risk_mean_abs_diff_seed_mean",
                                "value": float(r["high_risk_mean_abs_diff_mean"]),
                            }
                        )
                    if "high_risk_downside_vol_abs_diff_mean" in cross.columns:
                        ordering_seed_rows.append(
                            {
                                "asset": safe_name(asset),
                                "K": int(k),
                                "model": str(r["model"]),
                                "seed": int(r["seed"]),
                                "metric": "ordering_cross_rep_high_risk_downside_vol_abs_diff_seed_mean",
                                "value": float(r["high_risk_downside_vol_abs_diff_mean"]),
                            }
                        )

            # Temporal seed summaries (use all-reps aggregate)
            temporal = od[(od["kind"].astype(str) == "temporal") & (od["scope"].astype(str) == "all_reps")].copy()
            if not temporal.empty:
                for _, r in temporal.iterrows():
                    ordering_seed_rows.append(
                        {
                            "asset": safe_name(asset),
                            "K": int(k),
                            "model": str(r["model"]),
                            "seed": int(r["seed"]),
                            "metric": "ordering_temporal_top1_seed_mean",
                            "value": float(r["top1_high_risk_consistency_mean"]),
                        }
                    )
                    ordering_seed_rows.append(
                        {
                            "asset": safe_name(asset),
                            "K": int(k),
                            "model": str(r["model"]),
                            "seed": int(r["seed"]),
                            "metric": "ordering_temporal_spearman_seed_mean",
                            "value": float(r["spearman_rank_consistency_mean"]),
                        }
                    )
                    if "high_risk_mean_sign_consistency_mean" in temporal.columns:
                        ordering_seed_rows.append(
                            {
                                "asset": safe_name(asset),
                                "K": int(k),
                                "model": str(r["model"]),
                                "seed": int(r["seed"]),
                                "metric": "ordering_temporal_high_risk_mean_sign_seed_mean",
                                "value": float(r["high_risk_mean_sign_consistency_mean"]),
                            }
                        )
                    if "high_risk_mean_abs_diff_mean" in temporal.columns:
                        ordering_seed_rows.append(
                            {
                                "asset": safe_name(asset),
                                "K": int(k),
                                "model": str(r["model"]),
                                "seed": int(r["seed"]),
                                "metric": "ordering_temporal_high_risk_mean_abs_diff_seed_mean",
                                "value": float(r["high_risk_mean_abs_diff_mean"]),
                            }
                        )
                    if "high_risk_downside_vol_abs_diff_mean" in temporal.columns:
                        ordering_seed_rows.append(
                            {
                                "asset": safe_name(asset),
                                "K": int(k),
                                "model": str(r["model"]),
                                "seed": int(r["seed"]),
                                "metric": "ordering_temporal_high_risk_downside_vol_abs_diff_seed_mean",
                                "value": float(r["high_risk_downside_vol_abs_diff_mean"]),
                            }
                        )

    if ordering_seed_rows:
        ordering_seed_df = pd.DataFrame(ordering_seed_rows)
        ordering_seed_df.to_csv(outputs_dir / "ordering_seed_metrics.csv", index=False)

        ordering_grouped = (
            ordering_seed_df.groupby(["asset", "K", "model", "metric"], as_index=False)["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"count": "n_seeds"})
        )
        ordering_grouped["ci95"] = [
            _ci95(float(s), int(n))
            for s, n in zip(ordering_grouped["std"].astype(float), ordering_grouped["n_seeds"].astype(int))
        ]
        ordering_grouped.to_csv(outputs_dir / "ordering_ci_summary.csv", index=False)

        # Plot: per-asset subplots across K (HMM vs GMM), for Top1 and Spearman.
        try:
            import matplotlib.pyplot as plt

            assets_sorted = [safe_name(a) for a in assets]
            for metric, fname, ylab, ylim in [
                ("ordering_cross_rep_top1_seed_mean", "ordering_crossrep_top1_ci_by_k.png", "cross-rep Top-1 high-risk (mean across seeds)", (-0.05, 1.05)),
                ("ordering_temporal_top1_seed_mean", "ordering_temporal_top1_ci_by_k.png", "temporal Top-1 high-risk (mean across seeds)", (-0.05, 1.05)),
                ("ordering_cross_rep_spearman_seed_mean", "ordering_crossrep_spearman_ci_by_k.png", "cross-rep risk-rank Spearman (mean across seeds)", (-1.05, 1.05)),
                ("ordering_temporal_spearman_seed_mean", "ordering_temporal_spearman_ci_by_k.png", "temporal risk-rank Spearman (mean across seeds)", (-1.05, 1.05)),
            ]:
                fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
                axes = axes.flatten()
                for ax, asset_name in zip(axes, assets_sorted):
                    sub = ordering_grouped[(ordering_grouped["asset"] == asset_name) & (ordering_grouped["metric"] == metric)]
                    if sub.empty:
                        ax.set_title(asset_name)
                        ax.axis("off")
                        continue
                    for model_name in sorted(set(str(x) for x in sub["model"].dropna().unique())):
                        m = sub[sub["model"].astype(str) == model_name].sort_values("K")
                        ax.errorbar(
                            m["K"].astype(int).values,
                            m["mean"].astype(float).values,
                            yerr=m["ci95"].astype(float).values,
                            marker="o",
                            linewidth=1.5,
                            capsize=3,
                            label=model_name,
                        )
                    ax.set_title(asset_name)
                    ax.set_xlabel("K")
                    ax.set_ylabel(ylab)
                    ax.set_ylim(ylim[0], ylim[1])
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best", fontsize=8)
                fig.suptitle(f"Ordering consistency vs K ({metric}; 95% CI across seeds)")
                fig.tight_layout()
                fig.savefig(outputs_dir / fname)
                plt.close(fig)
        except Exception:
            logger.exception("Failed to plot ordering consistency CI figure.")

