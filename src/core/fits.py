"""
Fit primitives for the Paper 1 pipeline.

Per-(rep, model, K, seed) GMM / HMM fitting and shard writing. These
helpers are the bottom of the call stack: they take feature panels and
emit windows_states_hard CSVs without touching higher-level
orchestration.

Functions:
  _fit_one                        single GMM / HMM fit
  _fit_slice_collect              fit across reps / models for one slice
  _append_csv_row                 generic CSV row appender
  _fit_slice_write_shard          fit + immediate shard write
  _run_parallel_sharded_fits      top-level parallel fit orchestrator
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError

from src.core.features import RepConfig, build_representation_single
from src.core.models import fit_gmm, fit_hmm
from src.core.utils import ensure_dir, safe_name

logger = logging.getLogger(__name__)


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
            "hmm_diag_fallback",
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
            "hmm_diag_fallback": scores.get("hmm_diag_fallback"),
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
