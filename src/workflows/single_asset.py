"""
Per-asset orchestrator for the Paper 1 pipeline.

`_run_single_asset` wires together the bottom-up call stack:
  - download / load price data (core.data)
  - build representations (core.features)
  - fit GMM / HMM models on rolling windows (core.fits)
  - compute cross-representation and temporal stability (core.stability)
  - compute ordering consistency + null baselines (core.ordering)
  - write CSV / JSON / figure outputs (workflows.outputs)

This is the integration glue called from workflows.pipeline.run() and
from workflows.sweeps.* for step / robustness sweeps.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.data import download_tickers, load_prices
from src.core.features import build_representation_single
from src.core.fits import _fit_slice_collect, _run_parallel_sharded_fits
from src.core.metrics import semantic_drift
from src.core.ordering import (
    _compute_ordering_consistency_crossrep_seed_summary,
    _compute_ordering_consistency_temporal_seed_summary,
    _compute_ordering_null_baseline,
)
from src.core.stability import (
    _compute_rep_stability_from_map,
    _compute_semantic_crossrep_wasserstein_from_map,
    _compute_semantic_temporal_wasserstein_from_map,
    _compute_window_stability_from_map,
    _load_hard_map_from_rep_csv,
)
from src.core.utils import (
    _build_rep_configs,
    _fmt_hms,
    _rmtree_with_retries,
    _window_roll_name,
    ensure_dir,
    rolling_slices,
    safe_name,
    save_json,
)
from src.workflows.outputs import _write_key_outputs, _write_plots

logger = logging.getLogger(__name__)


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

    # Filter representations by asset_filter and apply to current asset.
    reps = [r for r in reps if r.asset_filter is None or asset in r.asset_filter]
    if not reps:
        logger.warning("[%s] No representations applicable for this asset; skipping.", asset)
        return float(time.perf_counter() - t_asset0)

    # Load auxiliary series (e.g. ^VIX) for any reps that need them.
    vix_features = {"vix_level", "vix_change", "vix_percentile"}
    aux: Dict[str, pd.Series] = {}
    if any(vix_features.intersection(r.features) for r in reps):
        data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
        vix_missing = not (raw_dir / "VIX.csv").exists()
        if vix_missing:
            logger.info("[%s] Downloading ^VIX for VIX-based representation.", asset)
            download_tickers(
                ["^VIX"],
                output_dir=raw_dir,
                start_date=str(data_cfg.get("start_date", "2005-01-01")),
                end_date=str(data_cfg.get("end_date")) if data_cfg.get("end_date") else None,
            )
        vix_prices = load_prices(["^VIX"], raw_dir, price_col=None)
        if "^VIX" in vix_prices:
            aux["^VIX"] = vix_prices["^VIX"]
        else:
            logger.warning("[%s] ^VIX could not be loaded; skipping VIX-based reps.", asset)
            reps = [r for r in reps if not vix_features.intersection(r.features)]

    # Build all representations once, then align to a common post-warmup index.
    X_by_rep: Dict[str, pd.DataFrame] = {}
    first_valid: Dict[str, pd.Timestamp] = {}
    for rep in reps:
        X_raw = build_representation_single(price, rep, aux=aux if aux else None)
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
    fit_quality_rows: List[Dict] = []

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
        expected_by_model: Dict[str, int] = {str(m): 0 for m in models}
        for model_name, *_ in tasks:
            expected_by_model[str(model_name)] = int(expected_by_model.get(str(model_name), 0) + 1)
        success_by_model: Dict[str, int] = {str(m): 0 for m in models}

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
                    sc_rows = sc.to_dict(orient="records")
                    score_rows.extend(sc_rows)
                    success_by_model[str(m)] = int(len(sc_rows))

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
            success_by_model[model_name] = int(success_by_model.get(model_name, 0) + 1)

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
        fit_quality_rows.append(
            {
                "rep": rep.name,
                "model": "all",
                "expected": int(len(tasks)),
                "success": int(ok),
                "failed": int(fail),
                "success_rate": (float(ok) / float(len(tasks))) if len(tasks) > 0 else float("nan"),
            }
        )
        for model_name in sorted(expected_by_model.keys()):
            exp_m = int(expected_by_model.get(model_name, 0))
            ok_m = int(success_by_model.get(model_name, 0))
            fit_quality_rows.append(
                {
                    "rep": rep.name,
                    "model": str(model_name),
                    "expected": exp_m,
                    "success": ok_m,
                    "failed": int(max(0, exp_m - ok_m)),
                    "success_rate": (float(ok_m) / float(exp_m)) if exp_m > 0 else float("nan"),
                }
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
        n_jobs=n_jobs,
    )
    window_stability = _compute_window_stability_from_map(
        hard_map=hard_map,
        rep_names=rep_names,
        models=models,
        k=k,
        window=w,
        seeds=seeds,
        rolls=rolls,
        n_jobs=n_jobs,
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
        n_jobs=n_jobs,
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
        n_jobs=n_jobs,
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

    # Chance-level baseline for ordering metrics (random permutation null).
    ordering_null = _compute_ordering_null_baseline(
        hard_map=hard_map,
        returns=returns,
        rep_names=rep_names,
        models=models,
        k=k,
        seeds=seeds,
        rolls=rolls,
        n_perm=500,
        alpha=0.05,
    )

    fit_quality = pd.DataFrame(fit_quality_rows)
    scores.to_csv(plots_dir / "scores_summary.csv", index=False)
    stability.to_csv(plots_dir / "stability_summary.csv", index=False)
    if not semantic.empty:
        semantic.to_csv(plots_dir / "semantic_summary.csv", index=False)
    if not ordering.empty:
        ordering.to_csv(plots_dir / "ordering_consistency_seed_summary.csv", index=False)
    # Save null baseline
    if ordering_null.get("null_n", 0) > 0:
        import json as _json
        (plots_dir / "ordering_null_baseline.json").write_text(
            _json.dumps(ordering_null, indent=2)
        )
        try:
            plot_ordering_consistency_summary(ordering, plots_dir / "ordering_consistency.png")
        except Exception as e:
            logger.warning("[%s] Could not plot ordering consistency: %s", ctx, e)
    if not fit_quality.empty:
        fit_quality.to_csv(plots_dir / "fit_quality_summary.csv", index=False)
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
    _write_key_outputs(scores, stability, semantic, ordering, fit_quality, out_dir)

    elapsed_s = float(time.perf_counter() - t_asset0)
    logger.info("Run finished; asset=%s elapsed_s=%.1f (%s)", asset, elapsed_s, _fmt_hms(elapsed_s))
    return elapsed_s


