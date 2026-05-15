"""
Step-size and robustness sweep machinery for the Paper 1 pipeline.

Each sweep iterates `_run_single_asset` over a parameter grid (alternative
step sizes for the rolling window; alternative K-state and window settings
for robustness analyses), aggregating per-config results into a sweep
summary. Workers are launched in parallel via joblib.

Public functions:
  _step_sweep_asset_worker     joblib worker for step-sweep entry
  _robustness_asset_worker     joblib worker for robustness-sweep entry
  _run_step_sweep              top-level step-sweep orchestrator
  _run_robustness_sweep        top-level robustness-sweep orchestrator
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from src.core.runtime import configure_global_file_logging
from src.core.utils import (
    _fmt_hms,
    _timing_summary_lines,
    ensure_dir,
    safe_name,
)
from src.workflows.single_asset import _run_single_asset

logger = logging.getLogger(__name__)


def _step_sweep_asset_worker(args_tuple):
    """Module-level worker for ProcessPoolExecutor; runs one (asset, step) combo.

    Each call lives in a freshly spawned Python process, so joblib loky inside
    is fully independent (no nested-loky downgrade to threading).
    """
    asset, cfg, outputs_dir_str, step, inner_jobs, log_path_str = args_tuple
    outputs_dir_local = Path(outputs_dir_str)
    log_path_local = Path(log_path_str)
    try:
        configure_global_file_logging(log_path_local)
    except Exception:
        pass
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["grid"] = cfg_copy.get("grid") or {}
    cfg_copy["grid"]["step"] = int(step)
    cfg_copy["grid"]["n_jobs"] = int(inner_jobs)
    out_dir = outputs_dir_local / safe_name(asset) / f"step_{int(step)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return float(_run_single_asset(asset, cfg_copy, outputs_dir_local, out_dir_override=out_dir))


def _robustness_asset_worker(args_tuple):
    """Module-level worker for ProcessPoolExecutor; runs one (asset, K) combo."""
    asset, cfg, outputs_dir_str, k, inner_jobs, log_path_str = args_tuple
    outputs_dir_local = Path(outputs_dir_str)
    log_path_local = Path(log_path_str)
    try:
        configure_global_file_logging(log_path_local)
    except Exception:
        pass
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["grid"] = cfg_copy.get("grid") or {}
    cfg_copy["grid"]["n_states"] = [int(k)]
    cfg_copy["grid"]["n_jobs"] = int(inner_jobs)
    out_dir = outputs_dir_local / safe_name(asset) / "robustness" / f"K_{int(k)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return float(_run_single_asset(asset, cfg_copy, outputs_dir_local, out_dir_override=out_dir))


def _run_step_sweep(cfg: Dict, assets: List[str], outputs_dir: Path, steps: List[int]) -> None:
    """Run pipeline for each step in steps; asset-first layout; write step_sweep_summary.csv and ari_vs_step.png."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as _mp
    t0 = time.perf_counter()
    logger.info("Starting step sweep; assets=%s steps=%s", ",".join(assets), steps)

    cfg["grid"] = cfg.get("grid") or {}
    n_jobs_total = int(cfg["grid"].get("n_jobs", 1))
    n_asset_workers = max(1, int(cfg["grid"].get("n_asset_workers", 1)))
    inner_jobs = max(1, n_jobs_total // n_asset_workers)
    if n_asset_workers > 1:
        logger.info(
            "Step sweep: ProcessPool outer parallelism n_asset_workers=%d inner_jobs=%d (total=%d)",
            n_asset_workers, inner_jobs, n_asset_workers * inner_jobs,
        )

    summary_rows: List[Dict] = []
    totals_by_asset: Dict[str, float] = {}
    totals_by_step: Dict[str, float] = {}
    log_path_str = str(outputs_dir / "run.log")
    outputs_dir_str = str(outputs_dir)

    def _run_one_asset(asset: str, step: int) -> float:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["grid"]["step"] = int(step)
        cfg_copy["grid"]["n_jobs"] = inner_jobs
        out_dir = outputs_dir / safe_name(asset) / f"step_{int(step)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Step sweep: asset=%s step=%d -> %s", asset, int(step), out_dir)
        return _run_single_asset(asset, cfg_copy, outputs_dir, out_dir_override=out_dir)

    for step in steps:
        t_step0 = time.perf_counter()
        if n_asset_workers > 1:
            args_list = [
                (asset, cfg, outputs_dir_str, int(step), int(inner_jobs), log_path_str)
                for asset in assets
            ]
            ctx = _mp.get_context("spawn")
            elapsed_by_asset: Dict[str, float] = {}
            with ProcessPoolExecutor(max_workers=n_asset_workers, mp_context=ctx) as executor:
                futures = {executor.submit(_step_sweep_asset_worker, a): a[0] for a in args_list}
                for fut in as_completed(futures):
                    asset = futures[fut]
                    try:
                        elapsed = float(fut.result())
                    except Exception:
                        logger.exception("Step sweep worker failed for asset=%s step=%d", asset, int(step))
                        elapsed = 0.0
                    elapsed_by_asset[safe_name(asset)] = elapsed
                    logger.info("Step sweep: asset=%s step=%d done in %.1fs", asset, int(step), elapsed)
            for asset in assets:
                key = safe_name(asset)
                totals_by_asset[key] = totals_by_asset.get(key, 0.0) + float(elapsed_by_asset.get(key, 0.0))
        else:
            for asset in assets:
                elapsed = _run_one_asset(asset, step)
                totals_by_asset[safe_name(asset)] = totals_by_asset.get(safe_name(asset), 0.0) + float(elapsed)
        # Lazy import to avoid circular dependency with src.workflows.pipeline.
        from src.workflows.pipeline import _extract_metrics_from_key_results
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

    n_jobs_total = int(cfg["grid"].get("n_jobs", 1))
    n_asset_workers = max(1, int(cfg["grid"].get("n_asset_workers", 1)))
    inner_jobs = max(1, n_jobs_total // n_asset_workers)
    if n_asset_workers > 1:
        logger.info(
            "Robustness sweep: asset-level parallelism n_asset_workers=%d inner_jobs=%d",
            n_asset_workers, inner_jobs,
        )

    def _run_one_asset_k(asset: str, k: int) -> float:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["grid"]["n_states"] = [int(k)]
        cfg_copy["grid"]["n_jobs"] = inner_jobs
        out_dir = outputs_dir / safe_name(asset) / "robustness" / f"K_{int(k)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Robustness: asset=%s K=%d -> %s", asset, int(k), out_dir)
        return _run_single_asset(asset, cfg_copy, outputs_dir, out_dir_override=out_dir)

    # Run K sweep (each K runs all seeds internally)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as _mp
    log_path_str = str(outputs_dir / "run.log")
    outputs_dir_str = str(outputs_dir)
    totals_by_asset: Dict[str, float] = {}
    totals_by_k: Dict[str, float] = {}
    for k in ks:
        t_k0 = time.perf_counter()
        if n_asset_workers > 1:
            args_list = [
                (asset, cfg, outputs_dir_str, int(k), int(inner_jobs), log_path_str)
                for asset in assets
            ]
            ctx = _mp.get_context("spawn")
            elapsed_by_asset: Dict[str, float] = {}
            with ProcessPoolExecutor(max_workers=n_asset_workers, mp_context=ctx) as executor:
                futures = {executor.submit(_robustness_asset_worker, a): a[0] for a in args_list}
                for fut in as_completed(futures):
                    asset = futures[fut]
                    try:
                        elapsed = float(fut.result())
                    except Exception:
                        logger.exception("Robustness worker failed for asset=%s K=%d", asset, int(k))
                        elapsed = 0.0
                    elapsed_by_asset[safe_name(asset)] = elapsed
                    logger.info("Robustness: asset=%s K=%d done in %.1fs", asset, int(k), elapsed)
            for asset in assets:
                key = safe_name(asset)
                totals_by_asset[key] = totals_by_asset.get(key, 0.0) + float(elapsed_by_asset.get(key, 0.0))
        else:
            for asset in assets:
                elapsed = _run_one_asset_k(asset, k)
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

        # Aggregate per-K independent null values from each asset/K subdirectory.
        null_rows_k: List[Dict] = []
        for asset in assets:
            for k in ks:
                null_path = (
                    outputs_dir / safe_name(asset) / "robustness" / f"K_{int(k)}"
                    / "plots" / "ordering_null_baseline.json"
                )
                if not null_path.exists():
                    continue
                try:
                    nb = json.loads(null_path.read_text())
                except Exception:
                    continue
                null_rows_k.append({
                    "asset": safe_name(asset),
                    "K": int(k),
                    "null_top1_mean": nb.get("null_top1_mean"),
                    "null_spearman_mean": nb.get("null_spearman_mean"),
                    "indep_null_top1_mean": nb.get("indep_null_top1_mean"),
                    "indep_null_spearman_mean": nb.get("indep_null_spearman_mean"),
                })
        if null_rows_k:
            pd.DataFrame(null_rows_k).to_csv(outputs_dir / "ordering_null_by_k.csv", index=False)

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

