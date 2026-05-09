"""
Output writers for the Paper 1 pipeline.

Centralises the per-asset CSV / JSON / figure emission that was
previously inlined in workflows.pipeline. Imported by
workflows.single_asset and workflows.pipeline.

Functions:
  _mean_or_nan        nan-safe mean
  _write_key_outputs  scores / stability / semantic / ordering CSVs
  _write_plots        cross-rep box, line, pairwise heatmap PNGs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.core.utils import ensure_dir, safe_name
from src.visualization.plots import (
    plot_ari_vs_step,
    plot_cross_rep_box_by_rep,
    plot_line_by_group,
    plot_ordering_consistency_summary,
    plot_pairwise_matrix_heatmap,
)

logger = logging.getLogger(__name__)




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
    fit_quality: pd.DataFrame,
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
            eval_mode = (
                str(tmp["temporal_eval_mode"].mode().iloc[0])
                if "temporal_eval_mode" in tmp.columns and not tmp["temporal_eval_mode"].dropna().empty
                else "overlap"
            )
            rows.append(
                {
                    "metric": "temporal_ari_mean",
                    "scope": "all",
                    "value": _mean_or_nan(tmp["ari"]),
                    "n": int(len(tmp)),
                }
            )
            rows.append(
                {
                    "metric": "temporal_eval_mode",
                    "scope": "all",
                    "value": eval_mode,
                    "n": int(len(tmp)),
                }
            )
            if "overlap_ratio" in tmp.columns:
                rows.append(
                    {
                        "metric": "temporal_overlap_ratio_mean",
                        "scope": "all",
                        "value": _mean_or_nan(pd.to_numeric(tmp["overlap_ratio"], errors="coerce")),
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
            # Keep overlap-only diagnostics to make temporal inflation auditable.
            for metric_col in ("ari", "ami", "vi"):
                col = f"{metric_col}_overlap"
                if col in tmp.columns:
                    rows.append(
                        {
                            "metric": f"temporal_overlap_{metric_col}_mean",
                            "scope": "all",
                            "value": _mean_or_nan(pd.to_numeric(tmp[col], errors="coerce")),
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
                    if "overlap_ratio" in g.columns:
                        rows.append(
                            {
                                "metric": "temporal_overlap_ratio_mean",
                                "scope": f"model={model_name}",
                                "value": _mean_or_nan(pd.to_numeric(g["overlap_ratio"], errors="coerce")),
                                "n": int(len(g)),
                            }
                        )

    # Seed-level CIs (emitted when ≥3 seeds so paper_autofill can report mean ± CI)
    if "seed" in stability.columns and stability["seed"].nunique() >= 3:
        from scipy import stats as _sp_stats

        def _seed_ci(series: pd.Series) -> tuple[float, float]:
            """Return (mean, 95% CI half-width) across seed-level means."""
            arr = series.dropna().values.astype(float)
            if len(arr) < 2:
                return float(np.nanmean(arr)), float("nan")
            sem = float(_sp_stats.sem(arr))
            hw = sem * float(_sp_stats.t.ppf(0.975, len(arr) - 1))
            return float(np.mean(arr)), hw

        # Cross-rep seed CIs
        if {"rep_a", "rep_b", "ari"}.issubset(stability.columns):
            cross_df = stability.dropna(subset=["rep_a", "rep_b", "ari"]).copy()
            cross_df = cross_df[cross_df["rep_a"] != cross_df["rep_b"]]
            if not cross_df.empty and "model" in cross_df.columns:
                for model_name, g in cross_df.groupby("model"):
                    seed_means = g.groupby("seed")["ari"].mean()
                    if len(seed_means) >= 3:
                        mu, ci = _seed_ci(seed_means)
                        rows.append({"metric": "cross_rep_ari_seed_mean", "scope": f"model={model_name}", "value": mu, "n": int(len(seed_means))})
                        rows.append({"metric": "cross_rep_ari_seed_ci95", "scope": f"model={model_name}", "value": ci, "n": int(len(seed_means))})

        # Temporal seed CIs
        if {"roll_a", "roll_b", "ari"}.issubset(stability.columns):
            tmp = stability.dropna(subset=["ari", "roll_a", "roll_b"]).copy()
            if not tmp.empty and "model" in tmp.columns:
                for model_name, g in tmp.groupby("model"):
                    seed_means = g.groupby("seed")["ari"].mean()
                    if len(seed_means) >= 3:
                        mu, ci = _seed_ci(seed_means)
                        rows.append({"metric": "temporal_ari_seed_mean", "scope": f"model={model_name}", "value": mu, "n": int(len(seed_means))})
                        rows.append({"metric": "temporal_ari_seed_ci95", "scope": f"model={model_name}", "value": ci, "n": int(len(seed_means))})

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
    if not scores.empty and {"model", "hmm_diag_fallback"}.issubset(scores.columns):
        hmm_scores = scores[scores["model"].astype(str) == "hmm"].copy()
        if not hmm_scores.empty:
            fb = pd.to_numeric(hmm_scores["hmm_diag_fallback"], errors="coerce").dropna()
            if not fb.empty:
                rows.append(
                    {
                        "metric": "hmm_diag_fallback_rate",
                        "scope": "all",
                        "value": float(fb.mean()),
                        "n": int(len(fb)),
                    }
                )
    if not fit_quality.empty and {"expected", "success", "failed"}.issubset(fit_quality.columns):
        fq = fit_quality.copy()
        expected_all = float(pd.to_numeric(fq["expected"], errors="coerce").sum())
        success_all = float(pd.to_numeric(fq["success"], errors="coerce").sum())
        failed_all = float(pd.to_numeric(fq["failed"], errors="coerce").sum())
        if expected_all > 0:
            rows.append(
                {
                    "metric": "fit_success_rate",
                    "scope": "all",
                    "value": success_all / expected_all,
                    "n": int(expected_all),
                }
            )
            rows.append(
                {
                    "metric": "fit_failure_rate",
                    "scope": "all",
                    "value": failed_all / expected_all,
                    "n": int(expected_all),
                }
            )
        if "model" in fq.columns:
            fq_model = fq[fq["model"].astype(str) != "all"]
            for model_name, g in fq_model.groupby("model"):
                expected_m = float(pd.to_numeric(g["expected"], errors="coerce").sum())
                success_m = float(pd.to_numeric(g["success"], errors="coerce").sum())
                failed_m = float(pd.to_numeric(g["failed"], errors="coerce").sum())
                if expected_m <= 0:
                    continue
                rows.append(
                    {
                        "metric": "fit_success_rate",
                        "scope": f"model={model_name}",
                        "value": success_m / expected_m,
                        "n": int(expected_m),
                    }
                )
                rows.append(
                    {
                        "metric": "fit_failure_rate",
                        "scope": f"model={model_name}",
                        "value": failed_m / expected_m,
                        "n": int(expected_m),
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
        "# Paper 1: Unified Run Analysis",
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


