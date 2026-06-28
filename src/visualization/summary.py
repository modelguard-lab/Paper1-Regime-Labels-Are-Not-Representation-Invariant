"""
Summary plots and aggregation figures (Paper 1).

Functions:
  plot_cross_rep_box_by_rep                       cross-rep ARI box plot
  plot_box_by_group                               generic grouped box
  plot_line_by_group                              generic grouped line
  plot_ari_vs_step                                ARI vs step rolling plot
  plot_synth_ari_vs_step_by_model                 synthetic ARI by model
  plot_ordering_consistency_summary               ordering consistency
  plot_ari_gap_distribution_from_key_results      ARI gap distribution
  plot_model_split_grouped_bar_from_key_results   model-split grouped bar
"""

from __future__ import annotations

from pathlib import Path
import logging

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


def plot_cross_rep_box_by_rep(
    pairs: pd.DataFrame,
    out_path: Path,
    value_col: str,
    title: str,
) -> None:
    """
    Boxplot of cross-representation agreement, grouped by representation.

    Each pair contributes to both endpoint representations (rep_a and rep_b),
    so all representations appear.
    """

    if pairs.empty:
        return
    if not {"rep_a", "rep_b", value_col}.issubset(pairs.columns):
        raise ValueError(f"pairs must contain rep_a, rep_b, and {value_col}")

    a = pairs[["rep_a", value_col]].rename(columns={"rep_a": "rep"})
    b = pairs[["rep_b", value_col]].rename(columns={"rep_b": "rep"})
    long = pd.concat([a, b], axis=0, ignore_index=True).dropna(
        subset=["rep", value_col]
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    long.boxplot(column=value_col, by="rep", ax=ax, grid=False, rot=45)
    ax.set_title(title)
    ax.set_xlabel("rep")
    ax.set_ylabel(value_col)
    fig.suptitle("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_box_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    df.boxplot(column=value_col, by=group_col, ax=ax, grid=False, rot=45)
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    fig.suptitle("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_line_by_group(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, sub in df.groupby(group):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], label=str(name), marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(loc="best", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ari_vs_step(summary: pd.DataFrame, out_path: Path) -> None:
    """
    Plot temporal ARI and cross-rep ARI vs step (window overlap sensitivity).
    summary must have columns: step, asset, cross_rep_ari_mean, temporal_ari_mean.
    """
    if summary.empty or not {
        "step",
        "asset",
        "cross_rep_ari_mean",
        "temporal_ari_mean",
    }.issubset(summary.columns):
        return
    steps = sorted(summary["step"].unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for asset, sub in summary.groupby("asset"):
        sub = sub.sort_values("step")
        ax1.plot(
            sub["step"],
            sub["temporal_ari_mean"],
            label=asset,
            marker="o",
            linewidth=1.5,
        )
        ax2.plot(
            sub["step"],
            sub["cross_rep_ari_mean"],
            label=asset,
            marker="o",
            linewidth=1.5,
        )
    mean_t = summary.groupby("step")["temporal_ari_mean"].mean()
    mean_c = summary.groupby("step")["cross_rep_ari_mean"].mean()
    ax1.plot(mean_t.index, mean_t.values, "k--", label="mean", linewidth=2)
    ax2.plot(mean_c.index, mean_c.values, "k--", label="mean", linewidth=2)
    ax1.set_xticks(steps)
    ax1.set_xlabel("step (days)")
    ax1.set_ylabel("temporal ARI (mean)")
    ax1.set_title("Temporal ARI vs step (overlap ↓ as step ↑)")
    ax1.legend(loc="best", fontsize=7)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax2.set_xticks(steps)
    ax2.set_xlabel("step (days)")
    ax2.set_ylabel("cross-representation ARI (mean)")
    ax2.set_title("Cross-rep ARI vs step")
    ax2.legend(loc="best", fontsize=7)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_synth_ari_vs_step_by_model(summary: pd.DataFrame, out_path: Path) -> None:
    """
    Synthetic sanity check plot: temporal ARI and cross-rep ARI vs step, with models as lines.

    Expects columns:
    - step (int)
    - model (str) in {"gmm","hmm"} (or any)
    - temporal_mean, temporal_ci95
    - cross_mean, cross_ci95
    """
    required = {"step", "model", "temporal_mean", "cross_mean"}
    if summary.empty or not required.issubset(summary.columns):
        return

    s = summary.copy()
    s["step"] = pd.to_numeric(s["step"], errors="coerce")
    s = s.dropna(subset=["step"])
    s["step"] = s["step"].astype(int)
    steps = sorted(s["step"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for model, sub in s.groupby("model", sort=False):
        sub = sub.sort_values("step")
        y_t = pd.to_numeric(sub.get("temporal_mean"), errors="coerce")
        y_c = pd.to_numeric(sub.get("cross_mean"), errors="coerce")
        e_t = pd.to_numeric(
            sub.get("temporal_ci95", pd.Series(index=sub.index, data=np.nan)),
            errors="coerce",
        )
        e_c = pd.to_numeric(
            sub.get("cross_ci95", pd.Series(index=sub.index, data=np.nan)),
            errors="coerce",
        )

        ax1.errorbar(
            sub["step"].values,
            y_t.values,
            yerr=e_t.values,
            label=str(model),
            marker="o",
            linewidth=1.5,
            capsize=3,
        )
        ax2.errorbar(
            sub["step"].values,
            y_c.values,
            yerr=e_c.values,
            label=str(model),
            marker="o",
            linewidth=1.5,
            capsize=3,
        )

    ax1.set_xticks(steps)
    ax1.set_xlabel("step (days)")
    ax1.set_ylabel("temporal ARI (mean ± 95% CI)")
    ax1.set_title("Synthetic: Temporal ARI vs step (overlap ↓ as step ↑)")
    ax1.legend(loc="best", fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.set_xticks(steps)
    ax2.set_xlabel("step (days)")
    ax2.set_ylabel("cross-representation ARI (mean ± 95% CI)")
    ax2.set_title("Synthetic: Cross-rep ARI vs step")
    ax2.legend(loc="best", fontsize=8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ordering_consistency_summary(ordering: pd.DataFrame, out_path: Path) -> None:
    """
    Plot ordering consistency summary (Top-1 high-risk alignment + Spearman rank consistency).

    Expects seed-level rows with columns:
    - kind in {"cross_rep","temporal"}
    - scope in {"all_rep_pairs","all_reps"} (we plot these two)
    - model, seed
    - top1_high_risk_consistency_mean, spearman_rank_consistency_mean
    """
    required = {
        "kind",
        "scope",
        "model",
        "seed",
        "top1_high_risk_consistency_mean",
        "spearman_rank_consistency_mean",
    }
    if ordering.empty or not required.issubset(ordering.columns):
        return

    cross = ordering[
        (ordering["kind"].astype(str) == "cross_rep")
        & (ordering["scope"].astype(str) == "all_rep_pairs")
    ].copy()
    temp = ordering[
        (ordering["kind"].astype(str) == "temporal")
        & (ordering["scope"].astype(str) == "all_reps")
    ].copy()
    if cross.empty and temp.empty:
        return

    def _agg(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        g = (
            df.groupby("model", as_index=False)[
                ["top1_high_risk_consistency_mean", "spearman_rank_consistency_mean"]
            ]
            .mean(numeric_only=True)
            .rename(
                columns={
                    "top1_high_risk_consistency_mean": "top1",
                    "spearman_rank_consistency_mean": "spearman",
                }
            )
        )
        return g

    c = _agg(cross)
    t = _agg(temp)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey="row")
    ax = axes[0, 0]
    if not c.empty:
        ax.bar(c["model"].astype(str), c["top1"].astype(float))
    ax.set_title("Cross-rep Top-1 high-risk consistency")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 1]
    if not c.empty:
        ax.bar(c["model"].astype(str), c["spearman"].astype(float))
    ax.set_title("Cross-rep risk-rank consistency (Spearman)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    if not t.empty:
        ax.bar(t["model"].astype(str), t["top1"].astype(float))
    ax.set_title("Temporal Top-1 high-risk consistency")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    if not t.empty:
        ax.bar(t["model"].astype(str), t["spearman"].astype(float))
    ax.set_title("Temporal risk-rank consistency (Spearman)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    for a in axes.flatten():
        a.set_xlabel("model")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ari_gap_distribution_from_key_results(
    key_results: pd.DataFrame,
    out_path: Path,
    title: str = "Temporal-minus-cross ARI across assets and models",
) -> None:
    """
    Plot gap = temporal_ari_mean - cross_rep_ari_mean for each (asset, scope).

    - Each dot is one (asset, scope) pair (e.g., model class or other scope).
    - Includes a subtle 0 reference line and a highlighted mean line.
    - Avoids histogram bin-interpretation issues for small n.

    Expects columns: asset, metric, scope, value.
    """
    required = {"asset", "metric", "scope", "value"}
    if (
        key_results is None
        or key_results.empty
        or not required.issubset(key_results.columns)
    ):
        return

    df = key_results.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["asset", "metric", "scope", "value"])

    # Keep only needed metrics
    df = df[df["metric"].isin(["cross_rep_ari_mean", "temporal_ari_mean"])]
    if df.empty:
        return

    # Wide form per (asset, scope). aggfunc=mean guards against duplicates.
    wide = df.pivot_table(
        index=["asset", "scope"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).dropna(subset=["cross_rep_ari_mean", "temporal_ari_mean"])

    if wide.empty:
        return

    wide["gap"] = wide["temporal_ari_mean"] - wide["cross_rep_ari_mean"]
    gaps = wide["gap"].astype(float).to_numpy()

    n = int(gaps.size)
    mean_gap = float(np.mean(gaps))

    # --- Figure (paper-friendly defaults) ---
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=300)

    # Jitter in y so points don't overlap visually
    rng = np.random.default_rng(0)
    y = rng.normal(loc=0.0, scale=0.03, size=n)

    ax.scatter(
        gaps,
        y,
        s=34,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )

    # Subtle zero line (background reference)
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="gray",
        alpha=0.6,
        zorder=1,
    )

    # Mean line (primary emphasis); use solid to avoid "chunky" dashed artifacts
    ax.axvline(
        mean_gap,
        linestyle="-",
        linewidth=2.0,
        zorder=2,
        label=f"mean = {mean_gap:.3f}",
    )

    # Make line caps crisp (helps on some backends)
    for line in ax.lines:
        line.set_solid_capstyle("butt")

    ax.set_title(f"{title} (n={n})", fontsize=12)
    ax.set_xlabel("gap = temporal ARI − cross-representation ARI", fontsize=11)

    # Strip-plot aesthetics: no y-axis meaning
    ax.set_yticks([])
    ax.set_ylim(-0.12, 0.12)

    # Tight x-limits while still showing 0
    xmin = min(float(np.min(gaps)), 0.0) - 0.02
    xmax = max(float(np.max(gaps)), 0.0) + 0.02
    ax.set_xlim(xmin, xmax)

    # Light grid on x only
    ax.grid(True, axis="x", alpha=0.2, linewidth=0.7)
    ax.grid(False, axis="y")

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.legend(loc="upper left", frameon=False, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_model_split_grouped_bar_from_key_results(
    key_results: pd.DataFrame,
    out_path: Path,
    title: str = "Cross-representation vs temporal ARI by model class",
) -> None:
    """
    Grouped bar chart of cross-representation vs temporal ARI by model class.

    Expects rows with:
      - metric in {"cross_rep_ari_mean", "temporal_ari_mean"}
      - scope in {"model=gmm", "model=hmm"}
      - value numeric

    Aggregation across assets is done by simple averaging.
    """
    required = {"metric", "scope", "value"}
    if (
        key_results is None
        or key_results.empty
        or not required.issubset(key_results.columns)
    ):
        return

    df = key_results.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    def _scope_to_model(s: str) -> str | None:
        if isinstance(s, str) and s.startswith("model="):
            return s.split("=", 1)[1].strip().lower()
        return None

    df["model"] = df["scope"].map(_scope_to_model)
    df = df[df["model"].isin(["gmm", "hmm"])]
    df = df[df["metric"].isin(["cross_rep_ari_mean", "temporal_ari_mean"])]
    if df.empty:
        return

    grouped = df.groupby(["model", "metric"], as_index=False)["value"].mean(
        numeric_only=True
    )

    pivot = grouped.pivot(index="model", columns="metric", values="value").reindex(
        ["gmm", "hmm"]
    )
    if pivot.isna().all().all():
        return

    models = [m.upper() for m in pivot.index.astype(str).tolist()]
    cross = pivot["cross_rep_ari_mean"].to_numpy()
    temp = pivot["temporal_ari_mean"].to_numpy()

    # --- Plot styling (paper-friendly) ---
    x = np.arange(len(models))  # [0, 1, ...]
    width = 0.34  # readable in print

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=200)

    b1 = ax.bar(
        x - width / 2,
        cross,
        width,
        label="Cross-rep ARI",
        edgecolor="black",
        linewidth=0.6,
    )
    b2 = ax.bar(
        x + width / 2,
        temp,
        width,
        label="Temporal ARI",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("ARI (mean across assets)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)

    # ARI is usually in [0,1]; keep axis tight and clean
    ax.set_ylim(0.0, 1.0)

    # Light y-grid only
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.grid(False, axis="x")

    # Legend: keep it unobtrusive
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    # --- Value labels on bars ---
    def _label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.02,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    _label_bars(b1)
    _label_bars(b2)

    # --- Emphasize the gap (Temp - Cross) ---
    # Adds an annotation per model showing the representation-uncertainty gap.
    for i, (c, t) in enumerate(zip(cross, temp)):
        if np.isfinite(c) and np.isfinite(t):
            gap = t - c
            y = max(c, t) + 0.10
            y = min(y, 0.95)  # avoid hitting the top
            ax.annotate(
                f"gap={gap:.2f}",
                xy=(x[i], max(c, t)),
                xytext=(x[i], y),
                ha="center",
                va="bottom",
                fontsize=9,
                arrowprops=dict(arrowstyle="-", linewidth=0.8, alpha=0.7),
            )

    # Clean spines (optional but helps in papers)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
