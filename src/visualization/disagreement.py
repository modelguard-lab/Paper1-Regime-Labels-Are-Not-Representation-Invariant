"""
Cross-resolution disagreement and pairwise heatmap figures (Paper 1).

Functions:
  plot_disagreement_timeseries     timeseries of cross-rep disagreement
  plot_stability_heatmap           per-roll-window stability heatmap
  plot_pairwise_matrix_heatmap     pairwise-rep ARI matrix heatmap
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


def plot_disagreement_timeseries(
    hard_map: dict,
    price: pd.Series,
    out_path: Path,
    reps: list[str],
    model: str = "hmm",
    seed: int = 1,
    roll: str | None = None,
    smooth_window: int = 63,
    stress_periods: list[tuple[str, str, str]] | None = None,
) -> None:
    """Plot rolling cross-representation disagreement rate over the full sample.

    For each date, computes the fraction of representation pairs that assign
    different coarse risk labels (high-risk vs not).  A rolling average smooths
    the series for readability.

    Parameters
    ----------
    hard_map : dict
        Mapping (rep, model, seed, roll) -> pd.Series of state labels.
    price : pd.Series
        Price series (used as background).
    reps : list[str]
        Representation names to include.
    model, seed, roll : str, int, str
        Selectors for the state sequences. If roll is None, the first
        available roll is used.
    smooth_window : int
        Rolling-average window for the disagreement rate.
    stress_periods : list of (start, end, label)
        Date ranges to shade (e.g. COVID, 2022 inflation).
    """
    from itertools import combinations

    if stress_periods is None:
        stress_periods = [
            ("2020-02-19", "2020-06-08", "COVID-19"),
            ("2022-01-03", "2022-10-14", "2022 inflation"),
        ]

    # Collect risk labels across ALL rolls for each rep.
    # State numbering can differ between rolls, so we compute risk labels
    # (binary: 1=high-risk, 0=not) per roll, then stitch them together.
    ret = price.pct_change()
    risk_labels: dict[str, pd.Series] = {}
    for rep in reps:
        keys = [k for k in hard_map if k[0] == rep and k[1] == model and k[2] == seed]
        if roll is not None:
            keys = [k for k in keys if k[3] == roll]
        if not keys:
            continue
        # Process rolls in order so later rolls overwrite earlier ones
        # on overlapping dates.
        roll_risk: list[pd.Series] = []
        for key in sorted(keys, key=lambda k: k[3]):
            s = hard_map[key].dropna().astype(int)
            if s.empty:
                continue
            n_states = int(s.max()) + 1
            state_means = {}
            for st in range(n_states):
                idx = s[s == st].index
                r = ret.reindex(idx).dropna()
                state_means[st] = float(r.mean()) if len(r) > 0 else 0.0
            worst_state = min(state_means, key=state_means.get)
            roll_risk.append(s.map(lambda x, ws=worst_state: 1 if x == ws else 0))
        if roll_risk:
            # Concatenate; later rolls overwrite earlier ones on duplicate dates
            combined = pd.concat(roll_risk)
            risk_labels[rep] = combined[~combined.index.duplicated(keep="last")].sort_index()

    available_reps = list(risk_labels.keys())
    if len(available_reps) < 2:
        logger.warning("plot_disagreement_timeseries: fewer than 2 reps available; skipping.")
        return

    # For each date, compute fraction of pairs that disagree
    all_dates = sorted(set().union(*(rl.index for rl in risk_labels.values())))
    all_dates = pd.DatetimeIndex(all_dates)
    pairs = list(combinations(available_reps, 2))
    n_pairs = len(pairs)

    disagree = pd.Series(np.nan, index=all_dates, name="disagreement_rate")
    for dt in all_dates:
        n_disagree = 0
        n_valid = 0
        for ra, rb in pairs:
            la = risk_labels[ra]
            lb = risk_labels[rb]
            if dt in la.index and dt in lb.index:
                n_valid += 1
                if la[dt] != lb[dt]:
                    n_disagree += 1
        if n_valid > 0:
            disagree[dt] = n_disagree / n_valid

    disagree_smooth = disagree.rolling(smooth_window, min_periods=1).mean()

    # Plot
    fig, (ax_price, ax_dis) = plt.subplots(
        2, 1, figsize=(6.4, 4.0), dpi=300, sharex=True,
        gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.08},
    )

    # Price panel
    p = price.reindex(all_dates).dropna()
    ax_price.plot(p.index, p.values, color="black", linewidth=0.8)
    ax_price.set_ylabel(price.name or "Price", fontsize=8)
    ax_price.tick_params(labelsize=7)
    ax_price.grid(True, alpha=0.2, linewidth=0.5)

    # Disagreement panel
    ax_dis.fill_between(
        disagree_smooth.index, 0, disagree_smooth.values,
        color="#1f77b4", alpha=0.3, linewidth=0,
    )
    ax_dis.plot(
        disagree_smooth.index, disagree_smooth.values,
        color="#1f77b4", linewidth=1.0,
    )
    ax_dis.set_ylabel("Disagreement rate", fontsize=8)
    ax_dis.set_ylim(0, 1)
    ax_dis.tick_params(labelsize=7)
    ax_dis.grid(True, alpha=0.2, linewidth=0.5)

    # Stress-period shading on both panels
    for start, end, label in stress_periods:
        for ax in (ax_price, ax_dis):
            ax.axvspan(
                pd.Timestamp(start), pd.Timestamp(end),
                color="#ff7f0e", alpha=0.12, zorder=0,
            )
        # Label at the top of each shaded band
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax_price.annotate(
            label, xy=(mid, ax_price.get_ylim()[1]),
            xytext=(0, 2), textcoords="offset points",
            ha="center", va="bottom", fontsize=5.5, color="#cc6600",
            annotation_clip=False,
        )

    ax_dis.set_xlabel("Date", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("plot_disagreement_timeseries: saved to %s", out_path)


def plot_stability_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    value_col: str,
    index_col: str,
    columns_col: str,
) -> None:
    pivot = df.pivot_table(
        values=value_col, index=index_col, columns=columns_col, aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{value_col} heatmap")
    fig.colorbar(im, ax=ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pairwise_matrix_heatmap(
    pairs: pd.DataFrame,
    out_path: Path,
    value_col: str,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    """
    Plot a symmetric rep×rep matrix heatmap from pairwise rows.

    Expects columns: rep_a, rep_b, and `value_col`. The matrix is filled symmetrically.
    """

    if pairs.empty:
        return
    if not {"rep_a", "rep_b", value_col}.issubset(pairs.columns):
        raise ValueError(f"pairs must contain rep_a, rep_b, and {value_col}")

    reps = sorted(
        set(pairs["rep_a"].dropna().astype(str))
        | set(pairs["rep_b"].dropna().astype(str))
    )
    idx = {r: i for i, r in enumerate(reps)}
    mat = np.full((len(reps), len(reps)), np.nan, dtype=float)
    np.fill_diagonal(mat, 1.0)

    grouped = (
        pairs.dropna(subset=["rep_a", "rep_b", value_col])
        .groupby(["rep_a", "rep_b"], as_index=False)[value_col]
        .mean()
    )
    for _, row in grouped.iterrows():
        a = str(row["rep_a"])
        b = str(row["rep_b"])
        v = float(row[value_col])
        if a in idx and b in idx:
            i = idx[a]
            j = idx[b]
            mat[i, j] = v
            mat[j, i] = v

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(reps)))
    ax.set_xticklabels(reps, rotation=45, ha="right")
    ax.set_yticks(range(len(reps)))
    ax.set_yticklabels(reps)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


