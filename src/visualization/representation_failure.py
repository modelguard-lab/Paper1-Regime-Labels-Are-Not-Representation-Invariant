"""
Representation Failure Matrix figure (Paper 1 signature plot).

The 517-line implementation that draws the per-asset state band layout
showing where representations disagree on regime label assignments. Used
by the pipeline and by post-hoc figure regeneration.

Public entry point: ``plot_representation_failure_matrix``.
Internal helpers: ``_compute_state_risk_ranks``, ``_color_for_level``,
``_draw_state_band_no_legend``.
"""

from __future__ import annotations

from pathlib import Path
import logging

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


logger = logging.getLogger(__name__)


def _compute_state_risk_ranks(
    returns: pd.Series, states: pd.Series, n_states: int
) -> dict:
    """
    Compute a simple volatility-based risk ranking for discrete states.

    Returns a dict mapping state_id -> rank (0=lowest volatility, n_states-1=highest).
    """
    aligned = returns.dropna().to_frame("ret").join(states.rename("state"), how="inner")
    if aligned.empty:
        return {int(s): int(s) for s in range(int(n_states))}
    vols = aligned.groupby("state")["ret"].std().replace({0.0: np.nan}).to_dict()

    missing = {int(s) for s in range(int(n_states))} - {int(k) for k in vols.keys()}
    for s in missing:
        vols[int(s)] = np.nan

    ordered = sorted(vols.items(), key=lambda kv: (np.isnan(kv[1]), kv[1]))
    ranks = {int(state): rank for rank, (state, _) in enumerate(ordered)}
    return ranks


def _color_for_level(level: str, palette: str) -> str:
    """
    palette:
      - 'returns' : warm/neutral (A)
      - 'risk'    : cool/neutral (B)
    """
    if palette == "returns":
        # Muted forest-green / indian-red tones for print-friendly contrast.
        return {"on": "#8FA88F", "off": "#C28B82"}.get(level, "#e0e0e0")
    # Muted steel-blue shades.
    return {"on": "#A9B7C7", "off": "#6E879F"}.get(level, "#e0e0e0")


def _draw_state_band_no_legend(
    ax,
    idx,
    states,
    label_map: dict[int, str],
    ymin: float,
    ymax: float,
    alpha: float,
    palette: str,
) -> None:
    """
    Draw contiguous state segments as axvspan blocks.
    No legend labels are attached here (legend is constructed manually).
    """
    if len(idx) == 0:
        return

    last_state = None
    seg_start = None

    for t, s in zip(idx, states):
        if pd.isna(s):
            if last_state is not None and seg_start is not None:
                level = label_map.get(int(last_state), "missing")
                ax.axvspan(
                    seg_start,
                    t,
                    ymin=ymin,
                    ymax=ymax,
                    facecolor=_color_for_level(level, palette),
                    edgecolor="none",
                    alpha=alpha,
                    zorder=1,
                )
            last_state = None
            seg_start = None
            continue

        s_int = int(s)
        if last_state is None:
            last_state = s_int
            seg_start = t
        elif s_int != last_state:
            level = label_map.get(int(last_state), "missing")
            ax.axvspan(
                seg_start,
                t,
                ymin=ymin,
                ymax=ymax,
                facecolor=_color_for_level(level, palette),
                edgecolor="none",
                alpha=alpha,
                zorder=1,
            )
            last_state = s_int
            seg_start = t

    if last_state is not None and seg_start is not None:
        level = label_map.get(int(last_state), "missing")
        ax.axvspan(
            seg_start,
            idx[-1],
            ymin=ymin,
            ymax=ymax,
            facecolor=_color_for_level(level, palette),
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )


def plot_representation_failure_matrix(
    price: pd.Series,
    states_a: pd.Series,
    states_b: pd.Series,
    conflict_path: Path,
    matrix_path: Path,
    start: str = "2020-02-01",
    end: str = "2020-06-30",
    title_left: str = "Representation conflict under COVID-19 stress",
    title_right: str = "Drawdown under representation-misaligned decision and validation",
) -> None:
    """
    Generate two figures:
      (1) Conflict view: price + Layer A/B state bands + conflict strip.
      (2) Failure matrix: drawdown value placed in off-diagonal cells.

    NOTE: Paper 1 usually uses only (1). (2) is more Paper 3-ish, but kept here.
    """
    logger.info(
        "plot_representation_failure_matrix: start; conflict_path=%s matrix_path=%s",
        conflict_path,
        matrix_path,
    )
    if price is None or price.empty:
        logger.warning("plot_representation_failure_matrix: empty price series; abort.")
        return
    if states_a is None or states_a.dropna().empty:
        logger.warning("plot_representation_failure_matrix: empty states_a; abort.")
        return
    if states_b is None or states_b.dropna().empty:
        logger.warning("plot_representation_failure_matrix: empty states_b; abort.")
        return

    price = price.sort_index()

    idx_start = np.datetime64(start)
    idx_end = np.datetime64(end)
    price_win = price.loc[
        (price.index.values >= idx_start) & (price.index.values <= idx_end)
    ]

    common_idx = price_win.index.intersection(states_a.dropna().index).intersection(
        states_b.dropna().index
    )
    if common_idx.empty:
        common_idx = price.index.intersection(states_a.dropna().index).intersection(
            states_b.dropna().index
        )
        price_win = price.loc[common_idx]
    if common_idx.empty:
        logger.warning(
            "plot_representation_failure_matrix: no common index between price and states; abort."
        )
        return

    logger.info(
        "plot_representation_failure_matrix: window points=%d full_common=%d",
        len(price_win),
        len(common_idx),
    )

    s_a = states_a.reindex(common_idx).astype("Int64")
    s_b = states_b.reindex(common_idx).astype("Int64")

    # --- risk-on/off mapping per representation (highest vol state = risk-off) ---
    ret = price.pct_change()
    n_states = int(max(s_a.max(skipna=True), s_b.max(skipna=True)) + 1)

    ranks_a = _compute_state_risk_ranks(ret, s_a, n_states)
    ranks_b = _compute_state_risk_ranks(ret, s_b, n_states)

    max_rank_a = max((v for v in ranks_a.values() if not np.isnan(v)), default=0.0)
    max_rank_b = max((v for v in ranks_b.values() if not np.isnan(v)), default=0.0)

    label_map_a: dict[int, str] = {}
    label_map_b: dict[int, str] = {}

    for st, r in ranks_a.items():
        if np.isnan(r):
            label_map_a[int(st)] = "missing"
        elif r == max_rank_a:
            label_map_a[int(st)] = "off"
        else:
            label_map_a[int(st)] = "on"

    for st, r in ranks_b.items():
        if np.isnan(r):
            label_map_b[int(st)] = "missing"
        elif r == max_rank_b:
            label_map_b[int(st)] = "off"
        else:
            label_map_b[int(st)] = "on"

    risk_a = s_a.map(lambda x: label_map_a.get(int(x), "missing"))
    risk_b = s_b.map(lambda x: label_map_b.get(int(x), "missing"))

    conflict_mask = (
        risk_a.ne(risk_b) & risk_a.isin(["on", "off"]) & risk_b.isin(["on", "off"])
    ).to_numpy(dtype=bool)

    logger.info(
        "plot_representation_failure_matrix: conflict points=%d",
        int(conflict_mask.sum()),
    )

    # --- drawdown inside conflict zone (stress-test scalar for matrix) ---
    extra_dd = np.nan
    idx_conflict = common_idx[conflict_mask]
    if len(idx_conflict) > 2:
        conf_rets = ret.reindex(idx_conflict).dropna()
        if len(conf_rets) > 2:
            conf_cum = (1.0 + conf_rets).cumprod()
            running_max = conf_cum.cummax()
            dd = (running_max - conf_cum) / running_max
            extra_dd = float(dd.max())

    if not np.isfinite(extra_dd):
        full_rets = price_win.pct_change().dropna()
        if len(full_rets) > 2:
            full_cum = (1.0 + full_rets).cumprod()
            running_max_full = full_cum.cummax()
            dd_full = (running_max_full - full_cum) / running_max_full
            extra_dd = float(dd_full.max())

    # =========================================================================
    # Figure 1: Conflict view, two subplots
    #   Top (3/4 height): price line + two state-band strips
    #   Bottom (1/4 height): binary conflict indicator
    # =========================================================================
    fig_conf, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.4, 4.2), dpi=300,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # --- Top panel: price ---
    ax_top.plot(price_win.index, price_win.values, color="black", linewidth=1.2, zorder=3)
    ax_top.set_title(title_left, fontsize=10)
    ax_top.set_ylabel(price.name or "Price", fontsize=9)
    ax_top.set_axisbelow(True)
    ax_top.grid(True, alpha=0.20, linewidth=0.5)

    # State bands drawn as thin horizontal strips within the top axes (axes fraction).
    # rep_a: upper strip; rep_c1: lower strip; separated so neither overlaps price line.
    _draw_state_band_no_legend(
        ax=ax_top,
        idx=common_idx,
        states=s_a.to_numpy(),
        label_map=label_map_a,
        ymin=0.88,
        ymax=0.96,
        alpha=0.55,
        palette="returns",
    )
    _draw_state_band_no_legend(
        ax=ax_top,
        idx=common_idx,
        states=s_b.to_numpy(),
        label_map=label_map_b,
        ymin=0.78,
        ymax=0.86,
        alpha=0.55,
        palette="risk",
    )

    # Compact legend for the state bands
    legend_elements = [
        Patch(facecolor="none", edgecolor="none", label="rep_a (upper):"),
        Patch(facecolor=_color_for_level("on", "returns"), edgecolor="none", label=" low-risk"),
        Patch(facecolor=_color_for_level("off", "returns"), edgecolor="none", label=" high-risk"),
        Patch(facecolor="none", edgecolor="none", label="rep_c1 (lower):"),
        Patch(facecolor=_color_for_level("on", "risk"), edgecolor="none", label=" low-risk"),
        Patch(facecolor=_color_for_level("off", "risk"), edgecolor="none", label=" high-risk"),
    ]
    ax_top.legend(handles=legend_elements, loc="upper left", fontsize=6,
                  framealpha=0.7, ncol=2, columnspacing=0.4)

    # --- Bottom panel: conflict indicator as filled step function ---
    conflict_series = conflict_mask.astype(float)
    ax_bot.fill_between(
        common_idx,
        conflict_series,
        step="mid",
        color="#C0392B",
        alpha=0.60,
        linewidth=0,
        label="Conflict",
    )
    ax_bot.set_ylim(-0.05, 1.35)
    ax_bot.set_yticks([0, 1])
    ax_bot.set_yticklabels(["agree", "conflict"], fontsize=7)
    ax_bot.set_xlabel("Date", fontsize=9)
    ax_bot.grid(True, axis="x", alpha=0.20, linewidth=0.5)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    fig_conf.subplots_adjust(hspace=0.08)
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    fig_conf.savefig(conflict_path, bbox_inches="tight", dpi=300)
    plt.close(fig_conf)
    logger.info(
        "plot_representation_failure_matrix: saved conflict figure to %s",
        conflict_path,
    )
    # Legacy panel removed: stop here to avoid duplicate overwrite.
    return

    # Restrict to common index and the specified calendar window.
    price = price.sort_index()
    idx_start = np.datetime64(start)
    idx_end = np.datetime64(end)
    price_win = price.loc[
        (price.index.values >= idx_start) & (price.index.values <= idx_end)
    ]
    common_idx = price_win.index.intersection(states_a.dropna().index).intersection(
        states_b.dropna().index
    )
    if common_idx.empty:
        # If the requested window has no overlap, fall back to full common range.
        common_idx = price.index.intersection(states_a.dropna().index).intersection(
            states_b.dropna().index
        )
        price_win = price.loc[common_idx]
    if common_idx.empty:
        return

    s_a = states_a.reindex(common_idx).astype("Int64")
    s_b = states_b.reindex(common_idx).astype("Int64")

    # Compute simple volatility-based risk ranks per state (0=lowest risk, higher=more volatile).
    ret = price.pct_change()
    n_states = int(max(s_a.max(skipna=True), s_b.max(skipna=True)) + 1)
    ranks_a = _compute_state_risk_ranks(ret, s_a, n_states)
    ranks_b = _compute_state_risk_ranks(ret, s_b, n_states)

    # Compress to two qualitative levels per layer: risk-on vs risk-off.
    # For each representation, the highest-volatility state is treated as risk-off;
    # all other states are treated as risk-on.
    max_rank_a = max((v for v in ranks_a.values() if not np.isnan(v)), default=0.0)
    max_rank_b = max((v for v in ranks_b.values() if not np.isnan(v)), default=0.0)
    label_map_a: dict[int, str] = {}
    label_map_b: dict[int, str] = {}
    for s, r in ranks_a.items():
        if np.isnan(r):
            label_map_a[int(s)] = "missing"
        elif r == max_rank_a:
            label_map_a[int(s)] = "off"
        else:
            label_map_a[int(s)] = "on"
    for s, r in ranks_b.items():
        if np.isnan(r):
            label_map_b[int(s)] = "missing"
        elif r == max_rank_b:
            label_map_b[int(s)] = "off"
        else:
            label_map_b[int(s)] = "on"

    risk_a = s_a.map(lambda x: label_map_a.get(int(x), "missing"))
    risk_b = s_b.map(lambda x: label_map_b.get(int(x), "missing"))
    conflict_mask = (
        risk_a.ne(risk_b) & risk_a.isin(["on", "off"]) & risk_b.isin(["on", "off"])
    ).to_numpy(dtype=bool)

    # Compute drawdown in the conflict zone when the model (A) is effectively "long"
    # while the conservative representation (B) flags high risk.
    idx_conflict = common_idx[conflict_mask]
    extra_dd = np.nan
    if len(idx_conflict) > 0:
        conf_rets = ret.reindex(idx_conflict).dropna()
        if not conf_rets.empty:
            conf_cum = (1.0 + conf_rets).cumprod()
            running_max = conf_cum.cummax()
            dd = (running_max - conf_cum) / running_max
            extra_dd = float(dd.max())

    # Fallback: if conflict-only drawdown is undefined (e.g. too few points),
    # compute drawdown over the full window to ensure a visible stress-test value.
    if not np.isfinite(extra_dd):
        full_rets = price_win.pct_change().dropna()
        if not full_rets.empty:
            full_cum = (1.0 + full_rets).cumprod()
            running_max_full = full_cum.cummax()
            dd_full = (running_max_full - full_cum) / running_max_full
            extra_dd = float(dd_full.max())

    # -------------------------
    # Figure 1: Conflict view.
    # -------------------------
    fig_conf, ax_left = plt.subplots(figsize=(6.0, 3.5), dpi=300)

    ax_left.plot(price_win.index, price_win.values, color="0.4", linewidth=1.2)
    ax_left.set_title(title_left, fontsize=11)
    ax_left.set_xlabel("Date")
    ax_left.set_ylabel(price.name or "Price")

    # Helper to draw horizontal state bands for a given representation.
    def _draw_state_band(
        idx,
        states,
        label_map,
        ymin,
        ymax,
        label_prefix: str,
        alpha: float,
        palette: str,
    ) -> None:
        if len(idx) == 0:
            return
        last_state = None
        seg_start = None
        legend_done = set()
        for t, s in zip(idx, states):
            if pd.isna(s):
                if last_state is not None and seg_start is not None:
                    level = label_map.get(last_state, "missing")
                    if palette == "returns":
                        if level == "on":
                            color = "#c8e6c9"  # green: risk-on
                        elif level == "off":
                            color = "#ffcdd2"  # red: risk-off
                        else:
                            color = "#e0e0e0"
                    else:  # risk-based layer uses blue palette
                        if level == "on":
                            color = "#bbdefb"  # light blue: risk-on
                        elif level == "off":
                            color = "#64b5f6"  # deeper blue: risk-off
                        else:
                            color = "#e0e0e0"
                    band_label = (
                        f"{label_prefix}: risk-{level}"
                        if level in {"on", "off"}
                        else None
                    )
                    show_label = band_label not in legend_done
                    ax_left.axvspan(
                        seg_start,
                        t,
                        ymin=ymin,
                        ymax=ymax,
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.7,
                        label=band_label if show_label else None,
                    )
                    legend_done.add(band_label)
                    last_state = None
                    seg_start = None
                continue
            s_int = int(s)
            if last_state is None:
                last_state = s_int
                seg_start = t
            elif s_int != last_state:
                level = label_map.get(last_state, "missing")
                if palette == "returns":
                    if level == "on":
                        color = "#c8e6c9"
                    elif level == "off":
                        color = "#ffcdd2"
                    else:
                        color = "#e0e0e0"
                else:
                    if level == "on":
                        color = "#bbdefb"
                    elif level == "off":
                        color = "#64b5f6"
                    else:
                        color = "#e0e0e0"
                band_label = (
                    f"{label_prefix}: risk-{level}" if level in {"on", "off"} else None
                )
                show_label = band_label not in legend_done
                ax_left.axvspan(
                    seg_start,
                    t,
                    ymin=ymin,
                    ymax=ymax,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.7,
                    label=band_label if show_label else None,
                )
                legend_done.add(band_label)
                last_state = s_int
                seg_start = t
        # Close final segment
        if last_state is not None and seg_start is not None:
            level = label_map.get(last_state, "missing")
            if palette == "returns":
                if level == "on":
                    color = "#c8e6c9"
                elif level == "off":
                    color = "#ffcdd2"
                else:
                    color = "#e0e0e0"
            else:
                if level == "on":
                    color = "#bbdefb"
                elif level == "off":
                    color = "#64b5f6"
                else:
                    color = "#e0e0e0"
            band_label = (
                f"{label_prefix}: risk-{level}" if level in {"on", "off"} else None
            )
            show_label = band_label not in legend_done
            ax_left.axvspan(
                seg_start,
                idx[-1],
                ymin=ymin,
                ymax=ymax,
                facecolor=color,
                edgecolor="none",
                alpha=0.7,
                label=band_label if show_label else None,
            )
            legend_done.add(band_label)

    # Draw Layer A and Layer B state bands.
    _draw_state_band(
        common_idx,
        s_a.to_numpy(),
        label_map_a,
        ymin=0.70,
        ymax=0.82,
        label_prefix="rep_a",
        alpha=0.65,
        palette="returns",
    )
    _draw_state_band(
        common_idx,
        s_b.to_numpy(),
        label_map_b,
        ymin=0.52,
        ymax=0.64,
        label_prefix="rep_c1",
        alpha=0.40,
        palette="risk",
    )

    # Conflict indicator band (0/1) placed below Layer B.
    if conflict_mask.any():
        # Base strip
        ax_left.axvspan(
            common_idx[0],
            common_idx[-1],
            ymin=0.34,
            ymax=0.46,
            facecolor="#ffffff",
            edgecolor="none",
            alpha=0.95,
        )
        # Shade contiguous conflict segments so they are visually wide enough.
        in_zone = False
        seg_start = None
        first_label = True
        for t, is_conflict in zip(common_idx, conflict_mask):
            if is_conflict and not in_zone:
                in_zone = True
                seg_start = t
            elif not is_conflict and in_zone:
                ax_left.axvspan(
                    seg_start,
                    t,
                    ymin=0.34,
                    ymax=0.46,
                    facecolor="#ffec99",
                    edgecolor="none",
                    alpha=0.9,
                    label="Conflict zone" if first_label else None,
                )
                first_label = False
                in_zone = False
                seg_start = None
        if in_zone and seg_start is not None:
            ax_left.axvspan(
                seg_start,
                common_idx[-1],
                ymin=0.34,
                ymax=0.46,
                facecolor="#ffec99",
                edgecolor="none",
                alpha=0.9,
                label="Conflict zone" if first_label else None,
            )

    # Avoid duplicate legend entries and place legend in the lower-right corner.
    handles, labels = ax_left.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        # Enforce intuitive ordering: risk-on then risk-off for each layer, then conflict.
        desired = [
            "rep_a: risk-on",
            "rep_a: risk-off",
            "rep_c1: risk-on",
            "rep_c1: risk-off",
            "Conflict zone",
        ]
        ordered_labels = [lab for lab in desired if lab in by_label]
        ordered_handles = [by_label[lab] for lab in ordered_labels]
        ax_left.legend(
            ordered_handles,
            ordered_labels,
            fontsize=7,
            loc="lower right",
            framealpha=0.7,
        )

    ax_left.grid(True, alpha=0.3, linewidth=0.5)

    fig_conf.tight_layout()
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    fig_conf.savefig(conflict_path, bbox_inches="tight", dpi=300)
    plt.close(fig_conf)


