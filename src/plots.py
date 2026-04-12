from __future__ import annotations

from pathlib import Path
import logging

import matplotlib
import pandas as pd

# Force a non-interactive backend for reproducible file output (headless-safe).
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
    # Figure 1: Conflict view (Paper 1)
    # =========================================================================
    fig_conf, ax = plt.subplots(figsize=(6.4, 3.6), dpi=300)

    ax.plot(price_win.index, price_win.values, color="black", linewidth=1.2, zorder=3)
    ax.set_title(title_left, fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel(price.name or "Price")
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    # Bands: keep them thin, separated, and behind price
    _draw_state_band_no_legend(
        ax=ax,
        idx=common_idx,
        states=s_a.to_numpy(),
        label_map=label_map_a,
        ymin=0.72,
        ymax=0.82,
        alpha=0.40,
        palette="returns",
    )
    _draw_state_band_no_legend(
        ax=ax,
        idx=common_idx,
        states=s_b.to_numpy(),
        label_map=label_map_b,
        ymin=0.56,
        ymax=0.66,
        alpha=0.35,
        palette="risk",
    )

    # Conflict strip (independent visual language)
    # Base white strip
    ax.axvspan(
        common_idx[0],
        common_idx[-1],
        ymin=0.46,
        ymax=0.52,
        facecolor="white",
        edgecolor="none",
        alpha=0.95,
        zorder=2,
    )

    if conflict_mask.any():
        in_zone = False
        seg_start = None
        for t, is_conflict in zip(common_idx, conflict_mask):
            if is_conflict and not in_zone:
                in_zone = True
                seg_start = t
            elif (not is_conflict) and in_zone:
                ax.axvspan(
                    seg_start,
                    t,
                    ymin=0.46,
                    ymax=0.52,
                    facecolor="#ffec99",
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.95,
                    zorder=4,
                )
                in_zone = False
                seg_start = None
        if in_zone and seg_start is not None:
            ax.axvspan(
                seg_start,
                common_idx[-1],
                ymin=0.46,
                ymax=0.52,
                facecolor="#ffec99",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.95,
                zorder=4,
            )

    # --- Grouped legend (3 blocks) ---
    legend_elements = [
        Patch(facecolor="none", edgecolor="none", label="rep_a"),
        Patch(
            facecolor=_color_for_level("on", "returns"),
            edgecolor="none",
            label=" low-risk",
        ),
        Patch(
            facecolor=_color_for_level("off", "returns"),
            edgecolor="none",
            label=" high-risk",
        ),
        Patch(facecolor="none", edgecolor="none", label="rep_c1"),
        Patch(
            facecolor=_color_for_level("on", "risk"),
            edgecolor="none",
            label=" low-risk",
        ),
        Patch(
            facecolor=_color_for_level("off", "risk"),
            edgecolor="none",
            label=" high-risk",
        ),
        Patch(facecolor="#ffec99", edgecolor="black", label="Conflict"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=6, framealpha=0.6)

    fig_conf.tight_layout()
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    fig_conf.savefig(conflict_path, bbox_inches="tight", dpi=300)
    plt.close(fig_conf)
    logger.info(
        "plot_representation_failure_matrix: saved conflict figure to %s",
        conflict_path,
    )

    # (Legacy failure-matrix panel for Paper 3 has been removed. The ``matrix_path``
    # argument is retained for backwards-compatibility but no longer used here.)
    """
    Generate two separate figures illustrating representation conflict and its risk impact.

    (1) A conflict-only figure with price, layered regimes (Layer A/B), and a conflict strip.
    (2) A decision--validation failure matrix summarising drawdown in the conflict zones.
    """
    if price is None or price.empty:
        return
    if states_a is None or states_a.dropna().empty:
        return
    if states_b is None or states_b.dropna().empty:
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

    # Draw Layer A (returns-based representation) and Layer B (risk-based representation).
    _draw_state_band(
        common_idx,
        s_a.to_numpy(),
        label_map_a,
        ymin=0.70,
        ymax=0.82,
        label_prefix="Layer A (returns)",
        alpha=0.65,
        palette="returns",
    )
    _draw_state_band(
        common_idx,
        s_b.to_numpy(),
        label_map_b,
        ymin=0.52,
        ymax=0.64,
        label_prefix="Layer B (risk)",
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
            "Layer A (returns): risk-on",
            "Layer A (returns): risk-off",
            "Layer B (risk): risk-on",
            "Layer B (risk): risk-off",
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

    # Mean line (primary emphasis) — use solid to avoid "chunky" dashed artifacts
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
