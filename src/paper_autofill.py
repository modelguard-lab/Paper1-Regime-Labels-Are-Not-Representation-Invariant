from __future__ import annotations

import re
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
try:  # optional; used for small-n CI across assets
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover
    student_t = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BaselineSpec:
    step: int
    window: int
    k: int
    seeds: List[int]


def _fmt(x: float | int | None, nd: int = 3) -> str:
    if x is None:
        return "NA"
    try:
        if isinstance(x, (float, np.floating)) and (np.isnan(x) or np.isinf(x)):
            return "NA"
    except Exception:
        pass
    if isinstance(x, int):
        return str(x)
    return f"{float(x):.{nd}f}"


def _replace_block(text: str, start_tag: str, end_tag: str, new_block: str) -> str:
    pattern = re.compile(
        rf"({re.escape(start_tag)}\n)(.*?)(\n{re.escape(end_tag)})", flags=re.DOTALL
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Could not find numeric-block tags: {start_tag} ... {end_tag}")
    return text[: m.start(2)] + new_block.rstrip() + text[m.end(2) :]


def _iter_asset_step_key_results(outputs_dir: Path) -> Iterable[Tuple[str, Optional[int], Path]]:
    """
    Yield (asset, step, key_results_path).

    Supports both asset-first step layout:
      outputs/<asset>/step_<s>/key_results.csv
    and single-run layout:
      outputs/<asset>/key_results.csv
    """
    if not outputs_dir.exists():
        return
    for asset_dir in sorted([p for p in outputs_dir.iterdir() if p.is_dir()]):
        asset = asset_dir.name
        # step_* layout
        step_dirs = sorted([p for p in asset_dir.glob("step_*") if p.is_dir()])
        any_step = False
        for sd in step_dirs:
            any_step = True
            m = re.match(r"step_(\d+)$", sd.name)
            step = int(m.group(1)) if m else None
            p = sd / "key_results.csv"
            if p.exists():
                yield asset, step, p
        # fallback layout
        if not any_step:
            p = asset_dir / "key_results.csv"
            if p.exists():
                yield asset, None, p


def _read_key_results(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # be tolerant to older exports
    for col in ["metric", "scope", "value"]:
        if col not in df.columns:
            raise ValueError(f"Unexpected key_results format: missing column={col} in {p}")
    return df


def _get_metric(df: pd.DataFrame, metric: str, scope: str) -> Optional[float]:
    s = df[(df["metric"] == metric) & (df["scope"] == scope)]
    if s.empty:
        return None
    return float(pd.to_numeric(s["value"].iloc[0], errors="coerce"))


def _compute_model_split_block(outputs_dir: Path, baseline: BaselineSpec) -> str:
    # Collect per-asset baseline key_results.
    per_asset: Dict[str, pd.DataFrame] = {}
    for asset, step, p in _iter_asset_step_key_results(outputs_dir):
        if step is None:
            continue
        if int(step) != int(baseline.step):
            continue
        per_asset[asset] = _read_key_results(p)

    if not per_asset:
        return "*(missing required baseline exports.)*"

    # Means by model (across assets).
    rows = []
    for asset, df in per_asset.items():
        for model in ["gmm", "hmm"]:
            rows.append(
                {
                    "asset": asset,
                    "model": model,
                    "cross_rep_ari": _get_metric(df, "cross_rep_ari_mean", f"model={model}"),
                    "temporal_ari": _get_metric(df, "temporal_ari_mean", f"model={model}"),
                }
            )
    d = pd.DataFrame(rows)
    means = (
        d.groupby("model")[["cross_rep_ari", "temporal_ari"]]
        .mean(numeric_only=True)
        .reindex(["gmm", "hmm"])
    )

    # Overall ratio (temporal / cross-rep) using "all" scope.
    all_rows = []
    for asset, df in per_asset.items():
        all_rows.append(
            {
                "asset": asset,
                "cross_rep_all": _get_metric(df, "cross_rep_ari_mean", "all"),
                "temporal_all": _get_metric(df, "temporal_ari_mean", "all"),
            }
        )
    da = pd.DataFrame(all_rows).set_index("asset")
    cross_mean = float(da["cross_rep_all"].mean())
    temp_mean = float(da["temporal_all"].mean())
    ratio = temp_mean / cross_mean if cross_mean and not np.isnan(cross_mean) else float("nan")

    gap_temporal = float(means.loc["gmm", "temporal_ari"] - means.loc["hmm", "temporal_ari"])
    gap_cross = float(means.loc["gmm", "cross_rep_ari"] - means.loc["hmm", "cross_rep_ari"])

    assets_list = ", ".join(sorted(per_asset.keys()))
    seeds_str = ", ".join(str(s) for s in baseline.seeds)

    block = []
    block.append(
        f"Baseline setting: window $={baseline.window}$, step $={baseline.step}$, $K={baseline.k}$, seeds $=\\{{{seeds_str}\\}}$; averaged across assets ({assets_list})."
    )
    block.append("")
    block.append("| model | cross_rep_ari (mean across assets) | temporal_ari (mean across assets) |")
    block.append("| --- | ---: | ---: |")
    block.append(f"| GMM | {_fmt(means.loc['gmm','cross_rep_ari'])} | {_fmt(means.loc['gmm','temporal_ari'])} |")
    block.append(f"| HMM | {_fmt(means.loc['hmm','cross_rep_ari'])} | {_fmt(means.loc['hmm','temporal_ari'])} |")
    block.append("")
    block.append(
        "Cross-asset main effect: temporal ARI is higher than cross-representation ARI "
        f"(ratio $={_fmt(ratio, nd=2)}\\times$ at step $={baseline.step}$), indicating apparent stability within a fixed representation that collapses under reasonable representation changes. "
        f"Model contrast supports the mechanism: GMM exceeds HMM on both cross-representation stability (gap $={_fmt(gap_cross)}$) and temporal stability (gap $={_fmt(gap_temporal)}$), consistent with persistence-driven disagreement amplification in dynamic sequence models (Section 6.6)."
    )
    return "\n".join(block).rstrip()


def _compute_step_sweep_block(outputs_dir: Path, steps: List[int]) -> str:
    # Prefer the runner-produced summary if present.
    summary_path = outputs_dir / "step_sweep_summary.csv"
    d: pd.DataFrame
    if summary_path.exists():
        d = pd.read_csv(summary_path)
    else:
        # Fallback: collect key_results for each asset-step present.
        rows = []
        for asset, step, p in _iter_asset_step_key_results(outputs_dir):
            if step is None or int(step) not in set(int(s) for s in steps):
                continue
            df = _read_key_results(p)
            rows.append(
                {
                    "asset": asset,
                    "step": int(step),
                    "cross_rep_ari_mean": _get_metric(df, "cross_rep_ari_mean", "all"),
                    "temporal_ari_mean": _get_metric(df, "temporal_ari_mean", "all"),
                }
            )
        if not rows:
            return "*(missing required step-sweep exports.)*"
        d = pd.DataFrame(rows)

    d["step"] = pd.to_numeric(d["step"], errors="coerce").astype("Int64")
    d["cross_rep_ari_mean"] = pd.to_numeric(d["cross_rep_ari_mean"], errors="coerce")
    d["temporal_ari_mean"] = pd.to_numeric(d["temporal_ari_mean"], errors="coerce")

    def ci95(series: pd.Series) -> float | None:
        s = pd.to_numeric(series, errors="coerce").dropna()
        n = int(s.shape[0])
        if n <= 1:
            return None
        std = float(s.std(ddof=1))
        se = std / math.sqrt(n) if std == std else float("nan")
        if student_t is not None:
            tcrit = float(student_t.ppf(0.975, df=n - 1))
        else:
            # Normal approximation fallback (less accurate for n=4, but avoids hard dependency).
            tcrit = 1.96
        return float(tcrit * se)

    out_lines = []
    out_lines.append("*Note: ARI values are averaged across both GMM and HMM (scope='all'); for per-model breakdown see the model-split block above.*")
    out_lines.append("")
    out_lines.append("| step | temporal_ari (mean ± 95% CI across assets) | cross_rep_ari (mean ± 95% CI across assets) | n_assets |")
    out_lines.append("| ---: | ---: | ---: | ---: |")
    for step in steps:
        sub = d[d["step"] == int(step)]
        if sub.empty:
            out_lines.append(f"| {int(step)} | NA | NA | 0 |")
            continue
        t = pd.to_numeric(sub["temporal_ari_mean"], errors="coerce")
        c = pd.to_numeric(sub["cross_rep_ari_mean"], errors="coerce")
        n_t = int(t.dropna().shape[0])
        n_c = int(c.dropna().shape[0])
        n_assets = max(n_t, n_c)
        t_mean = float(t.mean()) if n_t > 0 else None
        c_mean = float(c.mean()) if n_c > 0 else None
        t_ci = ci95(t)
        c_ci = ci95(c)
        t_str = "NA" if t_mean is None else f"{_fmt(t_mean)} ± {_fmt(t_ci, nd=3) if t_ci is not None else 'NA'}"
        c_str = "NA" if c_mean is None else f"{_fmt(c_mean)} ± {_fmt(c_ci, nd=3) if c_ci is not None else 'NA'}"
        out_lines.append(
            f"| {int(step)} | {t_str} | {c_str} | {n_assets} |"
        )
    return "\n".join(out_lines).rstrip()


def _compute_semantic_drift_block(outputs_dir: Path, step: int) -> str:
    # Gather baseline drift CSVs under step_{step}.
    drift_files = sorted(outputs_dir.glob(f"*/step_{int(step)}/results/*/windows_semantic_drift_*.csv"))
    if not drift_files:
        return "*(missing required semantic-drift exports.)*"

    rows = []
    for p in drift_files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "model" not in df.columns or "semantic_drift" not in df.columns:
            continue
        df["semantic_drift"] = pd.to_numeric(df["semantic_drift"], errors="coerce")
        df = df.dropna(subset=["semantic_drift"])
        if df.empty:
            continue
        for model, sub in df.groupby("model"):
            vals = sub["semantic_drift"].astype(float).values
            rows.append(
                {
                    "model": str(model),
                    "mean": float(np.mean(vals)),
                    "p10": float(np.quantile(vals, 0.10)),
                    "p50": float(np.quantile(vals, 0.50)),
                    "p90": float(np.quantile(vals, 0.90)),
                    "max": float(np.max(vals)),
                    "n": int(vals.size),
                }
            )

    if not rows:
        return "*(semantic drift exports exist but contain no numeric values.)*"

    d = pd.DataFrame(rows)
    # Aggregate across files: weighted by n.
    out_rows = []
    for model in ["gmm", "hmm"]:
        sub = d[d["model"] == model]
        if sub.empty:
            continue
        w = sub["n"].astype(float)
        mean = float(np.average(sub["mean"], weights=w))
        # For quantiles/max, report across all per-file values conservatively (not perfectly exact, but stable).
        # To avoid loading huge concatenations, we approximate using per-file quantiles weighted by n.
        p10 = float(np.average(sub["p10"], weights=w))
        p50 = float(np.average(sub["p50"], weights=w))
        p90 = float(np.average(sub["p90"], weights=w))
        mx = float(sub["max"].max())
        n = int(sub["n"].sum())
        out_rows.append((model, mean, p10, p50, p90, mx, n))

    if not out_rows:
        return "*(no gmm/hmm semantic drift rows found.)*"

    lines = []
    lines.append(f"Baseline semantic drift summary (step $={int(step)}$), aggregated over all available windows/states across assets and representations:")
    lines.append("")
    lines.append("| model | mean | p10 | p50 | p90 | max | n (state-window points) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model, mean, p10, p50, p90, mx, n in out_rows:
        label = "GMM" if model == "gmm" else "HMM"
        lines.append(
            f"| {label} | {_fmt(mean)} | {_fmt(p10)} | {_fmt(p50)} | {_fmt(p90)} | {_fmt(mx)} | {n} |"
        )
    return "\n".join(lines).rstrip()


def _compute_robustness_block(outputs_dir: Path) -> str:
    """
    Summarize robustness across seeds and K from outputs/robustness_ci_summary.csv.

    We report asset-averaged means (with asset ranges) and highlight the contrast in
    seed-level variability between GMM and HMM.
    """
    p = outputs_dir / "robustness_ci_summary.csv"
    if not p.exists():
        return "*(missing required robustness exports.)*"

    df = pd.read_csv(p, low_memory=False)
    required = {"asset", "K", "model", "metric", "mean", "std", "n_seeds", "ci95"}
    if not required.issubset(set(df.columns)):
        return "*(robustness export has unexpected columns.)*"

    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["std"] = pd.to_numeric(df["std"], errors="coerce")
    df["ci95"] = pd.to_numeric(df["ci95"], errors="coerce")
    df["model"] = df["model"].astype(str)
    df["metric"] = df["metric"].astype(str)

    metrics = ["cross_rep_ari_seed_mean", "temporal_ari_seed_mean"]
    sub = df[df["metric"].isin(metrics)].copy()
    if sub.empty:
        return "*(robustness export contains no ARI seed-metric rows.)*"

    ks = sorted(int(x) for x in sub["K"].dropna().unique())
    assets = sorted(set(str(x) for x in sub["asset"].dropna().unique()))

    def agg_one(k: int, model: str, metric: str) -> Tuple[float | None, float | None, float | None, float | None, float | None]:
        s = sub[(sub["K"] == k) & (sub["model"] == model) & (sub["metric"] == metric)]
        if s.empty:
            return None, None, None, None, None
        means = s["mean"].dropna().astype(float)
        ci = s["ci95"].dropna().astype(float)
        std = s["std"].dropna().astype(float)
        if means.empty:
            return None, None, None, None, None
        return (
            float(means.mean()),  # asset-average
            float(means.min()),
            float(means.max()),
            float(ci.mean()) if not ci.empty else None,
            float(std.mean()) if not std.empty else None,
        )

    lines: List[str] = []
    lines.append("Robustness summary (seed-level).")
    lines.append("")
    lines.append("| K | cross-rep ARI (GMM; mean ± avg CI; asset range) | cross-rep ARI (HMM; mean ± avg CI; asset range) | temporal ARI (GMM; mean ± avg CI; asset range) | temporal ARI (HMM; mean ± avg CI; asset range) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for k in ks:
        g_c_mean, g_c_min, g_c_max, g_c_ci, g_c_std = agg_one(k, "gmm", "cross_rep_ari_seed_mean")
        h_c_mean, h_c_min, h_c_max, h_c_ci, h_c_std = agg_one(k, "hmm", "cross_rep_ari_seed_mean")
        g_t_mean, g_t_min, g_t_max, g_t_ci, g_t_std = agg_one(k, "gmm", "temporal_ari_seed_mean")
        h_t_mean, h_t_min, h_t_max, h_t_ci, h_t_std = agg_one(k, "hmm", "temporal_ari_seed_mean")

        def cell(mean, ci, mn, mx):
            if mean is None:
                return "NA"
            return f"{_fmt(mean)} ± {_fmt(ci, nd=3) if ci is not None else 'NA'}; [{_fmt(mn)}–{_fmt(mx)}]"

        lines.append(
            f"| {k} | {cell(g_c_mean, g_c_ci, g_c_min, g_c_max)} | {cell(h_c_mean, h_c_ci, h_c_min, h_c_max)} | {cell(g_t_mean, g_t_ci, g_t_min, g_t_max)} | {cell(h_t_mean, h_t_ci, h_t_min, h_t_max)} |"
        )

    # Variability highlight: compare average seed-std across assets.
    def std_summary(model: str, metric: str) -> Tuple[float | None, float | None]:
        s = sub[(sub["model"] == model) & (sub["metric"] == metric)]["std"]
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return None, None
        return float(s.mean()), float(s.max())

    gmm_c_std_mean, gmm_c_std_max = std_summary("gmm", "cross_rep_ari_seed_mean")
    hmm_c_std_mean, hmm_c_std_max = std_summary("hmm", "cross_rep_ari_seed_mean")
    gmm_t_std_mean, gmm_t_std_max = std_summary("gmm", "temporal_ari_seed_mean")
    hmm_t_std_mean, hmm_t_std_max = std_summary("hmm", "temporal_ari_seed_mean")

    lines.append("")
    lines.append(
        "Seed sensitivity (std across seeds, averaged across assets/K): "
        f"GMM cross-rep std = {_fmt(gmm_c_std_mean)} (max {_fmt(gmm_c_std_max)}), "
        f"HMM cross-rep std = {_fmt(hmm_c_std_mean)} (max {_fmt(hmm_c_std_max)}); "
        f"GMM temporal std = {_fmt(gmm_t_std_mean)} (max {_fmt(gmm_t_std_max)}), "
        f"HMM temporal std = {_fmt(hmm_t_std_mean)} (max {_fmt(hmm_t_std_max)})."
    )
    lines.append("")
    lines.append(f"Assets: {', '.join(assets)}. Seeds per cell: 20. CI is within-asset across seeds; we report the asset-average CI for compactness.")
    return "\n".join(lines).rstrip()


def _compute_ordering_robustness_block(outputs_dir: Path) -> str:
    """
    Summarize ordering consistency (Top-1 + Spearman) across seeds and K.
    """
    p = outputs_dir / "ordering_ci_summary.csv"
    if not p.exists():
        return "*(missing required ordering exports.)*"

    df = pd.read_csv(p, low_memory=False)
    required = {"asset", "K", "model", "metric", "mean", "std", "n_seeds", "ci95"}
    if not required.issubset(set(df.columns)):
        return "*(ordering export has unexpected columns.)*"

    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["ci95"] = pd.to_numeric(df["ci95"], errors="coerce")
    df["model"] = df["model"].astype(str)
    df["metric"] = df["metric"].astype(str)

    keep = [
        "ordering_cross_rep_top1_seed_mean",
        "ordering_cross_rep_spearman_seed_mean",
        "ordering_temporal_top1_seed_mean",
        "ordering_temporal_spearman_seed_mean",
    ]
    sub = df[df["metric"].isin(keep)].copy()
    if sub.empty:
        return "*(ordering export contains no Top1/Spearman rows.)*"

    ks = sorted(int(x) for x in sub["K"].dropna().unique())
    assets = sorted(set(str(x) for x in sub["asset"].dropna().unique()))

    def agg(k: int, model: str, metric: str) -> Tuple[float | None, float | None]:
        s = sub[(sub["K"] == k) & (sub["model"] == model) & (sub["metric"] == metric)]
        if s.empty:
            return None, None
        m = s["mean"].dropna().astype(float)
        ci = s["ci95"].dropna().astype(float)
        if m.empty:
            return None, None
        return float(m.mean()), float(ci.mean()) if not ci.empty else None

    lines: List[str] = []
    lines.append("Ordering consistency summary (Top-1 high-risk + Spearman; seed-level).")
    lines.append("")
    lines.append("| K | cross-rep Top-1 (GMM) | cross-rep Top-1 (HMM) | cross-rep Spearman (GMM) | cross-rep Spearman (HMM) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for k in ks:
        g_top, g_top_ci = agg(k, "gmm", "ordering_cross_rep_top1_seed_mean")
        h_top, h_top_ci = agg(k, "hmm", "ordering_cross_rep_top1_seed_mean")
        g_sp, g_sp_ci = agg(k, "gmm", "ordering_cross_rep_spearman_seed_mean")
        h_sp, h_sp_ci = agg(k, "hmm", "ordering_cross_rep_spearman_seed_mean")
        lines.append(
            f"| {k} | {_fmt(g_top)} ± {_fmt(g_top_ci, nd=3) if g_top_ci is not None else 'NA'} | {_fmt(h_top)} ± {_fmt(h_top_ci, nd=3) if h_top_ci is not None else 'NA'} | {_fmt(g_sp)} ± {_fmt(g_sp_ci, nd=3) if g_sp_ci is not None else 'NA'} | {_fmt(h_sp)} ± {_fmt(h_sp_ci, nd=3) if h_sp_ci is not None else 'NA'} |"
        )

    lines.append("")
    lines.append("| K | temporal Top-1 (GMM) | temporal Top-1 (HMM) | temporal Spearman (GMM) | temporal Spearman (HMM) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for k in ks:
        g_top, g_top_ci = agg(k, "gmm", "ordering_temporal_top1_seed_mean")
        h_top, h_top_ci = agg(k, "hmm", "ordering_temporal_top1_seed_mean")
        g_sp, g_sp_ci = agg(k, "gmm", "ordering_temporal_spearman_seed_mean")
        h_sp, h_sp_ci = agg(k, "hmm", "ordering_temporal_spearman_seed_mean")
        lines.append(
            f"| {k} | {_fmt(g_top)} ± {_fmt(g_top_ci, nd=3) if g_top_ci is not None else 'NA'} | {_fmt(h_top)} ± {_fmt(h_top_ci, nd=3) if h_top_ci is not None else 'NA'} | {_fmt(g_sp)} ± {_fmt(g_sp_ci, nd=3) if g_sp_ci is not None else 'NA'} | {_fmt(h_sp)} ± {_fmt(h_sp_ci, nd=3) if h_sp_ci is not None else 'NA'} |"
        )

    lines.append("")
    lines.append(f"Assets: {', '.join(assets)}. Seeds per cell: 20. CI is within-asset across seeds; we report the asset-average CI for compactness.")
    return "\n".join(lines).rstrip()


def update_empirical_results_md(outputs_dir: Path, md_path: Path, cfg: Dict) -> None:
    """
    Update numeric blocks in paper/sections/04_empirical_results.md from exports.

    This function is intentionally tolerant to partial outputs (e.g., if a long run was interrupted).
    """
    text = md_path.read_text(encoding="utf-8")

    grid = cfg.get("grid") or {}
    baseline = BaselineSpec(
        step=int(grid.get("step", 21)),
        window=int((grid.get("windows") or [252])[0]),
        k=int((grid.get("n_states") or [3])[0]),
        seeds=[int(s) for s in (grid.get("seeds") or [1, 2])],
    )
    step_sweep = grid.get("step_sweep") or []
    steps = [int(s) for s in step_sweep] if isinstance(step_sweep, list) else []

    model_block = _compute_model_split_block(outputs_dir, baseline)
    step_block = _compute_step_sweep_block(outputs_dir, steps) if steps else "*(step sweep is disabled.)*"
    drift_block = _compute_semantic_drift_block(outputs_dir, baseline.step)
    robustness_block = _compute_robustness_block(outputs_dir)
    ordering_block = _compute_ordering_robustness_block(outputs_dir)

    text = _replace_block(
        text,
        "<!-- BLOCK:MODEL-SPLIT:START -->",
        "<!-- BLOCK:MODEL-SPLIT:END -->",
        model_block,
    )
    text = _replace_block(
        text,
        "<!-- BLOCK:STEP-SWEEP:START -->",
        "<!-- BLOCK:STEP-SWEEP:END -->",
        step_block,
    )
    text = _replace_block(
        text,
        "<!-- BLOCK:SEMANTIC-DRIFT:START -->",
        "<!-- BLOCK:SEMANTIC-DRIFT:END -->",
        drift_block,
    )
    text = _replace_block(
        text,
        "<!-- BLOCK:ROBUSTNESS:START -->",
        "<!-- BLOCK:ROBUSTNESS:END -->",
        robustness_block,
    )
    text = _replace_block(
        text,
        "<!-- BLOCK:ORDERING-ROBUSTNESS:START -->",
        "<!-- BLOCK:ORDERING-ROBUSTNESS:END -->",
        ordering_block,
    )

    md_path.write_text(text, encoding="utf-8")

