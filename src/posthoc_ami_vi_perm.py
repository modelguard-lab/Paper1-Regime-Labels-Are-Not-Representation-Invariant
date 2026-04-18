"""
Post-hoc computation of AMI, VI, and permutation p-values from existing outputs.

Reads windows_states_hard_*.csv files already on disk, recomputes cross-rep and
temporal stability (temporal uses disjoint-window segments to avoid overlap inflation)
with the new metrics (AMI, VI), runs aggregate permutation tests,
and appends the results to each asset/step key_results.csv.

Run from the project root:
    python src/posthoc_ami_vi_perm.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
)
from scipy.stats import entropy as _scipy_entropy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
REPS = ["rep_a", "rep_a_unscaled", "rep_b", "rep_c1", "rep_c2", "rep_c3", "rep_d"]
MODELS = ["gmm", "hmm"]
K_DEFAULT = 3
N_PERM = 99  # min p = 1/100 = 0.01 < 0.05; sufficient for paper's p<0.05 claim
PERM_SEED = 42
PERM_MAX_PAIRS = 2000  # subsample for speed; all pairs have p << 0.05 trivially


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def variation_of_information(a: np.ndarray, b: np.ndarray) -> float:
    """Variation of Information = H(X|Y) + H(Y|X), in nats."""
    if len(a) == 0:
        return float("nan")
    mi = float(mutual_info_score(a, b))
    _, ca = np.unique(a, return_counts=True)
    _, cb = np.unique(b, return_counts=True)
    h_a = float(_scipy_entropy(ca))
    h_b = float(_scipy_entropy(cb))
    return h_a + h_b - 2.0 * mi


def align_and_ari(a: np.ndarray, b: np.ndarray) -> float:
    return float(adjusted_rand_score(a, b))


def permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
) -> float:
    """
    One-sided permutation p-value: P(ARI_perm >= ARI_obs) under H0 of independence.

    Uses the observed ARI as test statistic and permutes b.
    """
    rng = np.random.default_rng(seed)
    observed = float(adjusted_rand_score(a, b))
    count = 0
    for _ in range(n_perm):
        b_perm = rng.permutation(b)
        if float(adjusted_rand_score(a, b_perm)) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hard_states(
    asset_dir: Path, step: Optional[int], rep: str, model: str
) -> Optional[pd.DataFrame]:
    """Load windows_states_hard_<model>.csv for step-layout or root-layout outputs."""
    candidates: List[Path] = []
    if step is not None:
        candidates.append(
            asset_dir / f"step_{step}" / "results" / rep / f"windows_states_hard_{model}.csv"
        )
    candidates.append(asset_dir / "results" / rep / f"windows_states_hard_{model}.csv")
    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_hard_map(
    asset_dir: Path,
    step: Optional[int],
    reps: List[str] = REPS,
    models: List[str] = MODELS,
) -> Dict[Tuple[str, str, int, str], pd.Series]:
    """
    Build {(rep, model, seed, roll): state_series} from disk.
    state_series is indexed by date.
    """
    hard_map: Dict[Tuple[str, str, int, str], pd.Series] = {}
    for rep in reps:
        for model in models:
            df = load_hard_states(asset_dir, step, rep, model)
            if df is None:
                continue
            for (seed, roll), grp in df.groupby(["seed", "roll"]):
                s = grp.set_index("date")["state"].astype(int).sort_index()
                # Drop duplicate dates (keep last, consistent with runner)
                s = s[~s.index.duplicated(keep="last")]
                hard_map[(rep, model, int(seed), str(roll))] = s
    return hard_map


# ---------------------------------------------------------------------------
# Cross-rep and temporal metric computation
# ---------------------------------------------------------------------------

def compute_pair_metrics(
    s_a: pd.Series, s_b: pd.Series, mode: str = "intersection"
) -> Dict[str, float]:
    """
    Compute ARI, AMI, VI for a pair of state series.

    mode="intersection": compare on common timestamps.
    mode="disjoint": compare non-overlapping segments only (ordered pairing).
    """
    if mode == "disjoint":
        idx_a_only = s_a.index.difference(s_b.index)
        idx_b_only = s_b.index.difference(s_a.index)
        if len(idx_a_only) < 2 or len(idx_b_only) < 2:
            return {"ari": float("nan"), "ami": float("nan"), "vi": float("nan")}
        a_raw = s_a.loc[idx_a_only].values.astype(int)
        b_raw = s_b.loc[idx_b_only].values.astype(int)
        n = int(min(len(a_raw), len(b_raw)))
        if n < 2:
            return {"ari": float("nan"), "ami": float("nan"), "vi": float("nan")}
        a = a_raw[:n]
        b = b_raw[:n]
    else:
        idx = s_a.index.intersection(s_b.index)
        if len(idx) < 2:
            return {"ari": float("nan"), "ami": float("nan"), "vi": float("nan")}
        a = s_a.loc[idx].values.astype(int)
        b = s_b.loc[idx].values.astype(int)
    return {
        "ari": float(adjusted_rand_score(a, b)),
        "ami": float(adjusted_mutual_info_score(a, b)),
        "vi": float(variation_of_information(a, b)),
    }


def _cross_rep_one_seed(
    model: str, seed: int,
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    reps: List[str], rolls: List[str],
) -> List[Dict]:
    records = []
    for roll in rolls:
        series = {
            rep: hard_map[(rep, model, seed, roll)]
            for rep in reps
            if (rep, model, seed, roll) in hard_map
        }
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                a_name, b_name = reps[i], reps[j]
                if a_name not in series or b_name not in series:
                    continue
                m = compute_pair_metrics(series[a_name], series[b_name])
                records.append({"rep_a": a_name, "rep_b": b_name, "model": model,
                                "seed": seed, "roll": roll, **m})
    return records


def cross_rep_records(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    reps: List[str],
    models: List[str],
    seeds: List[int],
    rolls: List[str],
    n_jobs: int = 18,
) -> List[Dict]:
    from joblib import Parallel, delayed
    tasks = [(m, s) for m in models for s in seeds]
    shards = {(m, s): {k: v for k, v in hard_map.items() if k[1] == m and k[2] == s}
              for m, s in tasks}
    logger.info("  Cross-rep AMI/VI: %d tasks, n_jobs=%d", len(tasks), n_jobs)
    nested = Parallel(n_jobs=min(n_jobs, len(tasks)), backend="loky")(
        delayed(_cross_rep_one_seed)(m, s, shards[(m, s)], reps, rolls)
        for m, s in tasks
    )
    return [r for batch in nested for r in batch]


def _temporal_one_unit(
    rep: str, model: str, seed: int,
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    rolls: List[str],
) -> List[Dict]:
    records = []
    for i in range(len(rolls) - 1):
        roll_a, roll_b = rolls[i], rolls[i + 1]
        sa = hard_map.get((rep, model, seed, roll_a))
        sb = hard_map.get((rep, model, seed, roll_b))
        if sa is None or sb is None:
            continue
        m = compute_pair_metrics(sa, sb, mode="disjoint")
        records.append({"rep": rep, "model": model, "seed": seed,
                        "roll_a": roll_a, "roll_b": roll_b,
                        "temporal_eval_mode": "disjoint", **m})
    return records


def temporal_records(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    reps: List[str],
    models: List[str],
    seeds: List[int],
    rolls: List[str],
    n_jobs: int = 18,
) -> List[Dict]:
    from joblib import Parallel, delayed
    tasks = [(rep, m, s) for rep in reps for m in models for s in seeds]
    shards = {(rep, m, s): {k: v for k, v in hard_map.items() if k[0] == rep and k[1] == m and k[2] == s}
              for rep, m, s in tasks}
    logger.info("  Temporal AMI/VI: %d tasks, n_jobs=%d", len(tasks), n_jobs)
    nested = Parallel(n_jobs=min(n_jobs, len(tasks)), backend="loky")(
        delayed(_temporal_one_unit)(rep, m, s, shards[(rep, m, s)], rolls)
        for rep, m, s in tasks
    )
    return [r for batch in nested for r in batch]


# ---------------------------------------------------------------------------
# Aggregate permutation test
# ---------------------------------------------------------------------------

def _perm_one_pair(
    a: np.ndarray, b: np.ndarray, n_perm: int, seed: int,
) -> Dict[str, float]:
    """Permutation test for one pair — parallelisable unit."""
    rng = np.random.default_rng(seed)
    obs = float(adjusted_rand_score(a, b))
    exceed = sum(
        1 for _ in range(n_perm)
        if float(adjusted_rand_score(a, rng.permutation(b))) >= obs
    )
    pval = (exceed + 1) / (n_perm + 1)
    return {"obs": obs, "exceed": exceed, "pval": pval}


def aggregate_permutation_pvalue(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    reps: List[str],
    models: List[str],
    seeds: List[int],
    rolls: List[str],
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
    n_sample_pairs: Optional[int] = None,
    n_jobs: int = 18,
) -> Dict[str, float]:
    """
    Efficient one-sided permutation p-value: H0: cross-rep ARI <= 0 (random labelling).

    Parallelised over pairs via joblib.
    """
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)

    # Collect all valid (key_a, key_b) pair candidates
    candidates = []
    for model in models:
        for s in seeds:
            for roll in rolls:
                for i in range(len(reps)):
                    for j in range(i + 1, len(reps)):
                        ka = (reps[i], model, s, roll)
                        kb = (reps[j], model, s, roll)
                        if ka in hard_map and kb in hard_map:
                            candidates.append((ka, kb))

    if not candidates:
        return {"pvalue_all": float("nan")}

    if n_sample_pairs is None:
        chosen = list(candidates)
    else:
        chosen_idx = rng.choice(
            len(candidates),
            size=min(int(n_sample_pairs), len(candidates)),
            replace=False,
        )
        chosen = [candidates[i] for i in chosen_idx]

    # Prepare arrays (avoid pickling hard_map)
    pair_data = []
    for ka, kb in chosen:
        sa = hard_map[ka]
        sb = hard_map[kb]
        idx = sa.index.intersection(sb.index)
        if len(idx) < 10:
            continue
        a = sa.loc[idx].values.astype(int)
        b = sb.loc[idx].values.astype(int)
        pair_data.append((a, b))

    if not pair_data:
        return {"pvalue_all": float("nan")}

    logger.info("  Permutation test: %d pairs, n_perm=%d, n_jobs=%d", len(pair_data), n_perm, n_jobs)

    # Parallel permutation tests — each pair gets a unique seed
    results = Parallel(n_jobs=min(n_jobs, len(pair_data)), backend="loky")(
        delayed(_perm_one_pair)(a, b, n_perm, seed + i)
        for i, (a, b) in enumerate(pair_data)
    )

    total_exceed = sum(r["exceed"] + 1 for r in results)
    total_trials = len(results) * (n_perm + 1)
    pvalue = total_exceed / total_trials
    obs_aris = [r["obs"] for r in results]
    mean_obs = float(np.mean(obs_aris))
    per_pair_pvals = [r["pval"] for r in results]

    logger.info(
        "  Permutation test: mean_obs_ARI=%.4f, aggregate p-value=%.6f "
        "(n_pairs=%d, n_perm=%d each)",
        mean_obs, pvalue, len(obs_aris), n_perm,
    )
    if per_pair_pvals:
        logger.info(
            "  Per-pair p-values: min=%.4f, median=%.4f, max=%.4f (n=%d)",
            float(np.min(per_pair_pvals)), float(np.median(per_pair_pvals)),
            float(np.max(per_pair_pvals)), len(per_pair_pvals),
        )

    result: Dict[str, float] = {"pvalue_all": pvalue, "obs_ari": mean_obs}
    if per_pair_pvals:
        result["pvalue_perpair_min"] = float(np.min(per_pair_pvals))
        result["pvalue_perpair_median"] = float(np.median(per_pair_pvals))
        result["pvalue_perpair_max"] = float(np.max(per_pair_pvals))
        result["pvalue_perpair_n"] = float(len(per_pair_pvals))
    return result


# ---------------------------------------------------------------------------
# Key-results update
# ---------------------------------------------------------------------------

def _mean_or_nan(x) -> float:
    try:
        v = float(pd.to_numeric(pd.Series(x), errors="coerce").mean())
        return v
    except Exception:
        return float("nan")


def update_key_results(key_results_path: Path, new_rows: List[Dict]) -> None:
    """Append new metric rows to an existing key_results.csv (no duplicates)."""
    existing = pd.read_csv(key_results_path)
    # Remove any previously written posthoc rows (idempotent re-run)
    posthoc_metrics = {r["metric"] for r in new_rows}
    existing = existing[~existing["metric"].isin(posthoc_metrics)]
    new_df = pd.DataFrame(new_rows)
    out = pd.concat([existing, new_df], ignore_index=True)
    out.to_csv(key_results_path, index=False)
    logger.info("  Updated %s (+%d rows)", key_results_path.name, len(new_rows))


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_asset_step(
    asset_dir: Path,
    step: Optional[int],
    reps: List[str],
    models: List[str],
    key_results_path: Path,
) -> None:
    asset = asset_dir.name
    step_label = "root" if step is None else f"step={step}"
    logger.info("Processing %s / %s", asset, step_label)

    # Infer seeds and rolls from disk
    hard_map = build_hard_map(asset_dir, step, reps=reps, models=models)
    if not hard_map:
        logger.warning("  No hard states found; skipping.")
        return

    present_reps = sorted({k[0] for k in hard_map})
    present_models = sorted({k[1] for k in hard_map})
    seeds = sorted({k[2] for k in hard_map})
    rolls = sorted({k[3] for k in hard_map})

    logger.info("  reps=%s models=%s seeds=%s rolls=%d",
                present_reps, present_models, seeds, len(rolls))

    # --- Cross-rep metrics ---
    cr_records = cross_rep_records(hard_map, present_reps, present_models, seeds, rolls)
    cr_df = pd.DataFrame(cr_records)

    # --- Temporal metrics ---
    tmp_records = temporal_records(hard_map, present_reps, present_models, seeds, rolls)
    tmp_df = pd.DataFrame(tmp_records)

    # --- Permutation test (aggregate, only for baseline step) ---
    perm_result = aggregate_permutation_pvalue(
        hard_map, present_reps, present_models, seeds, rolls,
        n_sample_pairs=PERM_MAX_PAIRS,
    )

    # --- Build new key_results rows ---
    new_rows: List[Dict] = []

    def _add(metric, scope, value, n=0):
        if not np.isnan(float(value if value is not None else float("nan"))):
            new_rows.append({"metric": metric, "scope": scope, "value": value, "n": n})

    if not cr_df.empty:
        _add("cross_rep_ami_mean", "all", _mean_or_nan(cr_df["ami"]), len(cr_df))
        _add("cross_rep_vi_mean",  "all", _mean_or_nan(cr_df["vi"]),  len(cr_df))
        if "model" in cr_df.columns:
            for model_name, g in cr_df.groupby("model"):
                _add(f"cross_rep_ami_mean", f"model={model_name}",
                     _mean_or_nan(g["ami"]), len(g))
                _add(f"cross_rep_vi_mean",  f"model={model_name}",
                     _mean_or_nan(g["vi"]),  len(g))

    if not tmp_df.empty:
        _add("temporal_ami_mean", "all", _mean_or_nan(tmp_df["ami"]), len(tmp_df))
        _add("temporal_vi_mean",  "all", _mean_or_nan(tmp_df["vi"]),  len(tmp_df))
        if "model" in tmp_df.columns:
            for model_name, g in tmp_df.groupby("model"):
                _add(f"temporal_ami_mean", f"model={model_name}",
                     _mean_or_nan(g["ami"]), len(g))
                _add(f"temporal_vi_mean",  f"model={model_name}",
                     _mean_or_nan(g["vi"]),  len(g))

    if not np.isnan(perm_result.get("pvalue_all", float("nan"))):
        _add("crossrep_ari_perm_pvalue", "all", perm_result["pvalue_all"])
    for suffix in ("perpair_min", "perpair_median", "perpair_max", "perpair_n"):
        key = f"pvalue_{suffix}"
        if key in perm_result and not np.isnan(perm_result[key]):
            _add(f"crossrep_ari_perm_{suffix}", "all", perm_result[key])

    # --- Update key_results.csv ---
    if key_results_path.exists() and new_rows:
        update_key_results(key_results_path, new_rows)
    else:
        logger.warning("  key_results.csv not found at %s", key_results_path)


def update_step_sweep_summary(outputs_dir: Path, steps: List[int]) -> None:
    """Regenerate step_sweep_summary.csv with ami/vi columns added."""
    summary_path = outputs_dir / "step_sweep_summary.csv"
    rows = []
    for asset_dir in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        asset = asset_dir.name
        for step in steps:
            kr_path = asset_dir / f"step_{step}" / "key_results.csv"
            if not kr_path.exists():
                continue
            kr = pd.read_csv(kr_path)
            def _get(metric, scope="all"):
                s = kr[(kr["metric"] == metric) & (kr["scope"] == scope)]
                return float(pd.to_numeric(s["value"].iloc[0], errors="coerce")) if not s.empty else float("nan")
            rows.append({
                "step": step,
                "asset": asset,
                "cross_rep_ari_mean":  _get("cross_rep_ari_mean"),
                "temporal_ari_mean":   _get("temporal_ari_mean"),
                "temporal_overlap_ari_mean": _get("temporal_overlap_ari_mean"),
                "cross_rep_ami_mean":  _get("cross_rep_ami_mean"),
                "temporal_ami_mean":   _get("temporal_ami_mean"),
                "cross_rep_vi_mean":   _get("cross_rep_vi_mean"),
                "temporal_vi_mean":    _get("temporal_vi_mean"),
                "crossrep_ari_perm_pvalue": _get("crossrep_ari_perm_pvalue"),
            })
    if rows:
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        logger.info("Updated step_sweep_summary.csv (%d rows)", len(rows))


def _resolve_run_defs(cfg: Optional[Dict]) -> Tuple[List[int], List[str], List[str]]:
    steps = [21, 63, 126, 252]
    reps = list(REPS)
    models = list(MODELS)
    if isinstance(cfg, dict):
        grid = cfg.get("grid", {}) if isinstance(cfg.get("grid", {}), dict) else {}
        if isinstance(grid.get("step_sweep"), list) and grid.get("step_sweep"):
            steps = [int(x) for x in grid.get("step_sweep")]
        elif "step" in grid:
            steps = [int(grid.get("step"))]
        reps_cfg = cfg.get("representations", {})
        if isinstance(reps_cfg, dict) and reps_cfg:
            reps = [str(k) for k in reps_cfg.keys()]
        models_cfg = cfg.get("models", {})
        if isinstance(models_cfg, dict) and models_cfg:
            models = [
                str(m)
                for m in ("gmm", "hmm")
                if isinstance(models_cfg.get(m, {}), dict)
                and bool(models_cfg.get(m, {}).get("enabled", True))
            ]
    return steps, reps, models


def main(cfg: Optional[Dict] = None) -> None:
    if cfg is None:
        cfg_path = ROOT / "config.yaml"
        if cfg_path.exists():
            try:
                import yaml

                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = None
    if cfg is None:
        logger.error("posthoc_ami_vi_perm requires config context (cfg or ROOT/config.yaml).")
        sys.exit(1)
    steps, reps, models = _resolve_run_defs(cfg)
    outputs_dir = Path(cfg.get("outputs_dir", str(OUTPUTS_DIR))) if isinstance(cfg, dict) else OUTPUTS_DIR
    asset_dirs = sorted(
        p for p in outputs_dir.iterdir()
        if p.is_dir() and any((p / f"step_{s}").exists() for s in steps)
    )
    if not asset_dirs:
        # Fallback root-layout assets
        asset_dirs = sorted(
            p for p in outputs_dir.iterdir() if p.is_dir() and (p / "key_results.csv").exists()
        )
    if not asset_dirs:
        logger.error("No asset directories found under %s", outputs_dir)
        sys.exit(1)

    for asset_dir in asset_dirs:
        step_dirs = [asset_dir / f"step_{step}" for step in steps]
        has_step_layout = any(p.exists() for p in step_dirs)
        if has_step_layout:
            for step in steps:
                step_dir = asset_dir / f"step_{step}"
                if step_dir.exists():
                    process_asset_step(
                        asset_dir,
                        step,
                        reps=reps,
                        models=models,
                        key_results_path=asset_dir / f"step_{step}" / "key_results.csv",
                    )
        else:
            process_asset_step(
                asset_dir,
                None,
                reps=reps,
                models=models,
                key_results_path=asset_dir / "key_results.csv",
            )

    update_step_sweep_summary(outputs_dir, steps)
    logger.info("Done.")


if __name__ == "__main__":
    main()
