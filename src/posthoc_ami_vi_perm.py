"""
Post-hoc computation of AMI, VI, and permutation p-values from existing outputs.

Reads windows_states_hard_*.csv files already on disk, recomputes cross-rep and
temporal stability with the new metrics (AMI, VI), runs aggregate permutation tests,
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
REPS = ["rep_a", "rep_a_unscaled", "rep_b", "rep_c1", "rep_c2", "rep_c3"]
MODELS = ["gmm", "hmm"]
K_DEFAULT = 3
N_PERM = 999
PERM_SEED = 42


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
    asset_dir: Path, step: int, rep: str, model: str
) -> Optional[pd.DataFrame]:
    """Load windows_states_hard_<model>.csv for a given asset/step/rep/model."""
    p = asset_dir / f"step_{step}" / "results" / rep / f"windows_states_hard_{model}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_hard_map(
    asset_dir: Path,
    step: int,
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
    s_a: pd.Series, s_b: pd.Series
) -> Dict[str, float]:
    """Compute ARI, AMI, VI on the time-index intersection of two state series."""
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


def cross_rep_records(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    reps: List[str],
    models: List[str],
    seeds: List[int],
    rolls: List[str],
) -> List[Dict]:
    records = []
    for model in models:
        for seed in seeds:
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
                        records.append(
                            {
                                "rep_a": a_name,
                                "rep_b": b_name,
                                "model": model,
                                "seed": seed,
                                "roll": roll,
                                **m,
                            }
                        )
    return records


def temporal_records(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    reps: List[str],
    models: List[str],
    seeds: List[int],
    rolls: List[str],
) -> List[Dict]:
    records = []
    for rep in reps:
        for model in models:
            for seed in seeds:
                for i in range(len(rolls) - 1):
                    roll_a, roll_b = rolls[i], rolls[i + 1]
                    sa = hard_map.get((rep, model, seed, roll_a))
                    sb = hard_map.get((rep, model, seed, roll_b))
                    if sa is None or sb is None:
                        continue
                    m = compute_pair_metrics(sa, sb)
                    records.append(
                        {
                            "rep": rep,
                            "model": model,
                            "seed": seed,
                            "roll_a": roll_a,
                            "roll_b": roll_b,
                            **m,
                        }
                    )
    return records


# ---------------------------------------------------------------------------
# Aggregate permutation test
# ---------------------------------------------------------------------------

def aggregate_permutation_pvalue(
    hard_map: Dict[Tuple[str, str, int, str], pd.Series],
    reps: List[str],
    models: List[str],
    seeds: List[int],
    rolls: List[str],
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
    n_sample_pairs: int = 60,
) -> Dict[str, float]:
    """
    Efficient one-sided permutation p-value: H0: cross-rep ARI <= 0 (random labelling).

    Strategy: randomly sample n_sample_pairs cross-rep (a, b) pairs; for each, run
    n_perm permutations of b and record whether permuted ARI >= observed.  Pool counts
    across pairs to obtain an aggregate p-value.  Using a sample of pairs keeps runtime
    manageable while maintaining the spirit of a permutation test.
    """
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

    # Sample without replacement (or take all if fewer than n_sample_pairs)
    chosen_idx = rng.choice(len(candidates),
                             size=min(n_sample_pairs, len(candidates)),
                             replace=False)
    chosen = [candidates[i] for i in chosen_idx]

    total_exceed = 0
    total_trials = 0
    obs_aris = []

    for ka, kb in chosen:
        sa = hard_map[ka]
        sb = hard_map[kb]
        idx = sa.index.intersection(sb.index)
        if len(idx) < 10:
            continue
        a = sa.loc[idx].values.astype(int)
        b = sb.loc[idx].values.astype(int)
        obs = float(adjusted_rand_score(a, b))
        obs_aris.append(obs)
        exceed = sum(
            1 for _ in range(n_perm)
            if float(adjusted_rand_score(a, rng.permutation(b))) >= obs
        )
        total_exceed += exceed + 1
        total_trials += n_perm + 1

    if total_trials == 0 or not obs_aris:
        return {"pvalue_all": float("nan")}

    pvalue = total_exceed / total_trials
    mean_obs = float(np.mean(obs_aris))
    logger.info(
        "  Permutation test: mean_obs_ARI=%.4f, aggregate p-value=%.6f "
        "(n_pairs=%d, n_perm=%d each)",
        mean_obs, pvalue, len(obs_aris), n_perm,
    )
    return {"pvalue_all": pvalue, "obs_ari": mean_obs}


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

def process_asset_step(asset_dir: Path, step: int) -> None:
    asset = asset_dir.name
    logger.info("Processing %s / step=%d", asset, step)

    # Infer seeds and rolls from disk
    hard_map = build_hard_map(asset_dir, step)
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
        hard_map, present_reps, present_models, seeds, rolls
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

    # --- Update key_results.csv ---
    key_results_path = asset_dir / f"step_{step}" / "key_results.csv"
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
                "cross_rep_ami_mean":  _get("cross_rep_ami_mean"),
                "temporal_ami_mean":   _get("temporal_ami_mean"),
                "cross_rep_vi_mean":   _get("cross_rep_vi_mean"),
                "temporal_vi_mean":    _get("temporal_vi_mean"),
                "crossrep_ari_perm_pvalue": _get("crossrep_ari_perm_pvalue"),
            })
    if rows:
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        logger.info("Updated step_sweep_summary.csv (%d rows)", len(rows))


def main() -> None:
    steps = [21, 63, 126, 252]
    asset_dirs = sorted(
        p for p in OUTPUTS_DIR.iterdir()
        if p.is_dir() and any((p / f"step_{s}").exists() for s in steps)
    )
    if not asset_dirs:
        logger.error("No asset directories found under %s", OUTPUTS_DIR)
        sys.exit(1)

    for asset_dir in asset_dirs:
        for step in steps:
            step_dir = asset_dir / f"step_{step}"
            if step_dir.exists():
                process_asset_step(asset_dir, step)

    update_step_sweep_summary(OUTPUTS_DIR, steps)
    logger.info("Done.")


if __name__ == "__main__":
    main()
