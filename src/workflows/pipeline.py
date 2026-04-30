"""
Top-level pipeline entry for Paper 1.

Orchestrates the full run:
  1. Load config, validate assets, prepare outputs directory and run.log
  2. Download missing tickers via core.data
  3. Dispatch to step / robustness sweeps (workflows.sweeps) or to a single
     per-asset run (workflows.single_asset) when no sweep is configured
  4. Write a multi-asset summary CSV and (best-effort) auto-fill paper
     Results numbers from outputs

The heavy lifting lives elsewhere: per-asset orchestration in
workflows.single_asset, sweep machinery in workflows.sweeps, fit
primitives in core.fits, stability / ordering / Wasserstein primitives
in core.{stability,ordering}, output writers in workflows.outputs,
visualisation in visualization.{representation_failure,disagreement,summary}.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from src.core.data import download_tickers
from src.core.runtime import (
    configure_console_logging,
    configure_global_file_logging,
    set_thread_env_defaults,
)
from src.core.utils import _fmt_hms, _rmtree_with_retries, safe_name
from src.experiments.synthetic_sanity import run_synthetic_sanity_check
from src.workflows.paper_autofill import (
    update_empirical_results_md,
    update_main_tex_tables,
)
from src.workflows.single_asset import _run_single_asset
from src.workflows.sweeps import _run_robustness_sweep, _run_step_sweep

# Console logging + thread env defaults early (before numpy / scikit-learn).
configure_console_logging()
set_thread_env_defaults(1)

logger = logging.getLogger(__name__)


def _extract_metrics_from_key_results(p: Path) -> dict | None:
    """Read key_results.csv; return cross_rep_ari_mean, temporal_ari_mean (scope=all) or None."""
    if not p.exists():
        return None
    df = pd.read_csv(p)
    all_rows = df[df["scope"] == "all"]
    cross = all_rows[all_rows["metric"] == "cross_rep_ari_mean"]
    temp = all_rows[all_rows["metric"] == "temporal_ari_mean"]
    if cross.empty:
        return None
    return {
        "cross_rep_ari_mean": float(cross["value"].iloc[0]),
        # Temporal ARI can still be missing if an upstream run did not emit
        # temporal rows (e.g., partial / incomplete outputs).
        "temporal_ari_mean": (
            pd.to_numeric(temp["value"].iloc[0], errors="coerce")
            if not temp.empty
            else float("nan")
        ),
    }


def _write_key_results_all_assets(
    outputs_dir: Path,
    assets: List[str],
    candidate_paths_by_asset: Dict[str, Path],
    label: str,
) -> None:
    """Write a concise multi-asset key results table.

    Used for paper tables that need a single CSV (e.g., baseline step=21 run),
    even when the main run was a sweep.
    """
    rows: List[pd.DataFrame] = []
    for asset in assets:
        name = safe_name(asset)
        p = candidate_paths_by_asset.get(name)
        if p is None or not p.exists():
            continue
        df = pd.read_csv(p)
        df.insert(0, "asset", name)
        rows.append(df)
    if not rows:
        logger.warning("No key_results found for multi-asset summary (%s).", label)
        return
    out_csv = Path(outputs_dir) / "key_results_all_assets.csv"
    pd.concat(rows, axis=0, ignore_index=True).to_csv(out_csv, index=False)
    (Path(outputs_dir) / "analysis_all_assets.md").write_text(
        "# Multi-asset summary\n\n"
        + f"- label: {label}\n"
        + f"- assets: {', '.join(safe_name(a) for a in assets)}\n"
        + f"- csv: {out_csv.name}\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s (%s)", out_csv, label)


def run(config_path: Path) -> None:
    t_run0 = time.perf_counter()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    assets = cfg.get("assets", None)
    if not isinstance(assets, list) or not assets:
        raise ValueError("Config requires `assets` as a non-empty list.")
    assets = [str(a) for a in assets]

    outputs_dir = Path(cfg.get("outputs_dir", "outputs"))
    raw_dir = Path(cfg.get("raw_dir", cfg.get("data", {}).get("raw_dir", "data")))
    grid = cfg.get("grid") or {}
    step_sweep = grid.get("step_sweep")
    robustness = grid.get("robustness") or {}
    robustness_enabled = bool(robustness.get("enabled", False))

    # Clear outputs once per invocation (requested).
    if outputs_dir.exists():
        _rmtree_with_retries(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    configure_global_file_logging(outputs_dir / "run.log")

    try:
        logger.info(
            "Starting run; assets=%s raw_dir=%s outputs_dir=%s config=%s",
            ",".join(assets),
            raw_dir,
            outputs_dir,
            Path(config_path),
        )
        data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
        download_start = str(data_cfg.get("start_date", "2005-01-01"))
        download_end = data_cfg.get("end_date", None)
        if download_end is not None:
            download_end = str(download_end)
        missing = [a for a in assets if not (raw_dir / f"{safe_name(a)}.csv").exists()]
        if missing:
            print(f"[runner] Downloading missing tickers to {raw_dir}: {missing}")
            download_tickers(
                missing,
                output_dir=raw_dir,
                start_date=download_start,
                end_date=download_end,
            )
        manifest = {
            "assets": assets,
            "raw_dir": str(raw_dir),
            "outputs_dir": str(outputs_dir),
            "config_path": str(Path(config_path)),
            "download_start_date": download_start,
            "download_end_date": (download_end if download_end is not None else "today_utc"),
            "downloaded_missing_assets": missing,
        }
        (outputs_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        ran_any = False

        # 1) Step sweep (overlap sensitivity).
        if isinstance(step_sweep, list) and len(step_sweep) > 0:
            _run_step_sweep(cfg, assets, outputs_dir, [int(x) for x in step_sweep])
            if 21 in [int(x) for x in step_sweep]:
                candidates: Dict[str, Path] = {}
                for asset in assets:
                    name = safe_name(asset)
                    candidates[name] = outputs_dir / name / "step_21" / "key_results.csv"
                _write_key_results_all_assets(
                    outputs_dir=outputs_dir,
                    assets=assets,
                    candidate_paths_by_asset=candidates,
                    label="baseline step=21 (from step_sweep)",
                )
            ran_any = True

        # 2) Robustness sweep (K x seeds).
        if robustness_enabled:
            _run_robustness_sweep(cfg, assets, outputs_dir, robustness)
            ran_any = True

        # 3) Fallback: single-step run when no sweeps configured.
        if not ran_any:
            experiment_dir = outputs_dir
            for asset in assets:
                _run_single_asset(asset, cfg, experiment_dir)

        # Optional: one-page synthetic sanity check for Appendix (disabled by default).
        try:
            run_synthetic_sanity_check(cfg, outputs_dir)
        except Exception as e:
            logger.warning("Synthetic sanity check failed (continuing without it): %s", e)

        # Auto-fill paper Results numbers from outputs.
        try:
            try:
                from src.experiments.posthoc_ami_vi_perm import main as posthoc_ami_vi_perm_main

                posthoc_ami_vi_perm_main(cfg=cfg)
                logger.info("Post-hoc AMI/VI/permutation metrics updated.")
            except Exception as e:
                logger.warning("Could not run post-hoc AMI/VI/permutation metrics: %s", e)

            project_dir = Path(__file__).resolve().parents[2]
            md_path = project_dir / "paper" / "sections" / "04_empirical_results.md"
            if md_path.exists():
                update_empirical_results_md(outputs_dir=outputs_dir, md_path=md_path, cfg=cfg)
                logger.info("Auto-filled Results numbers in %s", md_path)
            tex_path = project_dir / "paper" / "main.tex"
            if tex_path.exists():
                update_main_tex_tables(outputs_dir=outputs_dir, tex_path=tex_path, cfg=cfg)
                logger.info("Synchronized numeric table rows in %s", tex_path)
        except Exception as e:
            logger.warning("Could not auto-fill paper Results numbers: %s", e)

        # Always try to write a baseline multi-asset summary CSV for the paper.
        candidates: Dict[str, Path] = {}
        if isinstance(step_sweep, list) and len(step_sweep) > 0 and 21 in [int(x) for x in step_sweep]:
            for asset in assets:
                name = safe_name(asset)
                candidates[name] = outputs_dir / name / "step_21" / "key_results.csv"
            _write_key_results_all_assets(
                outputs_dir=outputs_dir,
                assets=assets,
                candidate_paths_by_asset=candidates,
                label="baseline step=21 (from step_sweep)",
            )
        else:
            for asset in assets:
                name = safe_name(asset)
                candidates[name] = outputs_dir / name / "key_results.csv"
            _write_key_results_all_assets(
                outputs_dir=outputs_dir,
                assets=assets,
                candidate_paths_by_asset=candidates,
                label="baseline (per-asset root)",
            )
    except Exception:
        logger.exception("Run failed with an uncaught exception.")
        raise
    finally:
        elapsed_run = float(time.perf_counter() - t_run0)
        logger.info("Run complete; total_elapsed_s=%.1f (%s)", elapsed_run, _fmt_hms(elapsed_run))
