from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aggregate_scores(results_dir: Path) -> pd.DataFrame:
    """Aggregate per-rep windows_scores_<model>.csv files under ``results_dir``.

    The modern pipeline writes one CSV per (rep, model) at
    ``<results_dir>/<rep>/windows_scores_<model>.csv``. Each row already carries
    ``rep, model, K, W, seed, roll`` plus model-fit columns. This function unions
    those CSVs and tags each record with its source path.
    """
    base = Path(results_dir)
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(base.rglob("windows_scores_*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # corrupt or empty csv
            logger.warning("aggregate_scores: failed to read %s (%s)", csv_path, exc)
            continue
        if df.empty:
            continue
        df["path"] = str(csv_path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0, ignore_index=True)
    logger.debug("aggregate_scores: %d records from %s", len(out), results_dir)
    return out


def aggregate_stability(results_dir: Path) -> pd.DataFrame:
    records: List[Dict] = []
    base = Path(results_dir)
    legacy_paths = sorted(base.rglob("stability.json"))
    legacy_dirs = {p.parent.resolve() for p in legacy_paths}

    for p in sorted(base.rglob("rep_stability.json")):
        if p.parent.resolve() in legacy_dirs:
            logger.warning("aggregate_stability: skipping %s because stability.json exists nearby", p)
            continue
        payload = _read_json(p)
        records.extend(payload.get("rep_stability", []))
    for p in sorted(base.rglob("window_stability.json")):
        if p.parent.resolve() in legacy_dirs:
            logger.warning("aggregate_stability: skipping %s because stability.json exists nearby", p)
            continue
        payload = _read_json(p)
        records.extend(payload.get("window_stability", []))
    for p in legacy_paths:
        payload = _read_json(p)
        records.extend(payload.get("stability", []))
    df = pd.DataFrame(records)
    if not df.empty:
        dedup_keys = [
            "rep_a",
            "rep_b",
            "rep",
            "model",
            "K",
            "window",
            "seed",
            "roll",
            "roll_a",
            "roll_b",
            "ari",
            "nmi",
            "ami",
            "vi",
        ]
        present_keys = [c for c in dedup_keys if c in df.columns]
        if present_keys:
            before = int(len(df))
            df = df.drop_duplicates(subset=present_keys, keep="last")
            if len(df) < before:
                logger.warning(
                    "aggregate_stability: dropped %d duplicate rows (from %d to %d)",
                    before - len(df),
                    before,
                    len(df),
                )
    if not df.empty:
        logger.debug("aggregate_stability: %d records from %s", len(df), results_dir)
    return df


def main(cfg=None) -> None:
    """CLI entry: scan outputs/<asset>/step_21/results/ for each asset and write
    consolidated scores_aggregate.csv and stability_aggregate.csv at outputs root.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    project = Path(__file__).resolve().parent.parent.parent
    outputs_dir = project / "outputs"
    if cfg is None:
        cfg_path = project / "config.yaml"
        if cfg_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = None
    assets = (cfg or {}).get("assets") or ["IEF", "^GSPC", "GLD", "BTC-USD"]

    def _safe_name(a: str) -> str:
        return a.replace("^", "").replace("/", "_")

    score_frames = []
    stab_frames = []
    for asset in assets:
        results_dir = outputs_dir / _safe_name(asset) / "step_21" / "results"
        if not results_dir.exists():
            logger.warning("aggregate: %s not found; skipping asset=%s", results_dir, asset)
            continue
        s = aggregate_scores(results_dir)
        if not s.empty:
            s.insert(0, "asset", _safe_name(asset))
            score_frames.append(s)
        st = aggregate_stability(results_dir)
        if not st.empty:
            st.insert(0, "asset", _safe_name(asset))
            stab_frames.append(st)

    if score_frames:
        out = pd.concat(score_frames, axis=0, ignore_index=True)
        out_path = outputs_dir / "scores_aggregate.csv"
        out.to_csv(out_path, index=False)
        logger.info("aggregate: wrote %d score records to %s", len(out), out_path)
    else:
        logger.warning("aggregate: no scores aggregated")

    if stab_frames:
        out = pd.concat(stab_frames, axis=0, ignore_index=True)
        out_path = outputs_dir / "stability_aggregate.csv"
        out.to_csv(out_path, index=False)
        logger.info("aggregate: wrote %d stability records to %s", len(out), out_path)
    else:
        logger.warning("aggregate: no stability aggregated")


if __name__ == "__main__":
    main()
