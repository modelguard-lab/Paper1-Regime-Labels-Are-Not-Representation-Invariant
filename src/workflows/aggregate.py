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
    records: List[Dict] = []
    for score_path in Path(results_dir).rglob("scores.json"):
        scores = _read_json(score_path)
        parts = score_path.parts
        record = {"path": str(score_path)}
        if "hmm" in parts:
            record["model"] = "hmm"
        if "gmm" in parts:
            record["model"] = "gmm"
        for p in parts:
            if p.startswith("K"):
                record["K"] = int(p[1:])
            if p.startswith("W"):
                record["W"] = int(p[1:])
            if p.startswith("S"):
                record["seed"] = int(p[1:])
            if p.startswith("roll_"):
                record["roll"] = p
        record.update(scores)
        records.append(record)
    df = pd.DataFrame(records)
    if not df.empty:
        logger.debug("aggregate_scores: %d records from %s", len(df), results_dir)
    return df


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

