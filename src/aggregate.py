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
    for p in Path(results_dir).rglob("rep_stability.json"):
        payload = _read_json(p)
        records.extend(payload.get("rep_stability", []))
    for p in Path(results_dir).rglob("window_stability.json"):
        payload = _read_json(p)
        records.extend(payload.get("window_stability", []))
    for p in Path(results_dir).rglob("stability.json"):
        payload = _read_json(p)
        records.extend(payload.get("stability", []))
    df = pd.DataFrame(records)
    if not df.empty:
        logger.debug("aggregate_stability: %d records from %s", len(df), results_dir)
    return df

