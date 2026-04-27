from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def reps_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    """Return representation names from a loaded config.yaml dict.

    Raises on missing/empty cfg.representations. No silent fallback: hardcoded
    module-level rep lists previously drifted from config (e.g., rep_e was
    dropped from S&P-500 post-hoc analyses), passing silently incomplete
    outputs through the pipeline.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict (loaded from config.yaml); got {type(cfg).__name__}")
    reps = cfg.get("representations")
    if not isinstance(reps, dict) or not reps:
        raise KeyError("cfg.representations must be a non-empty dict of rep_name -> spec")
    return [str(k) for k in reps.keys()]


def enabled_models_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    """Return ('gmm','hmm') filtered by cfg.models[*].enabled (default True)."""
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict; got {type(cfg).__name__}")
    models_cfg = cfg.get("models")
    if not isinstance(models_cfg, dict) or not models_cfg:
        raise KeyError("cfg.models must be a non-empty dict")
    models = [
        str(m)
        for m in ("gmm", "hmm")
        if isinstance(models_cfg.get(m, {}), dict)
        and bool(models_cfg.get(m, {}).get("enabled", True))
    ]
    if not models:
        raise ValueError("cfg.models has no enabled entries among ('gmm', 'hmm')")
    return models


def assets_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    """Return the asset list from cfg.assets. Raises on missing/empty."""
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict; got {type(cfg).__name__}")
    assets = cfg.get("assets")
    if not isinstance(assets, list) or not assets:
        raise KeyError("cfg.assets must be a non-empty list")
    return [str(a) for a in assets]


def safe_name(name: str) -> str:
    """Convert an identifier (e.g. '^GSPC') into a filesystem-safe token."""
    return (
        str(name)
        .replace("^", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .strip()
    )


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def rolling_slices(n: int, window: int, step: int) -> List[Tuple[int, int]]:
    """Generate (start, end) slices for rolling windows over [0..n)."""
    out: List[Tuple[int, int]] = []
    for start in range(0, max(0, n - window + 1), step):
        end = start + window
        if end <= n:
            out.append((start, end))
    return out


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

