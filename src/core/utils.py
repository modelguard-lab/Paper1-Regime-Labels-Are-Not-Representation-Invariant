from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.core.features import RepConfig


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


def _fmt_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS (rounded)."""
    try:
        s = int(round(float(seconds)))
    except Exception:
        return "NA"
    if s < 0:
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _timing_summary_lines(title: str, totals: Dict[str, float], top_k: int = 12) -> List[str]:
    """Return a compact timing summary line (largest first)."""
    if not totals:
        return [f"{title}: (no timings collected)"]
    items = sorted(((str(k), float(v)) for k, v in totals.items()), key=lambda kv: kv[1], reverse=True)
    shown = items[: max(1, int(top_k))]
    parts = [f"{k}={_fmt_hms(v)}" for k, v in shown]
    more = "" if len(items) <= len(shown) else f" (+{len(items) - len(shown)} more)"
    total_all = sum(v for _, v in items)
    return [f"{title}: total={_fmt_hms(total_all)}; " + ", ".join(parts) + more]


def _rmtree_with_retries(path: Path, retries: int = 8, base_sleep_s: float = 0.3) -> None:
    """Robust rmtree for Windows.

    Handles transient "directory not empty" / permission errors that can happen
    when a previous run crashed and left temporary shard files behind, or when
    the OS is still releasing file handles.
    """
    path = Path(path)
    if not path.exists():
        return

    def _onerror(func, p, exc_info):  # type: ignore[no-untyped-def]
        try:
            os.chmod(p, 0o666)
        except Exception:
            pass
        try:
            func(p)
        except Exception:
            pass

    last_err: Exception | None = None
    for i in range(int(retries)):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except Exception as e:
            last_err = e
            time.sleep(base_sleep_s * (2**i))
    if last_err is not None:
        raise last_err


def _window_roll_name(i: int) -> str:
    return f"roll_{i:03d}"


def _build_rep_configs(cfg: Dict) -> List[RepConfig]:
    reps: List[RepConfig] = []
    for name, rep in (cfg.get("representations", {}) or {}).items():
        reps.append(
            RepConfig(
                name=name,
                features=rep.get("features", []),
                windows=rep.get("windows", {}) or {},
                drop_features=rep.get("drop_features", None),
                standardization=rep.get("standardization", None),
                asset_filter=rep.get("asset_filter", None),
            )
        )
    if not reps:
        raise ValueError("No representations configured.")
    return reps
