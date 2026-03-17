from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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

