"""
Logging configuration for Paper 1 (Representation-Invariant).

Re-exports from the legacy src/runtime module and adds Paper 3-aligned
helpers for UTF-8 console and UTF-16 log cleanup.
"""

from __future__ import annotations

# Re-export everything from the legacy runtime module.
from src.runtime import (              # noqa: F401
    configure_console_logging,
    configure_global_file_logging,
    set_thread_env_defaults,
    tee_console_to_file,
)

import sys
from pathlib import Path


def _remove_utf16_shell_tee_log(log_path: Path) -> None:
    """PowerShell Tee-Object writes UTF-16LE by default; mixing with UTF-8 FileHandler corrupts the file."""
    if not log_path.is_file():
        return
    try:
        head = log_path.read_bytes()[:4]
    except OSError:
        return
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass


_utf8_done = False


def ensure_utf8_console():
    """Force UTF-8 on Windows console streams (idempotent)."""
    global _utf8_done
    if _utf8_done:
        return
    import io
    if sys.platform == "win32":
        for stream in ("stdout", "stderr"):
            current = getattr(sys, stream)
            if hasattr(current, "buffer") and getattr(current, "encoding", "").lower() != "utf-8":
                setattr(sys, stream, io.TextIOWrapper(current.buffer, encoding="utf-8", errors="replace"))
    _utf8_done = True
