from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
import warnings
from typing import TextIO


def configure_console_logging(level: int = logging.INFO) -> None:
    """
    Ensure logs are visible in the console.

    This is intentionally lightweight and only adds a StreamHandler if the root
    logger has no handlers yet (so it won't duplicate logs in notebooks/IDEs
    that already configured logging).
    """

    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class _TeeTextIO:
    """
    Minimal tee wrapper for text streams (stdout/stderr).

    This is used to persist console output to disk, which is helpful when
    third-party libraries (or loky workers) emit warnings/prints that do not
    go through the main process' logging handlers.
    """

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:
        n = self._primary.write(s)
        self._secondary.write(s)
        return n

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:  # pragma: no cover
        try:
            return bool(self._primary.isatty())
        except Exception:
            return False


def tee_console_to_file(log_path: Path) -> None:
    """
    Duplicate sys.stdout and sys.stderr to a file (append, line-buffered).

    Captures warnings/prints that bypass Python logging, including output
    forwarded from joblib/loky worker processes.
    """

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeTextIO(sys.stdout, f)  # type: ignore[assignment]
    sys.stderr = _TeeTextIO(sys.stderr, f)  # type: ignore[assignment]


def configure_global_file_logging(log_path: Path, level: int = logging.INFO) -> None:
    """
    Append a single global FileHandler for the run.

    Unlike `basicConfig`, this will add a FileHandler even if logging was already
    configured elsewhere (e.g. IDE / notebooks).
    """

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    existing = []
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                existing.append(Path(h.baseFilename).resolve())
            except Exception:
                continue
    if log_path.resolve() not in existing:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        root.addHandler(fh)

    if root.level > level:
        root.setLevel(level)

    # Keep third-party libraries quiet during large grids.
    #
    # NOTE: we intentionally DO NOT silence hmmlearn to ERROR here, because
    # the paper workflow relies on being able to inspect convergence warnings
    # (e.g., "Model is not converging") in the single run.log.
    logging.getLogger("hmmlearn").setLevel(logging.WARNING)
    logging.captureWarnings(True)
    warnings.filterwarnings("default")
    # Avoid noisy sklearn KMeans warning on Windows+MKL; we also set OMP_NUM_THREADS=1.
    warnings.filterwarnings(
        "ignore", message="KMeans is known to have a memory leak on Windows with MKL*"
    )


def set_thread_env_defaults(n_threads: int = 1) -> None:
    """
    Set conservative thread defaults for BLAS/numpy stack.

    Uses `setdefault` so user-provided env vars win.
    Must be called before importing numpy/scikit-learn for best effect.
    """

    v = str(int(n_threads))
    os.environ.setdefault("OMP_NUM_THREADS", v)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", v)
    os.environ.setdefault("MKL_NUM_THREADS", v)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", v)

