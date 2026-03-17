"""
Paper 1 experiment runner.

This entrypoint is intentionally thin: all Paper1 logic lives under
`src/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local package imports (src layout) work when invoked from repo root.
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime import configure_console_logging, set_thread_env_defaults

# Limit threads per process early (before numpy/scikit-learn imports).
configure_console_logging()
set_thread_env_defaults(1)

from runner import run


if __name__ == "__main__":
    config_path = PROJECT_DIR / "config.yaml"
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        config_path = p if p.is_absolute() else PROJECT_DIR / p
    run(config_path)

