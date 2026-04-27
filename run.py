"""
Paper 1 -- Regime Labels Are Not Representation-Invariant: unified entry point.

Usage
-----
  python run.py pipeline              # Full experiment pipeline (default)
  python run.py posthoc_figs          # Post-hoc figure generation
  python run.py posthoc_ami           # Post-hoc AMI/VI permutation analysis
  python run.py posthoc_synthetic     # Post-hoc synthetic ground-truth
  python run.py posthoc_var_spread    # Post-hoc variance-spread analysis
  python run.py paper_autofill        # Auto-fill paper tables from results
  python run.py aggregate             # Aggregate results across assets
  python run.py all                   # Pipeline + all post-hoc analyses;
                                      #   tees stdout/stderr to
                                      #   outputs/run.log (incl. joblib
                                      #   worker output).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Paper 1 modules use bare imports (from data import ...) that rely on
# src/ being on sys.path.  Keep this for backward compatibility.
_SRC_DIR = PROJECT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from src.commands.cli_registry import COMMANDS, run_module_command

ALL_EXPERIMENT_COMMANDS: tuple[str, ...] = (
    "pipeline",
    "posthoc_figs",
    "posthoc_ami",
    "posthoc_synthetic",
    "posthoc_var_spread",
    "paper_autofill",
    "aggregate",
)


def _print_help() -> None:
    text = (__doc__ or "").strip()
    try:
        print(text)
    except UnicodeEncodeError:
        safe = text.encode("ascii", errors="replace").decode("ascii")
        print(safe)


def _run_pipeline() -> None:
    from src.core.runtime import configure_console_logging, set_thread_env_defaults
    configure_console_logging()
    set_thread_env_defaults(1)
    from src.runner import run
    config_path = PROJECT_DIR / "config.yaml"
    run(config_path)


def _run_module(name: str) -> None:
    from src.core.runtime import configure_console_logging, set_thread_env_defaults
    configure_console_logging()
    # Post-hoc modules use joblib/loky for pair-level parallelism. On Windows,
    # MKL segfaults inside forked workers unless thread counts are pinned to 1
    # *before* numpy/scipy import in the workers. Set this in the parent so
    # loky inherits the env.
    set_thread_env_defaults(1)
    run_module_command(name)


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else "pipeline"

    if cmd in ("-h", "--help", "help"):
        _print_help()
        return

    if len(args) > 1 and args[1] in ("-h", "--help", "help"):
        if cmd in COMMANDS or cmd in {"pipeline", "all"}:
            _print_help()
            return
        raise SystemExit(f"Unknown command: {cmd}. Use --help to list commands.")

    if cmd == "all":
        from src.core.runtime import (
            configure_console_logging,
            set_thread_env_defaults,
            tee_console_to_file,
        )
        outputs_dir = PROJECT_DIR / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        log_path = outputs_dir / "run.log"
        # Start fresh per invocation so the log reflects the current run only.
        log_path.unlink(missing_ok=True)
        # Tee BEFORE the console handler so logger records, raw prints, and
        # joblib/loky worker output all reach the file.
        tee_console_to_file(log_path)
        configure_console_logging()
        set_thread_env_defaults(1)
        import logging
        _logger = logging.getLogger("run")
        _logger.info("Log file: %s", log_path)
        _run_pipeline()
        for name in ALL_EXPERIMENT_COMMANDS:
            if name == "pipeline":
                continue
            if name not in COMMANDS:
                _logger.warning("Skip missing command: %s", name)
                continue
            _logger.info("=" * 60)
            _logger.info("  %s", name)
            _logger.info("=" * 60)
            _run_module(name)
        _logger.info("ALL_DONE")
        return

    if cmd == "pipeline":
        _run_pipeline()
        return

    if cmd in COMMANDS:
        _run_module(cmd)
        return

    raise SystemExit(f"Unknown command: {cmd}. Use --help to list commands.")


if __name__ == "__main__":
    main()
