"""
Parallelism helper with sequential fallback for the analysis pipeline.

Wraps joblib.Parallel with a TerminatedWorkerError catch (Windows + MKL
forks). Used by every stability / ordering / fit batch in the pipeline.
"""

from __future__ import annotations

import logging

from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError

logger = logging.getLogger(__name__)


def _parallel_or_sequential(func, args_list, n_jobs, label="parallel"):
    """Run func(*args) for each args in args_list via joblib, with sequential fallback.

    Automatically falls back to sequential when:
    - n_jobs <= 1 or only 1 task
    - A TerminatedWorkerError occurs (MKL/loky instability on Windows)

    Logs detailed diagnostics on failure to aid debugging.

    Root cause of TerminatedWorkerError on Windows:
    numpy/scipy link to Intel MKL which has known thread-safety bugs when
    forked via loky.  Setting MKL_NUM_THREADS=1 in worker init prevents most
    segfaults.  If it still occurs, we fall back to sequential.
    """
    import time as _t
    import traceback as _tb
    t0 = _t.perf_counter()
    n_tasks = len(args_list)
    # Analysis tasks pass large per-task dicts (hard_map shards) that must be pickled once
    # per worker.  At n_jobs > ~8, pickle overhead + concurrent MKL init on Windows
    # triggers TerminatedWorkerError.  Cap at 8 for stability; each task takes seconds,
    # so 8-way parallelism is sufficient (5x speedup vs sequential, which is what matters).
    analysis_cap = 8
    effective = min(n_jobs if n_jobs > 0 else 4, n_tasks, analysis_cap)

    if effective <= 1 or n_tasks <= 2:
        results = [func(*a) for a in args_list]
        logger.info("%s: done in %.1fs (sequential, %d tasks)", label, _t.perf_counter() - t0, n_tasks)
        return results

    logger.info("%s: starting parallel; %d tasks, n_jobs=%d (backend=loky)", label, n_tasks, effective)
    try:
        # loky with mmap of large args + inner_max_num_threads=1 avoids MKL crashes.
        from joblib import parallel_config as _parallel_config
        with _parallel_config(backend="loky", n_jobs=effective, inner_max_num_threads=1):
            results = Parallel(max_nbytes="50M", mmap_mode="r")(
                delayed(func)(*a) for a in args_list
            )
        logger.info("%s: done in %.1fs (parallel loky, %d tasks, n_jobs=%d)", label, _t.perf_counter() - t0, n_tasks, effective)
        return results
    except TerminatedWorkerError:
        # Halve workers and retry once with loky before falling through to threading.
        retry_jobs = max(2, effective // 2)
        logger.warning("%s: TerminatedWorkerError at n_jobs=%d; retrying with n_jobs=%d.", label, effective, retry_jobs)
        t0 = _t.perf_counter()
        try:
            from joblib import parallel_config as _parallel_config
            with _parallel_config(backend="loky", n_jobs=retry_jobs, inner_max_num_threads=1):
                results = Parallel(max_nbytes="50M", mmap_mode="r")(
                    delayed(func)(*a) for a in args_list
                )
            logger.info("%s: done in %.1fs (loky retry at n_jobs=%d, %d tasks)", label, _t.perf_counter() - t0, retry_jobs, n_tasks)
            return results
        except Exception:
            logger.warning("%s: loky retry also failed; falling back to threading.", label)
        t0 = _t.perf_counter()
        try:
            results = Parallel(n_jobs=effective, backend="threading")(
                delayed(func)(*a) for a in args_list
            )
            logger.info("%s: done in %.1fs (threading fallback, %d tasks)", label, _t.perf_counter() - t0, n_tasks)
            return results
        except Exception:
            pass
        t0 = _t.perf_counter()
        results = [func(*a) for a in args_list]
        logger.info("%s: done in %.1fs (sequential final fallback, %d tasks)", label, _t.perf_counter() - t0, n_tasks)
        return results
    except Exception as e:
        logger.error(
            "%s: parallel error (%s); falling back to sequential.\n%s",
            label, e, _tb.format_exc(),
        )
        t0 = _t.perf_counter()
        results = [func(*a) for a in args_list]
        logger.info("%s: done in %.1fs (sequential fallback, %d tasks)", label, _t.perf_counter() - t0, n_tasks)
        return results

