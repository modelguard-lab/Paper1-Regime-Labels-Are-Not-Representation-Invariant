"""
Post-hoc convergence audit for HMM fits.

`hmmlearn` emits a "Model is not converging" warning whenever the EM log-
likelihood update for the final iteration is non-positive (or below tol).
Most of these warnings carry a |delta| at or below 1e-6 -- i.e. the EM has
effectively converged but failed the strict monotonicity check at machine
precision. A non-trivial subset, however, terminates with |delta| > 1e-3,
indicating the fit truly did not stabilise within the iteration budget.

This script parses ``outputs/run.log`` (or any path passed on the CLI), counts
warnings by |delta| magnitude bucket, and writes a small summary CSV that the
paper appendix references. It does not re-fit anything.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DELTA_RE = re.compile(r"Delta is (-?[0-9.eE+-]+)")
THRESHOLDS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def parse_deltas(log_path: Path) -> np.ndarray:
    deltas: list[float] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "not converging" not in line:
                continue
            m = DELTA_RE.search(line)
            if not m:
                continue
            try:
                deltas.append(abs(float(m.group(1))))
            except ValueError:
                continue
    return np.asarray(deltas, dtype=float)


def count_total_hmm_fits(outputs_dir: Path) -> int:
    """Sum unique (K, W, seed, roll) groups across every windows_states_hard_hmm.csv.

    Each group corresponds to exactly one HMM fit; the total is the denominator
    against which the warning count is meaningful.
    """
    total = 0
    for p in outputs_dir.rglob("windows_states_hard_hmm.csv"):
        try:
            df = pd.read_csv(p, usecols=["K", "W", "seed", "roll"], low_memory=False)
        except Exception:
            continue
        if df.empty:
            continue
        total += df.drop_duplicates(["K", "W", "seed", "roll"]).shape[0]
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=ROOT / "outputs" / "run.log")
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "hmm_convergence_audit.csv")
    args = parser.parse_args()

    if not args.log.exists():
        logger.error("run.log not found: %s", args.log)
        raise SystemExit(1)

    deltas = parse_deltas(args.log)
    if deltas.size == 0:
        logger.warning("No 'not converging' warnings found in %s", args.log)

    total_fits = count_total_hmm_fits(args.outputs_dir)
    rows = [
        {
            "metric": "n_hmm_fits_total",
            "value": int(total_fits),
            "share_of_fits": 1.0 if total_fits > 0 else float("nan"),
        },
        {
            "metric": "n_warnings_total",
            "value": int(deltas.size),
            "share_of_fits": float(deltas.size / total_fits) if total_fits > 0 else float("nan"),
        },
    ]
    if deltas.size > 0:
        rows.extend([
            {"metric": "delta_median", "value": float(np.median(deltas)), "share_of_fits": float("nan")},
            {"metric": "delta_p95", "value": float(np.percentile(deltas, 95)), "share_of_fits": float("nan")},
            {"metric": "delta_max", "value": float(deltas.max()), "share_of_fits": float("nan")},
        ])
        for thresh in THRESHOLDS:
            n_above = int(np.sum(deltas > thresh))
            rows.append({
                "metric": f"n_warnings_abs_delta_gt_{thresh:g}",
                "value": n_above,
                "share_of_fits": float(n_above / total_fits) if total_fits > 0 else float("nan"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    logger.info("Wrote %s", args.out)

    # Human-readable report.
    print()
    print(f"Total HMM fits in pipeline:        {total_fits:>10,}")
    print(f"Total non-converging warnings:     {deltas.size:>10,}  ({100*deltas.size/max(total_fits,1):.2f}% of fits)")
    if deltas.size > 0:
        print(f"  median |delta|:                  {np.median(deltas):.2e}")
        print(f"  95th percentile |delta|:         {np.percentile(deltas, 95):.2e}")
        print(f"  max |delta|:                     {deltas.max():.2e}")
        print()
        print("|delta| > threshold breakdown (truly non-converged at increasing strictness):")
        for thresh in THRESHOLDS:
            n_above = int(np.sum(deltas > thresh))
            pct_warnings = 100 * n_above / deltas.size
            pct_fits = 100 * n_above / max(total_fits, 1)
            print(f"  > {thresh:>6g}: {n_above:>8,} warnings ({pct_warnings:5.1f}% of warnings; {pct_fits:5.2f}% of fits)")


if __name__ == "__main__":
    main()
