"""
Verify that committed synthetic_groundtruth_psweep_summary.csv is the
mean / SE / count aggregation of the raw synthetic_groundtruth_psweep.csv,
to within numerical tolerance.

This guards against drift between the per-seed log and its summary,
which is the only source the manuscript Tables read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
_RAW = _REPO / "outputs" / "synthetic_groundtruth_psweep.csv"
_SUMMARY = _REPO / "outputs" / "synthetic_groundtruth_psweep_summary.csv"


@pytest.mark.skipif(
    not (_RAW.exists() and _SUMMARY.exists()),
    reason="synthetic psweep outputs not present in this clone",
)
def test_psweep_summary_matches_raw() -> None:
    raw = pd.read_csv(_RAW)
    summary = pd.read_csv(_SUMMARY)

    grouped = raw.groupby(["persistence_p", "model"], sort=True)

    derived = grouped.agg(
        ari_truth_mean=("ari_vs_truth", "mean"),
        ari_truth_se=("ari_vs_truth", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        ari_cross_mean=("ari_cross_rep_mean", "mean"),
        ari_cross_se=("ari_cross_rep_mean", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        n=("ari_vs_truth", "count"),
    ).reset_index()

    summary_sorted = summary.sort_values(["persistence_p", "model"]).reset_index(drop=True)
    derived_sorted = derived.sort_values(["persistence_p", "model"]).reset_index(drop=True)

    # Schema match.
    assert list(summary_sorted["persistence_p"]) == list(derived_sorted["persistence_p"])
    assert list(summary_sorted["model"]) == list(derived_sorted["model"])
    assert list(summary_sorted["n"]) == list(derived_sorted["n"])

    # Numeric agreement within reasonable tolerance.
    for col in ("ari_truth_mean", "ari_truth_se", "ari_cross_mean", "ari_cross_se"):
        np.testing.assert_allclose(
            summary_sorted[col].to_numpy(),
            derived_sorted[col].to_numpy(),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"summary[{col}] disagrees with raw aggregation",
        )
