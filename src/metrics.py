from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy as _scipy_entropy
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StabilityScores:
    ari: float
    nmi: float
    ami: float
    vi: float


def variation_of_information(a: np.ndarray, b: np.ndarray) -> float:
    """Variation of Information = H(X|Y) + H(Y|X), in nats."""
    if len(a) == 0:
        return float("nan")
    mi = float(mutual_info_score(a, b))
    _, ca = np.unique(a, return_counts=True)
    _, cb = np.unique(b, return_counts=True)
    h_a = float(_scipy_entropy(ca))  # natural log, nats
    h_b = float(_scipy_entropy(cb))
    return h_a + h_b - 2.0 * mi


def align_states(reference: np.ndarray, target: np.ndarray, n_states: int) -> np.ndarray:
    """Align target labels to reference via maximum agreement matching."""
    conf = np.zeros((n_states, n_states), dtype=int)
    for i in range(n_states):
        for j in range(n_states):
            conf[i, j] = int(np.sum((reference == i) & (target == j)))
    row_ind, col_ind = linear_sum_assignment(conf.max() - conf)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(int(s), int(s)) for s in target], dtype=int)


def stability_metrics(labels_a: pd.Series, labels_b: pd.Series, n_states: int) -> StabilityScores:
    """
    Compute ARI/NMI after aligning labels_b to labels_a.

    Important: we align by **time index**, and we must be robust to duplicate indices
    that can arise from imperfect merges of per-window exports. Using index intersection
    + `.loc[...]` can produce length mismatches when one side has duplicate timestamps.
    """
    if not isinstance(labels_a, pd.Series):
        labels_a = pd.Series(labels_a)
    if not isinstance(labels_b, pd.Series):
        labels_b = pd.Series(labels_b)

    # Robust alignment: inner-join on index using pandas alignment semantics.
    ab = pd.concat([labels_a, labels_b], axis=1, join="inner")
    if ab.empty:
        logger.warning("No common index between state sequences for stability_metrics")
        return StabilityScores(ari=float("nan"), nmi=float("nan"), ami=float("nan"), vi=float("nan"))
    ab = ab.dropna()
    if ab.empty:
        logger.warning("No non-NA overlap between state sequences for stability_metrics")
        return StabilityScores(ari=float("nan"), nmi=float("nan"), ami=float("nan"), vi=float("nan"))

    a = ab.iloc[:, 0].astype(int).to_numpy()
    b = ab.iloc[:, 1].astype(int).to_numpy()
    if a.shape[0] != b.shape[0]:
        # Should be impossible after concat alignment, but keep a guardrail.
        logger.warning("Aligned arrays have different lengths (a=%d, b=%d); truncating.", a.shape[0], b.shape[0])
        n = int(min(a.shape[0], b.shape[0]))
        a = a[:n]
        b = b[:n]

    return StabilityScores(
        ari=float(adjusted_rand_score(a, b)),
        nmi=float(normalized_mutual_info_score(a, b)),
        ami=float(adjusted_mutual_info_score(a, b)),
        vi=float(variation_of_information(a, b)),
    )


def semantic_drift(features: pd.DataFrame, states: pd.Series, feature_cols: Iterable[str]) -> pd.Series:
    """
    Measure within-window semantic drift of state risk profiles.

    For a given rolling window (features + inferred states), split the time index
    into two consecutive halves and, for each state, compute the mean absolute
    change in the feature-mean vector between the first half and the second half
    (using only timestamps assigned to that state in each half).

    Returns a Series indexed by `state_<k>`; aggregate drift summaries can be
    computed by taking mean/std over this Series.
    """

    if features.empty or states.dropna().empty:
        return pd.Series(dtype=float, name="semantic_drift")

    cols = list(feature_cols)
    idx = states.dropna().index.intersection(features.index)
    if len(idx) == 0:
        return pd.Series(dtype=float, name="semantic_drift")

    df = features.loc[idx, cols].sort_index()
    z = states.loc[df.index].dropna().astype(int)
    if df.empty or z.empty:
        return pd.Series(dtype=float, name="semantic_drift")

    # Split window into two consecutive halves (by time order).
    mid = int(len(df) // 2)
    if mid <= 0 or mid >= len(df):
        return pd.Series(dtype=float, name="semantic_drift")
    first_idx = df.index[:mid]
    second_idx = df.index[mid:]

    out: dict[str, float] = {}
    for s in sorted(set(int(x) for x in z.dropna().unique())):
        s_idx = z[z == s].index
        a_idx = s_idx.intersection(first_idx)
        b_idx = s_idx.intersection(second_idx)
        if len(a_idx) == 0 or len(b_idx) == 0:
            out[f"state_{s}"] = float("nan")
            continue
        m1 = df.loc[a_idx].mean()
        m2 = df.loc[b_idx].mean()
        out[f"state_{s}"] = float((m2 - m1).abs().mean())

    return pd.Series(out, name="semantic_drift", dtype=float)

