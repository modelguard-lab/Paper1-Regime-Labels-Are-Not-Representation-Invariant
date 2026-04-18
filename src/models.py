from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class ModelResult:
    states_hard: pd.Series
    states_soft: pd.DataFrame
    model_params: Dict[str, Any]
    scores: Dict[str, float]


def _drop_na_rows(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    mask = ~X.isna().any(axis=1)
    return X.loc[mask], X.index[mask]


def _align_outputs(
    index: pd.Index,
    valid_index: pd.Index,
    hard: np.ndarray,
    soft: np.ndarray,
    n_states: int,
) -> tuple[pd.Series, pd.DataFrame]:
    hard_series = pd.Series(index=index, dtype=float)
    hard_series.loc[valid_index] = hard
    hard_series = hard_series.astype("Int64")

    soft_df = pd.DataFrame(
        data=np.nan,
        index=index,
        columns=[f"state_{k}" for k in range(n_states)],
    )
    soft_df.loc[valid_index] = soft
    return hard_series, soft_df


def _num_hmm_params(model: GaussianHMM, n_features: int) -> int:
    k = model.n_components
    cov_type = model.covariance_type
    startprob = k - 1
    transmat = k * (k - 1)
    means = k * n_features
    if cov_type == "full":
        covars = k * n_features * (n_features + 1) // 2
    elif cov_type == "diag":
        covars = k * n_features
    elif cov_type == "spherical":
        covars = k
    elif cov_type == "tied":
        covars = n_features * (n_features + 1) // 2
    else:
        covars = k * n_features
    return int(startprob + transmat + means + covars)


def fit_hmm(
    X: pd.DataFrame,
    n_states: int,
    covariance_type: str = "full",
    n_iter: int = 200,
    min_covar: float = 1e-4,
    random_state: int | None = 0,
) -> ModelResult:
    X_clean, valid_index = _drop_na_rows(X)
    if X_clean.empty:
        raise ValueError("Input X has no valid rows after dropping NaNs.")

    def _fit(model: GaussianHMM) -> GaussianHMM:
        # hmmlearn can be noisy on stderr/stdout; silence it.
        with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(
            fnull
        ), contextlib.redirect_stderr(fnull):
            model.fit(X_clean.values)
        return model

    requested_covariance_type = str(covariance_type)
    hmm_diag_fallback = False
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=requested_covariance_type,
        n_iter=n_iter,
        min_covar=min_covar,
        random_state=random_state,
    )
    try:
        model = _fit(model)
    except ValueError:
        hmm_diag_fallback = True
        model = _fit(
            GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=n_iter,
                min_covar=min_covar,
                random_state=random_state,
            )
        )

    try:
        loglik = float(model.score(X_clean.values))
        hard = model.predict(X_clean.values)
        soft = model.predict_proba(X_clean.values)
    except ValueError:
        hmm_diag_fallback = True
        model = _fit(
            GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=n_iter,
                min_covar=min_covar,
                random_state=random_state,
            )
        )
        loglik = float(model.score(X_clean.values))
        hard = model.predict(X_clean.values)
        soft = model.predict_proba(X_clean.values)

    hard_series, soft_df = _align_outputs(X.index, valid_index, hard, soft, n_states)
    n_params = _num_hmm_params(model, X_clean.shape[1])
    n_samples = X_clean.shape[0]
    aic = 2 * n_params - 2 * loglik
    bic = np.log(n_samples) * n_params - 2 * loglik

    model_params = {
        "startprob": model.startprob_.tolist(),
        "transmat": model.transmat_.tolist(),
        "means": model.means_.tolist(),
        "covars": model.covars_.tolist(),
        "covariance_type": model.covariance_type,
        "n_components": model.n_components,
        "requested_covariance_type": requested_covariance_type,
        "hmm_diag_fallback": bool(hmm_diag_fallback),
    }
    scores = {
        "loglik": loglik,
        "aic": float(aic),
        "bic": float(bic),
        "hmm_diag_fallback": float(1.0 if hmm_diag_fallback else 0.0),
    }
    return ModelResult(hard_series, soft_df, model_params, scores)


def fit_gmm(
    X: pd.DataFrame,
    n_states: int,
    covariance_type: str = "full",
    n_init: int = 5,
    random_state: int | None = 0,
) -> ModelResult:
    X_clean, valid_index = _drop_na_rows(X)
    if X_clean.empty:
        raise ValueError("Input X has no valid rows after dropping NaNs.")

    model = GaussianMixture(
        n_components=n_states,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=random_state,
    )
    model.fit(X_clean.values)
    hard = model.predict(X_clean.values)
    soft = model.predict_proba(X_clean.values)

    hard_series, soft_df = _align_outputs(X.index, valid_index, hard, soft, n_states)
    scores = {
        "loglik": float(model.score(X_clean.values) * X_clean.shape[0]),
        "aic": float(model.aic(X_clean.values)),
        "bic": float(model.bic(X_clean.values)),
    }
    model_params = {
        "weights": model.weights_.tolist(),
        "means": model.means_.tolist(),
        "covars": model.covariances_.tolist(),
        "covariance_type": model.covariance_type,
        "n_components": model.n_components,
    }
    return ModelResult(hard_series, soft_df, model_params, scores)

