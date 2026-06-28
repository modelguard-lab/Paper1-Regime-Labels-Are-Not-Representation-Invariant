"""
Smoke tests for src.core.fits.

These tests catch import-level breakage (e.g. missing imports for os, json,
semantic_drift, configure_global_file_logging, _rmtree_with_retries) and
confirm that the per-slice fit primitives return a successful result on
synthetic data. Without this coverage, the kind of NameError that makes
every fit silently fail (broad except + ok=False return) can ship to a
fresh clone undetected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.fits import _fit_slice_collect, _fit_slice_write_shard


def _make_synthetic_panel(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Two clear regimes by mean to make any reasonable fit converge.
    half = n // 2
    a = rng.normal(loc=0.0, scale=1.0, size=(half, 2))
    b = rng.normal(loc=3.0, scale=1.0, size=(n - half, 2))
    X = np.vstack([a, b])
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(X, index=idx, columns=["f0", "f1"])


@pytest.mark.parametrize("model_name", ["gmm", "hmm"])
def test_fit_slice_collect_returns_ok(model_name: str) -> None:
    panel = _make_synthetic_panel(n=200, seed=0)
    model_cfg = {
        "gmm": {"covariance_type": "full", "n_init": 1},
        "hmm": {"covariance_type": "full", "n_iter": 50},
    }
    result = _fit_slice_collect(
        model_name=model_name,
        X_values=panel.values,
        X_index_values=panel.index.values,
        X_columns=list(panel.columns),
        start=0,
        end=len(panel),
        k=2,
        seed=0,
        rep_name="rep_smoke",
        roll="roll_0",
        w=len(panel),
        model_cfg=model_cfg,
    )
    assert result["ok"] is True
    assert result["model"] == model_name
    assert result["K"] == 2
    assert "scores" in result and isinstance(result["scores"], dict)
    assert "semantic_drift_mean" in result["scores"]


@pytest.mark.parametrize("model_name", ["gmm", "hmm"])
def test_fit_slice_write_shard_writes_files(model_name: str, tmp_path: Path) -> None:
    panel = _make_synthetic_panel(n=200, seed=1)
    model_cfg = {
        "gmm": {"covariance_type": "full", "n_init": 1},
        "hmm": {"covariance_type": "full", "n_iter": 50},
    }
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    log_path = tmp_path / "run.log"
    ok = _fit_slice_write_shard(
        shard_dir=shard_dir,
        log_path=str(log_path),
        asset="SMOKE",
        model_name=model_name,
        X_values=panel.values,
        X_index_values=panel.index.values,
        X_columns=list(panel.columns),
        start=0,
        end=len(panel),
        k=2,
        seed=0,
        rep_name="rep_smoke",
        roll="roll_0",
        w=len(panel),
        model_cfg=model_cfg,
    )
    assert ok is True
    # At least the hard-states and scores shards must exist.
    hard_files = list(shard_dir.glob(f"states_hard_{model_name}_*.csv"))
    score_files = list(shard_dir.glob(f"scores_{model_name}_*.csv"))
    assert hard_files, "states_hard shard not written"
    assert score_files, "scores shard not written"
