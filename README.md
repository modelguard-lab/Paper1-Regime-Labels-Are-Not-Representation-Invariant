# Paper 1: Representation Dependence of Market State Identification

**Question**: Can market states be identified as stable statistical objects?

## Scope and disclaimer

**This tool does not certify regulatory compliance.**  
It provides diagnostic evidence on representation-related model instability to support validation, documentation, and governance review.

**Tagline (for open-source / model-validator):** *Detect when regime models appear stable but are not trustworthy.*

*(When this codebase is released as the model-validator CLI, the same disclaimer will appear in the tool README and in every generated report.)*

## Installation

- Python 3.10+ recommended

```bash
pip install -r requirements.txt
```

## Run

Use the unified runner:

```bash
python run.py
```

**Implementation note**: Code is self-contained under `src/`.

## Experiment definition

- **Fixed**:
  - Rolling window alignment: window \(=252\), step \(=21\) (monthly update cadence)
  - \(K=3\) (fixed for cross-representation label matching)
- **Varied**:
  - **Representations (8)**:
    - `rep_a` (standardized baseline)
    - `rep_a_unscaled` (no standardization; preprocessing ablation)
    - `rep_b` (alternative feature family)
    - `rep_c1` (longer feature horizons)
    - `rep_c2` (feature subset perturbation)
    - `rep_c3` (alternative baseline feature subset)
    - `rep_d` (GARCH(1,1) conditional volatility; heterogeneous vol estimator)
    - `rep_e` (VIX-family implied volatility; ^GSPC only, gated by `asset_filter`)
  - **Model**: HMM vs GMM
  - **Seeds**: 1..20 (20 seeds; see `config.yaml:grid.seeds`)

**Representations**:

- 8 reps defined in `config.yaml`: `rep_a`, `rep_a_unscaled`, `rep_b`, `rep_c1`, `rep_c2`, `rep_c3`, `rep_d`, `rep_e`.
- `rep_e` is restricted to `^GSPC` via `asset_filter` (VIX is the S&P 500 implied-vol index), so non-S&P assets run with 7 reps.
- Default: rolling z-score standardization; `rep_a_unscaled` is the only unstandardized control.

Data requirement: `data/` CSVs (e.g. `GSPC.csv`, `IEF.csv`). If a CSV is missing,
`run.py` will automatically download it via yfinance at startup.
For reproducibility, sample bounds are configured in `config.yaml` under `data.start_date` and `data.end_date`.

## Multi-asset usage (optional)

In `config.yaml`:

- Set `assets` to a list of assets (e.g. `["^GSPC", "QQQ", "IWM"]`).
- If you provide multiple assets, each asset is run independently and outputs are saved under
  `outputs/<asset>/`.
  - Common tickers: gold = `GLD`, bitcoin = `BTC-USD`.

## Outputs

Per-asset step outputs are written under `outputs/<asset>/step_<N>/`:

- `outputs/<asset>/step_<N>/results/`: per-rep CSV files (hard states, stability metrics, etc.)
- `outputs/<asset>/step_<N>/plots/`: figures for that step configuration
- `outputs/<asset>/step_<N>/key_results.csv`: aggregated metrics for this (asset, step)
- `outputs/<asset>/step_<N>/analysis.md`: human-readable summary

Robustness (K-sweep) outputs are written under `outputs/<asset>/robustness/K_<N>/` with the same sub-layout.

**Multi-asset rollup**: `outputs/key_results_all_assets.csv` combines all assets.

**Paper-relevant CSVs** (written to `outputs/` root by post-hoc scripts):

| File | Content |
| ---- | ------- |
| `step_sweep_summary.csv` | ARI/NMI vs step size |
| `robustness_ci_summary.csv` | K-state robustness confidence intervals |
| `var_spread_summary.csv` | VaR spread across representations |
| `rank_aligned_ordering_summary.csv` | Rank-aligned ordering metrics (Jaccard, Spearman) |
| `kmeans_robustness_summary.csv` | K-means vs GMM/HMM cross-rep ARI |
| `repr_decomp_summary.csv` | Representation-dimension decomposition |
| `synthetic_groundtruth.csv` / `synthetic_groundtruth_psweep.csv` / `synthetic_groundtruth_ksweep.csv` | Synthetic ground-truth recovery results |

## What to report (paper-ready)

### Core thesis evidence (states are not invariant)

- **Cross-representation agreement** (ARI/NMI): show low/moderate agreement across “reasonable” representations.
- **Temporal stability** (consecutive-window ARI/NMI on disjoint segments): show drift over time without overlap inflation.
- **Semantic drift**: show that state meanings (risk profiles) shift across representations.

### Preprocessing ablation (rep_a_unscaled)

Report `rep_a` vs `rep_a_unscaled`:

- Cross-rep ARI/NMI (same features, same windows, only scaling differs)
- Any systematic shift in semantic profiles

Interpretation template:
> “Even holding the feature set fixed, standardization changes the inferred state structure, reinforcing that ‘market states’ are representation-dependent.”

### Fit diagnostics (non-central, optional)

Use `outputs/<asset>/step_<N>/plots/scores_summary.csv` and `outputs/run.log` as diagnostics (not the main claim):

- Failure rate / logged exceptions
- LogLik/AIC/BIC distributions by model (HMM vs GMM)

**If your machine freezes or BSODs**: The runs are CPU- and memory-heavy. Default is `n_jobs: 24`.
If it still crashes, set `grid.n_jobs: 1` or `grid.n_jobs: 2` in
`config.yaml` to reduce load.

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this code, please cite the paper and/or this repository.
See [CITATION.cff](CITATION.cff).
