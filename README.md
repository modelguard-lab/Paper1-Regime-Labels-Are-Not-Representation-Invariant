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
  - **Representations (6)**:
    - `rep_a` (standardized baseline)
    - `rep_a_unscaled` (no standardization; preprocessing ablation)
    - `rep_b` (alternative feature family)
    - `rep_c1` (longer feature horizons)
    - `rep_c2` (feature subset perturbation)
    - `rep_c3` (alternative baseline feature subset)
  - **Model**: HMM vs GMM
  - **Seeds**: 1, 2

**Representations**:

- 6 reps (`rep_a`, `rep_a_unscaled`, `rep_b`, `rep_c1`, `rep_c2`, `rep_c3`)
- Default: rolling z-score standardization; `rep_a_unscaled` is the only unstandardized control.

Data requirement: `data/` CSVs (e.g. `GSPC.csv`, `IEF.csv`). If a CSV is missing,
`run.py` will automatically download it via yfinance at startup.

## Multi-asset usage (optional)

In `config.yaml`:

- Set `assets` to a list of assets (e.g. `["^GSPC", "QQQ", "IWM"]`).
- If you provide multiple assets, each asset is run independently and outputs are saved under
  `outputs/<asset>/`.
  - Common tickers: gold = `GLD`, bitcoin = `BTC-USD`.

## Outputs

- **Per-asset outputs**: `outputs/<asset>/`
  - **Results**: `outputs/<asset>/results/`
  - **Plots**: `outputs/<asset>/plots/`

## One-file exports (requested)

- **Per-asset results data**: `outputs/<asset>/key_results.csv`
- **Per-asset results analysis**: `outputs/<asset>/analysis.md`
- **All-assets combined (when assets > 1)**:
  - `outputs/key_results_all_assets.csv`
  - `outputs/analysis_all_assets.md`

## What to report (paper-ready)

### Core thesis evidence (states are not invariant)

- **Cross-representation agreement** (ARI/NMI): show low/moderate agreement across “reasonable” representations.
- **Temporal stability** (consecutive-window ARI/NMI): show drift over time (well below 1.0).
- **Semantic drift**: show that state meanings (risk profiles) shift across representations.

### Preprocessing ablation (rep_a_unscaled)

Report `rep_a` vs `rep_a_unscaled`:

- Cross-rep ARI/NMI (same features, same windows, only scaling differs)
- Any systematic shift in semantic profiles

Interpretation template:
> “Even holding the feature set fixed, standardization changes the inferred state structure, reinforcing that ‘market states’ are representation-dependent.”

### Fit diagnostics (non-central, optional)

Use `plots/scores_summary.csv` and `results/run.log` as diagnostics (not the main claim):

- Failure rate / logged exceptions
- LogLik/AIC/BIC distributions by model (HMM vs GMM)

**If your machine freezes or BSODs**: The runs are CPU- and memory-heavy. Default is `n_jobs: 12`.
If it still crashes, set `grid.n_jobs: 1` or `grid.n_jobs: 2` in
`config.yaml` to reduce load.

**Note**: Per-asset outputs are written under `outputs/{asset}/`.

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this code, please cite the paper and/or this repository.
See [CITATION.cff](CITATION.cff).
