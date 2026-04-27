# Revision Notes — 2026-04-17

Addressed 7 substantive critiques + 11 minor items + 6 self-identified weaknesses.

**Re-run required:** `python run.py` → `python src/posthoc_figs.py` → `python src/posthoc_ami_vi_perm.py` → `python src/posthoc_var_spread.py` → `python src/posthoc_synthetic_groundtruth.py` to regenerate results with the new config (20 baseline seeds, 8 representations: rep_a--rep_d applied to all four assets, rep_e (VIX-family) applied only to ^GSPC via `asset_filter`). `paper_autofill.py` will auto-sync Tables 1 and 3 numbers after re-run; Table 2 (CVaR spread) ^GSPC rows and the line-205 aggregate prose must be patched manually.

---

## Critique 1 — Admissible class too narrow; redundancy unacknowledged

**Reviewer:** The six representations are highly redundant (5/6 share VaR/CVaR, 4/6 share the same windows). This should be acknowledged as strengthening the lower-bound argument. A genuinely heterogeneous representation (e.g. GARCH conditional volatility) should be added.

**Response:**
- Added `rep_d` (GARCH(1,1) conditional volatility + drawdown + VaR/CVaR) — a parametric autoregressive filter that is structurally different from rolling realized volatility.
- Rewrote §2.2 to explicitly enumerate the four independent variation axes and acknowledge the redundancy structure, reframing low ARI among near-redundant specifications as a conservative lower bound.
- `arch>=5.0` added to dependencies.

**Files:** `config.yaml`, `requirements.txt`, `src/features.py`, `src/posthoc_ami_vi_perm.py`, `paper/main.tex`

---

## Critique 2 — Seed stability vs main table precision illusion

**Reviewer:** Table 2 reports 3-decimal precision from 2 seeds, creating a false sense of accuracy. The 20-seed robustness results (Table 4) should be the main table.

**Response:**
- Promoted baseline seeds from [1,2] to [1..20] in the config.
- Table 2 rewritten: 2-column format (Cross-rep ARI, Temporal ARI) with mean ± 95% CI from 20 seeds. AMI moved to table notes.
- §4.3 renamed from "Robustness" to "State-number sensitivity" — its K-sweep role is now primary, not the seed-robustness check.
- Added seed-level CI computation (`cross_rep_ari_seed_mean/ci95`, `temporal_ari_seed_mean/ci95`) to key_results.csv export.
- `paper_autofill.py` updated for the new table format and auto-sync.

**Files:** `config.yaml`, `src/runner.py`, `src/paper_autofill.py`, `paper/main.tex`

---

## Critique 3 — Permutation test pooling ignores dependence

**Reviewer:** Pooling exceedance counts across non-independent pairs understates p-values. Report per-pair p-values; add dependence caveat; compress the section (the test merely confirms ARI > 0, which is a low bar).

**Response:**
- Added per-pair p-value computation to `posthoc_ami_vi_perm.py` (each pair tested independently; min/median/max exported to key_results.csv).
- §2.5: replaced pooled aggregate description with per-pair tests; added dependence caveat ("pairs constructed from the same asset share underlying data"); added sentence acknowledging the test is intentionally weak.
- §4.1 and Table 2 notes: switched from "aggregate p < 0.019" to "all individual pairs p < 0.05".

**Files:** `src/posthoc_ami_vi_perm.py`, `src/paper_autofill.py`, `paper/main.tex`

---

## Critique 4 — Disjoint temporal metric semantics unclear

**Reviewer:** The disjoint metric at step=252 compares labels from independently estimated models separated by one year. Low ARI there could reflect estimation instability rather than low regime persistence. A traditional overlap-based metric should be reported as a comparator.

**Response:**
- §2.5: added a dedicated paragraph ("Temporal agreement: two complementary metrics") defining both the overlap metric (upper bound, but contaminated) and the disjoint metric (lower bound, but conflates persistence with estimation stability at large steps).
- Table 3: added "Temporal ARI (overlap)" column with real values from existing outputs (0.577 at step=21 vs disjoint 0.384; gap = 0.19 quantifies overlap inflation).
- §4.2: rewritten to discuss both metrics, quantify the inflation gap, and add an explicit caveat for the step=252 interpretation.

**Files:** `src/paper_autofill.py`, `src/posthoc_ami_vi_perm.py`, `paper/main.tex`

---

## Critique 5 — Figure 1 legend mismatch; limited quantification

**Reviewer:** "Layer A (returns)" / "Layer B (risk)" is misleading — both are risk-based representations. A single qualitative figure for COVID is insufficient; add a quantitative time series.

**Response:**
- Fixed legend labels: "Layer A (returns)" → "rep_a", "Layer B (risk)" → "rep_c1" in both figure variants. Rewrote Figure 1 caption.
- Added `plot_disagreement_timeseries()`: for each trading day, computes the fraction of representation pairs assigning different coarse risk labels. 63-day rolling average, with COVID-19 and 2022 inflation stress periods highlighted.
- Added Figure 2 (`fig_disagreement_timeseries`) to the paper, cross-referenced from Figure 1.

**Files:** `src/plots.py`, `src/posthoc_figs.py`, `paper/main.tex`

---

## Critique 6 — Ordering consistency needs chance baseline

**Reviewer:** Top-1 agreement of 0.91–0.94 and Spearman of 0.886–0.907 appear high, but the Hungarian-matched chance baseline at K=3 may be much higher than 1/K. Compute the null distribution; with only 3 ranks, Spearman has limited discriminative power.

**Response:**
- **Key finding:** permutation null under Hungarian matching at K=3 gives top-1 ≈ 0.953, Spearman ≈ 0.907. The observed values (0.91–0.94 / 0.886–0.907) are **not distinguishable from chance**.
- Added `_ordering_null_distribution()` and `_compute_ordering_null_baseline()` to runner.py.
- Appendix table: added "Chance (null)" row with null values; rewrote preceding paragraph.
- **Major reframing throughout:** Abstract, §4.1, §4.3, and Conclusion all updated to remove the "coarse ordering is more stable" claim. The paper now states that representation uncertainty extends to coarse ordering as well as fine partitions.

**Files:** `src/runner.py`, `paper/main.tex`

---

## Critique 7 — D'Amour positioning; instability vs plurality

**Reviewer:** The paper correctly notes regime identification lacks held-out performance (unlike D'Amour et al.'s underspecification), but doesn't explore the deeper question: does low ARI mean "instability" (a defect) or "multiple equally valid partitions" (epistemic pluralism)? This is the most theoretically interesting tension.

**Response:**
- Introduction: expanded the underspecification paragraph to name the two interpretations explicitly and note their different governance implications (repair vs ensemble reporting).
- Added new §4.4 "Instability or plurality?" — a dedicated discussion subsection that:
  - Restates D'Amour et al.'s framework and why it doesn't apply
  - Defines the two readings precisely
  - Acknowledges the paper cannot distinguish them (requires ground truth or downstream payoffs)
  - Argues the key conclusion holds under either reading: single-partition reporting is inadequate regardless

**Files:** `paper/main.tex` (no code changes)

---

## Minor corrections (11 items)

1. **Abstract:** qualified "cross-rep < temporal" with K-dependence ("$K \in \{2,3\}$... reverses at $K=4$")
2. **§2.3 footnotes:** unified citation style; named `hmmlearn` (v0.2.8+) and `scikit-learn` explicitly with doc URLs
3. **§2.2:** added "realized skewness and volatility-of-volatility" to feature summary (was missing rep_b)
4. **§3:** "post-2005 portion of the global financial crisis" → "the 2007–2009 global financial crisis"
5. **Table 1:** added BTC min/max date footnote (Min: 12 March 2020, COVID liquidity crisis; Max: 7 December 2017)
6. **§4.1:** reminded reader of $(e+1)/(B+1)$ correction for permutation p-values
7. **Table 3:** clarified CI = 95% confidence interval across the four assets ($t_{3,0.975} \times \text{SE}$)
8. **§4.3:** added economic explanation for K=4 GMM reversal (GMM lacks transition dynamics → finer partitions fragile to window shifts)
9. **Conclusion:** added practitioner-actionable recommendation (≥3 specs, 0.65 threshold triggers escalation, ensemble reporting)
10. **§1:** expanded Ampountolas (2025) and Wang et al. (2026) discussion with connections to this paper
11. **§2.4:** fixed ARI formula outer brackets (`\big` → `\Big` for proper rendering)

---

## Self-identified weaknesses — additional fixes (same session)

### Weakness #1 + #3 — Downstream economic impact / pure negative finding

**Problem:** Paper proved labels disagree but never answered "so what?" — no downstream consequence quantified. After ordering null correction, paper became purely negative with no constructive anchor.

**Response:**
- Added `src/posthoc_var_spread.py`: computes regime-conditional CVaR (5% expected shortfall) under each representation for each date; measures cross-representation spread.
- **Key finding:** CVaR spread averages 53–59% of unconditional CVaR across assets/models (range: 37% IEF/GMM to 80% BTC/HMM). During COVID-19 stress for S&P 500, spread reaches 260% of unconditional CVaR.
- Added "Downstream economic impact" paragraph to §4.1 with these numbers.
- Paper is no longer purely negative: "labels disagree *and* this changes risk estimates by 53–80%."

### Weakness #4 — §4.4 lacks empirical support for instability-vs-plurality

**Problem:** The instability-vs-plurality distinction was conceptual only; a synthetic experiment could distinguish the two readings.

**Response:**
- Added `src/posthoc_synthetic_groundtruth.py`: generates data from known 3-state regime-switching DGP (Markov chain, persistence=0.97, 3 vol levels, drift), runs the full representation pipeline, compares inferred labels to ground truth.
- **Key finding:** ARI vs ground truth averages 0.264 (HMM) / 0.273 (GMM) — comparable to cross-rep ARI (0.305 / 0.383). This is **instability, not plurality**: the pipeline fails to recover the true partition even when one exists.
- Updated §4.4 with the synthetic evidence; the discussion now provides "partial resolution" rather than purely conceptual framing.

### Weakness #2 + #6 — N=4 and class homogeneity

**Problem:** N=4 assets limits statistical inference; admissible class remains homogeneous (daily, single-asset, 252-day window).

**Response:** Expanded conclusion limitations paragraph to explicitly state: (1) four assets are case studies, not a statistical sample — no formal cross-asset inference supported; (2) representation class restricted to daily-frequency, single-asset, univariate features with fixed estimation window.

### Weakness #5 — Some representations might be better

**Problem:** Paper treats all representations as equally valid, but a domain expert might privilege one.

**Response:** Partially addressed by the synthetic experiment: the experiment framework can rank representations by ARI vs ground truth. In the synthetic setting, no representation is clearly superior (all have ARI ≈ 0.26–0.27), but the framework is extensible. Not further pursued for real data (would require external ground truth like NBER dates — out of scope for this paper).

---

## Files modified (summary)

| File | Changes |
|------|---------|
| `config.yaml` | rep_d added; baseline seeds → 20 |
| `requirements.txt` | arch>=5.0 |
| `src/features.py` | `_compute_garch_vol()`, wired into `build_representation_single` |
| `src/runner.py` | seed-level CIs, `_ordering_null_distribution()`, `_compute_ordering_null_baseline()` |
| `src/paper_autofill.py` | new table format, overlap column, per-pair p-values |
| `src/posthoc_ami_vi_perm.py` | per-pair p-values, rep_d in REPS, overlap ARI in summary |
| `src/posthoc_figs.py` | disagreement time-series plot wiring |
| `src/plots.py` | `plot_disagreement_timeseries()`, fixed legend labels |
| `src/posthoc_var_spread.py` | **NEW** — regime-conditional CVaR spread analysis |
| `src/posthoc_synthetic_groundtruth.py` | **NEW** — synthetic ground-truth experiment |
| `paper/main.tex` | all text changes above |

---

## Second revision round — 2026-04-19

After the initial revision above, the full pipeline was re-run with the new config (20 seeds × 7 reps including GARCH); a further round of reviewer audits surfaced additional inconsistencies that were fixed after inspecting the actual outputs.

## 2a. Full pipeline re-run and number refresh

All tables previously used placeholder values (Table 4 K=3 row from the old 6-rep × 20-seed robustness run). After re-running, the baseline values moved slightly because 7 reps (including rep_d/GARCH) contribute 21 cross-rep pairs instead of 15:

| Quantity | Placeholder | Confirmed |
|---|---|---|
| GMM cross-rep ARI (K=3) | 0.390 ± 0.002 | **0.415 ± 0.002** |
| HMM cross-rep ARI (K=3) | 0.334 ± 0.029 | **0.355 ± 0.034** |
| K=2 GMM cross | 0.402 ± 0.002 | **0.436 ± 0.002** |
| K=4 GMM cross | 0.400 ± 0.001 | **0.420 ± 0.002** |
| Step-sweep overlap ARI | placeholder | confirmed (0.617 → 0.490) |
| VaR spread (HMM mean) | 59% | **71%** |
| VaR spread (GMM mean) | 53% | **56%** |
| SPX COVID spread | 260% | **240%** (BTC >280%) |
| Ordering null top-1 (4-asset mean) | 0.953 | **0.932** |

All three ARI tables (baseline, step-sweep, K-sweep) now reference the same `20 seeds × 7 representations` run and carry a `7 representations including rep_d (GARCH-vol)` note.

**GMM/HMM ranking correction.** The original text said "GMM exhibits higher stability than HMM in both dimensions" — but with the new numbers GMM cross (0.415) > HMM cross (0.355) while GMM temp (0.396) < HMM temp (0.425). Rewritten to describe the actual split.

## 2b. Per-pair permutation test redesign

Initial implementation used B=999 permutations × all 148,680 pairs per asset — computationally infeasible (>1h single-threaded per asset). Redesigned:

- **Parallelised** `aggregate_permutation_pvalue` via `joblib Parallel(backend="loky")`, per-pair as independent task.
- **Subsampled** to 2,000 pairs per asset (plenty of power for the claim — ARI ≫ 0).
- **Reduced** B=999 → **B=99** (minimum detectable p = 1/100 = 0.01; sufficient for "p<0.05" claim).

**Claim correction**: initially claimed "all individual pairs p<0.05". Actual data shows max p = 1.0 across all assets. The claim is now "97.9–98.5% of pairs reject at p<0.05; failing 1.5–2.1% correspond to specific (model, seed, roll) slices where the observed ARI is near zero by chance (range [-0.13, 0.05])." Added new Supplementary Table `tab:perm_per_asset` with the per-asset distribution.

## 2c. Ordering null per-asset split surfaced

Reviewer noted that the "Mean=0.932 vs null=0.932" in the appendix table hid an asset-level split. Confirmed in data:

| Asset | Top-1 obs | Top-1 null |
|---|---|---|
| BTC | 0.946 | **0.976** (null > obs) |
| GLD | 0.919 | **0.962** (null > obs) |
| GSPC | 0.943 | 0.894 (obs > null) |
| IEF | 0.919 | 0.896 (obs > null) |

Restructured Table `tab:ordering` to show both Observed and Null per asset. Added "high-kurtosis assets (BTC, GLD) inflate the null" to the Methodological caveat. Updated all in-text claims: "do not exceed the null" → "do not consistently exceed; per-asset kurtosis-dependent split."

## 2d. Paper 2 alignment (companion submission)

Both Paper 1 and Paper 2 submitted to FRL; aligned for stylistic consistency:

- **Bibliography**: `elsarticle-harv` → `apalike` (Paper 2's style)
- **Supplementary Material block**: Appendix sections renumbered with `S1, S2, …` via `\renewcommand{\thesection}{S\arabic{section}}`; "Appendix Table" → "Supplementary Table"
- **Governance recommendation**: introduced **representation envelope** concept (parallel to Paper 2's **resolution envelope**): compute regime-conditional CVaR under each admissible representation, report the range `[min_Φ, max_Φ]`, escalate if width exceeds materiality threshold or ARI < 0.65.
- **Code availability section**: added formal section pointing to `mrv-lib` (matching Paper 2).
- **Companion citation**: added `\citep{companion_res}` in the governance paragraph.
- **Abstract wording**: "framework" → "ARI diagnostic" (avoids over-claiming); "confirms" → "suggests ... not fully resolved" (matches §4.4 body).

## 2e. Pipeline performance fixes (engineering)

The full pipeline took ~5 hours when first run; identified and fixed:

- **MKL segfault in loky workers on Windows** → set `MKL_NUM_THREADS=OMP_NUM_THREADS=...=1` at module load of `runner.py`.
- **Parallelised all post-processing** (`_compute_rep_stability_from_map`, `_compute_window_stability_from_map`, `_compute_semantic_*`, `_compute_ordering_*`) via a `_parallel_or_sequential` helper with automatic fallback to sequential on `TerminatedWorkerError`.
- Runtime dropped from ~5h to **~3.5h** (baseline + step sweep + robustness K=2/3/4 + posthoc).

## Additional files modified (post-commit)

| File | Change |
|------|--------|
| `paper/main.tex` | All 2a–2d updates |
| `config.yaml` | `n_jobs: 12 → 18` |
| `src/runner.py` | `_parallel_or_sequential` helper; parallelised 6 post-processing functions; MKL thread env forced at module load |
| `src/posthoc_ami_vi_perm.py` | Parallelised `cross_rep_records`, `temporal_records`, `aggregate_permutation_pvalue`; subsample 2000 pairs, B=99 |
| `src/posthoc_synthetic_groundtruth.py` | Uses `random_state` (not `seed`) to match `fit_hmm`/`fit_gmm` signature |
