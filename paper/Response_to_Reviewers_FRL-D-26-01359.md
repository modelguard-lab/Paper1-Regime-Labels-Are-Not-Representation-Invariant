# Response to Reviewers — FRL-D-26-01359

**Manuscript:** Regime Labels Are Not Representation-Invariant: Implications for Model Risk Governance
**Authors:** Kai Zheng, Rand Low, Ruili Wang
**Date:** 2 May 2026

We thank the editor and both reviewers for the careful and constructive reading. The revision addresses every comment; below we list each reviewer point, summarise the change, and point to the relevant section. New material is highlighted in the colour-coded revised manuscript (additions in blue, removed text struck through).

---

## Reviewer #1

### R1.1 — Admissible class is not justified strongly enough; results may be specific to the authors' design

We agree that the strength of the universal claim is bounded by the admissible class chosen, and we have addressed this on three fronts.

**(i) Class expanded.** The admissible class now contains **eight** representations (up from six in the originally submitted draft) varying along **five** documentation-relevant dimensions: standardisation, feature subset, rolling horizon, volatility estimator (rolling realised vs **GARCH(1,1) — newly added rep_d**), and information source (backward-looking realised vs forward-looking VIX, S&P 500 only, rep_e). The GARCH addition specifically addresses the heterogeneity question: it is structurally distinct from a rolling realised filter and is the canonical methodological fork in FRTB market-risk modelling.

**(ii) Per-dimension justification.** A new Supplementary Section S2 *Admissible Class as Registered Scope* gives a one-paragraph justification for each of the five dimensions, anchoring each to a specific entry point in routine model paperwork (SR 11-7 data-and-assumptions principle, FRTB volatility-estimator choice, etc.) rather than ad-hoc design.

**(iii) Registered-scope framing of conclusions.** We now state explicitly in the Abstract, Introduction (§1, last paragraph), Methodology (§3.2, last sentence), and Supplementary §S2 that conclusions are statements *within* this registered class, not universal claims. Critically, the class includes near-redundant specifications by design (five of seven share VaR/CVaR; four share the same rolling windows), so the reported ARI is a **conservative upper bound**: a broader class can only weaken pairwise overlap, since admissible variation can only add disagreement. This is now the explicit framing throughout the paper.

In short, the partition-instability finding is an upper bound on the disagreement realised in production specification spaces; broadening the class would *strengthen* the result, not weaken it.

### R1.2 — Coarse risk ordering is stable (top-1 ≥ 0.91, Spearman 0.89–0.91); conclusion should be framed more carefully

This is critical, and the current manuscript **explicitly retracts** the "high ordering stability" reading. The 0.91–0.94 numbers Reviewer 1 cites were produced by a Hungarian + CVaR matching rule that mechanically rank-aligns CVaR extremes regardless of whether the partitions agree; under the same matching rule the empirical independent-partition null is itself **0.78–0.96**, so the apparent excess collapses (Supplementary Section S3 *Ordering Consistency*, Table tab:ordering, "Chance (null)" row).

Under a **matching-free rank-aligned metric** with a rigorous 1/K null:
- Top-1 Jaccard at K=3: 0.47 (vs null 0.20, a 2.4× excess), not 0.93
- Pointwise agreement: 0.55 (vs 0.33)
- Spearman rank: 0.48 (vs 0)

Across K ∈ {2, 3, 4, 5} the honest excess is 1.6× to 2.4×, comparable in magnitude to the partition-level ARI rather than categorically higher. The retraction is now visible in:
- Abstract (Contribution tag): replaces the biased Hungarian + CVaR rule with "a matching-free rank-aligned metric calibrated against a rigorous 1/K null"
- §1 (Introduction): "we retract that claim — the matching rule's null is itself 0.78–0.96"
- §4.1 (Baseline comparison), paragraph "Coarse ordering: a corrected reading"
- §6 (Conclusion): "Coarse risk ordering is genuinely above chance but at comparable magnitude to the partition (top-1 Jaccard 0.47 vs 1/K=0.20), not categorically higher"
- Supplementary §S5 (Robustness and Alternative Interpretations), point (iii)

The corrected message is that **neither stability dimension is a safe harbour for the other** — both partition-level and ordering-level agreement are materially above chance, but both are well below any operationally meaningful threshold. The governance recommendation in §6 is therefore based on the matching-free metric, not the Hungarian one, and treats ordering at the same level of scepticism as partition.

### R1.3 — Governance implications should be more cautious; minimal single-asset design limits external scope

Agreed. We now state the scope explicitly as a four-asset case-study design and reframe the governance recommendation as a calibrated, institution-specific trigger. Specifically:

- **Abstract** ends with the explicit limitation: "Method: Four single-asset case studies"
- **§6 Conclusion** ends with: "institutions are expected to calibrate the triggers to their own admissible class. Scope: four-asset case-study, daily univariate features, fixed 252-day window, discrete-state HMM/GMM and non-parametric k-means/spectral; extension to multivariate features, higher frequencies, and continuous-state model classes is left for future work"
- **Conclusion limitations paragraph** explicitly says "four assets are case studies, not a statistical sample — no formal cross-asset inference supported"

The **representation envelope** workflow we propose is presented as a calibrable diagnostic that an institution registers against its own admissible class — not a one-size-fits-all rule.

---

## Reviewer #2

### R2.1 — Abstract is unbalanced; please restructure to cover methodology, novelty, gap, significance

The abstract has been rewritten with **five explicit labelled sections** corresponding directly to the elements requested:

> *Background and gap.* Regime models underpin capital, risk-limit, and hedging decisions. Their outputs depend on feature construction, rolling-window, and preprocessing choices, but the existing literature evaluates a single fitted specification rather than multiple admissible specifications against each other. Under Basel IV and SR 11-7, models must instead demonstrate structural stability across admissible specifications.
>
> *Contribution.* We provide the first systematic audit of representation invariance for discrete-state regime models. We measure partition-level agreement by ARI and coarse risk-ordering agreement by a matching-free rank-aligned metric calibrated against a rigorous 1/K null.
>
> *Method.* We test four single-asset case studies (S&P 500, IEF, GLD, BTC) under three model classes (HMM, GMM, non-parametric k-means) at K ∈ {2,3,4,5}. Eight admissible feature representations vary along five dimensions: standardisation, feature subset, rolling horizon, volatility estimator (rolling vs GARCH), and information source (realised vs VIX). A step-size sweep separates regime persistence from rolling-window overlap.
>
> *Findings.* Cross-representation ARI is 0.35–0.45 across all assets and model classes, well below Steinley (2004)'s 0.65 poor-recovery threshold. The disagreement persists under k-means (no Gaussian assumption) and at fully non-overlapping windows. Under the matching-free metric, top-1 ordering Jaccard at K=3 is 0.47 against a 1/K=0.20 null, a 2.4× excess that is genuinely above chance but comparable in magnitude to the partition-level signal rather than categorically higher.
>
> *Significance.* Representation-conditional CVaR diverges by 39–77% of the unconditional CVaR on average and exceeds 300% during COVID-19 stress. We recommend a cross-representation envelope as a diagnostic complement to temporal backtesting. The admissible class is registered and intentionally narrow (Supplementary Section S2), so the reported ARI is a conservative upper bound: a broader class would only weaken agreement.

### R2.2 — Generalisation / circularity: the conclusion of instability may be sensitive to the chosen representation space

This is a sharp point and we engage it directly. The **monotonicity argument** that resolves the apparent circularity is now explicit:

> "The class includes near-redundant specifications by design, so the reported ARI is a conservative upper bound on agreement: a broader admissible class would only weaken pairwise overlap, since admissible variation can only add disagreement." (§3.2)

The logic: any specification we *added* to the admissible class is, by definition, an additional admissible alternative, and adding an additional alternative cannot *increase* the pairwise mean ARI (the new pairs are constrained to be either no more similar than existing ones, or strictly less similar). So our reported numbers are a **conservative bound on disagreement** under the registered scope. The framing thus inverts the circularity concern: the result becomes less rather than more sensitive to the choice of class as the class broadens.

A reader working under a strictly *narrower* class can read off the relevant cell of Supplementary Table tab:decomp: the highest within-class mean ARI we observe is 0.55 ± 0.04 (feature-subset variation under common windows), still below the 0.65 threshold. A reader working under a *broader* class would by construction obtain lower agreement.

This argument is now in: Abstract (last paragraph), §1 (last paragraph), §3.2, and Supplementary §S2.

### R2.3 — Theoretical foundation: dedicated "theoretical background" section

A full section §2 *Theoretical background* now precedes the Methodology, with three subsections:
- §2.1 *Identifiability of finite mixtures and Markov-switching models* (Teicher 1963, Yakowitz–Spragins 1968, Hamilton 1989, Guidolin 2011)
- §2.2 *Limitations of the existing theory for representation invariance* — three failure modes of the asymptotic guarantee (non-stationarity, fixed-Φ premise, parametric misspecification) and the connection to D'Amour et al.'s underspecification framework
- §2.3 *Empirical fragility of regime inference in finance* — 2007–2009 GFC, 2020 COVID-19 crash, and 2022 inflation shock

The section explicitly states *what* the empirical diagnostic does and does not test: "The descriptive diagnostic of Section 3.4 is therefore the operative validation tool *until a theory of representation-invariant stability exists*." This bounds the empirical claims; a closed-form bound from feature-distribution divergence to label ARI is left for future work.

### R2.4 — Robustness should incorporate broader range of models, including non-parametric / no-discrete-state

The revision adds **two non-parametric robustness benchmarks** alongside the parametric HMM and GMM:

1. **k-means clustering** (centroid-only assumption, no Gaussian likelihood, no transition dynamics) — Table tab:kmeans_robustness shows cross-rep ARI 0.437–0.465 across assets, still well below 0.65.

2. **Spectral clustering** (k-NN affinity matrix eigendecomposition, no parametric emission and no Euclidean centroid assumption) — newly added in this revision. Cross-rep ARI 0.291–0.348 across assets (mean 0.326 ± 0.040), still well below 0.65 — confirms the same conclusion under the strongest available non-parametric benchmark we can run on these features.

Both methods preserve the ranking HMM < GMM < non-parametric < 0.65 on every asset.

**Continuous-state and tree-ensemble models** (score-driven volatility, stochastic volatility, copula-based regimes, deep classifiers) are explicitly out-of-scope for this paper and are noted as such in §3.3 (last sentence). We considered including a continuous-state demonstration in this paper but judged it would push us substantially over the FRL 2,500-word limit; the present paper's contribution is in operationalising the diagnostic for the discrete-state class that dominates current production practice (Hamilton 1989; Ang–Timmermann 2012; Guidolin 2011).

### R2.5 — Practical implications should be more specific

The §6 Conclusion now contains an operational workflow and a worked numerical example:

**Workflow** — three numbers per asset per quarter:
1. Cross-representation ARI under the registered admissible class
2. Envelope width = max_Φ − min_Φ of regime-conditional CVaR
3. Worst-asset percentile across the admissible class

**Triggers** — fire when:
- ARI < 0.65 (Steinley 2004 partition-recovery threshold), OR
- Envelope width / |unconditional CVaR| > 0.10 (10% materiality)

**Action** — escalate to MRM committee; resolve by adopting conservative representation, reporting an ensemble, or documenting the limitation under SR 11-7's known-limitation provisions.

**Worked example (S&P 500, 23 March 2020)** is now in §6 (in-line summary) and Supplementary §S13 (full step-by-step):
- Four representations assign S&P 500 to the high-volatility state (conditional CVaR ~−700 bp/day)
- Two assign the medium-volatility state (~−200 bp/day)
- Envelope width ≈ 500 bp = 200% of unconditional CVaR
- Trailing cross-rep ARI well below 0.65
- Both triggers fire by two orders of magnitude → classification escalates

The intent is to make the recommendation a *concrete, repeatable, timestamped* operational step, not a conceptual prescription.

### R2.6 — Numeric examples + GitHub repository link

Both are now provided.

**Worked numeric example** — Supplementary Section S13 *Worked Numeric Example: S&P 500 2020-Q1* walks through the full computation for a single asset on a single date, with per-step inputs, intermediate quantities, trigger evaluation, and admissible action.

**GitHub repositories** — two are now linked in the Code Availability section:

1. **Paper repository** (frozen reproducibility artifact):
   `https://github.com/modelguard-lab/Paper1-Regime-Labels-Are-Not-Representation-Invariant`
   Contains the complete experiment code, four-asset case studies, robustness sweeps, and post-hoc analysis scripts that produced every table and figure in the paper.

2. **mrv-lib** (productised diagnostic library, maintained):
   `https://github.com/modelguard-lab/mrv-lib` (also at `https://mrv-lib.org`, `pip install mrv-lib`)
   The implementation referenced in §6's recommendation. Intended for use by model validators on their own institution-specific admissible classes.

The two repositories serve distinct purposes (the paper repo is a frozen artifact; mrv-lib is a maintained tool), and Supplementary §S13 explicitly references mrv-lib as the implementation reference.

---

## Additional changes not driven by reviewer comments

In addition to the reviewer-driven changes, this revision incorporates two methodological strengthenings that we believe materially improve the paper:

1. **Politis–Romano (1992) circular block bootstrap as a second null** for the per-pair permutation test, with Patton–Politis–White (2009) optimal block length. The iid permutation null is mildly anti-conservative on persistent rolling-window labels (raw p-values too small); the CBB null preserves within-series autocorrelation under random alignment. Aggregate CBB p-values are 3–4× larger than iid (0.063–0.073 vs 0.016–0.021) and the BH-adjusted survival fraction at q=0.05 drops from 97.7–98.5% under iid to 67.6–71.2% under CBB — a quantitative restatement of the iid null's anti-conservatism. The substantive conclusion (cross-rep ARI well below 0.65) survives both nulls. Details in Supplementary §S3 *Per-pair Permutation Test Details*.

2. **Benjamini–Hochberg (1995) step-up FDR control at q=0.05** applied to the per-pair p-values under each null. Bonferroni at n=2,000 pairs requires raw p < 2.5×10⁻⁵, below the smallest attainable p-value at B=99 replications (1/100 = 0.01), so it is uninformative in this design. BH-FDR is the appropriate correction given that the principal claim is the aggregate magnitude of cross-representation ARI, not the discovery of any specific pair. Supplementary Table tab:perm_per_asset reports the BH-survival fractions per asset and per null.

These additions strengthen the per-pair claim with appropriate dependence and multiplicity corrections; they do not alter any of the headline numbers in the main tables.

---

## Summary of word count

The revised main text is **2,499 words** (FRL 2,500-word limit), excluding abstract, references, tables, figures and captions, acknowledgements, declarations, and supplementary material. Supplementary material totals 18 pages (S1–S13).

We hope these revisions address all reviewer concerns. We are happy to make further changes in response to a second round of review.

Yours sincerely,

Kai Zheng, Rand Low, Ruili Wang
