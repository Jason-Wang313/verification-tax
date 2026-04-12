# Calibrated NeurIPS 2026 Main Track Review
## Paper: "The Verification Tax: Fundamental Limits of AI Auditing in the Rare-Error Regime"

**Calibration source:** 20,581 NeurIPS 2021-2025 main track papers (81,758 reviews), VTax pipeline v2.0.
**Pipeline date:** 2026-04-11. **Review version:** v11.
**Gate:** 3,621 theory-relevant papers filtered; 30 comparables found (2 Tier A, 28 Tier C).

**DATA LIMITATION:** NeurIPS does not publish rejected papers by default. Calibration is anchored to accepted paper scores. We cannot compute fatality rates or decision boundaries. All analysis is relative positioning against accepted baselines.

---

## 1. Summary and Contributions

This paper establishes the fundamental sample complexity of verifying calibration in the rare-error regime ($\varepsilon \to 0$). The main result is the **verification tax**: estimating binned ECE to accuracy $\delta$ from $m$ holdout samples requires $m = \Omega(L\varepsilon/\delta^3)$ for $L$-Lipschitz calibration functions, and this is achieved by histogram binning with $B^* = (L^2 m/\varepsilon)^{1/3}$ bins. The key novelty over prior work (Sun et al. 2023, Futami & Fujisawa 2024) is the explicit $\varepsilon$-dependence, which reveals a phase transition at $m \sim 1/\varepsilon$ and enables scaling-law analysis.

The paper contains 17 formal results organized into 7 contribution groups:

1. **The verification tax** (T1-T3): Le Cam lower bound $\Omega(\sqrt{\varepsilon/m})$, minimax lower bound $\Omega((L\varepsilon/m)^{1/3})$ via reduction to Lepski et al. (1999), matched upper bound via histogram binning.
2. **Universality** (TG): General verification functional covering calibration, fairness, robustness.
3. **Impossibility results** (T4, TB, P3): White-box irrelevance, self-verification impossibility, ungameability.
4. **Adaptive evaluation** (TA): Active verification achieves $\Theta(\sqrt{\varepsilon/m})$, eliminating the Lipschitz constant.
5. **Scaling consequences** (TH, PR, TC): Verification horizon, recalibration trap, compositional tax.
6. **Temporal decay** (THL, TD): Verification half-life and dynamics.
7. **Empirical validation** (planned): 5 falsifiable predictions on MMLU with 4 LLMs.

**Page count:** ~15 pages main text. NeurIPS limit is 9 pages + unlimited appendix/references. **This is a critical issue** (see Section 4, W6).

---

## 2. Strengths

**S1. Novel $\varepsilon$-dependence — the core technical contribution.**
Prior work (Sun et al. 2023, Futami & Fujisawa 2024) established the $n^{-1/3}$ rate for ECE estimation but treated the error rate $\varepsilon$ as a fixed constant. This paper makes $\varepsilon$ a first-class parameter, revealing:
- A phase transition at $m \sim 1/\varepsilon$ (Corollary 1)
- Scaling laws connecting model size to verification cost (Corollary 3)
- The verification horizon where models outpace auditors (TH)

This is a genuine contribution absent from all prior work. No paper in the 20,581-paper NeurIPS corpus addresses this.

*Anchored:* "Auditing Fairness by Betting" (NeurIPS 2023, spotlight, mean score 7.25) similarly provided a genuinely new framework (sequential testing for fairness) not present in prior work. That paper received scores {6, 7, 8, 8}. VTax's novelty is comparable in kind.

**S2. Matched upper and lower bounds.**
T2 and T3 establish $(L\varepsilon/m)^{1/3}$ as the minimax rate (up to logs). The log gap is honestly attributed to an open problem in $L_1$ functional estimation (Lepski et al. 1999). The Le Cam bound (T1) provides a separate $\sqrt{\varepsilon/m}$ floor. The active rate (TA) matches the Le Cam bound, establishing that adaptivity is worth exactly the factor $(L^2 m/\varepsilon)^{1/6}$.

*Anchored:* Theory papers with matched bounds are well-received at NeurIPS. "Nearly Tight Bounds For Differentially Private Multiway Cut" (NeurIPS 2023, oral, mean 6.67) and "Optimal Top-Two Method for Best Arm Identification" (NeurIPS 2024, poster, mean 5.8) demonstrate this pattern.

**S3. Breadth of implications — 7 contribution groups.**
The paper explores an unusually wide landscape: impossibility results (T4, TB), game-theoretic robustness (P3), sequential protocols (TS), transfer learning (P2), compositional systems (TC), temporal dynamics (THL, TD). Each builds on the core rate using the same 3 properties (Lipschitz integrand, Bernoulli noise, $L_1$ functional). The General Verification Tax (TG) unifies these cleanly.

*Caveat:* This breadth is simultaneously the paper's greatest strength and its most dangerous weakness (see W1, W6).

**S4. Regulatory and practical relevance.**
The verification horizon (TH), half-life (THL), recalibration trap (PR), and compositional tax (TC) have direct implications for AI regulation. The practitioner tools (adaptive holdout sizing, stratified evaluation, budget calculator) are immediately useful. The self-verification impossibility (TB) is a clean result with a clear policy implication.

**S5. Honest reporting.**
The log gap in T2 is attributed to an open problem, not claimed as tight. The parametric exception (T4 corollary) honestly notes when white-box access helps. The temperature scaling analysis (P1) includes a misspecification caveat.

---

## 3. Weaknesses

**W1. Scope too broad — 17 results in 9 pages (CRITICAL).**
The paper currently has ~15 pages of main text but NeurIPS allows only 9. Even compressed, 17 formal results cannot receive adequate exposition in 9 pages. The risk is not just density — it's that reviewers will perceive the paper as a "theorem dump" that prioritises quantity over depth.

*Calibration:* "Scaling Up Active Testing to Large Language Models" (NeurIPS 2025, poster, mean 3.75, scores {3, 3, 4, 5}) — a comparable paper that attempts breadth over depth — received the lowest scores in our comparable set. The low scores suggest reviewers penalise papers that spread too thin.

**Specific recommendation:** The 9-page main text should contain at most 7-8 results. Suggested main text: T1, T2, T3, T4, TA, TB, TH. Move to appendix: TG, TS, P1, P2, P3, PR, TD, THL, TC, and most corollaries. Promote TC (compositional) or THL (half-life) to main text only if space allows.

**W2. Empirical validation is planned but not yet executed.**
The empirical section describes 5 falsifiable predictions but no actual data. NeurIPS reviewers will treat this as a pure theory submission. While pure theory papers are accepted, the practical claims (verification horizon, half-life numbers, MMLU worked examples) ring hollow without empirical backing.

*Calibration:* Among the 25 accepted comparables, papers with empirical validation (e.g., "Auditing Fairness by Betting" with 3 benchmark datasets) scored higher on average (mean 7.25) than purely theoretical papers. "Bernstein-von Mises for Adaptively Collected Data" (NeurIPS 2025, poster, mean 4.6) — a pure theory paper — shows the score floor for theory without experiments.

**W3. The log gap between T2 and T3.**
The lower bound (T2) has a $(\log m)^{-c_3}$ factor that the upper bound (T3) lacks. While honestly attributed to Lepski et al. (1999), a reviewer may view this as an incomplete result. The gap means the minimax rate is not fully characterised.

*Calibration:* "gap_bounds" is a standard weakness for theory papers. Papers with fully matched bounds (like "Nearly Tight Bounds For Differentially Private Multiway Cut", mean 6.67) score higher than those with log gaps.

**W4. Lipschitz assumption on calibration functions.**
The entire framework assumes $\Delta \in \mathcal{F}(L)$. While Lipschitz continuity is standard in nonparametric statistics, real calibration functions may have discontinuities (e.g., from binning artifacts, distribution shift). The paper does not discuss robustness of the bounds to violations of the Lipschitz assumption.

*Calibration:* "assumptions_strong" is a frequent weakness category. Among accepted comparables, assumption concerns appear regularly but are survivable when other strengths compensate.

**W5. Reduction to Lepski et al. (1999) — limited technical novelty in T2.**
The core lower bound (T2) is obtained by reducing to the known minimax rate for $L_1$ functional estimation via the Brown-Low equivalence (1996). While the reduction itself requires care (score-mass concentration, effective sample size), the heavy lifting is done by existing tools. A reviewer may argue: "The main lower bound is a corollary of Lepski et al., not a new technique."

*Counter:* The $\varepsilon$-dependence IS new and not obtainable from Lepski et al. without the reduction. The phase transition and scaling law consequences are genuine contributions.

**W6. Presentation density.**
Even after compression to 9 pages, the paper will be extremely dense. The combination of W1 (too many results) and W6 (density) is the most dangerous compounding pair, because a reviewer who finds the paper hard to follow AND questions the practical relevance of any individual result reaches: "unfocused and hard to follow."

*Calibration:* "presentation_dense" and "scope_too_broad" are common weakness categories. In the global NeurIPS data, "clarity_writing" is the 3rd most frequent weakness (24,828 mentions).

---

## 4. Questions for Authors

1. **Page limit compliance:** The paper is currently ~15 pages. What is the planned 9-page structure? Which results move to the appendix?

2. **Empirical timeline:** When will the MMLU/NIM experiments be completed? Will results be available before the submission deadline?

3. **Log gap:** Is there a path to closing the log gap between T2 and T3? Have you considered whether the log factor is an artifact of the reduction or a genuine feature of the ECE estimation problem?

4. **Lipschitz constant estimation:** In practice, $L$ is unknown. How sensitive are the practical tools (holdout sizing, bin count) to misspecification of $L$?

5. **Beyond binned ECE:** Do the results extend to kernel-based or debiased ECE estimators? The Bridge Lemma ties binned ECE to smooth CE, but the binning discretisation is itself a limitation.

---

## 5. Limitations

The paper's limitation discussion is embedded throughout (misspecification caveat in P1, honest log gap attribution, Lipschitz assumption discussions). A consolidated limitations section would strengthen the paper.

Key limitations:
- Lipschitz assumption may not hold universally
- Log gap between T2 and T3 is unresolved
- Empirical validation is planned but not executed
- Single-author paper from independent researcher (no institutional bias, but limited external validation)
- The compositional tax (TC) assumes Lipschitz composition, which fails for discontinuous components (retrieval, tool use)

---

## 6. Soundness: 3/4 (Good)

The proofs follow standard information-theoretic techniques (Le Cam, Fano, Brown-Low equivalence). T1 and T3 are clean and self-contained. T2 relies on the established Lepski et al. (1999) result via a careful reduction. The active verification proof (TA) is technically involved but the two-phase strategy is well-motivated. The self-verification impossibility (TB) is elegant and airtight.

Minor concerns: the Brown-Low equivalence (T2, Step 2) requires technical conditions (score-mass concentration, minimum $m$) that should be verified more carefully. The temperature scaling analysis (P1) requires correct specification, which is a strong assumption.

## 7. Presentation: 2/4 (Below Average)

At 15 pages, the paper violates the 9-page limit. Even compressed, 17 results will create a very dense read. The contribution is real but the packaging needs significant work. A reviewer encountering this paper will struggle to identify the main contribution amid the breadth.

## 8. Contribution: 4/4 (Excellent)

The $\varepsilon$-dependent minimax rate for ECE estimation is a genuinely new result. The verification horizon, phase transition, and compositional tax are novel consequences with practical significance. No paper in the 20,581-paper NeurIPS corpus addresses this specific question. The breadth of implications (impossibility, adaptivity, transfer, composition, temporal dynamics) is remarkable.

---

## 9. Overall Scores — Three Reviewer Archetypes

### Reviewer A: Statistician (values proof novelty, tight bounds)

**Score: 5 (Accept)**

*Reasoning:* The matched minimax rate with $\varepsilon$-dependence is a clean contribution to nonparametric functional estimation. The reduction to Lepski et al. is well-executed. The active verification rate (TA) eliminating the Lipschitz constant is a nice result. The log gap is unfortunate but honest.

*Anchored to:* "Auditing Fairness by Betting" (NeurIPS 2023, spotlight, scores {6, 7, 8, 8}) — a similarly structured statistical theory paper with practical implications. VTax has comparable theoretical depth but weaker experiments (none yet). Adjusting down from 7.25 to ~5 for missing empirics and page-limit violation.

Also anchored to: "Nearly Tight Bounds For Differentially Private Multiway Cut" (NeurIPS 2023, oral, mean 6.67) — tight bounds paper. VTax has a log gap where that paper does not.

### Reviewer B: ML Experimentalist (values practical impact, real experiments)

**Score: 3 (Leaning Reject)**

*Reasoning:* The paper has no experiments. The practical tools (holdout sizing, bin count) are useful but untested. The verification horizon numbers depend on Chinchilla scaling assumptions. The 15-page paper needs major surgery to fit the format.

*Anchored to:* "Scaling Up Active Testing to Large Language Models" (NeurIPS 2025, poster, scores {3, 3, 4, 5}) — an active testing paper that DID have experiments but still scored poorly. VTax lacks even this paper's empirical validation. Scoring at the low end of accepted papers.

### Reviewer C: AI Safety/Policy (values regulatory implications)

**Score: 5 (Accept)**

*Reasoning:* The self-verification impossibility (TB) alone is worth publishing — it directly addresses a live policy debate about whether model developers can self-audit. The verification horizon connects calibration theory to frontier AI governance. The compositional tax quantifies a real concern about agentic systems. The regulatory implications are clearly stated and well-grounded in the theory.

*Anchored to:* "Position: If Innovation in AI systematically Violates Fundamental Rights" (NeurIPS 2025, oral, mean 6.0) — a policy-adjacent paper that received strong acceptance despite being less technically deep. VTax has stronger theory supporting its policy claims.

### Predicted Distribution: **{3, 5, 5}** — moderate variance

Mean: 4.33. Likely outcome: **Borderline Accept / Poster** contingent on rebuttal addressing Reviewer B's concerns (experiments, page limit).

---

## 10. Score Anchoring Analysis

### Score Floor Among Accepted Comparables
**3.75** ("Scaling Up Active Testing to Large Language Models", NeurIPS 2025 poster, scores {3, 3, 4, 5}). This is the lowest mean score among accepted papers in our comparable set. VTax should score above this — the theoretical contribution is substantially stronger.

### Most Similar Comparable
**"Auditing Fairness by Betting"** (NeurIPS 2023, spotlight, sim=0.80 [manually assigned], scores {6, 7, 8, 8}, mean 7.25). This paper is the closest structural match: sequential statistical testing framework for AI auditing, theory + experiments, practical implications. VTax is stronger on theoretical depth (matched bounds, 17 results) but weaker on experiments (none vs 3 benchmarks) and presentation (15 pages vs well-packaged).

**Expected delta from this anchor:** -1.5 to -2.0 for missing experiments and page-limit violation. Predicted: ~5.0-5.5.

### Per-Paper Mean Distribution of Accepted Comparables
Range: 3.75 to 7.25. Median: ~5.8. VTax likely falls in the 4.0-5.5 range given its pure-theory status and format issues.

---

## 11. CRITICAL — Page Limit Problem

**Current state:** ~15 pages main text. NeurIPS limit: 9 pages.

**Assessment:** The paper CANNOT fit 17 results in 9 pages without becoming unreadably dense. This will directly cost 0.5-1.0 points per reviewer.

### Recommended 9-Page Structure

**Pages 1-1.5: Introduction** (base rate framing, 4 grouped contributions, outline)

**Pages 1.5-3.5: Core Theory (Section 2)**
- Definition 1 (ECE), Definition 4 (Lipschitz class), Definition 5 (minimax risk)
- T1 (Le Cam lower bound) — full proof (~0.5 page)
- T2 (Minimax lower bound) — proof sketch (~0.5 page)
- T3 (Matched upper bound) — full proof (~0.5 page)
- Phase transition corollary

**Pages 3.5-5.5: Extensions (Section 3)**
- T4 (White-box irrelevance) — full proof (~0.5 page)
- TA (Active verification) — proof sketch (~0.5 page)
- TB (Self-verification impossibility) — full proof (~0.3 page)

**Pages 5.5-7: Scaling Consequences (Section 4)**
- TH (Verification horizon) — theorem + proof + numbers
- TC (Compositional tax) — theorem + proof + agent loop example

**Pages 7-8.5: Empirical Validation (Section 5)**
- 3-4 experiments with MMLU data (if available)
- Or: "Empirical Predictions" with concrete falsifiable claims

**Pages 8.5-9: Related Work + Conclusion**

**Moved to Appendix:** TG, TS, P1, P2, P3, PR, TD, THL, all corollaries, full proofs of T2 and TA.

This reduces from 17 to 7 main-text results: **T1, T2, T3, T4, TA, TB, TH + TC**. The core story becomes: "verification is expensive (T1-T3), white-box doesn't help (T4), adaptive does (TA), self-verification is impossible (TB), and here's what that means for scaling (TH) and composition (TC)."

### Does Compression Change the Predicted Score?
**Yes, positively.** A focused 9-page paper with 7 results would likely gain +0.5 to +1.0 from Reviewer B (less "unfocused" criticism) and hold steady with Reviewers A and C. Predicted distribution shifts from {3, 5, 5} to **{4, 5, 5}** — solidly in poster territory.

---

## 12. Single Highest-ROI Change

**Complete the MMLU experiments.**

*Evidence:* "Auditing Fairness by Betting" (7.25 mean, spotlight) had 3 benchmark experiments. "Scaling Up Active Testing" (3.75 mean, poster) had LLM experiments but questionable novelty. The score gap (7.25 vs 3.75) between these two Tier A comparables is partially explained by the quality of empirical validation. VTax currently has 0 experiments.

Adding even a basic empirical study (4 models, MMLU, ECE vs $m$ curves with theoretical bounds overlaid, phase transition validation) would shift VTax from "pure theory" to "theory + empirical" — a category that scores substantially higher in the calibration data.

**Expected impact:** +0.5 to +1.5 across all three reviewer archetypes. This is the single change with the highest expected ROI.

---

## 13. Diff-Ready Revision List (Ordered by Expected Impact)

### 1. Compress to 9 pages with 7 main-text results
**Expected impact:** +0.5 to +1.0 (especially Reviewer B)
**Evidence:** "Scaling Up Active Testing" (15→accepted at poster but scored 3.75) demonstrates that scope-breadth is penalised. Focused theory papers ("Nearly Tight Bounds", mean 6.67) score higher.

### 2. Complete MMLU empirical validation
**Expected impact:** +0.5 to +1.5 (all reviewers)
**Evidence:** "Auditing Fairness by Betting" (theory + 3 benchmarks = 7.25) vs "Bernstein-von Mises" (pure theory = 4.6). The theory-to-empirical upgrade is worth ~2 points.

### 3. Promote TB (self-verification impossibility) to a highlighted result
**Expected impact:** +0.3 to +0.5 (Reviewer C)
**Evidence:** Clean impossibility results with policy implications are valued. This is the paper's most citable individual result.

### 4. Close or explicitly scope the log gap
**Expected impact:** +0.2 to +0.5 (Reviewer A)
**Evidence:** State whether the log factor is conjectured to be necessary or an artifact. If there's a path to closing it, sketch it. "Nearly Tight Bounds" (oral) had fully matched bounds.

### 5. Add a consolidated limitations section
**Expected impact:** +0.1 to +0.3 (all reviewers)
**Evidence:** NeurIPS increasingly expects explicit limitations. The paper's honest caveats are scattered; consolidating them signals maturity.

### 6. Include 2-3 worked numerical examples in main text
**Expected impact:** +0.2 to +0.5 (Reviewer B)
**Evidence:** The MMLU worked example ($K^* = 4$ recalibration rounds) and domain-specific half-lives (medical: 2 months, trading: 17 hours) make the abstract theory tangible. Currently these are in remarks — promote to main text.

---

## Calibration Appendix

### A. Global NeurIPS Base Rates

| Year | Papers | Reviews | Mean Score | Scale |
|------|--------|---------|------------|-------|
| 2021 | 2,708 | 10,384 | 6.41 | 1-10 |
| 2022 | 3,054 | 11,219 | 6.04 | 1-10 |
| 2023 | 3,777 | 16,666 | 5.96 | 1-10 |
| 2024 | 4,830 | 18,562 | 5.94 | 1-10 |
| 2025 | 6,212 | 24,927 | 4.34 | 1-6 |

Total accepted: 16,422. Mean accepted score: 5.50.

### B. Comparable Paper Summary

| Rank | Paper | Year | Decision | Sim | Mean Score | Tier |
|------|-------|------|----------|-----|------------|------|
| 1 | Auditing Fairness by Betting | 2023 | spotlight | 0.80* | 7.25 | A |
| 2 | Scaling Up Active Testing to LLMs | 2025 | poster | 0.80* | 3.75 | A |
| 3 | On Calibrating Diffusion Probabilistic Models | 2023 | poster | 0.45 | 6.40 | C |
| 4 | Towards Calibrated Robust Fine-Tuning of VLMs | 2024 | poster | 0.42 | 5.25 | C |
| 5 | Online Clustering of Bandits | 2023 | poster | 0.38 | 5.00 | C |
| 6 | Best of Both Worlds: OOD Detection | 2024 | poster | 0.38 | 6.25 | C |
| 7 | Optimal Top-Two for Best Arm ID | 2024 | poster | 0.38 | 5.80 | C |
| 8 | Minimax Optimal Online Imitation Learning | 2022 | accept | 0.38 | 7.33 | C |
| 9 | Variance, Admissibility of ERM | 2023 | spotlight | 0.38 | 6.60 | C |
| 10 | Testing Calibration in Nearly-Linear Time | 2024 | poster | 0.30 | 6.25 | C |

*Manually assigned similarity for mandatory comparables.

### C. Missing Mandatory Comparables (not in NeurIPS scrape)

- Sun, Song & Hero, "On the Estimation of Expected Calibration Error" (NeurIPS 2023 Spotlight) — **THE most relevant paper**. Known scores: estimated ~7-8 range based on spotlight designation.
- Futami & Fujisawa (NeurIPS 2024) — directly comparable upper bound work.
- Blasiok et al., "A Unifying Theory of Distance from Calibration" (STOC 2023) — not NeurIPS.
- Ciosek et al., "Finite-Sample Guarantees for Calibration Error" (2025 preprint) — not yet published.

**Critical gap:** The two most relevant papers (Sun et al. 2023, Futami & Fujisawa 2024) are missing from the calibration data. Sun et al. as a NeurIPS 2023 Spotlight likely scored in the 7-8 range. VTax extends Sun et al. with the $\varepsilon$-dependence they left as an open question. This is a strong novelty claim.

### D. Review Text Availability

Tier A papers (2 papers, 8 reviews): **0 reviews with text** (2023 and 2025 NeurIPS reviews lack extracted strengths/weaknesses in the scrape). This limits our ability to identify survivable weakness patterns from the closest comparables.

Tier C papers (28 papers, ~95 reviews): Review text availability varies by year. 2021-2023 papers have limited text; 2024-2025 have more.

### E. Survivable Weakness Analysis

Due to missing review text for Tier A papers, survivable weakness analysis relies on global NeurIPS patterns:

| Weakness Category | Global Frequency | Survivability |
|-------------------|-----------------|---------------|
| empirical_validation | 36,921 | High (most common, most survivable) |
| significance | 27,006 | Medium (2nd most common, can be fatal) |
| clarity_writing | 24,828 | High (3rd most common, rarely fatal alone) |
| assumptions | 13,794 | High (7th, survivable with honest discussion) |
| novelty | 12,206 | Medium (fatal if core contribution questioned) |
| gap_bounds | N/A (theory-specific) | Medium (log gaps are common in accepted theory papers) |

---

## Decision Prediction

**Predicted decision: Borderline Accept (poster)** — contingent on page-limit compliance and ideally empirical results.

**Confidence:** 50% accept, 30% borderline/conditional, 20% reject.

**Risk factors:**
- Page limit violation is the single largest controllable risk
- Missing experiments is the single largest quality gap
- The W1+W6 compounding pair (scope + density) could pull Reviewer B to a 2 (Reject)
- Single author / independent researcher may face implicit bias (not capturable from data)

**Upside scenario ({5, 5, 5}, poster/spotlight):** If the paper is compressed to 9 pages with 7 focused results AND includes basic MMLU experiments, all three reviewer archetypes find enough to praise. The $\varepsilon$-dependent minimax rate is a clean contribution that sits well at NeurIPS.

**Downside scenario ({2, 4, 5}, reject):** If submitted at 15 pages or with no experiments, Reviewer B scores 2 (Reject: "no experiments, violates page limit"), Reviewer A scores 4 (Borderline: "interesting theory but unfocused"), Reviewer C scores 5 (Accept: "important for policy"). AC sides with Reviewer B due to format violation.
