# THE VERIFICATION TAX — CLAUDE.md Contract
# Single source of truth for agent execution. Do not deviate.

## Project Identity
- **Title:** The Verification Tax: Fundamental Limits of AI Auditing in the Rare-Error Regime
- **Target:** ICLR 2027 (deadline ~October 1, 2026)
- **Author:** Jason (solo, no institutional affiliation)
- **Format:** ICLR 2027 LaTeX template, 9pp main + appendix
- **ArXiv preprint:** June 5–8, 2026

## Agent Rules
1. **Typeset only what is specified below.** Do not invent notation, do not add results, do not rephrase theorem statements.
2. **Every symbol is frozen.** If something seems inconsistent, flag it — do not silently fix it.
3. **LaTeX quality:** Use `\newcommand` for all repeated notation. No hardcoded constants in theorem bodies.
4. **Proof structure:** Main text gets proof sketches (≤1 page each). Full proofs go in appendix.
5. **No placeholder text.** If a section is not yet specified, leave a `% TODO: [description]` comment and move on.

---

## FROZEN NOTATION

```
\newcommand{\ECE}{\mathrm{ECE}}
\newcommand{\CE}{\mathrm{CE}}
\newcommand{\cB}{\mathcal{B}}            % number of bins
\newcommand{\cF}{\mathcal{F}}            % function class
\newcommand{\cS}{\mathcal{S}}            % sigma-algebra (white-box)
\newcommand{\Lip}{L}                      % Lipschitz constant
\newcommand{\errrate}{\varepsilon}        % error rate
\newcommand{\nsamples}{m}                 % number of holdout samples
\newcommand{\calmapraw}{\eta}             % calibration function η(p) = E[Y|f(X)=p]
\newcommand{\calmap}{\Delta}              % calibration gap Δ(p) = η(p) - p
\newcommand{\scoredist}{\mu}              % score distribution
\newcommand{\ind}{\mathbb{1}}             % indicator
\newcommand{\E}{\mathbb{E}}              % expectation
\newcommand{\Var}{\mathrm{Var}}
\newcommand{\KL}{\mathrm{KL}}
\newcommand{\TV}{\mathrm{TV}}
\newcommand{\Bern}{\mathrm{Bern}}
\newcommand{\FisherI}{\mathcal{I}}       % Fisher information
\newcommand{\logit}{\mathrm{logit}}      % logit function
\newcommand{\sigmoid}{\sigma}             % sigmoid function
\newcommand{\shiftfn}{h}                  % fine-tuning shift function h = Δ₂ - Δ₁
\newcommand{\ftshift}{\delta_{\mathrm{ft}}} % fine-tuning shift bound
\newcommand{\Liph}{L_h}                   % Lipschitz constant of shift function
\newcommand{\activeR}{R^*_{\mathrm{active}}}  % active minimax risk
\newcommand{\selfR}{R^*_{\mathrm{self}}}      % self-verification risk
\newcommand{\veriffunc}{V}                % verification functional
\newcommand{\effnoise}{\sigma^2}          % effective noise variance
\newcommand{\effsamples}{m_{\mathrm{eff}}} % effective sample size
\newcommand{\stoppingtime}{\tau}          % sequential stopping time
\newcommand{\groupprop}{\pi}              % group proportion
\newcommand{\targetacc}{\delta}           % target accuracy
\newcommand{\confalpha}{\alpha}           % confidence level
\newcommand{\modelsize}{N}                 % model parameter count
\newcommand{\totaldata}{M_{\mathrm{total}}} % total available labeled data
\newcommand{\horizonN}{N^*}                % verification horizon critical model size
\newcommand{\scalingexp}{\alpha}            % error rate scaling exponent
\newcommand{\deployrate}{r}                % deployment rate (samples/time)
\newcommand{\verfloor}{\delta_{\mathrm{floor}}} % verification floor
\newcommand{\critexp}{\beta}               % improvement exponent over time
\newcommand{\recalround}{K}                % recalibration round
\newcommand{\maxround}{K^*}                % max verifiable rounds
\newcommand{\shrinkfactor}{\gamma}          % ECE reduction factor per round
\newcommand{\Lsystem}{L_{\mathrm{sys}}}    % system Lipschitz constant
\newcommand{\Lcomp}{L_k}                   % per-component Lipschitz constant
\newcommand{\maxpipedepth}{K_{\mathrm{max}}}   % max verifiable pipeline depth
\newcommand{\halflife}{t_{1/2}}            % verification half-life
\newcommand{\driftrate}{\lambda}            % ECE drift rate (per unit time)
\newcommand{\tvdrift}{\rho}                 % TV distance drift rate
\newcommand{\datrate}{r}                    % data collection rate (samples/time)
```

---

## FROZEN DEFINITIONS

### Definition 1: Binned Expected Calibration Error

Let $f: \mathcal{X} \to \mathcal{Y} \times [0,1]$ be a classifier outputting (prediction, confidence). Let $p(x) = f(x)_{\text{conf}}$ denote the confidence score. Partition $[0,1]$ into $B$ equal-width bins $I_1, \ldots, I_B$ where $I_b = ((b-1)/B, \, b/B]$.

$$\ECE_B = \sum_{b=1}^{B} \frac{n_b}{n} \left| \mathrm{acc}(I_b) - \mathrm{conf}(I_b) \right|$$

where $n_b = |\{i : p(x_i) \in I_b\}|$, $\mathrm{acc}(I_b) = \frac{1}{n_b}\sum_{i: p(x_i) \in I_b} \ind[y_i = \hat{y}_i]$, and $\mathrm{conf}(I_b) = \frac{1}{n_b}\sum_{i: p(x_i) \in I_b} p(x_i)$.

**Primary metric throughout this paper:** Top-label binned ECE with uniform-width binning.

### Definition 2: Calibration Gap Function

$$\Delta(p) = \eta(p) - p, \quad \text{where } \eta(p) = \E[Y = \hat{y} \mid p(X) = p]$$

So $\ECE_B = \sum_b \frac{n_b}{n} |\bar{\Delta}_b|$ where $\bar{\Delta}_b$ is the average calibration gap in bin $b$.

### Definition 3: Error Rate

$$\errrate = \Pr[Y \neq \hat{y}] = \E[1 - \eta(p(X))]$$

For a well-calibrated model, $\errrate = \E[1 - p(X)]$. In the rare-error regime, $\errrate \to 0$, and label variance near high-confidence predictions satisfies $\Var(Y \mid p(X) \approx 1-\errrate) = \errrate(1-\errrate) \approx \errrate$.

### Definition 4: Lipschitz Calibration Function Class

$$\cF(\Lip) = \left\{ \Delta : [0,1] \to [-1,1] \;\middle|\; |\Delta(p_1) - \Delta(p_2)| \leq \Lip |p_1 - p_2| \; \forall\, p_1, p_2 \right\}$$

### Definition 5: Minimax ECE Estimation Risk

$$R^*(\nsamples, \errrate, \Lip) = \inf_{\widehat{\ECE}} \sup_{\Delta \in \cF(\Lip)} \E\left[ \left| \widehat{\ECE} - \ECE(\Delta) \right| \right]$$

where the infimum is over all estimators based on $\nsamples$ i.i.d. labeled samples from the classifier.

### Bridge Lemma (Lemma 1): Binned ECE ↔ Smooth CE

Let $\CE_{\mathrm{smooth}} = \int |\Delta(p)| \, d\scoredist(p)$. Then for $B$ equal-width bins:

$$\ECE_B \leq \CE_{\mathrm{smooth}} \leq \ECE_B + \frac{\Lip}{B}$$

**Proof (3 lines):**
- Lower bound: By Jensen within each bin, $|\E[\Delta(p) \mid p \in I_b]| \leq \E[|\Delta(p)| \mid p \in I_b]$, so $\ECE_B \leq \CE_{\mathrm{smooth}}$.
- Upper bound: Within bin $I_b$ of width $1/B$, $\Delta$ varies by at most $\Lip/B$ (Lipschitz). So $|\Delta(p)| \leq |\bar{\Delta}_b| + \Lip/B$ for all $p \in I_b$. Integrating: $\CE_{\mathrm{smooth}} \leq \ECE_B + \Lip/B$. $\square$

**Purpose:** Justifies using smooth CE for temperature scaling analysis (Proposition P1) while all core bounds use binned ECE.

---

## THEOREMS

### Theorem 1 (Le Cam Lower Bound) — T1

**Statement:** For any $\Lip > 0$ and $\errrate \in (0, 1/2)$,

$$R^*(\nsamples, \errrate, \Lip) \geq c_1 \sqrt{\frac{\errrate}{\nsamples}}$$

where $c_1 > 0$ is a universal constant. In particular, $c_1 \geq 1/(4\sqrt{2}) \approx 0.177$.

**Proof (main text, full):**

Construct two hypotheses. Fix a classifier whose score distribution places all mass at $p_0 = 1 - \errrate$.

- $P_0$: $Y \mid p(X) = p_0 \sim \Bern(1 - \errrate)$. Calibrated: $\eta(p_0) = p_0$, so $\ECE = 0$.
- $P_1$: $Y \mid p(X) = p_0 \sim \Bern(1 - \errrate - \delta)$. Miscalibrated: $\ECE = \delta$.

Both are consistent with $\Delta \in \cF(\Lip)$ for any $\delta \leq \Lip$ (the calibration gap is constant, trivially Lipschitz).

KL divergence of $\nsamples$ i.i.d. samples:

$$\KL(P_0^{\nsamples} \| P_1^{\nsamples}) = \nsamples \cdot \KL(\Bern(1-\errrate) \| \Bern(1-\errrate-\delta)) = \frac{\nsamples \delta^2}{\errrate(1-\errrate)} \leq \frac{\nsamples \delta^2}{\errrate}$$

where the last inequality uses $1 - \errrate \leq 1$ and the KL approximation $\KL(\Bern(p) \| \Bern(p+\delta)) \approx \delta^2 / (p(1-p))$ for small $\delta$.

By Le Cam's lemma (Tsybakov, 2009, Theorem 2.2):

$$R^* \geq \frac{\delta}{2}\left(1 - \TV(P_0^{\nsamples}, P_1^{\nsamples})\right) \geq \frac{\delta}{2} e^{-\KL(P_0^{\nsamples} \| P_1^{\nsamples})}$$

where the second inequality uses Bretagnolle-Huber. Set $\KL = 1$, i.e., $\delta = \sqrt{\errrate / \nsamples}$:

$$R^* \geq \frac{1}{2e} \sqrt{\frac{\errrate}{\nsamples}} \approx 0.184 \sqrt{\frac{\errrate}{\nsamples}} \qquad \square$$

**Note for agent:** Also compute the exact Bernoulli KL (not quadratic approximation) numerically for $\errrate \in \{0.01, 0.05, 0.10, 0.20, 0.35\}$ and report the exact constant $c_1(\errrate)$ in a table in the appendix.

---

### Theorem 2 (Minimax Lower Bound via Reduction to L₁ Functional Estimation) — T2

**Statement:** For the class of $\Lip$-Lipschitz calibration functions with error rate $\errrate$, the minimax ECE estimation error satisfies:

$$R^*(\nsamples, \errrate, \Lip) \geq c_2 \left(\frac{\Lip \errrate}{\nsamples}\right)^{1/3} \cdot (\log \nsamples)^{-c_3}$$

where $c_2, c_3 > 0$ are universal constants.

**Proof strategy (main text sketch, ~1 page; full proof in Appendix A):**

**Step 1: Reduction to nonparametric regression.**

Within a window of width $w$ around $p = 1 - \errrate$, where the score distribution concentrates, the calibration verification problem reduces to estimating the L₁ norm of a regression function. Specifically:

For $p \in [1-\errrate - w/2, \, 1-\errrate + w/2]$, define $\Delta(p) = \eta(p) - p$. The labels $Y_i$ given $p(X_i) = p$ are $\Bern(\eta(p)) = \Bern(p + \Delta(p))$, with variance $\eta(p)(1 - \eta(p)) \approx \errrate$ in the rare-error regime.

The ECE restricted to this window is:

$$\ECE_{\text{window}} = \int_{1-\errrate-w/2}^{1-\errrate+w/2} |\Delta(p)| \, d\scoredist(p)$$

This is precisely the L₁ norm of $\Delta$ on the window, weighted by $\scoredist$.

**Step 2: Identification with L₁ functional estimation in white noise.**

By the asymptotic equivalence of nonparametric regression and Gaussian white noise (Brown & Low, 1996; Nussbaum, 1996), estimating $\int |\Delta(p)| dp$ from $\nsamples$ observations with per-observation variance $\errrate$ is asymptotically equivalent to estimating $\int |f(t)| dt$ in the white noise model:

$$dX(t) = f(t) dt + (\nsamples / \errrate)^{-1/2} dW(t), \quad t \in [0,1]$$

with effective sample size $n_{\mathrm{eff}} = \nsamples / \errrate$ and $f \in \Sigma(\beta=1, \Lip)$ (Lipschitz = Hölder with $\beta = 1$).

**Step 3: Application of Lepski, Nemirovski & Spokoiny (1999).**

By Theorem 2.2 of Lepski et al. (1999), the minimax rate for estimating the L₁ norm of a $\beta$-Hölder function in white noise with sample size $n$ is:

$$R^*_{\text{L}_1} \geq c \cdot n^{-\beta/(2\beta+1)} / (\log n)^{c'}$$

For $\beta = 1$ (Lipschitz) and $n = n_{\mathrm{eff}} = \nsamples / \errrate$:

$$R^*_{\text{L}_1} \geq c \cdot (\nsamples / \errrate)^{-1/3} / (\log(\nsamples/\errrate))^{c'} = c \cdot \left(\frac{\errrate}{\nsamples}\right)^{1/3} / (\log \nsamples)^{c'}$$

Incorporating the Lipschitz constant $\Lip$ (which scales the function class):

$$R^*(\nsamples, \errrate, \Lip) \geq c_2 \left(\frac{\Lip \errrate}{\nsamples}\right)^{1/3} / (\log \nsamples)^{c_3} \qquad \square$$

**Technical conditions for Step 2 (must verify in appendix):**

- Score-mass concentration: The score distribution $\scoredist$ must place mass $\geq \rho$ in a window of width $w$ around $1 - \errrate$. State as an assumption. Argue in a remark that this holds for well-trained classifiers (overconfident models concentrate predictions near the top).
- The Brown-Low equivalence requires $\nsamples$ sufficiently large relative to the smoothness. State the minimum $\nsamples$ explicitly.
- The Lipschitz constant $\Lip$ of $\Delta(p)$ must be related to the Lipschitz constant of $\eta(p)$. Since $\Delta(p) = \eta(p) - p$ and $p \mapsto p$ is 1-Lipschitz, we have $\Lip_\Delta \leq \Lip_\eta + 1$. Use $\Lip$ throughout for $\Lip_\Delta$.

---

### Theorem 3 (Matched Upper Bound) — T3

**Statement:** Histogram binning with $B^* = \left\lfloor (\Lip^2 \nsamples / \errrate)^{1/3} \right\rfloor$ bins achieves:

$$\E\left[ \left| \widehat{\ECE}_{B^*} - \ECE \right| \right] \leq C \left( \frac{\Lip \errrate}{\nsamples} \right)^{1/3}$$

where $C > 0$ depends only on the score distribution $\scoredist$.

**Proof (main text, ~1 page):**

**Bias-variance decomposition.** With $B$ equal-width bins and $\nsamples$ total samples:

1. **Bias (approximation error):** Within each bin of width $1/B$, the calibration gap $\Delta$ varies by at most $\Lip/B$ (Lipschitz). The binned ECE approximation error is:

$$|\ECE_B - \CE_{\mathrm{smooth}}| \leq \frac{\Lip}{B}$$

(This is the Bridge Lemma.)

2. **Variance (estimation error):** Each bin $b$ has $n_b \approx \nsamples/B$ samples (assuming roughly uniform score distribution). The empirical calibration gap $\hat{\Delta}_b$ has:

$$\Var(\hat{\Delta}_b) = \frac{\eta_b(1-\eta_b)}{n_b} \approx \frac{\errrate}{\nsamples/B} = \frac{\errrate B}{\nsamples}$$

The estimation error of the full ECE is:

$$\E\left[|\widehat{\ECE}_B - \ECE_B|\right] \leq \sqrt{\frac{\errrate B}{\nsamples}}$$

(This uses Jensen and the independence across bins for the dominant term.)

3. **Total error:**

$$\E\left[|\widehat{\ECE}_{B} - \CE_{\mathrm{smooth}}|\right] \leq \frac{\Lip}{B} + \sqrt{\frac{\errrate B}{\nsamples}}$$

4. **Optimization over $B$:** Set $\Lip/B = \sqrt{\errrate B / \nsamples}$, solve:

$$\frac{\Lip^2}{B^2} = \frac{\errrate B}{\nsamples} \implies B^3 = \frac{\Lip^2 \nsamples}{\errrate} \implies B^* = \left(\frac{\Lip^2 \nsamples}{\errrate}\right)^{1/3}$$

5. **Substituting back:**

$$\text{Error} = \frac{\Lip}{B^*} = \Lip \cdot \left(\frac{\errrate}{\Lip^2 \nsamples}\right)^{1/3} = \left(\frac{\Lip \errrate}{\nsamples}\right)^{1/3} \qquad \square$$

**Note:** The upper bound has NO logarithmic factor. The log gap between T2 and T3 is inherited from the open problem in L₁ functional estimation (Lepski et al., 1999). State this explicitly in one sentence after T3.

---

### General Verification Tax Theorem — TG

**Definition (Verification Functional):** A *verification functional* is any quantity of the form:

$$V(\Delta, \mu_G) = \int |\Delta(p)| \, d\mu_G(p)$$

where $\Delta: [0,1] \to \mathbb{R}$ is an $\Lip$-Lipschitz deviation function and $\mu_G$ is a (possibly reweighted) score distribution over a subpopulation or domain $G$.

**Instantiations:** (i) *Calibration*: $G$ = full population, $\Delta(p) = \eta(p) - p$, $V = \ECE$. (ii) *Group fairness*: for group $A$ with proportion $\pi_A$, effective sample size $m_{\mathrm{eff}} = m\pi_A$. (iii) *Robustness*: under input perturbation, shifted scores with shifted calibration gap.

**Statement:** For any verification functional $V$ with $\Lip$-Lipschitz integrand, effective noise variance $\sigma^2$, and effective sample size $m_{\mathrm{eff}}$:

(i) *Le Cam bound:* $R^*(m_{\mathrm{eff}}, \sigma^2, \Lip) \geq c\sqrt{\sigma^2/m_{\mathrm{eff}}}$

(ii) *Minimax rate:* $R^*(m_{\mathrm{eff}}, \sigma^2, \Lip) = \Theta((\Lip\sigma^2/m_{\mathrm{eff}})^{1/3})$ up to logarithmic factors

(iii) *Active rate:* $R^*_{\mathrm{active}}(m_{\mathrm{eff}}, \sigma^2, \Lip) = \Theta(\sqrt{\sigma^2/m_{\mathrm{eff}}})$

**Proof (3 lines):** The proofs of T1–T3 and TA use only three properties: (a) the integrand is $\Lip$-Lipschitz, (b) observations are Bernoulli with variance $\sigma^2$, (c) the functional is an $L_1$ norm. All verification functionals share this structure with $\varepsilon$ replaced by $\sigma^2$ and $m$ replaced by $m_{\mathrm{eff}}$. $\square$

**Corollary (Fairness Verification Tax):** For a group with proportion $\pi$: $m \geq (1/\pi) \cdot L\varepsilon/\delta^3$. For a 5% minority group, this is 20× the cost of the full population.

**Corollary (Robustness Verification Tax):** Under perturbation with error rate $\varepsilon_{\mathrm{pert}} > \varepsilon$, robustness verification is strictly more expensive than standard calibration verification.

---

### Theorem 4 (White-Box Irrelevance) — T4

**Statement:** Let $\cS = \sigma(\theta, A, D_{\mathrm{train}})$ be the sigma-algebra generated by model weights $\theta$, architecture $A$, and training data $D_{\mathrm{train}}$. For any estimator that is $\cS$-measurable and uses $\nsamples$ fresh labeled samples, the minimax ECE over $\cF(\Lip)$ satisfies:

$$R^*_{\cS}(\nsamples, \errrate, \Lip) \geq c \sqrt{\frac{\errrate}{\nsamples}}$$

That is, white-box access does not improve the minimax rate beyond the black-box Le Cam bound (T1).

**Proof (main text, full, ~0.5 page):**

The proof constructs two worlds with the *identical model* but different label distributions.

Fix a model — same weights $\theta$, same architecture $A$, same training data $D_{\mathrm{train}}$. This model assigns confidence scores $p(x)$ to inputs. The sigma-algebra $\cS = \sigma(\theta, A, D_{\mathrm{train}})$ is identical under both hypotheses. Any $\cS$-measurable estimator sees exactly the same model internals under both.

Define two worlds:

- $P_0$: $Y \mid p(X) = p \sim \Bern(p)$. The model is perfectly calibrated. $\ECE = 0$.
- $P_1$: $Y \mid p(X) = p \sim \Bern(p - \delta)$ for $p$ near $1 - \errrate$. The model is miscalibrated by $\delta$. $\ECE = \delta$.

The model is identical in both worlds — same weights, same predictions, same confidence scores. The only difference is whether the labels agree with the confidences at rate $p$ (world $P_0$) or at rate $p - \delta$ (world $P_1$).

An $\cS$-measurable estimator receives no additional information for distinguishing $P_0$ from $P_1$, because $\cS$ is identical under both. The only distinguishing information comes from the $\nsamples$ fresh labeled samples.

Formally: conditioned on $\cS$ and on the test inputs $x_1, \ldots, x_{\nsamples}$ (which determine scores $p_1, \ldots, p_{\nsamples}$), the labels $y_1, \ldots, y_{\nsamples}$ are independent Bernoullis. Under $P_0$ they are $\Bern(p_i)$; under $P_1$ they are $\Bern(p_i - \delta)$ for relevant $p_i$. The conditional KL divergence is:

$$\KL(P_0^{\nsamples} \| P_1^{\nsamples} \mid \cS) = \sum_{i: p_i \approx 1-\errrate} \KL(\Bern(p_i) \| \Bern(p_i - \delta)) \approx \frac{\nsamples \delta^2}{\errrate}$$

identical to T1. Applying Le Cam's lemma to the conditional problem:

$$R^*_{\cS} \geq \frac{\delta}{2} e^{-\KL} = c \sqrt{\frac{\errrate}{\nsamples}} \qquad \square$$

**Corollary (Parametric Exception):** If $\cS$ reveals that $\Delta \in \cF_d$ for a known $d$-dimensional parametric family (e.g., temperature scaling: $\Delta(p) = \sigma(\mathrm{logit}(p)/T) - p$ with unknown $T \in \mathbb{R}$), then the function class shrinks from infinite-dimensional $\cF(\Lip)$ to a $d$-dimensional manifold, and verification requires only $\nsamples = O(d/\errrate)$ samples. The rate improves from $(\Lip\errrate/\nsamples)^{1/3}$ to the parametric $O(1/\sqrt{\nsamples})$.

White-box access helps not through the model internals per se, but by constraining the calibration function class. If the parametric form is not trusted, the nonparametric rate applies regardless of model access.

**Regulatory implication (one sentence in conclusion):** An AI auditor who can inspect model weights has no fundamental advantage over a black-box auditor, unless the developer provides (and the auditor trusts) a parametric specification of the calibration structure.

---

## PROPOSITIONS (proved)

### Proposition P1 (Temperature Scaling Rate)

**Statement:** Under correct specification (the calibration map belongs to the temperature scaling family $\Delta(p; T) = \sigma(\mathrm{logit}(p)/T) - p$ with true parameter $T^* = 1$), the MLE $\hat{T}$ on $\nsamples$ samples achieves:

$$\CE_{\mathrm{smooth}} = O_p\left(\frac{1}{\sqrt{\nsamples}}\right)$$

for the smooth calibration error. Via the Bridge Lemma, $\ECE_B = O(1/\sqrt{\nsamples} + \Lip/B)$.

**Proof (main text, ~0.75 page):**

**Step 1: MLE for the temperature parameter.**

Given scores $p_1, \ldots, p_{\nsamples}$ and labels $y_1, \ldots, y_{\nsamples}$, the negative log-likelihood is:

$$\ell(T) = -\sum_{i=1}^{\nsamples} \left[ y_i \log \sigma(\mathrm{logit}(p_i)/T) + (1 - y_i) \log(1 - \sigma(\mathrm{logit}(p_i)/T)) \right]$$

This is a 1D convex optimization. The MLE $\hat{T}$ is unique.

**Step 2: Asymptotic normality of MLE.**

By standard M-estimation theory (van der Vaart, 1998, Chapter 5):

$$\sqrt{\nsamples}(\hat{T} - T^*) \xrightarrow{d} N(0, I(T^*)^{-1})$$

where $I(T^*)$ is the Fisher information. At $T^* = 1$, the score function is:

$$\frac{\partial}{\partial T} \log p(Y \mid p, T) \bigg|_{T=1} = -(Y - p) \cdot \mathrm{logit}(p)$$

So:

$$I(1) = \E[(Y - p)^2 \cdot \mathrm{logit}(p)^2] = \E[p(1-p) \cdot \mathrm{logit}(p)^2]$$

In the rare-error regime with scores concentrated near $1 - \errrate$: $\mathrm{logit}(1-\errrate) \approx \log(1/\errrate)$ and $p(1-p) \approx \errrate$, giving:

$$I(1) \approx \errrate \cdot \log^2(1/\errrate)$$

This is positive and grows as $\errrate \to 0$ — rare errors make temperature estimation EASIER because logits are more spread out.

The MLE satisfies: $|\hat{T} - T^*| = O_p(1/\sqrt{\nsamples \cdot I(T^*)})$.

**Step 3: Translating to calibration error.**

By Taylor expansion of $\Delta(p; T)$ around $T^* = 1$:

$$\Delta(p; \hat{T}) \approx \frac{\partial \Delta}{\partial T}\bigg|_{T=1} \cdot (\hat{T} - 1) = p(1-p) \cdot \mathrm{logit}(p) \cdot (\hat{T} - 1)$$

So the smooth CE is:

$$\CE_{\mathrm{smooth}} = \int |\Delta(p; \hat{T})| \, d\scoredist(p) \approx |\hat{T} - 1| \cdot \underbrace{\int p(1-p) |\mathrm{logit}(p)| \, d\scoredist(p)}_{\kappa}$$

where $\kappa$ is a constant depending on the score distribution. Substituting $|\hat{T} - 1| = O_p(1/\sqrt{\nsamples \cdot I(T^*)})$:

$$\CE_{\mathrm{smooth}} = O_p\left(\frac{\kappa}{\sqrt{\nsamples \cdot I(T^*)}}\right) = O_p\left(\frac{1}{\sqrt{\nsamples}}\right)$$

with a constant that depends on $\errrate$ through $I(T^*)$ and $\kappa$. The key point: this is the **parametric rate** $\nsamples^{-1/2}$, not the nonparametric rate $(\Lip\errrate/\nsamples)^{1/3}$.

**Step 4: Connection to binned ECE.**

Via the Bridge Lemma: $\ECE_B \leq \CE_{\mathrm{smooth}} + \Lip/B$. Taking $B = O(\sqrt{\nsamples}/\Lip)$:

$$\ECE_B = O(1/\sqrt{\nsamples})$$

**Comparison to nonparametric rate:** The ratio is $(1/\sqrt{\nsamples}) / (\Lip\errrate/\nsamples)^{1/3} = (\Lip\errrate)^{-1/3} / \nsamples^{1/6}$, which shrinks with $\nsamples$. For $\nsamples = 10{,}000$ and $\Lip\errrate = 0.05$: ratio $\approx 0.05$. Temperature scaling is $\sim$20$\times$ more data-efficient than nonparametric ECE estimation.

**Caveat (state in paper):** Under *misspecification* (temperature scaling is wrong), the MLE converges to the best approximation in the family, and the residual miscalibration is not captured. The $1/\sqrt{\nsamples}$ rate applies to the parametric error only. Any misspecification bias persists regardless of $\nsamples$. This is the price of the parametric assumption.

**Narrative role:** T1–T3 say verification is fundamentally expensive. P1 says: unless you assume a parametric form — then it's cheap. T4 ties them together: white-box access helps only insofar as it justifies the parametric assumption.

---

### Proposition P2 (Verification Transfer)

**Statement:** Let $M_1$ have verified calibration (ECE known to accuracy $\delta_1$). Let $M_2$ be obtained from $M_1$ by a procedure that shifts the calibration gap by at most $\delta_{\mathrm{ft}}$ in $L^\infty$, with the shift function $h = \Delta_2 - \Delta_1$ being $\Lip_h$-Lipschitz. To verify $M_2$'s ECE to accuracy $\delta$:

(i) **From scratch (no transfer):** $\nsamples_2 = \Theta(\Lip \errrate_2 / \delta^3)$ \quad [from T3]

(ii) **With transfer:** $\nsamples_2 = \Theta(\Lip_h \errrate_2 / \delta_{\mathrm{ft}}^3)$ suffices, provided $\delta_{\mathrm{ft}} \leq \delta$.

**Transfer saving factor:** $(\Lip / \Lip_h) \cdot (\delta / \delta_{\mathrm{ft}})^3$

**Proof (main text, ~0.75 page):**

**Step 1: Problem reduction.**

Since $\Delta_1(p)$ is known (from prior verification), estimating $\ECE(M_2) = \int |\Delta_2(p)| \, d\scoredist(p)$ reduces to estimating the shift $h(p) = \Delta_2(p) - \Delta_1(p)$, then computing $\int |\Delta_1(p) + \hat{h}(p)| \, d\scoredist(p)$.

**Step 2: Lower bound (detection cost).**

Two-sample Le Cam between:
- $H_0$: $M_2$ has calibration gap $\Delta_2 = \Delta_1$ (fine-tuning didn't change calibration)
- $H_1$: $\Delta_2 = \Delta_1 + h$ with $\|h\|_\infty = \delta_{\mathrm{ft}}$

From $\nsamples_2$ fresh samples of $M_2$ near $p \approx 1 - \errrate_2$:

$$\KL(H_0^{\nsamples_2} \| H_1^{\nsamples_2}) \approx \frac{\nsamples_2 \delta_{\mathrm{ft}}^2}{\errrate_2}$$

Indistinguishable when $\KL \leq 1$, i.e., $\nsamples_2 \leq \errrate_2 / \delta_{\mathrm{ft}}^2$. So detecting any shift requires $\nsamples_2 = \Omega(\errrate_2 / \delta_{\mathrm{ft}}^2)$.

**Step 3: Upper bound (estimation with transfer).**

Apply T3 (histogram binning) to the function $h$ instead of $\Delta_2$. The observations are fresh labels from $M_2$; the target is $h(p) = \eta_2(p) - \eta_1(p)$, where $\eta_1$ is known. The histogram estimator for $h$ has:

- Bias: $\Lip_h / B$ (Lipschitz constant of $h$, not of $\Delta_2$)
- Variance: $\sqrt{\errrate_2 B / \nsamples_2}$

Optimizing over $B$: $B^* = (\Lip_h^2 \nsamples_2 / \errrate_2)^{1/3}$, giving estimation error:

$$\|\hat{h} - h\| = O\left(\left(\frac{\Lip_h \errrate_2}{\nsamples_2}\right)^{1/3}\right)$$

The ECE estimate is $\int |\Delta_1(p) + \hat{h}(p)| \, d\scoredist(p)$. By the triangle inequality:

$$\left| \ECE(M_2) - \widehat{\ECE}(M_2) \right| \leq \int |\hat{h}(p) - h(p)| \, d\scoredist(p) = O\left(\left(\frac{\Lip_h \errrate_2}{\nsamples_2}\right)^{1/3}\right)$$

Setting this equal to $\delta_{\mathrm{ft}}$ and solving: $\nsamples_2 = O(\Lip_h \errrate_2 / \delta_{\mathrm{ft}}^3)$.

**Step 4: When does transfer help?**

From-scratch verification (T3): $\nsamples_2 = \Theta(\Lip \errrate_2 / \delta^3)$.

With transfer: $\nsamples_2 = \Theta(\Lip_h \errrate_2 / \delta_{\mathrm{ft}}^3)$.

Transfer helps when $\delta_{\mathrm{ft}} < \delta$ AND $\Lip_h \leq \Lip$. With a gentle fine-tune ($\delta_{\mathrm{ft}} = \delta/10$, $\Lip_h = \Lip$), the saving is $(10)^3 = 1000\times$ fewer samples.

Transfer does NOT help when $\delta_{\mathrm{ft}} \geq \delta$ (fine-tuning changes calibration by more than target accuracy). In this case, the prior verification is useless and you start from scratch.

**Practical interpretation:** If you've verified GPT-4 and the developer releases GPT-4.1 (minor fine-tune with small $\delta_{\mathrm{ft}}$), you don't need a full re-audit — a small number of fresh samples suffices. But if the update involves major changes (RLHF, domain adaptation), $\delta_{\mathrm{ft}}$ is large and prior verification provides no benefit.

**Regulatory implication (one sentence in conclusion):** This result supports a "continuous auditing" framework where regular small updates require proportionally small verification budgets, rather than a full re-audit at each model release.

---

### Proposition P3 (Ungameability of the Verification Tax) — P3

**Statement:** Let a developer choose any $\Delta \in \cF(\Lip)$ with $\ECE(\Delta) = \delta$, seeking to minimize the auditor's ability to detect miscalibration. The minimax value of this verification game satisfies:

$$\inf_{\widehat{\ECE}} \sup_{\substack{\Delta \in \cF(\Lip) \\ \ECE(\Delta) = \delta}} \E\left[|\widehat{\ECE} - \ECE(\Delta)|\right] = \Theta\left(\sqrt{\frac{\errrate}{\nsamples}}\right)$$

The developer's optimal strategy is constant miscalibration $\Delta(p) = \delta$ near $p = 1 - \errrate$. No sophisticated adversarial pattern improves upon this.

**Proof (6 lines):** The lower bound is T1: the Le Cam construction already uses constant $\Delta$, which is the hardest case. For the converse, any $\Delta$ with $\ECE = \delta$ satisfies $\|\Delta\|_1 = \delta$ and $\|\Delta\|_2 \geq \|\Delta\|_1$ with equality iff $\Delta$ is constant on its support. The KL is $m\|\Delta\|_2^2 / \varepsilon \geq m\delta^2/\varepsilon$, with equality for constant $\Delta$. Constant miscalibration minimizes KL, making detection hardest. Any non-constant $\Delta$ with the same ECE is *easier* to detect. $\square$

**Remark:** Auditors need not design detection protocols for adversarial miscalibration patterns. The standard holdout-based ECE estimator is robust to strategic behavior by the model developer.

---

### Theorem A (Active Verification Rate) — TA

**Statement:** For the class of $\Lip$-Lipschitz calibration functions with error rate $\errrate$, adaptive verification — where the auditor sequentially selects confidence levels $p_1, p_2, \ldots, p_{\nsamples}$, with each choice depending on all previous observations — achieves minimax rate:

$$R^*_{\mathrm{active}}(\nsamples, \errrate, \Lip) = \Theta\left(\sqrt{\frac{\errrate}{\nsamples}}\right)$$

In particular: (i) no adaptive strategy can beat $c\sqrt{\errrate/\nsamples}$, and (ii) a two-phase explore-exploit strategy achieves $C\sqrt{\errrate/\nsamples}$, with universal constants $c, C > 0$.

**Proof (main text, full):**

**Lower bound.** The Le Cam lower bound from T1 applies unchanged to adaptive strategies. Under adaptive querying, the auditor selects $p_i$ based on all previous observations $(p_1, y_1, \ldots, p_{i-1}, y_{i-1})$. By the chain rule of KL divergence:

$$\KL(P_0^{\nsamples} \| P_1^{\nsamples}) = \sum_{i=1}^{\nsamples} \E\left[\KL(P_0(\cdot \mid p_i) \| P_1(\cdot \mid p_i))\right]$$

Each term is at most $\delta^2 / \errrate$. Adaptivity does not increase the per-observation information: conditioned on $p_i$, the label $y_i$ is a single Bernoulli draw whose KL is $\delta^2 / \errrate$ regardless of previous observations. Therefore $\KL \leq \nsamples \delta^2 / \errrate$, and Le Cam gives $R^*_{\mathrm{active}} \geq c\sqrt{\errrate / \nsamples}$.

**Upper bound.** Two-phase explore-exploit strategy:

*Phase 1 (Exploration):* Select $N$ equally-spaced confidence levels in $[1 - \errrate - w, 1 - \errrate + w]$. Query each level $\nsamples/(2N)$ times. Classify bins as *resolved* ($|\hat{\Delta}(p_j)| > 2\sigma_j$, sign known) or *unresolved* ($|\hat{\Delta}(p_j)| \leq 2\sigma_j$).

*Phase 2 (Exploitation):* Concentrate remaining $\nsamples/2$ queries on resolved bins where $|\Delta|$ can be estimated without the absolute value problem.

Choose $N = \Lip\sqrt{\nsamples/\errrate}$ to balance resolved-bin estimation error ($O(\sqrt{\errrate/\nsamples})$) with unresolved-zone contribution ($O(\errrate N / (\nsamples \Lip))$). Both become $O(\sqrt{\errrate/\nsamples})$. $\square$

**Remark (Passive vs. Active Gap):** The gap is $(\Lip^2 \nsamples / \errrate)^{1/6}$. For $\Lip = 1$, $\errrate = 0.05$, $\nsamples = 10{,}000$, active verification is ~8× more sample-efficient. The Lipschitz constant $\Lip$ disappears entirely from the active rate.

---

### Theorem S (Sequential Verification) — TS

**Definition (Sequential Verification Protocol):** A sequential verification protocol is a pair $(\tau, \widehat{\ECE}_\tau)$ where $\tau$ is a stopping time adapted to $\mathcal{F}_t = \sigma(p_1, y_1, \ldots, p_t, y_t)$ and $\widehat{\ECE}_\tau$ is the ECE estimate at time $\tau$. The protocol has accuracy $\delta$ at confidence level $1 - \alpha$ if $\Pr(|\widehat{\ECE}_\tau - \ECE| > \delta) \leq \alpha$.

**Statement:** For a model with calibration gap $\Delta$ and error rate $\varepsilon$:

(i) *Lower bound:* Any sequential protocol with accuracy $\delta$ at confidence $1 - \alpha$ satisfies: $\E[\tau] \geq c \cdot (\varepsilon/\delta^2) \cdot \log(1/\alpha)$

(ii) *Upper bound:* There exists a sequential protocol achieving this with: $\E[\tau] \leq C \cdot (\varepsilon/\delta^2) \cdot \log(1/\alpha)$

The expected stopping time is $\Theta(\varepsilon / \delta^2 \cdot \log(1/\alpha))$, matching the fixed-sample Le Cam bound with an additional $\log(1/\alpha)$ factor for confidence control.

**Proof sketch:** *Lower bound:* Under Le Cam hypotheses, the log-likelihood ratio after $t$ observations is a random walk with drift $\delta^2/\varepsilon$. By Wald's identity, $\E[\tau] \geq \log(1/\alpha) / (\delta^2/\varepsilon) = \varepsilon\log(1/\alpha)/\delta^2$. *Upper bound:* Confidence sequence via method of mixtures (Howard et al., 2021) with width $O(\sqrt{\varepsilon \log(1/\alpha) / t})$. Stop when width $\leq \delta$, giving $\tau \approx \varepsilon\log(1/\alpha)/\delta^2$. $\square$

**Remark (Instance-Dependent Behavior):** Bad models ($\ECE \gg \delta$) are cheap to catch after $O(\varepsilon/\ECE^2)$ samples. Good models ($\ECE \approx 0$) require the full $O(\varepsilon/\delta^2)$ budget. A regulator auditing 100 models will spend most of the budget on the well-calibrated ones.

---

### Theorem B (Self-Verification Impossibility) — TB

**Statement:** Let $f$ be a classifier with confidence scores $p(x)$ and let $z(x) = g(x, \theta)$ be any auxiliary signal derived from the model (internal activations, MC dropout variance, ensemble disagreement, attention entropy, or any other function of the input and model weights). Any estimator $V(x_1, p_1, z_1, \ldots, x_{\nsamples}, p_{\nsamples}, z_{\nsamples})$ that does not use true labels $y_i$ satisfies:

$$\sup_{\Delta \in \cF(\Lip)} \E\left[|V - \ECE(\Delta)|\right] \geq \frac{\Lip}{2}$$

That is, self-verification without ground truth has worst-case error proportional to the Lipschitz constant, regardless of the number of queries $\nsamples$.

**Proof (main text, short ~0.3 page):**

Fix any model with weights $\theta$. The outputs $(p(x_i), z(x_i))$ are deterministic functions of $x_i$ and $\theta$; they do not depend on $\Delta$. Construct:

- $\Delta_0(p) = 0$ for all $p$. $\ECE(\Delta_0) = 0$.
- $\Delta_1(p) = \delta$ near $1 - \errrate$, extended $\Lip$-Lipschitz. $\ECE(\Delta_1) = \delta$.

Both produce identical observations $(x_i, p_i, z_i)$. Any label-free estimator $V$ returns the same value under both. Therefore:

$$\E[|V - \ECE(\Delta_0)|] + \E[|V - \ECE(\Delta_1)|] \geq |\ECE(\Delta_0) - \ECE(\Delta_1)| = \delta$$

Taking $\delta = \Lip$: $\sup_\Delta \E[|V - \ECE|] \geq \Lip/2$. $\square$

**Corollary (Pseudo-Label Circularity):** Self-labeling with $\hat{y}_i = \arg\max f(x_i)$ yields ECE = 0 for any deterministic classifier. Self-labeling measures internal consistency, not calibration.

**Corollary (Distribution Shift Vulnerability):** A calibration monitor trained on $D_{\mathrm{train}}$ cannot detect miscalibration on $D_{\mathrm{test}}$ without labeled samples from $D_{\mathrm{test}}$.

**Corollary (Regulatory Implication):** Self-reported calibration claims by model developers have no information-theoretic value. Independent auditing with fresh labeled data is necessary, not merely desirable.

---

## THE VERIFICATION HORIZON (Section 5)

### Theorem: Verification Horizon — TH

**Statement:** Let $\varepsilon(N) = c_0 N^{-\alpha}$ be the error rate scaling law, where $N$ is parameter count, $c_0 > 0$, $\alpha > 0$. Let $M_{\mathrm{total}}$ be total available labeled data. Define the *verification floor*:

$$\delta_{\mathrm{floor}}(N) = \left(\frac{L \varepsilon(N)}{M_{\mathrm{total}}}\right)^{1/3} = \left(\frac{Lc_0}{M_{\mathrm{total}}}\right)^{1/3} N^{-\alpha/3}$$

Verification is *meaningful* only when $\delta_{\mathrm{floor}}(N) < \varepsilon(N)$. This holds iff $N < N^*$, where:

$$N^* = \left(\frac{c_0^2 M_{\mathrm{total}}}{L}\right)^{1/(2\alpha)}$$

For $N > N^*$, no verifier can verify calibration to a precision finer than the model's actual ECE. The model enters *unverifiable improvement*.

**Proof (short):** From T3, verification to accuracy $\delta$ requires $m \geq L\varepsilon/\delta^3$. With $m = M_{\mathrm{total}}$, best accuracy is $\delta_{\mathrm{floor}} = (L\varepsilon/M_{\mathrm{total}})^{1/3}$. Meaningful when $\delta_{\mathrm{floor}} < \varepsilon$, giving $\varepsilon > \sqrt{L/M_{\mathrm{total}}}$. Substituting scaling law: $N < (c_0^2 M_{\mathrm{total}}/L)^{1/(2\alpha)} = N^*$. $\square$

**Remark (Quantitative Estimates):** Under Chinchilla scaling ($\alpha \approx 0.5$, $c_0 \approx 1$) with $L = 1$, $M_{\mathrm{total}} = 10^7$: $N^* = 10^7$. Frontier models exceed $10^{11}$ parameters. Active verification shifts horizon to $N^*_{\mathrm{active}} = (c_0 M_{\mathrm{total}}/L)^{1/\alpha}$.

**Corollary (Active Verification Horizon):** Under active verification, the active horizon is $N^*_{\mathrm{active}} = (c_0 M_{\mathrm{total}}/L)^{1/\alpha}$, strictly larger than $N^*$. Ratio $N^*_{\mathrm{active}}/N^* = (c_0 M_{\mathrm{total}}/L)^{1/(2\alpha)}$.

---

### Theorem: Verification Dynamics — TD

**Statement:** Suppose a model is deployed at rate $r$ samples/time, yielding verified labels. At time $t$: $m(t) = rt$, $\varepsilon(t) = c_0 t^{-\beta}$ for improvement exponent $\beta > 0$.

(i) If $\beta < 1/2$: verification *catches up* — meaningful verification eventually possible.

(ii) If $\beta > 1/2$: verification *falls behind* — meaningful verification eventually impossible despite growing data.

(iii) If $\beta = 1/2$: perpetually marginal, depending on $L/(rc_0^2) < 1$.

Critical improvement exponent: $\beta^* = 1/2$.

**Proof (full):** Verification floor at time $t$: $\delta_{\mathrm{floor}}(t) = (Lc_0/r)^{1/3} t^{-(\beta+1)/3}$. Meaningful when $\delta_{\mathrm{floor}}(t) < \varepsilon(t) = c_0 t^{-\beta}$, simplifying to $(L/(rc_0^2))^{1/3} < t^{(1-2\beta)/3}$. Exponent $(1-2\beta)/3$ determines behavior: positive ($\beta < 1/2$) → RHS grows, negative ($\beta > 1/2$) → RHS shrinks, zero ($\beta = 1/2$) → fixed condition. $\square$

**Remark (Interpretation):** $\beta^* = 1/2$ means: verification cost grows as $\varepsilon^{-2}$, data grows linearly. If $\varepsilon$ shrinks faster than $t^{-1/2}$, verification loses the race. This is independent of $L$, $r$, and the verification algorithm.

**Remark (Active Dynamics):** Under active verification, critical exponent shifts to $\beta^*_{\mathrm{active}} = 1$. Adaptive evaluation doubles the critical exponent but does not eliminate the horizon.

---

### Proposition: Recalibration Trap — PR

**Statement:** Consider iterative recalibration reducing ECE by factor $\gamma \in (0,1)$ per round, so $\ECE_K = \gamma^K \cdot \ECE_0$. Verification cost of confirming round $K+1$ produced genuine improvement:

$$m_K = \Omega\left(\frac{\varepsilon}{\gamma^{2K} \cdot \ECE_0^2 \cdot (1-\gamma)^2}\right)$$

Exponential in rounds: $(1/\gamma^2)^K$. Maximum verifiable rounds with budget $M_{\mathrm{total}}$:

$$K^* = \left\lfloor \frac{\log(M_{\mathrm{total}} \cdot (1-\gamma)^2 \cdot \ECE_0^2 / \varepsilon)}{2\log(1/\gamma)} \right\rfloor$$

**Proof (short):** At round $K$, improvement is $\Delta_K = \gamma^K \ECE_0(1-\gamma)$. By T1, distinguishing requires $m \geq \varepsilon/\Delta_K^2$. Setting $m = M_{\mathrm{total}}$ and solving gives $K^*$. $\square$

**Worked Example:** $\gamma = 0.5$, $\varepsilon = 0.05$, $\ECE_0 = 0.10$, $M_{\mathrm{total}} = 14{,}000$ (MMLU): $K^* = 4$. After 4 rounds, MMLU cannot verify further improvement.

**Remark:** Regulatory requirements for "continuous calibration improvement" are infeasible beyond $K^*$ rounds without growing the evaluation budget.

---

### Theorem: Verification Half-Life — THL

**Definition (ECE Drift Rate):** The *ECE drift rate* $\lambda > 0$ bounds the rate of ECE change: $|\ECE(t) - \ECE(0)| \leq \lambda t$. Implied by distribution TV drift rate $\rho$: $\lambda \leq L \cdot \rho$.

**Statement:** Let a model be verified at $t = 0$ with accuracy $\delta$, and let ECE drift rate be $\lambda$.

(i) *Half-life:* Verification becomes invalid after $t_{1/2} = \delta / \lambda$.

(ii) *Perpetual verification:* Re-verifying every $t_{1/2}$ requires sustained data rate $r \geq L\varepsilon\lambda / \delta^4$ samples/time. Below this rate, verification permanently expires.

(iii) *Accelerating drift:* If $\lambda(t)$ is increasing, $t_{1/2}(t) \to 0$ and $r(t) \to \infty$. Eventually no data rate suffices.

**Proof (short):** (i) Triangle inequality: $|\widehat{\ECE} - \ECE(t)| \leq \delta + \lambda t$, exceeds $2\delta$ when $t > \delta/\lambda$. (ii) Cost per re-verification: $m = L\varepsilon/\delta^3$ (from T3). Rate: $r = m/t_{1/2} = L\varepsilon\lambda/\delta^4$. (iii) Immediate. $\square$

**Remark (Domain-Specific Half-Lives):** Medical AI: $t_{1/2} \approx 2$ months, $r \approx 3{,}100$/month. Content moderation: 6 months, 620/month. Financial trading: 17 hours, $10^6$/week. Autonomous driving: 15 days, 100{,}000/month. (Assumes $L = 1$, $\varepsilon = 0.05$.)

**Remark (Benchmark Half-Lives):** Static benchmarks also decay at rate $\lambda$. Formal argument for periodic benchmark refresh.

---

## COMPOSITIONAL VERIFICATION (Section 6)

### Theorem: Compositional Verification Tax — TC

**Definition (Compositional AI Pipeline):** A $K$-component pipeline produces confidence $p_K$ through: $p_k = g_k(p_{k-1}, x)$, $k = 1, \ldots, K$, where each $g_k$ is $L_k$-Lipschitz in its first argument.

**Statement:** The end-to-end Lipschitz constant is $L_{\mathrm{sys}} \leq \prod_{k=1}^K L_k + 1$. End-to-end verification to accuracy $\delta$ requires:

$$m_{\mathrm{sys}} \geq c \cdot \frac{L_{\mathrm{sys}} \cdot \varepsilon}{\delta^3}$$

For homogeneous pipelines ($L_k = L > 1$): $m_{\mathrm{sys}} = \Omega(L^K \cdot \varepsilon / \delta^3)$. Exponential in pipeline depth.

**Proof (~0.5 page):** Chain rule: composition of $L_k$-Lipschitz functions is $(\prod L_k)$-Lipschitz. Triangle inequality: $L_{\mathrm{sys}} \leq \prod L_k + 1$. Apply T3 with $L = L_{\mathrm{sys}}$. $\square$

**Corollary (Maximum Verifiable Depth):** $K_{\mathrm{max}} = \lfloor \log_L (M_{\mathrm{total}} \cdot \delta^3 / \varepsilon) \rfloor$.

**Corollary (Agent Loop Bound):** $K = 10$ iterations with $L = 2$: verification cost $1{,}024\times$ single-step. $K = 20$: over $10^6\times$.

**Remark (Compositionality–Verifiability Tradeoff):** Longer reasoning chains become exponentially harder to verify. Applies to chain-of-thought, iterative refinement, multi-hop retrieval, agentic loops.

**Remark (Lipschitz Assumption):** The Lipschitz model is optimistic. Discontinuous transformations (e.g., retriever returning different documents) have $L_k = \infty$, making end-to-end verification impossible at any sample size.

---

## COROLLARIES (one-line derivations from theorems)

### Corollary 1: Phase Transition

Holdout-based calibration assessment (multi-bin) beats the trivial calibrator $\hat{c}(p) = 1 - \hat{\errrate}$ (single-bin) if and only if $\nsamples \gtrsim 1/\errrate$. [From T1: when $\sqrt{\errrate/\nsamples} > \errrate$, i.e., $\nsamples < 1/\errrate$, even detecting miscalibration is impossible.]

### Corollary 2: Fairness Extension

For a demographic group with population proportion $\pi$, verifying equalized odds requires:

$$\nsamples \geq \frac{1}{\errrate \cdot \pi}$$

[From T1 applied to the subgroup with effective sample size $\nsamples \cdot \pi$.]

### Corollary 3: Verification Scaling Law

If model capability scales as $\errrate(N) = c_0 N^{-\alpha}$ (where $N$ is parameter count), then verification requires:

$$\nsamples(N) = \Omega\left(N^{\alpha}\right)$$

samples. Verification cost grows polynomially with model size. [Algebra from T1/T2.]

### Corollary 4: Parametric Exception

White-box access helps if and only if it identifies a $d$-dimensional parametric family for $\Delta$. In that case, the rate improves from $(\Lip\errrate/\nsamples)^{1/3}$ to $O(d/\errrate\nsamples)^{1/2}$. [From T4 + P1.]

---

## PAPER STRUCTURE

### LaTeX Setup
- Template: ICLR 2027 (use `iclr2027_conference.sty`)
- If not available, use `iclr2025_conference.sty` as placeholder
- Packages: `amsmath, amssymb, amsthm, mathtools, algorithm2e, booktabs, hyperref, cleveref, natbib`
- Theorem environments: `theorem, proposition, lemma, corollary, definition, remark, example`

### Page Allocation (9 pages main text)

**Pages 1–1.5: Introduction** ✓ WRITTEN
- Base rate framing: rare conditions are hard to detect (Bayes' theorem analogy)
- Verification tax as a law, not a limitation
- Relief: adaptive evaluation, parametric assumptions, transfer
- 7 grouped contributions:
  1. The verification tax (T1–T3, phase transition)
  2. Universality (TG: calibration, fairness, robustness)
  3. Impossibility results (T4, TB, P3)
  4. Adaptive evaluation (TA, tight rate)
  5. Scaling consequences (TH, PR, TC)
  6. Temporal decay (THL)
  7. Empirical validation

**Pages 2–3: Setup + Core Theory**
- Section 2.1: Formal setup (Definitions 1–5, Bridge Lemma)
- Section 2.2: T1 (Le Cam) — full proof in main text
- Section 2.3: T2 (Minimax lower bound) — proof sketch, cite Lepski et al.
- Section 2.4: T3 (Matched upper bound) — full proof in main text
- One sentence after T3: "The lower and upper bounds match in the polynomial rate $(L\varepsilon/m)^{1/3}$; the logarithmic gap is inherited from the open problem of $L_1$ functional estimation \citep{lepski1999estimation}."
- Phase transition corollary (Corollary 1)
- Section 2.6: TG (General Verification Tax) — definition + theorem + 2 corollaries (fairness, robustness)

**Pages 4–6: Extensions**
- Section 3.1: T4 (White-box irrelevance) with parametric exception corollary
- Section 3.2: P3 (Ungameability) — short proof + remark
- Section 3.3: TA (Active verification) — proof sketch in main text, full proof in appendix
- Section 3.4: TS (Sequential verification) — proof sketch in main text, full proof in appendix
- Section 3.5: TB (Self-verification impossibility) — full proof in main text (short)
- Section 3.6: P1 (Temperature scaling rate) — proof sketch in main text, full proof in appendix
- Section 3.7: Corollaries (fairness, verification scaling law)
- Section 3.8: P2 (Verification transfer) — proof sketch in main text, full proof in appendix

**Pages 5.5–7: Empirical Study**
- % TODO: Populate after NIM data collection (Apr 28–May 4)
- Section 4.1: Experimental setup (models, benchmarks, confidence definition)
- Section 4.2: ECE vs m curves (Fig 1), theoretical bounds overlaid
- Section 4.3: Calibration efficiency and effective exponent (Fig 2, 3)
- Section 4.4: Phase transition validation (Fig 4)
- Section 4.5: Stratified evaluation (Fig 5)

**Pages 7–8: Practical Tools**
- Fix 1: ε-Adaptive holdout sizing formula
- Fix 3: Stratified evaluation report
- Fix 4: Verification budget calculator
- Fix 2: ε-Adaptive bin count (ONLY if T2+T3 proved without log gap; otherwise label as "conjectured")

**Pages 8–9: The Verification Horizon (Section 5 — climax)**
- TH (Verification Horizon) — theorem + short proof + remark with numbers + active corollary
- TD (Verification Dynamics) — theorem + full proof + 2 remarks (interpretation, active dynamics)
- PR (Recalibration Trap) — proposition + short proof + worked example + regulatory remark
- THL (Verification Half-Life) — definition + theorem + proof + domain table + benchmark remark

**Pages 9–10: Compositional Verification (Section 6)**
- TC (Compositional Tax) — definition + theorem + proof + max depth corollary + agent loop corollary + 2 remarks + worked example

**Pages 10–11: Related Work + Conclusion**
- Differentiation table
- Summary paragraph: complete landscape (TB → T1–T3 → T4 → TA → TH/TD → PR → TC)
- Horizon paragraph + compositional paragraph + regulatory implications

### Appendix Structure
- Appendix A: Full proof of T2 (reduction to Lepski et al.) — the core technical contribution
- Appendix B: Full proof of T4 (white-box irrelevance) — supplement to main-text proof
- Appendix C: Full proof of TA (active verification) — two-phase strategy details and chain rule KL argument
- Appendix D: Full proof of TS (sequential verification) — Wald's identity + confidence sequences
- Appendix E: Temperature scaling analysis (P1) — MLE asymptotics details, Fisher information computation
- Appendix F: Verification transfer (P2) — full histogram analysis for shift estimation
- Appendix G: Corrections (correlation, heterogeneity, pre-calibration)
- Appendix H: Numerical sanity checks — sharp Le Cam constants table (DONE: c₁ ∈ [0.28, 0.38], computed via exact Bernoulli TV)
- Appendix I: Full experimental details, additional figures, robustness checks

---

## CITATION KEYS (for BibTeX)

```
lepski1999estimation    — Lepski, Nemirovski, Spokoiny (PTRF 1999)
brown1996equivalence    — Brown & Low (Ann. Statist. 1996)
nussbaum1996equivalence — Nussbaum (Ann. Statist. 1996)
tsybakov2009nonparametric — Tsybakov, Introduction to Nonparametric Estimation (2009)
sun2023recalibration    — Sun, Song, Hero (NeurIPS 2023 Spotlight)
futami2024information   — Futami & Fujisawa (NeurIPS 2024)
futami2026smooth        — Futami & Nitanda (ICLR 2026)
hu2024testing           — Hu et al. (NeurIPS 2024 / STOC 2025)
kossen2021active        — Kossen et al. (ICML 2021)
blasiok2023calibration  — Blasiok et al. (STOC 2023)
ciosek2025finite        — Ciosek et al. (2025 preprint)
guo2017calibration      — Guo et al. (ICML 2017) — On Calibration of Modern Neural Networks
kumar2019verified       — Kumar, Liang, Ma (NeurIPS 2019)
donoho1991geometrizing  — Donoho & Liu (Ann. Statist. 1991) — modulus of continuity
vandervaart1998asymptotic — van der Vaart (1998)
ibragimov1986estimation — Ibragimov, Nemirovski, Khasminski (1986) — smooth functional estimation
goldenshluger2020Lr     — Goldenshluger et al. (PTRF 2020) — L_r norms in Besov spaces
howard2021timeuniform   — Howard, Ramdas et al. (Ann. Statist. 2021) — confidence sequences
hoffmann2022chinchilla  — Hoffmann et al. (NeurIPS 2022) — Chinchilla scaling laws
kaplan2020scaling       — Kaplan et al. (2020) — scaling laws for neural language models
wei2022chainofthought   — Wei et al. (NeurIPS 2022) — chain-of-thought prompting
lewis2020rag            — Lewis et al. (NeurIPS 2020) — retrieval-augmented generation
yao2023react            — Yao et al. (ICLR 2023) — ReAct: reasoning and acting
davis2020calibrationdrift — Davis et al. (JAMIA 2020) — calibration drift in clinical models
gama2014concept         — Gama et al. (ACM Comput. Surv. 2014) — concept drift adaptation
```

---

## KEY REFERENCES TO INCLUDE IN RELATED WORK

| Paper | Their result | Our differentiation |
|-------|-------------|---------------------|
| Sun, Song & Hero (NeurIPS 2023 Spotlight) | UB $O(n^{-2/3})$, $B \sim n^{1/3}$ | We provide matching LB + $\errrate$-dependence they left open |
| Futami & Fujisawa (NeurIPS 2024) | General $n^{-1/3}$ UB for ECE bias | We add $\errrate$-dependence revealing phase transition + first LB |
| Futami & Nitanda (ICLR 2026) | Smooth CE uniform convergence | Different question: data requirements vs algorithm guarantees |
| Hu et al. (NeurIPS 2024 / STOC 2025) | $\Omega(\errrate^{-2.5})$ testing complexity | Different problem: testing vs estimation; their $\errrate$ = distance, ours = error rate |
| Lepski, Nemirovski & Spokoiny (PTRF 1999) | $L_1$ norm minimax rate $n^{-1/3}$ up to logs | We specialize to calibration with $\errrate$-dependent noise; their result is our key technical tool |
| Blasiok et al. (STOC 2023) | Calibration distance framework | Foundational; different metric (distance to calibration vs ECE) |

---

## EMPIRICAL STUDY SPECIFICATION (for later execution)

### Benchmarks
- MMLU (~14,000 questions, 4-way MCQ)
- TruthfulQA (~800 questions) or a binary NLI task (~3,000 questions)

### Confidence Definition (FROZEN)
- Constrain generation to single token from {A, B, C, D} (MMLU) or {Yes, No} (binary)
- Confidence = softmax probability of predicted token
- Requires token-level logprobs from NIM API

### Models
- Tier 1 (must-have, 6 models): Spanning ε from ~0.05 to ~0.35
- Tier 2 (supplementary, up to 8 more): Fill gaps in ε coverage
- Drop any model that doesn't expose token-level logprobs

### Analysis Protocol
1. Stage 1: Reserve 500 samples per model as pilot to estimate ε̂
2. Stage 2: Subsample at m ∈ {50, 100, 200, 500, 1000, 2000, 5000, 10000}
3. Compute ECE with B = B*(m, ε̂, L̂) from T3 formula
4. 1000 bootstrap replicates per (model, m) pair
5. Robustness: Also compute with fixed B = 15

### Pre-Registered Decision Threshold
- Slopes within 0.1 of theory (−0.5 for T1, −0.33 for T2): Empirics validate theory in main text
- Slopes deviate 0.1–0.2: Include with diagnosis of WHY
- Slopes deviate >0.2: Drop to appendix as preliminary evidence

---

## EXECUTION ORDER FOR AGENT

Phase 1 (Typesetting — DONE):
1. Set up ICLR LaTeX template with all notation commands ✓
2. Typeset Definitions 1–6 and Bridge Lemma ✓
3. Typeset T1 with full proof ✓
4. Typeset T2 with proof sketch (main text) + full proof (Appendix A) ✓
5. Typeset T3 with full proof ✓
6. Typeset TG (General Verification Tax) with definition + theorem + 2 corollaries ✓
7. Typeset T4 with full proof ✓
8. Typeset P3 (Ungameability) with short proof + remark ✓
9. Typeset TA (Active Verification) with proof sketch + full proof (Appendix C) ✓
10. Typeset TS (Sequential Verification) with proof sketch + full proof (Appendix D) ✓
11. Typeset TB (Self-Verification Impossibility) with full proof + 3 corollaries ✓
12. Typeset P1 with proof sketch (main text) + full proof (Appendix E) ✓
13. Typeset P2 with proof sketch (main text) + full proof (Appendix F) ✓
14. Typeset all corollaries ✓
15. Typeset TH (Verification Horizon) with proof + remark + active corollary ✓
16. Typeset TD (Verification Dynamics) with full proof + 2 remarks ✓
17. Typeset PR (Recalibration Trap) with proof + worked example + regulatory remark ✓
18. Typeset TC (Compositional Tax) with definition + theorem + proof + 2 corollaries + 2 remarks + example ✓
19. Typeset THL (Verification Half-Life) with definition + theorem + proof + domain table + 2 remarks ✓
20. Compute sharp Le Cam constants via Python script (scripts/sharp_constants.py) ✓
21. Insert constants table into Appendix (Numerical Sanity Checks) ✓
22. Update Fix 1 with exact holdout sizing formula ✓
23. Write full introduction (base rate framing, 7 grouped contributions) ✓
24. Create BibTeX file with all citation keys (24 entries) ✓
25. Set up section structure with TODO comments for unfilled sections ✓
26. Compile and verify — 23 pages, zero errors ✓

Phase 2 (Writing — DONE):
1. Write related work section (4 subsections: calibration estimation, testing/distance, active evaluation, nonparametric) ✓
2. Write introduction (base rate framing, 7 grouped contributions, outline paragraph) ✓
3. Write conclusion (3 paragraphs + limitations + open problems + broader implications) ✓
4. Fill all Practical Tools fixes (1-4) ✓
5. Rewrite empirical section as "Empirical Predictions" (5 falsifiable predictions) ✓
6. Fill all appendix TODOs (B: T4 supplement, G: corrections, I: experimental protocol) ✓
7. Zero TODOs remaining ✓
8. Compile and verify — 25 pages, zero errors, zero undefined references ✓
9. Paper is ready for ArXiv submission ✓

Phase 3 (Empirics — after data collection, Apr 28–May 4):
1. Generate all figures
2. Typeset empirical study section
3. Typeset practical tools section

Phase 4 (Polish — after external review):
1. Fill remaining appendix sections (E, F)
2. Final compilation and proofreading
