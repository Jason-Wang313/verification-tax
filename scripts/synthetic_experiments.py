"""
Synthetic validation experiments for The Verification Tax.

DGP (FIX-5): Beta distribution Beta(α, 1) with α = (1-ε)/ε to ensure E[1-p] = ε.
This properly enforces the target error rate, unlike Uniform[0,1] which has ε = 0.5.

Calibration gap: Δ(p) = A sin(2πk p), Lipschitz constant L = 2πkA.
Labels: Y ~ Bern(p + Δ(p)), clipped to [0,1].

Validates: (1) m^{-1/3} passive rate, (2) phase transition, (3) active-passive gap,
(4) hard instance approaches minimax.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)


def sample_scores(m, eps):
    """Sample scores from Beta((1-ε)/ε, 1) so that E[1-p] ≈ ε."""
    alpha = (1 - eps) / eps
    return np.random.beta(alpha, 1, size=m)


def realized_eps(m_test, eps_target):
    """Empirically verify the realized error rate."""
    p = sample_scores(m_test, eps_target)
    # For a perfectly calibrated classifier, P(Y≠ŷ) = E[1-p]
    return np.mean(1 - p)


def empirical_ece(p_samples, y_samples, B):
    n = len(p_samples)
    edges = np.linspace(0, 1, B + 1)
    ece = 0.0
    for b in range(B):
        mask = (p_samples >= edges[b]) & (p_samples < edges[b + 1])
        nb = mask.sum()
        if nb > 0:
            acc = np.mean(y_samples[mask])
            conf = np.mean(p_samples[mask])
            ece += (nb / n) * abs(acc - conf)
    return ece


def true_ece_for_dgp(eps, delta_func, n_huge=2_000_000):
    """Compute true ECE by Monte Carlo on a huge sample."""
    p = sample_scores(n_huge, eps)
    eta = np.clip(p + delta_func(p), 1e-10, 1 - 1e-10)
    # ECE = E[ |E[Y|p in I_b] - E[p|p in I_b]| ] over fine bins
    B = 200
    edges = np.linspace(0, 1, B + 1)
    ece = 0.0
    for b in range(B):
        mask = (p >= edges[b]) & (p < edges[b + 1])
        if mask.sum() > 0:
            true_acc = np.mean(eta[mask])
            mean_conf = np.mean(p[mask])
            ece += (mask.sum() / n_huge) * abs(true_acc - mean_conf)
    return ece


# ============================================================
# Verify DGP: realized ε matches target
# ============================================================
print("=== DGP verification ===")
for eps_t in [0.05, 0.10, 0.20]:
    realized = [realized_eps(100000, eps_t) for _ in range(5)]
    print(f"  Target ε={eps_t}: realized = {np.mean(realized):.4f} ± {np.std(realized):.4f}")
print()

# ============================================================
# Benign calibration function
# ============================================================
A_benign = 0.03
k_benign = 1
L_benign = 2 * np.pi * k_benign * A_benign
delta_benign = lambda p: A_benign * np.sin(2 * np.pi * k_benign * p)

# Hard calibration function
A_hard = 0.015
k_hard = 8
L_hard = 2 * np.pi * k_hard * A_hard
delta_hard = lambda p: A_hard * np.sin(2 * np.pi * k_hard * p)

# Precompute true ECE for benign and hard at each ε
print("=== True ECE values ===")
ECE_TRUE_BENIGN = {}
ECE_TRUE_HARD = {}
for eps in [0.05, 0.10, 0.20]:
    ECE_TRUE_BENIGN[eps] = true_ece_for_dgp(eps, delta_benign)
    ECE_TRUE_HARD[eps] = true_ece_for_dgp(eps, delta_hard)
    print(f"  ε={eps}: ECE_benign={ECE_TRUE_BENIGN[eps]:.6f}, ECE_hard={ECE_TRUE_HARD[eps]:.6f}")
print()

ms = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
n_boot = 400


# ============================================================
# Figure 1: Benign passive scaling
# ============================================================
print("=== Figure 1: Benign Passive ===")
results_benign = {}
slopes_benign = {}

for eps in [0.05, 0.10, 0.20]:
    errors = []
    for m in ms:
        B_star = max(3, int((L_benign**2 * m / eps) ** (1/3)))
        boot_errors = []
        for _ in range(n_boot):
            p_samp = sample_scores(m, eps)
            eta = np.clip(p_samp + delta_benign(p_samp), 1e-10, 1 - 1e-10)
            y_samp = np.random.binomial(1, eta)
            ece_hat = empirical_ece(p_samp, y_samp, B_star)
            boot_errors.append(abs(ece_hat - ECE_TRUE_BENIGN[eps]))
        mean_err = np.mean(boot_errors)
        errors.append(mean_err)
        print(f"  eps={eps:.2f}, m={m:6d}, B*={B_star:3d}, err={mean_err:.6f}")
    results_benign[eps] = errors

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
log_m = np.log10(ms)
for eps in [0.05, 0.10, 0.20]:
    log_err = np.log10(results_benign[eps])
    mask = np.array(ms) >= 1000
    slope, _ = np.polyfit(np.array(log_m)[mask], np.array(log_err)[mask], 1)
    slopes_benign[eps] = slope
    ax.plot(log_m, log_err, 'o-', markersize=5, linewidth=1.5,
            label=rf'$\varepsilon = {eps}$ (slope={slope:.2f})')
    print(f"  eps={eps:.2f}: slope = {slope:.3f}")

ref = np.log10(results_benign[0.10][0]) + (-1/3) * (np.array(log_m) - log_m[0])
ax.plot(log_m, ref, 'k--', alpha=0.5, linewidth=2, label=r'Theory: slope $= -1/3$')

ax.set_xlabel(r'$\log_{10}(m)$', fontsize=12)
ax.set_ylabel(r'$\log_{10}(\mathrm{mean}\;|\widehat{\mathrm{ECE}} - \mathrm{ECE}|)$', fontsize=12)
ax.set_title(r'ECE Estimation Error vs Sample Size (Beta DGP, Optimal $B^*$)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig1_scaling_exponent.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig1_scaling_exponent.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 2: Phase transition
# ============================================================
print("\n=== Figure 2: Phase Transition ===")
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
n_trials = 800

for eps in [0.02, 0.05, 0.10, 0.20]:
    ms_phase = np.unique(np.logspace(np.log10(max(20, int(0.1/eps))),
                                      np.log10(int(100/eps)), 18).astype(int))
    power = []
    for m in ms_phase:
        B = max(3, int((L_benign**2 * m / eps) ** (1/3)))
        detections = 0
        for _ in range(n_trials):
            p_samp = sample_scores(m, eps)
            # H1: miscalibrated
            eta = np.clip(p_samp + delta_benign(p_samp), 1e-10, 1 - 1e-10)
            y1 = np.random.binomial(1, eta)
            ece_h1 = empirical_ece(p_samp, y1, B)
            # H0: calibrated
            y0 = np.random.binomial(1, p_samp)
            ece_h0 = empirical_ece(p_samp, y0, B)
            if ece_h1 > ece_h0 * 1.5:
                detections += 1
        power.append(detections / n_trials)

    norm_m = ms_phase * eps
    ax.semilogx(norm_m, power, 'o-', markersize=4, linewidth=1.5,
                label=rf'$\varepsilon = {eps}$')

ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
           label=r'$m \cdot \varepsilon = 1$')
ax.set_xlabel(r'Normalized sample size $m \cdot \varepsilon$', fontsize=12)
ax.set_ylabel('Detection power', fontsize=12)
ax.set_title('Phase Transition in Calibration Detection (Beta DGP)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('figures/fig2_phase_transition.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig2_phase_transition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Done.")

# ============================================================
# Figure 3: Active vs Passive
# ============================================================
print("\n=== Figure 3: Active vs Passive ===")
eps_ap = 0.10
passive_errors = []
active_errors = []

for m in ms:
    B_star = max(3, int((L_benign**2 * m / eps_ap) ** (1/3)))
    p_errs = []
    a_errs = []
    for _ in range(n_boot):
        # Passive: random sampling from Beta DGP
        p_samp = sample_scores(m, eps_ap)
        eta = np.clip(p_samp + delta_benign(p_samp), 1e-10, 1 - 1e-10)
        y_samp = np.random.binomial(1, eta)
        ece_passive = empirical_ece(p_samp, y_samp, B_star)
        p_errs.append(abs(ece_passive - ECE_TRUE_BENIGN[eps_ap]))

        # Active: stratified sampling from a pre-computed Beta pool
        # The active oracle has a pool of unlabeled inputs sorted by score
        # and labels samples per bin equally (mimicking Definition 8)
        from scipy.stats import beta as beta_dist
        alpha = (1 - eps_ap) / eps_ap
        B_active = max(3, int(np.sqrt(m)))
        per_bin = max(1, m // B_active)
        # Stratify over [0.5, 1.0] (high-density region for these ε values)
        edges = np.linspace(0.5, 1.0, B_active + 1)
        # Bin weights = probability mass under Beta in each bin
        cdf_vals = beta_dist.cdf(edges, alpha, 1)
        bin_weights = cdf_vals[1:] - cdf_vals[:-1]
        # Sample uniformly within each bin (active oracle: query inputs with p ∈ I_b)
        ece_active = 0.0
        for b in range(B_active):
            if bin_weights[b] < 1e-8:
                continue
            p_b = np.random.uniform(edges[b], edges[b+1], size=per_bin)
            eta_b = np.clip(p_b + delta_benign(p_b), 1e-10, 1 - 1e-10)
            y_b = np.random.binomial(1, eta_b)
            delta_hat = np.mean(y_b) - np.mean(p_b)
            ece_active += bin_weights[b] * abs(delta_hat)
        a_errs.append(abs(ece_active - ECE_TRUE_BENIGN[eps_ap]))

    passive_errors.append(np.mean(p_errs))
    active_errors.append(np.mean(a_errs))
    print(f"  m={m:6d}: passive={np.mean(p_errs):.6f}, active={np.mean(a_errs):.6f}")

log_m = np.log10(ms)
mask = np.array(ms) >= 1000
slope_p, _ = np.polyfit(np.array(log_m)[mask], np.log10(np.array(passive_errors))[mask], 1)
slope_a, _ = np.polyfit(np.array(log_m)[mask], np.log10(np.array(active_errors))[mask], 1)
print(f"  Passive slope: {slope_p:.3f}")
print(f"  Active slope:  {slope_a:.3f}")

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(log_m, np.log10(passive_errors), 'bs-', markersize=6, linewidth=2,
        label=f'Passive (slope={slope_p:.2f})')
ax.plot(log_m, np.log10(active_errors), 'r^-', markersize=6, linewidth=2,
        label=f'Active (slope={slope_a:.2f})')

ref_p = np.log10(passive_errors[0]) + (-1/3) * (np.array(log_m) - log_m[0])
ref_a = np.log10(active_errors[0]) + (-1/2) * (np.array(log_m) - log_m[0])
ax.plot(log_m, ref_p, 'b--', alpha=0.4, linewidth=1.5, label=r'$m^{-1/3}$ reference')
ax.plot(log_m, ref_a, 'r--', alpha=0.4, linewidth=1.5, label=r'$m^{-1/2}$ reference')

ax.set_xlabel(r'$\log_{10}(m)$', fontsize=12)
ax.set_ylabel(r'$\log_{10}(\mathrm{mean}\;|\widehat{\mathrm{ECE}} - \mathrm{ECE}|)$', fontsize=12)
ax.set_title(rf'Active vs Passive Verification ($\varepsilon = {eps_ap}$, Beta DGP)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig3_active_passive.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig3_active_passive.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 4: Hard instance
# ============================================================
print("\n=== Figure 4: Hard Instance ===")
hard_results = {}
hard_slopes = {}
for eps in [0.05, 0.10]:
    errors = []
    for m in ms:
        B_star = max(3, int((L_hard**2 * m / eps) ** (1/3)))
        boot_errors = []
        for _ in range(n_boot):
            p_samp = sample_scores(m, eps)
            eta = np.clip(p_samp + delta_hard(p_samp), 1e-10, 1 - 1e-10)
            y_samp = np.random.binomial(1, eta)
            ece_hat = empirical_ece(p_samp, y_samp, B_star)
            boot_errors.append(abs(ece_hat - ECE_TRUE_HARD[eps]))
        mean_err = np.mean(boot_errors)
        errors.append(mean_err)
        print(f"  [HARD] eps={eps:.2f}, m={m:6d}, B*={B_star:3d}, err={mean_err:.6f}")
    hard_results[eps] = errors

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
for eps in [0.05, 0.10]:
    log_m_h = np.log10(ms)
    log_err_h = np.log10(hard_results[eps])
    mask_h = np.array(ms) >= 1000
    slope_h, _ = np.polyfit(np.array(log_m_h)[mask_h], np.array(log_err_h)[mask_h], 1)
    hard_slopes[eps] = slope_h
    ax.plot(log_m_h, log_err_h, 'o-', markersize=5, linewidth=1.5,
            label=rf'$\varepsilon = {eps}$ (slope={slope_h:.2f})')
    print(f"  [HARD] eps={eps:.2f}: slope = {slope_h:.3f}")

ref_hard = np.log10(hard_results[0.10][0]) + (-1/3) * (np.array(log_m_h) - log_m_h[0])
ax.plot(log_m_h, ref_hard, 'k--', alpha=0.5, linewidth=2, label=r'Theory: slope $= -1/3$')

ax.set_xlabel(r'$\log_{10}(m)$', fontsize=12)
ax.set_ylabel(r'$\log_{10}(\mathrm{mean}\;|\widehat{\mathrm{ECE}} - \mathrm{ECE}|)$', fontsize=12)
ax.set_title(r'Hard Instance: High-Frequency Calibration ($k=8$, Beta DGP)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig4_hard_instance.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig4_hard_instance.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n=== SUMMARY ===")
print("Benign passive slopes:")
for eps in [0.05, 0.10, 0.20]:
    print(f"  ε={eps}: {slopes_benign[eps]:.3f}")
print(f"Active vs passive (ε={eps_ap}): passive={slope_p:.3f}, active={slope_a:.3f}")
print("Hard passive slopes:")
for eps in [0.05, 0.10]:
    print(f"  ε={eps}: {hard_slopes[eps]:.3f}")
