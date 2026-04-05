"""
Synthetic validation experiments for The Verification Tax.
Validates: (1) m^{-1/3} passive rate, (2) phase transition, (3) active-passive gap.

Key design: Use smooth CE (integral |Δ(p)| dμ(p)) as ground truth, not binned ECE.
This avoids the confound of ECE_true changing with B.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad

np.random.seed(42)

# Calibration gap: Δ(p) = A sin(2πp), Lipschitz constant L = 2πA
A = 0.03
L_lip = 2 * np.pi * A  # ≈ 0.188

delta_func = lambda p: A * np.sin(2 * np.pi * p)

# True smooth CE = ∫|Δ(p)| dp over [0.5, 1.0] / 0.5 (normalize for uniform on [0.5,1])
# = (1/0.5) ∫_{0.5}^{1.0} |A sin(2πp)| dp
TRUE_SMOOTH_CE, _ = quad(lambda p: abs(delta_func(p)), 0.5, 1.0)
TRUE_SMOOTH_CE /= 0.5  # normalize by support width for density=2 on [0.5,1]
# Actually for uniform on [0.5,1], density = 2, so CE = ∫ |Δ(p)| * 2 dp
# But ECE with bins just uses empirical weights, so let's use the binned version
# with a very large number of bins as ground truth.
def compute_true_ece_large_B():
    """True ECE computed with B=1000 bins on a huge uniform sample."""
    p = np.linspace(0.5001, 0.9999, 1000000)
    d = delta_func(p)
    B = 1000
    edges = np.linspace(0.5, 1.0, B + 1)
    ece = 0.0
    for b in range(B):
        mask = (p >= edges[b]) & (p < edges[b + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(p)) * abs(np.mean(d[mask]))
    return ece

ECE_TRUE = compute_true_ece_large_B()
print(f"True ECE (B=1000): {ECE_TRUE:.6f}")


def empirical_ece(p_samples, y_samples, B):
    n = len(p_samples)
    edges = np.linspace(0.5, 1.0, B + 1)
    ece = 0.0
    for b in range(B):
        mask = (p_samples >= edges[b]) & (p_samples < edges[b + 1])
        nb = mask.sum()
        if nb > 0:
            acc = np.mean(y_samples[mask])
            conf = np.mean(p_samples[mask])
            ece += (nb / n) * abs(acc - conf)
    return ece


ms = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
n_boot = 500

# ============================================================
# Figure 1: Passive scaling
# ============================================================
print("\n=== Figure 1: Passive ECE Scaling ===")
epsilons = [0.05, 0.10, 0.20]
results = {}

for eps in epsilons:
    errors = []
    for m in ms:
        B_star = max(3, int((L_lip**2 * m / eps) ** (1/3)))
        boot_errors = []
        for _ in range(n_boot):
            p_samp = np.random.uniform(0.5, 1.0, size=m)
            eta = p_samp + delta_func(p_samp)
            eta = np.clip(eta, 1e-10, 1 - 1e-10)
            y_samp = np.random.binomial(1, eta)
            ece_hat = empirical_ece(p_samp, y_samp, B_star)
            boot_errors.append(abs(ece_hat - ECE_TRUE))
        mean_err = np.mean(boot_errors)
        errors.append(mean_err)
        print(f"  eps={eps:.2f}, m={m:6d}, B*={B_star:3d}, mean|err|={mean_err:.6f}")
    results[eps] = errors

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
log_m = np.log10(ms)
slopes = {}
for eps in epsilons:
    log_err = np.log10(results[eps])
    mask = np.array(ms) >= 1000
    slope, intercept = np.polyfit(np.array(log_m)[mask], np.array(log_err)[mask], 1)
    slopes[eps] = slope
    ax.plot(log_m, log_err, 'o-', markersize=5, linewidth=1.5,
            label=rf'$\varepsilon = {eps}$ (slope={slope:.2f})')
    print(f"  eps={eps:.2f}: slope = {slope:.3f}")

# Theory reference
ref = np.log10(results[0.10][0]) + (-1/3) * (np.array(log_m) - log_m[0])
ax.plot(log_m, ref, 'k--', alpha=0.5, linewidth=2, label=r'Theory: slope $= -1/3$')

ax.set_xlabel(r'$\log_{10}(m)$', fontsize=12)
ax.set_ylabel(r'$\log_{10}(\mathrm{mean}\;|\widehat{\mathrm{ECE}} - \mathrm{ECE}|)$', fontsize=12)
ax.set_title(r'ECE Estimation Error vs Sample Size (Optimal $B^*$)', fontsize=13, fontweight='bold')
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
eps_vals_phase = [0.02, 0.05, 0.10, 0.20]
n_trials = 1000

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
for eps in eps_vals_phase:
    ms_phase = np.unique(np.logspace(np.log10(max(10, int(0.1/eps))),
                                      np.log10(int(100/eps)), 20).astype(int))
    power = []
    for m in ms_phase:
        B = max(3, int((L_lip**2 * m / eps) ** (1/3)))
        detections = 0
        for _ in range(n_trials):
            p_samp = np.random.uniform(0.5, 1.0, size=m)
            # H1
            eta = p_samp + delta_func(p_samp)
            eta = np.clip(eta, 1e-10, 1 - 1e-10)
            y1 = np.random.binomial(1, eta)
            ece_h1 = empirical_ece(p_samp, y1, B)
            # H0
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
ax.set_title('Phase Transition in Calibration Detection', fontsize=13, fontweight='bold')
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
    B_star = max(3, int((L_lip**2 * m / eps_ap) ** (1/3)))

    p_errs = []
    a_errs = []
    for _ in range(n_boot):
        # === Passive: random uniform sampling ===
        p_samp = np.random.uniform(0.5, 1.0, size=m)
        eta = p_samp + delta_func(p_samp)
        eta = np.clip(eta, 1e-10, 1 - 1e-10)
        y_samp = np.random.binomial(1, eta)
        ece_passive = empirical_ece(p_samp, y_samp, B_star)
        p_errs.append(abs(ece_passive - ECE_TRUE))

        # === Active: stratified sampling (equal per bin) ===
        B_active = B_star
        per_bin = max(1, m // B_active)
        edges = np.linspace(0.5, 1.0, B_active + 1)
        p_act = []
        y_act = []
        for b in range(B_active):
            p_b = np.random.uniform(edges[b], edges[b+1], size=per_bin)
            eta_b = p_b + delta_func(p_b)
            eta_b = np.clip(eta_b, 1e-10, 1 - 1e-10)
            y_b = np.random.binomial(1, eta_b)
            p_act.extend(p_b)
            y_act.extend(y_b)
        p_act = np.array(p_act)
        y_act = np.array(y_act)
        ece_active = empirical_ece(p_act, y_act, B_active)
        a_errs.append(abs(ece_active - ECE_TRUE))

    passive_errors.append(np.mean(p_errs))
    active_errors.append(np.mean(a_errs))
    print(f"  m={m:6d}: passive={np.mean(p_errs):.6f}, active={np.mean(a_errs):.6f}")

log_m = np.log10(ms)
mask = np.array(ms) >= 1000
slope_p, _ = np.polyfit(np.array(log_m)[mask], np.log10(np.array(passive_errors))[mask], 1)
slope_a, _ = np.polyfit(np.array(log_m)[mask], np.log10(np.array(active_errors))[mask], 1)
print(f"  Passive slope: {slope_p:.3f} (theory: -0.333)")
print(f"  Active slope:  {slope_a:.3f} (theory: -0.500)")

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
ax.set_title(rf'Active vs Passive Verification ($\varepsilon = {eps_ap}$)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig3_active_passive.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig3_active_passive.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll figures saved to figures/")
print(f"\nSummary of slopes:")
for eps in epsilons:
    print(f"  Passive eps={eps}: {slopes[eps]:.3f}")
print(f"  Active-passive (eps={eps_ap}): passive={slope_p:.3f}, active={slope_a:.3f}")
