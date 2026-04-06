"""
v3 synthetic experiments addressing reviewer FIX-D and FIX-E.

FIX-D: Slope vs number of zero-crossings (k ∈ {1, 8, 32, 64})
       Demonstrates slope → -1/3 as k increases (worst-case minimax)

FIX-E: Active vs passive with large L (k=8 with L≈3, k=16 with L≈6)
       Shows the active-passive gap widens with L
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

np.random.seed(42)


def sample_beta(m, eps):
    """Beta((1-ε)/ε, 1) DGP enforcing E[1-p] = ε."""
    alpha = (1 - eps) / eps
    return np.random.beta(alpha, 1, size=m)


def empirical_ece(p, y, B):
    n = len(p)
    edges = np.linspace(0, 1, B + 1)
    ece = 0.0
    for b in range(B):
        mask = (p >= edges[b]) & (p < edges[b + 1])
        nb = mask.sum()
        if nb > 0:
            acc = np.mean(y[mask])
            conf = np.mean(p[mask])
            ece += (nb / n) * abs(acc - conf)
    return ece


def true_ece_mc(eps, delta_func, n_huge=2_000_000):
    """Monte Carlo ground truth ECE."""
    p = sample_beta(n_huge, eps)
    eta = np.clip(p + delta_func(p), 1e-10, 1 - 1e-10)
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


def measure_passive_slope(eps, delta_func, L, ms, n_boot=300):
    """Run passive estimation, return errors and fitted slope."""
    ece_true = true_ece_mc(eps, delta_func)
    errors = []
    for m in ms:
        B_star = max(3, int((L**2 * m / eps) ** (1/3)))
        boot_errs = []
        for _ in range(n_boot):
            p = sample_beta(m, eps)
            eta = np.clip(p + delta_func(p), 1e-10, 1 - 1e-10)
            y = np.random.binomial(1, eta)
            boot_errs.append(abs(empirical_ece(p, y, B_star) - ece_true))
        errors.append(np.mean(boot_errs))
    log_m = np.log10(ms)
    log_e = np.log10(errors)
    mask = np.array(ms) >= 1000
    slope, _ = np.polyfit(log_m[mask], log_e[mask], 1)
    return errors, slope, ece_true


def measure_active_slope(eps, delta_func, L, ms, n_boot=300):
    """Active stratified sampling on Beta DGP."""
    ece_true = true_ece_mc(eps, delta_func)
    alpha = (1 - eps) / eps
    errors = []
    for m in ms:
        # Active uses sqrt(m) bins as the theoretical optimum suggests
        B_active = max(3, int(np.sqrt(m)))
        per_bin = max(1, m // B_active)
        edges = np.linspace(0.5, 1.0, B_active + 1)
        cdf_vals = beta_dist.cdf(edges, alpha, 1)
        bin_weights = cdf_vals[1:] - cdf_vals[:-1]
        boot_errs = []
        for _ in range(n_boot):
            ece_active = 0.0
            for b in range(B_active):
                if bin_weights[b] < 1e-10:
                    continue
                p_b = np.random.uniform(edges[b], edges[b+1], size=per_bin)
                eta_b = np.clip(p_b + delta_func(p_b), 1e-10, 1 - 1e-10)
                y_b = np.random.binomial(1, eta_b)
                delta_hat = np.mean(y_b) - np.mean(p_b)
                ece_active += bin_weights[b] * abs(delta_hat)
            boot_errs.append(abs(ece_active - ece_true))
        errors.append(np.mean(boot_errs))
    log_m = np.log10(ms)
    log_e = np.log10(errors)
    mask = np.array(ms) >= 1000
    slope, _ = np.polyfit(log_m[mask], log_e[mask], 1)
    return errors, slope


# ============================================================
# FIX-D: Slope vs zero-crossings
# ============================================================
print("=== FIX-D: Slope vs zero-crossings ===")
eps = 0.10
ms = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
ks = [1, 4, 8, 16, 32, 64]

# Use a fixed amplitude budget so all instances have the same ECE scale
# but different L. Δ(p) = A sin(2π k p), L = 2π k A.
# Use A = 0.02 / k so L = 2π·0.02 ≈ 0.126 is constant. But this defeats the purpose.
# Instead use A = 0.025 (fixed), L = 2π k · 0.025, increasing with k.
A_fixed = 0.025

slopes_by_k = {}
errors_by_k = {}

for k in ks:
    A = A_fixed
    L = 2 * np.pi * k * A
    delta = lambda p, kk=k, AA=A: AA * np.sin(2 * np.pi * kk * p)
    errors, slope, true_ece = measure_passive_slope(eps, delta, L, ms, n_boot=250)
    slopes_by_k[k] = slope
    errors_by_k[k] = errors
    print(f"  k={k:3d}, L={L:.3f}, ECE_true={true_ece:.5f}, slope={slope:+.3f}")

# Plot 1: error curves for each k
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
log_m = np.log10(ms)
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ks)))
for k, color in zip(ks, colors):
    log_e = np.log10(errors_by_k[k])
    ax.plot(log_m, log_e, 'o-', color=color, markersize=5, linewidth=1.5,
            label=rf'$k={k}$ (slope={slopes_by_k[k]:+.2f})')
# Reference line at -1/3
ref_idx = ks[-1]
ref_y = np.log10(errors_by_k[ref_idx][0]) + (-1/3) * (np.array(log_m) - log_m[0])
ax.plot(log_m, ref_y, 'k--', alpha=0.5, linewidth=2, label=r'Theory: $-1/3$')

ax.set_xlabel(r'$\log_{10}(m)$', fontsize=12)
ax.set_ylabel(r'$\log_{10}(\mathrm{mean}\;|\widehat{\mathrm{ECE}} - \mathrm{ECE}|)$', fontsize=12)
ax.set_title(r'Convergence Rate vs Number of Zero-Crossings ($\varepsilon=0.10$)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig5_slope_vs_k.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig5_slope_vs_k.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: slope vs k
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.semilogx(ks, [slopes_by_k[k] for k in ks], 'ko-', markersize=8, linewidth=2,
            label='Empirical slopes')
ax.axhline(y=-1/3, color='red', linestyle='--', linewidth=2,
           label=r'Minimax theory: $-1/3$')
ax.axhline(y=-1/2, color='blue', linestyle='--', linewidth=2,
           label=r'Parametric rate: $-1/2$')
ax.set_xlabel(r'Number of zero-crossings $k$', fontsize=12)
ax.set_ylabel('Fitted slope', fontsize=12)
ax.set_title(r'Slope Convergence to Minimax as $k \to \infty$',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig6_slope_convergence.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig6_slope_convergence.png', dpi=150, bbox_inches='tight')
plt.close()

print()

# ============================================================
# FIX-E: Active vs passive with large L
# ============================================================
print("=== FIX-E: Active vs passive at large L ===")
configs = [
    (0.19, 1, 0.03),    # baseline (small L)
    (3.0, 8, 3.0/(2*np.pi*8)),    # L≈3
    (6.0, 16, 6.0/(2*np.pi*16)),  # L≈6
]

ap_results = []
for L_target, k, A in configs:
    L = 2 * np.pi * k * A
    delta = lambda p, kk=k, AA=A: AA * np.sin(2 * np.pi * kk * p)
    p_errs, p_slope, _ = measure_passive_slope(eps, delta, L, ms, n_boot=200)
    a_errs, a_slope = measure_active_slope(eps, delta, L, ms, n_boot=200)
    ap_results.append({
        'L': L, 'k': k,
        'p_errs': p_errs, 'p_slope': p_slope,
        'a_errs': a_errs, 'a_slope': a_slope
    })
    print(f"  L={L:.2f} (k={k}): passive slope={p_slope:+.3f}, active slope={a_slope:+.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, res in zip(axes, ap_results):
    log_m = np.log10(ms)
    ax.plot(log_m, np.log10(res['p_errs']), 'bs-', markersize=6, linewidth=2,
            label=f"Passive ({res['p_slope']:+.2f})")
    ax.plot(log_m, np.log10(res['a_errs']), 'r^-', markersize=6, linewidth=2,
            label=f"Active ({res['a_slope']:+.2f})")
    ax.set_xlabel(r'$\log_{10}(m)$', fontsize=11)
    ax.set_ylabel(r'$\log_{10}(\mathrm{error})$', fontsize=11)
    ax.set_title(rf'$L \approx {res["L"]:.1f}$ ($k={res["k"]}$)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Active vs Passive Verification: Gap Widens with $L$',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig7_active_vs_L.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures/fig7_active_vs_L.png', dpi=150, bbox_inches='tight')
plt.close()

print()
print("=== SUMMARY ===")
print("FIX-D: Slope vs k (passive, ε=0.10):")
for k in ks:
    print(f"  k={k:3d}: slope={slopes_by_k[k]:+.3f}")
print()
print("FIX-E: Active vs Passive at varying L:")
for res in ap_results:
    print(f"  L={res['L']:.2f}: passive={res['p_slope']:+.3f}, active={res['a_slope']:+.3f}, gap={res['p_slope'] - res['a_slope']:+.3f}")
