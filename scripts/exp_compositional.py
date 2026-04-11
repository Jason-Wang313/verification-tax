"""
Compositional Verification Tax (Theorem C) -- Empirical Validation

Validates that verification cost grows exponentially with pipeline depth
at rate L^K for a K-stage pipeline.

Key design: Each pipeline stage applies p -> p + A*sin(2*pi*freq*p + phase).
The per-stage perturbation has Lipschitz constant L_pert = 2*pi*freq*A.
The full stage map p -> p + delta(p) has derivative 1 + delta'(p), so its
max derivative (Lipschitz) is 1 + L_pert.  By the chain rule, the end-to-end
map has L_sys = (1 + L_pert)^K, and the calibration gap Lipschitz is similar.

We set L_pert = 1.5 (i.e. per-stage MAP Lipschitz ~ 2.5), so L_sys grows
exponentially with base ~2.5.  The theory (Theorem C) predicts:
  - Error at fixed m: proportional to (L_sys * eps / m)^{1/3}
  - Cost to reach fixed delta: proportional to L_sys * eps / delta^3
Both grow exponentially in K.

Outputs:
  figures/fig_compositional.pdf/.png  (2-panel, 300 DPI)
  results/analysis/compositional_results.json
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = r"C:\Users\wangz\verification tax"
FIG_DIR = os.path.join(BASE, "figures")
RES_DIR = os.path.join(BASE, "results", "analysis")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Beta DGP
# ---------------------------------------------------------------------------
def sample_beta(m, eps):
    """Beta((1-eps)/eps, 1) DGP enforcing E[1-p] = eps."""
    alpha_param = (1 - eps) / eps
    return np.random.beta(alpha_param, 1, size=m)

# ---------------------------------------------------------------------------
# Binned ECE
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_pipeline(p0, K, A, freq, phases):
    """K-stage pipeline: p_k = clip(p_{k-1} + A*sin(2*pi*freq*p_{k-1} + phase_k))."""
    p = p0.copy()
    for k in range(K):
        p = np.clip(p + A * np.sin(2 * np.pi * freq * p + phases[k]), 0.001, 0.999)
    return p

def measure_L_sys(K, A, freq, phases, n_grid=100000):
    """Empirical Lipschitz of the end-to-end MAP p0 -> pK via finite differences."""
    p = np.linspace(0.001, 0.999, n_grid)
    pK = build_pipeline(p, K, A, freq, phases)
    dp = p[1] - p[0]
    return float(np.max(np.abs(np.diff(pK) / dp)))

# ---------------------------------------------------------------------------
# Parameters (matching the spec)
# ---------------------------------------------------------------------------
eps = 0.10
L_pert = 1.5                         # perturbation Lipschitz per stage
freq = 2                             # zero-crossings
A = L_pert / (2 * np.pi * freq)      # ~ 0.119

Ks = [1, 2, 3, 4, 5]
ms = np.array([200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
n_reps = 200
n_huge = 500_000

# The theoretical per-stage MAP Lipschitz (max |1 + delta'(p)|) is 1 + L_pert = 2.5
# So L_sys_theory = (1 + L_pert)^K = 2.5^K
L_stage = 1 + L_pert  # = 2.5

np.random.seed(123)
all_phases = np.random.uniform(0, 2 * np.pi, size=max(Ks))
np.random.seed(42)

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
print("=" * 70)
print("Compositional Verification Tax -- Empirical Validation (Theorem C)")
print("=" * 70)
print(f"  eps             = {eps}")
print(f"  L_pert          = {L_pert}")
print(f"  L_stage (1+Lp)  = {L_stage}")
print(f"  A               = {A:.5f}")
print(f"  freq            = {freq}")
print(f"  Ks              = {Ks}")
print(f"  ms              = {[int(x) for x in ms]}")
print(f"  n_reps          = {n_reps}")
print(f"  n_huge          = {n_huge}")
print()
sys.stdout.flush()

results = {}

for K in Ks:
    print(f"--- K = {K} ---")
    sys.stdout.flush()

    L_sys_theory = L_stage ** K
    L_sys_emp = measure_L_sys(K, A, freq, all_phases[:K])

    # Ground truth from large pool
    # The verifier observes (p0, y) where y ~ Bern(pK(p0)).
    # The calibration gap is Delta(p0) = pK(p0) - p0.
    p0_pool = sample_beta(n_huge, eps)
    pK_pool = build_pipeline(p0_pool, K, A, freq, all_phases[:K])
    y_pool = np.random.binomial(1, pK_pool)
    ece_true = empirical_ece(p0_pool, y_pool, B=500)

    print(f"  L_sys theory ((1+Lp)^K) = {L_sys_theory:.2f}")
    print(f"  L_sys empirical         = {L_sys_emp:.2f}")
    print(f"  ECE_true                = {ece_true:.5f}")
    sys.stdout.flush()

    # For the B* formula, use the calibration-gap Lipschitz.
    # The gap Delta(p0)=pK(p0)-p0 has Lipschitz = L_sys - 1 (subtracting identity).
    # But empirically it can be larger due to clipping. Use L_sys_emp as an upper bound.
    L_for_B = L_sys_emp

    errors_by_m = []
    for m in ms:
        B_star = max(2, int((L_for_B ** 2 * m / eps) ** (1.0 / 3.0)))
        errs = []
        for _ in range(n_reps):
            idx = np.random.choice(n_huge, size=m, replace=False)
            ece_hat = empirical_ece(p0_pool[idx], y_pool[idx], B_star)
            errs.append(abs(ece_hat - ece_true))
        mean_err = np.mean(errs)
        errors_by_m.append(mean_err)
        print(f"    m={m:6d}  B*={B_star:5d}  error={mean_err:.5f}")
        sys.stdout.flush()

    results[K] = {
        'L_sys_theory': float(L_sys_theory),
        'L_sys_empirical': float(L_sys_emp),
        'ece_true': float(ece_true),
        'ms': [int(x) for x in ms],
        'errors': [float(e) for e in errors_by_m],
    }
    print()
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Cost to reach delta for each K -- use multiple targets, pick the one that
# gives clean interpolation for most K values
# ---------------------------------------------------------------------------
# Try delta targets and find one where all K values can be interpolated
# reliably (the target falls within the observed error range for all K)
delta_candidates = [0.005, 0.008, 0.01, 0.015, 0.02, 0.03]

# Pick the target where the most K-values have the target within their
# [min_error, max_error] range
best_delta = None
best_count = -1
for dt in delta_candidates:
    count = sum(1 for K in Ks
                if np.array(results[K]['errors'])[0] >= dt >= np.array(results[K]['errors'])[-1])
    if count > best_count:
        best_count = count
        best_delta = dt

delta_target = best_delta
print("=" * 70)
print(f"Cost to reach delta = {delta_target} (best coverage: {best_count}/{len(Ks)} K-values)")
print("=" * 70)

cost_vs_K = {}
for K in Ks:
    errs = np.array(results[K]['errors'])
    ms_arr = np.array(results[K]['ms'], dtype=float)
    log_m = np.log10(ms_arr)
    log_e = np.log10(errs)
    log_target = np.log10(delta_target)

    if errs[-1] > delta_target:
        # Extrapolate forward
        sl, ic = np.polyfit(log_m[-5:], log_e[-5:], 1)
        m_req = 10 ** ((log_target - ic) / sl)
        method = 'extrapolated'
    elif errs[0] < delta_target:
        # Extrapolate backward
        sl, ic = np.polyfit(log_m[:4], log_e[:4], 1)
        m_req = 10 ** ((log_target - ic) / sl)
        method = 'extrapolated_back'
    else:
        for i in range(len(errs) - 1):
            if errs[i] >= delta_target >= errs[i + 1]:
                frac = (log_target - log_e[i]) / (log_e[i + 1] - log_e[i])
                m_req = 10 ** (log_m[i] + frac * (log_m[i + 1] - log_m[i]))
                method = 'interpolated'
                break
        else:
            sl, ic = np.polyfit(log_m[-4:], log_e[-4:], 1)
            m_req = 10 ** ((log_target - ic) / sl)
            method = 'fallback'

    L_sys = results[K]['L_sys_empirical']
    m_thy = L_sys * eps / delta_target ** 3

    cost_vs_K[K] = {
        'm_required': float(m_req),
        'm_theory': float(m_thy),
        'method': method,
    }
    print(f"  K={K}:  m_emp = {m_req:>12,.0f} ({method:18s})  m_thy = {m_thy:>12,.0f}")

# ---------------------------------------------------------------------------
# Fit log(m) vs K — theory predicts slope = log(L_stage)
# ---------------------------------------------------------------------------
K_arr = np.array(Ks, dtype=float)
log_m_emp = np.array([np.log(cost_vs_K[K]['m_required']) for K in Ks])
log_m_thy = np.array([np.log(cost_vs_K[K]['m_theory']) for K in Ks])

slope_emp, _ = np.polyfit(K_arr, log_m_emp, 1)
slope_thy, _ = np.polyfit(K_arr, log_m_thy, 1)
theory_slope = np.log(L_stage)

print()
print("=" * 70)
print("SUMMARY: Exponential Scaling of Verification Cost")
print("=" * 70)
print(f"  Per-stage MAP Lipschitz L = {L_stage}")
print(f"  Theory predicts slope(log m vs K) = log(L) = {theory_slope:.4f}")
print()
print(f"  From empirical error curves:")
print(f"    Slope            = {slope_emp:.4f}")
print(f"    Exp base         = {np.exp(slope_emp):.3f}")
print()
print(f"  From theoretical m = L_sys*eps/delta^3:")
print(f"    Slope            = {slope_thy:.4f}")
print(f"    Exp base         = {np.exp(slope_thy):.3f}")
print()

# Per-K slopes
print("Per-K convergence slopes (log10 error vs log10 m):")
slopes_K = {}
for K in Ks:
    lm = np.log10(np.array(results[K]['ms'], dtype=float))
    le = np.log10(np.array(results[K]['errors']))
    mask = np.array(results[K]['ms']) >= 1000
    sl, _ = np.polyfit(lm[mask], le[mask], 1) if mask.sum() >= 2 else np.polyfit(lm, le, 1)
    slopes_K[K] = float(sl)
    print(f"  K={K}: {sl:+.3f}  (theory: -1/3 = -0.333)")

print()
print("L_sys growth (empirical vs theory):")
for K in Ks:
    Le = results[K]['L_sys_empirical']
    Lt = results[K]['L_sys_theory']
    print(f"  K={K}: emp={Le:.2f}  thy={Lt:.2f}  ratio={Le/Lt:.2f}")

sys.stdout.flush()

# ---------------------------------------------------------------------------
# Figure: 2-panel
# ---------------------------------------------------------------------------
colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# LEFT: log-log error vs m
for idx, K in enumerate(Ks):
    ms_k = np.array(results[K]['ms'], dtype=float)
    errs_k = np.array(results[K]['errors'])
    L_sys = results[K]['L_sys_empirical']

    ax1.plot(ms_k, errs_k, 'o-', color=colors[idx], markersize=5, linewidth=1.8,
             label=f'$K={K}$  ($L_{{\\mathrm{{sys}}}}\\approx{L_sys:.0f}$)', zorder=3)

    # Theory dashed: calibrated (L_sys*eps/m)^{1/3}
    log_err = np.log(errs_k)
    log_thy = np.log((L_sys * eps / ms_k) ** (1.0 / 3.0))
    C = np.exp(np.mean(log_err - log_thy))
    m_fine = np.logspace(np.log10(ms_k[0]), np.log10(ms_k[-1]), 100)
    ax1.plot(m_fine, C * (L_sys * eps / m_fine) ** (1.0 / 3.0),
             '--', color=colors[idx], linewidth=1.0, alpha=0.5, zorder=2)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Sample size $m$', fontsize=10)
ax1.set_ylabel(r'Mean $|\widehat{\mathrm{ECE}} - \mathrm{ECE}|$', fontsize=10)
ax1.set_title('(a) Estimation error vs sample size', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8.5, loc='upper right', title='Depth $K$', title_fontsize=9)
ax1.grid(True, alpha=0.25, which='both')
ax1.tick_params(labelsize=9)

# Slope triangle
x0, x1 = 2e3, 2e4
y0 = 0.05
y1 = y0 * (x1 / x0) ** (-1.0 / 3.0)
ax1.plot([x0, x1], [y0, y1], 'k-', linewidth=2, alpha=0.5)
ax1.plot([x1, x1], [y1, y0], 'k-', linewidth=1, alpha=0.3)
ax1.plot([x0, x1], [y0, y0], 'k-', linewidth=1, alpha=0.3)
ax1.annotate('slope $=-1/3$', xy=(3.5e3, y0 * 1.12),
             fontsize=9, ha='center', color='black', alpha=0.6)

# RIGHT: cost vs K
K_plot = np.array(Ks)
m_emp_arr = np.array([cost_vs_K[K]['m_required'] for K in Ks])
m_thy_arr = np.array([cost_vs_K[K]['m_theory'] for K in Ks])

ax2.semilogy(K_plot, m_emp_arr, 's-', color='#0072B2', markersize=8, linewidth=2.2,
             label=f'Empirical ($\\delta={delta_target}$)', zorder=3)
ax2.semilogy(K_plot, m_thy_arr, 'D--', color='#009E73', markersize=7, linewidth=1.8,
             label=r'Theory: $L_{\mathrm{sys}}\varepsilon/\delta^3$', zorder=3)

# Fit separate exponential to empirical (only on interpolated points)
interp_mask = np.array([cost_vs_K[K]['method'] == 'interpolated' for K in Ks])
if interp_mask.sum() >= 2:
    K_interp = K_arr[interp_mask]
    log_m_interp = np.array([np.log(cost_vs_K[K]['m_required'])
                             for K, m in zip(Ks, interp_mask) if m])
    slope_emp_clean, _ = np.polyfit(K_interp, log_m_interp, 1)
else:
    slope_emp_clean = slope_emp

# Reference line: anchor at median K interpolated point
K_fine = np.linspace(Ks[0], Ks[-1], 100)
# Anchor to K=3 (middle) empirical point
anchor_K = 3
anchor_m = cost_vs_K[anchor_K]['m_required']
ref_line_emp = anchor_m * np.exp(slope_emp_clean * (K_fine - anchor_K))
ax2.semilogy(K_fine, ref_line_emp, 'r:', linewidth=2, alpha=0.6,
             label=f'Fit: base $= {np.exp(slope_emp_clean):.2f}$', zorder=2)

ax2.annotate(
    f'Fitted base: {np.exp(slope_emp_clean):.2f}\n'
    f'$L_{{\\mathrm{{sys}}}}$ base: ~{np.exp(slope_thy):.1f}',
    xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9,
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

ax2.set_xlabel('Pipeline depth $K$', fontsize=10)
ax2.set_ylabel(f'Sample cost $m$ for $\\delta={delta_target}$', fontsize=10)
ax2.set_title('(b) Verification cost vs pipeline depth', fontsize=11, fontweight='bold')
ax2.set_xticks(Ks)
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(True, alpha=0.25, which='both')
ax2.tick_params(labelsize=9)

plt.tight_layout()
fig_pdf = os.path.join(FIG_DIR, "fig_compositional.pdf")
fig_png = os.path.join(FIG_DIR, "fig_compositional.png")
plt.savefig(fig_pdf, dpi=300, bbox_inches='tight')
plt.savefig(fig_png, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {fig_pdf}")
print(f"Figure saved: {fig_png}")

# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
output = {
    'parameters': {
        'eps': eps, 'L_pert': L_pert, 'L_stage': L_stage,
        'A': float(A), 'freq': freq,
        'Ks': Ks, 'ms': [int(x) for x in ms],
        'n_reps': n_reps, 'n_huge': n_huge,
        'phases': [float(x) for x in all_phases],
        'delta_target': delta_target,
    },
    'per_K_results': {
        str(K): {
            'L_sys_theory': results[K]['L_sys_theory'],
            'L_sys_empirical': results[K]['L_sys_empirical'],
            'ece_true': results[K]['ece_true'],
            'errors': results[K]['errors'],
            'slope': slopes_K[K],
        }
        for K in Ks
    },
    'cost_vs_K': {str(K): cost_vs_K[K] for K in Ks},
    'exponential_fit': {
        'empirical': {'slope': float(slope_emp), 'base': float(np.exp(slope_emp))},
        'theoretical_m': {'slope': float(slope_thy), 'base': float(np.exp(slope_thy))},
        'reference': {'slope_log_L': float(theory_slope), 'L_stage': float(L_stage)},
    },
}
json_path = os.path.join(RES_DIR, "compositional_results.json")
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Results saved: {json_path}")
print("\nDone.")
