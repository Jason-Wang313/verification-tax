"""
Compute sharp Le Cam constants c_1(ε) for the verification tax lower bound:
    R* ≥ c_1(ε) · √(ε/m)

The optimal δ scales as √(ε/m), so we parametrize δ = c · √(ε/m) and optimize over c.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binom


def exact_bernoulli_kl(p, q):
    """Exact KL(Bern(p) || Bern(q))"""
    if p <= 1e-15 or p >= 1 - 1e-15 or q <= 1e-15 or q >= 1 - 1e-15:
        return float('inf')
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def le_cam_bh_constant(eps, m):
    """Find c_1 via Bretagnolle-Huber, optimizing over c where δ = c·√(ε/m)"""
    scale = np.sqrt(eps / m)

    def neg_c1(c):
        delta = c * scale
        if delta <= 0 or delta >= 1 - eps:
            return 0.0
        kl = exact_bernoulli_kl(1 - eps, 1 - eps - delta)
        bound = (delta / 2) * np.exp(-m * kl)
        c1 = bound / scale
        return -c1

    result = minimize_scalar(neg_c1, bounds=(0.01, 10.0), method='bounded')
    return -result.fun


def le_cam_pinsker_constant(eps, m):
    """Find c_1 via Pinsker, optimizing over c where δ = c·√(ε/m)"""
    scale = np.sqrt(eps / m)

    def neg_c1(c):
        delta = c * scale
        if delta <= 0 or delta >= 1 - eps:
            return 0.0
        kl = exact_bernoulli_kl(1 - eps, 1 - eps - delta)
        mkl = m * kl
        tv_upper = min(np.sqrt(mkl / 2), 1.0)
        bound = (delta / 2) * (1 - tv_upper)
        if bound <= 0:
            return 0.0
        c1 = bound / scale
        return -c1

    result = minimize_scalar(neg_c1, bounds=(0.01, 10.0), method='bounded')
    return -result.fun


def le_cam_exact_tv_constant(eps, m):
    """Exact Le Cam with exact TV (vectorized, for moderate m)"""
    scale = np.sqrt(eps / m)

    def neg_c1(c):
        delta = c * scale
        if delta <= 0 or delta >= 1 - eps:
            return 0.0
        p0 = 1 - eps
        p1 = 1 - eps - delta
        if p1 <= 0:
            return 0.0
        ks = np.arange(m + 1)
        pmf0 = binom.pmf(ks, m, p0)
        pmf1 = binom.pmf(ks, m, p1)
        tv = 0.5 * np.sum(np.abs(pmf0 - pmf1))
        tv = min(tv, 1.0)
        bound = (delta / 2) * (1 - tv)
        if bound <= 0:
            return 0.0
        c1 = bound / scale
        return -c1

    result = minimize_scalar(neg_c1, bounds=(0.01, 10.0), method='bounded')
    return -result.fun


if __name__ == "__main__":
    epsilons = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    print("=" * 80)
    print("Sharp Le Cam Constants c_1(ε)")
    print("=" * 80)
    print(f"{'ε':>6s} | {'BH(1e5)':>8s} | {'Pinsker':>8s} | {'ExactTV':>8s} | {'Best':>8s} | {'m for ECE≤0.01':>15s}")
    print("-" * 72)

    results = []
    for eps in epsilons:
        # BH and Pinsker at large m
        c_bh = le_cam_bh_constant(eps, 100000)
        c_pinsker = le_cam_pinsker_constant(eps, 100000)
        # Exact TV at moderate m (1000 is fast enough with vectorized binom)
        c_exact = le_cam_exact_tv_constant(eps, 1000)

        c_best = max(c_bh, c_pinsker, c_exact)

        # Required m for ECE ≤ 0.01: from R* = c_1 √(ε/m) ≤ 0.01
        # => m ≥ c_1² · ε / 0.01²
        m_required = int(np.ceil(c_best**2 * eps / 0.01**2))

        results.append((eps, c_bh, c_pinsker, c_exact, c_best, m_required))
        print(f"{eps:6.2f} | {c_bh:8.4f} | {c_pinsker:8.4f} | {c_exact:8.4f} | {c_best:8.4f} | {m_required:>15,d}")

    print()
    print("LaTeX table rows:")
    print()
    for eps, c_bh, c_pinsker, c_exact, c_best, m_req in results:
        print(f"    {eps:.2f} & {c_best:.4f} & {m_req:,d} \\\\")

    all_c = [r[4] for r in results]
    c_max = max(all_c)
    print(f"\nMax constant: c_1 = {c_max:.4f}")
    print(f"Conservative formula: m >= {c_max**2:.4f} * ε / δ^2")

    # Verify: 1/(2e) ≈ 0.184 is the quadratic-KL approximation
    print(f"\nTheoretical 1/(2e) = {1/(2*np.e):.4f}")
