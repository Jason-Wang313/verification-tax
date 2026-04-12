"""
vtax: Verification Tax Calculator

Computes information-theoretic verification floors for AI evaluation metrics.
Based on "The Verification Tax" (Wang, 2026).

Usage:
    from vtax import verification_floor, holdout_size, is_verifiable, active_floor
"""

import math


def verification_floor(eps: float, n: int, L: float = 1.0) -> float:
    """
    Compute the verification floor: the minimum detectable ECE difference.

    Below this floor, ECE estimates are indistinguishable from noise,
    regardless of the estimator used. This is an information-theoretic
    limit, not a limitation of any particular method.

    Args:
        eps: Model error rate (1 - accuracy). Must be in (0, 1).
        n: Number of evaluation items (benchmark size).
        L: Lipschitz constant of the calibration function. Default 1.0.
            Typical range: [0.5, 5.0]. Use 1.0 as a conservative default.

    Returns:
        delta_floor: Minimum detectable ECE difference.

    Example:
        >>> verification_floor(eps=0.15, n=14042, L=1.5)
        0.025
        >>> verification_floor(eps=0.05, n=250)  # MMLU per-subject
        0.058

    Reference:
        Theorem 3 of "The Verification Tax" (Wang, 2026):
        R*(m, eps, L) = Theta((L * eps / m)^{1/3})
    """
    if not (0 < eps < 1):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}")
    return (L * eps / n) ** (1/3)


def holdout_size(eps: float, delta: float, L: float = 1.0) -> int:
    """
    Compute the minimum holdout set size for ECE estimation to accuracy delta.

    Args:
        eps: Model error rate (1 - accuracy).
        delta: Target estimation accuracy. Must be > 0.
        L: Lipschitz constant. Default 1.0.

    Returns:
        m: Minimum number of labeled samples needed.

    Example:
        >>> holdout_size(eps=0.15, delta=0.02, L=1.5)
        28125
        >>> holdout_size(eps=0.05, delta=0.01)
        50000

    Reference:
        Theorem 3: m >= L * eps / delta^3
    """
    if not (0 < eps < 1):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta}")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}")
    return math.ceil(L * eps / delta**3)


def active_floor(eps: float, n: int) -> float:
    """
    Compute the active verification floor (when the auditor chooses inputs).

    Active querying eliminates the Lipschitz constant L entirely.
    The active floor is always <= the passive floor.

    Args:
        eps: Model error rate.
        n: Number of evaluation items (budget).

    Returns:
        delta_active: Active verification floor.

    Example:
        >>> active_floor(eps=0.15, n=14042)
        0.003  # Much smaller than passive floor

    Reference:
        Theorem A: R*_active = Theta(sqrt(eps / m))
    """
    if not (0 < eps < 1):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return math.sqrt(eps / n)


def is_verifiable(claimed_improvement: float, eps: float, n: int, L: float = 1.0) -> dict:
    """
    Check whether a claimed metric improvement is verifiable.

    Args:
        claimed_improvement: The claimed improvement in ECE or accuracy.
        eps: Model error rate.
        n: Benchmark size.
        L: Lipschitz constant. Default 1.0.

    Returns:
        dict with keys:
            - floor: the verification floor
            - ratio: claimed_improvement / floor
            - verdict: "VERIFIED" (ratio > 2), "MARGINAL" (1-2), or "NOISE" (< 1)
            - active_floor: floor under active verification

    Example:
        >>> is_verifiable(0.04, eps=0.15, n=14042)
        {'floor': 0.022, 'ratio': 1.8, 'verdict': 'MARGINAL', 'active_floor': 0.003}
    """
    floor = verification_floor(eps, n, L)
    a_floor = active_floor(eps, n)
    ratio = claimed_improvement / floor if floor > 0 else float('inf')

    if ratio >= 2.0:
        verdict = "VERIFIED"
    elif ratio >= 1.0:
        verdict = "MARGINAL"
    else:
        verdict = "NOISE"

    return {
        "floor": round(floor, 6),
        "ratio": round(ratio, 2),
        "verdict": verdict,
        "active_floor": round(a_floor, 6),
    }


def phase_transition(eps: float) -> int:
    """
    Compute the phase transition threshold: minimum samples to detect miscalibration.

    Below this threshold, even detecting whether a model is miscalibrated
    is impossible, regardless of the estimator.

    Args:
        eps: Model error rate.

    Returns:
        m_star: Phase transition threshold (m * eps ≈ 1).

    Reference:
        Corollary 1: Detection impossible for m < 1/eps.
    """
    return math.ceil(1.0 / eps)


def max_pipeline_depth(M_total: int, eps: float, delta: float, L: float = 2.0) -> int:
    """
    Compute the maximum verifiable pipeline depth.

    For a K-component pipeline with per-component Lipschitz L,
    verification cost grows as L^K. This function returns the
    maximum K such that verification is feasible with M_total samples.

    Args:
        M_total: Total available labeled samples.
        eps: Error rate.
        delta: Target accuracy.
        L: Per-component Lipschitz constant. Default 2.0.

    Returns:
        K_max: Maximum verifiable pipeline depth.

    Reference:
        Theorem C: K_max = floor(log_L(M * delta^3 / eps))
    """
    if L <= 1:
        return float('inf')  # Non-expanding pipeline
    arg = M_total * delta**3 / eps
    if arg <= 0:
        return 0
    return max(0, int(math.log(arg) / math.log(L)))
