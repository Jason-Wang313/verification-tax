"""
Anonymous verification-floor utilities for calibration auditing.

This module provides small, dependency-free helpers implementing the closed-form
verification laws used in the submission.
"""

import math


def verification_floor(eps: float, n: int, L: float = 1.0) -> float:
    """
    Return the passive verification floor for calibration differences.

    Args:
        eps: Error rate in (0, 1).
        n: Number of evaluation items.
        L: Lipschitz constant of the calibration function.
    """
    if not (0 < eps < 1):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}")
    return (L * eps / n) ** (1 / 3)


def holdout_size(eps: float, delta: float, L: float = 1.0) -> int:
    """
    Return the minimum passive holdout size needed to resolve delta.
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
    Return the active-verification floor when the auditor chooses the queries.
    """
    if not (0 < eps < 1):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return math.sqrt(eps / n)


def is_verifiable(claimed_improvement: float, eps: float, n: int, L: float = 1.0) -> dict:
    """
    Compare a claimed improvement against the passive verification floor.
    """
    floor = verification_floor(eps, n, L)
    ratio = claimed_improvement / floor if floor > 0 else float("inf")

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
        "active_floor": round(active_floor(eps, n), 6),
    }


def phase_transition(eps: float) -> int:
    """
    Return the threshold m* where m * eps is on the order of one.
    """
    if not (0 < eps < 1):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    return math.ceil(1.0 / eps)


def max_pipeline_depth(M_total: int, eps: float, delta: float, L: float = 2.0) -> int:
    """
    Return the largest pipeline depth that remains passively verifiable.
    """
    if M_total < 1:
        raise ValueError(f"M_total must be >= 1, got {M_total}")
    if not (0 < eps < 1):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta}")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}")
    if L <= 1:
        return math.inf

    arg = M_total * delta**3 / eps
    if arg <= 1:
        return 0
    return max(0, int(math.log(arg) / math.log(L)))

