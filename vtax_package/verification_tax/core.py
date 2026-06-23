"""
verification_tax: information-theoretic floors and auditing tools for AI evaluation.

Based on "The Verification Tax" (Wang, 2026).

Usage:
    from verification_tax import (
        verification_floor, active_floor, is_verifiable, phase_transition,
        audit_predictions, confidence_quality_screen,
    )
"""

import math
from typing import Iterable

import numpy as np

# Confidence-quality screen constants — frozen to match paper.
HIGH_COVERAGE = 0.70
STD_MIN = 0.02
UNIQUE_MIN = 20
DOMINANT_SHARE_MAX = 0.95
FALLBACK_SHARE_MAX = 0.95
FALLBACK_VALUES = (0.25, 0.50, 1.00)


def verification_floor(eps: float, n: int, L: float = 1.0) -> float:
    """Passive verification floor: minimum detectable ECE from Theorem 3.

    R*(m, eps, L) = Theta((L * eps / m)^{1/3})
    """
    if not (0 < eps < 1):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}")
    return (L * eps / n) ** (1 / 3)


def holdout_size(eps: float, delta: float, L: float = 1.0) -> int:
    """Minimum holdout size: m >= L * eps / delta^3."""
    if not (0 < eps < 1):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta}")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}")
    return math.ceil(L * eps / delta ** 3)


def active_floor(eps: float, n: int) -> float:
    """Active verification floor: Theta(sqrt(eps / m)). Lipschitz constant disappears."""
    if not (0 < eps < 1):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return math.sqrt(eps / n)


def phase_transition(eps: float) -> int:
    """Detection phase-transition threshold: m_star ≈ 1 / eps."""
    if not (0 < eps < 1):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    return math.ceil(1.0 / eps)


def optimal_bins(n: int, eps: float, L: float = 1.0) -> int:
    """Optimal bin count for histogram ECE: B* = floor((L^2 m / eps)^{1/3})."""
    if n < 1 or not (0 < eps < 1) or L <= 0:
        raise ValueError("invalid inputs")
    return max(1, int((L ** 2 * n / eps) ** (1 / 3)))


def max_pipeline_depth(M_total: int, eps: float, delta: float, L: float = 2.0) -> int:
    """Max verifiable pipeline depth: K_max = floor(log_L(M * delta^3 / eps))."""
    if L <= 1:
        return 10 ** 9
    arg = M_total * delta ** 3 / eps
    if arg <= 0:
        return 0
    return max(0, int(math.log(arg) / math.log(L)))


def is_verifiable(claimed_improvement: float, eps: float, n: int, L: float = 1.0) -> dict:
    """Verdict: VERIFIED (>=2x floor), MARGINAL (>=1x), NOISE (<1x)."""
    floor = verification_floor(eps, n, L)
    a_floor = active_floor(eps, n)
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
        "active_floor": round(a_floor, 6),
    }


def estimate_lipschitz(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 20,
    min_per_bin: int | None = None,
    cap: float = 5.0,
) -> float:
    """Estimate Lipschitz L as 75th-percentile adjacent-bin finite diff (cap 5.0).

    Matches scripts/analyze_screened_roster.py:estimate_lipschitz verbatim so
    CLI verdicts reproduce the paper's screened-roster table.
    """
    p = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if min_per_bin is None:
        min_per_bin = max(10, len(p) // 100)
    edges = np.linspace(0, 1, n_bins + 1)
    centers: list[float] = []
    accs: list[float] = []
    for idx in range(n_bins):
        if idx == n_bins - 1:
            mask = (p >= edges[idx]) & (p <= edges[idx + 1])
        else:
            mask = (p >= edges[idx]) & (p < edges[idx + 1])
        if int(mask.sum()) >= min_per_bin:
            centers.append((edges[idx] + edges[idx + 1]) / 2)
            accs.append(float(np.mean(y[mask])))
    if len(centers) < 2:
        return 1.0
    centers_arr = np.asarray(centers)
    gaps = np.asarray(accs) - centers_arr
    slopes: list[float] = []
    for idx in range(len(gaps) - 1):
        delta = abs(float(centers_arr[idx + 1] - centers_arr[idx]))
        if delta > 0:
            slopes.append(abs(float(gaps[idx + 1] - gaps[idx])) / delta)
    if not slopes:
        return 1.0
    return float(min(np.percentile(slopes, 75), cap))


def confidence_quality_screen(confidences: Iterable[float]) -> dict:
    """Confidence-quality screen from the paper's screened-roster criteria.

    Returns dict with {passed: bool, reasons: [...], metrics: {...}}.
    Rejects if: std < 0.02, unique<20 (rounded to 3dp), dominant share > 95%,
    or fallback-values {0.25, 0.50, 1.00} share > 95%.
    """
    p = np.asarray(list(confidences), dtype=float)
    if p.size == 0:
        return {"passed": False, "reasons": ["empty"], "metrics": {}}
    rounded = np.round(p, 3)
    values, counts = np.unique(rounded, return_counts=True)
    dominant_idx = int(np.argmax(counts))
    fallback_shares = {
        f"{v:.2f}": float(np.mean(np.isclose(p, v, atol=1e-6)))
        for v in FALLBACK_VALUES
    }
    metrics = {
        "confidence_std": float(np.std(p)),
        "rounded_unique": int(len(values)),
        "dominant_value": float(values[dominant_idx]),
        "dominant_share": float(counts[dominant_idx] / len(p)),
        "fallback_shares": fallback_shares,
        "max_fallback_share": float(max(fallback_shares.values())),
    }
    reasons: list[str] = []
    if metrics["confidence_std"] < STD_MIN:
        reasons.append("std<0.02")
    if metrics["rounded_unique"] < UNIQUE_MIN:
        reasons.append("unique<20")
    if metrics["dominant_share"] > DOMINANT_SHARE_MAX:
        reasons.append("dominant>95pct")
    if metrics["max_fallback_share"] > FALLBACK_SHARE_MAX:
        reasons.append("fallback>95pct")
    return {"passed": len(reasons) == 0, "reasons": reasons, "metrics": metrics}


class VerificationAudit:
    """Object-oriented one-liner for auditing per-item prediction traces.

    Example:
        >>> from verification_tax import VerificationAudit
        >>> audit = VerificationAudit(confidences, labels)
        >>> audit.report()
        >>> audit.passive_floor
        0.0264
    """

    def __init__(
        self,
        confidences: Iterable[float],
        correct: Iterable[int],
        L: float | None = None,
    ):
        self._result = audit_predictions(confidences, correct, L=L)

    def report(self, as_json: bool = False) -> None:
        """Print a human-readable audit table (or JSON if as_json=True)."""
        if as_json:
            import json
            print(json.dumps(self._result, indent=2))
            return
        # Lazy import to avoid a hard circular dep at import time.
        from verification_tax.cli import _format_table
        print(_format_table(self._result))

    def to_dict(self) -> dict:
        """Return the full audit dict (same shape as audit_predictions)."""
        return dict(self._result)

    @property
    def m(self) -> int:
        return self._result["m"]

    @property
    def eps(self) -> float:
        return self._result["eps"]

    @property
    def L_hat(self) -> float:
        return self._result["L_hat"]

    @property
    def passive_floor(self) -> float:
        return self._result["passive_floor"]

    @property
    def active_floor(self) -> float:
        return self._result["active_floor"]

    @property
    def phase_transition_m(self) -> int:
        return self._result["phase_transition_m"]

    @property
    def optimal_bins(self) -> int:
        return self._result["optimal_bins"]

    @property
    def confidence_quality(self) -> dict:
        return self._result["confidence_quality"]


def audit_predictions(
    confidences: Iterable[float],
    correct: Iterable[int],
    L: float | None = None,
) -> dict:
    """Full audit of a per-item (confidence, correct) stream.

    Returns epsilon, m, Lipschitz estimate, passive/active floors, phase-transition
    threshold, optimal bin count, and confidence-quality verdict.
    """
    p = np.asarray(list(confidences), dtype=float)
    y = np.asarray(list(correct), dtype=float)
    if p.size != y.size:
        raise ValueError("confidences and correct must have the same length")
    if p.size == 0:
        raise ValueError("empty inputs")
    m = int(p.size)
    eps = float(1.0 - y.mean())
    eps_clamped = min(max(eps, 1e-4), 1 - 1e-4)
    L_hat = L if L is not None else estimate_lipschitz(p, y)
    floor = verification_floor(eps_clamped, m, L_hat)
    a_floor = active_floor(eps_clamped, m)
    m_star = phase_transition(eps_clamped)
    B_star = optimal_bins(m, eps_clamped, L_hat)
    cq = confidence_quality_screen(p)
    return {
        "m": m,
        "eps": round(eps, 6),
        "L_hat": round(L_hat, 4),
        "passive_floor": round(floor, 6),
        "active_floor": round(a_floor, 6),
        "phase_transition_m": m_star,
        "optimal_bins": B_star,
        "confidence_quality": cq,
    }
