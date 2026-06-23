"""Round-trip tests for verification-tax core API and CLI."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from verification_tax import (
    VerificationAudit,
    audit_predictions,
    confidence_quality_screen,
    is_verifiable,
    verification_floor,
    active_floor,
    phase_transition,
    optimal_bins,
)


def test_verification_floor_scales_as_cubic():
    f1 = verification_floor(eps=0.1, n=1000, L=1.0)
    f8 = verification_floor(eps=0.1, n=8000, L=1.0)
    assert f1 / f8 == pytest.approx(2.0, rel=1e-6)


def test_active_floor_below_passive_at_frontier():
    eps, n = 0.05, 10_000
    assert active_floor(eps, n) < verification_floor(eps, n, L=1.0)


def test_phase_transition_at_inverse_eps():
    assert phase_transition(0.1) == 10
    assert phase_transition(0.01) == 100


def test_is_verifiable_verdicts():
    eps, n = 0.15, 14_042
    floor = verification_floor(eps, n)
    assert is_verifiable(3 * floor, eps, n)["verdict"] == "VERIFIED"
    assert is_verifiable(1.5 * floor, eps, n)["verdict"] == "MARGINAL"
    assert is_verifiable(0.5 * floor, eps, n)["verdict"] == "NOISE"


def test_confidence_quality_screen_rejects_fallback_collapse():
    # 96% of mass at 0.25 triggers the fallback rejection.
    rng = np.random.default_rng(0)
    p = np.concatenate([np.full(960, 0.25), rng.random(40)])
    out = confidence_quality_screen(p)
    assert not out["passed"]
    assert "fallback>95pct" in out["reasons"]


def test_confidence_quality_screen_accepts_spread():
    rng = np.random.default_rng(0)
    p = rng.beta(5, 2, size=2_000)
    out = confidence_quality_screen(p)
    assert out["passed"], out["reasons"]


def test_audit_predictions_returns_shape():
    rng = np.random.default_rng(0)
    p = rng.beta(5, 2, size=1000)
    y = (rng.random(1000) < p).astype(int)
    r = audit_predictions(p, y)
    assert {"m", "eps", "L_hat", "passive_floor", "active_floor", "phase_transition_m", "optimal_bins", "confidence_quality"} <= set(r)
    assert r["m"] == 1000
    assert 0 < r["passive_floor"] < 1


def test_oop_audit_matches_functional():
    rng = np.random.default_rng(42)
    p = rng.beta(5, 2, size=500)
    y = (rng.random(500) < p).astype(int)
    func = audit_predictions(p, y)
    oop = VerificationAudit(p, y)
    assert oop.passive_floor == func["passive_floor"]
    assert oop.active_floor == func["active_floor"]
    assert oop.phase_transition_m == func["phase_transition_m"]
    assert oop.m == func["m"]
    assert oop.eps == func["eps"]
    assert oop.L_hat == func["L_hat"]
    assert oop.optimal_bins == func["optimal_bins"]
    assert oop.confidence_quality == func["confidence_quality"]
    assert oop.to_dict() == func


def test_oop_report_prints_table(capsys):
    rng = np.random.default_rng(1)
    p = rng.beta(5, 2, size=200)
    y = (rng.random(200) < p).astype(int)
    VerificationAudit(p, y).report()
    captured = capsys.readouterr().out
    assert "passive floor" in captured
    assert "confidence quality" in captured


def test_optimal_bins_formula():
    # B* = (L^2 m / eps)^{1/3}; at m=1000, eps=0.1, L=1 => ~21
    b = optimal_bins(1000, 0.1, 1.0)
    assert 18 <= b <= 24


@pytest.mark.skipif(
    not Path("C:/Users/wangz/verification tax/data/mmlu/results_llama-3.1-405b-instruct.jsonl").exists(),
    reason="paper data not available",
)
def test_cli_audit_reproduces_paper_numbers():
    """The 405B MMLU audit must land in the paper's reported floor range (0.02-0.04)."""
    trace = Path("C:/Users/wangz/verification tax/data/mmlu/results_llama-3.1-405b-instruct.jsonl")
    result = subprocess.run(
        [sys.executable, "-m", "verification_tax.cli", "audit", str(trace), "--json"],
        check=True, capture_output=True, text=True,
    )
    out = json.loads(result.stdout)
    assert out["m"] == 14_042
    assert 0.13 <= out["eps"] <= 0.20
    assert 0.020 <= out["passive_floor"] <= 0.040
    assert out["confidence_quality"]["passed"] is True
