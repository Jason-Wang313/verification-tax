# verification-tax

Information-theoretic floors and auditing tools for AI evaluation, based on *The Verification Tax* (Wang, 2026).

Given per-item `(confidence, correct)` traces, compute:
- the passive verification floor `(L·ε/m)^{1/3}` (Theorem 3),
- the active verification floor `√(ε/m)` (Theorem A),
- the phase-transition threshold `m* ≈ 1/ε` (Corollary 1),
- a confidence-quality screen (rejects traces that collapse to fallback values).

## Install

```bash
pip install verification-tax
```

## 5-minute reviewer demo

```bash
# JSONL with per-item {"confidence": ..., "correct": 0/1} rows
verification-tax audit your_results.jsonl

# Or CSV with columns: confidence,correct
verification-tax audit your_results.csv --json
```

Output on a real MMLU trace (Llama-3.1-405B, n=14,042):

```
  items (m)                : 14,042
  error rate (eps)         : 0.1596
  Lipschitz estimate (L)   : 1.610
  passive floor            : 0.0264
  active floor             : 0.0034
  phase-transition m*      : 7
  optimal bin count (B*)   : 61
  confidence quality       : PASS
```

Any claimed calibration improvement below the passive floor is statistically indistinguishable from noise. The same budget under active querying reaches a much smaller floor.

## Python API

One-liner (OOP):

```python
from verification_tax import VerificationAudit

audit = VerificationAudit(confidences, labels)
audit.report()
# items (m)                : 14,042
# error rate (eps)         : 0.1596
# Lipschitz estimate (L)   : 1.610
# passive floor            : 0.0264
# active floor             : 0.0034
# phase-transition m*      : 7
# optimal bin count (B*)   : 61
# confidence quality       : PASS

audit.passive_floor     # 0.0264
audit.active_floor      # 0.0034
audit.to_dict()         # full dict
```

Functional API (same data, more direct):

```python
from verification_tax import audit_predictions, is_verifiable, verification_floor

report = audit_predictions(confidences, correct)

is_verifiable(claimed_improvement=0.02, eps=0.05, n=14_042)
# {'floor': 0.01747, 'ratio': 1.14, 'verdict': 'MARGINAL', 'active_floor': 0.00224}

verification_floor(eps=0.05, n=14_042, L=1.4)   # 0.01747
```

## What the confidence-quality screen catches

~40% of high-coverage model runs we audited produced confidence outputs that collapse to fallback values (constant 0.25, 0.50, or 1.00). Accuracy is still measurable, but calibration shape is not. The screen flags this failure mode explicitly so reviewers can exclude those runs from calibration claims.

Rejection criteria (from `scripts/analyze_screened_roster.py`):
- `std < 0.02`
- `< 20` unique confidence levels (rounded to 3 decimals)
- any single value carries `> 95%` of the mass
- fallback values {0.25, 0.50, 1.00} share `> 95%` of the mass

## Citation

```bibtex
@article{wang2026verification,
  title   = {The Verification Tax: Fundamental Limits of AI Auditing in the Rare-Error Regime},
  author  = {Wang, Jason},
  year    = {2026}
}
```

## Compatibility

The package was previously named `vtax` (v0.1.0). A deprecation shim is kept in `vtax/` so old imports still work; please migrate to `verification_tax` for new code.
