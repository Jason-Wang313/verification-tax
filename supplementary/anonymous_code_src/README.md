# Anonymous Verification Tools

This supplementary package contains a minimal Python implementation of the
verification-floor, holdout-sizing, verifiability-assessment, phase-transition,
and pipeline-depth utilities used in the submission.

The artifact is intentionally anonymized for double-blind review. Public package
names, repository links, and author metadata are omitted on purpose.

## Contents

- `verification_tools/core.py`: core utility functions
- `verification_tools/__init__.py`: package exports
- `tests/test_core.py`: lightweight smoke tests

## Installation

```bash
pip install .
```

## Quick Start

```python
from verification_tools import verification_floor, holdout_size, is_verifiable

floor = verification_floor(eps=0.15, n=14042, L=1.5)
needed = holdout_size(eps=0.15, delta=0.02, L=1.5)
verdict = is_verifiable(claimed_improvement=0.03, eps=0.15, n=14042, L=1.5)

print(floor)
print(needed)
print(verdict)
```

## Notes

The utilities implement the closed-form scaling laws used in the paper:

- passive verification floor: `(L * eps / n) ** (1/3)`
- passive holdout size: `ceil(L * eps / delta**3)`
- active verification floor: `sqrt(eps / n)`
- phase transition threshold: `ceil(1 / eps)`

