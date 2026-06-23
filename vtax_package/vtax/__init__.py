"""Deprecation shim: `vtax` was renamed to `verification_tax` in 1.0.0."""

import warnings as _warnings

_warnings.warn(
    "The `vtax` import name is a compatibility shim; please migrate to `verification_tax`.",
    DeprecationWarning,
    stacklevel=2,
)

from verification_tax.core import (  # noqa: E402
    VerificationAudit,
    verification_floor,
    holdout_size,
    active_floor,
    is_verifiable,
    phase_transition,
    max_pipeline_depth,
    optimal_bins,
    audit_predictions,
    confidence_quality_screen,
    estimate_lipschitz,
)

__version__ = "1.1.0"
__all__ = [
    "VerificationAudit",
    "verification_floor",
    "holdout_size",
    "active_floor",
    "is_verifiable",
    "phase_transition",
    "max_pipeline_depth",
    "optimal_bins",
    "audit_predictions",
    "confidence_quality_screen",
    "estimate_lipschitz",
]
