from verification_tools import (
    active_floor,
    holdout_size,
    is_verifiable,
    max_pipeline_depth,
    phase_transition,
    verification_floor,
)


def run_smoke_tests() -> None:
    floor = verification_floor(eps=0.15, n=14042, L=1.5)
    assert 0.0 < floor < 0.1

    needed = holdout_size(eps=0.15, delta=0.02, L=1.5)
    assert needed == 28125

    active = active_floor(eps=0.15, n=14042)
    assert active < floor

    verdict = is_verifiable(claimed_improvement=0.03, eps=0.15, n=14042, L=1.5)
    assert verdict["verdict"] in {"MARGINAL", "VERIFIED", "NOISE"}

    assert phase_transition(0.05) == 20
    assert max_pipeline_depth(M_total=100000, eps=0.1, delta=0.02, L=2.0) >= 0


if __name__ == "__main__":
    run_smoke_tests()
    print("anonymous supplement smoke tests passed")
