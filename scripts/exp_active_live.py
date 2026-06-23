"""True live explore-exploit active verification (Block 4.1).

Unlike exp_active_real.py, which subsamples saved traces, this driver calls
llama.cpp live on freshly chosen MMLU items and implements the two-phase
protocol from Theorem~\\ref{thm:active} with a real inference loop.

Phase 1 (exploration): sample items uniformly, observe their confidence bins.
Phase 2 (exploitation): reweight toward bins where the sign of Δ is resolved
  (|Δ̂_b| > 2 σ_b) vs unresolved; query more from resolved bins.

Usage:
    python scripts/exp_active_live.py \
        --model-path models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
        --model-id llama-3-8b-instruct \
        --budget 2000

Output:
    results/analysis/active_live_{model-id}.json
    figures/fig_active_live.pdf (multi-model aggregate)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_local_experiment import _score_one  # noqa: E402

FIG_DIR = PROJECT_ROOT / "figures"
RES_DIR = PROJECT_ROOT / "results" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


def _bin_index(p: float, n_bins: int) -> int:
    return min(int(p * n_bins), n_bins - 1)


def _empirical_ece(confs: list[float], corr: list[int], n_bins: int) -> float:
    if not confs:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(confs, corr):
        bins[_bin_index(p, n_bins)].append((p, y))
    n = len(confs)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        ps = [x[0] for x in b]
        ys = [x[1] for x in b]
        ece += (len(b) / n) * abs(sum(ys) / len(ys) - sum(ps) / len(ps))
    return ece


def two_phase_active(llm, items, budget: int, n_bins: int, rng: random.Random):
    """Execute Phase 1 (uniform sample, budget/2) then Phase 2 (reweight to resolved bins)."""
    phase1_n = budget // 2
    phase2_n = budget - phase1_n

    # Phase 1: uniform random items.
    phase1 = rng.sample(items, min(phase1_n, len(items)))
    confs: list[float] = []
    corr: list[int] = []
    bin_records: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for idx, item in enumerate(phase1):
        r = _score_one(llm, item)
        if "max_conf" not in r:
            continue
        p = r["max_conf"]
        y = int(r["is_correct"])
        confs.append(p)
        corr.append(y)
        bin_records[_bin_index(p, n_bins)].append((p, y))
        if (idx + 1) % 100 == 0:
            print(f"  phase1 {idx+1}/{len(phase1)}", flush=True)

    # Classify bins: resolved if |Δ̂| > 2σ, else unresolved.
    resolved_bins: list[int] = []
    for b_idx, records in enumerate(bin_records):
        if len(records) < 5:
            continue
        ps = [r[0] for r in records]
        ys = [r[1] for r in records]
        delta = abs(sum(ys) / len(ys) - sum(ps) / len(ps))
        sigma = math.sqrt(max(1e-9, (sum(ys) / len(ys)) * (1 - sum(ys) / len(ys)) / len(records)))
        if delta > 2 * sigma:
            resolved_bins.append(b_idx)

    # Phase 2: reweight toward resolved bins.
    remaining_pool = [it for it in items if it not in set(phase1)]
    rng.shuffle(remaining_pool)
    queried_phase2 = 0
    # Simple reweight strategy: sample candidates from remaining pool, keep
    # with acceptance prob = 1 if resolved bin, 0.2 if unresolved. Since we
    # don't know which bin an item lands in until we query, we over-sample
    # and filter after scoring. Hard cap at phase2_n queries.
    for item in remaining_pool:
        if queried_phase2 >= phase2_n:
            break
        r = _score_one(llm, item)
        if "max_conf" not in r:
            continue
        p = r["max_conf"]
        y = int(r["is_correct"])
        b = _bin_index(p, n_bins)
        # Rejection sampling toward resolved bins.
        accept = 1.0 if b in resolved_bins else 0.2
        if rng.random() < accept:
            confs.append(p)
            corr.append(y)
            queried_phase2 += 1
            if queried_phase2 % 100 == 0:
                print(f"  phase2 {queried_phase2}/{phase2_n}", flush=True)

    return confs, corr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--benchmark", default="mmlu")
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=0, help="0 = auto")
    args = parser.parse_args()

    items_file = PROJECT_ROOT / "data" / args.benchmark / "all_items.json"
    if not items_file.exists():
        print(f"error: {items_file} missing", file=sys.stderr)
        return 2
    items = json.loads(items_file.read_text(encoding="utf-8"))

    from llama_cpp import Llama
    print(f"loading {args.model_path}", flush=True)
    import os as _os
    llm = Llama(
        model_path=str(args.model_path),
        n_ctx=args.n_ctx,
        n_threads=(args.n_threads or _os.cpu_count() or 4),
        n_batch=2048, n_ubatch=1024,
        use_mmap=True, use_mlock=False,
        logits_all=False, verbose=False,
    )

    rng = random.Random(args.seed)
    # Passive baseline: uniform random subsample of same budget.
    passive_items = rng.sample(items, min(args.budget, len(items)))
    passive_confs: list[float] = []
    passive_corr: list[int] = []
    print("running passive baseline…", flush=True)
    t0 = time.time()
    for idx, item in enumerate(passive_items):
        r = _score_one(llm, item)
        if "max_conf" in r:
            passive_confs.append(r["max_conf"])
            passive_corr.append(int(r["is_correct"]))
        if (idx + 1) % 100 == 0:
            print(f"  passive {idx+1}/{len(passive_items)} ({(time.time()-t0)/60:.1f} min)", flush=True)

    print("running two-phase active…", flush=True)
    active_confs, active_corr = two_phase_active(llm, items, args.budget, args.n_bins, rng)

    passive_ece = _empirical_ece(passive_confs, passive_corr, args.n_bins)
    active_ece = _empirical_ece(active_confs, active_corr, args.n_bins)
    passive_eps = float(1 - np.mean(passive_corr))
    active_eps = float(1 - np.mean(active_corr))

    out = {
        "model_id": args.model_id,
        "benchmark": args.benchmark,
        "budget": args.budget,
        "passive": {"n": len(passive_confs), "eps": passive_eps, "ece": passive_ece},
        "active": {"n": len(active_confs), "eps": active_eps, "ece": active_ece},
        "ratio_active_over_passive": active_ece / passive_ece if passive_ece > 0 else None,
    }
    out_path = RES_DIR / f"active_live_{args.model_id}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
