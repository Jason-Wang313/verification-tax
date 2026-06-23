"""Multi-benchmark / multi-stage pipeline verification tax (extends exp_pipeline_real.py).

Runs the same composition logic as exp_pipeline_real.py across:
  - 2-stage on MMLU (Llama-405B + Qwen3-80B) — the original experiment.
  - 2-stage on ARC-Challenge (Llama-405B + Qwen3-80B) — new in v5.
  - 3-stage on MMLU (Llama-405B + Qwen3-80B + DeepSeek-R1-Qwen-32B as adjudicator
    on disagreement) — new in v5, subject to traces being available.

Outputs:
  figures/fig_pipeline_multi.pdf       — 3-panel comparison.
  results/analysis/pipeline_multi_results.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "figures"
RES_DIR = PROJECT_ROOT / "results" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# Delegate reusable helpers to exp_pipeline_real's module.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from exp_pipeline_real import load_model, empirical_ece, estimate_lipschitz  # noqa: E402


def compose_two_stage(recs_a: dict, recs_b: dict, shared: list[str]):
    conf_A = np.array([recs_a[q]["conf"] for q in shared])
    corr_A = np.array([recs_a[q]["correct"] for q in shared])
    conf_B = np.array([recs_b[q]["conf"] for q in shared])
    corr_B = np.array([recs_b[q]["correct"] for q in shared])
    agree = conf_A == conf_A  # placeholder; agree defined below
    agree = corr_A == corr_B
    disagree = ~agree
    pipe_conf = np.empty(len(shared))
    pipe_corr = np.empty(len(shared), dtype=int)
    pipe_conf[agree] = np.maximum(conf_A[agree], conf_B[agree])
    pipe_corr[agree] = corr_A[agree]
    pipe_conf[disagree] = conf_B[disagree] * (1 - conf_A[disagree])
    pipe_corr[disagree] = corr_B[disagree]
    return {"conf": pipe_conf, "correct": pipe_corr,
            "frac_disagree": float(disagree.mean())}


def compose_three_stage(recs_a, recs_b, recs_c, shared):
    """A screens, B decides; if B and A disagree AND B is low-confidence (<0.6),
    delegate to adjudicator C. Confidence propagates multiplicatively on handoff."""
    conf_A = np.array([recs_a[q]["conf"] for q in shared])
    corr_A = np.array([recs_a[q]["correct"] for q in shared])
    conf_B = np.array([recs_b[q]["conf"] for q in shared])
    corr_B = np.array([recs_b[q]["correct"] for q in shared])
    conf_C = np.array([recs_c[q]["conf"] for q in shared])
    corr_C = np.array([recs_c[q]["correct"] for q in shared])

    agree_AB = corr_A == corr_B
    disagree_AB = ~agree_AB
    escalate = disagree_AB & (conf_B < 0.6)
    stop_at_B = disagree_AB & ~escalate

    pipe_conf = np.empty(len(shared))
    pipe_corr = np.empty(len(shared), dtype=int)
    pipe_conf[agree_AB] = np.maximum(conf_A[agree_AB], conf_B[agree_AB])
    pipe_corr[agree_AB] = corr_A[agree_AB]
    pipe_conf[stop_at_B] = conf_B[stop_at_B] * (1 - conf_A[stop_at_B])
    pipe_corr[stop_at_B] = corr_B[stop_at_B]
    pipe_conf[escalate] = conf_C[escalate] * (1 - conf_A[escalate]) * (1 - conf_B[escalate])
    pipe_corr[escalate] = corr_C[escalate]
    return {"conf": pipe_conf, "correct": pipe_corr,
            "frac_escalate": float(escalate.mean()),
            "frac_stop_at_B": float(stop_at_B.mean())}


def audit_panel(conf, correct, label, m_grid, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(conf)
    L_hat = estimate_lipschitz(conf, correct)
    eps_hat = float(1.0 - correct.mean())
    floors = [(L_hat * eps_hat / max(m, 1)) ** (1 / 3) for m in m_grid]
    # Bootstrap sampling errors at each m.
    boot = []
    for m in m_grid:
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=m)
            vals.append(empirical_ece(conf[idx], correct[idx], B=20))
        boot.append((float(np.mean(vals)), float(np.std(vals))))
    return {
        "label": label, "m_grid": list(m_grid), "L_hat": L_hat, "eps": eps_hat,
        "floor_curve": floors,
        "ece_mean": [b[0] for b in boot],
        "ece_std": [b[1] for b in boot],
    }


def _shared(records_list):
    keys = set(records_list[0].keys())
    for r in records_list[1:]:
        keys &= set(r.keys())
    return sorted(keys)


def run_single_config(name: str, paths: list[Path], compose_fn, m_grid):
    """Load all paths (single record set per path), compose, and audit."""
    for p in paths:
        if not p.exists():
            print(f"  [{name}] skipped: {p.name} not found", flush=True)
            return None
    records = []
    for p in paths:
        recs, _ = load_model(str(p))
        records.append(recs)
    shared = _shared(records)
    if len(shared) < 500:
        print(f"  [{name}] skipped: only {len(shared)} shared items", flush=True)
        return None
    composed = compose_fn(*records, shared=shared)
    conf = composed["conf"]
    corr = composed["correct"]
    # Single-model baseline = the first record set for reference.
    baseline_recs = records[0]
    b_conf = np.array([baseline_recs[q]["conf"] for q in shared])
    b_corr = np.array([baseline_recs[q]["correct"] for q in shared])
    return {
        "name": name,
        "n_shared": len(shared),
        "extras": {k: v for k, v in composed.items() if k not in ("conf", "correct")},
        "baseline": audit_panel(b_conf, b_corr, "single-model baseline", m_grid),
        "composed": audit_panel(conf, corr, "composed pipeline", m_grid),
    }


def main() -> int:
    MMLU = PROJECT_ROOT / "data" / "mmlu"
    ARC = PROJECT_ROOT / "data" / "arc_challenge"
    m_grid = [100, 200, 500, 1000, 2000, 5000]
    configs = []

    # Two-stage MMLU (original).
    configs.append(run_single_config(
        "2-stage MMLU",
        [MMLU / "results_llama-3.1-405b-instruct.jsonl",
         MMLU / "results_qwen3-next-80b.jsonl"],
        lambda a, b, shared: compose_two_stage(a, b, shared),
        m_grid,
    ))
    # Two-stage ARC-Challenge.
    configs.append(run_single_config(
        "2-stage ARC-Challenge",
        [ARC / "results_llama-3.1-405b-instruct.jsonl",
         ARC / "results_qwen3-next-80b.jsonl"],
        lambda a, b, shared: compose_two_stage(a, b, shared),
        m_grid,
    ))
    # Three-stage MMLU (Llama-405B + Qwen3-80B + DeepSeek-R1-Qwen-32B adjudicator).
    configs.append(run_single_config(
        "3-stage MMLU",
        [MMLU / "results_llama-3.1-405b-instruct.jsonl",
         MMLU / "results_qwen3-next-80b.jsonl",
         MMLU / "results_deepseek-r1-qwen-32b.jsonl"],
        lambda a, b, c, shared: compose_three_stage(a, b, c, shared),
        m_grid,
    ))

    configs = [c for c in configs if c is not None]

    # Save JSON.
    out_json = RES_DIR / "pipeline_multi_results.json"
    out_json.write_text(json.dumps(configs, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")

    # Plot 3-panel comparison.
    n_panels = len(configs)
    if n_panels == 0:
        print("no configs succeeded — nothing to plot.")
        return 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4.8 * n_panels, 4.5), sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, cfg in zip(axes, configs):
        ax.plot(cfg["baseline"]["m_grid"], cfg["baseline"]["ece_mean"], "o-", label=f"baseline (L={cfg['baseline']['L_hat']:.2f})")
        ax.plot(cfg["composed"]["m_grid"], cfg["composed"]["ece_mean"], "s--", label=f"composed (L={cfg['composed']['L_hat']:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(cfg["name"])
        ax.set_xlabel("m")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("ECE estimate")
    fig.tight_layout()
    out_pdf = FIG_DIR / "fig_pipeline_multi.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150)
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
