#!/usr/bin/env python3
"""
exp_benchmark_demolition.py
===========================
Computes the verification floor for major AI benchmarks and outputs
a LaTeX table (booktabs, NeurIPS-ready) and a JSON summary.

Verification floor formula:
    delta_floor = (L * epsilon / n)^(1/3)

where L = Lipschitz constant, epsilon = frontier error rate, n = benchmark size.
"""

import json
import math
import os
from datetime import datetime

# ── Paths (absolute) ─────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\wangz\verification tax"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
TEX_OUT = os.path.join(RESULTS_DIR, "benchmark_demolition_table.tex")
JSON_OUT = os.path.join(ANALYSIS_DIR, "benchmark_demolition.json")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Benchmark parameters ─────────────────────────────────────────────────────
BENCHMARKS = [
    {
        "name": "MMLU (per-subj)",
        "n": 250,
        "epsilon": 0.15,
        "delta_low": 0.02,
        "delta_high": 0.05,
        "notes": "Median subject size ~173-250",
    },
    {
        "name": "MMLU (full)",
        "n": 14042,
        "epsilon": 0.15,
        "delta_low": 0.02,
        "delta_high": 0.05,
        "notes": "Full test set",
    },
    {
        "name": "TruthfulQA",
        "n": 817,
        "epsilon": 0.40,
        "delta_low": 0.05,
        "delta_high": 0.15,
        "notes": "MC format",
    },
    {
        "name": "BBQ (per-cat)",
        "n": 500,
        "epsilon": 0.30,
        "delta_low": 0.03,
        "delta_high": 0.10,
        "notes": "9 bias categories",
    },
    {
        "name": "ToxiGen",
        "n": 940,
        "epsilon": 0.25,
        "delta_low": 0.05,
        "delta_high": 0.12,
        "notes": "Binary toxicity",
    },
    {
        "name": "HumanEval",
        "n": 164,
        "epsilon": 0.30,
        "delta_low": 0.05,
        "delta_high": 0.20,
        "notes": "Code gen pass@1",
    },
    {
        "name": "SWE-bench Verified",
        "n": 500,
        "epsilon": 0.70,
        "delta_low": 0.05,
        "delta_high": 0.15,
        "notes": "Resolved tasks",
    },
    {
        "name": "GPQA Diamond",
        "n": 198,
        "epsilon": 0.50,
        "delta_low": 0.05,
        "delta_high": 0.15,
        "notes": "Expert QA",
    },
    {
        "name": "WinoGender",
        "n": 720,
        "epsilon": 0.10,
        "delta_low": 0.02,
        "delta_high": 0.08,
        "notes": "Gender bias",
    },
    {
        "name": "GSM8K",
        "n": 1319,
        "epsilon": 0.08,
        "delta_low": 0.02,
        "delta_high": 0.05,
        "notes": "Math reasoning",
    },
    {
        "name": "ARC-Challenge",
        "n": 1172,
        "epsilon": 0.10,
        "delta_low": 0.02,
        "delta_high": 0.05,
        "notes": "Science QA",
    },
]


def verification_floor(L: float, epsilon: float, n: int) -> float:
    """Compute delta_floor = (L * epsilon / n)^(1/3)."""
    return (L * epsilon / n) ** (1.0 / 3.0)


def analyze_benchmarks():
    """Compute verification floors and classification for every benchmark."""
    results = []
    for b in BENCHMARKS:
        floor_L1 = verification_floor(1.0, b["epsilon"], b["n"])
        floor_L2 = verification_floor(2.0, b["epsilon"], b["n"])
        typical_delta = (b["delta_low"] + b["delta_high"]) / 2.0
        ratio_L1 = typical_delta / floor_L1 if floor_L1 > 0 else float("inf")
        ratio_L2 = typical_delta / floor_L2 if floor_L2 > 0 else float("inf")

        # A claimed improvement is "verifiable" if the typical claim
        # exceeds the floor at L=1 by a comfortable margin (ratio > 1).
        verifiable = ratio_L1 > 1.0

        # More granular classification
        if ratio_L1 < 1.0:
            status = "Below floor"
        elif ratio_L1 < 2.0:
            status = "Marginal"
        elif ratio_L1 < 5.0:
            status = "Plausible"
        else:
            status = "Robust"

        results.append(
            {
                "name": b["name"],
                "n": b["n"],
                "epsilon": b["epsilon"],
                "delta_low": b["delta_low"],
                "delta_high": b["delta_high"],
                "typical_delta": typical_delta,
                "floor_L1": floor_L1,
                "floor_L2": floor_L2,
                "ratio_L1": ratio_L1,
                "ratio_L2": ratio_L2,
                "verifiable": verifiable,
                "status": status,
                "notes": b["notes"],
            }
        )
    return results


def make_latex_table(results: list) -> str:
    """Generate a booktabs LaTeX table (NeurIPS-ready)."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Verification floor analysis for major AI benchmarks. "
                 r"$\delta_{\mathrm{floor}} = (L \varepsilon / n)^{1/3}$ with $L{=}1$. "
                 r"Ratio $= \Delta_{\mathrm{typ}} / \delta_{\mathrm{floor}}$; "
                 r"values $<1$ indicate claims below the verification floor.}")
    lines.append(r"\label{tab:benchmark-demolition}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrccrc}")
    lines.append(r"\toprule")
    lines.append(
        r"Benchmark & $n$ & $\varepsilon$ & $\delta_{\mathrm{floor}}$ "
        r"& Typical $\Delta$ & Ratio & Verifiable? \\"
    )
    lines.append(r"\midrule")

    for r in results:
        # Format the "Verifiable?" column with color cues
        if r["status"] == "Below floor":
            ver_str = r"\textcolor{red}{\ding{55}}"
        elif r["status"] == "Marginal":
            ver_str = r"$\sim$"
        else:
            ver_str = r"\checkmark"

        delta_range = f"{r['delta_low']:.2f}--{r['delta_high']:.2f}"

        lines.append(
            f"  {r['name']} & {r['n']:,} & {r['epsilon']:.2f} "
            f"& {r['floor_L1']:.4f} & {delta_range} "
            f"& {r['ratio_L1']:.1f}$\\times$ & {ver_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_json_summary(results: list) -> dict:
    """Build the JSON summary dict."""
    n_below = sum(1 for r in results if r["status"] == "Below floor")
    n_marginal = sum(1 for r in results if r["status"] == "Marginal")
    n_plausible = sum(1 for r in results if r["status"] == "Plausible")
    n_robust = sum(1 for r in results if r["status"] == "Robust")

    return {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "formula": "delta_floor = (L * epsilon / n)^(1/3)",
            "L_values": [1, 2],
            "description": (
                "Verification floor analysis: benchmarks where typical "
                "claimed improvements fall near or below the verification "
                "floor are unreliable for measuring true progress."
            ),
        },
        "summary": {
            "total_benchmarks": len(results),
            "below_floor": n_below,
            "marginal": n_marginal,
            "plausible": n_plausible,
            "robust": n_robust,
        },
        "benchmarks": results,
    }


def print_summary(results: list):
    """Print a human-readable summary to stdout."""
    hdr = (
        f"{'Benchmark':<20s} {'n':>6s} {'eps':>5s} {'floor(L=1)':>10s} "
        f"{'floor(L=2)':>10s} {'Typ Delta':>10s} {'Ratio':>6s} {'Status':<12s}"
    )
    print("=" * len(hdr))
    print("VERIFICATION FLOOR ANALYSIS — BENCHMARK DEMOLITION TABLE")
    print("=" * len(hdr))
    print(f"Formula: delta_floor = (L * epsilon / n)^(1/3)")
    print()
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        typ = f"{r['delta_low']:.2f}-{r['delta_high']:.2f}"
        print(
            f"{r['name']:<20s} {r['n']:>6d} {r['epsilon']:>5.2f} "
            f"{r['floor_L1']:>10.4f} {r['floor_L2']:>10.4f} "
            f"{typ:>10s} {r['ratio_L1']:>5.1f}x {r['status']:<12s}"
        )

    print("-" * len(hdr))

    n_below = sum(1 for r in results if r["status"] == "Below floor")
    n_marginal = sum(1 for r in results if r["status"] == "Marginal")
    n_plausible = sum(1 for r in results if r["status"] == "Plausible")
    n_robust = sum(1 for r in results if r["status"] == "Robust")

    print()
    print(f"Below floor : {n_below}/{len(results)}")
    print(f"Marginal    : {n_marginal}/{len(results)}")
    print(f"Plausible   : {n_plausible}/{len(results)}")
    print(f"Robust      : {n_robust}/{len(results)}")
    print()

    # Highlight the worst offenders
    worst = [r for r in results if r["ratio_L1"] < 2.0]
    if worst:
        print("BENCHMARKS MOST VULNERABLE TO VERIFICATION TAX:")
        for r in worst:
            print(f"  - {r['name']}: ratio = {r['ratio_L1']:.2f}x "
                  f"(floor = {r['floor_L1']:.4f}, typical claim = "
                  f"{r['delta_low']:.2f}-{r['delta_high']:.2f})")
    print()


def main():
    results = analyze_benchmarks()

    # 1. Print summary to stdout
    print_summary(results)

    # 2. Write LaTeX table
    tex = make_latex_table(results)
    with open(TEX_OUT, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"LaTeX table written to: {TEX_OUT}")

    # 3. Write JSON summary
    summary = make_json_summary(results)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary written to: {JSON_OUT}")


if __name__ == "__main__":
    main()
