"""
Expanded empirical audit across all available benchmark-model result files.

This script differs from analyze_all_benchmarks.py in two ways:
  1. it auto-discovers every results_*.jsonl file under data/<benchmark>/
  2. it reports coverage explicitly, so partial runs can be separated from
     high-coverage headline analyses

Outputs:
  - results/full_roster_benchmarks_table.tex
  - results/analysis/full_roster_benchmarks_summary.json
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

RESULTS_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)

BENCHMARKS = ["mmlu", "truthfulqa", "arc_challenge", "hellaswag", "winogrande"]
MIN_VALID = 100
HIGH_COVERAGE = 0.70


def load_results(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec:
                continue
            if "max_conf" not in rec or "is_correct" not in rec:
                continue
            records.append(
                {
                    "conf": float(rec["max_conf"]),
                    "correct": int(bool(rec["is_correct"])),
                }
            )
    return records


def empirical_ece(p, y, B):
    n = len(p)
    edges = np.linspace(0, 1, B + 1)
    ece = 0.0
    for b in range(B):
        if b == B - 1:
            mask = (p >= edges[b]) & (p <= edges[b + 1])
        else:
            mask = (p >= edges[b]) & (p < edges[b + 1])
        nb = mask.sum()
        if nb > 0:
            ece += (nb / n) * abs(np.mean(y[mask]) - np.mean(p[mask]))
    return ece


def optimal_B(N, eps, L_hat):
    if eps <= 0:
        return 15
    raw = (L_hat ** 2 * N / eps) ** (1.0 / 3.0)
    return max(2, min(50, int(np.floor(raw))))


def estimate_lipschitz(p, y, n_bins=20, min_per_bin=None, N=None):
    if min_per_bin is None:
        min_per_bin = max(10, (N or len(p)) // 100)

    edges = np.linspace(0, 1, n_bins + 1)
    centers = []
    accs = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.sum() >= min_per_bin:
            centers.append((edges[i] + edges[i + 1]) / 2)
            accs.append(y[mask].mean())

    centers = np.array(centers)
    accs = np.array(accs)
    if len(centers) < 2:
        return 1.0

    gaps = accs - centers
    slopes = []
    for i in range(len(gaps) - 1):
        d = abs(centers[i + 1] - centers[i])
        if d > 0:
            slopes.append(abs(gaps[i + 1] - gaps[i]) / d)
    if not slopes:
        return 1.0
    return float(min(np.percentile(slopes, 75), 5.0))


def estimate_lipschitz_90(p, y, n_bins=20, min_per_bin=None, N=None):
    """90th percentile estimator for L-hat (less conservative)."""
    if min_per_bin is None:
        min_per_bin = max(10, (N or len(p)) // 100)

    edges = np.linspace(0, 1, n_bins + 1)
    centers = []
    accs = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.sum() >= min_per_bin:
            centers.append((edges[i] + edges[i + 1]) / 2)
            accs.append(y[mask].mean())

    centers = np.array(centers)
    accs = np.array(accs)
    if len(centers) < 2:
        return 1.0

    gaps = accs - centers
    slopes = []
    for i in range(len(gaps) - 1):
        d = abs(centers[i + 1] - centers[i])
        if d > 0:
            slopes.append(abs(gaps[i + 1] - gaps[i]) / d)
    if not slopes:
        return 1.0
    return float(min(np.percentile(slopes, 90), 10.0))


def floor_sensitivity(eps, N, L_values=(1.0, 2.0, 3.0, 5.0)):
    """Compute passive floors at multiple L values for sensitivity analysis."""
    return {f"L={L:.0f}": float((L * eps / N) ** (1.0 / 3.0)) for L in L_values}


def self_eval_spearman(p, y, n_bins=20, min_per_bin=None, N=None):
    if min_per_bin is None:
        min_per_bin = max(10, (N or len(p)) // 100)

    edges = np.linspace(0, 1, n_bins + 1)
    mean_confs = []
    abs_gaps = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.sum() >= min_per_bin:
            mc = np.mean(p[mask])
            acc = np.mean(y[mask])
            mean_confs.append(mc)
            abs_gaps.append(abs(acc - mc))

    if len(mean_confs) < 3:
        return None, None

    rho, pval = stats.spearmanr(mean_confs, abs_gaps)
    return rho, pval


def load_total_items(benchmark: str):
    with open(DATA_DIR / benchmark / "all_items.json", encoding="utf-8") as f:
        return len(json.load(f))


def benchmark_label(key: str):
    labels = {
        "mmlu": "MMLU",
        "truthfulqa": "TruthfulQA",
        "arc_challenge": "ARC-Challenge",
        "hellaswag": "HellaSwag",
        "winogrande": "WinoGrande",
    }
    return labels[key]


def prettify_model(stem: str):
    return (
        stem.replace("results_", "")
        .replace("-instruct", "")
        .replace("-", " ")
        .replace("qwen", "Qwen")
        .replace("llama", "Llama")
        .replace("mistral", "Mistral")
        .replace("gemma", "Gemma")
        .replace("deepseek", "DeepSeek")
        .replace("nemotron", "Nemotron")
        .replace("phi", "Phi")
        .replace("next", "Next")
        .replace("mini", "Mini")
        .title()
    )


def main():
    rows = []
    total_items = {bench: load_total_items(bench) for bench in BENCHMARKS}

    print("=" * 88)
    print("FULL-ROSTER EMPIRICAL AUDIT")
    print("=" * 88)

    for bench in BENCHMARKS:
        print(f"\n--- {benchmark_label(bench)} ---")
        for path in sorted((DATA_DIR / bench).glob("results_*.jsonl")):
            records = load_results(path)
            N = len(records)
            if N < MIN_VALID:
                continue

            p = np.array([r["conf"] for r in records])
            y = np.array([r["correct"] for r in records])
            eps = float(1.0 - y.mean())
            L_hat = estimate_lipschitz(p, y, N=N)
            L_hat_90 = estimate_lipschitz_90(p, y, N=N)
            B_opt = optimal_B(N, eps, L_hat)
            ece = empirical_ece(p, y, B_opt)
            floor = (L_hat * eps / N) ** (1.0 / 3.0)
            rho, pval = self_eval_spearman(p, y, N=N)
            coverage = N / total_items[bench]
            sensitivity = floor_sensitivity(eps, N)

            row = {
                "benchmark": bench,
                "benchmark_label": benchmark_label(bench),
                "model_key": path.stem.replace("results_", ""),
                "model_label": prettify_model(path.stem),
                "N": N,
                "total_items": total_items[bench],
                "coverage": coverage,
                "eps": eps,
                "ece": ece,
                "L_hat": L_hat,
                "L_hat_90": L_hat_90,
                "B_opt": B_opt,
                "delta_floor": floor,
                "floor_sensitivity": sensitivity,
                "spearman_rho": rho,
                "spearman_p": pval,
                "high_coverage": coverage >= HIGH_COVERAGE,
            }
            rows.append(row)

            print(
                f"  {row['model_label'][:28]:28} "
                f"N={N:5d}/{total_items[bench]:5d} "
                f"cov={coverage:5.1%} eps={eps:0.3f} "
                f"ECE={ece:0.3f} L={L_hat:0.2f} floor={floor:0.3f}"
            )

    high_cov = [r for r in rows if r["high_coverage"]]
    nonsig = [r for r in high_cov if r["spearman_rho"] is not None and r["spearman_p"] > 0.05]
    with_rho = [r for r in high_cov if r["spearman_rho"] is not None]

    summary = {
        "min_valid_threshold": MIN_VALID,
        "high_coverage_threshold": HIGH_COVERAGE,
        "total_pairs_with_N_ge_100": len(rows),
        "total_high_coverage_pairs": len(high_cov),
        "benchmarks": {bench: total_items[bench] for bench in BENCHMARKS},
        "coverage_range": [min(r["coverage"] for r in rows), max(r["coverage"] for r in rows)],
        "eps_range": [min(r["eps"] for r in rows), max(r["eps"] for r in rows)],
        "delta_floor_range": [min(r["delta_floor"] for r in rows), max(r["delta_floor"] for r in rows)],
        "L_hat_range": [min(r["L_hat"] for r in rows), max(r["L_hat"] for r in rows)],
        "L_hat_90_range": [min(r["L_hat_90"] for r in rows), max(r["L_hat_90"] for r in rows)],
        "self_eval_fraction_nonsignificant_high_coverage": (
            len(nonsig) / len(with_rho) if with_rho else None
        ),
        "self_eval_nonsig_high_coverage_count": len(nonsig),
        "self_eval_high_coverage_total_with_rho": len(with_rho),
        "per_pair_results": rows,
    }

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Expanded full-roster audit across all usable benchmark--model result files. Coverage is the fraction of the benchmark with valid per-item outputs. High-coverage pairs have coverage $\\geq 70\\%$.}",
        "\\label{tab:full-roster}",
        "\\small",
        "\\begin{tabular}{llrrcccc}",
        "\\toprule",
        "Benchmark & Model & $N$ & Cov. & $\\varepsilon$ & ECE & $\\hat{L}$ & $\\delta_{\\mathrm{floor}}$ \\\\",
        "\\midrule",
    ]

    for bench in BENCHMARKS:
        bench_rows = [r for r in rows if r["benchmark"] == bench]
        bench_rows.sort(key=lambda r: (-r["high_coverage"], -r["coverage"], r["model_label"]))
        for row in bench_rows:
            cov_str = f"{100 * row['coverage']:.0f}\\%"
            tex_lines.append(
                f"{row['benchmark_label']} & {row['model_label']} & {row['N']:,} & {cov_str} & "
                f"{row['eps']:.3f} & {row['ece']:.3f} & {row['L_hat']:.2f} & {row['delta_floor']:.3f} \\\\"
            )
        tex_lines.append("\\midrule")

    tex_lines[-1] = "\\bottomrule"
    tex_lines.extend(["\\end{tabular}", "\\end{table}"])

    (RESULTS_DIR / "full_roster_benchmarks_table.tex").write_text(
        "\n".join(tex_lines), encoding="utf-8"
    )
    (ANALYSIS_DIR / "full_roster_benchmarks_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print(f"Usable pairs (N >= {MIN_VALID}): {len(rows)}")
    print(f"High-coverage pairs (coverage >= {HIGH_COVERAGE:.0%}): {len(high_cov)}")
    if with_rho:
        print(
            "High-coverage self-eval non-significant fraction: "
            f"{len(nonsig)}/{len(with_rho)} = {len(nonsig)/len(with_rho):.2f}"
        )
    print(f"Wrote {RESULTS_DIR / 'full_roster_benchmarks_table.tex'}")
    print(f"Wrote {ANALYSIS_DIR / 'full_roster_benchmarks_summary.json'}")


if __name__ == "__main__":
    main()
