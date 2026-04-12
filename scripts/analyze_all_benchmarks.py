"""
Unified analysis across all 5 benchmarks for the Verification Tax paper.

Produces:
  1. results/all_benchmarks_table.tex         -- LaTeX summary table
  2. results/analysis/all_benchmarks_summary.json  -- Cross-benchmark JSON summary
  3. figures/fig_all_benchmarks.pdf / .png     -- Bar chart of verification floors
"""

import json
import os
import sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths (absolute)
# ---------------------------------------------------------------------------
BASE_DIR = r"C:/Users/wangz/verification tax"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Benchmarks & models
# ---------------------------------------------------------------------------
BENCHMARKS = ["mmlu", "truthfulqa", "arc_challenge", "hellaswag", "winogrande"]

MODELS = {
    "llama-3.1-405b-instruct": "Llama-3.1-405B",
    "llama-4-maverick":        "Llama-4-Maverick",
    "qwen3-next-80b":          "Qwen3-Next-80B",
}

MIN_VALID = 100  # minimum valid items to include a (benchmark, model) pair

# ---------------------------------------------------------------------------
# Utility: load JSONL results
# ---------------------------------------------------------------------------
def load_results(path):
    """Load valid (no-error) records from a results JSONL file."""
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
            records.append({
                "conf": float(rec["max_conf"]),
                "correct": int(bool(rec["is_correct"])),
            })
    return records

# ---------------------------------------------------------------------------
# ECE with specified B
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Optimal bin count from T3
# ---------------------------------------------------------------------------
def optimal_B(N, eps, L_hat):
    """B = max(2, min(50, floor((L^2 * n / eps)^{1/3})))"""
    if eps <= 0:
        return 15  # fallback
    raw = (L_hat ** 2 * N / eps) ** (1.0 / 3.0)
    return max(2, min(50, int(np.floor(raw))))

# ---------------------------------------------------------------------------
# Lipschitz estimate
# ---------------------------------------------------------------------------
def estimate_lipschitz(p, y, n_bins=20, min_per_bin=None, N=None):
    """
    Estimate L via finite-difference on the empirical calibration curve.
    20 bins, min occupancy = max(10, N//100), 75th percentile slopes, cap at 5.
    """
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

    gaps = accs - centers  # calibration gap Delta(p)
    slopes = []
    for i in range(len(gaps) - 1):
        d = abs(centers[i + 1] - centers[i])
        if d > 0:
            slopes.append(abs(gaps[i + 1] - gaps[i]) / d)

    if not slopes:
        return 1.0

    return float(min(np.percentile(slopes, 75), 5.0))

# ---------------------------------------------------------------------------
# Self-eval: Spearman correlation of confidence vs |calibration gap| across bins
# ---------------------------------------------------------------------------
def self_eval_spearman(p, y, n_bins=20, min_per_bin=None, N=None):
    """
    Compute Spearman correlation between bin-level mean confidence
    and |calibration gap| across bins.
    Returns (rho, pvalue) or (None, None) if not enough bins.
    """
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
            gap = abs(acc - mc)
            mean_confs.append(mc)
            abs_gaps.append(gap)

    if len(mean_confs) < 3:
        return None, None

    rho, pval = stats.spearmanr(mean_confs, abs_gaps)
    return rho, pval

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("UNIFIED ANALYSIS ACROSS ALL 5 BENCHMARKS")
    print("=" * 80)
    print()

    # Collect all (benchmark, model) results
    all_rows = []

    for bench in BENCHMARKS:
        print(f"--- {bench.upper()} ---")
        for model_key, model_label in MODELS.items():
            fname = f"results_{model_key}.jsonl"
            path = os.path.join(DATA_DIR, bench, fname)
            if not os.path.exists(path):
                print(f"  {model_label}: FILE MISSING ({fname})")
                continue

            records = load_results(path)
            N = len(records)
            if N < MIN_VALID:
                print(f"  {model_label}: only {N} valid items (< {MIN_VALID}), SKIPPED")
                continue

            p = np.array([r["conf"] for r in records])
            y = np.array([r["correct"] for r in records])

            # 1. Error rate
            eps = float(1.0 - y.mean())

            # 4. Lipschitz estimate
            L_hat = estimate_lipschitz(p, y, n_bins=20, N=N)

            # 3. ECE with optimal B
            B_opt = optimal_B(N, eps, L_hat)
            ece = empirical_ece(p, y, B_opt)

            # 5. Verification floor
            delta_floor = (L_hat * eps / N) ** (1.0 / 3.0)

            # 6. Self-eval Spearman
            rho, pval = self_eval_spearman(p, y, n_bins=20, N=N)

            row = {
                "benchmark": bench,
                "model_key": model_key,
                "model_label": model_label,
                "N": N,
                "eps": eps,
                "B_opt": B_opt,
                "ece": ece,
                "L_hat": L_hat,
                "delta_floor": delta_floor,
                "spearman_rho": rho,
                "spearman_p": pval,
            }
            all_rows.append(row)

            rho_str = f"{rho:+.3f} (p={pval:.3f})" if rho is not None else "N/A"
            print(f"  {model_label}: N={N:,}  eps={eps:.4f}  ECE={ece:.4f}  "
                  f"L_hat={L_hat:.3f}  floor={delta_floor:.4f}  self-eval r={rho_str}")
        print()

    if not all_rows:
        print("ERROR: No valid (benchmark, model) pairs found!")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # OUTPUT 1: LaTeX summary table
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("GENERATING LaTeX TABLE: results/all_benchmarks_table.tex")
    print("=" * 80)

    # Pretty benchmark names for the table
    bench_names = {
        "mmlu": "MMLU",
        "truthfulqa": "TruthfulQA",
        "arc_challenge": "ARC-Challenge",
        "hellaswag": "HellaSwag",
        "winogrande": "WinoGrande",
    }

    tex_lines = []
    tex_lines.append(r"\begin{table}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{Verification tax across five benchmarks and three models. "
                     r"$\hat{L}$ is the estimated Lipschitz constant, "
                     r"$\delta_{\mathrm{floor}}$ is the verification floor, "
                     r"and Self-eval $r$ is the Spearman correlation between "
                     r"confidence and $|\text{calibration gap}|$ across bins "
                     r"(non-significant $p > 0.05$ supports Theorem~\ref{thm:self-verif}).}")
    tex_lines.append(r"\label{tab:all-benchmarks}")
    tex_lines.append(r"\small")
    tex_lines.append(r"\begin{tabular}{llrccccl}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"Benchmark & Model & $N$ & $\varepsilon$ & ECE & $\hat{L}$ "
                     r"& $\delta_{\mathrm{floor}}$ & Self-eval $r$ ($p$) \\")
    tex_lines.append(r"\midrule")

    prev_bench = None
    for row in all_rows:
        bench = row["benchmark"]
        if bench != prev_bench:
            if prev_bench is not None:
                tex_lines.append(r"\midrule")
            prev_bench = bench

        bname = bench_names.get(bench, bench)
        model = row["model_label"]
        N = row["N"]
        eps = row["eps"]
        ece = row["ece"]
        L_hat = row["L_hat"]
        delta_floor = row["delta_floor"]
        rho = row["spearman_rho"]
        pval = row["spearman_p"]

        if rho is not None:
            rho_str = f"${rho:+.2f}$ ($p={pval:.2f}$)"
        else:
            rho_str = "---"

        tex_lines.append(
            f"{bname} & {model} & {N:,} & {eps:.3f} & {ece:.4f} & "
            f"{L_hat:.2f} & {delta_floor:.4f} & {rho_str} \\\\"
        )

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\end{table}")

    tex_path = os.path.join(RESULTS_DIR, "all_benchmarks_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"  Saved to {tex_path}")
    print()

    # Print the table content for review
    print("--- LaTeX table content ---")
    for line in tex_lines:
        print(line)
    print()

    # -----------------------------------------------------------------------
    # OUTPUT 2: Cross-benchmark summary JSON
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("GENERATING SUMMARY: results/analysis/all_benchmarks_summary.json")
    print("=" * 80)

    eps_vals = [r["eps"] for r in all_rows]
    ece_vals = [r["ece"] for r in all_rows]
    L_vals = [r["L_hat"] for r in all_rows]
    floor_vals = [r["delta_floor"] for r in all_rows]

    # Self-eval: fraction with p > 0.05 (non-significant)
    pairs_with_rho = [r for r in all_rows if r["spearman_rho"] is not None]
    non_significant = [r for r in pairs_with_rho if r["spearman_p"] > 0.05]
    frac_nonsig = len(non_significant) / len(pairs_with_rho) if pairs_with_rho else None

    # Mean verification floor per benchmark
    mean_floor_by_bench = {}
    for bench in BENCHMARKS:
        bench_rows = [r for r in all_rows if r["benchmark"] == bench]
        if bench_rows:
            mean_floor_by_bench[bench] = float(np.mean([r["delta_floor"] for r in bench_rows]))

    summary = {
        "total_benchmarks": len(BENCHMARKS),
        "total_pairs_with_N_ge_100": len(all_rows),
        "eps_range": [float(min(eps_vals)), float(max(eps_vals))],
        "ece_range": [float(min(ece_vals)), float(max(ece_vals))],
        "L_hat_range": [float(min(L_vals)), float(max(L_vals))],
        "delta_floor_range": [float(min(floor_vals)), float(max(floor_vals))],
        "self_eval_fraction_nonsignificant": frac_nonsig,
        "self_eval_nonsig_count": len(non_significant),
        "self_eval_total_with_rho": len(pairs_with_rho),
        "mean_verification_floor_per_benchmark": mean_floor_by_bench,
        "per_pair_results": [
            {
                "benchmark": r["benchmark"],
                "model": r["model_label"],
                "N": r["N"],
                "eps": round(r["eps"], 6),
                "ece": round(r["ece"], 6),
                "L_hat": round(r["L_hat"], 4),
                "delta_floor": round(r["delta_floor"], 6),
                "spearman_rho": round(r["spearman_rho"], 4) if r["spearman_rho"] is not None else None,
                "spearman_p": round(r["spearman_p"], 4) if r["spearman_p"] is not None else None,
            }
            for r in all_rows
        ],
    }

    json_path = os.path.join(ANALYSIS_DIR, "all_benchmarks_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {json_path}")
    print()

    # Print summary
    print("--- Cross-Benchmark Summary ---")
    print(f"  Total benchmarks:                    {summary['total_benchmarks']}")
    print(f"  Total (benchmark, model) pairs:      {summary['total_pairs_with_N_ge_100']}")
    print(f"  eps range:                           [{summary['eps_range'][0]:.4f}, {summary['eps_range'][1]:.4f}]")
    print(f"  ECE range:                           [{summary['ece_range'][0]:.4f}, {summary['ece_range'][1]:.4f}]")
    print(f"  L_hat range:                         [{summary['L_hat_range'][0]:.3f}, {summary['L_hat_range'][1]:.3f}]")
    print(f"  delta_floor range:                   [{summary['delta_floor_range'][0]:.4f}, {summary['delta_floor_range'][1]:.4f}]")
    if frac_nonsig is not None:
        print(f"  Self-eval non-significant fraction:  {frac_nonsig:.2f} "
              f"({len(non_significant)}/{len(pairs_with_rho)})")
    else:
        print(f"  Self-eval non-significant fraction:  N/A (no pairs with rho)")
    print(f"  Mean verification floor by benchmark:")
    for bench, mf in mean_floor_by_bench.items():
        print(f"    {bench_names.get(bench, bench):20s}: {mf:.4f}")
    print()

    # -----------------------------------------------------------------------
    # OUTPUT 3: Bar chart of verification floor per benchmark
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("GENERATING FIGURE: figures/fig_all_benchmarks.pdf / .png")
    print("=" * 80)

    bench_labels = []
    bench_floors = []
    bench_colors = []
    color_map = {
        "mmlu":          "#2176AE",
        "truthfulqa":    "#E36414",
        "arc_challenge": "#57A773",
        "hellaswag":     "#8B5CF6",
        "winogrande":    "#E74C3C",
    }

    for bench in BENCHMARKS:
        if bench in mean_floor_by_bench:
            bench_labels.append(bench_names.get(bench, bench))
            bench_floors.append(mean_floor_by_bench[bench])
            bench_colors.append(color_map.get(bench, "#888888"))

    if not bench_labels:
        print("  ERROR: No benchmarks with data to plot!")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Style: serif fonts
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    plt.rcParams["mathtext.fontset"] = "cm"

    bars = ax.bar(
        range(len(bench_labels)),
        bench_floors,
        color=bench_colors,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
        zorder=3,
    )

    # Reference line at delta = 0.05
    ref_line = 0.05
    ax.axhline(y=ref_line, color="red", linestyle="--", linewidth=1.2, zorder=2,
               label=r"$\delta = 0.05$ (typical claimed improvement)")

    # Annotate bars above the reference line
    for i, (label, floor_val) in enumerate(zip(bench_labels, bench_floors)):
        y_offset = floor_val + 0.001
        if floor_val >= ref_line:
            ax.annotate(
                r"$\delta_{\mathrm{floor}} > 0.05$",
                xy=(i, floor_val),
                xytext=(i, floor_val + 0.006),
                ha="center", va="bottom",
                fontsize=8,
                fontweight="bold",
                color="red",
                arrowprops=dict(arrowstyle="-", color="red", lw=0.5),
            )

    ax.set_xticks(range(len(bench_labels)))
    ax.set_xticklabels(bench_labels, fontsize=10, fontfamily="serif")
    ax.set_ylabel(r"Verification floor $\delta_{\mathrm{floor}}$", fontsize=11, fontfamily="serif")
    ax.set_title("Verification Floor by Benchmark (averaged across models)", fontsize=12, fontfamily="serif")

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Grid
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Add value labels on bars
    for i, v in enumerate(bench_floors):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontfamily="serif")

    plt.tight_layout()

    pdf_path = os.path.join(FIG_DIR, "fig_all_benchmarks.pdf")
    png_path = os.path.join(FIG_DIR, "fig_all_benchmarks.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {pdf_path}")
    print(f"  Saved to {png_path}")
    print()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("DONE. All outputs generated.")
    print(f"  1. LaTeX table:   {tex_path}")
    print(f"  2. JSON summary:  {json_path}")
    print(f"  3. Figure (PDF):  {pdf_path}")
    print(f"  3. Figure (PNG):  {png_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
