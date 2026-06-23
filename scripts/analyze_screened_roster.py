"""
Screened full-roster audit for direct calibration claims.

This script formalizes the inclusion rule for publication-grade
item-level calibration analyses. It separates:
  1. all usable pairs with N >= 100
  2. high-coverage pairs (coverage >= 70%)
  3. screened headline pairs with high coverage and non-degenerate
     confidence support

Outputs:
  - results/analysis/screened_roster_summary.json
  - results/screened_roster_table.tex
  - results/confidence_quality_table.tex
  - figures/fig_confidence_quality_audit.pdf/.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

BENCHMARKS = ["mmlu", "truthfulqa", "arc_challenge", "hellaswag", "winogrande"]
MIN_VALID = 100
HIGH_COVERAGE = 0.70
STD_MIN = 0.02
UNIQUE_MIN = 20
DOMINANT_SHARE_MAX = 0.95
FALLBACK_SHARE_MAX = 0.95
FALLBACK_VALUES = (0.25, 0.50, 1.00)


def benchmark_label(key: str) -> str:
    labels = {
        "mmlu": "MMLU",
        "truthfulqa": "TruthfulQA",
        "arc_challenge": "ARC-Challenge",
        "hellaswag": "HellaSwag",
        "winogrande": "WinoGrande",
    }
    return labels[key]


def prettify_model(stem: str) -> str:
    return (
        stem.replace("results_", "")
        .replace("-instruct", "")
        .replace("-", " ")
        .replace("qwen", "Qwen")
        .replace("llama", "Llama")
        .replace("mistral", "Mistral")
        .replace("gemma", "Gemma")
        .replace("deepseek", "DeepSeek")
        .replace("next", "Next")
        .title()
    )


def load_total_items(benchmark: str) -> int:
    return len(json.loads((DATA_DIR / benchmark / "all_items.json").read_text(encoding="utf-8")))


def load_results(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec or "max_conf" not in rec or "is_correct" not in rec:
                continue
            records.append({"conf": float(rec["max_conf"]), "correct": int(bool(rec["is_correct"]))})
    return records


def empirical_ece(p: np.ndarray, y: np.ndarray, bins: int) -> float:
    n = len(p)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (p >= edges[idx]) & (p <= edges[idx + 1])
        else:
            mask = (p >= edges[idx]) & (p < edges[idx + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / n) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(total)


def estimate_lipschitz(p: np.ndarray, y: np.ndarray, n_bins: int = 20, min_per_bin: int | None = None) -> float:
    if min_per_bin is None:
        min_per_bin = max(10, len(p) // 100)
    edges = np.linspace(0, 1, n_bins + 1)
    centers = []
    accs = []
    for idx in range(n_bins):
        if idx == n_bins - 1:
            mask = (p >= edges[idx]) & (p <= edges[idx + 1])
        else:
            mask = (p >= edges[idx]) & (p < edges[idx + 1])
        if int(mask.sum()) >= min_per_bin:
            centers.append((edges[idx] + edges[idx + 1]) / 2)
            accs.append(float(np.mean(y[mask])))
    if len(centers) < 2:
        return 1.0
    centers = np.array(centers)
    gaps = np.array(accs) - centers
    slopes = []
    for idx in range(len(gaps) - 1):
        delta = abs(float(centers[idx + 1] - centers[idx]))
        if delta > 0:
            slopes.append(abs(float(gaps[idx + 1] - gaps[idx])) / delta)
    if not slopes:
        return 1.0
    return float(min(np.percentile(slopes, 75), 5.0))


def optimal_bins(n: int, eps: float, l_hat: float) -> int:
    if eps <= 0:
        return 15
    raw = (l_hat**2 * n / eps) ** (1.0 / 3.0)
    return max(2, min(50, int(np.floor(raw))))


def self_eval_spearman(p: np.ndarray, y: np.ndarray, n_bins: int = 20, min_per_bin: int | None = None):
    if min_per_bin is None:
        min_per_bin = max(10, len(p) // 100)
    edges = np.linspace(0, 1, n_bins + 1)
    mean_confs = []
    abs_gaps = []
    for idx in range(n_bins):
        if idx == n_bins - 1:
            mask = (p >= edges[idx]) & (p <= edges[idx + 1])
        else:
            mask = (p >= edges[idx]) & (p < edges[idx + 1])
        if int(mask.sum()) >= min_per_bin:
            mean_conf = float(np.mean(p[mask]))
            mean_acc = float(np.mean(y[mask]))
            mean_confs.append(mean_conf)
            abs_gaps.append(abs(mean_acc - mean_conf))
    if len(mean_confs) < 3:
        return None, None
    rho, p_val = stats.spearmanr(mean_confs, abs_gaps)
    return float(rho), float(p_val)


def quality_metrics(p: np.ndarray) -> dict[str, float]:
    rounded = np.round(p, 3)
    values, counts = np.unique(rounded, return_counts=True)
    dominant_idx = int(np.argmax(counts))
    fallback_shares = {f"{value:.2f}": float(np.mean(np.isclose(p, value, atol=1e-6))) for value in FALLBACK_VALUES}
    return {
        "confidence_std": float(np.std(p)),
        "rounded_unique": int(len(values)),
        "dominant_value": float(values[dominant_idx]),
        "dominant_share": float(counts[dominant_idx] / len(p)),
        "fallback_shares": fallback_shares,
        "max_fallback_share": float(max(fallback_shares.values())),
    }


def screen_row(row: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if row["coverage"] < HIGH_COVERAGE:
        reasons.append("coverage<70pct")
    if row["confidence_std"] < STD_MIN:
        reasons.append("std<0.02")
    if row["rounded_unique"] < UNIQUE_MIN:
        reasons.append("unique<20")
    if row["dominant_share"] > DOMINANT_SHARE_MAX:
        reasons.append("dominant>95pct")
    if row["max_fallback_share"] > FALLBACK_SHARE_MAX:
        reasons.append("fallback>95pct")
    return len(reasons) == 0, reasons


def write_screened_table(screened_rows: list[dict]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Screened headline set for direct item-level calibration claims. Pairs are included only when they have at least 100 valid items, at least 70\\% benchmark coverage, and non-degenerate confidence support.}",
        "\\label{tab:screened-roster}",
        "\\small",
        "\\begin{tabular}{llrrcccc}",
        "\\toprule",
        "Benchmark & Model & $m$ & Cov. & $\\varepsilon$ & ECE & $\\hat{L}$ & $\\delta_{\\mathrm{floor}}$ \\\\",
        "\\midrule",
    ]
    current_benchmark = None
    for row in screened_rows:
        if current_benchmark is not None and current_benchmark != row["benchmark"]:
            lines.append("\\midrule")
        current_benchmark = row["benchmark"]
        lines.append(
            f"{row['benchmark_label']} & {row['model_label']} & {row['N']:,} & "
            f"{100 * row['coverage']:.0f}\\% & {row['eps']:.3f} & {row['ece']:.3f} & "
            f"{row['L_hat']:.2f} & {row['delta_floor']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (RESULTS_DIR / "screened_roster_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_quality_table(rows: list[dict]) -> None:
    high_cov_excluded = [row for row in rows if row["coverage"] >= HIGH_COVERAGE and not row["screened_headline"]]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Confidence-quality audit for high-coverage pairs excluded from the headline set. These runs retain usable accuracy information but their confidence traces are too degenerate for calibration-shape claims.}",
        "\\label{tab:confidence-quality}",
        "\\footnotesize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{p{1.8cm}p{2.6cm}rrp{1.2cm}p{2.0cm}p{4.6cm}}",
        "\\toprule",
        "Benchmark & Model & $m$ & Std(conf.) & Unique$_{0.001}$ & Dominant share & Exclusion reason \\\\",
        "\\midrule",
    ]
    for row in high_cov_excluded:
        lines.append(
            f"{row['benchmark_label']} & {row['model_label']} & {row['N']:,} & "
            f"{row['confidence_std']:.3f} & {row['rounded_unique']} & "
            f"{100 * row['dominant_share']:.0f}\\% @ {row['dominant_value']:.2f} & "
            f"{'; '.join(row['screen_fail_reasons'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table*}"])
    (RESULTS_DIR / "confidence_quality_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_quality_figure(rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    colors = []
    markers = []
    labels_done = set()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for row in rows:
        if row["screened_headline"]:
            color = "#009E73"
            label = "Headline set"
            marker = "o"
        elif row["coverage"] >= HIGH_COVERAGE:
            color = "#D55E00"
            label = "High coverage, excluded"
            marker = "X"
        else:
            color = "#999999"
            label = "Low coverage"
            marker = "o"

        plot_label = label if label not in labels_done else None
        labels_done.add(label)
        ax.scatter(
            100 * row["coverage"],
            row["confidence_std"],
            s=55 if row["screened_headline"] else 45,
            color=color,
            marker=marker,
            alpha=0.85,
            label=plot_label,
            edgecolor="black" if row["screened_headline"] else "none",
            linewidth=0.4,
        )

    for row in rows:
        if row["coverage"] >= HIGH_COVERAGE and not row["screened_headline"]:
            short_model = row["model_label"].replace("DeepSeek ", "").replace("Qwen2.5 ", "Q2.5 ")
            ax.annotate(
                short_model,
                (100 * row["coverage"], row["confidence_std"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7.5,
            )

    ax.axvline(100 * HIGH_COVERAGE, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(STD_MIN, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Benchmark coverage (%)")
    ax.set_ylabel("Confidence standard deviation")
    ax.set_title("Confidence-quality audit of saved per-item traces")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.set_xlim(10, 103)
    ax.set_ylim(-0.005, max(row["confidence_std"] for row in rows) + 0.03)
    ax.text(
        0.02,
        0.97,
        "Excluded high-coverage runs collapse to fallback confidence values\n"
        "or too few unique levels for calibration-shape estimation.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_confidence_quality_audit.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_confidence_quality_audit.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = []
    totals = {bench: load_total_items(bench) for bench in BENCHMARKS}

    for bench in BENCHMARKS:
        for path in sorted((DATA_DIR / bench).glob("results_*.jsonl")):
            records = load_results(path)
            n = len(records)
            if n < MIN_VALID:
                continue
            p = np.array([record["conf"] for record in records])
            y = np.array([record["correct"] for record in records])
            coverage = n / totals[bench]
            eps = float(1.0 - np.mean(y))
            l_hat = estimate_lipschitz(p, y)
            bins = optimal_bins(n, eps, l_hat)
            ece = empirical_ece(p, y, bins)
            rho, p_val = self_eval_spearman(p, y)
            quality = quality_metrics(p)
            row = {
                "benchmark": bench,
                "benchmark_label": benchmark_label(bench),
                "model_key": path.stem.replace("results_", ""),
                "model_label": prettify_model(path.stem),
                "N": n,
                "total_items": totals[bench],
                "coverage": float(coverage),
                "eps": eps,
                "ece": ece,
                "L_hat": l_hat,
                "B_opt": bins,
                "delta_floor": float((l_hat * max(eps, 1e-9) / n) ** (1.0 / 3.0)),
                "spearman_rho": rho,
                "spearman_p": p_val,
                **quality,
            }
            screened, reasons = screen_row(row)
            row["screened_headline"] = screened
            row["screen_fail_reasons"] = reasons
            rows.append(row)

    rows.sort(key=lambda row: (BENCHMARKS.index(row["benchmark"]), row["model_label"]))
    screened_rows = [row for row in rows if row["screened_headline"]]
    screened_with_rho = [row for row in screened_rows if row["spearman_rho"] is not None]
    screened_nonsig = [row for row in screened_with_rho if row["spearman_p"] > 0.05]
    high_cov_rows = [row for row in rows if row["coverage"] >= HIGH_COVERAGE]
    high_cov_excluded = [row for row in high_cov_rows if not row["screened_headline"]]

    summary = {
        "thresholds": {
            "min_valid": MIN_VALID,
            "high_coverage": HIGH_COVERAGE,
            "confidence_std_min": STD_MIN,
            "rounded_unique_min": UNIQUE_MIN,
            "dominant_share_max": DOMINANT_SHARE_MAX,
            "fallback_share_max": FALLBACK_SHARE_MAX,
            "fallback_values": FALLBACK_VALUES,
        },
        "counts": {
            "total_pairs_with_N_ge_100": len(rows),
            "high_coverage_pairs": len(high_cov_rows),
            "screened_headline_pairs": len(screened_rows),
            "high_coverage_excluded_pairs": len(high_cov_excluded),
        },
        "screened_benchmark_counts": {
            benchmark_label(bench): sum(1 for row in screened_rows if row["benchmark"] == bench)
            for bench in BENCHMARKS
        },
        "screened_self_eval": {
            "nonsignificant_count": len(screened_nonsig),
            "total_with_rho": len(screened_with_rho),
            "fraction_nonsignificant": (
                len(screened_nonsig) / len(screened_with_rho) if screened_with_rho else None
            ),
        },
        "screened_ranges": {
            "eps": [
                min(row["eps"] for row in screened_rows),
                max(row["eps"] for row in screened_rows),
            ],
            "delta_floor": [
                min(row["delta_floor"] for row in screened_rows),
                max(row["delta_floor"] for row in screened_rows),
            ],
            "L_hat": [
                min(row["L_hat"] for row in screened_rows),
                max(row["L_hat"] for row in screened_rows),
            ],
        },
        "rows": rows,
    }

    (ANALYSIS_DIR / "screened_roster_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_screened_table(screened_rows)
    write_quality_table(rows)
    write_quality_figure(rows)

    print("=" * 88)
    print("SCREENED FULL-ROSTER AUDIT")
    print("=" * 88)
    print(f"Usable pairs (N >= {MIN_VALID}): {len(rows)}")
    print(f"High-coverage pairs (coverage >= {HIGH_COVERAGE:.0%}): {len(high_cov_rows)}")
    print(f"Headline screened pairs: {len(screened_rows)}")
    print(f"Excluded high-coverage pairs: {len(high_cov_excluded)}")
    if screened_with_rho:
        print(
            "Screened self-eval non-significant fraction: "
            f"{len(screened_nonsig)}/{len(screened_with_rho)} = "
            f"{len(screened_nonsig) / len(screened_with_rho):.2f}"
        )
    print(f"Wrote {ANALYSIS_DIR / 'screened_roster_summary.json'}")
    print(f"Wrote {RESULTS_DIR / 'screened_roster_table.tex'}")
    print(f"Wrote {RESULTS_DIR / 'confidence_quality_table.tex'}")
    print(f"Wrote {FIGURES_DIR / 'fig_confidence_quality_audit.pdf'}")


if __name__ == "__main__":
    main()
