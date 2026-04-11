#!/usr/bin/env python3
"""
exp_leaderboard_noise.py
========================
Demonstrates that per-subject MMLU rankings are mostly noise.

Computes:
  1. Per-subject verification floor and pairwise accuracy gaps
  2. Ranking stability via bootstrap resampling
  3. Summary statistics
  4. Two-panel figure (histogram of gaps + scatter of instability vs size)

Outputs:
  - figures/fig_leaderboard_noise.pdf and .png
  - results/leaderboard_noise_table.tex
  - results/analysis/leaderboard_noise.json
"""

import json
import math
import os
import itertools
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths (absolute) ─────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\wangz\verification tax"
DATA_DIR = os.path.join(BASE_DIR, "data", "mmlu")
FIG_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Model files ──────────────────────────────────────────────────────────────
MODEL_FILES = {
    "Llama-3.1-405B": "results_llama-3.1-405b-instruct.jsonl",
    "Llama-4-Maverick": "results_llama-4-maverick.jsonl",
    "Qwen3-Next-80B": "results_qwen3-next-80b.jsonl",
}

MODEL_NAMES = list(MODEL_FILES.keys())
MODEL_PAIRS = list(itertools.combinations(MODEL_NAMES, 2))

# Verification floor parameter
L = 1  # Lipschitz constant
N_SUBJ = 57  # number of MMLU subjects

# Bootstrap parameters
N_BOOTSTRAP = 200
SUBSAMPLE_FRAC = 0.80
SEED = 42

# ── Colorblind-safe palette (Okabe-Ito) ─────────────────────────────────────
COLOR_NOISE = "#E69F00"      # orange
COLOR_VERIFIABLE = "#0072B2"  # blue
COLOR_REF = "#D55E00"         # vermillion


def load_model_data(model_name):
    """Load JSONL file, skipping records with errors or missing fields."""
    filepath = os.path.join(DATA_DIR, MODEL_FILES[model_name])
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Skip records with error field or missing required fields
            if "error" in rec:
                continue
            if "max_conf" not in rec or "is_correct" not in rec:
                continue
            if "subject" not in rec:
                continue
            records.append(rec)
    return records


def compute_subject_stats(records):
    """Compute per-subject accuracy and item count from records."""
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    subject_items = defaultdict(list)  # store (question_id, is_correct) for bootstrap
    for rec in records:
        subj = rec["subject"]
        subject_total[subj] += 1
        if rec["is_correct"]:
            subject_correct[subj] += 1
        subject_items[subj].append(1 if rec["is_correct"] else 0)
    stats = {}
    for subj in subject_total:
        n = subject_total[subj]
        acc = subject_correct[subj] / n if n > 0 else 0.0
        stats[subj] = {
            "n_items": n,
            "accuracy": acc,
            "error_rate": 1.0 - acc,
            "items": subject_items[subj],
        }
    return stats


def main():
    rng = random.Random(SEED)
    np_rng = np.random.RandomState(SEED)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading model data...")
    model_data = {}
    model_stats = {}
    for name in MODEL_NAMES:
        records = load_model_data(name)
        model_data[name] = records
        model_stats[name] = compute_subject_stats(records)
        print(f"  {name}: {len(records)} valid records")

    # Get union of subjects across all models
    all_subjects = set()
    for name in MODEL_NAMES:
        all_subjects.update(model_stats[name].keys())
    all_subjects = sorted(all_subjects)
    print(f"Total subjects: {len(all_subjects)}")

    # ── Part 1: Per-subject verification floor ───────────────────────────────
    print("\nPart 1: Per-subject verification floor")
    subject_results = {}
    all_gaps = []           # (gap_value, subject, pair_label, is_verifiable)
    all_floors = []

    for subj in all_subjects:
        # Collect per-model stats for this subject
        model_eps = {}
        model_acc = {}
        model_n = {}
        for name in MODEL_NAMES:
            if subj in model_stats[name]:
                s = model_stats[name][subj]
                model_eps[name] = s["error_rate"]
                model_acc[name] = s["accuracy"]
                model_n[name] = s["n_items"]
            else:
                model_eps[name] = None
                model_acc[name] = None
                model_n[name] = 0

        # Skip subjects missing from any model
        if any(v is None for v in model_eps.values()):
            continue

        # Mean error rate across models
        eps_vals = [model_eps[n] for n in MODEL_NAMES]
        eps_avg = sum(eps_vals) / len(eps_vals)

        # n_items: use minimum across models (conservative)
        n_items_list = [model_n[n] for n in MODEL_NAMES]
        n_items_min = min(n_items_list)
        n_items_max = max(n_items_list)
        n_items_avg = sum(n_items_list) / len(n_items_list)

        # Verification floor: delta_floor = (L * eps_avg / n_subj)^(1/3)
        delta_floor = (L * eps_avg / N_SUBJ) ** (1.0 / 3.0)
        all_floors.append(delta_floor)

        # Pairwise gaps
        pair_results = {}
        for (a, b) in MODEL_PAIRS:
            gap = abs(model_eps[a] - model_eps[b])
            is_verifiable = gap > delta_floor
            pair_label = f"{a} vs {b}"
            pair_results[pair_label] = {
                "gap": gap,
                "delta_floor": delta_floor,
                "verifiable": is_verifiable,
            }
            all_gaps.append({
                "gap": gap,
                "subject": subj,
                "pair": pair_label,
                "verifiable": is_verifiable,
                "delta_floor": delta_floor,
            })

        # Rank models by accuracy (descending)
        ranked = sorted(MODEL_NAMES, key=lambda n: model_acc[n], reverse=True)
        ranking = {n: rank + 1 for rank, n in enumerate(ranked)}

        subject_results[subj] = {
            "n_items_min": n_items_min,
            "n_items_max": n_items_max,
            "n_items_avg": n_items_avg,
            "eps_avg": eps_avg,
            "delta_floor": delta_floor,
            "model_accuracy": {n: model_acc[n] for n in MODEL_NAMES},
            "model_error_rate": {n: model_eps[n] for n in MODEL_NAMES},
            "ranking": ranking,
            "pairwise": pair_results,
        }

    print(f"  Subjects with data from all models: {len(subject_results)}")

    # ── Part 2: Ranking stability via bootstrap ──────────────────────────────
    print("\nPart 2: Bootstrap ranking stability")
    instability_fractions = {}

    for subj in subject_results:
        # Get item-level correctness for each model
        model_items = {}
        for name in MODEL_NAMES:
            model_items[name] = np.array(model_stats[name][subj]["items"])

        n_items = len(model_items[MODEL_NAMES[0]])
        subsample_size = max(1, int(n_items * SUBSAMPLE_FRAC))

        # Original ranking
        orig_accs = {n: model_items[n].mean() for n in MODEL_NAMES}
        orig_ranked = sorted(MODEL_NAMES, key=lambda n: orig_accs[n], reverse=True)

        # Bootstrap
        n_changed = 0
        for _ in range(N_BOOTSTRAP):
            idx = np_rng.choice(n_items, size=subsample_size, replace=False)
            boot_accs = {n: model_items[n][idx].mean() for n in MODEL_NAMES}
            boot_ranked = sorted(MODEL_NAMES, key=lambda n: boot_accs[n], reverse=True)
            if boot_ranked != orig_ranked:
                n_changed += 1

        instability = n_changed / N_BOOTSTRAP
        instability_fractions[subj] = instability
        subject_results[subj]["instability"] = instability

    mean_instability = np.mean(list(instability_fractions.values()))
    print(f"  Mean instability fraction: {mean_instability:.3f}")

    # ── Part 3: Summary statistics ───────────────────────────────────────────
    print("\nPart 3: Summary statistics")

    n_completely_unranked = 0  # ALL pairwise gaps < delta_floor
    n_partially_unranked = 0   # at least one gap < delta_floor

    for subj, res in subject_results.items():
        all_noise = all(not p["verifiable"] for p in res["pairwise"].values())
        any_noise = any(not p["verifiable"] for p in res["pairwise"].values())
        if all_noise:
            n_completely_unranked += 1
        if any_noise:
            n_partially_unranked += 1

    n_verifiable_pairs = sum(1 for g in all_gaps if g["verifiable"])
    n_noise_pairs = sum(1 for g in all_gaps if not g["verifiable"])

    summary = {
        "total_subjects": len(subject_results),
        "completely_unranked": n_completely_unranked,
        "partially_unranked": n_partially_unranked,
        "fully_verifiable": len(subject_results) - n_partially_unranked,
        "mean_instability": float(mean_instability),
        "median_delta_floor": float(np.median(all_floors)),
        "total_pairwise_comparisons": len(all_gaps),
        "verifiable_pairs": n_verifiable_pairs,
        "noise_pairs": n_noise_pairs,
        "noise_fraction": n_noise_pairs / len(all_gaps) if all_gaps else 0,
    }

    print(f"  Total subjects: {summary['total_subjects']}")
    print(f"  Completely unranked (all gaps < floor): {n_completely_unranked}")
    print(f"  Partially unranked (>= 1 gap < floor): {n_partially_unranked}")
    print(f"  Fully verifiable: {summary['fully_verifiable']}")
    print(f"  Noise pairs: {n_noise_pairs}/{len(all_gaps)} "
          f"({summary['noise_fraction']:.1%})")
    print(f"  Mean instability: {mean_instability:.3f}")
    print(f"  Median delta_floor: {summary['median_delta_floor']:.4f}")

    # ── Part 4: Figure ───────────────────────────────────────────────────────
    print("\nPart 4: Creating figure...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # LEFT: Histogram of pairwise accuracy gaps
    gaps_noise = [g["gap"] for g in all_gaps if not g["verifiable"]]
    gaps_verif = [g["gap"] for g in all_gaps if g["verifiable"]]
    median_floor = np.median(all_floors)

    # Determine common bin edges
    all_gap_vals = [g["gap"] for g in all_gaps]
    bin_edges = np.linspace(0, max(all_gap_vals) * 1.05, 30)

    ax1.hist(gaps_noise, bins=bin_edges, color=COLOR_NOISE, alpha=0.85,
             label=f"Noise (n={len(gaps_noise)})", edgecolor="white", linewidth=0.5)
    ax1.hist(gaps_verif, bins=bin_edges, color=COLOR_VERIFIABLE, alpha=0.85,
             label=f"Verifiable (n={len(gaps_verif)})", edgecolor="white", linewidth=0.5)
    ax1.axvline(median_floor, color=COLOR_REF, linestyle="--", linewidth=2,
                label=f"Median $\\delta_{{floor}}$ = {median_floor:.3f}")
    ax1.set_xlabel("Pairwise accuracy gap $|\\varepsilon_A - \\varepsilon_B|$",
                   fontsize=11)
    ax1.set_ylabel("Count (subject-pair combinations)", fontsize=11)
    ax1.set_title("Per-Subject Pairwise Accuracy Gaps", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # RIGHT: Scatter of subject size vs ranking instability
    sizes = []
    instabilities = []
    for subj, res in subject_results.items():
        sizes.append(res["n_items_min"])
        instabilities.append(res["instability"])

    ax2.scatter(sizes, instabilities, s=30, alpha=0.7, color=COLOR_VERIFIABLE,
                edgecolors="white", linewidth=0.3, zorder=3)

    # Reference curve: instability ~ 1/sqrt(n) (heuristic)
    sizes_arr = np.array(sizes)
    inst_arr = np.array(instabilities)
    x_ref = np.linspace(min(sizes_arr), max(sizes_arr), 200)
    # Fit scale: instability ~ c / sqrt(n)
    # Use least-squares on valid points
    valid = inst_arr > 0
    if valid.sum() > 0:
        c_fit = np.median(inst_arr[valid] * np.sqrt(sizes_arr[valid]))
        y_ref = c_fit / np.sqrt(x_ref)
        y_ref = np.clip(y_ref, 0, 1)
        ax2.plot(x_ref, y_ref, color=COLOR_REF, linestyle="--", linewidth=2,
                 label=f"Reference: $c/\\sqrt{{n}}$", zorder=2)

    ax2.set_xlabel("Subject size (number of items)", fontsize=11)
    ax2.set_ylabel("Ranking instability fraction", fontsize=11)
    ax2.set_title("Subject Size vs. Ranking Instability", fontsize=12,
                  fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylim(-0.02, 1.02)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()

    fig_pdf = os.path.join(FIG_DIR, "fig_leaderboard_noise.pdf")
    fig_png = os.path.join(FIG_DIR, "fig_leaderboard_noise.png")
    fig.savefig(fig_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_pdf}")
    print(f"  Saved: {fig_png}")

    # ── LaTeX table: top-20 largest subjects ─────────────────────────────────
    print("\nGenerating LaTeX table...")

    # Sort subjects by n_items_min descending
    sorted_subjects = sorted(subject_results.items(),
                             key=lambda x: x[1]["n_items_min"], reverse=True)
    top20 = sorted_subjects[:20]

    # Short model labels for table
    short_names = {
        "Llama-3.1-405B": "L405B",
        "Llama-4-Maverick": "L4-Mav",
        "Qwen3-Next-80B": "Q3-80B",
    }

    tex_lines = []
    tex_lines.append(r"\begin{table}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{Per-subject MMLU rankings for the 20 largest subjects. "
                     r"$\delta_{\mathrm{floor}}$ is the verification floor; "
                     r"gaps below $\delta_{\mathrm{floor}}$ are noise (marked \ding{55}). "
                     r"Instability = fraction of 200 bootstraps where ranking changes.}")
    tex_lines.append(r"\label{tab:leaderboard-noise}")
    tex_lines.append(r"\resizebox{\textwidth}{!}{%")
    tex_lines.append(r"\begin{tabular}{lrccccccc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"Subject & $n$ & "
                     r"L405B & L4-Mav & Q3-80B & "
                     r"Max gap & $\delta_{\mathrm{floor}}$ & Verif. & Instab. \\")
    tex_lines.append(r"\midrule")

    for subj, res in top20:
        n = res["n_items_min"]
        # Model accuracies
        accs = [f"{res['model_accuracy'][m]:.2f}" for m in MODEL_NAMES]
        # Max pairwise gap
        max_gap = max(p["gap"] for p in res["pairwise"].values())
        floor = res["delta_floor"]
        # Any verifiable?
        any_verif = any(p["verifiable"] for p in res["pairwise"].values())
        verif_str = r"\ding{51}" if any_verif else r"\ding{55}"
        instab = res["instability"]

        # Format subject name
        subj_fmt = subj.replace("_", " ").title()
        if len(subj_fmt) > 28:
            subj_fmt = subj_fmt[:26] + "..."

        row = (f"{subj_fmt} & {n} & "
               f"{accs[0]} & {accs[1]} & {accs[2]} & "
               f"{max_gap:.3f} & {floor:.3f} & {verif_str} & {instab:.2f} \\\\")
        tex_lines.append(row)

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}}")
    tex_lines.append(r"\end{table}")

    tex_out = os.path.join(RESULTS_DIR, "leaderboard_noise_table.tex")
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"  Saved: {tex_out}")

    # ── JSON output ──────────────────────────────────────────────────────────
    print("\nSaving JSON results...")

    # Strip numpy types and item lists for JSON serialization
    json_subjects = {}
    for subj, res in subject_results.items():
        r = dict(res)
        r.pop("model_accuracy", None)
        r.pop("model_error_rate", None)
        # Re-add as plain dicts with float values
        r["model_accuracy"] = {
            n: float(res["model_accuracy"][n]) for n in MODEL_NAMES
        }
        r["model_error_rate"] = {
            n: float(res["model_error_rate"][n]) for n in MODEL_NAMES
        }
        # Convert numpy floats
        for key in ["n_items_avg", "eps_avg", "delta_floor", "instability"]:
            if key in r:
                r[key] = float(r[key])
        json_subjects[subj] = r

    json_output = {
        "metadata": {
            "description": "Per-subject MMLU leaderboard noise analysis",
            "models": MODEL_NAMES,
            "n_bootstrap": N_BOOTSTRAP,
            "subsample_frac": SUBSAMPLE_FRAC,
            "seed": SEED,
            "L": L,
            "n_subj": N_SUBJ,
        },
        "summary": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in summary.items()},
        "subjects": json_subjects,
        "all_pairwise_gaps": [
            {
                "subject": g["subject"],
                "pair": g["pair"],
                "gap": float(g["gap"]),
                "delta_floor": float(g["delta_floor"]),
                "verifiable": g["verifiable"],
            }
            for g in all_gaps
        ],
    }

    json_out = os.path.join(ANALYSIS_DIR, "leaderboard_noise.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    print(f"  Saved: {json_out}")

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LEADERBOARD NOISE ANALYSIS — SUMMARY")
    print("=" * 60)
    print(f"Subjects analyzed:           {summary['total_subjects']}")
    print(f"Completely unranked:         {summary['completely_unranked']} "
          f"({summary['completely_unranked']/summary['total_subjects']:.0%})")
    print(f"Partially unranked:          {summary['partially_unranked']} "
          f"({summary['partially_unranked']/summary['total_subjects']:.0%})")
    print(f"Fully verifiable:            {summary['fully_verifiable']} "
          f"({summary['fully_verifiable']/summary['total_subjects']:.0%})")
    print(f"Noise pairs:                 {summary['noise_pairs']}/{summary['total_pairwise_comparisons']} "
          f"({summary['noise_fraction']:.1%})")
    print(f"Mean ranking instability:    {summary['mean_instability']:.3f}")
    print(f"Median verification floor:   {summary['median_delta_floor']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
