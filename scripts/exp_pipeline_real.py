#!/usr/bin/env python3
"""
exp_pipeline_real.py  --  Real 2-stage pipeline verification tax

Validates the compositional verification tax (TC) on a real pipeline
constructed from existing MMLU data (Llama-3.1-405B + Qwen3-Next-80B).

Pipeline logic:
  - If both models agree: confidence = max(conf_A, conf_B), correct = correct_A
  - If models disagree:   confidence = conf_B * (1 - conf_A), correct = correct_B
  (Model A screens, Model B decides on uncertain cases.)

Outputs:
  figures/fig_pipeline_real.pdf  (and .png)
  results/analysis/pipeline_real_results.json
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ===================================================================
# Paths (all absolute)
# ===================================================================
BASE = r"C:\Users\wangz\verification tax"
DATA_DIR = os.path.join(BASE, "data", "mmlu")
FIG_DIR = os.path.join(BASE, "figures")
RES_DIR = os.path.join(BASE, "results", "analysis")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


# ===================================================================
# Data loading
# ===================================================================
def load_model(filepath):
    """Load valid (non-error) records from a JSONL file, keyed by question_id."""
    records = {}
    n_skipped = 0
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                n_skipped += 1
                continue
            if "max_conf" not in obj or "is_correct" not in obj:
                n_skipped += 1
                continue
            qid = obj["question_id"]
            records[qid] = {
                "conf": float(obj["max_conf"]),
                "correct": int(obj["is_correct"]),
            }
    return records, n_skipped


# ===================================================================
# ECE computation (same as analyze_mmlu.py)
# ===================================================================
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
            acc = np.mean(y[mask])
            conf = np.mean(p[mask])
            ece += (nb / n) * abs(acc - conf)
    return ece


# ===================================================================
# Lipschitz estimation (same as analyze_mmlu.py)
# ===================================================================
def estimate_lipschitz(p, y, n_bins=20, min_per_bin=30):
    """Estimate L via finite-difference on a smoothed empirical calibration
    curve. 20 bins, min 30 per bin, 75th percentile of slopes, cap at 5."""
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
    gaps = accs - centers  # Delta(p)
    if len(gaps) < 2:
        return 1.0
    slopes = []
    for i in range(len(gaps) - 1):
        d = abs(centers[i + 1] - centers[i])
        if d > 0:
            slopes.append(abs(gaps[i + 1] - gaps[i]) / d)
    if not slopes:
        return 1.0
    return float(min(np.percentile(slopes, 75), 5.0))


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("Real 2-Stage Pipeline Verification Tax (Theorem C)")
    print("=" * 70)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 1: Load both models, match by question_id
    # ------------------------------------------------------------------
    path_a = os.path.join(DATA_DIR, "results_llama-3.1-405b-instruct.jsonl")
    path_b = os.path.join(DATA_DIR, "results_qwen3-next-80b.jsonl")

    recs_a, skip_a = load_model(path_a)
    recs_b, skip_b = load_model(path_b)

    print(f"  Model A (Llama-405B):   {len(recs_a):,} valid  ({skip_a:,} skipped)")
    print(f"  Model B (Qwen3-80B):    {len(recs_b):,} valid  ({skip_b:,} skipped)")

    shared_qids = sorted(set(recs_a.keys()) & set(recs_b.keys()))
    print(f"  Shared question_ids:    {len(shared_qids):,}")
    sys.stdout.flush()

    if len(shared_qids) == 0:
        print("ERROR: No shared questions. Aborting.")
        return

    # Build paired arrays
    conf_A = np.array([recs_a[q]["conf"] for q in shared_qids])
    correct_A = np.array([recs_a[q]["correct"] for q in shared_qids])
    conf_B = np.array([recs_b[q]["conf"] for q in shared_qids])
    correct_B = np.array([recs_b[q]["correct"] for q in shared_qids])

    # ------------------------------------------------------------------
    # Step 2: Construct 2-stage pipeline
    # ------------------------------------------------------------------
    agree = (correct_A == correct_B)
    disagree = ~agree

    pipeline_conf = np.empty(len(shared_qids))
    pipeline_correct = np.empty(len(shared_qids), dtype=int)

    # Agreement: pipeline confidence = max, correctness follows A (same as B)
    pipeline_conf[agree] = np.maximum(conf_A[agree], conf_B[agree])
    pipeline_correct[agree] = correct_A[agree]

    # Disagreement: pipeline confidence = conf_B * (1 - conf_A), follow B
    pipeline_conf[disagree] = conf_B[disagree] * (1 - conf_A[disagree])
    pipeline_correct[disagree] = correct_B[disagree]

    n_agree = agree.sum()
    n_disagree = disagree.sum()
    pipeline_acc = pipeline_correct.mean()
    pipeline_mean_conf = pipeline_conf.mean()

    print(f"\n  Pipeline construction:")
    print(f"    Agree:     {n_agree:,} ({100*n_agree/len(shared_qids):.1f}%)")
    print(f"    Disagree:  {n_disagree:,} ({100*n_disagree/len(shared_qids):.1f}%)")
    print(f"    Pipeline accuracy:    {pipeline_acc:.4f}")
    print(f"    Pipeline mean conf:   {pipeline_mean_conf:.4f}")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 3: Compute pipeline ECE_true (B=50)
    # ------------------------------------------------------------------
    ece_true_pipeline = empirical_ece(pipeline_conf, pipeline_correct, B=50)
    print(f"    Pipeline ECE_true (B=50): {ece_true_pipeline:.5f}")

    # Also compute single-model ECE_true for Llama-405B
    ece_true_single = empirical_ece(conf_A, correct_A, B=50)
    print(f"    Single-model (Llama-405B) ECE_true (B=50): {ece_true_single:.5f}")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 4: Estimate Lipschitz constants
    # ------------------------------------------------------------------
    L_pipeline = estimate_lipschitz(pipeline_conf, pipeline_correct)
    L_single = estimate_lipschitz(conf_A, correct_A)

    eps_pipeline = float(1 - pipeline_correct.mean())
    eps_single = float(1 - correct_A.mean())

    print(f"\n  Lipschitz estimates:")
    print(f"    Pipeline:     L_hat = {L_pipeline:.3f}")
    print(f"    Single model: L_hat = {L_single:.3f}")
    print(f"  Error rates:")
    print(f"    Pipeline:     eps = {eps_pipeline:.4f}")
    print(f"    Single model: eps = {eps_single:.4f}")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 5: Subsampling experiment
    # ------------------------------------------------------------------
    m_values = [100, 200, 500, 1000, 2000, 5000, 10000]
    n_reps = 200
    N = len(shared_qids)

    print(f"\n  Subsampling experiment (N={N}, {n_reps} replicates)")
    print("=" * 70)
    sys.stdout.flush()

    # --- Pipeline ---
    pipeline_results = {}
    print("\n  Pipeline:")
    for m in m_values:
        if m > N:
            continue
        errors = []
        for _ in range(n_reps):
            idx = np.random.choice(N, size=m, replace=False)
            sub_p = pipeline_conf[idx]
            sub_y = pipeline_correct[idx]
            B_star = max(2, int((L_pipeline ** 2 * m / max(eps_pipeline, 1e-3)) ** (1 / 3)))
            est = empirical_ece(sub_p, sub_y, B=B_star)
            errors.append(abs(est - ece_true_pipeline))
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))
        floor_val = (L_pipeline * eps_pipeline / m) ** (1 / 3)
        pipeline_results[m] = {
            "mean_abs_error": mean_err,
            "std_abs_error": std_err,
            "verification_floor": float(floor_val),
            "B_star": int(max(2, int((L_pipeline ** 2 * m / max(eps_pipeline, 1e-3)) ** (1 / 3)))),
        }
        print(f"    m={m:5d}: |err|={mean_err:.5f} +/- {std_err:.5f}, "
              f"floor={floor_val:.5f}, B*={pipeline_results[m]['B_star']}")
        sys.stdout.flush()

    # --- Single model (Llama-405B) ---
    single_results = {}
    print("\n  Single model (Llama-405B):")
    for m in m_values:
        if m > N:
            continue
        errors = []
        for _ in range(n_reps):
            idx = np.random.choice(N, size=m, replace=False)
            sub_p = conf_A[idx]
            sub_y = correct_A[idx]
            B_star = max(2, int((L_single ** 2 * m / max(eps_single, 1e-3)) ** (1 / 3)))
            est = empirical_ece(sub_p, sub_y, B=B_star)
            errors.append(abs(est - ece_true_single))
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))
        floor_val = (L_single * eps_single / m) ** (1 / 3)
        single_results[m] = {
            "mean_abs_error": mean_err,
            "std_abs_error": std_err,
            "verification_floor": float(floor_val),
            "B_star": int(max(2, int((L_single ** 2 * m / max(eps_single, 1e-3)) ** (1 / 3)))),
        }
        print(f"    m={m:5d}: |err|={mean_err:.5f} +/- {std_err:.5f}, "
              f"floor={floor_val:.5f}, B*={single_results[m]['B_star']}")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 6: Generate figure (2-panel)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Generating figure...")
    sys.stdout.flush()

    plt.rcParams.update({
        "font.size":        10,
        "axes.labelsize":   11,
        "axes.titlesize":   12,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linewidth":   0.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- LEFT PANEL: log-log error curves ---
    ms_p = sorted(pipeline_results.keys())
    err_p = [pipeline_results[m]["mean_abs_error"] for m in ms_p]
    std_p = [pipeline_results[m]["std_abs_error"] for m in ms_p]
    flr_p = [pipeline_results[m]["verification_floor"] for m in ms_p]

    ms_s = sorted(single_results.keys())
    err_s = [single_results[m]["mean_abs_error"] for m in ms_s]
    std_s = [single_results[m]["std_abs_error"] for m in ms_s]
    flr_s = [single_results[m]["verification_floor"] for m in ms_s]

    # Pipeline
    ax1.errorbar(ms_p, err_p, yerr=std_p, fmt="s-", color="#D55E00",
                 markersize=6, linewidth=2, capsize=3,
                 label=f"Pipeline ($\\hat{{L}}={L_pipeline:.2f}$, "
                       f"$\\varepsilon={eps_pipeline:.3f}$)",
                 zorder=3)
    ax1.plot(ms_p, flr_p, "--", color="#D55E00", alpha=0.5, linewidth=1.2,
             label="Pipeline floor $(L\\varepsilon/m)^{1/3}$")

    # Single model
    ax1.errorbar(ms_s, err_s, yerr=std_s, fmt="o-", color="#0072B2",
                 markersize=6, linewidth=2, capsize=3,
                 label=f"Single model ($\\hat{{L}}={L_single:.2f}$, "
                       f"$\\varepsilon={eps_single:.3f}$)",
                 zorder=3)
    ax1.plot(ms_s, flr_s, "--", color="#0072B2", alpha=0.5, linewidth=1.2,
             label="Single-model floor")

    # Reference slope -1/3
    m_ref = np.array([200, 10000])
    y_ref_start = 0.06
    y_ref = y_ref_start * (m_ref / m_ref[0]) ** (-1.0 / 3.0)
    ax1.plot(m_ref, y_ref, "k:", linewidth=1.5, alpha=0.4)
    ax1.annotate("slope $= -1/3$", xy=(800, y_ref_start * 0.7),
                 fontsize=8, color="black", alpha=0.5)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Sample size $m$")
    ax1.set_ylabel(r"Mean $|\widehat{\mathrm{ECE}} - \mathrm{ECE}_{\mathrm{true}}|$")
    ax1.set_title("(a) Pipeline vs single-model estimation error", fontweight="bold")
    ax1.legend(fontsize=7.5, loc="upper right")

    # --- RIGHT PANEL: bar chart of L_hat and verification floors ---
    bar_labels = ["Single model\n(Llama-405B)", "2-stage\npipeline"]
    L_vals = [L_single, L_pipeline]
    eps_vals = [eps_single, eps_pipeline]

    # Verification floor at m=1000
    m_ref_floor = 1000
    floor_single = (L_single * eps_single / m_ref_floor) ** (1 / 3)
    floor_pipeline = (L_pipeline * eps_pipeline / m_ref_floor) ** (1 / 3)
    floor_vals = [floor_single, floor_pipeline]

    x_pos = np.arange(len(bar_labels))
    width = 0.35

    bars1 = ax2.bar(x_pos - width / 2, L_vals, width, color=["#0072B2", "#D55E00"],
                    alpha=0.85, label="$\\hat{L}$ (Lipschitz)")
    # Add value labels
    for bar, val in zip(bars1, L_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x_pos + width / 2, floor_vals, width,
                         color=["#0072B2", "#D55E00"],
                         alpha=0.45, hatch="//",
                         label=f"Verif. floor ($m$={m_ref_floor})")
    for bar, val in zip(bars2, floor_vals):
        ax2_twin.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                      f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bar_labels)
    ax2.set_ylabel("$\\hat{L}$ (Lipschitz constant)")
    ax2_twin.set_ylabel(f"Verification floor at $m={m_ref_floor}$")
    ax2.set_title("(b) System complexity vs verification floor", fontweight="bold")

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # Annotation box
    ratio_L = L_pipeline / max(L_single, 1e-6)
    ratio_floor = floor_pipeline / max(floor_single, 1e-6)
    ax2.text(
        0.95, 0.55,
        f"Pipeline / Single:\n"
        f"  $L$ ratio:      {ratio_L:.2f}x\n"
        f"  Floor ratio:  {ratio_floor:.2f}x\n\n"
        f"Theorem C predicts\nhigher $L_{{\\mathrm{{sys}}}}$ for\ncomposed systems",
        transform=ax2.transAxes, fontsize=7.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="grey", alpha=0.9),
    )

    plt.tight_layout(w_pad=2.5)

    pdf_path = os.path.join(FIG_DIR, "fig_pipeline_real.pdf")
    png_path = os.path.join(FIG_DIR, "fig_pipeline_real.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {pdf_path}")
    print(f"  Saved {png_path}")

    # ------------------------------------------------------------------
    # Step 7: Save JSON
    # ------------------------------------------------------------------
    output = {
        "description": (
            "Real 2-stage pipeline verification tax validation (Theorem C). "
            "Pipeline = Llama-405B screens + Qwen3-80B decides on disagreements. "
            "Theory predicts: pipeline has higher L_sys, hence higher verification floor."
        ),
        "data": {
            "n_shared": len(shared_qids),
            "n_agree": int(n_agree),
            "n_disagree": int(n_disagree),
        },
        "pipeline": {
            "accuracy": float(pipeline_acc),
            "mean_conf": float(pipeline_mean_conf),
            "ece_true_B50": float(ece_true_pipeline),
            "L_hat": float(L_pipeline),
            "eps": float(eps_pipeline),
        },
        "single_model": {
            "accuracy": float(correct_A.mean()),
            "mean_conf": float(conf_A.mean()),
            "ece_true_B50": float(ece_true_single),
            "L_hat": float(L_single),
            "eps": float(eps_single),
        },
        "subsampling": {
            "m_values": m_values,
            "n_reps": n_reps,
            "pipeline": {str(m): v for m, v in pipeline_results.items()},
            "single_model": {str(m): v for m, v in single_results.items()},
        },
        "comparison": {
            "L_ratio_pipeline_over_single": float(ratio_L),
            "floor_ratio_pipeline_over_single_at_m1000": float(ratio_floor),
        },
    }

    json_path = os.path.join(RES_DIR, "pipeline_real_results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"  Saved {json_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<30s} {'Single':>12s} {'Pipeline':>12s} {'Ratio':>8s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")
    print(f"  {'L_hat':<30s} {L_single:>12.3f} {L_pipeline:>12.3f} {ratio_L:>8.2f}")
    print(f"  {'eps (error rate)':<30s} {eps_single:>12.4f} {eps_pipeline:>12.4f} "
          f"{eps_pipeline/max(eps_single,1e-9):>8.2f}")
    print(f"  {'ECE_true (B=50)':<30s} {ece_true_single:>12.5f} {ece_true_pipeline:>12.5f} "
          f"{ece_true_pipeline/max(ece_true_single,1e-9):>8.2f}")
    print(f"  {'Floor at m=1000':<30s} {floor_single:>12.5f} {floor_pipeline:>12.5f} "
          f"{ratio_floor:>8.2f}")
    print()
    if L_pipeline > L_single:
        print("  RESULT: Pipeline L_hat > Single-model L_hat, consistent with TC.")
    else:
        print("  NOTE: Pipeline L_hat <= Single-model L_hat. Pipeline composition")
        print("        may not always increase Lipschitz in practice.")
    print("=" * 70)


if __name__ == "__main__":
    main()
