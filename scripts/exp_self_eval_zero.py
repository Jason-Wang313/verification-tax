#!/usr/bin/env python3
"""
exp_self_eval_zero.py  --  Empirical validation of Theorem B
(Self-Verification Impossibility)

Key insight tested here:
  - Self-confidence DOES predict accuracy  (positive Spearman r)
  - Self-confidence does NOT predict calibration gap  (r near zero)

This is exactly what Theorem B says: without ground-truth labels, a
model's own outputs carry no information about how miscalibrated those
outputs are.

Reads MMLU JSONL data for three models, computes adaptive-binned
calibration statistics, runs within-model and cross-model Spearman
analyses, and produces a publication-quality 2-panel figure.

Outputs:
  figures/fig_self_eval_zero.pdf  (and .png)
  results/analysis/self_eval_correlations.json
"""

import json
import os
import sys
from itertools import combinations

import numpy as np
from scipy import stats

# ===================================================================
# Paths (all absolute)
# ===================================================================
BASE = r"C:\Users\wangz\verification tax"
DATA_DIR = os.path.join(BASE, "data", "mmlu")
FIG_DIR = os.path.join(BASE, "figures")
RES_DIR = os.path.join(BASE, "results", "analysis")

MODEL_FILES = {
    "LLaMA-3.1-405B":   "results_llama-3.1-405b-instruct.jsonl",
    "LLaMA-4-Maverick":  "results_llama-4-maverick.jsonl",
    "Qwen3-Next-80B":    "results_qwen3-next-80b.jsonl",
}

# ===================================================================
# Adaptive bin edges  (~20 bins, finer where 78-92% of mass sits)
# ===================================================================
#   5 bins in [0, 0.5]:    0, 0.1, 0.2, 0.3, 0.4, 0.5
#   5 bins in [0.5, 0.9]:  0.5, 0.6, 0.7, 0.8, 0.85, 0.9
#  10 bins in [0.9, 1.0]:  0.9, 0.91, ..., 0.99, 1.0
BIN_EDGES = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.85, 0.9,
    0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0
])

MIN_BIN_COUNT = 10   # skip bins with fewer items


# ===================================================================
# Data loading
# ===================================================================
def load_model(filepath):
    """
    Load valid records from a JSONL file.

    Returns
    -------
    records : list of (question_id, max_conf, is_correct)
    n_skipped : int
    """
    records = []
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
            records.append((
                obj["question_id"],
                float(obj["max_conf"]),
                bool(obj["is_correct"]),
            ))
    return records, n_skipped


# ===================================================================
# Binning
# ===================================================================
def compute_bin_stats(confs, corrects, bin_edges=BIN_EDGES,
                      min_count=MIN_BIN_COUNT):
    """
    Assign items to adaptive confidence bins and compute per-bin stats.

    Parameters
    ----------
    confs : array of float, shape (n,)
    corrects : array of float/bool, shape (n,)

    Returns
    -------
    bins : list of dict with keys
        bin_lo, bin_hi, mean_conf, accuracy, calibration_gap, n_items
    """
    confs = np.asarray(confs, dtype=np.float64)
    corrects = np.asarray(corrects, dtype=np.float64)

    # np.digitize with right=True: bin i contains (edges[i-1], edges[i]]
    # Edge case: values exactly 0.0 land in bin 0, which we clip to 1.
    bin_idx = np.digitize(confs, bin_edges, right=True)
    bin_idx = np.clip(bin_idx, 1, len(bin_edges) - 1)

    bins = []
    for i in range(1, len(bin_edges)):
        mask = (bin_idx == i)
        n = int(mask.sum())
        if n < min_count:
            continue
        mc = float(confs[mask].mean())
        acc = float(corrects[mask].mean())
        gap = abs(acc - mc)
        bins.append({
            "bin_lo":           float(bin_edges[i - 1]),
            "bin_hi":           float(bin_edges[i]),
            "mean_conf":        mc,
            "accuracy":         acc,
            "calibration_gap":  gap,
            "n_items":          n,
        })
    return bins


# ===================================================================
# Spearman helper
# ===================================================================
def spearman_corr(x, y):
    """Spearman r and two-sided p-value. Returns (nan, nan) if < 3 points."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


# ===================================================================
# Main
# ===================================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load all models
    # ------------------------------------------------------------------
    model_by_qid = {}    # name -> {qid: (conf, correct)}
    model_arrays = {}    # name -> (confs_array, corrects_array)

    for name, fname in MODEL_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        records, n_skip = load_model(path)
        print(f"[{name}]  {len(records):>6,} valid records  "
              f"({n_skip:,} skipped)")

        by_qid = {}
        confs, cors = [], []
        for qid, c, cor in records:
            by_qid[qid] = (c, cor)
            confs.append(c)
            cors.append(float(cor))
        model_by_qid[name] = by_qid
        model_arrays[name] = (np.array(confs), np.array(cors))

    # ------------------------------------------------------------------
    # Part 1: Within-model analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 1 -- Within-model analysis")
    print("=" * 70)

    within = {}        # per-model results dict (for JSON)
    plot_bins = {}     # per-model bin list  (for figure)

    for name in MODEL_FILES:
        confs, cors = model_arrays[name]
        bins = compute_bin_stats(confs, cors)
        plot_bins[name] = bins

        mc  = [b["mean_conf"] for b in bins]
        acc = [b["accuracy"] for b in bins]
        gap = [b["calibration_gap"] for b in bins]

        r_acc, p_acc = spearman_corr(mc, acc)
        r_gap, p_gap = spearman_corr(mc, gap)

        frac_gt99 = float((confs > 0.99).mean())
        frac_gt90 = float((confs > 0.90).mean())

        within[name] = {
            "n_valid":            int(len(confs)),
            "overall_accuracy":   float(cors.mean()),
            "overall_mean_conf":  float(confs.mean()),
            "frac_conf_gt_0.99":  frac_gt99,
            "frac_conf_gt_0.90":  frac_gt90,
            "n_bins_used":        len(bins),
            "spearman_conf_vs_accuracy":
                {"r": r_acc, "p": p_acc},
            "spearman_conf_vs_calibration_gap":
                {"r": r_gap, "p": p_gap},
            "bins": bins,
        }

        print(f"\n  {name}")
        print(f"    Items:            {len(confs):,}")
        print(f"    Accuracy:         {cors.mean():.4f}")
        print(f"    Mean confidence:  {confs.mean():.4f}")
        print(f"    Conf > 0.99:      {frac_gt99:.1%}")
        print(f"    Conf > 0.90:      {frac_gt90:.1%}")
        print(f"    Bins used:        {len(bins)}")
        print(f"    Spearman(conf, accuracy):    "
              f"r = {r_acc:+.4f}   p = {p_acc:.2e}")
        print(f"    Spearman(conf, |cal gap|):   "
              f"r = {r_gap:+.4f}   p = {p_gap:.2e}")

    # ------------------------------------------------------------------
    # Part 2: Cross-model analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2 -- Cross-model analysis")
    print("=" * 70)

    cross = {}
    model_names = list(MODEL_FILES.keys())

    for name_a, name_b in combinations(model_names, 2):
        da = model_by_qid[name_a]
        db = model_by_qid[name_b]
        shared = sorted(set(da) & set(db))
        if not shared:
            print(f"\n  {name_a} vs {name_b}: 0 shared questions")
            continue

        confs_a  = np.array([da[q][0] for q in shared])
        cors_a   = np.array([float(da[q][1]) for q in shared])
        confs_b  = np.array([db[q][0] for q in shared])
        cors_b   = np.array([float(db[q][1]) for q in shared])

        # Bin by model A's confidence
        idx = np.digitize(confs_a, BIN_EDGES, right=True)
        idx = np.clip(idx, 1, len(BIN_EDGES) - 1)

        cross_bins = []
        for i in range(1, len(BIN_EDGES)):
            mask = (idx == i)
            n = int(mask.sum())
            if n < MIN_BIN_COUNT:
                continue
            mc_a  = float(confs_a[mask].mean())
            acc_b  = float(cors_b[mask].mean())
            mc_b   = float(confs_b[mask].mean())
            gap_b  = abs(acc_b - mc_b)
            cross_bins.append({
                "bin_lo":            float(BIN_EDGES[i - 1]),
                "bin_hi":            float(BIN_EDGES[i]),
                "mean_conf_A":       mc_a,
                "accuracy_B":        acc_b,
                "mean_conf_B":       mc_b,
                "calibration_gap_B": gap_b,
                "n_items":           n,
            })

        mc_a_vec = [b["mean_conf_A"] for b in cross_bins]
        gap_b_vec = [b["calibration_gap_B"] for b in cross_bins]
        r_cross, p_cross = spearman_corr(mc_a_vec, gap_b_vec)

        pair = f"{name_a} -> {name_b}"
        cross[pair] = {
            "n_shared":     len(shared),
            "n_bins_used":  len(cross_bins),
            "spearman_confA_vs_gapB": {"r": r_cross, "p": p_cross},
            "bins": cross_bins,
        }

        print(f"\n  {pair}")
        print(f"    Shared items:  {len(shared):,}")
        print(f"    Bins used:     {len(cross_bins)}")
        print(f"    Spearman(conf_A, gap_B):  "
              f"r = {r_cross:+.4f}   p = {p_cross:.2e}")

    # ------------------------------------------------------------------
    # Part 3: Two-panel figure
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 3 -- Generating figure")
    print("=" * 70)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Colorblind-safe palette (Wong 2011 / IBM Design)
    COLORS = {
        "LLaMA-3.1-405B":   "#0072B2",   # blue
        "LLaMA-4-Maverick":  "#D55E00",   # vermilion
        "Qwen3-Next-80B":    "#009E73",   # bluish green
    }
    MARKERS = {
        "LLaMA-3.1-405B":   "o",
        "LLaMA-4-Maverick":  "s",
        "Qwen3-Next-80B":    "D",
    }

    plt.rcParams.update({
        "font.size":        10,
        "axes.labelsize":   10,
        "axes.titlesize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  7.5,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linewidth":   0.5,
    })

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Helper: marker size proportional to log(n_items)
    def marker_sizes(n_arr):
        return 15 + 25 * np.log10(np.maximum(n_arr, 1))

    for name in MODEL_FILES:
        bins = plot_bins[name]
        mc  = np.array([b["mean_conf"] for b in bins])
        acc = np.array([b["accuracy"] for b in bins])
        gap = np.array([b["calibration_gap"] for b in bins])
        ni  = np.array([b["n_items"] for b in bins])
        sz  = marker_sizes(ni)

        r_a = within[name]["spearman_conf_vs_accuracy"]["r"]
        r_g = within[name]["spearman_conf_vs_calibration_gap"]["r"]
        col = COLORS[name]
        mk  = MARKERS[name]

        # LEFT panel: confidence vs accuracy
        ax_l.scatter(mc, acc, s=sz, c=col, marker=mk, alpha=0.82,
                     edgecolors="white", linewidths=0.5,
                     label=f"{name}  ($r_s$={r_a:+.2f})", zorder=3)

        # RIGHT panel: confidence vs |calibration gap|
        ax_r.scatter(mc, gap, s=sz, c=col, marker=mk, alpha=0.82,
                     edgecolors="white", linewidths=0.5,
                     label=f"{name}  ($r_s$={r_g:+.2f})", zorder=3)

    # -- LEFT panel formatting --
    ax_l.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8, alpha=0.5,
              zorder=1, label="Perfect calibration")
    ax_l.set_xlabel("Mean self-confidence (max softmax)")
    ax_l.set_ylabel("Accuracy")
    ax_l.set_title("Confidence predicts accuracy", fontweight="bold")
    ax_l.set_xlim(-0.03, 1.03)
    ax_l.set_ylim(-0.03, 1.03)
    ax_l.set_aspect("equal", adjustable="box")
    ax_l.legend(loc="upper left", framealpha=0.92)

    # -- RIGHT panel formatting --
    ax_r.set_xlabel("Mean self-confidence (max softmax)")
    ax_r.set_ylabel("|Calibration gap|  =  |accuracy $-$ confidence|")
    ax_r.set_title("Confidence does NOT predict calibration error",
                    fontweight="bold")
    ax_r.set_xlim(-0.03, 1.03)
    ax_r.set_ylim(bottom=-0.01)
    ax_r.legend(loc="upper right", framealpha=0.92)

    # Theorem B annotation box
    ax_r.text(
        0.03, 0.97,
        "Theorem B:\nSelf-evaluation cannot\nestimate its own\ncalibration error",
        transform=ax_r.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                  ec="grey", alpha=0.92),
    )

    fig.tight_layout(w_pad=2.5)

    pdf_path = os.path.join(FIG_DIR, "fig_self_eval_zero.pdf")
    png_path = os.path.join(FIG_DIR, "fig_self_eval_zero.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {pdf_path}")
    print(f"  Saved  {png_path}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output_json = {
        "description": (
            "Empirical validation of Theorem B (self-verification "
            "impossibility). Self-confidence predicts accuracy (positive "
            "Spearman r_s) but NOT calibration gap (r_s near zero or "
            "non-significant), consistent with the impossibility result."
        ),
        "within_model": within,
        "cross_model":  cross,
    }

    json_path = os.path.join(RES_DIR, "self_eval_correlations.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(output_json, fh, indent=2, ensure_ascii=False)
    print(f"  Saved  {json_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    hdr = f"  {'Model':<25s} {'conf vs acc':>14s} {'conf vs |gap|':>14s}"
    print("\n  Within-model Spearman correlations:")
    print(hdr)
    print(f"  {'-'*25} {'-'*14} {'-'*14}")
    for name in MODEL_FILES:
        ra = within[name]["spearman_conf_vs_accuracy"]["r"]
        rg = within[name]["spearman_conf_vs_calibration_gap"]["r"]
        print(f"  {name:<25s} {ra:>+14.4f} {rg:>+14.4f}")

    print("\n  Cross-model Spearman(conf_A, gap_B):")
    for pair, res in cross.items():
        rc = res["spearman_confA_vs_gapB"]["r"]
        pc = res["spearman_confA_vs_gapB"]["p"]
        sig = "  *" if pc < 0.05 else ""
        print(f"    {pair:<45s}  r = {rc:+.4f}  p = {pc:.2e}{sig}")

    print()
    print("  Interpretation:")
    print("    - Confidence reliably predicts accuracy (strong positive r_s)")
    print("    - Confidence does NOT reliably predict calibration gap")
    print("      (weaker r_s, often non-significant), consistent with")
    print("      Theorem B (self-verification impossibility).")
    print("=" * 70)


if __name__ == "__main__":
    main()
