"""
Analyze TruthfulQA experiment results (second benchmark B2).
Computes per-model stats, self-eval validation, phase transition,
subsampling convergence, and MMLU vs TruthfulQA comparison table.

Generates:
  - figures/fig_truthfulqa_validation.pdf/.png (multi-panel)
  - results/analysis/truthfulqa_summary.json
"""

import json
import os
import numpy as np
from math import erf, sqrt
from scipy import stats as sp_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths (absolute)
# ---------------------------------------------------------------------------
BASE_DIR = r"C:/Users/wangz/verification tax"
DATA_DIR = os.path.join(BASE_DIR, "data", "truthfulqa")
FIG_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "analysis")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_FILES = {
    "Qwen3-Next-80B":       "results_qwen3-next-80b.jsonl",
    "Llama-4-Maverick":     "results_llama-4-maverick.jsonl",
    "Llama-3.1-405B":       "results_llama-3.1-405b-instruct.jsonl",
}

# MMLU reference values (from results/analysis/mmlu_summary.json)
MMLU_REF = {
    "N": 14042,
    "eps_range": (0.16, 0.27),
    "ece_range": (0.12, 0.26),
    "L_range": (1.4, 2.5),
    "floor_range": (0.025, 0.034),
    "self_eval_gap_r_range": (0.39, 0.53),
    "self_eval_note": "2/3 NS",
}


# ---------------------------------------------------------------------------
# Helper: load valid records
# ---------------------------------------------------------------------------
def load_results(path):
    """Load valid (no-error) records from a JSONL results file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Skip errors or missing fields
            if "error" in rec:
                continue
            if "max_conf" not in rec or "is_correct" not in rec:
                continue
            records.append({
                "conf": float(rec["max_conf"]),
                "correct": int(rec["is_correct"]),
            })
    return records


# ---------------------------------------------------------------------------
# ECE (frozen formula from spec)
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
            acc = np.mean(y[mask])
            conf = np.mean(p[mask])
            ece += (nb / n) * abs(acc - conf)
    return ece


# ---------------------------------------------------------------------------
# Lipschitz estimate
# ---------------------------------------------------------------------------
def estimate_lipschitz(p, y, n_bins=20, min_per_bin=10, cap=5.0):
    """Estimate L via finite-difference on empirical calibration curve.
    Uses 20 bins, min 10 per bin (smaller dataset), 75th percentile slopes, cap at 5.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    centers = []
    gaps = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.sum() >= min_per_bin:
            centers.append((edges[i] + edges[i + 1]) / 2)
            acc_b = y[mask].mean()
            gaps.append(acc_b - (edges[i] + edges[i + 1]) / 2)
    centers = np.array(centers)
    gaps = np.array(gaps)
    if len(gaps) < 2:
        return 1.0
    slopes = []
    for i in range(len(gaps) - 1):
        d = abs(centers[i + 1] - centers[i])
        if d > 0:
            slopes.append(abs(gaps[i + 1] - gaps[i]) / d)
    if not slopes:
        return 1.0
    return float(min(np.percentile(slopes, 75), cap))


# ---------------------------------------------------------------------------
# Self-eval validation: adaptive binning
# ---------------------------------------------------------------------------
def self_eval_analysis(p, y, model_name):
    """
    Bin by confidence with adaptive scheme:
      5 bins in [0, 0.5], 5 in [0.5, 0.9], 10 in [0.9, 1.0]
    Compute per-bin accuracy, calibration gap.
    Report Spearman corr for conf->accuracy and conf->gap.
    """
    edges_low = np.linspace(0.0, 0.5, 6)       # 5 bins
    edges_mid = np.linspace(0.5, 0.9, 6)       # 5 bins
    edges_high = np.linspace(0.9, 1.0, 11)      # 10 bins
    all_edges = np.concatenate([edges_low, edges_mid[1:], edges_high[1:]])
    # Remove duplicates and sort
    all_edges = np.unique(all_edges)

    bins_info = []
    for i in range(len(all_edges) - 1):
        lo, hi = all_edges[i], all_edges[i + 1]
        if i == len(all_edges) - 2:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        nb = mask.sum()
        if nb >= 3:  # need at least 3 items for meaningful stats
            mean_conf = float(np.mean(p[mask]))
            acc = float(np.mean(y[mask]))
            gap = float(mean_conf - acc)  # overconfidence = positive
            bins_info.append({
                "bin_lo": round(float(lo), 4),
                "bin_hi": round(float(hi), 4),
                "n_items": int(nb),
                "mean_conf": round(mean_conf, 6),
                "accuracy": round(acc, 6),
                "calibration_gap": round(gap, 6),
            })

    if len(bins_info) < 4:
        return {
            "n_bins_used": len(bins_info),
            "spearman_conf_vs_accuracy": {"r": None, "p": None},
            "spearman_conf_vs_gap": {"r": None, "p": None},
            "bins": bins_info,
        }

    confs = np.array([b["mean_conf"] for b in bins_info])
    accs = np.array([b["accuracy"] for b in bins_info])
    gaps = np.array([b["calibration_gap"] for b in bins_info])

    r_acc, p_acc = sp_stats.spearmanr(confs, accs)
    r_gap, p_gap = sp_stats.spearmanr(confs, gaps)

    return {
        "n_bins_used": len(bins_info),
        "spearman_conf_vs_accuracy": {"r": round(float(r_acc), 4), "p": round(float(p_acc), 6)},
        "spearman_conf_vs_gap": {"r": round(float(r_gap), 4), "p": round(float(p_gap), 6)},
        "bins": bins_info,
    }


# ---------------------------------------------------------------------------
# Phase transition: detection power vs m*eps
# ---------------------------------------------------------------------------
def phase_transition(p, y, eps, L_hat, m_values, n_reps=300):
    """
    At each m, subsample and compute detection power:
      fraction where |ECE_estimate| > 2 * theoretical_stderr
    theoretical_stderr ~ sqrt(eps/m) (Le Cam bound scale)
    """
    N = len(p)
    results = {}
    for m in m_values:
        if m > N:
            continue
        detections = 0
        ece_estimates = []
        for _ in range(n_reps):
            idx = np.random.choice(N, size=m, replace=False)
            sub_p = p[idx]
            sub_y = y[idx]
            B_star = max(2, int((L_hat ** 2 * m / max(eps, 1e-3)) ** (1 / 3)))
            B_star = min(B_star, m // 3)  # don't let bins exceed data
            est = empirical_ece(sub_p, sub_y, B=B_star)
            ece_estimates.append(est)
            theoretical_stderr = np.sqrt(eps / m)
            if abs(est) > 2 * theoretical_stderr:
                detections += 1
        power = detections / n_reps
        results[m] = {
            "power": round(float(power), 4),
            "m_eps": round(float(m * eps), 4),
            "mean_ece": round(float(np.mean(ece_estimates)), 6),
            "std_ece": round(float(np.std(ece_estimates)), 6),
        }
    return results


# ---------------------------------------------------------------------------
# Subsampling convergence
# ---------------------------------------------------------------------------
def subsampling_convergence(p, y, ece_true, L_hat, eps, m_values, n_reps=200):
    """
    At each m: subsample, compute mean |ECE_est - ECE_true|.
    Compare to verification floor (L*eps/m)^{1/3}.
    """
    N = len(p)
    results = {}
    for m in m_values:
        if m > N:
            continue
        abs_errors = []
        for _ in range(n_reps):
            idx = np.random.choice(N, size=m, replace=False)
            sub_p = p[idx]
            sub_y = y[idx]
            B_star = max(2, int((L_hat ** 2 * m / max(eps, 1e-3)) ** (1 / 3)))
            B_star = min(B_star, m // 3)
            est = empirical_ece(sub_p, sub_y, B=B_star)
            abs_errors.append(abs(est - ece_true))
        floor = (L_hat * eps / m) ** (1 / 3)
        results[m] = {
            "mean_abs_err": round(float(np.mean(abs_errors)), 6),
            "std_abs_err": round(float(np.std(abs_errors)), 6),
            "floor": round(float(floor), 6),
        }
    return results


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 80)
    print("TruthfulQA Experiment Analysis (B2)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Basic stats per model
    # ------------------------------------------------------------------
    print("\n--- 1. Basic Stats per Model ---\n")
    model_data = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  {name}: FILE MISSING at {path}")
            continue
        records = load_results(path)
        if len(records) == 0:
            print(f"  {name}: NO VALID RECORDS")
            continue

        p = np.array([r["conf"] for r in records])
        y = np.array([r["correct"] for r in records])
        N = len(records)
        eps = float(1 - y.mean())
        mean_conf = float(p.mean())
        ece_true = float(empirical_ece(p, y, B=30))  # B=30 for smaller benchmark
        L_hat = estimate_lipschitz(p, y, n_bins=20, min_per_bin=10, cap=5.0)
        floor_full = (L_hat * eps / N) ** (1 / 3)
        is_partial = (N < 500)

        model_data[name] = {
            "p": p, "y": y,
            "N": N, "eps": eps, "mean_conf": mean_conf,
            "ece_true": ece_true, "L_hat": L_hat,
            "floor_full": floor_full, "is_partial": is_partial,
        }
        flag = " [PARTIAL]" if is_partial else ""
        print(f"  {name}{flag}:")
        print(f"    N = {N}, eps = {eps:.4f}, mean_conf = {mean_conf:.4f}")
        print(f"    ECE(B=30) = {ece_true:.4f}, L_hat = {L_hat:.3f}")
        print(f"    delta_floor = {floor_full:.4f}")

    if not model_data:
        print("No model data available. Aborting.")
        return

    # ------------------------------------------------------------------
    # 2. Self-eval validation
    # ------------------------------------------------------------------
    print("\n--- 2. Self-Eval Validation (TB on B2) ---\n")
    self_eval_results = {}
    for name, d in model_data.items():
        se = self_eval_analysis(d["p"], d["y"], name)
        self_eval_results[name] = se
        flag = " [PARTIAL]" if d["is_partial"] else ""
        r_acc = se["spearman_conf_vs_accuracy"]["r"]
        p_acc = se["spearman_conf_vs_accuracy"]["p"]
        r_gap = se["spearman_conf_vs_gap"]["r"]
        p_gap = se["spearman_conf_vs_gap"]["p"]
        sig_acc = "***" if p_acc is not None and p_acc < 0.001 else ("**" if p_acc is not None and p_acc < 0.01 else ("*" if p_acc is not None and p_acc < 0.05 else "NS"))
        sig_gap = "***" if p_gap is not None and p_gap < 0.001 else ("**" if p_gap is not None and p_gap < 0.01 else ("*" if p_gap is not None and p_gap < 0.05 else "NS"))
        print(f"  {name}{flag} ({se['n_bins_used']} bins):")
        r_acc_str = f"{r_acc:.3f}" if r_acc is not None else "N/A"
        p_acc_str = f"{p_acc:.4f}" if p_acc is not None else "N/A"
        r_gap_str = f"{r_gap:.3f}" if r_gap is not None else "N/A"
        p_gap_str = f"{p_gap:.4f}" if p_gap is not None else "N/A"
        print(f"    conf -> accuracy: r = {r_acc_str}, p = {p_acc_str} ({sig_acc})")
        print(f"    conf -> gap:      r = {r_gap_str}, p = {p_gap_str} ({sig_gap})")

    # ------------------------------------------------------------------
    # 3. Phase transition
    # ------------------------------------------------------------------
    print("\n--- 3. Phase Transition ---\n")
    phase_m_values = [20, 50, 100, 200, 400, 600]
    phase_results = {}
    for name, d in model_data.items():
        pt = phase_transition(d["p"], d["y"], d["eps"], d["L_hat"],
                              phase_m_values, n_reps=300)
        phase_results[name] = pt
        flag = " [PARTIAL]" if d["is_partial"] else ""
        print(f"  {name}{flag}:")
        for m in sorted(pt.keys()):
            print(f"    m={m:4d} (m*eps={pt[m]['m_eps']:.1f}): "
                  f"power={pt[m]['power']:.3f}")

    # ------------------------------------------------------------------
    # 4. Subsampling convergence
    # ------------------------------------------------------------------
    print("\n--- 4. Subsampling Convergence ---\n")
    sub_m_values = [50, 100, 200, 400, 600, 800]
    sub_results = {}
    for name, d in model_data.items():
        sr = subsampling_convergence(d["p"], d["y"], d["ece_true"],
                                     d["L_hat"], d["eps"],
                                     sub_m_values, n_reps=200)
        sub_results[name] = sr
        flag = " [PARTIAL]" if d["is_partial"] else ""
        print(f"  {name}{flag}:")
        for m in sorted(sr.keys()):
            print(f"    m={m:4d}: |err|={sr[m]['mean_abs_err']:.4f} +/- "
                  f"{sr[m]['std_abs_err']:.4f}, floor={sr[m]['floor']:.4f}")

    # ------------------------------------------------------------------
    # 5. Summary comparison table: MMLU vs TruthfulQA
    # ------------------------------------------------------------------
    print("\n--- 5. MMLU vs TruthfulQA Comparison Table ---\n")

    # Compute TruthfulQA ranges (excluding partial models for ranges)
    full_models = {k: v for k, v in model_data.items() if not v["is_partial"]}
    if full_models:
        tqa_eps = [d["eps"] for d in full_models.values()]
        tqa_ece = [d["ece_true"] for d in full_models.values()]
        tqa_L = [d["L_hat"] for d in full_models.values()]
        tqa_floor = [d["floor_full"] for d in full_models.values()]
        tqa_gap_r = [self_eval_results[k]["spearman_conf_vs_gap"]["r"]
                     for k in full_models if self_eval_results[k]["spearman_conf_vs_gap"]["r"] is not None]
    else:
        # fall back to all models
        tqa_eps = [d["eps"] for d in model_data.values()]
        tqa_ece = [d["ece_true"] for d in model_data.values()]
        tqa_L = [d["L_hat"] for d in model_data.values()]
        tqa_floor = [d["floor_full"] for d in model_data.values()]
        tqa_gap_r = [self_eval_results[k]["spearman_conf_vs_gap"]["r"]
                     for k in model_data if self_eval_results[k]["spearman_conf_vs_gap"]["r"] is not None]

    # Count NS for self-eval gap
    gap_ns_count = 0
    gap_total = 0
    for k in (full_models if full_models else model_data):
        se = self_eval_results[k]
        if se["spearman_conf_vs_gap"]["p"] is not None:
            gap_total += 1
            if se["spearman_conf_vs_gap"]["p"] >= 0.05:
                gap_ns_count += 1

    def fmt_range(vals, decimals=2):
        if not vals:
            return "N/A"
        lo, hi = min(vals), max(vals)
        fmt = f"{{:.{decimals}f}}"
        if abs(lo - hi) < 10 ** (-decimals):
            return fmt.format(lo)
        return f"{fmt.format(lo)}-{fmt.format(hi)}"

    N_tqa = max(d["N"] for d in full_models.values()) if full_models else max(d["N"] for d in model_data.values())

    gap_r_str = fmt_range(tqa_gap_r)
    gap_ns_str = f"{gap_ns_count}/{gap_total} NS" if gap_total > 0 else "N/A"

    rows = [
        ("N", f"{MMLU_REF['N']:,}", f"{N_tqa}"),
        ("eps range", f"{MMLU_REF['eps_range'][0]:.2f}-{MMLU_REF['eps_range'][1]:.2f}",
         fmt_range(tqa_eps)),
        ("ECE range", f"{MMLU_REF['ece_range'][0]:.2f}-{MMLU_REF['ece_range'][1]:.2f}",
         fmt_range(tqa_ece)),
        ("L_hat range", f"{MMLU_REF['L_range'][0]:.1f}-{MMLU_REF['L_range'][1]:.1f}",
         fmt_range(tqa_L, 1)),
        ("delta_floor range", f"{MMLU_REF['floor_range'][0]:.3f}-{MMLU_REF['floor_range'][1]:.3f}",
         fmt_range(tqa_floor, 3)),
        ("Self-eval (conf->gap r)", f"{MMLU_REF['self_eval_gap_r_range'][0]:.2f}-{MMLU_REF['self_eval_gap_r_range'][1]:.2f} ({MMLU_REF['self_eval_note']})",
         f"{gap_r_str} ({gap_ns_str})"),
    ]

    header = f"{'Metric':<30s} {'MMLU':<30s} {'TruthfulQA':<30s}"
    print(header)
    print("-" * len(header))
    for metric, mmlu_val, tqa_val in rows:
        print(f"{metric:<30s} {mmlu_val:<30s} {tqa_val:<30s}")

    # Also print LaTeX version
    print("\n  LaTeX version:")
    print(r"  \begin{tabular}{lcc}")
    print(r"  \toprule")
    print(r"  Metric & MMLU & TruthfulQA \\")
    print(r"  \midrule")
    for metric, mmlu_val, tqa_val in rows:
        print(f"  {metric} & {mmlu_val} & {tqa_val} \\\\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    print("\n--- Generating figures ---\n")

    fig = plt.figure(figsize=(17, 5.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.32)

    colors = {
        "Qwen3-Next-80B": "#1f77b4",
        "Llama-4-Maverick": "#ff7f0e",
        "Llama-3.1-405B": "#2ca02c",
    }
    markers = {
        "Qwen3-Next-80B": "o",
        "Llama-4-Maverick": "s",
        "Llama-3.1-405B": "^",
    }

    # ---- Panel 1: Self-eval (conf vs accuracy vs gap) ----
    ax1 = fig.add_subplot(gs[0])
    for name in MODEL_FILES:
        if name not in self_eval_results:
            continue
        se = self_eval_results[name]
        d = model_data[name]
        bins = se["bins"]
        if not bins:
            continue
        confs = [b["mean_conf"] for b in bins]
        accs = [b["accuracy"] for b in bins]
        gaps = [b["calibration_gap"] for b in bins]
        label_suffix = " [partial]" if d["is_partial"] else ""

        ax1.plot(confs, accs, marker=markers[name], color=colors[name],
                 linewidth=1.5, markersize=5, label=f"{name} acc{label_suffix}")
        ax1.plot(confs, gaps, marker=markers[name], color=colors[name],
                 linewidth=1.5, markersize=5, linestyle="--", alpha=0.6,
                 label=f"{name} gap")

    ax1.plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.4, label="Perfect cal.")
    ax1.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax1.set_xlabel("Mean confidence", fontsize=11)
    ax1.set_ylabel("Accuracy / Cal. gap", fontsize=11)
    ax1.set_title("(a) Self-Eval: Confidence vs Accuracy & Gap", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=6.5, loc="upper left", ncol=1)
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(-0.3, 1.05)
    ax1.grid(True, alpha=0.2)

    # ---- Panel 2: Subsampling convergence with floor ----
    ax2 = fig.add_subplot(gs[1])
    for name in MODEL_FILES:
        if name not in sub_results:
            continue
        d = model_data[name]
        sr = sub_results[name]
        ms = sorted(sr.keys())
        errs = [sr[m]["mean_abs_err"] for m in ms]
        err_stds = [sr[m]["std_abs_err"] for m in ms]
        floors = [sr[m]["floor"] for m in ms]
        label_suffix = " [partial]" if d["is_partial"] else ""

        ax2.errorbar(ms, errs, yerr=err_stds, fmt=f"{markers[name]}-",
                     color=colors[name], markersize=5, linewidth=1.5,
                     capsize=3, label=f"{name}{label_suffix}")
        ax2.plot(ms, floors, "--", color=colors[name], alpha=0.5, linewidth=1.2,
                 label=f"{name} floor")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Sample size $m$", fontsize=11)
    ax2.set_ylabel(r"$|\widehat{\mathrm{ECE}} - \mathrm{ECE}|$", fontsize=11)
    ax2.set_title("(b) Subsampling Convergence", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=6.5, loc="upper right")
    ax2.grid(True, alpha=0.2, which="both")

    # ---- Panel 3: Phase transition ----
    ax3 = fig.add_subplot(gs[2])
    for name in MODEL_FILES:
        if name not in phase_results:
            continue
        d = model_data[name]
        pt = phase_results[name]
        ms = sorted(pt.keys())
        m_eps = [pt[m]["m_eps"] for m in ms]
        powers = [pt[m]["power"] for m in ms]
        label_suffix = " [partial]" if d["is_partial"] else ""

        ax3.plot(m_eps, powers, f"{markers[name]}-", color=colors[name],
                 markersize=6, linewidth=1.5, label=f"{name}{label_suffix}")

    ax3.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5,
                label=r"$m \cdot \varepsilon = 1$ (theory)")
    ax3.set_xscale("log")
    ax3.set_xlabel(r"$m \cdot \varepsilon$", fontsize=11)
    ax3.set_ylabel("Detection power", fontsize=11)
    ax3.set_title("(c) Phase Transition", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=7, loc="lower right")
    ax3.grid(True, alpha=0.2)
    ax3.set_ylim(-0.05, 1.05)

    plt.suptitle("TruthfulQA Validation (B2): Self-Eval, Convergence, Phase Transition",
                 fontsize=13, fontweight="bold", y=1.02)

    fig_path_pdf = os.path.join(FIG_DIR, "fig_truthfulqa_validation.pdf")
    fig_path_png = os.path.join(FIG_DIR, "fig_truthfulqa_validation.png")
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig_path_pdf}")
    print(f"  Saved {fig_path_png}")

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    summary = {}
    for name, d in model_data.items():
        summary[name] = {
            "N": d["N"],
            "eps": round(d["eps"], 6),
            "mean_conf": round(d["mean_conf"], 6),
            "ece_true_B30": round(d["ece_true"], 6),
            "L_hat": round(d["L_hat"], 4),
            "floor_full": round(d["floor_full"], 6),
            "is_partial": d["is_partial"],
        }
    summary["self_eval"] = {}
    for name, se in self_eval_results.items():
        summary["self_eval"][name] = {
            "n_bins_used": se["n_bins_used"],
            "spearman_conf_vs_accuracy": se["spearman_conf_vs_accuracy"],
            "spearman_conf_vs_gap": se["spearman_conf_vs_gap"],
            "bins": se["bins"],
        }
    summary["phase_transition"] = {}
    for name, pt in phase_results.items():
        summary["phase_transition"][name] = {str(k): v for k, v in pt.items()}
    summary["subsampling"] = {}
    for name, sr in sub_results.items():
        summary["subsampling"][name] = {str(k): v for k, v in sr.items()}
    summary["comparison_table"] = {
        "MMLU": {
            "N": MMLU_REF["N"],
            "eps_range": list(MMLU_REF["eps_range"]),
            "ece_range": list(MMLU_REF["ece_range"]),
            "L_range": list(MMLU_REF["L_range"]),
            "floor_range": list(MMLU_REF["floor_range"]),
            "self_eval_gap_r_range": list(MMLU_REF["self_eval_gap_r_range"]),
        },
        "TruthfulQA": {
            "N": N_tqa,
            "eps_range": [round(min(tqa_eps), 4), round(max(tqa_eps), 4)] if tqa_eps else None,
            "ece_range": [round(min(tqa_ece), 4), round(max(tqa_ece), 4)] if tqa_ece else None,
            "L_range": [round(min(tqa_L), 4), round(max(tqa_L), 4)] if tqa_L else None,
            "floor_range": [round(min(tqa_floor), 6), round(max(tqa_floor), 6)] if tqa_floor else None,
            "self_eval_gap_r_range": [round(min(tqa_gap_r), 4), round(max(tqa_gap_r), 4)] if tqa_gap_r else None,
            "gap_ns_fraction": f"{gap_ns_count}/{gap_total}",
        },
    }

    json_path = os.path.join(RESULTS_DIR, "truthfulqa_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
