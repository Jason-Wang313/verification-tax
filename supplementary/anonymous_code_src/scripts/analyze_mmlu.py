"""
Analyze MMLU experiment results: compute ε, ECE_true, L̂, and run subsampling.
Generates figures/real_model_verification.pdf and the per-model audit table.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

DATA_DIR = "data/mmlu"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_FILES = {
    "Llama-3.1-405B": "results_llama-3.1-405b-instruct.jsonl",
    "Llama-3.1-70B": "results_llama-3.1-70b-instruct.jsonl",
    "Llama-4-Maverick-17B": "results_llama-4-maverick.jsonl",
    "Qwen3-Next-80B": "results_qwen3-next-80b.jsonl",
}


def load_results(path):
    """Load valid (no-error) records from a results file."""
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "error" in rec or "max_conf" not in rec or "is_correct" not in rec:
                continue
            records.append({
                "conf": rec["max_conf"],
                "correct": int(rec["is_correct"]),
            })
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
            acc = np.mean(y[mask])
            conf = np.mean(p[mask])
            ece += (nb / n) * abs(acc - conf)
    return ece


def estimate_lipschitz(p, y, n_bins=20, min_per_bin=30):
    """Estimate L via finite-difference on a smoothed empirical calibration curve.
    Uses fewer, wider bins (n_bins=20) and a minimum bin occupancy to suppress noise.
    Returns the 75th percentile of adjacent-bin slopes for a robust estimate.
    """
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
    gaps = accs - centers  # Δ(p)
    if len(gaps) < 2:
        return 1.0
    slopes = []
    for i in range(len(gaps) - 1):
        d = abs(centers[i + 1] - centers[i])
        if d > 0:
            slopes.append(abs(gaps[i + 1] - gaps[i]) / d)
    if not slopes:
        return 1.0
    # Use the 75th percentile to get a robust estimate; cap at a reasonable max
    return float(min(np.percentile(slopes, 75), 5.0))


def main():
    print("=" * 80)
    print("MMLU Experiment Analysis")
    print("=" * 80)

    model_data = {}

    # Step 1: Compute ε, ECE_true, L̂ for each model
    for name, fname in MODEL_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  {name}: FILE MISSING")
            continue
        records = load_results(path)
        if len(records) == 0:
            print(f"  {name}: NO VALID RECORDS")
            continue
        p = np.array([r["conf"] for r in records])
        y = np.array([r["correct"] for r in records])

        eps = float(1 - y.mean())
        ece_true = float(empirical_ece(p, y, B=50))
        L_hat = estimate_lipschitz(p, y)

        # Verification floor at full N
        N = len(records)
        floor_full = (L_hat * eps / N) ** (1/3)

        model_data[name] = {
            "p": p, "y": y, "N": N,
            "eps": eps, "ece_true": ece_true, "L_hat": L_hat,
            "floor_full": floor_full,
        }
        print(f"  {name}: N={N}, eps={eps:.4f}, ECE_true={ece_true:.4f}, "
              f"L_hat={L_hat:.2f}, floor(N)={floor_full:.4f}")

    if not model_data:
        print("No model data — aborting analysis.")
        return

    # Step 2: Subsampling experiment
    print()
    print("=" * 80)
    print("Subsampling experiment")
    print("=" * 80)

    m_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    n_reps = 200

    sub_results = {}
    for name, d in model_data.items():
        N = d["N"]
        L = d["L_hat"]
        eps = d["eps"]
        ece_true = d["ece_true"]
        per_m = {}
        for m in m_values:
            if m > N:
                continue
            estimates = []
            for _ in range(n_reps):
                idx = np.random.choice(N, size=m, replace=False)
                sub_p = d["p"][idx]
                sub_y = d["y"][idx]
                B_star = max(2, int((L ** 2 * m / max(eps, 1e-3)) ** (1 / 3)))
                est = empirical_ece(sub_p, sub_y, B=B_star)
                estimates.append(est)
            mean_ece = float(np.mean(estimates))
            std_ece = float(np.std(estimates))
            floor = (L * eps / m) ** (1/3)
            per_m[m] = {
                "mean": mean_ece,
                "std": std_ece,
                "floor": floor,
                "abs_err": abs(mean_ece - ece_true),
            }
            print(f"  {name} m={m:5d}: ECE={mean_ece:.4f} ± {std_ece:.4f}, "
                  f"floor={floor:.4f}, |err|={abs(mean_ece - ece_true):.4f}")
        sub_results[name] = per_m
        print()

    # Step 3: Generate figure
    print("Generating figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = plt.cm.tab10.colors

    for i, (name, d) in enumerate(model_data.items()):
        if name not in sub_results:
            continue
        ms = sorted(sub_results[name].keys())
        means = [sub_results[name][m]["mean"] for m in ms]
        stds = [sub_results[name][m]["std"] for m in ms]
        floors = [sub_results[name][m]["floor"] for m in ms]
        ece_true = d["ece_true"]
        eps = d["eps"]

        # Left panel: ECE estimate ± std vs m, floor overlay
        ax1.errorbar(ms, means, yerr=stds, fmt="o-", color=colors[i],
                     markersize=5, linewidth=1.5, capsize=3,
                     label=f"{name} ($\\varepsilon$={eps:.2f})")
        ax1.plot(ms, floors, "--", color=colors[i], alpha=0.5, linewidth=1)
        ax1.axhline(y=ece_true, color=colors[i], linestyle=":", alpha=0.4, linewidth=1)

    ax1.set_xscale("log")
    ax1.set_xlabel(r"Sample size $m$", fontsize=12)
    ax1.set_ylabel("ECE", fontsize=12)
    ax1.set_title("Real-Model ECE Estimation vs Verification Floor", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Right panel: detection power vs m·ε
    # Detection power: fraction of replicates where |ECE_hat| > 2*floor (very simple)
    for i, (name, d) in enumerate(model_data.items()):
        if name not in sub_results:
            continue
        ms = sorted(sub_results[name].keys())
        eps = d["eps"]
        # Simple measure: power = 1 if mean_ece > 2*floor, else 0; smoother: use std
        powers = []
        for m in ms:
            r = sub_results[name][m]
            # Power proxy: P(estimate > 2*floor) under normal approx
            # = 1 - Phi((2*floor - mean) / std)
            from math import erf, sqrt
            mu = r["mean"]
            sd = max(r["std"], 1e-8)
            thresh = 2 * r["floor"]
            z = (thresh - mu) / sd
            power = 0.5 * (1 - erf(z / sqrt(2)))
            powers.append(power)
        norm_m = [m * eps for m in ms]
        ax2.plot(norm_m, powers, "o-", color=colors[i], markersize=5, linewidth=1.5,
                 label=f"{name}")

    ax2.axvline(x=1.0, color="red", linestyle="--", linewidth=2,
                label=r"$m \cdot \varepsilon = 1$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Normalized sample size $m \cdot \varepsilon$", fontsize=12)
    ax2.set_ylabel("Detection power", fontsize=12)
    ax2.set_title("Real-Model Phase Transition", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "real_model_verification.pdf"),
                dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "real_model_verification.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figures/real_model_verification.{pdf,png}")

    # Step 4: Print audit table for paper
    print()
    print("=" * 80)
    print("Audit table (LaTeX-ready):")
    print("=" * 80)
    for name, d in model_data.items():
        print(f"{name} & {d['N']:,} & {d['eps']:.3f} & {d['L_hat']:.2f} & "
              f"{d['ece_true']:.4f} & {d['floor_full']:.4f} \\\\")

    # Save analysis for reuse
    summary = {
        name: {k: v for k, v in d.items() if k not in ("p", "y")}
        for name, d in model_data.items()
    }
    summary["subsampling"] = sub_results
    with open(os.path.join("results/analysis", "mmlu_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results/analysis/mmlu_summary.json")


if __name__ == "__main__":
    os.makedirs("results/analysis", exist_ok=True)
    main()
