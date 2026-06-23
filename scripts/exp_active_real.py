"""
Offline active vs passive verification on saved MMLU traces.

This experiment treats the benchmark as a finite unlabeled pool with hidden
correctness labels. The auditor may reveal labels adaptively from the saved
per-item outputs, which gives an offline replay version of the two-phase
explore-exploit protocol from Theorem 12.

Outputs:
  - figures/fig_active_real.pdf and .png
  - results/analysis/active_real_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "mmlu"
FIG_DIR = PROJECT_ROOT / "figures"
RES_DIR = PROJECT_ROOT / "results" / "analysis"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

MODEL_FILES = {
    "Llama 3.1 405B": "results_llama-3.1-405b-instruct.jsonl",
    "Llama 4 Maverick": "results_llama-4-maverick.jsonl",
    "Mistral Small 3.1": "results_mistral-small-3.1.jsonl",
    "Qwen3 Next 80B": "results_qwen3-next-80b.jsonl",
}

M_VALUES = [100, 200, 500, 1000, 2000, 5000, 10000]
N_REPS = 200


def load_results(path: Path) -> list[dict[str, float]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "error" in rec or "max_conf" not in rec or "is_correct" not in rec:
                continue
            rows.append({"conf": float(rec["max_conf"]), "correct": int(bool(rec["is_correct"]))})
    return rows


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


def estimate_lipschitz(p: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
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
        delta = float(abs(centers[idx + 1] - centers[idx]))
        if delta > 0:
            slopes.append(abs(float(gaps[idx + 1] - gaps[idx])) / delta)
    if not slopes:
        return 1.0
    return float(min(np.percentile(slopes, 75), 5.0))


def optimal_bins(n: int, eps: float, l_hat: float) -> int:
    if eps <= 0:
        return 15
    return max(2, min(50, int(np.floor((l_hat**2 * n / eps) ** (1 / 3)))))


def passive_estimate(p_pool: np.ndarray, y_pool: np.ndarray, m: int, l_hat: float, eps: float) -> float:
    idx = RNG.choice(len(p_pool), size=m, replace=False)
    sub_p = p_pool[idx]
    sub_y = y_pool[idx]
    bins = optimal_bins(m, eps, l_hat)
    return empirical_ece(sub_p, sub_y, bins)


def _build_quantile_bins(p_pool: np.ndarray, n_grid: int) -> np.ndarray:
    quantiles = np.linspace(0, 100, n_grid + 1)
    edges = np.percentile(p_pool, quantiles)
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1e-12
    return edges


def active_estimate(p_pool: np.ndarray, y_pool: np.ndarray, m: int, l_hat: float, eps: float) -> float:
    budget_explore = m // 2
    budget_exploit = m - budget_explore

    n_grid = min(int(max(l_hat, 1.0) * np.sqrt(m / max(eps, 1e-3))), budget_explore // 4)
    n_grid = min(max(n_grid, 4), 50)
    edges = _build_quantile_bins(p_pool, n_grid)
    n_bins = len(edges) - 1

    pool_bin_indices = []
    pool_bin_counts = np.zeros(n_bins, dtype=int)
    for idx in range(n_bins):
        if idx == n_bins - 1:
            mask = (p_pool >= edges[idx]) & (p_pool <= edges[idx + 1])
        else:
            mask = (p_pool >= edges[idx]) & (p_pool < edges[idx + 1])
        chosen = np.where(mask)[0]
        pool_bin_indices.append(chosen)
        pool_bin_counts[idx] = len(chosen)
    pool_fractions = pool_bin_counts / max(len(p_pool), 1)

    explore_target = max(1, budget_explore // n_bins)
    phase1 = {}
    phase1_delta = np.zeros(n_bins)
    phase1_n = np.zeros(n_bins, dtype=int)

    for idx in range(n_bins):
        avail = pool_bin_indices[idx]
        if len(avail) == 0:
            phase1[idx] = np.array([], dtype=int)
            continue
        n_sample = min(explore_target, len(avail))
        chosen = RNG.choice(avail, size=n_sample, replace=False)
        phase1[idx] = chosen
        phase1_n[idx] = n_sample
        phase1_delta[idx] = float(np.mean(y_pool[chosen]) - np.mean(p_pool[chosen]))

    resolved = np.zeros(n_bins, dtype=bool)
    for idx in range(n_bins):
        if phase1_n[idx] < 2:
            continue
        stderr = np.sqrt(max(eps, 1e-3) * (1 - max(eps, 1e-3)) / phase1_n[idx])
        if abs(phase1_delta[idx]) > 2 * stderr:
            resolved[idx] = True

    resolved_weights = pool_fractions * resolved.astype(float)
    alloc = np.zeros(n_bins, dtype=int)
    total_resolved = float(resolved_weights.sum())
    if total_resolved > 0:
        alloc = (resolved_weights / total_resolved * budget_exploit).astype(int)
        leftover = budget_exploit - int(alloc.sum())
        if leftover > 0:
            order = np.argsort(-resolved_weights)
            for idx in order[:leftover]:
                alloc[idx] += 1
    else:
        non_empty = pool_bin_counts > 0
        count = int(non_empty.sum())
        if count > 0:
            alloc[non_empty] = budget_exploit // count

    phase2 = {}
    for idx in range(n_bins):
        if alloc[idx] == 0 or pool_bin_counts[idx] == 0:
            phase2[idx] = np.array([], dtype=int)
            continue
        already = set(phase1[idx].tolist()) if len(phase1[idx]) else set()
        remaining = np.array([obs for obs in pool_bin_indices[idx] if obs not in already])
        if len(remaining) == 0:
            phase2[idx] = np.array([], dtype=int)
            continue
        n_sample = min(int(alloc[idx]), len(remaining))
        phase2[idx] = RNG.choice(remaining, size=n_sample, replace=False)

    total = 0.0
    for idx in range(n_bins):
        all_idx = np.concatenate([phase1[idx], phase2[idx]]).astype(int)
        if len(all_idx) == 0:
            continue
        gap = float(np.mean(y_pool[all_idx]) - np.mean(p_pool[all_idx]))
        total += pool_fractions[idx] * abs(gap)
    return float(total)


def main() -> None:
    print("=" * 80)
    print("Offline Active vs Passive Verification on Saved MMLU Traces")
    print("=" * 80)

    model_data = {}
    for model_name, filename in MODEL_FILES.items():
        path = DATA_DIR / filename
        rows = load_results(path)
        p = np.array([row["conf"] for row in rows])
        y = np.array([row["correct"] for row in rows])
        eps = float(1.0 - np.mean(y))
        l_hat = estimate_lipschitz(p, y)
        ece_true = empirical_ece(p, y, 50)
        model_data[model_name] = {
            "p": p,
            "y": y,
            "N": len(rows),
            "eps": eps,
            "L_hat": l_hat,
            "ece_true": ece_true,
            "frac_conf_gt_099": float(np.mean(p > 0.99)),
        }
        print(
            f"{model_name:20s} N={len(rows):5d} eps={eps:.3f} "
            f"L_hat={l_hat:.2f} ECE={ece_true:.3f} frac(conf>0.99)={np.mean(p > 0.99):.3f}"
        )

    results = {}
    for model_name, data in model_data.items():
        print(f"\n--- {model_name} ---")
        passive_errors = {}
        active_errors = {}
        for m in M_VALUES:
            if m > data["N"]:
                continue
            passive = []
            active = []
            for _ in range(N_REPS):
                passive.append(
                    abs(passive_estimate(data["p"], data["y"], m, data["L_hat"], data["eps"]) - data["ece_true"])
                )
                active.append(
                    abs(active_estimate(data["p"], data["y"], m, data["L_hat"], data["eps"]) - data["ece_true"])
                )
            passive_errors[m] = {
                "mean": float(np.mean(passive)),
                "std": float(np.std(passive)),
                "median": float(np.median(passive)),
            }
            active_errors[m] = {
                "mean": float(np.mean(active)),
                "std": float(np.std(active)),
                "median": float(np.median(active)),
            }
            print(
                f"m={m:5d}: passive={np.mean(passive):.4f} +/- {np.std(passive):.4f}   "
                f"active={np.mean(active):.4f} +/- {np.std(active):.4f}   "
                f"ratio={np.mean(passive)/max(np.mean(active), 1e-8):.2f}"
            )
        results[model_name] = {
            "eps": data["eps"],
            "L_hat": data["L_hat"],
            "ece_true": data["ece_true"],
            "passive": passive_errors,
            "active": active_errors,
        }

    slopes = {}
    for model_name, data in results.items():
        slopes[model_name] = {}
        for strategy in ("passive", "active"):
            ms = sorted(m for m in data[strategy] if m >= 500)
            if len(ms) < 2:
                continue
            log_m = np.log(np.array(ms, dtype=float))
            log_err = np.log(np.array([data[strategy][m]["mean"] for m in ms]))
            slope, intercept = np.polyfit(log_m, log_err, 1)
            slopes[model_name][strategy] = {"slope": float(slope), "intercept": float(intercept)}

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    plt.rcParams.update({"font.size": 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    for idx, (model_name, data) in enumerate(results.items()):
        ms_p = sorted(data["passive"])
        errs_p = [data["passive"][m]["mean"] for m in ms_p]
        ms_a = sorted(data["active"])
        errs_a = [data["active"][m]["mean"] for m in ms_a]
        ax1.plot(ms_p, errs_p, "o-", color=colors[idx], linewidth=1.6, markersize=5,
                 label=f"{model_name} passive ($\\hat{{L}}$={data['L_hat']:.2f})")
        ax1.plot(ms_a, errs_a, "s--", color=colors[idx], linewidth=1.6, markersize=5,
                 label=f"{model_name} active")

    m_ref = np.array(M_VALUES, dtype=float)
    ax1.plot(m_ref, 0.8 * m_ref ** (-1 / 3), "k:", alpha=0.5, label=r"$m^{-1/3}$")
    ax1.plot(m_ref, 0.45 * m_ref ** (-1 / 2), "k-.", alpha=0.5, label=r"$m^{-1/2}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Label budget $m$")
    ax1.set_ylabel(r"Mean $|\widehat{\mathrm{ECE}} - \mathrm{ECE}_{\mathrm{true}}|$")
    ax1.set_title("Offline adaptive audit vs passive subsampling", fontweight="bold")
    ax1.legend(fontsize=7.4, loc="upper right", framealpha=0.92)
    ax1.grid(True, alpha=0.25, which="both")

    ordered = sorted(results, key=lambda name: results[name]["L_hat"])
    x = np.arange(len(ordered))
    passive_2k = [results[name]["passive"][2000]["mean"] for name in ordered]
    active_2k = [results[name]["active"][2000]["mean"] for name in ordered]
    ax2.bar(x - 0.18, passive_2k, 0.36, color=[colors[list(results.keys()).index(name)] for name in ordered],
            alpha=0.85, edgecolor="black", linewidth=0.5, label="Passive")
    ax2.bar(x + 0.18, active_2k, 0.36, color=[colors[list(results.keys()).index(name)] for name in ordered],
            alpha=0.45, hatch="///", edgecolor="black", linewidth=0.5, label="Active")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{name}\n($\\hat{{L}}$={results[name]['L_hat']:.2f})" for name in ordered], fontsize=8.5)
    ax2.set_ylabel(r"Mean $|\widehat{\mathrm{ECE}} - \mathrm{ECE}_{\mathrm{true}}|$")
    ax2.set_title(r"Fixed budget $m=2000$ across saved MMLU traces", fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.25, axis="y")

    active_cv = float(np.std(active_2k) / np.mean(active_2k))
    passive_cv = float(np.std(passive_2k) / np.mean(passive_2k))
    ax2.text(
        0.98,
        0.95,
        f"Passive CV = {passive_cv:.2f}\nActive CV = {active_cv:.2f}\n"
        "Main signal: steeper active slopes",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )

    fig.tight_layout(w_pad=3)
    fig.savefig(FIG_DIR / "fig_active_real.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_active_real.png", bbox_inches="tight")
    plt.close(fig)

    slope_ratios = {}
    for model_name in results:
        if "passive" in slopes.get(model_name, {}) and "active" in slopes.get(model_name, {}):
            passive_slope = slopes[model_name]["passive"]["slope"]
            active_slope = slopes[model_name]["active"]["slope"]
            slope_ratios[model_name] = float(active_slope / passive_slope)

    output = {
        "description": "Offline active vs passive ECE estimation on saved MMLU traces",
        "n_reps": N_REPS,
        "m_values": M_VALUES,
        "active_cv_at_m2000": active_cv,
        "passive_cv_at_m2000": passive_cv,
        "slope_ratios_active_over_passive": slope_ratios,
        "models": {},
    }
    for model_name, data in results.items():
        output["models"][model_name] = {
            "eps": data["eps"],
            "L_hat": data["L_hat"],
            "ece_true": data["ece_true"],
            "frac_conf_gt_099": model_data[model_name]["frac_conf_gt_099"],
            "passive": {str(k): v for k, v in data["passive"].items()},
            "active": {str(k): v for k, v in data["active"].items()},
            "slopes": slopes.get(model_name, {}),
        }

    with open(RES_DIR / "active_real_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nSummary")
    print("-" * 80)
    for model_name in ordered:
        if "passive" in slopes.get(model_name, {}) and "active" in slopes.get(model_name, {}):
            print(
                f"{model_name:20s} passive slope={slopes[model_name]['passive']['slope']:+.3f}   "
                f"active slope={slopes[model_name]['active']['slope']:+.3f}   "
                f"ratio={slope_ratios[model_name]:.2f}"
            )
    print(f"Passive CV at m=2000: {passive_cv:.3f}")
    print(f"Active CV at m=2000:  {active_cv:.3f}")


if __name__ == "__main__":
    main()
