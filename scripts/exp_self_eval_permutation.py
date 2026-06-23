#!/usr/bin/env python3
"""
exp_self_eval_permutation.py  --  Permutation test for self-eval impossibility

Strengthens the self-eval claim (Theorem B) with a proper permutation test,
not just Spearman correlation.

Key test:
  H0: calibration gap is independent of confidence level
  Test statistic: |Spearman(mean_conf, calibration_gap)| across adaptive bins

Positive control: |Spearman(mean_conf, accuracy)| should be significant.

Auto-discovers all results_*.jsonl files across all 5 benchmark directories.
Applies the same confidence-quality screen as analyze_screened_roster.py.
Applies Benjamini-Hochberg FDR correction across all p-values.
Computes post-hoc power analysis against rho=0.5 alternative.

Outputs:
  results/analysis/self_eval_permutation.json
  Printed summary table
"""

import json
import os
import sys
import glob
import numpy as np
from scipy import stats

np.random.seed(42)

# ===================================================================
# Paths
# ===================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = PROJECT_ROOT
DATA_DIR = os.path.join(BASE, "data")
RES_DIR = os.path.join(BASE, "results", "analysis")
os.makedirs(RES_DIR, exist_ok=True)

BENCHMARKS = ["mmlu", "truthfulqa", "arc_challenge", "hellaswag", "winogrande"]

# ===================================================================
# Confidence-quality screen (matches analyze_screened_roster.py)
# ===================================================================
MIN_VALID = 100
HIGH_COVERAGE = 0.70
STD_MIN = 0.02
UNIQUE_MIN = 20
FALLBACK_SHARE_MAX = 0.95
FALLBACK_VALUES = (0.25, 0.50, 1.00)

# ===================================================================
# Adaptive bin edges (same as exp_self_eval_zero.py)
# ===================================================================
BIN_EDGES = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.85, 0.9,
    0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0
])

MIN_BIN_COUNT = 10


# ===================================================================
# Data loading
# ===================================================================
def load_model(filepath):
    """Load valid (non-error) records from a JSONL file."""
    confs = []
    corrects = []
    n_skipped = 0
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            if "error" in obj:
                n_skipped += 1
                continue
            if "max_conf" not in obj or "is_correct" not in obj:
                n_skipped += 1
                continue
            confs.append(float(obj["max_conf"]))
            corrects.append(float(obj["is_correct"]))
    return np.array(confs), np.array(corrects), n_skipped


def load_total_items(benchmark):
    """Load total items for a benchmark to compute coverage."""
    items_path = os.path.join(DATA_DIR, benchmark, "all_items.json")
    if not os.path.exists(items_path):
        return None
    with open(items_path, "r", encoding="utf-8") as fh:
        return len(json.load(fh))


def passes_screen(confs, total_items):
    """Apply confidence-quality screen matching analyze_screened_roster.py."""
    n = len(confs)
    if n < MIN_VALID:
        return False, "n<100"
    if total_items and n / total_items < HIGH_COVERAGE:
        return False, "coverage<70%"
    if np.std(confs) < STD_MIN:
        return False, "std<0.02"
    rounded = np.round(confs, 2)
    if len(np.unique(rounded)) < UNIQUE_MIN:
        return False, "unique<20"
    for fv in FALLBACK_VALUES:
        if np.mean(np.abs(confs - fv) < 0.005) > FALLBACK_SHARE_MAX:
            return False, f"fallback={fv}"
    return True, ""


# ===================================================================
# Auto-discover all screened benchmark-model pairs
# ===================================================================
def discover_pairs():
    """Find all results_*.jsonl across benchmarks, apply screen."""
    pairs = []
    for bench in BENCHMARKS:
        bench_dir = os.path.join(DATA_DIR, bench)
        if not os.path.isdir(bench_dir):
            continue
        total = load_total_items(bench)
        for path in sorted(glob.glob(os.path.join(bench_dir, "results_*.jsonl"))):
            fname = os.path.basename(path)
            model_id = fname.replace("results_", "").replace(".jsonl", "")
            confs, corrects, n_skip = load_model(path)
            if len(confs) == 0:
                continue
            ok, reason = passes_screen(confs, total)
            if ok:
                pairs.append({
                    "benchmark": bench,
                    "model_id": model_id,
                    "path": path,
                    "confs": confs,
                    "corrects": corrects,
                    "n_valid": len(confs),
                })
    return pairs


# ===================================================================
# Adaptive binning
# ===================================================================
def compute_bin_stats(confs, corrects, bin_edges=BIN_EDGES,
                      min_count=MIN_BIN_COUNT):
    """Compute per-bin statistics using adaptive binning."""
    confs = np.asarray(confs, dtype=np.float64)
    corrects = np.asarray(corrects, dtype=np.float64)

    bin_idx = np.digitize(confs, bin_edges, right=True)
    bin_idx = np.clip(bin_idx, 1, len(bin_edges) - 1)

    mean_confs = []
    accuracies = []
    cal_gaps = []
    n_items_list = []

    for i in range(1, len(bin_edges)):
        mask = (bin_idx == i)
        n = int(mask.sum())
        if n < min_count:
            continue
        mc = float(confs[mask].mean())
        acc = float(corrects[mask].mean())
        gap = abs(acc - mc)
        mean_confs.append(mc)
        accuracies.append(acc)
        cal_gaps.append(gap)
        n_items_list.append(n)

    return (np.array(mean_confs), np.array(accuracies),
            np.array(cal_gaps), np.array(n_items_list))


# ===================================================================
# Permutation test
# ===================================================================
def permutation_test(x, y, n_perms=10000):
    """
    Permutation test for |Spearman(x, y)|.

    Returns
    -------
    observed_r : float  -- observed |Spearman r|
    p_value : float     -- fraction of permuted |r| >= observed
    null_dist : array   -- permutation distribution of |r|
    """
    if len(x) < 3:
        return float("nan"), float("nan"), np.array([])

    observed_r = abs(stats.spearmanr(x, y).statistic)

    count_ge = 0
    null_dist = np.empty(n_perms)
    y_perm = y.copy()

    for i in range(n_perms):
        np.random.shuffle(y_perm)
        r_perm = abs(stats.spearmanr(x, y_perm).statistic)
        null_dist[i] = r_perm
        if r_perm >= observed_r:
            count_ge += 1

    p_value = count_ge / n_perms
    return float(observed_r), float(p_value), null_dist


# ===================================================================
# Benjamini-Hochberg FDR correction
# ===================================================================
def benjamini_hochberg(p_values, alpha=0.05):
    """Apply BH correction. Returns adjusted p-values."""
    p = np.array(p_values)
    n = len(p)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]
    adjusted = np.empty(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            adjusted[sorted_idx[i + 1]],
            sorted_p[i] * n / (i + 1)
        )
    return np.clip(adjusted, 0, 1)


# ===================================================================
# Power analysis via Monte Carlo
# ===================================================================
def power_analysis(n_bins, rho_alt=0.5, alpha=0.05, n_sims=5000):
    """
    Estimate power of the Spearman test to detect rho = rho_alt.

    Uses scipy's Spearman p-value (exact for small n, approximation for large n)
    rather than nested permutation tests, which is much faster.
    """
    if n_bins < 3:
        return float("nan")

    rejections = 0
    for _ in range(n_sims):
        x = np.arange(n_bins, dtype=float)
        noise = np.random.randn(n_bins)
        y = rho_alt * x + np.sqrt(1 - rho_alt**2) * noise
        r, p = stats.spearmanr(x, y)
        # Two-sided test, but we use |r| so halve the threshold
        if p <= alpha:
            rejections += 1

    return rejections / n_sims


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("Permutation Test for Self-Verification Impossibility (Theorem B)")
    print("  Auto-discovers all screened benchmark-model pairs")
    print("  Applies BH correction + power analysis")
    print("=" * 70)
    print(f"  Permutations:  10,000")
    print(f"  Bins:          adaptive ({len(BIN_EDGES)-1} edges, min {MIN_BIN_COUNT}/bin)")
    print()
    sys.stdout.flush()

    pairs = discover_pairs()
    print(f"Discovered {len(pairs)} screened benchmark-model pairs")
    for p in pairs:
        print(f"  {p['benchmark']}/{p['model_id']} ({p['n_valid']:,} items)")
    print()
    sys.stdout.flush()

    results = {}
    all_gap_pvalues = []
    pair_names = []

    for pair in pairs:
        name = f"{pair['benchmark']}/{pair['model_id']}"
        print(f"--- {name} ---")
        sys.stdout.flush()

        confs = pair["confs"]
        corrects = pair["corrects"]

        # Compute bin stats
        mean_confs, accuracies, cal_gaps, n_items = compute_bin_stats(confs, corrects)
        n_bins = len(mean_confs)
        print(f"  Items: {len(confs):,}, Bins used: {n_bins}")

        if n_bins < 3:
            print(f"  WARNING: Too few bins ({n_bins}). Skipping.")
            results[name] = {"error": "too few bins", "n_bins": n_bins}
            continue

        # --- PRIMARY TEST: conf -> calibration gap ---
        obs_r_gap, p_gap, null_gap = permutation_test(mean_confs, cal_gaps, n_perms=10000)
        sig_gap = "SIG" if p_gap <= 0.05 else "ns"
        print(f"  Gap test:  |r|={obs_r_gap:.4f}  p={p_gap:.4f}  [{sig_gap}]")

        # --- POSITIVE CONTROL: conf -> accuracy ---
        obs_r_acc, p_acc, null_acc = permutation_test(mean_confs, accuracies, n_perms=10000)
        sig_acc = "SIG" if p_acc <= 0.01 else "ns"
        print(f"  Acc ctrl:  |r|={obs_r_acc:.4f}  p={p_acc:.4f}  [{sig_acc}]")

        # Standard Spearman for reference
        r_gap_std, p_gap_std = stats.spearmanr(mean_confs, cal_gaps)
        r_acc_std, p_acc_std = stats.spearmanr(mean_confs, accuracies)

        all_gap_pvalues.append(p_gap)
        pair_names.append(name)

        results[name] = {
            "benchmark": pair["benchmark"],
            "model_id": pair["model_id"],
            "n_valid": int(len(confs)),
            "n_bins": int(n_bins),
            "overall_accuracy": float(corrects.mean()),
            "overall_mean_conf": float(confs.mean()),
            "primary_test_conf_vs_gap": {
                "description": "H0: calibration gap is independent of confidence",
                "observed_abs_spearman_r": obs_r_gap,
                "permutation_p_value": p_gap,
                "n_permutations": 10000,
                "significant_at_0.05": bool(p_gap <= 0.05),
                "standard_spearman_r": float(r_gap_std),
                "standard_spearman_p": float(p_gap_std),
            },
            "positive_control_conf_vs_accuracy": {
                "description": "Positive control: confidence should predict accuracy",
                "observed_abs_spearman_r": obs_r_acc,
                "permutation_p_value": p_acc,
                "n_permutations": 10000,
                "significant_at_0.01": bool(p_acc <= 0.01),
                "standard_spearman_r": float(r_acc_std),
                "standard_spearman_p": float(p_acc_std),
            },
            "bin_details": [
                {
                    "mean_conf": float(mean_confs[i]),
                    "accuracy": float(accuracies[i]),
                    "calibration_gap": float(cal_gaps[i]),
                    "n_items": int(n_items[i]),
                }
                for i in range(n_bins)
            ],
        }
        print()
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Benjamini-Hochberg FDR correction
    # ------------------------------------------------------------------
    if all_gap_pvalues:
        raw_p = np.array(all_gap_pvalues)
        adjusted_p = benjamini_hochberg(raw_p, alpha=0.05)

        for i, name in enumerate(pair_names):
            if name in results and "error" not in results[name]:
                results[name]["primary_test_conf_vs_gap"]["bh_adjusted_p"] = float(adjusted_p[i])
                results[name]["primary_test_conf_vs_gap"]["significant_bh_0.05"] = bool(adjusted_p[i] <= 0.05)

    # ------------------------------------------------------------------
    # Power analysis (for representative bin counts)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("POWER ANALYSIS (rho_alt=0.5, alpha=0.05, 5k sims)")
    print("=" * 70)
    sys.stdout.flush()

    bin_counts_seen = set()
    power_results = {}
    for name in pair_names:
        if name in results and "error" not in results[name]:
            nb = results[name]["n_bins"]
            bin_counts_seen.add(nb)

    for nb in sorted(bin_counts_seen):
        print(f"  Computing power for n_bins={nb}...", end=" ", flush=True)
        pwr = power_analysis(nb, rho_alt=0.5, alpha=0.05, n_sims=5000)
        power_results[nb] = pwr
        print(f"power = {pwr:.3f}")

    # Attach power to each result
    for name in pair_names:
        if name in results and "error" not in results[name]:
            nb = results[name]["n_bins"]
            results[name]["power_at_rho_0.5"] = power_results.get(nb, float("nan"))

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print()
    hdr = (f"  {'Pair':<40s} {'bins':>4s} "
           f"{'|r|(gap)':>8s} {'p(gap)':>8s} {'p(BH)':>8s} {'sig?':>5s} "
           f"{'|r|(acc)':>8s} {'p(acc)':>8s} {'power':>6s}")
    print(hdr)
    print(f"  {'-'*40} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*6}")

    n_tested = 0
    n_gap_nonsig_raw = 0
    n_gap_nonsig_bh = 0
    n_acc_sig = 0

    for name in pair_names:
        if name not in results or "error" in results[name]:
            print(f"  {name:<40s}   (skipped)")
            continue
        r = results[name]
        n_tested += 1
        rg = r["primary_test_conf_vs_gap"]["observed_abs_spearman_r"]
        pg = r["primary_test_conf_vs_gap"]["permutation_p_value"]
        pbh = r["primary_test_conf_vs_gap"].get("bh_adjusted_p", float("nan"))
        sg = "BH" if pbh <= 0.05 else ("raw" if pg <= 0.05 else "no")
        ra = r["positive_control_conf_vs_accuracy"]["observed_abs_spearman_r"]
        pa = r["positive_control_conf_vs_accuracy"]["permutation_p_value"]
        pwr = r.get("power_at_rho_0.5", float("nan"))
        nb = r["n_bins"]

        if pg > 0.05:
            n_gap_nonsig_raw += 1
        if pbh > 0.05:
            n_gap_nonsig_bh += 1
        if pa <= 0.01:
            n_acc_sig += 1

        print(f"  {name:<40s} {nb:>4d} "
              f"{rg:>8.4f} {pg:>8.4f} {pbh:>8.4f} {sg:>5s} "
              f"{ra:>8.4f} {pa:>8.4f} {pwr:>6.3f}")

    print()
    frac_raw = n_gap_nonsig_raw / n_tested if n_tested > 0 else 0
    frac_bh = n_gap_nonsig_bh / n_tested if n_tested > 0 else 0
    print(f"  Tested: {n_tested}")
    print(f"  Gap non-significant (raw p>0.05):  {n_gap_nonsig_raw}/{n_tested} = {frac_raw:.1%}")
    print(f"  Gap non-significant (BH p>0.05):   {n_gap_nonsig_bh}/{n_tested} = {frac_bh:.1%}")
    print(f"  Accuracy significant (p<0.01):     {n_acc_sig}/{n_tested}")
    print()
    print(f"  Power analysis: power to detect rho=0.5")
    for nb, pwr in sorted(power_results.items()):
        print(f"    n_bins={nb}: power = {pwr:.3f}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output = {
        "description": (
            "Permutation test for self-verification impossibility (Theorem B). "
            "Auto-discovers all screened benchmark-model pairs across 5 benchmarks. "
            "Tests whether confidence predicts calibration gap (should NOT be "
            "significant) vs accuracy (positive control, should be significant). "
            "10,000 permutations per test. BH FDR correction applied. "
            "Power analysis against rho=0.5 alternative."
        ),
        "n_pairs_tested": n_tested,
        "n_gap_nonsignificant_raw": n_gap_nonsig_raw,
        "n_gap_nonsignificant_bh": n_gap_nonsig_bh,
        "fraction_nonsignificant_raw": frac_raw,
        "fraction_nonsignificant_bh": frac_bh,
        "n_accuracy_significant": n_acc_sig,
        "power_by_nbins": {str(k): v for k, v in power_results.items()},
        "per_pair": results,
    }

    json_path = os.path.join(RES_DIR, "self_eval_permutation.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved {json_path}")


if __name__ == "__main__":
    main()
