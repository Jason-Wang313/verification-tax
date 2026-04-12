#!/usr/bin/env python3
"""
exp_self_eval_permutation.py  --  Permutation test for self-eval impossibility

Strengthens the self-eval=zero claim (Theorem B) with a proper permutation test,
not just Spearman correlation.

Key test:
  H0: calibration gap is independent of confidence level
  Test statistic: |Spearman(mean_conf, calibration_gap)| across adaptive bins

Positive control: |Spearman(mean_conf, accuracy)| should be significant.

Expected result:
  - p-value for conf -> gap:      > 0.05 (non-significant)
  - p-value for conf -> accuracy:  < 0.01 (significant)
  This contrast validates Theorem B.

Data: 3 MMLU JSONL files (Llama-405B, Llama-4-Maverick, Qwen3-Next-80B).

Outputs:
  results/analysis/self_eval_permutation.json
  Printed summary table
"""

import json
import os
import sys
import numpy as np
from scipy import stats

np.random.seed(42)

# ===================================================================
# Paths (all absolute)
# ===================================================================
BASE = r"C:\Users\wangz\verification tax"
DATA_DIR = os.path.join(BASE, "data", "mmlu")
RES_DIR = os.path.join(BASE, "results", "analysis")
os.makedirs(RES_DIR, exist_ok=True)

MODEL_FILES = {
    "LLaMA-3.1-405B":   "results_llama-3.1-405b-instruct.jsonl",
    "LLaMA-4-Maverick":  "results_llama-4-maverick.jsonl",
    "Qwen3-Next-80B":    "results_qwen3-next-80b.jsonl",
}

# ===================================================================
# Adaptive bin edges (same as exp_self_eval_zero.py)
#   5 bins in [0, 0.5]:    0, 0.1, 0.2, 0.3, 0.4, 0.5
#   5 bins in [0.5, 0.9]:  0.5, 0.6, 0.7, 0.8, 0.85, 0.9
#  10 bins in [0.9, 1.0]:  0.9, 0.91, ..., 0.99, 1.0
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
            obj = json.loads(line)
            if "error" in obj:
                n_skipped += 1
                continue
            if "max_conf" not in obj or "is_correct" not in obj:
                n_skipped += 1
                continue
            confs.append(float(obj["max_conf"]))
            corrects.append(float(obj["is_correct"]))
    return np.array(confs), np.array(corrects), n_skipped


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
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("Permutation Test for Self-Verification Impossibility (Theorem B)")
    print("=" * 70)
    print(f"  Permutations:  10,000")
    print(f"  Bins:          adaptive ({len(BIN_EDGES)-1} edges, min {MIN_BIN_COUNT}/bin)")
    print()
    sys.stdout.flush()

    results = {}

    for name, fname in MODEL_FILES.items():
        print(f"--- {name} ---")
        sys.stdout.flush()

        path = os.path.join(DATA_DIR, fname)
        confs, corrects, n_skip = load_model(path)
        print(f"  Loaded {len(confs):,} valid records ({n_skip:,} skipped)")

        # Compute bin stats
        mean_confs, accuracies, cal_gaps, n_items = compute_bin_stats(confs, corrects)
        n_bins = len(mean_confs)
        print(f"  Bins used: {n_bins}")

        if n_bins < 3:
            print(f"  WARNING: Too few bins ({n_bins}). Skipping.")
            results[name] = {"error": "too few bins"}
            continue

        # --- PRIMARY TEST: conf -> calibration gap ---
        print(f"\n  PRIMARY TEST: |Spearman(conf, |cal_gap|)|")
        sys.stdout.flush()
        obs_r_gap, p_gap, null_gap = permutation_test(mean_confs, cal_gaps, n_perms=10000)
        print(f"    Observed |r| = {obs_r_gap:.4f}")
        print(f"    Permutation p-value = {p_gap:.4f}")
        if p_gap > 0.05:
            print(f"    -> NOT SIGNIFICANT (p > 0.05): consistent with Theorem B")
        else:
            print(f"    -> Significant (p <= 0.05): unexpectedly significant")

        # --- POSITIVE CONTROL: conf -> accuracy ---
        print(f"\n  POSITIVE CONTROL: |Spearman(conf, accuracy)|")
        sys.stdout.flush()
        obs_r_acc, p_acc, null_acc = permutation_test(mean_confs, accuracies, n_perms=10000)
        print(f"    Observed |r| = {obs_r_acc:.4f}")
        print(f"    Permutation p-value = {p_acc:.4f}")
        if p_acc < 0.01:
            print(f"    -> SIGNIFICANT (p < 0.01): confidence predicts accuracy")
        else:
            print(f"    -> Not significant (p >= 0.01): unexpected")

        # Also compute standard Spearman for reference
        r_gap_std, p_gap_std = stats.spearmanr(mean_confs, cal_gaps)
        r_acc_std, p_acc_std = stats.spearmanr(mean_confs, accuracies)

        results[name] = {
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
            "null_distribution_stats": {
                "gap_null_mean": float(null_gap.mean()) if len(null_gap) > 0 else None,
                "gap_null_std": float(null_gap.std()) if len(null_gap) > 0 else None,
                "gap_null_95th": float(np.percentile(null_gap, 95)) if len(null_gap) > 0 else None,
                "acc_null_mean": float(null_acc.mean()) if len(null_acc) > 0 else None,
                "acc_null_std": float(null_acc.std()) if len(null_acc) > 0 else None,
                "acc_null_95th": float(np.percentile(null_acc, 95)) if len(null_acc) > 0 else None,
            },
        }
        print()
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print()
    hdr = (f"  {'Model':<25s} "
           f"{'|r|(gap)':>10s} {'p(gap)':>10s} {'sig?':>6s} "
           f"{'|r|(acc)':>10s} {'p(acc)':>10s} {'sig?':>6s}")
    print(hdr)
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*6}")

    all_pass = True
    for name in MODEL_FILES:
        if name not in results or "error" in results[name]:
            print(f"  {name:<25s}   (skipped)")
            all_pass = False
            continue
        r = results[name]
        rg = r["primary_test_conf_vs_gap"]["observed_abs_spearman_r"]
        pg = r["primary_test_conf_vs_gap"]["permutation_p_value"]
        sg = "YES" if pg <= 0.05 else "no"
        ra = r["positive_control_conf_vs_accuracy"]["observed_abs_spearman_r"]
        pa = r["positive_control_conf_vs_accuracy"]["permutation_p_value"]
        sa = "YES" if pa <= 0.01 else "no"
        print(f"  {name:<25s} {rg:>10.4f} {pg:>10.4f} {sg:>6s} "
              f"{ra:>10.4f} {pa:>10.4f} {sa:>6s}")

        # Check expected pattern
        gap_ok = pg > 0.05
        acc_ok = pa < 0.01
        if not (gap_ok and acc_ok):
            all_pass = False

    print()
    print("  Expected pattern (Theorem B):")
    print("    conf -> |gap|:    NOT significant (p > 0.05)")
    print("    conf -> accuracy: SIGNIFICANT (p < 0.01)")
    print()
    if all_pass:
        print("  RESULT: All models match expected pattern. Theorem B validated.")
    else:
        print("  RESULT: Some models deviate from expected pattern.")
        print("          See per-model details above.")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output = {
        "description": (
            "Permutation test for self-verification impossibility (Theorem B). "
            "Tests whether confidence predicts calibration gap (should NOT be "
            "significant) vs accuracy (positive control, should be significant). "
            "10,000 permutations per test."
        ),
        "per_model": results,
    }

    json_path = os.path.join(RES_DIR, "self_eval_permutation.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved {json_path}")


if __name__ == "__main__":
    main()
