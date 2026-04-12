"""
Bootstrap confidence intervals for all key experimental claims in
"The Verification Tax" paper.

Computes 95% CIs (2.5th / 97.5th percentiles) via 1000 bootstrap
replicates for:
  1. ECE and error rate per model
  2. Lipschitz estimate per model
  3. Verification floor per model
  4. Self-eval Spearman correlation (Theorem B validation)
  5. Leaderboard noise fraction (unranked subjects)
  6. Active vs passive slope CIs
  7. Benchmark demolition floor CIs

Output:
  - results/analysis/bootstrap_confidence_intervals.json
  - Summary table to stdout
"""

import json
import os
import sys
import time
import numpy as np
from scipy import stats

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ── Configuration ──────────────────────────────────────────────────────────
np.random.seed(42)
N_BOOT = 1000
N_ITEMS = 14042          # expected valid items per model
B_ECE = 50               # bins for ECE computation
ALPHA_LO = 2.5           # percentile for CI lower
ALPHA_HI = 97.5          # percentile for CI upper

BASE = "C:/Users/wangz/verification tax"
DATA_DIR = os.path.join(BASE, "data", "mmlu")
OUT_DIR = os.path.join(BASE, "results", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_FILES = {
    "Llama-3.1-405B":    "results_llama-3.1-405b-instruct.jsonl",
    "Llama-4-Maverick":  "results_llama-4-maverick.jsonl",
    "Qwen3-Next-80B":    "results_qwen3-next-80b.jsonl",
}


# ── Data loading ───────────────────────────────────────────────────────────
def load_results(path):
    """Load valid records, returning arrays of conf, correct, subject."""
    confs, corrects, subjects = [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "error" in rec:
                continue
            if "max_conf" not in rec or "is_correct" not in rec:
                continue
            confs.append(float(rec["max_conf"]))
            corrects.append(int(rec["is_correct"]))
            subjects.append(rec["subject"])
    return (np.array(confs, dtype=np.float64),
            np.array(corrects, dtype=np.float64),
            np.array(subjects))


# ── Core metrics ───────────────────────────────────────────────────────────
def compute_ece(p, y, B=B_ECE):
    """Binned ECE with B equal-width bins."""
    n = len(p)
    edges = np.linspace(0, 1, B + 1)
    ece = 0.0
    for b in range(B):
        lo, hi = edges[b], edges[b + 1]
        if b < B - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        nb = mask.sum()
        if nb > 0:
            ece += (nb / n) * abs(np.mean(y[mask]) - np.mean(p[mask]))
    return ece


def compute_ece_vectorized(p_all, y_all, B=B_ECE):
    """Compute ECE for multiple bootstrap samples simultaneously.
    p_all: (n_boot, n_items), y_all: (n_boot, n_items)
    Returns: (n_boot,) array of ECE values.
    """
    n_boot, n = p_all.shape
    edges = np.linspace(0, 1, B + 1)
    ece = np.zeros(n_boot)
    for b in range(B):
        lo, hi = edges[b], edges[b + 1]
        if b < B - 1:
            mask = (p_all >= lo) & (p_all < hi)
        else:
            mask = (p_all >= lo) & (p_all <= hi)
        nb = mask.sum(axis=1).astype(np.float64)
        safe_nb = np.where(nb > 0, nb, 1.0)
        sum_y = (y_all * mask).sum(axis=1)
        sum_p = (p_all * mask).sum(axis=1)
        mean_y = sum_y / safe_nb
        mean_p = sum_p / safe_nb
        contrib = (nb / n) * np.abs(mean_y - mean_p)
        contrib = np.where(nb > 0, contrib, 0.0)
        ece += contrib
    return ece


def estimate_lipschitz(p, y, n_bins=20, min_per_bin=30):
    """Estimate L via finite-difference on smoothed empirical calibration curve.
    75th percentile of adjacent-bin slopes, capped at 5.0."""
    edges = np.linspace(0, 1, n_bins + 1)
    centers, gaps = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        if mask.sum() >= min_per_bin:
            centers.append((lo + hi) / 2)
            gaps.append(y[mask].mean() - p[mask].mean())
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


def compute_spearman_conf_vs_gap(p, y, n_bins=20, min_per_bin=20):
    """Compute Spearman correlation between bin center (confidence) and
    calibration gap across bins. Returns (r, pvalue)."""
    edges = np.linspace(0, 1, n_bins + 1)
    centers, cal_gaps = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        if mask.sum() >= min_per_bin:
            mc = np.mean(p[mask])
            acc = np.mean(y[mask])
            centers.append(mc)
            cal_gaps.append(acc - mc)
    if len(centers) < 4:
        return (np.nan, 1.0)
    r, pval = stats.spearmanr(centers, cal_gaps)
    return (r, pval)


# ── Bootstrap helpers ──────────────────────────────────────────────────────
def ci(arr):
    """Return (point_estimate=median, lo, hi) from bootstrap distribution."""
    return {
        "point": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "lo": float(np.percentile(arr, ALPHA_LO)),
        "hi": float(np.percentile(arr, ALPHA_HI)),
    }


def print_ci(name, c):
    """Pretty-print a CI dict."""
    pt = c.get("point")
    if pt is None:
        print(f"  {name:40s}: N/A")
        return
    lo = c.get("lo", float("nan"))
    hi = c.get("hi", float("nan"))
    print(f"  {name:40s}: {pt:.4f}  [{lo:.4f}, {hi:.4f}]")


# ── Main computation ───────────────────────────────────────────────────────
def main():
    t0 = time.time()
    results = {"n_bootstrap": N_BOOT, "seed": 42, "models": {}}

    # Load all model data
    all_data = {}
    for model_name, fname in MODEL_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        p, y, subj = load_results(path)
        assert len(p) == N_ITEMS, f"{model_name}: expected {N_ITEMS}, got {len(p)}"
        all_data[model_name] = {"p": p, "y": y, "subj": subj}
        print(f"Loaded {model_name}: {len(p)} items, "
              f"eps={1-y.mean():.4f}, "
              f"ECE={compute_ece(p, y):.4f}, "
              f"L_hat={estimate_lipschitz(p, y):.3f}")

    # Pre-generate bootstrap indices for item-level resampling
    print(f"\nPre-generating {N_BOOT} bootstrap index sets...")
    boot_idx = np.random.randint(0, N_ITEMS, size=(N_BOOT, N_ITEMS))

    # Storage for Lipschitz bootstrap samples (reuse across steps 2, 3)
    L_boot_cache = {}

    # ================================================================
    # 1. ECE and error rate CIs per model
    # ================================================================
    print("\n" + "=" * 70)
    print("1. ECE and Error Rate CIs")
    print("=" * 70)

    # Also store eps bootstrap samples for reuse in step 3
    eps_boot_cache = {}

    for model_name in MODEL_FILES:
        p = all_data[model_name]["p"]
        y = all_data[model_name]["y"]

        eps_point = 1.0 - y.mean()
        ece_point = compute_ece(p, y)

        p_boot = p[boot_idx]   # (N_BOOT, N_ITEMS)
        y_boot = y[boot_idx]

        ece_samples = compute_ece_vectorized(p_boot, y_boot, B=B_ECE)
        eps_samples = 1.0 - y_boot.mean(axis=1)
        eps_boot_cache[model_name] = eps_samples

        ece_ci = ci(ece_samples)
        eps_ci = ci(eps_samples)
        ece_ci["point"] = float(ece_point)
        eps_ci["point"] = float(eps_point)

        results["models"][model_name] = {
            "ECE": ece_ci,
            "epsilon": eps_ci,
        }

        print(f"\n  {model_name}:")
        print_ci("ECE (B=50)", ece_ci)
        print_ci("Error rate (epsilon)", eps_ci)

    print(f"  [Step 1 done in {time.time()-t0:.1f}s]")

    # ================================================================
    # 2. Lipschitz estimate CI per model
    # ================================================================
    t1 = time.time()
    print("\n" + "=" * 70)
    print("2. Lipschitz Estimate CIs")
    print("=" * 70)

    for model_name in MODEL_FILES:
        p = all_data[model_name]["p"]
        y = all_data[model_name]["y"]

        L_point = estimate_lipschitz(p, y)
        L_samples = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            idx = boot_idx[b]
            L_samples[b] = estimate_lipschitz(p[idx], y[idx])
            if (b + 1) % 250 == 0:
                print(f"    {model_name}: {b+1}/{N_BOOT} Lipschitz bootstraps done...")

        L_boot_cache[model_name] = L_samples

        L_ci = ci(L_samples)
        L_ci["point"] = float(L_point)
        results["models"][model_name]["L_hat"] = L_ci

        print(f"\n  {model_name}:")
        print_ci("L_hat", L_ci)

    print(f"  [Step 2 done in {time.time()-t1:.1f}s]")

    # ================================================================
    # 3. Verification floor CI per model (reuse cached L and eps)
    # ================================================================
    t1 = time.time()
    print("\n" + "=" * 70)
    print("3. Verification Floor CIs (delta_floor)")
    print("=" * 70)

    for model_name in MODEL_FILES:
        p = all_data[model_name]["p"]
        y = all_data[model_name]["y"]

        eps_point = 1.0 - y.mean()
        L_point = estimate_lipschitz(p, y)
        floor_point = (L_point * eps_point / N_ITEMS) ** (1.0 / 3.0)

        # Reuse cached bootstrap distributions
        eps_samples = eps_boot_cache[model_name]
        L_boot = L_boot_cache[model_name]

        floor_samples = (L_boot * eps_samples / N_ITEMS) ** (1.0 / 3.0)
        floor_ci = ci(floor_samples)
        floor_ci["point"] = float(floor_point)

        results["models"][model_name]["delta_floor"] = floor_ci

        print(f"\n  {model_name}:")
        print_ci("delta_floor", floor_ci)

    print(f"  [Step 3 done in {time.time()-t1:.1f}s]")

    # ================================================================
    # 4. Self-eval correlation CIs (Theorem B validation)
    # ================================================================
    t1 = time.time()
    print("\n" + "=" * 70)
    print("4. Self-Eval Spearman Correlation CIs (conf vs calibration gap)")
    print("=" * 70)

    for model_name in MODEL_FILES:
        p = all_data[model_name]["p"]
        y = all_data[model_name]["y"]

        r_point, pval_point = compute_spearman_conf_vs_gap(p, y)

        r_samples = np.zeros(N_BOOT)
        p_samples = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            idx = boot_idx[b]
            r_b, pval_b = compute_spearman_conf_vs_gap(p[idx], y[idx])
            r_samples[b] = r_b
            p_samples[b] = pval_b

        valid_mask = ~np.isnan(r_samples)
        r_valid = r_samples[valid_mask]
        p_valid = p_samples[valid_mask]

        r_ci_val = ci(r_valid) if len(r_valid) > 0 else {"point": np.nan, "mean": np.nan, "lo": np.nan, "hi": np.nan}
        r_ci_val["point"] = float(r_point) if not np.isnan(r_point) else None

        frac_nonsig = float((p_valid > 0.05).mean()) if len(p_valid) > 0 else np.nan

        results["models"][model_name]["self_eval_spearman"] = {
            "r": r_ci_val,
            "p_value_point": float(pval_point) if not np.isnan(pval_point) else None,
            "frac_p_gt_005": frac_nonsig,
            "n_valid_bootstrap": int(valid_mask.sum()),
        }

        print(f"\n  {model_name}:")
        print_ci("Spearman r (conf vs gap)", r_ci_val)
        print(f"  {'Frac bootstrap p > 0.05':40s}: {frac_nonsig:.4f}")
        print(f"  {'Point p-value':40s}: {pval_point:.6f}")

    print(f"  [Step 4 done in {time.time()-t1:.1f}s]")

    # ================================================================
    # 5. Leaderboard noise fraction CI
    # ================================================================
    t1 = time.time()
    print("\n" + "=" * 70)
    print("5. Leaderboard Noise Fraction CI (unranked subjects)")
    print("=" * 70)

    subject_list = sorted(set(all_data["Llama-3.1-405B"]["subj"]))
    n_subj = len(subject_list)
    model_names = list(MODEL_FILES.keys())
    print(f"  Number of subjects: {n_subj}")

    # Build per-subject data for each model
    subj_data = {}
    for model_name in MODEL_FILES:
        subj_data[model_name] = {}
        p = all_data[model_name]["p"]
        y = all_data[model_name]["y"]
        subj = all_data[model_name]["subj"]
        for s in subject_list:
            mask = (subj == s)
            subj_data[model_name][s] = {
                "p": p[mask],
                "y": y[mask],
                "n": int(mask.sum()),
            }

    def compute_unranked_fraction(subj_data_dict, L_override=None):
        """Count fraction of subjects that are completely unranked."""
        n_unranked = 0
        for s in subject_list:
            eps_vals = []
            n_vals = []
            for mn in model_names:
                sd = subj_data_dict[mn][s]
                eps_vals.append(1.0 - sd["y"].mean())
                n_vals.append(sd["n"])
            eps_avg = np.mean(eps_vals)
            n_avg = np.mean(n_vals)
            L = L_override if L_override is not None else 1.0
            delta_floor = (L * eps_avg / n_avg) ** (1.0 / 3.0)

            all_unranked = True
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    acc_i = subj_data_dict[model_names[i]][s]["y"].mean()
                    acc_j = subj_data_dict[model_names[j]][s]["y"].mean()
                    gap = abs(acc_i - acc_j)
                    if gap >= delta_floor:
                        all_unranked = False
                        break
                if not all_unranked:
                    break
            if all_unranked:
                n_unranked += 1
        return n_unranked / n_subj

    unranked_point = compute_unranked_fraction(subj_data)

    unranked_samples = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        boot_subj_data = {}
        for mn in model_names:
            boot_subj_data[mn] = {}
            for s in subject_list:
                orig = subj_data[mn][s]
                n_s = orig["n"]
                idx_s = np.random.randint(0, n_s, size=n_s)
                boot_subj_data[mn][s] = {
                    "p": orig["p"][idx_s],
                    "y": orig["y"][idx_s],
                    "n": n_s,
                }
        unranked_samples[b] = compute_unranked_fraction(boot_subj_data)
        if (b + 1) % 250 == 0:
            print(f"    Leaderboard noise: {b+1}/{N_BOOT} bootstraps done...")

    unranked_ci = ci(unranked_samples)
    unranked_ci["point"] = float(unranked_point)

    results["leaderboard_noise"] = {
        "n_subjects": n_subj,
        "unranked_fraction": unranked_ci,
    }

    print(f"\n  Point estimate: {unranked_point:.4f} ({int(unranked_point * n_subj)}/{n_subj} subjects)")
    print_ci("Unranked fraction", unranked_ci)
    print(f"  [Step 5 done in {time.time()-t1:.1f}s]")

    # ================================================================
    # 6. Active vs Passive slope CIs
    # ================================================================
    t1 = time.time()
    print("\n" + "=" * 70)
    print("6. Active vs Passive Slope CIs")
    print("=" * 70)

    # Strategy: For each bootstrap, subsample at a few m values from the
    # bootstrapped pool, compute ECE error, fit log-log slope.
    # Use 200 bootstrap replicates (still reliable for CIs) with fewer
    # subsampling reps per point to keep runtime reasonable.
    N_BOOT_SLOPE = 200
    m_values = [200, 500, 1000, 2000, 5000]
    n_sub_reps = 30  # subsampling reps per (bootstrap, m) pair

    for model_name in MODEL_FILES:
        p = all_data[model_name]["p"]
        y = all_data[model_name]["y"]
        ece_true = compute_ece(p, y)
        log_m = np.log10(np.array(m_values, dtype=float))

        # --- Point estimates (50 reps) ---
        def subsample_errors(p_pool, y_pool, ece_ref, m_val, n_reps):
            """Compute mean |ECE_hat - ECE_ref| over n_reps subsamples."""
            errs = np.zeros(n_reps)
            for r in range(n_reps):
                idx = np.random.randint(0, len(p_pool), size=m_val)
                errs[r] = abs(compute_ece(p_pool[idx], y_pool[idx]) - ece_ref)
            return errs.mean()

        # Passive point estimate
        log_err_passive = np.array([
            np.log10(max(subsample_errors(p, y, ece_true, m, 50), 1e-10))
            for m in m_values
        ])
        passive_slope, passive_int = np.polyfit(log_m, log_err_passive, 1)

        # Active point estimate: stratified sampling by confidence decile
        strat_edges = np.linspace(0, 1, 11)
        def compute_active_errors(p_pool, y_pool, ece_ref, m_val, n_reps):
            """Active sampling: allocate proportional to sqrt(variance)."""
            bin_vars = []
            bin_indices = []
            for sb in range(10):
                lo, hi = strat_edges[sb], strat_edges[sb + 1]
                if sb < 9:
                    mask = (p_pool >= lo) & (p_pool < hi)
                else:
                    mask = (p_pool >= lo) & (p_pool <= hi)
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    acc = y_pool[indices].mean()
                    var_proxy = acc * (1 - acc) + 0.01
                else:
                    var_proxy = 0.01
                bin_vars.append(var_proxy)
                bin_indices.append(indices)

            weights = np.sqrt(np.array(bin_vars))
            weights /= weights.sum()
            allocs = np.maximum((weights * m_val).astype(int), 1)

            errs = np.zeros(n_reps)
            for r in range(n_reps):
                p_samp = []
                y_samp = []
                for sb in range(10):
                    if len(bin_indices[sb]) == 0:
                        continue
                    chosen = np.random.choice(bin_indices[sb],
                                              size=min(allocs[sb], len(bin_indices[sb])),
                                              replace=True)
                    p_samp.append(p_pool[chosen])
                    y_samp.append(y_pool[chosen])
                if p_samp:
                    ps = np.concatenate(p_samp)
                    ys = np.concatenate(y_samp)
                    errs[r] = abs(compute_ece(ps, ys, B=B_ECE) - ece_ref)
                else:
                    errs[r] = 1.0
            return errs.mean()

        log_err_active = np.array([
            np.log10(max(compute_active_errors(p, y, ece_true, m, 50), 1e-10))
            for m in m_values
        ])
        active_slope, active_int = np.polyfit(log_m, log_err_active, 1)

        print(f"\n  {model_name} point estimates:")
        print(f"    Passive slope: {passive_slope:.4f}")
        print(f"    Active slope:  {active_slope:.4f}")

        # --- Bootstrap CIs (N_BOOT_SLOPE replicates) ---
        passive_slopes = np.zeros(N_BOOT_SLOPE)
        active_slopes = np.zeros(N_BOOT_SLOPE)

        for b in range(N_BOOT_SLOPE):
            idx = boot_idx[b]  # reuse first N_BOOT_SLOPE indices
            p_b, y_b = p[idx], y[idx]
            ece_b = compute_ece(p_b, y_b)

            # Passive
            le_p = np.array([
                np.log10(max(subsample_errors(p_b, y_b, ece_b, m, n_sub_reps), 1e-10))
                for m in m_values
            ])
            passive_slopes[b], _ = np.polyfit(log_m, le_p, 1)

            # Active
            le_a = np.array([
                np.log10(max(compute_active_errors(p_b, y_b, ece_b, m, n_sub_reps), 1e-10))
                for m in m_values
            ])
            active_slopes[b], _ = np.polyfit(log_m, le_a, 1)

            if (b + 1) % 50 == 0:
                elapsed_b = time.time() - t1
                rate = (b + 1) / elapsed_b
                remaining = (N_BOOT_SLOPE - b - 1) / rate
                print(f"    {model_name}: {b+1}/{N_BOOT_SLOPE} slope bootstraps "
                      f"({elapsed_b:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        # Ratio
        ratio_samples = np.where(
            np.abs(passive_slopes) > 1e-6,
            active_slopes / passive_slopes,
            np.nan
        )
        valid_ratio = ratio_samples[~np.isnan(ratio_samples)]

        passive_ci = ci(passive_slopes)
        passive_ci["point"] = float(passive_slope)
        active_ci = ci(active_slopes)
        active_ci["point"] = float(active_slope)
        ratio_ci = ci(valid_ratio) if len(valid_ratio) > 0 else {
            "point": np.nan, "mean": np.nan, "lo": np.nan, "hi": np.nan
        }

        results["models"][model_name]["slopes"] = {
            "passive": passive_ci,
            "active": active_ci,
            "active_passive_ratio": ratio_ci,
            "n_bootstrap_slope": N_BOOT_SLOPE,
        }

        print(f"\n  {model_name} CIs:")
        print_ci("Passive slope", passive_ci)
        print_ci("Active slope", active_ci)
        print_ci("Active/Passive ratio", ratio_ci)

    print(f"  [Step 6 done in {time.time()-t1:.1f}s]")

    # ================================================================
    # 7. Benchmark Demolition Floor CIs
    # ================================================================
    t1 = time.time()
    print("\n" + "=" * 70)
    print("7. Benchmark Demolition Floor CIs")
    print("=" * 70)

    # 7a. Full MMLU floor (already done in step 3 per model)
    print("\n  [Full MMLU floors -- same as step 3, repeated for clarity]")
    for model_name in MODEL_FILES:
        floor_ci = results["models"][model_name]["delta_floor"]
        print_ci(f"  {model_name} delta_floor", floor_ci)

    # 7b. Per-subject floors (N_BOOT_SUBJ replicates for speed)
    N_BOOT_SUBJ = 500
    print(f"\n  [Per-subject floors -- {N_BOOT_SUBJ} bootstrap replicates per (subject, model)]")

    per_subject_floors = {}
    for si, s in enumerate(subject_list):
        subj_floors = {}
        for model_name in model_names:
            sd = subj_data[model_name][s]
            p_s, y_s = sd["p"], sd["y"]
            n_s = sd["n"]

            if n_s < 30:
                subj_floors[model_name] = {
                    "point": None, "lo": None, "hi": None,
                    "note": f"too few items ({n_s})"
                }
                continue

            n_bins_s = max(3, min(10, n_s // 15))
            min_per_s = max(3, n_s // 30)

            eps_point_s = 1.0 - y_s.mean()
            L_point_s = estimate_lipschitz(p_s, y_s, n_bins=n_bins_s, min_per_bin=min_per_s)
            floor_point_s = (L_point_s * eps_point_s / n_s) ** (1.0 / 3.0)

            floor_boot = np.zeros(N_BOOT_SUBJ)
            for b in range(N_BOOT_SUBJ):
                idx_b = np.random.randint(0, n_s, size=n_s)
                p_b = p_s[idx_b]
                y_b = y_s[idx_b]
                eps_b = 1.0 - y_b.mean()
                L_b = estimate_lipschitz(p_b, y_b, n_bins=n_bins_s, min_per_bin=min_per_s)
                floor_boot[b] = (L_b * eps_b / n_s) ** (1.0 / 3.0)

            fc = ci(floor_boot)
            fc["point"] = float(floor_point_s)
            subj_floors[model_name] = fc

        per_subject_floors[s] = subj_floors
        if (si + 1) % 10 == 0:
            print(f"    Per-subject floors: {si+1}/{n_subj} subjects done...")

    results["benchmark_demolition"] = {
        "full_mmlu": {mn: results["models"][mn]["delta_floor"] for mn in model_names},
        "per_subject": per_subject_floors,
    }

    # Print summary of per-subject floors (5 examples)
    example_subjects = subject_list[:5]
    for s in example_subjects:
        print(f"\n  Subject: {s}")
        for mn in model_names:
            fc = per_subject_floors[s][mn]
            if fc.get("point") is not None:
                print_ci(f"    {mn}", fc)
            else:
                print(f"    {mn:40s}: {fc.get('note', 'N/A')}")

    print(f"  [Step 7 done in {time.time()-t1:.1f}s]")

    # ================================================================
    # Summary Table
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: 95% Bootstrap Confidence Intervals")
    print("=" * 70)

    header = f"{'Metric':35s} | {'Model':20s} | {'Point':>8s} | {'95% CI':>20s}"
    print(header)
    print("-" * len(header))

    for model_name in MODEL_FILES:
        m = results["models"][model_name]
        for metric, key in [("ECE (B=50)", "ECE"),
                            ("Error rate (eps)", "epsilon"),
                            ("Lipschitz L_hat", "L_hat"),
                            ("Verif. floor (delta)", "delta_floor")]:
            c = m[key]
            print(f"{metric:35s} | {model_name:20s} | {c['point']:8.4f} | [{c['lo']:.4f}, {c['hi']:.4f}]")

        # Self-eval
        se = m["self_eval_spearman"]
        rc = se["r"]
        r_pt = rc["point"] if rc["point"] is not None else float("nan")
        lo = rc.get("lo", float("nan"))
        hi = rc.get("hi", float("nan"))
        print(f"{'Spearman r (conf vs gap)':35s} | {model_name:20s} | {r_pt:8.4f} | [{lo:.4f}, {hi:.4f}]")
        print(f"{'  frac p>0.05':35s} | {model_name:20s} | {se['frac_p_gt_005']:8.4f} |")

        # Slopes
        sl = m["slopes"]
        print(f"{'Passive slope':35s} | {model_name:20s} | {sl['passive']['point']:8.4f} | [{sl['passive']['lo']:.4f}, {sl['passive']['hi']:.4f}]")
        print(f"{'Active slope':35s} | {model_name:20s} | {sl['active']['point']:8.4f} | [{sl['active']['lo']:.4f}, {sl['active']['hi']:.4f}]")
        print("-" * len(header))

    # Leaderboard noise
    uc = results["leaderboard_noise"]["unranked_fraction"]
    print(f"{'Unranked subject fraction':35s} | {'all models':20s} | {uc['point']:8.4f} | [{uc['lo']:.4f}, {uc['hi']:.4f}]")
    print("-" * len(header))

    # ── Save results ──
    out_path = os.path.join(OUT_DIR, "bootstrap_confidence_intervals.json")

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
