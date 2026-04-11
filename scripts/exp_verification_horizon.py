#!/usr/bin/env python3
"""
exp_verification_horizon.py

Computes the verification horizon N* for frontier models across domains
and outputs a LaTeX table plus JSON analysis.

Formula (passive):  N* = (c0^2 * M_total / L)^(1 / (2*alpha))
Formula (active):   N*_active = (c0 * M_total / L)^(1 / alpha)

Scaling law: eps(N) = c0 * N^(-alpha)
  Chinchilla approximation: alpha ~ 0.5, c0 ~ 1
"""

import json
import os
import math

# ── Paths (absolute) ────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\wangz\verification tax"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Scaling law parameters ──────────────────────────────────────────────────
ALPHA = 0.5          # Chinchilla scaling exponent
C0 = 1.0             # Chinchilla scaling constant
L = 1.0              # Lipschitz constant

# ── Frontier models ─────────────────────────────────────────────────────────
MODELS = [
    {"name": "GPT-4",              "params": 1.8e12,  "eps_mmlu": 0.12},
    {"name": "GPT-4o",             "params": 2.0e11,  "eps_mmlu": 0.13},
    {"name": "Claude 3 Opus",      "params": 1.75e11, "eps_mmlu": 0.15},
    {"name": "Claude 3.5 Sonnet",  "params": 7.0e10,  "eps_mmlu": 0.12},
    {"name": "Gemini 1.5 Pro",     "params": 5.4e11,  "eps_mmlu": 0.10},
    {"name": "Llama-3-405B",       "params": 4.05e11, "eps_mmlu": 0.16},
    {"name": "Llama-4-Maverick",   "params": 1.7e10,  "eps_mmlu": 0.27},
    {"name": "Qwen3-Next-80B",     "params": 8.0e10,  "eps_mmlu": 0.17},
]

# ── Domain data availability ────────────────────────────────────────────────
DOMAINS = {
    "General NLP (MMLU-scale)": 14_000,
    "Medical imaging":          500_000,
    "Legal":                    10_000,
    "Financial":                20_000,
    "Code (HumanEval-scale)":   5_000,
    "Autonomous driving":       100_000,
}


def compute_n_star_passive(c0, m_total, l, alpha):
    """N* = (c0^2 * M_total / L)^(1 / (2*alpha))"""
    base = (c0 ** 2) * m_total / l
    exponent = 1.0 / (2.0 * alpha)
    return base ** exponent


def compute_n_star_active(c0, m_total, l, alpha):
    """N*_active = (c0 * M_total / L)^(1 / alpha)"""
    base = c0 * m_total / l
    exponent = 1.0 / alpha
    return base ** exponent


def fmt_sci(x):
    """Format a number in scientific notation like 1.2e+06."""
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mantissa = x / (10 ** exp)
    return f"{mantissa:.2f}e{exp:+d}"


def main():
    rows = []  # for JSON
    latex_rows = []  # for LaTeX

    print("=" * 100)
    print("VERIFICATION HORIZON ANALYSIS")
    print(f"Scaling law: eps(N) = {C0} * N^(-{ALPHA}),  L = {L}")
    print("=" * 100)

    # Header
    header = (
        f"{'Model':<22} {'Domain':<26} {'N':>12} "
        f"{'N*(pass)':>14} {'N*(act)':>14} {'Gap(N/N*)':>12} {'Exceeds?':>10}"
    )
    print(header)
    print("-" * len(header))

    cmark = "\\cmark"
    xmark = "\\xmark"

    for model in MODELS:
        for domain_name, m_total in DOMAINS.items():
            n_star = compute_n_star_passive(C0, m_total, L, ALPHA)
            n_star_active = compute_n_star_active(C0, m_total, L, ALPHA)
            gap = model["params"] / n_star
            exceeds = model["params"] > n_star

            row = {
                "model": model["name"],
                "params_N": model["params"],
                "eps_mmlu": model["eps_mmlu"],
                "domain": domain_name,
                "M_total": m_total,
                "N_star_passive": n_star,
                "N_star_active": n_star_active,
                "gap_N_over_Nstar": gap,
                "exceeds_horizon": exceeds,
            }
            rows.append(row)

            exceeds_str = "YES" if exceeds else "no"
            print(
                f"{model['name']:<22} {domain_name:<26} {fmt_sci(model['params']):>12} "
                f"{fmt_sci(n_star):>14} {fmt_sci(n_star_active):>14} "
                f"{gap:>12.2f} {exceeds_str:>10}"
            )

            # LaTeX row
            latex_rows.append(
                f"        {model['name']:<22} & {domain_name:<26} & "
                f"${fmt_sci(model['params'])}$ & "
                f"${fmt_sci(n_star)}$ & "
                f"${fmt_sci(n_star_active)}$ & "
                f"{gap:.2f} & "
                f"{cmark if exceeds else xmark} \\\\"
            )

        # visual separator between models
        print()

    # ── Summary statistics ──────────────────────────────────────────────────
    n_exceeds = sum(1 for r in rows if r["exceeds_horizon"])
    n_total = len(rows)
    print("=" * 100)
    print(f"SUMMARY: {n_exceeds}/{n_total} (model, domain) pairs EXCEED the passive verification horizon.")
    print()

    # Per-domain summary
    for domain_name in DOMAINS:
        domain_rows = [r for r in rows if r["domain"] == domain_name]
        exc = sum(1 for r in domain_rows if r["exceeds_horizon"])
        avg_gap = sum(r["gap_N_over_Nstar"] for r in domain_rows) / len(domain_rows)
        print(f"  {domain_name:<30}: {exc}/{len(domain_rows)} exceed horizon, avg gap = {avg_gap:.1f}x")

    print()

    # ── Write LaTeX ─────────────────────────────────────────────────────────
    tex_path = os.path.join(RESULTS_DIR, "verification_horizon_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by exp_verification_horizon.py\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Verification horizon $N^*$ for frontier models across domains. "
                "Gap $= N / N^*$; values $> 1$ indicate the model exceeds the horizon.}\n")
        f.write("\\label{tab:verification_horizon}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llccccc}\n")
        f.write("\\toprule\n")
        f.write("        Model & Domain & $N$ & $N^*_{\\text{pass}}$ & "
                "$N^*_{\\text{act}}$ & Gap & Exceeds \\\\\n")
        f.write("\\midrule\n")

        prev_model = None
        for i, lr in enumerate(latex_rows):
            # extract model name from row data
            current_model = rows[i]["model"]
            if prev_model is not None and current_model != prev_model:
                f.write("        \\addlinespace\n")
            prev_model = current_model
            f.write(lr + "\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table written to: {tex_path}")

    # ── Write JSON ──────────────────────────────────────────────────────────
    json_path = os.path.join(ANALYSIS_DIR, "verification_horizon.json")
    output = {
        "description": "Verification horizon analysis for frontier models",
        "scaling_law": {
            "alpha": ALPHA,
            "c0": C0,
            "L": L,
            "formula_passive": "N* = (c0^2 * M_total / L)^(1/(2*alpha))",
            "formula_active": "N*_active = (c0 * M_total / L)^(1/alpha)",
        },
        "models": [
            {"name": m["name"], "params": m["params"], "eps_mmlu": m["eps_mmlu"]}
            for m in MODELS
        ],
        "domains": DOMAINS,
        "results": rows,
        "summary": {
            "total_pairs": n_total,
            "pairs_exceeding_horizon": n_exceeds,
            "fraction_exceeding": n_exceeds / n_total,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"JSON analysis written to: {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
