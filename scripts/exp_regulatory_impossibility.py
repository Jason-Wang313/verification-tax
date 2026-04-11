#!/usr/bin/env python3
"""
exp_regulatory_impossibility.py

Computes data requirements for regulatory frameworks and demonstrates
their infeasibility with currently available labeled data.

Formula (passive):  m_required = K / pi_min * L * eps / delta^3
Formula (active):   m_active   = K / pi_min * eps / delta^2
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

# ── Regulatory frameworks ───────────────────────────────────────────────────
FRAMEWORKS = [
    {
        "name": "EU AI Act Annex III",
        "description": "High-risk medical AI",
        "K": 10,           # 2 sex x 5 age brackets
        "pi_min": 0.05,
        "delta": 0.02,     # clinically meaningful precision
        "eps": 0.05,       # frontier medical AI error rate
        "L": 1.0,
        "domain": "Medical imaging",
    },
    {
        "name": "NIST AI RMF",
        "description": "Performance monitoring",
        "K": 5,            # use contexts
        "pi_min": 0.10,
        "delta": 0.01,
        "eps": 0.03,
        "L": 1.0,
        "domain": "General medical",
    },
    {
        "name": "FDA SaMD",
        "description": "Clinical subgroup validation",
        "K": 8,            # clinical subgroups
        "pi_min": 0.05,
        "delta": 0.01,
        "eps": 0.02,
        "L": 1.0,
        "domain": "Medical imaging",
    },
]

# ── Available data by domain ────────────────────────────────────────────────
AVAILABLE_DATA = {
    "Medical imaging": {
        "total": 707_000,
        "breakdown": {
            "CheXpert": 224_000,
            "MIMIC-CXR": 371_000,
            "NIH ChestXray14": 112_000,
        },
    },
    "General medical": {
        "total": 500_000,
        "breakdown": {},
    },
    "Clinical trials": {
        "total": 200_000,
        "breakdown": {},
    },
}


def compute_m_required(K, pi_min, L, eps, delta):
    """m_required = K / pi_min * L * eps / delta^3"""
    return K / pi_min * L * eps / (delta ** 3)


def compute_m_active(K, pi_min, eps, delta):
    """m_active = K / pi_min * eps / delta^2"""
    return K / pi_min * eps / (delta ** 2)


def fmt_num(x):
    """Format large numbers with commas."""
    if x >= 1e9:
        return f"{x:.2e}"
    return f"{x:,.0f}"


def fmt_sci_latex(x):
    """Format number as LaTeX scientific notation."""
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mantissa = x / (10 ** exp)
    if exp == 0:
        return f"{mantissa:.2f}"
    return f"{mantissa:.2f} \\times 10^{{{exp}}}"


def main():
    results = []

    print("=" * 110)
    print("REGULATORY IMPOSSIBILITY ANALYSIS")
    print("=" * 110)
    print()

    header = (
        f"{'Framework':<24} {'K':>4} {'pi_min':>8} {'delta':>8} {'eps':>8} "
        f"{'m_req (passive)':>18} {'m_active':>18} {'m_avail':>12} "
        f"{'Gap (pass)':>12} {'Gap (act)':>12} {'Feasible?':>10}"
    )
    print(header)
    print("-" * len(header))

    for fw in FRAMEWORKS:
        K = fw["K"]
        pi_min = fw["pi_min"]
        delta = fw["delta"]
        eps = fw["eps"]
        L = fw["L"]
        domain = fw["domain"]

        m_required = compute_m_required(K, pi_min, L, eps, delta)
        m_active = compute_m_active(K, pi_min, eps, delta)
        m_available = AVAILABLE_DATA[domain]["total"]

        gap_passive = m_required / m_available
        gap_active = m_active / m_available

        feasible_passive = m_required <= m_available
        feasible_active = m_active <= m_available

        result = {
            "framework": fw["name"],
            "description": fw["description"],
            "domain": domain,
            "parameters": {
                "K": K,
                "pi_min": pi_min,
                "delta": delta,
                "eps": eps,
                "L": L,
            },
            "m_required_passive": m_required,
            "m_required_active": m_active,
            "m_available": m_available,
            "gap_passive": gap_passive,
            "gap_active": gap_active,
            "feasible_passive": feasible_passive,
            "feasible_active": feasible_active,
        }
        results.append(result)

        feas_str = "yes" if feasible_passive else "NO"
        print(
            f"{fw['name']:<24} {K:>4} {pi_min:>8.2f} {delta:>8.3f} {eps:>8.3f} "
            f"{fmt_num(m_required):>18} {fmt_num(m_active):>18} {fmt_num(m_available):>12} "
            f"{gap_passive:>12.1f}x {gap_active:>12.1f}x {feas_str:>10}"
        )

    print()
    print("=" * 110)
    print("DETAILED BREAKDOWN")
    print("=" * 110)

    for i, fw in enumerate(FRAMEWORKS):
        r = results[i]
        print()
        print(f"--- {fw['name']} ({fw['description']}) ---")
        print(f"  Domain:               {fw['domain']}")
        print(f"  Subgroups K:          {fw['K']}")
        print(f"  Min group fraction:   {fw['pi_min']}")
        print(f"  Required precision:   delta = {fw['delta']}")
        print(f"  Model error rate:     eps = {fw['eps']}")
        print(f"  Lipschitz constant:   L = {fw['L']}")
        print()
        print(f"  Passive requirement:  {fmt_num(r['m_required_passive'])} samples")
        print(f"  Active requirement:   {fmt_num(r['m_required_active'])} samples")
        print(f"  Available data:       {fmt_num(r['m_available'])} samples")
        print()
        print(f"  Passive feasibility gap: {r['gap_passive']:.1f}x  "
              f"({'INFEASIBLE' if not r['feasible_passive'] else 'feasible'})")
        print(f"  Active feasibility gap:  {r['gap_active']:.1f}x  "
              f"({'INFEASIBLE' if not r['feasible_active'] else 'feasible'})")

        if not r["feasible_passive"]:
            # How much delta would need to relax for feasibility
            # m_required = K / pi_min * L * eps / delta^3 <= m_available
            # delta^3 >= K / pi_min * L * eps / m_available
            delta_min = (fw["K"] / fw["pi_min"] * fw["L"] * fw["eps"]
                         / r["m_available"]) ** (1.0 / 3.0)
            print(f"  -> To be feasible (passive), delta must relax to >= {delta_min:.4f} "
                  f"(currently {fw['delta']})")

    # ── Available data breakdown ────────────────────────────────────────────
    print()
    print("=" * 110)
    print("AVAILABLE DATA SUMMARY")
    print("=" * 110)
    for domain, info in AVAILABLE_DATA.items():
        print(f"\n  {domain}: {fmt_num(info['total'])} total")
        for ds, count in info["breakdown"].items():
            print(f"    - {ds}: {fmt_num(count)}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    print("=" * 110)
    print("CONCLUSION")
    print("=" * 110)
    n_infeasible_passive = sum(1 for r in results if not r["feasible_passive"])
    n_infeasible_active = sum(1 for r in results if not r["feasible_active"])
    n_total = len(results)
    print(f"  Passive verification: {n_infeasible_passive}/{n_total} frameworks are INFEASIBLE")
    print(f"  Active verification:  {n_infeasible_active}/{n_total} frameworks are INFEASIBLE")
    print()

    # ── Write LaTeX ─────────────────────────────────────────────────────────
    tex_path = os.path.join(RESULTS_DIR, "regulatory_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by exp_regulatory_impossibility.py\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Data requirements for regulatory compliance verification. "
                "Gap $= m_{\\text{req}} / m_{\\text{avail}}$; "
                "values $> 1$ indicate infeasibility with current data.}\n")
        f.write("\\label{tab:regulatory_impossibility}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcccccccc}\n")
        f.write("\\toprule\n")
        f.write("        Framework & $K$ & $\\pi_{\\min}$ & $\\delta$ & $\\varepsilon$ "
                "& $m_{\\text{req}}$ & $m_{\\text{active}}$ & $m_{\\text{avail}}$ "
                "& Gap \\\\\n")
        f.write("\\midrule\n")

        for i, fw in enumerate(FRAMEWORKS):
            r = results[i]
            f.write(
                f"        {fw['name']} & {fw['K']} & {fw['pi_min']:.2f} "
                f"& {fw['delta']:.3f} & {fw['eps']:.3f} "
                f"& ${fmt_sci_latex(r['m_required_passive'])}$ "
                f"& ${fmt_sci_latex(r['m_required_active'])}$ "
                f"& ${fmt_sci_latex(r['m_available'])}$ "
                f"& {r['gap_passive']:.1f}$\\times$ \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table written to: {tex_path}")

    # ── Write JSON ──────────────────────────────────────────────────────────
    json_path = os.path.join(ANALYSIS_DIR, "regulatory_impossibility.json")
    output = {
        "description": "Regulatory impossibility analysis for AI verification",
        "formulas": {
            "m_required_passive": "K / pi_min * L * eps / delta^3",
            "m_required_active": "K / pi_min * eps / delta^2",
        },
        "frameworks": [
            {
                "name": fw["name"],
                "description": fw["description"],
                "K": fw["K"],
                "pi_min": fw["pi_min"],
                "delta": fw["delta"],
                "eps": fw["eps"],
                "L": fw["L"],
            }
            for fw in FRAMEWORKS
        ],
        "available_data": {
            domain: info["total"]
            for domain, info in AVAILABLE_DATA.items()
        },
        "results": results,
        "summary": {
            "total_frameworks": n_total,
            "infeasible_passive": n_infeasible_passive,
            "infeasible_active": n_infeasible_active,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"JSON analysis written to: {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
