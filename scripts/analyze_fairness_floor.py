"""Fairness-tax validation: per-subgroup verification floors on BBQ.

Computes per-subgroup eps, L_hat, passive floor, and the minority/majority
floor ratio. Outputs a LaTeX table for the paper and a figure showing
floor vs subgroup proportion.

Theory check: for fixed ε and m, the subgroup floor scales as (1/π_g)^{1/3}
because the effective sample size for subgroup g is m·π_g.

Usage:
    python scripts/analyze_fairness_floor.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BBQ_DIR = PROJECT_ROOT / "data" / "bbq"
FIG_DIR = PROJECT_ROOT / "figures"
RES_DIR = PROJECT_ROOT / "results"
ANA_DIR = RES_DIR / "analysis"
for d in (FIG_DIR, RES_DIR, ANA_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _load_jsonl(path: Path):
    confs, correct, subgroups = [], [], []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "max_conf" not in row or "is_correct" not in row:
                continue
            confs.append(float(row["max_conf"]))
            correct.append(int(bool(row["is_correct"])))
            subgroups.append(row.get("subgroup") or row.get("subject") or "unknown")
    return np.asarray(confs), np.asarray(correct), np.asarray(subgroups)


def _floor(eps: float, n: int, L: float = 1.0) -> float:
    if n <= 0:
        return float("inf")
    return (L * max(eps, 1e-4) / n) ** (1 / 3)


def main() -> int:
    traces = sorted(BBQ_DIR.glob("results_*.jsonl"))
    if not traces:
        print("error: no data/bbq/results_*.jsonl — run the BBQ sweep first.")
        return 2

    rows = []
    for trace in traces:
        model_id = trace.stem.replace("results_", "")
        confs, corr, subgroups = _load_jsonl(trace)
        if len(confs) == 0:
            continue
        m_total = len(confs)
        majority_n = max(int((subgroups == g).sum()) for g in np.unique(subgroups))
        majority_floor = _floor(float(1 - corr.mean()), majority_n)
        for g in sorted(np.unique(subgroups)):
            mask = subgroups == g
            n_g = int(mask.sum())
            if n_g == 0:
                continue
            eps_g = float(1 - corr[mask].mean())
            pi_g = n_g / m_total
            floor_g = _floor(eps_g, n_g)
            rows.append({
                "model": model_id,
                "subgroup": g,
                "n_g": n_g,
                "pi_g": pi_g,
                "eps_g": eps_g,
                "floor": floor_g,
                "floor_ratio_vs_majority": floor_g / majority_floor if majority_floor else None,
            })

    out_json = ANA_DIR / "fairness_floor_results.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out_json} ({len(rows)} rows)")

    # LaTeX summary table (per-subgroup floors aggregated across models).
    agg: dict[str, list[dict]] = {}
    for r in rows:
        agg.setdefault(r["subgroup"], []).append(r)

    tex = ["\\begin{table}[h]\n\\centering",
           "\\caption{Fairness tax: verification floor by BBQ subgroup (averaged across audited models).}",
           "\\label{tab:fairness-floor}",
           "\\small",
           "\\begin{tabular}{lrrrr}\n\\toprule",
           "Subgroup & $\\overline{\\pi_g}$ & $\\overline{\\errrate}_g$ & $\\overline{\\verfloor}_g$ & ratio vs.\\ majority \\\\",
           "\\midrule"]
    for sub in sorted(agg):
        recs = agg[sub]
        avg_pi = float(np.mean([r["pi_g"] for r in recs]))
        avg_eps = float(np.mean([r["eps_g"] for r in recs]))
        avg_floor = float(np.mean([r["floor"] for r in recs]))
        avg_ratio = float(np.mean([r["floor_ratio_vs_majority"] for r in recs if r["floor_ratio_vs_majority"]]))
        tex.append(f"{sub.replace('_', ' ')} & {avg_pi:.3f} & {avg_eps:.3f} & {avg_floor:.4f} & {avg_ratio:.2f} \\\\")
    tex.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    tex_path = RES_DIR / "fairness_floor_table.tex"
    tex_path.write_text("\n".join(tex), encoding="utf-8")
    print(f"wrote {tex_path}")

    # Figure: floor vs pi_g (log-log), with 1/pi^{1/3} theory line.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model in sorted({r["model"] for r in rows}):
        model_rows = [r for r in rows if r["model"] == model]
        xs = [r["pi_g"] for r in model_rows]
        ys = [r["floor"] for r in model_rows]
        ax.scatter(xs, ys, label=model, alpha=0.7)
    pis = np.logspace(-2, 0, 50)
    avg_eps = float(np.mean([r["eps_g"] for r in rows]))
    m_total_avg = int(np.mean([len(_load_jsonl(t)[0]) for t in traces]))
    theory = [(avg_eps / (p * m_total_avg)) ** (1 / 3) for p in pis]
    ax.plot(pis, theory, "k--", label=r"theory: $(\varepsilon/(m\pi_g))^{1/3}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"subgroup proportion $\pi_g$")
    ax.set_ylabel(r"verification floor $\delta_{\mathrm{floor},g}$")
    ax.set_title("Fairness tax: per-subgroup floors on BBQ")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = FIG_DIR / "fig_fairness_floor.pdf"
    fig.savefig(fig_path)
    fig.savefig(fig_path.with_suffix(".png"), dpi=150)
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
