"""Safety-claim floor validation on XSTest.

Shows that typical safety claims ("refusal rate 95% → 97%") sit below the
accuracy floor at the dataset's sample size (n≈450). Computes the smallest
refusal-rate improvement that is statistically distinguishable at XSTest's n,
and emits a LaTeX table + figure for the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
XSTEST_DIR = PROJECT_ROOT / "data" / "xstest"
TOXIGEN_DIR = PROJECT_ROOT / "data" / "toxigen"
FIG_DIR = PROJECT_ROOT / "figures"
RES_DIR = PROJECT_ROOT / "results"
ANA_DIR = RES_DIR / "analysis"
for d in (FIG_DIR, RES_DIR, ANA_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _accuracy_floor(eps: float, n: int) -> float:
    """2 * sqrt(eps*(1-eps)/n) — the standard proportion-difference noise floor."""
    if n <= 0 or eps <= 0 or eps >= 1:
        return float("inf")
    return 2 * np.sqrt(eps * (1 - eps) / n)


def _rows_for_dir(bench_name: str, directory: Path) -> list[dict]:
    traces = sorted(directory.glob("results_*.jsonl"))
    rows: list[dict] = []
    for trace in traces:
        model_id = trace.stem.replace("results_", "")
        n = 0
        correct = 0
        with trace.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "is_correct" not in r:
                    continue
                n += 1
                correct += int(bool(r["is_correct"]))
        if n == 0:
            continue
        acc = correct / n
        eps = 1 - acc
        floor = _accuracy_floor(eps, n)
        rows.append({
            "benchmark": bench_name,
            "model": model_id,
            "n": n,
            "accuracy": acc,
            "eps": eps,
            "accuracy_floor": floor,
            "minimum_verifiable_improvement_pct": 100 * floor,
        })
    return rows


def main() -> int:
    rows = _rows_for_dir("xstest", XSTEST_DIR) + _rows_for_dir("toxigen", TOXIGEN_DIR)
    if not rows:
        print("error: no safety data — run the XSTest/ToxiGen sweeps first.")
        return 2

    # Annotate verdicts for common "95 -> 97" and "90 -> 95" claims.
    verdicts = []
    for claim_baseline, claim_new in [(0.95, 0.97), (0.90, 0.95), (0.80, 0.90), (0.70, 0.80)]:
        for r in rows:
            floor = _accuracy_floor(min(1-claim_baseline, 1-claim_new), r["n"])
            gap = claim_new - claim_baseline
            verdict = "VERIFIED" if gap >= 2 * floor else ("MARGINAL" if gap >= floor else "NOISE")
            verdicts.append({
                "model": r["model"], "n": r["n"],
                "claim": f"{100*claim_baseline:.0f}% -> {100*claim_new:.0f}%",
                "gap": round(gap, 4), "floor": round(floor, 4), "verdict": verdict,
            })

    out_json = ANA_DIR / "safety_floor_results.json"
    out_json.write_text(json.dumps({"per_model": rows, "claim_verdicts": verdicts}, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")

    # Canonical table: verdicts at XSTest's n=450 and ToxiGen's n=940.
    n_xstest = next((r["n"] for r in rows if r["benchmark"] == "xstest"), 450)
    n_toxigen = next((r["n"] for r in rows if r["benchmark"] == "toxigen"), 940)
    tex = ["\\begin{table}[h]\n\\centering",
           "\\caption{Safety-claim verification. Common small refusal-rate claims fall below the accuracy floor at both XSTest ($n{=}" + str(n_xstest) + "$) and ToxiGen ($n{=}" + str(n_toxigen) + "$).}",
           "\\label{tab:safety-floor}",
           "\\small",
           "\\begin{tabular}{lrrll}\n\\toprule",
           "Claim & Gap & Floor (XSTest) & Floor (ToxiGen) & Verdict \\\\",
           "\\midrule"]
    for claim_baseline, claim_new in [(0.95, 0.97), (0.90, 0.95), (0.80, 0.90), (0.70, 0.80)]:
        gap = claim_new - claim_baseline
        floor_x = _accuracy_floor(min(1-claim_baseline, 1-claim_new), n_xstest)
        floor_t = _accuracy_floor(min(1-claim_baseline, 1-claim_new), n_toxigen)
        def _v(g, f):
            return "VERIFIED" if g >= 2*f else ("MARGINAL" if g >= f else "NOISE")
        verdicts = f"{_v(gap, floor_x)} / {_v(gap, floor_t)}"
        tex.append(
            f"{100*claim_baseline:.0f}\\%$\\rightarrow${100*claim_new:.0f}\\% & "
            f"{100*gap:.1f}\\% & {100*floor_x:.2f}\\% & {100*floor_t:.2f}\\% & \\textbf{{{verdicts}}} \\\\"
        )
    tex.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    tex_path = RES_DIR / "safety_floor_table.tex"
    tex_path.write_text("\n".join(tex), encoding="utf-8")
    print(f"wrote {tex_path}")

    # Figure: floor vs n for common claim magnitudes.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ns = np.logspace(2, 5, 50)
    for claim_gap, label in [(0.02, "+2 pp claim"), (0.05, "+5 pp claim"), (0.10, "+10 pp claim")]:
        floors = [_accuracy_floor(0.05, int(n)) for n in ns]
        ax.axhline(claim_gap, linestyle=":", alpha=0.4)
    # Plot the accuracy floor curve.
    floor_curve = [_accuracy_floor(0.05, int(n)) for n in ns]
    ax.plot(ns, floor_curve, "k-", label=r"$\delta_{\rm floor}=2\sqrt{\varepsilon(1-\varepsilon)/n}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(450, color="red", linestyle="--", alpha=0.5, label="XSTest $n{=}450$")
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel("verifiable improvement threshold")
    ax.set_title("Safety claims vs. accuracy floor")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig_path = FIG_DIR / "fig_safety_floor.pdf"
    fig.savefig(fig_path)
    fig.savefig(fig_path.with_suffix(".png"), dpi=150)
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
