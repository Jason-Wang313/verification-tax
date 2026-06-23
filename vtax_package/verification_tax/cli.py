"""verification-tax CLI — audit a per-item (confidence, correct) stream."""

import argparse
import csv
import json
import sys
from pathlib import Path

from verification_tax.core import audit_predictions


def _load_jsonl(path: Path) -> tuple[list[float], list[int]]:
    confs: list[float] = []
    corr: list[int] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            c = row.get("confidence", row.get("max_conf"))
            y = row.get("correct", row.get("is_correct"))
            if c is None or y is None:
                continue
            confs.append(float(c))
            corr.append(int(bool(y)))
    return confs, corr


def _load_csv(path: Path) -> tuple[list[float], list[int]]:
    confs: list[float] = []
    corr: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            c = row.get("confidence") or row.get("max_conf")
            y = row.get("correct") or row.get("is_correct")
            if c is None or y is None:
                continue
            confs.append(float(c))
            corr.append(int(float(y) > 0.5))
    return confs, corr


def _load(path: Path) -> tuple[list[float], list[int]]:
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    raise ValueError(f"unsupported extension: {path.suffix}")


def _format_table(result: dict) -> str:
    cq = result["confidence_quality"]
    lines = [
        f"  items (m)                : {result['m']:,}",
        f"  error rate (eps)         : {result['eps']:.4f}",
        f"  Lipschitz estimate (L)   : {result['L_hat']:.3f}",
        f"  passive floor            : {result['passive_floor']:.4f}",
        f"  active floor             : {result['active_floor']:.4f}",
        f"  phase-transition m*      : {result['phase_transition_m']:,}",
        f"  optimal bin count (B*)   : {result['optimal_bins']}",
        f"  confidence quality       : {'PASS' if cq['passed'] else 'FAIL'}",
    ]
    if not cq["passed"]:
        lines.append(f"    reasons               : {', '.join(cq['reasons'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verification-tax",
        description="Information-theoretic audit of per-item prediction traces.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Audit a JSONL or CSV of (confidence, correct) items.")
    p_audit.add_argument("path", type=Path, help="Path to JSONL (keys: confidence|max_conf, correct|is_correct) or CSV.")
    p_audit.add_argument("--L", type=float, default=None, help="Override Lipschitz constant; otherwise estimated.")
    p_audit.add_argument("--json", dest="as_json", action="store_true", help="Emit machine-readable JSON.")

    args = parser.parse_args(argv)
    if args.command == "audit":
        if not args.path.exists():
            print(f"error: file not found: {args.path}", file=sys.stderr)
            return 2
        confs, corr = _load(args.path)
        if not confs:
            print("error: no usable rows", file=sys.stderr)
            return 2
        result = audit_predictions(confs, corr, L=args.L)
        if args.as_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"verification-tax audit: {args.path}")
            print(_format_table(result))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
