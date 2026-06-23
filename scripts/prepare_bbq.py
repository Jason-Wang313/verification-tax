"""Download BBQ (Bias Benchmark for QA) and convert to the paper's all_items.json schema.

BBQ has 11 category folders on https://github.com/nyu-mll/BBQ. Each .jsonl item
carries a question, 3 answer options, a correct-answer index, and a subgroup
category. We flatten all categories into a single all_items.json compatible
with run_local_experiment.py, and add a `subgroup` field that the fairness
analysis script reads.

Usage:
    pip install datasets
    python scripts/prepare_bbq.py   # writes data/bbq/all_items.json
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "bbq"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("error: pip install datasets", flush=True)
        return 2

    try:
        ds = load_dataset("oskarvanderwal/bbq", split="test")
    except Exception as e:
        print(f"error: could not load oskarvanderwal/bbq: {e}")
        return 2

    items: list[dict] = []
    for i, row in enumerate(ds):
        choices = [row.get("ans0", ""), row.get("ans1", ""), row.get("ans2", "")]
        if any(c is None for c in choices):
            continue
        try:
            label = int(row.get("label", -1))
        except (TypeError, ValueError):
            continue
        if label < 0 or label >= len(choices):
            continue
        cat = row.get("category", "unknown")
        items.append({
            "question_id": f"bbq_{i:06d}",
            "subject": cat,
            "subgroup": cat,
            "context": row.get("context", ""),
            "question": (row.get("context", "") + " " + row.get("question", "")).strip(),
            "choices": choices,
            "answer": label,
        })
    out_path = OUT_DIR / "all_items.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(items):,} items to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
