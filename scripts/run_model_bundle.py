"""Bundled-benchmark runner: load one model once, sweep multiple benchmarks.

Saves the 10-30 s model-load penalty per (model, benchmark) pair by keeping
the Llama object in memory across benchmarks. Crash-safe: each item is
fsync'd to its benchmark's JSONL; on restart the resume logic picks up where
we stopped, bundle-style launch re-uses that state for the remaining items.

Usage:
    python scripts/run_model_bundle.py \
        --model-path models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
        --model-id llama-3-2-3b-instruct-q4-k-m \
        --benchmarks mmlu:500,truthfulqa:0,arc_challenge:0 \
        --n-ctx 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_local_experiment import (  # noqa: E402
    BENCHMARK_CONFIG,
    _completed_ids,
    _extract_correct_letter,
    _letter_token_ids,
    _score_one,
)


def _item_is_answerable(item: dict, choices: str) -> bool:
    """Skip items whose gold answer falls outside the benchmark's allowed choice set,
    or whose prompt is too long to score reliably."""
    if _extract_correct_letter(item, choices) not in choices:
        return False
    # Prune items with excessively long cumulative choice text (>3000 chars total);
    # they'd push past n_ctx and skew timing.
    total = len(item.get("question", "")) + sum(len(c) for c in item.get("choices", [])[:len(choices)])
    return total <= 3000


def _parse_spec(spec: str) -> list[tuple[str, int]]:
    """Parse 'mmlu:500,truthfulqa:0,arc_challenge:0' → [('mmlu', 500), ...]."""
    out: list[tuple[str, int]] = []
    for entry in spec.split(","):
        if not entry.strip():
            continue
        parts = entry.split(":")
        bench = parts[0].strip()
        n = int(parts[1]) if len(parts) > 1 else 0
        out.append((bench, n))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bundled model × multiple benchmarks.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--benchmarks", type=str, required=True,
                        help="Comma-separated BENCH:N (N=0 for full).")
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--n-threads", type=int, default=0)
    args = parser.parse_args(argv)

    schedule = _parse_spec(args.benchmarks)
    # Validate all benchmarks and load item lists eagerly.
    loaded: list[tuple[str, list[dict], int, Path, str]] = []
    for bench, cap in schedule:
        cfg = BENCHMARK_CONFIG.get(bench)
        if not cfg:
            print(f"[skip] {bench} — not in BENCHMARK_CONFIG", flush=True)
            continue
        items_file: Path = cfg["items_file"]
        out_dir: Path = cfg["output_dir"]
        choices: str = cfg["choices"]
        if not items_file.exists():
            print(f"[skip] {bench} — {items_file} missing", flush=True)
            continue
        items = json.loads(items_file.read_text(encoding="utf-8"))
        # Keep only items whose gold answer is inside the benchmark's choice set
        # and whose prompt isn't pathologically long (prevents ctx-ceiling stalls).
        items = [it for it in items if _item_is_answerable(it, choices)]
        if cap:
            items = items[:cap]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"results_{args.model_id}.jsonl"
        completed = _completed_ids(out_path)
        pending = [it for it in items if it["question_id"] not in completed]
        loaded.append((bench, pending, len(items), out_path, choices))
        print(f"{args.model_id} / {bench}: {len(items)} answerable, "
              f"{len(completed)} done, {len(pending)} pending.", flush=True)

    if not any(p for _, p, *_ in loaded):
        print("nothing to do.", flush=True)
        return 0

    # Load model once.
    from llama_cpp import Llama
    n_threads = args.n_threads or os.cpu_count() or 4
    print(f"loading {args.model_path} (n_ctx={args.n_ctx}, threads={n_threads})", flush=True)
    t_load = time.time()
    llm = Llama(
        model_path=str(args.model_path),
        n_ctx=args.n_ctx,
        n_threads=n_threads,
        n_batch=2048,           # large prompt-ingest batch
        n_ubatch=1024,          # avoid micro-batching stalls
        use_mmap=True,
        use_mlock=False,
        logits_all=False,       # fast: only final-token logits
        verbose=False,
    )
    print(f"  loaded in {time.time() - t_load:.1f}s", flush=True)

    # Run each benchmark.
    for bench, pending, total, out_path, choices in loaded:
        if not pending:
            continue
        letter_ids = _letter_token_ids(llm, choices)
        t0 = time.time()
        with out_path.open("a", encoding="utf-8") as fh:
            fd = fh.fileno()
            for i, item in enumerate(pending):
                scored = _score_one(llm, item, choices=choices, letter_ids=letter_ids)
                row = {
                    "question_id": item["question_id"],
                    "subject": item.get("subject"),
                    "subgroup": item.get("subgroup"),
                    **scored,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if (i + 1) % 10 == 0:
                    try:
                        os.fsync(fd)
                    except OSError:
                        pass
                if (i + 1) % 50 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    eta = (len(pending) - i - 1) / max(rate, 1e-6) / 60
                    print(f"  [{args.model_id}/{bench}] {i+1}/{len(pending)} @ "
                          f"{rate:.2f}/s ETA {eta:.1f}min", flush=True)
            try:
                os.fsync(fd)
            except OSError:
                pass
        elapsed = time.time() - t0
        print(f"done: {args.model_id} / {bench} — "
              f"{len(pending)} items in {elapsed/60:.1f}min "
              f"({len(pending)/max(elapsed,1e-6):.2f}/s)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
