"""Local-model verification sweep via llama.cpp (CPU only, no provider API).

Mirrors run_mmlu_experiment.py's JSONL schema so the existing analysis scripts
pick up local-model traces without code changes. Supports MMLU, TruthfulQA,
and ARC-Challenge with a multiple-choice {A,B,C,D} confidence definition.

Usage:
    python scripts/run_local_experiment.py \
        --model-path models/llama-3-8b-instruct-q4_k_m.gguf \
        --model-id llama-3-8b-instruct \
        --benchmark mmlu \
        --n 1000

Output: data/{benchmark}/results_{model_id}.jsonl with one line per item.
Checkpoint/resume: re-running skips items already in the file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_CONFIG = {
    "mmlu": {
        "items_file": PROJECT_ROOT / "data" / "mmlu" / "all_items.json",
        "output_dir": PROJECT_ROOT / "data" / "mmlu",
        "choices": "ABCD",
    },
    "truthfulqa": {
        "items_file": PROJECT_ROOT / "data" / "truthfulqa" / "all_items.json",
        "output_dir": PROJECT_ROOT / "data" / "truthfulqa",
        "choices": "ABCD",  # cap at 4 to match the frozen confidence def; items >4 choices are skipped
    },
    "arc_challenge": {
        "items_file": PROJECT_ROOT / "data" / "arc_challenge" / "all_items.json",
        "output_dir": PROJECT_ROOT / "data" / "arc_challenge",
        "choices": "ABCD",  # cap at 4; ARC items with >4 choices skipped
    },
    "bbq": {
        "items_file": PROJECT_ROOT / "data" / "bbq" / "all_items.json",
        "output_dir": PROJECT_ROOT / "data" / "bbq",
        "choices": "ABC",
    },
    "xstest": {
        "items_file": PROJECT_ROOT / "data" / "xstest" / "all_items.json",
        "output_dir": PROJECT_ROOT / "data" / "xstest",
        "choices": "AB",  # A=refuse, B=answer
    },
    "toxigen": {
        "items_file": PROJECT_ROOT / "data" / "toxigen" / "all_items.json",
        "output_dir": PROJECT_ROOT / "data" / "toxigen",
        "choices": "AB",  # A=toxic, B=not toxic
    },
}


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _item_choices(item: dict, max_choices: int) -> str:
    """Return the choice letters for this item, e.g. 'ABC' or 'ABCD'."""
    n = min(len(item.get("choices", [])), max_choices, len(ALPHABET))
    return ALPHABET[:n] if n > 0 else "ABCD"


def _build_prompt(item: dict, max_choices: int = 4) -> str:
    q = item["question"]
    c = item.get("choices", [])
    letters = _item_choices(item, max_choices)
    lines = [
        "Answer the following multiple choice question. Reply with ONLY the letter.",
        "",
        f"Question: {q}",
    ]
    for i, choice in enumerate(c[: len(letters)]):
        lines.append(f"{letters[i]}) {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def _softmax(values):
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _completed_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    ids: set[str] = set()
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["question_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def _letter_token_ids(llm, choices: str) -> dict[str, int]:
    """Resolve each choice letter to a single token id (prefer ' A' leading-space variant)."""
    ids: dict[str, int] = {}
    for letter in choices:
        picked = None
        for variant in (f" {letter}", letter):
            t = llm.tokenize(variant.encode(), add_bos=False)
            if len(t) == 1:
                picked = t[0]
                break
        if picked is None:
            t = llm.tokenize(letter.encode(), add_bos=False)
            picked = t[-1]
        ids[letter] = picked
    return ids


def _extract_correct_letter(item: dict, choices: str) -> str:
    """Normalize: some benchmarks carry 'answer' (int idx), others 'correct_letter' or 'correct_idx'."""
    if "answer" in item:
        try:
            idx = int(item["answer"])
            if 0 <= idx < len(choices):
                return choices[idx]
        except (TypeError, ValueError):
            pass
    if "correct_idx" in item:
        try:
            idx = int(item["correct_idx"])
            if 0 <= idx < len(choices):
                return choices[idx]
        except (TypeError, ValueError):
            pass
    if "correct_letter" in item:
        letter = str(item["correct_letter"]).strip().upper()
        if letter and letter[0] in choices:
            return letter[0]
    return "?"


def _score_one(llm, item: dict, choices: str = "ABCD", letter_ids: dict[str, int] | None = None) -> dict:
    """Fast single-item scoring: one forward pass via llm.eval(), read _scores[-1]."""
    import numpy as np
    # Use the intersection of the benchmark's declared choice set with the
    # item's actual choice count (e.g. TruthfulQA has variable 2–13 options).
    per_item_choices = _item_choices(item, len(choices))
    prompt = _build_prompt(item, max_choices=len(per_item_choices))
    correct_letter = _extract_correct_letter(item, per_item_choices)

    if letter_ids is None:
        letter_ids = _letter_token_ids(llm, choices)
    # Ensure letter_ids covers every letter we need (fallback if caller gave a subset).
    for L in per_item_choices:
        if L not in letter_ids:
            letter_ids = dict(letter_ids)
            letter_ids.update(_letter_token_ids(llm, per_item_choices))
            break

    try:
        tokens = llm.tokenize(prompt.encode(), add_bos=True)
        llm.reset()
        llm.eval(tokens)
        final_logits = np.asarray(llm._scores[-1], dtype=np.float64)
    except Exception as e:  # noqa: BLE001
        return {"error": f"eval_failed:{type(e).__name__}", "message": str(e)[:200]}

    m = float(np.max(final_logits))
    log_sum = m + math.log(float(np.sum(np.exp(final_logits - m))))
    logps: dict[str, float] = {}
    for letter in per_item_choices:
        tid = letter_ids[letter]
        logps[letter] = float(final_logits[tid] - log_sum)

    sm = _softmax([logps[L] for L in per_item_choices])
    softmax_abcd = dict(zip(per_item_choices, sm))
    pred_letter = max(softmax_abcd, key=softmax_abcd.get)
    max_conf = softmax_abcd[pred_letter]
    return {
        "response": pred_letter,
        "first_letter": pred_letter,
        "correct_letter": correct_letter,
        "is_correct": pred_letter == correct_letter,
        "logprobs_abcd": logps,
        "softmax_abcd": softmax_abcd,
        "max_conf": max_conf,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local llama.cpp sweep.")
    parser.add_argument("--model-path", type=Path, required=True, help="GGUF model path.")
    parser.add_argument("--model-id", type=str, required=True, help="Short slug used in output filename.")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARK_CONFIG), required=True)
    parser.add_argument("--n", type=int, default=None, help="Cap item count (for smoke tests).")
    parser.add_argument("--n-ctx", type=int, default=1024, help="llama.cpp context length (default 1024; fits all MCQ benchmarks).")
    parser.add_argument("--n-threads", type=int, default=0, help="CPU threads (0 = physical core count).")
    args = parser.parse_args(argv)

    cfg = BENCHMARK_CONFIG[args.benchmark]
    items_file: Path = cfg["items_file"]
    out_dir: Path = cfg["output_dir"]
    choices: str = cfg["choices"]

    if not items_file.exists():
        print(f"error: benchmark items missing: {items_file}", file=sys.stderr)
        return 2
    if not args.model_path.exists():
        print(f"error: model not found: {args.model_path}", file=sys.stderr)
        return 2

    items = json.loads(items_file.read_text(encoding="utf-8"))
    if args.n:
        items = items[: args.n]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{args.model_id}.jsonl"
    completed = _completed_ids(out_path)
    pending = [it for it in items if it["question_id"] not in completed]
    print(f"{args.model_id} / {args.benchmark}: {len(items)} items total, "
          f"{len(completed)} done, {len(pending)} pending.", flush=True)

    if not pending:
        return 0

    from llama_cpp import Llama
    print(f"  Loading {args.model_path} (n_ctx={args.n_ctx}, threads={args.n_threads or 'auto'})", flush=True)
    # Use physical-core count when --n-threads=0 (auto) for best llama.cpp throughput.
    import os
    n_threads = args.n_threads or os.cpu_count() or 4
    llm = Llama(
        model_path=str(args.model_path),
        n_ctx=args.n_ctx,
        n_threads=n_threads,
        n_batch=2048,
        n_ubatch=1024,
        use_mmap=True,
        use_mlock=False,
        logits_all=False,
        verbose=False,
    )
    letter_ids = _letter_token_ids(llm, choices)

    t0 = time.time()
    import os as _os
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
            # fsync every 10 items to bound data loss on a hard crash without
            # paying the syscall on every line.
            if (i + 1) % 10 == 0:
                try:
                    _os.fsync(fd)
                except OSError:
                    pass
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(pending) - i - 1) / max(rate, 1e-6) / 60
                print(f"  [{args.model_id}] {i+1}/{len(pending)} @ {rate:.2f}/s "
                      f"ETA {eta:.1f}min", flush=True)
        try:
            _os.fsync(fd)
        except OSError:
            pass

    print(f"done: {args.model_id} / {args.benchmark}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
