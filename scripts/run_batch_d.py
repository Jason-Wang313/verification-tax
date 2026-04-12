"""Batch D: 3 models x 5 benchmarks via NIM API.

Models: llama-4-scout, qwen2.5-7b, mistral-small-3.1
Benchmarks (small-first): truthfulqa, arc_challenge, winogrande, hellaswag, mmlu

Features:
  - Round-robin async client pool (18 NIM API keys)
  - Fast-fail retry (3 attempts, 15s timeout)
  - JSONL checkpoint/resume, flush every 10 items
  - Logprob extraction -> softmax -> max_conf
  - Graceful no-logprobs fallback (uniform confidence)
  - Early bail if model fails >90% on first 50 items

Output: data/{benchmark}/results_{model_name}.jsonl
  Each line: {question_id, is_correct, max_conf, first_letter, correct_letter}
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import string
import time
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/wangz/verification tax")

# ─────────────────────────── Models ───────────────────────────
MODELS = [
    ("llama-4-scout", "meta/llama-4-scout-17b-16e-instruct"),
    ("qwen2.5-7b", "qwen/qwen2.5-7b-instruct"),
    ("mistral-small-3.1", "mistralai/mistral-small-3.1-24b-instruct-2503"),
]

# ─────────────────────────── Benchmarks (small -> large) ──────
BENCHMARKS = [
    ("truthfulqa",    PROJECT_ROOT / "data" / "truthfulqa"    / "all_items.json"),
    ("arc_challenge", PROJECT_ROOT / "data" / "arc_challenge" / "all_items.json"),
    ("winogrande",    PROJECT_ROOT / "data" / "winogrande"    / "all_items.json"),
    ("hellaswag",     PROJECT_ROOT / "data" / "hellaswag"     / "all_items.json"),
    ("mmlu",          PROJECT_ROOT / "data" / "mmlu"          / "all_items.json"),
]

# ─────────────────────────── Tuning ───────────────────────────
CONCURRENCY = 64
MAX_TOKENS = 5
TIMEOUT_S = 15
LETTERS = list(string.ascii_uppercase)

# ─────────────────────────── API Client Pool ──────────────────
_NIM_CLIENTS: list = []
_NIM_IDX = [0]


def _load_keys(prefix: str) -> list[str]:
    env = Path("C:/Users/wangz/MIRROR/.env").read_text(encoding="utf-8")
    return [
        line.split("=", 1)[1].strip()
        for line in env.splitlines()
        if line.startswith(prefix) and "=" in line
        and line.split("=", 1)[1].strip()
    ]


def _init_clients() -> None:
    from openai import AsyncOpenAI
    global _NIM_CLIENTS
    keys = _load_keys("NVIDIA_NIM_API_KEY")
    if not keys:
        raise RuntimeError("No NVIDIA_NIM_API_KEY found in MIRROR/.env")
    _NIM_CLIENTS = [
        AsyncOpenAI(api_key=k, base_url="https://integrate.api.nvidia.com/v1")
        for k in keys
    ]
    print(f"  Initialized {len(_NIM_CLIENTS)} NIM clients", flush=True)


async def _next_nim_client():
    idx = _NIM_IDX[0] % len(_NIM_CLIENTS)
    _NIM_IDX[0] += 1
    return _NIM_CLIENTS[idx]


# ─────────────────────────── Prompt Builders ──────────────────

def _build_prompt_mcq(question: str, choices: list[str], letters: list[str]) -> str:
    """Generic MCQ prompt: Question + lettered choices."""
    lines = [f"Question: {question}"]
    for letter, choice in zip(letters, choices):
        lines.append(f"{letter}) {choice}")
    lines.append("")
    lines.append("Answer with just the letter.")
    return "\n".join(lines)


def _build_prompt_winogrande(sentence: str, option1: str, option2: str) -> str:
    return (
        "Fill in the blank:\n"
        f"{sentence}\n"
        f"A) {option1}\n"
        f"B) {option2}\n\n"
        "Answer with just the letter (A or B)."
    )


def _build_prompt_hellaswag(ctx: str, endings: list[str]) -> str:
    return (
        f"Context: {ctx}\n\n"
        "Which ending is most plausible?\n"
        f"A) {endings[0]}\n"
        f"B) {endings[1]}\n"
        f"C) {endings[2]}\n"
        f"D) {endings[3]}\n\n"
        "Answer with just the letter."
    )


def build_prompt(benchmark: str, item: dict) -> tuple[str, list[str]]:
    """Return (prompt_text, valid_letters) for the given benchmark + item."""
    if benchmark == "mmlu":
        choices = item["choices"]
        letters = LETTERS[:len(choices)]
        prompt = _build_prompt_mcq(item["question"], choices, letters)
        return prompt, letters

    elif benchmark == "truthfulqa":
        choices = item["choices"]
        letters = LETTERS[:len(choices)]
        prompt = _build_prompt_mcq(item["question"], choices, letters)
        return prompt, letters

    elif benchmark == "arc_challenge":
        choices = item["choices"]
        labels = item.get("labels", LETTERS[:len(choices)])
        prompt = _build_prompt_mcq(item["question"], choices, labels)
        return prompt, labels

    elif benchmark == "hellaswag":
        prompt = _build_prompt_hellaswag(item["ctx"], item["endings"])
        return prompt, list("ABCD")

    elif benchmark == "winogrande":
        prompt = _build_prompt_winogrande(
            item["sentence"], item["option1"], item["option2"]
        )
        return prompt, list("AB")

    raise ValueError(f"Unknown benchmark: {benchmark}")


def get_correct_letter(benchmark: str, item: dict) -> str:
    """Return the correct answer letter for the given benchmark + item."""
    if benchmark == "mmlu":
        return chr(65 + item["answer"])
    elif benchmark == "truthfulqa":
        return item["correct_letter"]
    elif benchmark == "arc_challenge":
        return item["correct_letter"]
    elif benchmark == "hellaswag":
        return item["correct_letter"]
    elif benchmark == "winogrande":
        return item["correct_letter"]
    raise ValueError(f"Unknown benchmark: {benchmark}")


# ─────────────────────────── Logprob Extraction ───────────────

def _extract_choice_logprobs(top_logprobs_list, valid_letters: set) -> dict:
    """Extract logprobs for choice letters from top_logprobs list."""
    out = {letter: -1e9 for letter in valid_letters}
    for alt in top_logprobs_list:
        tok = alt.token.strip().upper()
        matched_letter = None
        if tok in valid_letters:
            matched_letter = tok
        elif len(tok) >= 2 and tok[-1] in valid_letters and tok[:-1] in ("(", "'", '"', "."):
            matched_letter = tok[-1]
        elif len(tok) >= 1 and tok[0] in valid_letters and (len(tok) == 1 or tok[1] in ").' \"."):
            matched_letter = tok[0]
        if matched_letter and matched_letter in valid_letters:
            out[matched_letter] = max(out[matched_letter], alt.logprob)
    return out


def _softmax(lps: dict) -> dict:
    """Compute softmax over logprob dict."""
    letters = sorted(lps.keys())
    vals = [lps[k] for k in letters]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps) or 1.0
    return {k: round(e / s, 6) for k, e in zip(letters, exps)}


# ─────────────────────────── Core Scorer ──────────────────────

async def _score_one(
    model_id: str, benchmark: str, item: dict
) -> dict:
    """Run one item through one model. Returns result dict."""
    client = await _next_nim_client()
    prompt, valid_letters_list = build_prompt(benchmark, item)
    valid_letters = set(valid_letters_list)
    correct_letter = get_correct_letter(benchmark, item)
    num_choices = len(valid_letters)

    for attempt in range(3):
        try:
            r = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=20,
                ),
                timeout=TIMEOUT_S,
            )
            text = r.choices[0].message.content or ""
            first_letter = next(
                (c for c in text.upper() if c in valid_letters), "?"
            )

            # Try to extract logprobs
            content = r.choices[0].logprobs.content if r.choices[0].logprobs else None
            if content:
                first_tok = content[0]
                top = first_tok.top_logprobs or []
                choice_lps = _extract_choice_logprobs(top, valid_letters)
                sm = _softmax(choice_lps)
                if first_letter == "?" and sm:
                    first_letter = max(sm, key=sm.get)
                return {
                    "first_letter": first_letter,
                    "correct_letter": correct_letter,
                    "is_correct": first_letter == correct_letter,
                    "max_conf": max(sm.values()),
                }
            else:
                # No logprobs available -- use uniform confidence
                if first_letter == "?":
                    for c in text.strip():
                        if c.upper() in valid_letters:
                            first_letter = c.upper()
                            break
                return {
                    "first_letter": first_letter,
                    "correct_letter": correct_letter,
                    "is_correct": first_letter == correct_letter,
                    "max_conf": round(1.0 / num_choices, 6),
                    "no_logprobs": True,
                }

        except asyncio.TimeoutError:
            if attempt < 2:
                continue
            return {
                "error": f"timeout_{TIMEOUT_S}s",
                "first_letter": "?",
                "correct_letter": correct_letter,
            }
        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "rate limit" in msg:
                if attempt < 2:
                    await asyncio.sleep(0.5)
                    continue
            if any(t in msg for t in ("500", "502", "503")):
                if attempt < 2:
                    await asyncio.sleep(0.5)
                    continue
            return {
                "error": str(exc)[:200],
                "first_letter": "?",
                "correct_letter": correct_letter,
            }
    return {
        "error": "max_retries",
        "first_letter": "?",
        "correct_letter": correct_letter,
    }


# ─────────────────────────── Model+Benchmark Runner ──────────

async def run_benchmark(
    model_name: str,
    model_id: str,
    benchmark: str,
    items: list[dict],
) -> dict:
    """Run all items for one (model, benchmark) pair.
    Returns summary dict: {n_valid, n_total, eps, mean_conf, skipped}.
    """
    data_dir = PROJECT_ROOT / "data" / benchmark
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"results_{model_name}.jsonl"

    # -- Resume from checkpoint --
    completed_ids = set()
    if output_path.exists():
        for line in open(output_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if not r.get("error"):
                    completed_ids.add(r["question_id"])
            except json.JSONDecodeError:
                pass

    pending = [it for it in items if it["question_id"] not in completed_ids]
    total = len(items)
    print(f"  [{model_name}][{benchmark}] {len(completed_ids)} done, "
          f"{len(pending)} pending out of {total}", flush=True)

    if not pending:
        return _compute_summary(output_path, total)

    # -- Early-bail probe: test first 50 items --
    if len(completed_ids) == 0 and len(pending) >= 50:
        probe_items = pending[:50]
        probe_results = await asyncio.gather(
            *[_score_one(model_id, benchmark, it) for it in probe_items]
        )
        probe_errors = sum(1 for r in probe_results if r.get("error"))
        if probe_errors > 45:  # >90% errors
            print(f"  [{model_name}][{benchmark}] SKIPPED: {probe_errors}/50 "
                  f"probe items failed (>90%)", flush=True)
            return {"n_valid": 0, "n_total": total, "eps": None,
                    "mean_conf": None, "skipped": True,
                    "skip_reason": f"probe_fail_{probe_errors}/50"}
        # Save successful probe results
        with output_path.open("a", encoding="utf-8") as f:
            for it, res in zip(probe_items, probe_results):
                rec = {"question_id": it["question_id"], **res}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if not res.get("error"):
                    completed_ids.add(it["question_id"])
        pending = [it for it in items if it["question_id"] not in completed_ids]
        print(f"    Probe OK: {probe_errors}/50 errors. "
              f"Continuing with {len(pending)} remaining.", flush=True)

    if not pending:
        return _compute_summary(output_path, total)

    # -- Worker pool --
    queue: asyncio.Queue = asyncio.Queue()
    for it in pending:
        queue.put_nowait(it)
    write_queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    semaphore = asyncio.Semaphore(CONCURRENCY)
    counter = {"done": len(completed_ids), "errors": 0}
    start = time.time()

    async def _process(item: dict) -> None:
        async with semaphore:
            result = await _score_one(model_id, benchmark, item)
        rec = {"question_id": item["question_id"], **result}
        await write_queue.put(rec)

    async def _flusher():
        buffer = []
        last_print = 0
        while True:
            try:
                item = await asyncio.wait_for(write_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if buffer:
                    with output_path.open("a", encoding="utf-8") as f:
                        for r in buffer:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    counter["done"] += len(buffer)
                    counter["errors"] += sum(
                        1 for r in buffer if r.get("error")
                    )
                    if counter["done"] - last_print >= 100:
                        elapsed = (time.time() - start) / 60
                        new_done = counter["done"] - len(completed_ids)
                        rate = new_done / max(elapsed, 0.01)
                        remain = total - counter["done"]
                        eta = remain / max(rate, 0.1)
                        print(
                            f"    [{model_name}][{benchmark}] "
                            f"{counter['done']}/{total} "
                            f"({counter['done']*100//total}%) "
                            f"err={counter['errors']} "
                            f"rate={rate:.0f}/min eta={eta:.1f}min",
                            flush=True,
                        )
                        last_print = counter["done"]
                    buffer = []
                continue
            if item is DONE:
                if buffer:
                    with output_path.open("a", encoding="utf-8") as f:
                        for r in buffer:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    counter["done"] += len(buffer)
                return
            buffer.append(item)
            if len(buffer) >= 10:
                with output_path.open("a", encoding="utf-8") as f:
                    for r in buffer:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                counter["done"] += len(buffer)
                counter["errors"] += sum(
                    1 for r in buffer if r.get("error")
                )
                if counter["done"] - last_print >= 100:
                    elapsed = (time.time() - start) / 60
                    new_done = counter["done"] - len(completed_ids)
                    rate = new_done / max(elapsed, 0.01)
                    remain = total - counter["done"]
                    eta = remain / max(rate, 0.1)
                    print(
                        f"    [{model_name}][{benchmark}] "
                        f"{counter['done']}/{total} "
                        f"({counter['done']*100//total}%) "
                        f"err={counter['errors']} "
                        f"rate={rate:.0f}/min eta={eta:.1f}min",
                        flush=True,
                    )
                    last_print = counter["done"]
                buffer = []

    async def _worker():
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await _process(item)
            except Exception as exc:
                print(f"    [{model_name}][{benchmark}] worker exc: {exc}",
                      flush=True)
            finally:
                queue.task_done()

    flusher_task = asyncio.create_task(_flusher())
    await asyncio.gather(*[_worker() for _ in range(CONCURRENCY)])
    await write_queue.put(DONE)
    await flusher_task

    elapsed = (time.time() - start) / 60
    print(
        f"  [{model_name}][{benchmark}] COMPLETE: "
        f"{counter['done']}/{total} errors={counter['errors']} "
        f"runtime={elapsed:.1f}min",
        flush=True,
    )

    return _compute_summary(output_path, total)


def _compute_summary(output_path: Path, total: int) -> dict:
    """Compute summary stats from a results JSONL file."""
    if not output_path.exists():
        return {"n_valid": 0, "n_total": total, "eps": None,
                "mean_conf": None, "skipped": False}

    records = []
    for line in open(output_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    valid = [r for r in records if not r.get("error") and r.get("max_conf") is not None]
    if not valid:
        return {"n_valid": 0, "n_total": total, "eps": None,
                "mean_conf": None, "skipped": False}

    n_correct = sum(1 for r in valid if r.get("is_correct"))
    eps = 1 - n_correct / len(valid)
    confs = [r["max_conf"] for r in valid]
    mean_conf = sum(confs) / len(confs)
    n_no_logprobs = sum(1 for r in valid if r.get("no_logprobs"))

    return {
        "n_valid": len(valid),
        "n_total": total,
        "eps": round(eps, 4),
        "mean_conf": round(mean_conf, 4),
        "skipped": False,
        "n_no_logprobs": n_no_logprobs,
    }


# ─────────────────────────── Main ─────────────────────────────

async def main() -> None:
    print("=" * 72)
    print("BATCH D: 3 models x 5 benchmarks")
    print("=" * 72)
    _init_clients()

    # Load all benchmark items
    bench_items: dict[str, list[dict]] = {}
    for bname, bpath in BENCHMARKS:
        items = json.loads(bpath.read_text(encoding="utf-8"))
        bench_items[bname] = items
        print(f"  Loaded {bname}: {len(items)} items", flush=True)

    # Results table: (model_name, benchmark) -> summary
    results_table: dict[tuple[str, str], dict] = {}
    grand_start = time.time()

    # Process models sequentially
    for mi, (model_name, model_id) in enumerate(MODELS):
        print(f"\n{'='*72}")
        print(f"MODEL {mi+1}/{len(MODELS)}: {model_name} ({model_id})")
        print(f"{'='*72}")

        for bname, _ in BENCHMARKS:
            items = bench_items[bname]
            print(f"\n--- {model_name} / {bname} ({len(items)} items) ---",
                  flush=True)
            summary = await run_benchmark(model_name, model_id, bname, items)
            results_table[(model_name, bname)] = summary

            if summary.get("skipped"):
                print(f"  -> SKIPPED: {summary.get('skip_reason', '?')}")
            elif summary["n_valid"] > 0:
                print(f"  -> n_valid={summary['n_valid']}/{summary['n_total']} "
                      f"eps={summary['eps']:.4f} "
                      f"mean_conf={summary['mean_conf']:.4f}")
                if summary.get("n_no_logprobs", 0) > 0:
                    print(f"     (no_logprobs: {summary['n_no_logprobs']})")
            else:
                print(f"  -> No valid results")

    # ─────────────────────────── Final Summary ────────────────
    total_elapsed = (time.time() - grand_start) / 60
    print(f"\n\n{'='*72}")
    print(f"FINAL SUMMARY  (total runtime: {total_elapsed:.1f} min)")
    print(f"{'='*72}")

    # Header
    bench_names = [b[0] for b in BENCHMARKS]
    header = f"{'Model':<25s}"
    for bn in bench_names:
        header += f" | {bn:>14s}"
    print(header)
    print("-" * len(header))

    for model_name, _ in MODELS:
        row = f"{model_name:<25s}"
        for bn in bench_names:
            s = results_table.get((model_name, bn))
            if s is None:
                row += f" | {'---':>14s}"
            elif s.get("skipped"):
                row += f" | {'SKIP':>14s}"
            elif s["n_valid"] == 0:
                row += f" | {'FAIL':>14s}"
            else:
                eps_str = f"{s['eps']:.3f}"
                row += f" | {s['n_valid']:>5d} e={eps_str}"
        print(row)

    # Detailed stats
    print(f"\n--- Detailed ---")
    print(f"{'Model':<25s} {'Benchmark':<15s} {'N_valid':>7s} {'N_total':>7s} "
          f"{'eps':>7s} {'mean_conf':>10s} {'no_lp':>6s}")
    print("-" * 85)
    for model_name, _ in MODELS:
        for bn in bench_names:
            s = results_table.get((model_name, bn))
            if s is None:
                continue
            eps_s = f"{s['eps']:.4f}" if s['eps'] is not None else "---"
            mc_s = f"{s['mean_conf']:.4f}" if s['mean_conf'] is not None else "---"
            nlp = str(s.get("n_no_logprobs", 0))
            skip = " SKIP" if s.get("skipped") else ""
            print(f"{model_name:<25s} {bn:<15s} {s['n_valid']:>7d} "
                  f"{s['n_total']:>7d} {eps_s:>7s} {mc_s:>10s} {nlp:>6s}{skip}")

    # Count completed pairs
    n_completed = sum(
        1 for s in results_table.values()
        if s and s["n_valid"] > 0 and not s.get("skipped")
    )
    n_total_pairs = len(MODELS) * len(BENCHMARKS)
    print(f"\nCompleted: {n_completed}/{n_total_pairs} (model, benchmark) pairs")
    print(f"Total runtime: {total_elapsed:.1f} minutes")


if __name__ == "__main__":
    asyncio.run(main())
