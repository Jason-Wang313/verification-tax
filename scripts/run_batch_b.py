"""Batch B: Evaluate 2 models on all 5 benchmarks via NIM API.

Models:
  - mistral-large-2 (mistralai/mistral-large-2-instruct)
  - gemma-3-27b (google/gemma-3-27b-it)

Benchmarks (small first):
  truthfulqa, arc_challenge, winogrande, hellaswag, mmlu

Output: data/{benchmark}/results_{model_name}.jsonl
  Each line: {question_id, ..., first_letter, correct_letter, is_correct,
              softmax, max_conf, error}

Resume from checkpoint by re-running -- already-completed question_ids skipped.
Flush every 10 items. Skip benchmark if >90% errors in first 50 items.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import string
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    ("mistral-large-2", "mistralai/mistral-large-2-instruct"),
    ("gemma-3-27b", "google/gemma-3-27b-it"),
]

BENCHMARKS = ["truthfulqa", "arc_challenge", "winogrande", "hellaswag", "mmlu"]

PER_MODEL_CONCURRENCY = 64
NUM_WORKERS_PER_MODEL = 96
MAX_TOKENS = 5
TIMEOUT_S = 15
FLUSH_EVERY = 10

LETTERS = list(string.ascii_uppercase)

# ---------------------------------------------------------------------------
# API key loading + NIM client pool
# ---------------------------------------------------------------------------


def _load_keys(prefix: str) -> list[str]:
    env = Path("C:/Users/wangz/MIRROR/.env").read_text(encoding="utf-8")
    return [
        line.split("=", 1)[1].strip()
        for line in env.splitlines()
        if line.startswith(prefix) and "=" in line
        and line.split("=", 1)[1].strip()
    ]


_NIM_CLIENTS: list = []
_NIM_IDX = [0]


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


# ---------------------------------------------------------------------------
# Prompt builders (per-benchmark)
# ---------------------------------------------------------------------------


def _build_prompt_truthfulqa(item: dict) -> str:
    q = item["question"]
    choices = item["choices"]
    lines = [f"Question: {q}"]
    for i, choice in enumerate(choices):
        lines.append(f"{LETTERS[i]}) {choice}")
    lines.append("")
    lines.append("Answer with just the letter.")
    return "\n".join(lines)


def _build_prompt_arc(item: dict) -> str:
    q = item["question"]
    labels = item["labels"]
    choices = item["choices"]
    lines = [f"Question: {q}"]
    for label, choice in zip(labels, choices):
        lines.append(f"{label}) {choice}")
    lines.append("")
    lines.append("Answer with just the letter.")
    return "\n".join(lines)


def _build_prompt_winogrande(item: dict) -> str:
    sentence = item["sentence"]
    option1 = item["option1"]
    option2 = item["option2"]
    return (
        "Fill in the blank:\n"
        f"{sentence}\n"
        f"A) {option1}\n"
        f"B) {option2}\n\n"
        "Answer with just the letter (A or B)."
    )


def _build_prompt_hellaswag(item: dict) -> str:
    ctx = item["ctx"]
    endings = item["endings"]
    lines = [f"Question: {ctx}"]
    for i, ending in enumerate(endings):
        lines.append(f"{LETTERS[i]}) {ending}")
    lines.append("")
    lines.append("Answer with just the letter.")
    return "\n".join(lines)


def _build_prompt_mmlu(item: dict) -> str:
    q = item["question"]
    c = item["choices"]
    return (
        f"Question: {q}\n"
        f"A) {c[0]}\n"
        f"B) {c[1]}\n"
        f"C) {c[2]}\n"
        f"D) {c[3]}\n\n"
        "Answer with just the letter."
    )


PROMPT_BUILDERS = {
    "truthfulqa": _build_prompt_truthfulqa,
    "arc_challenge": _build_prompt_arc,
    "winogrande": _build_prompt_winogrande,
    "hellaswag": _build_prompt_hellaswag,
    "mmlu": _build_prompt_mmlu,
}


# ---------------------------------------------------------------------------
# Valid-letters helpers (per-benchmark)
# ---------------------------------------------------------------------------


def _valid_letters_for(benchmark: str, item: dict) -> set[str]:
    if benchmark == "truthfulqa":
        return set(LETTERS[:item["num_choices"]])
    elif benchmark == "arc_challenge":
        return set(item["labels"])
    elif benchmark == "winogrande":
        return {"A", "B"}
    elif benchmark == "hellaswag":
        return {"A", "B", "C", "D"}
    elif benchmark == "mmlu":
        return {"A", "B", "C", "D"}
    return {"A", "B", "C", "D"}


def _correct_letter_for(benchmark: str, item: dict) -> str:
    if benchmark == "mmlu":
        return chr(65 + item["answer"])
    return item["correct_letter"]


# ---------------------------------------------------------------------------
# Logprob extraction + softmax
# ---------------------------------------------------------------------------


def _extract_choice_logprobs(top_logprobs_list, valid_letters: set) -> dict:
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


def _softmax_choices(lps: dict) -> dict:
    letters = sorted(lps.keys())
    vals = [lps[k] for k in letters]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps) or 1.0
    return {k: round(e / s, 6) for k, e in zip(letters, exps)}


# ---------------------------------------------------------------------------
# Score one item
# ---------------------------------------------------------------------------


async def _score_one(model_id: str, benchmark: str, item: dict) -> dict:
    client = await _next_nim_client()
    prompt = PROMPT_BUILDERS[benchmark](item)
    valid_letters = _valid_letters_for(benchmark, item)
    correct_letter = _correct_letter_for(benchmark, item)
    num_choices = len(valid_letters)

    delay = 1
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
            first_letter = next((c for c in text.upper() if c in valid_letters), "?")
            content = r.choices[0].logprobs.content if r.choices[0].logprobs else None
            if not content:
                # No logprobs: max_conf = 1/num_choices
                return {
                    "response": text,
                    "first_letter": first_letter,
                    "correct_letter": correct_letter,
                    "is_correct": first_letter == correct_letter,
                    "softmax": {},
                    "max_conf": round(1.0 / num_choices, 6),
                    "error": "no_logprobs",
                }
            first = content[0]
            top = first.top_logprobs or []
            choice_lps = _extract_choice_logprobs(top, valid_letters)
            sm = _softmax_choices(choice_lps)
            if first_letter == "?" and sm:
                first_letter = max(sm, key=sm.get)
            return {
                "response": text,
                "first_letter": first_letter,
                "correct_letter": correct_letter,
                "is_correct": first_letter == correct_letter,
                "logprobs": choice_lps,
                "softmax": sm,
                "max_conf": max(sm.values()),
            }
        except asyncio.TimeoutError:
            if attempt < 2:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8)
                continue
            return {"error": f"timeout_{TIMEOUT_S}s"}
        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "rate limit" in msg:
                if attempt < 2:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 8)
                    continue
            if any(t in msg for t in ("500", "502", "503")):
                if attempt < 2:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 8)
                    continue
            return {"error": str(exc)[:120]}
    return {"error": "max_retries"}


# ---------------------------------------------------------------------------
# Run one (model, benchmark) pair
# ---------------------------------------------------------------------------


async def run_model_benchmark(
    model_name: str,
    model_id: str,
    benchmark: str,
    items: list[dict],
) -> None:
    data_dir = PROJECT_ROOT / "data" / benchmark
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"results_{model_name}.jsonl"

    # Resume from checkpoint
    completed_ids: set[str] = set()
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
    tag = f"{model_name}/{benchmark}"
    print(f"  [{tag}] {len(completed_ids)} done, {len(pending)} pending", flush=True)

    if not pending:
        return

    # Worker pool with semaphore + buffered writer
    queue: asyncio.Queue = asyncio.Queue()
    for it in pending:
        queue.put_nowait(it)
    write_queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    semaphore = asyncio.Semaphore(PER_MODEL_CONCURRENCY)
    counter = {"done": len(completed_ids), "errors": 0, "early_errors": 0, "early_total": 0}
    total = len(items)
    start = time.time()
    skipped = {"value": False}

    def _build_record(item: dict, result: dict) -> dict:
        rec = {"question_id": item["question_id"]}
        # Add benchmark-specific context fields
        if benchmark == "truthfulqa":
            rec["question"] = item["question"]
            rec["num_choices"] = item["num_choices"]
        elif benchmark == "arc_challenge":
            pass  # question_id is enough
        elif benchmark == "winogrande":
            rec["sentence"] = item["sentence"][:200]
        elif benchmark == "hellaswag":
            rec["ctx"] = item["ctx"][:200]
        elif benchmark == "mmlu":
            rec["subject"] = item["subject"]
        rec.update(result)
        return rec

    async def _process(item: dict) -> None:
        async with semaphore:
            result = await _score_one(model_id, benchmark, item)
        rec = _build_record(item, result)
        await write_queue.put(rec)

    async def _flusher():
        buffer: list[dict] = []
        last_print = 0
        while True:
            try:
                item = await asyncio.wait_for(write_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if buffer:
                    with output_path.open("a", encoding="utf-8") as f:
                        for r in buffer:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_err = sum(1 for r in buffer if r.get("error"))
                    counter["done"] += len(buffer)
                    counter["errors"] += n_err
                    # Early error check
                    counter["early_total"] += len(buffer)
                    counter["early_errors"] += n_err
                    if counter["done"] - last_print >= 50:
                        elapsed = (time.time() - start) / 60
                        rate = (counter["done"] - len(completed_ids)) / max(elapsed, 0.01)
                        eta = (total - counter["done"]) / max(rate, 0.1)
                        print(
                            f"    [{tag}] {counter['done']}/{total} "
                            f"({counter['done']*100//total}%) "
                            f"err={counter['errors']} rate={rate:.0f}/min eta={eta:.1f}min",
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
            if len(buffer) >= FLUSH_EVERY:
                with output_path.open("a", encoding="utf-8") as f:
                    for r in buffer:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_err = sum(1 for r in buffer if r.get("error"))
                counter["done"] += len(buffer)
                counter["errors"] += n_err
                counter["early_total"] += len(buffer)
                counter["early_errors"] += n_err
                # Check >90% errors in first 50 items
                if counter["early_total"] >= 50 and not skipped["value"]:
                    err_rate = counter["early_errors"] / counter["early_total"]
                    if err_rate > 0.90:
                        print(
                            f"  [{tag}] SKIPPING: {err_rate*100:.0f}% errors in "
                            f"first {counter['early_total']} items",
                            flush=True,
                        )
                        skipped["value"] = True
                if counter["done"] - last_print >= 50:
                    elapsed = (time.time() - start) / 60
                    rate = (counter["done"] - len(completed_ids)) / max(elapsed, 0.01)
                    eta = (total - counter["done"]) / max(rate, 0.1)
                    print(
                        f"    [{tag}] {counter['done']}/{total} "
                        f"({counter['done']*100//total}%) "
                        f"err={counter['errors']} rate={rate:.0f}/min eta={eta:.1f}min",
                        flush=True,
                    )
                    last_print = counter["done"]
                buffer = []

    async def _worker():
        while True:
            if skipped["value"]:
                return
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await _process(item)
            except Exception as exc:
                print(f"    [{tag}] worker exception: {exc}", flush=True)
            finally:
                queue.task_done()

    flusher_task = asyncio.create_task(_flusher())
    await asyncio.gather(*[_worker() for _ in range(NUM_WORKERS_PER_MODEL)])
    await write_queue.put(DONE)
    await flusher_task

    status = "SKIPPED" if skipped["value"] else "COMPLETE"
    print(
        f"  [{tag}] {status}: {counter['done']}/{total} "
        f"errors={counter['errors']} runtime={(time.time()-start)/60:.1f}min",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Item loaders (from cached all_items.json)
# ---------------------------------------------------------------------------


def _load_items(benchmark: str) -> list[dict]:
    cache = PROJECT_ROOT / "data" / benchmark / "all_items.json"
    if not cache.exists():
        raise FileNotFoundError(
            f"Missing {cache}. Run the individual benchmark script first to download data."
        )
    items = json.loads(cache.read_text(encoding="utf-8"))
    print(f"  [{benchmark}] Loaded {len(items)} items from cache", flush=True)
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 70)
    print("BATCH B: 2 MODELS x 5 BENCHMARKS")
    print("=" * 70)
    _init_clients()

    for benchmark in BENCHMARKS:
        items = _load_items(benchmark)
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {benchmark} ({len(items)} items)")
        print(f"{'='*70}")

        # Run both models concurrently on this benchmark
        tasks = [
            run_model_benchmark(name, mid, benchmark, items)
            for name, mid in MODELS
        ]
        await asyncio.gather(*tasks)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, _ in MODELS:
        print(f"\n  {name}:")
        for benchmark in BENCHMARKS:
            path = PROJECT_ROOT / "data" / benchmark / f"results_{name}.jsonl"
            if not path.exists():
                print(f"    {benchmark}: no file")
                continue
            records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            valid = [r for r in records if not r.get("error") and r.get("max_conf") is not None]
            errors = [r for r in records if r.get("error")]
            if not valid:
                print(
                    f"    {benchmark}: no valid records "
                    f"({len(records)} total, {len(errors)} errors)"
                )
                continue
            n_correct = sum(1 for r in valid if r.get("is_correct"))
            accuracy = n_correct / len(valid)
            confs = [r["max_conf"] for r in valid]
            mean_conf = sum(confs) / len(confs)
            n_sat = sum(1 for c in confs if c > 0.99)
            print(
                f"    {benchmark}: {len(valid)}/{len(records)} valid, "
                f"{len(errors)} err | acc={accuracy:.3f} "
                f"mean_conf={mean_conf:.3f} sat={n_sat}/{len(valid)}"
            )


if __name__ == "__main__":
    asyncio.run(main())
