"""Batch C: Evaluate 3 models on all 5 benchmarks via NIM API.

Models:
  - phi-4-mini (microsoft/phi-4-mini-instruct)
  - deepseek-r1-qwen-32b (deepseek-ai/deepseek-r1-distill-qwen-32b)
  - nemotron-70b (nvidia/llama-3.1-nemotron-70b-instruct)

Benchmarks (small-first):
  1. truthfulqa      (817 items)
  2. arc_challenge    (1172 items)
  3. winogrande       (1267 items)
  4. hellaswag        (10042 items)
  5. mmlu             (14042 items)

Output: data/{benchmark}/results_{model_name}.jsonl
Format: {"question_id", "is_correct", "max_conf", "first_letter", "correct_letter"}

Crash-resistant: JSONL checkpoint, skip completed question_ids, flush every 10.
Early abort: >90% errors in first 50 items => skip that benchmark/model combo.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import string
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    ("phi-4-mini", "microsoft/phi-4-mini-instruct"),
    ("deepseek-r1-qwen-32b", "deepseek-ai/deepseek-r1-distill-qwen-32b"),
    ("nemotron-70b", "nvidia/llama-3.1-nemotron-70b-instruct"),
]

BENCHMARKS = [
    "truthfulqa",
    "arc_challenge",
    "winogrande",
    "hellaswag",
    "mmlu",
]

PER_MODEL_CONCURRENCY = 64
NUM_WORKERS = 96
MAX_TOKENS = 5
TIMEOUT_S = 15

LETTERS = list(string.ascii_uppercase)


# ---------------------------------------------------------------------------
# API key loading + client pool
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
# Prompt builders per benchmark
# ---------------------------------------------------------------------------

def _build_prompt_mmlu(item: dict) -> tuple[str, str, set[str]]:
    """Returns (prompt, correct_letter, valid_letters)."""
    q = item["question"]
    c = item["choices"]
    letters_used = LETTERS[:len(c)]
    lines = [f"Question: {q}"]
    for i, choice in enumerate(c):
        lines.append(f"{letters_used[i]}) {choice}")
    lines.append("")
    lines.append("Answer with just the letter.")
    correct_letter = chr(65 + item["answer"])
    return "\n".join(lines), correct_letter, set(letters_used)


def _build_prompt_truthfulqa(item: dict) -> tuple[str, str, set[str]]:
    q = item["question"]
    choices = item["choices"]
    num = item.get("num_choices", len(choices))
    letters_used = LETTERS[:num]
    lines = [f"Question: {q}"]
    for i, choice in enumerate(choices):
        lines.append(f"{letters_used[i]}) {choice}")
    lines.append("")
    lines.append("Answer with just the letter.")
    correct_letter = item["correct_letter"]
    return "\n".join(lines), correct_letter, set(letters_used)


def _build_prompt_arc(item: dict) -> tuple[str, str, set[str]]:
    q = item["question"]
    labels = item["labels"]
    choices = item["choices"]
    lines = [f"Question: {q}"]
    for label, choice in zip(labels, choices):
        lines.append(f"{label}) {choice}")
    lines.append("")
    lines.append("Answer with just the letter.")
    correct_letter = item["correct_letter"]
    return "\n".join(lines), correct_letter, set(labels)


def _build_prompt_hellaswag(item: dict) -> tuple[str, str, set[str]]:
    ctx = item["ctx"]
    endings = item["endings"]
    letters_used = LETTERS[:len(endings)]
    lines = [f"Context: {ctx}", "",
             "Which ending is most plausible?"]
    for i, ending in enumerate(endings):
        lines.append(f"{letters_used[i]}) {ending}")
    lines.append("")
    lines.append("Answer with just the letter.")
    correct_letter = item["correct_letter"]
    return "\n".join(lines), correct_letter, set(letters_used)


def _build_prompt_winogrande(item: dict) -> tuple[str, str, set[str]]:
    sentence = item["sentence"]
    option1 = item["option1"]
    option2 = item["option2"]
    prompt = (
        "Fill in the blank:\n"
        f"{sentence}\n"
        f"A) {option1}\n"
        f"B) {option2}\n\n"
        "Answer with just the letter (A or B)."
    )
    correct_letter = item["correct_letter"]
    return prompt, correct_letter, {"A", "B"}


PROMPT_BUILDERS = {
    "mmlu": _build_prompt_mmlu,
    "truthfulqa": _build_prompt_truthfulqa,
    "arc_challenge": _build_prompt_arc,
    "hellaswag": _build_prompt_hellaswag,
    "winogrande": _build_prompt_winogrande,
}


# ---------------------------------------------------------------------------
# Logprob extraction + softmax
# ---------------------------------------------------------------------------

def _extract_logprobs(top_logprobs_list, valid_letters: set[str]) -> dict:
    """Extract logprobs for valid choice letters from top_logprobs list."""
    out = {letter: -1e9 for letter in valid_letters}
    for alt in top_logprobs_list:
        tok = alt.token.strip().upper()
        matched = None
        if tok in valid_letters:
            matched = tok
        elif len(tok) >= 2 and tok[-1] in valid_letters and tok[:-1] in ("(", "'", '"', "."):
            matched = tok[-1]
        elif len(tok) >= 1 and tok[0] in valid_letters and (len(tok) == 1 or tok[1] in ").' \"."):
            matched = tok[0]
        if matched and matched in valid_letters:
            out[matched] = max(out[matched], alt.logprob)
    return out


def _softmax(lps: dict) -> dict:
    letters = sorted(lps.keys())
    vals = [lps[k] for k in letters]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps) or 1.0
    return {k: round(e / s, 6) for k, e in zip(letters, exps)}


# ---------------------------------------------------------------------------
# Score one item
# ---------------------------------------------------------------------------

async def _score_one(
    model_id: str,
    benchmark: str,
    item: dict,
) -> dict:
    """Run one item through one model. Returns compact result dict."""
    client = await _next_nim_client()
    builder = PROMPT_BUILDERS[benchmark]
    prompt, correct_letter, valid_letters = builder(item)
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
            content = (
                r.choices[0].logprobs.content
                if r.choices[0].logprobs else None
            )
            if content:
                top = content[0].top_logprobs or []
                lps = _extract_logprobs(top, valid_letters)
                sm = _softmax(lps)
                max_conf = max(sm.values())
                # If text parse failed, use argmax of softmax
                if first_letter == "?" and sm:
                    first_letter = max(sm, key=sm.get)
            else:
                # No logprobs available -> uniform confidence
                max_conf = round(1.0 / num_choices, 6)

            return {
                "question_id": item["question_id"],
                "is_correct": first_letter == correct_letter,
                "max_conf": max_conf,
                "first_letter": first_letter,
                "correct_letter": correct_letter,
            }

        except asyncio.TimeoutError:
            if attempt < 2:
                continue
            return {
                "question_id": item["question_id"],
                "error": f"timeout_{TIMEOUT_S}s",
                "is_correct": False,
                "max_conf": round(1.0 / num_choices, 6),
                "first_letter": "?",
                "correct_letter": correct_letter,
            }
        except Exception as exc:
            msg = str(exc).lower()
            retryable = (
                "429" in msg or "rate limit" in msg
                or any(t in msg for t in ("500", "502", "503"))
            )
            if retryable and attempt < 2:
                await asyncio.sleep(0.5)
                continue
            return {
                "question_id": item["question_id"],
                "error": str(exc)[:120],
                "is_correct": False,
                "max_conf": round(1.0 / num_choices, 6),
                "first_letter": "?",
                "correct_letter": correct_letter,
            }

    return {
        "question_id": item["question_id"],
        "error": "max_retries",
        "is_correct": False,
        "max_conf": round(1.0 / num_choices, 6),
        "first_letter": "?",
        "correct_letter": correct_letter,
    }


# ---------------------------------------------------------------------------
# Run one model on one benchmark
# ---------------------------------------------------------------------------

async def run_model_benchmark(
    model_name: str,
    model_id: str,
    benchmark: str,
    items: list[dict],
) -> bool:
    """Run all items for one model/benchmark combo.

    Returns True if completed (or skipped), False if early-aborted.
    """
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
    total = len(items)
    print(f"  [{model_name}/{benchmark}] {len(completed_ids)} done, "
          f"{len(pending)} pending of {total}", flush=True)

    if not pending:
        return True

    # --- Worker pool ---
    queue: asyncio.Queue = asyncio.Queue()
    for it in pending:
        queue.put_nowait(it)
    write_queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    semaphore = asyncio.Semaphore(PER_MODEL_CONCURRENCY)
    counter = {"done": len(completed_ids), "errors": 0, "early_items": 0,
               "early_errors": 0, "abort": False}
    start = time.time()

    async def _process(item: dict) -> None:
        async with semaphore:
            result = await _score_one(model_id, benchmark, item)
        await write_queue.put(result)

    async def _flusher():
        buffer: list[dict] = []
        last_print = 0
        while True:
            try:
                item = await asyncio.wait_for(write_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if buffer:
                    _flush(buffer, output_path, counter, completed_ids,
                           start, total, model_name, benchmark)
                    last_print = counter["done"]
                    buffer = []
                continue
            if item is DONE:
                if buffer:
                    _flush(buffer, output_path, counter, completed_ids,
                           start, total, model_name, benchmark)
                return
            buffer.append(item)
            if len(buffer) >= 10:
                _flush(buffer, output_path, counter, completed_ids,
                       start, total, model_name, benchmark)
                # Progress printing
                if counter["done"] - last_print >= 100:
                    elapsed = (time.time() - start) / 60
                    rate = (counter["done"] - len(completed_ids)) / max(elapsed, 0.01)
                    eta = (total - counter["done"]) / max(rate, 0.1)
                    print(f"    [{model_name}/{benchmark}] "
                          f"{counter['done']}/{total} "
                          f"({counter['done']*100//total}%) "
                          f"err={counter['errors']} "
                          f"rate={rate:.0f}/min eta={eta:.1f}min",
                          flush=True)
                    last_print = counter["done"]
                buffer = []

    async def _worker():
        while True:
            if counter["abort"]:
                return
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await _process(item)
            except Exception as exc:
                print(f"    [{model_name}/{benchmark}] worker exc: {exc}",
                      flush=True)
            finally:
                queue.task_done()

    flusher_task = asyncio.create_task(_flusher())
    await asyncio.gather(*[_worker() for _ in range(NUM_WORKERS)])
    await write_queue.put(DONE)
    await flusher_task

    if counter["abort"]:
        print(f"  [{model_name}/{benchmark}] ABORTED: >90% errors "
              f"in first 50 items ({counter['early_errors']}/{counter['early_items']})",
              flush=True)
        return False

    elapsed = (time.time() - start) / 60
    print(f"  [{model_name}/{benchmark}] COMPLETE: {counter['done']}/{total} "
          f"errors={counter['errors']} runtime={elapsed:.1f}min", flush=True)
    return True


def _flush(
    buffer: list[dict],
    output_path: Path,
    counter: dict,
    completed_ids: set[str],
    start: float,
    total: int,
    model_name: str,
    benchmark: str,
) -> None:
    """Write buffer to JSONL and update counters. Check early-abort."""
    with output_path.open("a", encoding="utf-8") as f:
        for rec in buffer:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_errs = sum(1 for r in buffer if r.get("error"))
    counter["done"] += len(buffer)
    counter["errors"] += n_errs

    # Track first 50 items for early-abort check
    prev_early = counter["early_items"]
    if prev_early < 50:
        for r in buffer:
            if counter["early_items"] >= 50:
                break
            counter["early_items"] += 1
            if r.get("error"):
                counter["early_errors"] += 1

        # Check threshold
        if counter["early_items"] >= 50:
            err_pct = counter["early_errors"] / counter["early_items"]
            if err_pct > 0.90:
                counter["abort"] = True


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_items(benchmark: str) -> list[dict]:
    """Load cached benchmark items from all_items.json."""
    path = PROJECT_ROOT / "data" / benchmark / "all_items.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    items = json.loads(path.read_text(encoding="utf-8"))
    print(f"  Loaded {len(items)} items for {benchmark}", flush=True)
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 70)
    print("BATCH C: 3 MODELS x 5 BENCHMARKS")
    print("=" * 70)
    _init_clients()

    for benchmark in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {benchmark}")
        print(f"{'='*60}")
        items = _load_items(benchmark)

        for model_name, model_id in MODELS:
            print(f"\n--- {model_name} on {benchmark} ---", flush=True)
            await run_model_benchmark(model_name, model_id, benchmark, items)

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("BATCH C SUMMARY")
    print("=" * 70)
    for benchmark in BENCHMARKS:
        print(f"\n  {benchmark}:")
        for model_name, _ in MODELS:
            path = PROJECT_ROOT / "data" / benchmark / f"results_{model_name}.jsonl"
            if not path.exists():
                print(f"    {model_name}: no file")
                continue
            records = [
                json.loads(l)
                for l in open(path, encoding="utf-8")
                if l.strip()
            ]
            valid = [
                r for r in records
                if not r.get("error") and r.get("max_conf") is not None
            ]
            errors = [r for r in records if r.get("error")]
            if not valid:
                print(f"    {model_name}: no valid records "
                      f"({len(records)} total, {len(errors)} errors)")
                continue
            n_correct = sum(1 for r in valid if r.get("is_correct"))
            accuracy = n_correct / len(valid)
            confs = [r["max_conf"] for r in valid]
            mean_conf = sum(confs) / len(confs)
            n_sat = sum(1 for c in confs if c > 0.99)
            print(f"    {model_name}: {len(valid)}/{len(records)} valid, "
                  f"acc={accuracy:.3f}, mean_conf={mean_conf:.3f}, "
                  f"sat(>0.99)={n_sat}/{len(valid)}")


if __name__ == "__main__":
    asyncio.run(main())
