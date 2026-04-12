"""WinoGrande real-model verification experiment.

Runs the WinoGrande (winogrande_xl) validation set (~1,267 items) on 3
NIM-hosted LLMs, extracts logprob-based confidence scores (softmax over
A/B logprobs at the first generated position), and saves incrementally
for crash-resistance.

Output: data/winogrande/results_{model_name}.jsonl
  Each line: {question_id, sentence, response, first_letter, correct_letter,
              logprobs_ab, softmax_ab, max_conf, is_correct, error}

Resume from checkpoint by re-running — already-completed question_ids skipped.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "winogrande"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("llama-3.1-405b-instruct", "meta/llama-3.1-405b-instruct"),
    ("llama-4-maverick", "meta/llama-4-maverick-17b-128e-instruct"),
    ("qwen3-next-80b", "qwen/qwen3-next-80b-a3b-instruct"),
]

# Small dataset (~1,267 items) — moderate concurrency
PER_MODEL_CONCURRENCY = 32
NUM_WORKERS_PER_MODEL = 48
MAX_TOKENS = 5
TIMEOUT_S = 15

LETTERS = "AB"


def _load_keys(prefix: str) -> list[str]:
    env = Path("C:/Users/wangz/MIRROR/.env").read_text(encoding="utf-8")
    return [
        line.split("=", 1)[1].strip()
        for line in env.splitlines()
        if line.startswith(prefix) and "=" in line
        and line.split("=", 1)[1].strip()
    ]


# --- NIM client pool with round-robin ---
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


def _build_prompt(item: dict) -> str:
    sentence = item["sentence"]
    option1 = item["option1"]
    option2 = item["option2"]
    return (
        "Fill in the blank with the correct option.\n\n"
        f"Sentence: {sentence}\n"
        f"A) {option1}\n"
        f"B) {option2}\n\n"
        "Answer with just the letter (A or B)."
    )


def _extract_ab_logprobs(top_logprobs_list) -> dict:
    """Extract logprobs for tokens A/B from top_logprobs list."""
    out = {"A": -1e9, "B": -1e9}
    for alt in top_logprobs_list:
        tok = alt.token.strip().upper()
        if tok in ("A", "B"):
            out[tok] = max(out[tok], alt.logprob)
        elif len(tok) >= 1 and tok[-1] in "AB" and tok[:-1] in ("(", "'", '"', "."):
            letter = tok[-1]
            out[letter] = max(out[letter], alt.logprob)
        elif len(tok) >= 1 and tok[0] in "AB" and (len(tok) == 1 or tok[1] in ").' \"."):
            letter = tok[0]
            out[letter] = max(out[letter], alt.logprob)
    return out


def _softmax_ab(lps: dict) -> dict:
    vals = [lps[k] for k in "AB"]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps) or 1.0
    return {k: e / s for k, e in zip("AB", exps)}


async def _score_one(model_id: str, item: dict) -> dict:
    """Run one WinoGrande question through one model."""
    client = await _next_nim_client()
    prompt = _build_prompt(item)
    correct_letter = item["correct_letter"]

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
            first_letter = next((c for c in text.upper() if c in "AB"), "?")
            content = r.choices[0].logprobs.content if r.choices[0].logprobs else None
            if not content:
                return {"error": "no_logprobs", "response": text}
            first = content[0]
            top = first.top_logprobs or []
            ab_lps = _extract_ab_logprobs(top)
            sm = _softmax_ab(ab_lps)
            return {
                "response": text,
                "first_letter": first_letter,
                "correct_letter": correct_letter,
                "is_correct": first_letter == correct_letter,
                "logprobs_ab": ab_lps,
                "softmax_ab": sm,
                "max_conf": max(sm.values()),
            }
        except asyncio.TimeoutError:
            if attempt < 2:
                continue
            return {"error": f"timeout_{TIMEOUT_S}s"}
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
            return {"error": str(exc)[:120]}
    return {"error": "max_retries"}


async def run_model(model_name: str, model_id: str, items: list[dict]) -> None:
    """Run all WinoGrande items on one model with worker pool + crash-safe writes."""
    output_path = DATA_DIR / f"results_{model_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
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
    print(f"  [{model_name}] {len(completed_ids)} done, {len(pending)} pending", flush=True)

    if not pending:
        return

    # Worker pool with semaphore + buffered writer
    queue: asyncio.Queue = asyncio.Queue()
    for it in pending:
        queue.put_nowait(it)
    write_queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    semaphore = asyncio.Semaphore(PER_MODEL_CONCURRENCY)
    counter = {"done": len(completed_ids), "errors": 0}
    total = len(items)
    start = time.time()

    async def _process(item: dict) -> None:
        async with semaphore:
            result = await _score_one(model_id, item)
        rec = {
            "question_id": item["question_id"],
            "sentence": item["sentence"][:200],  # truncate for space
            **result,
        }
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
                    if any(r.get("error") for r in buffer):
                        counter["errors"] += sum(1 for r in buffer if r.get("error"))
                    if counter["done"] - last_print >= 50:
                        elapsed = (time.time() - start) / 60
                        rate = (counter["done"] - len(completed_ids)) / max(elapsed, 0.01)
                        eta = (total - counter["done"]) / max(rate, 0.1)
                        print(f"    [{model_name}] {counter['done']}/{total} "
                              f"({counter['done']*100//total}%) "
                              f"err={counter['errors']} rate={rate:.0f}/min eta={eta:.1f}min",
                              flush=True)
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
                if counter["done"] - last_print >= 50:
                    elapsed = (time.time() - start) / 60
                    rate = (counter["done"] - len(completed_ids)) / max(elapsed, 0.01)
                    eta = (total - counter["done"]) / max(rate, 0.1)
                    print(f"    [{model_name}] {counter['done']}/{total} "
                          f"({counter['done']*100//total}%) "
                          f"err={counter['errors']} rate={rate:.0f}/min eta={eta:.1f}min",
                          flush=True)
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
                print(f"    [{model_name}] worker exception: {exc}", flush=True)
            finally:
                queue.task_done()

    flusher_task = asyncio.create_task(_flusher())
    await asyncio.gather(*[_worker() for _ in range(NUM_WORKERS_PER_MODEL)])
    await write_queue.put(DONE)
    await flusher_task

    print(f"  [{model_name}] COMPLETE: {counter['done']}/{total} "
          f"errors={counter['errors']} runtime={(time.time()-start)/60:.1f}min",
          flush=True)


def load_winogrande_items() -> list[dict]:
    """Load WinoGrande validation items and assign question_ids.

    Cached to data/winogrande/all_items.json to avoid re-downloading.
    """
    cache = DATA_DIR / "all_items.json"
    if cache.exists():
        items = json.loads(cache.read_text(encoding="utf-8"))
        print(f"  Loaded {len(items)} items from cache", flush=True)
        return items

    from datasets import load_dataset

    print("  Downloading WinoGrande (winogrande_xl) validation split...", flush=True)
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")

    items = []
    for i, row in enumerate(ds):
        # answer is "1" or "2" -> map to "A" or "B"
        answer_str = row["answer"]
        correct_letter = "A" if answer_str == "1" else "B"
        items.append({
            "question_id": f"wg_{i:06d}",
            "sentence": row["sentence"],
            "option1": row["option1"],
            "option2": row["option2"],
            "answer": answer_str,
            "correct_letter": correct_letter,
        })

    print(f"  Loaded {len(items)} items", flush=True)
    cache.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return items


async def main() -> None:
    print("=" * 70)
    print("WINOGRANDE REAL-MODEL VERIFICATION EXPERIMENT")
    print("=" * 70)
    _init_clients()
    items = load_winogrande_items()

    # Run all 3 models concurrently — each has its own worker pool
    tasks = [run_model(name, mid, items) for name, mid in MODELS]
    await asyncio.gather(*tasks)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, _ in MODELS:
        path = DATA_DIR / f"results_{name}.jsonl"
        if not path.exists():
            print(f"  {name}: no file")
            continue
        records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        valid = [r for r in records if not r.get("error") and r.get("max_conf") is not None]
        errors = [r for r in records if r.get("error")]
        if not valid:
            print(f"  {name}: no valid records ({len(records)} total, {len(errors)} errors)")
            continue
        n_correct = sum(1 for r in valid if r.get("is_correct"))
        accuracy = n_correct / len(valid)
        err_rate = 1 - accuracy
        confs = [r["max_conf"] for r in valid]
        mean_conf = sum(confs) / len(confs)
        n_sat = sum(1 for c in confs if c > 0.99)
        print(f"  {name}:")
        print(f"    Items: {len(valid)}/{len(records)} valid, {len(errors)} errors")
        print(f"    Accuracy: {accuracy:.3f} (error rate: {err_rate:.3f})")
        print(f"    Mean confidence: {mean_conf:.3f}")
        print(f"    Saturated (>0.99): {n_sat}/{len(valid)} ({n_sat*100/len(valid):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
