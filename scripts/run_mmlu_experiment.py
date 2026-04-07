"""MMLU real-model verification experiment.

Runs the full MMLU test set on 4 NIM-hosted LLMs, extracts logprob-based
confidence scores (softmax over A/B/C/D logprobs at the first generated
position), and saves incrementally for crash-resistance.

Output: data/mmlu/results_{model_name}.jsonl
  Each line: {question_id, subject, model_answer, correct_answer,
              logprobs_abcd, softmax_abcd, max_conf, is_correct, error}

Resume from checkpoint by re-running — already-completed query_ids skipped.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "mmlu"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("llama-3.1-405b-instruct", "meta/llama-3.1-405b-instruct"),
    ("llama-4-maverick", "meta/llama-4-maverick-17b-128e-instruct"),
    ("qwen3-next-80b", "qwen/qwen3-next-80b-a3b-instruct"),
    ("llama-3.1-70b-instruct", "meta/llama-3.1-70b-instruct"),
]

# Per-model concurrency: keep low enough to avoid 429 storms.
# With 4 models running in parallel and ~16 concurrent calls each = 64 total.
PER_MODEL_CONCURRENCY = 16
NUM_WORKERS_PER_MODEL = 24
MAX_TOKENS = 5
TIMEOUT_S = 60


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
_NIM_LOCK: asyncio.Lock | None = None


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
    global _NIM_LOCK
    if _NIM_LOCK is None:
        _NIM_LOCK = asyncio.Lock()
    async with _NIM_LOCK:
        idx = _NIM_IDX[0] % len(_NIM_CLIENTS)
        _NIM_IDX[0] += 1
    return _NIM_CLIENTS[idx]


def _build_prompt(item: dict) -> str:
    q = item["question"]
    c = item["choices"]
    return (
        "Answer the following multiple choice question. Reply with ONLY "
        "the letter (A, B, C, or D).\n\n"
        f"Question: {q}\n"
        f"A) {c[0]}\n"
        f"B) {c[1]}\n"
        f"C) {c[2]}\n"
        f"D) {c[3]}\n\n"
        "Answer:"
    )


def _extract_abcd_logprobs(top_logprobs_list) -> dict:
    """Extract logprobs for tokens A/B/C/D from top_logprobs list.

    Tries to match exact 'A','B','C','D' first, then variants like '(A',
    'A)', ' A', etc. Takes the max logprob seen for each letter.
    """
    out = {"A": -1e9, "B": -1e9, "C": -1e9, "D": -1e9}
    for alt in top_logprobs_list:
        tok = alt.token.strip().upper()
        if tok in ("A", "B", "C", "D"):
            out[tok] = max(out[tok], alt.logprob)
        elif len(tok) >= 1 and tok[-1] in "ABCD" and tok[:-1] in ("(", "'", '"', "."):
            letter = tok[-1]
            out[letter] = max(out[letter], alt.logprob)
        elif len(tok) >= 1 and tok[0] in "ABCD" and (len(tok) == 1 or tok[1] in ").' \"."):
            letter = tok[0]
            out[letter] = max(out[letter], alt.logprob)
    return out


def _softmax_abcd(lps: dict) -> dict:
    vals = [lps[k] for k in "ABCD"]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps) or 1.0
    return {k: e / s for k, e in zip("ABCD", exps)}


async def _score_one(model_id: str, item: dict) -> dict:
    """Run one MMLU question through one model."""
    client = await _next_nim_client()
    prompt = _build_prompt(item)
    correct_letter = chr(65 + item["answer"])

    delay = 15
    for attempt in range(5):
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
            first_letter = next((c for c in text if c in "ABCD"), "?")
            content = r.choices[0].logprobs.content if r.choices[0].logprobs else None
            if not content:
                return {"error": "no_logprobs", "response": text}
            first = content[0]
            top = first.top_logprobs or []
            abcd_lps = _extract_abcd_logprobs(top)
            sm = _softmax_abcd(abcd_lps)
            return {
                "response": text,
                "first_letter": first_letter,
                "correct_letter": correct_letter,
                "is_correct": first_letter == correct_letter,
                "logprobs_abcd": abcd_lps,
                "softmax_abcd": sm,
                "max_conf": max(sm.values()),
                "first_token": first.token,
            }
        except asyncio.TimeoutError:
            if attempt < 2:
                await asyncio.sleep(5)
                continue
            return {"error": f"timeout_{TIMEOUT_S}s"}
        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "rate limit" in msg:
                if attempt < 4:
                    await asyncio.sleep(min(delay, 90))
                    delay = min(delay * 2, 90)
                    continue
            if any(t in msg for t in ("500", "502", "503")):
                if attempt < 3:
                    await asyncio.sleep(10)
                    continue
            return {"error": str(exc)[:120]}
    return {"error": "max_retries"}


async def run_model(model_name: str, model_id: str, items: list[dict]) -> None:
    """Run all MMLU items on one model with worker pool + crash-safe writes."""
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
            "subject": item["subject"],
            "answer_letter": chr(65 + item["answer"]),
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
                    if counter["done"] - last_print >= 200:
                        elapsed = (time.time() - start) / 60
                        rate = (counter["done"] - len(completed_ids)) / max(elapsed, 0.01)
                        eta = (total - counter["done"]) / max(rate, 0.1)
                        print(f"    [{model_name}] {counter['done']}/{total} "
                              f"({counter['done']*100//total}%) "
                              f"err={counter['errors']} rate={rate:.0f}/min eta={eta:.0f}min",
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
            if len(buffer) >= 20:
                with output_path.open("a", encoding="utf-8") as f:
                    for r in buffer:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                counter["done"] += len(buffer)
                if counter["done"] - last_print >= 200:
                    elapsed = (time.time() - start) / 60
                    rate = (counter["done"] - len(completed_ids)) / max(elapsed, 0.01)
                    eta = (total - counter["done"]) / max(rate, 0.1)
                    print(f"    [{model_name}] {counter['done']}/{total} "
                          f"({counter['done']*100//total}%) "
                          f"err={counter['errors']} rate={rate:.0f}/min eta={eta:.0f}min",
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
          f"errors={counter['errors']} runtime={(time.time()-start)/60:.0f}min",
          flush=True)


def load_mmlu_items() -> list[dict]:
    """Load all MMLU test items and assign global question_ids.

    Cached to data/mmlu/all_items.json to avoid re-downloading.
    """
    cache = DATA_DIR / "all_items.json"
    if cache.exists():
        items = json.loads(cache.read_text(encoding="utf-8"))
        print(f"  Loaded {len(items)} items from cache", flush=True)
        return items

    from datasets import load_dataset

    print("  Downloading MMLU 'all' subset (full test split)...", flush=True)
    ds = load_dataset("cais/mmlu", "all", split="test")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "question_id": f"mmlu_{i:06d}",
            "subject": row["subject"],
            "question": row["question"],
            "choices": list(row["choices"]),
            "answer": int(row["answer"]),
        })
    print(f"  Loaded {len(items)} items", flush=True)
    cache.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return items


async def main() -> None:
    print("=" * 70)
    print("MMLU REAL-MODEL VERIFICATION EXPERIMENT")
    print("=" * 70)
    _init_clients()
    items = load_mmlu_items()

    # Run all 4 models concurrently — each has its own worker pool
    tasks = [run_model(name, mid, items) for name, mid in MODELS]
    await asyncio.gather(*tasks)

    print("\nAll models complete. Computing summary...")
    for name, _ in MODELS:
        path = DATA_DIR / f"results_{name}.jsonl"
        if not path.exists():
            print(f"  {name}: no file")
            continue
        records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        valid = [r for r in records if not r.get("error") and r.get("max_conf") is not None]
        if not valid:
            print(f"  {name}: no valid records ({len(records)} total, all errors)")
            continue
        n_correct = sum(1 for r in valid if r.get("is_correct"))
        eps = 1 - n_correct / len(valid)
        confs = [r["max_conf"] for r in valid]
        n_sat = sum(1 for c in confs if c > 0.99)
        print(f"  {name}: {len(valid)}/{len(records)} valid, "
              f"epsilon={eps:.3f}, mean_conf={sum(confs)/len(confs):.3f}, "
              f"saturated={n_sat}/{len(valid)}")


if __name__ == "__main__":
    asyncio.run(main())
