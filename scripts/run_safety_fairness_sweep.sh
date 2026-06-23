#!/usr/bin/env bash
# v6 fairness/safety sweep: bundled per model for (bbq:1500, xstest:0, toxigen:0).
# Keeps each model loaded across all three safety benchmarks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

MODELS=(
  "llama-3-2-3b-instruct-q4-k-m|models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
  "phi-3-mini-4k-instruct-q4-k-m|models/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
  "gemma-2-2b-it-q4-k-m|models/gemma-2-2b-it-Q4_K_M.gguf"
)

BENCHSPEC="bbq:500,xstest:0,toxigen:0"

main_log="$LOGDIR/safety_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "safety/fairness sweep started $(date) (bundle spec: $BENCHSPEC)" | tee -a "$main_log"

for entry in "${MODELS[@]}"; do
  MODEL_ID="${entry%|*}"
  MODEL_PATH="${entry#*|}"
  if [[ ! -f "$ROOT/$MODEL_PATH" ]]; then
    echo "[skip] $MODEL_ID — path missing" | tee -a "$main_log"
    continue
  fi
  echo "=== $MODEL_ID (bundled bbq+xstest+toxigen) $(date +%H:%M:%S) ===" | tee -a "$main_log"
  python "$ROOT/scripts/run_model_bundle.py" \
    --model-path "$ROOT/$MODEL_PATH" \
    --model-id   "$MODEL_ID" \
    --benchmarks "$BENCHSPEC" \
    --n-ctx 1536 2>&1 | tee -a "$main_log"
done

echo "safety/fairness sweep finished $(date)" | tee -a "$main_log"
