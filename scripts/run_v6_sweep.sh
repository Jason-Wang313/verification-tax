#!/usr/bin/env bash
# v6 main sweep: one bundled invocation per model, covering MMLU(n=500), TQA, ARC.
# Loading the GGUF once per model instead of 3× cuts ~20-60s per model.
# Resume-safe (run_model_bundle.py uses run_local_experiment._completed_ids).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/models/MANIFEST.csv"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

if [[ ! -f "$MANIFEST" ]]; then
  python "$ROOT/scripts/build_manifest.py"
fi

BENCHSPEC="mmlu:300,truthfulqa:200,arc_challenge:200"

main_log="$LOGDIR/v6_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "v6 sweep started $(date) (bundle spec: $BENCHSPEC)" | tee -a "$main_log"

while IFS=, read -r MODEL_ID MODEL_PATH; do
  if [[ -z "$MODEL_ID" ]]; then continue; fi
  if [[ ! -f "$ROOT/$MODEL_PATH" ]]; then
    echo "[skip] $MODEL_ID — path missing: $MODEL_PATH" | tee -a "$main_log"
    continue
  fi
  echo "=== $MODEL_ID (bundled 3 benchmarks) $(date +%H:%M:%S) ===" | tee -a "$main_log"
  python "$ROOT/scripts/run_model_bundle.py" \
    --model-path "$ROOT/$MODEL_PATH" \
    --model-id   "$MODEL_ID" \
    --benchmarks "$BENCHSPEC" \
    --n-ctx 2048 2>&1 | tee -a "$main_log"
done < "$MANIFEST"

echo "v6 sweep finished $(date)" | tee -a "$main_log"
