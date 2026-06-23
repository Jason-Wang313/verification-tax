#!/usr/bin/env bash
# Run one sweep lane: iterates its own sub-manifest with a fixed thread count.
# Usage: bash scripts/run_lane.sh <lane_name> <sub_manifest_path> <n_threads> <benchspec> <n_ctx>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANE="$1"
MANIFEST="$2"
NTHR="${3:-4}"
BENCHSPEC="${4:-mmlu:300,truthfulqa:200,arc_challenge:200}"
NCTX="${5:-2048}"

LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/lane_${LANE}_$(date +%Y%m%d_%H%M%S).log"

echo "[$LANE] started $(date) manifest=$MANIFEST threads=$NTHR spec=$BENCHSPEC" | tee -a "$LOG"

while IFS=, read -r MODEL_ID MODEL_PATH; do
  [[ -z "$MODEL_ID" ]] && continue
  if [[ ! -f "$ROOT/$MODEL_PATH" ]]; then
    echo "[$LANE skip] $MODEL_ID — path missing" | tee -a "$LOG"
    continue
  fi
  echo "[$LANE] === $MODEL_ID $(date +%H:%M:%S) ===" | tee -a "$LOG"
  python "$ROOT/scripts/run_model_bundle.py" \
    --model-path "$ROOT/$MODEL_PATH" \
    --model-id   "$MODEL_ID" \
    --benchmarks "$BENCHSPEC" \
    --n-ctx      "$NCTX" \
    --n-threads  "$NTHR" 2>&1 | tee -a "$LOG"
done < "$MANIFEST"

echo "[$LANE] finished $(date)" | tee -a "$LOG"
