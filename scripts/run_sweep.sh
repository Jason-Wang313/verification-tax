#!/usr/bin/env bash
# Drive the local-model sweep across all (model, benchmark) pairs.
# Requires: models/MANIFEST.csv (run scripts/build_manifest.py first).
# Usage:   bash scripts/run_sweep.sh [--n 1000]   # optional cap for smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/models/MANIFEST.csv"
BENCHMARKS=(mmlu truthfulqa arc_challenge)

if [[ ! -f "$MANIFEST" ]]; then
  echo "missing $MANIFEST — run scripts/build_manifest.py first" >&2
  exit 2
fi

N_CAP=""
if [[ "${1:-}" == "--n" && -n "${2:-}" ]]; then
  N_CAP="--n $2"
fi

while IFS=, read -r MODEL_ID MODEL_PATH; do
  for BENCH in "${BENCHMARKS[@]}"; do
    echo "=== $MODEL_ID / $BENCH ==="
    python "$ROOT/scripts/run_local_experiment.py" \
      --model-path "$ROOT/$MODEL_PATH" \
      --model-id   "$MODEL_ID" \
      --benchmark  "$BENCH" \
      $N_CAP
  done
done < "$MANIFEST"

echo "sweep done — re-run analysis:"
echo "  python scripts/analyze_all_benchmarks.py"
echo "  python scripts/analyze_screened_roster.py"
