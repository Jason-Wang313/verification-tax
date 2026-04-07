#!/bin/bash
# Auto-restart the MMLU experiment when it exits.
# Runs until 405B reaches 13,500 valid records (97% — accounts for hard 403s).
# Each restart re-tries previously errored items.

cd "$(dirname "$0")/.."

while true; do
  # Check current valid count
  VALID=$(python -c "
import json
v = 0
try:
    for line in open('data/mmlu/results_llama-3.1-405b-instruct.jsonl'):
        rec = json.loads(line)
        if 'error' not in rec and 'max_conf' in rec:
            v += 1
    print(v)
except FileNotFoundError:
    print(0)
")
  echo "[$(date +%H:%M:%S)] 405B valid: $VALID/14042"

  if [ "$VALID" -ge 14042 ]; then
    echo "[$(date +%H:%M:%S)] 405B reached $VALID valid, target met. Stopping watchdog."
    break
  fi

  # Run one pass; will exit naturally when queue empty
  python -u scripts/run_mmlu_experiment.py >> results/mmlu_experiment.log 2>&1

  echo "[$(date +%H:%M:%S)] Pass complete; sleeping 30s before retry pass..."
  sleep 30
done

echo "[$(date +%H:%M:%S)] Watchdog finished."
