#!/usr/bin/env bash
# Watch the sweep and auto-regenerate screened-roster analysis + PDF each
# time a new model completes. Run alongside run_v6_sweep.sh.
#
# Detection: periodically checksum the data/mmlu/ directory listing; when it
# changes AND at least one new model has ≥500 items, trigger a rebuild.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

last_sig=""
while true; do
  sig=$(ls -la "$ROOT/data/mmlu/"results_*.jsonl "$ROOT/data/truthfulqa/"results_*.jsonl "$ROOT/data/arc_challenge/"results_*.jsonl 2>/dev/null | md5sum | cut -d' ' -f1)
  if [[ "$sig" != "$last_sig" && -n "$last_sig" ]]; then
    echo "[$(date +%H:%M)] sweep progress detected, regenerating analysis + PDF"
    (cd "$ROOT" && python scripts/analyze_screened_roster.py 2>&1 | tail -5)
    (cd "$ROOT" && python scripts/analyze_all_benchmarks.py 2>&1 | tail -3)
    (cd "$ROOT" && pdflatex -interaction=nonstopmode main.tex > /tmp/p1.log 2>&1 && pdflatex -interaction=nonstopmode main.tex > /tmp/p2.log 2>&1)
    cp "$ROOT/main.pdf" "$HOME/Downloads/main.pdf" 2>/dev/null && echo "  refreshed main.pdf"
  fi
  last_sig="$sig"
  sleep 300  # every 5 minutes
done
