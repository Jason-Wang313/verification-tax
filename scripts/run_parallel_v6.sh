#!/usr/bin/env bash
# Two-worker parallel sweep: split the manifest into odd/even lanes, run each
# with 4 threads concurrently. Saturates 8 logical cores while allowing two
# models to make progress simultaneously on memory-bandwidth-bound workloads.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/models/MANIFEST.csv"
LANE_DIR="$ROOT/logs/lane_manifests"
mkdir -p "$LANE_DIR"

LANE_A="$LANE_DIR/lane_a.csv"
LANE_B="$LANE_DIR/lane_b.csv"
: > "$LANE_A"; : > "$LANE_B"

i=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  if (( i % 2 == 0 )); then
    echo "$line" >> "$LANE_A"
  else
    echo "$line" >> "$LANE_B"
  fi
  i=$((i + 1))
done < "$MANIFEST"

echo "lane A:"; cat "$LANE_A"
echo "lane B:"; cat "$LANE_B"

# Main-sweep spec: MMLU 300 + TQA 200 + ARC 200.
MAIN_SPEC="mmlu:300,truthfulqa:200,arc_challenge:200"

bash "$ROOT/scripts/run_lane.sh" A "$LANE_A" 4 "$MAIN_SPEC" 2048 &
PID_A=$!
bash "$ROOT/scripts/run_lane.sh" B "$LANE_B" 4 "$MAIN_SPEC" 2048 &
PID_B=$!

echo "launched lane A pid=$PID_A, lane B pid=$PID_B — waiting..."
wait "$PID_A" "$PID_B"
echo "both lanes finished $(date)"
