#!/usr/bin/env bash
# FOUR-worker parallel sweep: 4 lanes × 2 threads each on 8 physical cores.
# Hard caps at 100 items per benchmark for max speed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${MANIFEST_OVERRIDE:-$ROOT/models/MANIFEST.csv}"
LANE_DIR="$ROOT/logs/lane_manifests"
mkdir -p "$LANE_DIR"

LANE_A="$LANE_DIR/lane_a.csv"
LANE_B="$LANE_DIR/lane_b.csv"
LANE_C="$LANE_DIR/lane_c.csv"
LANE_D="$LANE_DIR/lane_d.csv"
: > "$LANE_A"; : > "$LANE_B"; : > "$LANE_C"; : > "$LANE_D"

i=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  case $((i % 4)) in
    0) echo "$line" >> "$LANE_A" ;;
    1) echo "$line" >> "$LANE_B" ;;
    2) echo "$line" >> "$LANE_C" ;;
    3) echo "$line" >> "$LANE_D" ;;
  esac
  i=$((i + 1))
done < "$MANIFEST"

echo "lane A:"; cat "$LANE_A"
echo "lane B:"; cat "$LANE_B"
echo "lane C:"; cat "$LANE_C"
echo "lane D:"; cat "$LANE_D"

SPEC="${BENCHSPEC:-mmlu:100,truthfulqa:100,arc_challenge:100}"
NCTX="${NCTX:-2048}"

bash "$ROOT/scripts/run_lane.sh" A "$LANE_A" 2 "$SPEC" "$NCTX" &
PA=$!
bash "$ROOT/scripts/run_lane.sh" B "$LANE_B" 2 "$SPEC" "$NCTX" &
PB=$!
bash "$ROOT/scripts/run_lane.sh" C "$LANE_C" 2 "$SPEC" "$NCTX" &
PC=$!
bash "$ROOT/scripts/run_lane.sh" D "$LANE_D" 2 "$SPEC" "$NCTX" &
PD=$!

echo "4 lanes launched pids=$PA,$PB,$PC,$PD — waiting..."
wait "$PA" "$PB" "$PC" "$PD"
echo "all 4 lanes finished $(date)"
