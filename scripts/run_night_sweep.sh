#!/usr/bin/env bash
# Overnight 2-lane sweep. Balanced small+big per lane, 4 threads each.
# Resume-safe (checkpoints per item). Hard-stopped by alarm_kill.sh.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANE_DIR="$ROOT/logs/lane_manifests"
LOG="$ROOT/logs/night_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "[night] start $(date)" | tee -a "$LOG"

SPEC="${BENCHSPEC:-mmlu:300,truthfulqa:200,arc_challenge:200}"
NCTX="${NCTX:-1536}"
NTHR="${NTHR:-4}"

bash "$ROOT/scripts/run_lane.sh" A "$LANE_DIR/lane_a_night.csv" "$NTHR" "$SPEC" "$NCTX" >> "$LOG" 2>&1 &
PA=$!
bash "$ROOT/scripts/run_lane.sh" B "$LANE_DIR/lane_b_night.csv" "$NTHR" "$SPEC" "$NCTX" >> "$LOG" 2>&1 &
PB=$!
echo "[night] lanes launched pidA=$PA pidB=$PB" | tee -a "$LOG"
wait "$PA" "$PB" 2>/dev/null || true
echo "[night] end $(date)" | tee -a "$LOG"
