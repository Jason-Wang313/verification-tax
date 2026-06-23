#!/usr/bin/env bash
# Parallel safety+fairness sweep: 3 models split into 2 lanes, 4 threads each.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANE_DIR="$ROOT/logs/lane_manifests"
mkdir -p "$LANE_DIR"
LANE_A="$LANE_DIR/safety_a.csv"
LANE_B="$LANE_DIR/safety_b.csv"
cat > "$LANE_A" <<EOF
llama-3-2-3b-instruct-q4-k-m,models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
gemma-2-2b-it-q4-k-m,models/gemma-2-2b-it-Q4_K_M.gguf
EOF
cat > "$LANE_B" <<EOF
phi-3-mini-4k-instruct-q4-k-m,models/Phi-3-mini-4k-instruct-Q4_K_M.gguf
EOF

SAFETY_SPEC="bbq:300,xstest:0,toxigen:0"

bash "$ROOT/scripts/run_lane.sh" safetyA "$LANE_A" 4 "$SAFETY_SPEC" 2048 &
PA=$!
bash "$ROOT/scripts/run_lane.sh" safetyB "$LANE_B" 4 "$SAFETY_SPEC" 2048 &
PB=$!
wait "$PA" "$PB"
echo "safety lanes done $(date)"
