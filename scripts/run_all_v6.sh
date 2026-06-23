#!/usr/bin/env bash
# Unified v6 execution: main sweep → fairness/safety sweep → live active → paper refresh.
# Resume-safe throughout (run_local_experiment checkpoints by question_id).
# Expected wall-clock: ~50 hours on CPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

DRIVER_LOG="$LOGDIR/run_all_v6_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date)] run_all_v6 driver started" | tee -a "$DRIVER_LOG"

# Phase 1 — manifest must exist. Do NOT auto-rebuild (would overwrite hand-curated rosters).
if [[ ! -f "$ROOT/models/MANIFEST.csv" ]]; then
  python "$ROOT/scripts/build_manifest.py" > /dev/null
fi
echo "[$(date)] manifest: $(wc -l < "$ROOT/models/MANIFEST.csv") models" | tee -a "$DRIVER_LOG"

# Phase 2 — 4-WORKER parallel main sweep: 4 lanes × 2 threads × 5 models, caps 100/100/100.
echo "[$(date)] === PHASE 2: 4-lane main sweep ===" | tee -a "$DRIVER_LOG"
BENCHSPEC="mmlu:100,truthfulqa:100,arc_challenge:100" NCTX=1536 \
  bash "$ROOT/scripts/run_4lane_v6.sh" 2>&1 | tee -a "$DRIVER_LOG" || true

# Phase 3 — 4-WORKER parallel safety+fairness on 3-model sub-manifest.
echo "[$(date)] === PHASE 3: 4-lane safety+fairness ===" | tee -a "$DRIVER_LOG"
mkdir -p "$ROOT/logs/lane_manifests"
cat > "$ROOT/logs/lane_manifests/safety_manifest.csv" <<EOF
llama-3-2-3b-instruct-q4-k-m,models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
phi-3-mini-4k-instruct-q4-k-m,models/Phi-3-mini-4k-instruct-Q4_K_M.gguf
gemma-2-2b-it-q4-k-m,models/gemma-2-2b-it-Q4_K_M.gguf
EOF
BENCHSPEC="bbq:200,xstest:0,toxigen:0" NCTX=1536 \
  MANIFEST_OVERRIDE="$ROOT/logs/lane_manifests/safety_manifest.csv" \
  bash "$ROOT/scripts/run_4lane_v6.sh" 2>&1 | tee -a "$DRIVER_LOG" || true

# Phase 4 — live active parallel: 3 models × budget=400 (passive + active = 800 inferences each).
echo "[$(date)] === PHASE 4: parallel live active ===" | tee -a "$DRIVER_LOG"
active_pids=()
for m in llama-3-2-3b-instruct-q4-k-m phi-3-mini-4k-instruct-q4-k-m gemma-2-2b-it-q4-k-m; do
    p=$(grep "^${m}," "$ROOT/models/MANIFEST.csv" | cut -d, -f2)
    if [[ -z "$p" ]]; then
        echo "[skip] $m — not in manifest" | tee -a "$DRIVER_LOG"; continue
    fi
    out="$ROOT/results/analysis/active_live_${m}.json"
    if [[ -s "$out" ]]; then
        echo "[skip] $m — $out exists" | tee -a "$DRIVER_LOG"; continue
    fi
    echo "[$(date)] live active bg on $m" | tee -a "$DRIVER_LOG"
    python "$ROOT/scripts/exp_active_live.py" \
        --model-path "$ROOT/$p" --model-id "$m" \
        --budget 400 --benchmark mmlu --n-ctx 1536 --n-threads 2 \
        >> "$DRIVER_LOG" 2>&1 &
    active_pids+=($!)
done
for pid in "${active_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
done

# Phase 5 — analysis refresh + paper rebuild.
echo "[$(date)] === PHASE 5: refresh analysis + paper ===" | tee -a "$DRIVER_LOG"
cd "$ROOT"
python scripts/analyze_screened_roster.py 2>&1 | tee -a "$DRIVER_LOG" || true
python scripts/analyze_all_benchmarks.py 2>&1 | tee -a "$DRIVER_LOG" || true
python scripts/analyze_fairness_floor.py 2>&1 | tee -a "$DRIVER_LOG" || true
python scripts/analyze_safety_floor.py 2>&1 | tee -a "$DRIVER_LOG" || true
python scripts/exp_named_model_comparison.py 2>&1 | tee -a "$DRIVER_LOG" || true
python scripts/exp_pipeline_multi.py 2>&1 | tee -a "$DRIVER_LOG" || true

pdflatex -interaction=nonstopmode main.tex > /tmp/v6_p1.log 2>&1
pdflatex -interaction=nonstopmode main.tex > /tmp/v6_p2.log 2>&1
cp "$ROOT/main.pdf" "$HOME/Downloads/main.pdf" 2>/dev/null || true

echo "[$(date)] run_all_v6 driver FINISHED" | tee -a "$DRIVER_LOG"
