#!/usr/bin/env bash
# Download the 10-model GGUF roster for the v6 local sweep.
# Resume-safe: huggingface-cli download verifies existing files.
# Total size: ~30 GB. Expect 6-10 hours on a home link.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

# (repo, filename) pairs
ENTRIES=(
  "bartowski/Meta-Llama-3-8B-Instruct-GGUF|Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
  "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF|Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
  "bartowski/Llama-3.2-3B-Instruct-GGUF|Llama-3.2-3B-Instruct-Q4_K_M.gguf"
  "bartowski/gemma-2-9b-it-GGUF|gemma-2-9b-it-Q4_K_M.gguf"
  "bartowski/gemma-2-2b-it-GGUF|gemma-2-2b-it-Q4_K_M.gguf"
  "bartowski/Phi-3-mini-4k-instruct-GGUF|Phi-3-mini-4k-instruct-Q4_K_M.gguf"
  "bartowski/Phi-3.5-mini-instruct-GGUF|Phi-3.5-mini-instruct-Q4_K_M.gguf"
  "bartowski/Mistral-7B-Instruct-v0.3-GGUF|Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
  "bartowski/Qwen2.5-7B-Instruct-GGUF|Qwen2.5-7B-Instruct-Q4_K_M.gguf"
  "bartowski/Qwen2.5-3B-Instruct-GGUF|Qwen2.5-3B-Instruct-Q4_K_M.gguf"
)

for entry in "${ENTRIES[@]}"; do
  repo="${entry%|*}"
  fname="${entry#*|}"
  target="$MODELS_DIR/$fname"
  if [[ -s "$target" ]]; then
    size_mb=$(du -m "$target" | cut -f1)
    echo "[skip] $fname already present (${size_mb} MB)"
    continue
  fi
  echo "[down] $repo :: $fname"
  huggingface-cli download "$repo" "$fname" --local-dir "$MODELS_DIR" \
    --local-dir-use-symlinks False 2>&1 | tail -5 || {
      echo "[warn] failed: $repo :: $fname (will retry on next run)"
      continue
    }
done

echo "done. $(ls -1 "$MODELS_DIR"/*.gguf 2>/dev/null | wc -l) GGUF files present."
