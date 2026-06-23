# Local GGUF model download list (CPU sweep)

All downloads land in this `models/` directory. Q4_K_M quantization unless noted.
Roughly 4-6 GB per 7B-8B model, 5-7 GB per 9B, 2 GB per 3B. Total footprint
for the target roster: ~55 GB.

Install helper once:

```bash
pip install huggingface-hub
```

## Primary roster (10 models — required for Block 3)

```bash
# Llama family
huggingface-cli download bartowski/Meta-Llama-3-8B-Instruct-GGUF \
    Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir models

huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models

huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
    Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models

# Gemma family
huggingface-cli download bartowski/gemma-2-9b-it-GGUF \
    gemma-2-9b-it-Q4_K_M.gguf --local-dir models

huggingface-cli download bartowski/gemma-2-2b-it-GGUF \
    gemma-2-2b-it-Q4_K_M.gguf --local-dir models

# Phi family
huggingface-cli download bartowski/Phi-3-mini-4k-instruct-GGUF \
    Phi-3-mini-4k-instruct-Q4_K_M.gguf --local-dir models

huggingface-cli download bartowski/Phi-3.5-mini-instruct-GGUF \
    Phi-3.5-mini-instruct-Q4_K_M.gguf --local-dir models

# Mistral
huggingface-cli download bartowski/Mistral-7B-Instruct-v0.3-GGUF \
    Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --local-dir models

# Qwen family
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models

huggingface-cli download bartowski/Qwen2.5-3B-Instruct-GGUF \
    Qwen2.5-3B-Instruct-Q4_K_M.gguf --local-dir models
```

## Fill-coverage roster (5 more — optional)

```bash
huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF \
    DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf --local-dir models

huggingface-cli download bartowski/Yi-1.5-9B-Chat-GGUF \
    Yi-1.5-9B-Chat-Q4_K_M.gguf --local-dir models

huggingface-cli download bartowski/stablelm-2-12b-chat-GGUF \
    stablelm-2-12b-chat-Q4_K_M.gguf --local-dir models

huggingface-cli download bartowski/OLMo-2-1124-7B-Instruct-GGUF \
    OLMo-2-1124-7B-Instruct-Q4_K_M.gguf --local-dir models

huggingface-cli download bartowski/SmolLM2-1.7B-Instruct-GGUF \
    SmolLM2-1.7B-Instruct-Q4_K_M.gguf --local-dir models
```

## Sweep model list

The sweep driver reads `models/MANIFEST.csv` for `(model_id, model_path)` pairs.
After downloading, `cd` to this directory and run:

```bash
python ../scripts/build_manifest.py   # scans for *.gguf and writes MANIFEST.csv
```

Or edit `MANIFEST.csv` by hand; see `scripts/run_sweep.sh` for the driver.
