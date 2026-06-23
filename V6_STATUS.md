# v6 Session Status (2026-04-15)

## Done in this session

- **Block D — OOP API.** `VerificationAudit(conf, labels).report()` ships in `verification-tax` v1.1.0. 11 tests pass. README carries the one-liner. `main.tex` advertises the CLI + OOP one-liner as a new paragraph.
- **Block A — Downloads.** All 10 GGUF weights in `models/` (≈30 GB): Llama-3-8B, Llama-3.1-8B, Llama-3.2-3B, Gemma-2-2B, Gemma-2-9B, Mistral-7B-v0.3, Phi-3-mini, Phi-3.5-mini, Qwen2.5-3B, Qwen2.5-7B. `models/MANIFEST.csv` regenerated with LF line endings. `scripts/build_manifest.py` rewrites it with forward slashes and LF only.
- **Block A — Sweep infrastructure.** `scripts/run_v6_sweep.sh` iterates the manifest over MMLU (n=500), TruthfulQA (full, 817), ARC-Challenge (full, 1172). Resume-safe. Logs to `logs/v6_sweep_*.log`.
- **Block C — Data prep.**
  - `data/bbq/all_items.json` (58,492 items, 11 subgroups) via `oskarvanderwal/bbq`.
  - `data/xstest/all_items.json` (450 items) via `natolambert/xstest-v2-copy`.
  - `data/toxigen/all_items.json` (940 annotated items) via `skg/toxigen-data/annotated`.
- **Block C — Scripts.** `prepare_toxigen.py` written; `run_local_experiment.py` BENCHMARK_CONFIG now includes `toxigen`; `analyze_safety_floor.py` reports both XSTest (n=450) and ToxiGen (n=940) floor thresholds in one table.
- **Paper.** Still 9 pages, References starts top of page 10. `pip install verification-tax` paragraph added.

## Running in background (at session end)

- Task `bgunt8j4k`: `bash scripts/run_v6_sweep.sh` (8-model MMLU/TQA/ARC sweep). Progress at session close: `gemma-2-2b / mmlu` at 35/500 items (~12 s/item on CPU).

Expected wall-clock to sweep completion: ~2-3 days at observed rate (500 items × 8 models × 3 benchmarks ≈ 12,000 inferences × 12 s).

## To resume in next session

1. **Check sweep progress.**
   ```bash
   tail -20 "logs/v6_sweep_"*.log
   wc -l data/mmlu/results_*.jsonl data/truthfulqa/results_*.jsonl data/arc_challenge/results_*.jsonl
   ```

2. **If main sweep is still running, leave it.** Otherwise relaunch:
   ```bash
   python scripts/build_manifest.py   # pick up Qwen2.5-3B/7B that weren't in earlier manifest
   bash scripts/run_v6_sweep.sh       # resume-safe; skips completed (model, benchmark) pairs
   ```

3. **Launch fairness + safety sweep** (parallel OK once main is ≥50% done, else serial):
   ```bash
   bash scripts/run_safety_fairness_sweep.sh
   ```
   This runs 3 small models (Llama-3.2-3B, Phi-3-mini, Gemma-2-2B) × (BBQ n=1500, XSTest full, ToxiGen full). ~8-12 hours.

4. **Block B — Live active querying** (once Block A has ≥3 small models at n=500):
   ```bash
   for m in llama-3-2-3b-instruct-q4-k-m phi-3-mini-4k-instruct-q4-k-m gemma-2-2b-it-q4-k-m; do
     p=$(grep "^${m}," models/MANIFEST.csv | cut -d, -f2)
     python scripts/exp_active_live.py --model-path "$p" --model-id "$m" --budget 2000 --benchmark mmlu
   done
   ```

5. **Block E — Paper integration** (once enough data lands):
   ```bash
   python scripts/analyze_screened_roster.py    # refreshes screened_roster_table.tex, confidence_quality_table.tex, fig_confidence_quality_audit.pdf
   python scripts/analyze_all_benchmarks.py
   python scripts/analyze_fairness_floor.py
   python scripts/analyze_safety_floor.py
   python scripts/exp_pipeline_multi.py         # optional; current output already in paper
   pdflatex main.tex && pdflatex main.tex
   cp main.pdf ~/Downloads/main.pdf
   ```
   Then update these numbers in `main.tex`:
   - Abstract line 100: "screened headline set of 16 non-degenerate runs" → actual count from `results/analysis/screened_roster_summary.json`.
   - Empirics line 345: "$12/13$ are non-significant" → actual ratio from `analyze_screened_roster.py` stdout.
   - Confidence-quality section line 248: "~40%" rejection rate → actual ratio if different.
   - Add a fairness paragraph summarising minority/majority floor ratio from `results/fairness_floor_table.tex`.
   - Add a safety paragraph summarising the 95→97 verdict at n=450 vs n=940 from `results/safety_floor_table.tex`.

6. **Alternative**: run `bash scripts/watch_and_refresh.sh` in a second terminal to auto-refresh tables/figures/PDF every 5 minutes as new data lands.

## Files added this session

- `vtax_package/verification_tax/core.py` — `VerificationAudit` class appended.
- `vtax_package/verification_tax/__init__.py` — export.
- `vtax_package/vtax/__init__.py` — compat re-export.
- `vtax_package/README.md` — OOP one-liner section.
- `vtax_package/pyproject.toml` — v1.1.0.
- `vtax_package/tests/test_audit.py` — 2 new tests.
- `scripts/download_models.sh` — 10-model batch download.
- `scripts/run_v6_sweep.sh` — MMLU/TQA/ARC × 10 models.
- `scripts/run_safety_fairness_sweep.sh` — BBQ/XSTest/ToxiGen × 3 models.
- `scripts/prepare_toxigen.py` — HF dataset → all_items.json.
- `scripts/watch_and_refresh.sh` — auto-refresh paper on new data.
- `data/bbq/`, `data/xstest/`, `data/toxigen/` — all_items.json files.
- `models/*.gguf` — 10 GGUF weights (~30 GB).
- `logs/v6_sweep_*.log` — sweep run logs.

## Files modified this session

- `main.tex` — pip-tool paragraph added; broader-impacts trimmed for page budget.
- `scripts/prepare_bbq.py` — switched from deprecated `heegyu/bbq` script loader to `oskarvanderwal/bbq` parquet dataset.
- `scripts/run_local_experiment.py` — `logits_all=True`, `toxigen` benchmark, `subgroup` field propagated.
- `scripts/build_manifest.py` — POSIX paths + LF line endings so bash read-loops on Windows don't carry `\r`.
- `scripts/analyze_safety_floor.py` — reads both `xstest/` and `toxigen/` directories, emits n=450 and n=940 verdicts side by side.

## Open questions / risks

- **Qwen models** (Qwen2.5-3B, Qwen2.5-7B) finished downloading after the running sweep had already read `MANIFEST.csv`. They'll be included on the next sweep invocation; re-running `scripts/run_v6_sweep.sh` after the current run picks them up automatically (resume-safe).
- **CPU wall-clock**: at observed ~12 s/item, the full sweep matrix is ~48 hours. Smaller models (2-3B) run closer to 3-5 s/item; 7-9B closer to 15-20 s/item.
- **Subgroup floor ratio target**: for π=0.05, theory predicts $1/\pi^{1/3} \approx 2.71\times$ higher floor vs the majority (at fixed ε). Current `analyze_fairness_floor.py` plots this alongside the empirical scatter.
