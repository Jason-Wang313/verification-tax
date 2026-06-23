# Anonymous Reproducibility Artifact

This supplementary package is the anonymous reproducibility artifact for
`The Verification Tax`.

It is designed to let reviewers regenerate the paper's main tables and figures
from saved outputs without rerunning hosted-model inference.

## What is included

- `scripts/`: anonymized analysis and figure-generation scripts
- `data/*/results_*.jsonl`: derived per-item model outputs used by the analysis
- `results/analysis/*.json`: saved summary outputs for cross-checking
- `results/*.tex`: generated LaTeX tables referenced by the paper
- `verification_tools/`: lightweight utility package for the closed-form laws
- `tests/`: lightweight smoke tests for the utility package
- `requirements-repro.txt`: Python dependencies for the analysis scripts
- `reproduce_all.ps1`: one-command Windows reproduction runner

## What is intentionally not included

- Benchmark question text or answer choices
- Model weights
- Provider credentials or API keys
- Any institution-identifying metadata

The package redistributes only our code and derived evaluation outputs.

## Environment

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements-repro.txt
pip install -e .
```

The analysis scripts were written against a standard Python scientific stack and
do not require GPU access for figure/table regeneration from saved outputs.

## Exact commands

From the root of this artifact, the main paper outputs can be regenerated with:

```bash
python scripts/analyze_mmlu.py
python scripts/analyze_truthfulqa.py
python scripts/analyze_all_benchmarks.py
python scripts/exp_self_eval_zero.py
python scripts/exp_self_eval_permutation.py
python scripts/exp_leaderboard_noise.py
python scripts/exp_benchmark_demolition.py
python scripts/exp_named_model_comparison.py
python scripts/exp_active_real.py
python scripts/exp_compositional.py
python scripts/exp_pipeline_real.py
python scripts/exp_rigor_bootstrap.py
python scripts/exp_regulatory_impossibility.py
python scripts/exp_verification_horizon.py
python scripts/fig_sun_comparison.py
```

On Windows, the same sequence can be run with:

```powershell
powershell -ExecutionPolicy Bypass -File .\\reproduce_all.ps1
```

## Hosted-model reruns

The full hosted-model collection scripts are included only for transparency of
methodology. Re-running those scripts requires access to the same hosted-model
provider and a local `.env` file containing one or more `NVIDIA_NIM_API_KEY`
entries. The scripts read the `.env` file from the artifact root by default, or
from the path specified by the `NIM_ENV_FILE` environment variable.

Because the original experiments used a provider-managed inference API, exact
server-side GPU type, memory, and total wall-clock are not exposed through the
API interface. For that reason, this artifact focuses on figure/table
reproduction from saved outputs rather than claiming hardware-level reruns.

## Limitations

- This artifact is sufficient to verify the reported tables and figures from
  saved outputs, but it is not a full turnkey rerun of the hosted inference.
- Benchmark question text is intentionally excluded, so this package is not a
  benchmark redistribution.
- Some scripts generate the same output files as the checked-in JSON or LaTeX
  summaries; those saved files are included for cross-checking.

## Verification tools quick start

```python
from verification_tools import verification_floor, holdout_size, is_verifiable

floor = verification_floor(eps=0.15, n=14042, L=1.5)
needed = holdout_size(eps=0.15, delta=0.02, L=1.5)
verdict = is_verifiable(claimed_improvement=0.03, eps=0.15, n=14042, L=1.5)

print(floor)
print(needed)
print(verdict)
```

The utilities implement the closed-form scaling laws used in the paper:

- passive verification floor: `(L * eps / n) ** (1/3)`
- passive holdout size: `ceil(L * eps / delta**3)`
- active verification floor: `sqrt(eps / n)`
- phase transition threshold: `ceil(1 / eps)`
