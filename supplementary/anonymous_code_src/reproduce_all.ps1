$ErrorActionPreference = "Stop"

$scripts = @(
    "scripts/analyze_mmlu.py",
    "scripts/analyze_truthfulqa.py",
    "scripts/analyze_all_benchmarks.py",
    "scripts/exp_self_eval_zero.py",
    "scripts/exp_self_eval_permutation.py",
    "scripts/exp_leaderboard_noise.py",
    "scripts/exp_benchmark_demolition.py",
    "scripts/exp_named_model_comparison.py",
    "scripts/exp_active_real.py",
    "scripts/exp_compositional.py",
    "scripts/exp_pipeline_real.py",
    "scripts/exp_rigor_bootstrap.py",
    "scripts/exp_regulatory_impossibility.py",
    "scripts/exp_verification_horizon.py",
    "scripts/fig_sun_comparison.py"
)

foreach ($script in $scripts) {
    Write-Host "`n==> python $script"
    & python $script
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: python $script"
    }
}
