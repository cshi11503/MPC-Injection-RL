# Launch one training run in its own titled PowerShell window.
# Each run also writes results/runs/<run_name>/stdout.log (UTF-8).
#
# Usage (from repo root):
#   .\experiments\run_job.ps1 -Title "baseline seed1" -TrainArgs "--seed 1 --threads 2"
#   .\experiments\run_job.ps1 -Title "r1 seed1" -TrainArgs "--obstacle-radius 1.0 --seed 1 --threads 2"
#
# Several runs in parallel: call it once per run. ~6 runs x 2 threads is the
# measured sweet spot on this laptop (each run ~1.7x slower, ~3.6x total throughput).
param(
    [Parameter(Mandatory = $true)][string]$Title,
    [Parameter(Mandatory = $true)][string]$TrainArgs
)
$repo = Split-Path -Parent $PSScriptRoot
$cmd = "Set-Location '$repo'; `$host.UI.RawUI.WindowTitle = '$Title'; python -u -m training.train $TrainArgs"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd
Write-Host "Started '$Title': python -m training.train $TrainArgs"
