# setup_env_applio.ps1 — environment bootstrap for the v2 runtime venv.
# Dot-source this BEFORE any pip / python / model-download command so EVERYTHING
# stays inside RVC\ and nothing is ever written to C: (CLAUDE.md §6, hard rule).
#
#   . .\setup_env_applio.ps1
#
# This activates .venv-applio (Python 3.10 + the modern, fairseq-free Applio
# inference stack: torch/cu128 + numpy 2.x + transformers + torchfcpe). It is
# the SOLE runtime venv (the program is v2-only).

$RVC = "D:\Users\Palovil\Desktop\Tvoice\RVC"

# --- cache / temp redirection (keep the C: drive untouched) ---------------
$env:PIP_CACHE_DIR          = "$RVC\.cache\pip"
$env:HF_HOME                = "$RVC\.cache\hf"
$env:HUGGINGFACE_HUB_CACHE  = "$RVC\.cache\hf"
$env:TRANSFORMERS_CACHE     = "$RVC\.cache\hf"
$env:TORCH_HOME             = "$RVC\.cache\torch"
$env:XDG_CACHE_HOME         = "$RVC\.cache"
$env:NUMBA_CACHE_DIR        = "$RVC\.cache\numba"
$env:TMP                    = "$RVC\.tmp"
$env:TEMP                   = "$RVC\.tmp"

# UTF-8 console so non-GBK device names print instead of crashing.
$env:PYTHONIOENCODING       = "utf-8"

# --- make sure the redirect targets exist ---------------------------------
foreach ($d in @("$RVC\.cache\pip", "$RVC\.cache\hf", "$RVC\.cache\torch",
                 "$RVC\.cache\numba", "$RVC\.tmp")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

# --- activate the v2 runtime venv (Python 3.10 + modern fairseq-free stack) -
& "$RVC\.venv-applio\Scripts\Activate.ps1"

Write-Host "RVC applio env ready: caches -> $RVC\.cache , temp -> $RVC\.tmp , venv -> .venv-applio" -ForegroundColor Green
