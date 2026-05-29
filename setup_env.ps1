# setup_env.ps1 — dot-source this at the start of every session BEFORE any
# pip / python / model-download command, so EVERYTHING stays inside RVC\
# and nothing is ever written to the C: drive (CLAUDE.md §6, hard rule).
#
#   . .\setup_env.ps1
#
# It redirects all package / model / temp caches into RVC\.cache and RVC\.tmp,
# forces UTF-8 console output (so device names with non-GBK characters do not
# crash --list-devices on a CN locale), and activates the runtime venv.

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

# --- activate the runtime venv (Python 3.10 + the full RVC stack) ---------
& "$RVC\.venv310\Scripts\Activate.ps1"

Write-Host "RVC env ready: caches -> $RVC\.cache , temp -> $RVC\.tmp , venv -> .venv310" -ForegroundColor Green
