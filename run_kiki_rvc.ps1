# run_kiki_rvc.ps1 — one-click launcher for the Kiki RVC realtime runtime.
# Launched by run_kiki_rvc.bat (which sets the console to UTF-8 and keeps the
# window open afterwards). Can also be run directly from a PowerShell prompt.

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

# Anchor to this script's folder (= RVC), independent of the caller's cwd, so
# the relative --config / --model-profile paths always resolve.
Set-Location -LiteralPath $PSScriptRoot

Write-Host "========================================"
Write-Host " Starting Kiki RVC Runtime"
Write-Host " Input  : Windows default microphone (系统默认麦克风)"
Write-Host " Output : CABLE Input"
Write-Host " App    : select 'CABLE Output' as its microphone"
Write-Host "========================================"
Write-Host ""

# Preflight: the runtime venv and env script must exist (check BEFORE loading
# setup_env.ps1, whose own venv activation would otherwise fail first).
$py = ".\.venv310\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "ERROR: runtime venv python not found at $py" -ForegroundColor Red
    Write-Host "Expected the runtime venv at .venv310 (Python 3.10 + the RVC stack)." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path ".\setup_env.ps1")) {
    Write-Host "ERROR: setup_env.ps1 not found next to this script." -ForegroundColor Red
    exit 1
}

# Load env: cache/temp redirection (keep C: untouched) + venv activation + UTF-8.
. .\setup_env.ps1

# Pause so you can switch your app's microphone to 'CABLE Output' first.
Write-Host ""
Write-Host "Set your app's (Discord/OBS/recorder) microphone to 'CABLE Output'." -ForegroundColor Yellow
Write-Host "After you press Enter, the model loads + warms up (~30 s on a cold" -ForegroundColor Yellow
Write-Host "start) BEFORE audio starts flowing - that pause is normal, not a hang." -ForegroundColor Yellow
Write-Host "Press Ctrl+C in this window to stop the runtime." -ForegroundColor Yellow
Read-Host "Press Enter to start"

# Use the venv python by absolute path (do not rely on PATH).
& $py -m src.main `
    --config config/runtime.example.json `
    --model-profile config/model_profiles/kiki.example.json `
    --device cuda
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "Runtime exited normally." -ForegroundColor Green
} else {
    Write-Host "Runtime exited with code $code - see the messages above." -ForegroundColor Red
    Write-Host "Common codes: 4=device not found, 11=model load failed (CUDA?), 3=feedback-loop guard." -ForegroundColor Red
}
