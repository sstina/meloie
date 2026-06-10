# run_A_direct.ps1 — DIRECT engine launcher for model 'A' (v2 / 768-dim,
# models/A.pth + models/V2.index). The v2-only realtime entry point: default
# mic -> Applio persistent-buffer engine (.venv-applio) -> CABLE Input, with
# the model profile config/model_profiles/A.json.

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "========================================"
Write-Host " Starting A RVC Runtime (DIRECT / Path-A)"
Write-Host " Model  : models/A.pth  (v2, +index models/V2.index)"
Write-Host " Input  : Windows default microphone"
Write-Host " Output : CABLE Input   (app mic -> CABLE Output)"
Write-Host " Engine : direct (Applio persistent-buffer, .venv-applio)"
Write-Host "========================================"
Write-Host ""

$py = ".\.venv-applio\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "ERROR: .venv-applio python not found at $py" -ForegroundColor Red
    Write-Host "Build it first (the Path-A migration venv)." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path ".\setup_env_applio.ps1")) {
    Write-Host "ERROR: setup_env_applio.ps1 not found." -ForegroundColor Red
    exit 1
}

# Load env: cache/temp redirection (keep C: untouched) + activate .venv-applio.
. .\setup_env_applio.ps1

Write-Host ""
Write-Host "Set your app's (Discord/OBS/recorder) microphone to 'CABLE Output'." -ForegroundColor Yellow
Write-Host "After Enter, the model loads + warms up (~30 s cold) BEFORE audio flows." -ForegroundColor Yellow
Write-Host "Ctrl+C to stop. Tune with: --direct-block-ms / --direct-context-ms / --direct-crossfade-ms." -ForegroundColor Yellow

# 输入降噪 / Input denoise — manual choice each launch. Skipped if the caller
# already passed --direct-denoise or --no-direct-denoise via @args (their flag wins).
$denoiseArg = @()
$hasDenoiseChoice = $false
foreach ($a in $args) {
    if ($a -eq "--direct-denoise" -or $a -eq "--no-direct-denoise") { $hasDenoiseChoice = $true }
}
if (-not $hasDenoiseChoice) {
    Write-Host ""
    Write-Host "输入降噪 / Input denoise - reduce ambient noise BEFORE conversion." -ForegroundColor Cyan
    Write-Host "  y = ON  : cleaner in a noisy room (can muffle very soft speech)" -ForegroundColor DarkGray
    Write-Host "  n = OFF : most faithful to your raw mic  [default, best for a quiet mic]" -ForegroundColor DarkGray
    $ans = Read-Host "Enable input denoise? (y/N)"
    if ($ans -match "^(y|yes)$") {
        $denoiseArg = @("--direct-denoise")
        Write-Host "  -> input denoise ON" -ForegroundColor Green
    } else {
        $denoiseArg = @("--no-direct-denoise")
        Write-Host "  -> input denoise OFF" -ForegroundColor Green
    }
}

Read-Host "Press Enter to start"

# Confirmed-good defaults: fcpe (smoother + ~30% faster F0 on this stack).
# Input denoise is chosen interactively above (default OFF). A's profile sets
# index_rate=0.0 (V2.index NOT loaded; raw features sounded best) + pitch_shift=12
# (seller range 10-14). Override via @args, e.g.
#   run_A_direct.bat --direct-denoise               (force denoise ON, skip prompt)
#   run_A_direct.bat --direct-denoise-strength 0.8  (if ON: clean harder)
#   run_A_direct.bat --pitch 14                     (one-off transpose +14)
#   run_A_direct.bat --direct-silence-dbfs -50      (silence gate / 响应阈值)
& $py -m meloie.main `
    --config config/runtime.example.json `
    --model-profile config/model_profiles/A.json `
    --device cuda --direct-f0 fcpe @denoiseArg @args
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "Runtime exited normally." -ForegroundColor Green
} else {
    Write-Host "Runtime exited with code $code - see messages above." -ForegroundColor Red
    Write-Host "Codes: 3=feedback-loop guard, 4=device not found, 5=runtime, 11=engine load." -ForegroundColor Red
}
