# run_gui.ps1 — launch the Meloie GUI (PySide6 + QML, RVC engine) in .venv-applio.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "========================================"
Write-Host " Meloie GUI (PySide6 + QML)"
Write-Host " Model -> realtime RVC -> CABLE Input"
Write-Host "========================================"

$py = ".\.venv-applio\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "ERROR: .venv-applio python not found at $py" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path ".\setup_env_applio.ps1")) {
    Write-Host "ERROR: setup_env_applio.ps1 not found." -ForegroundColor Red
    exit 1
}

# cache/temp redirection (keep C: untouched) + activate .venv-applio
. .\setup_env_applio.ps1

& $py -m src.ui @args
$code = $LASTEXITCODE
Write-Host ""
if ($code -ne 0) {
    Write-Host "GUI exited with code $code - see messages above." -ForegroundColor Red
}
