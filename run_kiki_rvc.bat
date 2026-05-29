@echo off
chcp 65001 >nul
title Kiki RVC Runtime
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -NoProfile -File ".\run_kiki_rvc.ps1"
echo.
echo ========================================
echo  Runtime stopped. Press any key to close.
echo ========================================
pause >nul
