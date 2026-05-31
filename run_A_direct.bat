@echo off
chcp 65001 >nul
title A RVC Runtime (DIRECT / Path-A)
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -NoProfile -File ".\run_A_direct.ps1" %*
echo.
echo ========================================
echo  Runtime stopped. Press any key to close.
echo ========================================
pause >nul
