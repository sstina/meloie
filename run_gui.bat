@echo off
chcp 65001 >nul
title Meloie GUI
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -NoProfile -File ".\run_gui.ps1" %*
echo.
echo ========================================
echo  GUI closed. Press any key to close.
echo ========================================
pause >nul
