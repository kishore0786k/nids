@echo off
setlocal
title Neuro-Symbolic NIDS - Full Pipeline

cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo ============================================
echo  Launch Neuro-Symbolic NIDS backend + dashboard
echo ============================================
cd backend
"%PYTHON%" app.py

pause
