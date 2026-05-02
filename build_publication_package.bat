@echo off
setlocal
title Build Neuro-Symbolic NIDS IEEE Package

cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo ============================================
echo  Generate publication package
echo ============================================
"%PYTHON%" -m backend.generate_publication_package

echo.
echo Outputs:
echo  - results\publication_package
echo  - paper\generated
echo  - paper\figures\generated
pause
