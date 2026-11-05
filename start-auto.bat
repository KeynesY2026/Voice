@echo off
setlocal EnableExtensions DisableDelayedExpansion

cd /d "%~dp0" || (
    echo Failed to open the project directory.
    pause
    exit /b 1
)

if exist ".venv-gpu\Scripts\python.exe" (
    set "UV_PROJECT_ENVIRONMENT=.venv-gpu"
    set "VOICE_EXTRA=gpu"
) else if exist ".venv\Scripts\python.exe" (
    set "UV_PROJECT_ENVIRONMENT=.venv"
    set "VOICE_EXTRA=cpu"
    for /d %%D in (".venv\Lib\site-packages\torch-*+cu*.dist-info") do set "VOICE_EXTRA=gpu"
) else (
    echo No installed Python environment was found.
    echo First run: uv sync --extra gpu --group dev
    echo Or for CPU: uv sync --extra cpu --group dev
    pause
    exit /b 1
)

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found on PATH. Install it from https://docs.astral.sh/uv/
    pause
    exit /b 1
)

echo Starting Local Voice Reader with automatic device selection.
echo Environment: %UV_PROJECT_ENVIRONMENT% ^(%VOICE_EXTRA%^)
echo Open http://127.0.0.1:7860 after the server starts.
echo Keep this window open while using the app; close it to stop.
echo.

call uv run --extra %VOICE_EXTRA% voice-reader-web --device auto
set "VOICE_EXIT_CODE=%ERRORLEVEL%"
if not "%VOICE_EXIT_CODE%"=="0" (
    echo.
    echo Startup failed with exit code %VOICE_EXIT_CODE%.
    pause
)
exit /b %VOICE_EXIT_CODE%
