@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [SETUP] Preparing Python environment on first launch...
    call "%~dp0setup_windows.bat"
    if errorlevel 1 (
        echo [ERROR] Python dependencies could not be set up.
        pause
        exit /b 1
    )
)
"%PY%" "%~dp0launcher.py"
if errorlevel 1 echo [ERROR] Please inspect the error above.
pause
