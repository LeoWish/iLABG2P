@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3.11 -m venv .venv 2>nul
if errorlevel 1 (
    python -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11 or later required'" 2>nul
    if errorlevel 1 (
        echo [ERROR] Install Python 3.11 or later from python.org first.
        exit /b 1
    )
    python -m venv .venv
    if errorlevel 1 exit /b 1
)
:install
set "REQ=requirements.txt"
if exist "requirements_portable_lock.txt" set "REQ=requirements_portable_lock.txt"
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r "%REQ%"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" verify_package.py --basic
if errorlevel 1 exit /b 1
exit /b 0
