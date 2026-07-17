@echo off
REM Convenience launcher for the Personal Testing UI -- starts the real
REM trading API and this UI's local server together, each in its own
REM window, then opens the UI in your browser. Never touches engine code;
REM just runs the same two commands documented in README.md so you don't
REM have to open two terminals by hand every session.

setlocal
set ROOT=%~dp0..
set PY=%ROOT%\.venv\Scripts\python.exe

if not exist "%PY%" (
    echo Could not find %PY% -- is the venv set up at %ROOT%\.venv?
    pause
    exit /b 1
)

echo Starting trading API on http://127.0.0.1:8000 ...
start "Trading API (port 8000)" cmd /k ""%PY%" -m uvicorn api.main:app --port 8000"

echo Starting testing UI server on http://127.0.0.1:8765 ...
start "Testing UI server (port 8765)" cmd /k ""%PY%" "%~dp0server.py""

timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8765"

endlocal
