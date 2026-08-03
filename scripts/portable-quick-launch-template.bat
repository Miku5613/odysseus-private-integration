@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

title Odysseus - Portable Quick Launch
echo.
echo ============================================================
echo  Odysseus Portable Quick Launch
echo ============================================================
echo  Running mandatory, optional, and reachability preflight...
echo.

set "VENVPY=venv\Scripts\python.exe"
set /a OPTIONAL_WARNINGS=0

echo [MANDATORY]
if not exist "app.py" (
  echo   [ABSENT] app.py
  goto :required_failed
)
if not exist "requirements.txt" (
  echo   [ABSENT] requirements.txt
  goto :required_failed
)
if not exist "static\index.html" (
  echo   [ABSENT] static\index.html
  goto :required_failed
)
if not exist "%VENVPY%" (
  echo   [ABSENT] venv\Scripts\python.exe
  echo   Run Express Installation.bat first.
  goto :required_failed
)

"%VENVPY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if errorlevel 1 (
  echo   [UNAVAILABLE] Python 3.11+ virtual environment
  goto :required_failed
)
echo   [READY] Python virtual environment

"%VENVPY%" scripts\dependency_status.py requirements requirements.txt
if errorlevel 1 (
  echo   [UNAVAILABLE] Mandatory Python dependencies
  echo   Run Express Installation.bat to repair them.
  goto :required_failed
)

"%VENVPY%" -c "from pathlib import Path; p=Path('data'); p.mkdir(exist_ok=True); q=p/'.odysseus-launch-check.tmp'; q.write_text('ok',encoding='utf-8'); q.unlink()" >nul 2>&1
if errorlevel 1 (
  echo   [UNAVAILABLE] Writable data directory
  goto :required_failed
)
echo   [READY] Writable data directory

"%VENVPY%" -c "from pathlib import Path; import sqlite3; p=Path('data/app.db'); c=sqlite3.connect(('file:' + p.resolve().as_posix() + '?mode=ro') if p.exists() else ':memory:', uri=p.exists()); c.execute('SELECT 1'); c.close()" >nul 2>&1
if errorlevel 1 (
  echo   [UNAVAILABLE] SQLite database
  goto :required_failed
)
echo   [READY] SQLite database

echo.
echo [OPTIONAL]
"%VENVPY%" scripts\dependency_status.py requirements requirements-langgraph.txt
if errorlevel 1 set /a OPTIONAL_WARNINGS+=1

"%VENVPY%" scripts\dependency_status.py opencode --installed-only
set "OPENCODE_STATE=!ERRORLEVEL!"
if "!OPENCODE_STATE!"=="10" set /a OPTIONAL_WARNINGS+=1
if "!OPENCODE_STATE!"=="20" set /a OPTIONAL_WARNINGS+=1

"%VENVPY%" scripts\dependency_status.py requirements requirements-optional.txt
if errorlevel 1 set /a OPTIONAL_WARNINGS+=1

"%VENVPY%" scripts\runtime_reachability.py chroma
if errorlevel 1 set /a OPTIONAL_WARNINGS+=1

"%VENVPY%" scripts\runtime_reachability.py model
if errorlevel 1 set /a OPTIONAL_WARNINGS+=1

if !OPTIONAL_WARNINGS! GTR 0 (
  echo.
  echo !OPTIONAL_WARNINGS! optional component check^(s^) are absent, outdated, or unreachable.
  choice /C YN /N /M "Do you still want to boot Odysseus without them? [Y/N] "
  if errorlevel 2 exit /b 2
)

echo.
echo [PORT]
"%VENVPY%" -c "import socket,urllib.request; url='http://127.0.0.1:7000/api/health'; running=False; occupied=False; exec(\"try:\\n urllib.request.urlopen(url,timeout=1).read(); running=True\\nexcept Exception: pass\"); s=socket.socket(); s.settimeout(1); occupied=(s.connect_ex(('127.0.0.1',7000))==0); s.close(); raise SystemExit(2 if running else (1 if occupied else 0))" >nul 2>&1
set "PORT_STATE=!ERRORLEVEL!"
if "!PORT_STATE!"=="2" (
  echo   [READY] Odysseus is already running at http://127.0.0.1:7000
  pause
  exit /b 0
)
if "!PORT_STATE!"=="1" (
  echo   [UNAVAILABLE] Port 7000 is occupied by another application.
  goto :required_failed
)
echo   [READY] Port 7000 is available

echo.
echo Starting Odysseus at http://127.0.0.1:7000
echo Press Ctrl+C to stop the server.
echo.
"%VENVPY%" -m uvicorn app:app --host 127.0.0.1 --port 7000
set "SERVER_EXIT=!ERRORLEVEL!"
echo.
echo Odysseus stopped with exit code !SERVER_EXIT!.
pause
exit /b !SERVER_EXIT!

:required_failed
echo.
echo Odysseus cannot start because the required dependency above is absent or unavailable.
echo The launcher will now close.
pause
exit /b 1
