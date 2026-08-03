@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "NO_PAUSE=0"
if /I "%~1"=="/nopause" set "NO_PAUSE=1"

title Odysseus - Verify Files Integrity
echo.
echo ============================================================
echo  Odysseus startup readiness and dependency inventory
echo ============================================================
echo  This is a non-installing startup simulation. Its temporary
echo  writeability probe is removed immediately.
echo.

set /a REQUIRED_FAILURES=0
set "PYEXE="
set "PYARGS="

if exist "venv\Scripts\python.exe" (
  set "PYEXE=venv\Scripts\python.exe"
) else (
  py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "PYEXE=py"
    set "PYARGS=-3.11"
  ) else (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYEXE=python"
  )
)

echo [MANDATORY] Core startup path
if defined PYEXE (
  echo   [INSTALLED] Python 3.11+ interpreter
) else (
  echo   [NOT INSTALLED] Python 3.11+ interpreter
  set /a REQUIRED_FAILURES+=1
)

if exist "app.py" if exist "requirements.txt" if exist "static\index.html" (
  echo   [INSTALLED] Required server and UI files
) else (
  echo   [NOT INSTALLED] One or more required files: app.py, requirements.txt, static\index.html
  set /a REQUIRED_FAILURES+=1
)

if defined PYEXE (
  %PYEXE% %PYARGS% -c "import fastapi,uvicorn,sqlalchemy,bcrypt,httpx,dotenv,pydantic,pyotp; print('  [INSTALLED] Core Python dependency set')" 2>nul
  if errorlevel 1 (
    echo   [NOT INSTALLED] Core Python dependency set - run Express Installation.bat
    set /a REQUIRED_FAILURES+=1
  )

  %PYEXE% %PYARGS% -c "from pathlib import Path; p=Path('data'); p.mkdir(exist_ok=True); q=p/'.odysseus-write-check.tmp'; q.write_text('ok',encoding='utf-8'); q.unlink(); print('  [INSTALLED] Writable data directory')" 2>nul
  if errorlevel 1 (
    echo   [UNREACHABLE] Writable data directory
    set /a REQUIRED_FAILURES+=1
  )

  %PYEXE% %PYARGS% -c "from pathlib import Path; import sqlite3; p=Path('data/app.db'); c=sqlite3.connect(('file:' + p.resolve().as_posix() + '?mode=ro') if p.exists() else ':memory:', uri=p.exists()); c.execute('SELECT 1'); c.close(); print('  [INSTALLED] SQLite database can be opened')" 2>nul
  if errorlevel 1 (
    echo   [UNREACHABLE] SQLite database open check
    set /a REQUIRED_FAILURES+=1
  )
) else (
  echo   [UNREACHABLE] Python import and database startup checks
)

echo.
echo [OPTIONAL] Added runtime upgrades
if defined PYEXE (
  %PYEXE% %PYARGS% -c "import langgraph,langchain_core; from langgraph.checkpoint.sqlite import SqliteSaver; print('  [INSTALLED] LangGraph + LangChain + SQLite checkpoints')" 2>nul
  if errorlevel 1 echo   [FALLBACK] LangGraph runtime Not Installed - Legacy remains available
) else (
  echo   [UNREACHABLE] LangGraph runtime check
)

set "OPENCODE_FOUND="
for /f "delims=" %%I in ('where opencode.cmd 2^>nul') do if not defined OPENCODE_FOUND set "OPENCODE_FOUND=%%I"
if not defined OPENCODE_FOUND if exist "%APPDATA%\npm\opencode.cmd" set "OPENCODE_FOUND=%APPDATA%\npm\opencode.cmd"
if not defined OPENCODE_FOUND (
  echo   [FALLBACK] OpenCode CLI Not Installed - Legacy remains available
) else (
  echo   [INSTALLED] OpenCode CLI: !OPENCODE_FOUND!
)

echo.
echo [OPTIONAL] Supporting tools and feature packs
where git >nul 2>&1 && (echo   [INSTALLED] Git) || (echo   [NOT INSTALLED] Git - needed only to clone or update)
where npm >nul 2>&1 && (echo   [INSTALLED] Node.js/npm for OpenCode maintenance) || (echo   [FALLBACK] Node.js/npm Not Installed)
if exist "requirements-optional.txt" (
  if defined PYEXE (
    %PYEXE% %PYARGS% -c "import importlib.util; mods=('faster_whisper','ddgs','fitz','markitdown'); raise SystemExit(0 if all(importlib.util.find_spec(x) for x in mods) else 1)" >nul 2>&1
    if errorlevel 1 (echo   [FALLBACK] Upstream optional feature pack is incomplete) else (echo   [INSTALLED] Upstream optional feature pack)
  )
)
where docker >nul 2>&1 && (echo   [INSTALLED] Docker - alternative deployment path) || (echo   [FALLBACK] Docker Not Installed - native launch remains available)
if defined PYEXE (
  %PYEXE% %PYARGS% scripts\runtime_reachability.py chroma >nul 2>&1 && (echo   [REACHABLE] ChromaDB service) || (echo   [FALLBACK] ChromaDB service Unreachable)
  %PYEXE% %PYARGS% scripts\runtime_reachability.py model >nul 2>&1 && (echo   [REACHABLE] Local model endpoint) || (echo   [CONFIGURATION] Local model endpoint Unreachable)
)
echo   [CONFIGURATION] A model provider is still required for AI responses.
echo                   Model libraries are user-managed and never installed here.

echo.
if %REQUIRED_FAILURES% EQU 0 (
  echo RESULT: [READY] Mandatory native-server startup checks passed.
) else (
  echo RESULT: [NOT READY] %REQUIRED_FAILURES% mandatory check^(s^) failed.
)
echo Optional failures do not prevent Legacy runtime startup.
echo.
if "%NO_PAUSE%"=="0" pause
exit /b %REQUIRED_FAILURES%
