@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

title Odysseus - Express Installation
echo.
echo ============================================================
echo  Odysseus Express Installation
echo ============================================================
echo  Installed and compatible dependencies are skipped.
echo  Missing optional components use an INSTALL confirmation.
echo  Incompatible or outdated components use an UPDATE confirmation.
echo  Ollama, LM Studio, and other model libraries are never offered.
echo.

set "BASEPY="
set "BASEARGS="
py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "BASEPY=py"
  set "BASEARGS=-3.11"
) else (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
  if not errorlevel 1 set "BASEPY=python"
)

if not defined BASEPY (
  where winget >nul 2>&1
  if errorlevel 1 goto :python_missing
  echo Installing mandatory Python 3.11...
  winget install --id Python.Python.3.11 -e --accept-package-agreements --accept-source-agreements
  if errorlevel 1 goto :failed
  set "BASEPY=py"
  set "BASEARGS=-3.11"
)

if not exist "venv\Scripts\python.exe" (
  echo.
  echo Creating the mandatory Odysseus virtual environment...
  %BASEPY% %BASEARGS% -m venv venv
  if errorlevel 1 goto :failed
) else (
  echo [skip] Odysseus virtual environment already exists.
)
set "VENVPY=venv\Scripts\python.exe"

echo.
echo Checking mandatory Python dependencies...
"%VENVPY%" scripts\dependency_status.py requirements requirements.txt
set "CORE_STATE=!ERRORLEVEL!"
if "!CORE_STATE!"=="0" (
  echo [skip] Mandatory Python dependencies are installed and compatible.
) else (
  if "!CORE_STATE!"=="20" (
    echo Updating mandatory Python dependencies to the required versions...
  ) else (
    echo Installing missing mandatory Python dependencies...
  )
  "%VENVPY%" -m pip install -r requirements.txt
  if errorlevel 1 goto :failed
)

echo.
echo Running safe first-time setup and migrations...
"%VENVPY%" setup.py
if errorlevel 1 goto :failed

echo.
echo Checking optional LangGraph runtime...
"%VENVPY%" scripts\dependency_status.py requirements requirements-langgraph.txt
set "LANGGRAPH_STATE=!ERRORLEVEL!"
if "!LANGGRAPH_STATE!"=="0" (
  echo [skip] LangGraph/LangChain runtime is already installed and compatible.
) else if "!LANGGRAPH_STATE!"=="20" (
  choice /C YN /N /M "Do you want to update the current version of LangGraph/LangChain? [Y/N] "
  if not errorlevel 2 (
    "%VENVPY%" -m pip install --upgrade -r requirements-langgraph.txt
    if errorlevel 1 echo WARNING: LangGraph update failed; Legacy remains available.
  )
) else (
  choice /C YN /N /M "Install the missing optional LangGraph/LangChain runtime? [Y/N] "
  if not errorlevel 2 (
    "%VENVPY%" -m pip install -r requirements-langgraph.txt
    if errorlevel 1 echo WARNING: LangGraph installation failed; Legacy remains available.
  )
)

echo.
echo Checking optional OpenCode runtime...
"%VENVPY%" scripts\dependency_status.py opencode
set "OPENCODE_STATE=!ERRORLEVEL!"
if "!OPENCODE_STATE!"=="0" (
  echo [skip] OpenCode is installed and current.
) else if "!OPENCODE_STATE!"=="30" (
  echo [skip] OpenCode is installed; the update registry was unavailable.
) else if "!OPENCODE_STATE!"=="20" (
  choice /C YN /N /M "Do you want to update the current version of OpenCode? [Y/N] "
  if not errorlevel 2 call :install_opencode update
) else (
  choice /C YN /N /M "Install the missing optional OpenCode CLI runtime? [Y/N] "
  if not errorlevel 2 call :install_opencode install
)

echo.
echo Checking upstream optional feature extras...
"%VENVPY%" scripts\dependency_status.py requirements requirements-optional.txt
set "EXTRAS_STATE=!ERRORLEVEL!"
if "!EXTRAS_STATE!"=="0" (
  echo [skip] Optional document, speech, search, and PDF extras are installed.
) else if "!EXTRAS_STATE!"=="20" (
  choice /C YN /N /M "Do you want to update the current optional feature extras? [Y/N] "
  if not errorlevel 2 (
    "%VENVPY%" -m pip install --upgrade -r requirements-optional.txt
    if errorlevel 1 echo WARNING: Some optional extras failed to update.
  )
) else (
  choice /C YN /N /M "Install the missing optional document, speech, search, and PDF extras? [Y/N] "
  if not errorlevel 2 (
    "%VENVPY%" -m pip install -r requirements-optional.txt
    if errorlevel 1 echo WARNING: Some optional extras failed to install.
  )
)

echo.
echo Express installation finished. Running the readiness inventory...
call "%~dp0Verify Files Integrity.bat" /nopause
if errorlevel 1 goto :failed

echo.
set "ODYSSEUS_PROJECT_ROOT=%~dp0"
set "QUICK_LAUNCH=%~dp0Quick Launch Odysseus.bat"
set "QUICK_CANDIDATE=%TEMP%\odysseus_quick_launch_%RANDOM%_%RANDOM%.bat"

"%VENVPY%" -c "import base64,pathlib,os; pathlib.Path(r'!QUICK_CANDIDATE!').write_bytes(base64.b64decode('QGVjaG8gb2ZmCnNldGxvY2FsIEVuYWJsZUV4dGVuc2lvbnMgRW5hYmxlRGVsYXllZEV4cGFuc2lvbgpzZXQgIlBST0pFQ1RfUk9PVD1fX09EWVNTRVVTX1BST0pFQ1RfUk9PVF9fIgppZiBub3QgZXhpc3QgIiVQUk9KRUNUX1JPT1QlXGxhdW5jaC13aW5kb3dzLnBzMSIgKAogIGVjaG8uCiAgZWNobyBbQUJTRU5UXSBTdG9yZWQgT2R5c3NldXMgcHJvamVjdCBmb2xkZXI6ICVQUk9KRUNUX1JPT1QlCiAgZWNobyBUaGUgcHJvamVjdCBtYXkgaGF2ZSBiZWVuIG1vdmVkLiBSdW4gRXhwcmVzcyBJbnN0YWxsYXRpb24uYmF0IGFnYWluIGZyb20gdGhlIHByb2plY3Qgcm9vdC4KICBlY2hvLgogIHBhdXNlCiAgZXhpdCAvYiAxCikKY2QgL2QgIiVQUk9KRUNUX1JPT1QlIgoKdGl0bGUgT2R5c3NldXMgLSBQb3J0YWJsZSBRdWljayBMYXVuY2gKZWNoby4KZWNobyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KZWNobyAgT2R5c3NldXMgUG9ydGFibGUgUXVpY2sgTGF1bmNoCmVjaG8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CmVjaG8gIENoZWNraW5nIHRoZSBjb25maWd1cmVkIE9keXNzZXVzIHByb2plY3QsIGRlcGVuZGVuY2llcywgYW5kIHNlcnZpY2VzLi4uCmVjaG8uCgpzZXQgIlZFTlZQWT12ZW52XFNjcmlwdHNccHl0aG9uLmV4ZSIKc2V0IC9hIE9QVElPTkFMX1dBUk5JTkdTPTAKCmVjaG8gW01BTkRBVE9SWV0KaWYgbm90IGV4aXN0ICJsYXVuY2gtd2luZG93cy5wczEiICgKICBlY2hvICAgW0FCU0VOVF0gbGF1bmNoLXdpbmRvd3MucHMxCiAgZ290byA6cmVxdWlyZWRfZmFpbGVkCikKZWNobyAgIFtSRUFEWV0gRGVkaWNhdGVkIGxhdW5jaGVyOiBsYXVuY2gtd2luZG93cy5wczEKCmlmIG5vdCBleGlzdCAiYXBwLnB5IiAoCiAgZWNobyAgIFtBQlNFTlRdIGFwcC5weQogIGdvdG8gOnJlcXVpcmVkX2ZhaWxlZAopCmlmIG5vdCBleGlzdCAicmVxdWlyZW1lbnRzLnR4dCIgKAogIGVjaG8gICBbQUJTRU5UXSByZXF1aXJlbWVudHMudHh0CiAgZ290byA6cmVxdWlyZWRfZmFpbGVkCikKaWYgbm90IGV4aXN0ICJzdGF0aWNcaW5kZXguaHRtbCIgKAogIGVjaG8gICBbQUJTRU5UXSBzdGF0aWNcaW5kZXguaHRtbAogIGdvdG8gOnJlcXVpcmVkX2ZhaWxlZAopCmVjaG8gICBbUkVBRFldIFJlcXVpcmVkIE9keXNzZXVzIHNlcnZlciBhbmQgVUkgZmlsZXMKCmlmIG5vdCBleGlzdCAiJVZFTlZQWSUiICgKICBlY2hvICAgW0FCU0VOVF0gdmVudlxTY3JpcHRzXHB5dGhvbi5leGUKICBlY2hvICAgUnVuIEV4cHJlc3MgSW5zdGFsbGF0aW9uLmJhdCBmaXJzdC4KICBnb3RvIDpyZXF1aXJlZF9mYWlsZWQKKQoKIiVWRU5WUFklIiAtYyAiaW1wb3J0IHN5czsgcmFpc2UgU3lzdGVtRXhpdCgwIGlmIHN5cy52ZXJzaW9uX2luZm8gPj0gKDMsMTEpIGVsc2UgMSkiID5udWwgMj4mMQppZiBlcnJvcmxldmVsIDEgKAogIGVjaG8gICBbVU5BVkFJTEFCTEVdIFB5dGhvbiAzLjExKyB2aXJ0dWFsIGVudmlyb25tZW50CiAgZ290byA6cmVxdWlyZWRfZmFpbGVkCikKZWNobyAgIFtSRUFEWV0gUHl0aG9uIHZpcnR1YWwgZW52aXJvbm1lbnQKCiIlVkVOVlBZJSIgc2NyaXB0c1xkZXBlbmRlbmN5X3N0YXR1cy5weSByZXF1aXJlbWVudHMgcmVxdWlyZW1lbnRzLnR4dCA+bnVsIDI+JjEKaWYgZXJyb3JsZXZlbCAxICgKICBlY2hvICAgW1VOQVZBSUxBQkxFXSBNYW5kYXRvcnkgUHl0aG9uIGRlcGVuZGVuY2llcwogIGVjaG8gICBSdW4gRXhwcmVzcyBJbnN0YWxsYXRpb24uYmF0IHRvIHJlcGFpciB0aGVtLgogIGdvdG8gOnJlcXVpcmVkX2ZhaWxlZAopCmVjaG8gICBbUkVBRFldIE1hbmRhdG9yeSBQeXRob24gZGVwZW5kZW5jaWVzCgoiJVZFTlZQWSUiIC1jICJmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGg7IHA9UGF0aCgnZGF0YScpOyBwLm1rZGlyKGV4aXN0X29rPVRydWUpOyBxPXAvJy5vZHlzc2V1cy1sYXVuY2gtY2hlY2sudG1wJzsgcS53cml0ZV90ZXh0KCdvaycsZW5jb2Rpbmc9J3V0Zi04Jyk7IHEudW5saW5rKCkiID5udWwgMj4mMQppZiBlcnJvcmxldmVsIDEgKAogIGVjaG8gICBbVU5BVkFJTEFCTEVdIFdyaXRhYmxlIGRhdGEgZGlyZWN0b3J5CiAgZ290byA6cmVxdWlyZWRfZmFpbGVkCikKZWNobyAgIFtSRUFEWV0gV3JpdGFibGUgZGF0YSBkaXJlY3RvcnkKCiIlVkVOVlBZJSIgLWMgImZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aDsgaW1wb3J0IHNxbGl0ZTM7IHA9UGF0aCgnZGF0YS9hcHAuZGInKTsgYz1zcWxpdGUzLmNvbm5lY3QoKCdmaWxlOicgKyBwLnJlc29sdmUoKS5hc19wb3NpeCgpICsgJz9tb2RlPXJvJykgaWYgcC5leGlzdHMoKSBlbHNlICc6bWVtb3J5OicsIHVyaT1wLmV4aXN0cygpKTsgYy5leGVjdXRlKCdTRUxFQ1QgMScpOyBjLmNsb3NlKCkiID5udWwgMj4mMQppZiBlcnJvcmxldmVsIDEgKAogIGVjaG8gICBbVU5BVkFJTEFCTEVdIFNRTGl0ZSBkYXRhYmFzZQogIGdvdG8gOnJlcXVpcmVkX2ZhaWxlZAopCmVjaG8gICBbUkVBRFldIFNRTGl0ZSBkYXRhYmFzZQoKd2hlcmUgcG93ZXJzaGVsbC5leGUgPm51bCAyPiYxCmlmIGVycm9ybGV2ZWwgMSAoCiAgZWNobyAgIFtVTkFWQUlMQUJMRV0gV2luZG93cyBQb3dlclNoZWxsCiAgZ290byA6cmVxdWlyZWRfZmFpbGVkCikKZWNobyAgIFtSRUFEWV0gV2luZG93cyBQb3dlclNoZWxsCgplY2hvLgplY2hvIFtPUFRJT05BTF0KCiIlVkVOVlBZJSIgc2NyaXB0c1xkZXBlbmRlbmN5X3N0YXR1cy5weSByZXF1aXJlbWVudHMgcmVxdWlyZW1lbnRzLWxhbmdncmFwaC50eHQgPm51bCAyPiYxCmlmIGVycm9ybGV2ZWwgMSAoCiAgZWNobyAgIFtVTkFWQUlMQUJMRV0gTGFuZ0dyYXBoL0xhbmdDaGFpbiBydW50aW1lIC0gTGVnYWN5IHJ1bnRpbWUgcmVtYWlucyBhdmFpbGFibGUKICBjYWxsIDphZGRfb3B0aW9uYWwgIkxhbmdHcmFwaC9MYW5nQ2hhaW4gcnVudGltZSIKKSBlbHNlICgKICBlY2hvICAgW1JFQURZXSBMYW5nR3JhcGgvTGFuZ0NoYWluIHJ1bnRpbWUKKQoKIiVWRU5WUFklIiBzY3JpcHRzXGRlcGVuZGVuY3lfc3RhdHVzLnB5IG9wZW5jb2RlIC0taW5zdGFsbGVkLW9ubHkgPm51bCAyPiYxCnNldCAiT1BFTkNPREVfU1RBVEU9IUVSUk9STEVWRUwhIgppZiAiIU9QRU5DT0RFX1NUQVRFISI9PSIwIiAoCiAgZWNobyAgIFtSRUFEWV0gT3BlbkNvZGUgQ0xJIHJ1bnRpbWUKKSBlbHNlIGlmICIhT1BFTkNPREVfU1RBVEUhIj09IjIwIiAoCiAgZWNobyAgIFtPVVREQVRFRF0gT3BlbkNvZGUgQ0xJIHJ1bnRpbWUKICBjYWxsIDphZGRfb3B0aW9uYWwgIk9wZW5Db2RlIENMSSBydW50aW1lIGlzIG91dGRhdGVkIgopIGVsc2UgKAogIGVjaG8gICBbQUJTRU5UXSBPcGVuQ29kZSBDTEkgcnVudGltZSAtIExlZ2FjeSBydW50aW1lIHJlbWFpbnMgYXZhaWxhYmxlCiAgY2FsbCA6YWRkX29wdGlvbmFsICJPcGVuQ29kZSBDTEkgcnVudGltZSIKKQoKIiVWRU5WUFklIiBzY3JpcHRzXGRlcGVuZGVuY3lfc3RhdHVzLnB5IHJlcXVpcmVtZW50cyByZXF1aXJlbWVudHMtb3B0aW9uYWwudHh0ID5udWwgMj4mMQppZiBlcnJvcmxldmVsIDEgKAogIGVjaG8gICBbSU5DT01QTEVURV0gT3B0aW9uYWwgZG9jdW1lbnQsIHNwZWVjaCwgc2VhcmNoLCBhbmQgUERGIGZlYXR1cmUgcGFjawogIGNhbGwgOmFkZF9vcHRpb25hbCAiT3B0aW9uYWwgZG9jdW1lbnQvc3BlZWNoL3NlYXJjaC9QREYgZmVhdHVyZSBwYWNrIgopIGVsc2UgKAogIGVjaG8gICBbUkVBRFldIE9wdGlvbmFsIGRvY3VtZW50LCBzcGVlY2gsIHNlYXJjaCwgYW5kIFBERiBmZWF0dXJlIHBhY2sKKQoKIiVWRU5WUFklIiBzY3JpcHRzXHJ1bnRpbWVfcmVhY2hhYmlsaXR5LnB5IGNocm9tYSA+bnVsIDI+JjEKaWYgZXJyb3JsZXZlbCAxICgKICBlY2hvICAgW1VOUkVBQ0hBQkxFXSBDaHJvbWFEQiBzZXJ2aWNlIGF0IGxvY2FsaG9zdDo4MTAwCiAgY2FsbCA6YWRkX29wdGlvbmFsICJDaHJvbWFEQiBzZXJ2aWNlIGF0IGxvY2FsaG9zdDo4MTAwIgopIGVsc2UgKAogIGVjaG8gICBbUkVBQ0hBQkxFXSBDaHJvbWFEQiBzZXJ2aWNlIGF0IGxvY2FsaG9zdDo4MTAwCikKCiIlVkVOVlBZJSIgc2NyaXB0c1xydW50aW1lX3JlYWNoYWJpbGl0eS5weSBtb2RlbCA+bnVsIDI+JjEKaWYgZXJyb3JsZXZlbCAxICgKICBlY2hvICAgW1VOUkVBQ0hBQkxFXSBDb25maWd1cmVkIGxvY2FsIG1vZGVsIGVuZHBvaW50CiAgY2FsbCA6YWRkX29wdGlvbmFsICJDb25maWd1cmVkIGxvY2FsIG1vZGVsIGVuZHBvaW50IgopIGVsc2UgKAogIGVjaG8gICBbUkVBQ0hBQkxFXSBDb25maWd1cmVkIGxvY2FsIG1vZGVsIGVuZHBvaW50CikKCmlmICFPUFRJT05BTF9XQVJOSU5HUyEgR1RSIDAgKAogIGVjaG8uCiAgZWNobyBUaGUgZm9sbG93aW5nIG9wdGlvbmFsIGNvbXBvbmVudF4oc14pIGFyZSB1bmF2YWlsYWJsZSwgb3V0ZGF0ZWQsIG9yIHVucmVhY2hhYmxlOgogIGZvciAvTCAlJU4gaW4gKDEsMSwhT1BUSU9OQUxfV0FSTklOR1MhKSBkbyBlY2hvICAgLSAhT1BUSU9OQUxfJSVOIQogIGVjaG8uCiAgY2hvaWNlIC9DIFlOIC9OIC9NICJEbyB5b3Ugc3RpbGwgd2FudCB0byBib290IE9keXNzZXVzIHdpdGhvdXQgdGhlbT8gW1kvTl0gIgogIGlmIGVycm9ybGV2ZWwgMiBleGl0IC9iIDIKKQoKZWNoby4KZWNobyBbTEFVTkNIXQplY2hvICAgW1JFQURZXSBQcm9qZWN0IGRpcmVjdG9yeTogJUNEJQplY2hvICAgU3RhcnRpbmcgdGhlIGRlZGljYXRlZCBPZHlzc2V1cyBsYXVuY2hlci4uLgplY2hvLgoKcG93ZXJzaGVsbC5leGUgLU5vUHJvZmlsZSAtRXhlY3V0aW9uUG9saWN5IEJ5cGFzcyAtRmlsZSAiJUNEJVxsYXVuY2gtd2luZG93cy5wczEiCnNldCAiTEFVTkNIX0VYSVQ9IUVSUk9STEVWRUwhIgoKZWNoby4KaWYgIiFMQVVOQ0hfRVhJVCEiPT0iMCIgKAogIGVjaG8gT2R5c3NldXMgbGF1bmNoZXIgZmluaXNoZWQgc3VjY2Vzc2Z1bGx5LgopIGVsc2UgKAogIGVjaG8gT2R5c3NldXMgbGF1bmNoZXIgc3RvcHBlZCB3aXRoIGV4aXQgY29kZSAhTEFVTkNIX0VYSVQhLgopCnBhdXNlCmV4aXQgL2IgIUxBVU5DSF9FWElUIQoKOmFkZF9vcHRpb25hbApzZXQgL2EgT1BUSU9OQUxfV0FSTklOR1MrPTEKc2V0ICJPUFRJT05BTF8hT1BUSU9OQUxfV0FSTklOR1MhPSV+MSIKZXhpdCAvYiAwCgo6cmVxdWlyZWRfZmFpbGVkCmVjaG8uCmVjaG8gT2R5c3NldXMgY2Fubm90IHN0YXJ0IGJlY2F1c2UgdGhlIG1hbmRhdG9yeSBpdGVtIGlkZW50aWZpZWQgYWJvdmUgaXMgYWJzZW50IG9yIHVuYXZhaWxhYmxlLgplY2hvIFJ1biBFeHByZXNzIEluc3RhbGxhdGlvbi5iYXQgdG8gcmVwYWlyIGluc3RhbGxhYmxlIGRlcGVuZGVuY2llcy4KZWNoby4KcGF1c2UKZXhpdCAvYiAxCg==').decode('utf-8').replace('__ODYSSEUS_PROJECT_ROOT__', os.environ['ODYSSEUS_PROJECT_ROOT'].rstrip('\\/')).encode('utf-8'))"
if errorlevel 1 (
  echo WARNING: Could not prepare the portable Quick Launch file.
  goto :quick_done
)

if exist "!QUICK_LAUNCH!" (
  fc /b "!QUICK_CANDIDATE!" "!QUICK_LAUNCH!" >nul 2>&1
  if not errorlevel 1 (
    echo [skip] Quick Launch Odysseus.bat already exists and is current.
  ) else (
    choice /C YN /N /M "Do you want to update the current Quick Launch Odysseus.bat? [Y/N] "
    if not errorlevel 2 (
      copy /Y "!QUICK_CANDIDATE!" "!QUICK_LAUNCH!" >nul
      if errorlevel 1 (
        echo WARNING: Quick Launch Odysseus.bat could not be updated.
      ) else (
        echo [updated] Quick Launch Odysseus.bat
      )
    )
  )
) else (
  choice /C YN /N /M "Would you like to create Quick Launch Odysseus.bat? [Y/N] "
  if not errorlevel 2 (
    copy /Y "!QUICK_CANDIDATE!" "!QUICK_LAUNCH!" >nul
    if errorlevel 1 (
      echo WARNING: Quick Launch Odysseus.bat could not be created.
    ) else (
      echo [created] Quick Launch Odysseus.bat
    )
  )
)

:quick_done
if exist "!QUICK_CANDIDATE!" del /Q "!QUICK_CANDIDATE!" >nul 2>&1

echo.
echo === Ready to use ===
echo Start with launch-windows.ps1 or Quick Launch Odysseus.bat if created.
echo.
pause
exit /b 0

:install_opencode
set "NPMCMD="
for /f "delims=" %%I in ('where npm.cmd 2^>nul') do if not defined NPMCMD set "NPMCMD=%%I"
if not defined NPMCMD if exist "%ProgramFiles%\nodejs\npm.cmd" set "NPMCMD=%ProgramFiles%\nodejs\npm.cmd"
if not defined NPMCMD (
  where winget >nul 2>&1
  if errorlevel 1 (
    echo WARNING: npm is required for OpenCode. Install Node.js LTS, then run:
    echo          npm install -g opencode-ai
    exit /b 1
  )
  echo Installing Node.js LTS as part of the approved OpenCode action...
  winget install --id OpenJS.NodeJS.LTS -e --accept-package-agreements --accept-source-agreements
  if exist "%ProgramFiles%\nodejs\npm.cmd" set "NPMCMD=%ProgramFiles%\nodejs\npm.cmd"
)
if not defined NPMCMD (
  echo WARNING: Node.js installed but npm is not visible yet. Open a new terminal and run:
  echo          npm install -g opencode-ai
  exit /b 1
)
if /I "%~1"=="update" (
  call "%NPMCMD%" install -g opencode-ai@latest
) else (
  call "%NPMCMD%" install -g opencode-ai
)
if errorlevel 1 echo WARNING: OpenCode action failed; Legacy remains available.
exit /b 0

:python_missing
echo ERROR: Python 3.11+ is mandatory and winget is unavailable.
echo Install Python from https://www.python.org/downloads/ and run this file again.
goto :failed_pause

:failed
echo.
echo ERROR: A mandatory installation or verification step failed.
:failed_pause
echo.
pause
exit /b 1
