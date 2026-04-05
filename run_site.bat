@echo off
setlocal
cd /d %~dp0
set PY_EXE=C:\Python314\python.exe

:: start backend server in a dedicated console and log output
start "Vigilant Server" cmd /k "cd /d %~dp0 && %PY_EXE% -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --log-level info"

:: small delay for startup
ping -n 3 127.0.0.1 > nul

:: open the site in the default browser
start "" http://127.0.0.1:8000/
endlocal
