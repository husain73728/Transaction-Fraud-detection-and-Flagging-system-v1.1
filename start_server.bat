@echo off
setlocal
cd /d %~dp0
set PY_EXE=C:\Python314\python.exe

echo Starting Vigilant Lens server... > server.log
"%PY_EXE%" -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --log-level info >> server.log 2>&1
