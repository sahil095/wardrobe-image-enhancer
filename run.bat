@echo off
REM ---------------------------------------------------------------------------
REM Start the web app. Run from an Anaconda Prompt in this folder:
REM     run.bat
REM Then open http://127.0.0.1:8000
REM ---------------------------------------------------------------------------
setlocal
cd /d %~dp0

call conda activate wardrobe-ie
if errorlevel 1 (
  echo Could not activate the "wardrobe-ie" conda env. Run setup.bat first.
  exit /b 1
)

cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
endlocal
