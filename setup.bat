@echo off
REM ---------------------------------------------------------------------------
REM One-time setup. Run from an Anaconda Prompt in this folder:
REM     setup.bat
REM It creates/uses the "wardrobe" conda env and installs everything.
REM ---------------------------------------------------------------------------
setlocal
cd /d %~dp0

echo === Creating conda env "wardrobe" (python 3.11) if needed ===
call conda create -n wardrobe python=3.11 -y
call conda activate wardrobe
if errorlevel 1 (
  echo Could not activate the "wardrobe" conda env. Open an Anaconda Prompt and retry.
  exit /b 1
)

echo === Installing PyTorch (CUDA 12.4) ===
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo === Installing project requirements ===
pip install -r backend\requirements.txt

echo === Environment check ===
python scripts\check_env.py

echo.
echo Setup done. Next: run.bat   (then open http://127.0.0.1:8000)
endlocal
