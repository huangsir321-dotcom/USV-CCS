@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Install CUDA-enabled PyTorch into the CURRENT Python env
echo ============================================================
python --version
if errorlevel 1 (
  echo Python was not found. Activate your Anaconda environment first.
  pause
  exit /b 2
)
echo.
echo 1. PyTorch 2.15.0 + CUDA 13.2 (Python 3.11+, newer NVIDIA driver)
echo 2. PyTorch 2.14.1 + CUDA 12.6 (Python 3.10+, broader driver compatibility)
echo.
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)"
if errorlevel 1 (
  set "GPU_CHOICE=2"
  echo Python 3.10 detected: selecting the CUDA 12.6 compatibility wheel.
) else (
  set "GPU_CHOICE=1"
  set /p GPU_CHOICE=Select CUDA build [1/2, default 1]:
  if "%GPU_CHOICE%"=="" set "GPU_CHOICE=1"
)
if "%GPU_CHOICE%"=="1" goto CUDA132
if "%GPU_CHOICE%"=="2" goto CUDA126
echo Invalid selection.
exit /b 2

:CUDA132
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)"
if errorlevel 1 (
  echo PyTorch 2.15 CUDA 13.2 requires Python 3.11 or later.
  exit /b 2
)
echo Installing PyTorch 2.15.0 CUDA 13.2 into %CD% ...
python -m pip install --upgrade torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
if errorlevel 1 goto INSTALL_FAILED
goto VERIFY

:CUDA126
echo Installing PyTorch 2.14.1 CUDA 12.6 into %CD% ...
python -m pip install --upgrade torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto INSTALL_FAILED
goto VERIFY

:INSTALL_FAILED
echo.
echo CUDA wheel installation failed. No CPU fallback was installed automatically.
echo The current environment was not deliberately uninstalled first.
echo Check the pip error, Python version, network and NVIDIA driver; then retry.
exit /b 3

:VERIFY
python gpu_check.py
if errorlevel 1 (
  echo.
  echo The CUDA wheel was installed or selected, but the GPU is not available to PyTorch.
  echo Check NVIDIA hardware/driver with nvidia-smi and confirm this is the correct Python env.
  echo The simulator can still run; select CPU only if you explicitly want CPU training.
  exit /b 0
)
echo.
echo CUDA training is ready in this Python environment.
endlocal
exit /b 0
