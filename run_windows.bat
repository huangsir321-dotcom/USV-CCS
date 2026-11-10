@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo USV COLREGs Simulator - Windows launcher
echo The CURRENT Python interpreter will be used.
echo ============================================================
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found in PATH.
    echo Open Anaconda Prompt, activate your environment, then run this file.
    pause
    exit /b 1
)
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo GUI/ML dependency installation failed.
    pause
    exit /b 1
)
echo.
python -c "import torch,sys; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if errorlevel 1 (
    echo No working CUDA GPU was detected in this Python environment.
    choice /C YN /M "Install a CUDA-enabled PyTorch wheel now"
    if errorlevel 2 goto START_GUI
    call install_cuda_torch.bat
    if errorlevel 1 (
        echo CUDA installation did not complete. The simulator can still run.
        echo GPU training will remain unavailable until CUDA is fixed.
    )
) else (
    echo CUDA is already available. Skipping PyTorch reinstallation.
)
:START_GUI
echo.
python gpu_check.py
python run.py
if errorlevel 1 pause
endlocal
