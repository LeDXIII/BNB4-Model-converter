@echo off
chcp 65001 >nul

echo ==========================================
echo AI Model BNB4 Converter (Windows) - Universal Installer
echo ==========================================

:: Check Python version
echo [0/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found.
    echo Please install Python 3.8 or later from https://www.python.org/downloads/
    echo Make sure Python is added to your PATH environment variable.
    pause
    exit /b 1
)

:: Get Python version and check compatibility
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
for /f "tokens=1,2 delims=." %%m in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%m
    set PYTHON_MINOR=%%n
)

if %PYTHON_MAJOR% LSS 3 (
    echo ERROR: Python 3.8 or later is required. Your version is %PYTHON_VERSION%.
    pause
    exit /b 1
)

if %PYTHON_MAJOR% EQU 3 (
    if %PYTHON_MINOR% LSS 8 (
        echo ERROR: Python 3.8 or later is required. Your version is %PYTHON_VERSION%.
        pause
        exit /b 1
    )
)

echo Compatible Python version %PYTHON_VERSION% detected.

:: Create virtual environment
echo [1/6] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip, setuptools, wheel
echo [2/6] Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel

:: Check NVIDIA GPU and install PyTorch with CUDA support
echo [3/6] Detecting CUDA and installing PyTorch...
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: NVIDIA GPU not detected. CUDA support required for this project.
    echo Please install NVIDIA drivers and ensure nvidia-smi is available in PATH.
    pause
    exit /b 1
)

:: Get NVIDIA driver version
set DRIVER_VERSION=
for /f "tokens=3" %%i in ('nvidia-smi 2^>nul ^| findstr /C:"Driver Version"') do set DRIVER_VERSION=%%i

if not defined DRIVER_VERSION (
    echo ERROR: Could not read NVIDIA driver version.
    echo Please ensure NVIDIA drivers are properly installed.
    pause
    exit /b 1
)

echo Detected NVIDIA Driver Version: %DRIVER_VERSION%

:: Determine compatible CUDA version and PyTorch
if "%DRIVER_VERSION%" LSS "525.60.13" (
    set CUDA_VERSION=cu118
    set TORCH_VERSION=2.6.0+cu118
    set TORCHVISION_VERSION=0.21.0+cu118
    set TORCHAUDIO_VERSION=2.6.0+cu118
    set PYTORCH_INDEX=https://download.pytorch.org/whl/cu118
    echo Selected CUDA version: 11.8
) else if "%DRIVER_VERSION%" LSS "535.104.05" (
    set CUDA_VERSION=cu121
    set TORCH_VERSION=2.6.0+cu121
    set TORCHVISION_VERSION=0.21.0+cu121
    set TORCHAUDIO_VERSION=2.6.0+cu121
    set PYTORCH_INDEX=https://download.pytorch.org/whl/cu121
    echo Selected CUDA version: 12.1
) else (
    set CUDA_VERSION=cu124
    set TORCH_VERSION=2.6.0+cu124
    set TORCHVISION_VERSION=0.21.0+cu124
    set TORCHAUDIO_VERSION=2.6.0+cu124
    set PYTORCH_INDEX=https://download.pytorch.org/whl/cu124
    echo Selected CUDA version: 12.4
)

echo Installing PyTorch for CUDA from: %PYTORCH_INDEX%
pip install torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION% --index-url %PYTORCH_INDEX%

:: Install other dependencies
echo [4/6] Installing other dependencies...
pip install "dill>=0.3.0,<0.3.9" "fsspec[http]>=2023.1.0,<=2025.3.0" "multiprocess<0.70.17" "tzdata>=2022.7"

:: Install dependencies with no-deps to avoid conflicts
echo [5/6] Installing remaining packages...
pip install --no-deps -r requirements.txt

:: Install additional packages
echo [6/6] Installing additional packages...
pip install xformers>=0.0.26 --no-deps
pip install optimum

:: Final message
echo.
echo Installation completed successfully!
echo To run the application, execute: run.bat
echo.
pause
