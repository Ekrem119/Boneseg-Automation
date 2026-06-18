@echo off
setlocal
echo ====================================================
echo BoneSeg - Clean Environment Setup
echo ====================================================

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python is not installed. Please install it before continuing.
    pause
    exit
)

:: 2. Delete old environment to start fresh
if exist "env" (
    echo [+] Removing old environment for a clean start...
    rd /s /q env
)

echo [+] Creating new virtual environment...
python -m venv env

:: 3. Activation and Installation
call env\Scripts\activate.bat

echo [+] Upgrading pip...
python -m pip install --upgrade pip

:: 4. Installing Core Dependencies first
echo [+] Installing PyTorch and dependencies...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install "numpy<2.0.0" "pillow<11.0.0"

echo [+] Installing Imaging and Analysis tools...
python -m pip install nd2reader pims scikit-image scikit-learn pandas matplotlib customtkinter

echo [+] Installing Build tools...
python -m pip install pyinstaller
:: --------------------------------------

:: 5. Install the rest from requirements
if exist "requirements.txt" (
    echo [+] Installing remaining libraries from requirements.txt...
    python -m pip install -r requirements.txt
)

:: 6. Install SAM2
if exist "sam2" (
    echo [+] Installing SAM2...
    cd sam2
    python -m pip install -e . --no-deps
    cd ..
)

echo ====================================================
echo FINAL VERIFICATION
echo ====================================================

set "MISSING="

python -c "import nd2reader" 2>nul || set "MISSING=%MISSING% nd2reader"
python -c "import sklearn" 2>nul || set "MISSING=%MISSING% scikit-learn"
python -c "import torch" 2>nul || set "MISSING=%MISSING% torch"
python -c "import torchvision" 2>nul || set "MISSING=%MISSING% torchvision"
python -m PyInstaller --version >nul 2>&1 || set "MISSING=%MISSING% pyinstaller"

if "%MISSING%"=="" (
    echo [SUCCESS] All critical libraries found!
    echo You can now build the EXE.
) else (
    echo [ERROR] The following libraries failed to install: %MISSING%
    pause
    exit /b
)

pause

