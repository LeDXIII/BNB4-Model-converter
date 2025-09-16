@echo off
chcp 65001 >nul
echo ==========================================
echo AI Model BNB4 Converter — Run
echo ==========================================

if not exist venv (
  echo ERROR: Virtual environment not found.
  echo Please run install.bat first.
  pause & exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Starting converter...
python gui.py

if errorlevel 1 (
  echo.
  echo ERROR: Application exited with errors.
  pause
)
