@echo off
echo === Face Mask Detection System Setup ===
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Python found. Installing required packages...
echo.

REM Install required packages
pip install opencv-python numpy

if %errorlevel% neq 0 (
    echo Error installing packages. Please check your internet connection.
    pause
    exit /b 1
)

echo.
echo Setup complete! Starting Face Mask Detection System...
echo.

REM Run the detection system
python mask_detection_simple.py

pause
