@echo off
REM BudSimulator Setup Script for Windows

echo Starting BudSimulator Setup...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "env" (
    echo Creating virtual environment...
    python -m venv env
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install critical dependencies first
echo Installing critical dependencies...
pip install psutil python-multipart aiofiles colorama

REM Install other dependencies (filtering out problematic ones)
echo Installing dependencies from requirements.txt...
findstr /v "^genz>=" requirements.txt | findstr /v "^genz-llm" > temp_requirements.txt
pip install -r temp_requirements.txt
del temp_requirements.txt

REM Install the local GenZ package in editable mode
if exist "GenZ" (
    echo Installing local GenZ package...
    pip install -e .
    echo GenZ package installed in editable mode
)

REM Run the main setup script
echo Running main setup script...
python setup.py

REM Verify installation
echo Verifying installation...
python verify_setup.py

echo.
echo Setup complete!
echo.
echo Next steps:
echo   1. To start the API server: python run_api.py
echo   2. To enable hot-reload (development): set RELOAD=true && python run_api.py
echo   3. To run tests: pytest tests/
echo.
echo Note: The virtual environment is activated.
pause 