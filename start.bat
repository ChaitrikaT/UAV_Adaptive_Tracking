@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Starting Condition-Aware UAV Tracking Streamlit App...
echo ============================================================

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv\
    echo Please create the virtual environment and install dependencies first:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Activating .venv and launching Streamlit...
streamlit run app.py

pause

