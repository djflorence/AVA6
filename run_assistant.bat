@echo off
REM Run the AI Assistant with a specific profile

REM Check if profile is provided
if "%1"=="" (
    echo Usage: run_assistant.bat [profile] [--web]
    echo Example: run_assistant.bat development --web
    exit /b 1
)

REM Set profile
set PROFILE=%1

REM Check for web flag
set WEB_FLAG=
if "%2"=="--web" set WEB_FLAG=--web

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
)

REM Run the assistant
echo Running assistant with profile: %PROFILE%
python -m src.main --config %PROFILE% %WEB_FLAG%

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat 