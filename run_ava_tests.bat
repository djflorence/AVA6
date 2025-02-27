@echo off
echo Ava Testing Suite
echo =================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    exit /b 1
)

echo Step 1: Preparing Ava for testing...
python prepare_ava_test.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Preparation failed
    echo Please check the error messages above
    exit /b 1
)

echo.
echo Step 2: Running Ava tests...
echo.

REM Check if a specific test suite was requested
set TEST_SUITE=all
set SAVE_RESULTS=

:parse_args
if "%~1"=="" goto run_tests
if /i "%~1"=="--test-suite" (
    set TEST_SUITE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--save-results" (
    set SAVE_RESULTS=--save-results
    shift
    goto parse_args
)
shift
goto parse_args

:run_tests
if "%TEST_SUITE%"=="all" (
    python test_ava_chat.py %SAVE_RESULTS%
) else (
    python test_ava_chat.py --test-suite %TEST_SUITE% %SAVE_RESULTS%
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Tests failed
    echo Please check the error messages above
    exit /b 1
)

echo.
echo All tests completed successfully!
echo.
echo To view detailed results, check the test_results directory if you used --save-results
echo. 