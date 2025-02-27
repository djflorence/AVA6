@echo off
REM Backup script for AVA6 project

echo AVA6 Backup Script
echo ==================
echo.

REM Get the current date and time for the commit message
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YYYY=%dt:~0,4%"
set "MM=%dt:~4,2%"
set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%"
set "Min=%dt:~10,2%"
set "Sec=%dt:~12,2%"

set "DATE=%YYYY%-%MM%-%DD% %HH%:%Min%:%Sec%"
set "COMMIT_MSG=Backup: %DATE%"

REM Check if a custom commit message was provided
if not "%~1"=="" (
    set "COMMIT_MSG=%~1"
)

echo Using commit message: '%COMMIT_MSG%'
echo.

REM Add all changes
echo Adding all changes...
git add .

REM Commit changes
echo Committing changes...
git commit -m "%COMMIT_MSG%"

REM Push to GitHub
echo Pushing to GitHub...
git push origin clean-develop

echo.
echo Backup completed successfully!
echo. 