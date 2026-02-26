@echo off
echo Starting Drug Interaction Chatbot Server...
echo ------------------------------------------

:: Launch the main PowerShell script
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\run_all.ps1"

:: Pause if something goes wrong
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred. Press any key to exit.
    pause >nul
)
