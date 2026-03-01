@echo off
REM ============================================================
REM  GIGA TRADER - Auto-Start Launcher (Windows Task Scheduler)
REM ============================================================
REM
REM  This script is designed to be registered with Windows Task
REM  Scheduler to auto-restart the giga_trader system on reboot
REM  or after a crash.
REM
REM  To register (run once as Administrator):
REM    schtasks /create /tn "GigaTrader" /tr "C:\Users\amare\OneDrive\Documents\giga_trader\scripts\autostart_giga_trader.bat" /sc onlogon /rl highest /f
REM
REM  To remove:
REM    schtasks /delete /tn "GigaTrader" /f
REM
REM  To check status:
REM    schtasks /query /tn "GigaTrader"
REM
REM ============================================================

setlocal

set PROJECT_ROOT=C:\Users\amare\OneDrive\Documents\giga_trader
set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe
set LOG_FILE=%PROJECT_ROOT%\logs\autostart_%date:~-4,4%%date:~-7,2%%date:~-10,2%.log

REM Create logs directory if needed
if not exist "%PROJECT_ROOT%\logs" mkdir "%PROJECT_ROOT%\logs"

echo [%date% %time%] GIGA TRADER auto-start triggered >> "%LOG_FILE%"

REM Wait 30 seconds for network/services to settle after login
timeout /t 30 /nobreak > nul

REM Kill any stale Python processes from previous session
echo [%date% %time%] Checking for stale processes... >> "%LOG_FILE%"
tasklist /fi "imagename eq python.exe" /fo csv 2>nul | find /i "start_system" > nul
if %errorlevel%==0 (
    echo [%date% %time%] Found stale start_system.py process, skipping start >> "%LOG_FILE%"
    goto :eof
)

REM Change to project directory
cd /d "%PROJECT_ROOT%"

REM Load .env file (for environment variables)
if exist "%PROJECT_ROOT%\.env" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%PROJECT_ROOT%\.env") do (
        set "%%a=%%b"
    )
)

REM Launch the system in --no-trading mode (experiment + training only)
echo [%date% %time%] Starting giga_trader system... >> "%LOG_FILE%"
start "GigaTrader" /min "%VENV_PYTHON%" "%PROJECT_ROOT%\scripts\start_system.py" --no-trading >> "%LOG_FILE%" 2>&1

echo [%date% %time%] System launched successfully >> "%LOG_FILE%"

endlocal
