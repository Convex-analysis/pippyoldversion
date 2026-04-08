@echo off

:: EdgePipe Jetson Run Script for Windows
:: This script simplifies the process of running EdgePipe on two Jetson devices

:: Default configuration
set SERVER_HOST=0.0.0.0
set SERVER_PORT=5000
set DEVICE0_ID=orin
set DEVICE1_ID=nano
set DEVICE0_HOST=
set DEVICE1_HOST=
set ROUNDS=1
set MICRO_BATCHES=4
set TEMPLATE_ID=vit_b16_2stage_v1
set DATA_DIR=./data
set DOWNLOAD=false
set AUTO_EXIT=true
set COMMAND=

:: Usage function
:usage
echo Usage: %0 [options] [command]
echo.
echo Commands:
echo   server          Start the EdgePipe server
echo   device0         Start EdgePipe on device0 (Jetson Orin)
echo   device1         Start EdgePipe on device1 (Jetson Nano)
echo   status          Check status of EdgePipe system
echo   cleanup         Clean up EdgePipe processes
echo.
echo Options:
echo   --server-host   Server host address (default: %SERVER_HOST%)
echo   --server-port   Server port (default: %SERVER_PORT%)
echo   --device0-id    Device0 ID (default: %DEVICE0_ID%)
echo   --device1-id    Device1 ID (default: %DEVICE1_ID%)
echo   --device0-host  Device0 host address
echo   --device1-host  Device1 host address
echo   --rounds        Number of training rounds (default: %ROUNDS%)
echo   --micro-batches Number of micro-batches (default: %MICRO_BATCHES%)
echo   --template-id   Model template ID (default: %TEMPLATE_ID%)
echo   --data-dir      Data directory (default: %DATA_DIR%)
echo   --download      Download dataset if not present (default: %DOWNLOAD%)
echo   --auto-exit     Automatically exit after training (default: %AUTO_EXIT%)
echo   -h, --help      Show this help message
echo.
echo Examples:
echo   :: Start server
echo   %0 server
echo.
echo   :: Start device0
echo   %0 device0 --server-host 192.168.1.100
echo.
goto :eof

:: Parse arguments
:parse_args
if "%~1"=="" goto :validate_args

if "%~1"=="server" set COMMAND=server && shift && goto :parse_args
if "%~1"=="device0" set COMMAND=device0 && shift && goto :parse_args
if "%~1"=="device1" set COMMAND=device1 && shift && goto :parse_args
if "%~1"=="status" set COMMAND=status && shift && goto :parse_args
if "%~1"=="cleanup" set COMMAND=cleanup && shift && goto :parse_args

if "%~1"=="--server-host" set SERVER_HOST=%~2 && shift && shift && goto :parse_args
if "%~1"=="--server-port" set SERVER_PORT=%~2 && shift && shift && goto :parse_args
if "%~1"=="--device0-id" set DEVICE0_ID=%~2 && shift && shift && goto :parse_args
if "%~1"=="--device1-id" set DEVICE1_ID=%~2 && shift && shift && goto :parse_args
if "%~1"=="--device0-host" set DEVICE0_HOST=%~2 && shift && shift && goto :parse_args
if "%~1"=="--device1-host" set DEVICE1_HOST=%~2 && shift && shift && goto :parse_args
if "%~1"=="--rounds" set ROUNDS=%~2 && shift && shift && goto :parse_args
if "%~1"=="--micro-batches" set MICRO_BATCHES=%~2 && shift && shift && goto :parse_args
if "%~1"=="--template-id" set TEMPLATE_ID=%~2 && shift && shift && goto :parse_args
if "%~1"=="--data-dir" set DATA_DIR=%~2 && shift && shift && goto :parse_args
if "%~1"=="--download" set DOWNLOAD=%~2 && shift && shift && goto :parse_args
if "%~1"=="--auto-exit" set AUTO_EXIT=%~2 && shift && shift && goto :parse_args
if "%~1"=="-h" goto :usage
if "%~1"=="--help" goto :usage

echo Error: Unknown argument %~1
echo.
goto :usage

:validate_args
if "%COMMAND%"=="" (
    echo Error: No command specified
    echo.
    goto :usage
)
goto :eof

:: Start server
:start_server
echo Starting EdgePipe server...
echo Server host: %SERVER_HOST%:%SERVER_PORT%
echo Device IDs: %DEVICE0_ID%, %DEVICE1_ID%
echo Rounds: %ROUNDS%
echo Micro-batches: %MICRO_BATCHES%
echo Template ID: %TEMPLATE_ID%
echo.

python -m fhdp.EdgePipe.edgepipe_jetson ^
    --mode server ^
    --host "%SERVER_HOST%" ^
    --port "%SERVER_PORT%" ^
    --device0-id "%DEVICE0_ID%" ^
    --device1-id "%DEVICE1_ID%" ^
    --rounds "%ROUNDS%" ^
    --micro-batches "%MICRO_BATCHES%" ^
    --template-id "%TEMPLATE_ID%" ^
    --auto-exit "%AUTO_EXIT%"
goto :eof

:: Start device0
:start_device0
echo Starting EdgePipe device0...
echo Device ID: %DEVICE0_ID%
echo Server: %SERVER_HOST%:%SERVER_PORT%
echo Rounds: %ROUNDS%
echo Micro-batches: %MICRO_BATCHES%
echo Data directory: %DATA_DIR%
echo Download: %DOWNLOAD%
echo.

python -m fhdp.EdgePipe.edgepipe_jetson ^
    --mode device ^
    --role device0 ^
    --device-id "%DEVICE0_ID%" ^
    --server-host "%SERVER_HOST%" ^
    --server-port "%SERVER_PORT%" ^
    --device0-id "%DEVICE0_ID%" ^
    --device1-id "%DEVICE1_ID%" ^
    --listen-host "0.0.0.0" ^
    --listen-port "5001" ^
    --rounds "%ROUNDS%" ^
    --micro-batches "%MICRO_BATCHES%" ^
    --data-dir "%DATA_DIR%" ^
    --download "%DOWNLOAD%" ^
    --auto-exit "%AUTO_EXIT%"
goto :eof

:: Start device1
:start_device1
echo Starting EdgePipe device1...
echo Device ID: %DEVICE1_ID%
echo Server: %SERVER_HOST%:%SERVER_PORT%
echo Rounds: %ROUNDS%
echo Micro-batches: %MICRO_BATCHES%
echo Data directory: %DATA_DIR%
echo Download: %DOWNLOAD%
echo.

python -m fhdp.EdgePipe.edgepipe_jetson ^
    --mode device ^
    --role device1 ^
    --device-id "%DEVICE1_ID%" ^
    --server-host "%SERVER_HOST%" ^
    --server-port "%SERVER_PORT%" ^
    --device0-id "%DEVICE0_ID%" ^
    --device1-id "%DEVICE1_ID%" ^
    --listen-host "0.0.0.0" ^
    --listen-port "5002" ^
    --rounds "%ROUNDS%" ^
    --micro-batches "%MICRO_BATCHES%" ^
    --data-dir "%DATA_DIR%" ^
    --download "%DOWNLOAD%" ^
    --auto-exit "%AUTO_EXIT%"
goto :eof

:: Check status
:check_status
echo Checking EdgePipe system status...
echo.

:: Check if server is running
tasklist | findstr "python.exe" | findstr "edgepipe_jetson.*--mode server" >nul
if %errorlevel% equ 0 (
    echo ✓ Server is running
) else (
    echo ✗ Server is not running
)

:: Check if device0 is running
tasklist | findstr "python.exe" | findstr "edgepipe_jetson.*--role device0" >nul
if %errorlevel% equ 0 (
    echo ✓ Device0 is running
) else (
    echo ✗ Device0 is not running
)

:: Check if device1 is running
tasklist | findstr "python.exe" | findstr "edgepipe_jetson.*--role device1" >nul
if %errorlevel% equ 0 (
    echo ✓ Device1 is running
) else (
    echo ✗ Device1 is not running
)
echo.
goto :eof

:: Cleanup processes
:cleanup
echo Cleaning up EdgePipe processes...
echo.

:: Kill server process
for /f "tokens=2" %%a in ('tasklist ^| findstr "python.exe" ^| findstr "edgepipe_jetson.*--mode server"') do (
    echo Killing server process: %%a
    taskkill /PID %%a /F >nul 2>&1
)

:: Kill device0 process
for /f "tokens=2" %%a in ('tasklist ^| findstr "python.exe" ^| findstr "edgepipe_jetson.*--role device0"') do (
    echo Killing device0 process: %%a
    taskkill /PID %%a /F >nul 2>&1
)

:: Kill device1 process
for /f "tokens=2" %%a in ('tasklist ^| findstr "python.exe" ^| findstr "edgepipe_jetson.*--role device1"') do (
    echo Killing device1 process: %%a
    taskkill /PID %%a /F >nul 2>&1
)
echo.
echo Cleanup completed
goto :eof

:: Main function
:main
call :parse_args %*

if "%COMMAND%"=="server" call :start_server
if "%COMMAND%"=="device0" call :start_device0
if "%COMMAND%"=="device1" call :start_device1
if "%COMMAND%"=="status" call :check_status
if "%COMMAND%"=="cleanup" call :cleanup

:eof
