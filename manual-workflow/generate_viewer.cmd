@echo off
cd /d "%~dp0"
if "%~2"=="" (echo Usage: generate_viewer.cmd INPUT_DIRECTORY OUTPUT_HTML [MEASUREMENTS_DIRECTORY] & exit /b 1)
if "%~3"=="" (python manual-confirmation/build_viewer.py "%~1" "%~2") else (python manual-confirmation/build_viewer.py "%~1" "%~2" --measurements "%~3")
exit /b %errorlevel%
