@echo off
cd /d "%~dp0"
if "%~2"=="" (echo Usage: repair_features.cmd INPUT_DIRECTORY OUTPUT_DIRECTORY & exit /b 1)
build\gap_replay.exe "%~1" "%~2"
if errorlevel 1 exit /b 1
python gap_report.py "%~1" "%~2"
