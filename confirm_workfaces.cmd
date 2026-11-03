@echo off
setlocal
cd /d "%~dp0"
set "RANGE_INPUT=%~1"
if not defined RANGE_INPUT set "RANGE_INPUT=%~dp0run-output"
if not exist "%RANGE_INPUT%\classification.csv" if exist "%RANGE_INPUT%\run-output\classification.csv" set "RANGE_INPUT=%RANGE_INPUT%\run-output"
if not exist "%RANGE_INPUT%\classification.csv" (
  echo Results not found. Drag the folder containing classification.csv onto this script.
  pause
  exit /b 1
)
if not exist "%~dp0build\workface_range.exe" (
  echo Please run build.cmd first.
  pause
  exit /b 1
)
"%~dp0build\workface_range.exe" "%RANGE_INPUT%" "%RANGE_INPUT%\range-review"
if errorlevel 1 (
  echo Generation failed. See the error above.
  pause
  exit /b 1
)
echo Open this file in Edge or Chrome:
echo "%RANGE_INPUT%\range-review\workface_range_review.html"
pause
