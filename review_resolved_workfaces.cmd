@echo off
setlocal
cd /d "%~dp0"
set "RESOLVED_INPUT=%~1"
if not defined RESOLVED_INPUT set "RESOLVED_INPUT=%~dp0run-output"
if not exist "%RESOLVED_INPUT%\classification.csv" if exist "%RESOLVED_INPUT%\run-output\classification.csv" set "RESOLVED_INPUT=%RESOLVED_INPUT%\run-output"
if not exist "%RESOLVED_INPUT%\classification.csv" (
  echo Drag a result folder containing classification.csv onto this script.
  pause
  exit /b 1
)
if not exist "%~dp0build\workface_range.exe" (
  echo Build first, or use the compiled tool package.
  pause
  exit /b 1
)
"%~dp0build\workface_range.exe" "%RESOLVED_INPUT%" "%RESOLVED_INPUT%\resolved-review"
if errorlevel 1 (
  echo Generation failed. See the error above.
  pause
  exit /b 1
)
echo Open this file in Edge or Chrome:
echo "%RESOLVED_INPUT%\resolved-review\resolved_workface_review.html"
pause
