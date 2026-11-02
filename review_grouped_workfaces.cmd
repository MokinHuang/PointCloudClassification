@echo off
setlocal
cd /d "%~dp0"
set "GROUP_INPUT=%~1"
if not defined GROUP_INPUT set "GROUP_INPUT=%~dp0run-output"
if not exist "%GROUP_INPUT%\classification.csv" if exist "%GROUP_INPUT%\run-output\classification.csv" set "GROUP_INPUT=%GROUP_INPUT%\run-output"
if not exist "%GROUP_INPUT%\classification.csv" (
  echo Drag a result folder containing classification.csv onto this script.
  pause
  exit /b 1
)
if not exist "%~dp0build\workface_range.exe" (
  echo Please run build.cmd first, or use the compiled tool package.
  pause
  exit /b 1
)
"%~dp0build\workface_range.exe" "%GROUP_INPUT%" "%GROUP_INPUT%\grouped-review"
if errorlevel 1 (
  echo Generation failed. See the error above.
  pause
  exit /b 1
)
echo Open this file in Edge or Chrome:
echo "%GROUP_INPUT%\grouped-review\grouped_workface_review.html"
pause
