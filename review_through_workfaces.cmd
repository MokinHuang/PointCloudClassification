@echo off
setlocal
cd /d "%~dp0"
set "THROUGH_INPUT=%~1"
if not defined THROUGH_INPUT set "THROUGH_INPUT=%~dp0run-output"
if not exist "%THROUGH_INPUT%\classification.csv" if exist "%THROUGH_INPUT%\run-output\classification.csv" set "THROUGH_INPUT=%THROUGH_INPUT%\run-output"
if not exist "%THROUGH_INPUT%\classification.csv" (
  echo Drag a result folder containing classification.csv, feature_lines.csv and grid_origin.txt here.
  pause
  exit /b 1
)
if not exist "%~dp0build\workface_range.exe" (
  echo Build first or use the compiled replay tool.
  pause
  exit /b 1
)
"%~dp0build\workface_range.exe" "%THROUGH_INPUT%" "%THROUGH_INPUT%\through-review"
if errorlevel 1 (
  echo Generation failed. See the error above.
  pause
  exit /b 1
)
start "" "%THROUGH_INPUT%\through-review\through_resolved_workface_review.html"
pause
