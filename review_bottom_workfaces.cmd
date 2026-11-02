@echo off
setlocal
cd /d "%~dp0"
set "BOTTOM_INPUT=%~1"
if not defined BOTTOM_INPUT set "BOTTOM_INPUT=%~dp0run-output"
if not exist "%BOTTOM_INPUT%\classification.csv" if exist "%BOTTOM_INPUT%\run-output\classification.csv" set "BOTTOM_INPUT=%BOTTOM_INPUT%\run-output"
if not exist "%BOTTOM_INPUT%\classification.csv" (
  echo Drag the result folder containing classification.csv, feature_lines.csv and grid_origin.txt here.
  pause
  exit /b 1
)
if not exist "%~dp0build\workface_range.exe" (
  echo Build first or use the compiled replay tool.
  pause
  exit /b 1
)
"%~dp0build\workface_range.exe" "%BOTTOM_INPUT%" "%BOTTOM_INPUT%\bottom-review"
if errorlevel 1 (
  echo Generation failed. See the error above.
  pause
  exit /b 1
)
start "" "%BOTTOM_INPUT%\bottom-review\through_resolved_workface_review.html"
pause
