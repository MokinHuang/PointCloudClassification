@echo off
setlocal
cd /d "%~dp0"
set "OWNED_INPUT=%~1"
if not defined OWNED_INPUT set "OWNED_INPUT=%~dp0run-output"
if not exist "%OWNED_INPUT%\classification.csv" if exist "%OWNED_INPUT%\run-output\classification.csv" set "OWNED_INPUT=%OWNED_INPUT%\run-output"
if not exist "%OWNED_INPUT%\classification.csv" (
  echo Drag the folder containing classification.csv, feature_lines.csv and grid_origin.txt here.
  pause
  exit /b 1
)
if not exist "%~dp0build\workface_range.exe" (
  echo Build first or use the compiled replay tool.
  pause
  exit /b 1
)
"%~dp0build\workface_range.exe" "%OWNED_INPUT%" "%OWNED_INPUT%\owned-review"
if errorlevel 1 (
  echo Generation failed. See the error above.
  pause
  exit /b 1
)
start "" "%OWNED_INPUT%\owned-review\owned_resolved_workface_review.html"
pause
