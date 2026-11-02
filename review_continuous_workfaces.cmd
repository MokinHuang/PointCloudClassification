@echo off
setlocal
cd /d "%~dp0"
set "CONTINUOUS_INPUT=%~1"
if not defined CONTINUOUS_INPUT set "CONTINUOUS_INPUT=%~dp0run-output"
if not exist "%CONTINUOUS_INPUT%\classification.csv" if exist "%CONTINUOUS_INPUT%\run-output\classification.csv" set "CONTINUOUS_INPUT=%CONTINUOUS_INPUT%\run-output"
if not exist "%CONTINUOUS_INPUT%\classification.csv" (
  echo Drag the result folder containing classification.csv, feature_lines.csv and grid_origin.txt here.
  pause
  exit /b 1
)
if not exist "%~dp0build\workface_range.exe" (
  echo Build first, or use the compiled tool package.
  pause
  exit /b 1
)
"%~dp0build\workface_range.exe" "%CONTINUOUS_INPUT%" "%CONTINUOUS_INPUT%\continuous-review"
if errorlevel 1 (
  echo Generation failed. See the error above.
  pause
  exit /b 1
)
start "" "%CONTINUOUS_INPUT%\continuous-review\continuous_resolved_workface_review.html"
pause
