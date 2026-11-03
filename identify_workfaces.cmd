@echo off
setlocal
if not exist "%~dp0build\workface_preview.exe" (
  echo Run build.cmd first.
  exit /b 1
)
if not exist "%~dp0run-output\classification.csv" (
  echo Process a PCD first with run.cmd.
  exit /b 1
)
"%~dp0build\workface_preview.exe" "%~dp0run-output" "%~dp0run-output"
if errorlevel 1 exit /b 1
echo Automatic candidates: run-output\auto_workface_selector.html
echo Candidate geometry only. Production status and design limits are not confirmed.
