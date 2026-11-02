@echo off
setlocal
if "%~1"=="" (
  echo Drag the downloaded workfaces.csv onto this script.
  exit /b 1
)
if not exist "%~dp0run-output\classification.csv" (
  echo Run the point cloud processing first to create run-output diagnostics.
  exit /b 1
)
if not exist "%~dp0build\workface_preview.exe" (
  echo Run build.cmd first.
  exit /b 1
)
"%~dp0build\workface_preview.exe" "%~dp0run-output" "%~dp0run-output" "%~f1"
if errorlevel 1 exit /b 1
echo Preview updated: run-output\workface_selector.html
