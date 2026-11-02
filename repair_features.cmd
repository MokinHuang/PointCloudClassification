@echo off
setlocal
cd /d "%~dp0"
if not exist build\gap_replay.exe (
  echo Please run build.cmd first.
  exit /b 1
)
build\gap_replay.exe run-output run-output
exit /b %errorlevel%
