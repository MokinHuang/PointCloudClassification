@echo off
setlocal
set "PCL_ROOT=D:\ComputerVisual\PCL\PCL 1.15.1"
set "PATH=%PCL_ROOT%\bin;%PCL_ROOT%\3rdParty\VTK\bin;%PCL_ROOT%\3rdParty\OpenNI\Redist;%PATH%"
set "CLOUD_INPUT=%~f1"
set "WORKFACE_CONFIG=%~f2"
if "%CLOUD_INPUT%"=="" set "CLOUD_INPUT=%~dp0Pointcloudclassification\test2.pcd"
if not exist "%CLOUD_INPUT%" (
  echo Point cloud not found. Drag a PCD file onto run.cmd or pass its full path.
  exit /b 1
)
if not exist "%~dp0build\Pointcloudclassification.exe" (
  echo Build first using build.cmd.
  exit /b 1
)
if not exist "%~dp0run-output" mkdir "%~dp0run-output"
cd /d "%~dp0run-output"
"%~dp0build\Pointcloudclassification.exe" "%CLOUD_INPUT%" "%WORKFACE_CONFIG%"
exit /b %errorlevel%
