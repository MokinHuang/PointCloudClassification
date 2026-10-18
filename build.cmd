@echo off
setlocal
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b 1
set "PCL_ROOT=D:\ComputerVisual\PCL\PCL 1.15.1"
set INC=/I"%PCL_ROOT%\3rdParty\VTK\include\vtk-9.4" /I"%PCL_ROOT%\3rdParty\Qhull\include" /I"%PCL_ROOT%\3rdParty\OpenNI\Include" /I"%PCL_ROOT%\3rdParty\FLANN\include" /I"%PCL_ROOT%\3rdParty\Eigen3\include\eigen3" /I"%PCL_ROOT%\3rdParty\Boost\include\boost-1_87" /I"%PCL_ROOT%\include\pcl-1.15"
set OPT=/nologo /O2 /Gy /EHsc /arch:AVX /std:c++17 /permissive- /utf-8 /DBOOST_USE_WINDOWS_H /DNOMINMAX /D_CRT_SECURE_NO_DEPRECATE
if not exist build mkdir build
cl.exe %OPT% /c %INC% /Fobuild\PointCloudClassification.obj Pointcloudclassification\PointCloudClassification.cpp
if errorlevel 1 exit /b 1
cl.exe %OPT% /c %INC% /Fobuild\FeatureLine.obj Pointcloudclassification\FeatureLine.cpp
if errorlevel 1 exit /b 1
cl.exe %OPT% /c %INC% /Fobuild\main.obj Pointcloudclassification\main.cpp
if errorlevel 1 exit /b 1
link.exe /nologo /out:build\Pointcloudclassification.exe /subsystem:console /opt:ref /libpath:"%PCL_ROOT%\lib" /libpath:"%PCL_ROOT%\3rdParty\Boost\lib" build\PointCloudClassification.obj build\FeatureLine.obj build\main.obj pcl_io.lib pcl_common.lib
exit /b %errorlevel%
