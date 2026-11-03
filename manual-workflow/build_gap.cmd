@echo off
setlocal
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b 1
set "PCL_ROOT=D:\ComputerVisual\PCL\PCL 1.15.1"
set INC=/I"%PCL_ROOT%\3rdParty\VTK\include\vtk-9.4" /I"%PCL_ROOT%\3rdParty\Qhull\include" /I"%PCL_ROOT%\3rdParty\OpenNI\Include" /I"%PCL_ROOT%\3rdParty\FLANN\include" /I"%PCL_ROOT%\3rdParty\Eigen3\include\eigen3" /I"%PCL_ROOT%\3rdParty\Boost\include\boost-1_87" /I"%PCL_ROOT%\include\pcl-1.15"
set OPT=/nologo /O2 /Gy /EHsc /arch:AVX /std:c++17 /permissive- /utf-8 /DBOOST_USE_WINDOWS_H /DNOMINMAX /D_CRT_SECURE_NO_DEPRECATE
if not exist build mkdir build
cl.exe %OPT% %INC% /Fobuild\gap_replay.obj /Febuild\gap_replay.exe gap_replay.cpp
if errorlevel 1 exit /b 1
cl.exe %OPT% %INC% /Fobuild\gap_tests.obj /Febuild\gap_tests.exe tests\gap_tests.cpp
if errorlevel 1 exit /b 1
build\gap_tests.exe
exit /b %errorlevel%
