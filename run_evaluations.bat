@echo off
setlocal

:: Save current PATH
set ORIGINAL_PATH=%PATH%

:: Set a minimal PATH for pixi activation
set PATH=C:\Windows\System32;C:\Windows;%USERPROFILE%\.pixi\bin;%LOCALAPPDATA%\pixi\bin

:: Try to run with pixi
echo Testing pixi environment...
F:\hok\DATN\Project\traffic-monitor-resources\.pixi\envs\default\python.exe -c "import pandas, matplotlib, seaborn; print('All packages available')"

if %errorlevel% equ 0 (
    echo Running evaluation scripts...
    F:\hok\DATN\Project\traffic-monitor-resources\.pixi\envs\default\python.exe evaluation_tools\run_all_visualizations.py
) else (
    echo Pixi environment failed, using system Python...
    python evaluation_tools\run_all_visualizations.py
)

:: Restore original PATH
set PATH=%ORIGINAL_PATH%
pause 

$clean = ($env:Path -split ';' | Select-Object -Unique) -join ';' [Environment]::SetEnvironmentVariable('Path',$clean,'User')