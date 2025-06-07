@echo off
rem This batch script kills pythonw processes that are running specific Python scripts

rem Set the script name or pattern to match (adjust this to your script's name)
set script_name=demo_depth_anything_new_threshold_algorithm.py

rem Loop through each pythonw process and check for the script in the command line
for /f "tokens=2,* delims=," %%a in ('tasklist /fi "imagename eq pythonw.exe" /v /fo csv') do (
    rem %%a is the PID and %%b contains the command line arguments (script name)
    echo %%a
    taskkill /f /pid %%a
)

echo All matching pythonw processes have been killed.
pause
