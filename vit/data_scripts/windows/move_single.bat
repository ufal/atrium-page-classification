@echo off

:: Define the target directory
set "TARGET_DIR=onepagers"

:: Create the target directory if it doesn't exist
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

:: Iterate through all subdirectories in the current directory
for /d %%D in (*) do (
    if not "%%~nxD" == "%TARGET_DIR%" (
        :: Initialize a counter
        set "FILE_COUNT=0"
        for %%F in ("%%D\*") do (
            set /a FILE_COUNT+=1
            set "LAST_FILE=%%F"
        )

        :: Use delayed expansion to evaluate variables within loops
        setlocal enabledelayedexpansion
        if !FILE_COUNT! equ 1 (
            echo Moving !LAST_FILE! to %TARGET_DIR%
            move "!LAST_FILE!" "%TARGET_DIR%"
            echo Removing empty directory %%D
            rmdir "%%D"
        )
        endlocal
    )
)

echo Files moved to "%TARGET_DIR%" and empty directories removed.
