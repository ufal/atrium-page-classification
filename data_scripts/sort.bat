@echo off

:: Set the input and output directories
set "INPUT_DIR=C:\Users\user\Documents\code\atrium\atrium-ufal\data_scripts"
set "OUTPUT_DIR=C:\Users\user\Documents\code\atrium\atrium-ufal\pages"
set "INPUT_CSV=C:\Users\user\Documents\code\atrium\atrium-ufal\data_scripts\labels.csv"

:: Ensure the output directory exists
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Check if the input CSV file exists
if not exist "%INPUT_CSV%" (
    echo ERROR: The file "%INPUT_CSV%" does not exist.
    exit /b 1
)

:: Skip the header line and process the CSV file line by line
for /f "tokens=1,2,3 delims=, skip=1" %%A in (%INPUT_CSV%) do (
    setlocal enabledelayedexpansion
    
    set "FILENAME=%%A"
    set "PAGE_NUMBER=%%B"
    set "CATEGORY=%%C"
    
    echo Processing: !FILENAME! !PAGE_NUMBER! !CATEGORY!
    
    :: Create category subdirectory inside output directory if it doesn't exist
    set "CATEGORY_DIR=%OUTPUT_DIR%\!CATEGORY!"
    if not exist "!CATEGORY_DIR!" mkdir "!CATEGORY_DIR!"

    :: Check if subdirectory exists in input directory
    set "INPUT_SUBDIR=%INPUT_DIR%\!FILENAME!"
    if exist "!INPUT_SUBDIR!" (
        :: Find the matching file in the subdirectory
        for %%F in ("!INPUT_SUBDIR!\*-!PAGE_NUMBER!.png") do (
            echo Copying %%F to !CATEGORY_DIR!
            copy "%%F" "!CATEGORY_DIR!"
        )
    ) else (
        :: Use 'onepagers' as the default subdirectory
        set "DEFAULT_SUBDIR=%INPUT_DIR%\onepagers"
        for %%F in ("!DEFAULT_SUBDIR!\!FILENAME!-!PAGE_NUMBER!.png") do (
            echo Copying %%F to !CATEGORY_DIR!
            copy "%%F" "!CATEGORY_DIR!"
        )
    )

    endlocal
)

echo Processing completed.
