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

        :: --- REFINED: Avoid wildcard searches which cause I/O bottlenecks ---
        :: Generate padded versions of the page number natively (e.g. 1 -> 01, 001, 0001)
        set "P0=!PAGE_NUMBER!"
        set "P2=0!PAGE_NUMBER!"
        set "P2=!P2:~-2!"
        set "P3=00!PAGE_NUMBER!"
        set "P3=!P3:~-3!"
        set "P4=000!PAGE_NUMBER!"
        set "P4=!P4:~-4!"

        set "FILE_FOUND=false"

        :: Check exact paths directly instead of searching the directory
        for %%P in (!P0! !P2! !P3! !P4!) do (
            set "EXPECTED_FILE=!INPUT_SUBDIR!\!FILENAME!-%%P.png"
            if exist "!EXPECTED_FILE!" (
                echo Copying !EXPECTED_FILE! to !CATEGORY_DIR!
                copy "!EXPECTED_FILE!" "!CATEGORY_DIR!" >nul
                set "FILE_FOUND=true"
            )
        )

        if "!FILE_FOUND!"=="false" (
            echo File ending with !PAGE_NUMBER! not found in !INPUT_SUBDIR!.
        )

    ) else (
        :: Use 'onepagers' as the default subdirectory
        set "DEFAULT_SUBDIR=%INPUT_DIR%\onepagers"
        set "EXPECTED_FILE=!DEFAULT_SUBDIR!\!FILENAME!-!PAGE_NUMBER!.png"

        if exist "!EXPECTED_FILE!" (
            echo Copying !EXPECTED_FILE! to !CATEGORY_DIR!
            copy "!EXPECTED_FILE!" "!CATEGORY_DIR!" >nul
        ) else (
            echo File starting with !FILENAME! and ending with !PAGE_NUMBER! not found in !DEFAULT_SUBDIR!.
        )
    )

    endlocal
)

echo Processing completed.