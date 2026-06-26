@echo off
setlocal enabledelayedexpansion

:: sort.bat — Copy (or move) annotated PNG pages into label-specific subdirectories.
::
:: Reads a CSV annotation file with FILE, PAGE, CLASS columns and copies the
:: matching PNG from the document-specific subdirectory structure into a
:: label-sorted output directory suitable for model training.
::
:: Usage:
::   sort.bat /i INPUT_DIR /o OUTPUT_DIR /c CSV_FILE [/move] [/?]
::
:: Options:
::   /i INPUT_DIR   Directory containing document-specific PNG subdirectories
::   /o OUTPUT_DIR  Target directory for label-sorted training pages
::   /c CSV_FILE    CSV annotation file with columns: FILE, PAGE, CLASS
::   /move          Move files instead of copying (default: copy)
::   /?             Show this help message and exit
::
:: The script handles zero-padded page numbers automatically (1, 01, 001, 0001).
:: Documents with no subdirectory fall back to an "onepagers" subdirectory inside
:: INPUT_DIR, mirroring the behaviour of pdf2png.bat / move_single.bat.
::
:: Example:
::   sort.bat /i C:\data\pages /o C:\data\train /c C:\data\annotations.csv
::   sort.bat /i C:\data\pages /o C:\data\train /c C:\data\annotations.csv /move

:: ── Argument parsing ──────────────────────────────────────────────────────
set "INPUT_DIR="
set "OUTPUT_DIR="
set "INPUT_CSV="
set "USE_MOVE=false"

:parse_args
if "%~1"=="" goto check_args
if /i "%~1"=="/i" ( set "INPUT_DIR=%~2"  & shift & shift & goto parse_args )
if /i "%~1"=="/o" ( set "OUTPUT_DIR=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="/c" ( set "INPUT_CSV=%~2"  & shift & shift & goto parse_args )
if /i "%~1"=="/move" ( set "USE_MOVE=true" & shift & goto parse_args )
if /i "%~1"=="/?" goto show_help
echo Unknown option: %~1
goto show_help

:show_help
echo.
echo Usage: sort.bat /i INPUT_DIR /o OUTPUT_DIR /c CSV_FILE [/move] [/?]
echo.
echo Options:
echo   /i INPUT_DIR   Directory with document-specific PNG subdirectories
echo   /o OUTPUT_DIR  Target directory for label-sorted training data
echo   /c CSV_FILE    CSV file with columns: FILE, PAGE, CLASS ^(header expected^)
echo   /move          Move files instead of copying ^(default: copy^)
echo   /?             Show this help message
echo.
echo Example:
echo   sort.bat /i C:\data\pages /o C:\data\train /c C:\data\annotations.csv
exit /b 0

:check_args
if "%INPUT_DIR%"==""  ( echo Error: /i INPUT_DIR is required.  & echo. & goto show_help )
if "%OUTPUT_DIR%"=="" ( echo Error: /o OUTPUT_DIR is required. & echo. & goto show_help )
if "%INPUT_CSV%"==""  ( echo Error: /c CSV_FILE is required.   & echo. & goto show_help )

:: ── Validation ────────────────────────────────────────────────────────────
if not exist "%INPUT_DIR%\"  ( echo Error: INPUT_DIR "%INPUT_DIR%" does not exist.  & exit /b 1 )
if not exist "%INPUT_CSV%"   ( echo Error: CSV_FILE "%INPUT_CSV%" does not exist.   & exit /b 1 )
if not exist "%OUTPUT_DIR%\" mkdir "%OUTPUT_DIR%"

echo Input dir : %INPUT_DIR%
echo Output dir: %OUTPUT_DIR%
echo CSV file  : %INPUT_CSV%
if "%USE_MOVE%"=="true" (echo Mode      : move) else (echo Mode      : copy)
echo.

:: ── Counters ──────────────────────────────────────────────────────────────
set /a "COPIED=0"
set /a "NOT_FOUND=0"

:: ── Main loop ─────────────────────────────────────────────────────────────
for /f "tokens=1,2,3 delims=, skip=1" %%A in (%INPUT_CSV%) do (
    set "FILENAME=%%A"
    set "PAGE_NUMBER=%%B"
    set "CATEGORY=%%C"

    :: Strip any trailing carriage-return from values read by for /f
    for /f "delims=" %%X in ("!CATEGORY!") do set "CATEGORY=%%X"
    set "CATEGORY=!CATEGORY: =!"

    :: Create the label subdirectory on first encounter
    set "CATEGORY_DIR=%OUTPUT_DIR%\!CATEGORY!"
    if not exist "!CATEGORY_DIR!\" mkdir "!CATEGORY_DIR!"

    set "INPUT_SUBDIR=%INPUT_DIR%\!FILENAME!"
    set "FILE_FOUND=false"

    if exist "!INPUT_SUBDIR!\" (
        :: P3 FIX: try all padding widths (0-4 leading zeros) by building each
        :: expected filename directly rather than scanning the directory.
        :: This handles both Windows ImageMagick output (no padding) and Unix
        :: pdftoppm output (auto-padded to match the page count length).
        for %%W in (0 1 2 3 4) do (
            if "!FILE_FOUND!"=="false" (
                call :pad_page "!PAGE_NUMBER!" %%W PADDED_PAGE
                set "EXPECTED=!INPUT_SUBDIR!\!FILENAME!-!PADDED_PAGE!.png"
                if exist "!EXPECTED!" (
                    if "%USE_MOVE%"=="true" (
                        move /y "!EXPECTED!" "!CATEGORY_DIR!\" >nul
                    ) else (
                        copy /y "!EXPECTED!" "!CATEGORY_DIR!\" >nul
                    )
                    set "FILE_FOUND=true"
                    set /a "COPIED+=1"
                )
            )
        )
    ) else (
        :: Fall back to the onepagers flat directory
        set "DEFAULT_SUBDIR=%INPUT_DIR%\onepagers"
        set "EXPECTED=!DEFAULT_SUBDIR!\!FILENAME!-!PAGE_NUMBER!.png"
        if exist "!EXPECTED!" (
            if "%USE_MOVE%"=="true" (
                move /y "!EXPECTED!" "!CATEGORY_DIR!\" >nul
            ) else (
                copy /y "!EXPECTED!" "!CATEGORY_DIR!\" >nul
            )
            set "FILE_FOUND=true"
            set /a "COPIED+=1"
        )
    )

    if "!FILE_FOUND!"=="false" (
        echo Not found: !FILENAME! page !PAGE_NUMBER!
        set /a "NOT_FOUND+=1"
    )
)

:: ── Summary ───────────────────────────────────────────────────────────────
echo.
if "%USE_MOVE%"=="true" (
    echo Done. Moved %COPIED% file^(s^) ^| %NOT_FOUND% page^(s^) not found.
) else (
    echo Done. Copied %COPIED% file^(s^) ^| %NOT_FOUND% page^(s^) not found.
)
endlocal
exit /b 0

:: ── Subroutine: left-pad PAGE_NUMBER with zeros to width W ────────────────
:pad_page <number> <width> <result_var>
set "_N=%~1"
set "_W=%~2"
set "_PAD=00000"
set "_PADDED=!_PAD:~0,%_W%!!_N!"
set "%~3=!_PADDED!"
exit /b 0
