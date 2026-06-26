@echo off
setlocal enabledelayedexpansion

:: move_single.bat — Move single-file subdirectories into a shared flat folder.
::
:: Scans a source directory for subdirectories that contain exactly one file
:: and moves that file into a common target directory, then removes the now-
:: empty subdirectory.  Used as a pre-processing step so single-page PDFs land
:: in a flat "onepagers" folder rather than redundant one-entry subdirectories.
::
:: Usage:
::   move_single.bat [OPTIONS]
::
:: Options:
::   /s DIR   Source directory to scan          (default: current directory)
::   /t DIR   Target directory for moved files  (default: .\onepagers)
::   /n       Dry-run: preview moves without executing them
::   /?       Show this help message and exit
::
:: Examples:
::   move_single.bat
::   move_single.bat /s C:\data\converted_pdfs /t C:\data\onepagers
::   move_single.bat /s C:\data\converted_pdfs /n

:: ── Defaults ──────────────────────────────────────────────────────────────
set "SOURCE_DIR=%CD%"
set "TARGET_DIR=%CD%\onepagers"
set "DRY_RUN=false"

:: ── Argument parsing ───────────────────────────────────────────────────────
:parse_args
if "%~1"=="" goto validate
if /i "%~1"=="/s" ( set "SOURCE_DIR=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="/t" ( set "TARGET_DIR=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="/n" ( set "DRY_RUN=true"   & shift         & goto parse_args )
if /i "%~1"=="/?" goto show_help
echo Unknown option: %~1
goto show_help

:show_help
echo.
echo Usage: move_single.bat [OPTIONS]
echo.
echo Options:
echo   /s DIR   Source directory to scan          ^(default: current directory^)
echo   /t DIR   Target directory for moved files  ^(default: .\onepagers^)
echo   /n       Dry-run: preview moves without making any changes
echo   /?       Show this help message
echo.
echo Example:
echo   move_single.bat /s C:\data\converted_pdfs /t C:\data\onepagers
exit /b 0

:validate
if not exist "%SOURCE_DIR%\" (
    echo Error: source directory "%SOURCE_DIR%" does not exist.
    exit /b 1
)

if "%DRY_RUN%"=="false" (
    if not exist "%TARGET_DIR%\" mkdir "%TARGET_DIR%"
)

echo Source dir: %SOURCE_DIR%
echo Target dir: %TARGET_DIR%
if "%DRY_RUN%"=="true" echo Dry-run   : yes
echo.

:: ── Counters ──────────────────────────────────────────────────────────────
set /a "MOVED=0"
set /a "SKIPPED=0"

:: ── Main loop ─────────────────────────────────────────────────────────────
for /d %%D in ("%SOURCE_DIR%\*") do (
    :: Skip the target directory itself to avoid moving files into onepagers
    :: and then re-scanning them in the same run.
    if /i not "%%~fD"=="%TARGET_DIR%" (
        :: Count files in this subdirectory (not recursive).
        :: setlocal/endlocal is used here so FILE_COUNT and LAST_FILE are
        :: reset cleanly for every iteration of the outer for /d loop.
        setlocal
        set /a "_COUNT=0"
        set "_LAST="
        for %%F in ("%%D\*") do (
            set /a "_COUNT+=1"
            set "_LAST=%%F"
        )

        if !_COUNT! equ 1 (
            if "%DRY_RUN%"=="true" (
                echo [dry-run] move:   !_LAST! → %TARGET_DIR%\
                echo [dry-run] remove: %%D\
            ) else (
                move "!_LAST!" "%TARGET_DIR%\" >nul
                rmdir "%%D"
            )
            endlocal
            set /a "MOVED+=1"
        ) else (
            endlocal
            set /a "SKIPPED+=1"
        )
    )
)

:: ── Summary ───────────────────────────────────────────────────────────────
echo.
if "%DRY_RUN%"=="true" (
    echo [dry-run] Would move %MOVED% file^(s^) to "%TARGET_DIR%" ^| %SKIPPED% multi-file director^(ies^) skipped.
) else (
    echo Done. Moved %MOVED% file^(s^) to "%TARGET_DIR%" ^| %SKIPPED% multi-file director^(ies^) skipped.
)
endlocal
