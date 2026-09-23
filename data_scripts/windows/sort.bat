@echo off
setlocal enabledelayedexpansion

:: sort.bat — Copy (or move) annotated PNG pages into label-specific subdirectories.
::
:: Reads a CSV annotation file with FILE, PAGE, CLASS columns and copies the
:: matching PNG from the document-specific subdirectory structure into a
:: label-sorted output directory suitable for model training.
::
:: Usage:
::   sort.bat /i INPUT_DIR /o OUTPUT_DIR /c CSV_FILE [/move] [/n] [/?]
::
:: Options:
::   /i INPUT_DIR   Directory containing document-specific PNG subdirectories
::   /o OUTPUT_DIR  Target directory for label-sorted training pages
::   /c CSV_FILE    CSV annotation file: a header row, then FILE, PAGE, CLASS as
::                  the FIRST THREE columns (any further columns are ignored)
::   /move          Move files instead of copying (default: copy)
::   /n             Dry run: show what would happen without making any changes
::   /?             Show this help message and exit
::
:: Unlike sort.sh, this script reads the columns by position, not by header name.
::
:: Rows that are skipped and reported, never acted on:
::   * a PAGE that is not a whole number (leading zeros are fine: 08 is page 8);
::   * an empty CLASS, "." or "..", or one containing "\" or "/".
:: A label folder is only created once a page has been found for it.
::
:: The script handles zero-padded page numbers automatically (1, 01, 001, 0001),
:: both in the document's own subdirectory and in the "onepagers" fallback.
:: Documents with no subdirectory fall back to an "onepagers" subdirectory inside
:: INPUT_DIR, mirroring the behaviour of pdf2png.bat / move_single.bat.
::
:: Example:
::   sort.bat /i C:\data\pages /o C:\data\train /c C:\data\annotations.csv
::   sort.bat /i C:\data\pages /o C:\data\train /c C:\data\annotations.csv /move
::   sort.bat /i C:\data\pages /o C:\data\train /c C:\data\annotations.csv /n

:: ── Argument parsing ──────────────────────────────────────────────────────
set "INPUT_DIR="
set "OUTPUT_DIR="
set "INPUT_CSV="
set "USE_MOVE=false"
set "DRY_RUN=false"

:parse_args
if "%~1"=="" goto check_args
if /i "%~1"=="/i" ( set "INPUT_DIR=%~2"  & shift & shift & goto parse_args )
if /i "%~1"=="/o" ( set "OUTPUT_DIR=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="/c" ( set "INPUT_CSV=%~2"  & shift & shift & goto parse_args )
if /i "%~1"=="/move" ( set "USE_MOVE=true" & shift & goto parse_args )
if /i "%~1"=="/n" ( set "DRY_RUN=true" & shift & goto parse_args )
if /i "%~1"=="/?" goto show_help
echo Unknown option: %~1
goto show_help

:show_help
echo.
echo Usage: sort.bat /i INPUT_DIR /o OUTPUT_DIR /c CSV_FILE [/move] [/n] [/?]
echo.
echo Options:
echo   /i INPUT_DIR   Directory with document-specific PNG subdirectories
echo   /o OUTPUT_DIR  Target directory for label-sorted training data
echo   /c CSV_FILE    CSV file: header row, then FILE, PAGE, CLASS as the first
echo                  three columns ^(further columns are ignored^)
echo   /move          Move files instead of copying ^(default: copy^)
echo   /n             Dry run: show what would happen, change nothing
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
if "%DRY_RUN%"=="false" if not exist "%OUTPUT_DIR%\" mkdir "%OUTPUT_DIR%"

echo Input dir : %INPUT_DIR%
echo Output dir: %OUTPUT_DIR%
echo CSV file  : %INPUT_CSV%
if "%USE_MOVE%"=="true" (echo Mode      : move) else (echo Mode      : copy)
if "%DRY_RUN%"=="true" echo Dry-run   : yes
echo.

:: ── Counters ──────────────────────────────────────────────────────────────
set /a "COPIED=0"
set /a "NOT_FOUND=0"
set /a "INVALID=0"

:: ── Main loop ─────────────────────────────────────────────────────────────
:: usebackq + a quoted path, so a CSV path containing spaces is read as a file
:: rather than taken as a literal string.
for /f "usebackq tokens=1,2,3 delims=, skip=1" %%A in ("%INPUT_CSV%") do (
    set "FILENAME=%%A"
    set "PAGE_NUMBER=%%B"
    set "CATEGORY=%%C"

    rem Strip any trailing carriage-return from values read by for /f
    for /f "delims=" %%X in ("!CATEGORY!") do set "CATEGORY=%%X"
    rem Drop the double quotes of quoted CSV fields, and spaces around PAGE and CLASS
    if defined FILENAME set "FILENAME=!FILENAME:"=!"
    if defined PAGE_NUMBER set "PAGE_NUMBER=!PAGE_NUMBER:"=!"
    if defined PAGE_NUMBER set "PAGE_NUMBER=!PAGE_NUMBER: =!"
    if defined CATEGORY set "CATEGORY=!CATEGORY:"=!"
    if defined CATEGORY set "CATEGORY=!CATEGORY: =!"

    rem PAGE must be a whole number: remove every digit and see whether anything is left.
    set "PAGE_OK=false"
    if defined PAGE_NUMBER (
        set "PAGE_REST=!PAGE_NUMBER!"
        for %%D in (0 1 2 3 4 5 6 7 8 9) do if defined PAGE_REST set "PAGE_REST=!PAGE_REST:%%D=!"
        if not defined PAGE_REST set "PAGE_OK=true"
    )

    rem CLASS becomes a folder name: refuse anything that is not one path component.
    set "LABEL_OK=true"
    if not defined CATEGORY (
        set "LABEL_OK=false"
    ) else (
        if "!CATEGORY!"=="." set "LABEL_OK=false"
        if "!CATEGORY!"==".." set "LABEL_OK=false"
        if not "!CATEGORY:\=!"=="!CATEGORY!" set "LABEL_OK=false"
        if not "!CATEGORY:/=!"=="!CATEGORY!" set "LABEL_OK=false"
    )

    if "!PAGE_OK!"=="false" (
        echo Invalid page: !FILENAME! page '!PAGE_NUMBER!' -- skipped
        set /a "INVALID+=1"
    ) else if "!LABEL_OK!"=="false" (
        echo Invalid label: !FILENAME! page !PAGE_NUMBER! label '!CATEGORY!' -- skipped
        set /a "INVALID+=1"
    ) else (
        rem Strip leading zeros, so 08 becomes 8: the padding below starts from the
        rem plain number, and 08 is never read as an octal value.
        for /l %%Z in (1,1,10) do (
            if not "!PAGE_NUMBER!"=="0" if "!PAGE_NUMBER:~0,1!"=="0" set "PAGE_NUMBER=!PAGE_NUMBER:~1!"
        )

        rem Documents without their own subdirectory live in the flat onepagers one.
        set "SEARCH_DIR=%INPUT_DIR%\!FILENAME!"
        if not exist "!SEARCH_DIR!\" set "SEARCH_DIR=%INPUT_DIR%\onepagers"

        rem Try zero to four leading zeros by building each expected filename
        rem directly rather than scanning the directory. This handles both Windows
        rem ImageMagick output, unpadded, and Unix pdftoppm output, auto-padded to
        rem the width of the page count. The same widths apply in onepagers.
        set "FOUND="
        for %%W in (0 1 2 3 4) do (
            if not defined FOUND (
                call :pad_page "!PAGE_NUMBER!" %%W PADDED_PAGE
                if exist "!SEARCH_DIR!\!FILENAME!-!PADDED_PAGE!.png" set "FOUND=!SEARCH_DIR!\!FILENAME!-!PADDED_PAGE!.png"
            )
        )

        if not defined FOUND (
            echo Not found: !FILENAME! page !PAGE_NUMBER!
            set /a "NOT_FOUND+=1"
        ) else (
            rem Only now, with a page in hand, is the label folder created: an
            rem empty folder would still count as a category to training.
            set "CATEGORY_DIR=%OUTPUT_DIR%\!CATEGORY!"
            if "%DRY_RUN%"=="true" (
                if "%USE_MOVE%"=="true" (
                    echo [dry-run] move: !FOUND! to !CATEGORY_DIR!\
                ) else (
                    echo [dry-run] copy: !FOUND! to !CATEGORY_DIR!\
                )
            ) else (
                if not exist "!CATEGORY_DIR!\" mkdir "!CATEGORY_DIR!"
                if "%USE_MOVE%"=="true" (
                    move /y "!FOUND!" "!CATEGORY_DIR!\" >nul
                ) else (
                    copy /y "!FOUND!" "!CATEGORY_DIR!\" >nul
                )
            )
            set /a "COPIED+=1"
        )
    )
)

:: ── Summary ───────────────────────────────────────────────────────────────
echo.
set "PREFIX="
if "%DRY_RUN%"=="true" set "PREFIX=[dry-run] "
if "%USE_MOVE%"=="true" (
    echo %PREFIX%Done. Moved %COPIED% file^(s^) ^| %NOT_FOUND% page^(s^) not found ^| %INVALID% invalid row^(s^) skipped.
) else (
    echo %PREFIX%Done. Copied %COPIED% file^(s^) ^| %NOT_FOUND% page^(s^) not found ^| %INVALID% invalid row^(s^) skipped.
)
endlocal
exit /b 0

:: ── Subroutine: prefix PAGE_NUMBER with W zeros (W = 0..4) ─────────────────
:pad_page <number> <zeros> <result_var>
set "_N=%~1"
set "_W=%~2"
set "_PAD=00000"
set "_PADDED=!_PAD:~0,%_W%!!_N!"
set "%~3=!_PADDED!"
exit /b 0
