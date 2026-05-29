@echo off
setlocal enabledelayedexpansion

:: pdf2png.bat — Convert PDF files to page images using ImageMagick + Ghostscript.
::
:: Converts every PDF in a directory to per-page image files.  Each PDF gets its
:: own subdirectory of numbered images.
::
:: Note on parallelism: this script processes PDFs sequentially.  For parallel
:: conversion on Windows, use the companion PowerShell script pdf2png.ps1 which
:: leverages ForEach-Object -Parallel (requires PowerShell 7+).
::
:: Usage:
::   pdf2png.bat [OPTIONS]
::
:: Options:
::   /f FORMAT    Output format: png or jpg    (default: png)
::   /r DPI       Output resolution in DPI     (default: 300)
::   /d DIR       Directory with PDF files     (default: current directory)
::   /o DIR       Output root directory        (default: same as /d)
::   /k           Keep original PDFs after conversion (default: delete on success)
::   /?           Show this help message and exit
::
:: Output structure:
::   <output>\<pdf-name>\<pdf-name>-1.png   (ImageMagick does NOT zero-pad)
::
:: Examples:
::   pdf2png.bat
::   pdf2png.bat /f jpg /r 200
::   pdf2png.bat /d C:\data\pdfs /o C:\data\pages /k

:: ── Defaults ──────────────────────────────────────────────────────────────
set "FORMAT=png"
set "DPI=300"
set "SOURCE_DIR=%CD%"
set "OUTPUT_DIR="
set "KEEP=false"

:: ── Argument parsing ───────────────────────────────────────────────────────
:parse_args
if "%~1"=="" goto check_deps
if /i "%~1"=="/f" ( set "FORMAT=%~2"     & shift & shift & goto parse_args )
if /i "%~1"=="/r" ( set "DPI=%~2"        & shift & shift & goto parse_args )
if /i "%~1"=="/d" ( set "SOURCE_DIR=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="/o" ( set "OUTPUT_DIR=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="/k" ( set "KEEP=true"      & shift         & goto parse_args )
if /i "%~1"=="/?" goto show_help
echo Unknown option: %~1
goto show_help

:show_help
echo.
echo Usage: pdf2png.bat [OPTIONS]
echo.
echo Options:
echo   /f FORMAT    Output format: png or jpg  ^(default: png^)
echo   /r DPI       Output resolution in DPI   ^(default: 300^)
echo   /d DIR       Directory with PDF files   ^(default: current directory^)
echo   /o DIR       Output root directory      ^(default: same as /d^)
echo   /k           Keep original PDFs after conversion
echo   /?           Show this help message
echo.
echo Example:
echo   pdf2png.bat /f jpg /r 200 /d C:\data\pdfs
exit /b 0

:: ── Dependency checks ─────────────────────────────────────────────────────
:check_deps

:: Validate format
if /i "%FORMAT%"=="png" ( set "MAGICK_FORMAT=png24" & set "EXT=png" & goto check_gs )
if /i "%FORMAT%"=="jpg" ( set "MAGICK_FORMAT=jpeg"  & set "EXT=jpg" & goto check_gs )
if /i "%FORMAT%"=="jpeg"( set "MAGICK_FORMAT=jpeg"  & set "EXT=jpg" & goto check_gs )
echo Error: /f FORMAT must be 'png' or 'jpg' ^(got '%FORMAT%'^).
exit /b 1

:check_gs
:: Check for Ghostscript — ImageMagick PDF conversion requires it.
:: Try 64-bit first, then 32-bit as a fallback.
set "GS_FOUND=false"
gswin64c -v >nul 2>&1 && set "GS_FOUND=true"
if "%GS_FOUND%"=="false" (
    gswin32c -v >nul 2>&1 && set "GS_FOUND=true"
)
if "%GS_FOUND%"=="false" (
    echo [ERROR] Ghostscript is not installed or not in your system PATH.
    echo         ImageMagick PDF conversion requires Ghostscript.
    echo         Download from: https://www.ghostscript.com/releases/gsdnld.html
    exit /b 1
)

:: Check for ImageMagick
magick -version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ImageMagick ^(magick^) is not installed or not in your system PATH.
    echo         Download from: https://imagemagick.org/script/download.php#windows
    exit /b 1
)

:: ── Resolve output directory ──────────────────────────────────────────────
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=%SOURCE_DIR%"
if not exist "%OUTPUT_DIR%\" mkdir "%OUTPUT_DIR%"

if not exist "%SOURCE_DIR%\" (
    echo Error: source directory "%SOURCE_DIR%" does not exist.
    exit /b 1
)

echo Source dir : %SOURCE_DIR%
echo Output dir : %OUTPUT_DIR%
echo Format     : %EXT%  ^(DPI: %DPI%^)
if "%KEEP%"=="true" (echo Keep PDFs  : yes) else (echo Keep PDFs  : no ^(delete on success^))
echo.

:: ── Main conversion loop ──────────────────────────────────────────────────
:: Uses an inline for /f loop — no temporary file required.
set /a "CONVERTED=0"
set /a "FAILED=0"

for /f "tokens=*" %%F in ('dir /b "%SOURCE_DIR%\*.pdf" 2^>nul') do (
    set "PDF_FILE=%SOURCE_DIR%\%%F"
    set "BASENAME=%%~nF"
    set "OUT_SUBDIR=%OUTPUT_DIR%\%%~nF"

    if not exist "!OUT_SUBDIR!\" mkdir "!OUT_SUBDIR!"

    magick -density %DPI% -scene 1 "!PDF_FILE!" ^
           -background white -alpha remove -alpha off ^
           "%MAGICK_FORMAT%:!OUT_SUBDIR!\!BASENAME!-%%d.%EXT%"

    if errorlevel 1 (
        echo Failed:    !PDF_FILE!
        set /a "FAILED+=1"
    ) else (
        echo Converted: !PDF_FILE!  →  !OUT_SUBDIR!\!BASENAME!-*.%EXT%
        set /a "CONVERTED+=1"
        if "%KEEP%"=="false" del "!PDF_FILE!"
    )
)

:: ── Summary ───────────────────────────────────────────────────────────────
echo.
echo Done. Converted %CONVERTED% PDF^(s^) ^| %FAILED% failed.
endlocal
