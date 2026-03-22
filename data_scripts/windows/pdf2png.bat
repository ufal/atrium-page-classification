@echo off

:: --- REFINED: Safety check for Ghostscript ---
:: ImageMagick PDF conversion relies on Ghostscript. We verify its existence first.
gswin64c -v >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ghostscript is not installed or not in your system PATH.
    echo ImageMagick PDF conversion will silently fail without it.
    exit /b 1
)

:: Set output image quality (300 DPI)
set DPI=300

:: Create a list of all PDF files in the current directory
dir /b *.pdf > pdf_files.txt

:: Iterate through the list of PDF files
for /f "tokens=*" %%F in (pdf_files.txt) do (
    echo Processing: %%F

    :: Get the filename without extension
    set "FILENAME=%%~nF"

    :: Create a directory for the images from this PDF file
    if not exist "%%~nF" mkdir "%%~nF"

    :: Convert each page of the PDF to a PNG image in the specified folder
    magick -density %DPI% -scene 1 "%%F" -background white -alpha remove -alpha off png24:"%%~nF/%%~nF-%%d.png"

    :: Check if conversion was successful
    if not errorlevel 1 (
        echo Converted %%F to PNG images in folder %%~nF
        del "%%F"
    ) else (
        echo Failed to convert %%F
    )
)

:: Clean up temporary file
del pdf_files.txt
echo All PDFs processed.