@echo off
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
@REM     magick -density %DPI% -background white -scene 1 -quality 100 "%%F" "%%~nF\%%~nF-%%d.png"
    magick -density %DPI% "%%F" -background white -alpha remove -alpha off png24:"%%~nF/%%~nF-%%d.png"
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
