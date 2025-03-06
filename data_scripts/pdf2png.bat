@echo off

:: Set output image quality (300 DPI)
set DPI=300

:: Create a list of all PDF files in the current directory
setlocal enabledelayedexpansion
set FILE_LIST=
for %%F in (*.pdf) do (
    set FILE_LIST=!FILE_LIST!"%%F"
)

:: Iterate through the list of PDF files
for %%F in (%FILE_LIST%) do (
    :: Get the filename without extension
    set "FILENAME=%%~nF"

    :: Create a directory for the images from this PDF file
    if not exist "!FILENAME!" mkdir "!FILENAME!"

    :: Convert each page of the PDF to a PNG image in the specified folder
    magick -density %DPI% -background white -scene 1 -quality 100 "%%~F" "!FILENAME!\%%~nF-%%d.png"

    :: Check if conversion was successful
    if not errorlevel 1 (
        echo Converted %%~F to PNG images in folder !FILENAME!

        del "%%~F"
    ) else (
        echo Failed to convert %%~F
    )
)

echo All PDFs processed.
endlocal
