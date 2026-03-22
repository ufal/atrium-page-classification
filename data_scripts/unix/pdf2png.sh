#!/bin/bash

# Set output image quality (300 DPI)
DPI=300

# --- REFINED: Bounded parallelism using xargs ---
# Find all PDF files in the current directory and process them in parallel safely.
# -print0 and -0 safely handle filenames containing spaces.
# -P $(nproc) limits the number of concurrent jobs to the number of available CPU cores.
find . -name "*.pdf" -print0 | xargs -0 -I {} -P $(nproc) sh -c '
    pdf_file="{}"

    # Get the filename without extension
    filename=$(basename "$pdf_file" .pdf)

    # Create a directory for the images from this PDF file
    mkdir -p "$filename"

    # Convert each page of the PDF to a PNG image in the specified folder
    pdftoppm -png -r '$DPI' "$pdf_file" "$filename/$filename"

    # Check if conversion was successful
    if [ $? -eq 0 ]; then
        echo "Converted $pdf_file to PNG images in folder $filename"
        # Remove the original PDF file
        rm "$pdf_file"
    else
        echo "Failed to convert $pdf_file"
    fi
'

echo "All PDFs processed."