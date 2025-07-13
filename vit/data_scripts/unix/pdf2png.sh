#!/bin/bash

# Set output image quality (300 DPI)
DPI=300

# Find all PDF files in the current directory and process them in parallel
find . -name "*.pdf" | while read -r pdf_file; do
    {
        # Get the filename without extension
        filename=$(basename "$pdf_file" .pdf)

        # Create a directory for the images from this PDF file
        mkdir -p "$filename"

        # Convert each page of the PDF to a PNG image in the specified folder
        pdftoppm -png -r $DPI "$pdf_file" "$filename/$filename"

        # Check if conversion was successful
        if [ $? -eq 0 ]; then
            echo "Converted $pdf_file to PNG images in folder $filename"
            # Remove the original PDF file
            rm "$pdf_file"
        else
            echo "Failed to convert $pdf_file"
        fi
    } &
done

# Wait for all parallel jobs to complete
wait

