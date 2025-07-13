#!/bin/bash

# Define the target directory
TARGET_DIR="./onepagers"

# Create the target directory if it doesn't exist
#mkdir -p "$TARGET_DIR"

# Iterate through all subdirectories in the current directory
for dir in */; do
    # Check if the subdirectory contains exactly one file
    if [ "$(find "$dir" -type f | wc -l)" -eq 1 ]; then
        # Find the single file in the subdirectory
        file=$(find "$dir" -type f)
        echo $file
        # Move the file to the target directory
        mv "$file" "$TARGET_DIR/"
        # Remove the now-empty subdirectory
        rmdir "$dir"
    fi
done

echo "Files moved to '$TARGET_DIR' and empty directories removed."

