#!/bin/bash

# Set the input and output directories
input_dir="/lnet/work/projects/atrium/ARUP_HDD_PDF"
output_dir="/lnet/work/people/lutsai/pythonProject/pages/train_final"
input_csv="/lnet/work/people/lutsai/pythonProject/pages/input.csv"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Read the CSV file line by line (skip the header row if necessary)
# CSV file should have exactly 3 columns for file, page, and category
tail -n +2 "$input_csv" | while IFS=',' read -r filename page_number category; do
  # Create category subdirectory inside output directory if it doesn't exist
  category_dir="$output_dir/$category"
  mkdir -p "$category_dir"

  # Check if subdirectory exists in input directory
  input_subdir="$input_dir/$filename"
  if [ -d "$input_subdir" ]; then
    # Copy the file ending with the page number to the category subdirectory

    file_count=$(ls -1q "$input_subdir"/*.png 2>/dev/null | wc -l)
    pn=$(printf "%0${#file_count}d" "$page_number")


    file_to_copy=$(find "$input_subdir" -type f -name "*-$pn.png" -print -quit)
    if [ -n "$file_to_copy" ]; then
      cp "$file_to_copy" "$category_dir/"
    else
      echo "File ending with $page_number not found in $input_subdir."
    fi
  else
    # Use 'onepagers' as the default subdirectory
    default_subdir="$input_dir/onepagers"
    file_to_copy=$(find "$default_subdir" -type f -name "$filename-$page_number.png" -print -quit)
    if [ -n "$file_to_copy" ]; then
      cp "$file_to_copy" "$category_dir/"
    else
      echo "File starting with $filename and ending with $page_number not found in $default_subdir."
    fi
  fi
done

echo "Processing completed."
