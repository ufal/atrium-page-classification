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

    # --- REFINED: Construct the target path directly to avoid I/O bottlenecks ---
    # Instead of executing 'find' which scans the whole directory on every loop,
    # we check for the file using known padding formats natively.
    file_found=false
    for pad in "" "%02d" "%03d" "%04d"; do
        if [ -z "$pad" ]; then
            pn="$page_number"
        else
            pn=$(printf "$pad" "$page_number")
        fi

        expected_file="$input_subdir/${filename}-${pn}.png"

        if [ -f "$expected_file" ]; then
            cp "$expected_file" "$category_dir/"
            file_found=true
            break
        fi
    done

    if [ "$file_found" = false ]; then
      echo "File ending with $page_number not found in $input_subdir."
    fi

  else
    # Use 'onepagers' as the default subdirectory
    default_subdir="$input_dir/onepagers"
    expected_file="$default_subdir/${filename}-${page_number}.png"

    if [ -f "$expected_file" ]; then
      cp "$expected_file" "$category_dir/"
    else
      echo "File starting with $filename and ending with $page_number not found in $default_subdir."
    fi
  fi
done

echo "Processing completed."