import csv
import os
import sys
from collections import defaultdict

def split_csv_and_aggregate(input_file, output_dir='./per_document/'):
    # Normalize the path to ensure it ends with a separator
    output_dir = os.path.join(output_dir, '')

    # 1. Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        print(f"Processing '{input_file}'...")

        with open(input_file, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)

            # 2. Grab the header (to write it to every new file)
            try:
                header = next(reader)
            except StopIteration:
                print("Error: Input file is empty.")
                return

            # 3. Group rows in memory (Aggregating Duplicates)
            # We use a dictionary where Key = Filename, Value = List of Rows
            grouped_rows = defaultdict(list)

            row_count = 0
            for row in reader:
                if row:
                    # Key is the first column (e.g., "document1.pdf")
                    # We strip whitespace to avoid "file.pdf" and "file.pdf " being separate
                    key = row[0].strip()
                    grouped_rows[key].append(row)
                    row_count += 1

            print(f"Loaded {row_count} rows. Aggregated into {len(grouped_rows)} unique files.")

            # 4. Write the grouped data to separate files
            for filename_key, rows in grouped_rows.items():
                # Create a safe filename (e.g., "document1.pdf" -> "document1.pdf.csv")
                # We replace path separators just in case the csv contains paths like "folder/doc.pdf"
                safe_filename = filename_key.replace('/', '_').replace('\\', '_') + ".csv"
                output_path = os.path.join(output_dir, safe_filename)

                with open(output_path, mode='w', encoding='utf-8', newline='') as out_f:
                    writer = csv.writer(out_f)
                    writer.writerow(header)  # Write header first
                    writer.writerows(rows)   # Write all aggregated rows for this file

                # Optional: Print progress for large batches
                # print(f"Wrote: {output_path} ({len(rows)} rows)")

        print(f"Done! Check the '{output_dir}' folder.")

    except FileNotFoundError:
        print(f"Error: Could not find file '{input_file}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_by_file.py <your_file.csv> [output_dir]")
    else:
        input_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else './per_document/'
        split_csv_and_aggregate(input_file, output_dir)