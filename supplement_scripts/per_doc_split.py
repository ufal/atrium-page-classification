import csv
import os
import sys
from collections import defaultdict


def split_csv_and_aggregate(input_file, output_dir='./per_document/'):
    output_dir = os.path.join(output_dir, '')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        print(f"Processing '{input_file}' in memory-safe stream mode...")
        seen_files = set()
        row_count = 0

        with open(input_file, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                print("Error: Input file is empty.")
                return

            # --- REFINED: Write sequentially directly to disk (OOM Safe) ---
            for row in reader:
                if not row: continue

                key = row[0].strip()
                safe_filename = key.replace('/', '_').replace('\\', '_') + ".csv"
                output_path = os.path.join(output_dir, safe_filename)

                # If we haven't seen this file yet, overwrite ('w') and write header.
                # If we have, append ('a').
                mode = 'a' if key in seen_files else 'w'

                with open(output_path, mode=mode, encoding='utf-8', newline='') as out_f:
                    writer = csv.writer(out_f)
                    if mode == 'w':
                        writer.writerow(header)
                        seen_files.add(key)
                    writer.writerow(row)

                row_count += 1

        print(f"Loaded {row_count} rows. Aggregated into {len(seen_files)} unique files.")
        print(f"Done! Check the '{output_dir}' folder.")

    except Exception as e:
        print(f"An error occurred: {e}")




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_by_file.py <your_file.csv> [output_dir]")
    else:
        input_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else './per_document/'
        split_csv_and_aggregate(input_file, output_dir)