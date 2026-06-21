import argparse
import csv
import os


def split_csv_and_aggregate(input_file: str, output_dir: str = './per_document/') -> None:
    """Split a flat result CSV into one file per unique document in the FILE column.

    Rows are streamed directly to disk so arbitrarily large CSVs can be split
    without loading the entire file into memory.

    Args:
        input_file: Path to the result CSV (FILE, PAGE, CLASS-N, SCORE-N …).
        output_dir: Directory where per-document CSV files will be written.
                    Created automatically if it does not exist.
    """
    output_dir = os.path.join(output_dir, '')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        print(f"Processing '{input_file}' in memory-safe stream mode...")
        seen_files: set = set()
        row_count = 0

        with open(input_file, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                print("Error: Input file is empty.")
                return

            for row in reader:
                if not row:
                    continue

                key = row[0].strip()
                safe_filename = key.replace('/', '_').replace('\\', '_') + ".csv"
                output_path = os.path.join(output_dir, safe_filename)

                # First occurrence of a document key: write header + row.
                # Subsequent occurrences: append the row only.
                mode = 'a' if key in seen_files else 'w'

                with open(output_path, mode=mode, encoding='utf-8', newline='') as out_f:
                    writer = csv.writer(out_f)
                    if mode == 'w':
                        writer.writerow(header)
                        seen_files.add(key)
                    writer.writerow(row)

                row_count += 1

        print(f"Processed {row_count} rows → {len(seen_files)} per-document file(s) in '{output_dir}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# P1 FIX: the original used raw sys.argv which made -i/--input (as documented
# in the README) silently pass the literal string "-i" as a filename.
# Replaced with argparse to match the documented interface.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a flat result CSV into one file per document in the FILE column.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python per_doc_split.py -i result/tables/model_v43_TOP-3.csv
  python per_doc_split.py -i result/tables/model_v43_TOP-3.csv -o ./split_results/
        """,
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='CSV_FILE',
        help="Input result CSV file with FILE, PAGE, CLASS-N, SCORE-N columns",
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='./per_document/',
        metavar='DIR',
        help="Output directory for per-document CSV files (default: ./per_document/)",
    )
    args = parser.parse_args()
    split_csv_and_aggregate(args.input, args.output_dir)
