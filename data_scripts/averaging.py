import pandas as pd
import os
import argparse
import glob
import re


def parse_arguments():
    parser = argparse.ArgumentParser(description="Aggregate and average model prediction CSVs.")

    parser.add_argument(
        '-f', '--files',
        nargs='+',
        required=True,
        help="List of input CSV files or glob pattern (e.g., 'data/*.csv')"
    )

    parser.add_argument(
        '-n', '--top_n',
        type=int,
        default=3,
        help="Number of top predictions to keep in the final output (default: 3)"
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default="averaged_results_sorted.csv",
        help="Filename for the output CSV"
    )

    return parser.parse_args()


def load_and_melt(file_paths):
    long_dfs = []

    # Handle glob patterns if provided as a single string (e.g. in quotes)
    expanded_files = []
    for f in file_paths:
        expanded_files.extend(glob.glob(f))

    # Remove duplicates and check existence
    unique_files = sorted(list(set(expanded_files)))
    if not unique_files:
        print("Error: No files found matching the provided paths.")
        return None, 0

    print(f"Processing {len(unique_files)} files...")

    for fpath in unique_files:
        try:
            df = pd.read_csv(fpath)

            # Ensure PAGE is numeric
            df['PAGE'] = pd.to_numeric(df['PAGE'], errors='coerce')

            # Dynamically identify how many CLASS/SCORE pairs exist in this file
            # We look for columns starting with "CLASS-" and extract the number
            class_cols = [c for c in df.columns if c.startswith('CLASS-')]
            indices = [int(c.split('-')[1]) for c in class_cols if c.split('-')[1].isdigit()]

            if not indices:
                print(f"Warning: No CLASS-x columns found in {fpath}. Skipping.")
                continue

            # Melt each pair (CLASS-i, SCORE-i)
            file_parts = []
            for i in indices:
                cls_col = f'CLASS-{i}'
                scr_col = f'SCORE-{i}'

                if cls_col in df.columns and scr_col in df.columns:
                    temp = df[['FILE', 'PAGE', cls_col, scr_col]].rename(
                        columns={cls_col: 'CLASS', scr_col: 'SCORE'}
                    )
                    file_parts.append(temp)

            if file_parts:
                melted = pd.concat(file_parts, ignore_index=True)
                long_dfs.append(melted)

        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    if not long_dfs:
        return None, 0

    return pd.concat(long_dfs, ignore_index=True), len(unique_files)


def main():
    args = parse_arguments()

    # 1. Load and Normalize
    combined_df, num_models = load_and_melt(args.files)

    if combined_df is None:
        return

    # 2. Aggregate Scores
    print("Aggregating scores...")
    # Group by File, Page, and Class to sum scores
    grouped = combined_df.groupby(['FILE', 'PAGE', 'CLASS'])['SCORE'].sum().reset_index()

    # Calculate Average
    grouped['AVG_SCORE'] = grouped['SCORE'] / num_models

    # 3. Rank and Select Top N
    print(f"Ranking and selecting Top {args.top_n}...")
    # Sort by File, Page (Numerical), and Avg Score (Descending)
    grouped.sort_values(by=['FILE', 'PAGE', 'AVG_SCORE'], ascending=[True, True, False], inplace=True)

    # Create a rank column
    grouped['rank'] = grouped.groupby(['FILE', 'PAGE']).cumcount() + 1

    # Filter for the requested TOP-N
    top_n_df = grouped[grouped['rank'] <= args.top_n].copy()

    # 4. Pivot back to Wide Format
    top_n_pivot = top_n_df.pivot_table(
        index=['FILE', 'PAGE'],
        columns='rank',
        values=['CLASS', 'AVG_SCORE'],
        aggfunc='first'
    )

    # Flatten MultiIndex columns
    flat_df = pd.DataFrame(index=top_n_pivot.index)

    # Dynamically create columns based on the actual max rank found (up to args.top_n)
    # This handles cases where some pages might have fewer than N predictions
    max_rank_found = top_n_df['rank'].max()

    for r in range(1, int(max_rank_found) + 1):
        # Access pivot columns safely
        if ('CLASS', r) in top_n_pivot.columns:
            flat_df[f'CLASS-{r}'] = top_n_pivot[('CLASS', r)]
        if ('AVG_SCORE', r) in top_n_pivot.columns:
            flat_df[f'SCORE-{r}'] = top_n_pivot[('AVG_SCORE', r)]

    # Reset index and sort
    final_output = flat_df.reset_index()
    final_output.sort_values(by=['FILE', 'PAGE'], ascending=[True, True], inplace=True)

    # 5. Save
    final_output.to_csv(args.output, index=False)

    print(f"Successfully processed {len(final_output)} pages.")
    print(f"Results saved to: {args.output}")

    # Optional: Preview
    print("\nPreview:")
    print(final_output.head(5).to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()