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
    file_page_sets = []
    top1_dfs = []

    # Handle glob patterns if provided as a single string
    expanded_files = []
    for f in file_paths:
        expanded_files.extend(glob.glob(f))

    # Remove duplicates and check existence
    unique_files = sorted(list(set(expanded_files)))
    if not unique_files:
        print("Error: No files found matching the provided paths.")
        return None, None, None, 0

    print(f"Processing {len(unique_files)} files...")

    loaded_models_count = 0

    for fpath in unique_files:
        try:
            df = pd.read_csv(fpath)

            # Normalize filenames
            if 'FILE' in df.columns:
                df['FILE'] = df['FILE'].astype(str).str.replace('-', '_', regex=False)

            # Ensure PAGE is numeric and drop invalid rows
            df['PAGE'] = pd.to_numeric(df['PAGE'], errors='coerce')
            df = df.dropna(subset=['PAGE'])

            # Extract Top-1 class and dynamically generate column header
            filename = os.path.basename(fpath)
            match = re.search(r'_v(\d+)3_', filename)
            col_name = f"V{match.group(1)}.3" if match else f"Model_{loaded_models_count + 1}"

            if 'CLASS-1' in df.columns:
                t_df = df.drop_duplicates(subset=['FILE', 'PAGE'])[['FILE', 'PAGE', 'CLASS-1']].copy()
                t_df.rename(columns={'CLASS-1': col_name}, inplace=True)
                top1_dfs.append(t_df)

            # Dynamically identify how many CLASS/SCORE pairs exist
            class_cols = [c for c in df.columns if c.startswith('CLASS-')]
            indices = [int(c.split('-')[1]) for c in class_cols if c.split('-')[1].isdigit()]

            if not indices:
                print(f"Warning: No CLASS-x columns found in {fpath}. Skipping.")
                continue

            # Melt each pair
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
                melted = pd.concat(file_parts, ignore_index=True).dropna(subset=['CLASS'])

                # Deduplicate WITHIN the model
                melted = melted.groupby(['FILE', 'PAGE', 'CLASS'], as_index=False)['SCORE'].max()

                # Track unique (FILE, PAGE) pairs for intersection
                unique_pairs = set(zip(melted['FILE'], melted['PAGE']))
                file_page_sets.append(unique_pairs)

                long_dfs.append(melted)
                loaded_models_count += 1

        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    if not long_dfs:
        return None, None, None, 0

    return long_dfs, file_page_sets, top1_dfs, loaded_models_count


def main():
    args = parse_arguments()

    # 1. Load, Normalize, and get per-model DataFrames
    long_dfs, file_page_sets, top1_dfs, num_models = load_and_melt(args.files)

    if not long_dfs or num_models == 0:
        print("No valid data to process. Exiting.")
        return

    # 2. Find strict intersection of (FILE, PAGE)
    print("Finding intersection of pages across all models...")
    common_file_pages = set.intersection(*file_page_sets)
    print(f"Found {len(common_file_pages)} unique pages present in all {num_models} models.")

    if not common_file_pages:
        print("Error: No common pages found across all models. Exiting.")
        return

    # Combine all models
    combined_df = pd.concat(long_dfs, ignore_index=True)

    # Filter combined_df to only include the intersection
    combined_df['FILE_PAGE'] = list(zip(combined_df['FILE'], combined_df['PAGE']))
    combined_df = combined_df[combined_df['FILE_PAGE'].isin(common_file_pages)].drop(columns=['FILE_PAGE'])

    # 3. Aggregate Scores
    print("Aggregating scores...")
    grouped = combined_df.groupby(['FILE', 'PAGE', 'CLASS'])['SCORE'].sum().reset_index()
    grouped['AVG_SCORE'] = grouped['SCORE'] / num_models
    grouped['AVG_SCORE'] = grouped['AVG_SCORE'].clip(upper=1.0)

    # 4. Rank and Select Top N
    print(f"Ranking and selecting Top {args.top_n}...")
    grouped.sort_values(by=['FILE', 'PAGE', 'AVG_SCORE'], ascending=[True, True, False], inplace=True)
    grouped['rank'] = grouped.groupby(['FILE', 'PAGE']).cumcount() + 1
    top_n_df = grouped[grouped['rank'] <= args.top_n].copy()

    # 5. Pivot back to Wide Format
    top_n_pivot = top_n_df.pivot_table(
        index=['FILE', 'PAGE'],
        columns='rank',
        values=['CLASS', 'AVG_SCORE'],
        aggfunc='first'
    )

    flat_df = pd.DataFrame(index=top_n_pivot.index)
    max_rank_found = top_n_df['rank'].max()
    if pd.isna(max_rank_found):
        max_rank_found = 0

    for r in range(1, int(max_rank_found) + 1):
        if ('CLASS', r) in top_n_pivot.columns:
            flat_df[f'CLASS-{r}'] = top_n_pivot[('CLASS', r)]
        else:
            flat_df[f'CLASS-{r}'] = pd.NA

        if ('AVG_SCORE', r) in top_n_pivot.columns:
            flat_df[f'SCORE-{r}'] = top_n_pivot[('AVG_SCORE', r)]
        else:
            flat_df[f'SCORE-{r}'] = pd.NA

    final_output = flat_df.reset_index()

    # 6. Merge the individual model Top-1 predictions into the final output
    for t_df in top1_dfs:
        final_output = final_output.merge(t_df, on=['FILE', 'PAGE'], how='left')

    # Reorder columns to group [FILE, PAGE, Model-Headers..., CLASS-1, SCORE-1...]
    model_cols = [col for t_df in top1_dfs for col in t_df.columns if col not in ['FILE', 'PAGE']]
    avg_cols = [col for col in final_output.columns if col not in ['FILE', 'PAGE'] + model_cols]
    final_output = final_output[['FILE', 'PAGE'] + model_cols + avg_cols]

    # 7. Final Sort, Clean, and Save
    final_output.sort_values(by=['FILE', 'PAGE'], ascending=[True, True], inplace=True)

    # Replace 0 and 0.0 with empty strings before saving
    final_output = final_output.replace({0: "", 0.0: ""})

    final_output.to_csv(args.output, index=False)

    print(f"Successfully processed {len(final_output)} pages.")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()