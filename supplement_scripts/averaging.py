import pandas as pd
import os
import argparse
import glob
import re


# Compiled once: matches the wide multi-model column format produced by the
# --best prediction run, e.g. CLASS-1-v1.3, CLASS-1-v4.3, CLASS-2-v2.3.
# Group 1 = rank (1, 2, 3 …), Group 2 = model version string (v1.3, v4.3 …).
_WIDE_MODEL_RE = re.compile(r'^CLASS-(\d+)-(.+)$')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Aggregate and average prediction scores from multiple model CSV outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Average all TOP-3 CSVs in a folder, keep 3 predictions per page:
  python averaging.py --files "result/tables/*_TOP-3.csv" --top_n 3

  # Combine two specific models, keep only the single best prediction:
  python averaging.py --files result/tables/model_v53.csv result/tables/model_v43.csv -n 1

  # Use the BEST_5_models_TOP-1 wide format directly (majority vote):
  python averaging.py --files result/tables/20260529-1215_BEST_5_models_TOP-1.csv -n 1

  # Mix a wide multi-model file with additional standard model CSVs:
  python averaging.py --files "result/tables/BEST_5*TOP-1.csv" "result/tables/model_v53_TOP-3.csv"

  # Preserve zero scores in the output instead of clearing them:
  python averaging.py --files "result/tables/*_TOP-3.csv" --keep-zeros

  # Disable automatic filename normalisation (hyphen → underscore):
  python averaging.py --files "result/tables/*_TOP-3.csv" --no-normalize
        """,
    )

    parser.add_argument(
        '-f', '--files',
        nargs='+',
        required=True,
        help="Input CSV files or a glob pattern (e.g., 'result/tables/*.csv')",
    )
    parser.add_argument(
        '-n', '--top_n',
        type=int,
        default=3,
        help="Number of top predictions to keep in the output (default: 3)",
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default="averaged_results_sorted.csv",
        help="Output CSV filename (default: averaged_results_sorted.csv)",
    )

    # P1 FIX: original had action="store_true", default=True which made the
    # flag permanently True regardless of whether it was passed.  The correct
    # pattern is default=True (zeros ARE cleared by default) with a companion
    # --keep-zeros flag that sets zeros=False to opt out.
    parser.add_argument(
        '--keep-zeros',
        dest='zeros',
        action='store_false',
        help="Keep zero scores in the output; by default zero scores and their "
             "paired CLASS labels are replaced with empty strings",
    )

    # P4: make the filename normalisation opt-out so users with filenames that
    # legitimately differ by hyphens vs underscores aren't silently mismatched.
    parser.add_argument(
        '--no-normalize',
        dest='normalize_filenames',
        action='store_false',
        help="Disable automatic replacement of '-' with '_' in FILE column values "
             "(normalisation is ON by default to improve cross-model key matching)",
    )

    parser.set_defaults(zeros=True, normalize_filenames=True)
    return parser.parse_args()


def load_and_melt(file_paths, normalize_filenames=True):
    long_dfs = []
    file_page_sets = []
    top1_dfs = []

    # Expand any glob patterns passed as a single string element
    expanded_files = []
    for f in file_paths:
        expanded_files.extend(glob.glob(f))

    unique_files = sorted(list(set(expanded_files)))
    if not unique_files:
        print("Error: No files found matching the provided paths.")
        return None, None, None, 0

    print(f"Processing {len(unique_files)} file(s)...")

    loaded_models_count = 0

    for fpath in unique_files:
        try:
            df = pd.read_csv(fpath)

            # Optional filename normalisation: improves join reliability when
            # the same document appears with hyphens in one CSV and underscores
            # in another.  Controlled by --no-normalize to opt out.
            if normalize_filenames and 'FILE' in df.columns:
                df['FILE'] = df['FILE'].astype(str).str.replace('-', '_', regex=False)

            df['PAGE'] = pd.to_numeric(df['PAGE'], errors='coerce')
            df = df.dropna(subset=['PAGE'])

            # Guard for top_N=1 output from utils.py which writes a CATEGORY
            # column instead of CLASS-1; normalise to CLASS-1 for consistency.
            if 'CLASS-1' not in df.columns and 'CATEGORY' in df.columns:
                df = df.rename(columns={'CATEGORY': 'CLASS-1'})

            # Extract Top-1 column header from the filename (e.g. V4.3, V2.3)
            filename = os.path.basename(fpath)
            match = re.search(r'_v(\d+)3_', filename)
            col_name = f"V{match.group(1)}.3" if match else f"Model_{loaded_models_count + 1}"

            if 'CLASS-1' in df.columns:
                t_df = df.drop_duplicates(subset=['FILE', 'PAGE'])[['FILE', 'PAGE', 'CLASS-1']].copy()
                t_df.rename(columns={'CLASS-1': col_name}, inplace=True)
                top1_dfs.append(t_df)

            # ── Detect wide multi-model format (CLASS-1-v1.3, CLASS-1-v2.3, …) ─
            # This format is produced by the --best prediction run.  All model
            # predictions sit in a single CSV as CLASS-{rank}-{model} columns
            # with no SCORE columns.  Each prediction is assigned a uniform
            # weight of 1 so the downstream average becomes a majority vote:
            #   AVG_SCORE = (# models that predicted this class) / (# models)
            #
            # Each model column is unpacked as a separate virtual input so that
            # loaded_models_count and the intersection logic work correctly —
            # the same result as loading N individual Top-1 CSVs.
            mm_cols = []
            for c in df.columns:
                m = _WIDE_MODEL_RE.match(c)
                if m:
                    mm_cols.append((c, m.group(1), m.group(2)))

            if mm_cols:
                rank1_cols = [(col, model_id) for col, rank, model_id in mm_cols if rank == '1']
                skipped_ranks = [col for col, rank, _ in mm_cols if rank != '1']

                if not rank1_cols:
                    print(f"Warning: wide multi-model format detected in {os.path.basename(fpath)} "
                          f"but no rank-1 columns found. Skipping.")
                    continue
                if skipped_ranks:
                    print(f"Note ({os.path.basename(fpath)}): "
                          f"{len(skipped_ranks)} rank-2+ column(s) ignored "
                          f"(no confidence scores available for weighting).")

                print(f"  → Wide multi-model format: "
                      f"{len(rank1_cols)} model prediction(s) in one file.")

                # Collect existing display names to avoid column collisions if
                # this wide file is mixed with standard per-model CSVs.
                existing_display = {
                    col for t in top1_dfs for col in t.columns if col not in ('FILE', 'PAGE')
                }

                for col, model_id in rank1_cols:
                    # Normalise: v1.3 → V1.3
                    display_name = model_id[0].upper() + model_id[1:] if model_id else f"Model_{loaded_models_count + 1}"
                    if display_name in existing_display:
                        display_name = f"{display_name}_{loaded_models_count + 1}"
                    existing_display.add(display_name)

                    # Per-model top-1 column for the output preview section
                    t_df = (
                        df.drop_duplicates(subset=['FILE', 'PAGE'])[['FILE', 'PAGE', col]]
                        .copy()
                        .rename(columns={col: display_name})
                    )
                    top1_dfs.append(t_df)

                    # Build long-form with score=1; downstream sum÷num_models gives
                    # the fraction of models that voted for each class.
                    melted = (
                        df[['FILE', 'PAGE', col]]
                        .copy()
                        .rename(columns={col: 'CLASS'})
                        .assign(SCORE=1.0)
                        .dropna(subset=['CLASS'])
                    )

                    if not melted.empty:
                        melted = melted.groupby(
                            ['FILE', 'PAGE', 'CLASS'], as_index=False
                        )['SCORE'].max()
                        file_page_sets.append(set(zip(melted['FILE'], melted['PAGE'])))
                        long_dfs.append(melted)
                        loaded_models_count += 1

                continue  # skip the standard pd.wide_to_long path below

            # ── Standard format: CLASS-N / SCORE-N column pairs ─────────────
            class_cols = [c for c in df.columns if c.startswith('CLASS-')]
            indices = [int(c.split('-')[1]) for c in class_cols if c.split('-')[1].isdigit()]

            if not indices:
                print(f"Warning: No CLASS-x columns found in {fpath}. Skipping.")
                continue

            cols_to_keep = (
                ['FILE', 'PAGE']
                + [f'CLASS-{i}' for i in indices]
                + [f'SCORE-{i}' for i in indices]
            )
            cols_to_keep = [c for c in cols_to_keep if c in df.columns]
            df_subset = df[cols_to_keep].copy()

            # Vectorised melt — replaces the original row-level for-loop
            melted = pd.wide_to_long(
                df_subset,
                stubnames=['CLASS', 'SCORE'],
                i=['FILE', 'PAGE'],
                j='rank',
                sep='-',
                suffix=r'\d+',
            ).reset_index().dropna(subset=['CLASS'])

            # P1 FIX: the original code had a dead `if file_parts:` block below
            # this point that referenced an undefined variable `file_parts`
            # (leftover from a prior refactor), causing a NameError at runtime.
            # That block has been removed; the vectorised path above is the only
            # code path.

            if not melted.empty:
                # Deduplicate within this model's predictions
                melted = melted.groupby(['FILE', 'PAGE', 'CLASS'], as_index=False)['SCORE'].max()

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

    long_dfs, file_page_sets, top1_dfs, num_models = load_and_melt(
        args.files,
        normalize_filenames=args.normalize_filenames,
    )

    if not long_dfs or num_models == 0:
        print("No valid data to process. Exiting.")
        return

    # ── Intersect pages present in ALL models ────────────────────────────────
    print("Finding intersection of pages across all models...")
    common_file_pages = set.intersection(*file_page_sets)
    print(f"Found {len(common_file_pages)} unique page(s) present in all {num_models} model(s).")

    if not common_file_pages:
        print("Error: No common pages found across all models. Exiting.")
        return

    # Filter each per-model DataFrame to the intersection BEFORE concat to
    # avoid building a large intermediate DataFrame that is immediately discarded.
    filtered = []
    for df in long_dfs:
        mask = pd.Series(list(zip(df['FILE'], df['PAGE']))).isin(common_file_pages)
        filtered.append(df[mask.values])
    combined_df = pd.concat(filtered, ignore_index=True)

    # ── Aggregate ────────────────────────────────────────────────────────────
    print("Aggregating scores...")
    grouped = combined_df.groupby(['FILE', 'PAGE', 'CLASS'])['SCORE'].sum().reset_index()
    grouped['AVG_SCORE'] = (grouped['SCORE'] / num_models).clip(upper=1.0)

    # ── Rank and select Top-N ────────────────────────────────────────────────
    print(f"Ranking and selecting Top-{args.top_n}...")
    grouped.sort_values(
        by=['FILE', 'PAGE', 'AVG_SCORE'],
        ascending=[True, True, False],
        inplace=True,
    )
    grouped['rank'] = grouped.groupby(['FILE', 'PAGE']).cumcount() + 1
    top_n_df = grouped[grouped['rank'] <= args.top_n].copy()

    # ── Pivot to wide format ─────────────────────────────────────────────────
    top_n_pivot = top_n_df.pivot_table(
        index=['FILE', 'PAGE'],
        columns='rank',
        values=['CLASS', 'AVG_SCORE'],
        aggfunc='first',
    )

    flat_df = pd.DataFrame(index=top_n_pivot.index)
    max_rank_found = int(top_n_df['rank'].max()) if not top_n_df.empty else 0

    for r in range(1, max_rank_found + 1):
        flat_df[f'CLASS-{r}'] = top_n_pivot.get(('CLASS', r), pd.NA)
        flat_df[f'SCORE-{r}'] = top_n_pivot.get(('AVG_SCORE', r), pd.NA)

    final_output = flat_df.reset_index()

    # ── Merge individual model Top-1 columns ────────────────────────────────
    for t_df in top1_dfs:
        final_output = final_output.merge(t_df, on=['FILE', 'PAGE'], how='left')

    model_cols = [
        col
        for t_df in top1_dfs
        for col in t_df.columns
        if col not in ('FILE', 'PAGE')
    ]
    avg_cols = [c for c in final_output.columns if c not in ['FILE', 'PAGE'] + model_cols]
    final_output = final_output[['FILE', 'PAGE'] + model_cols + avg_cols]

    # ── Final sort and zero cleanup ──────────────────────────────────────────
    final_output.sort_values(by=['FILE', 'PAGE'], ascending=[True, True], inplace=True)
    final_output = final_output.replace({0: "", 0.0: ""})

    if args.zeros:
        # When a SCORE-N slot is empty (was zero) also clear its CLASS-N label
        # so the reader never sees a class name paired with a blank score.
        for i in range(2, args.top_n + 1):
            score_col = f'SCORE-{i}'
            class_col = f'CLASS-{i}'
            if score_col in final_output.columns and class_col in final_output.columns:
                final_output.loc[final_output[score_col] == "", class_col] = ""

    final_output.to_csv(args.output, index=False)
    print(f"Successfully processed {len(final_output)} page(s).")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()