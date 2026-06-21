import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt


# ── Category definitions ──────────────────────────────────────────────────
CATEGORIES = [
    'DRAW', 'DRAW_L', 'LINE_HW', 'LINE_P', 'LINE_T',
    'PHOTO', 'PHOTO_L', 'TEXT', 'TEXT_HW', 'TEXT_P', 'TEXT_T',
]

LABEL_COLOR_MAP = {
    'DRAW':    'indigo',
    'DRAW_L':  'orchid',
    'LINE_HW': 'cyan',
    'LINE_P':  'deepskyblue',
    'LINE_T':  'royalblue',
    'PHOTO':   'tomato',
    'PHOTO_L': 'firebrick',
    'TEXT':    'gold',
    'TEXT_HW': 'limegreen',
    'TEXT_P':  'olive',
    'TEXT_T':  'darkgreen',
}

DEFAULT_DATE_REGEX = r'((?:19|20)\d{2})'


def parse_csv_by_category(csv_path: str, date_regex: str) -> pd.DataFrame:
    """Read the annotation CSV and return a (year × category) count pivot table.

    Uses vectorised Pandas operations rather than iterrows() for performance on
    large datasets.

    Args:
        csv_path:   Path to the annotation CSV (columns: file/FILE, page/PAGE,
                    category/CLASS — case-insensitive column matching).
        date_regex: Regex with one capturing group that extracts the year from
                    the filename.  Default matches 1920–2029.

    Returns:
        DataFrame indexed by year with one column per category and integer counts.
    """
    df = pd.read_csv(csv_path)

    # Normalise column names to lower-case for flexible input CSVs
    df.columns = df.columns.str.lower()

    file_col = next((c for c in df.columns if c in ('file', 'filename')), None)
    cat_col  = next((c for c in df.columns if c in ('category', 'class')), None)

    if file_col is None:
        raise ValueError(f"Cannot find a 'file' column in {csv_path}. "
                         f"Found columns: {list(df.columns)}")
    if cat_col is None:
        raise ValueError(f"Cannot find a 'category' or 'class' column in {csv_path}. "
                         f"Found columns: {list(df.columns)}")

    # Vectorised year extraction — replaces the original iterrows() loop
    df['year'] = df[file_col].astype(str).str.extract(date_regex, expand=False)
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    skipped = len(df) - len(df.dropna(subset=['year']))
    if skipped > 0:
        print(f"Skipped {skipped} row(s) without a matching year in the filename.")

    # Count pages per (year, category) and pivot to wide format
    counts = (
        df.groupby(['year', cat_col])
          .size()
          .reset_index(name='count')
          .pivot_table(index='year', columns=cat_col, values='count', fill_value=0)
    )
    counts.columns.name = None

    # Ensure all expected categories are present (fill missing with 0)
    for cat in CATEGORIES:
        if cat not in counts.columns:
            counts[cat] = 0

    return counts.sort_index()


def plot_stacked_timeline(
    counts: pd.DataFrame,
    output_path: str = 'dataset_timeline.png',
    show: bool = False,
) -> None:
    """Render a stacked bar chart of page counts over time and save it.

    Args:
        counts:      DataFrame returned by parse_csv_by_category().
        output_path: Path for the saved PNG.
        show:        If True, also call plt.show() for interactive display.
    """
    present_cats = [c for c in CATEGORIES if c in counts.columns and counts[c].sum() > 0]

    fig, ax = plt.subplots(figsize=(18, 8))
    bottom = pd.Series(0, index=counts.index)

    for cat in present_cats:
        ax.bar(
            counts.index,
            counts[cat],
            bottom=bottom,
            label=cat,
            color=LABEL_COLOR_MAP.get(cat, 'gray'),
            width=0.8,
        )
        bottom += counts[cat]

    ax.set_title('Document Page Counts Over Time by Category', fontsize=16)
    ax.set_xlabel('Year',           fontsize=12)
    ax.set_ylabel('Number of Pages', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Category', bbox_to_anchor=(0.05, 1), loc='upper left')

    print(f"\nTotal pages by category:")
    for cat in present_cats:
        total = int(counts[cat].sum())
        if total > 0:
            print(f"  {cat}: {total}")
    print(f"\nYear range : {counts.index.min()} – {counts.index.max()}")
    print(f"Total pages: {int(counts.values.sum())}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"\nSaved timeline chart → {output_path}")

    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot a stacked bar chart of annotated page counts per year and category.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_timeline.py -i data_annotation.csv
  python dataset_timeline.py -i data_annotation.csv -o timeline.png --show
  python dataset_timeline.py -i data_annotation.csv --regex "(19[89]\\d|20[012]\\d)"
        """,
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='CSV_FILE',
        help="Annotation CSV file with FILE (or FILENAME) and CATEGORY (or CLASS) columns",
    )
    parser.add_argument(
        '-o', '--output',
        default='dataset_timeline.png',
        metavar='PNG_FILE',
        help="Output plot filename (default: dataset_timeline.png)",
    )
    parser.add_argument(
        '--regex',
        default=DEFAULT_DATE_REGEX,
        metavar='PATTERN',
        help=f"Regex with one capturing group to extract the year from filenames "
             f"(default: {DEFAULT_DATE_REGEX!r})",
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help="Open an interactive Matplotlib window after saving",
    )
    args = parser.parse_args()

    counts_df = parse_csv_by_category(args.input, args.regex)
    plot_stacked_timeline(counts_df, output_path=args.output, show=args.show)
