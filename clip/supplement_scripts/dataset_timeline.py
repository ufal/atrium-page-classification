import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------- CONFIGURATION ------------
CSV_PATH = 'data_annotation_5.csv'  # Update this path
DATE_FILE_REGEX = r'(19|20)\d{2}'
OUTPUT_PATH = 'stacked_timeline_graph_5.png'
# --------------------------------------

CATEGORIES = [
    'DRAW', 'DRAW_L', 'LINE_HW', 'LINE_P', 'LINE_T',
    'PHOTO', 'PHOTO_L', 'TEXT', 'TEXT_HW', 'TEXT_P', 'TEXT_T'
]

label_color_map = {
    'DRAW': 'indigo',
    'DRAW_L': 'orchid',
    'LINE_HW': 'cyan',
    'LINE_P': 'deepskyblue',
    'LINE_T': 'royalblue',
    'PHOTO': 'tomato',
    'PHOTO_L': 'firebrick',
    'TEXT': 'gold',
    'TEXT_HW': 'limegreen',
    'TEXT_P': 'olive',
    'TEXT_T': 'darkgreen'
}

def parse_csv_by_category(csv_path):
    """
    Read CSV with columns: file, page, category
    Extract year from filename and count pages by year and category.
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['file', 'page', 'category']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    counts = defaultdict(lambda: defaultdict(int))
    skipped = []
    
    for _, row in df.iterrows():
        filename = str(row['file'])
        category = row['category']
        
        # Extract year from filename
        year_match = re.search(DATE_FILE_REGEX, filename)
        if year_match:
            year = int(year_match.group())
            counts[year][category] += 1
        else:
            skipped.append(filename)
    
    if skipped:
        print(f"Skipped {len(skipped)} rows without valid year in filename")
        print(f"First few examples: {skipped[:5]}")
    
    return counts

def plot_stacked_timeline(timeline_counts, output_path='stacked_timeline_graph.png'):
    """
    Create stacked bar chart from timeline counts.
    """
    # Build DataFrame
    rows = []
    for year, cat_counts in timeline_counts.items():
        row = {'Year': year}
        row.update(cat_counts)
        rows.append(row)
    
    if not rows:
        print("No data to plot!")
        return
    
    df = pd.DataFrame(rows).fillna(0).set_index('Year').sort_index()
    
    # Ensure all categories exist
    for cat in CATEGORIES:
        if cat not in df.columns:
            df[cat] = 0
    df = df[CATEGORIES]
    
    # Plot
    fig, ax = plt.subplots(figsize=(18, 8))
    bottom = pd.Series(0, index=df.index)
    
    for cat in CATEGORIES:
        ax.bar(
            df.index, df[cat],
            bottom=bottom,
            label=cat,
            color=label_color_map.get(cat, 'gray'),
            width=0.8
        )
        bottom += df[cat]
    
    ax.set_title('Document Page Counts Over Time by Category', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Pages', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Category', bbox_to_anchor=(0.05, 1), loc='upper left')
    
    # Print summary statistics
    print(f"\nTotal pages by category:")
    for cat in CATEGORIES:
        total = df[cat].sum()
        if total > 0:
            print(f"  {cat}: {int(total)}")
    print(f"\nYear range: {df.index.min()} - {df.index.max()}")
    print(f"Total pages: {int(df.sum().sum())}")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"\nSaved stacked bar chart as {output_path}")

if __name__ == '__main__':
    counts = parse_csv_by_category(CSV_PATH)
    plot_stacked_timeline(counts, OUTPUT_PATH)
