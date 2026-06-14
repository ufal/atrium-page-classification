import argparse

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


# ── Model-type detection ──────────────────────────────────────────────────
MODEL_TYPE_PREFIXES = {
    'EffNet-v2-':          'EffNetV2',
    'tf_efficientnetv2_':  'EffNetV2',
    'RegNetY-':            'RegNetY',
    'regnety_':            'RegNetY',
    'Vit-':                'ViT',
    'vit-':                'ViT',
    'CLIP-ViT-':           'CLIP',
    'Dit-':                'DiT',
    'dit-':                'DiT',
}

MODEL_MARKERS = {
    'EffNetV2': 'o',   # circle
    'RegNetY':  's',   # square
    'ViT':      'D',   # diamond
    'CLIP':     '^',   # triangle up
    'DiT':      'v',   # triangle down
    'Other':    'X',   # fallback
}


def get_model_type(model_name: str) -> str:
    for prefix, mtype in MODEL_TYPE_PREFIXES.items():
        if model_name.startswith(prefix):
            return mtype
    return 'Other'


def short_model_name(model_name: str) -> str:
    for prefix in MODEL_TYPE_PREFIXES:
        if model_name.startswith(prefix):
            return model_name[len(prefix):].lstrip('-_. ')
    return model_name


def plot_comparison(
    input_csv: str,
    output_png: str,
    title: str,
    show: bool,
) -> None:
    """Render and save the model parameters-vs-accuracy scatter plot.

    Args:
        input_csv:  CSV with at minimum 'model', 'param', and 'acc' columns.
        output_png: Path for the saved PNG.
        title:      Chart title.
        show:       Call plt.show() after saving when True.
    """
    df = pd.read_csv(input_csv)

    required = {'model', 'param', 'acc'}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Input CSV is missing required column(s): {sorted(missing_cols)}. "
            f"Found: {sorted(df.columns)}"
        )

    df['model_type'] = df['model'].apply(get_model_type)
    df['short_name'] = df['model'].apply(short_model_name)

    unique_types = df['model_type'].unique()
    cmap = plt.get_cmap('tab10')
    color_map = {mt: cmap(i % cmap.N) for i, mt in enumerate(unique_types)}

    fig, ax = plt.subplots(figsize=(10, 6))

    # Global trend line
    X = df['param'].values.reshape(-1, 1)
    y = df['acc'].values
    if len(X) > 1:
        reg = LinearRegression().fit(X, y)
        x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        ax.plot(x_range, reg.predict(x_range),
                linestyle='--', alpha=0.5, color='gray', label='Trendline')

    for model_type, group in df.groupby('model_type'):
        ax.scatter(
            group['param'], group['acc'],
            label=model_type,
            marker=MODEL_MARKERS.get(model_type, 'X'),
            s=100, edgecolor='k', alpha=0.8,
            color=color_map[model_type],
        )
        if len(group) > 1:
            g_sorted = group.sort_values('param')
            ax.plot(
                g_sorted['param'].values,
                g_sorted['acc'].values,
                linestyle=':', linewidth=1.5,
                marker=None, alpha=0.8,
                label='_nolegend_',
                color=color_map[model_type],
            )

    for _, row in df.iterrows():
        ax.text(row['param'], row['acc'] - 0.03, row['short_name'],
                fontsize=9, ha='center', va='top')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Parameters (Millions)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)',    fontsize=12)
    ax.legend(loc='lower right', bbox_to_anchor=(0.5, 0.02),
              frameon=True, fontsize=9, title='Model Type').get_frame().set_alpha(0.9)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Saved comparison chart → {output_png}")

    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot model parameter count vs. Top-1 accuracy for comparison.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The input CSV must have at least these columns:
  model   — model name / identifier string
  param   — parameter count in millions
  acc     — Top-1 accuracy (%)

Examples:
  python visualize.py -i model_accuracies_new.csv
  python visualize.py -i model_accuracies_new.csv -o comparison.png --show
  python visualize.py -i model_accuracies_new.csv --title "vX.3 model comparison"
        """,
    )
    parser.add_argument(
        '-i', '--input',
        default='model_accuracies_new.csv',
        metavar='CSV_FILE',
        help="Input CSV with 'model', 'param', 'acc' columns "
             "(default: model_accuracies_new.csv)",
    )
    parser.add_argument(
        '-o', '--output',
        default='model_acc_compared.png',
        metavar='PNG_FILE',
        help="Output chart filename (default: model_acc_compared.png)",
    )
    parser.add_argument(
        '--title',
        default='Model Comparison: Parameters vs. Top-1 Accuracy',
        help="Chart title",
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help="Open an interactive Matplotlib window after saving",
    )
    args = parser.parse_args()

    plot_comparison(
        input_csv=args.input,
        output_png=args.output,
        title=args.title,
        show=args.show,
    )
