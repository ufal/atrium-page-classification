import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib as mpl

# Load the data
df = pd.read_csv('model_accuracies_new.csv')

# Define symbols for different model types
model_markers = {
    'EffNetV2': 'o',  # Circle
    'RegNetY': 's',   # Square
    'ViT': 'D',       # Diamond
    'CLIP': '^',      # Triangle
    'DiT': 'v',       # Inverted Triangle
    'Other': 'X',     # Fallback marker
}

model_types = {
    'EffNet-v2-': 'EffNetV2',
    'RegNetY-': 'RegNetY',
    'Vit-': 'ViT',
    'CLIP-ViT-': 'CLIP',
    'Dit-': 'DiT',
    'tf_efficientnetv2_': 'EffNetV2',
    'regnety_': 'RegNetY',
    'vit-': 'ViT',
    'dit-': 'DiT',
}

# Helper to map model name to type
def get_model_type(model_name):
    for key in model_types.keys():
        if model_name.startswith(key):
            return model_types[key]
    return 'Other'

# Helper to extract a shorter display name (removes known prefix and any leading separators)
def short_model_name(model_name):
    for key in model_types.keys():
        if model_name.startswith(key):
            # remove the prefix and strip leading non-alphanumeric characters
            rest = model_name[len(key):]
            return rest.lstrip("-_ .")
    return model_name

df['model_type'] = df['model'].apply(get_model_type)
df['short_name'] = df['model'].apply(short_model_name)

# Prepare colors (one distinct color per model_type)
unique_types = df['model_type'].unique()
cmap = plt.get_cmap('tab10')  # stable, distinct colors
color_map = {mt: cmap(i % cmap.N) for i, mt in enumerate(unique_types)}

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Fit a linear model (global trend)
X = df['param'].values.reshape(-1, 1)
y = df['acc'].values
if len(X) > 1:  # Ensure there's enough data for fitting
    reg = LinearRegression().fit(X, y)
    x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_pred = reg.predict(x_range)
    ax.plot(x_range, y_pred, linestyle='--', alpha=0.5, color="gray", label='Trendline')

# Scatter plot by model_type
for model_type, group in df.groupby('model_type'):
    ax.scatter(
        group['param'], group['acc'],
        label=model_type,
        marker=model_markers.get(model_type, 'X'),
        s=100, edgecolor='k', alpha=0.8,
        color=color_map[model_type]
    )

    # Draw dotted colored lines connecting models of the same type (sorted by param)
    # Use label='_nolegend_' to avoid adding the line to legend
    if len(group) > 1:
        g_sorted = group.sort_values('param')
        ax.plot(
            g_sorted['param'].values,
            g_sorted['acc'].values,
            linestyle=':', linewidth=1.5,
            marker=None,
            alpha=0.8,
            label='_nolegend_',
            color=color_map[model_type]
        )

# Annotate points with shortened model names (adjust vertical offset)
for i, row in df.iterrows():
    ax.text(row['param'], row['acc'] - 0.03, row['short_name'],
            fontsize=9, ha='center', va='top')

# Labels and title
ax.set_title('Model Comparison: Parameters vs. Top-1 Accuracy', fontsize=14)
ax.set_xlabel('Parameters (Millions)', fontsize=12)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)

# Move legend inside the plot (compact)
legend = ax.legend(loc='lower right', bbox_to_anchor=(0.5, 0.02), frameon=True, fontsize=9, title='Model Type')
legend.get_frame().set_alpha(0.9)

# Grid
ax.grid(True, linestyle='--', alpha=0.7)

# Tight layout and save
plt.tight_layout()
plt.savefig('model_acc_compared.png', dpi=300, bbox_inches='tight')
# plt.show()