"""
Behavioral Correlograms

Extended Data Fig 8a

IDE: Spyder
Date: 08/2025
"""
#%% Libraries
from IPython import get_ipython
get_ipython().magic('reset -sf')
get_ipython().magic('clear')

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns

#%% Functions
def plot_correlogram_axes(ax, corr_matrix, labels, title=''):
    n = len(labels)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)

    # Draw grid lines
    for i in range(n + 1):
        ax.axhline(i, color='lightgray', linewidth=0.5)
        ax.axvline(i, color='lightgray', linewidth=0.5)

    # Use Seaborn's 'vlag' colormap and invert it
    cmap = sns.color_palette("vlag", as_cmap=True)
    cmap_reversed = cmap.reversed()

    for i in range(n):
        for j in range(i + 1):  # Lower triangle only
            corr = corr_matrix[i, j]
            color = cmap_reversed((corr + 1) / 2)  # normalize to [0, 1]
            size = np.abs(corr) * 1500
            circle = plt.Circle((j + 0.5, i + 0.5), radius=np.sqrt(size)/100,
                                color=color, lw=0)  # Removed ec, lw=0 disables edge
            ax.add_patch(circle)
            # Removed the text annotation

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8, pad=10)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
#%% Parameters
brain_region = 'ACC' # ACC or CLA
brain_regions = {
    'CLA': ['sub016A', 'sub016B', 'sub017', 'sub024A'],
    'ACC': ['sub016A', 'sub016B', 'sub019A', 'sub019B', 'sub020', 'sub023', 'sub024A']
}
subjects = brain_regions.get(brain_region, [])
columns_to_keep = ["A_safety_variance", "B_safety_variance",
                   "A_absolute_prediction_error", "B_absolute_prediction_error"]
corr_matrices = []

data_path = rf'define data path'
save_path = rf'define save path'

#%% Load and compute correlation
for sub in subjects:
    file_path = os.path.join(
        f'{data_path}/Appear/{brain_region}/',
        f'{sub}/{brain_region}_{sub}.xlsx'
    )
    behavior = pd.read_excel(file_path)
    behavior = behavior[columns_to_keep]
    corr_matrix = behavior.corr(method='pearson').to_numpy()
    corr_matrices.append(corr_matrix)

#%% Plot correlograms grid
n_matrices = len(corr_matrices)
n_cols = 4
n_rows = int(np.ceil(n_matrices / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 4.5))
axes = axes.flatten()

for i in range(n_cols * n_rows):
    if i < n_matrices:
        plot_correlogram_axes(axes[i], corr_matrices[i], columns_to_keep, title=subjects[i])
    else:
        axes[i].axis('off')

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
vlag_cmap = sns.color_palette("vlag", as_cmap=True).reversed()
norm = Normalize(vmin=-1, vmax=1)
sm = cm.ScalarMappable(cmap=vlag_cmap, norm=norm)
fig.colorbar(sm, cax=cbar_ax, label='Pearson r')

fig.suptitle(f'Correlograms – {brain_region}', fontsize=14)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 5
fig.savefig(rf"{save_path}\correlograms_{brain_region}.svg", format='svg', bbox_inches='tight')
plt.close()