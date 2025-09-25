"""
This script analyzes neuronal firing rates in relation to behavioral measures 
for CLA and ACC brain regions. It computes mutual information between neural 
activity and behavior, performs permutation-based statistical testing, generates 
raster plots and smoothed firing rate curves, visualizes results with heatmaps 
and cluster analyses, and saves all outputs (plots and statistics) for further
inspection.

This code generates the following figures:
    Main Figures:
        4l-n
        5i-k
    Extended Data Figures:
        5e-f, h-k
        6g-h, j-m
        
IDE: Spyder
Date: 06/25
"""
#%% Libraries
from IPython import get_ipython
get_ipython().magic('reset -sf')  # Reset environment, '-sf' forces a reset without confirmation
get_ipython().magic('clear')

import os
import numpy as np
import pandas as pd

import scipy.io
from scipy.stats import spearmanr
from scipy.ndimage import uniform_filter1d
from scipy.stats import zscore
from scipy import stats
from scipy.stats import rankdata
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics import mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import minmax_scale

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
matplotlib.use("Agg")  # non-interactive backend

from joblib import Parallel, delayed
    
#%% Functions
def sklearn_mi_metric(disc_x, disc_y):
    """
    Compute mutual information between two discrete variables using sklearn
    """
    return mutual_info_score(disc_x, disc_y)

def individual_permutation_mi(x, y, n_perm, compute_p=True):
    """
    Compute mutual information between features (x) and target (y) with
    permutation-based p-values
    """
    # Convert to numpy
    x = x.values
    y = np.array(y)
    n_samples, n_features = x.shape
    # Pre-discretize behavior
    y_disc = y
    # Pre-discretize all features once
    x_disc_all = np.round(x * 1000).astype(int)
    # Observed MI for all features
    mi = [mutual_info_score(x_disc_all[:, j], y_disc) for j in range(n_features)]
    # Pre-generate permutations
    rng = np.random.default_rng()
    perm_indices_list = [rng.permutation(n_samples) for _ in range(n_perm)]
    # Compute null distribution for all features in parallel
    def permuted_metrics(perm_indices):
        x_perm = x_disc_all[perm_indices, :]
        return [mutual_info_score(x_perm[:, j], y_disc) for j in range(n_features)]
    null_distributions = Parallel(n_jobs=-1)(
        delayed(permuted_metrics)(perm_indices) for perm_indices in perm_indices_list)
    null_distributions = np.array(null_distributions)  # shape (n_perm, n_features)
    # Compute p-values
    if compute_p:
        p_mi = []
        for j in range(n_features):
            p = (np.sum(null_distributions[:, j] >= mi[j]) + 1) / (n_perm + 1)
            p_mi.append(p)
    else:
        p_mi = [np.nan] * n_features
    return np.array(mi), np.array(p_mi)

def build_entry(method, brain_region, file_name, behavior_name, event,
                firing_rate, behavior_array, rho, p_value, fr_all, outcome):
    """
    Create a dictionary entry for storing neuron-behavior analysis results
    """
    return {'method': method, 'brain_region': brain_region, 'file_name': file_name,
            'behavior_name': behavior_name, 'event': event, 'firing_rate': firing_rate,
            'behavior': behavior_array, 'rho': rho, 'p_value': p_value,
            'fr_all': fr_all, 'outcome': outcome}

def save_results_to_excel(brain_region, result_mi_it, result_mi_a, save_folder):
    """Save lists of MI results (intertrial and asteroid) to Excel files"""
    # Helper to convert and save each list to Excel
    def save_list_to_excel(data_list, filename):
        if len(data_list) == 0:
            print(f"No data to save for {filename}")
            return
        df = pd.DataFrame(data_list)
        full_path = f"{save_folder}{filename}"
        df.to_excel(full_path, index=False)
        print(f"Saved {full_path}")
    # Save intertrial results    
    save_list_to_excel(result_mi_it, f'neuron_id_intertrial_MI_score_{brain_region}.xlsx')
    # Save asteroid onset results    
    save_list_to_excel(result_mi_a, f'neuron_id_asteroid_MI_score_{brain_region}.xlsx')

def permutation_test(low_vals, high_vals, n_permutations=10000):
    """
    Perform a permutation test between two sets of values and return
    difference and p-value
    """
    combined = np.concatenate([low_vals, high_vals])
    n_low = len(low_vals)
    observed_diff = np.mean(high_vals) - np.mean(low_vals)    
    perm_diffs = np.zeros(n_permutations)
    rng = np.random.default_rng(42)
    for i in range(n_permutations):        
        perm = rng.permutation(len(combined))
        perm_low = combined[perm[:n_low]]
        perm_high = combined[perm[n_low:]]
        perm_diffs[i] = np.mean(perm_high) - np.mean(perm_low)        
    p_val = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    return observed_diff, p_val

def plot_version(fr, x, outcome, bin_size=40):
    """
    Smooth firing rates as a function of behavior percentile for plotting
    """
    fr = np.array(fr)
    x = np.array(x)
    o = np.array(outcome)
    # Remove NaNs
    valid_mask = ~np.isnan(fr) & ~np.isnan(x)
    fr = fr[valid_mask]
    x = x[valid_mask]
    o = o[valid_mask]
    # Sort by behavior
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    fr_sorted = fr[sort_idx]        
    if len(fr_sorted) % 2 == 1:
        fr_sorted = fr_sorted[:-1]
        x_sorted = x_sorted[:-1]    
    # Percentile based x-axis
    x_percentile = rankdata(x_sorted, method='average') / len(x_sorted)
    # Smooth fr using sliding window in percentile order
    mean_frs_smooth = np.full_like(fr_sorted, np.nan, dtype=np.float32)
    half_window = bin_size // 2
    for i in range(len(fr_sorted)):
        start = max(0, i - half_window)
        end = min(len(fr_sorted), i + half_window + 1)
        mean_frs_smooth[i] = np.mean(fr_sorted[start:end])
    return {'x_percentile': x_percentile, 'fr_sorted': fr_sorted,
            'mean_frs_smooth': mean_frs_smooth}

def plot_individual_neurons(results, method_name, time_point, save_dir,
                            collect_curves=None, brain_region=None, dpi=600):
    """
    Plot smoothed firing rates vs behavior for individual neurons and save
    statistics
    """
    if brain_region is None:
        raise ValueError("brain_region must be provided.")
    os.makedirs(save_dir, exist_ok=True)
    all_stat_data = []  # List to collect stats per neuron
    for entry in results:
        behavior = np.array(entry['behavior']).flatten()
        fr = np.array(entry['firing_rate']).flatten()
        fr_all = np.array(entry['fr_all'])
        file_name = entry['file_name']
        behavior_name = entry['behavior_name']
        outcome = entry['outcome']
        if behavior.shape != fr.shape:
            print(f"Skipping {file_name}: shape mismatch.")
            continue
        # Percentile binning
        curves = plot_version(fr, behavior, outcome)
        xq = curves['x_percentile']
        fr_sorted = curves['fr_sorted']
        fr_smooth = curves['mean_frs_smooth']        
        # Stats (mean ± SEM and p-value between low and high quantiles)
        low_vals = fr_smooth[xq <= 0.3]
        high_vals = fr_smooth[xq >= 0.7]        
        # Means and SEMs
        mean_low = np.mean(low_vals)
        sem_low = stats.sem(low_vals)
        mean_high = np.mean(high_vals)
        sem_high = stats.sem(high_vals)        
        # Statistical test
        _, p_val = permutation_test(low_vals, high_vals, n_permutations=10000)
        if p_val < 0.0001:
            p_val = 0.0001
        # Save stat data
        stat_data = {'brain_region': entry.get('brain_region', brain_region),
            'neuron_id': file_name, 'behavior': behavior_name, 'p_val': p_val,
            'event': entry.get('event', 'unknown'), 'mean_low': mean_low,
            'sem_low': sem_low, 'mean_high': mean_high, 'sem_high': sem_high}
        all_stat_data.append(stat_data)
        # Plot
        if behavior_name in ['A_safety_variance', 'B_safety_variance']:
            color_high = "orange"
            color_low = "blue"
        else:
            color_high = "#ff408c"
            color_low = "#00aaff"  
        fig, ax = plt.subplots(1, 1, figsize=(1.5, .75), sharex=True, dpi=dpi)
        # Smoothed
        ax.plot(xq, fr_smooth, '-', color='black', linewidth=0.5, alpha=0.7)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.axvspan(0, 0.3, color=color_low, alpha=0.1)   # low: 0–30%
        ax.axvspan(0.7, 1.0, color=color_high, alpha=0.1)  # high: 70–100%
        ax.set_xticks([0.0, 0.3, 0.5, 0.7, 1.0])
        ax.set_xticklabels(['0', '30', '50', '70', '100'], fontsize=5)
        ax.set_xlabel('Behavior Percentile (%)', fontsize=5)
        ax.set_ylabel('Smoothed Spike Counts', fontsize=5)
        ax.tick_params(labelsize=6, width=0.6, length=3)
        ax.spines[['top', 'right']].set_visible(False)
        # Annotate
        textstr = (f"Low: {mean_low:.2f}±{sem_low:.2f}\n"
                   f"High: {mean_high:.2f}±{sem_high:.2f}\n"
                   f"p = {p_val:.5f}")
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                 verticalalignment='top', fontsize=4.5,
                 bbox=dict(boxstyle='round,pad=0.2', edgecolor='none', facecolor='white', alpha=0.8))
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        fig.subplots_adjust(hspace=0.35, left=0.22, right=0.95, top=0.88, bottom=0.15)
        fig.suptitle(f'{behavior_name}', fontsize=5, y=0.95)
        if collect_curves is not None:
            collect_curves.append({'curve_smooth': fr_smooth, 'curve': fr_sorted,
                'behavior_name': behavior_name, 'behavior': behavior,
                'method': method_name, 'time_point': time_point,
                'file_name': file_name, 'fr_all': fr_all, 'outcome': outcome})        
        fig_path = os.path.join(save_dir, f"{brain_region}_{method_name}_{time_point}_{file_name}_{behavior_name}.svg")
        plt.savefig(fig_path, dpi=dpi)
        plt.close(fig)
    return all_stat_data
        
def plot_mi_heatmaps(mi_it, mi_a, columns_to_keep, neuron_labels, 
                     brain_region, method_label, save_path):
    """
    Plot heatmaps of normalized mutual information for intertrial and
    asteroid periods
    """
    # Normalize function (column-wise between 0 and 1)
    def normalize_columns(matrix):
        min_vals = np.nanmin(matrix, axis=0, keepdims=True)
        max_vals = np.nanmax(matrix, axis=0, keepdims=True)
        denom = max_vals - min_vals
        denom[denom == 0] = 1  # avoid division by zero
        return (matrix - min_vals) / denom
    # Normalize each matrix
    mi_it = normalize_columns(mi_it)
    mi_a = normalize_columns(mi_a)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 10))
    # intertrial
    im = axes[0].imshow(mi_it, aspect='equal', cmap='inferno')
    fig.colorbar(im, ax=axes[0], label='Mutual Information (norm)')
    axes[0].set_xticks(np.arange(3))
    labels = columns_to_keep[:2] + columns_to_keep[-2:-1]
    axes[0].set_xticklabels(labels, rotation=90, fontsize=5)
    axes[0].set_yticks(np.arange(len(neuron_labels)))
    axes[0].set_yticklabels(neuron_labels, fontsize=5)
    axes[0].set_title('Intertrial', fontsize=5)
    # asteroid
    im = axes[1].imshow(mi_a, aspect='equal', cmap='inferno')
    fig.colorbar(im, ax=axes[1], label='Mutual Information (norm)')
    axes[1].set_xticks(np.arange(8))    
    axes[1].set_xticklabels(columns_to_keep, rotation=90, fontsize=5)
    axes[1].set_yticks(np.arange(len(neuron_labels)))
    axes[1].set_yticklabels(neuron_labels, fontsize=5)
    axes[1].set_title('Asteroid Period', fontsize=5)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{save_path}/{brain_region}_{method_label}_mutual_information_heatmap.svg', format='svg')
    plt.close(fig)

def plot_pval_heatmap(pval, neuron_labels, columns_to_keep, brain_region, 
                              method_label, save_path, cell_type_df, title_suffix, scale):
    """
    Plot heatmap of p-values per neuron and overlay cell type information
    """
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad(color='white')
    neuron_to_type = {row['Neuron_ID_mapped']: row['Cell_Type'] for _, row in cell_type_df.iterrows()}
    cell_type_colors = {
        'Narrow Interneuron': '#8EAA77',
        'Pyramidal Cell': '#CC9900',
        'Wide Interneuron': '#8B8069',
        'Unknown': '#B0B0B0'}
    mask_sig = np.any(pval < 0.05, axis=1)
    pval_sig = pval[mask_sig, :]
    neuron_labels_sig = [neuron_labels[i] for i in np.where(mask_sig)[0]]
    neuron_types = [neuron_to_type.get(lbl, 'Unknown') for lbl in neuron_labels_sig]
    color_vector = [cell_type_colors[ct] for ct in neuron_types]
    n_behaviors = pval.shape[1]
    n_neurons = pval_sig.shape[0]
    # --- Figure width: scale with neurons but keep within limits ---
    fig_width = scale*min(max(6, n_neurons * 0.22), 20)  # min width=6, max width=20
    fig_height = scale*max(2, n_behaviors * 0.18 + 0.5)
    fig = plt.figure(figsize=(fig_width, fig_height))    
    gs = GridSpec(2, 1, height_ratios=[n_behaviors, 1], hspace=0.05, figure=fig)
    # --- Heatmap ---
    ax_heat = fig.add_subplot(gs[0])
    masked = np.ma.masked_where(pval_sig >= 0.05, pval_sig)
    c = ax_heat.pcolormesh(np.arange(n_neurons+1), np.arange(n_behaviors+1), masked.T[::-1],
                           cmap=cmap, edgecolors='gray', linewidth=0.2, vmin=0, vmax=0.05)
    ax_heat.set_aspect('equal')
    ax_heat.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_heat.set_yticks(np.arange(n_behaviors)+0.5)
    ax_heat.set_yticklabels(columns_to_keep[::-1], fontsize=5, fontname='Arial')
    ax_heat.set_xlim(0, n_neurons)
    ax_heat.set_ylim(0, n_behaviors)
    ax_heat.set_title(title_suffix, fontsize=5, fontname='Arial')
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.03, 0.7])
    cbar = plt.colorbar(c, cax=cbar_ax, orientation='vertical')
    cbar.set_label('p-Value', fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    # --- Cell type row ---
    ax_bar = fig.add_subplot(gs[1], sharex=ax_heat)
    rgba_colors = np.array([mcolors.to_rgba(c) for c in color_vector]).reshape(1, -1, 4)
    ax_bar.pcolormesh(np.arange(n_neurons+1), [0, 1], rgba_colors, edgecolors='gray', linewidth=0.2)
    ax_bar.set_aspect('equal')
    ax_bar.set_xticks(np.arange(n_neurons)+0.5)    
    ax_bar.set_xticklabels(neuron_labels_sig, fontsize=5, fontname='Arial', rotation=90)
    ax_bar.set_yticks([])
    ax_bar.set_ylabel('Cell Type', fontsize=5, fontname='Arial')
    ax_bar.set_xlim(0, n_neurons)
    ax_bar.set_ylim(0, 1)    
    for ax in [ax_heat, ax_bar]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
    plt.savefig(rf'{save_path}/{brain_region}_{method_label}_p_values_{title_suffix}.svg', format='svg', dpi=300)
    plt.close(fig)

def plot_cell_type_proportions(pval_mat, neuron_labels, cell_type_df,
                               behavior_labels, brain_region, title, save_path=None):
    """
    Compute and plot proportion of each cell type among significant neurons
    per behavior
    """
    # Compute proportion of each cell type per behavior
    cell_type_colors = {'Narrow Interneuron': '#8EAA77',
                        'Pyramidal Cell': '#CC9900',
                        'Wide Interneuron': '#8B8069',
                        'Unknown': '#B0B0B0'}
    proportions = []
    for beh_idx in range(pval_mat.shape[1]):
        sig_neurons = np.where(pval_mat[:, beh_idx] < 0.05)[0]
        if len(sig_neurons) == 0:
            props = {ct: 0 for ct in cell_type_colors.keys()}
        else:
            sig_ids = [neuron_labels[i] for i in sig_neurons]
            sig_types = cell_type_df[cell_type_df['Neuron_ID_mapped'].isin(sig_ids)]['Cell_Type']
            counts = sig_types.value_counts()
            total = counts.sum()
            props = {ct: counts.get(ct, 0)/total*100 for ct in cell_type_colors.keys()}
        proportions.append(props)    
    df_prop = pd.DataFrame(proportions, index=behavior_labels)
    df_plot = df_prop.drop(columns=['Unknown'], errors='ignore')    
    fig, ax = plt.subplots(figsize=(len(behavior_labels)*0.35, 0.5))
    bar_container = df_plot.plot(
        kind='bar', stacked=True,
        color=[cell_type_colors[ct] for ct in df_plot.columns],
        ax=ax, width=0.25, legend=False)    
    # Annotate
    for idx, patch_list in enumerate(bar_container.containers):
        for bar in patch_list:
            height = bar.get_height()
            if height > 3:  # only label segments > 3%
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + height / 2
                ax.text(x, y, f'{height:.0f}', ha='center', va='center', fontname='Arial', fontsize=5)    
    ax.set_ylabel('%', fontsize=5, fontname='Arial')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', length=1, width=0.5, rotation=90, labelsize=5)
    ax.tick_params(axis='y', length=1, width=0.5, labelsize=5)    
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    if save_path:
        fig.savefig(rf'{save_path}/{brain_region}_{title}_cell_type_proportions.svg', format='svg', dpi=300)
    plt.show()
    plt.close(fig)

def generate_raster(trials, start_time=-4, end_time=2):
    """
    Convert spike time data into a raster format for plotting
    """    
    raster = []
    for trial in trials:
        try:
            flat_trial = np.concatenate([np.ravel(np.array(t, dtype=float)) for t in trial])
            valid_times = flat_trial[(flat_trial >= start_time) & (flat_trial <= end_time)]
            raster.append(valid_times)
        except Exception:
            raster.append([])  # If any error occurs (e.g., empty or malformed trial), append empty list
    return raster
                
def plot_raster(trials, fr, behavior, time_axis, column, brain_region,
                file_name, folder_path, start_time=-4, end_time=2):
    """
    Plot raster and smoothed firing rate for all trials and split by behavior
    quantiles
    """
    os.makedirs(folder_path, exist_ok=True)
    # Generate raster
    raster = generate_raster(trials, start_time=start_time, end_time=end_time)    
    # Calculate 30-70th percentile of behavior
    low_cutoff = np.percentile(behavior, 30)
    high_cutoff = np.percentile(behavior, 70)    
    low_idx = np.where(behavior <= low_cutoff)[0]
    high_idx = np.where(behavior >= high_cutoff)[0]
    # Split raster and firing rate
    raster_low = [raster[i] for i in low_idx]
    raster_high = [raster[i] for i in high_idx]
    fr_low = fr[low_idx]
    fr_high = fr[high_idx]    
    # Demeaning
    baseline_mask = (time_axis >= 1.5) & (time_axis <= 2)
    fr = (fr - np.mean(fr[:,baseline_mask]))
    fr_low = (fr_low - np.mean(fr_low[:,baseline_mask]))
    fr_high = (fr_high - np.mean(fr_high[:,baseline_mask]))
    # Smooth firing rates
    bin_size = int(200/50)  # Assuming 50 ms bins
    fr_low_smooth = uniform_filter1d(np.mean(fr_low, axis=0), bin_size)
    fr_high_smooth = uniform_filter1d(np.mean(fr_high, axis=0), bin_size)
    fr_all_smooth = uniform_filter1d(np.mean(fr, axis=0), bin_size)
    # Set up 2-column figure
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3), constrained_layout=True)
    fig.suptitle(column, fontsize=5)
    # Left Panel – Original Raster
    raster_x, raster_y = [], []
    for i, trial_times in enumerate(raster):
        raster_x.extend(trial_times)
        raster_y.extend([i] * len(trial_times))
    ax_raster = axes[0, 0]
    ax_raster.scatter(raster_x, raster_y, s=1, c='#22205F', marker='s')
    ax_raster.axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax_raster.axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax_raster.set_xlim(start_time, end_time)
    ax_raster.set_xticks([])
    ax_raster.set_ylabel("trial #")
    ax_raster.set_title("All Trials")
    ax_raster.tick_params(axis='both', which='both', length=1, width=0.5)
    for spine in ax_raster.spines.values():
        spine.set_linewidth(0.5)
    ax_raster.spines['top'].set_visible(False)
    ax_raster.spines['right'].set_visible(False)
    ax_raster.spines['bottom'].set_visible(False)
    # Left Bottom – Mean FR
    ax_fr = axes[1, 0]
    ax_fr.plot(time_axis, fr_all_smooth, color='#22205F', linewidth=0.5)
    ax_fr.axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax_fr.axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax_fr.set_xlim(start_time, end_time)
    ax_fr.set_xticks(np.arange(start_time, end_time + 1, 2))
    ax_fr.set_xlabel("time (s)")
    ax_fr.set_ylabel("mean D rate")
    ax_fr.tick_params(axis='both', which='both', length=1, width=0.5)
    for spine in ax_fr.spines.values():
        spine.set_linewidth(0.5)
    ax_fr.spines['top'].set_visible(False)
    ax_fr.spines['right'].set_visible(False)
    # Right Panel – Split Raster
    raster_x_low, raster_y_low = [], []
    for i, trial_times in enumerate(raster_low):
        raster_x_low.extend(trial_times)
        raster_y_low.extend([i] * len(trial_times))    
    raster_x_high, raster_y_high = [], []
    for i, trial_times in enumerate(raster_high):
        raster_x_high.extend(trial_times)
        raster_y_high.extend([i + len(raster_low)] * len(trial_times))
    ax_split_raster = axes[0, 1]
    if column in ['A_safety_variance', 'B_safety_variance']:
        ax_split_raster.scatter(raster_x_low, raster_y_low, s=1, c='blue', marker='s')
        ax_split_raster.scatter(raster_x_high, raster_y_high, s=1, c='orange', marker='s')
    else:
        ax_split_raster.scatter(raster_x_low, raster_y_low, s=1, c='#00aaff', marker='s')
        ax_split_raster.scatter(raster_x_high, raster_y_high, s=1, c='#ff408c', marker='s')
    ax_split_raster.axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax_split_raster.axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax_split_raster.set_xlim(start_time, end_time)
    ax_split_raster.set_xticks([])
    ax_split_raster.set_title(column)
    ax_split_raster.tick_params(axis='both', which='both', length=1, width=0.5)
    for spine in ax_split_raster.spines.values():
        spine.set_linewidth(0.5)
    ax_split_raster.spines['top'].set_visible(False)
    ax_split_raster.spines['right'].set_visible(False)
    ax_split_raster.spines['bottom'].set_visible(False)
    # Right Bottom – Split FR
    ax_split_fr = axes[1, 1]
    if column in ['A_safety_variance', 'B_safety_variance']:
        ax_split_fr.plot(time_axis, fr_low_smooth, color='blue', linewidth=0.5)
        ax_split_fr.plot(time_axis, fr_high_smooth, color='orange', linewidth=0.5)
    else:
        ax_split_fr.plot(time_axis, fr_low_smooth, color='#00aaff', linewidth=0.5)
        ax_split_fr.plot(time_axis, fr_high_smooth, color='#ff408c', linewidth=0.5)
    ax_split_fr.axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax_split_fr.axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax_split_fr.set_xlim(start_time, end_time)
    ax_split_fr.set_xticks(np.arange(start_time, end_time + 1, 2))
    ax_split_fr.set_xlabel("time (s)")
    ax_split_fr.tick_params(axis='both', which='both', length=1, width=0.5)
    for spine in ax_split_fr.spines.values():
        spine.set_linewidth(0.5)
    ax_split_fr.spines['top'].set_visible(False)
    ax_split_fr.spines['right'].set_visible(False)
    # Save figure
    fname = f"{brain_region}_{file_name}_{column}"
    save_path_svg = os.path.join(folder_path, f"{fname}.svg")
    os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)
    fig.savefig(save_path_svg, format='svg', dpi=300, transparent=True)
    plt.close(fig)

def process_and_cluster(curves, title, brain_region, n_points=200):
    """
    Interpolate z-scored curves, perform hierarchical clustering, and plot
    cluster-averaged curves
    """
    # Custom grayscale palette
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#1a1a1a", "#4d4d4d", "#808080", "#b3b3b3", "#d9d9d9"])
    # --- Interpolate curves ---
    labels = []
    interp_curves_z = []     # for plotting (z-score)
    for c in curves:
        x_old = np.linspace(0, 1, len(c['curve_smooth']))
        x_new = np.linspace(0, 1, n_points)
        f = interp1d(x_old, c['curve_smooth'], kind='linear')
        curve_interp = f(x_new)
        # For plotting → z-score
        curve_z = zscore(curve_interp)
        interp_curves_z.append(curve_z)
        labels.append(c['file_name'])    
    interp_curves_z = np.vstack(interp_curves_z)
    n_samples = interp_curves_z.shape[0]
    dist_mat = pdist(interp_curves_z, metric='correlation')
    # --- Hierarchical clustering ---
    Z_corr = linkage(dist_mat, method='average')
    # --- Silhouette-based optimal threshold ---
    best_t = 0
    best_score = -np.inf
    dist_square = squareform(dist_mat)
    for t in np.linspace(0.05, 1.0, 200):
        clusters_try = fcluster(Z_corr, t=t * max(Z_corr[:, 2]), criterion='distance')
        n_clusters_try = len(np.unique(clusters_try))
        if 2 <= n_clusters_try < n_samples:
            score = silhouette_score(dist_square, clusters_try, metric='precomputed')
            if score > best_score:
                best_score = score
                best_t = t * max(Z_corr[:, 2])
    clusters = fcluster(Z_corr, t=best_t, criterion='distance')
    n_clusters = len(np.unique(clusters))
    x = np.linspace(0, 100, n_points)
    # --- Combined figure ---
    fig = plt.figure(figsize=(2.25, 0.7 * n_clusters))
    gs = GridSpec(n_clusters, 2, width_ratios=[0.75, 2],
                  height_ratios=[1]*n_clusters, wspace=0.7, hspace=0.7)
    # --- Dendrogram ---
    ax_dendro = fig.add_subplot(gs[:, 0])
    dendro = dendrogram(Z_corr, labels=labels, orientation='left',
                        leaf_font_size=5, color_threshold=best_t, ax=ax_dendro)
    ax_dendro.tick_params(axis='y', labelsize=5)
    ax_dendro.tick_params(axis='x', bottom=False, labelbottom=False)
    for side in ['top', 'right', 'left', 'bottom']:
        ax_dendro.spines[side].set_visible(False)
    for lc in ax_dendro.collections:
        lc.set_linewidth(0.5)
    leaf_colors = {label: color for label, color in zip(dendro['ivl'], dendro['leaves_color_list'])}
    # --- Cluster plots (z-score!) ---
    for i, cid in enumerate(np.unique(clusters)):
        cluster_curves = interp_curves_z[clusters == cid]
        mean_curve = cluster_curves.mean(axis=0)
        sem_curve = cluster_curves.std(axis=0) / np.sqrt(cluster_curves.shape[0])
        first_label = labels[np.where(clusters == cid)[0][0]]
        color = leaf_colors[first_label]
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(x, mean_curve, color=color, linewidth=0.5)
        ax.fill_between(x, mean_curve - sem_curve, mean_curve + sem_curve,
                        color=color, alpha=0.25, edgecolor='none')
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.tick_params(axis='both', length=1, width=0.5, labelsize=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.3)
        ax.spines['bottom'].set_linewidth(0.3)
        n_units = cluster_curves.shape[0]
        y_max = mean_curve.max() + sem_curve.max() * 1.1 - 0.5
        ax.text(95, y_max, f"N={n_units}", ha='center', va='bottom',
                fontsize=5, fontname='Arial', color=color)
    plt.suptitle(f"{title}", fontsize=5, fontname='Arial', y=1.02)
    plt.tight_layout()
    plt.savefig(rf"path\{brain_region}_clusters_{title}.svg",
                format='svg')
    plt.close()   
    
#%% Main Analysis
brain_region = 'CLA' # ACC or CLA
brain_regions = {
    'CLA': ['sub016A', 'sub016B', 'sub017', 'sub024A'],
    'ACC': ['sub016A', 'sub016B', 'sub019A', 'sub019B', 'sub020', 'sub023', 'sub024A']
}
brain_region_color = {'CLA': '#3F5D71', 'ACC': '#EE9658'}
brain_region_color = brain_region_color[brain_region]
sub_neurons = pd.read_csv(fr'path\good_neurons_{brain_region}.csv')

subjects = brain_regions.get(brain_region, [])
n_perm = 10000
n_jobs = -1

save_path = 'path/Results_MI'
# Create directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

cnt = 0
neuron_labels = []
file_cnt_mapping = []
mi_it_all, mi_a_all = [], []
result_mi_it, result_mi_a = [], []
p_values_it_mi, p_values_a_mi= [], []      
mi_curve_it, mi_curve_a = [], []
for sub in subjects:    
    good_neurons = sub_neurons.loc[sub_neurons['goodneurons'].str.contains(sub, na=False)]  # Extract subject-specific neurons
    file_names = good_neurons.iloc[:, 0].astype(str)  # Convert to string in case of numeric values 
    mat_dir = os.path.join(f'path/{brain_region}/', f'{sub}')
    for i, file_name in enumerate(file_names):        
        cnt = cnt + 1
        print(cnt)
        file_cnt_mapping.append({'file_name': file_name, 'cnt': cnt})
        neuron_labels.append(cnt) # store Neuron_ID
        # load firing rate
        fr_file_path = os.path.join(mat_dir, f'{brain_region}_{file_name}.mat')
        fr = scipy.io.loadmat(fr_file_path)
        fr = fr['fr']
        # load spike times
        st_file_path = os.path.join(mat_dir, f'{brain_region}_{file_name}_spike_times.mat')       
        st = scipy.io.loadmat(st_file_path)
        st = st['spike_times']
        # load behavior
        file_path = os.path.join(f'path/{brain_region}/', f'{sub}/{brain_region}_{sub}.xlsx')
        behavior = pd.read_excel(file_path)   
        columns_to_keep = ["A_safety_variance", "B_safety_variance", "A_absolute_prediction_error", "B_absolute_prediction_error", "A_outcome", "B_outcome"]
        outcome = behavior["outcome"] # Hit and Miss
        behavior = behavior[columns_to_keep]
        # load y-position
        file_path = os.path.join(r'path\yPos', f'{sub}_gameinfo_yPos.csv')
        yPos = pd.read_csv(file_path)  
        
        baseline_std = np.std(fr, axis=1, keepdims=True)
        zero_rows = np.where(baseline_std == 0)[0]
        fr = np.delete(fr, zero_rows, axis=0)
        st = np.delete(st, zero_rows, axis=0)
        baseline_std = np.delete(baseline_std, zero_rows, axis=0)
        behavior = behavior.drop(index=zero_rows).reset_index(drop=True)  
        outcome = outcome.drop(index=zero_rows).reset_index(drop=True)  
        yPos = yPos.drop(index=zero_rows).reset_index(drop=True)     
               
        # Parameters for statistics
        time_vector = np.linspace(-4, 2, fr.shape[1]) 
        intertrail_idx = (time_vector >= -4) & (time_vector <= 0)
        asteroid_idx = (time_vector >= 0) & (time_vector <= 1.5)        
        baseline_idx = (time_vector >=1.5) & (time_vector <= 2) # define baseline as 1.5 to 2      
        
        # Spike counts
        it_sp = np.sum(fr[:, intertrail_idx], axis=1)/20 # convert back to spikes (50ms (20hz))
        a_sp = np.sum(fr[:, asteroid_idx], axis=1)/20
        
        # Define spike counts
        intertrial = it_sp
        asteroid = a_sp
        
        # Mutual Information
        x = behavior
        z = yPos.copy()
        mi_it, p_it_mi, = individual_permutation_mi(
            pd.concat([x.iloc[:,:2], z.iloc[:, [1]]], axis=1), intertrial, n_perm=n_perm)        
        mi_a, p_a_mi = individual_permutation_mi(
            pd.concat([x, z.iloc[:, [2]], outcome.to_frame()],  axis=1), asteroid, n_perm=n_perm)
        # Store per-neuron        
        mi_it_all.append(mi_it), mi_a_all.append(mi_a)
        p_values_it_mi.append(p_it_mi), p_values_a_mi.append(p_a_mi)
        
        target_behaviors = ['A_safety_variance', 'B_safety_variance', 'A_absolute_prediction_error', 'B_absolute_prediction_error']
        
        # Compute MI Curves
        # Intertrial
        for idx in range(3):
            if idx == 2:
                behavior_name = 'y-Position'
                x = z.iloc[:, 1]
            else:
                behavior_name = behavior.columns[idx]
                x = behavior.iloc[:, idx]
            rho, _ = spearmanr(x, intertrial)
        
            if p_it_mi[idx] < 0.05:
                entry = build_entry('MI_score', brain_region, cnt, behavior_name, 'intertrial', it_sp, np.array(x), rho, p_it_mi[idx], fr, outcome)
                result_mi_it.append(entry)
                plot_raster(trials=st, fr=fr, behavior=np.array(x),
                    time_axis=np.linspace(-4, 2, fr.shape[1]),
                    column=behavior_name, brain_region=brain_region, file_name=file_name,
                    folder_path=r'C:/Users/rd883/Desktop/CLA_high_cog_feature/Figures/Results_MI/Rasters/intertrial')
   
        # Asteroid
        for idx in range(8):
            if idx == 7:
                behavior_name = 'Outcome'
                x = outcome.to_frame()
            elif idx == 6:
                behavior_name = 'y-Position'
                x = z.iloc[:, 2]
            else:
                behavior_name = behavior.columns[idx]
                x = behavior.iloc[:, idx]
            rho, _ = spearmanr(x, asteroid)
        
            if p_a_mi[idx] < 0.05:
                entry = build_entry('MI_score', brain_region, cnt, behavior_name, 'asteroid', a_sp, np.array(x), rho, p_a_mi[idx], fr, outcome)
                result_mi_a.append(entry)
                plot_raster(trials=st, fr=fr, behavior=np.array(x),
                    time_axis=np.linspace(-4, 2, fr.shape[1]),
                    column=behavior_name, brain_region=brain_region, file_name=file_name,
                    folder_path=r'C:/Users/rd883/Desktop/CLA_high_cog_feature/Figures/Results_MI/Rasters/asteroid')

# Save mapping to Excel
mapping_df = pd.DataFrame(file_cnt_mapping)
mapping_df.to_excel(rf'path/Results_MI/{brain_region}_file_name_to_cnt_mapping.xlsx', index=False)

#%% Chord Diagram Data
save_results_to_excel(brain_region=brain_region,
    result_mi_it=result_mi_it, result_mi_a=result_mi_a, save_folder=r'path/Results_MI/')    

#%% MI Plot
columns_to_keep += ['y-Position', 'Outcome'] if 'y-Position' not in columns_to_keep else []
plot_mi_heatmaps(
    np.vstack(mi_it_all), 
    np.vstack(mi_a_all),
    columns_to_keep, neuron_labels, brain_region,
    method_label="MI_score",
    save_path=r'path/Results_MI')

#%% p-Value Plot and cell type
# Cell Types
file_path = 'path\\cell_types.csv'  # Replace with your CSV file path
cell_type = pd.read_csv(file_path)
mapping_dict = dict(zip(mapping_df['file_name'], mapping_df['cnt']))
cell_type['Neuron_ID_mapped'] = cell_type['Neuron_ID'].map(mapping_dict)

if brain_region == 'ACC':
    scale_intertrial = 0.35
    scale_asteroid = 0.65
else:
    scale_intertrial = 0.35
    scale_asteroid = 0.52
# Intertrial period
plot_pval_heatmap(
    np.vstack(p_values_it_mi), neuron_labels,
    columns_to_keep[:2] + [columns_to_keep[6]],
    brain_region, "MI_score", save_path, cell_type, 'Intertrial', scale_intertrial)

# Asteroid period
plot_pval_heatmap(
    np.vstack(p_values_a_mi), neuron_labels,
    columns_to_keep, brain_region, "MI_score", save_path, cell_type, 'Asteroid', scale_asteroid)

# Intertrial
plot_cell_type_proportions(
    np.vstack(p_values_it_mi), neuron_labels,
    cell_type, columns_to_keep[:2]+[columns_to_keep[6]],
    brain_region, title='Intertrial',
    save_path=r'path/Results_MI')
# Asteroid
plot_cell_type_proportions(
    np.vstack(p_values_a_mi), neuron_labels,
    cell_type, columns_to_keep, brain_region, title='Asteroid',
    save_path=r'path/Results_MI')

#%% Plot Behavior vs FR
methods = {
    'mi': {'intertrial': result_mi_it, 'asteroid': result_mi_a}
}
exclude_behaviors = {'y-Position', 'A_outcome', 'B_outcome', 'Outcome'}
# Loop over each timepoint in the 'mi' method
for time_key in methods['mi']:
    results = methods['mi'][time_key]    
    # Normalize behavior_name just in case (e.g., trim and lowercase)
    for d in results:
        d['behavior_name'] = d['behavior_name'].strip()        
    # Step 1: Flag file_names that contain any excluded behavior
    flagged_filenames = {
        d['file_name'] for d in results
        if d['behavior_name'] in exclude_behaviors}
    # Step 2: Filter out entries by behavior or flagged file
    filtered_results = [
        d for d in results
        if d['behavior_name'] not in exclude_behaviors
        and d['file_name'] not in flagged_filenames]
     # Step 3: Save filtered results back
    methods['mi'][time_key] = filtered_results
base_save_dir = r'path\Results_MI'

# Call plot_individual_neurons with collect_curves
matplotlib.use("Agg")  # non-interactive backend
all_curves, all_stat = [], []
for method, time_dict in methods.items():
    for time_point, res in time_dict.items():
        save_folder = os.path.join(base_save_dir, f"{method}_{time_point}")
        stat_data = plot_individual_neurons(res, method, time_point, save_folder, collect_curves=all_curves, brain_region=brain_region)
        all_stat.extend(stat_data)

df_stats = pd.DataFrame(all_stat)
os.makedirs(base_save_dir, exist_ok=True)
# Build full output path with brain_region in the filename
output_path = os.path.join(base_save_dir, f"statistics_{brain_region}.xlsx")
# Save to Excel
df_stats.to_excel(output_path, index=False)

matplotlib.use("TkAgg")

#%% Separate neurons by shape (z-scored version)
intertrial_curves = [c for c in all_curves if c['time_point'] == "intertrial" 
                     and "safety_variance" in c['behavior_name']]
asteroid_safety = [c for c in all_curves if c['time_point'] == "asteroid"
                   and "safety_variance" in c['behavior_name']]
asteroid_ape = [c for c in all_curves if c['time_point'] == "asteroid"
                and "absolute_prediction_error" in c['behavior_name']]

process_and_cluster(intertrial_curves, "Intertrial (Safety Variance)", brain_region)
process_and_cluster(asteroid_safety, "Asteroid (Safety Variance)", brain_region)
process_and_cluster(asteroid_ape, "Asteroid (Absolute Prediction Error)", brain_region)
