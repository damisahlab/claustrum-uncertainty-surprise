"""
Cluster-based permutation test analysis (per neuron and behavior).
Includes raster plots and firing rate comparisons:
    - High vs Low
    - Crash vs Avoid

This code generates the following figures:
    Main Figures:
        4e-j
        5a-f
    Extended Data Figures:
        5a-e
        6a-f
        9f-h

IDE: Spyder
Date: 08/2025
"""
#%% Libraries
from IPython import get_ipython
get_ipython().magic('reset -sf')  # Reset environment (force, no confirmation)
get_ipython().magic('clear')      # Clear console

import gc
import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats as stats
from scipy.ndimage import uniform_filter1d, label

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (for saving plots only)
# matplotlib.use('TkAgg')  # Use GUI for interactive display
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Plot style
plt.rcParams['agg.path.chunksize'] = 10000  # reduce memory usage
mpl.rcParams['svg.fonttype'] = 'none'       # keep text as text
mpl.rcParams['patch.linewidth'] = 0         # no default stroke

#%% Functions
def process_data(fr_values, indices, baseline_idx):
    """Subtract baseline firing rate for selected trials."""
    fr_data = fr_values[indices, :]
    baseline = np.mean(fr_data[:, baseline_idx], axis=1, keepdims=True)
    return fr_data - baseline

def cbp(data, n_permutations=10000, p_threshold=0.05):
    """
    Cluster-based permutation test
    One-sided test: Test for regions where data > baseline.
    Based on Maris & Oostenveld (2007).
    """
    rng = np.random.default_rng() # set seed for reproducibility
    n_trials, n_time = data.shape    
    # 1. Observed t-values
    t_obs = np.mean(data, axis=0) / ((np.std(data, axis=0, ddof=1) / np.sqrt(n_trials)) + 1e-12)
    t_obs[~np.isfinite(t_obs)] = 0
    # 2. Cluster-forming threshold (one-sided)
    threshold = stats.t.ppf(1 - p_threshold, df=n_trials-1)
    # 3. Find clusters above threshold
    cluster_mask = t_obs > threshold
    cluster_labels, n_clusters = label(cluster_mask)   
    # Cluster masses
    cluster_stats = [np.sum(t_obs[cluster_labels == c]) for c in range(1, n_clusters+1)]
    # 4. Permutations
    max_cluster_stats = np.zeros(n_permutations)
    for perm in range(n_permutations):
        signs = rng.choice([-1, 1], size=n_trials)
        perm_data = data * signs[:, None]
        t_perm = np.mean(perm_data, axis=0) / ((np.std(perm_data, axis=0, ddof=1) / np.sqrt(n_trials)) + 1e-12)
        t_perm[~np.isfinite(t_perm)] = 0.0
        cluster_mask_perm = t_perm > threshold
        cluster_labels_perm, n_clusters_perm = label(cluster_mask_perm)
        perm_cluster_stats = [np.sum(t_perm[cluster_labels_perm == c]) for c in range(1, n_clusters_perm+1)]
        max_cluster_stats[perm] = np.max(perm_cluster_stats) if perm_cluster_stats else 0
    # 5. Compute cluster p-values
    cluster_pvals = [(np.sum(max_cluster_stats >= stat) + 1) / (n_permutations + 1) 
                 for stat in cluster_stats]
    # 6. Keep only significant clusters
    sig_pvals = []
    sig_labels = np.zeros_like(cluster_labels)
    for i, pval in enumerate(cluster_pvals, start=1):
        if pval < 0.05:  # significance threshold
            sig_pvals.append(pval)
            sig_labels[cluster_labels == i] = i  # keep only significant cluster
    return sig_pvals, sig_labels

def cbp_highvslow(high, low, n_permutations=10000, p_threshold=0.05, tail='both'):
    """
    Cluster-based permutation test comparing High vs Low firing rates
    (paired samples - same neurons, different conditions).
    Uses sign-flipping permutations appropriate for paired data.        
    Returns sig_pvals (list) and sig_labels (array).
    """
    rng = np.random.default_rng()
    n_neurons, n_time = high.shape
    # 1. Observed t-values
    diff = high - low
    se = np.std(diff, axis=0, ddof=1) / np.sqrt(n_neurons)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_obs = np.mean(diff, axis=0) / se
    t_obs[~np.isfinite(t_obs)] = 0.0
    # 2. Cluster-forming threshold
    if tail == 'both':
        threshold = stats.t.ppf(1 - p_threshold/2, df=n_neurons-1)
    else:
        threshold = stats.t.ppf(1 - p_threshold, df=n_neurons-1)
    # 3. Observed clusters
    pos_mask = t_obs > threshold if tail in ('both', 'larger') else np.zeros(n_time, dtype=bool)
    neg_mask = t_obs < -threshold if tail in ('both', 'smaller') else np.zeros(n_time, dtype=bool)
    cluster_labels_pos, n_pos = label(pos_mask)
    cluster_labels_neg, n_neg = label(neg_mask)
    # Shift negative labels to avoid overlap
    cluster_labels_neg[cluster_labels_neg > 0] += n_pos
    cluster_labels = cluster_labels_pos + cluster_labels_neg
    # Observed cluster masses
    n_clusters = n_pos + n_neg
    cluster_stats = [np.sum(t_obs[cluster_labels == c]) if c <= n_pos else
                     np.sum(-t_obs[cluster_labels == c]) for c in range(1, n_clusters+1)]
    # 4. Permutations
    max_cluster_stats = np.zeros(n_permutations)
    for perm in range(n_permutations):
        signs = rng.choice([1, -1], size=n_neurons)[:, None]
        perm_diff = diff * signs
        se_perm = np.std(perm_diff, axis=0, ddof=1) / np.sqrt(n_neurons)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_perm = np.mean(perm_diff, axis=0) / se_perm
        t_perm[~np.isfinite(t_perm)] = 0.0
        # Positive clusters (only if requested)
        if tail in ('both', 'larger'):
            pos_mask_perm = t_perm > threshold
            cluster_labels_pos_perm, n_pos_perm = label(pos_mask_perm)
        else:
            cluster_labels_pos_perm = np.zeros(n_time, dtype=int)
            n_pos_perm = 0
        # Negative clusters (only if requested)
        if tail in ('both', 'smaller'):
            neg_mask_perm = t_perm < -threshold
            cluster_labels_neg_perm, n_neg_perm = label(neg_mask_perm)
        else:
            cluster_labels_neg_perm = np.zeros(n_time, dtype=int)
            n_neg_perm = 0
        # Combine
        if n_neg_perm > 0:
            cluster_labels_neg_perm[cluster_labels_neg_perm > 0] += n_pos_perm
        cluster_labels_perm = cluster_labels_pos_perm + cluster_labels_neg_perm

        perm_cluster_stats = [np.sum(t_perm[cluster_labels_perm == c]) if c <= n_pos_perm else
                              np.sum(-t_perm[cluster_labels_perm == c])
                              for c in range(1, n_pos_perm + n_neg_perm + 1)]
        max_cluster_stats[perm] = np.max(perm_cluster_stats) if perm_cluster_stats else 0.0    
    # 5. Cluster p-values
    cluster_pvals = [(np.sum(max_cluster_stats >= stat) + 1) / (n_permutations + 1)
                     for stat in cluster_stats]
    # 6. Keep only significant clusters
    sig_pvals = []
    sig_labels = np.zeros_like(cluster_labels)
    for i, pval in enumerate(cluster_pvals, start=1):
        if pval < 0.05:
            sig_pvals.append(pval)
            sig_labels[cluster_labels == i] = i
    return sig_pvals, sig_labels

def clean_st_values(trials):
    """Efficiently clean spike time data."""
    cleaned_trials = []
    for trial in trials:
        try:
            flat_trial = np.concatenate([np.ravel(np.array(t, dtype=float)) for t in trial])
            cleaned_trials.append(flat_trial)
        except:
            cleaned_trials.append([])  # Handle errors without print statements
    return cleaned_trials

def gen_raster(trials, start_time=-2, end_time=4):    
    raster = []
    for trial in trials:
        try:
            flat_trial = np.concatenate([np.ravel(np.array(t, dtype=float)) for t in trial])
            valid_times = flat_trial[(flat_trial >= start_time) & (flat_trial <= end_time)]
            raster.append(valid_times)
        except Exception:
            raster.append([])  # If any error occurs (e.g., empty or malformed trial), append empty list
    return raster

def generate_raster(st_low, st_high, labels_high, labels_low, behavior_name, cond):
    """Generate raster plot data efficiently."""
    bar_high = (labels_high > 0).astype(int)
    bar_low = (labels_low > 0).astype(int)
    low_st_data, high_st_data = clean_st_values(st_low), clean_st_values(st_high)
    np.random.shuffle(low_st_data)   # shuffle trial order
    np.random.shuffle(high_st_data)  # shuffle trial order
    trial_spike_times = high_st_data + low_st_data    
    trial_lengths = [len(trial[(trial >= -2) & (trial <= 4)]) for trial in trial_spike_times]
    total_spikes = sum(trial_lengths)
    raster_x = np.zeros(total_spikes)
    raster_y = np.zeros(total_spikes)
    raster_colors = np.empty(total_spikes, dtype=object)
    counter = 0
    for t, trial in enumerate(trial_spike_times):
        valid_times = trial[(trial >= -2) & (trial <= 4)]
        num_spikes = len(valid_times)
        raster_x[counter:counter + num_spikes] = valid_times
        raster_y[counter:counter + num_spikes] = t
        if behavior_name in ['A_safety_variance', 'B_safety_variance']:
            color_high = "orange"
            color_low = "blue"
        else:
            color_high = "#ff408c"
            color_low = "#00aaff"
        raster_colors[counter:counter + num_spikes] = (
            color_high if t < len(trial_spike_times) // 2 else color_low
        )
        counter += num_spikes
        
    # Plot
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.115, 1.2), constrained_layout=True)    
    fig.suptitle(behavior_name, fontsize=5)
    
    # --- Raster Plot ---
    ax_raster = axes[0]
    ax_raster.scatter(raster_x, raster_y, s=.1, c=raster_colors, marker='s')
    ax_raster.axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax_raster.axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax_raster.set_xlim(-2, 4)
    ax_raster.set_xticks([])
    ax_raster.set_ylabel("trial #")
    ax_raster.tick_params(axis='both', which='both', length=1, width=0.5)
    ax_raster.spines['top'].set_visible(False)
    ax_raster.spines['right'].set_visible(False)
    ax_raster.spines['bottom'].set_visible(False)
    
    for spine in ax_raster.spines.values():
        spine.set_linewidth(0.3)
        
    # --- Firing Rate Plot ---
    ax_fr = axes[1]
    low_fr = fr_values[np.where(low_indices)[0]]
    high_fr = fr_values[np.where(high_indices)[0]]
    low_fr, high_fr = np.mean(low_fr, axis=0), np.mean(high_fr, axis=0)
    bin_size = int(400/50)  # smoothing
    high_fr = uniform_filter1d(high_fr, bin_size, axis=0)
    low_fr = uniform_filter1d(low_fr, bin_size, axis=0)
    ax_fr.plot(time_axis, high_fr, color=color_high, label="High", linewidth=0.5)
    ax_fr.plot(time_axis, low_fr, color=color_low, label="Low", linewidth=0.5)
    ax_fr.axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax_fr.axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax_fr.set_xlim(-2, 4)
    ax_fr.set_xticks([])    
    ax_fr.set_ylabel("mean $\Delta$ firing rate (Hz)")
    ax_fr.tick_params(axis='both', which='both', length=1, width=0.5)
    ax_fr.spines['top'].set_visible(False)
    ax_fr.spines['right'].set_visible(False)
    
    for spine in ax_fr.spines.values():
        spine.set_linewidth(0.3)
        
    cmap_high = ListedColormap(['white', color_high])
    cmap_low  = ListedColormap(['white', color_low])
    bar_high_ax = ax_fr.inset_axes([0, -0.04, 1, 0.02])
    bar_high_ax.imshow(np.expand_dims(bar_high, axis=0), aspect='auto', cmap=cmap_high, origin='lower', extent=[-2, 4, 0, 1])
    bar_high_ax.axis('off')    
    bar_low_ax = ax_fr.inset_axes([0, -0.07, 1, 0.02])
    bar_low_ax.imshow(np.expand_dims(bar_low, axis=0), aspect='auto', cmap=cmap_low, origin='lower', extent=[-2, 4, 0, 1])
    bar_low_ax.axis('off')
    
    # Save inside behavior folder
    fig.savefig(os.path.join(behavior_folder, f'{brain_region}_{file_name}_{cond}.svg'), format='svg', dpi=300, transparent=True)
    fig.savefig(os.path.join(behavior_folder, f'{brain_region}_{file_name}_{cond}.pdf'), format='pdf')
    plt.close(fig)
    gc.collect()

def plot_raster(trials, fr, time_axis, brain_region,
                file_name, folder_path, start_time=-2, end_time=4):
    """Generate raster + mean firing rate plots for a single neuron."""
    os.makedirs(folder_path, exist_ok=True)
    # Generate raster
    raster = gen_raster(trials, start_time=start_time, end_time=end_time)    
    # Demeaning
    baseline_mask = (time_axis >= 1.5) & (time_axis <= 2)
    fr = (fr - np.mean(fr[:,baseline_mask]))
    # Smooth firing rates
    bin_size = int(400/50)  # Assuming 50 ms bins
    fr_all_smooth = uniform_filter1d(np.mean(fr, axis=0), bin_size)
    
    # Set up 2-column figure
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.115, 1.2), constrained_layout=True)
    
    # Left Panel – Original Raster
    raster_x, raster_y = [], []
    for i, trial_times in enumerate(raster):
        raster_x.extend(trial_times)
        raster_y.extend([i] * len(trial_times))
    ax_raster = axes[0]
    ax_raster.scatter(raster_x, raster_y, s=.1, c='#22205F', marker='s')
    ax_raster.axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax_raster.axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax_raster.set_xlim(start_time, end_time)
    ax_raster.set_xticks([])
    ax_raster.set_ylabel("trial #")    
    ax_raster.tick_params(axis='both', which='both', length=1, width=0.5)
    
    for spine in ax_raster.spines.values():
        spine.set_linewidth(0.3)
    ax_raster.spines['top'].set_visible(False)
    ax_raster.spines['right'].set_visible(False)
    ax_raster.spines['bottom'].set_visible(False)
    
    # Left Bottom – Mean FR
    ax_fr = axes[1]
    ax_fr.plot(time_axis, fr_all_smooth, color='#22205F', linewidth=0.5)
    ax_fr.axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax_fr.axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax_fr.set_xlim(start_time, end_time)
    ax_fr.set_xticks([])    
    ax_fr.set_ylabel("mean $\Delta$ firing rate (Hz)")
    ax_fr.tick_params(axis='both', which='both', length=1, width=0.5)
    
    for spine in ax_fr.spines.values():
        spine.set_linewidth(0.3)
    ax_fr.spines['top'].set_visible(False)
    ax_fr.spines['right'].set_visible(False)  
    
    # Save figure
    fname = f"{brain_region}_{file_name}"
    save_path = os.path.join(folder_path, f"Rasters/{fname}.pdf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='pdf')
    plt.close(fig) 
    
def filter_clusters(labels, min_bins=1):        
    labels_filtered = np.zeros_like(labels)
    cluster_positions = []
    i = 0
    while i < len(labels):
        if labels[i] > 0:
            cluster_id = labels[i]
            start = i
            while i < len(labels) and labels[i] == cluster_id:
                i += 1
            end = i
            length = end - start
            if length >= min_bins:   # Keep only clusters >= min_bins
                labels_filtered[start:end] = cluster_id
                cluster_positions.append((start, end, cluster_id))
        else:
            i += 1
    return labels_filtered, cluster_positions

def sem_ci_2d(data):
    """Compute confidence intervals using SEM for each time point."""
    mean_vals = np.mean(data, axis=0)
    sem_vals = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])       
    low_bound = mean_vals - sem_vals
    high_bound = mean_vals + sem_vals
    return low_bound, high_bound

def plot_condition(high, low, crash, avoid, crash_dash=None, avoid_dash=None, behavior_name=None, cond=None, condition=None):
    """
    Plot 3-panel figure for:
      1) High vs Low (mean +/- CI) + cluster bar
      2) Crash vs Avoid (condition 1)
      3) Crash vs Avoid (condition 2)
    """
    label_high, label_low = 'High', 'Low'
    label_crash, label_avoid = 'Crash', 'Avoid'
    
    color_crash = "#8F39E6"
    color_avoid = "#00cc00"    
    if behavior_name in ['A_safety_variance', 'B_safety_variance']:
        color_high = "orange"
        color_low = "blue"
    else:
        color_high = "#ff408c"
        color_low = "#00aaff" 
        
    # High vs Low
    high = np.array(high)      
    low = np.array(low)      
    confidence_interval='sem_ci_2d'
    # Preprocessing raw data
    time_axis = np.linspace(-2, 4, fr_values.shape[1])
    baseline_idx = (time_axis >= -2) & (time_axis <= -1.5)
    baseline_low = low[:,baseline_idx]
    baseline_high = high[:,baseline_idx]     
    # Z-scoring    
    high = (high - np.mean(baseline_high, axis=1, keepdims=True)) / np.where(np.std(baseline_high, axis=1, keepdims=True) == 0, 1, np.std(baseline_high, axis=1, keepdims=True))
    low = (low - np.mean(baseline_low, axis=1, keepdims=True)) / np.where(np.std(baseline_low, axis=1, keepdims=True) == 0, 1, np.std(baseline_low, axis=1, keepdims=True))
    # Convolution 
    bin_size = int(400/50) # (sliding windows size/bin size)                                                      
    # using filter1d
    high = uniform_filter1d( high, bin_size, axis=1)
    low = uniform_filter1d( low, bin_size, axis=1)
    # bootstrap or sem
    ci_high = globals()[confidence_interval](high)
    ci_low = globals()[confidence_interval](low)
    
    # Crash vs Avoid
    crash = np.array(crash)      
    avoid = np.array(avoid)      
    confidence_interval='sem_ci_2d'
    # Preprocessing raw data
    time_axis = np.linspace(-2, 4, fr_values.shape[1])
    baseline_idx = (time_axis >= -2) & (time_axis <= -1.5)
    baseline_avoid = avoid[:,baseline_idx]
    baseline_crash = crash[:,baseline_idx]     
    # Z-scoring  
    crash = (crash - np.mean(baseline_crash, axis=1, keepdims=True)) / np.where(np.std(baseline_crash, axis=1, keepdims=True) == 0, 1, np.std(baseline_crash, axis=1, keepdims=True))
    avoid = (avoid - np.mean(baseline_avoid, axis=1, keepdims=True)) / np.where(np.std(baseline_avoid, axis=1, keepdims=True) == 0, 1, np.std(baseline_avoid, axis=1, keepdims=True))                                                   
    # using filter1d
    crash = uniform_filter1d( crash, bin_size, axis=1)
    avoid = uniform_filter1d( avoid, bin_size, axis=1)
    # bootstrap or sem
    ci_crash = globals()[confidence_interval](crash)
    ci_avoid = globals()[confidence_interval](avoid) 
    
    # cluster-based permutation
    p_highvslow, labels_highvslow = cbp_highvslow(high, low, n_permutations=10000, p_threshold=0.05, tail='both')
    p_avoidvscrash, labels_avoidvscrash = cbp_highvslow(avoid, crash, n_permutations=10000, p_threshold=0.05, tail='both')    
    p_highvslow, p_avoidvscrash = np.array(p_highvslow), np.array(p_avoidvscrash)
    labels_highvslow, labels_avoidvscrash = np.array(labels_highvslow), np.array(labels_avoidvscrash)    
    labels_highvslow, clusters_highvslow = filter_clusters(labels_highvslow, min_bins=2)
    labels_avoidvscrash, clusters_avoidvscrash = filter_clusters(labels_avoidvscrash, min_bins=2)
    
    # Plot
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5    
    fig, ax = plt.subplots(3, 1, figsize=(0.95, 2.7))                   
    plt.subplots_adjust(hspace=0.25)  # Increase space between subplots
    fig.suptitle(behavior_name, fontsize=5) 
    
    # Panel 1: High vs Low
    ax[0].plot(time_axis, np.mean(high, axis=0), color=color_high, label=label_high, linewidth = .5)
    ax[0].plot(time_axis, np.mean(low, axis=0), color=color_low, label=label_low, linewidth = .5)
    ax[0].fill_between(time_axis, ci_high[0], ci_high[1], color=color_high, alpha=0.25, edgecolor='none')
    ax[0].fill_between(time_axis, ci_low[0], ci_low[1], color=color_low, alpha=0.25, edgecolor='none')
    ax[0].axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax[0].axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax[0].set_xlim(-2, 4)
    ax[0].set_xticklabels([]) # hide x-axis labels
    ax[0].xaxis.set_ticks([]) # hide all ticks
    ax[0].grid(False)         # hide any grid lines
    ax[0].tick_params(axis='y', which='both', length=1, width=0.5)
    ax[0].set_title(f"(n = {high.shape[0]})", fontsize=5)
    ax[0].tick_params(axis='both', which='both', length=1, width=0.5)       
    # Cluster bars + p-values
    pos0 = ax[0].get_position()
    bar_ax0 = fig.add_axes([pos0.x0, pos0.y0 + 0.005, pos0.width, 0.005])
    bar_ax0.imshow(np.expand_dims(labels_highvslow > 0, axis=0), aspect='auto', cmap='Greys', origin='lower',
                   extent=[time_axis[0], time_axis[-1], 0, 1])
    bar_ax0.axis('off')
    
    # Panel 2: Crash vs Avoid (condition 1)
    ax[1].text(2.8, 3.3, cond, color='black', fontfamily='Arial', fontsize=5)
    ax[1].plot(time_axis, np.mean(crash, axis=0), color=color_crash, label=label_crash, linewidth=0.5)
    ax[1].plot(time_axis, np.mean(avoid, axis=0), color=color_avoid, label=label_avoid, linewidth=0.5)    
    ax[1].fill_between(time_axis, ci_crash[0], ci_crash[1], color=color_crash, alpha=0.25, edgecolor='none')
    ax[1].fill_between(time_axis, ci_avoid[0], ci_avoid[1], color=color_avoid, alpha=0.25, edgecolor='none')
    ax[1].axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax[1].axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax[1].set_xlim(-2, 4)
    ax[1].set_xticks(np.arange(-2, 5, 2))
    # Cluster bars + p-values
    pos1 = ax[1].get_position()
    bar_ax1 = fig.add_axes([pos1.x0, pos1.y0 + 0.005, pos1.width, 0.005])
    bar_ax1.imshow(np.expand_dims(labels_avoidvscrash > 0, axis=0), aspect='auto', cmap='Greys', origin='lower',
                   extent=[time_axis[0], time_axis[-1], 0, 1])
    bar_ax1.axis('off')
    
    # Panel 3: Crash vs Avoid (condition 2)
    crash_dash = np.array(crash_dash)
    avoid_dash = np.array(avoid_dash)    
    baseline_crash_d = crash_dash[:, baseline_idx]
    baseline_avoid_d = avoid_dash[:, baseline_idx]
    crash_dash = (crash_dash - np.mean(baseline_crash_d, axis=1, keepdims=True)) / np.where(np.std(baseline_crash_d, axis=1, keepdims=True) == 0, 1, np.std(baseline_crash_d, axis=1, keepdims=True))
    avoid_dash = (avoid_dash - np.mean(baseline_avoid_d, axis=1, keepdims=True)) / np.where(np.std(baseline_avoid_d, axis=1, keepdims=True) == 0, 1, np.std(baseline_avoid_d, axis=1, keepdims=True))
    crash_dash = uniform_filter1d(crash_dash, bin_size, axis=1)
    avoid_dash = uniform_filter1d(avoid_dash, bin_size, axis=1)   
    ci_crash_d = globals()[confidence_interval](crash_dash)
    ci_avoid_d = globals()[confidence_interval](avoid_dash) 
    p_avoidvscrash_d, labels_avoidvscrash_d = cbp_highvslow(avoid_dash, crash_dash, n_permutations=10000, p_threshold=0.05, tail='both')            
    labels_avoidvscrash_d = np.array(labels_avoidvscrash_d)
    labels_avoidvscrash_d, clusters_avoidvscrash_d = filter_clusters(labels_avoidvscrash_d, min_bins=2)
    if cond == 'High':
        ax[2].text(2.8, 3.7, 'Low', color='black', fontfamily='Arial', fontsize=5)
    else:
        ax[2].text(2.8, 3.7, 'High', color='black', fontfamily='Arial', fontsize=5)
    ax[2].plot(time_axis, np.mean(crash_dash, axis=0), color=color_crash, label=label_crash, linewidth=0.5)
    ax[2].plot(time_axis, np.mean(avoid_dash, axis=0), color=color_avoid, label=label_avoid, linewidth=0.5)    
    ax[2].fill_between(time_axis, ci_crash_d[0], ci_crash_d[1], color=color_crash, alpha=0.25, edgecolor='none')
    ax[2].fill_between(time_axis, ci_avoid_d[0], ci_avoid_d[1], color=color_avoid, alpha=0.25, edgecolor='none')
    ax[2].axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax[2].axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax[2].set_xlim(-2, 4)
    ax[2].set_xticks(np.arange(-2, 5, 2))          
    pos2 = ax[2].get_position()
    bar_ax2 = fig.add_axes([pos2.x0, pos2.y0 + 0.005, pos1.width, 0.005])
    bar_ax2.imshow(np.expand_dims(labels_avoidvscrash_d > 0, axis=0), aspect='auto', cmap='Greys', origin='lower',
                   extent=[time_axis[0], time_axis[-1], 0, 1])
    bar_ax2.axis('off')
    
    for i in range(3):  # Loop through both subplots
        for spine in ax[i].spines.values():
            spine.set_linewidth(0.3)  # Make surrounding lines thinner
        ax[i].spines['top'].set_visible(False)  # Remove top bar
        ax[i].spines['right'].set_visible(False)  # Remove right bar    
        # Adjust tick labels to be closer to axes
        ax[i].tick_params(axis='both', which='both', length=1, width=0.5, pad=1)    
        # Adjust axis labels to be closer to the tick labels
        ax[i].set_xlabel(ax[i].get_xlabel(), labelpad=1)
        ax[i].set_ylabel(ax[i].get_ylabel(), labelpad=1)  
        
    # Save the figure
    fig.savefig(os.path.join(folder_path, condition, f'{brain_region}_{behavior_name}_{cond}.svg'), dpi=300, bbox_inches='tight', format='svg')
    fig.savefig(os.path.join(folder_path, condition, f'{brain_region}_{behavior_name}_{cond}.pdf'), dpi=300, bbox_inches='tight', format='pdf')
    plt.close(fig)
    gc.collect()
    
#%% Main Analysis
brain_region = 'AMY' # ACC, AMY, CLA
low_percentile = 0.30 # Low condition cutoff
high_percentile = 1-low_percentile

# Subjects by region
brain_regions = {
    'ACC': ['sub016A', 'sub016B', 'sub019A', 'sub019B', 'sub020', 'sub023', 'sub024A'],
    'AMY': ['sub016A', 'sub016B', 'sub024A'],
    'CLA': ['sub016A', 'sub016B', 'sub017', 'sub024A']
}
subjects = brain_regions.get(brain_region, [])

# Output folder
folder_path = os.path.join(f'path/{brain_region}/Population_cbp/')

# Crete subfolder
for sub in ['Data', 'Rasters', 'Appear', 'Appear/rasters', 'Event', 'Event/rasters']:
    os.makedirs(os.path.join(folder_path, sub), exist_ok=True)

for condition, (a_s, a_e) in zip(['Appear', 'Event'], [(0, 0.5), (0.67, 1.17)]):   
    behavior_columns = ["A_absolute_prediction_error", "A_safety_variance", "B_absolute_prediction_error", "B_safety_variance"]    
    file_path = os.path.join('path to bad_neurons', f'{brain_region}_badneurons_{condition}.xlsx')
    badneurons = pd.read_excel(file_path)
    for behavior_name in behavior_columns:
        badneuron_high = badneurons[f"{behavior_name}_high"].dropna().tolist()
        badneuron_low  = badneurons[f"{behavior_name}_low"].dropna().tolist()
        # Create behavior-specific folder inside rasters
        behavior_folder = os.path.join(folder_path, condition, 'rasters', behavior_name)
        if not os.path.exists(behavior_folder):
            os.makedirs(behavior_folder)
        hh_pa, hl_pa, lh_pa, ll_pa, hh_pa_o, hl_pa_o, lh_pa_o, ll_pa_o, h_pa_units, l_pa_units, h_pa_bar, l_pa_bar, h_pa_p, l_pa_p = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
        hh_a, hl_a, lh_a, ll_a, hh_a_o, hl_a_o, lh_a_o, ll_a_o, h_a_units, l_a_units, h_a_bar, l_a_bar, h_a_p, l_a_p = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
        hh_e, hl_e, lh_e, ll_e, hh_e_o, hl_e_o, lh_e_o, ll_e_o, h_e_units, l_e_units, h_e_bar, l_e_bar, h_e_p, l_e_p = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
        for sub in subjects:
            good_neurons = pd.read_csv(fr'path/good_neurons_{brain_region}.csv')
            good_neurons = good_neurons.loc[good_neurons['goodneurons'].str.contains(sub, na=False)]  # Extract subject-specific neurons
            file_names = good_neurons.iloc[:, 0].astype(str)  # Convert to string in case of numeric values    
    
            if condition == 'Appear':
                mat_dir = os.path.join(f'path/Appear/{brain_region}/', f'{sub}')
            elif condition == 'Event':
                mat_dir = os.path.join(f'path/Event/{brain_region}/', f'{sub}')
            
            # Extract the percentage of high and low
            for i, file_name in enumerate(file_names):
                if file_name in badneuron_high and file_name in badneuron_low:
                    continue
                else:
                    if condition == 'Appear':
                        file_path = os.path.join(f'path/Appear/{brain_region}/', f'{sub}/{brain_region}_{sub}.xlsx')
                    elif condition == 'Event':
                        file_path = os.path.join(f'path/Event/{brain_region}/', f'{sub}/{brain_region}_{sub}.xlsx')
                        
                    behavior = pd.read_excel(file_path)   
                    outcome = behavior['outcome']
                    behavior = behavior[behavior_name]
                    
                    fr_mat_file_path = os.path.join(mat_dir, f'{brain_region}_{file_name}.mat')
                    fr_data = scipy.io.loadmat(fr_mat_file_path)
                    fr_values = fr_data['fr']
                    
                    st_mat_file_path = os.path.join(mat_dir, f'{brain_region}_{file_name}_spike_times.mat')     
                    st_data = scipy.io.loadmat(st_mat_file_path)
                    st_values = st_data['spike_times']
                    if condition == 'Event':
                        st_values = st_values + 0.67 # 670ms was added to firing rate to show the baseline
    
                    # Extract data in analysis window
                    time_axis = np.linspace(-2, 4, fr_values.shape[1])                                   
                    event_idx = (time_axis >= a_s) & (time_axis <= a_e) # Appear, or Event                                        
                    baseline_idx = (time_axis >=-2) & (time_axis <= -1.5) # define baseline as -2 to -1.5
                    
                    baseline_std = np.std(fr_values, axis=1, keepdims=True)
                    zero_rows = np.where(baseline_std == 0)[0]
                    fr_values = np.delete(fr_values, zero_rows, axis=0)
                    st_values = np.delete(st_values, zero_rows, axis=0)
                    baseline_std = np.delete(baseline_std, zero_rows, axis=0)
                    behavior = behavior.drop(index=zero_rows).reset_index(drop=True)
                    outcome = outcome.drop(index=zero_rows).reset_index(drop=True)                    
                            
                    # Plot Rasters (individual plot for each file_name)
                    plot_raster(trials=st_values, fr=fr_values, time_axis=np.linspace(-2, 4, fr_values.shape[1]),
                               brain_region=brain_region, file_name=file_name, folder_path=folder_path)
                    
                    # Split low and high trials based on quantile
                    low_threshold = behavior.quantile(low_percentile)
                    high_threshold = behavior.quantile(high_percentile)                              
                    
                    low_indices = behavior <= low_threshold
                    high_indices = behavior >= high_threshold
                    low_count = low_indices.sum()
                    high_count = high_indices.sum()        
                    
                    if low_count < 30 or high_count < 30:
                        continue
                    else:
                        low_fr_analysis = process_data(fr_values, np.where(low_indices)[0], baseline_idx)
                        high_fr_analysis = process_data(fr_values, np.where(high_indices)[0], baseline_idx)
                        p_low, labels_low = cbp(low_fr_analysis, n_permutations=10000, p_threshold=0.05) 
                        p_high, labels_high = cbp(high_fr_analysis, n_permutations=10000, p_threshold=0.05)
                        if file_name not in badneuron_high:
                            if np.any(np.array(p_high) < 0.05): # store neurons
                                hl_pa.append(np.mean(fr_values[np.where(low_indices)[0]], axis=0))
                                hh_pa.append(np.mean(fr_values[np.where(high_indices)[0]], axis=0))
                                h_pa_units.append(file_name)
                                generate_raster(st_values[low_indices], st_values[high_indices], labels_high, labels_low, behavior_name, 'High')
                                high_inds = np.where(high_indices)[0]                            
                                hh_pa_o.append(np.mean(fr_values[high_inds[outcome[high_indices] == 1]], axis=0)) # crash
                                hl_pa_o.append(np.mean(fr_values[high_inds[outcome[high_indices] == 0]], axis=0)) # avoidance
                        if file_name not in badneuron_low:                
                            if np.any(np.array(p_low) < 0.05):
                                ll_pa.append(np.mean(fr_values[np.where(low_indices)[0]], axis=0))
                                lh_pa.append(np.mean(fr_values[np.where(high_indices)[0]], axis=0))
                                l_pa_units.append(file_name)
                                generate_raster(st_values[low_indices], st_values[high_indices], labels_high, labels_low, behavior_name, 'Low')
                                low_inds = np.where(low_indices)[0]                            
                                lh_pa_o.append(np.mean(fr_values[low_inds[outcome[low_indices] == 1]], axis=0)) # crash
                                ll_pa_o.append(np.mean(fr_values[low_inds[outcome[low_indices] == 0]], axis=0))# avoidance    
                                
        #%% Save to Excel
        output_file = os.path.join(folder_path, f'Data/{brain_region}_{condition}_{behavior_name}.xlsx')
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:                        
            if len(hh_pa) > 0:
                df_high = pd.DataFrame({"Neuron": h_pa_units})
                df_high.to_excel(writer, sheet_name="High", index=False)        
            # Low significant neurons
            if len(ll_pa) > 0:
                df_low = pd.DataFrame({"Neuron": l_pa_units})
                df_low.to_excel(writer, sheet_name="Low", index=False)

        #%% Plots
        # High  
        if len(hh_pa) > 0:
            plot_condition(hh_pa, hl_pa, hh_pa_o, hl_pa_o, lh_pa_o, ll_pa_o, behavior_name, 'High', condition) if len(hh_pa) > 1 else None
        # Low
        if len(lh_pa) > 0:
            plot_condition(lh_pa, ll_pa, lh_pa_o, ll_pa_o, hh_pa, hl_pa_o, behavior_name, 'Low', condition) if len(lh_pa) > 1 else None
