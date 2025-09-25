"""
Computes permutation tests per neuron for Busters vs Pausers. Includes FDR
correction for multiple comparisons. Generates raster plots, line plots,
and heatmaps. Normalizes firing rates and plots all neurons sorted by event
response.
This code generates Extended Data Fig.:
    9a-e
    
Author: Rodrigo Dalvit
Data: 09/2025
"""
#%% Libraries
from IPython import get_ipython
get_ipython().magic('reset -sf')  # Reset environment (force, no confirmation)
get_ipython().magic('clear')      # Clear console

import gc
gc.collect()
import pandas as pd
import os
import numpy as np
import scipy.io
import scipy.stats as stats
from scipy.ndimage import uniform_filter1d
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (for saving plots only)
# matplotlib.use('TkAgg')  # Use GUI for interactive display
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['agg.path.chunksize'] = 10000  # Reduce memory usage
mpl.rcParams['svg.fonttype'] = 'none'  # keep text as text
mpl.rcParams['patch.linewidth'] = 0    # no default stroke

#%% Functions
def process_data(fr_data, baseline_idx):  
    """Subtract baseline firing rate"""
    baseline = np.mean(fr_data[:, baseline_idx], axis=1, keepdims=True)
    fr_data = fr_data - baseline
    return fr_data

def permutationTest(sample1, sample2, permutations, sidedness, exact=False):
    """
    Permutation test from:
    Laurens R Krol (2025). Permutation Test (https://github.com/lrkrol/permutationTest), GitHub. Retrieved March 20, 2025.        
    """        
    # p, observed_difference, effect_size = 10, 10, 10
    p, observed_difference = 10, 10    
    sample1, sample2 = np.asarray(sample1), np.asarray(sample2)
    observed_difference = np.nanmean(sample1) - np.nanmean(sample2)
    # pooled_std = np.sqrt(((len(sample1) - 1) * np.nanvar(sample1, ddof=1) + 
    #                       (len(sample2) - 1) * np.nanvar(sample2, ddof=1)) /
    #                      (len(sample1) + len(sample2) - 2))
    # effect_size = observed_difference / pooled_std    
    all_observations = np.concatenate([sample1, sample2])
    n1 = len(sample1)    
    if exact:
        all_combinations = list(combinations(range(len(all_observations)), n1))
        permutations = len(all_combinations)    
    random_differences = np.zeros(permutations)    
    for i in range(permutations):
        if exact:
            indices = np.array(all_combinations[i])
        else:
            # np.random.shuffle(all_observations)
            # indices = np.arange(n1)        
            indices = np.random.permutation(len(all_observations))[:n1]
        perm_sample1 = all_observations[indices]        
        perm_sample2 = np.delete(all_observations, indices)        
        random_differences[i] = np.nanmean(perm_sample1) - np.nanmean(perm_sample2)    
    if sidedness == 'both':
        p = (np.sum(np.abs(random_differences) >= abs(observed_difference)) + 1) / (permutations + 1)
    elif sidedness == 'smaller':
        p = (np.sum(random_differences <= observed_difference) + 1) / (permutations + 1)
    elif sidedness == 'larger':
        p = (np.sum(random_differences >= observed_difference) + 1) / (permutations + 1)
    else:
        raise ValueError("sidedness must be 'both', 'smaller', or 'larger'")       
    return p, observed_difference#, effect_size

def plot_raster(trials, fr, outcome, time_axis, brain_region,
                          file_name, folder_path, start_time=-2, end_time=4):
    """Plots raster and mean firing rate for all trials and outcomes."""
    # --- Common preprocessing ---
    raster = gen_raster(trials, start_time=start_time, end_time=end_time)
    baseline_mask = (time_axis >= 1.5) & (time_axis <= 2)
    fr = (fr - np.mean(fr[:, baseline_mask]))
    bin_size = int(400/50)  # assuming 50 ms bins
    # Smooth FR (all trials)
    fr_all_smooth = uniform_filter1d(np.mean(fr, axis=0), bin_size)
    # Trial indices for crash vs avoidance
    crash_idx = np.where(outcome == 1)[0]
    avoid_idx = np.where(outcome == 0)[0]
    # Smooth FR (by outcome)
    fr_crash_smooth = uniform_filter1d(np.mean(fr[crash_idx], axis=0), bin_size)
    fr_avoid_smooth = uniform_filter1d(np.mean(fr[avoid_idx], axis=0), bin_size)
    # --- Set up figure: 2 rows × 2 columns ---
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(2.18, 1.4), constrained_layout=True)
    # Raster (top left)
    ax_raster_all = axes[0, 0]
    raster_x, raster_y = [], []
    for i, trial_times in enumerate(raster):
        raster_x.extend(trial_times)
        raster_y.extend([i] * len(trial_times))
    ax_raster_all.scatter(raster_x, raster_y, s=.1, c='#22205F', marker='s')
    ax_raster_all.axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax_raster_all.axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax_raster_all.set_xlim(start_time, end_time)
    ax_raster_all.set_xticks([])
    ax_raster_all.set_ylabel("trial #")
    ax_raster_all.tick_params(axis='both', which='both', length=1, width=0.5)
    for spine in ax_raster_all.spines.values():
        spine.set_linewidth(0.3)
    ax_raster_all.spines['top'].set_visible(False)
    ax_raster_all.spines['right'].set_visible(False)
    ax_raster_all.spines['bottom'].set_visible(False)
    # Mean FR (bottom left)
    ax_fr_all = axes[1, 0]
    ax_fr_all.plot(time_axis, fr_all_smooth, color='#22205F', linewidth=0.5)
    ax_fr_all.axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax_fr_all.axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax_fr_all.set_xlim(start_time, end_time)
    ax_fr_all.set_xticks([])
    ax_fr_all.set_ylabel("mean $\Delta$ firing rate (Hz)")
    ax_fr_all.tick_params(axis='both', which='both', length=1, width=0.5)
    for spine in ax_fr_all.spines.values():
        spine.set_linewidth(0.3)
    ax_fr_all.spines['top'].set_visible(False)
    ax_fr_all.spines['right'].set_visible(False)
    # Raster (top right)
    ax_raster_outcome = axes[0, 1]
    for i, trial_times in enumerate(raster):
        if outcome[i] == 1:  # crash
            ax_raster_outcome.scatter(trial_times, [i] * len(trial_times),
                                      s=.1, c="#8F39E6", marker="s")
        else:  # avoidance
            ax_raster_outcome.scatter(trial_times, [i] * len(trial_times),
                                      s=.1, c="#00cc00", marker="s")
    ax_raster_outcome.axvline(0, color="red", linestyle="--", linewidth=0.5)
    ax_raster_outcome.axvline(0.67, color="gray", linestyle="--", linewidth=0.5)
    ax_raster_outcome.set_xlim(start_time, end_time)
    ax_raster_outcome.set_xticks([])
    ax_raster_outcome.tick_params(axis="both", which="both", length=1, width=0.5)
    for spine in ax_raster_outcome.spines.values():
        spine.set_linewidth(0.3)
    ax_raster_outcome.spines["top"].set_visible(False)
    ax_raster_outcome.spines["right"].set_visible(False)
    ax_raster_outcome.spines["bottom"].set_visible(False)
    # Mean FR (bottom right)
    ax_fr_outcome = axes[1, 1]
    ax_fr_outcome.plot(time_axis, fr_crash_smooth, color="#8F39E6", linewidth=0.5)
    ax_fr_outcome.plot(time_axis, fr_avoid_smooth, color="#00cc00", linewidth=0.5)
    ax_fr_outcome.axvline(0, color="red", linestyle="--", linewidth=0.5)
    ax_fr_outcome.axvline(0.67, color="gray", linestyle="--", linewidth=0.5)
    ax_fr_outcome.set_xlim(start_time, end_time)
    ax_fr_outcome.set_xticks([])
    ax_fr_outcome.tick_params(axis="both", which="both", length=1, width=0.5)
    for spine in ax_fr_outcome.spines.values():
        spine.set_linewidth(0.3)
    ax_fr_outcome.spines["top"].set_visible(False)
    ax_fr_outcome.spines["right"].set_visible(False)
    ax_fr_outcome.legend(loc="upper right", fontsize=4, frameon=False, handlelength=1)
    # --- Save figure ---
    fname = f"{brain_region}_{file_name}"
    save_dir = os.path.join(folder_path, "Rasters")
    fig.savefig(os.path.join(save_dir, f"{fname}.svg"), format="svg", dpi=300, transparent=True)
    fig.savefig(os.path.join(save_dir, f"{fname}.pdf"), format="pdf", transparent=True)
    plt.close(fig)

def gen_raster(trials, start_time=-2, end_time=4):
    """Generates raster data from trial spike times within a time window."""    
    raster = []
    for trial in trials:
        try:
            flat_trial = np.concatenate([np.ravel(np.array(t, dtype=float)) for t in trial])
            valid_times = flat_trial[(flat_trial >= start_time) & (flat_trial <= end_time)]
            raster.append(valid_times)
        except Exception:
            raster.append([])  # If any error occurs (e.g., empty or malformed trial), append empty list
    return raster
 
def filter_clusters(labels, min_bins=1):
    """Filters out clusters smaller than a minimum size."""        
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

def plot_busters_pausers(busters, busters_crash, busters_avoid, pausers, pausers_crash, pausers_avoid, condition):
    """Plots population activity of Busters vs Pausers across conditions."""
    color_bvsp = "#22205F"
    color_crash = "#8F39E6"
    color_avoid = "#00cc00"    
    # busters vs pausers
    busters = np.array(busters)
    pausers = np.array(pausers)
    # Preprocessing raw data
    time_axis = np.linspace(-2, 4, busters.shape[1])
    baseline_idx = (time_axis >= -2) & (time_axis <= -1.5)
    preappear_idx = (time_axis >= -0.5) & (time_axis <= -0.05)
    event_idx = []
    if condition == 'Appear':
        event_idx = (time_axis >= 0) & (time_axis <= 0.5)
    else:
        event_idx = (time_axis >= 0.67) & (time_axis <= 1.17)
    
    baseline_busters = busters[:,baseline_idx]
    baseline_pausers = pausers[:,baseline_idx]     
    busters_stat = busters - np.mean(baseline_busters, axis=1, keepdims=True)
    # Z-scoring    
    busters = (busters - np.mean(baseline_busters, axis=1, keepdims=True)) / np.where(np.std(baseline_busters, axis=1, keepdims=True) == 0, 1, np.std(baseline_busters, axis=1, keepdims=True))
    pausers = (pausers - np.mean(baseline_pausers, axis=1, keepdims=True)) / np.where(np.std(baseline_pausers, axis=1, keepdims=True) == 0, 1, np.std(baseline_pausers, axis=1, keepdims=True))
    # Convolution 
    bin_size = int(200/50) # (sliding windows size/bin size)                                                
    # using filter1d
    busters = uniform_filter1d(busters, bin_size, axis=1)
    pausers = uniform_filter1d(pausers, bin_size, axis=1)
    # bootstrap or sem
    ci_busters = sem_ci_2d(busters)
    ci_pausers = sem_ci_2d(pausers)    
    
    # Plot Busters vs Suppressors
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5    
    fig, ax = plt.subplots(3, 1, figsize=(0.95, 2.7))                   
    plt.subplots_adjust(hspace=0.25)  # Increase space between subplots    
    # Panel 1: Busters vs Pausers
    ax[0].plot(time_axis, np.mean(busters, axis=0), color=color_bvsp, linewidth = .5)
    ax[0].plot(time_axis, np.mean(pausers, axis=0), linestyle='--', color=color_bvsp, linewidth = .5)
    ax[0].fill_between(time_axis, ci_busters[0], ci_busters[1], color=color_bvsp, alpha=0.25, edgecolor='none')
    ax[0].fill_between(time_axis, ci_pausers[0], ci_pausers[1], color=color_bvsp, alpha=0.25, edgecolor='none')
    ax[0].axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax[0].axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax[0].set_xlim(-2, 4)
    ax[0].set_xticklabels([]) # hide x-axis labels
    ax[0].xaxis.set_ticks([]) # hide all ticks
    ax[0].grid(False)         # hide any grid lines
    ax[0].tick_params(axis='y', which='both', length=1, width=0.5)
    ax[0].set_title(f"Busters (n={busters.shape[0]}), Pausers (n={pausers.shape[0]})", fontsize=5)
    ax[0].tick_params(axis='both', which='both', length=1, width=0.5)       
    
    # avoid vs crash
    busters_avoid = np.array(busters_avoid)
    busters_crash = np.array(busters_crash)
    pausers_avoid = np.array(pausers_avoid)
    pausers_crash = np.array(pausers_crash)
    confidence_interval='sem_ci_2d'
    # Preprocessing raw data
    time_axis = np.linspace(-2, 4, busters.shape[1])
    baseline_idx = (time_axis >= -2) & (time_axis <= -1.5)
    baseline_busters_avoid = busters_avoid[:,baseline_idx]
    baseline_busters_crash = busters_crash[:,baseline_idx]
    baseline_pausers_avoid = pausers_avoid[:,baseline_idx]
    baseline_pausers_crash = pausers_crash[:,baseline_idx] 
    crash_stat = busters_crash - np.mean(baseline_busters_crash, axis=1, keepdims=True)
    avoid_stat = busters_avoid - np.mean(baseline_busters_avoid, axis=1, keepdims=True)
    # Z-scoring    
    busters_avoid = (busters_avoid - np.mean(baseline_busters_avoid, axis=1, keepdims=True)) / np.where(np.std(baseline_busters_avoid, axis=1, keepdims=True) == 0, 1, np.std(baseline_busters_avoid, axis=1, keepdims=True))
    busters_crash = (busters_crash - np.mean(baseline_busters_crash, axis=1, keepdims=True)) / np.where(np.std(baseline_busters_crash, axis=1, keepdims=True) == 0, 1, np.std(baseline_busters_crash, axis=1, keepdims=True))
    pausers_avoid = (pausers_avoid - np.mean(baseline_pausers_avoid, axis=1, keepdims=True)) / np.where(np.std(baseline_pausers_avoid, axis=1, keepdims=True) == 0, 1, np.std(baseline_pausers_avoid, axis=1, keepdims=True))
    pausers_crash = (pausers_crash - np.mean(baseline_pausers_crash, axis=1, keepdims=True)) / np.where(np.std(baseline_pausers_crash, axis=1, keepdims=True) == 0, 1, np.std(baseline_pausers_crash, axis=1, keepdims=True))

    # Convolution 
    bin_size = int(200/50) # (sliding windows size/bin size)                                                
    # using filter1d
    busters_avoid = uniform_filter1d(busters_avoid, bin_size, axis=1)
    busters_crash = uniform_filter1d(busters_crash, bin_size, axis=1)
    pausers_avoid = uniform_filter1d(pausers_avoid, bin_size, axis=1)
    pausers_crash = uniform_filter1d(pausers_crash, bin_size, axis=1)
    # bootstrap or sem
    ci_busters_avoid = sem_ci_2d(busters_avoid)
    ci_busters_crash = sem_ci_2d(busters_crash)
    ci_pausers_avoid = sem_ci_2d(pausers_avoid)
    ci_pausers_crash = sem_ci_2d(pausers_crash)
    # Panel 2: Crash vs Avoid
    ax[1].plot(time_axis, np.mean(busters_avoid, axis=0), color=color_avoid, linewidth=0.5)
    ax[1].plot(time_axis, np.mean(busters_crash, axis=0), color=color_crash, linewidth=0.5)
    ax[1].fill_between(time_axis, ci_busters_avoid[0], ci_busters_avoid[1], color=color_avoid, alpha=0.25, edgecolor='none')
    ax[1].fill_between(time_axis, ci_busters_crash[0], ci_busters_crash[1], color=color_crash, alpha=0.25, edgecolor='none')
    ax[1].plot(time_axis, np.mean(pausers_avoid, axis=0), linestyle='--', color=color_avoid, linewidth=0.5)
    ax[1].plot(time_axis, np.mean(pausers_crash, axis=0), linestyle='--', color=color_crash, linewidth=0.5)
    ax[1].fill_between(time_axis, ci_pausers_avoid[0], ci_pausers_avoid[1], color=color_avoid, alpha=0.25, edgecolor='none')
    ax[1].fill_between(time_axis, ci_pausers_crash[0], ci_pausers_crash[1], color=color_crash, alpha=0.25, edgecolor='none')
    ax[1].axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax[1].axvline(0.67, color='gray', linestyle='--', linewidth=0.5)
    ax[1].set_xlim(-2, 4)
    ax[1].set_xticks(np.arange(-2, 5, 2))    
    
    # Panel 3: Stats
    positions = [1, 2, 4, 5]
    _, p_busters = stats.wilcoxon(np.mean(busters_stat[:, event_idx], axis=1), alternative='two-sided')           
    # Scatter points for baseline and event
    ax[2].scatter(np.ones_like(np.mean(busters_stat[:, preappear_idx], axis=1))*positions[0], np.mean(busters_stat[:, preappear_idx], axis=1), 5, facecolors='white', edgecolors='black', marker='^', linewidths=0.125, zorder=2)
    ax[2].scatter(np.ones_like(np.mean(busters_stat[:, event_idx], axis=1))*positions[1], np.mean(busters_stat[:, event_idx], axis=1), 5, facecolors='white', edgecolors=color_bvsp, marker='^', linewidths=0.125, zorder=2)    
    # Connect baseline → event for each neuron
    for b, e in zip(np.mean(busters_stat[:, preappear_idx], axis=1), np.mean(busters_stat[:, event_idx], axis=1)):
        ax[2].plot([positions[0], positions[1]], [b, e], color='#cccccc', alpha=0.5, lw=0.3, zorder=1)    
    # Group means and connecting line
    ax[2].scatter(positions[0], np.mean(busters_stat[:, preappear_idx]), s=30, c='black', marker='_', linewidths=0.5, zorder=2)
    ax[2].scatter(positions[1], np.mean(busters_stat[:, event_idx]), s=30, c=color_bvsp, marker='_', linewidths=0.5, zorder=2)
    ax[2].plot([positions[0], positions[1]], [np.mean(busters_stat[:, preappear_idx]), np.mean(busters_stat[:, event_idx])], '-k', linewidth=0.5, zorder=1)   
    # Add stats text
    mean_post = np.mean(busters_stat[:, event_idx])
    sem_post = stats.sem(np.mean(busters_stat[:, event_idx], axis=1))
    mean_pre = np.mean(busters_stat[:, preappear_idx])
    sem_pre = stats.sem(np.mean(busters_stat[:, preappear_idx], axis=1))
    mean_sem_text = (
        f"{'*' if p_busters < 0.05 else 'ns'} (p={p_busters:.4f})\n"
        f"Pre: {mean_pre:.4f}±{sem_pre:.4f}\n "
        f"Post: {mean_post:.4f}±{sem_post:.4f}")
    ax[2].text(1.5, 0, mean_sem_text, ha='center', fontsize=5, linespacing=1.1) 
    
    # Avoid vs Crash
    _, p_avoidvscrash = stats.wilcoxon(np.mean(crash_stat[:, event_idx], axis=1), np.mean(avoid_stat[:, event_idx], axis=1), alternative='two-sided')           
    # Scatter points for baseline and event
    ax[2].scatter(np.ones_like(np.mean(avoid_stat[:, event_idx], axis=1))*positions[2], np.mean(avoid_stat[:, event_idx], axis=1), 5, facecolors='white', edgecolors=color_avoid, marker='^', linewidths=0.125, zorder=2)
    ax[2].scatter(np.ones_like(np.mean(crash_stat[:, event_idx], axis=1))*positions[3], np.mean(crash_stat[:, event_idx], axis=1), 5, facecolors='white', edgecolors=color_crash, marker='^', linewidths=0.125, zorder=2)    
    # Connect baseline → event for each neuron
    for b, e in zip(np.mean(avoid_stat[:, event_idx], axis=1), np.mean(crash_stat[:, event_idx], axis=1)):
        ax[2].plot([positions[2], positions[3]], [b, e], color='#cccccc', alpha=0.5, lw=0.3, zorder=1)    
    # Group means and connecting line
    ax[2].scatter(positions[2], np.mean(avoid_stat[:, event_idx]), s=30, c=color_avoid, marker='_', linewidths=0.5, zorder=2)
    ax[2].scatter(positions[3], np.mean(crash_stat[:, event_idx]), s=30, c=color_crash, marker='_', linewidths=0.5, zorder=2)
    ax[2].plot([positions[2], positions[3]], [np.mean(avoid_stat[:, event_idx]), np.mean(crash_stat[:, event_idx])], '-k', linewidth=0.5, zorder=1)   
    # Add stats text
    mean_crash = np.mean(crash_stat[:, event_idx])
    sem_crash = stats.sem(np.mean(crash_stat[:, event_idx], axis=1))
    mean_avoid = np.mean(avoid_stat[:, event_idx])
    sem_avoid = stats.sem(np.mean(avoid_stat[:, event_idx], axis=1))
    mean_sem_text = (
        f"{'*' if p_avoidvscrash < 0.05 else 'ns'} (p={p_avoidvscrash:.4f})\n"
        f"Crash: {mean_crash:.4f}±{sem_crash:.4f}\n "
        f"Avoid: {mean_avoid:.4f}±{sem_avoid:.4f}")
    ax[2].text(4.5, 0, mean_sem_text, ha='center', fontsize=5, linespacing=1.1) 

    # Update ticks
    ax[2].set_xticks([1, 2, 4, 5])
    ax[2].set_xticklabels(["Pre", "Post", "Avoid", "Crash"], fontsize=5)
    ax[2].set_ylabel("mean Δ rate")

    for i in range(len(ax)):  # Loop through both subplots
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
    fig.savefig(os.path.join(folder_path, f'{brain_region}_{condition}.svg'), dpi=300, bbox_inches='tight', format='svg')
    fig.savefig(os.path.join(folder_path, f'{brain_region}_{condition}.pdf'), dpi=300, bbox_inches='tight', format='pdf')
    plt.close(fig)
    gc.collect()

def normalize_fr(fr):
    """Normalizes firing rates to 0–1 for each neuron."""
    return (fr - np.min(fr, axis=1, keepdims=True)) / (
        np.max(fr, axis=1, keepdims=True) - np.min(fr, axis=1, keepdims=True) + 1e-10)

def plot_heatmap_square(fr, title, ylabel, brain_region, condition):
    """Plots a heatmap of firing rates with event markers."""
    n_neurons, n_timebins = fr.shape    
    # Fixed figure width in inches -> same x-axis length for all plots
    fig_width = 3    
    # Size of one bin in inches (square pixels enforced)
    bin_size = fig_width / n_timebins     
    # Compute height from number of neurons
    fig_height = bin_size * n_neurons     
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))    
    # Plot with square pixels
    im = ax.imshow(fr, cmap='inferno', interpolation='nearest',
                   aspect='equal', extent=[0, n_timebins, 0, n_neurons])    
    # Stimulus/event line
    ax.axvline(x=stim_idx, color='cyan', linestyle='--', lw=0.5)    
    # Ticks
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_vals, fontsize=5, fontname='Arial')    
    ax.set_yticks(np.arange(n_neurons))
    ax.set_yticklabels(np.arange(1, n_neurons+1), fontsize=5, fontname='Arial')    
    # Labels
    ax.set_xlabel('Time (s)', fontsize=5, fontname='Arial')
    ax.set_ylabel(ylabel, fontsize=5, fontname='Arial')
    ax.set_title(title, fontsize=5, fontname='Arial')    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, label='normalized firing rate')
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.yaxis.label.set_fontsize(5)
    cbar.ax.yaxis.label.set_fontname('Arial')    
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)    
    # Save
    os.makedirs(folder_path, exist_ok=True)
    fig.savefig(os.path.join(folder_path, f'{brain_region}_{condition}_{title}_heatmap.svg'),
                dpi=300, bbox_inches='tight', format='svg')
    fig.savefig(os.path.join(folder_path, f'{brain_region}_{condition}_{title}_heatmap.pdf'),
                dpi=300, bbox_inches='tight', format='pdf')
    plt.close(fig)
    
#%% Main Analysis
brain_region = 'AMY'
brain_regions = {'AMY': ['sub016A', 'sub016B', 'sub024A']}
subjects = brain_regions.get(brain_region, [])
folder_path = os.path.join(f'path/{brain_region}/Population_permutation_new/')

# Check if the folder exists, and if not, create it
for sub in ['Data', 'Rasters']:
    os.makedirs(os.path.join(folder_path, sub), exist_ok=True)

for condition, (a_s, a_e) in zip(['Appear'], [(0, 0.5)]):
    unit_id, fr_heatmap = [], []
    busters, busters_units, busters_crash, busters_avoid, p_busters = [], [], [], [], []
    pausers, pausers_units, pausers_crash, pausers_avoid, p_pausers = [], [], [], [], []
    for sub in subjects:
       good_neurons = pd.read_csv(fr'path/good_neurons_{brain_region}.csv')
       good_neurons = good_neurons.loc[good_neurons['goodneurons'].str.contains(sub, na=False)]
       file_names = good_neurons.iloc[:, 0].astype(str)
       mat_dir = os.path.join(f'path/Appear/{brain_region}/', f'{sub}')

       for i, file_name in enumerate(file_names):
           file_path = os.path.join(f'path/Appear/{brain_region}/', f'{sub}/{brain_region}_{sub}.xlsx')

           behavior = pd.read_excel(file_path)   
           outcome = behavior['outcome']
           
           fr_mat_file_path = os.path.join(mat_dir, f'{brain_region}_{file_name}.mat')
           fr_data = scipy.io.loadmat(fr_mat_file_path)
           fr_values = fr_data['fr']
           
           st_mat_file_path = os.path.join(mat_dir, f'{brain_region}_{file_name}_spike_times.mat')     
           st_data = scipy.io.loadmat(st_mat_file_path)
           st_values = st_data['spike_times']
           if condition == 'Event':
               st_values = st_values + 0.67

           # Extract data in analysis window
           time_axis = np.linspace(-2, 4, fr_values.shape[1])                                   
           event_idx = (time_axis >= a_s) & (time_axis <= a_e)
           baseline_idx = (time_axis >=-2) & (time_axis <= -1.5)
           
           baseline_std = np.std(fr_values, axis=1, keepdims=True)
           zero_rows = np.where(baseline_std == 0)[0]
           fr_values = np.delete(fr_values, zero_rows, axis=0)
           st_values = np.delete(st_values, zero_rows, axis=0)
           baseline_std = np.delete(baseline_std, zero_rows, axis=0)
           outcome = outcome.drop(index=zero_rows).reset_index(drop=True)                    
                   
           # Plot Rasters (individual plot for each file_name)
           plot_raster(trials=st_values, fr=fr_values, outcome=outcome, time_axis=np.linspace(-2, 4, fr_values.shape[1]),
                      brain_region=brain_region, file_name=file_name, folder_path=folder_path)  
           
           fr_analysis = process_data(fr_values, baseline_idx)
           p_buster, _ = permutationTest(np.mean(fr_analysis[:, event_idx],axis=1), np.mean(fr_analysis[:, baseline_idx],axis=1), permutations=10000, sidedness='larger')
           p_pauser, _ = permutationTest(np.mean(fr_analysis[:, event_idx],axis=1), np.mean(fr_analysis[:, baseline_idx],axis=1), permutations=10000, sidedness='smaller')           
           p_busters.append(p_buster)
           p_pausers.append(p_pauser)
           unit_id.append(file_name)
           fr_heatmap.append(np.mean(fr_values, axis=0)) 
           if np.any(np.array(p_buster) < 0.05): # store neurons               
               busters.append(np.mean(fr_values, axis=0))                
               busters_units.append(file_name)                                  
               busters_crash.append(np.mean(fr_values[outcome == 1], axis=0)) # crash
               busters_avoid.append(np.mean(fr_values[outcome == 0], axis=0)) # avoidance
           if np.any(np.array(p_pauser) < 0.05):               
               pausers.append(np.mean(fr_values, axis=0))                
               pausers_units.append(file_name)                                  
               pausers_crash.append(np.mean(fr_values[outcome == 1], axis=0)) # crash
               pausers_avoid.append(np.mean(fr_values[outcome == 0], axis=0)) # avoidance 

    #%% FDR correction
    # Apply multiple-comparison correction (FDR) to permutation test p-values    
    rej_busters, p_busters_corrected, _, _ = multipletests(p_busters, alpha=0.05, method='fdr_bh')
    rej_pausers, p_pausers_corrected, _, _ = multipletests(p_pausers, alpha=0.05, method='fdr_bh')        
    units_busters = np.array(unit_id)[rej_busters]
    units_pausers = np.array(unit_id)[rej_pausers]
    mask_busters = np.array([unit in units_busters for unit in busters_units])
    mask_pausers = np.array([unit in units_pausers for unit in pausers_units])
    # apply mask to busters and pausers
    busters = list(np.array(busters)[mask_busters])
    pausers = list(np.array(pausers)[mask_pausers])
    busters_units = list(np.array(busters_units)[mask_busters])
    pausers_units = list(np.array(pausers_units)[mask_pausers])
    busters_crash = list(np.array( busters_crash)[mask_busters])
    busters_avoid = list(np.array(busters_avoid)[mask_busters])   
    pausers_crash = list(np.array( pausers_crash)[mask_pausers])
    pausers_avoid = list(np.array(pausers_avoid)[mask_pausers])                         
    
    #%% Save to Excel
    output_file = os.path.join(folder_path, f'Data/{brain_region}_{condition}.xlsx')
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:                        
        if len(busters) > 0:
            df_busters = pd.DataFrame({"Neuron": busters_units})
            df_busters.to_excel(writer, sheet_name="busters", index=False)        
        # pausers significant neurons
        if len(pausers) > 0:
            df_pausers = pd.DataFrame({"Neuron": pausers_units})
            df_pausers.to_excel(writer, sheet_name="pausers", index=False)

    #%% Plots
    # Busters vs Pausers
    plot_busters_pausers(busters, busters_crash, busters_avoid, pausers, pausers_crash, pausers_avoid, condition)
    
    # Heatmap
    # Normalize
    fr_all = normalize_fr(fr_heatmap)
    fr_bp = normalize_fr(np.vstack([busters, pausers]))      
    mean_window = np.mean(fr_all[:, event_idx], axis=1)
    sorted_idx = np.argsort(-mean_window)
    fr_all_sorted = fr_all[sorted_idx, :]    
    # X-axis ticks
    tick_vals = [-2, -1, 0, 1, 2, 4]
    tick_idx = [np.argmin(np.abs(time_axis - t)) for t in tick_vals]
    stim_idx = np.argmin(np.abs(time_axis - 0))
    
    # Plot both heatmaps
    plot_heatmap_square(fr_all_sorted, 'All neurons sorted', 'Neurons (sorted)', brain_region, condition)
    plot_heatmap_square(fr_bp, 'Busters + Pausers', 'Neurons (Busters + Pausers)', brain_region, condition)