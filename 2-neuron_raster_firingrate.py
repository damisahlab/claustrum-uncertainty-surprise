#%% neuron_raster_firingrate.py (Single-Neuron Raster and Mean Firing Rate Plots by Behavioral Condition (Low vs High))
"""
Description: Neural spike analysis: Raster plots and firing rate comparisons (low vs. high behavioral condition)
             For each significant neuron (filtered by p-value threshold), this script generates:
                 - A raster plot of spike times
                 - A mean firing rate plot with bootstrap confidence intervals
             Data is separated by brain region (e.g., ACC, CLA), behavioral phase (Appear, PreAppear, Event), and significance threshold.

Created on Feb 18, 2025
@author: Rodrigo Dalvit"""

# %% Libraries
import os
import gc
import numpy as np
import pandas as pd
import scipy.io
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#%% Functions
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

def generate_raster_data(trial_spike_times, start_time=-2, end_time=4):
    """Generate raster plot data efficiently."""
    trial_lengths = [len(trial[(trial >= start_time) & (trial <= end_time)]) for trial in trial_spike_times]
    total_spikes = sum(trial_lengths)

    raster_x = np.zeros(total_spikes)
    raster_y = np.zeros(total_spikes)
    raster_colors = np.empty(total_spikes, dtype=object)

    counter = 0
    for t, trial in enumerate(trial_spike_times):
        valid_times = trial[(trial >= start_time) & (trial <= end_time)]
        num_spikes = len(valid_times)
        
        raster_x[counter:counter + num_spikes] = valid_times
        raster_y[counter:counter + num_spikes] = t
        raster_colors[counter:counter + num_spikes] = 'orange' if t < len(trial_spike_times) // 2 else 'blue'
        counter += num_spikes

    return raster_x, raster_y, raster_colors

# %% Parameters
brain_region = 'CLA'
p_value = 0.05
low_quantile = 0.3
high_quantile = 1 - low_quantile
sliding_window = 400

base_dir = f'Input' # Path to Dataset
folder_path = f'Output' # Path to Output
 
for subfolder in ['Appear', 'PreAppear', 'Event']:
    os.makedirs(os.path.join(folder_path, subfolder), exist_ok=True)
    
brain_regions = {
    'CLA': ['sub016A', 'sub016B', 'sub017', 'sub024A'],
    'ACC': ['sub016A', 'sub016B', 'sub019A', 'sub019B', 'sub020', 'sub023', 'sub024A']
}
subjects = brain_regions.get(brain_region, [])

#%% Main Loop
for subject in subjects:    
    for condition, (a_s, a_e) in zip(['Appear', 'PreAppear', 'Event'], [(0, 0.4), (-1, 0), (0.5, 1)]):                  
        pvalue_file = 'p_value_data' # path to p-value files (.csv) generated from permutation_analysis.py
        good_neurons = pd.read_csv(pvalue_file)
        good_neurons = good_neurons.rename(columns={good_neurons.columns[0]: 'ID'})
        good_neurons[['sub', 'channel', 'unit']] = good_neurons.iloc[:, 0].str.split('_', expand=True)
        columns_order = ['ID', 'sub', 'channel', 'unit'] + [col for col in good_neurons.columns if col not in ['ID', 'sub', 'channel', 'unit']]
        good_neurons = good_neurons[columns_order]
        columns_to_keep = ["ID","sub", "channel", "unit", "A_safety_value", "B_safety_value",
                           "A_safety_value_raw", "B_safety_value_raw", "A_safety_variance", "B_safety_variance",
                           "A_prediction_error", "B_prediction_error", "A_absolute_prediction_error", "B_absolute_prediction_error"]
        good_neurons = good_neurons[columns_to_keep]
        file_names = good_neurons.ID.astype(str)
        sub = good_neurons["sub"]        
        
        for i,file_name in enumerate(file_names):
            behavior_file = os.path.join(base_dir, f'{sub[i]}/{brain_region}_{sub[i]}.xlsx')
            behavior = pd.read_excel(behavior_file)            
            fr_mat_file_path = os.path.join(base_dir, f'{sub[i]}/{brain_region}_{file_name}.mat') 
            st_mat_file_path = os.path.join(base_dir, f'{sub[i]}/{brain_region}_{file_name}_spike_times.mat')     
        
            fr_data = scipy.io.loadmat(fr_mat_file_path)
            st_data = scipy.io.loadmat(st_mat_file_path)
            fr_values = fr_data['fr']
            st_values = st_data['spike_times']
            
            time_axis = np.linspace(-2, 4, fr_values.shape[1])
            baseline_std = np.std(fr_values, axis=1, keepdims=True)
            zero_rows = np.where(baseline_std == 0)[0]            
            fr_values = np.delete(fr_values, zero_rows, axis=0)
            st_values = np.delete(st_values, zero_rows, axis=0)
            col_names = behavior.columns 
            behavior = np.delete(behavior, zero_rows, axis=0)         
            behavior = pd.DataFrame(behavior, columns=col_names)
            
            low_indices = {col: behavior[behavior[col] <= behavior[col].quantile(low_quantile)].index.to_numpy() for col in behavior.columns[3:]}
            high_indices = {col: behavior[behavior[col] >= behavior[col].quantile(high_quantile)].index.to_numpy() for col in behavior.columns[3:]}
            
            for j,column in enumerate(behavior.columns[3:]):
                if good_neurons.iloc[i,j+4] < p_value:
                    plt.close()
                    plt.ioff()
                    plt.rcParams["font.family"] = "Arial"
                    plt.rcParams["font.size"] = 5                    
                    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.2, 1.6), constrained_layout=True)
                    plt.subplots_adjust(hspace=0.5)
                    fig.suptitle(column, fontsize=5)                    
                    ax_raster = axes[0]
                    ax_fr = axes[1]
        
                    low_idx, high_idx = low_indices[column], high_indices[column]
                    low_st_data, high_st_data = clean_st_values(st_values[low_idx]), clean_st_values(st_values[high_idx])
                    
                    raster_x, raster_y, raster_colors = generate_raster_data(high_st_data + low_st_data)
                    ax_raster.scatter(raster_x, raster_y, s=.0001, c=raster_colors, alpha=0.7, marker='o')
                    ax_raster.axvline(0, color='red', linestyle='--', linewidth=0.5)                    
                    ax_raster.axvline(0.67, color='red', linestyle='-', linewidth=0.5)
                    ax_raster.set_xticks([])
                    ax_raster.set_xlim(-2, 4)                    
                    ax_raster.set_ylabel("trial #")
                    ax_raster.tick_params(axis='both', which='both', length=1, width=0.5)
                    for spine_name, spine in ax_raster.spines.items():
                        spine.set_linewidth(0.5)
                        if spine_name in ['top', 'right', 'bottom']:
                            spine.set_visible(False)
    
                    time_vector = np.linspace(-2, 4, fr_values.shape[1])
                    baseline_idx = (time_vector >=-2) & (time_vector <= -1.5)
                    
                    low_fr = fr_values[low_idx]
                    high_fr = fr_values[high_idx]
                    low_fr = low_fr - np.mean(low_fr[:,baseline_idx], axis=1, keepdims=True)
                    high_fr = high_fr - np.mean(high_fr[:,baseline_idx], axis=1, keepdims=True)   
                    
                    low_fr, high_fr = np.mean(low_fr, axis=0), np.mean(high_fr, axis=0)
                    bin_size = int(sliding_window/50)                                                                                              
                    high_fr = uniform_filter1d(high_fr, bin_size, axis=0)
                    low_fr = uniform_filter1d(low_fr, bin_size, axis=0)
                    
                    ax_fr.plot(time_axis, high_fr, color='orange', label="High", linewidth=0.5)
                    ax_fr.plot(time_axis, low_fr, color='blue', label="Low", linewidth=0.5)               
                    ax_fr.axvline(0, color='red', linestyle='--', linewidth=0.5)
                    ax_fr.axvline(0.67, color='red', linestyle='-', linewidth=0.5)
                    ax_fr.set_xlim(-2, 4)
                    ax_fr.set_xticks(np.arange(-2, 5, 2))
                    ax_fr.set_xlabel("time (s)")
                    ax_fr.set_ylabel("mean $\Delta$ firing rate (Hz)")                    
                    ax_fr.tick_params(axis='both', which='both', length=1, width=0.5)
                    for spine in ax_fr.spines.values():
                        spine.set_linewidth(0.5)
                    ax_fr.spines['top'].set_visible(False)
                    ax_fr.spines['right'].set_visible(False)
                    
                    fig.savefig(os.path.join(folder_path, condition, f'{brain_region}_{file_name}_{column}_{low_quantile}_{condition}.svg'), format='svg')
                    fig.savefig(os.path.join(folder_path, condition, f'{brain_region}_{file_name}_{column}_{low_quantile}_{condition}.pdf'), format='pdf')
                    plt.close(fig)
                    gc.collect()
plt.ion()
gc.collect()
