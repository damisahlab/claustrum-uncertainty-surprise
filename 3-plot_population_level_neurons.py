# plot_population_level_neurons.py
"""
Description: This script analyzes and visualizes neuronal firing rates in response to cognitive features at the population level. 
             Specifically, it processes significant neurons' data for both low and high conditions, computes confidence intervals, 
             performs z-scoring, and generates plots to illustrate the differences between these conditions. 
             The analyzed conditions include 'PreAppear', 'Appear', and 'Event', with results saved in specified output folders.

Created: Feb 18, 2025
Author: Rodrigo Dalvit
"""

#%% Libraries
import os
import gc
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats as stats
from scipy.ndimage import uniform_filter1d
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple
# Ensure matplotlib uses 'TkAgg' backend
matplotlib.use('TkAgg')
# Configure plotting for high-resolution
plt.rcParams['agg.path.chunksize'] = 10000

#%% Functions
def sem_ci_2d(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean_vals = np.mean(data, axis=0)
    sem_vals = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
    low_bound = mean_vals - sem_vals
    high_bound = mean_vals + sem_vals
    return low_bound, high_bound

def plot_condition(ax: plt.Axes, time_axis: np.ndarray, all_high: np.ndarray, all_low: np.ndarray,
                   event_idx: np.ndarray, column: str, condition_label: str, p_threshold: float) -> float:
    baseline_idx = (time_axis >= -2) & (time_axis <= -1.5)
    baseline_low = all_low[:, baseline_idx]
    baseline_high = all_high[:, baseline_idx]
    
    high = np.mean(all_high[:, event_idx], axis=1) - np.mean(baseline_high, axis=1)
    low = np.mean(all_low[:, event_idx], axis=1) - np.mean(baseline_low, axis=1)
    _, p_vs = stats.wilcoxon(high, low, alternative='two-sided', method='exact')
    
    baseline_std_high = np.std(baseline_high, axis=1, keepdims=True)
    baseline_std_low = np.std(baseline_low, axis=1, keepdims=True)
    baseline_std_high[baseline_std_high == 0] = 1
    baseline_std_low[baseline_std_low == 0] = 1    
    all_high = (all_high - np.mean(baseline_high, axis=1, keepdims=True)) / baseline_std_high
    all_low = (all_low - np.mean(baseline_low, axis=1, keepdims=True)) / baseline_std_low
    
    bin_size = int(400/ 50)
    all_high = uniform_filter1d(all_high, bin_size, axis=1)
    all_low = uniform_filter1d(all_low, bin_size, axis=1)
    
    ci_high = sem_ci_2d(all_high)
    ci_low = sem_ci_2d(all_low)
    
    ax[0].plot(time_axis, np.mean(all_high, axis=0), color='orange', label='High', linewidth=.5)
    ax[0].plot(time_axis, np.mean(all_low, axis=0), color='blue', label='Low', linewidth=.5)
    ax[0].fill_between(time_axis, ci_high[0], ci_high[1], color='orange', alpha=0.25)
    ax[0].fill_between(time_axis, ci_low[0], ci_low[1], color='blue', alpha=0.25)
    ax[0].axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax[0].axvline(0.67, color='red', linestyle='-', linewidth=0.5)
    ax[0].set_xlim(-2, 4)
    ax[0].set_xticks(np.arange(-2, 5, 2))
    ax[0].tick_params(axis='both', which='both', length=1, width=0.5)

    ax[1].scatter(np.ones(len(low)), low, 5, facecolors='none', edgecolors='blue', marker='^', label='High Low', linewidths=0.25)
    ax[1].scatter(2 * np.ones(len(high)), high, 5, facecolors='none', edgecolors='orange', marker='^', label='High Hig', linewidths=0.25)
    ax[1].scatter(1, np.mean(low), 6, 'k', marker='^', label='Mean High Low', edgecolors='k', linewidths=0.2)
    ax[1].scatter(2, np.mean(high), 6, 'k', marker='v', label='Mean High Hig', edgecolors='k', linewidths=0.2)
    ax[1].set_xlim([0, 3])
    ax[1].set_xticks([1, 2])
    yl = [min(np.min(low), np.min(high)), max(np.max(low), np.max(high))]
    m = yl[1] - yl[0]
    ax[1].set_ylim([yl[0] - .05 * m, yl[1] + m * 0.25])
    ax[1].plot([1, 2], [low, high], '-k', linewidth=0.075)
    ax[1].plot([1, 2], [np.mean(low), np.mean(high)], '-k', linewidth=0.2)
    ax[1].tick_params(axis='both', which='both', length=1, width=0.5)
    
    asterisk_y = yl[1] + m * 0.125
    bar_y = asterisk_y + m * 0.07
    if p_vs < p_threshold:
        ax[1].text(1.5, bar_y, f'(p={p_vs:.4f})', ha='center')
    else:
        ax[1].text(1.5, bar_y, f'ns (p={p_vs:.4f})', ha='center')
    for i in range(2):
        for spine in ax[i].spines.values():
            spine.set_linewidth(0.3)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].tick_params(axis='both', which='both', length=1, width=0.5, pad=1)
        ax[i].set_xlabel(ax[i].get_xlabel(), labelpad=1)
        ax[i].set_ylabel(ax[i].get_ylabel(), labelpad=1)
    return p_vs

def create_output_folders(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(os.path.join(folder_path, 'Appear'))
        os.makedirs(os.path.join(folder_path, 'PreAppear'))
        os.makedirs(os.path.join(folder_path, 'Event'))
        os.makedirs(os.path.join(folder_path, 'Data'))

def load_neuron_data(brain_region: str, subjects: List[str], low_p: float, condition: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    good_neurons = pd.concat([
        pd.read_csv(f'path/{brain_region}/p_values_{brain_region}_{sub}_{low_p}_{condition}.csv')
        for sub in subjects], ignore_index=True) # path to p-value files (.csv) generated from permutation_analysis.py
    
    effect_size = pd.concat([
        pd.read_csv(f'path/{brain_region}/effect_sizes_{brain_region}_{sub}_{low_p}_{condition}.csv')
        for sub in subjects], ignore_index=True) # path to effect-size files (.csv) generated from permutation_analysis.py
    
    good_neurons = good_neurons.rename(columns={good_neurons.columns[0]: 'ID'})
    good_neurons[['sub', 'channel', 'unit']] = good_neurons['ID'].str.split('_', expand=True)
    good_neurons = good_neurons[['ID', 'sub', 'channel', 'unit'] + [col for col in good_neurons.columns if col not in ['ID', 'sub', 'channel', 'unit']]]

    effect_size = effect_size.rename(columns={effect_size.columns[0]: 'ID'})
    effect_size[['sub', 'channel', 'unit']] = effect_size['ID'].str.split('_', expand=True)
    effect_size = effect_size[['ID', 'sub', 'channel', 'unit'] + [col for col in effect_size.columns if col not in ['ID', 'sub', 'channel', 'unit']]]
    return good_neurons, effect_size

def analyze_condition(brain_region: str, condition: str, a_s: float, a_e: float, subjects: List[str], low_p: float,
                      high_p: float, p_value: float, es_thresh: float, sliding_window: int, folder_path: str) -> None:
    good_neurons, effect_size = load_neuron_data(brain_region, subjects, low_p, condition)
    base_dir = f'Input/{condition}/{brain_region}/' # Path to Data
    plt.ioff()
    output_file = os.path.join(folder_path, f'{brain_region}_{low_p}_{condition}_neuron_categories.xlsx')

    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        number_neurons = np.zeros((len(good_neurons.columns[4:]), 2))
        for idx, column in enumerate(good_neurons.columns[4:]):
            sig_neurons = good_neurons[good_neurons[column] < p_value]
            sig_neurons = sig_neurons.merge(effect_size[['ID', column]], on='ID', how='left')
            sig_neurons.rename(columns={sig_neurons.columns[-1]: 'effect_size'}, inplace=True)            
            if sig_neurons.empty:
                continue
            all_high_low, all_high_high, all_low_low, all_low_high = [], [], [], []
            event_idx, time_axis = None, None
            for _, row in sig_neurons.iterrows():
                fr_values = scipy.io.loadmat(os.path.join(base_dir, f'{row["sub"]}/{brain_region}_{row["ID"]}.mat'))['fr']
                behavior = pd.read_excel(os.path.join(base_dir, f'{row["sub"]}/{brain_region}_{row["sub"]}.xlsx'))
                time_axis = np.linspace(-2, 4, fr_values.shape[1])
                event_idx = (time_axis >= a_s) & (time_axis <= a_e)

                low_idx = behavior[behavior[column] <= behavior[column].quantile(low_p)].index
                high_idx = behavior[behavior[column] >= behavior[column].quantile(high_p)].index
                low_values = fr_values[low_idx]
                high_values = fr_values[high_idx]
                if row['effect_size'] > 0:
                    all_high_low.append(np.mean(low_values, axis=0))
                    all_high_high.append(np.mean(high_values, axis=0))
                else:
                    all_low_low.append(np.mean(low_values, axis=0))
                    all_low_high.append(np.mean(high_values, axis=0))

            all_high_high = np.vstack(all_high_high) if all_high_high else np.zeros((0, fr_values.shape[1]))
            all_high_low = np.vstack(all_high_low) if all_high_low else np.zeros((0, fr_values.shape[1]))
            all_low_high = np.vstack(all_low_high) if all_low_high else np.zeros((0, fr_values.shape[1]))
            all_low_low = np.vstack(all_low_low) if all_low_low else np.zeros((0, fr_values.shape[1]))

            number_neurons[idx, 0] = all_high_high.shape[0]
            number_neurons[idx, 1] = all_low_high.shape[0]

            df_high_high = pd.DataFrame(all_high_high)
            df_high_low = pd.DataFrame(all_high_low)
            df_low_high = pd.DataFrame(all_low_high)
            df_low_low = pd.DataFrame(all_low_low)

            data_output_path = os.path.join(folder_path, f'Data/{brain_region}_{condition}_{column}.xlsx')
            with pd.ExcelWriter(data_output_path, engine='xlsxwriter') as data_writer:
                df_high_high.to_excel(data_writer, sheet_name="High-High", index=False)
                df_high_low.to_excel(data_writer, sheet_name="High-Low", index=False)
                df_low_high.to_excel(data_writer, sheet_name="Low-High", index=False)
                df_low_low.to_excel(data_writer, sheet_name="Low-Low", index=False)

            plot_and_save_results(all_high_high, all_high_low, event_idx, time_axis, p_value, folder_path, condition,
                                  brain_region, column, low_p, 'High')
            plot_and_save_results(all_low_high, all_low_low, event_idx, time_axis, p_value, folder_path, condition,
                                  brain_region, column, low_p, 'Low')

        number_neurons_df = pd.DataFrame(number_neurons, columns=['High', 'Low'], index=good_neurons.columns[4:])
        number_neurons_df.to_csv(os.path.join(folder_path, f'number_neurons_{brain_region}_{low_p}_{condition}.csv'))
    plt.ion()

def plot_and_save_results(all_high: np.ndarray, all_low: np.ndarray, event_idx: np.ndarray, time_axis: np.ndarray,
                          p_value: float, folder_path: str, condition: str, brain_region: str, column: str, low_p: float,
                          label: str) -> None:
    if not (all_high.size or all_low.size):
        return
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5
    fig, ax = plt.subplots(2, 1, figsize=(.975, 1.575))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(column, fontsize=5)

    p_v = plot_condition(ax, time_axis, all_high, all_low, event_idx, column, label, p_value)
    plt.tight_layout()
    if p_v < 0.05:
        fig.savefig(os.path.join(folder_path, condition, f'{brain_region}_{column}_{low_p}_{condition}_{label}.svg'), dpi=300, bbox_inches='tight', format='svg')
        fig.savefig(os.path.join(folder_path, condition, f'{brain_region}_{column}_{low_p}_{condition}_{label}.pdf'), dpi=300, bbox_inches='tight', format='pdf')
    else:
        fig.savefig(os.path.join(folder_path, condition, 'ns', f'{brain_region}_{column}_{low_p}_{condition}_{label}.pdf'), dpi=300, bbox_inches='tight', format='pdf')

    plt.close(fig)
    gc.collect()

#%% Main
def main():
    # Parameters
    brain_region = 'ACC'
    low_p = 0.3
    high_p = 1 - low_p
    p_value = 0.05
    es_thresh = 0.225
    sliding_window = 400

    brain_regions = {'ACC': ['sub016A', 'sub016B', 'sub019A', 'sub019B', 'sub020', 'sub023', 'sub024A'],
                     'CLA': ['sub016A', 'sub016B', 'sub017', 'sub024A']}
    subjects = brain_regions.get(brain_region, [])

    folder_name = f'{low_p}'
    folder_path = os.path.join('Output', brain_region, 'Population', folder_name) # Path to Output
    create_output_folders(folder_path)

    conditions = [{'name': 'PreAppear', 'start': -1, 'end': 0}, {'name': 'Appear', 'start': 0, 'end': 0.4}, {'name': 'Event', 'start': 0.5, 'end': 1}]
    for condition in conditions:
        analyze_condition(brain_region, condition['name'], condition['start'], condition['end'], subjects,low_p,
                          high_p, p_value, es_thresh, sliding_window, folder_path)
if __name__ == "__main__":
    main()
