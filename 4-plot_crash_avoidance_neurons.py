# analyze_hit_vs_miss_conditions.py
"""
Description: This script analyzes and visualizes neuronal firing rates in response to cognitive features, focusing
             on hit vs. miss conditions at the population level. 
             It processes significant neurons' data, computes confidence intervals, performs z-scoring, and generates
             plots to illustrate the differences between these conditions. The analyzed conditions include 
             'PreAppear', 'Appear', and 'Event'. Results are saved in specified output folders.

Created: Mar 03, 2025
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['agg.path.chunksize'] = 10000

#%% Functions
def sem_ci_2d(data):
    mean_vals = np.mean(data, axis=0)
    sem_vals = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
    low_bound = mean_vals - sem_vals
    high_bound = mean_vals + sem_vals
    return low_bound, high_bound

def plot_condition(ax, time_axis, all_hit, all_miss, event_idx, neurons, column, condition_label, p_threshold):
    hit_color = "#8F39E6"
    miss_color = "#45B766"
    
    baseline_idx = (time_axis >= -2) & (time_axis <= -1.5)
    baseline_miss = all_miss[:, baseline_idx]
    baseline_hit = all_hit[:, baseline_idx]

    hit = np.mean(all_hit[:, event_idx], axis=1) - np.mean(baseline_hit, axis=1)
    miss = np.mean(all_miss[:, event_idx], axis=1) - np.mean(baseline_miss, axis=1)

    _, p_vs = stats.wilcoxon(hit, miss, alternative='two-sided', method='exact')

    all_hit = (all_hit - np.mean(baseline_hit, axis=1, keepdims=True)) / np.where(
        np.std(baseline_hit, axis=1, keepdims=True) == 0, 1, np.std(baseline_hit, axis=1, keepdims=True))
    all_miss = (all_miss - np.mean(baseline_miss, axis=1, keepdims=True)) / np.where(
        np.std(baseline_miss, axis=1, keepdims=True) == 0, 1, np.std(baseline_miss, axis=1, keepdims=True))

    bin_size = int(400/50)
    all_hit = uniform_filter1d(all_hit, bin_size, axis=1)
    all_miss = uniform_filter1d(all_miss, bin_size, axis=1)

    ci_hit = sem_ci_2d(all_hit)
    ci_miss = sem_ci_2d(all_miss)

    ax[0].plot(time_axis, np.mean(all_hit, axis=0), color=hit_color, label='hit', linewidth=0.5)
    ax[0].plot(time_axis, np.mean(all_miss, axis=0), color=miss_color, label='miss', linewidth=0.5)
    ax[0].fill_between(time_axis, ci_hit[0], ci_hit[1], color=hit_color, alpha=0.25)
    ax[0].fill_between(time_axis, ci_miss[0], ci_miss[1], color=miss_color, alpha=0.25)
    ax[0].axvline(0, color='red', linestyle='--', linewidth=0.5)
    ax[0].axvline(0.67, color='red', linestyle='-', linewidth=0.5)
    ax[0].set_xlim(-2, 4)
    ax[0].set_xticks(np.arange(-2, 5, 2))
    ax[0].set_xlabel("time (s)")
    ax[0].tick_params(axis='both', which='both', length=1, width=0.5)
    
    ax[1].scatter(np.ones(len(miss)), miss, 5, facecolors='none', edgecolors=miss_color, marker='^', label='hit miss', linewidths=0.25)
    ax[1].scatter(2*np.ones(len(hit)), hit, 5, facecolors='none', edgecolors=hit_color, marker='^', label='hit miss', linewidths=0.25)
    ax[1].scatter(1, np.mean(miss), 6, 'k', marker='^', label='Mean hit miss', edgecolors='k', linewidths=0.2)
    ax[1].scatter(2, np.mean(hit), 6, 'k', marker='v', label='Mean hit High', edgecolors='k', linewidths=0.2)
    ax[1].set_xlim([0, 3])
    ax[1].set_xticks([1, 2])
    ax[1].set_xticklabels(['miss', 'hit'])
    yl = [min(np.min(miss), np.min(hit)), max(np.max(miss), np.max(hit))]
    m = yl[1] - yl[0]
    ax[1].set_ylim([yl[0] - 0.05 * m, yl[1] + m * 0.25])
    ax[1].plot([1, 2], [miss, hit], '-k', linewidth=0.075)
    ax[1].plot([1, 2], [np.mean(miss), np.mean(hit)], '-k', linewidth=0.2)
    ax[1].tick_params(axis='both', which='both', length=1, width=0.5)
    asterisk_y = yl[1] + m * 0.125
    bar_y = asterisk_y + m * 0.07
    if p_vs < p_threshold:
        ax[1].text(1.5, bar_y, f'*(p={p_vs:.4f})', ha='center')
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

def setup_folders(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, 'Appear'))
        os.makedirs(os.path.join(folder_path, 'PreAppear'))
        os.makedirs(os.path.join(folder_path, 'Event'))

def load_data(brain_region, subjects, condition, low_p):
    good_neurons = pd.concat([
        pd.read_csv(f'path/{brain_region}/p_values_{brain_region}_{sub}_{low_p}_{condition}.csv')
        for sub in subjects], ignore_index=True) # path to p-value files (.csv) generated from permutation_analysis.py 
    
    effect_size = pd.concat([
        pd.read_csv(f'path/{brain_region}/effect_sizes_{brain_region}_{sub}_{low_p}_{condition}.csv')
        for sub in subjects], ignore_index=True) # path to effect-size files (.csv) generated from permutation_analysis.py
       
    good_neurons = good_neurons.rename(columns={good_neurons.columns[0]: 'ID'})
    good_neurons[['sub', 'channel', 'unit']] = good_neurons['ID'].str.split('_', expand=True)
    good_neurons = good_neurons[['ID', 'sub', 'channel', 'unit'] + [col for col in good_neurons.columns if col not in ['ID', 'sub', 'channel', 'unit']]]    
    columns_to_keep = ["ID","sub", "channel", "unit", "A_safety_value", "B_safety_value",
                       "A_safety_value_raw", "B_safety_value_raw", "A_safety_variance", "B_safety_variance",
                       "A_prediction_error", "B_prediction_error", "A_absolute_prediction_error", "B_absolute_prediction_error"]
    good_neurons = good_neurons[columns_to_keep]
    
    effect_size = effect_size.rename(columns={effect_size.columns[0]: 'ID'})
    effect_size[['sub', 'channel', 'unit']] = effect_size['ID'].str.split('_', expand=True)
    effect_size = effect_size[['ID', 'sub', 'channel', 'unit'] + [col for col in effect_size.columns if col not in ['ID', 'sub', 'channel', 'unit']]]
    return good_neurons, effect_size

def main():
    # Parameters
    brain_region = 'CLA'
    low_p = 0.3
    p_value = 0.05
    brain_regions = {'ACC': ['sub016A', 'sub016B', 'sub019A', 'sub019B', 'sub020', 'sub023', 'sub024A'],
                     'CLA': ['sub016A', 'sub016B', 'sub017', 'sub024A']}        
    
    subjects = brain_regions.get(brain_region, [])
    folder_path = f'Output/{brain_region}/Population/{low_p}' # Path to Output
    setup_folders(folder_path)
    
    for condition, (a_s, a_e) in zip(['PreAppear', 'Appear', 'Event'], [(-1, 0), (0, 0.4), (0.5, 1)]):
        good_neurons, effect_size = load_data(brain_region, subjects, condition, low_p)

        plt.ioff()
        output_file = os.path.join(folder_path, f'HITvsMISS_{brain_region}_{low_p}_{condition}_neuron_categories.xlsx')
        
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            hit_number_neurons = np.zeros((len(good_neurons.columns[4:]), 2))
            miss_number_neurons = np.zeros((len(good_neurons.columns[4:]), 2))
            
            for idx, column in enumerate(good_neurons.columns[4:]):
                sig_neurons = good_neurons[good_neurons[column] < p_value]
                if sig_neurons.empty:
                    continue                
                sig_neurons = sig_neurons.merge(effect_size[['ID', column]], on='ID', how='left')
                sig_neurons.rename(columns={sig_neurons.columns[-1]: 'effect_size'}, inplace=True)

                all_hit_low, all_hit_high, all_miss_low, all_miss_high = [], [], [], []
                hit_high_neurons, hit_low_neurons, miss_high_neurons, miss_low_neurons = [], [], [], []
                time_axis = None

                for _, row in sig_neurons.iterrows():
                    sub = row['sub']
                    file_name = row['ID']
                    
                    fr_mat_file_path = os.path.join(f'Input/{condition}/{brain_region}/{sub}/{brain_region}_{file_name}.mat') # Path to Dataset
                    fr_values = scipy.io.loadmat(fr_mat_file_path)['fr']

                    behavior_file = os.path.join(f'Input/{condition}/{brain_region}/{sub}/{brain_region}_{sub}.xlsx')
                    behavior = pd.read_excel(behavior_file)

                    control = pd.read_csv(f'Output/{sub}_gameinfo_yPos.csv') # Path to Output
                    control = control.iloc[1:-1].reset_index(drop=True)

                    time_axis = np.linspace(-2, 4, fr_values.shape[1])
                    event_idx = (time_axis >= a_s) & (time_axis <= a_e)

                    baseline_std = np.std(fr_values, axis=1, keepdims=True)
                    zero_rows = np.where(baseline_std == 0)[0]
                    fr_values = np.delete(fr_values, zero_rows, axis=0)
                    behavior = behavior.drop(index=zero_rows).reset_index(drop=True)
                    
                    hit_low_idx = behavior[(behavior[column] <= behavior[column].quantile(low_p)) & (behavior['outcome'] == 1)].index.to_numpy()
                    hit_high_idx = behavior[(behavior[column] >= behavior[column].quantile(low_p)) & (behavior['outcome'] == 1)].index.to_numpy()
                    miss_low_idx = behavior[(behavior[column] <= behavior[column].quantile(low_p)) & (behavior['outcome'] == 0)].index.to_numpy()
                    miss_high_idx = behavior[(behavior[column] >= behavior[column].quantile(low_p)) & (behavior['outcome'] == 0)].index.to_numpy()
                    
                    if row['effect_size'] > 0:
                        all_hit_high.append(np.mean(fr_values[hit_high_idx], axis=0))
                        all_miss_high.append(np.mean(fr_values[miss_high_idx], axis=0))
                        hit_high_neurons.append([column, row.ID, 'high', 'hit'])
                        miss_high_neurons.append([column, row.ID, 'high', 'miss'])
                    elif row['effect_size'] < 0:
                        all_hit_low.append(np.mean(fr_values[hit_low_idx], axis=0))
                        all_miss_low.append(np.mean(fr_values[miss_low_idx], axis=0))
                        hit_low_neurons.append([column, row.ID, 'low', 'hit'])
                        miss_low_neurons.append([column, row.ID, 'low', 'miss'])

                high_neuron_behavior = pd.DataFrame(hit_high_neurons + miss_high_neurons, 
                                                    columns=['Behavior', 'Neuron_ID', 'Category', 'Condition'])
                low_neuron_behavior = pd.DataFrame(hit_low_neurons + miss_low_neurons, 
                                                   columns=['Behavior', 'Neuron_ID', 'Category', 'Condition'])
                
                high_neuron_behavior.to_excel(writer, sheet_name=f"{column}_h", index=False)
                low_neuron_behavior.to_excel(writer, sheet_name=f"{column}_m", index=False)

                all_hit_high = np.vstack(all_hit_high) if all_hit_high else np.zeros((0, fr_values.shape[1]))
                all_miss_high = np.vstack(all_miss_high) if all_miss_high else np.zeros((0, fr_values.shape[1]))
                hit_number_neurons[idx, 0] = all_hit_high.shape[0]
                miss_number_neurons[idx, 0] = all_miss_high.shape[0]

                all_hit_low = np.vstack(all_hit_low) if all_hit_low else np.zeros((0, fr_values.shape[1]))
                all_miss_low = np.vstack(all_miss_low) if all_miss_low else np.zeros((0, fr_values.shape[1]))
                hit_number_neurons[idx, 1] = all_hit_low.shape[0]
                miss_number_neurons[idx, 1] = all_miss_low.shape[0]

                if all_hit_high.size == 0 and all_miss_high.size == 0:
                    continue
                else:
                    fig, ax = plt.subplots(2, 1, figsize=(.975, 1.575))
                    plt.subplots_adjust(hspace=0.5)
                    fig.suptitle(column, fontsize=5)
                    
                    p_v = plot_condition(ax, time_axis, all_hit_high, all_miss_high,
                                         event_idx, hit_high_neurons, column, "HitvsMiss-High", p_value)

                    if p_v < 0.05:
                        fig.savefig(os.path.join(folder_path, condition, f'HiVsMiss_{brain_region}_{column}_{low_p}_{condition}_high.svg'), format='svg')
                        fig.savefig(os.path.join(folder_path, condition, f'HiVsMiss_{brain_region}_{column}_{low_p}_{condition}_high.pdf'), format='pdf')
                    else:
                        fig.savefig(os.path.join(folder_path, condition, 'ns', f'HiVsMiss_{brain_region}_{column}_{low_p}_{condition}_high.svg'), format='svg')
                        fig.savefig(os.path.join(folder_path, condition, 'ns', f'HiVsMiss_{brain_region}_{column}_{low_p}_{condition}_high.pdf'), format='pdf')
                    
                    plt.close(fig)
                    gc.collect()

                if all_hit_low.size == 0 and all_miss_low.size == 0:
                    continue
                else:
                    fig, ax = plt.subplots(2, 1, figsize=(.975, 1.575))
                    plt.subplots_adjust(hspace=0.5)
                    fig.suptitle(column, fontsize=5)
                    
                    p_v = plot_condition(ax, time_axis, all_hit_low, all_miss_low,
                                         event_idx, hit_low_neurons, column, "HitvsMiss-Low", p_value)

                    if p_v < 0.05:
                        fig.savefig(os.path.join(folder_path, condition, f'HiVsMiss_{brain_region}_{column}_{low_p}_{condition}_low.svg'), format='svg')
                        fig.savefig(os.path.join(folder_path, condition, f'HiVsMiss_{brain_region}_{column}_{low_p}_{condition}_low.pdf'), format='pdf')
                    else:
                        fig.savefig(os.path.join(folder_path, condition, 'ns', f'HiVsMiss_{brain_region}_{column}_{low_p}_{condition}_low.svg'), format='svg')
                        fig.savefig(os.path.join(folder_path, condition, 'ns', f'HiVsMiss_{brain_region}_{column}_{low_p}_{condition}_low.pdf'), format='pdf')
                    
                    plt.close(fig)
                    gc.collect()
        plt.ion()
if __name__ == "__main__":
    main()
