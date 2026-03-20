"""
Claustrum / ACC / AMY conditional mutual information (CMI) analysis
==================================================================

Reproduces the conditional mutual information (CMI) analysis used in the study
“Human claustrum neurons encode uncertainty and prediction errors during aversive learning”
and generates:
    Fig. 5h-k, 6e-h
    Ext. Fig. 6c-g, 7c-g
    Table 3

Inputs:
- firing-rate matrices
- spike-time structures
- trial-level latent variables

Outputs:
- summary tables
- single-unit plots
- population summary plots

Citation
--------
Please cite the associated paper if you use this code.

Author: Rodrigo Dalvit
Copyright (c) 2026 Rodrigo Dalvit / DamisahLab

This code is provided solely for manuscript review and figure reproduction.
No reuse, redistribution, modification, or derivative use is permitted
without prior written permission from the copyright holder.
"""

#%% Imports
import os
import numpy as np
import pandas as pd

import scipy.io
from scipy.stats import spearmanr
from scipy.ndimage import uniform_filter1d
from scipy import stats
from scipy.stats import rankdata

from sklearn.metrics import mutual_info_score
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import plotly.graph_objects as go

from joblib import Parallel, delayed

#%% User Settings
brain_region = 'ACC'  # 'ACC', 'CLA'
brain_regions = {
    "ACC": ["subject list"],
    "AMY": ["subject list"],
    "CLA": ["subject list"],
}
sub_neurons_csv = r'path'
data_root = r'path'
ypos_root = r'path'
save_path = r'path'
n_perm = 1000
n_jobs = 12
seed = 42

#%% Helpers
def make_strata(Y):    
    strata, _ = pd.factorize(pd.util.hash_pandas_object(Y, index=False))
    return strata

def conditional_mi_with_conditional_permutation(R, X, Y, n_perm: int = 1000,
                                                n_jobs: int = -1, seed: int = 42):    
    def one_perm(k: int) -> float:        
        rng = np.random.default_rng(seed + k)
        val = 0.0
        for w, r_s, x_s in zip(weights, R_by, X_by):
            xp = x_s[rng.permutation(len(x_s))]
            val += w * mutual_info_score(r_s, xp)
        return val

    def chunk_perm(k0: int, m: int) -> np.ndarray:        
        vals = np.empty(m, dtype=float)
        for j in range(m):
            vals[j] = one_perm(k0 + j)
        return vals
    
    R = pd.Series(R).reset_index(drop=True)
    X = pd.Series(X).reset_index(drop=True)
    Y = Y.reset_index(drop=True)

    valid = (~R.isna()) & (~X.isna())
    for c in Y.columns:
        valid &= ~pd.Series(Y[c]).isna()
    
    R = R[valid].reset_index(drop=True)
    X = X[valid].reset_index(drop=True)
    Y = Y.loc[valid].reset_index(drop=True)
    
    Xb = pd.qcut(X, q=10, labels=False, duplicates="drop").to_numpy(dtype=np.int64)    
    if Xb is None:
        return np.nan, np.nan
    
    R = np.rint(R.to_numpy(dtype=float)).astype(np.int64)
    
    strata = make_strata(Y).astype(np.int64)
    uniq = np.unique(strata)
    
    idx_list = [np.where(strata == s)[0] for s in uniq]
    idx_list = [idx for idx in idx_list if len(idx) >= 3]
    if len(idx_list) == 0:
        return np.nan, np.nan     
    
    N = sum(len(idx) for idx in idx_list)
    weights = np.array([len(idx) for idx in idx_list], dtype=float) / N
    R_by = [R[idx] for idx in idx_list]
    X_by = [Xb[idx] for idx in idx_list]
    
    obs = 0.0
    for w, r_s, x_s in zip(weights, R_by, X_by):
        obs += w * mutual_info_score(r_s, x_s)
    
    chunk = 50
    starts = range(0, n_perm, chunk)
    chunks = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(chunk_perm)(k0, min(chunk, n_perm - k0)) for k0 in starts
    )
    null = np.concatenate(chunks)
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, p

def build_entry(method, brain_region, file_name, behavior_name, event,
                firing_rate, behavior_array, rho, p_value, fr_all, outcome):
    return {'method': method, 'brain_region': brain_region, 'file_name': file_name,
            'behavior_name': behavior_name, 'event': event, 'firing_rate': firing_rate,
            'behavior': behavior_array, 'rho': rho, 'p_value': p_value,
            'fr_all': fr_all, 'outcome': outcome}

def save_results_to_excel(brain_region, result_mi_it, result_mi_a, save_folder):    
    def save_list_to_excel(data_list, filename):
        if len(data_list) == 0:
            print(f"No data to save for {filename}")
            return
        df = pd.DataFrame(data_list)
        full_path = f"{save_folder}{filename}"
        df.to_excel(full_path, index=False)
        print(f"Saved {full_path}")    
    save_list_to_excel(result_mi_it, f'/neuron_id_intertrial_MI_score_{brain_region}.xlsx')      
    save_list_to_excel(result_mi_a, f'/neuron_id_asteroid_MI_score_{brain_region}.xlsx')

def permutation_test(low_vals, high_vals, n_permutations=10000):
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
    fr = np.array(fr)
    x = np.array(x)
    o = np.array(outcome)
    valid_mask = ~np.isnan(fr) & ~np.isnan(x)
    fr = fr[valid_mask]
    x = x[valid_mask]
    o = o[valid_mask]
    
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    fr_sorted = fr[sort_idx]        
    if len(fr_sorted) % 2 == 1:
        fr_sorted = fr_sorted[:-1]
        x_sorted = x_sorted[:-1]        
    x_percentile = rankdata(x_sorted, method='average') / len(x_sorted)    
    mean_frs_smooth = np.full_like(fr_sorted, np.nan, dtype=np.float32)
    half_window = bin_size // 2
    for i in range(len(fr_sorted)):
        start = max(0, i - half_window)
        end = min(len(fr_sorted), i + half_window + 1)
        mean_frs_smooth[i] = np.mean(fr_sorted[start:end])
    return {'x_percentile': x_percentile, 'fr_sorted': fr_sorted,
            'mean_frs_smooth': mean_frs_smooth}

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((nx - 1)*sx + (ny - 1)*sy) / (nx + ny - 2))
    return (np.mean(y) - np.mean(x)) / s_pooled

def plot_individual_neurons(results, method_name, time_point, save_dir,
                            collect_curves=None, brain_region=None, dpi=600):
    if brain_region is None:
        raise ValueError("brain_region must be provided.")
    os.makedirs(save_dir, exist_ok=True)
    all_stat_data = []
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
        curves = plot_version(fr, behavior, outcome)
        xq = curves['x_percentile']
        fr_sorted = curves['fr_sorted']
        fr_smooth = curves['mean_frs_smooth']        
        
        low_vals = fr_smooth[xq <= 0.3]
        high_vals = fr_smooth[xq >= 0.7]                
        mean_low = np.mean(low_vals)
        sem_low = stats.sem(low_vals)
        mean_high = np.mean(high_vals)
        sem_high = stats.sem(high_vals)        
        _, p_val = permutation_test(low_vals, high_vals, n_permutations=10000)
        d_val = cohens_d(low_vals, high_vals)
        if p_val < 0.0001:
            p_val = 0.0001
            
        stat_data = {'brain_region': entry.get('brain_region', brain_region),
            'neuron_id': file_name, 'behavior': behavior_name, 'p_val': p_val,
            'cohens_d': d_val,'event': entry.get('event', 'unknown'),
            'mean_low': mean_low,'sem_low': sem_low, 'mean_high': mean_high,
            'sem_high': sem_high}
        all_stat_data.append(stat_data)
        
        if behavior_name in ['A_safety_variance', 'B_safety_variance']:
            color_high = "orange"
            color_low = "blue"
        else:
            color_high = "#ff408c"
            color_low = "#00aaff"  
        fig, ax = plt.subplots(1, 1, figsize=(.8, .4), sharex=True, dpi=dpi)
        
        ax.plot(xq, fr_smooth, '-', color='black', linewidth=0.5, alpha=0.7)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.axvspan(0, 0.3, color=color_low, alpha=0.1)
        ax.axvspan(0.7, 1.0, color=color_high, alpha=0.1)
        ax.set_xticks([0.0, 0.3, 0.5, 0.7, 1.0])
        ax.set_xticklabels(['0', '30', '50', '70', '100'], fontsize=4)             
        ax.tick_params(axis='both', which='major', labelsize=4, width=0.3, length=.75, pad=.5)
        ax.spines[['top', 'right']].set_visible(False)        
        textstr = (f"Low: {mean_low:.2f}±{sem_low:.2f}\n"
                   f"High: {mean_high:.2f}±{sem_high:.2f}\n"
                   f"Cohens d: {d_val:.5f}\n"
                   f"p = {p_val:.5f}")
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                 verticalalignment='top', fontsize=4,
                 bbox=dict(boxstyle='round,pad=0.2', edgecolor='none', facecolor='white', alpha=0.8))
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
        fig.subplots_adjust(hspace=0.35, left=0.22, right=0.95, top=0.88, bottom=0.15)        
        if collect_curves is not None:
            collect_curves.append({'curve_smooth': fr_smooth, 'curve': fr_sorted,
                'behavior_name': behavior_name, 'behavior': behavior,
                'method': method_name, 'time_point': time_point,
                'file_name': file_name, 'fr_all': fr_all, 'outcome': outcome})        
        fig_path = os.path.join(save_dir, f"{brain_region}_{method_name}_{time_point}_{file_name}_{behavior_name}.png")
        plt.savefig(fig_path, dpi=dpi)
        plt.close(fig)
    return all_stat_data

def plot_pval_heatmap(pval, neuron_labels, columns_to_keep, brain_region,
                      method_label, save_path, title_suffix, scale):    
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad(color='white')

    mask_sig = np.any(pval < 0.05, axis=1)
    pval_sig = pval[mask_sig, :]
    neuron_labels_sig = [neuron_labels[i] for i in np.where(mask_sig)[0]]

    if pval_sig.shape[0] == 0:
        print(f"[{title_suffix}] No significant neurons (p < 0.05). Skipping heatmap.")
        return

    n_behaviors = pval_sig.shape[1]
    n_neurons = pval_sig.shape[0]

    fig_width = scale * min(max(6, n_neurons * 0.22), 20)
    fig_height = scale * max(2, n_behaviors * 0.18 + 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    masked = np.ma.masked_where(pval_sig >= 0.05, pval_sig)
    c = ax.pcolormesh(
        np.arange(n_neurons + 1),
        np.arange(n_behaviors + 1),
        masked.T[::-1],
        cmap=cmap,
        edgecolors='gray',
        linewidth=0.2,
        vmin=0,
        vmax=0.05
    )

    ax.set_aspect('equal')
    ax.tick_params(axis='x', bottom=False, labelbottom=True)
    ax.set_xticks(np.arange(n_neurons) + 0.5)
    ax.set_xticklabels(neuron_labels_sig, fontsize=6, fontname='Arial', rotation=90)

    ax.set_yticks(np.arange(n_behaviors) + 0.5)
    ax.set_yticklabels(columns_to_keep[::-1], fontsize=6, fontname='Arial')

    ax.set_xlim(0, n_neurons)
    ax.set_ylim(0, n_behaviors)
    ax.set_title(title_suffix, fontsize=6, fontname='Arial')

    cbar = plt.colorbar(c, ax=ax, orientation='vertical')
    cbar.set_label('p-Value', fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

    plt.tight_layout()
    plt.savefig(
        rf'{save_path}/{brain_region}_{method_label}_p_values_{title_suffix}_behaviorsOnly.png',
        format='png',
        dpi=300
    )
    plt.close(fig)

def generate_raster(trials, start_time=-4, end_time=2):    
    raster = []
    for trial in trials:
        try:
            flat_trial = np.concatenate([np.ravel(np.array(t, dtype=float)) for t in trial])
            valid_times = flat_trial[(flat_trial >= start_time) & (flat_trial <= end_time)]
            raster.append(valid_times)
        except Exception:
            raster.append([])
    return raster
                
def plot_raster(trials, fr, behavior, time_axis, column, brain_region,
                file_name, folder_path, start_time=-4, end_time=2):   
    os.makedirs(folder_path, exist_ok=True)
    raster = generate_raster(trials, start_time=start_time, end_time=end_time)
    low_cutoff = np.percentile(behavior, 30)
    high_cutoff = np.percentile(behavior, 70)    
    low_idx = np.where(behavior <= low_cutoff)[0]
    high_idx = np.where(behavior >= high_cutoff)[0]
    
    raster_low = [raster[i] for i in low_idx]
    raster_high = [raster[i] for i in high_idx]
    fr_low = fr[low_idx]
    fr_high = fr[high_idx]    
    
    baseline_mask = (time_axis >= 1.5) & (time_axis <= 2)
    fr = (fr - np.mean(fr[:,baseline_mask]))
    fr_low = (fr_low - np.mean(fr_low[:,baseline_mask]))
    fr_high = (fr_high - np.mean(fr_high[:,baseline_mask]))
    
    bin_size = int(200/50)
    fr_low_smooth = uniform_filter1d(np.mean(fr_low, axis=0), bin_size)
    fr_high_smooth = uniform_filter1d(np.mean(fr_high, axis=0), bin_size)
    fr_all_smooth = uniform_filter1d(np.mean(fr, axis=0), bin_size)
    
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3), constrained_layout=True)
    fig.suptitle(column, fontsize=5)
    
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
    
    raster_x_low, raster_y_low = [], []
    for i, trial_times in enumerate(raster_low):
        raster_x_low.extend(trial_times)
        raster_y_low.extend([i] * len(trial_times))    
    raster_x_high, raster_y_high = [], []
    for i, trial_times in enumerate(raster_high):
        raster_x_high.extend(trial_times)
        raster_y_high.extend([i + len(raster_low)] * len(trial_times))
    ax_split_raster = axes[0, 1]
    if column in ['A_safety_variance', 'B_safety_variance', 'uncertainty']:
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
    
    ax_split_fr = axes[1, 1]
    if column in ['A_safety_variance', 'B_safety_variance', 'uncertainty']:
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
    
    fname = f"{brain_region}_{file_name}_{column}"
    save_path_png = os.path.join(folder_path, f"{fname}.png")
    os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
    fig.savefig(save_path_png, format='png', dpi=300, transparent=True)
    plt.close(fig)

def normalize_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """Remove A_/B_ prefixes from Behavior names (like R's gsub('^[AB]_', '', ...))."""
    df = df.copy()
    df["Behavior"] = df["Behavior"].astype(str).str.replace(r"^[AB]_", "", regex=True)
    return df

def assign_behavior_category(df: pd.DataFrame) -> pd.DataFrame:    
    df = df.copy()
    prediction_terms = ("absolute_prediction_error", "prediction_error")
    uncertainty_terms = ("safety_variance", "uncertainty")

    def _cat(b) -> str:
        s = str(b).lower()

        if any(term in s for term in prediction_terms):
            return "Prediction Error"
        if any(term in s for term in uncertainty_terms):
            return "Uncertainty"
        return "Other"

    df["BehaviorCategory"] = df["Behavior"].apply(_cat)
    return df

def sankey_diagram(df_intertrial: pd.DataFrame, df_asteroid: pd.DataFrame,
                   out_file_png: str, value_scale: float = 0.2, node_thickness: int = 12,
                   node_padding: int = 20, left_behaviors: Optional[List[str]] = None,
                   right_behaviors: Optional[List[str]] = None, left_suffix: str = " (intertrial)",
                   right_suffix: str = " (asteroid)"):
    def hex_to_rgba(hex_color: str, alpha: float = 0.3) -> str:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    TRANSPARENT = "rgba(0,0,0,0)"
    pal = {
        "absolute_prediction_error": "#AF9BCA",
        "safety_variance": "#B1A0A1",
        "uncertainty": "#B1A0A1",
        "prediction_error": "#AF9BCA",
        "Neuron": "#000000",
        "PaddingTop": "#000000",
        "PaddingBottom": "#000000",
    }
    df_i = assign_behavior_category(df_intertrial.copy())
    df_i = df_i.astype({"Behavior": "string", "NeuronID": "string"})
    df_i = df_i[df_i["BehaviorCategory"] != "Other"]
    df_i = df_i[["NeuronID", "Behavior", "BehaviorCategory"]].drop_duplicates()

    df_a = assign_behavior_category(df_asteroid.copy())
    df_a = df_a.astype({"Behavior": "string", "NeuronID": "string"})
    df_a = df_a[df_a["BehaviorCategory"] != "Other"]
    df_a = df_a[["NeuronID", "Behavior", "BehaviorCategory"]].drop_duplicates()
    
    if left_behaviors is not None:
        df_i = df_i[df_i["Behavior"].isin(left_behaviors)]
    if right_behaviors is not None:
        df_a = df_a[df_a["Behavior"].isin(right_behaviors)]

    if df_i.empty and df_a.empty:
        print("No neuron–behavior pairs to plot (both epochs empty after filters).")
        return

    df_i = df_i.copy()
    df_a = df_a.copy()
    df_i["BehaviorEpoch"] = df_i["Behavior"] + left_suffix
    df_a["BehaviorEpoch"] = df_a["Behavior"] + right_suffix

    left_beh = (
        df_i[["BehaviorEpoch", "BehaviorCategory"]]
        .drop_duplicates()
        .sort_values(["BehaviorCategory", "BehaviorEpoch"], ignore_index=True)
        .copy()
    )
    left_beh["group"] = "BehaviorLeft"
    left_beh = left_beh.rename(columns={"BehaviorEpoch": "name"})

    neuron_ids = pd.concat([df_i[["NeuronID"]], df_a[["NeuronID"]]], ignore_index=True).drop_duplicates()
    neurons = neuron_ids.sort_values("NeuronID", ignore_index=True).copy()
    neurons["name"] = neurons["NeuronID"]
    neurons["group"] = "Neuron"
    neurons = neurons[["name", "group"]]

    right_beh = (
        df_a[["BehaviorEpoch", "BehaviorCategory"]]
        .drop_duplicates()
        .sort_values(["BehaviorCategory", "BehaviorEpoch"], ignore_index=True)
        .copy()
    )
    right_beh["group"] = "BehaviorRight"
    right_beh = right_beh.rename(columns={"BehaviorEpoch": "name"})

    nodes = pd.concat(
        [left_beh[["name", "group"]],
         neurons[["name", "group"]],
         right_beh[["name", "group"]]],
        ignore_index=True
    ).copy()
    nodes["node_id"] = np.arange(len(nodes))
    name_to_id = dict(zip(nodes["name"], nodes["node_id"]))

    BASE_LINK = 1.0

    links_L = df_i[["NeuronID", "BehaviorEpoch", "BehaviorCategory"]].copy()
    links_L["source"] = links_L["BehaviorEpoch"].map(name_to_id)
    links_L["target"] = links_L["NeuronID"].map(name_to_id)
    links_L["link_group"] = links_L["BehaviorCategory"]
    links_L = links_L.drop_duplicates(subset=["source", "target", "link_group"])
    links_L["value"] = BASE_LINK

    links_R = df_a[["NeuronID", "BehaviorEpoch", "BehaviorCategory"]].copy()
    links_R["source"] = links_R["NeuronID"].map(name_to_id)
    links_R["target"] = links_R["BehaviorEpoch"].map(name_to_id)
    links_R["link_group"] = links_R["BehaviorCategory"]
    links_R = links_R.drop_duplicates(subset=["source", "target", "link_group"])
    links_R["value"] = BASE_LINK

    links = pd.concat([links_L, links_R], ignore_index=True)
    if links.empty:
        print("No links to plot after behavior filtering.")
        return

    links["value"] = links["value"] * value_scale

    neuron_node_ids = set(nodes.loc[nodes["group"] == "Neuron", "node_id"].tolist())
    in_sum = links.groupby("target")["value"].sum()
    out_sum = links.groupby("source")["value"].sum()

    DESIRED_TOTAL = 2.0 * value_scale

    max_id = int(nodes["node_id"].max())
    dummy_in_top  = max_id + 1
    dummy_in_bot  = max_id + 2
    dummy_out_top = max_id + 3
    dummy_out_bot = max_id + 4

    nodes_pad = pd.DataFrame([
        {"name": "", "group": "PaddingTop",    "node_id": dummy_in_top},
        {"name": "", "group": "PaddingBottom", "node_id": dummy_in_bot},
        {"name": "", "group": "PaddingTop",    "node_id": dummy_out_top},
        {"name": "", "group": "PaddingBottom", "node_id": dummy_out_bot},
    ])
    nodes = pd.concat([nodes, nodes_pad], ignore_index=True).sort_values("node_id", ignore_index=True)

    pad_links = []
    for n in neuron_node_ids:
        ins  = float(in_sum.get(n, 0.0))
        outs = float(out_sum.get(n, 0.0))

        target_total = max(DESIRED_TOTAL, ins, outs)

        if ins < target_total:
            pad = target_total - ins
            pad_top = pad / 2.0
            pad_bot = pad - pad_top
            pad_links.append({"source": dummy_in_top, "target": n, "value": pad_top, "link_group": "PaddingTop"})
            pad_links.append({"source": dummy_in_bot, "target": n, "value": pad_bot, "link_group": "PaddingBottom"})

        if outs < target_total:
            pad = target_total - outs
            pad_top = pad / 2.0
            pad_bot = pad - pad_top
            pad_links.append({"source": n, "target": dummy_out_top, "value": pad_top, "link_group": "PaddingTop"})
            pad_links.append({"source": n, "target": dummy_out_bot, "value": pad_bot, "link_group": "PaddingBottom"})

    if pad_links:
        links = pd.concat([links, pd.DataFrame(pad_links)], ignore_index=True)

    def _rank_link(row):
        if row["target"] in neuron_node_ids:
            key_node = row["target"]; direction = 0
        elif row["source"] in neuron_node_ids:
            key_node = row["source"]; direction = 1
        else:
            key_node = -1; direction = 2

        lg = row["link_group"]
        if lg == "PaddingTop":
            order = 0
        elif lg == "PaddingBottom":
            order = 2
        else:
            order = 1
        return (key_node, direction, order, int(row["source"]), int(row["target"]))

    links["_sortkey"] = links.apply(_rank_link, axis=1)
    links = links.sort_values("_sortkey", kind="mergesort").drop(columns=["_sortkey"]).reset_index(drop=True)

    nodes_sorted = nodes.sort_values("node_id").copy()
    node_labels = nodes_sorted["name"].tolist()
    node_groups = nodes_sorted["group"].tolist()

    col_idx = []
    for g in node_groups:
        if g == "BehaviorLeft":
            col_idx.append(0)
        elif g == "Neuron":
            col_idx.append(1)
        elif g == "BehaviorRight":
            col_idx.append(2)
        else:
            col_idx.append(1)
    nodes_sorted["col_idx"] = col_idx

    nodes_sorted.loc[nodes_sorted["node_id"].isin([dummy_in_top, dummy_in_bot]), "col_idx"] = 0
    nodes_sorted.loc[nodes_sorted["node_id"].isin([dummy_out_top, dummy_out_bot]), "col_idx"] = 2

    col_to_x = {0: 0.05, 1: 0.5, 2: 0.95}
    node_x = [col_to_x[c] for c in nodes_sorted["col_idx"]]

    node_y = np.zeros(len(nodes_sorted))
    left_beh_y_min, left_beh_y_max   = 0.47, 0.53
    right_beh_y_min, right_beh_y_max = 0.35, 0.65
    neu_y_min, neu_y_max             = 0.05, 0.95

    for c in [0, 1, 2]:
        idx = np.where(nodes_sorted["col_idx"].values == c)[0]
        if len(idx) == 0:
            continue
        if c == 0:
            y_min, y_max = left_beh_y_min, left_beh_y_max
        elif c == 2:
            y_min, y_max = right_beh_y_min, right_beh_y_max
        else:
            y_min, y_max = neu_y_min, neu_y_max
        if len(idx) == 1:
            node_y[idx] = (y_min + y_max) / 2
        else:
            node_y[idx] = np.linspace(y_min, y_max, len(idx))
    
    def _set_y_by_name(node_name: str, yval: float):
        j = np.where(nodes_sorted["name"].values == node_name)[0]
        if len(j) > 0:
            node_y[j[0]] = yval
    
    sv_right  = "safety_variance" + right_suffix
    ape_right = "absolute_prediction_error" + right_suffix
    
    _set_y_by_name(sv_right,  0.35)
    _set_y_by_name(ape_right, 0.65)

    left_anchor_ids = nodes_sorted.loc[nodes_sorted["group"] == "BehaviorLeft", "node_id"].tolist()
    right_anchor_ids = nodes_sorted.loc[nodes_sorted["group"] == "BehaviorRight", "node_id"].tolist()

    left_anchor_y = (left_beh_y_min + left_beh_y_max) / 2
    right_anchor_y = (right_beh_y_min + right_beh_y_max) / 2

    if left_anchor_ids:
        left_anchor_y = float(node_y[np.where(nodes_sorted["node_id"].values == left_anchor_ids[0])[0][0]])
    if right_anchor_ids:
        right_anchor_y = float(node_y[np.where(nodes_sorted["node_id"].values == right_anchor_ids[0])[0][0]])

    def _set_xy(node_id, xval, yval):
        j = np.where(nodes_sorted["node_id"].values == node_id)[0]
        if len(j) > 0:
            node_x[j[0]] = xval
            node_y[j[0]] = yval

    _set_xy(dummy_in_top,  col_to_x[0], left_anchor_y)
    _set_xy(dummy_in_bot,  col_to_x[0], left_anchor_y)
    _set_xy(dummy_out_top, col_to_x[2], right_anchor_y)
    _set_xy(dummy_out_bot, col_to_x[2], right_anchor_y)

    node_colors = []
    for g, nm in zip(node_groups, node_labels):
        if g.startswith("Padding"):
            node_colors.append(TRANSPARENT)
        elif g == "Neuron":
            node_colors.append(pal["Neuron"])
        else:
            base = nm.replace(left_suffix, "").replace(right_suffix, "")
            node_colors.append(pal.get(base, "#999999"))

    link_colors = []
    for g in links["link_group"]:
        if str(g).startswith("Padding"):
            link_colors.append(TRANSPARENT)
        else:
            link_colors.append(hex_to_rgba(pal.get(g, "#999999"), 0.3))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=node_padding,
                    thickness=node_thickness,
                    line=dict(color=TRANSPARENT, width=0.0),
                    label=node_labels,
                    color=node_colors,
                    x=node_x,
                    y=node_y,
                ),
                link=dict(
                    source=links["source"],
                    target=links["target"],
                    value=links["value"],
                    color=link_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Sankey – Intertrial behaviors → Neurons → Asteroid behaviors",
        font=dict(size=6),
        margin=dict(l=80, r=80, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.write_image(out_file_png)
    print(f"Saved png: {out_file_png}")
    
#%% Main
def main():
    os.makedirs(save_path, exist_ok=True)

    subjects = brain_regions.get(brain_region, [])
    if len(subjects) == 0:
        raise ValueError(f"No subjects defined for brain_region='{brain_region}'")

    sub_neurons = pd.read_csv(sub_neurons_csv)

    raster_it_dir = os.path.join(save_path, 'Rasters', 'intertrial')
    raster_a_dir  = os.path.join(save_path, 'Rasters', 'asteroid')
    os.makedirs(raster_it_dir, exist_ok=True)
    os.makedirs(raster_a_dir, exist_ok=True)

    map_df_path = os.path.join(save_path, f'{brain_region}_file_name_to_cnt_mapping.xlsx')

    cnt = 0
    neuron_labels = []
    file_cnt_mapping = []
    result_mi_it, result_mi_a = [], []
    p_values_it_mi, p_values_a_mi = [], []

    for sub in subjects:
        good_neurons = sub_neurons.loc[sub_neurons['goodneurons'].astype(str).str.contains(sub, na=False)]
        file_names = good_neurons.iloc[:, 0].astype(str)

        mat_dir = os.path.join(data_root, brain_region, sub)

        for file_name in file_names:
            cnt += 1
            print(f"Processing neuron #{cnt} ({sub} / {file_name})")

            file_cnt_mapping.append({'file_name': file_name, 'cnt': cnt})
            neuron_labels.append(cnt)

            fr_file_path = os.path.join(mat_dir, f'{brain_region}_{file_name}.mat')
            st_file_path = os.path.join(mat_dir, f'{brain_region}_{file_name}_spike_times.mat')

            fr = scipy.io.loadmat(fr_file_path)['fr']
            st = scipy.io.loadmat(st_file_path)['spike_times']

            behavior_path = os.path.join(data_root, brain_region, sub, f'{brain_region}_{sub}.xlsx')
            behavior = pd.read_excel(behavior_path)

            ypos_path = os.path.join(ypos_root, f'{sub}_gameinfo_yPos.csv')
            yPos = pd.read_csv(ypos_path)

            trials_keep = behavior["trial"]
            cols = ['intertrial (-4 to 0)', 'asteroid (0 to 1.5)']
            yPos = yPos[yPos["trial"].isin(trials_keep)].copy()
            yPos_ordered = behavior[['trial']].copy()
            yPos = yPos_ordered.merge(yPos[['trial'] + cols], on='trial', how='left', sort=False)

            trial_sorted_idx = behavior["trial"].astype(int).sort_values().index.to_numpy()
            behavior = behavior.iloc[trial_sorted_idx].reset_index(drop=True)
            fr = fr[trial_sorted_idx, :]
            st = st[trial_sorted_idx, :]

            columns_to_keep = [
                "A_safety_variance", "B_safety_variance",
                "A_absolute_prediction_error", "B_absolute_prediction_error",
                "A_outcome", "B_outcome"
            ]
            outcome = behavior["outcome"]
            behavior = behavior[columns_to_keep]

            time_axis = np.linspace(-4, 2, fr.shape[1])
            intertrial_idx = (time_axis >= -4) & (time_axis <= 0)
            asteroid_idx = (time_axis >= 0) & (time_axis <= 1.5)

            it_sp = np.sum(fr[:, intertrial_idx], axis=1) / 20.0
            a_sp  = np.sum(fr[:, asteroid_idx], axis=1) / 20.0

            intertrial = it_sp.astype(int)
            asteroid = a_sp.astype(int)

            targets_it = ["A_safety_variance", "B_safety_variance"]
            targets_a  = ["A_safety_variance", "B_safety_variance",
                          "A_absolute_prediction_error", "B_absolute_prediction_error"]

            Y_it = pd.DataFrame({"yPos": yPos.iloc[:, 1]})
            Y_a = pd.DataFrame({
                "yPos": yPos.iloc[:, 2],
                "outcome": outcome,
                "A_outcome": behavior["A_outcome"],
                "B_outcome": behavior["B_outcome"]
            })

            if 'yPos' in Y_it.columns:
                Y_it['yPos'] = np.floor(Y_it['yPos'].astype(float) / 10).astype(np.int64)
            for c in Y_it.columns:
                if c != 'yPos':
                    Y_it[c] = Y_it[c].astype(int)

            if 'yPos' in Y_a.columns:
                Y_a['yPos'] = np.floor(Y_a['yPos'].astype(float) / 10).astype(np.int64)
            for c in Y_a.columns:
                if c != 'yPos':
                    Y_a[c] = Y_a[c].astype(int)

            mi_it_vec, p_it_vec = [], []
            for beh_name in targets_it:
                X = behavior[beh_name]
                score, pval = conditional_mi_with_conditional_permutation(
                    R=intertrial, X=X, Y=Y_it, n_perm=n_perm, n_jobs=n_jobs, seed=seed
                )
                mi_it_vec.append(score)
                p_it_vec.append(pval)

                if np.isfinite(pval) and (pval < 0.05):
                    rho, _ = spearmanr(X, intertrial)
                    entry = build_entry(
                        "CMI_condperm", brain_region, cnt, beh_name, 'intertrial',
                        it_sp, np.array(X), float(rho), float(pval), fr, outcome
                    )
                    result_mi_it.append(entry)
                    plot_raster(trials=st, fr=fr, behavior=np.array(X), time_axis=time_axis,
                                column=beh_name, brain_region=brain_region,
                                file_name=file_name, folder_path=raster_it_dir)

            mi_a_vec, p_a_vec = [], []
            for beh_name in targets_a:
                X = behavior[beh_name]
                score, pval = conditional_mi_with_conditional_permutation(
                    R=asteroid, X=X, Y=Y_a, n_perm=n_perm, n_jobs=n_jobs, seed=seed
                )
                mi_a_vec.append(score)
                p_a_vec.append(pval)

                if np.isfinite(pval) and (pval < 0.05):
                    rho, _ = spearmanr(X, asteroid)
                    entry = build_entry(
                        "CMI_condperm", brain_region, cnt, beh_name, 'asteroid',
                        a_sp, np.array(X), float(rho), float(pval), fr, outcome
                    )
                    result_mi_a.append(entry)
                    plot_raster(trials=st, fr=fr, behavior=np.array(X), time_axis=time_axis,
                                column=beh_name, brain_region=brain_region,
                                file_name=file_name, folder_path=raster_a_dir)

            p_values_it_mi.append(np.array(p_it_vec, dtype=float))
            p_values_a_mi.append(np.array(p_a_vec, dtype=float))

    pd.DataFrame(file_cnt_mapping).to_excel(map_df_path, index=False)

    save_results_to_excel(brain_region=brain_region,
                          result_mi_it=result_mi_it,
                          result_mi_a=result_mi_a,
                          save_folder=save_path)

    intertrial_file = os.path.join(save_path, f"neuron_id_intertrial_MI_score_{brain_region}.xlsx")
    asteroid_file   = os.path.join(save_path, f"neuron_id_asteroid_MI_score_{brain_region}.xlsx")

    if os.path.exists(intertrial_file) and os.path.exists(asteroid_file):
        df_intertrial = pd.read_excel(intertrial_file).rename(columns={"behavior_name": "Behavior", "file_name": "NeuronID"})
        df_intertrial["Event"] = "intertrial"
        df_intertrial = normalize_behavior(df_intertrial)

        df_asteroid = pd.read_excel(asteroid_file).rename(columns={"behavior_name": "Behavior", "file_name": "NeuronID"})
        df_asteroid["Event"] = "asteroid"
        df_asteroid = normalize_behavior(df_asteroid)

        out_three_png = os.path.join(save_path, f"sankey_{brain_region}.png")
        sankey_diagram(
            df_intertrial=df_intertrial,
            df_asteroid=df_asteroid,
            out_file_png=out_three_png,
            left_behaviors=["safety_variance"],
            right_behaviors=["safety_variance", "absolute_prediction_error"],
            value_scale=0.2,
            node_thickness=10,
            node_padding=25
        )

    if len(p_values_it_mi) > 0:
        p_it = np.vstack(p_values_it_mi)
        plot_pval_heatmap(p_it, neuron_labels, ["A_safety_variance", "B_safety_variance"],
                          brain_region, "MI_score", save_path, 'Intertrial', scale=0.35)

    if len(p_values_a_mi) > 0:
        p_a = np.vstack(p_values_a_mi)
        plot_pval_heatmap(p_a, neuron_labels,
                          ["A_safety_variance", "B_safety_variance", "A_absolute_prediction_error", "B_absolute_prediction_error"],
                          brain_region, "MI_score", save_path, 'Asteroid', scale=0.52)

    methods = {'mi': {'intertrial': result_mi_it, 'asteroid': result_mi_a}}
    all_curves, all_stat = [], []

    for method, time_dict in methods.items():
        for time_point, res in time_dict.items():
            save_folder = os.path.join(save_path, f"{method}_{time_point}")
            stat_data = plot_individual_neurons(res, method, time_point, save_folder,
                                                collect_curves=all_curves, brain_region=brain_region)
            all_stat.extend(stat_data)

    df_stats = pd.DataFrame(all_stat)
    output_path = os.path.join(save_path, f"statistics_{brain_region}.xlsx")
    df_stats.to_excel(output_path, index=False)
    print(f"Saved stats: {output_path}")

if __name__ == "__main__":
    main()