#%% permutation_analysis.py
"""
Description: This script performs a permutation-based statistical analysis to assess the relationship
             between neuronal firing rates and various behavioral metrics across different cognitive
             conditions, identifying significant activity modulations and effect directions for each neuron.

Created on Feb 17,2025
@author: Rodrigo Dalvit"""

#%%
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
from permutationTest import permutationTest

def load_behavior_data(sub, brain_region, condition, base_dir):
    xlsx_path = base_dir / condition / brain_region / sub / f"{brain_region}_{sub}.xlsx"
    behavior = pd.read_excel(xlsx_path)
    cols = [
        "Subject", "trial", "outcome",
        "A_safety_value", "B_safety_value",
        "A_safety_value_raw", "B_safety_value_raw",
        "A_safety_variance", "B_safety_variance",
        "A_prediction_error", "B_prediction_error",
        "A_absolute_prediction_error", "B_absolute_prediction_error"        
    ]
    return behavior[cols]

def analyze_neuron(file_name, fr_values, behavior, a_s, a_e, low_percentile, high_percentile):
    results = {"p_value": {}, "effect_size": {}}
    time_vector = np.linspace(-2, 4, fr_values.shape[1])
    event_idx = (time_vector >= a_s) & (time_vector <= a_e)
    baseline_idx = (time_vector >= -2) & (time_vector <= -1.5)

    for behavior_name in behavior.columns[3:]:
        low_thresh = behavior[behavior_name].quantile(low_percentile)
        high_thresh = behavior[behavior_name].quantile(high_percentile)

        low_idx = behavior[behavior_name] <= low_thresh
        high_idx = behavior[behavior_name] >= high_thresh

        if low_idx.sum() < 30 or high_idx.sum() < 30:
            results["p_value"][behavior_name] = 1
            results["effect_size"][behavior_name] = 0
            continue

        low_data = fr_values[low_idx, :]
        high_data = fr_values[high_idx, :]

        # Demean using baseline
        low_data = low_data - np.mean(low_data[:, baseline_idx], axis=1, keepdims=True)
        high_data = high_data - np.mean(high_data[:, baseline_idx], axis=1, keepdims=True)

        low_analysis = low_data[:, event_idx]
        high_analysis = high_data[:, event_idx]

        # Exclude suppressors: check if both low and high are not significantly above baseline
        p_high, _, _ = permutationTest(np.mean(high_analysis, axis=1), np.mean(high_data[:, baseline_idx], axis=1), 1000, 'larger')
        p_low, _, _ = permutationTest(np.mean(low_analysis, axis=1), np.mean(low_data[:, baseline_idx], axis=1), 1000, 'larger')

        if p_high >= 0.05 and p_low >= 0.05:
            results["p_value"][behavior_name] = 1
            results["effect_size"][behavior_name] = 0
            continue

        # Signed effect size (+1 or -1)
        mean_diff = np.mean(high_analysis) - np.mean(low_analysis)
        effect_size = np.sign(mean_diff)

        # Permutation test (two-sided)
        p_val, _, _ = permutationTest(high_analysis, low_analysis, permutations=1000, sidedness='both')
        results["p_value"][behavior_name] = p_val
        results["effect_size"][behavior_name] = effect_size

    return results


def main():
    # === Parameters === #
    # brain_region = 'CLA'
    brain_region = 'ACC'
    low_pct = 0.3
    high_pct = 1 - low_pct
    condition_windows = {
        "PreAppear": (-0.5, 0),
        "Appear": (0, 0.4),
        "Event": (0.5, 1)
    }

    base_dir = Path("Input") # Path to Dataset
    save_dir = Path("Output") / brain_region # Path to Output
    save_dir.mkdir(parents=True, exist_ok=True)

    subjects_dict = {
        'CLA': ['sub016A', 'sub016B', 'sub017', 'sub024A'],
        'ACC': ['sub016A', 'sub016B', 'sub019A', 'sub019B', 'sub020', 'sub023', 'sub024A']
    }
    subjects = subjects_dict.get(brain_region, [])

    good_neurons_df = pd.read_csv("List of neurons to use")

    for sub in subjects:
        sub_neurons = good_neurons_df[good_neurons_df['goodneurons'].str.contains(sub, na=False)]
        file_names = sub_neurons.iloc[:, 0].astype(str)

        for condition, (a_s, a_e) in condition_windows.items():
            mat_path = base_dir / condition / brain_region / sub
            p_results = {}
            es_results = {}

            behavior = load_behavior_data(sub, brain_region, condition, base_dir)

            for file_name in file_names:
                mat_file = mat_path / f"{brain_region}_{file_name}.mat"
                fr_data = scipy.io.loadmat(mat_file)['fr']
                stds = np.std(fr_data, axis=1, keepdims=True)

                # Drop rows with 0 std
                zero_rows = np.where(stds == 0)[0]
                fr_data = np.delete(fr_data, zero_rows, axis=0)
                behavior_cleaned = behavior.drop(index=zero_rows).reset_index(drop=True)

                res = analyze_neuron(file_name, fr_data, behavior_cleaned, a_s, a_e, low_pct, high_pct)
                p_results[file_name] = res["p_value"]
                es_results[file_name] = res["effect_size"]

            # Save results
            pd.DataFrame.from_dict(p_results, orient="index").to_csv(
                save_dir / f"p_values_{brain_region}_{sub}_{low_pct}_{condition}.csv", float_format="%.4f")
            pd.DataFrame.from_dict(es_results, orient="index").to_csv(
                save_dir / f"effect_sizes_{brain_region}_{sub}_{low_pct}_{condition}.csv", float_format="%.4f")

            print(f"Saved: {sub}, {condition}")

if __name__ == "__main__":
    main()
