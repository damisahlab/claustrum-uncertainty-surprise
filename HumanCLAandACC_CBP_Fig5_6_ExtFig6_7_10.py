"""
Cluster-based permutation (CBP) tests: per neuron × behavior × condition
=======================================================================

This script reproduces the cluster-based permutation (CBP) analysis pipeline used in the study
“Human claustrum neurons encode uncertainty and prediction errors during aversive learning”
and generates:

    Fig. 5c-f, 6a-d
    Extended Data Fig. 6a-b, 7a-b, 10f-g

Inputs:
- trial-aligned firing-rate matrices
- trial spike-time structures
- trial-level behavioral annotations

Outputs:
- unit inclusion tables
- single-unit summary plots
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
import gc
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats as stats
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#%% User Settings
BRAIN_REGION = "ACC"  # "ACC", "CLA", "AMY"
BRAIN_REGIONS = {
    "ACC": ["subject list"],
    "AMY": ["subject list"],
    "CLA": ["subject list"],
}
CONDITIONS = [
    ("Appear", 0.00, 0.50),
    ("Event",  0.67, 1.17),
]
BEHAVIORS = ["behaviors"]
LOW_PERCENTILE = 0.30
MIN_TRIALS_PER_TAIL = 30
N_PERM = 10000
P_THRESHOLD = 0.05
CBP_TAIL = "both"
T_START, T_END = -2.0, 4.0
BASELINE_START, BASELINE_END = -2.0, -1.5
BASE_DIR = r"path"

#%% Helpers
def make_time_axis(n_time: int, t_start: float, t_end: float) -> np.ndarray:
    return np.linspace(t_start, t_end, n_time)

def process_data(fr_values: np.ndarray, indices: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    fr_data = fr_values[indices, :]
    baseline = np.mean(fr_data[:, baseline_mask], axis=1, keepdims=True)
    return fr_data - baseline

def cbp_one_sample(data: np.ndarray, n_permutations: int = 10_000, p_threshold: float = 0.05):
    rng = np.random.default_rng()
    n_trials, n_time = data.shape

    se = (np.std(data, axis=0, ddof=1) / np.sqrt(n_trials)) + 1e-12
    t_obs = np.mean(data, axis=0) / se
    t_obs[~np.isfinite(t_obs)] = 0.0

    thr = stats.t.ppf(1 - p_threshold, df=n_trials - 1)

    mask = t_obs > thr
    labels_obs, n_clusters = label(mask)
    cluster_stats = [np.sum(t_obs[labels_obs == c]) for c in range(1, n_clusters + 1)]

    max_stats = np.zeros(n_permutations, float)
    for k in range(n_permutations):
        signs = rng.choice([-1, 1], size=n_trials)[:, None]
        perm = data * signs
        se_p = (np.std(perm, axis=0, ddof=1) / np.sqrt(n_trials)) + 1e-12
        t_p = np.mean(perm, axis=0) / se_p
        t_p[~np.isfinite(t_p)] = 0.0

        mask_p = t_p > thr
        lab_p, n_p = label(mask_p)
        stats_p = [np.sum(t_p[lab_p == c]) for c in range(1, n_p + 1)]
        max_stats[k] = np.max(stats_p) if stats_p else 0.0

    cluster_pvals = [(np.sum(max_stats >= s) + 1) / (n_permutations + 1) for s in cluster_stats]

    sig_labels = np.zeros_like(labels_obs)
    sig_pvals = []
    for i, p in enumerate(cluster_pvals, start=1):
        if p < 0.05:
            sig_pvals.append(float(p))
            sig_labels[labels_obs == i] = i

    return sig_pvals, sig_labels

def cbp_highvslow(high: np.ndarray, low: np.ndarray, n_permutations: int = 10_000,
                 p_threshold: float = 0.05, tail: str = "both"):
    rng = np.random.default_rng()
    n, n_time = high.shape
    diff = high - low

    se = np.std(diff, axis=0, ddof=1) / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_obs = np.mean(diff, axis=0) / se
    t_obs[~np.isfinite(t_obs)] = 0.0

    thr = stats.t.ppf(1 - (p_threshold / 2 if tail == "both" else p_threshold), df=n - 1)

    pos = (t_obs > thr) if tail in ("both", "larger") else np.zeros(n_time, bool)
    neg = (t_obs < -thr) if tail in ("both", "smaller") else np.zeros(n_time, bool)

    lab_pos, n_pos = label(pos)
    lab_neg, n_neg = label(neg)
    if n_neg > 0:
        lab_neg[lab_neg > 0] += n_pos
    labels_obs = lab_pos + lab_neg

    n_clusters = n_pos + n_neg
    cluster_stats = [
        np.sum(t_obs[labels_obs == c]) if c <= n_pos else np.sum(-t_obs[labels_obs == c])
        for c in range(1, n_clusters + 1)
    ]

    max_stats = np.zeros(n_permutations, float)
    for k in range(n_permutations):
        signs = rng.choice([1, -1], size=n)[:, None]
        perm_diff = diff * signs
        se_p = np.std(perm_diff, axis=0, ddof=1) / np.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_p = np.mean(perm_diff, axis=0) / se_p
        t_p[~np.isfinite(t_p)] = 0.0

        if tail in ("both", "larger"):
            pos_p = t_p > thr
            lab_pos_p, n_pos_p = label(pos_p)
        else:
            lab_pos_p = np.zeros(n_time, int); n_pos_p = 0

        if tail in ("both", "smaller"):
            neg_p = t_p < -thr
            lab_neg_p, n_neg_p = label(neg_p)
        else:
            lab_neg_p = np.zeros(n_time, int); n_neg_p = 0

        if n_neg_p > 0:
            lab_neg_p[lab_neg_p > 0] += n_pos_p
        lab_p = lab_pos_p + lab_neg_p

        stats_p = [
            np.sum(t_p[lab_p == c]) if c <= n_pos_p else np.sum(-t_p[lab_p == c])
            for c in range(1, n_pos_p + n_neg_p + 1)
        ]
        max_stats[k] = np.max(stats_p) if stats_p else 0.0

    cluster_pvals = [(np.sum(max_stats >= s) + 1) / (n_permutations + 1) for s in cluster_stats]

    sig_labels = np.zeros_like(labels_obs)
    sig_pvals = []
    for i, p in enumerate(cluster_pvals, start=1):
        if p < 0.05:
            sig_pvals.append(float(p))
            sig_labels[labels_obs == i] = i

    return sig_pvals, sig_labels

def filter_clusters(labels_arr: np.ndarray, min_bins: int = 2):
    labels_arr = np.asarray(labels_arr)
    out = np.zeros_like(labels_arr)
    i = 0
    while i < len(labels_arr):
        if labels_arr[i] > 0:
            cid = labels_arr[i]
            start = i
            while i < len(labels_arr) and labels_arr[i] == cid:
                i += 1
            end = i
            if (end - start) >= min_bins:
                out[start:end] = cid
        else:
            i += 1
    return out

def sem_ci_2d(data: np.ndarray):
    mean_vals = np.mean(data, axis=0)
    sem_vals = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
    return mean_vals - sem_vals, mean_vals + sem_vals

def gen_raster(trials, start_time=-2, end_time=4):
    raster = []
    for trial in trials:
        try:
            flat = np.concatenate([np.ravel(np.array(t, dtype=float)) for t in trial])
            raster.append(flat[(flat >= start_time) & (flat <= end_time)])
        except Exception:
            raster.append([])
    return raster

def plot_raster(trials, fr, time_axis, brain_region, file_name, out_dir,
                start_time=-2, end_time=4):
    os.makedirs(out_dir, exist_ok=True)
    raster = gen_raster(trials, start_time=start_time, end_time=end_time)

    baseline_mask = (time_axis >= 1.5) & (time_axis <= 2)
    fr = (fr - np.mean(fr[:, baseline_mask]))

    bin_size = int(400/50)
    fr_all_smooth = uniform_filter1d(np.mean(fr, axis=0), bin_size)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.115, 1.2), constrained_layout=True)

    rx, ry = [], []
    for i, tt in enumerate(raster):
        rx.extend(tt); ry.extend([i] * len(tt))
    axes[0].scatter(rx, ry, s=.1, c="#22205F", marker="s")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=0.5)
    axes[0].axvline(0.67, color="gray", linestyle="--", linewidth=0.5)
    axes[0].set_xlim(start_time, end_time)
    axes[0].set_xticks([])
    axes[0].set_ylabel("trial #")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["bottom"].set_visible(False)

    axes[1].plot(time_axis, fr_all_smooth, color="#22205F", linewidth=0.5)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=0.5)
    axes[1].axvline(0.67, color="gray", linestyle="--", linewidth=0.5)
    axes[1].set_xlim(start_time, end_time)
    axes[1].set_xticks([])
    axes[1].set_ylabel(r"mean $\Delta$ firing rate (Hz)")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    out_png = os.path.join(out_dir, f"{brain_region}_{file_name}.png")
    fig.savefig(out_png, format="png")
    plt.close(fig)

def plot_condition(high, low, crash, avoid, crash_dash, avoid_dash,
                   behavior_name=None, cond=None, condition=None,
                   brain_region=None, folder_path=None):
    if folder_path is None or brain_region is None:
        raise ValueError("plot_condition requires folder_path and brain_region.")

    color_crash = "#8F39E6"
    color_avoid = "#00cc00"
    if behavior_name in ["A_safety_variance", "B_safety_variance", "uncertainty"]:
        color_high, color_low = "orange", "blue"
    else:
        color_high, color_low = "#ff408c", "#00aaff"

    high = np.asarray(high); low = np.asarray(low)
    crash = np.asarray(crash); avoid = np.asarray(avoid)
    crash_dash = np.asarray(crash_dash); avoid_dash = np.asarray(avoid_dash)

    time_axis = make_time_axis(high.shape[1], T_START, T_END)
    baseline_mask = (time_axis >= BASELINE_START) & (time_axis <= BASELINE_END)

    def _z_by_baseline(A):
        A = np.asarray(A)
        base = A[:, baseline_mask]
        mu = np.mean(base, axis=1, keepdims=True)
        sd = np.std(base, axis=1, keepdims=True)
        sd = np.where(sd == 0, 1, sd)
        return (A - mu) / sd

    bin_size = int(400/50)
    high_z = uniform_filter1d(_z_by_baseline(high), bin_size, axis=1)
    low_z  = uniform_filter1d(_z_by_baseline(low),  bin_size, axis=1)

    crash_z = uniform_filter1d(_z_by_baseline(crash), bin_size, axis=1)
    avoid_z = uniform_filter1d(_z_by_baseline(avoid), bin_size, axis=1)

    crash_dz = uniform_filter1d(_z_by_baseline(crash_dash), bin_size, axis=1)
    avoid_dz = uniform_filter1d(_z_by_baseline(avoid_dash), bin_size, axis=1)

    ci_high  = sem_ci_2d(high_z);  ci_low   = sem_ci_2d(low_z)
    ci_crash = sem_ci_2d(crash_z); ci_avoid = sem_ci_2d(avoid_z)
    ci_crash_d = sem_ci_2d(crash_dz); ci_avoid_d = sem_ci_2d(avoid_dz)

    _, lab_hl = cbp_highvslow(high_z, low_z, n_permutations=N_PERM, p_threshold=P_THRESHOLD, tail="both")
    _, lab_ac = cbp_highvslow(avoid_z, crash_z, n_permutations=N_PERM, p_threshold=P_THRESHOLD, tail="both")
    _, lab_ac_d = cbp_highvslow(avoid_dz, crash_dz, n_permutations=N_PERM, p_threshold=P_THRESHOLD, tail="both")

    lab_hl   = filter_clusters(np.asarray(lab_hl),   min_bins=2)
    lab_ac   = filter_clusters(np.asarray(lab_ac),   min_bins=2)
    lab_ac_d = filter_clusters(np.asarray(lab_ac_d), min_bins=2)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5

    fig, ax = plt.subplots(3, 1, figsize=(0.95, 2.7))
    plt.subplots_adjust(hspace=0.25)
    fig.suptitle(behavior_name, fontsize=5)

    ax[0].plot(time_axis, high_z.mean(axis=0), color=color_high, linewidth=.5, label="High")
    ax[0].plot(time_axis, low_z.mean(axis=0),  color=color_low,  linewidth=.5, label="Low")
    ax[0].fill_between(time_axis, ci_high[0], ci_high[1], color=color_high, alpha=0.25, edgecolor="none")
    ax[0].fill_between(time_axis, ci_low[0],  ci_low[1],  color=color_low,  alpha=0.25, edgecolor="none")
    ax[0].axvline(0, color="red", linestyle="--", linewidth=0.5)
    ax[0].axvline(0.67, color="gray", linestyle="--", linewidth=0.5)
    ax[0].set_xlim(T_START, T_END)
    ax[0].set_xticklabels([]); ax[0].xaxis.set_ticks([])
    ax[0].set_title(f"{cond} (n={high_z.shape[0]})", fontsize=5)

    pos0 = ax[0].get_position()
    bar0 = fig.add_axes([pos0.x0, pos0.y0 + 0.005, pos0.width, 0.008])
    bar0.imshow(np.expand_dims(lab_hl > 0, axis=0), aspect="auto", cmap="Greys",
                origin="lower", extent=[time_axis[0], time_axis[-1], 0, 1])
    bar0.axis("off")

    ax[1].plot(time_axis, crash_z.mean(axis=0), color=color_crash, linewidth=.5, label="Crash")
    ax[1].plot(time_axis, avoid_z.mean(axis=0), color=color_avoid, linewidth=.5, label="Avoid")
    ax[1].fill_between(time_axis, ci_crash[0], ci_crash[1], color=color_crash, alpha=0.25, edgecolor="none")
    ax[1].fill_between(time_axis, ci_avoid[0], ci_avoid[1], color=color_avoid, alpha=0.25, edgecolor="none")
    ax[1].axvline(0, color="red", linestyle="--", linewidth=0.5)
    ax[1].axvline(0.67, color="gray", linestyle="--", linewidth=0.5)
    ax[1].set_xlim(T_START, T_END)
    ax[1].set_xticklabels([]); ax[1].xaxis.set_ticks([])
    ax[1].set_ylabel("z(FR)")

    pos1 = ax[1].get_position()
    bar1 = fig.add_axes([pos1.x0, pos1.y0 + 0.005, pos1.width, 0.008])
    bar1.imshow(np.expand_dims(lab_ac > 0, axis=0), aspect="auto", cmap="Greys",
                origin="lower", extent=[time_axis[0], time_axis[-1], 0, 1])
    bar1.axis("off")

    other_tail = "Low" if cond == "High" else "High"
    ax[2].plot(time_axis, crash_dz.mean(axis=0), color=color_crash, linewidth=.5)
    ax[2].plot(time_axis, avoid_dz.mean(axis=0), color=color_avoid, linewidth=.5)
    ax[2].fill_between(time_axis, ci_crash_d[0], ci_crash_d[1], color=color_crash, alpha=0.25, edgecolor="none")
    ax[2].fill_between(time_axis, ci_avoid_d[0], ci_avoid_d[1], color=color_avoid, alpha=0.25, edgecolor="none")
    ax[2].axvline(0, color="red", linestyle="--", linewidth=0.5)
    ax[2].axvline(0.67, color="gray", linestyle="--", linewidth=0.5)
    ax[2].set_xlim(T_START, T_END)
    ax[2].set_xticks(np.arange(-2, 5, 2))
    ax[2].set_xlabel("time (s)")
    ax[2].text(0.98, 0.90, other_tail, transform=ax[2].transAxes,
               ha="right", va="top", fontsize=5)

    pos2 = ax[2].get_position()
    bar2 = fig.add_axes([pos2.x0, pos2.y0 + 0.005, pos2.width, 0.008])
    bar2.imshow(np.expand_dims(lab_ac_d > 0, axis=0), aspect="auto", cmap="Greys",
                origin="lower", extent=[time_axis[0], time_axis[-1], 0, 1])
    bar2.axis("off")

    for a in ax:
        for spine in a.spines.values():
            spine.set_linewidth(0.3)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(axis="both", which="both", length=1, width=0.5, pad=1)

    out_dir = os.path.join(folder_path, condition)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{brain_region}_{behavior_name}_{cond}.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    gc.collect()

#%% Main
def main():
    subjects = BRAIN_REGIONS.get(BRAIN_REGION, [])
    if not subjects:
        raise ValueError(f"No subjects for brain region: {BRAIN_REGION}")
    
    DATA_ROOTS = {
        "Appear": os.path.join(BASE_DIR, "Data", "Appear"),
        "Event":  os.path.join(BASE_DIR, "Data", "Event"),
    }

    OUT_ROOT = os.path.join(BASE_DIR, "CBP", "{brain_region}", "Population_CBP")
    out_root = OUT_ROOT.format(brain_region=BRAIN_REGION)
    os.makedirs(out_root, exist_ok=True)

    for subf in ["Data", "Rasters", "Appear", "Event"]:
        os.makedirs(os.path.join(out_root, subf), exist_ok=True)
        
    GOOD_NEURONS_CSV = os.path.join(BASE_DIR, "good_neurons_{brain_region}.csv")
    good_neurons_path = GOOD_NEURONS_CSV.format(brain_region=BRAIN_REGION)
    good_neurons_all = pd.read_csv(good_neurons_path)

    for condition, a_s, a_e in CONDITIONS:
        print(f"\n=== Condition: {condition} | window: [{a_s}, {a_e}] ===")
        
        BAD_NEURONS_XLSX = os.path.join(BASE_DIR, "badneurons", "{brain_region}_badneurons_{condition}.xlsx")
        bad_path = BAD_NEURONS_XLSX.format(brain_region=BRAIN_REGION, condition=condition)
        badneurons = pd.read_excel(bad_path)

        data_root = DATA_ROOTS[condition]

        for behavior_name in BEHAVIORS:
            print(f"\n--- Behavior: {behavior_name} ---")

            bad_high = set(badneurons.get(f"{behavior_name}_high", pd.Series()).dropna().astype(str).tolist())
            bad_low  = set(badneurons.get(f"{behavior_name}_low",  pd.Series()).dropna().astype(str).tolist())

            hh, hl, lh, ll = [], [], [], []                        
            crash_h, avoid_h, crash_h_dash, avoid_h_dash = [], [], [], []
            crash_l, avoid_l, crash_l_dash, avoid_l_dash = [], [], [], []
            units_high, units_low = [], []

            for sub in subjects:
                gn = good_neurons_all.loc[good_neurons_all["goodneurons"].astype(str).str.contains(sub, na=False)]
                file_names = gn.iloc[:, 0].astype(str)

                mat_dir = os.path.join(data_root, BRAIN_REGION, sub)
                beh_xlsx = os.path.join(data_root, BRAIN_REGION, sub, f"{BRAIN_REGION}_{sub}.xlsx")

                behavior_df = pd.read_excel(beh_xlsx)

                if behavior_name == "prediction_error" and "prediction_error" not in behavior_df.columns:
                    behavior_df["prediction_error"] = (
                        behavior_df["A_absolute_prediction_error"] + behavior_df["B_absolute_prediction_error"]
                    ) / 2.0
                if behavior_name == "uncertainty" and "uncertainty" not in behavior_df.columns:
                    behavior_df["uncertainty"] = (
                        behavior_df["A_safety_variance"] + behavior_df["B_safety_variance"]
                    ) / 2.0

                beh = behavior_df[behavior_name]
                outcome = behavior_df["outcome"].astype(int).values

                low_thr = beh.quantile(LOW_PERCENTILE)
                high_thr = beh.quantile(1.0 - LOW_PERCENTILE)

                for file_name in file_names:
                    if (file_name in bad_high) and (file_name in bad_low):
                        continue

                    fr_path = os.path.join(mat_dir, f"{BRAIN_REGION}_{file_name}.mat")
                    st_path = os.path.join(mat_dir, f"{BRAIN_REGION}_{file_name}_spike_times.mat")

                    fr_values = scipy.io.loadmat(fr_path)["fr"]
                    st_values = scipy.io.loadmat(st_path)["spike_times"]

                    time_axis = make_time_axis(fr_values.shape[1], T_START, T_END)
                    baseline_mask = (time_axis >= BASELINE_START) & (time_axis <= BASELINE_END)

                    # split
                    low_idx = beh <= low_thr
                    high_idx = beh >= high_thr
                    if low_idx.sum() < MIN_TRIALS_PER_TAIL or high_idx.sum() < MIN_TRIALS_PER_TAIL:
                        continue
                    
                    low_inds = np.where(low_idx)[0]
                    high_inds = np.where(high_idx)[0]
                    
                    crash_low_inds  = low_inds[outcome[low_inds] == 1]
                    avoid_low_inds  = low_inds[outcome[low_inds] == 0]
                    crash_high_inds = high_inds[outcome[high_inds] == 1]
                    avoid_high_inds = high_inds[outcome[high_inds] == 0]
                                    
                    if (len(crash_low_inds) == 0 or len(avoid_low_inds) == 0 or
                        len(crash_high_inds) == 0 or len(avoid_high_inds) == 0):
                        continue

                    low_fr = process_data(fr_values, np.where(low_idx)[0], baseline_mask)
                    high_fr = process_data(fr_values, np.where(high_idx)[0], baseline_mask)

                    p_low, _ = cbp_one_sample(low_fr, n_permutations=N_PERM, p_threshold=P_THRESHOLD)
                    p_high, _ = cbp_one_sample(high_fr, n_permutations=N_PERM, p_threshold=P_THRESHOLD)

                    if (file_name not in bad_high) and (np.any(np.array(p_high) < 0.05)):
                        units_high.append(file_name)
                    
                        hl.append(fr_values[low_inds].mean(axis=0))
                        hh.append(fr_values[high_inds].mean(axis=0))                
                    
                        crash_h.append(fr_values[crash_high_inds].mean(axis=0))
                        avoid_h.append(fr_values[avoid_high_inds].mean(axis=0))
                        crash_h_dash.append(fr_values[crash_low_inds].mean(axis=0))
                        avoid_h_dash.append(fr_values[avoid_low_inds].mean(axis=0))
                    
                        plot_raster(st_values, fr_values, time_axis, BRAIN_REGION, file_name,
                                   out_dir=os.path.join(out_root, "Rasters"))

                    if (file_name not in bad_low) and (np.any(np.array(p_low) < 0.05)):
                        units_low.append(file_name)
                    
                        ll.append(fr_values[low_inds].mean(axis=0))
                        lh.append(fr_values[high_inds].mean(axis=0))
                    
                        crash_l.append(fr_values[crash_low_inds].mean(axis=0))
                        avoid_l.append(fr_values[avoid_low_inds].mean(axis=0))
                        crash_l_dash.append(fr_values[crash_high_inds].mean(axis=0))
                        avoid_l_dash.append(fr_values[avoid_high_inds].mean(axis=0))

            out_xlsx = os.path.join(out_root, "Data", f"{BRAIN_REGION}_{condition}_{behavior_name}.xlsx")
            with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as w:
                if units_high:
                    pd.DataFrame({"Neuron": units_high}).to_excel(w, sheet_name="High", index=False)
                if units_low:
                    pd.DataFrame({"Neuron": units_low}).to_excel(w, sheet_name="Low", index=False)
            print("Saved:", out_xlsx)

            if len(hh) > 1:
                plot_condition(
                    high=np.asarray(hh),
                    low=np.asarray(hl),
                    crash=np.asarray(crash_h),
                    avoid=np.asarray(avoid_h),
                    crash_dash=np.asarray(crash_h_dash),
                    avoid_dash=np.asarray(avoid_h_dash),
                    behavior_name=behavior_name,
                    cond="High",
                    condition=condition,
                    brain_region=BRAIN_REGION,
                    folder_path=out_root,
                )
            
            if len(lh) > 1:
                plot_condition(
                    high=np.asarray(lh),
                    low=np.asarray(ll),
                    crash=np.asarray(crash_l),
                    avoid=np.asarray(avoid_l),
                    crash_dash=np.asarray(crash_l_dash),
                    avoid_dash=np.asarray(avoid_l_dash),
                    behavior_name=behavior_name,
                    cond="Low",
                    condition=condition,
                    brain_region=BRAIN_REGION,
                    folder_path=out_root,
                )

if __name__ == "__main__":
    main()