"""
Permutation tests per neuron: bursters vs pausers
================================================

Reproduces the burster / pauser analysis used in:
“Human claustrum neurons encode uncertainty and prediction errors during aversive learning”

Generates:
    Ext. Fig. 10a-e

Inputs:
- firing-rate matrices
- spike-time structures

Outputs:
- unit classification tables
- population summary plots
- heatmaps

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
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.io
import scipy.stats as stats
from scipy.ndimage import uniform_filter1d
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["agg.path.chunksize"] = 10000
mpl.rcParams["png.fonttype"] = "none"
mpl.rcParams["patch.linewidth"] = 0

#%% Configuration
@dataclass(frozen=True)
class Config:
    base_dir: Path
    brain_region: str
    condition: str
    out_subdir: str = "path"

    t_start: float = -2.0
    t_end: float = 4.0
    baseline_win: Tuple[float, float] = (-2.0, -1.5)

    appear_win: Tuple[float, float] = (0.0, 0.5)
    event_win: Tuple[float, float] = (0.67, 1.17)

    bin_ms: int = 50
    smooth_ms: int = 200

    n_perm: int = 10_000
    alpha: float = 0.05

    appear_marker: float = 0.0
    event_marker: float = 0.67

BRAIN_REGIONS: Dict[str, List[str]] = {
    "AMY": ["subject list"]
    }

#%% Helpers
def make_time_axis(n_time: int, t_start: float, t_end: float) -> np.ndarray:
    return np.linspace(t_start, t_end, n_time)

def win_mask(time_axis: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return (time_axis >= t0) & (time_axis <= t1)

def baseline_subtract(fr: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    base = np.mean(fr[:, baseline_mask], axis=1, keepdims=True)
    return fr - base

def permutation_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    n_perm: int,
    sidedness: str,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(sample1, dtype=float)
    y = np.asarray(sample2, dtype=float)

    obs = np.nanmean(x) - np.nanmean(y)
    pooled = np.concatenate([x, y])
    n1 = len(x)

    diffs = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        idx = rng.permutation(len(pooled))[:n1]
        a = pooled[idx]
        b = np.delete(pooled, idx)
        diffs[k] = np.nanmean(a) - np.nanmean(b)

    if sidedness == "both":
        p = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1)
    elif sidedness == "smaller":
        p = (np.sum(diffs <= obs) + 1) / (n_perm + 1)
    elif sidedness == "larger":
        p = (np.sum(diffs >= obs) + 1) / (n_perm + 1)
    else:
        raise ValueError("sidedness must be 'both', 'smaller', or 'larger'")

    return float(p), float(obs)

def sem_ci_2d(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=float)
    mean_vals = np.nanmean(data, axis=0)
    n = data.shape[0]
    if n <= 1:
        sem_vals = np.zeros_like(mean_vals)
    else:
        sem_vals = np.nanstd(data, axis=0, ddof=1) / np.sqrt(n)
    return mean_vals - sem_vals, mean_vals + sem_vals

def normalize_01(fr: np.ndarray) -> np.ndarray:
    fr = np.asarray(fr, dtype=float)
    mn = np.min(fr, axis=1, keepdims=True)
    mx = np.max(fr, axis=1, keepdims=True)
    return (fr - mn) / (mx - mn + 1e-10)

def sort_by_peak_latency(fr: np.ndarray, time_axis: np.ndarray, t0: float = 0.0, t1: float = 2.0):
    fr = np.asarray(fr, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)

    post = (time_axis >= t0) & (time_axis <= t1)
    post_t = time_axis[post]
    fr_post = fr[:, post]

    peak_local = np.argmax(fr_post, axis=1)
    lat = post_t[peak_local]

    bad = ~np.isfinite(lat)
    lat2 = lat.copy()
    lat2[bad] = np.inf

    idx = np.argsort(lat2)
    return fr[idx, :], idx, lat

def gen_raster(trials, start_time=-2, end_time=4, shift: float = 0.0):
    raster = []
    for trial in trials:
        try:
            flat = np.concatenate([np.ravel(np.array(t, dtype=float)) for t in trial])
            flat = flat + shift
            raster.append(flat[(flat >= start_time) & (flat <= end_time)])
        except Exception:
            raster.append(np.array([], dtype=float))
    return raster

def plot_raster_panels(
    trials,
    fr: np.ndarray,
    outcome: np.ndarray,
    time_axis: np.ndarray,
    brain_region: str,
    file_name: str,
    out_dir: Path,
    appear_marker: float = 0.0,
    event_marker: float = 0.67,
    start_time: float = -2.0,
    end_time: float = 4.0,
    spike_shift: float = 0.0,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    raster = gen_raster(trials, start_time=start_time, end_time=end_time, shift=spike_shift)

    baseline_mask = (time_axis >= 1.5) & (time_axis <= 2.0)
    fr0 = fr - np.mean(fr[:, baseline_mask])

    bin_size = max(1, int(400 / 50))
    fr_all = uniform_filter1d(np.mean(fr0, axis=0), bin_size)

    crash_idx = np.where(outcome == 1)[0]
    avoid_idx = np.where(outcome == 0)[0]

    fr_crash = uniform_filter1d(np.mean(fr0[crash_idx], axis=0), bin_size) if len(crash_idx) else np.zeros(fr0.shape[1])
    fr_avoid = uniform_filter1d(np.mean(fr0[avoid_idx], axis=0), bin_size) if len(avoid_idx) else np.zeros(fr0.shape[1])

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(2.18, 1.4), constrained_layout=True)

    ax = axes[0, 0]
    rx, ry = [], []
    for i, tt in enumerate(raster):
        rx.extend(tt.tolist())
        ry.extend([i] * len(tt))
    ax.scatter(rx, ry, s=0.1, c="#22205F", marker="s")
    ax.axvline(appear_marker, color="red", linestyle="--", linewidth=0.5)
    ax.axvline(event_marker, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlim(start_time, end_time)
    ax.set_xticks([])
    ax.set_ylabel("trial #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)

    ax = axes[1, 0]
    ax.plot(time_axis, fr_all, color="#22205F", linewidth=0.5)
    ax.axvline(appear_marker, color="red", linestyle="--", linewidth=0.5)
    ax.axvline(event_marker, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlim(start_time, end_time)
    ax.set_xticks([])
    ax.set_ylabel(r"mean $\Delta$ firing rate (Hz)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)

    ax = axes[0, 1]
    for i, tt in enumerate(raster):
        if i >= len(outcome):
            break
        if outcome[i] == 1:
            ax.scatter(tt, [i] * len(tt), s=0.1, c="#8F39E6", marker="s")
        else:
            ax.scatter(tt, [i] * len(tt), s=0.1, c="#00cc00", marker="s")
    ax.axvline(appear_marker, color="red", linestyle="--", linewidth=0.5)
    ax.axvline(event_marker, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlim(start_time, end_time)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)

    ax = axes[1, 1]
    ax.plot(time_axis, fr_crash, color="#8F39E6", linewidth=0.5, label="Crash")
    ax.plot(time_axis, fr_avoid, color="#00cc00", linewidth=0.5, label="Avoid")
    ax.axvline(appear_marker, color="red", linestyle="--", linewidth=0.5)
    ax.axvline(event_marker, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlim(start_time, end_time)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
    ax.legend(loc="upper right", fontsize=4, frameon=False, handlelength=1)

    fname = f"{brain_region}_{file_name}"    
    fig.savefig(out_dir / f"{fname}.png", format="png", transparent=True)
    plt.close(fig)

def plot_heatmap_square(
    fr: np.ndarray,
    title: str,
    ylabel: str,
    brain_region: str,
    condition: str,
    out_dir: Path,
    time_axis: np.ndarray,
    stim_idx: int,
    tick_vals: List[float],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    n_neurons, n_time = fr.shape
    fig_width = 3.0
    bin_in = fig_width / n_time
    fig_height = max(0.4, bin_in * n_neurons)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(fr, cmap="inferno", interpolation="nearest", aspect="equal", origin="lower")

    ax.axvline(x=stim_idx, color="cyan", linestyle="--", lw=0.5)

    tick_idx = [int(np.argmin(np.abs(time_axis - t))) for t in tick_vals]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(t) for t in tick_vals], fontsize=5, fontname="Arial")

    ax.set_yticks(np.arange(n_neurons))
    ax.set_yticklabels(np.arange(1, n_neurons + 1), fontsize=5, fontname="Arial")

    ax.set_xlabel("Time (s)", fontsize=5, fontname="Arial")
    ax.set_ylabel(ylabel, fontsize=5, fontname="Arial")
    ax.set_title(title, fontsize=5, fontname="Arial")

    cbar = fig.colorbar(im, ax=ax, label="normalized firing rate")
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.yaxis.label.set_fontsize(5)
    cbar.ax.yaxis.label.set_fontname("Arial")

    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

    safe_title = title.replace(" ", "_")    
    fig.savefig(out_dir / f"{brain_region}_{condition}_{safe_title}_heatmap.png", dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)

def plot_bursters_pausers(
    bursters: np.ndarray,
    pausers: np.ndarray,
    bursters_crash: np.ndarray,
    bursters_avoid: np.ndarray,
    pausers_crash: np.ndarray,
    pausers_avoid: np.ndarray,
    condition: str,
    brain_region: str,
    out_dir: Path,
    time_axis: np.ndarray,
    baseline_mask: np.ndarray,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    event_mask: np.ndarray,
    smooth_bins: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    color_main = "#22205F"
    color_crash = "#8F39E6"
    color_avoid = "#00cc00"

    def z_by_base(A: np.ndarray) -> np.ndarray:
        base = A[:, baseline_mask]
        mu = np.mean(base, axis=1, keepdims=True)
        sd = np.std(base, axis=1, keepdims=True)
        sd = np.where(sd == 0, 1, sd)
        return (A - mu) / sd

    bur_z = uniform_filter1d(z_by_base(bursters), smooth_bins, axis=1) if len(bursters) else np.zeros((0, len(time_axis)))
    pau_z = uniform_filter1d(z_by_base(pausers), smooth_bins, axis=1) if len(pausers) else np.zeros((0, len(time_axis)))

    bur_ci = sem_ci_2d(bur_z) if len(bursters) else (np.zeros(len(time_axis)), np.zeros(len(time_axis)))
    pau_ci = sem_ci_2d(pau_z) if len(pausers) else (np.zeros(len(time_axis)), np.zeros(len(time_axis)))

    burA = uniform_filter1d(z_by_base(bursters_avoid), smooth_bins, axis=1) if len(bursters_avoid) else np.zeros((0, len(time_axis)))
    burC = uniform_filter1d(z_by_base(bursters_crash), smooth_bins, axis=1) if len(bursters_crash) else np.zeros((0, len(time_axis)))
    pauA = uniform_filter1d(z_by_base(pausers_avoid), smooth_bins, axis=1) if len(pausers_avoid) else np.zeros((0, len(time_axis)))
    pauC = uniform_filter1d(z_by_base(pausers_crash), smooth_bins, axis=1) if len(pausers_crash) else np.zeros((0, len(time_axis)))

    burA_ci = sem_ci_2d(burA) if len(bursters_avoid) else (np.zeros(len(time_axis)), np.zeros(len(time_axis)))
    burC_ci = sem_ci_2d(burC) if len(bursters_crash) else (np.zeros(len(time_axis)), np.zeros(len(time_axis)))
    pauA_ci = sem_ci_2d(pauA) if len(pausers_avoid) else (np.zeros(len(time_axis)), np.zeros(len(time_axis)))
    pauC_ci = sem_ci_2d(pauC) if len(pausers_crash) else (np.zeros(len(time_axis)), np.zeros(len(time_axis)))

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5

    fig, ax = plt.subplots(3, 1, figsize=(0.95, 2.7))
    plt.subplots_adjust(hspace=0.25)

    ax[0].plot(time_axis, bur_z.mean(axis=0) if len(bur_z) else 0*time_axis, color=color_main, lw=0.5)
    ax[0].plot(time_axis, pau_z.mean(axis=0) if len(pau_z) else 0*time_axis, color=color_main, lw=0.5, ls="--")
    ax[0].fill_between(time_axis, bur_ci[0], bur_ci[1], color=color_main, alpha=0.25, edgecolor="none")
    ax[0].fill_between(time_axis, pau_ci[0], pau_ci[1], color=color_main, alpha=0.25, edgecolor="none")
    ax[0].axvline(0, color="red", ls="--", lw=0.5)
    ax[0].axvline(0.67, color="gray", ls="--", lw=0.5)
    ax[0].set_xlim(time_axis[0], time_axis[-1])
    ax[0].set_xticks([])
    ax[0].set_title(f"bursters (n={len(bursters)}), pausers (n={len(pausers)})", fontsize=5)

    ax[1].plot(time_axis, burA.mean(axis=0) if len(burA) else 0*time_axis, color=color_avoid, lw=0.5)
    ax[1].plot(time_axis, burC.mean(axis=0) if len(burC) else 0*time_axis, color=color_crash, lw=0.5)
    ax[1].fill_between(time_axis, burA_ci[0], burA_ci[1], color=color_avoid, alpha=0.25, edgecolor="none")
    ax[1].fill_between(time_axis, burC_ci[0], burC_ci[1], color=color_crash, alpha=0.25, edgecolor="none")

    ax[1].plot(time_axis, pauA.mean(axis=0) if len(pauA) else 0*time_axis, color=color_avoid, lw=0.5, ls="--")
    ax[1].plot(time_axis, pauC.mean(axis=0) if len(pauC) else 0*time_axis, color=color_crash, lw=0.5, ls="--")
    ax[1].fill_between(time_axis, pauA_ci[0], pauA_ci[1], color=color_avoid, alpha=0.25, edgecolor="none")
    ax[1].fill_between(time_axis, pauC_ci[0], pauC_ci[1], color=color_crash, alpha=0.25, edgecolor="none")

    ax[1].axvline(0, color="red", ls="--", lw=0.5)
    ax[1].axvline(0.67, color="gray", ls="--", lw=0.5)
    ax[1].set_xlim(time_axis[0], time_axis[-1])
    ax[1].set_xticks(np.arange(-2, 5, 2))

    combined_raw = np.concatenate([np.abs(bursters - bursters[:, baseline_mask].mean(axis=1, keepdims=True)),
                                   np.abs(pausers - pausers[:, baseline_mask].mean(axis=1, keepdims=True))], axis=0)
    pre_vals = np.mean(combined_raw[:, pre_mask], axis=1)
    post_vals = np.mean(combined_raw[:, post_mask], axis=1)

    res = stats.wilcoxon(pre_vals, post_vals, alternative="two-sided")
    p_prepost = float(res.pvalue)

    positions = [1, 2, 4, 5]
    ax[2].scatter(np.ones_like(pre_vals) * positions[0], pre_vals, 5, facecolors="white", edgecolors="black",
                  marker="^", linewidths=0.125, zorder=2)
    ax[2].scatter(np.ones_like(post_vals) * positions[1], post_vals, 5, facecolors="white", edgecolors=color_main,
                  marker="^", linewidths=0.125, zorder=2)

    for b, e in zip(pre_vals, post_vals):
        ax[2].plot([positions[0], positions[1]], [b, e], color="#cccccc", alpha=0.5, lw=0.3, zorder=1)

    ax[2].scatter(positions[0], np.mean(pre_vals), s=30, c="black", marker="_", linewidths=0.5, zorder=2)
    ax[2].scatter(positions[1], np.mean(post_vals), s=30, c=color_main, marker="_", linewidths=0.5, zorder=2)
    ax[2].plot([positions[0], positions[1]], [np.mean(pre_vals), np.mean(post_vals)], "-k", lw=0.5, zorder=1)

    sem_pre = stats.sem(pre_vals, nan_policy="omit")
    sem_post = stats.sem(post_vals, nan_policy="omit")
    ax[2].text(
        1.5,
        0,
        f"{'*' if p_prepost < 0.05 else 'ns'} (p={p_prepost:.4f})\nPre: {np.mean(pre_vals):.4f}±{sem_pre:.4f}\nPost: {np.mean(post_vals):.4f}±{sem_post:.4f}",
        ha="center",
        fontsize=5,
        linespacing=1.1,
    )

    crash_stat = np.concatenate([np.mean(np.abs(bursters_crash - bursters_crash[:, baseline_mask].mean(axis=1, keepdims=True))[:, event_mask], axis=1),
                                 np.mean(np.abs(pausers_crash - pausers_crash[:, baseline_mask].mean(axis=1, keepdims=True))[:, event_mask], axis=1)], axis=0) if len(bursters_crash) or len(pausers_crash) else np.array([])
    avoid_stat = np.concatenate([np.mean(np.abs(bursters_avoid - bursters_avoid[:, baseline_mask].mean(axis=1, keepdims=True))[:, event_mask], axis=1),
                                 np.mean(np.abs(pausers_avoid - pausers_avoid[:, baseline_mask].mean(axis=1, keepdims=True))[:, event_mask], axis=1)], axis=0) if len(bursters_avoid) or len(pausers_avoid) else np.array([])

    if len(crash_stat) and len(avoid_stat) and len(crash_stat) == len(avoid_stat):
        res2 = stats.wilcoxon(avoid_stat, crash_stat, alternative="two-sided")
        p_ac = float(res2.pvalue)
    else:
        p_ac = np.nan

    ax[2].scatter(np.ones_like(avoid_stat) * positions[2], avoid_stat, 5, facecolors="white", edgecolors=color_avoid,
                  marker="^", linewidths=0.125, zorder=2)
    ax[2].scatter(np.ones_like(crash_stat) * positions[3], crash_stat, 5, facecolors="white", edgecolors=color_crash,
                  marker="^", linewidths=0.125, zorder=2)

    for b, e in zip(avoid_stat, crash_stat):
        ax[2].plot([positions[2], positions[3]], [b, e], color="#cccccc", alpha=0.5, lw=0.3, zorder=1)

    if len(avoid_stat):
        ax[2].scatter(positions[2], np.mean(avoid_stat), s=30, c=color_avoid, marker="_", linewidths=0.5, zorder=2)
    if len(crash_stat):
        ax[2].scatter(positions[3], np.mean(crash_stat), s=30, c=color_crash, marker="_", linewidths=0.5, zorder=2)
    if len(avoid_stat) and len(crash_stat):
        ax[2].plot([positions[2], positions[3]], [np.mean(avoid_stat), np.mean(crash_stat)], "-k", lw=0.5, zorder=1)

    if np.isfinite(p_ac):
        sem_avoid = stats.sem(avoid_stat, nan_policy="omit")
        sem_crash = stats.sem(crash_stat, nan_policy="omit")
        ax[2].text(
            4.5,
            0,
            f"{'*' if p_ac < 0.05 else 'ns'} (p={p_ac:.4f})\nAvoid: {np.mean(avoid_stat):.4f}±{sem_avoid:.4f}\nCrash: {np.mean(crash_stat):.4f}±{sem_crash:.4f}",
            ha="center",
            fontsize=5,
            linespacing=1.1,
        )

    ax[2].set_xticks([1, 2, 4, 5])
    ax[2].set_xticklabels(["Pre", "Post", "Avoid", "Crash"], fontsize=5)
    ax[2].set_ylabel("mean |Δ| rate")

    for a in ax:
        for sp in a.spines.values():
            sp.set_linewidth(0.3)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(axis="both", which="both", length=1, width=0.5, pad=1)

    fig.savefig(out_dir / f"{brain_region}_{condition}.png", dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    gc.collect()

def get_data_root(cfg: Config) -> Path:
    return cfg.base_dir / "Data" / cfg.condition

def good_neurons_path(cfg: Config) -> Path:
    return cfg.base_dir / f"good_neurons_{cfg.brain_region}.csv"

def behavior_xlsx_path(cfg: Config, subject: str) -> Path:
    data_root = get_data_root(cfg)
    return data_root / cfg.brain_region / subject / f"{cfg.brain_region}_{subject}.xlsx"

def mat_dir(cfg: Config, subject: str) -> Path:
    data_root = get_data_root(cfg)
    return data_root / cfg.brain_region / subject

def out_root(cfg: Config) -> Path:
    return cfg.base_dir / "CBP" / cfg.brain_region / cfg.out_subdir

#%% Main
def main(cfg: Config) -> None:
    subjects = BRAIN_REGIONS.get(cfg.brain_region, [])
    if not subjects:
        raise ValueError(f"Unknown brain region: {cfg.brain_region}")

    outdir = out_root(cfg)
    (outdir / "Data").mkdir(parents=True, exist_ok=True)
    (outdir / "Rasters").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    if cfg.condition == "Appear":
        a_s, a_e = cfg.appear_win
        spike_shift = 0.0
    elif cfg.condition == "Event":
        a_s, a_e = cfg.event_win
        spike_shift = 0.67
    else:
        raise ValueError("condition must be 'Appear' or 'Event'")

    unit_ids: List[str] = []
    fr_heatmap: List[np.ndarray] = []

    bursters: List[np.ndarray] = []
    bursters_units: List[str] = []
    bursters_crash: List[np.ndarray] = []
    bursters_avoid: List[np.ndarray] = []
    p_bursters: List[float] = []

    pausers: List[np.ndarray] = []
    pausers_units: List[str] = []
    pausers_crash: List[np.ndarray] = []
    pausers_avoid: List[np.ndarray] = []
    p_pausers: List[float] = []

    good_all = pd.read_csv(good_neurons_path(cfg))

    for subject in subjects:
        gn = good_all.loc[good_all["goodneurons"].astype(str).str.contains(subject, na=False)]
        file_names = gn.iloc[:, 0].astype(str).tolist()

        beh_df = pd.read_excel(behavior_xlsx_path(cfg, subject))
        outcome = beh_df["outcome"].to_numpy(dtype=int)

        mdir = mat_dir(cfg, subject)

        for file_name in file_names:
            fr_path = mdir / f"{cfg.brain_region}_{file_name}.mat"
            st_path = mdir / f"{cfg.brain_region}_{file_name}_spike_times.mat"

            fr_values = scipy.io.loadmat(fr_path)["fr"] 
            st_values = scipy.io.loadmat(st_path)["spike_times"] 

            time_axis = make_time_axis(fr_values.shape[1], cfg.t_start, cfg.t_end)
            baseline_mask = win_mask(time_axis, *cfg.baseline_win)
            event_mask = win_mask(time_axis, a_s, a_e)

            fr_bs = baseline_subtract(fr_values, baseline_mask)

            p_b, _ = permutation_test(
                np.mean(fr_bs[:, event_mask], axis=1),
                np.mean(fr_bs[:, baseline_mask], axis=1),
                n_perm=cfg.n_perm,
                sidedness="larger",
                rng=rng,
            )
            p_p, _ = permutation_test(
                np.mean(fr_bs[:, event_mask], axis=1),
                np.mean(fr_bs[:, baseline_mask], axis=1),
                n_perm=cfg.n_perm,
                sidedness="smaller",
                rng=rng,
            )

            p_bursters.append(p_b)
            p_pausers.append(p_p)
            unit_ids.append(file_name)
            fr_heatmap.append(np.mean(fr_bs, axis=0))

            if p_b < cfg.alpha:
                bursters.append(np.mean(fr_values, axis=0))
                bursters_units.append(file_name)
                bursters_crash.append(np.mean(fr_values[outcome == 1], axis=0))
                bursters_avoid.append(np.mean(fr_values[outcome == 0], axis=0))

            if p_p < cfg.alpha:
                pausers.append(np.mean(fr_values, axis=0))
                pausers_units.append(file_name)
                pausers_crash.append(np.mean(fr_values[outcome == 1], axis=0))
                pausers_avoid.append(np.mean(fr_values[outcome == 0], axis=0))

            plot_raster_panels(
                trials=st_values,
                fr=fr_values,
                outcome=outcome,
                time_axis=time_axis,
                brain_region=cfg.brain_region,
                file_name=file_name,
                out_dir=outdir / "Rasters",
                appear_marker=cfg.appear_marker,
                event_marker=cfg.event_marker,
                start_time=cfg.t_start,
                end_time=cfg.t_end,
                spike_shift=spike_shift,
            )

    rej_b, _, _, _ = multipletests(p_bursters, alpha=cfg.alpha, method="fdr_bh")
    rej_p, _, _, _ = multipletests(p_pausers, alpha=cfg.alpha, method="fdr_bh")

    units_b = set(np.array(unit_ids)[rej_b].tolist())
    units_p = set(np.array(unit_ids)[rej_p].tolist())

    mask_b = np.array([u in units_b for u in bursters_units], dtype=bool)
    mask_p = np.array([u in units_p for u in pausers_units], dtype=bool)

    bursters = list(np.array(bursters)[mask_b]) if len(bursters) else []
    bursters_units = list(np.array(bursters_units)[mask_b]) if len(bursters_units) else []
    bursters_crash = list(np.array(bursters_crash)[mask_b]) if len(bursters_crash) else []
    bursters_avoid = list(np.array(bursters_avoid)[mask_b]) if len(bursters_avoid) else []

    pausers = list(np.array(pausers)[mask_p]) if len(pausers) else []
    pausers_units = list(np.array(pausers_units)[mask_p]) if len(pausers_units) else []
    pausers_crash = list(np.array(pausers_crash)[mask_p]) if len(pausers_crash) else []
    pausers_avoid = list(np.array(pausers_avoid)[mask_p]) if len(pausers_avoid) else []

    xlsx_path = outdir / "Data" / f"{cfg.brain_region}_{cfg.condition}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
        if bursters_units:
            pd.DataFrame({"Neuron": bursters_units}).to_excel(w, sheet_name="bursters", index=False)
        if pausers_units:
            pd.DataFrame({"Neuron": pausers_units}).to_excel(w, sheet_name="pausers", index=False)

    baseline_mask = win_mask(time_axis, *cfg.baseline_win)
    pre_mask = win_mask(time_axis, -0.5, -0.05)
    post_mask = win_mask(time_axis, a_s, a_e)
    event_mask = post_mask

    smooth_bins = max(1, int(cfg.smooth_ms / cfg.bin_ms))

    plot_bursters_pausers(
        bursters=np.asarray(bursters),
        pausers=np.asarray(pausers),
        bursters_crash=np.asarray(bursters_crash),
        bursters_avoid=np.asarray(bursters_avoid),
        pausers_crash=np.asarray(pausers_crash),
        pausers_avoid=np.asarray(pausers_avoid),
        condition=cfg.condition,
        brain_region=cfg.brain_region,
        out_dir=outdir,
        time_axis=time_axis,
        baseline_mask=baseline_mask,
        pre_mask=pre_mask,
        post_mask=post_mask,
        event_mask=event_mask,
        smooth_bins=smooth_bins,
    )

    fr_all = normalize_01(np.vstack(fr_heatmap))
    fr_bp = normalize_01(np.vstack([np.vstack(bursters) if len(bursters) else np.zeros((0, fr_all.shape[1])),
                                    np.vstack(pausers) if len(pausers) else np.zeros((0, fr_all.shape[1]))]))

    fr_sorted, _, _ = sort_by_peak_latency(fr_all, time_axis, t0=0.0, t1=2.0)

    tick_vals = [-2, -1, 0, 1, 2, 4]
    stim_idx = int(np.argmin(np.abs(time_axis - 0)))

    plot_heatmap_square(
        fr=fr_sorted,
        title="All neurons sorted by peak latency",
        ylabel="Neurons (sorted)",
        brain_region=cfg.brain_region,
        condition=cfg.condition,
        out_dir=outdir,
        time_axis=time_axis,
        stim_idx=stim_idx,
        tick_vals=tick_vals,
    )
    plot_heatmap_square(
        fr=fr_bp,
        title="Bursters + Pausers",
        ylabel="Neurons (Bursters + Pausers)",
        brain_region=cfg.brain_region,
        condition=cfg.condition,
        out_dir=outdir,
        time_axis=time_axis,
        stim_idx=stim_idx,
        tick_vals=tick_vals,
    )

    print(f"Saved: {xlsx_path}")
    print(f"Figures: {outdir}")
    gc.collect()

if __name__ == "__main__":
    cfg = Config(
        base_dir=Path(r"path"),
        brain_region="brain region",
        condition="condition",
    )
    main(cfg)