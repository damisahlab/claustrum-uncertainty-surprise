"""
Displacement (CDT) and standard deviation analysis across subjects
=================================================================

Reproduces the displacement and standard deviation analysis used in:
“Human claustrum neurons encode uncertainty and prediction errors during aversive learning”

Generates:
    Fig. 5b

Inputs:
- subject- and condition-level .mat files

Outputs:
- summary plots

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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import scipy.io
import scipy.stats as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#%% Configuration
@dataclass(frozen=True)
class Config:
    base_dir: Path
    input_rel: Path = Path("path")
    output_rel: Path = Path("path")

    subjects: tuple[str, ...] = ("subject list")
    conditions: tuple[str, ...] = ("A_high", "A_low", "B_high", "B_low")

    disp_col: int = 1
    sd_col: int = 2

    font_family: str = "Arial"
    font_size: int = 5
    axes_linewidth: float = 0.5

#%% helpers
def wilcoxon_signrank(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    d = x - y
    mask = d != 0
    x_nz, y_nz = x[mask], y[mask]
    n = int(mask.sum())

    if n == 0:
        return np.nan, np.nan, 1.0, 0, np.nan

    Wplus, p = stats.wilcoxon(
        x_nz,
        y_nz,
        zero_method="wilcox",
        correction=False,
        alternative=alternative,
        mode="auto",
    )

    total_rank = n * (n + 1) / 2
    W = min(Wplus, total_rank - Wplus)

    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    Z = (W - mu) / sigma

    sign = np.sign(np.median(d[mask]))
    r_rb = sign * (1 - (2 * W) / (n * (n + 1)))

    return float(W), float(Z), float(p), n, float(r_rb)

def plot_metric_panel(ax: plt.Axes, low_vals, high_vals, p_value: float, z_value: float, ylabel: str) -> None:
    low_vals = np.asarray(low_vals, dtype=float)
    high_vals = np.asarray(high_vals, dtype=float)

    valid = np.isfinite(low_vals) & np.isfinite(high_vals)
    low_vals = low_vals[valid]
    high_vals = high_vals[valid]

    if low_vals.size == 0:
        ax.text(0.5, 0.5, "No valid paired data", ha="center", va="center")
        ax.set_axis_off()
        return

    mean_low = float(np.mean(low_vals))
    sd_low = float(np.std(low_vals, ddof=1)) if low_vals.size > 1 else 0.0
    mean_high = float(np.mean(high_vals))
    sd_high = float(np.std(high_vals, ddof=1)) if high_vals.size > 1 else 0.0

    ax.scatter(np.ones_like(low_vals), low_vals, 6, facecolors="none", edgecolors="blue",
               marker="^", linewidths=0.2)
    ax.scatter(3 * np.ones_like(high_vals), high_vals, 6, facecolors="none", edgecolors="orange",
               marker="^", linewidths=0.2)

    ax.scatter(1, mean_low, 30, "blue", marker="_", edgecolors="blue", linewidths=0.5)
    ax.scatter(3, mean_high, 30, "orange", marker="_", edgecolors="orange", linewidths=0.5)

    for lo, hi in zip(low_vals, high_vals):
        ax.plot([1, 3], [lo, hi], color="#cccccc", linewidth=0.075)
    ax.plot([1, 3], [mean_low, mean_high], "-k", linewidth=0.5)

    ax.set_xlim([0, 4])
    ax.set_xticks([1, 3])
    ax.set_xticklabels(["low", "high"])

    yl_min = min(float(np.min(low_vals)), float(np.min(high_vals)))
    yl_max = max(float(np.max(low_vals)), float(np.max(high_vals)))
    m = (yl_max - yl_min) if yl_max > yl_min else 1.0
    ax.set_ylim([yl_min - 0.1 * m, yl_max + 0.25 * m])

    sig_label = "**" if p_value < 0.05 else "ns"
    txt = (
        f"{sig_label} (p={p_value:.3f})\n"
        f"Low: {mean_low:.2f}±{sd_low:.2f}\n"
        f"High: {mean_high:.2f}±{sd_high:.2f}\n"
        f"Z-value: {z_value:.2f}"
    )
    ax.text(1.5, yl_max + 0.125 * m, txt, ha="center", va="bottom")

    ax.set_ylabel(ylabel)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.3)
    ax.spines["bottom"].set_linewidth(0.3)
    ax.tick_params(axis="both", which="both", length=1, width=0.5, pad=1)
    
def mat_path(cfg: Config, subject: str, condition: str) -> Path:
    return cfg.base_dir / cfg.input_rel / f"sub{subject}_{condition}.mat"

def load_condition_matrix(fp: Path, varname: str) -> np.ndarray:
    M = scipy.io.loadmat(fp)
    if varname not in M:
        raise KeyError(f"Variable '{varname}' not found in {fp.name}")
    return np.squeeze(M[varname])

def extract_col_from_mat(mat_arr: np.ndarray, col_idx: int) -> np.ndarray:
    a = np.asarray(mat_arr)
    a = np.squeeze(a)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f"Expected 2D array (n×3). Got shape {a.shape}")
    v = a[:, col_idx].astype(float)
    return v[np.isfinite(v)]

#%% Main
def main(cfg: Config) -> None:
    plt.rcParams["font.family"] = cfg.font_family
    plt.rcParams["font.size"] = cfg.font_size
    plt.rcParams["axes.linewidth"] = cfg.axes_linewidth

    out_dir = cfg.base_dir / cfg.output_rel
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_A_low, mean_A_high, mean_B_low, mean_B_high = [], [], [], []
    meanSD_A_low, meanSD_A_high, meanSD_B_low, meanSD_B_high = [], [], [], []

    for subject in cfg.subjects:
        data: Dict[str, np.ndarray] = {}

        ok = True
        for condition in cfg.conditions:
            fp = mat_path(cfg, subject, condition)
            if not fp.exists():
                print(f"[WARN] Missing file: {fp}")
                ok = False
                break
            try:
                data[condition] = load_condition_matrix(fp, varname=condition)
            except Exception as e:
                print(f"[WARN] Skipping sub{subject} {condition}: {e}")
                ok = False
                break

        if not ok or not all(k in data for k in cfg.conditions):
            print(f"[WARN] Skipping sub{subject}: missing one or more conditions.")
            continue

        A_low_disp = extract_col_from_mat(data["A_low"], cfg.disp_col)
        A_high_disp = extract_col_from_mat(data["A_high"], cfg.disp_col)
        B_low_disp = extract_col_from_mat(data["B_low"], cfg.disp_col)
        B_high_disp = extract_col_from_mat(data["B_high"], cfg.disp_col)

        A_low_sd = extract_col_from_mat(data["A_low"], cfg.sd_col)
        A_high_sd = extract_col_from_mat(data["A_high"], cfg.sd_col)
        B_low_sd = extract_col_from_mat(data["B_low"], cfg.sd_col)
        B_high_sd = extract_col_from_mat(data["B_high"], cfg.sd_col)

        mean_A_low.append(np.mean(A_low_disp))
        mean_A_high.append(np.mean(A_high_disp))
        mean_B_low.append(np.mean(B_low_disp))
        mean_B_high.append(np.mean(B_high_disp))

        meanSD_A_low.append(np.mean(A_low_sd))
        meanSD_A_high.append(np.mean(A_high_sd))
        meanSD_B_low.append(np.mean(B_low_sd))
        meanSD_B_high.append(np.mean(B_high_sd))

    mean_A_low = np.asarray(mean_A_low, dtype=float)
    mean_A_high = np.asarray(mean_A_high, dtype=float)
    mean_B_low = np.asarray(mean_B_low, dtype=float)
    mean_B_high = np.asarray(mean_B_high, dtype=float)

    meanSD_A_low = np.asarray(meanSD_A_low, dtype=float)
    meanSD_A_high = np.asarray(meanSD_A_high, dtype=float)
    meanSD_B_low = np.asarray(meanSD_B_low, dtype=float)
    meanSD_B_high = np.asarray(meanSD_B_high, dtype=float)

    mean_low_all = np.r_[mean_A_low, mean_B_low]
    mean_high_all = np.r_[mean_A_high, mean_B_high]

    meanSD_low_all = np.r_[meanSD_A_low, meanSD_B_low]
    meanSD_high_all = np.r_[meanSD_A_high, meanSD_B_high]

    W_m, Z_m, p_m, n_m, r_m = wilcoxon_signrank(mean_high_all, mean_low_all)
    W_s, Z_s, p_s, n_s, r_s = wilcoxon_signrank(meanSD_high_all, meanSD_low_all)

    fig, ax = plt.subplots(1, 1, figsize=(0.78, 1.15))
    plot_metric_panel(ax, mean_low_all, mean_high_all, p_m, Z_m, ylabel="mean CDT")
    ax.set_xlabel("uncertainty", labelpad=1)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_cdt.png", dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(0.78, 1.15))
    plot_metric_panel(ax, meanSD_low_all, meanSD_high_all, p_s, Z_s, ylabel="mean SD (CDT)")
    ax.set_xlabel("uncertainty", labelpad=1)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_SD.png", dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)

    print(f"Saved plots to: {out_dir}")
    print(f"[Displacement] n={n_m}, W={W_m:.3f}, Z={Z_m:.3f}, p={p_m:.4g}, r_rb={r_m:.3f}")
    print(f"[SD]           n={n_s}, W={W_s:.3f}, Z={Z_s:.3f}, p={p_s:.4g}, r_rb={r_s:.3f}")

if __name__ == "__main__":
    cfg = Config(
        base_dir=Path("path"),
    )
    main(cfg)
