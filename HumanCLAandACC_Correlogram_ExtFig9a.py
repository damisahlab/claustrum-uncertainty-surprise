"""
Behavioral correlograms (per-subject) as circle correlograms (lower triangle)
======================================================================

What this script does
---------------------
This script reproduces correlograms used in the study
**“Human claustrum neurons encode uncertainty and prediction errors during aversive learning”**.
and generates:
    Ext. Fig. 9a
    
This script:
1) Loads per-subject behavioral spreadsheets.
2) Computes Pearson correlation matrices on selected columns.
3) Plots a grid of circle correlograms with a shared colorbar.
4) Save figures.

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
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import seaborn as sns

#%% Configuration
@dataclass(frozen=True)
class Config:
    base_dir: Path
    brain_region: str
    data_condition: str = "Appear"
    out_dir: Path = Path("path")
    out_prefix: str = "correlograms"
    
    n_cols: int = 4
    fig_size: tuple = (6, 4.5)
    
    columns: tuple = (
        "A_safety_variance",
        "B_safety_variance",
        "A_absolute_prediction_error",
        "B_absolute_prediction_error",
    )

BRAIN_REGIONS: Dict[str, List[str]] = {    
    "ACC": ["subject list"],
    "CLA": ["subject list"]
}

#%% Helpers
def plot_correlogram_axes(
    ax: plt.Axes,
    corr: np.ndarray,
    labels: Sequence[str],
    title: str = "",
    cmap=None,
) -> None:
    labels = list(labels)
    n = len(labels)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

    for i in range(n + 1):
        ax.axhline(i, color="lightgray", linewidth=0.5)
        ax.axvline(i, color="lightgray", linewidth=0.5)

    if cmap is None:
        cmap = sns.color_palette("vlag", as_cmap=True).reversed()

    for i in range(n):
        for j in range(i + 1):
            r = float(corr[i, j])
            color = cmap((r + 1) / 2.0)

            radius = np.sqrt(abs(r)) * 0.45

            circ = plt.Circle(
                (j + 0.5, i + 0.5),
                radius=radius,
                color=color,
                ec="none",
                lw=0,
            )
            ax.add_patch(circ)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def subject_behavior_path(cfg: Config, subject: str) -> Path:
    return (
        cfg.base_dir
        / "Data"
        / cfg.data_condition
        / cfg.brain_region
        / subject
        / f"{cfg.brain_region}_{subject}.xlsx"
    )

def compute_corr_matrices(cfg: Config, subjects: Sequence[str]) -> List[np.ndarray]:
    corr_mats: List[np.ndarray] = []
    cols = list(cfg.columns)

    for sub in subjects:
        fp = subject_behavior_path(cfg, sub)
        df = pd.read_excel(fp)

        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{fp} is missing columns: {missing}")

        data = df[cols]
        corr = data.corr(method="pearson").to_numpy()
        corr_mats.append(corr)

    return corr_mats

#%% Main
def main(cfg: Config) -> Path:
    subjects = BRAIN_REGIONS.get(cfg.brain_region, [])
    if not subjects:
        raise ValueError(f"Unknown brain_region='{cfg.brain_region}'. Options: {list(BRAIN_REGIONS)}")

    corr_matrices = compute_corr_matrices(cfg, subjects)

    n_mats = len(corr_matrices)
    n_cols = cfg.n_cols
    n_rows = int(np.ceil(n_mats / n_cols))

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=cfg.fig_size)
    axes = np.array(axes).reshape(-1)

    cmap = sns.color_palette("vlag", as_cmap=True).reversed()

    for i in range(n_rows * n_cols):
        if i < n_mats:
            plot_correlogram_axes(
                axes[i],
                corr_matrices[i],
                cfg.columns,
                title=subjects[i],
                cmap=cmap,
            )
        else:
            axes[i].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
    norm = Normalize(vmin=-1, vmax=1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax, label="Pearson r")

    fig.suptitle(f"Correlograms – {cfg.brain_region}", fontsize=14)

    out_dir = cfg.base_dir / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.out_prefix}_{cfg.brain_region}.png"
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path

if __name__ == "__main__":
    cfg = Config(
        base_dir=Path(r"path"),
        brain_region="", # CLA, ACC
        data_condition="Appear",
        out_dir=Path("path"),
    )
    main(cfg)