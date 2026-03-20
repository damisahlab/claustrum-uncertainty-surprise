"""
Behavior distributions per subject
==================================

Reproduces the behavior distributions used in:
“Human claustrum neurons encode uncertainty and prediction errors during aversive learning”

Generates:
    Ext. Fig. 9b

Inputs:
- per-subject behavioral spreadsheets

Outputs:
- distribution plots

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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

#%% Configuration
@dataclass(frozen=True)
class Config:
    base_dir: Path
    subjects: tuple[str, ...] = ("subject list")
    input_rel: Path = Path("path")
    output_rel: Path = Path("path")
    columns_to_keep: tuple[str, ...] = (
        "subject", "trial", "outcome", "A_safety_variance", "B_safety_variance"
    )
    low_percentile: float = 0.30
    kde_bw: float = 0.15
    hist_bins: int = 20
    kde_points: int = 1000
    figsize: tuple[float, float] = (1.5, 1.0)
    font_family: str = "Arial"
    font_size: int = 5
    axes_linewidth: float = 0.5

#%% Helpers
def behavior_file(cfg: Config, subject: str) -> Path:
    return cfg.base_dir / cfg.input_rel / f"{subject}.xlsx"

def save_dual(fig: plt.Figure, out_svg: Path, out_png: Path) -> None:
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", format="png")

def plot_hist_kde_percentiles(
    data: np.ndarray,
    metric_name: str,
    p_low: float,
    p_high: float,
    cfg: Config,
) -> plt.Figure:
    kde = gaussian_kde(data, bw_method=cfg.kde_bw)
    x = np.linspace(float(np.min(data)), float(np.max(data)), cfg.kde_points)
    y = kde(x)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.hist(
        data,
        bins=cfg.hist_bins,
        alpha=0.7,
        color="gray",
        edgecolor="black",
        linewidth=0.3,
        density=True,
    )
    ax.plot(x, y, color="red", linewidth=0.5, label="KDE")

    ax.axvline(p_low, color="blue", linestyle="dashed", linewidth=0.5, label=f"P30 = {p_low:.4f}")
    ax.axvline(p_high, color="orange", linestyle="dashed", linewidth=0.5, label=f"P70 = {p_high:.4f}")

    ax.set_xlabel(metric_name, labelpad=1)
    ax.set_ylabel("Density", labelpad=1)
    ax.locator_params(axis="x", nbins=3)
    ax.locator_params(axis="y", nbins=3)
    ax.legend(frameon=False, loc="upper right", fontsize=4)

    for spine in ax.spines.values():
        spine.set_linewidth(cfg.axes_linewidth)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=1, width=0.5, pad=1)

    fig.tight_layout()
    return fig

#%% Main
def main(cfg: Config) -> None:
    plt.rcParams["font.family"] = cfg.font_family
    plt.rcParams["font.size"] = cfg.font_size
    plt.rcParams["axes.linewidth"] = cfg.axes_linewidth

    out_dir = cfg.base_dir / cfg.output_rel
    out_dir.mkdir(parents=True, exist_ok=True)

    high_percentile = 1.0 - cfg.low_percentile

    for sub in cfg.subjects:
        fp = behavior_file(cfg, sub)
        if not fp.exists():
            print(f"[WARN] File not found: {fp}")
            continue

        df = pd.read_excel(fp)

        missing = [c for c in cfg.columns_to_keep if c not in df.columns]
        if missing:
            raise KeyError(f"{fp} is missing columns: {missing}")

        df = df[list(cfg.columns_to_keep)]

        metric_cols = df.columns[3:]

        for metric in metric_cols:
            series = df[metric].dropna()
            if series.empty:
                print(f"[WARN] No data for {sub}-{metric}")
                continue

            data = series.to_numpy(dtype=float)
            p_low = float(np.quantile(data, cfg.low_percentile))
            p_high = float(np.quantile(data, high_percentile))

            fig = plot_hist_kde_percentiles(
                data=data,
                metric_name=str(metric),
                p_low=p_low,
                p_high=p_high,
                cfg=cfg,
            )

            out_png = out_dir / f"{sub}-{metric}.png"
            save_dual(fig, out_png)
            plt.close(fig)

if __name__ == "__main__":
    cfg = Config(
        base_dir=Path(r"path"),
    )
    main(cfg)
