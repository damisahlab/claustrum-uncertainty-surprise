"""
Claustrum / ACC / AMY SVM decoding
=================================

Reproduces the SVM decoding analysis used in:
“Human claustrum neurons encode uncertainty and prediction errors during aversive learning”

Generates:
    Ext. Fig. 6h, 7h, 10h    
    Table 4

Inputs:
- firing-rate matrices
- trial-level latent variables

Outputs:
- per-neuron decoding tables
- summary tables
- panel figure

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
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.base import clone

from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

#%% User Settings
BRAIN_REGION = "CLA"  # "ACC", "CLA", "AMY"
BRAIN_REGIONS = {
    "AMY": ["subject list"],
    "ACC": ["subject list"],
    "CLA": ["subject list"]
}
SUB_NEURONS_CSV = r"path"
DATA_ROOT = r"path"
SAVE_PATH = r"path"
BEHAVIORS = ["behaviors"]
LOW_PERCENTILE = 0.30
HIGH_PERCENTILE = 1.0 - LOW_PERCENTILE
N_REPEATS = 100
TEST_SIZE = 0.20
DO_SHUFFLE = True
N_PERM_SIGNFLIP = 10000
FDR_ALPHA = 0.05
SVM_KERNEL = "linear"
N_JOBS = -1 

#%% Helpers
def split_datset(X, y, clf, n_repeats=50, test_size=0.2,
                 do_shuffle=True, n_jobs=-1):
    splitter = StratifiedShuffleSplit(n_splits=n_repeats,
                                      test_size=test_size, random_state=None)
    splits = list(splitter.split(X, y))

    def one_split(k, tr_idx, te_idx):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        nte = len(yte)

        model = clone(clf)
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        bacc_t = balanced_accuracy_score(yte, yhat)
        acc_t  = accuracy_score(yte, yhat)
        K_t    = int(np.sum(yhat == yte))
        cm_t   = confusion_matrix(yte, yhat, labels=[0, 1])
        
        bacc_s = np.nan
        acc_s  = np.nan
        K_s    = 0
        cm_s   = np.zeros((2, 2), dtype=int)
        if do_shuffle:
            rng = np.random.default_rng()
            ytr_shuf = rng.permutation(ytr)
            model_s = clone(clf)
            model_s.fit(Xtr, ytr_shuf)
            yhat_s = model_s.predict(Xte)
            bacc_s = balanced_accuracy_score(yte, yhat_s)
            acc_s  = accuracy_score(yte, yhat_s)
            K_s    = int(np.sum(yhat_s == yte))
            cm_s   = confusion_matrix(yte, yhat_s, labels=[0, 1])

        return k, bacc_t, acc_t, bacc_s, acc_s, K_t, nte, K_s, cm_t, cm_s

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(one_split)(k, tr, te) for k, (tr, te) in enumerate(splits)
    )

    bacc_true = np.empty(n_repeats, float)
    acc_true  = np.empty(n_repeats, float)
    bacc_shuf = np.full(n_repeats, np.nan, float)
    acc_shuf  = np.full(n_repeats, np.nan, float)

    K_true = 0
    N_true = 0
    K_shuf = 0
    N_shuf = 0

    cm_true_agg = np.zeros((2, 2), dtype=int)
    cm_shuf_agg = np.zeros((2, 2), dtype=int)

    for k, b_t, a_t, b_s, a_s, K_t, nte, K_s, cm_t, cm_s in results:
        bacc_true[k] = b_t
        acc_true[k]  = a_t
        bacc_shuf[k] = b_s
        acc_shuf[k]  = a_s

        K_true += K_t
        N_true += nte
        cm_true_agg += cm_t

        if do_shuffle:
            K_shuf += K_s
            N_shuf += nte
            cm_shuf_agg += cm_s

    return {
        "bacc_true": bacc_true,
        "acc_true": acc_true,
        "bacc_shuf": bacc_shuf,
        "acc_shuf": acc_shuf,
        "K_true": K_true,
        "N_true": N_true,
        "acc_true_agg": (K_true / N_true) if N_true > 0 else np.nan,
        "K_shuf": K_shuf,
        "N_shuf": N_shuf,
        "acc_shuf_agg": (K_shuf / N_shuf) if N_shuf > 0 else np.nan,
        "cm_true_agg": cm_true_agg,
        "cm_shuf_agg": cm_shuf_agg,
    }

def paired_signflip_pvalue(x, y, n_perm=10000, alternative="greater"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    d = x[m] - y[m]
    if d.size == 0:
        return np.nan, np.nan
    obs = d.mean()
    rng = np.random.default_rng()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size), replace=True)
    perm_stats = (signs * d).mean(axis=1)
    if alternative == "greater":
        p = (1 + np.sum(perm_stats >= obs)) / (n_perm + 1)
    elif alternative == "less":
        p = (1 + np.sum(perm_stats <= obs)) / (n_perm + 1)
    else:
        p = (1 + np.sum(np.abs(perm_stats) >= abs(obs))) / (n_perm + 1)
    return float(p), float(obs)

def bootstrap_ci_mean(x, n_boot=10000, ci=95, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo = np.percentile(boots, (100-ci)/2)
    hi = np.percentile(boots, 100-(100-ci)/2)
    return float(x.mean()), float(lo), float(hi)

def cohen_dz_paired(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    d = x[m] - y[m]
    if d.size < 2:
        return np.nan
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else np.nan

def aggregate_cm_f1(df_sub):
    TN = int(df_sub["cm_TN"].sum())
    FP = int(df_sub["cm_FP"].sum())
    FN = int(df_sub["cm_FN"].sum())
    TP = int(df_sub["cm_TP"].sum())
    cm = np.array([[TN, FP],
                   [FN, TP]], dtype=int)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return cm, float(f1)

def build_clf(kernel: str = "linear") -> Pipeline:
    if kernel == "linear":
        return Pipeline([("scaler", StandardScaler()),
                         ("svm", SVC(kernel="linear", class_weight="balanced", probability=False))])
    if kernel == "rbf":
        return Pipeline([("scaler", StandardScaler()),
                         ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                                     class_weight="balanced", probability=False))])
    if kernel == "poly":
        return Pipeline([("scaler", StandardScaler()),
                         ("svm", SVC(kernel="poly", degree=1, coef0=1, C=1.0,
                                     class_weight="balanced", probability=False))])
    raise ValueError(f"Unknown kernel: {kernel}")

def run_svm_pipeline(
    brain_region: str,
    subjects: list[str],
    sub_neurons_csv: str,
    data_root: str,
    save_path: str,
    behaviors: list[str],
    low_percentile: float,
    n_repeats: int,
    test_size: float,
    do_shuffle: bool,
    n_perm_signflip: int,
    fdr_alpha: float,
    clf_kernel: str,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    os.makedirs(save_path, exist_ok=True)

    sub_neurons_path = sub_neurons_csv.format(brain_region=brain_region)
    sub_neurons = pd.read_csv(sub_neurons_path)

    clf = build_clf(clf_kernel)

    svm_rows = []
    cnt = 0

    for behavior_name in behaviors:
        for sub in subjects:
            good_neurons = sub_neurons.loc[sub_neurons["goodneurons"].astype(str).str.contains(sub, na=False)]
            file_names = good_neurons.iloc[:, 0].astype(str)

            mat_dir = os.path.join(data_root, brain_region, sub)
            behavior_xlsx = os.path.join(data_root, brain_region, sub, f"{brain_region}_{sub}.xlsx")
            behavior_df = pd.read_excel(behavior_xlsx)

            if behavior_name == "prediction_error" and "prediction_error" not in behavior_df.columns:
                behavior_df["prediction_error"] = (
                    behavior_df["A_absolute_prediction_error"] + behavior_df["B_absolute_prediction_error"]
                ) / 2.0
            if behavior_name == "uncertainty" and "uncertainty" not in behavior_df.columns:
                behavior_df["uncertainty"] = (
                    behavior_df["A_safety_variance"] + behavior_df["B_safety_variance"]
                ) / 2.0

            beh = behavior_df[behavior_name].to_numpy()

            low_th  = np.nanquantile(beh, low_percentile)
            high_th = np.nanquantile(beh, 1.0 - low_percentile)

            for file_name in file_names:
                cnt += 1
                print(f"{cnt} | {brain_region} | {sub} | {file_name} | {behavior_name}")

                fr_path = os.path.join(mat_dir, f"{brain_region}_{file_name}.mat")
                fr = scipy.io.loadmat(fr_path)["fr"]

                low_idx  = beh <= low_th
                high_idx = beh >= high_th
                keep_idx = low_idx | high_idx

                labels = np.zeros(int(np.sum(keep_idx)), dtype=int)
                labels[high_idx[keep_idx]] = 1
                fr_keep = fr[keep_idx, :]

                time_vector = np.linspace(-4, 2, fr_keep.shape[1])
                intertrial_idx = (time_vector >= -4) & (time_vector < 0)
                asteroid_idx   = (time_vector >= 0) & (time_vector <= 1.5)

                intertrial = fr_keep[:, intertrial_idx]
                asteroid   = fr_keep[:, asteroid_idx]

                intertrial = intertrial[:, :(intertrial.shape[1] // 10) * 10].reshape(intertrial.shape[0], -1, 10).mean(axis=2)
                asteroid   = asteroid[:,   :(asteroid.shape[1]   // 10) * 10].reshape(asteroid.shape[0],   -1, 10).mean(axis=2)

                res_it = split_datset(intertrial, labels, clf, n_repeats=n_repeats,
                                      test_size=test_size, do_shuffle=do_shuffle, n_jobs=n_jobs)
                res_ast = split_datset(asteroid, labels, clf, n_repeats=n_repeats,
                                       test_size=test_size, do_shuffle=do_shuffle, n_jobs=n_jobs)

                p_it, eff_it = paired_signflip_pvalue(res_it["acc_true"], res_it["acc_shuf"],
                                                      n_perm=n_perm_signflip, alternative="greater")
                p_ast, eff_ast = paired_signflip_pvalue(res_ast["acc_true"], res_ast["acc_shuf"],
                                                        n_perm=n_perm_signflip, alternative="greater")

                cm_it = res_it["cm_true_agg"]
                svm_rows.append({
                    "neuron": file_name, "behavior": behavior_name, "epoch": "intertrial",
                    "bal_acc_mean": float(np.mean(res_it["bacc_true"])),
                    "bal_acc_sd": float(np.std(res_it["bacc_true"], ddof=1)),
                    "bal_acc_shuf_mean": float(np.mean(res_it["bacc_shuf"])),
                    "bal_acc_shuf_sd": float(np.std(res_it["bacc_shuf"], ddof=1)),
                    "acc_agg": float(res_it["acc_true_agg"]),
                    "acc_shuf_agg": float(res_it["acc_shuf_agg"]),
                    "K": int(res_it["K_true"]), "N": int(res_it["N_true"]),
                    "p_value": float(p_it),
                    "cm_TN": int(cm_it[0,0]), "cm_FP": int(cm_it[0,1]),
                    "cm_FN": int(cm_it[1,0]), "cm_TP": int(cm_it[1,1]),
                    "eff_size": float(eff_it),
                })

                cm_ast = res_ast["cm_true_agg"]
                svm_rows.append({
                    "neuron": file_name, "behavior": behavior_name, "epoch": "asteroid",
                    "bal_acc_mean": float(np.mean(res_ast["bacc_true"])),
                    "bal_acc_sd": float(np.std(res_ast["bacc_true"], ddof=1)),
                    "bal_acc_shuf_mean": float(np.mean(res_ast["bacc_shuf"])),
                    "bal_acc_shuf_sd": float(np.std(res_ast["bacc_shuf"], ddof=1)),
                    "acc_agg": float(res_ast["acc_true_agg"]),
                    "acc_shuf_agg": float(res_ast["acc_shuf_agg"]),
                    "K": int(res_ast["K_true"]), "N": int(res_ast["N_true"]),
                    "p_value": float(p_ast),
                    "cm_TN": int(cm_ast[0,0]), "cm_FP": int(cm_ast[0,1]),
                    "cm_FN": int(cm_ast[1,0]), "cm_TP": int(cm_ast[1,1]),
                    "eff_size": float(eff_ast),
                })

    svm_df = pd.DataFrame(svm_rows)

    svm_df["p_fdr"] = np.nan
    svm_df["sig_fdr"] = False
    for beh in svm_df["behavior"].unique():
        for ep in svm_df["epoch"].unique():
            mask = (svm_df["behavior"] == beh) & (svm_df["epoch"] == ep)
            pvals = svm_df.loc[mask, "p_value"].values
            if len(pvals) == 0:
                continue
            reject, pvals_fdr, _, _ = multipletests(pvals, alpha=fdr_alpha, method="fdr_bh")
            svm_df.loc[mask, "p_fdr"] = pvals_fdr
            svm_df.loc[mask, "sig_fdr"] = reject

    summary_tbl = make_summary_table(svm_df, brain_region=brain_region, ci=95, n_boot=10000, seed=0)

    return svm_df, summary_tbl

def make_summary_table(svm_df, brain_region, ci=95, n_boot=10000, seed=0):
    rows = []
    if not isinstance(svm_df, pd.DataFrame):
        svm_df = pd.DataFrame(svm_df)

    for (ep, beh), df_cond in svm_df.groupby(["epoch", "behavior"], sort=True):
        x_true = df_cond["acc_agg"].to_numpy(float)
        x_shuf = df_cond["acc_shuf_agg"].to_numpy(float)
        
        m_t, lo_t, hi_t = bootstrap_ci_mean(x_true, n_boot=n_boot, ci=ci, seed=seed)
        m_s, lo_s, hi_s = bootstrap_ci_mean(x_shuf, n_boot=n_boot, ci=ci, seed=seed)
        
        half_t = (hi_t - lo_t) / 2.0
        half_s = (hi_s - lo_s) / 2.0
        
        true_txt = f"{m_t:.3f} ± {half_t:.3f}"
        shuf_txt = f"{m_s:.3f} ± {half_s:.3f}"

        dz = cohen_dz_paired(x_true, x_shuf)

        p_fdr_min = np.nan
        if "p_fdr" in df_cond.columns:
            p_fdr_min = float(np.nanmin(df_cond["p_fdr"].to_numpy(float)))

        rows.append({
            "Brain Region": brain_region,
            "Epoch": ep,
            "Latent Variable": beh,
            "True: mean ± CI": true_txt,
            "Shuffle: mean ± CI": shuf_txt,
            "FDR p-value": (f"{p_fdr_min:.3g}" if np.isfinite(p_fdr_min) else "NA"),
            "Cohen's d": (f"{dz:.2f}" if np.isfinite(dz) else "NA"),
        })

    out = pd.DataFrame(rows)

    epoch_order = ["intertrial", "asteroid"]
    beh_order = ["uncertainty", "prediction_error"]
    out["Epoch"] = pd.Categorical(out["Epoch"], categories=epoch_order, ordered=True)
    out["Latent Variable"] = pd.Categorical(out["Latent Variable"], categories=beh_order, ordered=True)
    out = out.sort_values(["Epoch", "Latent Variable"]).reset_index(drop=True)
    return out

def plot_fig(svm_df, brain_region, save_path,
             epochs=("intertrial", "asteroid"),
             behaviors=("uncertainty", "prediction_error"),
             beh_colors=None,
             ci=95,
             normalize_cm="true",
             pmax=0.05,
             top_width=0.70,
             mid_width=0.70,
             mid_height=0.70):

    if beh_colors is None:
        beh_colors = {"uncertainty": "#B1A0A1", "prediction_error": "#AF9BCA"}

    col_specs = [
        ("intertrial", "uncertainty"),
        ("intertrial", "prediction_error"),
        ("asteroid",   "uncertainty"),
        ("asteroid",   "prediction_error"),
    ]

    fig = plt.figure(figsize=(5, 3.5))
    gs = fig.add_gridspec(
        nrows=3, ncols=4,
        wspace=0.05, hspace=0.05
    )

    def centered_subax(parent_spec, width_frac=0.7, height_frac=1.0):
        w_margin = max(0.0, (1.0 - width_frac) / 2.0)
        h_margin = max(0.0, (1.0 - height_frac) / 2.0)

        if height_frac >= 0.999:
            sub = parent_spec.subgridspec(
                1, 3,
                width_ratios=[w_margin, width_frac, w_margin]
            )
            return sub[0, 1]
        else:
            sub = parent_spec.subgridspec(
                3, 3,
                height_ratios=[h_margin, height_frac, h_margin],
                width_ratios=[w_margin, width_frac, w_margin]
            )
            return sub[1, 1]

    for j, (ep, beh) in enumerate(col_specs):
        df_cond = svm_df[(svm_df["epoch"] == ep) & (svm_df["behavior"] == beh)].copy()

        ax_stats = fig.add_subplot(centered_subax(gs[0, j], width_frac=top_width, height_frac=1.0))
        plot_stats_panel(ax_stats, df_cond, beh, ep, beh_colors, ci=ci)
        n_neurons = len(df_cond)
        ax_stats.text(
            0.5, 0.98, f"n={n_neurons}",
            transform=ax_stats.transAxes,
            ha="right", va="top",
            fontsize=5)

        ax_stats.set_title(f"{ep}\n{beh})", fontsize=5, pad=8)

        if j != 0:
            ax_stats.set_ylabel("")

        ax_cm = fig.add_subplot(centered_subax(gs[1, j], width_frac=mid_width, height_frac=mid_height))
        if df_cond.empty:
            ax_cm.axis("off")
            ax_cm.text(0.5, 0.5, "no neurons",
                       ha="center", va="center", transform=ax_cm.transAxes, fontsize=5)
        else:
            cm, _ = aggregate_cm_f1(df_cond)     
            plot_cm(ax_cm, cm, normalize=normalize_cm, chance=0.5,
                below="#2B6CB0", above="#C53030",
                class_names=("Low", "High"),
                tight_labels=True,
                show_cbar=False)

            if j != 0:
                ax_cm.set_ylabel("")
            ax_cm.set_xlabel("Predicted", fontsize=5)

        ax_vol = fig.add_subplot(centered_subax(gs[2, j], width_frac=top_width, height_frac=1.0))
        plot_volcano_hsegments_panel(ax_vol, df_cond, beh, ep, beh_colors, pmax=pmax)        

    fig.suptitle(f"{brain_region} — True vs Shuffle (stats), Confusion Matrix, and Radial p_fdr Map",
                 fontsize=8, y=0.995)

    plt.tight_layout()
    outp = os.path.join(save_path, f"{brain_region}_plot.png")
    plt.savefig(outp, dpi=300)
    plt.show()
    print("Saved:", outp)

def plot_stats_panel(ax, df_cond, beh, ep, beh_colors, ci=95):
    style_ax(ax)
    ax.tick_params(axis='both', which='major', pad=1, length=2, width=0.25)

    ax.set_xlim(0, 4)
    ax.set_xticks([1, 3])
    ax.set_xticklabels(["True", "Shuffle"], fontsize=5)
    ax.set_ylabel("Accuracy", fontsize=5)

    if df_cond.empty:
        ax.set_title(f"{ep} - {beh}\nno neurons", pad=4, fontsize=5)
        ax.text(0.5, 0.5, "no neurons", ha="center", va="center",
                transform=ax.transAxes, fontsize=5)
        return

    x_true = df_cond["acc_agg"].to_numpy(float)
    x_shuf = df_cond["acc_shuf_agg"].to_numpy(float)

    m_t, lo_t, hi_t = bootstrap_ci_mean(x_true, ci=ci)
    m_s, lo_s, hi_s = bootstrap_ci_mean(x_shuf, ci=ci)

    p_signflip, obs = paired_signflip_pvalue(x_true, x_shuf, n_perm=10000, alternative="greater")

    dz = cohen_dz_paired(x_true, x_shuf)

    yerr_true = np.array([[m_t - lo_t], [hi_t - m_t]])
    yerr_shuf = np.array([[m_s - lo_s], [hi_s - m_s]])

    true_color = beh_colors.get(beh, "#000000")

    ax.errorbar([1], [m_t], yerr=yerr_true, fmt='none',
                ecolor=true_color, elinewidth=0.8, capsize=2, capthick=0.8, zorder=2)
    ax.plot([0.7, 1.3], [m_t, m_t], color=true_color, linewidth=2.5,
            solid_capstyle='butt', zorder=3)

    ax.errorbar([3], [m_s], yerr=yerr_shuf, fmt='none',
                ecolor="#9E9E9E", elinewidth=0.8, capsize=2, capthick=0.8, zorder=2)
    ax.plot([2.7, 3.3], [m_s, m_s], color="#9E9E9E", linewidth=2.5,
            solid_capstyle='butt', zorder=3)

    y_max = max(hi_t, hi_s)
    y_bar = y_max + 0.0025
    ax.plot([1, 3], [y_bar, y_bar], color='black', linewidth=0.5, zorder=4)
    ax.plot([1, 1], [y_bar-0.001, y_bar+0.001], color='black', linewidth=0.5, zorder=4)
    ax.plot([3, 3], [y_bar-0.001, y_bar+0.001], color='black', linewidth=0.5, zorder=4)

    if p_signflip < 0.001:
        sig_text = "***"
    elif p_signflip < 0.01:
        sig_text = "**"
    elif p_signflip < 0.05:
        sig_text = "*"
    else:
        sig_text = "ns"

    ax.text(2, y_bar, sig_text, ha='center', va='bottom', fontsize=5)

    lines = [
        f"p={p_signflip:.3g}",
        f"d\u209B={dz:.2f}",
        f"True: {m_t:.3f} [{lo_t:.3f},{hi_t:.3f}]",
        f"Shuf: {m_s:.3f} [{lo_s:.3f},{hi_s:.3f}]",
    ]
    annotate_stats_block(ax, lines, loc="upper right")
    ax.set_title(f"{ep} - {beh}\nn={len(df_cond)}", pad=4, fontsize=5)

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.3)
    ax.spines["bottom"].set_linewidth(0.3)
    ax.tick_params(axis="both", which="both", width=0.3, length=2)

def annotate_stats_block(ax, lines, loc="upper right"):
    if loc == "upper right":
        x, ha = 0.98, "right"
    else:
        x, ha = 0.02, "left"
    ax.text(
        x, 0.98, "\n".join(lines),
        transform=ax.transAxes,
        ha=ha, va="top",
        fontsize=5,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.6", linewidth=0.4, alpha=0.95)
    )

def plot_cm(ax, cm, normalize="true", chance=0.5,
            below="#2B6CB0", above="#C53030",
            show_cbar=False,
            class_names=("Low", "High"),
            tight_labels=True):

    cm = np.asarray(cm, dtype=float)

    if normalize == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        disp = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        fmt = "{:.2f}"
    else:
        disp = cm
        fmt = "{:.0f}"

    n = disp.shape[0]

    cmap = LinearSegmentedColormap.from_list(
        "below_white_above",
        [(0.0, below), (0.5, "white"), (1.0, above)],
        N=256
    )

    if normalize == "true":
        norm = TwoSlopeNorm(vmin=0.0, vcenter=chance, vmax=1.0)
        im = ax.imshow(disp, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        im = ax.imshow(disp, cmap="Blues", interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    if class_names is None or len(class_names) != n:
        class_names = [str(i) for i in range(n)]

    ax.set_xticklabels(class_names, fontsize=5)
    ax.set_yticklabels(class_names, fontsize=5)

    if tight_labels:
        ax.tick_params(axis="x", which="both", pad=1, length=2, width=0.25)
        ax.tick_params(axis="y", which="both", pad=1, length=2, width=0.25)
        ax.set_xlabel("Pred", fontsize=5, labelpad=1)
        ax.set_ylabel("True", fontsize=5, labelpad=1)
    else:
        ax.tick_params(labelsize=5, length=2, width=0.25)
        ax.set_xlabel("Predicted", fontsize=5)
        ax.set_ylabel("True", fontsize=5)

    if normalize == "true":
        for i in range(n):
            for j in range(n):
                v = disp[i, j]
                dist = abs(v - chance)
                ax.text(j, i, fmt.format(v),
                        ha="center", va="center", fontsize=5,
                        color=("white" if dist > 0.20 else "black"))
    else:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, fmt.format(disp[i, j]),
                        ha="center", va="center", fontsize=5, color="black")

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.3")

    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=5)
        if normalize == "true":
            cbar.set_label("Row-normalized", fontsize=5)
            cbar.set_ticks([0.0, chance, 1.0])
            cbar.set_ticklabels([f"{0.0:.2f}", f"{chance:.2f}", f"{1.0:.2f}"])
        else:
            cbar.set_label("Count", fontsize=5)
    return im

def plot_volcano_hsegments_panel(ax, df_subset, beh, ep, beh_colors, pmax=0.05):
    style_ax(ax)
    ax.tick_params(axis='both', which='major', pad=1, length=2, width=0.25)

    if df_subset.empty:
        ax.set_title("no neurons", fontsize=5, pad=4)
        ax.text(0.5, 0.5, "no neurons", ha="center", va="center",
                transform=ax.transAxes, fontsize=5)
        return mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, pmax),
                                     cmap=mpl.cm.get_cmap("inferno"))

    df = df_subset.copy()
    df["delta"] = df["acc_agg"].to_numpy(float) - df["acc_shuf_agg"].to_numpy(float)

    delta = df["delta"].to_numpy(float)
    acc = df["acc_agg"].to_numpy(float)

    p = df["p_fdr"].to_numpy(float)
    p = np.where(np.isfinite(p), p, pmax)
    p = np.clip(p, 0.0, pmax)

    sig = df["sig_fdr"].to_numpy(bool)

    order = np.argsort(delta)
    delta, acc, p, sig = delta[order], acc[order], p[order], sig[order]

    cmap = mpl.cm.get_cmap("inferno")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=pmax)

    for d, a, is_s in zip(delta, acc, sig):
        if not is_s:
            ax.hlines(a, 0, d, color=(0.75, 0.75, 0.75, 0.35),
                      linewidth=1.6, zorder=1)

    for d, a, pv, is_s in zip(delta, acc, p, sig):
        if is_s:
            col = cmap(norm(pv))
            ax.hlines(a, 0, d, color=col, linewidth=2.2, zorder=3)
    
    ax.axvline(0, color='black', linewidth=0.6, alpha=0.35, zorder=0)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=.5, alpha=0.7, zorder=0)

    xmin, xmax = np.nanmin(delta), np.nanmax(delta)
    xr = max(1e-6, xmax - xmin)
    ax.set_xlim(xmin - 0.12 * xr, xmax + 0.12 * xr)

    ymin, ymax = np.nanmin(acc), np.nanmax(acc)
    yr = max(1e-6, ymax - ymin)
    ax.set_ylim(ymin - 0.12 * yr, ymax + 0.12 * yr)

    ax.set_xlabel("Δ Accuracy \n(True − Shuffle)", fontsize=5)
    ax.set_ylabel("Accuracy", fontsize=5)
    ax.set_title(f"{sig.sum()}/{len(df)}", fontsize=5, pad=4)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm

#%% Main
def main():
    os.makedirs(SAVE_PATH, exist_ok=True)

    subjects = BRAIN_REGIONS.get(BRAIN_REGION, [])
    if not subjects:
        raise ValueError(f"No subjects defined for brain_region='{BRAIN_REGION}'")

    svm_df, summary_tbl = run_svm_pipeline(
        brain_region=BRAIN_REGION,
        subjects=subjects,
        sub_neurons_csv=SUB_NEURONS_CSV,
        data_root=DATA_ROOT,
        save_path=SAVE_PATH,
        behaviors=BEHAVIORS,
        low_percentile=LOW_PERCENTILE,
        n_repeats=N_REPEATS,
        test_size=TEST_SIZE,
        do_shuffle=DO_SHUFFLE,
        n_perm_signflip=N_PERM_SIGNFLIP,
        fdr_alpha=FDR_ALPHA,
        clf_kernel=SVM_KERNEL,
        n_jobs=N_JOBS,
    )

    out_csv = os.path.join(SAVE_PATH, f"SVM_{BRAIN_REGION}.csv")
    svm_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    out_sum = os.path.join(SAVE_PATH, f"SVM_summary_{BRAIN_REGION}.csv")
    summary_tbl.to_csv(out_sum, index=False)
    print("Saved:", out_sum)

    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 5,
        "axes.titlesize": 5, "axes.labelsize": 5,
        "xtick.labelsize": 5, "ytick.labelsize": 5,
        "legend.fontsize": 5
    })
    plot_fig(
        svm_df, BRAIN_REGION, SAVE_PATH,
        beh_colors={"uncertainty": "#B1A0A1", "prediction_error": "#AF9BCA"},
        top_width=0.60, mid_width=0.60, mid_height=0.60
    )

if __name__ == "__main__":
    main()