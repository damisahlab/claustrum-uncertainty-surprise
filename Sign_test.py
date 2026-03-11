#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:52:29 2026

@author: mingyuehu
"""
import pandas as pd
import numpy as np
import scipy.stats as st

# -----------------------------
# Load workbook
# -----------------------------
path = "Ming_TaskRespANOVA.xlsx"
cla = pd.read_excel(path, sheet_name="CLA")
acc = pd.read_excel(path, sheet_name="ACC")

# -----------------------------
# Build neuron-level table
# Assumes:
# col A = subject
# col B + C together help define the neuron
# col F = HIT firing rate
# col H = MISS firing rate
# -----------------------------
def make_wide(df, region_name):
    out = pd.DataFrame({
        "subject": df.iloc[:, 0].astype(str),
        "field_b": df.iloc[:, 1].astype(str),
        "field_c": df.iloc[:, 2].astype(str),
        "region": region_name,
        "fr_hit": pd.to_numeric(df.iloc[:, 5], errors="coerce"),
        "fr_miss": pd.to_numeric(df.iloc[:, 7], errors="coerce"),
    })

    out["neuron"] = (
        out["subject"] + "_" + out["field_b"] + "_" + out["field_c"]
    )

    # Neuron-wise outcome preference index
    # Positive = MISS-preferring, Negative = HIT-preferring
    out["delta"] = out["fr_miss"] - out["fr_hit"]

    return out

cla_wide = make_wide(cla, "CLA")
acc_wide = make_wide(acc, "ACC")

# -----------------------------
# Optional: restrict to task-responsive neurons only
# If your sheets already contain only task-responsive neurons,
# leave this as True and do nothing else.
# If not, add your own filter here.
# -----------------------------
use_all_rows_as_task_responsive = True

if not use_all_rows_as_task_responsive:
    raise ValueError("Add your task-responsive filter here before running the analysis.")

dat = pd.concat([cla_wide, acc_wide], ignore_index=True)

# -----------------------------
# Bootstrap CI helper
# -----------------------------
def bootstrap_ci(x, func=np.mean, n_boot=10000, alpha=0.05, random_state=123):
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan

    boots = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        boots[i] = func(sample)

    low = np.quantile(boots, alpha / 2)
    high = np.quantile(boots, 1 - alpha / 2)
    return low, high

# -----------------------------
# Region-wise heterogeneity analysis
# -----------------------------
def analyze_region(df_region):
    delta = df_region["delta"].dropna().to_numpy()

    n_total = len(delta)
    n_pos = int(np.sum(delta > 0))
    n_neg = int(np.sum(delta < 0))
    n_zero = int(np.sum(delta == 0))

    # Exact two-sided sign test (binomial test), excluding ties
    n_nonzero = n_pos + n_neg
    if n_nonzero > 0:
        sign_res = st.binomtest(k=n_pos, n=n_nonzero, p=0.5, alternative="two-sided")
        p_sign = sign_res.pvalue
    else:
        p_sign = np.nan

    # Wilcoxon signed-rank test against 0
    # Uses only nonzero differences by default under scipy's "wilcox"/"pratt" options
    try:
        wilc = st.wilcoxon(delta, zero_method="wilcox", alternative="two-sided", mode="auto")
        W = wilc.statistic
        p_wilc = wilc.pvalue
    except Exception:
        W = np.nan
        p_wilc = np.nan

    # One-sample t-test against 0 (optional)
    try:
        t_res = st.ttest_1samp(delta, popmean=0.0, nan_policy="omit")
        t_stat = t_res.statistic
        p_t = t_res.pvalue
    except Exception:
        t_stat = np.nan
        p_t = np.nan

    # Summary statistics
    mean_delta = np.nanmean(delta)
    median_delta = np.nanmedian(delta)
    sd_delta = np.nanstd(delta, ddof=1) if len(delta) > 1 else np.nan

    mean_ci_low, mean_ci_high = bootstrap_ci(delta, func=np.mean, n_boot=10000, alpha=0.05, random_state=123)
    med_ci_low, med_ci_high = bootstrap_ci(delta, func=np.median, n_boot=10000, alpha=0.05, random_state=456)

    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "sd_delta": sd_delta,
        "mean_ci_low": mean_ci_low,
        "mean_ci_high": mean_ci_high,
        "median_ci_low": med_ci_low,
        "median_ci_high": med_ci_high,
        "W_wilcoxon": W,
        "p_wilcoxon": p_wilc,
        "t_stat": t_stat,
        "p_ttest": p_t,
        "p_sign": p_sign,
    }

# -----------------------------
# Run analysis
# -----------------------------
results = {}
for region_name in ["CLA", "ACC"]:
    results[region_name] = analyze_region(dat.loc[dat["region"] == region_name].copy())

# -----------------------------
# Print results
# -----------------------------
for region_name, res in results.items():
    print(f"\n=== {region_name} ===")
    print(f"Total neurons: {res['n_total']}")
    print(f"Positive delta (MISS-preferring): {res['n_pos']}")
    print(f"Negative delta (HIT-preferring): {res['n_neg']}")
    print(f"Zero delta: {res['n_zero']}")
    print(f"Mean delta: {res['mean_delta']:.6f}")
    print(f"Median delta: {res['median_delta']:.6f}")
    print(f"Bootstrap 95% CI for mean: [{res['mean_ci_low']:.6f}, {res['mean_ci_high']:.6f}]")
    print(f"Bootstrap 95% CI for median: [{res['median_ci_low']:.6f}, {res['median_ci_high']:.6f}]")
    print(f"Wilcoxon signed-rank: W = {res['W_wilcoxon']}, p = {res['p_wilcoxon']:.6g}")
    print(f"One-sample t-test: t = {res['t_stat']:.6f}, p = {res['p_ttest']:.6g}")
    print(f"Exact binomial sign test: p = {res['p_sign']:.6g}")

# -----------------------------
# Optional subject-level summary
# This can help show whether effects vary across subjects
# -----------------------------
subject_summary = (
    dat.groupby(["region", "subject"], as_index=False)["delta"]
       .mean()
       .rename(columns={"delta": "mean_delta_by_subject"})
)

print("\n=== SUBJECT-LEVEL MEAN DELTA ===")
print(subject_summary.sort_values(["region", "subject"]))

# Optional paired subject-level comparison for subjects who have both CLA and ACC
pivot_subj = subject_summary.pivot(index="subject", columns="region", values="mean_delta_by_subject")
pivot_subj = pivot_subj.dropna(subset=["CLA", "ACC"], how="any")

if len(pivot_subj) >= 2:
    try:
        subj_w = st.wilcoxon(pivot_subj["ACC"], pivot_subj["CLA"], alternative="two-sided", mode="auto")
        print("\n=== SUBJECT-LEVEL ACC vs CLA DELTA COMPARISON ===")
        print(f"Paired Wilcoxon: W = {subj_w.statistic}, p = {subj_w.pvalue:.6g}")
    except Exception as e:
        print("\nSubject-level paired Wilcoxon failed:", e)

    try:
        subj_t = st.ttest_rel(pivot_subj["ACC"], pivot_subj["CLA"], nan_policy="omit")
        print(f"Paired t-test: t = {subj_t.statistic:.6f}, p = {subj_t.pvalue:.6g}")
    except Exception as e:
        print("Subject-level paired t-test failed:", e)
