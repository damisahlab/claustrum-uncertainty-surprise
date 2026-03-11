#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:48:53 2026

@author: mingyuehu
"""
#This repository contains the mixed-effects model analysis for CLA and ACC firing rates.
#Files:
#- Ming_TaskRespANOVA.xlsx: input data

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as st

# firing rate depends on region, outcome, 
#and whether the effect of outcome changes depending on the region, 
#while also accounting for repeated measures within neurons and clustering within subjects.

# -----------------------------
# Load workbook
# -----------------------------
path = "Ming_TaskRespANOVA.xlsx"
cla = pd.read_excel(path, sheet_name="CLA")
acc = pd.read_excel(path, sheet_name="ACC")

# -----------------------------
# Reshape to long format
# Assumes:
# col A = subject
# col B + C help define neuron identity
# col F = HIT firing rate
# col H = MISS firing rate
# -----------------------------
def make_long(df, region_name):
    wide = pd.DataFrame({
        "subject": df.iloc[:, 0].astype(str),
        "field_b": df.iloc[:, 1].astype(str),
        "field_c": df.iloc[:, 2].astype(str),
        "region": region_name,
        "fr_hit": pd.to_numeric(df.iloc[:, 5], errors="coerce"),
        "fr_miss": pd.to_numeric(df.iloc[:, 7], errors="coerce"),
    })

    # Unique neuron ID from subject + B + C
    wide["neuron"] = (
        wide["subject"] + "_" + wide["field_b"] + "_" + wide["field_c"]
    )

    long = pd.concat([
        wide[["subject", "neuron", "region", "fr_hit"]]
            .rename(columns={"fr_hit": "fr"})
            .assign(outcome="HIT"),
        wide[["subject", "neuron", "region", "fr_miss"]]
            .rename(columns={"fr_miss": "fr"})
            .assign(outcome="MISS")
    ], ignore_index=True)

    return long

cla_long = make_long(cla, "CLA")
acc_long = make_long(acc, "ACC")
dat = pd.concat([cla_long, acc_long], ignore_index=True)

dat["region"] = pd.Categorical(dat["region"], categories=["CLA", "ACC"])
dat["outcome"] = pd.Categorical(dat["outcome"], categories=["HIT", "MISS"])

# -----------------------------
# Full and reduced models
# -----------------------------
def fit_mixed_model(df, response_col="fr"):
    full = smf.mixedlm(
        f"{response_col} ~ region * outcome",
        df,
        groups=df["subject"],
        vc_formula={"neuron": "0 + C(neuron)"},
        re_formula="1"
    ).fit(method="lbfgs", reml=False)

    reduced = smf.mixedlm(
        f"{response_col} ~ region + outcome",
        df,
        groups=df["subject"],
        vc_formula={"neuron": "0 + C(neuron)"},
        re_formula="1"
    ).fit(method="lbfgs", reml=False)

    lrt = 2 * (full.llf - reduced.llf)
    p_lrt = st.chi2.sf(lrt, df=1)

    return full, reduced, lrt, p_lrt

full_model, reduced_model, lrt, p_lrt = fit_mixed_model(dat)

print("\n=== FULL MODEL ===")
print(full_model.summary())

print("\n=== LIKELIHOOD RATIO TEST ===")
print(f"LRT = {lrt:.4f}, p = {p_lrt:.6g}")

# -----------------------------
# Extract main interaction term
# -----------------------------
interaction_name = "region[T.ACC]:outcome[T.MISS]"
if interaction_name in full_model.params.index:
    beta = full_model.params[interaction_name]
    se = full_model.bse[interaction_name]
    z = beta / se
    p = full_model.pvalues[interaction_name]
    ci_low, ci_high = full_model.conf_int().loc[interaction_name]

    print("\n=== INTERACTION TERM ===")
    print(f"{interaction_name}: beta = {beta:.6f}, SE = {se:.6f}, z = {z:.6f}, "
          f"p = {p:.6g}, 95% CI = [{ci_low:.6f}, {ci_high:.6f}]")

# -----------------------------
# Post hoc simple effects:
# outcome effect within each region
# -----------------------------
print("\n=== POST HOC SIMPLE EFFECTS: HIT vs MISS WITHIN REGION ===")

def wald_test_single_param(model, param_name):
    """Test whether a single fixed-effect parameter differs from 0."""
    if param_name not in model.params.index:
        raise ValueError(f"{param_name} not found in model parameters.")

    beta = model.params[param_name]
    se = model.bse[param_name]
    z = beta / se
    p = model.pvalues[param_name]
    ci_low, ci_high = model.conf_int().loc[param_name]

    return {
        "contrast": param_name,
        "beta": beta,
        "SE": se,
        "z": z,
        "p": p,
        "CI_low": ci_low,
        "CI_high": ci_high
    }

def wald_test_linear_combo(model, weights_dict, label):
    """
    Wald test for a linear combination of fixed effects:
    sum(w_i * beta_i)
    """
    param_names = list(model.params.index)
    w = np.zeros(len(param_names), dtype=float)

    for name, weight in weights_dict.items():
        if name not in param_names:
            raise ValueError(f"{name} not found in model parameters.")
        w[param_names.index(name)] = weight

    beta = float(np.dot(w, model.params.values))
    cov = model.cov_params().loc[param_names, param_names].values
    var = float(np.dot(w, np.dot(cov, w)))
    se = np.sqrt(var)

    z = beta / se
    p = 2 * st.norm.sf(abs(z))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    return {
        "contrast": label,
        "beta": beta,
        "SE": se,
        "z": z,
        "p": p,
        "CI_low": ci_low,
        "CI_high": ci_high
    }

posthoc_results = []

# 1) CLA: MISS vs HIT
# Under treatment coding, this is just outcome[T.MISS]
cla_simple = wald_test_single_param(full_model, "outcome[T.MISS]")
cla_simple["contrast"] = "CLA: MISS vs HIT"
posthoc_results.append(cla_simple)

# 2) ACC: MISS vs HIT
# This is outcome[T.MISS] + region[T.ACC]:outcome[T.MISS]
acc_simple = wald_test_linear_combo(
    full_model,
    {
        "outcome[T.MISS]": 1.0,
        "region[T.ACC]:outcome[T.MISS]": 1.0
    },
    "ACC: MISS vs HIT"
)
posthoc_results.append(acc_simple)

posthoc_df = pd.DataFrame(posthoc_results)
print(posthoc_df.to_string(index=False))

# -----------------------------
# 1) Leave-one-subject-out robustness
# -----------------------------
print("\n=== LEAVE-ONE-SUBJECT-OUT ROBUSTNESS ===")
loo_results = []

subjects = sorted(dat["subject"].unique())
for subj in subjects:
    df_sub = dat.loc[dat["subject"] != subj].copy()
    try:
        full_sub, red_sub, lrt_sub, p_lrt_sub = fit_mixed_model(df_sub)
        beta_sub = full_sub.params.get(interaction_name, np.nan)
        se_sub = full_sub.bse.get(interaction_name, np.nan)
        z_sub = beta_sub / se_sub if pd.notnull(beta_sub) and pd.notnull(se_sub) else np.nan
        p_sub = full_sub.pvalues.get(interaction_name, np.nan)

        loo_results.append({
            "left_out_subject": subj,
            "beta_interaction": beta_sub,
            "SE_interaction": se_sub,
            "z_interaction": z_sub,
            "p_interaction": p_sub,
            "lrt": lrt_sub,
            "p_lrt": p_lrt_sub,
        })
    except Exception as e:
        loo_results.append({
            "left_out_subject": subj,
            "beta_interaction": np.nan,
            "SE_interaction": np.nan,
            "z_interaction": np.nan,
            "p_interaction": np.nan,
            "lrt": np.nan,
            "p_lrt": np.nan,
            "error": str(e)
        })

loo_df = pd.DataFrame(loo_results)
print(loo_df)

# Optional summary line
valid_p = loo_df["p_interaction"].dropna()
if len(valid_p) > 0:
    print("\nLOO summary:")
    print(f"all significant at p < 0.05? {(valid_p < 0.05).all()}")
    print(f"all significant at p < 0.01? {(valid_p < 0.01).all()}")
    print(f"max p across LOO refits = {valid_p.max():.6g}")

# -----------------------------
# 2) Log-transform robustness
# Uses log(1 + fr)
# -----------------------------
print("\n=== LOG-TRANSFORM ROBUSTNESS ===")
dat_log = dat.copy()
dat_log["fr_log"] = np.log1p(dat_log["fr"])

full_log, red_log, lrt_log, p_lrt_log = fit_mixed_model(dat_log, response_col="fr_log")
print(full_log.summary())
print(f"LRT = {lrt_log:.4f}, p = {p_lrt_log:.6g}")

if interaction_name in full_log.params.index:
    beta = full_log.params[interaction_name]
    se = full_log.bse[interaction_name]
    z = beta / se
    p = full_log.pvalues[interaction_name]
    ci_low, ci_high = full_log.conf_int().loc[interaction_name]

    print("\nLog-transform interaction:")
    print(f"beta = {beta:.6f}, SE = {se:.6f}, z = {z:.6f}, "
          f"p = {p:.6g}, 95% CI = [{ci_low:.6f}, {ci_high:.6f}]")


# -----------------------------
# 3) Raw firing-rate normality checks (optional)
# -----------------------------
print("\n=== RAW FIRING-RATE NORMALITY BY CELL ===")
for region in ["CLA", "ACC"]:
    for outcome in ["HIT", "MISS"]:
        vals = dat.loc[(dat["region"] == region) & (dat["outcome"] == outcome), "fr"].dropna()
        W, pval = st.shapiro(vals)
        skew = st.skew(vals, bias=False)
        kurt = st.kurtosis(vals, fisher=True, bias=False)
        print(f"{region} {outcome}: n = {len(vals)}, W = {W:.6f}, p = {pval:.6g}, "
              f"skew = {skew:.6f}, excess kurtosis = {kurt:.6f}")
