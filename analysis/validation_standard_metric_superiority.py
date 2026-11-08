#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Superiority check: delta-C versus standard BOLD metrics as tremor predictors.

For each tremor outcome (postural, rest; VIM- and ZI-treated sessions
separately), every candidate biomarker change is correlated across subjects
(Spearman) with the normalized tremor change
((Post - Pre) / (Post + Pre) of 4-8 Hz accelerometer power; negative = better).

Candidate families, evaluated on the same 19-subject set (16 primary, the
three non-converging subjects excluded only for the delta-C family where the
LOSO model of that subject is unavailable):
  deltaC   BRICK per-ROI control-gain change (loso_delta_summary.csv)
  rsFC     partial-correlation change, target vs 22 ROIs
           (rs-FC/*_fc-partialcorr.mat)
  fALFF    fractional ALFF change per ROI (fALFF/*_ALFF.mat)
  BOLDpow  raw slow-band (0.01-0.10 Hz) power change per ROI
           (timeseries/*_meants.mat, Welch, fs = 1/TR)

Per family: number of tests, the strongest |rho| and its p, BH-FDR q within
the family, and the count of q < 0.05 survivors. The comparison table answers
whether standard metrics reach the predictive strength of delta-C.

Outputs (results/validation/standard_metric_superiority/):
    metric_tremor_spearman.csv
    metric_family_comparison.csv
    fig_metric_family_comparison.png

Usage:
    python analysis/validation_standard_metric_superiority.py

Dependencies: numpy, scipy, pandas, matplotlib.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import io as sio
from scipy.stats import spearmanr
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARC = ROOT.parent / "fhqwd-osfstorage-archive"
OUT = ROOT / "results" / "validation" / "standard_metric_superiority"
TR = 2.0
SLOW = (0.01, 0.10)
ROI_FC22 = ["Left-Caudate", "Left-GPe", "Left-GPi", "Left-Putamen", "Left-STN",
            "Left-CBM-Dentate", "Left-CBM-Motor", "Left-Paracentral",
            "Left-Postcentral", "Left-Precentral", "Left-SupFront",
            "Right-Caudate", "Right-GPe", "Right-GPi", "Right-Putamen",
            "Right-STN", "Right-CBM-Dentate", "Right-CBM-Motor",
            "Right-Paracentral", "Right-Postcentral", "Right-Precentral",
            "Right-SupFront"]
# Archive timeseries .mat files still carry the pre-rename subthalamic
# spelling; map it onto the current STN label when matching columns.
LEGACY_FN = {"lh_STH": "lh_STN", "rh_STH": "rh_STN"}


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1.0)
    return q


def load_tremor():
    subj_list = [s.strip() for s in
                 (ARC / "rsfMRI" / "scripts" / "subject_list.txt")
                 .read_text(encoding="utf-8", errors="replace")
                 .splitlines() if s.strip()]
    m = sio.loadmat(ARC / "rsfMRI" / "tremor" / "tremor_pos_rest.mat")
    return pd.DataFrame({
        "subject_id": subj_list,
        "vim_pos": np.asarray(m["vim_pos"]).ravel(),
        "vim_rest": np.asarray(m["vim_rest"]).ravel(),
        "zi_pos": np.asarray(m["zi_pos"]).ravel(),
        "zi_rest": np.asarray(m["zi_rest"]).ravel()})


def slow_band_power(x):
    f, pxx = welch(x - x.mean(), fs=1.0 / TR, nperseg=128)
    sel = (f >= SLOW[0]) & (f <= SLOW[1])
    if not sel.sum():
        return np.nan
    return float(np.log(pxx[sel].mean() + 1e-12))


def delta_c_matrix(dsum, subs, target):
    """(n_sub, 24) array of delta-C, NaN where the fold is missing."""
    rois = list(dsum["roi"].unique())
    rois = [r for r in rois if r in ROI_ORDER]
    out = np.full((len(subs), len(ROI_ORDER)), np.nan)
    for i, s in enumerate(subs):
        for j, r in enumerate(ROI_ORDER):
            v = dsum[(dsum["subject_id"] == s) & (dsum["target"] == target)
                     & (dsum["roi"] == r)]["delta"]
            if len(v):
                out[i, j] = float(v.iloc[0])
    return out, ROI_ORDER


ROI_ORDER = ["lh_Ca", "lh_GPe", "lh_GPi", "lh_Pu", "lh_STN",
             "lh_cerebellum_dentate", "lh_cerebellum_motor",
             "lh_paracentral_smooth3mm", "lh_postcentral_smooth3mm",
             "lh_precentral_smooth3mm", "lh_superiorfrontal_smooth3mm",
             "lh_vim", "lh_zi", "rh_Ca", "rh_GPe", "rh_GPi", "rh_Pu",
             "rh_STN", "rh_cerebellum_dentate", "rh_cerebellum_motor",
             "rh_paracentral_smooth3mm", "rh_postcentral_smooth3mm",
             "rh_precentral_smooth3mm", "rh_superiorfrontal_smooth3mm"]


def build_metric_matrices(subs, target):
    """Return {family: (matrix n_sub x n_feature, feature_names)}."""
    dsum = pd.read_csv(ROOT / "results" / "figures" / "loso"
                       / "loso_19_fold_beta_0.2_13to1to5_split"
                       / "loso_delta_summary.csv")
    fams = {}

    m = np.full((len(subs), 24), np.nan)
    for i, s in enumerate(subs):
        for j, r in enumerate(ROI_ORDER):
            v = dsum[(dsum["subject_id"] == s) & (dsum["target"] == target)
                     & (dsum["roi"] == r)]["delta"]
            if len(v):
                m[i, j] = float(v.iloc[0])
    fams["deltaC"] = (m, list(ROI_ORDER))

    fc = np.full((len(subs), 22), np.nan)
    fa = np.full((len(subs), 24), np.nan)
    bp = np.full((len(subs), 24), np.nan)
    for i, s in enumerate(subs):
        fp = ARC / "rsfMRI" / "rs-FC" / f"{s}_ses-2_roi-{target}_fc-partialcorr.mat"
        if fp.exists():
            mm = sio.loadmat(fp)
            pre = np.asarray(mm["fcavg_prelifus"][0, 0][target]).ravel()
            post = np.asarray(mm["fcavg_postlifus"][0, 0][target]).ravel()
            k = min(len(pre), len(post), 22)
            fc[i, :k] = post[:k] - pre[:k]
        fa_p = ARC / "rsfMRI" / "fALFF" / f"{s}_ses-2_roi-{target}_ALFF.mat"
        if fa_p.exists():
            mm = sio.loadmat(fa_p)
            for j in range(24):
                try:
                    a = np.asarray(mm["fALFF_pre"][0, j], float).ravel()
                    b = np.asarray(mm["fALFF_post"][0, j], float).ravel()
                    fa[i, j] = np.log(b.mean() + 1e-12) - np.log(a.mean() + 1e-12)
                except Exception:
                    pass
        ts_p = ARC / "rsfMRI" / "timeseries" / f"{s}_ses-2_roi-{target}_meants.mat"
        if ts_p.exists():
            mm = sio.loadmat(ts_p)
            names = [LEGACY_FN.get(str(x[0]), str(x[0]))
                     for x in np.ravel(mm["fn"])]
            for j in range(min(mm["mpre"].shape[1], 24)):
                nm = names[j] if j < len(names) else None
                if nm not in ROI_ORDER:
                    continue
                col = ROI_ORDER.index(nm)
                pre = slow_band_power(mm["mpre"][:, j].astype(float))
                post = slow_band_power(mm["mpost"][:, j].astype(float))
                if not (np.isnan(pre) or np.isnan(post)):
                    bp[i, col] = post - pre
    fams["rsFC"] = (fc, list(ROI_FC22))
    fams["fALFF"] = (fa, list(ROI_ORDER))
    fams["BOLDpow"] = (bp, list(ROI_ORDER))
    return fams


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trem = load_tremor()
    subs = list(trem["subject_id"])

    rows = []
    for target in ("vim", "zi"):
        fams = build_metric_matrices(subs, target)
        for task in ("pos", "rest"):
            y = trem[f"{target}_{task}"].to_numpy(float)
            for fam, (M, feats) in fams.items():
                for j, feat in enumerate(feats):
                    col = M[:, j]
                    keep = ~np.isnan(col)
                    if keep.sum() < 12:
                        continue
                    r, p = spearmanr(col[keep], y[keep])
                    rows.append({"metric": fam, "target": target, "task": task,
                                 "feature": feat, "n": int(keep.sum()),
                                 "rho": float(r), "p": float(p)})
    df = pd.DataFrame(rows)
    df["q_within_target_task_family"] = np.nan
    for (m, t, k), g in df.groupby(["metric", "target", "task"]):
        df.loc[g.index, "q_within_target_task_family"] = bh_fdr(g["p"].to_numpy())
    df.to_csv(OUT / "metric_tremor_spearman.csv", index=False)

    comp = []
    for (m, t, k), g in df.groupby(["metric", "target", "task"]):
        i_best = g["p"].idxmin()
        comp.append({"metric": m, "target": t, "task": k, "n_tests": len(g),
                     "best_rho": float(g.loc[i_best, "rho"]),
                     "best_feature": str(g.loc[i_best, "feature"]),
                     "best_p": float(g.loc[i_best, "p"]),
                     "best_q": float(g.loc[i_best, "q_within_target_task_family"]),
                     "n_fdr_survive": int((g["q_within_target_task_family"] < 0.05).sum())})
    cdf = pd.DataFrame(comp).sort_values(["target", "task", "metric"])
    cdf.to_csv(OUT / "metric_family_comparison.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, (t, k) in zip(axes.ravel(), [("vim", "pos"), ("vim", "rest"),
                                          ("zi", "pos"), ("zi", "rest")]):
        g = cdf[(cdf["target"] == t) & (cdf["task"] == k)]
        colors = {"deltaC": "C3", "rsFC": "C0", "fALFF": "C2", "BOLDpow": "C1"}
        ax.bar(range(len(g)), g["best_rho"].abs(),
               color=[colors.get(x, "grey") for x in g["metric"]])
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g["metric"], rotation=30, fontsize=9)
        for i, (_, r) in enumerate(g.iterrows()):
            star = " *" if r["best_q"] < 0.05 else ""
            ax.text(i, abs(r["best_rho"]) + 0.02,
                    f"{r['best_rho']:+.2f}{star}", ha="center", fontsize=9)
        ax.axhline(0.456, color="k", ls=":", lw=0.8)
        ax.set_title(f"{t.upper()} treated, {k}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("|Spearman rho| (strongest feature)")
    fig.suptitle("Strongest tremor association per metric family "
                 "(* = survives BH-FDR; dotted = 80%-power threshold)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_metric_family_comparison.png", dpi=120)
    plt.close(fig)

    print(cdf.to_string(index=False))
    print("done:", OUT)


if __name__ == "__main__":
    main()