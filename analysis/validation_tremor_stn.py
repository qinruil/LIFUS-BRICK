#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STN delta-C contrast and delta-C versus tremor change validation.

Computes, for the 24-ROI BRICK model (19-patient LIFUS cohort):

Part 1. STN delta-C after ZI versus after VIM.
        Pairs the within-subject control-gain changes (roi_c_post - roi_c_pre)
        of lh_STN / rh_STN across the two sonication targets, with a two-sided
        Wilcoxon signed-rank test and a paired Cohen's dz. Runs on both the
        19-subject fold set and the 16-subject set (three divergent folds
        excluded).

Part 2. Correlation between delta-C and tremor change.
        Tremor change is the normalized power difference
        (Post - Pre)/(Post + Pre) from accelerometry, positive meaning higher
        post-sonication tremor power. Spearman correlation between the
        per-subject delta-C of a given ROI and the tremor change of the same
        target (VIM or ZI), for postural and rest tasks separately. Reported
        for the pre-specified ROIs (lh_GPi, lh_STN, rh_STN) and for all 24
        ROIs, with BH-FDR marking inside each target x task family.

Inputs (read only):
    results/figures/loso/loso_19_fold_beta_0.2_13to1to5_split/loso_delta_summary.csv
    results/figures/loso/loso_19_fold_beta_0.2_13to1to5_split/loso_delta_summary_filtered.csv
    fhqwd-osfstorage-archive/rsfMRI/tremor/tremor_pos_rest.mat
    fhqwd-osfstorage-archive/rsfMRI/scripts/subject_list.txt

Outputs (all under the output directory passed via --out):
    stn_deltaC_comparison.csv
    specified_roi_tremor_correlation.csv
    roi_tremor_spearman_all.csv
    fdr_summary.csv
    scatter_roi_tremor_{vim,zi}_{pos,rest}.png
    heatmap_roi_tremor_spearman.png

Dependencies: numpy, scipy, pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy.stats import spearmanr, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "fhqwd-osfstorage-archive"
TARGET_ROIS = [
    "lh_Ca", "lh_GPe", "lh_GPi", "lh_Pu", "lh_STN", "lh_cerebellum_dentate",
    "lh_cerebellum_motor", "lh_paracentral_smooth3mm", "lh_postcentral_smooth3mm",
    "lh_precentral_smooth3mm", "lh_superiorfrontal_smooth3mm", "lh_vim", "lh_zi",
    "rh_Ca", "rh_GPe", "rh_GPi", "rh_Pu", "rh_STN", "rh_cerebellum_dentate",
    "rh_cerebellum_motor", "rh_paracentral_smooth3mm", "rh_postcentral_smooth3mm",
    "rh_precentral_smooth3mm", "rh_superiorfrontal_smooth3mm",
]
SPECIFIED_ROIS = ["lh_GPi", "lh_STN", "rh_STN"]
DIVERGENT_FOLDS = {"sub-fuspd09", "sub-fuspd15", "sub-fuspd19"}


def load_delta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["subject_id"] = df["subject_id"].str.strip()
    df["roi"] = df["roi"].str.strip()
    df["target"] = df["target"].str.strip().str.lower()
    return df


def load_tremor(base: Path) -> pd.DataFrame:
    subj_list = [
        s.strip() for s in
        (base / "rsfMRI/scripts/subject_list.txt").read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    m = sio.loadmat(base / "rsfMRI/tremor/tremor_pos_rest.mat")
    rows = []
    for i, sid in enumerate(subj_list):
        rows.append({
            "subject_id": sid,
            "vim_pos": float(m["vim_pos"][i, 0]),
            "vim_rest": float(m["vim_rest"][i, 0]),
            "zi_pos": float(m["zi_pos"][i, 0]),
            "zi_rest": float(m["zi_rest"][i, 0]),
        })
    return pd.DataFrame(rows)


def spearman_boot(x: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                  seed: int = 42) -> tuple:
    """Spearman rho with percentile bootstrap 95% CI (bivariate resampling)."""
    n = len(x)
    rng = np.random.default_rng(seed)
    r, p = spearmanr(x, y)
    if n < 4:
        return float(r), float(p), float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bx, by = x[idx], y[idx]
        if np.unique(bx).size > 1 and np.unique(by).size > 1:
            boots.append(spearmanr(bx, by)[0])
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return float(r), float(p), float(lo), float(hi)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values (statsmodels is not required)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    q = np.ones(n)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[valid] = ranked[np.argsort(order)]
    return q


def paired_dz(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def run_stn_comparison(d19: pd.DataFrame, d16: pd.DataFrame) -> pd.DataFrame:
    out = []
    for label, df in (("19-subject", d19), ("16-subject", d16)):
        for roi in ("lh_STN", "rh_STN"):
            rows = []
            for tgt in ("vim", "zi"):
                sub = df[(df["roi"] == roi) & (df["target"] == tgt)]
                for _, r in sub.iterrows():
                    rows.append({"subject_id": r["subject_id"], tgt: float(r["delta"])})
            pivot = pd.DataFrame(rows).groupby("subject_id", as_index=False).first()
            paired = pivot.dropna(subset=["vim", "zi"])
            if len(paired) < 3:
                continue
            vim_d, zi_d = paired["vim"].to_numpy(), paired["zi"].to_numpy()
            try:
                w_stat, w_p = wilcoxon(vim_d, zi_d)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            out.append({
                "cohort": label, "roi": roi, "n_pairs": len(paired),
                "deltaC_vim_median": float(np.median(vim_d)),
                "deltaC_zi_median": float(np.median(zi_d)),
                "deltaC_vim_mean": float(vim_d.mean()),
                "deltaC_zi_mean": float(zi_d.mean()),
                "paired_dz": round(paired_dz(zi_d, vim_d), 4),
                "wilcoxon_p": float(w_p),
                "median_zi_minus_vim": float(np.median(zi_d - vim_d)),
            })
    return pd.DataFrame(out)


def run_tremor_correlation(d19: pd.DataFrame, d16: pd.DataFrame,
                           tremor: pd.DataFrame) -> tuple:
    fams = [("vim", "pos"), ("vim", "rest"), ("zi", "pos"), ("zi", "rest")]
    recs_all = []
    for cohort, df in (("19-subject", d19), ("16-subject", d16)):
        sub_ids = sorted(df["subject_id"].unique())
        tr = tremor[tremor["subject_id"].isin(sub_ids)]
        for roi in TARGET_ROIS:
            for tgt, task in fams:
                col = f"{tgt}_{task}"
                merged = []
                for sid in sub_ids:
                    drow = df[(df["subject_id"] == sid) & (df["roi"] == roi)
                              & (df["target"] == tgt)]
                    trow = tr[tr["subject_id"] == sid]
                    if drow.empty or trow.empty:
                        continue
                    merged.append((float(drow["delta"].iloc[0]),
                                   float(trow[col].iloc[0])))
                if len(merged) < 4:
                    continue
                x = np.array([m[0] for m in merged])
                y = np.array([m[1] for m in merged])
                r, p, lo, hi = spearman_boot(x, y)
                recs_all.append({
                    "cohort": cohort, "roi": roi, "target": tgt, "task": task,
                    "n": len(merged), "spearman_rho": round(r, 4),
                    "p_value": float(p), "boot_ci_lo": round(lo, 4),
                    "boot_ci_hi": round(hi, 4),
                })
    df_all = pd.DataFrame(recs_all)

    # BH-FDR within each cohort x target x task family
    fdr_rows = []
    df_c = df_all.copy()
    qs = []
    for (cohort, tgt, task), g in df_c.groupby(["cohort", "target", "task"]):
        q = bh_fdr(g["p_value"].to_numpy())
        df_c.loc[g.index, "q_value"] = q
    df_c["q_value"] = df_c["q_value"].astype(float)
    df_c["passes_fdr"] = df_c["q_value"] < 0.05
    for (cohort, tgt, task), g in df_c.groupby(["cohort", "target", "task"]):
        sig = int((g["q_value"] < 0.05).sum())
        fdr_rows.append({
            "cohort": cohort, "target": tgt, "task": task,
            "n_roi_family": len(g), "n_fdr_survive": sig,
            "min_q": float(g["q_value"].min()) if len(g) else np.nan,
        })
    return df_c, pd.DataFrame(fdr_rows)


def save_figures(df: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11, "axes.linewidth": 0.8, "figure.dpi": 150,
    })
    d19 = df[df["cohort"] == "19-subject"]
    dl = pd.read_csv(ROOT / "results/figures/loso"
                     / "loso_19_fold_beta_0.2_13to1to5_split"
                     / "loso_delta_summary.csv")
    dl["subject_id"] = dl["subject_id"].str.strip()
    dl["roi"] = dl["roi"].str.strip()
    dl["target"] = dl["target"].str.strip().str.lower()
    tremor = load_tremor(ARCHIVE)

    for tgt, task in (("vim", "pos"), ("vim", "rest"), ("zi", "pos"), ("zi", "rest")):
        col = f"{tgt}_{task}"
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
        for ax, roi in zip(axes, SPECIFIED_ROIS):
            xs, ys = [], []
            for _, rr in tremor.iterrows():
                drow = dl[(dl["subject_id"] == rr["subject_id"])
                          & (dl["roi"] == roi) & (dl["target"] == tgt)]
                if drow.empty:
                    continue
                xs.append(float(drow["delta"].iloc[0]))
                ys.append(float(rr[col]))
            xs, ys = np.array(xs), np.array(ys)
            keep = ~(np.isnan(xs) | np.isnan(ys))
            xs, ys = xs[keep], ys[keep]
            ax.scatter(xs, ys, s=42, alpha=0.75, edgecolor="k", linewidth=0.4)
            if len(xs) >= 4:
                r, p, lo, hi = spearman_boot(xs, ys)
                rhos = np.polyfit(xs, ys, 1)
                xx = np.linspace(xs.min(), xs.max(), 50)
                ax.plot(xx, np.polyval(rhos, xx), color="C3", lw=1.2)
                ax.set_title(f"{roi}\nrho={r:.2f}, p={p:.3f}")
            else:
                ax.set_title(roi)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xlabel(r"$\Delta C$")
            ax.set_ylabel(f"tremor change ({tgt}, {task})")
        fig.tight_layout()
        fig.savefig(out / f"scatter_roi_tremor_{tgt}_{task}.png")
        plt.close(fig)

    # heatmap: roi x (target-task) spearman rho (19-subject)
    piv = d19.pivot_table(index="roi", columns=["target", "task"],
                          values="spearman_rho", aggfunc="first")
    piv = piv.reindex(TARGET_ROIS)
    fig, ax = plt.subplots(figsize=(7.2, 9.5))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1,
                   aspect="auto")
    ax.set_yticks(range(len(TARGET_ROIS)))
    ax.set_yticklabels(TARGET_ROIS, fontsize=8)
    fams = [("vim", "pos"), ("vim", "rest"), ("zi", "pos"), ("zi", "rest")]
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{a} {b}" for a, b in fams], fontsize=9)
    for i, roi in enumerate(TARGET_ROIS):
        for j, (tgt, task) in enumerate(fams):
            val = piv.loc[roi, (tgt, task)] if (tgt, task) in piv.columns else np.nan
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title("Delta-C vs tremor change (19-subject)")
    fig.tight_layout()
    fig.savefig(out / "heatmap_roi_tremor_spearman.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/validation/tremor_stn")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    d19 = load_delta(ROOT / "results/figures/loso"
                     / "loso_19_fold_beta_0.2_13to1to5_split"
                     / "loso_delta_summary.csv")
    d16 = load_delta(ROOT / "results/figures/loso"
                     / "loso_19_fold_beta_0.2_13to1to5_split"
                     / "loso_delta_summary_filtered.csv")
    tremor = load_tremor(ARCHIVE)

    df_stn = run_stn_comparison(d19, d16)
    df_stn.to_csv(out / "stn_deltaC_comparison.csv", index=False)
    print("stn_deltaC_comparison.csv:")
    print(df_stn.to_string(index=False))

    df_corr, df_fdr = run_tremor_correlation(d19, d16, tremor)
    df_corr.to_csv(out / "roi_tremor_spearman_all.csv", index=False)
    df_fdr.to_csv(out / "fdr_summary.csv", index=False)
    spec = df_corr[df_corr["roi"].isin(SPECIFIED_ROIS)]
    spec.to_csv(out / "specified_roi_tremor_correlation.csv", index=False)
    print("\nspecified_roi_tremor_correlation.csv:")
    print(spec.to_string(index=False))
    print("\nfdr_summary.csv:")
    print(df_fdr.to_string(index=False))

    save_figures(df_corr, out)

    manifest = {
        "inputs": {
            "delta_19": (ROOT / "results/figures/loso"
                         / "loso_19_fold_beta_0.2_13to1to5_split"
                         / "loso_delta_summary.csv").as_posix(),
            "delta_16": (ROOT / "results/figures/loso"
                         / "loso_19_fold_beta_0.2_13to1to5_split"
                         / "loso_delta_summary_filtered.csv").as_posix(),
            "tremor": (ROOT / "fhqwd-osfstorage-archive/rsfMRI/tremor"
                       / "tremor_pos_rest.mat").as_posix(),
        },
        "tremor_metric": "(Post - Pre)/(Post + Pre) of 4-8 Hz band power, "
                         "positive = worse (higher post power)",
        "fdr": "Benjamini-Hochberg within cohort x target x task family",
        "divergent_folds_excluded": sorted(DIVERGENT_FOLDS),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("\ndone:", out)


if __name__ == "__main__":
    main()