#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-subject raw-BOLD spectral change panels and delta-C showcase ranking.

For every subject x target session (19 x 2), Welch power spectra of each of
the 24 ROIs are estimated before and after sonication (TR = 2.0 s, Nyquist
0.25 Hz). Two products are produced:

1) A visibility score per session: the mean absolute log band-power ratio
   over ROIs in the slow (0.01-0.10 Hz) band, plus the fraction of ROIs with
   |log ratio| >= 0.2. Sessions are ranked; the top session is rendered as
   the showcase figure (24 ROI pre/post PSD panels annotated with that
   session's per-ROI delta-C).

2) ROI-level association between delta-C and the slow-band power change
   across subjects (Spearman, per target x band), to check that the
   showcase case is representative rather than arbitrary.

Outputs (results/validation/S2/):
    session_visibility_rank.csv
    fig_showcase_{subject}_{target}.png
    bandpower_deltaC_spearman.csv

Usage:
    python analysis/validation_S2_spectrum_deltaC.py

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
ARC = ROOT.parent / "fhqwd-osfstorage-archive"
OUT = ROOT / "results" / "validation" / "S2"
TR = 2.0
FS = 1.0 / TR
ROIS24 = ["lh_Ca", "lh_GPe", "lh_GPi", "lh_Pu", "lh_STH",
          "lh_cerebellum_dentate", "lh_cerebellum_motor",
          "lh_paracentral_smooth3mm", "lh_postcentral_smooth3mm",
          "lh_precentral_smooth3mm", "lh_superiorfrontal_smooth3mm",
          "lh_vim", "lh_zi", "rh_Ca", "rh_GPe", "rh_GPi", "rh_Pu", "rh_STH",
          "rh_cerebellum_dentate", "rh_cerebellum_motor",
          "rh_paracentral_smooth3mm", "rh_postcentral_smooth3mm",
          "rh_precentral_smooth3mm", "rh_superiorfrontal_smooth3mm"]
SLOW = (0.01, 0.10)
FAST = (0.10, 0.25)


def load_session(subj, target):
    p = ARC / "rsfMRI" / "timeseries" / f"{subj}_ses-2_roi-{target}_meants.mat"
    if not p.exists():
        return None
    m = sio.loadmat(p)
    fnames = [str(x[0]) for x in np.ravel(m["fn"])]
    out = {}
    for i, rname in enumerate(fnames):
        out[rname] = (m["mpre"][:, i].astype(float),
                      m["mpost"][:, i].astype(float))
    if "lh_vim" not in out and target == "vim":
        out["lh_vim"] = (m["target_meants_pre"].ravel().astype(float),
                         m["target_meants_post"].ravel().astype(float))
    return out


def log_band_ratio(pre, post, lo, hi):
    f, p1 = welch(pre - pre.mean(), fs=FS, nperseg=128)
    f, p2 = welch(post - post.mean(), fs=FS, nperseg=128)
    sel = (f >= lo) & (f <= hi)
    if sel.sum() == 0:
        return np.nan, f, p1, p2
    return float(np.log((p2[sel].mean() + 1e-12) / (p1[sel].mean() + 1e-12))), f, p1, p2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dsum = pd.read_csv(ROOT / "results" / "figures" / "loso"
                       / "loso_19_fold_beta_0.2_13to1to5_split"
                       / "loso_delta_summary.csv")
    subs = sorted(dsum["subject_id"].unique())

    rows, assoc = [], []
    per_session = {}
    for subj in subs:
        for target in ["vim", "zi"]:
            sess = load_session(subj, target)
            if sess is None:
                continue
            slow_ratios, pairs = [], []
            for roi in ROIS24:
                if roi not in sess:
                    continue
                pre, post = sess[roi]
                lr, _, _, _ = log_band_ratio(pre, post, *SLOW)
                if np.isnan(lr):
                    continue
                slow_ratios.append(lr)
                dc = dsum[(dsum["subject_id"] == subj)
                          & (dsum["target"] == target)
                          & (dsum["roi"] == roi)]["delta"]
                if len(dc):
                    pairs.append((float(dc.iloc[0]), lr, roi))
                rows.append({"subject_id": subj, "target": target, "roi": roi,
                             "log_slow_power_ratio": round(lr, 4),
                             "delta_c": float(dc.iloc[0]) if len(dc) else np.nan})
            if slow_ratios:
                per_session[(subj, target)] = {
                    "mean_abs_log_slow": float(np.mean(np.abs(slow_ratios))),
                    "frac_abs_ge_0p2": float(np.mean(np.abs(slow_ratios) >= 0.2)),
                    "n_roi": len(slow_ratios)}
            for dcv, lr, roi in pairs:
                assoc.append({"target": target, "roi": roi,
                              "delta_c": dcv, "log_slow_power_ratio": lr})
    pd.DataFrame(rows).to_csv(OUT / "bandpower_deltaC_records.csv", index=False)

    rk = (pd.DataFrame([{"subject_id": s, "target": t, **v}
                        for (s, t), v in per_session.items()])
          .sort_values("mean_abs_log_slow", ascending=False)
          .reset_index(drop=True))
    rk.to_csv(OUT / "session_visibility_rank.csv", index=False)

    ad = pd.DataFrame(assoc)
    sp = ad.groupby(["target", "roi"]).apply(
        lambda g: pd.Series({"n": len(g),
                             "spearman_rho": spearmanr(g["delta_c"],
                                                       g["log_slow_power_ratio"])[0],
                             "p_value": spearmanr(g["delta_c"],
                                                  g["log_slow_power_ratio"])[1]}),
        include_groups=False).reset_index()
    sp.to_csv(OUT / "bandpower_deltaC_spearman.csv", index=False)

    top = rk.iloc[0]
    sess = load_session(top["subject_id"], top["target"])
    fig, axes = plt.subplots(6, 4, figsize=(14, 12), sharex=True)
    for ax, roi in zip(axes.ravel(), ROIS24):
        if roi not in sess:
            ax.axis("off"); continue
        pre, post = sess[roi]
        lr, f, p1, p2 = log_band_ratio(pre, post, *SLOW)
        sel = (f >= 0.005) & (f <= 0.25)
        ax.semilogy(f[sel], p1[sel], color="C0", lw=1, label="pre")
        ax.semilogy(f[sel], p2[sel], color="C3", lw=1, label="post")
        dc = dsum[(dsum["subject_id"] == top["subject_id"])
                  & (dsum["target"] == top["target"])
                  & (dsum["roi"] == roi)]["delta"]
        val = float(dc.iloc[0]) if len(dc) else np.nan
        ax.set_title(f"{roi}\ndC={val:+.4f}, log-ratio={lr:+.2f}", fontsize=7)
        ax.set_xlim(0.005, 0.25)
        ax.set_ylim(1e-5, 1)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Showcase session: {top['subject_id']}, {top['target']}-treated"
                 f"  (visibility: mean|log slow-power ratio| = "
                 f"{top['mean_abs_log_slow']:.2f})")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / f"fig_showcase_{top['subject_id']}_{top['target']}.png",
                dpi=110)
    plt.close(fig)

    def psd_panel(subj, target, out_png, highlight=False):
        sess2 = load_session(subj, target)
        if sess2 is None:
            return
        fig, axes = plt.subplots(6, 4, figsize=(14, 12), sharex=True)
        for ax, roi in zip(axes.ravel(), ROIS24):
            if roi not in sess2:
                ax.axis("off"); continue
            pre, post = sess2[roi]
            lr, f, p1, p2 = log_band_ratio(pre, post, *SLOW)
            sel = (f >= 0.005) & (f <= 0.25)
            ax.semilogy(f[sel], p1[sel], color="C0", lw=1, label="pre")
            ax.semilogy(f[sel], p2[sel], color="C3", lw=1, label="post")
            dc = dsum[(dsum["subject_id"] == subj) & (dsum["target"] == target)
                      & (dsum["roi"] == roi)]["delta"]
            val = float(dc.iloc[0]) if len(dc) else np.nan
            ax.set_title(f"{roi}\ndC={val:+.4f}, log-ratio={lr:+.2f}", fontsize=7)
            ax.set_xlim(0.005, 0.25)
            ax.set_ylim(1e-5, 1)
        axes[0, 0].legend(fontsize=7)
        tag = " [SHOWCASE]" if highlight else ""
        fig.suptitle(f"{subj} ({target}-treated): pre vs post PSD per ROI{tag}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_png, dpi=90)
        plt.close(fig)

    for _, r in rk.iterrows():
        psd_panel(r["subject_id"], r["target"],
                  OUT / f"fig_subject_{r['subject_id']}_{r['target']}.png")

    top = rk.iloc[0]
    psd_panel(top["subject_id"], top["target"],
              OUT / f"fig_showcase_{top['subject_id']}_{top['target']}.png",
              highlight=True)

    print("top visibility sessions:")
    print(rk.head(5).to_string(index=False))
    print("median |rho| delta_c vs slow power:",
          sp["spearman_rho"].abs().median().round(3))
    print("done:", OUT)


if __name__ == "__main__":
    main()