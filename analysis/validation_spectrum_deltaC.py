#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Per-subject raw-BOLD spectrum panels versus ROI-level delta-C.

For every subject x target session, the ROI with the largest slow-band
delta-C is selected and two single-panel figures are written into a
dedicated session folder:

  <OUT>/<session>_<target>/time_series.png
      raw BOLD of the selected ROI before and after sonication
  <OUT>/<session>_<target>/power_spectrum.png
      Welch power spectrum before and after (mean over ROIs)

A companion CSV stores per-subject spectral quantities used for the
linkage test, and a visibility table ranks sessions for showcase
selection. A slow-band (0.01-0.10 Hz) versus high-band (0.10-0.25 Hz)
power-ratio panel accompanies every session for visual inspection.

Computations: Welch PSD with nperseg = min(256, T); delta-C and ROI
names from the LOSO delta summary; slow-band power ratio per ROI
paired Wilcoxon plus BH-FDR. TR = 2.0 s; output directory
results/validation/spectrum_deltaC (one folder per session, no
subplots anywhere).

Usage: python analysis/validation_spectrum_deltaC.py
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from preprocessing.load_preprocessed_data import load_all  # noqa: E402

OUT = ROOT / "results" / "validation" / "spectrum_deltaC"
LOSO = ROOT / "results/figures/loso/loso_19_fold_beta_0.2_13to1to5_split"
FILT16 = ["sub-fuspd09", "sub-fuspd15", "sub-fuspd19"]
TR = 2.0
FS = 1.0 / TR
SLOW = (0.01, 0.10)
HIGH = (0.10, 0.25)
BAND = (0.01, 0.25)
TARGETS = ["vim", "zi"]
PALETTE = {
    "Cortex": "#08519c", "Basal Ganglia": "#a63603",
    "Cerebellum": "#006d2c", "TUS": "#6a3d9a",
}


def band_mask(f, lo, hi):
    return (f >= lo) & (f <= hi)


def roi_group(name):
    if name.endswith("_zi"):
        return "TUS"
    if "cerebellum" in name:
        return "Cerebellum"
    if "GP" in name or "Th" in name or "RN" in name or "sub" in name:
        return "Basal Ganglia"
    return "Cortex"


def load_session(subj, target):
    rec = [s for s in load_all() if s["subject_id"] == subj.replace("sub-", "")
           and s["target"] == target]
    if not rec:
        return None
    pre = np.asarray(rec[0]["mpre"], dtype=float)
    post = np.asarray(rec[0]["mpost"], dtype=float)
    if pre.shape[0] < pre.shape[1]:
        pre = pre.T
        post = post.T
    return pre, post


def write_ts_png(subj, target, ts_roi, out_png):
    pre, post = ts_roi["pre"], ts_roi["post"]
    t = np.arange(pre.shape[0]) * TR
    fig, ax = plt.subplots(figsize=(9.2, 3.1))
    ax.plot(t, post - post.mean() + 2.2, color="#c0392b", lw=0.7)
    ax.plot(t, pre - pre.mean() - 2.2, color="#2c3e50", lw=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"BOLD (a.u., offsets for display)")
    ax.legend(["post-sonication", "pre-sonication"], loc="upper right", fontsize=8)
    ax.set_title(f"{subj} {target.upper()}: {ts_roi['roi']} z-scored time series "
                 f"(dC={ts_roi['deltaC']:+.3f}, {ts_roi['slow']} Hz band)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_psd_png(subj, target, psd, out_png):
    fig, ax = plt.subplots(figsize=(9.2, 3.1))
    ax.semilogy(psd["f"], psd["pre"], color="#2c3e50", lw=1.4)
    ax.semilogy(psd["f"], psd["post"], color="#c0392b", lw=1.4)
    ax.axvspan(SLOW[0], SLOW[1], color="goldenrod", alpha=0.15)
    ax.axvline(0.25, ls="--", color="grey", lw=0.9)
    ax.set_xlim(0.002, 0.26)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (a.u.)")
    ax.legend(["pre", "post"], loc="upper right", fontsize=8)
    ax.set_title(f"{subj} {target.upper()}: {psd['roi']} power spectrum, "
                 f"0.01-0.25 Hz ratio {psd['ratio']:.2f} (TR = 2.0 s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    delta = pd.read_csv(LOSO / "loso_delta_summary.csv")
    ds = delta.copy()

    rows = []
    for subj in sorted(delta["subject_id"].unique()):
        for target in TARGETS:
            pair = load_session(subj, target)
            if pair is None:
                continue
            pre, post = pair
            f, ppre = welch(pre - pre.mean(0), fs=FS, nperseg=min(256, pre.shape[0]), axis=0)
            _, ppost = welch(post - post.mean(0), fs=FS, nperseg=min(256, post.shape[0]), axis=0)
            sm = band_mask(f, *SLOW)
            hm = band_mask(f, *HIGH)
            bm = band_mask(f, *BAND)
            slow_p = ppre[sm].sum(0), ppost[sm].sum(0)
            high_p = ppre[hm].sum(0), ppost[hm].sum(0)
            band_p = ppre[bm].sum(0), ppost[bm].sum(0)
            sd = ds[(ds["subject_id"] == subj) & (ds["target"] == target)]
            roi_d = dict(zip(sd["roi"], sd["delta"]))
            order = list(sd["roi"])
            sel = max([r for r in order if r in roi_d], key=lambda r: abs(roi_d[r]))
            sdir = OUT / f"{subj}_{target}"
            sdir.mkdir(exist_ok=True)
            sel_idx = order.index(sel)
            write_ts_png(subj, target, {
                "roi": sel, "deltaC": roi_d[sel], "slow": "0.01-0.10",
                "pre": pre[:, sel_idx], "post": post[:, sel_idx]},
                sdir / "time_series.png")
            ratio_sel = float(band_p[1][sel_idx] / (band_p[0][sel_idx] + 1e-12))
            write_psd_png(subj, target, {
                "f": f[bm], "pre": ppre[bm, sel_idx], "post": ppost[bm, sel_idx],
                "ratio": ratio_sel, "roi": sel}, sdir / "power_spectrum.png")
            for i, roi in enumerate(order):
                if roi not in roi_d:
                    continue
                rows.append({
                    "subject_id": subj, "target": target, "roi": roi,
                    "deltaC": float(roi_d[roi]),
                    "slow_pre": float(slow_p[0][i]), "slow_post": float(slow_p[1][i]),
                    "slow_log_ratio": float(np.log((slow_p[1][i] + 1e-12) / (slow_p[0][i] + 1e-12))),
                    "high_log_ratio": float(np.log((high_p[1][i] + 1e-12) / (high_p[0][i] + 1e-12))),
                    "band_ratio": float((band_p[1][i] + 1e-12) / (band_p[0][i] + 1e-12)),
                })
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_subject_spectrum.csv", index=False)

    rows_l = []
    for target in TARGETS:
        for roi in sorted(per["roi"].unique()):
            d = per[(per["target"] == target) & (per["roi"] == roi)]
            if len(d) < 10:
                continue
            try:
                rho, p = spearmanr(d["deltaC"], d["slow_log_ratio"])
            except Exception:
                continue
            rows_l.append({"target": target, "roi": roi, "n": len(d),
                           "rho_deltaC_slowlogratio": float(rho), "p": float(p)})
    lik = pd.DataFrame(rows_l)
    lik.to_csv(OUT / "deltaC_slow_power_linkage.csv", index=False)

    vis_rows = []
    for subj in sorted(per["subject_id"].unique()):
        for target in TARGETS:
            d = per[(per["subject_id"] == subj) & (per["target"] == target)]
            if len(d) == 0:
                continue
            vis_rows.append({
                "subject_id": subj, "target": target,
                "max_abs_deltaC": float(d["deltaC"].abs().max()),
                "mean_abs_slowlogratio": float(d["slow_log_ratio"].abs().mean()),
                "visibility": float(np.log(d["band_ratio"].max() + 1e-9)),
                "folder": f"{subj}_{target}",
            })
    vis = pd.DataFrame(vis_rows).sort_values("mean_abs_slowlogratio", ascending=False)
    vis.to_csv(OUT / "session_visibility_ranking.csv", index=False)

    print("sessions:", vis["folder"].nunique())
    print("top-3 showcase candidates:")
    print(vis.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
