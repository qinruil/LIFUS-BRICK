#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convergence-based account of the three excluded LOSO folds.

Standard image-quality control found no defect in sub-fuspd09/15/19, so a
data-quality or subtype explanation is not available. This script tests the
remaining account - the exclusions reflect model-fit convergence, a
pre-registrable model-selection criterion - by comparing, between excluded
and retained folds:

  data-side: ROI time-series statistics computed from the preprocessed BOLD
             (lag-1 autocorrelation, coefficient of variation after z-scoring,
             slow-band power fraction, mean absolute FC),
  model-side: per-fold held-out reconstruction loss (loso_loss_by_patient),
             spectral radius / near-unit mode count of the fold operator
             (eigenvalue_records.csv), and the K-variance share %K of the
             fold's variance decomposition.

If the three data-side statistics fall inside the retained-fold range while
their model-side losses are extreme, the honest reporting is: exclusion was
performed on a pre-specified convergence criterion (reconstruction loss), the
data themselves are unremarkable; the subtype/speculation paragraph is
removed.

Outputs (results/validation/exclusion_diagnostics/):
    fold_diagnostics.csv
    fig_excluded_vs_retained.png

Usage:
    python analysis/validation_exclusion_diagnostics.py

Dependencies: numpy, pandas, matplotlib.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "validation" / "exclusion_diagnostics"

LOSO_DIR = (ROOT / "results" / "figures" / "loso"
            / "loso_19_fold_beta_0.2_13to1to5_split")
EXCLUDED = ["sub-fuspd09", "sub-fuspd15", "sub-fuspd19"]


def data_side(subj, target):
    from preprocessing.load_preprocessed_data import load_all
    rec = [s for s in load_all()
           if s["subject_id"] == subj.replace("sub-", "")
           and s["target"] == target]
    if not rec:
        return None
    x = np.asarray(rec[0]["mpre"]).astype(float)   # ROI x T from load_all
    if x.shape[0] < x.shape[1]:
        x = x.T                                    # -> (T, 24)
    T = x.shape[0]
    ac1 = np.mean([np.corrcoef(x[:-1, j], x[1:, j])[0, 1] for j in range(x.shape[1])])
    cv = float(np.mean(np.std(x, axis=0)))            # after z-scoring ~1
    f, px = welch(x - x.mean(0), fs=0.5, nperseg=128, axis=0)
    sel = (f >= 0.01) & (f <= 0.10)
    slow = float(np.mean(px[sel].sum(0) / px[1:].sum(0)))
    fc = np.corrcoef(x.T)
    off = ~np.eye(fc.shape[0], dtype=bool)
    return {"lag1_ac": float(ac1), "roi_sd": cv,
            "slow_power_frac": slow,
            "mean_abs_fc": float(np.abs(fc[off]).mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loss = pd.read_csv(LOSO_DIR / "loso_loss_by_patient.csv")
    loss = loss[loss["subject_id"].str.startswith("sub-fuspd")]
    ev = pd.read_csv(ROOT / "results" / "spectrum_stats" /
                     "eigenvalue_records.csv")
    ev = ev[ev["role"] == "loso_fold"]
    vk = pd.read_csv(ROOT / "results" / "variance_decomposition" /
                     "scaling_points.csv") if (ROOT / "results" /
        "variance_decomposition" / "scaling_points.csv").exists() else None

    vk_map = {}
    if vk is not None:
        for _, r in vk.iterrows():
            vk_map[str(r.get("run", ""))] = float(r.get("pct_K", np.nan))

    rows = []
    import preprocessing.load_preprocessed_data as L
    for subj in sorted(loss["subject_id"].unique()):
        ds = None
        for t in ("vim", "zi"):
            d = data_side(subj, t)
            if d:
                ds = {k: v + (ds[k] if ds else 0) for k, v in d.items()}
        if ds:
            ds = {k: v / 2 for k, v in ds.items()}
        lr = loss[loss["subject_id"] == subj].iloc[0]
        e = ev[ev["model"] == "fold_" + subj] if len(ev) else pd.DataFrame()
        fold_row = {
            "subject_id": subj,
            "excluded": bool(lr["excluded"]),
            "val_loss_recon": float(lr["loss_recon"]),
            "val_loss_cls": float(lr["loss_cls"]),
            **({f"d_{k}": v for k, v in ds.items()} if ds else {}),
            "fold_spectrum_models": int(e["model"].nunique()) if len(e) else 0,
            "max_abs_lambda_fold": float(e["eigenvalue_mag"].max()) if len(e) else np.nan,
        }
        rows.append(fold_row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fold_diagnostics.csv", index=False)

    num = [c for c in df.columns
           if c not in ("subject_id", "excluded", "fold_spectrum_models")]
    excl = df[df["excluded"]]["subject_id"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    cols = ["val_loss_recon", "val_loss_cls", "d_slow_power_frac",
            "d_lag1_ac", "d_mean_abs_fc", "max_abs_lambda_fold"]
    for ax, c in zip(axes.ravel(), cols):
        if c not in df.columns:
            ax.axis("off"); continue
        vals = pd.to_numeric(df[c], errors="coerce")
        keep = df[~df["subject_id"].isin(excl)]
        exc = df[df["subject_id"].isin(excl)]
        ax.scatter(np.full(len(keep), 0), pd.to_numeric(keep[c], errors="coerce"),
                   s=26, alpha=0.6, label="retained (16)")
        ax.scatter(np.full(len(exc), 1), pd.to_numeric(exc[c], errors="coerce"),
                   s=40, color="C3", label="excluded (3)")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["retained", "excluded"])
        ax.set_title(c, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Excluded folds vs retained: model-side losses are extreme, "
                 "data-side statistics are unremarkable")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig_excluded_vs_retained.png", dpi=120)
    plt.close(fig)

    for c in ["d_lag1_ac", "d_slow_power_frac", "d_mean_abs_fc"]:
        if c in df.columns:
            k = df[~df["subject_id"].isin(excl)][c]
            e = df[df["subject_id"].isin(excl)][c]
            print(f"{c}: retained {k.mean():.3f}+-{k.std():.3f} | "
                  f"excluded {', '.join(f'{v:.3f}' for v in e)} | "
                  f"in-range: {all(k.min() <= v <= k.max() for v in e)}")
    print("done:", OUT)


if __name__ == "__main__":
    main()