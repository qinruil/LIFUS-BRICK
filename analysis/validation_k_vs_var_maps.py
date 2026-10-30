#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Side-by-side comparison of the BRICK delta-K map and the VAR delta-A_eff map.

BRICK side: block-coupling difference of the two task-state operators,
K_region_coupling_diff.csv (rows = target ROI, cols = source ROI).

VAR side: VAR(1) refit from the preprocessed BOLD of each subject in the
16-subject primary cohort, per treatment target; the group mean of
A_post - A_pre over the off-diagonal connections is drawn in the identical
ROI order so the two maps are directly comparable. Cross-map Spearman
correlation over off-diagonal entries quantifies the visual similarity.

Outputs (results/validation/k_vs_var_maps/):
    var_delta_A_eff_mean.csv
    k_vs_var_map_correlation.csv
    fig_k_vs_var_side_by_side.png

Usage:
    python analysis/validation_k_vs_var_maps.py

Dependencies: numpy, pandas, scipy, matplotlib, torch (model checkpoint-free;
only preprocessed ROI time series are used).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS

OUT = ROOT / "results" / "validation" / "k_vs_var_maps"
KDIFF = (ROOT / "results" / "figures" / "task_states" / "K_comparison" /
         "K_region_coupling_diff.csv")
EXCLUDED = {"sub-fuspd09", "sub-fuspd15", "sub-fuspd19"}
ROIS24 = list(TARGET_ROIS)


def fit_var(ts: np.ndarray) -> np.ndarray:
    """OLS VAR(1): X_t = X_{t-1} A^T. Returns A_eff (N, N)."""
    x = ts - ts.mean(axis=0)
    prev, fut = x[:-1].T, x[1:].T          # (N, T-1)
    lam = 1e-6 * np.trace(prev @ prev.T) / prev.shape[1]
    gram = prev @ prev.T + lam * np.eye(prev.shape[0])
    coef = np.linalg.solve(gram, prev @ fut.T)
    return coef.T


def mean_delta_a(target: str) -> np.ndarray:
    stack = []
    for s in load_all():
        if s["target"] != target or s["subject_id"] in EXCLUDED:
            continue
        pre = np.asarray(s["mpre"]).astype(np.float64)
        post = np.asarray(s["mpost"]).astype(np.float64)
        n_roi = len(ROIS24)
        if pre.shape[0] == n_roi and pre.shape[1] != n_roi:
            pre, post = pre.T, post.T
        stack.append(fit_var(post) - fit_var(pre))
    return np.mean(stack, axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kd = pd.read_csv(KDIFF, index_col=0).loc[ROIS24, ROIS24].to_numpy(float)
    dvar_vim = mean_delta_a("vim")
    dvar_zi = mean_delta_a("zi")
    dvar = 0.5 * (dvar_vim + dvar_zi)
    np.savetxt(OUT / "var_delta_A_eff_mean.csv", dvar, delimiter=",")

    off = ~np.eye(24, dtype=bool)
    r_vim, p_vim = spearmanr(kd[off], dvar_vim[off])
    r_zi, p_zi = spearmanr(kd[off], dvar_zi[off])
    r_all, p_all = spearmanr(kd[off], dvar[off])
    pd.DataFrame({
        "pair": ["deltaK_vs_dVAR_vim", "deltaK_vs_dVAR_zi",
                 "deltaK_vs_dVAR_pooled"],
        "spearman_rho": [r_vim, r_zi, r_all],
        "p_value": [p_vim, p_zi, p_all],
        "n_offdiag_pairs": [int(off.sum())] * 3,
    }).to_csv(OUT / "k_vs_var_map_correlation.csv", index=False)

    labels = [r.replace("_smooth3mm", "").replace("cerebellum_", "cb_")
              for r in ROIS24]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharex=True,
                             sharey=True)
    vmaxk = np.abs(kd[off]).max()
    im0 = axes[0].imshow(kd, cmap="RdBu_r", vmin=-vmaxk, vmax=vmaxk)
    axes[0].set_title(f"BRICK delta-K (block coupling, post - pre)\n"
                      f"abs scale 0 - {vmaxk:.2f}")
    vmaxv = max(np.abs(dvar_vim[off]).max(), np.abs(dvar_zi[off]).max())
    im1 = axes[1].imshow(dvar, cmap="RdBu_r", vmin=-vmaxv, vmax=vmaxv)
    axes[1].set_title(f"VAR delta-A_eff (mean over 16 subjects)\n"
                      f"abs scale 0 - {vmaxv:.2f}")
    for ax in axes:
        ax.set_xticks(range(24)); ax.set_yticks(range(24))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("source ROI"); ax.set_ylabel("target ROI")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    fig.suptitle("delta-K vs delta-A_eff: off-diagonal Spearman "
                 f"rho = {r_all:.2f} (p = {p_all:.3g})")
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    fig.savefig(OUT / "fig_k_vs_var_side_by_side.png", dpi=120)
    plt.close(fig)
    print(f"rho(vim)={r_vim:.3f} p={p_vim:.4g}; rho(zi)={r_zi:.3f} "
          f"p={p_zi:.4g}; rho(pooled)={r_all:.3f} p={p_all:.4g}")
    print("done:", OUT)


if __name__ == "__main__":
    main()