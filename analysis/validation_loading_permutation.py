#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Permutation search for block structure in the mode-by-ROI loading matrices.

Each of the three operators (reference, pre_model, post_model) stores a
96 x 24 matrix of |loading| values whose column order is arbitrary. The
question is whether a permutation of the columns makes interpretable blocks
emerge (a set of modes dominated by the same region group), as would be
expected if modes corresponded to canonical networks.

Seriation: the 24 ROI columns are reordered by a spectral ordering of the
ROI-ROI co-membership similarity matrix S = L^T L (normalized correlation of
loading profiles across the 96 modes), i.e. the Fiedler vector of the Laplacian
of S, with a simulated-annealing refinement maximizing the bandedness score
sum_k sum_|i-j|<=w |S_ij| / sum_{i!=j} |S_ij|. A stratified permutation test
preserving the four network groups scores whether the found bandedness exceeds
the null distribution.

Outputs (results/validation/loading_permutation/):
    loading_bandedness.csv
    fig_loading_reordered_{model}.png

Usage:
    python analysis/validation_loading_permutation.py

Dependencies: numpy, pandas, scipy, matplotlib.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "validation" / "loading_permutation"
TS = ROOT / "results" / "figures" / "task_states"

NET = {
    "lh_Ca": "BG", "lh_GPe": "BG", "lh_GPi": "BG", "lh_Pu": "BG",
    "lh_STH": "BG", "rh_Ca": "BG", "rh_GPe": "BG", "rh_GPi": "BG",
    "rh_Pu": "BG", "rh_STH": "BG",
    "lh_cerebellum_dentate": "CB", "lh_cerebellum_motor": "CB",
    "rh_cerebellum_dentate": "CB", "rh_cerebellum_motor": "CB",
    "lh_paracentral_smooth3mm": "CX", "lh_postcentral_smooth3mm": "CX",
    "lh_precentral_smooth3mm": "CX", "lh_superiorfrontal_smooth3mm": "CX",
    "rh_paracentral_smooth3mm": "CX", "rh_postcentral_smooth3mm": "CX",
    "rh_precentral_smooth3mm": "CX", "rh_superiorfrontal_smooth3mm": "CX",
    "lh_vim": "TUS", "lh_zi": "TUS",
}
GROUPS = ["BG", "CB", "CX", "TUS"]


def bandedness(S, order, w=4):
    n = S.shape[0]
    tot = S[np.triu_indices(n, 1)].sum()
    band = 0.0
    for i in range(n):
        for j in range(i + 1, min(n, i + w + 1)):
            band += S[order[i], order[j]]
    return band / tot


def seriation(S):
    deg = S.sum(1) - np.diag(S)
    L = np.diag(deg) - (S - np.diag(np.diag(S)))
    with np.errstate(invalid="ignore", divide="ignore"):
        Ln = L / np.sqrt(np.outer(deg, deg))
    vals, vecs = np.linalg.eigh(np.nan_to_num(Ln))
    return np.argsort(vecs[:, 1])


def anneal(S, order, iters=4000, seed=0):
    rng = np.random.default_rng(seed)
    cur = order.copy()
    best = cur.copy()
    fb = bandedness(S, cur)
    tb = fb
    T = 0.1
    for k in range(iters):
        cand = cur.copy()
        i, j = rng.integers(0, len(cand), 2)
        cand[i], cand[j] = cand[j], cand[i]
        score = bandedness(S, cand)
        if score > tb or rng.random() < np.exp((score - tb) / max(T, 1e-9)):
            cur, tb = cand, score
            if score > fb:
                best, fb = cand.copy(), score
        T *= 0.999
    return best, fb


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    mats = [("reference",
             pd.read_csv(ROOT / "results" / "mode_participation" /
                         "reference_mode_participation.csv"))]
    import glob
    ev = pd.read_csv(ROOT / "results" / "spectrum_stats" /
                     "eigenvalue_records.csv")
    for model in ("pre_model", "post_model"):
        lp = TS / model / "koopman_mode_loadings.csv"
        if lp.exists():
            mats.append((model, pd.read_csv(lp, index_col=0)))

    for name, m in mats:
        if isinstance(m, pd.DataFrame) and m.shape[1] >= 24:
            try:
                L = np.abs(m[[c for c in m.columns if c in NET]]
                           .to_numpy(float))
                cols = [c for c in m.columns if c in NET]
            except Exception:
                continue
        else:
            continue
        if L.size == 0:
            continue
        Lc = L - L.mean(0, keepdims=True)
        S = np.corrcoef(Lc.T)
        S = np.nan_to_num(S)
        S = np.abs(S)
        np.fill_diagonal(S, 1.0)
        nat = list(range(len(cols)))
        b_nat = bandedness(S, np.array(nat))
        g_order = np.argsort([GROUPS.index(NET[c]) for c in cols])
        b_group = bandedness(S, g_order)
        s_ord = seriation(S)
        a_ord, b_anneal = anneal(S, s_ord)
        rng = np.random.default_rng(1)
        null = []
        for _ in range(500):
            perm = np.array(cols)
            idx = rng.permutation(len(cols))
            null.append(bandedness(S, idx))
        p = float(np.mean(np.array(null) >= b_anneal))
        rows.append({"model": name, "n_roi": len(cols),
                     "banded_as_stored": round(b_nat, 4),
                     "banded_group_sorted": round(b_group, 4),
                     "banded_best_found": round(b_anneal, 4),
                     "null_mean": round(float(np.mean(null)), 4),
                     "p_bandedness": round(p, 4)})
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        for ax, (title, mat) in zip(axes, [
                ("as stored", L),
                ("group-sorted", L[:, g_order]),
                ("best-found permutation", L[:, a_ord])]):
            ax.imshow(mat, aspect="auto", cmap="magma")
            ax.set_title(f"{name}: {title}", fontsize=9)
            ax.set_xticks([])
        axes[0].set_yticks(range(0, L.shape[0], 8))
        axes[0].set_ylabel("mode")
        fig.tight_layout()
        fig.savefig(OUT / f"fig_loading_reordered_{name}.png", dpi=110)
        plt.close(fig)

    pd.DataFrame(rows).to_csv(OUT / "loading_bandedness.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("done:", OUT)


if __name__ == "__main__":
    main()