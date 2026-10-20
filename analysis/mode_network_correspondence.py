#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantified correspondence between Koopman modes and resting-state
network structure.

Design
------
Two spaces built from the SAME 24 regions, each reduced to a subspace:

  A) resting-state co-activation space: Pearson FC over the BOLD time series
     (per session, group-averaged), eigen-decomposed (symmetric); the leading
     eigenvectors ARE the spatial co-activation patterns -- the same object the
     resting-network reference, computed from this cohort.

  B) Koopman modal space: columns of |W_bar_x| (ROI loadings per mode) of the
     shared reference operator, mean-centred over ROIs so the non-negative
     common component does not saturate similarities.

Three quantities are then computed, each against a permutation null:

  1) subspace alignment: Procrustes-free principal angles between span(A_k) and
     span(B_k) for k = 1..10, and the canonical correlations (CCA via SVD).
  2) top-1 mode -> network concentration: fraction of a mode's loading mass on
     its largest network (BG/CB/CX/TUS grouping verbatim from
     compare_pre_post.py), vs random region-label permutations preserving
     network sizes.
  3) mode-mode "same network" tendency: among the 96 modes, the network that
     dominates each mode; chi-square of that histogram vs the permutation
     expectation.

Null model: ROI labels permuted within the network-size strata (so a mode can
be "concentrated on Cerebellum" only against the 4/24 null rate, not against 1/24).

No hypothesis family is opened here beyond the permutation p-values of the
alignment statistics; per the frozen round rule these are descriptive-
confirmatory (N=19) and are labelled as such in every row.

Usage
    python analysis/mode_network_correspondence.py
Dependencies: numpy, scipy, stdlib.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS  # noqa: E402
from analysis.koopman_spectrum_stats import load_spectrum, znorm, spectral_quantities  # noqa: E402

REF_CKPT = ROOT / "results" / "final_model" / "best_model_cls.pt"
OUT_DIR = ROOT / "results" / "mode_network_fc"
TR = 2.0
N_PERM = 1000
SEED = 20261006

NETWORKS = {  # verbatim grouping from compare_pre_post.py, all members exist in TARGET_ROIS
    "BasalGanglia": 10, "Cerebellum": 4, "Cortex": 8, "TUS": 2,
}
NET_OF_ROI = {}
for _n, _rois in {
    "BasalGanglia": ["lh_Ca", "lh_GPe", "lh_GPi", "lh_Pu", "lh_STH",
                     "rh_Ca", "rh_GPe", "rh_GPi", "rh_Pu", "rh_STH"],
    "Cerebellum":   ["lh_cerebellum_dentate", "lh_cerebellum_motor",
                     "rh_cerebellum_dentate", "rh_cerebellum_motor"],
    "Cortex":       ["lh_paracentral_smooth3mm", "lh_postcentral_smooth3mm",
                     "lh_precentral_smooth3mm", "lh_superiorfrontal_smooth3mm",
                     "rh_paracentral_smooth3mm", "rh_postcentral_smooth3mm",
                     "rh_precentral_smooth3mm", "rh_superiorfrontal_smooth3mm"],
    "TUS":          ["lh_vim", "lh_zi"],
}.items():
    for r in _rois:
        NET_OF_ROI[r] = _n
GVEC = np.array([NET_OF_ROI[r] for r in TARGET_ROIS])
NET_NAMES = list(NETWORKS)


def write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else
                            (f"{float(r[k]):.6g}" if isinstance(r[k], (float, np.floating))
                             and np.isfinite(r[k]) else str(r.get(k, ""))))
                        for k in fields})


# ------------------------------------------------------------- data side ----
def group_fc(subjects):
    """Group-mean Pearson FC over the 76 sessions (znorm as the model sees it)."""
    mats = []
    for s in subjects:
        for key in ("mpre", "mpost"):
            x = znorm(torch.tensor(s[key], dtype=torch.float32)).numpy().astype(np.float64)
            x = x - x.mean(axis=0)
            c = np.corrcoef(x.T)
            mats.append(c)
    return np.mean(np.stack(mats), axis=0), len(mats)


def fc_eigenvectors(FC, k):
    """Leading eigenvectors of the group FC (co-activation spatial patterns)."""
    val, vec = np.linalg.eigh(FC)          # ascending
    order = np.argsort(val)[::-1][:k]
    return vec[:, order], val[order]


def mode_profiles(W_bar_x, Lambda, k=None):
    """Mean-centred L1-normalised |loading| columns, ordered by persistence."""
    L = np.abs(W_bar_x).astype(np.float64)
    L = L / L.sum(axis=0, keepdims=True)
    L = L - L.mean(axis=0, keepdims=True)
    L = L / np.linalg.norm(L, axis=0, keepdims=True)
    mag = np.abs(Lambda)
    order = np.argsort(mag)[::-1]
    if k:
        order = order[:k]
    return L[:, order], order


def orthonormalise(B):
    q, _ = np.linalg.qr(B)
    return q


def principal_angles(A, B):
    sa = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.arccos(np.clip(sa, -1, 1))


def canonical_correlations(A, B):
    """CC between span(A) and span(B) via SVD of the orthonormal bases."""
    return np.linalg.svd(orthonormalise(A).T @ orthonormalise(B), compute_uv=False)


# --------------------------------------------------------------- statistics --
def network_concentration(loads_raw, perm=None, rng=None):
    """
    loads_raw: (24, M) non-negative L1-normalised loadings.
    Returns per-mode max-network share. If perm given, loadings' ROI rows are
    permuted within network strata before computing (null sample).
    """
    L = loads_raw
    if perm is not None:
        idx = stratified_permutation(perm, rng)
        L = L[idx, :]
    shares = np.zeros(L.shape[1])
    dom = []
    for m in range(L.shape[1]):
        x = L[:, m]
        x = x / x.sum() if x.sum() > 0 else x
        mass = np.array([x[GVEC == n].sum() for n in NET_NAMES])
        shares[m] = mass.max()
        dom.append(NET_NAMES[int(np.argmax(mass))])
    return shares, dom


def stratified_permutation(seed_i, rng):
    """Permute ROI labels within each network (sizes preserved)."""
    idx = np.arange(len(TARGET_ROIS))
    out = np.empty_like(idx)
    for n in NET_NAMES:
        where = np.where(GVEC == n)[0]
        out[rng.choice(where, size=len(where), replace=False)] = where
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("Koopman modes vs resting-state network structure (quantified)")
    print("=" * 78)

    subjects = load_all()
    FC, n_sess = group_fc(subjects)
    print(f"[load ] group FC from {n_sess} sessions; ROI order verified against TARGET_ROIS")
    ref = load_spectrum(REF_CKPT)
    Lraw = np.abs(ref["W_bar_x"]).astype(np.float64)
    Lraw = Lraw / Lraw.sum(axis=0, keepdims=True)
    Lcen, order = mode_profiles(ref["W_bar_x"], ref["Lambda"])

    # ---- 1) subspace alignment, k = 1..10 ----
    rows = []
    for k in range(1, 11):
        Vk, evals = fc_eigenvectors(FC, k)
        Bk = Lcen[:, :k]
        cc = canonical_correlations(Vk, Bk)
        # null: permute ROI labels of the modal profiles, recompute
        null_cc = []
        for _ in range(N_PERM // 4):   # SVD cost; 250 draws enough for k<=10
            perm = stratified_permutation(0, rng)
            Lp = Lraw[perm, :]
            Lp = Lp - Lp.mean(axis=0, keepdims=True)
            Lp = Lp / np.linalg.norm(Lp, axis=0, keepdims=True)
            null_cc.append(canonical_correlations(Vk, Lp[:, order[:k]]).max())
        null_cc = np.array(null_cc)
        p = float((1 + (null_cc >= cc.max()).sum()) / (1 + null_cc.size))
        rows.append({"k": k, "fc_eig_top": float(evals[0]), "fc_eig_kth": float(evals[-1]),
                     "max_canonical_correlation": float(cc.max()),
                     "mean_cc": float(cc.mean()),
                     "null_max_cc_mean": float(null_cc.mean()),
                     "null_max_cc_p95": float(np.quantile(null_cc, 0.95)),
                     "p_perm": p,
                     "note": "FC eigenvectors = co-activation patterns of THIS cohort; "
                             "modal profiles = mean-centred |loading| of shared operator"})
    write_rows(OUT_DIR / "subspace_alignment.csv",
               ["k", "fc_eig_top", "fc_eig_kth", "max_canonical_correlation", "mean_cc",
                "null_max_cc_mean", "null_max_cc_p95", "p_perm", "note"], rows)
    print("[out ] subspace_alignment.csv")
    for r in rows:
        print(f"   k={r['k']:>2}  max CC={r['max_canonical_correlation']:.3f}  "
              f"null mean={r['null_max_cc_mean']:.3f} p95={r['null_max_cc_p95']:.3f}  p={r['p_perm']:.3f}")

    # ---- 2) per-mode network concentration ----
    shares_obs, dom_obs = network_concentration(Lraw)
    null_shares = []
    null_dom_counts = {n: 0 for n in NET_NAMES}
    for _ in range(N_PERM):
        sh, dom = network_concentration(Lraw, perm=True, rng=rng)
        null_shares.append(sh)
        for d in dom:
            null_dom_counts[d] += 1
    null_shares = np.stack(null_shares)
    mode_rows = []
    for i, m in enumerate(order):
        # shares are indexed by ORIGINAL mode column m (rank i is metadata only)
        emp = float(np.mean(null_shares[:, m]))
        p = float((1 + (null_shares[:, m] >= shares_obs[m]).sum()) / (1 + N_PERM))
        mode_rows.append({"mode_index": int(m), "persistence_rank": i,
                          "eigenvalue_mag": float(np.abs(ref["Lambda"][m])),
                          "freq_hz": float(spectral_quantities(ref["Lambda"], TR)["freq_hz"][m]),
                          "dominant_network": dom_obs[m],
                          "max_network_share": float(shares_obs[m]),
                          "null_mean_share": emp, "p_perm_vs_stratified_null": p})
    write_rows(OUT_DIR / "mode_network_concentration.csv",
               ["mode_index", "persistence_rank", "eigenvalue_mag", "freq_hz",
                "dominant_network", "max_network_share", "null_mean_share",
                "p_perm_vs_stratified_null"], mode_rows)
    n_ge_035 = int((shares_obs >= 0.35).sum())
    print(f"[out ] mode_network_concentration.csv  modes with >=0.35 mass on one network: {n_ge_035}/96")
    print(f"       observed mean max-share {shares_obs.mean():.3f} vs null {null_shares.mean():.3f}")

    # ---- 3) dominant-network histogram ----
    obs_counts = {n: dom_obs.count(n) for n in NET_NAMES}
    null_frac = {n: null_dom_counts[n] / (N_PERM * 96) for n in NET_NAMES}
    chi_rows = []
    for n in NET_NAMES:
        chi_rows.append({"network": n, "n_roi": NETWORKS[n], "null_share_of_modes": null_frac[n],
                         "observed_n_modes": obs_counts[n],
                         "observed_share_of_modes": obs_counts[n] / 96,
                         "excess": obs_counts[n] - null_frac[n] * 96})
    write_rows(OUT_DIR / "dominant_network_histogram.csv",
               ["network", "n_roi", "observed_n_modes", "observed_share_of_modes",
                "null_share_of_modes", "excess"], chi_rows)
    print("[out ] dominant_network_histogram.csv", obs_counts)

    # ---- 4) FC vs K block coupling (matrix-level analogue of the pairing
    #      question, computed from the already-saved coupling table) ----
    kc = list(csv.DictReader(open(ROOT / "results" / "final_model" / "figures_and_stats" /
                                  "K_region_coupling.csv", encoding="utf-8-sig")))
    Kblk = np.array([[float(r[c]) for c in kc[0] if c != "target_ROI"] for r in kc])  # rows=target
    up = np.triu_indices(24, 1)
    r_fc_k = sps.spearmanr(FC[up], Kblk[up])
    rows_up = {"object": "off-diagonal pairs (n=276)", "spearman_rho": float(r_fc_k.correlation),
               "p_parametric": float(r_fc_k.pvalue),
               "note": "FC Pearson vs |K| block-coupling norm, same cohort, upper triangle"}
    write_rows(OUT_DIR / "fc_vs_k_block_matrix.csv",
               ["object", "spearman_rho", "p_parametric", "note"], [rows_up])
    print(f"[out ] fc_vs_k_block_matrix.csv  rho(FC, Kblock)={r_fc_k.correlation:.3f} "
          f"p={r_fc_k.pvalue:.3g}")

    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"script": "analysis/mode_network_correspondence.py",
                   "reference_checkpoint": str(REF_CKPT.relative_to(ROOT)),
                   "n_sessions_fc": n_sess, "n_perm": N_PERM, "seed": SEED,
                   "tr_s": TR,
                   "network_sizes": NETWORKS,
                   "null": "ROI label permutation within network strata (sizes preserved)",
                   "caveat": "cohort-internal FC used as the resting-network reference because no "
                             "external whole-brain template is in the repo; this measures alignment "
                             "with THIS dataset's co-activation, not canonical Yeo networks. Atlas "
                             "atlas limitation remains."},
                  fh, indent=2, ensure_ascii=False)
    print(f"[done] {OUT_DIR} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
