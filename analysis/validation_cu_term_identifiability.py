#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cu-term identifiability check.

C and the control input u are learned jointly, so neither factor is
individually identifiable; what enters the rollout -- and what any
interpretation should rest on -- is the combined injection term
c_term = P_inv (C u_t), whose decoder image is x_C, with
x_hat = x_K + x_C exactly (same split as validation_residual_timeseries
and variance_decomposition_extended).

Two questions, same cohort, same fold models, same tremor convention:

Crosswalk: per subject-session, how consistent is the ROI ranking of
    Var(x_C)_post - Var(x_C)_pre with the ROI ranking of delta-C?
    (If high, delta-C is a reliable proxy for the Cu-term change; if low,
    the two measure different things and the text must speak of Cu.)

Tremor association: run the identical per-cell Spearman + within-family
    BH-FDR analysis (4 cells x 24 ROIs) on cu_delta that delta-C received in
    the standard-metric comparison, and count survivors for both arms in one
    table. Interpretation claims then attach to the comparable quantity.

Outputs (results/validation/cu_term_identifiability/):
    cu_variance_by_subject.csv          per subject-session per ROI variance
    cu_deltaC_crosswalk.csv             per subject-session Spearman (ROIs)
    cu_tremor_spearman.csv              both arms, cells, ROIs, rho/p/q
    family_survivor_summary.csv         survivors per arm per cell
    fig_cu_vs_dc_comparison.png         survivors + crosswalk distribution
    run_manifest.json

Usage:
    python analysis/validation_cu_term_identifiability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import io as sio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARC = ROOT.parent / "fhqwd-osfstorage-archive"
FOLD = (ROOT / "results" / "trained_models"
        / "loso_19_fold_beta_0.2_13to1to5_split")
LOSO_SUM = (ROOT / "results" / "figures" / "loso"
             / "loso_19_fold_beta_0.2_13to1to5_split"
             / "loso_delta_summary.csv")
OUT = ROOT / "results" / "validation" / "cu_term_identifiability"

from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS  # noqa: E402
from models.brick import BRICK  # noqa: E402
from models.koopman_utils import compute_lambda  # noqa: E402

ROI_ORDER = list(TARGET_ROIS)
EPS = 1e-10


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty(len(pv))
    tmp[order] = np.minimum(ranked, 1.0)
    q[ok] = tmp
    return q


def load_tremor():
    subj = [s.strip().replace("sub-", "") for s in
            (ARC / "rsfMRI" / "scripts" / "subject_list.txt")
            .read_text(encoding="utf-8").splitlines() if s.strip()]
    m = sio.loadmat(ARC / "rsfMRI" / "tremor" / "tremor_pos_rest.mat")
    return pd.DataFrame({
        "subject_id": subj,
        "vim_pos": np.asarray(m["vim_pos"]).ravel().astype(float),
        "vim_rest": np.asarray(m["vim_rest"]).ravel().astype(float),
        "zi_pos": np.asarray(m["zi_pos"]).ravel().astype(float),
        "zi_rest": np.asarray(m["zi_rest"]).ravel().astype(float)})


def zs(x):
    x = np.asarray(x, float)
    if x.shape[0] < x.shape[1]:
        x = x.T
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def rollout_components(model, x_z):
    """Exact k_term / c_term split; returns (x_K, x_C), decoder images."""
    xd = torch.tensor(x_z, dtype=torch.float32)
    with torch.no_grad():
        Lambda = compute_lambda(model.nu_log, model.theta_log)
        P_inv = model.P_inv
        g_0, _, _ = model.encoder(xd)
        C_base, u, _, _, _ = model.control(xd)
        g0_c = g_0.to(torch.complex64)
        u_c = u.to(torch.complex64)
        C_c = C_base.to(torch.complex64)
        g_bar = P_inv @ g0_c
        k_terms, c_terms = [], []
        for t in range(u_c.shape[0]):
            k_term = Lambda * g_bar
            c_term = P_inv @ (C_c @ u_c[t])
            g_bar = k_term + c_term
            k_terms.append(k_term)
            c_terms.append(c_term)
        k_traj = torch.stack(k_terms, dim=0)
        c_traj = torch.stack(c_terms, dim=0)
        x_K = (model.W_bar_x @ k_traj.T).T.real
        x_C = (model.W_bar_x @ c_traj.T).T.real
    return x_K.numpy(), x_C.numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    tremor = load_tremor()
    dsum = pd.read_csv(LOSO_SUM)
    for c in ("subject_id", "roi", "target"):
        dsum[c] = dsum[c].astype(str).str.strip()
    dsum["target"] = dsum["target"].str.lower()
    dsum["subject_id"] = dsum["subject_id"].apply(lambda x: x.replace("sub-", ""))

    var_rows, cross_rows = [], []
    cu_map, dc_map = {}, {}
    n_models = 0
    for s in sessions:
        sid, tgt = s["subject_id"], s["target"]
        ckpt = FOLD / f"fold_sub-{sid}" / "best_model_cls.pt"
        if not ckpt.exists():
            print(f"  no fold model for {sid}, skipped")
            continue
        cm = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = BRICK(use_control=cm["use_control"], use_ic=cm["use_ic"],
                      h=cm["h"], m=cm["m"])
        model.load_state_dict(cm["model_state_dict"])
        model.eval()
        n_models += 1

        pre = zs(s["mpre"])
        post = zs(s["mpost"])
        _, Cpre = rollout_components(model, pre)
        _, Cpost = rollout_components(model, post)
        vpre = Cpre.var(axis=0)
        vpost = Cpost.var(axis=0)
        cu_delta = vpost - vpre

        dsel = dsum[(dsum["subject_id"] == sid) & (dsum["target"] == tgt)]
        dc_by_roi = dict(zip(dsel["roi"], dsel["delta"]))
        dc_vec = np.array([dc_by_roi.get(r, np.nan) for r in ROI_ORDER], float)

        for j, roi in enumerate(ROI_ORDER):
            var_rows.append({"subject_id": sid, "target": tgt, "roi": roi,
                             "var_xC_pre": float(vpre[j]),
                             "var_xC_post": float(vpost[j]),
                             "cu_delta": float(cu_delta[j]),
                             "deltaC": float(dc_vec[j]) if np.isfinite(dc_vec[j]) else np.nan})
        cu_map[(sid, tgt)] = cu_delta
        dc_map[(sid, tgt)] = dc_vec

        keep = np.isfinite(cu_delta) & np.isfinite(dc_vec)
        if keep.sum() >= 8:
            r, p = spearmanr(cu_delta[keep], dc_vec[keep])
            cross_rows.append({"subject_id": sid, "target": tgt,
                               "n_rois": int(keep.sum()), "rho": float(r),
                               "p": float(p)})
        del model

    vdf = pd.DataFrame(var_rows)
    vdf.to_csv(OUT / "cu_variance_by_subject.csv", index=False)
    cdf = pd.DataFrame(cross_rows)
    cdf.to_csv(OUT / "cu_deltaC_crosswalk.csv", index=False)

    # ---- tremor association, identical cells for both arms ----
    rows = []
    for tgt in ("vim", "zi"):
        sids = sorted({s for (s, t) in cu_map if t == tgt})
        for task in ("pos", "rest"):
            pairs = []
            for s in sids:
                trow = tremor[tremor["subject_id"] == s]
                if not len(trow) or (s, tgt) not in cu_map:
                    continue
                pairs.append((s, cu_map[(s, tgt)], dc_map[(s, tgt)],
                              float(trow[f"{tgt}_{task}"].iloc[0])))
            if len(pairs) < 12:
                continue
            y = np.array([p[3] for p in pairs])
            for j, roi in enumerate(ROI_ORDER):
                xcu = np.array([p[1][j] for p in pairs])
                xdc = np.array([p[2][j] for p in pairs])
                for arm, x in (("cu_variance_delta", xcu), ("deltaC", xdc)):
                    keep = ~np.isnan(x) & ~np.isnan(y)
                    if keep.sum() < 12:
                        continue
                    r, p = spearmanr(x[keep], y[keep])
                    rows.append({"target": tgt, "task": task, "arm": arm,
                                 "roi": roi, "n": int(keep.sum()),
                                 "rho": float(r), "p": float(p)})
    adf = pd.DataFrame(rows)
    for (tg, tk, arm), g in adf.groupby(["target", "task", "arm"]):
        adf.loc[g.index, "q"] = bh_fdr(g["p"].to_numpy())
    adf.to_csv(OUT / "cu_tremor_spearman.csv", index=False)

    summary = (adf.assign(surv=(adf["q"] < 0.05).astype(int))
               .groupby(["target", "task", "arm"])
               .agg(n_tests=("roi", "size"), survivors=("surv", "sum"),
                    median_abs_rho=("rho", lambda x: float(np.nanmedian(np.abs(x)))),
                    max_abs_rho=("rho", lambda x: float(np.nanmax(np.abs(x)))))
               .reset_index())
    summary.to_csv(OUT / "family_survivor_summary.csv", index=False)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    cells = summary.copy()
    piv = cells.pivot_table(index=["target", "task"], columns="arm",
                            values="survivors")
    idx_keys = [(t, k) for t, k in piv.index]
    xs = np.arange(len(piv))
    for k, arm in enumerate(("deltaC", "cu_variance_delta")):
        ys = piv[arm].to_numpy(float)
        axes[0].bar(xs + k * 0.35, ys, 0.33,
                    label=arm.replace("_", " "))
        for xi, yi in zip(xs + k * 0.35, ys):
            axes[0].text(xi, yi + 0.15, f"{int(yi)}", ha="center", fontsize=9)
    axes[0].set_xticks(xs + 0.35 * 0.5)
    axes[0].set_xticklabels([f"{a}/{b}" for a, b in idx_keys], fontsize=9)
    axes[0].set_ylabel("BH-FDR survivors (within 24-ROI family)")
    axes[0].set_title("delta-C vs Cu-term variance change, same cells")
    axes[0].legend(fontsize=9)
    for tgt, col in (("vim", "C0"), ("zi", "C1")):
        v = cdf[cdf["target"] == tgt]["rho"]
        axes[1].hist(v, bins=8, alpha=0.55, color=col,
                     label=f"{tgt.upper()} (median |rho| = {np.nanmedian(np.abs(v)):.2f})")
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("Spearman rho across ROIs: cu_delta vs delta-C")
    axes[1].set_ylabel("subject-sessions")
    axes[1].set_title("Within-subject crosswalk (per-session, 24 ROIs)")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cu_vs_dc_comparison.png", dpi=150)
    plt.close(fig)

    manifest = {
        "fold_models_used": n_models,
        "subject_sessions": int(len(cu_map)),
        "crosswalk_median_abs_rho": float(np.nanmedian(np.abs(cdf["rho"]))) if len(cdf) else None,
        "convention": "x_C = decoder image of P_inv(C u_t) sequence; "
                      "cu_delta = Var(x_C)_post - Var(x_C)_pre on the "
                      "subject's own LOSO fold model; identical tremor and "
                      "BH-FDR family convention as the delta-C analysis",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(summary.to_string(index=False))
    print("\ncrosswalk:", manifest["crosswalk_median_abs_rho"])
    print("done:", OUT)


if __name__ == "__main__":
    main()
