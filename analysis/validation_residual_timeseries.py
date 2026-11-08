#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Residual-time-series test.

The circularity objection to a plain K-ablation: K is trained on pre AND post
data, so deleting K deletes a quantity shared by both conditions and "what
happens to delta-C afterwards" says nothing about the brain. The direct
replacement instead SUBTRACTS the K-predicted component
from the raw BOLD series and treats the residual as a new fMRI dataset:

    x_z  = per-ROI z-scored session (identical to the model embedding)
    x_K  = the K-propagated part of the model's reconstruction
           (manual eigenspace rollout, decomposition x_K + x_C = x_hat)
    x_R  = x_z - x_K          <- "the data after the shared filter is removed"

If the K component really hides clinically relevant signal, tremor correlations
of standard features computed on x_R should be STRONGER than on x_z. The test
is run model-free: conventional statistics are computed directly on the de-K'd
data rather than only comparing model quantities.

Pre-declared feature families (delta-C is not defined on residuals -- the
model is decomposed away -- so the three conventional families are compared):

    fcChange    per-ROI Pearson FC of each ROI against all others, mean
                Fisher-z connectivity, post minus pre;
    fALFFchange log(AFF) and fALFF changes (Welch PSD, 0.01-0.1 / 0.01-0.25);
    slowPower   log slow-band (0.01-0.10 Hz) power change.

Statistics identical to the standard-metric comparison: Spearman with the
normalized tremor change, BH-FDR within each
(data x family x target x task) declared family.

Residuals are computed with each subject's own LOSO fold model (the subject is
never in that fold's training set), so the subtraction is out-of-sample and
cannot be inflated by in-sample fit.

Inputs:  data/preprocessed_data/*.mat
         results/trained_models/loso_19_fold_beta_0.2_13to1to5_split/
             fold_*/best_model_cls.pt
         archive tremor files
Outputs (results/validation/residual_timeseries/):
    session_diagnostics.csv
    residual_feature_tremor.csv
    family_comparison_raw_vs_residual.csv
    fig_residual_tremor_comparison.png
    run_manifest.json

Usage:
    python analysis/validation_residual_timeseries.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import io as sio
from scipy.signal import welch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARC = ROOT.parent / "fhqwd-osfstorage-archive"
FOLD = (ROOT / "results" / "trained_models"
        / "loso_19_fold_beta_0.2_13to1to5_split")
OUT = ROOT / "results" / "validation" / "residual_timeseries"

TR = 2.0
SLOW = (0.01, 0.10)
AFF = (0.01, 0.25)

from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS  # noqa: E402
from models.brick import BRICK  # noqa: E402
from models.koopman_utils import compute_lambda  # noqa: E402


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


def rollout_xK(model, x_z):
    """K-propagated component of the reconstruction for one session.

    Mirrors variance_decomposition_extended.manual_rollout: at each step the
    state update splits into k_term = Lambda * g_prev (the part carried by the
    shared operator alone) and c_term = P_inv @ (C u_t) (the freshly injected
    control input). x_K is the decoder image of the k_term sequence, so
    x_K + x_C == x_hat exactly and Var(x_K) share matches the ~6% reported by
    the variance decomposition. Returns (x_K, x_hat).
    """
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
        k_terms, full_terms = [], []
        for t in range(u_c.shape[0]):
            k_term = Lambda * g_bar
            c_term = P_inv @ (C_c @ u_c[t])
            g_bar = k_term + c_term
            k_terms.append(k_term)
            full_terms.append(g_bar)
        k_traj = torch.stack(k_terms, dim=0)
        f_traj = torch.stack(full_terms, dim=0)
        x_K = (model.W_bar_x @ k_traj.T).T.real
        x_hat = (model.W_bar_x @ f_traj.T).T.real
    return x_K.numpy(), x_hat.numpy()


def mean_connectivity(x):
    """Per-ROI mean Fisher-z connectivity to all other ROIs (24-vector)."""
    c = np.corrcoef(x.T)
    z = np.arctanh(np.clip(c, -0.999999, 0.999999))
    out = np.empty(z.shape[0])
    for i in range(z.shape[0]):
        out[i] = np.nanmean(np.delete(z[i], i))
    return out


def psd_features(x):
    """Per-ROI log slow power, log AFF, log fALFF."""
    f, pxx = welch(x - x.mean(0), fs=1.0 / TR, nperseg=128, axis=0)
    slow = (f >= SLOW[0]) & (f <= SLOW[1])
    aff = (f >= AFF[0]) & (f <= AFF[1])
    ls = np.log(pxx[slow].mean(0) + 1e-12)
    la = np.log(pxx[aff].sum(0) / max(aff.sum(), 1) + 1e-12)
    lf = np.log((pxx[slow].sum(0) / pxx[aff].sum(0)) + 1e-12)
    return ls, la, lf


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    tremor = load_tremor()

    # ---- build raw / residual matrices per session ----
    feats = {}
    diag_rows = []
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

        pre = zs(s["mpre"]); post = zs(s["mpost"])
        Kpre, Hpre = rollout_xK(model, pre)
        Kpost, Hpost = rollout_xK(model, post)
        Rpre = pre - Kpre; Rpost = post - Kpost
        if not (np.isfinite(Hpre).all() and np.isfinite(Rpre).all()
                and np.isfinite(Rpost).all()):
            print(f"  {sid}/{tgt}: non-finite rollout, skipped")
            continue

        for name, (a, b) in (("raw", (pre, post)), ("residual", (Rpre, Rpost))):
            fcs_a = mean_connectivity(a); fcs_b = mean_connectivity(b)
            spa, fla, ffa = psd_features(a)
            spb, flb, ffb = psd_features(b)
            feats[(sid, tgt, name)] = {
                "fcChange": fcs_b - fcs_a,
                "fALFFchange": (ffb - ffa) - (flb - fla),
                "slowPower": spb - spa,
            }
        var_r = np.var(np.vstack([Rpre, Rpost]), axis=0).mean()
        var_x = np.var(np.vstack([pre, post]), axis=0).mean()
        nmse = float(np.mean((Hpre - pre) ** 2) / np.mean((pre - pre.mean(0)) ** 2)
                     + np.mean((Hpost - post) ** 2)
                     / np.mean((post - post.mean(0)) ** 2)) / 2.0
        if nmse > 1.0:
            # same pre-declared convergence criterion used for the excluded
            # folds: this fold model does not reconstruct its held-out
            # session, so its K component is not a usable filter estimate.
            print(f"  {sid}/{tgt}: NMSE {nmse:.2f} > 1, fold excluded from "
                  f"the residual test")
            feats.pop((sid, tgt, "raw"), None)
            feats.pop((sid, tgt, "residual"), None)
            diag_rows.append({"subject_id": sid, "target": tgt,
                              "residual_var_frac": float(var_r / var_x),
                              "k_var_frac": float(1 - var_r / var_x),
                              "nmse": nmse, "used": False})
            continue
        diag_rows.append({"subject_id": sid, "target": tgt,
                          "residual_var_frac": float(var_r / var_x),
                          "k_var_frac": float(1 - var_r / var_x),
                          "nmse": nmse, "used": True})
        print(f"  {sid}/{tgt}: residual variance fraction "
              f"{diag_rows[-1]['residual_var_frac']:.3f} "
              f"(NMSE {nmse:.3f})")

    pd.DataFrame(diag_rows).to_csv(OUT / "session_diagnostics.csv", index=False)

    # ---- correlate features with tremor, raw vs residual ----
    rows = []
    for data in ("raw", "residual"):
        for fam in ("fcChange", "fALFFchange", "slowPower"):
            for tgt in ("vim", "zi"):
                sids = sorted({k[0] for k in feats if k[1] == tgt and k[2] == data})
                for task in ("pos", "rest"):
                    for j, roi in enumerate(TARGET_ROIS):
                        x, y = [], []
                        for s in sids:
                            v = feats.get((s, tgt, data), {}).get(fam)
                            tw = tremor[tremor["subject_id"] == s]
                            if v is None or not len(tw):
                                continue
                            x.append(float(v[j])); y.append(float(tw[f"{tgt}_{task}"].iloc[0]))
                        if len(x) < 12:
                            continue
                        r, p = spearmanr(x, y)
                        rows.append({"data": data, "family": fam, "target": tgt,
                                     "task": task, "roi": roi, "n": len(x),
                                     "rho": float(r), "p": float(p)})
    df = pd.DataFrame(rows)
    for (d_, f_, t_, k_), g in df.groupby(["data", "family", "target", "task"]):
        df.loc[g.index, "q"] = bh_fdr(g["p"].to_numpy())
    df.to_csv(OUT / "residual_feature_tremor.csv", index=False)

    comp = []
    for (d_, f_, t_, k_), g in df.groupby(["data", "family", "target", "task"]):
        i = g["p"].idxmin()
        comp.append({"data": d_, "family": f_, "target": t_, "task": k_,
                     "n_tests": int(len(g)),
                     "best_abs_rho": float(abs(g.loc[i, "rho"])),
                     "best_roi": str(g.loc[i, "roi"]),
                     "best_p": float(g.loc[i, "p"]),
                     "best_q": float(g.loc[i, "q"]),
                     "n_fdr_survive": int((g["q"] < 0.05).sum()),
                     "median_abs_rho": float(g["rho"].abs().median())})
    cdf = pd.DataFrame(comp)
    cdf.to_csv(OUT / "family_comparison_raw_vs_residual.csv", index=False)
    print("\n=== family comparison: raw vs residual ===")
    print(cdf.to_string(index=False))

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 11, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fams = ["fcChange", "fALFFchange", "slowPower"]
    colors = {"raw": "0.55", "residual": "C2"}
    for ax, (tgt, task) in zip(axes.ravel(),
                               [("vim", "pos"), ("vim", "rest"),
                                ("zi", "pos"), ("zi", "rest")]):
        w = 0.35
        for fi, fam in enumerate(fams):
            for di, d_ in enumerate(("raw", "residual")):
                g = cdf[(cdf["data"] == d_) & (cdf["family"] == fam)
                        & (cdf["target"] == tgt) & (cdf["task"] == task)]
                v = float(g["best_abs_rho"].iloc[0]) if len(g) else 0.0
                ns = int(g["n_fdr_survive"].iloc[0]) if len(g) else 0
                ax.bar(fi + (di - 0.5) * w, v, w,
                       color=colors[d_], edgecolor="k", linewidth=0.4)
                if ns:
                    ax.text(fi + (di - 0.5) * w, v + 0.02, f"*{ns}",
                            ha="center", fontsize=9, color="C3")
        ax.axhline(0.456, color="k", ls=":", lw=0.8)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels(fams, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"{tgt.upper()} treated, {task}")
        ax.set_ylabel("|Spearman rho| (best ROI)")
    axes[0, 0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors["raw"]),
                               plt.Rectangle((0, 0), 1, 1, color=colors["residual"])],
                      labels=["raw z-scored BOLD", "residual (BOLD minus K term)"],
                      fontsize=9)
    fig.suptitle("Residual-time-series test: tremor association of standard "
                 "features on raw vs de-K'd data\n(*n = BH-FDR survivors in "
                 "the 24-ROI family; dotted = 80%-power threshold)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig_residual_tremor_comparison.png")
    plt.close(fig)

    (OUT / "run_manifest.json").write_text(json.dumps({
        "residual": "x_R = x_z - x_K, x_K = K-propagated rollout component, "
                    "computed with each subject's own LOSO fold model "
                    "(out-of-sample)",
        "families": ["fcChange", "fALFFchange", "slowPower"],
        "statistics": "Spearman vs normalized tremor change, BH-FDR within "
                      "data x family x target x task",
        "note": "delta-C is undefined on residuals by construction",
    }, indent=2), encoding="utf-8")
    print("\ndone:", OUT)


if __name__ == "__main__":
    main()
