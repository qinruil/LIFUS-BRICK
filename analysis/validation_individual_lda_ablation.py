#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Individual-model ablation: shared-K dynamics versus per-subject fits.

Drops the shared Koopman operator and refits a linear dynamical system
per subject (ridge VAR(1)) on the same preprocessed BOLD embedding used
by BRICK. For each subject and treatment target:

    A_pre, A_post       per-subject VAR(1) coefficient matrices
    delta_A[i]          Frobenius norm of row i of (A_post - A_pre),
                        the connection change driven from ROI i
    delta_C_VAR[i]      row i of C_post - C_pre (control-diagonal change)

Both are ROI-level change vectors with the same 24-region layout as the
BRICK delta-C from the LOSO analysis. The three vectors are correlated
across subjects with the same tremor outcomes used elsewhere
(postural / rest; VIM- and ZI-treated), Spearman per ROI with BH-FDR
inside each target x task x family, so the BRICK and individual-model
families are scored on identical tests.

Pairwise Spearman correlation between BRICK delta-C and delta_A
quantifies whether the shared-dynamics route and the per-subject route
recover the same or different structure.

Outputs (results/validation/individual_lda_ablation/):
    per_subject_delta_vectors.csv
    tremor_correlation_comparison.csv
    deltaC_deltaA_crosswalk.csv

Usage:
    python analysis/validation_individual_lda_ablation.py

Dependencies: numpy, pandas, scipy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARC = ROOT.parent / "fhqwd-osfstorage-archive"
OUT = ROOT / "results" / "validation" / "individual_lda_ablation"
EXCLUDED = {"sub-fuspd09", "sub-fuspd15", "sub-fuspd19"}


def fit_var(ts: np.ndarray) -> np.ndarray:
    """Ridge VAR(1): X_t = X_{t-1} A^T on (T, N) input. Returns A (N, N)."""
    x = ts - ts.mean(axis=0)
    prev, fut = x[:-1].T, x[1:].T
    lam = 1e-6 * np.trace(prev @ prev.T) / prev.shape[1]
    gram = prev @ prev.T + lam * np.eye(prev.shape[0])
    return np.linalg.solve(gram, prev @ fut.T).T


def load_session(subj, target):
    from preprocessing.load_preprocessed_data import load_all
    rec = [s for s in load_all()
           if s["subject_id"] == subj.replace("sub-", "")
           and s["target"] == target]
    if not rec:
        return None
    pre = np.asarray(rec[0]["mpre"], dtype=float)
    post = np.asarray(rec[0]["mpost"], dtype=float)
    if pre.shape[0] < pre.shape[1]:
        pre, post = pre.T, post.T
    return pre, post


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.ones(n)
    q[order] = ranked
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    delta = pd.read_csv(ROOT / "results/figures/loso"
                        / "loso_19_fold_beta_0.2_13to1to5_split"
                        / "loso_delta_summary.csv")
    subs = sorted(delta["subject_id"].unique())

    rows = []
    for subj in subs:
        for target in ("vim", "zi"):
            pair = load_session(subj, target)
            if pair is None:
                continue
            A_pre = fit_var(pair[0])
            A_post = fit_var(pair[1])
            dA = A_post - A_pre
            # C-side change vectors, same session-level quantities as BRICK
            # uses: BRICK delta-C is a per-ROI diagonal gain, VAR has no
            # diagonal-control slot, so the comparable per-ROI quantities
            # are the row norms of the change matrix (outgoing) and of the
            # original matrices (baseline scale).
            sd = delta[(delta["subject_id"] == subj) & (delta["target"] == target)]
            for _, r in sd.iterrows():
                roi = r["roi"]
                from preprocessing.load_preprocessed_data import TARGET_ROIS
                i = TARGET_ROIS.index(roi)
                rows.append({
                    "subject_id": subj, "target": target, "roi": roi,
                    "deltaC_brick": float(r["delta"]),
                    "deltaA_out_rownorm": float(np.linalg.norm(dA[i, :])),
                    "deltaA_row_rel": float(np.linalg.norm(dA[i, :])
                                            / (np.linalg.norm(A_pre[i, :]) + 1e-12)),
                })
    dv = pd.DataFrame(rows)
    keep = ~dv["subject_id"].isin(EXCLUDED)
    dv.to_csv(OUT / "per_subject_delta_vectors.csv", index=False)

    tremor = load_tremor()
    fam_rows = []
    for cohort, mask in (("19-subject", np.ones(len(dv), bool)),
                         ("16-subject", keep.values)):
        d = dv[mask]
        for target in ("vim", "zi"):
            for task in ("pos", "rest"):
                col = f"{target}_{task}"
                y = tremor.set_index("subject_id")[col]
                for fam in ("deltaC_brick", "deltaA_out_rownorm", "deltaA_row_rel"):
                    for roi in d["roi"].unique():
                        g = d[(d["target"] == target) & (d["roi"] == roi)]
                        vals = [(float(v), float(y[s]))
                                for s, v in g.set_index("subject_id")[fam].items()
                                if s in y.index]
                        if len(vals) < 12:
                            continue
                        a = np.array([v[0] for v in vals])
                        b = np.array([v[1] for v in vals])
                        r, p = spearmanr(a, b)
                        fam_rows.append({"cohort": cohort, "family": fam,
                                         "target": target, "task": task,
                                         "roi": roi, "n": len(vals),
                                         "rho": float(r), "p": float(p)})
    corr = pd.DataFrame(fam_rows)
    corr["q"] = np.nan
    for (c, f, t, k), g in corr.groupby(["cohort", "family", "target", "task"]):
        corr.loc[g.index, "q"] = bh_fdr(g["p"].to_numpy())
    corr.to_csv(OUT / "tremor_correlation_comparison.csv", index=False)

    # crosswalk: do the two routes recover the same structure?
    xw = []
    for cohort, m in (("19-subject", np.ones(len(dv), bool)),
                      ("16-subject", keep.values)):
        d = dv[m]
        for target in ("vim", "zi"):
            for roi in d["roi"].unique():
                g = d[(d["target"] == target) & (d["roi"] == roi)]
                if len(g) < 12:
                    continue
                r1, p1 = spearmanr(g["deltaC_brick"], g["deltaA_out_rownorm"])
                r2, p2 = spearmanr(g["deltaC_brick"], g["deltaA_row_rel"])
                xw.append({"cohort": cohort, "target": target, "roi": roi,
                           "rho_brick_vs_rownorm": float(r1), "p_rownorm": float(p1),
                           "rho_brick_vs_rowrel": float(r2), "p_rowrel": float(p2)})
    pd.DataFrame(xw).to_csv(OUT / "deltaC_deltaA_crosswalk.csv", index=False)

    # console summary
    c19 = corr[corr["cohort"] == "19-subject"]
    piv = c19.groupby(["family", "target", "task"]).apply(
        lambda g: pd.Series({
            "median_abs_rho": float(g["rho"].abs().median()),
            "best_abs_rho": float(g["rho"].abs().max()),
            "n_survive_q05": int((g["q"] < 0.05).sum())}),
        include_groups=False).reset_index()
    print(piv.to_string(index=False))
    x19 = pd.DataFrame(xw)
    x19 = x19[x19["cohort"] == "19-subject"]
    print("\ncrosswalk median |rho| brick vs VAR rownorm:",
          round(x19["rho_brick_vs_rownorm"].abs().median(), 3),
          "| vs row-rel:",
          round(x19["rho_brick_vs_rowrel"].abs().median(), 3))
    print("done:", OUT)


if __name__ == "__main__":
    main()
