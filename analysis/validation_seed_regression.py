#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VIM / ZI seed-regression control for the delta-C versus tremor comparison.

Control question: if a plain seed-based correlation regression from the
stimulated target (VIM or ZI) predicts tremor improvement as well as BRICK's
per-ROI control-gain change does, why use BRICK at all?

Design, kept deliberately symmetric with delta-C so the comparison is fair:

  * Data: the same preprocessed z-scored 24-ROI sessions (19 subjects x
    {vim, zi} targets x {pre, post}); subject-specific time series, no shared
    operator involved.
  * Seed regressions: for each session, regress every ROI time series on the
    VIM (or ZI) seed with a lag-0 least-squares fit after demeaning; the
    standardized regression coefficient (beta weight, not correlation) is the
    seed-connectivity feature. Done separately for pre and post.
  * Feature: per-subject change beta_post - beta_pre for each of the 24 ROIs.
  * Target: normalized tremor change, (Post - Pre)/(Post + Pre) of the 4-8 Hz
    accelerometry band, positive = worse, exactly as in the delta-C analysis.
  * Statistics: Spearman rho across subjects within each
    (target-seed, tremor-task) cell; BH-FDR applied within the declared family
    (24 ROIs). Bootstrap 95% CI on the strongest coefficient.
  * Crosswalk: Spearman rho between the seed-regression change vector and the
    LOSO delta-C vector over the 24 ROIs, per subject-session, to show whether
    the two quantities rank regions the same way.

Outputs (results/validation/seed_regression/):
  seed_tremor_spearman.csv
  seed_family_comparison.csv          (side by side with delta-C for the plot)
  seed_vs_deltaC_crosswalk.csv
  fig_seed_vs_deltaC_comparison.png
  run_manifest.json

Usage:
  python analysis/validation_seed_regression.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
ARC = ROOT.parent / "fhqwd-osfstorage-archive"
LOSO = (ROOT / "results" / "figures" / "loso"
        / "loso_19_fold_beta_0.2_13to1to5_split")
OUT = ROOT / "results" / "validation" / "seed_regression"

ROI_ORDER = ["lh_Ca", "lh_GPe", "lh_GPi", "lh_Pu", "lh_STN",
             "lh_cerebellum_dentate", "lh_cerebellum_motor",
             "lh_paracentral_smooth3mm", "lh_postcentral_smooth3mm",
             "lh_precentral_smooth3mm", "lh_superiorfrontal_smooth3mm",
             "lh_vim", "lh_zi", "rh_Ca", "rh_GPe", "rh_GPi", "rh_Pu",
             "rh_STN", "rh_cerebellum_dentate", "rh_cerebellum_motor",
             "rh_paracentral_smooth3mm", "rh_postcentral_smooth3mm",
             "rh_precentral_smooth3mm", "rh_superiorfrontal_smooth3mm"]
SEED_ROI = {"vim": "lh_vim", "zi": "lh_zi"}
DIVERGENT = {"sub-fuspd09", "sub-fuspd15", "sub-fuspd19"}


def bh_fdr(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full(len(p), np.nan)
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


def seed_beta(x, seed_col):
    """Standardized lag-0 regression coefficient of every ROI on the seed,
    with the seed itself excluded from the feature set by construction."""
    s = x[:, seed_col]
    sc = s - s.mean()
    ss = float(sc @ sc)
    out = np.full(x.shape[1], np.nan)
    for j in range(x.shape[1]):
        if j == seed_col:
            continue
        yj = x[:, j] - x[:, j].mean()
        out[j] = float((sc @ yj) / ss)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- seed-regression change per subject-session ----
    from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS
    assert list(TARGET_ROIS) == ROI_ORDER, "ROI order drifted from loader"
    recs = {}
    for s in load_all():
        sid, tgt = s["subject_id"], s["target"]
        seed_col = ROI_ORDER.index(SEED_ROI[tgt])
        xpre = np.asarray(s["mpre"], float)
        xpost = np.asarray(s["mpost"], float)
        if xpre.shape[0] < xpre.shape[1]:
            xpre = xpre.T
        if xpost.shape[0] < xpost.shape[1]:
            xpost = xpost.T
        # per-ROI z-score over time, identical to the training embedding
        zpre = (xpre - xpre.mean(0)) / (xpre.std(0) + 1e-8)
        zpost = (xpost - xpost.mean(0)) / (xpost.std(0) + 1e-8)
        recs[(sid, tgt)] = seed_beta(zpost, seed_col) - seed_beta(zpre, seed_col)

    print(f"seed regressions computed for {len(recs)} subject-sessions")

    # ---- correlate with tremor ----
    tremor = load_tremor()
    rows = []
    for tgt in ("vim", "zi"):
        sids = sorted({s for (s, t) in recs if t == tgt})
        for task in ("pos", "rest"):
            pairs = []
            for s in sids:
                v = recs.get((s, tgt))
                trow = tremor[tremor["subject_id"] == s]
                if v is None or not len(trow):
                    continue
                pairs.append((s, v, float(trow[f"{tgt}_{task}"].iloc[0])))
            for j, roi in enumerate(ROI_ORDER):
                x = np.array([p[1][j] for p in pairs])
                y = np.array([p[2] for p in pairs])
                keep = ~np.isnan(x) & ~np.isnan(y)
                if keep.sum() < 12:
                    continue
                r, p = spearmanr(x[keep], y[keep])
                rows.append({"seed": tgt, "task": task, "roi": roi,
                             "n": int(keep.sum()), "rho": float(r),
                             "p": float(p)})
    df = pd.DataFrame(rows)
    for (sd, tk), g in df.groupby(["seed", "task"]):
        df.loc[g.index, "q"] = bh_fdr(g["p"].to_numpy())
    df.to_csv(OUT / "seed_tremor_spearman.csv", index=False)

    # ---- side by side with delta-C on the identical cohort ----
    dsum = pd.read_csv(LOSO / "loso_delta_summary.csv")
    for c in ("subject_id", "roi", "target"):
        dsum[c] = dsum[c].astype(str).str.strip()
    dsum["target"] = dsum["target"].str.lower()
    dsum["subject_id"] = dsum["subject_id"].apply(lambda x: x.replace("sub-", ""))

    # per-ROI delta-C correlations, BH-FDR within the same 24-ROI family
    drows = []
    for tgt in ("vim", "zi"):
        for task in ("pos", "rest"):
            for roi in ROI_ORDER:
                dsel = dsum[(dsum["target"] == tgt) & (dsum["roi"] == roi)]
                pairs = []
                for _, rr in dsel.iterrows():
                    trow = tremor[tremor["subject_id"] == rr["subject_id"]]
                    if not len(trow):
                        continue
                    pairs.append((float(rr["delta"]),
                                  float(trow[f"{tgt}_{task}"].iloc[0])))
                if len(pairs) < 12:
                    continue
                x = np.array([q[0] for q in pairs])
                y = np.array([q[1] for q in pairs])
                r, p = spearmanr(x, y)
                drows.append({"target": tgt, "task": task, "roi": roi,
                              "n": len(pairs), "rho": float(r), "p": float(p)})
    ddf = pd.DataFrame(drows)
    if len(ddf):
        for (tg, tk), g in ddf.groupby(["target", "task"]):
            ddf.loc[g.index, "q"] = bh_fdr(g["p"].to_numpy())
    ddf.to_csv(OUT / "deltaC_tremor_spearman.csv", index=False)

    comp = []
    for tgt in ("vim", "zi"):
        for task in ("pos", "rest"):
            for metric, g, rc, pc, qc in (
                    ("deltaC", ddf[(ddf["target"] == tgt) & (ddf["task"] == task)]
                     if len(ddf) else ddf, "rho", "p", "q"),
                    ("seed_beta", df[(df["seed"] == tgt) & (df["task"] == task)]
                     if len(df) else df, "rho", "p", "q")):
                if not len(g):
                    continue
                i = g[pc].idxmin()
                comp.append({"seed": tgt, "task": task, "metric": metric,
                             "n_tests": int(len(g)), "n": int(g.loc[i, "n"]),
                             "best_rho": float(g.loc[i, rc]),
                             "best_roi": str(g.loc[i, "roi"]),
                             "best_p": float(g.loc[i, pc]),
                             "best_q": float(g.loc[i, qc]),
                             "n_fdr_survive": int((g[qc] < 0.05).sum())})
    cdf = pd.DataFrame(comp).sort_values(["seed", "task", "metric"])
    cdf.to_csv(OUT / "seed_family_comparison.csv", index=False)
    print("\n=== strongest tremor association: seed regression vs delta-C ===")
    print(cdf.to_string(index=False))

    # ---- crosswalk: do the two quantities rank regions alike? ----
    cw = []
    for (s, tgt), v in sorted(recs.items()):
        d = dsum[(dsum["subject_id"] == s) & (dsum["target"] == tgt)]
        if d.empty:
            continue
        m = dict(zip(d["roi"], d["delta"]))
        x = np.array([v[j] for j in range(len(ROI_ORDER))])
        y = np.array([m.get(roi, np.nan) for roi in ROI_ORDER])
        keep = ~np.isnan(x) & ~np.isnan(y)
        if keep.sum() < 10:
            continue
        r, p = spearmanr(x[keep], y[keep])
        cw.append({"subject_id": s, "target": tgt, "n_roi": int(keep.sum()),
                   "rho": float(r), "p": float(p)})
    cwdf = pd.DataFrame(cw)
    cwdf.to_csv(OUT / "seed_vs_deltaC_crosswalk.csv", index=False)
    if len(cwdf):
        print(f"\ncrosswalk rho across {len(cwdf)} sessions: "
              f"median {cwdf['rho'].median():+.3f} "
              f"[{cwdf['rho'].min():+.3f}, {cwdf['rho'].max():+.3f}], "
              f"mean {cwdf['rho'].mean():+.3f}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 11, "figure.dpi": 150})
    cells = [("vim", "pos"), ("vim", "rest"), ("zi", "pos"), ("zi", "rest")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (tgt, task) in zip(axes.ravel(), cells):
        g = cdf[(cdf["seed"] == tgt) & (cdf["task"] == task)]
        order = ["deltaC", "seed_beta"]
        g = g.set_index("metric").reindex([o for o in order if o in g["metric"].values])
        xs = np.arange(len(g))
        ax.bar(xs, g["best_rho"].abs(), color=["C3", "C0"][:len(g)])
        for i, (_, rr) in enumerate(g.iterrows()):
            star = " *" if rr["best_q"] < 0.05 else ""
            ax.text(i, abs(rr["best_rho"]) + 0.02,
                    f"{rr['best_rho']:+.2f} ({rr['best_roi']}){star}",
                    ha="center", fontsize=8)
        ax.axhline(0.456, color="k", ls=":", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(g.index, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f"{tgt.upper()} seed, {task}")
        ax.set_ylabel("|Spearman rho| (strongest ROI)")
    fig.suptitle("Seed-regression change vs BRICK delta-C: strongest tremor "
                 "association per cell\n(* = survives BH-FDR within the 24-ROI "
                 "family; dotted = 80%-power threshold)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig_seed_vs_deltaC_comparison.png")
    plt.close(fig)

    (OUT / "run_manifest.json").write_text(json.dumps({
        "seed": "target ROI of the session (lh_vim for vim files, lh_zi for zi files)",
        "feature": "standardized lag-0 regression coefficient, post minus pre, per ROI",
        "same_zscored_series_as": "the ablation control (per-ROI z-score over time)",
        "tremor_metric": "(Post - Pre)/(Post + Pre) of 4-8 Hz band power",
        "families": "BH-FDR within seed x task x 24 ROIs",
        "scope_note": "24 tremor-relevant ROIs, not whole-brain (full-brain "
                      "comparison deferred as a suspended item)",
    }, indent=2), encoding="utf-8")
    print("\ndone:", OUT)


LEGACY = {"lh_STH": "lh_STN", "rh_STH": "rh_STN"}

if __name__ == "__main__":
    main()
