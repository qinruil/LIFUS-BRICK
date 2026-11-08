#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fourth-outlier sensitivity check.

Two questions, answered once with a pre-declared candidate (sub-fuspd24) so the
analysis cannot be tuned after seeing the result:

  (a) Identity: is the point at the right edge of the delta-C versus tremor
      scatter -- the one described as pulling the slope up -- the subject that
      was also considered for exclusion at the time (sub-fuspd24)?
      Read operationally: among the pre-specified ROIs, is sub-fuspd24 the
      maximum delta-C observation, or within the top rank, while its tremor
      change is not correspondingly extreme?

  (b) Influence: does keeping or dropping it change the reported relationship?
      For each (target, task, ROI) cell we report the OLS slope of tremor change
      on delta-C, the Spearman rho, and the BH-FDR survivor count over the
      declared family, computed on the 16-subject primary cohort and on the
      same cohort minus sub-fuspd24 (15 subjects). Leave-one-out slope swings
      over all 16 subjects are reported as the internal yardstick, so a
      sub-fuspd24-specific effect can be distinguished from ordinary leverage.

Inputs (read only):
  results/figures/loso/loso_19_fold_beta_0.2_13to1to5_split/
      loso_delta_summary_filtered.csv
  fhqwd-osfstorage-archive/rsfMRI/tremor/tremor_pos_rest.mat
  fhqwd-osfstorage-archive/rsfMRI/scripts/subject_list.txt

Outputs (results/validation/fourth_outlier/):
  candidate_identity.csv
  influence_16_vs_15.csv
  leave_one_out_slope_range.csv
  fig_fourth_outlier.png
  run_manifest.json

Usage:
  python analysis/validation_fourth_outlier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
ARC = ROOT.parent / "fhqwd-osfstorage-archive"
LOSO = (ROOT / "results" / "figures" / "loso"
        / "loso_19_fold_beta_0.2_13to1to5_split")
OUT = ROOT / "results" / "validation" / "fourth_outlier"

CANDIDATE = "sub-fuspd24"
SPECIFIED_ROIS = ["lh_GPi", "lh_STN", "rh_STN"]
PRE_COHORT = ["lh_Ca", "lh_GPe", "lh_GPi", "lh_Pu", "lh_STN",
              "lh_cerebellum_dentate", "lh_cerebellum_motor",
              "lh_paracentral_smooth3mm", "lh_postcentral_smooth3mm",
              "lh_precentral_smooth3mm", "lh_superiorfrontal_smooth3mm",
              "lh_vim", "lh_zi", "rh_Ca", "rh_GPe", "rh_GPi", "rh_Pu",
              "rh_STN", "rh_cerebellum_dentate", "rh_cerebellum_motor",
              "rh_paracentral_smooth3mm", "rh_postcentral_smooth3mm",
              "rh_precentral_smooth3mm", "rh_superiorfrontal_smooth3mm"]


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    ok = ~np.isnan(p)
    q = np.full(n, np.nan)
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


def load_delta(path):
    df = pd.read_csv(path)
    for c in ("subject_id", "roi", "target"):
        df[c] = df[c].astype(str).str.strip()
    df["target"] = df["target"].str.lower()
    return df


def load_tremor():
    subj = [s.strip() for s in
            (ARC / "rsfMRI" / "scripts" / "subject_list.txt")
            .read_text(encoding="utf-8").splitlines() if s.strip()]
    m = sio.loadmat(ARC / "rsfMRI" / "tremor" / "tremor_pos_rest.mat")
    return pd.DataFrame({
        "subject_id": subj,
        "vim_pos": np.asarray(m["vim_pos"]).ravel().astype(float),
        "vim_rest": np.asarray(m["vim_rest"]).ravel().astype(float),
        "zi_pos": np.asarray(m["zi_pos"]).ravel().astype(float),
        "zi_rest": np.asarray(m["zi_rest"]).ravel().astype(float)})


def cell(dsum, tremor, sid_list, roi, tgt, task):
    """Return (x=deltaC, y=tremor change) aligned pairs for one cell."""
    x, y = [], []
    for sid in sid_list:
        d = dsum[(dsum["subject_id"] == sid) & (dsum["roi"] == roi)
                 & (dsum["target"] == tgt)]["delta"]
        t = tremor[tremor["subject_id"] == sid][f"{tgt}_{task}"]
        if len(d) == 0 or len(t) == 0:
            continue
        x.append(float(d.iloc[0]))
        y.append(float(t.iloc[0]))
    return np.array(x), np.array(y)


def ols(x, y):
    if len(x) < 3 or np.allclose(x.std(), 0):
        return np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dsum = load_delta(LOSO / "loso_delta_summary_filtered.csv")
    tremor = load_tremor()
    subs16 = sorted(dsum["subject_id"].unique())
    subs15 = [s for s in subs16 if s != CANDIDATE]
    cand_in = CANDIDATE in subs16
    print(f"primary cohort n={len(subs16)}, candidate present: {cand_in}, "
          f"sensitivity cohort n={len(subs15)}")

    # ---------------- (a) identity: rank of the candidate ----------------
    rows = []
    for tgt in ("vim", "zi"):
        for roi in SPECIFIED_ROIS + PRE_COHORT:
            x, y = cell(dsum, tremor, subs16, roi, tgt, "rest")
            if len(x) < 5:
                continue
            order = np.argsort(x)                 # ascending delta-C
            rank_hi = int(np.where(x[order] >= x.max() - 1e-15)[0][0]) if len(x) else 0
            top_ids = [subs16[i] for i in order[::-1][:3]]
            pairs = list(zip(subs16, x, y))
            cd = [xi for sid, xi, _ in pairs if sid == CANDIDATE]
            cy = [yi for sid, _, yi in pairs if sid == CANDIDATE]
            if not cd:
                continue
            cd, cy = float(cd[0]), float(cy[0])
            xs_sorted = np.sort(x)
            pctile = float(np.searchsorted(xs_sorted, cd) / (len(x) - 1))
            ys_sorted = np.sort(np.abs(y))
            y_abs_pct = float(np.searchsorted(ys_sorted, abs(cy)) / (len(y) - 1))
            rows.append({
                "target": tgt, "roi": roi, "n": len(x),
                "deltaC_candidate": round(cd, 6),
                "deltaC_max": float(x.max()),
                "deltaC_rank_desc": int((x > cd).sum() + 1),
                "deltaC_percentile": round(pctile, 3),
                "top3_deltaC_subjects": "|".join(top_ids),
                "candidate_is_top1": bool(top_ids and top_ids[0] == CANDIDATE),
                "tremor_change_candidate": round(cy, 4),
                "abs_tremor_percentile": round(y_abs_pct, 3),
                "candidate_is_extreme_x_not_y": bool(
                    pctile >= 0.90 and y_abs_pct < 0.75),
            })
    ident = pd.DataFrame(rows)
    ident.to_csv(OUT / "candidate_identity.csv", index=False)
    print("\n=== (a) identity, pre-specified ROIs ===")
    print(ident[ident["roi"].isin(SPECIFIED_ROIS)].to_string(index=False))

    # ---------------- (b) influence on slope and FDR ----------------
    fam_rows = []
    for tgt in ("vim", "zi"):
        for task in ("pos", "rest"):
            for cohort, sid_list, tag in ((subs16, subs16, "16-subject"),
                                          (subs15, subs15, "15-minus-candidate")):
                recs = []
                for roi in PRE_COHORT:
                    x, y = cell(dsum, tremor, sid_list, roi, tgt, task)
                    if len(x) < 5:
                        continue
                    b, a = ols(x, y)
                    r, p = (spearmanr(x, y) if len(x) > 2 else (np.nan, np.nan))
                    recs.append({"roi": roi, "n": len(x), "slope": b,
                                 "intercept": a, "rho": float(r),
                                 "p": float(p)})
                g = pd.DataFrame(recs)
                if not len(g):
                    continue
                g["q"] = bh_fdr(g["p"].to_numpy())
                for _, rr in g.iterrows():
                    fam_rows.append({"target": tgt, "task": task,
                                     "cohort": tag, "roi": rr["roi"],
                                     "n": int(rr["n"]), "slope": rr["slope"],
                                     "rho": rr["rho"], "p": rr["p"],
                                     "q": rr["q"],
                                     "fdr_survive": bool(rr["q"] < 0.05)})
    infl = pd.DataFrame(fam_rows)
    infl.to_csv(OUT / "influence_16_vs_15.csv", index=False)

    summ = (infl.groupby(["target", "task", "cohort"])
            .agg(n_tests=("roi", "size"),
                 n_fdr=("fdr_survive", "sum"),
                 median_abs_slope=("slope", lambda s: float(np.median(np.abs(s)))),
                 max_abs_rho=("rho", lambda s: float(np.nanmax(np.abs(s)))))
            .reset_index())
    wide = summ.pivot(index=["target", "task"], columns="cohort",
                      values=["n_fdr", "median_abs_slope", "max_abs_rho"])
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()
    for c in ("n_fdr",):
        pass
    print("\n=== (b) family-level influence ===")
    print(wide.to_string(index=False))

    # leave-one-out slope swing per cell, as internal yardstick
    loo_rows = []
    for tgt in ("vim", "zi"):
        for task in ("pos", "rest"):
            for roi in PRE_COHORT:
                x, y = cell(dsum, tremor, subs16, roi, tgt, task)
                if len(x) < 6:
                    continue
                b_full, _ = ols(x, y)
                sl = {}
                for i, sid in enumerate(subs16):
                    keep = [j for j in range(len(x)) if j != i]
                    b, _ = ols(x[keep], y[keep])
                    sl[sid] = b
                vals = np.array([v for v in sl.values() if not np.isnan(v)])
                if len(vals) < 3:
                    continue
                cand = sl.get(CANDIDATE, np.nan)
                loo_rows.append({
                    "target": tgt, "task": task, "roi": roi,
                    "slope_16": b_full,
                    "loo_min": float(vals.min()), "loo_max": float(vals.max()),
                    "loo_range": float(vals.max() - vals.min()),
                    "slope_without_candidate": float(cand),
                    "candidate_shift": float(cand - b_full) if not np.isnan(cand) else np.nan,
                })
    loo = pd.DataFrame(loo_rows)

    # leave-one-out FDR survivor counts over the 24-ROI family, per cell
    fam_loo = []
    for tgt in ("vim", "zi"):
        for task in ("pos", "rest"):
            base_surv = None
            for sid in [None] + list(subs16):
                sid_list = subs16 if sid is None else [s for s in subs16 if s != sid]
                ps = []
                for roi2 in PRE_COHORT:
                    x2, y2 = cell(dsum, tremor, sid_list, roi2, tgt, task)
                    if len(x2) < 5:
                        continue
                    ps.append(spearmanr(x2, y2)[1])
                surv = int((bh_fdr(np.array(ps)) < 0.05).sum()) if ps else np.nan
                if sid is None:
                    base_surv = surv
                else:
                    fam_loo.append({"target": tgt, "task": task,
                                    "dropped": sid, "n": len(sid_list),
                                    "n_fdr_survivors": surv,
                                    "base_16_survivors": base_surv})
    fam_loo = pd.DataFrame(fam_loo)
    fam_loo.to_csv(OUT / "family_fdr_leave_one_out.csv", index=False)
    if len(fam_loo):
        piv = fam_loo.groupby(["target", "task"]).agg(
            base=("base_16_survivors", "first"),
            loo_min=("n_fdr_survivors", "min"),
            loo_max=("n_fdr_survivors", "max"),
            without_candidate=("n_fdr_survivors",
                               lambda s: float(s[
                                   fam_loo.loc[s.index, "dropped"] == CANDIDATE
                               ].iloc[0]) if (
                                   fam_loo.loc[s.index, "dropped"] == CANDIDATE
                               ).any() else np.nan),
            max_shift_dropped=("n_fdr_survivors", "idxmax"),
        ).reset_index()
        print("\n=== family-level FDR survivors, leave-one-out range ===")
        print(piv.to_string(index=False))
    loo.to_csv(OUT / "leave_one_out_slope_range.csv", index=False)
    if len(loo):
        sig = loo[loo["roi"].isin(SPECIFIED_ROIS)]
        print("\n=== leave-one-out slope swing, pre-specified ROIs ===")
        print(sig.to_string(index=False))
        print(f"\nmax |candidate shift| over {len(loo)} cells: "
              f"{loo['candidate_shift'].abs().max():.4f}")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 11, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, roi in enumerate(SPECIFIED_ROIS):
        for i, tgt in enumerate(("vim", "zi")):
            ax = axes[i, j]
            x, y = cell(dsum, tremor, subs16, roi, tgt, "rest")
            if len(x) < 5:
                ax.axis("off"); continue
            other = np.array([s != CANDIDATE for s in subs16])
            ax.scatter(x[other], y[other], s=40, alpha=0.75, edgecolor="k",
                       linewidth=0.4, label="cohort")
            ax.scatter(x[~other], y[~other], s=95, marker="D", color="C3",
                       edgecolor="k", linewidth=0.6,
                       label=f"{CANDIDATE} (candidate)")
            b16, a16 = ols(x, y)
            b15, a15 = ols(x[other], y[other])
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, a16 + b16 * xx, color="C0", lw=1.3,
                    label=f"slope 16 = {b16:+.3f}")
            ax.plot(xx, a15 + b15 * xx, color="C3", lw=1.3, ls="--",
                    label=f"slope 15 = {b15:+.3f}")
            r16 = spearmanr(x, y).statistic
            r15 = spearmanr(x[other], y[other]).statistic
            ax.set_title(f"{roi}, {tgt.upper()} (rho {r16:+.2f} -> {r15:+.2f})",
                         fontsize=11)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xlabel(r"$\Delta C$")
            if j == 0:
                ax.set_ylabel(f"tremor change, {tgt.upper()} (rest)")
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Fourth-outlier sensitivity: sub-fuspd24 influence on the "
                 "delta-C / tremor relationship (16 vs 15 subjects)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_fourth_outlier.png")
    plt.close(fig)

    manifest = {
        "candidate": CANDIDATE,
        "candidate_in_primary_cohort": bool(cand_in),
        "primary_cohort_n": len(subs16),
        "sensitivity_cohort_n": len(subs15),
        "families": "per target x task x 24 ROIs, BH-FDR within family",
        "pre_specified_rois": SPECIFIED_ROIS,
        "tremor_metric": "(Post - Pre)/(Post + Pre) of 4-8 Hz band power",
        "figures": ["fig_fourth_outlier.png"],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    print("\ndone:", OUT)


if __name__ == "__main__":
    main()
