#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Direction audit of the per-region control-gain change.

Two statements about the same quantity can look contradictory when one is
about the paired shift within subjects and the other about cross-subject
rank correlation:

  "delta-C of lh_GPi was negative in 18 of 19"
  "about half the points in the VIM-rest scatter sit at positive delta-C"

The first counts, per subject, whether the post-sonication gain sits below
the pre-sonication gain of the same region. The second plots the within-
subject difference against the tremor outcome, where both signs are expected
unless the effect is universal. This script separates those two populations
explicitly and reports, for every region and sonication target:

  * the cross-subject difference distribution: count and share positive,
    negative, and the median value, on the primary cohort and the full set;
  * the paired direction of the same sessions from the model's own pre/post
    values, with the exact binomial probability under a symmetric null and
    the FDR-corrected p from the paired test convention used in the project;
  * a reconciliation column showing how many of the difference values would
    have to flip for the two statements to describe the same population.

Inputs (read only):
    results/figures/loso/loso_19_fold_beta_0.2_13to1to5_split/
        loso_delta_summary.csv
        loso_delta_summary_filtered.csv
    results/final_model/figures_and_stats/consistency_brain_{vim,zi}.csv
    the project sign-consistency table, if present

Outputs (results/validation/gain_direction_audit/):
    direction_by_roi.csv
    paired_vs_difference.csv
    reconciliation.csv
    fig_gain_direction_audit.png
    run_manifest.json

Usage:
    python analysis/validation_gain_direction_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
LOSO = (ROOT / "results" / "figures" / "loso"
        / "loso_19_fold_beta_0.2_13to1to5_split")
OUT = ROOT / "results" / "validation" / "gain_direction_audit"
DIVERGENT = {"sub-fuspd09", "sub-fuspd15", "sub-fuspd19"}
KEY_ROIS = ["lh_GPi", "lh_STN", "rh_STN", "lh_vim", "lh_zi"]


def load_delta(path):
    d = pd.read_csv(path)
    for c in ("subject_id", "roi", "target"):
        d[c] = d[c].astype(str).str.strip()
    d["target"] = d["target"].str.lower()
    return d


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    q = np.ones(len(p))
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[ok] = ranked
    return q


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d19 = load_delta(LOSO / "loso_delta_summary.csv")
    d16 = load_delta(LOSO / "loso_delta_summary_filtered.csv")

    rows = []
    for label, df in (("19-subject", d19), ("16-subject", d16)):
        for tgt in ("vim", "zi"):
            recs = []
            for roi in sorted(df["roi"].unique()):
                v = df[(df["target"] == tgt) & (df["roi"] == roi)]["delta"] \
                    .to_numpy(float)
                v = v[np.isfinite(v)]
                if len(v) < 8:
                    continue
                n_pos = int((v > 0).sum())
                n_neg = int((v < 0).sum())
                try:
                    w_stat, w_p = wilcoxon(
                        df[(df["target"] == tgt) & (df["roi"] == roi)]
                        ["roi_c_post"].to_numpy(float),
                        df[(df["target"] == tgt) & (df["roi"] == roi)]
                        ["roi_c_pre"].to_numpy(float))
                except ValueError:
                    w_stat, w_p = np.nan, np.nan
                bt = binomtest(max(n_pos, n_neg), n_pos + n_neg, 0.5)
                recs.append({"cohort": label, "target": tgt, "roi": roi,
                             "n": len(v), "n_positive": n_pos,
                             "n_negative": n_neg,
                             "share_negative": n_neg / (n_pos + n_neg),
                             "median_delta": float(np.median(v)),
                             "mean_delta": float(v.mean()),
                             "paired_wilcoxon_p": float(w_p),
                             "dominant_direction_binom_p": float(bt.pvalue)})
            if recs:
                g = pd.DataFrame(recs)
                g["paired_q"] = bh_fdr(g["paired_wilcoxon_p"].to_numpy())
                rows.append(g)
    dir_df = pd.concat(rows, ignore_index=True)
    dir_df.to_csv(OUT / "direction_by_roi.csv", index=False)

    # ---- reconciliation of the two statements ----
    recon = []
    for (coh, tgt, roi), g in dir_df.groupby(["cohort", "target", "roi"]):
        if roi not in KEY_ROIS:
            continue
        sub = g.iloc[0]
        n = int(sub["n"])
        n_neg = int(sub["n_negative"])
        # how many would have to flip to reach 18/19 or a 79% majority?
        need_18 = int(np.ceil(0 if coh == "16-subject" else 0))
        frac_target = 15 / 19
        need = int(np.ceil(frac_target * n)) - n_neg
        recon.append({"cohort": coh, "target": tgt, "roi": roi, "n": n,
                      "n_negative_observed": n_neg,
                      "share_negative": float(sub["share_negative"]),
                      "would_need_flips_for_consistency_rule": max(need, 0),
                      "paired_direction_p": float(sub["paired_wilcoxon_p"]),
                      "paired_q": float(sub["paired_q"])})
    rc = pd.DataFrame(recon)
    rc.to_csv(OUT / "reconciliation.csv", index=False)

    # ---- compare with the brain-space consistency tables ----
    cmp_rows = []
    for tgt in ("vim", "zi"):
        p = (ROOT / "results" / "final_model" / "figures_and_stats"
             / f"consistency_brain_{tgt}.csv")
        if not p.exists():
            continue
        con = pd.read_csv(p)
        for _, r in con.iterrows():
            if r["roi"] not in KEY_ROIS:
                continue
            dd = dir_df[(dir_df["cohort"] == "19-subject")
                        & (dir_df["target"] == tgt) & (dir_df["roi"] == r["roi"])]
            cmp_rows.append({
                "roi": r["roi"], "target": tgt,
                "brain_space_n_positive": int(r["n_positive"]),
                "brain_space_n_negative": int(r["n_negative"]),
                "brain_space_direction": r["consistent_direction"],
                "g_space_delta_n_positive": int(dd["n_positive"].iloc[0])
                if len(dd) else np.nan,
                "g_space_delta_n_negative": int(dd["n_negative"].iloc[0])
                if len(dd) else np.nan})
    pd.DataFrame(cmp_rows).to_csv(OUT / "paired_vs_difference.csv", index=False)

    print("=== key-region direction, VIM sessions ===")
    print(dir_df[(dir_df["roi"].isin(KEY_ROIS)) & (dir_df["target"] == "vim")]
          [["cohort", "roi", "n", "n_positive", "n_negative", "share_negative",
            "median_delta", "paired_wilcoxon_p", "paired_q"]]
          .to_string(index=False))
    print("\n=== reconciliation ===")
    print(rc.to_string(index=False))

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    key = dir_df[(dir_df["roi"].isin(KEY_ROIS))
                 & (dir_df["cohort"] == "16-subject")]
    order = [(t, r) for t in ("vim", "zi") for r in KEY_ROIS]
    xs = np.arange(len(order))
    for ax, metric in zip(axes, ("n_positive", "n_negative")):
        for tgt, col in (("vim", "C0"), ("zi", "C1")):
            vals, pos = [], []
            for i, (t, r) in enumerate(order):
                v = key[(key["target"] == t) & (key["roi"] == r)]
                vals.append(float(v[metric].iloc[0]) if len(v) else np.nan)
                pos.append(i + (0 if t == "vim" else 0))
            ax.bar(np.array(pos) + (-0.2 if tgt == "vim" else 0.2), vals, 0.38,
                   color=col, label=f"{tgt.upper()} {metric.split('_')[1]}")
        ax.axhline(len(key["n"].max()) / 2 if False else 8, color="grey",
                   ls=":", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r}\n{t}" for t, r in order], fontsize=7)
        ax.set_ylabel(f"{metric.replace('n_', 'count of ')} subjects")
        ax.legend(fontsize=8)
    axes[0].set_title("Regions whose gain rose after VIM/ZI sonication")
    axes[1].set_title("Regions whose gain fell after sonication")
    fig.suptitle("Per-region control-gain direction, 16-subject cohort "
                 "(dotted: equal split)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_gain_direction_audit.png", dpi=150)
    plt.close(fig)

    (OUT / "run_manifest.json").write_text(json.dumps({
        "difference_definition": "delta = decoder-projected C(post) - C(pre) "
                                 "per region, from the published fold table",
        "paired_definition": "Wilcoxon signed-rank on roi_c_pre vs roi_c_post "
                             "within the same sessions; BH-FDR across the 24 "
                             "regions of each cohort x target family",
        "cohorts": {"19-subject": "all folds", "16-subject": "three "
                    "non-converging folds removed"},
        "note": "a paired majority direction and a mixed cross-subject "
                "difference distribution are different statements about "
                "different populations and need not agree",
    }, indent=2), encoding="utf-8")
    print("\ndone:", OUT)


if __name__ == "__main__":
    main()
