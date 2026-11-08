#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Showcase ranking and ablation comparison figure.

Two independent rankings of candidate case-study sessions:

  A. largest single-ROI absolute delta-C per session, with the spectral
     visibility score (mean absolute log slow-band power ratio) used as the
     tie-break and figure-quality proxy;
  B. absolute delta-C at a named key region (lh_GPi, VIM-treated sessions
     and rh_STN, ZI-treated sessions), so the picked subject has a large
     control-gain change at the region discussed in the text.

Both are written as CSV plus a printed top-3 table. A companion figure
compares the tremor-tracking strength of BRICK delta-C against the
per-subject individual-model quantities (median absolute Spearman rho per
family, VIM-rest and ZI-rest panels).

Usage:
    python analysis/validation_showcase_ranking.py

Dependencies: numpy, pandas, matplotlib.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "results" / "validation"
OUT = V / "individual_lda_ablation"

KEY_REGION = {"vim": "lh_GPi", "zi": "rh_STN"}
DIVERGENT = {"sub-fuspd09", "sub-fuspd15", "sub-fuspd19"}


def load_deltas():
    d = pd.read_csv(ROOT / "results/figures/loso"
                    / "loso_19_fold_beta_0.2_13to1to5_split"
                    / "loso_delta_summary.csv")
    return d[~d["subject_id"].isin(DIVERGENT)]


def rank_a(deltas, spec):
    rows = []
    for (subj, tgt), g in deltas.groupby(["subject_id", "target"]):
        i = g["delta"].abs().idxmax()
        roi = g.loc[i, "roi"]
        v = spec[(spec["subject_id"] == subj) & (spec["target"] == tgt)]
        vis = float(v["mean_abs_slowlogratio"].iloc[0]) if len(v) else np.nan
        rows.append({"subject_id": subj, "target": tgt, "max_abs_deltaC_roi": roi,
                     "max_abs_deltaC": float(g.loc[i, "delta"]),
                     "visibility": vis})
    r = pd.DataFrame(rows)
    return r.sort_values(["max_abs_deltaC", "visibility"], ascending=False)


def rank_b(deltas):
    rows = []
    for tgt, roi in KEY_REGION.items():
        d = deltas[(deltas["target"] == tgt) & (deltas["roi"] == roi)].copy()
        d["key_region"] = roi
        d["abs_deltaC"] = d["delta"].abs()
        rows.append(d[["subject_id", "target", "key_region", "delta", "abs_deltaC"]])
    r = pd.concat(rows)
    return r.sort_values("abs_deltaC", ascending=False)


def ablation_figure():
    corr = pd.read_csv(OUT / "tremor_correlation_comparison.csv")
    c = corr[(corr["cohort"] == "16-subject") & (corr["task"] == "rest")]
    fams = ["deltaC_brick", "deltaA_out_rownorm", "deltaA_row_rel"]
    labels = ["BRICK\ndelta-C", "individual\nVAR row-norm", "individual\nVAR row-rel"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, tgt in zip(axes, ["vim", "zi"]):
        med, best, surv = [], [], []
        for f in fams:
            g = c[(c["family"] == f) & (c["target"] == tgt)]
            med.append(float(g["rho"].abs().median()))
            best.append(float(g["rho"].abs().max()))
            surv.append(int((g["q"] < 0.05).sum()))
        x = np.arange(len(fams))
        ax.bar(x - 0.2, med, 0.38, color="#2c3e50", label="median |rho|")
        ax.bar(x + 0.2, best, 0.38, color="#c0392b", label="best |rho|")
        for xi, (m, b, s) in enumerate(zip(med, best, surv)):
            ax.text(xi + 0.2, b + 0.02, f"{s} FDR", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.axhline(0.456, color="grey", ls=":", lw=0.9)
        ax.set_title(f"{tgt.upper()}-treated, rest task")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|Spearman rho| with tremor change")
    fig.suptitle("Shared-K control gain versus per-subject linear model change "
                 "(dotted: 80%-power threshold)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_ablation_tremor_comparison.png", dpi=140)
    plt.close(fig)


def main():
    deltas = load_deltas()
    spec = pd.read_csv(V / "spectrum_deltaC" / "session_visibility_ranking.csv")
    ra = rank_a(deltas, spec)
    rb = rank_b(deltas)
    ra.to_csv(OUT / "showcase_rank_A_max_roi.csv", index=False)
    rb.to_csv(OUT / "showcase_rank_B_key_region.csv", index=False)
    print("=== A. by largest single-ROI |delta-C| per session ===")
    print(ra.head(3).to_string(index=False))
    print("=== B. by |delta-C| at key region (lh_GPi VIM / rh_STN ZI) ===")
    print(rb.head(3).to_string(index=False))
    ablation_figure()
    print("figure:", OUT / "fig_ablation_tremor_comparison.png")


if __name__ == "__main__":
    main()
