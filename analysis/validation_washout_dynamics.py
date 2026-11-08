#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Washout-phase dynamics of the control gain versus K's decay time constants.

Four-phase per-subject C trajectories are assembled from the LOSO delta
summary: for each subject the first-treated session gives (C_pre1, C_post1)
and the second-treated session gives (C_pre2, C_post2), where the phase order
follows the treatment group (VIM_first or ZI_first). Three contrasts per
target x ROI x subject:

    effect1   = C_post1 - C_pre1          (first sonication)
    washout   = C_pre2  - C_post1         (4 h interval, no sonication)
    effect2   = C_post2 - C_pre2          (second sonication)

Per treatment order the washout contrast is tested (one-sample Wilcoxon
signed-rank) across subjects, ROI by ROI, plus the order-by-target
interaction; and the slowest K-decay time constants (tau from the reference
operator) are reported against the 4 h washout window to state what decay
time scale the propagated-history term would imply.

Outputs (results/validation/washout_dynamics/):
    washout_contrasts.csv
    washout_group_summary.csv
    k_tau_vs_washout_window.csv
    fig_washout_dynamics.png

Usage:
    python analysis/validation_washout_dynamics.py

Dependencies: numpy, pandas, scipy, matplotlib.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "validation" / "washout_dynamics"
LOSO = (ROOT / "results" / "figures" / "loso"
        / "loso_19_fold_beta_0.2_13to1to5_split")
WASH_S = 4 * 3600.0
TR = 2.0


def assemble():
    rows = []
    cache = LOSO / "cache"
    for p in sorted(cache.glob("sub-*.csv")):
        d = pd.read_csv(p)
        subj = d["subject_id"].iloc[0]
        order = d["group_str"].iloc[0]          # VIM_first / ZI_first
        first, second = (("vim", "zi") if order == "VIM_first"
                         else ("zi", "vim"))
        for roi in d["roi"].unique():
            def cv(t, ph):
                v = d[(d["target"] == t) & (d["roi"] == roi)
                      & (d[f"roi_c_{ph}"])]
                return float(v[f"roi_c_{ph}"].iloc[0]) if len(v) else np.nan
            rows.append({"subject_id": subj, "order": order,
                         "first_target": first, "second_target": second,
                         "roi": roi,
                         "c_pre1": cv(first, "pre"), "c_post1": cv(first, "post"),
                         "c_pre2": cv(second, "pre"), "c_post2": cv(second, "post")})
    df = pd.DataFrame(rows)
    df["effect1"] = df["c_post1"] - df["c_pre1"]
    df["washout"] = df["c_pre2"] - df["c_post1"]
    df["effect2"] = df["c_post2"] - df["c_pre2"]
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = assemble()
    df.to_csv(OUT / "washout_contrasts.csv", index=False)

    summ = []
    for (order, phase), g in df.groupby(["order", "second_target"]):
        for roi, gg in g.groupby("roi"):
            vals = gg[{"vim": "effect1", "zi": "effect2"}.get(phase, "effect1")]
    summ_rows = []
    for (order, roi), g in df.groupby(["order", "roi"]):
        w = g["washout"].dropna()
        e1 = g["effect1"].dropna()
        if len(w) >= 6:
            try:
                stat, p = wilcoxon(w)
            except ValueError:
                stat, p = np.nan, 1.0
            summ_rows.append({
                "order": order, "roi": roi, "n": len(w),
                "median_washout_delta": float(w.median()),
                "washout_p": float(p),
                "median_effect1": float(e1.median()),
                "washout_over_effect1": (float(w.median() / e1.median())
                                          if len(e1) and abs(e1.median()) > 1e-12
                                          else np.nan)})
    sdf = pd.DataFrame(summ_rows)
    sdf["q_washout"] = np.nan
    for order, g in sdf.groupby("order"):
        p = g["washout_p"].to_numpy()
        n = len(p)
        order_idx = np.argsort(p)
        ranked = p[order_idx] * n / np.arange(1, n + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        q = np.empty(n); q[order_idx] = np.minimum(ranked, 1.0)
        sdf.loc[g.index, "q_washout"] = q
    sdf.to_csv(OUT / "washout_group_summary.csv", index=False)

    ev = pd.read_csv(ROOT / "results" / "spectrum_stats" /
                     "eigenvalue_records.csv")
    ref = ev[ev["model"] == ev["model"].iloc[0]] if len(ev) else ev
    try:
        tau_tab = pd.read_csv(ROOT / "results" / "validation" / "k_filter_modes" /
                              "stability_dc_check.csv")
    except FileNotFoundError:
        tau_tab = None
    slow = ev.groupby("model")["eigenvalue_mag"].max().reset_index()
    slow["tau_s_at_max_lambda"] = -TR / np.log(slow["eigenvalue_mag"])
    slow["washout_window_s"] = WASH_S
    slow["steps_in_washout"] = WASH_S / TR
    slow["residual_at_washout"] = slow["eigenvalue_mag"] ** (WASH_S / TR)
    slow.to_csv(OUT / "k_tau_vs_washout_window.csv", index=False)

    for order in ["VIM_first", "ZI_first"]:
        g = sdf[sdf["order"] == order].sort_values(
            "median_washout_delta", key=lambda s: s.abs(), ascending=False)
        colors = ["C3" if q < 0.05 else "#999999" for q in g["q_washout"]]
        n_surv = int((g["q_washout"] < 0.05).sum())
        fig, ax = plt.subplots(figsize=(7.5, 8))
        ax.barh(range(len(g)), g["median_washout_delta"], color=colors)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g["roi"], fontsize=9)
        ax.set_ylabel("ROI (control gain projected to brain space)")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{order}: median C change across 4 h washout\n"
                     f"red = survives within-order BH-FDR "
                     f"({n_surv}/{len(g)} here)", fontsize=12)
        ax.set_xlabel("change in ROI control gain C, arbitrary C units\n"
                      "C(session 2, pre-sonication)\n"
                      "  minus C(session 1, post-sonication)\n"
                      "i.e. drift over the 4 h no-stimulation interval")
        fig.tight_layout()
        fig.savefig(OUT / f"fig_washout_{order.lower()}.png", dpi=120)
        plt.close(fig)

    piv = sdf.pivot_table(index="roi", columns="order",
                          values="median_washout_delta")
    both = piv.dropna()
    print("washout delta sign by order (counts):")
    print({"VIM_first>0": int((sdf[sdf['order']=='VIM_first']['median_washout_delta'] > 0).sum()),
           "VIM_first<0": int((sdf[sdf['order']=='VIM_first']['median_washout_delta'] < 0).sum()),
           "ZI_first>0": int((sdf[sdf['order']=='ZI_first']['median_washout_delta'] > 0).sum()),
           "ZI_first<0": int((sdf[sdf['order']=='ZI_first']['median_washout_delta'] < 0).sum())})
    print("FDR survivors:", int((sdf["q_washout"] < 0.05).sum()))
    print(slow.to_string(index=False))
    print("done:", OUT)


if __name__ == "__main__":
    main()