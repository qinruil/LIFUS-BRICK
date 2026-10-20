#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Effect sizes and bootstrap confidence intervals for delta-C.

For each target x ROI x order group (VIM1st, VIM2nd, ZI1st, ZI2nd), delta-C is
a within-subject pre-to-post change; the standardized effect is therefore the
one-sample Cohen's d = mean(delta)/sd(delta). Bias-corrected and accelerated
(BCa) is not needed for the group-level summary; we give percentile bootstrap
95% CIs instead (2000 resamples, fixed seed).

A group-level summary (median d, share of ROI with |d| >= 0.5) is also written,
together with the power-relevant reference thresholds used elsewhere:
|rho| >= 0.545 (n = 24 ROI correlation) and d = 1.106 for 80% power under
the alpha used in the VAR comparison.

Outputs: results/statistics_supplement/delta_c_effect_sizes_grouped.csv
         results/statistics_supplement/effect_size_summary.csv
"""

from pathlib import Path
import csv
import numpy as np
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "statistics_supplement"
FILT_CSV = (ROOT / "results" / "figures" / "loso"
            / "loso_19_fold_beta_0.2_13to1to5_split" / "loso_delta_summary_filtered.csv")
RNG = np.random.default_rng(42)
N_BOOT = 2000


def main():
    RES.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(FILT_CSV, encoding="utf-8-sig")))
    # group id per subject per target
    cells = defaultdict(list)
    for r in rows:
        tgt, g = r["target"], r["group_str"]
        if tgt == "vim":
            gid = "VIM1st" if g == "VIM_first" else "VIM2nd"
        else:
            gid = "ZI1st" if g == "ZI_first" else "ZI2nd"
        cells[(tgt, r["roi"], gid)].append(float(r["delta"]))

    out = []
    for (tgt, roi, gid), vals in sorted(cells.items()):
        a = np.array(vals)
        n = len(a)
        m = float(a.mean())
        sd = float(a.std(ddof=1)) if n > 1 else 0.0
        d = m / sd if sd > 0 else float("nan")
        ci_lo = ci_hi = float("nan")
        if sd > 0 and n >= 3:
            boots = []
            for _ in range(N_BOOT):
                b = RNG.choice(a, size=n, replace=True)
                bs = b.std(ddof=1)
                if bs > 0:
                    boots.append(float(b.mean()) / bs)
            if boots:
                ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        out.append({
            "target": tgt, "group": gid, "roi": roi, "n": n,
            "mean_delta": round(m, 6), "sd_delta": round(sd, 6),
            "cohen_d": round(d, 4) if d == d else "",
            "boot_ci_lo": round(float(ci_lo), 4) if ci_lo == ci_lo else "",
            "boot_ci_hi": round(float(ci_hi), 4) if ci_hi == ci_hi else "",
        })
    p = RES / "delta_c_effect_sizes_grouped.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    # summary per target x group
    summ = defaultdict(list)
    for r in out:
        if r["cohen_d"] != "":
            summ[(r["target"], r["group"])].append(float(r["cohen_d"]))
    rows_s = []
    for (tgt, gid), ds in sorted(summ.items()):
        a = np.array(ds)
        rows_s.append({
            "target": tgt, "group": gid, "n_roi": len(ds),
            "median_d": round(float(np.median(a)), 4),
            "q1_d": round(float(np.percentile(a, 25)), 4),
            "q3_d": round(float(np.percentile(a, 75)), 4),
            "n_roi_abs_d_ge_0.5": int((np.abs(a) >= 0.5).sum()),
            "min_d": round(float(a.min()), 4), "max_d": round(float(a.max()), 4),
        })
    p = RES / "effect_size_summary.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_s[0].keys()))
        w.writeheader(); w.writerows(rows_s)

    print("Per-group median |d| and share with |d|>=0.5:")
    for r in rows_s:
        print(f"  {r['target']}/{r['group']:<6} median d={r['median_d']:>6} "
              f"abs>=0.5: {r['n_roi_abs_d_ge_0.5']}/{r['n_roi']}")
    # cross-group overlap of |d| >= 0.5 sets
    strong = defaultdict(set)
    for r in out:
        if r["cohen_d"] != "" and abs(float(r["cohen_d"])) >= 0.5:
            strong[(r["target"], r["group"])].add(r["roi"])
    print("ROI with |d|>=0.5 per group:", {f"{k[0]}/{k[1]}": len(v)
                                            for k, v in sorted(strong.items())})
    # reference thresholds for context
    print("Reference thresholds: |rho|>=0.545 (n=24), d=1.106 (80% power, alpha=9.06e-5)")


if __name__ == "__main__":
    main()