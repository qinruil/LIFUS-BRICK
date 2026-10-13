#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sensitivity analysis for delta-C.

Two layers of perturbation are quantified:

A) Fold exclusion sensitivity. The main report excludes three divergent LOSO
   folds (reconstruction loss far above the rest). Here we compare, per
   target and ROI, the sign and magnitude of delta-C between the 19-fold and
   16-fold summaries, and report the fraction of ROI whose sign flips.

B) Training/division seed sensitivity. The seed-sweep runs (3 training seeds
   x 3 division seeds, single-split models) are compared with the reference
   delta-C direction (16-patient LOSO mean) per target and ROI. Agreement
   rates and correlation are reported as descriptive robustness, with the
   caveat that seed runs use a different split and a smaller sample.

Outputs: results/statistics_supplement/fold_exclusion_sensitivity.csv
         results/statistics_supplement/seed_direction_agreement.csv
         results/statistics_supplement/seed_robustness_summary.csv
"""

from pathlib import Path
import csv
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "statistics_supplement"
LOSO_DIR = ROOT / "results" / "figures" / "loso" / "loso_19_fold_beta_0.2_13to1to5_split"
FULL_CSV = LOSO_DIR / "loso_delta_summary.csv"
FILT_CSV = LOSO_DIR / "loso_delta_summary_filtered.csv"
SEED_CSV = ROOT / "results" / "figures" / "seed_effects" / "seed_effects_roi_summary.csv"


def load_dc(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    out = {}
    for r in rows:
        key = (r["target"], r["roi"])
        out.setdefault(key, []).append(float(r["delta"]))
    means = {k: float(np.mean(v)) for k, v in out.items()}
    return means


def main():
    RES.mkdir(parents=True, exist_ok=True)
    full = load_dc(FULL_CSV)
    filt = load_dc(FILT_CSV)
    keys = sorted(set(full) & set(filt))

    fold_rows = []
    flip_count = 0
    for k in keys:
        d19, d16 = full[k], filt[k]
        s19 = 1 if d19 > 0 else (-1 if d19 < 0 else 0)
        s16 = 1 if d16 > 0 else (-1 if d16 < 0 else 0)
        flip = (s19 != 0 and s16 != 0 and s19 != s16)
        flip_count += int(flip)
        fold_rows.append({
            "target": k[0], "roi": k[1],
            "mean_dC_19fold": round(d19, 6), "mean_dC_16fold": round(d16, 6),
            "sign_19": s19, "sign_16": s16, "sign_flip": int(flip),
            "abs_change": round(abs(d19 - d16), 6),
        })
    p = RES / "fold_exclusion_sensitivity.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_rows[0].keys()))
        w.writeheader(); w.writerows(fold_rows)
    print(f"Fold exclusion: {len(keys)} target-ROI cells, {flip_count} sign flips "
          f"({flip_count/len(keys):.1%})")

    # seed sensitivity
    seed_rows = list(csv.DictReader(open(SEED_CSV, encoding="utf-8-sig")))
    # reference direction from the filtered LOSO means
    ref = {k: (1 if filt[k] > 0 else (-1 if filt[k] < 0 else 0)) for k in keys}
    agree_rows = []
    for r in seed_rows:
        key = (r["target"], r["roi"])
        ref_s = ref.get(key)
        if ref_s is None:
            continue
        d = float(r["delta"])
        s = 1 if d > 0 else (-1 if d < 0 else 0)
        agree = int(ref_s != 0 and s == ref_s)
        agree_rows.append({
            "target": r["target"], "roi": r["roi"],
            "seed_type": r["seed_type"], "seed_value": r["seed_value"],
            "delta": r["delta"], "sign": s, "ref_sign": ref_s,
            "agree_with_ref": agree,
            "significant_fdr": r.get("significant_fdr", ""),
        })
    p = RES / "seed_direction_agreement.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agree_rows[0].keys()))
        w.writeheader(); w.writerows(agree_rows)

    # summary per seed type/value
    summ = {}
    for r in agree_rows:
        k = (r["seed_type"], r["seed_value"])
        summ.setdefault(k, {"n": 0, "agree": 0, "signif": 0})
        d = summ[k]; d["n"] += 1; d["agree"] += r["agree_with_ref"]
        if str(r.get("significant_fdr", "")).lower() == "true":
            d["signif"] += 1
    out = [{
        "seed_type": k[0], "seed_value": k[1],
        "n_roi_target_cells": v["n"],
        "agree_with_reference": v["agree"],
        "agreement_rate": round(v["agree"] / v["n"], 4),
        "n_significant_fdr": v["signif"],
    } for k, v in sorted(summ.items())]
    p = RES / "seed_robustness_summary.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    for o in out:
        print(f"  seed {o['seed_type']}/{o['seed_value']}: agreement "
              f"{o['agree_with_reference']}/{o['n_roi_target_cells']} "
              f"({o['agreement_rate']:.1%}), signif {o['n_significant_fdr']}")

    # Spearman between seed delta and reference delta
    refvals = []
    for r in agree_rows:
        key = (r["target"], r["roi"])
        if key in filt:
            refvals.append((float(r["delta"]), filt[key]))
    if len(refvals) > 10:
        ry = np.array([a for a, b in refvals])
        rx = np.array([b for a, b in refvals])
        rho, pv = stats.spearmanr(rx, ry)
        print(f"Seed-vs-ref Spearman rho = {rho:.3f} (p={pv:.2e}, n={len(refvals)})")


if __name__ == "__main__":
    main()