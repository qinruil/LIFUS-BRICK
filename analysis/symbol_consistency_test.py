#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical support for the sign-consistency criterion.

Purpose
-------
Adds inferential statistics to the sign-consistency criterion: a cell counts
as consistent when >= 79% of the group's subjects share one delta-C direction.
This script adds the exact binomial/sign p-value, the expected consistency
under the null, and a confidence bound on the consistency fraction. It does NOT retrain or
re-forward anything; it recomputes direction counts from the stored per-subject
delta table and cross-checks them against the stored group statistics.

Criterion reproduced EXACTLY as in analysis/loso_analyze.py
-----------------------------------------------------------
    CONSISTENCY_FRACTION = 15 / 19
    per group x ROI:  n_pos = #(delta > 0), n_neg = #(delta < 0)
                      n_consistent = max(n_pos, n_neg)      # zeros counted in NEITHER
                      threshold    = ceil(15/19 * n)         # n INCLUDES any zeros
                      consistent   = n_consistent >= threshold
    The four groups are (target, order) cells; each subject contributes to both
    of its target cells, so the two "first tx" groups and the two "second tx"
    groups overlap in subjects WITHIN a target column. Groups are NOT disjoint:
    VIM(1st tx) and ZI(2nd tx) share the same VIM_first subjects.

What is added
-------------
1. Exact two-sided binomial p (H0: direction ~ Bernoulli(1/2)), computed on
   n_trials = n_pos + n_neg (conditional on non-zero; zeros reported separately
   and, as a check, also included in the denominator).
2. One-sided 95% Clopper-Pearson LOWER bound on the consistency fraction.
3. Null calibration: P(blue | H0) per group size and the EXPECTED number of
   blue cells across the 96 group x ROI table if direction were pure chance --
   the missing "expected level" that makes a 79% threshold interpretable.
4. BH-FDR within each group across its 24 ROIs (matching the t-test convention
   of loso_analyze.py), and pooled across all families, both reported.
5. Same three statistics for the whole-cohort brain-space tables (N=19) used by
   compare_pre_post (results/final_model/figures_and_stats/consistency_brain_*.csv).

Inputs (read only)
    results/figures/loso/loso_19_fold_beta_0.2_13to1to5_split/loso_delta_summary.csv
    results/figures/loso/loso_19_fold_beta_0.2_13to1to5_split/loso_delta_summary_filtered.csv
    results/figures/loso/loso_19_fold_beta_0.2_13to1to5_split/loso_pooled_stats.csv
    results/final_model/figures_and_stats/consistency_brain_vim.csv
    results/final_model/figures_and_stats/consistency_brain_zi.csv

Outputs: results/sign_consistency/
    group_binomial.csv           per group x ROI: counts, thresholds, p, CI, q
    null_calibration.csv         P(blue|H0) and expected blue cells by group size
    crosscheck_vs_pooled_stats.csv  reproduced vs stored criterion, mismatches
    cohort_brain_binomial.csv    N=19 brain-space tables + binomial support
    summary_line.csv             one line per group: max/min evidence after FDR
    run_manifest.json

Dependencies: numpy, scipy, stdlib (no pandas/matplotlib/statsmodels).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
LOSO_DIR = ROOT / "results" / "figures" / "loso" / "loso_19_fold_beta_0.2_13to1to5_split"
FULL_CSV = LOSO_DIR / "loso_delta_summary.csv"
FILTERED_CSV = LOSO_DIR / "loso_delta_summary_filtered.csv"
POOLED_CSV = LOSO_DIR / "loso_pooled_stats.csv"
COHORT_VIM = ROOT / "results/final_model/figures_and_stats/consistency_brain_vim.csv"
COHORT_ZI = ROOT / "results/final_model/figures_and_stats/consistency_brain_zi.csv"
OUT_DIR = ROOT / "results" / "sign_consistency"

# verbatim from loso_analyze.py
CONSISTENCY_FRACTION = 15 / 19
TREATMENT_GROUPS = [
    ("VIM (1st tx)", "vim", "VIM_first"),
    ("VIM (2nd tx)", "vim", "ZI_first"),
    ("ZI (1st tx)",  "zi",  "ZI_first"),
    ("ZI (2nd tx)",  "zi",  "VIM_first"),
]


# ------------------------------------------------------------------ helpers --
def read_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return "nan" if math.isnan(f) else f"{f:.6g}"
    return str(v)


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in fields})


def bh_fdr(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(np.asarray(pvals, dtype=float))
    q, prev = np.empty(n), 1.0
    for rank in range(n, 0, -1):
        idx = order[rank - 1]
        prev = min(prev, pvals[idx] * n / rank)
        q[idx] = prev
    return [float(x) for x in q]


def binom_support(k: int, n: int) -> dict:
    """Exact two-sided p, one-sided (greater) p, one-sided 95% CP lower bound."""
    if n == 0:
        return {"p_two": float("nan"), "p_one": float("nan"), "cp_lo95": float("nan")}
    bt = stats.binomtest(k, n, 0.5)
    # one-sided 95% lower bound == low end of the two-sided 90% exact interval
    ci = bt.proportion_ci(confidence_level=0.90, method="exact")
    # P(X >= k) under H0: p=0.5  (sf(k-1) == 1 - cdf(k-1), numerically stabler)
    p_one = float(stats.binom.sf(k - 1, n, 0.5))
    return {"p_two": float(bt.pvalue), "p_one": p_one,
            "cp_lo95": float(ci.low)}


def p_blue_null(n: int, thr: int) -> float:
    """P(max(#pos,#neg) >= thr | direction ~ Bernoulli(1/2), no zeros)."""
    if n == 0 or thr <= 0:
        return float("nan")
    upper = stats.binom.cdf(n, n, 0.5) - stats.binom.cdf(thr - 1, n, 0.5)
    return float(min(1.0, 2.0 * upper))


# --------------------------------------------------------------- load data --
def load_delta(path: Path) -> dict:
    rows = read_rows(path)
    roi_order = []
    for r in rows:
        if r["roi"] not in roi_order:
            roi_order.append(r["roi"])
    vals = {}
    for r in rows:
        vals.setdefault((r["group_str"], r["target"], r["roi"]), []).append(float(r["delta"]))
    subjects = sorted({r["subject_id"] for r in rows})
    return {"roi_order": roi_order, "vals": vals, "subjects": subjects, "n_rows": len(rows),
            "source": path.name}


# ------------------------------------------------------- group-level tests --
def group_table(data: dict) -> list[dict]:
    out = []
    for label, target, group_str in TREATMENT_GROUPS:
        for roi in data["roi_order"]:
            d = np.asarray(data["vals"].get((group_str, target, roi), []), dtype=float)
            n = d.size
            if n == 0:
                continue
            n_pos = int((d > 0).sum())
            n_neg = int((d < 0).sum())
            n_zero = n - n_pos - n_neg
            k = max(n_pos, n_neg)
            direction = "increase" if n_pos >= n_neg else "decrease"
            thr = math.ceil(CONSISTENCY_FRACTION * n)
            consistent = k >= thr
            # conditional on non-zero (primary): zeros carry no direction
            b = binom_support(k, n_pos + n_neg)
            # inclusive denominator (check): zeros counted against consistency
            b_inc = binom_support(k, n)
            row = {"group_label": label, "target": target, "group_str": group_str,
                   "roi": roi, "n": n, "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero,
                   "n_consistent": k, "consistency_direction": direction,
                   "consistency_fraction": k / n if n else float("nan"),
                   "threshold": thr, "consistent_direction": consistent,
                   "p_two_sided_binom": b["p_two"], "p_one_sided_binom": b["p_one"],
                   "cp_lower95_fraction": b["cp_lo95"],
                   "p_two_sided_zeros_included": b_inc["p_two"],
                   "expected_p_blue_given_H0": p_blue_null(n, thr)}
            out.append(row)
    return out


def crosscheck_vs_stored(computed: list[dict]) -> list[dict]:
    if not POOLED_CSV.exists():
        return []
    stored = read_rows(POOLED_CSV)
    s_idx = {(r["group_label"], r["roi"]): r for r in stored}
    rows, n_match, n_mismatch = [], 0, 0
    for c in computed:
        s = s_idx.get((c["group_label"], c["roi"]))
        if s is None:
            continue
        checks = {
            "n": (c["n"], int(float(s["n_roi"]))),
            "n_consistent": (c["n_consistent"], int(float(s["n_consistent"]))),
            "threshold": (c["threshold"], int(float(s["consistency_threshold"]))),
            "consistent_direction": (bool(c["consistent_direction"]),
                                     s["consistent_direction"].strip().lower() == "true"),
        }
        ok = all(a == b for a, b in checks.values())
        n_match += ok
        n_mismatch += (not ok)
        rows.append({"group_label": c["group_label"], "roi": c["roi"],
                     "match": ok,
                     "detail": "; ".join(f"{k}: computed={fmt(v[0])} stored={fmt(v[1])}"
                                         for k, v in checks.items() if v[0] != v[1])})
    print(f"[crosscheck] criterion fields identical for {n_match} cells, "
          f"mismatched {n_mismatch} (of {len(rows)} matched to stored pooled stats)")
    return rows


def add_fdr(computed: list[dict]) -> None:
    """BH within each group across its ROIs (matches the t-test convention),
    plus pooled across the four groups."""
    for label, _, _ in TREATMENT_GROUPS:
        sel = [r for r in computed if r["group_label"] == label]
        q = bh_fdr([float(r["p_two_sided_binom"]) for r in sel])
        for r, qq in zip(sel, q):
            r["q_bh_within_group"] = qq
    q_all = bh_fdr([float(r["p_two_sided_binom"]) for r in computed])
    for r, qq in zip(computed, q_all):
        r["q_bh_pooled_all_groups"] = qq


# --------------------------------------------------- cohort brain-space ------
def cohort_table() -> list[dict]:
    out = []
    for tgt, p in (("vim", COHORT_VIM), ("zi", COHORT_ZI)):
        if not p.exists():
            continue
        for r in read_rows(p):
            n_pos = int(r["n_positive"]); n_neg = int(r["n_negative"])
            n = n_pos + n_neg
            k = max(n_pos, n_neg)
            b = binom_support(k, n)
            out.append({"table": p.name, "target": tgt, "roi": r["roi"],
                        "n": n, "n_pos": n_pos, "n_neg": n_neg,
                        "n_consistent": k,
                        "consistency_direction": r.get("consistent_direction", ""),
                        "consistency_fraction": k / n if n else float("nan"),
                        "mean_delta": r.get("mean_delta", ""),
                        "p_two_sided_binom": b["p_two"], "p_one_sided_binom": b["p_one"],
                        "cp_lower95_fraction": b["cp_lo95"]})
    sel = out
    q = bh_fdr([float(r["p_two_sided_binom"]) for r in sel]) if sel else []
    for r, qq in zip(sel, q):
        r["q_bh_pooled_cohort_family"] = qq
    return out


# ------------------------------------------------------- null calibration ----
def calibration_rows(data: dict) -> list[dict]:
    """P(blue | H0) and expected number of blue cells per group size."""
    rows, seen = [], set()
    for label, target, group_str in TREATMENT_GROUPS:
        sizes = [len(data["vals"].get((group_str, target, roi), []))
                 for roi in data["roi_order"]]
        n = int(np.median([s for s in sizes if s > 0])) if any(sizes) else 0
        if n == 0:
            continue
        thr = math.ceil(CONSISTENCY_FRACTION * n)
        p_blue = p_blue_null(n, thr)
        rows.append({"group_label": label, "n_subjects": n, "threshold": thr,
                     "consistency_fraction_at_threshold": thr / n,
                     "p_blue_given_H0_two_sided": p_blue,
                     "expected_blue_cells_of_24": p_blue * 24})
        seen.add((n, thr))
    for n, thr in sorted(seen):
        p_blue = p_blue_null(n, thr)
        rows.append({"group_label": f"ALL groups with n={n}", "n_subjects": n,
                     "threshold": thr,
                     "consistency_fraction_at_threshold": thr / n,
                     "p_blue_given_H0_two_sided": p_blue,
                     "expected_blue_cells_of_24": p_blue * 24})
    return rows


# --------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser(description="sign-consistency binomial support")
    ap.add_argument("--primary", choices=("filtered", "full"), default="filtered",
                    help="delta table for the headline analysis (stored pooled "
                         "stats were produced on the filtered/16-fold set)")
    args = ap.parse_args()

    global OUT_DIR
    if args.primary == "full":
        OUT_DIR = ROOT / "results" / "sign_consistency_all19"

    t0 = time.time()
    print("=" * 78)
    print("sign-consistency criterion: binomial tests, CIs, null calibration")
    print("=" * 78)

    primary_path = FILTERED_CSV if args.primary == "filtered" else FULL_CSV
    data = load_delta(primary_path)
    print(f"[load ] primary={data['source']} rows={data['n_rows']} "
          f"subjects={len(data['subjects'])} ROIs={len(data['roi_order'])}")

    computed = group_table(data)
    add_fdr(computed)
    check_rows = crosscheck_vs_stored(computed)
    cal = calibration_rows(data)

    # ---- headline numbers ---------------------------------------------------
    n_cells = len(computed)
    n_blue = sum(1 for r in computed if r["consistent_direction"])
    exp_blue = sum(r["expected_p_blue_given_H0"] for r in computed)
    min_p = min(float(r["p_two_sided_binom"]) for r in computed
                if not math.isnan(float(r["p_two_sided_binom"])))
    min_qg = min(float(r["q_bh_within_group"]) for r in computed)
    min_qp = min(float(r["q_bh_pooled_all_groups"]) for r in computed)
    n_sig005 = sum(1 for r in computed if float(r["p_two_sided_binom"]) < 0.05)
    n_sig005_q = sum(1 for r in computed if float(r["q_bh_within_group"]) < 0.05)
    print(f"[group] {n_cells} group x ROI cells | blue (>=ceil(15n/19)): {n_blue} | "
          f"EXPECTED blue under H0 (pure chance): {exp_blue:.1f}")
    print(f"       p<0.05 raw: {n_sig005} | surviving within-group BH: {n_sig005_q} | "
          f"min p={min_p:.4g} min q_within={min_qg:.4g} min q_pooled={min_qp:.4g}")

    coh = cohort_table()
    if coh:
        c_blue = sum(1 for r in coh if r["consistency_fraction"] >= CONSISTENCY_FRACTION)
        c_minp = min(float(r["p_two_sided_binom"]) for r in coh)
        c_minq = min(float(r["q_bh_pooled_cohort_family"]) for r in coh)
        print(f"[cohort] N=19 brain-space tables: {len(coh)} cells | fraction>=79%: {c_blue} | "
              f"min p={c_minp:.4g} min q={c_minq:.4g}")

    # ---- outputs ------------------------------------------------------------
    gf = ["group_label", "target", "group_str", "roi", "n", "n_pos", "n_neg", "n_zero",
          "n_consistent", "consistency_direction", "consistency_fraction", "threshold",
          "consistent_direction", "p_two_sided_binom", "p_one_sided_binom",
          "cp_lower95_fraction", "p_two_sided_zeros_included",
          "expected_p_blue_given_H0", "q_bh_within_group", "q_bh_pooled_all_groups"]
    write_rows(OUT_DIR / "group_binomial.csv", gf, computed)
    write_rows(OUT_DIR / "null_calibration.csv",
               ["group_label", "n_subjects", "threshold",
                "consistency_fraction_at_threshold", "p_blue_given_H0_two_sided",
                "expected_blue_cells_of_24"], cal)
    write_rows(OUT_DIR / "crosscheck_vs_pooled_stats.csv",
               ["group_label", "roi", "match", "detail"], check_rows)
    write_rows(OUT_DIR / "cohort_brain_binomial.csv",
               ["table", "target", "roi", "n", "n_pos", "n_neg", "n_consistent",
                "consistency_direction", "consistency_fraction", "mean_delta",
                "p_two_sided_binom", "p_one_sided_binom", "cp_lower95_fraction",
                "q_bh_pooled_cohort_family"], coh)
    write_rows(OUT_DIR / "summary_line.csv",
               ["metric", "value"],
               [{"metric": "primary_table", "value": data["source"]},
                {"metric": "n_cells", "value": n_cells},
                {"metric": "n_blue_observed", "value": n_blue},
                {"metric": "expected_blue_under_H0", "value": round(exp_blue, 2)},
                {"metric": "min_p_two_sided", "value": min_p},
                {"metric": "n_p_below_0.05", "value": n_sig005},
                {"metric": "n_surviving_within_group_BH", "value": n_sig005_q},
                {"metric": "min_q_within_group", "value": min_qg},
                {"metric": "min_q_pooled_all_groups", "value": min_qp}])

    manifest = {
        "script": "analysis/symbol_consistency_test.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_input": str(primary_path.relative_to(ROOT)),
        "criterion_reproduced_from": "analysis/loso_analyze.py (CONSISTENCY_FRACTION=15/19, "
                                     "threshold=ceil(frac*n), zeros counted in neither sign "
                                     "but included in n)",
        "null_model": "direction ~ Bernoulli(1/2) i.i.d. across subjects within a group",
        "notes": [
            "binomial p conditional on non-zero delta is primary; zero-inclusive "
            "denominator reported as a check column.",
            "within-group BH over 24 ROIs mirrors the t-test correction convention; "
            "pooled-over-4-groups BH is the conservative second reading.",
            "groups overlap in subjects across target columns (VIM(1st tx) and "
            "ZI(2nd tx) share the VIM_first subjects), so the pooled family is not "
            "independent; treat its q as indicative.",
        ],
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "numpy": np.__version__, "scipy": __import__("scipy").__version__},
        "elapsed_sec": round(time.time() - t0, 2),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"[done ] {OUT_DIR}  elapsed={manifest['elapsed_sec']}s")


if __name__ == "__main__":
    main()
